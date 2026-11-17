"""
Executive AI Chief of Staff - Email Summarizer
Extractive summarization for single emails and thread conversations.
No external API — all local sentence scoring.
"""
import logging
import math
import re
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _sentence_split(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _word_frequencies(text: str) -> Dict[str, float]:
    """Compute normalized TF for words in text."""
    words = re.findall(r"\b[a-zA-Z]\w{2,}\b", text.lower())
    # Remove stopwords inline
    _SW = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
           "but", "if", "as", "with", "from", "by", "not", "no", "is", "are",
           "was", "were", "be", "been", "have", "has", "do", "does", "did",
           "will", "would", "can", "could", "should", "may", "might", "this",
           "that", "these", "those", "i", "we", "you", "he", "she", "it", "they"}
    words = [w for w in words if w not in _SW]
    if not words:
        return {}
    freq = Counter(words)
    max_f = max(freq.values())
    return {w: c / max_f for w, c in freq.items()}


def _score_sentence(sentence: str, freq: Dict[str, float], position: int, total: int) -> float:
    """Score a sentence by TF-weighted word overlap + position bias."""
    words = re.findall(r"\b[a-zA-Z]\w{2,}\b", sentence.lower())
    if not words:
        return 0.0
    tf_score = sum(freq.get(w, 0) for w in words) / len(words)
    # Position bias: first and last sentences get a boost
    pos_ratio = position / max(total - 1, 1)
    pos_boost = 1.2 if position == 0 else (0.9 if pos_ratio > 0.85 else 1.0)
    # Length penalty for very short sentences
    len_penalty = min(len(words) / 15.0, 1.0)
    return tf_score * pos_boost * len_penalty


class EmailSummarizer:
    """Extractive email and thread summarizer."""

    def summarize_email(self, text: str, max_sentences: int = 3) -> str:
        """Extractive summarization of a single email body."""
        if not text or len(text) < 100:
            return text or ""
        sentences = _sentence_split(text)
        if len(sentences) <= max_sentences:
            return " ".join(sentences)
        freq = _word_frequencies(text)
        scored = [
            (i, s, _score_sentence(s, freq, i, len(sentences)))
            for i, s in enumerate(sentences)
        ]
        # Sort by score, pick top N, then re-order by original position
        top = sorted(scored, key=lambda x: x[2], reverse=True)[:max_sentences]
        top_sorted = sorted(top, key=lambda x: x[0])
        return " ".join(s for _, s, _ in top_sorted)

    def summarize_thread(self, emails: List[Dict], max_sentences: int = 5) -> str:
        """Summarize an entire thread chronologically."""
        if not emails:
            return "No emails in thread."

        # Sort by date
        def _parse_date(e: Dict) -> datetime:
            d = e.get("date")
            if isinstance(d, datetime):
                return d
            try:
                return datetime.fromisoformat(str(d))
            except Exception:
                return datetime(2000, 1, 1)

        sorted_emails = sorted(emails, key=_parse_date)

        # Combine subjects for context
        subjects = list(dict.fromkeys(
            e.get("subject", "").replace("Re: ", "").replace("Fwd: ", "").strip()
            for e in sorted_emails if e.get("subject")
        ))

        # Build combined text from most recent (most relevant) emails
        combined = " ".join(
            e.get("cleaned_body") or e.get("body") or ""
            for e in sorted_emails[-5:]  # last 5 emails of thread
        )

        summary = self.summarize_email(combined, max_sentences=max_sentences)
        subject_str = subjects[0] if subjects else "Unknown"
        return f"Thread: '{subject_str}' ({len(emails)} messages). Summary: {summary}"

    def extract_key_points(self, text: str) -> List[str]:
        """Extract bullet-point key points from email text."""
        points: List[str] = []
        sentences = _sentence_split(text)
        freq = _word_frequencies(text)

        for i, s in enumerate(sentences):
            score = _score_sentence(s, freq, i, len(sentences))
            if score > 0.3 and len(s) > 30:
                # Clean up and add as bullet
                clean = re.sub(r"\s+", " ", s).strip()
                points.append(clean)
            if len(points) >= 5:
                break
        return points

    def generate_briefing_summary(
        self,
        emails: List[Dict],
        commitments: List[Dict],
        risks: List[Dict],
        deadlines: List[Dict],
    ) -> Dict:
        """Create structured briefing with sections."""
        # Top emails by priority
        top_emails = sorted(emails, key=lambda e: e.get("priority_score", 0), reverse=True)[:5]

        # Key developments: distinct topics from high-priority emails
        developments = []
        seen_subjects: set = set()
        for e in top_emails:
            subj = e.get("subject", "")[:80]
            if subj and subj.replace("Re: ", "") not in seen_subjects:
                seen_subjects.add(subj.replace("Re: ", ""))
                body = e.get("body") or ""
                summary = self.summarize_email(body, max_sentences=1)
                developments.append({
                    "subject": subj,
                    "sender": e.get("sender", ""),
                    "summary": summary,
                    "priority": e.get("priority_level", "medium"),
                })

        # Pending decisions: action_required + high priority
        pending_decisions = [
            e for e in emails
            if e.get("annotations", {}).get("action_required") and
               e.get("priority_level") in ("critical", "high")
        ][:5]

        # Upcoming deadlines
        upcoming = sorted(
            [d for d in deadlines if d.get("deadline")],
            key=lambda d: d.get("deadline", ""),
        )[:5]

        return {
            "top_emails": [{"subject": e.get("subject", ""), "sender": e.get("sender", ""),
                            "priority": e.get("priority_level", "medium"),
                            "score": e.get("priority_score", 0)} for e in top_emails],
            "key_developments": developments,
            "pending_decisions": [
                {"subject": e.get("subject", ""), "sender": e.get("sender", "")}
                for e in pending_decisions
            ],
            "upcoming_deadlines": upcoming,
            "overdue_commitments": [c for c in commitments if c.get("status") == "overdue"],
            "critical_risks": [r for r in risks if r.get("severity") in ("critical", "high")][:5],
            "recommended_actions": self._generate_actions(top_emails, commitments, risks),
        }

    def summarize_period(self, emails: List[Dict], start_date: datetime, end_date: datetime) -> Dict:
        """Summarize what happened in a date range, grouped by topic."""
        if not emails:
            return {"period": f"{start_date.date()} – {end_date.date()}", "sections": {}}

        def _parse_date(e: Dict) -> Optional[datetime]:
            d = e.get("date")
            if isinstance(d, datetime):
                return d
            try:
                return datetime.fromisoformat(str(d))
            except Exception:
                return None

        # Filter to date range
        in_range = [e for e in emails if (dt := _parse_date(e)) and start_date <= dt <= end_date]

        # Group by thread
        thread_groups: Dict[str, List[Dict]] = {}
        for e in in_range:
            tid = e.get("thread_id") or e.get("message_id", "none")
            thread_groups.setdefault(tid, []).append(e)

        # Summarize each thread
        thread_summaries = []
        for tid, thread_emails in list(thread_groups.items())[:20]:
            subj = thread_emails[-1].get("subject", "Unknown")[:60]
            summary = self.summarize_thread(thread_emails, max_sentences=2)
            thread_summaries.append({
                "thread_id": tid,
                "subject": subj,
                "email_count": len(thread_emails),
                "summary": summary,
                "latest_sender": thread_emails[-1].get("sender", ""),
            })

        return {
            "period": f"{start_date.date()} – {end_date.date()}",
            "total_emails": len(in_range),
            "total_threads": len(thread_groups),
            "thread_summaries": thread_summaries[:15],
        }

    @staticmethod
    def _generate_actions(emails: List[Dict], commitments: List[Dict], risks: List[Dict]) -> List[str]:
        actions: List[str] = []
        for e in emails[:3]:
            subj = e.get("subject", "")[:50]
            if e.get("priority_level") == "critical":
                actions.append(f"Respond immediately to: {subj}")
            elif e.get("annotations", {}).get("action_required"):
                actions.append(f"Review and act on: {subj}")
        for c in commitments[:3]:
            if c.get("status") == "overdue":
                actions.append(f"Follow up on overdue commitment: {c.get('description', '')[:50]}")
        for r in risks[:2]:
            if r.get("severity") == "critical":
                actions.append(f"Address critical {r.get('category', '')} risk")
        return actions[:7]

