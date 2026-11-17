"""
Executive AI Chief of Staff - Conversation Memory
Thread reconstruction, conversation tracking, and contextual analysis.
Uses header matching, subject-line heuristics, and optional semantic
similarity to group emails into coherent threads and extract insights.
"""

import hashlib
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import SIMILARITY_THRESHOLD
from src.utils.helpers import (
    clean_html,
    compute_cosine_similarity,
    normalize_email,
    safe_json_loads,
    truncate_string,
)

logger = logging.getLogger(__name__)

# Patterns for extracting questions and decisions
_QUESTION_PATTERN = re.compile(
    r"([A-Z][^.!?\n]*\?)", re.MULTILINE
)
_DECISION_KEYWORDS = [
    "decided", "agreed", "approved", "confirmed", "finalized",
    "we will", "we'll", "going with", "let's go with",
    "decision is", "plan is to", "moving forward with",
    "signed off", "greenlit", "green-lit", "go-ahead",
]
_ISSUE_KEYWORDS = [
    "issue", "problem", "concern", "bug", "error", "failure",
    "broken", "not working", "outage", "incident", "blocked",
    "impediment", "obstacle", "challenge", "risk",
]
_RESOLUTION_KEYWORDS = [
    "resolved", "fixed", "addressed", "closed", "done",
    "completed", "sorted", "handled", "mitigated",
]


class ConversationMemory:
    """
    Manages email thread reconstruction, conversation tracking, and
    contextual analysis for the executive assistant.

    Uses a multi-pass approach:
    1. Header-based linking (In-Reply-To, References)
    2. Subject-line matching (Re:, Fwd: prefix normalisation)
    3. Optional semantic similarity via embeddings
    """

    def __init__(self, db_manager, embedding_manager=None) -> None:
        """
        Initialise ConversationMemory.

        Args:
            db_manager: DatabaseManager instance.
            embedding_manager: Optional embedding manager for semantic
                               similarity (should expose an ``encode`` method).
        """
        self.db = db_manager
        self.embedding_manager = embedding_manager
        logger.info("ConversationMemory initialised (embeddings=%s)",
                     "yes" if embedding_manager else "no")

    # ------------------------------------------------------------------
    # Thread reconstruction
    # ------------------------------------------------------------------

    def reconstruct_threads(self, emails: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Reconstruct email threads using three passes:
        1. In-Reply-To / References headers
        2. Subject line matching (Re:, Fwd:)
        3. Semantic similarity of content

        Args:
            emails: List of email dicts with keys message_id, in_reply_to,
                    references_field, subject, body, date, sender, etc.

        Returns:
            Dict mapping thread_id -> list of email dicts ordered by date.
        """
        # Union-Find for thread grouping
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Index emails by message_id
        by_msg_id: Dict[str, Dict[str, Any]] = {}
        for email in emails:
            msg_id = email.get("message_id", "")
            if msg_id:
                by_msg_id[msg_id] = email
                parent.setdefault(msg_id, msg_id)

        # ---- Pass 1: Header-based linking ----
        for email in emails:
            msg_id = email.get("message_id", "")
            if not msg_id:
                continue

            # In-Reply-To
            reply_to = email.get("in_reply_to", "")
            if reply_to and reply_to in by_msg_id:
                union(msg_id, reply_to)

            # References
            refs = email.get("references_field", "") or ""
            if isinstance(refs, str):
                ref_ids = re.findall(r"<([^>]+)>", refs) if "<" in refs else refs.split()
            else:
                ref_ids = []
            for ref_id in ref_ids:
                if ref_id in by_msg_id:
                    union(msg_id, ref_id)

        # ---- Pass 2: Subject-line matching ----
        subject_groups: Dict[str, List[str]] = defaultdict(list)
        for email in emails:
            msg_id = email.get("message_id", "")
            if not msg_id:
                continue
            norm_subject = self._normalize_subject(email.get("subject", ""))
            if norm_subject:
                subject_groups[norm_subject].append(msg_id)

        for group_ids in subject_groups.values():
            if len(group_ids) > 1:
                for i in range(1, len(group_ids)):
                    union(group_ids[0], group_ids[i])

        # ---- Pass 3: Semantic similarity (optional) ----
        if self.embedding_manager and len(emails) <= 500:
            try:
                self._semantic_link(emails, by_msg_id, parent, find, union)
            except Exception as exc:
                logger.warning("Semantic linking failed: %s", exc)

        # ---- Build thread groups ----
        thread_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        assigned: Set[str] = set()

        for email in emails:
            msg_id = email.get("message_id", "")
            if msg_id:
                root = find(msg_id)
                thread_id = f"thread_{hashlib.md5(root.encode()).hexdigest()[:12]}"
                thread_groups[thread_id].append(email)
                assigned.add(msg_id)

        # Handle emails without message_id
        for email in emails:
            msg_id = email.get("message_id", "")
            if not msg_id or msg_id not in assigned:
                # Try to match by subject
                norm_subj = self._normalize_subject(email.get("subject", ""))
                placed = False
                if norm_subj:
                    for tid, thread_emails in thread_groups.items():
                        for te in thread_emails:
                            if self._normalize_subject(te.get("subject", "")) == norm_subj:
                                thread_groups[tid].append(email)
                                placed = True
                                break
                        if placed:
                            break
                if not placed:
                    solo_id = f"thread_{hashlib.md5(str(id(email)).encode()).hexdigest()[:12]}"
                    thread_groups[solo_id].append(email)

        # Sort each thread by date
        for tid in thread_groups:
            thread_groups[tid].sort(
                key=lambda e: self._parse_email_date(e.get("date"))
            )

        # Persist threads to database
        for tid, thread_emails in thread_groups.items():
            self._persist_thread(tid, thread_emails)

        logger.info(
            "Reconstructed %d threads from %d emails",
            len(thread_groups), len(emails),
        )
        return dict(thread_groups)

    def get_thread_summary(self, thread_id: str) -> Dict[str, Any]:
        """
        Summarise a thread: key points, decisions, pending items, participants.

        Args:
            thread_id: The thread identifier.

        Returns:
            Dict with thread summary information.
        """
        try:
            thread_emails = self._get_thread_emails(thread_id)
            if not thread_emails:
                return {"thread_id": thread_id, "error": "Thread not found"}

            participants: Set[str] = set()
            all_text = []
            subjects: Set[str] = set()
            dates: List[datetime] = []

            for email in thread_emails:
                sender = normalize_email(email.get("sender", ""))
                if sender:
                    participants.add(sender)
                recipients = email.get("recipients", [])
                if isinstance(recipients, str):
                    recipients = safe_json_loads(recipients, [])
                for r in recipients:
                    norm_r = normalize_email(r)
                    if norm_r:
                        participants.add(norm_r)

                body = clean_html(email.get("body", ""))
                if body:
                    all_text.append(body)
                subj = email.get("subject", "")
                if subj:
                    subjects.add(subj)
                dt = self._parse_email_date(email.get("date"))
                if dt:
                    dates.append(dt)

            combined_text = "\n".join(all_text)

            # Extract key points (sentences with keywords)
            key_points = self._extract_key_points(combined_text)

            # Extract decisions
            decisions = self._extract_decisions(combined_text)

            # Extract pending items (questions without answers)
            pending = self._find_pending_in_text(combined_text)

            # Latest topic
            latest_body = clean_html(thread_emails[-1].get("body", "")) if thread_emails else ""
            topic = self._normalize_subject(
                thread_emails[-1].get("subject", "") if thread_emails else ""
            )

            return {
                "thread_id": thread_id,
                "subject": list(subjects)[0] if subjects else "",
                "topic": topic,
                "participants": sorted(participants),
                "email_count": len(thread_emails),
                "date_range": {
                    "first": min(dates).isoformat() if dates else None,
                    "last": max(dates).isoformat() if dates else None,
                },
                "key_points": key_points[:10],
                "decisions": decisions[:5],
                "pending_items": pending[:5],
                "latest_message_preview": truncate_string(latest_body, 200),
            }
        except Exception as exc:
            logger.error("Error summarising thread %s: %s", thread_id, exc, exc_info=True)
            return {"thread_id": thread_id, "error": str(exc)}

    def get_pending_questions(self, thread_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find unanswered questions across threads or within a specific thread.

        Args:
            thread_id: If provided, limit search to this thread.

        Returns:
            List of dicts with question, source email info, and thread context.
        """
        try:
            if thread_id:
                thread_emails = self._get_thread_emails(thread_id)
                thread_map = {thread_id: thread_emails} if thread_emails else {}
            else:
                thread_map = self._get_all_threads()

            pending_questions: List[Dict[str, Any]] = []

            for tid, emails_in_thread in thread_map.items():
                if not emails_in_thread:
                    continue

                # Collect all questions from all emails except the last one
                questions_asked: List[Tuple[str, Dict]] = []
                all_response_text = ""

                for i, email in enumerate(emails_in_thread):
                    body = clean_html(email.get("body", ""))
                    found_questions = _QUESTION_PATTERN.findall(body)

                    if i < len(emails_in_thread) - 1:
                        # Questions from earlier emails — check if answered in later emails
                        for q in found_questions:
                            q = q.strip()
                            if len(q) > 15 and not self._is_rhetorical(q):
                                questions_asked.append((q, email))
                    else:
                        # Last email in thread — questions here are definitely pending
                        for q in found_questions:
                            q = q.strip()
                            if len(q) > 15 and not self._is_rhetorical(q):
                                pending_questions.append({
                                    "question": q,
                                    "thread_id": tid,
                                    "asked_by": email.get("sender", ""),
                                    "asked_date": email.get("date"),
                                    "email_subject": email.get("subject", ""),
                                    "status": "pending",
                                })

                    # Accumulate response text for answer checking
                    if i > 0:
                        all_response_text += " " + body

                # Check earlier questions against all subsequent response text
                response_lower = all_response_text.lower()
                for question, source_email in questions_asked:
                    # Simple heuristic: if key nouns from question don't appear
                    # in responses, it's likely unanswered
                    q_words = set(re.findall(r"\b\w{4,}\b", question.lower()))
                    # Remove common words
                    q_words -= {"what", "when", "where", "which", "will", "would",
                                "could", "should", "have", "does", "this", "that",
                                "there", "their", "they", "about", "from", "with"}
                    if not q_words:
                        continue

                    answered_words = sum(1 for w in q_words if w in response_lower)
                    answer_ratio = answered_words / len(q_words) if q_words else 0

                    if answer_ratio < 0.4:
                        pending_questions.append({
                            "question": question,
                            "thread_id": tid,
                            "asked_by": source_email.get("sender", ""),
                            "asked_date": source_email.get("date"),
                            "email_subject": source_email.get("subject", ""),
                            "status": "likely_unanswered",
                        })

            # Sort by date descending
            pending_questions.sort(
                key=lambda x: str(x.get("asked_date", "")), reverse=True
            )
            return pending_questions

        except Exception as exc:
            logger.error("Error finding pending questions: %s", exc, exc_info=True)
            return []

    def get_unresolved_issues(self, thread_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find issues raised but not resolved.

        Args:
            thread_id: If provided, limit search to this thread.

        Returns:
            List of dicts with issue description, source, and status.
        """
        try:
            if thread_id:
                thread_emails = self._get_thread_emails(thread_id)
                thread_map = {thread_id: thread_emails} if thread_emails else {}
            else:
                thread_map = self._get_all_threads()

            unresolved: List[Dict[str, Any]] = []

            for tid, emails_in_thread in thread_map.items():
                if not emails_in_thread:
                    continue

                all_text = ""
                issues_found: List[Tuple[str, Dict]] = []

                for email in emails_in_thread:
                    body = clean_html(email.get("body", ""))
                    body_lower = body.lower()
                    all_text += " " + body

                    # Find sentences mentioning issues
                    sentences = re.split(r"[.!?\n]+", body)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if len(sentence) < 15:
                            continue
                        sentence_lower = sentence.lower()
                        for keyword in _ISSUE_KEYWORDS:
                            if keyword in sentence_lower:
                                issues_found.append((sentence, email))
                                break

                # Check which issues have been resolved
                all_text_lower = all_text.lower()
                for issue_text, source_email in issues_found:
                    # Check if resolution keywords appear after the issue
                    is_resolved = False
                    for res_kw in _RESOLUTION_KEYWORDS:
                        if res_kw in all_text_lower:
                            # Check if resolution appears after the issue mention
                            issue_pos = all_text_lower.find(issue_text.lower()[:30])
                            res_pos = all_text_lower.find(res_kw)
                            if res_pos > issue_pos:
                                is_resolved = True
                                break

                    if not is_resolved:
                        unresolved.append({
                            "issue": truncate_string(issue_text, 200),
                            "thread_id": tid,
                            "raised_by": source_email.get("sender", ""),
                            "raised_date": source_email.get("date"),
                            "email_subject": source_email.get("subject", ""),
                            "status": "unresolved",
                        })

            return unresolved

        except Exception as exc:
            logger.error("Error finding unresolved issues: %s", exc, exc_info=True)
            return []

    def track_decisions(self, thread_id: str) -> List[Dict[str, Any]]:
        """
        Extract decisions made in a conversation thread.

        Args:
            thread_id: Thread identifier.

        Returns:
            List of decision dicts with text, who decided, and when.
        """
        try:
            thread_emails = self._get_thread_emails(thread_id)
            if not thread_emails:
                return []

            decisions: List[Dict[str, Any]] = []
            for email in thread_emails:
                body = clean_html(email.get("body", ""))
                extracted = self._extract_decisions(body)
                for decision_text in extracted:
                    decisions.append({
                        "decision": decision_text,
                        "decided_by": email.get("sender", ""),
                        "date": email.get("date"),
                        "email_subject": email.get("subject", ""),
                        "thread_id": thread_id,
                    })

            return decisions

        except Exception as exc:
            logger.error("Error tracking decisions for thread %s: %s", thread_id, exc, exc_info=True)
            return []

    def get_thread_timeline(self, thread_id: str) -> List[Dict[str, Any]]:
        """
        Chronological timeline of a thread with key events.

        Args:
            thread_id: Thread identifier.

        Returns:
            Ordered list of timeline events.
        """
        try:
            thread_emails = self._get_thread_emails(thread_id)
            if not thread_emails:
                return []

            timeline: List[Dict[str, Any]] = []
            for i, email in enumerate(thread_emails):
                body = clean_html(email.get("body", ""))
                event_type = "message"

                # Determine event type
                if i == 0:
                    event_type = "thread_started"
                elif email.get("is_reply"):
                    event_type = "reply"

                # Check for special content
                body_lower = body.lower()
                if any(kw in body_lower for kw in _DECISION_KEYWORDS):
                    event_type = "decision"
                elif any(kw in body_lower for kw in _ISSUE_KEYWORDS):
                    event_type = "issue_raised"
                elif any(kw in body_lower for kw in _RESOLUTION_KEYWORDS):
                    event_type = "issue_resolved"

                # Check for attachments or forwarding
                subject = email.get("subject", "")
                if subject.lower().startswith("fwd:") or subject.lower().startswith("fw:"):
                    event_type = "forwarded"

                timeline.append({
                    "index": i,
                    "event_type": event_type,
                    "date": email.get("date"),
                    "sender": email.get("sender", ""),
                    "subject": subject,
                    "preview": truncate_string(body, 150),
                    "has_questions": bool(_QUESTION_PATTERN.search(body)),
                })

            return timeline

        except Exception as exc:
            logger.error("Error building timeline for thread %s: %s", thread_id, exc, exc_info=True)
            return []

    def detect_topic_changes(self, thread_id: str) -> List[Dict[str, Any]]:
        """
        Detect when the conversation topic shifts within a thread.

        Uses subject changes and, if available, embedding cosine similarity
        between consecutive messages.

        Args:
            thread_id: Thread identifier.

        Returns:
            List of topic change events.
        """
        try:
            thread_emails = self._get_thread_emails(thread_id)
            if not thread_emails or len(thread_emails) < 2:
                return []

            changes: List[Dict[str, Any]] = []
            prev_subject = self._normalize_subject(
                thread_emails[0].get("subject", "")
            )
            prev_body = clean_html(thread_emails[0].get("body", ""))

            for i in range(1, len(thread_emails)):
                curr_subject = self._normalize_subject(
                    thread_emails[i].get("subject", "")
                )
                curr_body = clean_html(thread_emails[i].get("body", ""))
                change_detected = False
                reason = ""

                # Subject change
                if curr_subject and prev_subject and curr_subject != prev_subject:
                    change_detected = True
                    reason = f"Subject changed from '{prev_subject}' to '{curr_subject}'"

                # Semantic similarity check
                if not change_detected and self.embedding_manager:
                    try:
                        prev_emb = self.embedding_manager.encode(
                            truncate_string(prev_body, 512)
                        )
                        curr_emb = self.embedding_manager.encode(
                            truncate_string(curr_body, 512)
                        )
                        sim = compute_cosine_similarity(prev_emb, curr_emb)
                        if sim < 0.4:
                            change_detected = True
                            reason = f"Content similarity dropped to {sim:.2f}"
                    except Exception:
                        pass

                if change_detected:
                    changes.append({
                        "email_index": i,
                        "date": thread_emails[i].get("date"),
                        "sender": thread_emails[i].get("sender", ""),
                        "new_topic": curr_subject or truncate_string(curr_body, 80),
                        "previous_topic": prev_subject or truncate_string(prev_body, 80),
                        "reason": reason,
                    })

                prev_subject = curr_subject
                prev_body = curr_body

            return changes

        except Exception as exc:
            logger.error("Error detecting topic changes in thread %s: %s", thread_id, exc, exc_info=True)
            return []

    def find_related_threads(self, thread_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find threads discussing similar topics using embeddings or keyword overlap.

        Args:
            thread_id: The reference thread.
            top_k: Number of related threads to return.

        Returns:
            List of related thread dicts with similarity scores.
        """
        try:
            ref_emails = self._get_thread_emails(thread_id)
            if not ref_emails:
                return []

            # Build reference text
            ref_text = " ".join(
                clean_html(e.get("body", "")) for e in ref_emails
            )
            ref_subject = self._normalize_subject(ref_emails[0].get("subject", ""))
            ref_keywords = set(re.findall(r"\b\w{5,}\b", ref_text.lower()))

            # Get all threads
            all_threads = self._get_all_threads()
            similarities: List[Tuple[str, float, Dict]] = []

            # Embedding-based similarity
            if self.embedding_manager:
                try:
                    ref_emb = self.embedding_manager.encode(
                        truncate_string(ref_text, 512)
                    )
                except Exception:
                    ref_emb = None
            else:
                ref_emb = None

            for tid, thread_emails in all_threads.items():
                if tid == thread_id or not thread_emails:
                    continue

                other_text = " ".join(
                    clean_html(e.get("body", "")) for e in thread_emails
                )

                sim_score = 0.0

                # Embedding similarity
                if ref_emb is not None and self.embedding_manager:
                    try:
                        other_emb = self.embedding_manager.encode(
                            truncate_string(other_text, 512)
                        )
                        sim_score = compute_cosine_similarity(ref_emb, other_emb)
                    except Exception:
                        pass

                # Keyword overlap fallback
                if sim_score == 0.0:
                    other_keywords = set(re.findall(r"\b\w{5,}\b", other_text.lower()))
                    if ref_keywords and other_keywords:
                        overlap = len(ref_keywords & other_keywords)
                        total = len(ref_keywords | other_keywords)
                        sim_score = overlap / total if total > 0 else 0.0

                if sim_score > 0.1:
                    other_subject = self._normalize_subject(
                        thread_emails[0].get("subject", "")
                    )
                    similarities.append((tid, sim_score, {
                        "thread_id": tid,
                        "subject": other_subject,
                        "email_count": len(thread_emails),
                        "similarity": round(sim_score, 3),
                    }))

            # Sort by similarity descending and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [item[2] for item in similarities[:top_k]]

        except Exception as exc:
            logger.error("Error finding related threads for %s: %s", thread_id, exc, exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_subject(self, subject: str) -> str:
        """Strip Re:/Fwd:/Fw: prefixes and normalise whitespace."""
        if not subject:
            return ""
        cleaned = re.sub(r"^(?:re|fwd?|fw)\s*:\s*", "", subject.strip(), flags=re.I)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _parse_email_date(self, date_val: Any) -> datetime:
        """Parse a date value into datetime, with fallback to epoch."""
        if isinstance(date_val, datetime):
            return date_val
        if isinstance(date_val, str):
            from src.utils.helpers import parse_date
            parsed = parse_date(date_val)
            if parsed:
                return parsed
        return datetime(1970, 1, 1)

    def _get_thread_emails(self, thread_id: str) -> List[Dict[str, Any]]:
        """Retrieve emails belonging to a thread from the database."""
        try:
            from sqlalchemy import select
            from src.database.models import Email

            with self.db.get_session() as session:
                stmt = (
                    select(Email)
                    .where(Email.thread_id == thread_id)
                    .order_by(Email.date.asc())
                )
                emails = session.execute(stmt).scalars().all()
                return [self._email_to_dict(e) for e in emails]
        except Exception as exc:
            logger.warning("Failed to get thread emails for %s: %s", thread_id, exc)
            return []

    def _get_all_threads(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all threads from the database, grouped by thread_id."""
        try:
            from sqlalchemy import select
            from src.database.models import Email

            with self.db.get_session() as session:
                stmt = (
                    select(Email)
                    .where(Email.thread_id.isnot(None))
                    .order_by(Email.thread_id, Email.date.asc())
                )
                emails = session.execute(stmt).scalars().all()

                threads: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                for e in emails:
                    threads[e.thread_id].append(self._email_to_dict(e))
                return dict(threads)
        except Exception as exc:
            logger.warning("Failed to get all threads: %s", exc)
            return {}

    def _email_to_dict(self, email) -> Dict[str, Any]:
        """Convert an Email ORM object to a dict."""
        return {
            "id": email.id,
            "message_id": email.message_id,
            "subject": email.subject,
            "sender": email.sender,
            "recipients": email.recipients,
            "cc": email.cc,
            "date": email.date,
            "body": email.body,
            "thread_id": email.thread_id,
            "in_reply_to": email.in_reply_to,
            "references_field": email.references_field,
            "is_reply": email.is_reply,
        }

    def _persist_thread(self, thread_id: str, emails: List[Dict[str, Any]]) -> None:
        """Persist a thread record and update email thread_id references."""
        try:
            if not emails:
                return

            participants: Set[str] = set()
            dates: List[datetime] = []
            subjects: List[str] = []

            for email in emails:
                sender = normalize_email(email.get("sender", ""))
                if sender:
                    participants.add(sender)
                recipients = email.get("recipients", [])
                if isinstance(recipients, str):
                    recipients = safe_json_loads(recipients, [])
                for r in recipients:
                    nr = normalize_email(r)
                    if nr:
                        participants.add(nr)
                dt = self._parse_email_date(email.get("date"))
                if dt.year > 1970:
                    dates.append(dt)
                subj = email.get("subject", "")
                if subj:
                    subjects.append(subj)

            self.db.upsert_thread({
                "thread_id": thread_id,
                "subject": subjects[0] if subjects else "",
                "participants": sorted(participants),
                "email_count": len(emails),
                "first_date": min(dates) if dates else None,
                "last_date": max(dates) if dates else None,
                "status": "active",
            })

            # Update email records with thread_id
            from sqlalchemy import update
            from src.database.models import Email

            email_ids = [e.get("id") for e in emails if e.get("id")]
            if email_ids:
                with self.db.get_session() as session:
                    session.execute(
                        update(Email)
                        .where(Email.id.in_(email_ids))
                        .values(thread_id=thread_id)
                    )

        except Exception as exc:
            logger.warning("Failed to persist thread %s: %s", thread_id, exc)

    def _semantic_link(
        self,
        emails: List[Dict[str, Any]],
        by_msg_id: Dict[str, Dict[str, Any]],
        parent: Dict[str, str],
        find,
        union,
    ) -> None:
        """Link semantically similar emails that aren't yet in the same thread."""
        if not self.embedding_manager:
            return

        # Compute embeddings for all emails
        texts = []
        msg_ids = []
        for email in emails:
            mid = email.get("message_id", "")
            if not mid:
                continue
            text = (email.get("subject", "") + " " +
                    clean_html(email.get("body", "")))
            texts.append(truncate_string(text, 512))
            msg_ids.append(mid)

        if len(texts) < 2:
            return

        embeddings = []
        for t in texts:
            emb = self.embedding_manager.encode(t)
            embeddings.append(emb)

        embeddings_arr = np.array(embeddings)

        # Compare pairs that aren't already in the same group
        for i in range(len(msg_ids)):
            for j in range(i + 1, len(msg_ids)):
                if find(msg_ids[i]) == find(msg_ids[j]):
                    continue
                sim = compute_cosine_similarity(embeddings_arr[i], embeddings_arr[j])
                if sim >= SIMILARITY_THRESHOLD:
                    union(msg_ids[i], msg_ids[j])

    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from text — sentences with important keywords."""
        key_indicators = [
            "important", "key", "critical", "note", "highlight",
            "update", "decision", "agreed", "action", "deadline",
            "milestone", "deliverable", "blocker", "resolved",
        ]
        sentences = re.split(r"[.!?\n]+", text)
        key_points: List[str] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue
            sentence_lower = sentence.lower()
            if any(kw in sentence_lower for kw in key_indicators):
                key_points.append(truncate_string(sentence, 200))
        return key_points

    def _extract_decisions(self, text: str) -> List[str]:
        """Extract decision statements from text."""
        sentences = re.split(r"[.!?\n]+", text)
        decisions: List[str] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue
            sentence_lower = sentence.lower()
            if any(kw in sentence_lower for kw in _DECISION_KEYWORDS):
                decisions.append(truncate_string(sentence, 200))
        return decisions

    def _find_pending_in_text(self, text: str) -> List[str]:
        """Find pending items in text."""
        pending_keywords = [
            "pending", "awaiting", "waiting for", "to be done",
            "still need", "not yet", "follow up", "follow-up",
            "todo", "to do", "action item", "next step",
        ]
        sentences = re.split(r"[.!?\n]+", text)
        pending: List[str] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue
            sentence_lower = sentence.lower()
            if any(kw in sentence_lower for kw in pending_keywords):
                pending.append(truncate_string(sentence, 200))
        return pending

    def _is_rhetorical(self, question: str) -> bool:
        """Heuristic check for rhetorical questions."""
        rhetorical_starts = [
            "how are you", "how's it going", "hope you",
            "isn't it", "don't you think", "wouldn't you",
            "right?", "isn't that", "doesn't it",
        ]
        q_lower = question.lower().strip()
        return any(q_lower.startswith(r) or q_lower.endswith(r) for r in rhetorical_starts)
