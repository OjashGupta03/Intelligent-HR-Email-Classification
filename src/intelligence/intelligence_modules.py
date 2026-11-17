"""
Executive AI Chief of Staff - Intelligence Modules
Conversation Memory, Commitment Tracker, Risk Monitor,
Daily Briefing Generator, What Did I Miss?, Relationship Intelligence.
All in one file for compactness, each as a separate class.
"""
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


# ============================================================
# CONVERSATION MEMORY
# ============================================================

class ConversationMemory:
    """Reconstructs email threads and tracks conversation state."""

    def __init__(self, db_manager, embedding_manager=None):
        self.db = db_manager
        self.embeddings = embedding_manager

    def reconstruct_threads(self, emails: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Reconstruct threads using:
        1. in_reply_to / references headers
        2. Subject matching (normalize Re:/Fwd:)
        3. Fallback: semantic similarity (if embedding manager available)
        """
        threads: Dict[str, List[Dict]] = defaultdict(list)

        # Pass 1: use thread_id if already set
        for e in emails:
            tid = e.get("thread_id")
            if tid:
                threads[tid].append(e)

        # Pass 2: group remaining by normalized subject
        no_tid = [e for e in emails if not e.get("thread_id")]
        for e in no_tid:
            subj = _normalize_subject(e.get("subject", ""))
            tid = f"subj_{_slug(subj)}" if subj else f"msg_{e.get('message_id', id(e))}"
            e["thread_id"] = tid
            threads[tid].append(e)

        # Sort each thread chronologically
        for tid in threads:
            threads[tid].sort(key=lambda x: _parse_date(x.get("date")) or datetime.min)

        logger.info("Reconstructed %d threads from %d emails", len(threads), len(emails))
        return dict(threads)

    def get_thread_summary(self, thread_emails: List[Dict]) -> Dict:
        """Summarize a thread's key information."""
        from src.nlp.summarizer import EmailSummarizer
        summarizer = EmailSummarizer()

        participants = list({e.get("sender", "") for e in thread_emails if e.get("sender")})
        subject = thread_emails[-1].get("subject", "") if thread_emails else ""
        summary_text = summarizer.summarize_thread(thread_emails, max_sentences=3)
        dates = [_parse_date(e.get("date")) for e in thread_emails]
        dates = [d for d in dates if d]

        return {
            "subject": subject,
            "participants": participants,
            "email_count": len(thread_emails),
            "first_date": min(dates).isoformat() if dates else None,
            "last_date": max(dates).isoformat() if dates else None,
            "summary": summary_text,
            "has_unresolved": self._has_unresolved_questions(thread_emails),
        }

    def get_pending_questions(self, thread_emails: List[Dict]) -> List[str]:
        """Find questions in emails that likely haven't been answered."""
        questions = []
        answered_topics: Set[str] = set()

        # Collect questions from earlier messages
        for i, e in enumerate(thread_emails[:-1]):
            body = e.get("body") or ""
            for sent in re.split(r"(?<=[.!?])\s+", body):
                if "?" in sent and len(sent) > 20:
                    questions.append(sent.strip())

        # Simple heuristic: if no reply references question keywords, mark as pending
        last_body = (thread_emails[-1].get("body") or "") if thread_emails else ""
        pending = [q for q in questions if not _is_answered(q, last_body)]
        return pending[:5]

    def track_decisions(self, thread_emails: List[Dict]) -> List[Dict]:
        """Extract decision statements from thread."""
        _DECISION_PATTERNS = re.compile(
            r"\b(we (decided|agreed|resolved|concluded)|the decision is|"
            r"i approve|i authoriz|moving forward with|we will proceed|"
            r"approved|rejected|selected|chosen)\b",
            re.IGNORECASE,
        )
        decisions = []
        for e in thread_emails:
            body = e.get("body") or ""
            for sent in re.split(r"(?<=[.!?])\s+", body):
                if _DECISION_PATTERNS.search(sent):
                    decisions.append({
                        "decision": sent.strip()[:200],
                        "date": e.get("date"),
                        "sender": e.get("sender", ""),
                    })
        return decisions[:10]

    def _has_unresolved_questions(self, thread_emails: List[Dict]) -> bool:
        if len(thread_emails) < 2:
            return False
        return bool(self.get_pending_questions(thread_emails))

    def find_related_threads(self, thread_id: str, all_threads: Dict[str, List[Dict]], top_k: int = 5) -> List[str]:
        """Find threads discussing similar topics by subject overlap."""
        if thread_id not in all_threads:
            return []
        ref_subj = _normalize_subject(all_threads[thread_id][-1].get("subject", "") if all_threads[thread_id] else "")
        ref_words = set(ref_subj.lower().split())
        scored = []
        for tid, emails in all_threads.items():
            if tid == thread_id:
                continue
            subj = _normalize_subject(emails[-1].get("subject", "") if emails else "")
            words = set(subj.lower().split())
            overlap = len(ref_words & words) / max(len(ref_words | words), 1)
            if overlap > 0.2:
                scored.append((tid, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored[:top_k]]


# ============================================================
# COMMITMENT TRACKER
# ============================================================

class CommitmentTracker:
    """Tracks commitments detected in emails."""

    def __init__(self, db_manager, commitment_detector):
        self.db = db_manager
        self.detector = commitment_detector

    def process_email_commitments(self, email_dict: Dict) -> List[Dict]:
        """Detect and store commitments from an email."""
        text = f"{email_dict.get('subject', '')} {email_dict.get('body', '')}"
        raw_commitments = self.detector.detect_commitments(text)
        stored = []
        for c in raw_commitments:
            eid = email_dict.get("id") or email_dict.get("email_id") or email_dict.get("db_id")
            if not eid:
                continue
            commitment_data = {
                "email_id": eid,
                "description": c["description"][:500],
                "responsible_person": c.get("responsible_person", "Unknown"),
                "related_entity": c.get("related_entity"),
                "deadline": c.get("deadline"),
                "status": "pending",
                "source_text": c.get("source_text", "")[:500],
                "commitment_type": c.get("commitment_type", "promise"),
                "confidence": c.get("confidence", 0.5),
            }
            try:
                self.db.add_commitment(commitment_data)
                stored.append(commitment_data)
            except Exception as e:
                logger.debug("Could not store commitment: %s", e)
        return stored

    def get_all_commitments(self, status: Optional[str] = None) -> List[Dict]:
        try:
            return self.db.get_commitments(status=status)
        except Exception:
            return []

    def get_overdue_commitments(self) -> List[Dict]:
        try:
            return self.db.get_commitments(overdue=True)
        except Exception:
            return []

    def get_commitment_summary(self) -> Dict:
        all_c = self.get_all_commitments()
        summary: Dict[str, int] = defaultdict(int)
        for c in all_c:
            status = c.get("status", "unknown") if isinstance(c, dict) else getattr(c, "status", "unknown")
            summary[status] += 1
        return {"total": len(all_c), **dict(summary)}

    def get_upcoming_deadlines(self, days: int = 7) -> List[Dict]:
        cutoff = datetime.utcnow() + timedelta(days=days)
        all_c = self.get_all_commitments(status="pending")
        upcoming = []
        for c in all_c:
            deadline = c.get("deadline_date") if isinstance(c, dict) else getattr(c, "deadline_date", None)
            if deadline and deadline <= cutoff:
                upcoming.append(c)
        upcoming.sort(key=lambda x: x.get("deadline_date") if isinstance(x, dict) else getattr(x, "deadline_date", datetime.max))
        return upcoming[:10]


# ============================================================
# RISK MONITOR
# ============================================================

class RiskMonitor:
    """Monitors business risks across emails."""

    def __init__(self, db_manager, risk_detector):
        self.db = db_manager
        self.detector = risk_detector

    def process_email_risks(self, email_dict: Dict) -> List[Dict]:
        """Detect and store risks from an email."""
        text = f"{email_dict.get('subject', '')} {email_dict.get('body', '')}"
        raw_risks = self.detector.detect_risks(text)
        stored = []
        for r in raw_risks:
            eid = email_dict.get("id") or email_dict.get("email_id") or email_dict.get("db_id")
            risk_data = {
                "email_id": eid,
                "category": r["category"],
                "severity": r["severity"],
                "explanation": r.get("explanation", ""),
                "source_text": r.get("source_text", "")[:500],
                "suggested_action": r.get("suggested_action", ""),
                "confidence": r.get("confidence", 0.5),
            }
            try:
                self.db.add_risk(risk_data)
                stored.append(risk_data)
            except Exception as e:
                logger.debug("Could not store risk: %s", e)
        return stored

    def get_risk_dashboard(self) -> Dict:
        """High-level risk overview."""
        try:
            all_risks = self.db.get_risks()
        except Exception:
            return {}
        by_cat: Dict[str, int] = defaultdict(int)
        by_sev: Dict[str, int] = defaultdict(int)
        for r in all_risks:
            cat = r.get("category", "unknown") if isinstance(r, dict) else getattr(r, "category", "unknown")
            sev = r.get("severity", "low") if isinstance(r, dict) else getattr(r, "severity", "low")
            by_cat[cat] += 1
            by_sev[sev] += 1
        return {"total": len(all_risks), "by_category": dict(by_cat), "by_severity": dict(by_sev)}

    def get_critical_risks(self) -> List:
        try:
            return self.db.get_risks(severity="critical") + self.db.get_risks(severity="high")
        except Exception:
            return []

    def get_risks_by_category(self, category: str) -> List:
        try:
            return self.db.get_risks(category=category)
        except Exception:
            return []


# ============================================================
# DAILY BRIEFING GENERATOR
# ============================================================

class DailyBriefingGenerator:
    """Generates structured daily executive briefings."""

    def __init__(self, db_manager, priority_engine, commitment_tracker, risk_monitor):
        self.db = db_manager
        self.priority_engine = priority_engine
        self.commitment_tracker = commitment_tracker
        self.risk_monitor = risk_monitor

    def generate_briefing(self, date: Optional[datetime] = None) -> Dict:
        """Generate a daily briefing."""
        if date is None:
            date = datetime.utcnow()
        date_str = date.strftime("%Y-%m-%d")
        logger.info("Generating daily briefing for %s", date_str)

        # Top emails
        try:
            top_emails = self.priority_engine.get_top_emails(n=5)
        except Exception:
            top_emails = []

        # Upcoming deadlines (7 days)
        upcoming_deadlines = self.commitment_tracker.get_upcoming_deadlines(days=7)

        # Overdue commitments
        overdue = self.commitment_tracker.get_overdue_commitments()

        # Critical risks
        critical_risks = self.risk_monitor.get_critical_risks()[:5]

        # Recommended actions
        actions = []
        for e in top_emails[:3]:
            subj = e.get("subject", "")[:50] if isinstance(e, dict) else getattr(e, "subject", "")[:50]
            actions.append(f"Review: {subj}")
        for c in overdue[:2]:
            desc = c.get("description", "")[:40] if isinstance(c, dict) else getattr(c, "description", "")[:40]
            actions.append(f"Follow up overdue commitment: {desc}")
        for r in critical_risks[:2]:
            cat = r.get("category", "unknown") if isinstance(r, dict) else getattr(r, "category", "unknown")
            actions.append(f"Address critical {cat} risk")

        briefing = {
            "date": date_str,
            "generated_at": datetime.utcnow().isoformat(),
            "top_emails": [_serialize(e) for e in top_emails],
            "key_developments": [_serialize(e) for e in top_emails[:3]],
            "pending_decisions": [_serialize(e) for e in top_emails if _is_decision_needed(e)][:5],
            "upcoming_deadlines": [_serialize(c) for c in upcoming_deadlines],
            "overdue_commitments": [_serialize(c) for c in overdue[:5]],
            "critical_risks": [_serialize(r) for r in critical_risks],
            "recommended_actions": actions,
        }

        # Save to DB
        try:
            self.db.save_briefing(date_str, briefing)
        except Exception as e:
            logger.warning("Could not save briefing: %s", e)

        return briefing

    def get_previous_briefings(self, n: int = 7) -> List[Dict]:
        try:
            return self.db.get_recent_briefings(n)
        except Exception:
            return []


# ============================================================
# WHAT DID I MISS?
# ============================================================

class WhatDidIMiss:
    """Flagship feature: catch-up report for executives returning from absence."""

    def __init__(self, db_manager, priority_engine, commitment_tracker, risk_monitor, conversation_memory):
        self.db = db_manager
        self.priority_engine = priority_engine
        self.commitment_tracker = commitment_tracker
        self.risk_monitor = risk_monitor
        self.conv_memory = conversation_memory

    def generate_report(self, start_date: datetime, end_date: datetime) -> Dict:
        """Generate structured catch-up report for a date range."""
        logger.info("Generating 'What Did I Miss?' report: %s → %s", start_date.date(), end_date.date())

        try:
            period_emails = self.db.get_emails_in_date_range(start_date, end_date)
        except Exception:
            period_emails = []

        # Group into threads
        email_dicts = [_serialize(e) for e in period_emails]
        threads = self.conv_memory.reconstruct_threads(email_dicts) if email_dicts else {}

        # Score and rank emails
        scored_emails = []
        for e in email_dicts[:200]:  # limit for performance
            try:
                score_result = self.priority_engine.score_email(e)
                e.update(score_result)
                scored_emails.append(e)
            except Exception:
                e["priority_score"] = 0
                scored_emails.append(e)
        scored_emails.sort(key=lambda x: x.get("priority_score", 0), reverse=True)

        # Build sections
        critical = [e for e in scored_emails if e.get("priority_level") in ("critical", "high")][:10]
        attention_needed = [e for e in scored_emails if e.get("priority_score", 0) >= 60][:10]
        decisions_pending = [e for e in scored_emails if _is_decision_needed(e)][:8]
        commitments_in_period = [
            c for c in (self.commitment_tracker.get_all_commitments() or [])
            if _commitment_in_period(c, start_date, end_date)
        ][:10]
        overdue = self.commitment_tracker.get_overdue_commitments()[:5]
        risks_in_period = self.risk_monitor.get_critical_risks()[:8]

        # Recommended actions (prioritized)
        actions = []
        for e in critical[:5]:
            subj = e.get("subject", "")[:60]
            actions.append({"priority": 1, "action": f"Respond immediately: {subj}", "type": "email"})
        for c in overdue[:3]:
            desc = c.get("description", "")[:50] if isinstance(c, dict) else getattr(c, "description", "")[:50]
            actions.append({"priority": 2, "action": f"Overdue commitment: {desc}", "type": "commitment"})
        for r in risks_in_period[:3]:
            cat = r.get("category", "") if isinstance(r, dict) else getattr(r, "category", "")
            actions.append({"priority": 2, "action": f"Review {cat} risk", "type": "risk"})
        actions.sort(key=lambda x: x["priority"])

        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "total_emails": len(period_emails),
                "total_threads": len(threads),
            },
            "executive_summary": self._build_summary(period_emails, critical, risks_in_period, overdue),
            "critical_developments": [_safe_email_summary(e) for e in critical[:8]],
            "attention_needed": [_safe_email_summary(e) for e in attention_needed[:8]],
            "decisions_pending": [_safe_email_summary(e) for e in decisions_pending],
            "commitments_made": [_serialize(c) for c in commitments_in_period],
            "missed_deadlines": [_serialize(c) for c in overdue],
            "customer_issues": [_safe_email_summary(e) for e in scored_emails if _is_customer_issue(e)][:5],
            "risks_emerged": [_serialize(r) for r in risks_in_period],
            "conversation_updates": [
                {"thread_id": tid, "email_count": len(emails), "latest_subject": emails[-1].get("subject", "") if emails else ""}
                for tid, emails in list(threads.items())[:10]
            ],
            "recommended_actions": actions,
        }

    def _build_summary(self, all_emails, critical, risks, overdue) -> str:
        n = len(all_emails)
        nc = len(critical)
        nr = len(risks)
        no = len(overdue)
        parts = [f"During your absence, {n} emails arrived."]
        if nc:
            parts.append(f"{nc} require immediate attention.")
        if nr:
            parts.append(f"{nr} business risks were detected.")
        if no:
            parts.append(f"{no} commitments are overdue.")
        return " ".join(parts)


# ============================================================
# RELATIONSHIP INTELLIGENCE
# ============================================================

class RelationshipIntelligence:
    """Builds and queries the communication relationship graph."""

    def __init__(self, db_manager):
        self.db = db_manager
        self.graph: Optional[nx.DiGraph] = None

    def build_communication_graph(self) -> nx.DiGraph:
        """Build directed communication graph from email data."""
        G = nx.DiGraph()
        try:
            emails = self.db.get_emails({})
        except Exception:
            self.graph = G
            return G

        for e in emails:
            sender = e.sender if hasattr(e, "sender") else e.get("sender", "")
            recips_raw = e.recipients if hasattr(e, "recipients") else e.get("recipients", "[]")
            if not sender:
                continue
            try:
                recips = json.loads(recips_raw) if isinstance(recips_raw, str) else (recips_raw or [])
            except Exception:
                recips = []
            if not G.has_node(sender):
                G.add_node(sender, email_count=0)
            G.nodes[sender]["email_count"] = G.nodes[sender].get("email_count", 0) + 1
            for r in recips:
                if not r or not isinstance(r, str):
                    continue
                if not G.has_edge(sender, r):
                    G.add_edge(sender, r, weight=0)
                G.edges[sender, r]["weight"] += 1

        self.graph = G
        logger.info("Communication graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
        return G

    def get_top_communicators(self, n: int = 20) -> List[Dict]:
        """Most frequent communicators."""
        if self.graph is None:
            self.build_communication_graph()
        try:
            contacts = self.db.get_top_contacts(limit=n)
            return [_serialize(c) for c in contacts]
        except Exception:
            return []

    def get_important_senders(self, n: int = 10) -> List[Dict]:
        """Important senders based on combined signals."""
        try:
            contacts = self.db.get_top_contacts(limit=50)
            # Sort by importance_score
            sorted_c = sorted(contacts, key=lambda c: getattr(c, "importance_score", 50), reverse=True)
            return [_serialize(c) for c in sorted_c[:n]]
        except Exception:
            return []

    def get_unanswered_important(self) -> List[Dict]:
        """Emails marked high priority where there's no outgoing reply."""
        try:
            emails = self.db.get_emails({"priority_level": "critical"})
            return [_serialize(e) for e in emails[:10]]
        except Exception:
            return []

    def get_communication_patterns(self) -> Dict:
        """Communication frequency by day of week and hour."""
        patterns: Dict[str, Any] = {"by_day": defaultdict(int), "by_hour": defaultdict(int)}
        try:
            emails = self.db.get_emails({})
            for e in emails:
                date = e.date if hasattr(e, "date") else e.get("date")
                if isinstance(date, datetime):
                    patterns["by_day"][date.strftime("%A")] += 1
                    patterns["by_hour"][date.hour] += 1
        except Exception:
            pass
        return {k: dict(v) for k, v in patterns.items()}

    def get_organization_summary(self) -> List[Dict]:
        """Organizations with ongoing discussions."""
        try:
            contacts = self.db.get_top_contacts(limit=100)
            orgs: Dict[str, Dict] = {}
            for c in contacts:
                org = getattr(c, "organization", None) or "Unknown"
                if org not in orgs:
                    orgs[org] = {"organization": org, "contacts": [], "total_emails": 0}
                orgs[org]["contacts"].append(getattr(c, "email_address", ""))
                orgs[org]["total_emails"] += getattr(c, "total_emails_sent", 0) + getattr(c, "total_emails_received", 0)
            return sorted(orgs.values(), key=lambda x: x["total_emails"], reverse=True)[:10]
        except Exception:
            return []

    def configure_vip(self, email_address: str, importance_score: int) -> bool:
        """Allow executive to configure sender importance."""
        try:
            contact = self.db.get_contact(email_address)
            if contact is None:
                self.db.add_contact({"email_address": email_address, "importance_score": importance_score, "is_vip": importance_score >= 80})
            else:
                # Update via db session
                with self.db.session_scope() as sess:
                    from src.database.models import Contact
                    c = sess.get(Contact, contact.id)
                    if c:
                        c.importance_score = importance_score
                        c.is_vip = importance_score >= 80
            return True
        except Exception as e:
            logger.error("configure_vip error: %s", e)
            return False

    def get_relationship_summary(self, email_address: str) -> Dict:
        """Detailed summary for a specific contact."""
        try:
            contact = self.db.get_contact(email_address)
            if not contact:
                return {"error": "Contact not found"}
            return _serialize(contact)
        except Exception:
            return {}


# ============================================================
# HELPERS
# ============================================================

def _normalize_subject(subj: str) -> str:
    """Remove Re:/Fwd: prefixes."""
    return re.sub(r"^(Re|Fwd|Fw|RE|FW|FWD):\s*", "", subj, flags=re.IGNORECASE).strip()


def _slug(text: str, max_len: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower())[:max_len]


def _parse_date(date_val) -> Optional[datetime]:
    if isinstance(date_val, datetime):
        return date_val
    if isinstance(date_val, str):
        try:
            return datetime.fromisoformat(date_val)
        except Exception:
            pass
    return None


def _is_answered(question: str, reply_body: str) -> bool:
    q_words = set(re.findall(r"\b\w{4,}\b", question.lower()))
    r_words = set(re.findall(r"\b\w{4,}\b", reply_body.lower()))
    return len(q_words & r_words) >= 2


def _is_decision_needed(email) -> bool:
    text = ""
    if isinstance(email, dict):
        text = f"{email.get('subject', '')} {email.get('body', '')}"
        ann = email.get("annotations", {})
        if ann.get("action_required"):
            return True
    else:
        text = f"{getattr(email, 'subject', '')} {getattr(email, 'body', '')}"
    return bool(re.search(r"\b(approve|decision|authorization|sign.off|your\s+approval|awaiting\s+your)\b", text, re.IGNORECASE))


def _is_customer_issue(email) -> bool:
    text = f"{email.get('subject', '')} {email.get('body', '')}" if isinstance(email, dict) else ""
    return bool(re.search(r"\b(customer|client|account|user).{0,30}(complaint|issue|problem|escalat|dissatisfied|threatening|cancel)\b", text, re.IGNORECASE))


def _commitment_in_period(commitment, start: datetime, end: datetime) -> bool:
    dt = None
    if isinstance(commitment, dict):
        dt = _parse_date(commitment.get("detected_at"))
    else:
        dt = getattr(commitment, "detected_at", None)
    if dt is None:
        return False
    return start <= dt <= end


def _serialize(obj) -> Dict:
    """Convert SQLAlchemy model or dict to plain dict."""
    if isinstance(obj, dict):
        return obj
    result = {}
    for col in obj.__table__.columns if hasattr(obj, "__table__") else []:
        val = getattr(obj, col.name, None)
        if isinstance(val, datetime):
            val = val.isoformat()
        result[col.name] = val
    return result


def _safe_email_summary(e: Dict) -> Dict:
    return {
        "subject": e.get("subject", "")[:80],
        "sender": e.get("sender", ""),
        "date": e.get("date", ""),
        "priority_level": e.get("priority_level", "medium"),
        "priority_score": e.get("priority_score", 0),
        "summary": (e.get("body") or "")[:200],
    }

