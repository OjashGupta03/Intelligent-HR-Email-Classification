"""
Executive AI Chief of Staff - Commitment Tracker
Detects, stores, and monitors commitments and promises found in emails.
Tracks status, deadlines, responsible parties, and completion evidence.
"""

import json
import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import COMMITMENT_STATUSES
from src.utils.helpers import (
    clean_html,
    normalize_email,
    safe_json_loads,
    truncate_string,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Commitment detection patterns
# ---------------------------------------------------------------------------

_COMMITMENT_PATTERNS: List[re.Pattern] = [
    # "I will ..." / "I'll ..."
    re.compile(r"(?:I|we)\s+will\s+(.{10,80}?)(?:\.|$)", re.I | re.M),
    re.compile(r"(?:I|we)'ll\s+(.{10,80}?)(?:\.|$)", re.I | re.M),
    # "I promise ..."
    re.compile(r"(?:I|we)\s+(?:promise|commit|guarantee)\s+(?:to\s+)?(.{10,80}?)(?:\.|$)", re.I | re.M),
    # "I am going to ..."
    re.compile(r"(?:I|we)\s+(?:am|are)\s+going\s+to\s+(.{10,80}?)(?:\.|$)", re.I | re.M),
    # "Let me ... by ..."
    re.compile(r"let\s+me\s+(.{10,80}?)(?:\.|$)", re.I | re.M),
    # "I'll take care of ..."
    re.compile(r"(?:I|we)'ll\s+take\s+care\s+of\s+(.{10,80}?)(?:\.|$)", re.I | re.M),
    # "I'll make sure ..."
    re.compile(r"(?:I|we)'ll\s+(?:make sure|ensure)\s+(.{10,80}?)(?:\.|$)", re.I | re.M),
    # "You can expect ..."
    re.compile(r"you\s+can\s+expect\s+(.{10,80}?)(?:\.|$)", re.I | re.M),
    # "I'll get back to you ..."
    re.compile(r"(?:I|we)'ll\s+get\s+back\s+to\s+you\s+(.{10,60}?)(?:\.|$)", re.I | re.M),
    # "I'll send/share/provide ..."
    re.compile(r"(?:I|we)'ll\s+(?:send|share|provide|deliver|submit|forward|prepare)\s+(.{10,80}?)(?:\.|$)", re.I | re.M),
    # Deadline commitment: "by Friday / by end of day / by next week"
    re.compile(r"(?:I|we)\s+will\s+(.{5,60}?)\s+by\s+(.{3,30}?)(?:\.|$)", re.I | re.M),
]

_DEADLINE_EXTRACTION_PATTERNS = [
    re.compile(r"by\s+(today|tonight|tomorrow|end\s+of\s+(?:day|week|month)|EOD|COB|close\s+of\s+business)", re.I),
    re.compile(r"by\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)", re.I),
    re.compile(r"by\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?)", re.I),
    re.compile(r"by\s+(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)", re.I),
    re.compile(r"(?:within|in)\s+(\d+)\s+(?:hours?|days?|weeks?)", re.I),
    re.compile(r"(?:before|no later than)\s+(.{5,30}?)(?:\.|,|$)", re.I),
]

_COMPLETION_KEYWORDS = [
    "done", "completed", "finished", "delivered", "submitted",
    "sent", "shared", "provided", "attached", "here is",
    "here are", "as requested", "as promised", "please find",
    "i've completed", "we've completed", "task is done",
    "this is ready", "all set", "taken care of",
]


class CommitmentTracker:
    """
    Detects, stores, and monitors commitments found in emails.

    Works with a CommitmentDetector (or uses built-in pattern matching)
    to identify promises, deadlines, and responsible parties.  Tracks
    status over time and checks for completion evidence in subsequent
    emails.
    """

    def __init__(self, db_manager, commitment_detector=None) -> None:
        """
        Initialise CommitmentTracker.

        Args:
            db_manager: DatabaseManager instance.
            commitment_detector: Optional NLP commitment detector.
                                 If None, built-in patterns are used.
        """
        self.db = db_manager
        self.detector = commitment_detector
        logger.info("CommitmentTracker initialised (external_detector=%s)",
                     "yes" if commitment_detector else "no")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_email_commitments(self, email_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect and store commitments from an email.

        Args:
            email_dict: Email dict with at minimum body, sender, date, and id.

        Returns:
            List of detected commitment dicts persisted to the database.
        """
        try:
            body = clean_html(email_dict.get("body", ""))
            sender = email_dict.get("sender", "")
            email_id = email_dict.get("id") or email_dict.get("email_id")
            email_date = email_dict.get("date")

            if not body or not email_id:
                return []

            # Detect commitments (external detector or built-in)
            raw_commitments = self._detect_commitments(body, sender)

            # Deduplicate
            seen_descriptions: set = set()
            unique: List[Dict[str, Any]] = []
            for c in raw_commitments:
                desc_key = c["description"].lower().strip()[:80]
                if desc_key not in seen_descriptions:
                    seen_descriptions.add(desc_key)
                    unique.append(c)

            # Persist each commitment
            persisted: List[Dict[str, Any]] = []
            for commitment_data in unique:
                deadline = commitment_data.get("deadline")
                if isinstance(deadline, str):
                    deadline = self._parse_deadline_text(deadline, email_date)

                record = {
                    "email_id": email_id,
                    "description": commitment_data["description"],
                    "responsible_person": commitment_data.get("responsible_person", sender),
                    "related_entity": commitment_data.get("related_entity"),
                    "deadline": deadline,
                    "status": "pending",
                    "source_text": commitment_data.get("source_text", ""),
                    "confidence": commitment_data.get("confidence", 0.7),
                }

                try:
                    db_record = self.db.add_commitment(record)
                    record["id"] = db_record.id
                    persisted.append(record)
                except Exception as exc:
                    logger.warning("Failed to persist commitment: %s", exc)

            logger.info(
                "Detected %d commitments in email %s",
                len(persisted), email_id,
            )
            return persisted

        except Exception as exc:
            logger.error("Error processing commitments: %s", exc, exc_info=True)
            return []

    def get_all_commitments(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get commitments filtered by status.

        Args:
            status: Optional status filter (pending/completed/overdue/uncertain).

        Returns:
            List of commitment dicts.
        """
        try:
            commitments = self.db.get_commitments(status=status)
            return [self._commitment_to_dict(c) for c in commitments]
        except Exception as exc:
            logger.error("Error getting commitments: %s", exc, exc_info=True)
            return []

    def get_overdue_commitments(self) -> List[Dict[str, Any]]:
        """
        Get commitments past their deadline.

        Returns:
            List of overdue commitment dicts.
        """
        try:
            commitments = self.db.get_commitments(overdue=True)
            result = [self._commitment_to_dict(c) for c in commitments]

            # Also update their status to 'overdue' if still 'pending'
            for c in commitments:
                if c.status == "pending":
                    try:
                        self.db.update_commitment_status(c.id, "overdue")
                    except Exception:
                        pass

            return result
        except Exception as exc:
            logger.error("Error getting overdue commitments: %s", exc, exc_info=True)
            return []

    def update_commitment_status(
        self,
        commitment_id: int,
        new_status: str,
        evidence: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update a commitment's status with optional supporting evidence.

        Args:
            commitment_id: Database ID of the commitment.
            new_status: New status string.
            evidence: Optional evidence text supporting the status change.

        Returns:
            Updated commitment dict.
        """
        try:
            if new_status not in COMMITMENT_STATUSES:
                logger.warning("Invalid status '%s', defaulting to 'pending'", new_status)
                new_status = "pending"

            self.db.update_commitment_status(commitment_id, new_status)

            # If evidence provided, update the source_text
            if evidence:
                from sqlalchemy import select, update
                from src.database.models import Commitment

                with self.db.get_session() as session:
                    c = session.get(Commitment, commitment_id)
                    if c:
                        existing_source = c.source_text or ""
                        c.source_text = existing_source + f"\n[{new_status}] {evidence}"
                        c.updated_at = datetime.utcnow()

            logger.info(
                "Updated commitment %d to status '%s'", commitment_id, new_status
            )
            return {"id": commitment_id, "status": new_status, "evidence": evidence}

        except Exception as exc:
            logger.error("Error updating commitment %d: %s", commitment_id, exc, exc_info=True)
            return {"id": commitment_id, "error": str(exc)}

    def check_for_completion(
        self,
        commitment: Dict[str, Any],
        new_emails: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Check if a commitment has been fulfilled based on new emails.

        Args:
            commitment: Commitment dict with description, responsible_person, etc.
            new_emails: List of new email dicts to check.

        Returns:
            Dict with completion status and evidence.
        """
        try:
            desc = commitment.get("description", "").lower()
            responsible = normalize_email(commitment.get("responsible_person", ""))

            # Extract key action words from description
            desc_keywords = set(re.findall(r"\b\w{4,}\b", desc))
            desc_keywords -= {"will", "would", "going", "that", "this", "have", "make", "sure"}

            best_match_score = 0.0
            best_evidence = ""
            completion_detected = False

            for email in new_emails:
                sender = normalize_email(email.get("sender", ""))
                body = clean_html(email.get("body", "")).lower()

                # Sender relevance (same person or part of same thread)
                sender_match = sender == responsible

                # Check for completion keywords
                has_completion_keyword = any(kw in body for kw in _COMPLETION_KEYWORDS)

                # Check for keyword overlap with commitment description
                if desc_keywords:
                    body_words = set(re.findall(r"\b\w{4,}\b", body))
                    overlap = len(desc_keywords & body_words)
                    keyword_score = overlap / len(desc_keywords) if desc_keywords else 0
                else:
                    keyword_score = 0

                # Compute match score
                match_score = 0.0
                if sender_match:
                    match_score += 0.3
                if has_completion_keyword:
                    match_score += 0.3
                match_score += keyword_score * 0.4

                if match_score > best_match_score:
                    best_match_score = match_score
                    # Extract the most relevant sentence as evidence
                    sentences = re.split(r"[.!?\n]+", clean_html(email.get("body", "")))
                    for s in sentences:
                        s = s.strip()
                        if any(kw in s.lower() for kw in _COMPLETION_KEYWORDS):
                            best_evidence = truncate_string(s, 200)
                            break
                    if not best_evidence and sentences:
                        best_evidence = truncate_string(sentences[0], 200)

                if match_score >= 0.6:
                    completion_detected = True

            result = {
                "commitment_id": commitment.get("id"),
                "completion_detected": completion_detected,
                "confidence": round(best_match_score, 2),
                "evidence": best_evidence,
                "suggested_status": "completed" if completion_detected else "pending",
            }

            # Auto-update if high confidence
            if completion_detected and best_match_score >= 0.7:
                cid = commitment.get("id")
                if cid:
                    self.update_commitment_status(cid, "completed", best_evidence)

            return result

        except Exception as exc:
            logger.error("Error checking commitment completion: %s", exc, exc_info=True)
            return {
                "commitment_id": commitment.get("id"),
                "completion_detected": False,
                "confidence": 0.0,
                "evidence": "",
                "suggested_status": "uncertain",
            }

    def get_commitment_summary(self) -> Dict[str, Any]:
        """
        Summary statistics: total, pending, overdue, completed, by person.

        Returns:
            Dict with summary statistics.
        """
        try:
            from sqlalchemy import select, func
            from src.database.models import Commitment

            with self.db.get_session() as session:
                # Total
                total = session.execute(
                    select(func.count(Commitment.id))
                ).scalar() or 0

                # By status
                status_rows = session.execute(
                    select(Commitment.status, func.count(Commitment.id))
                    .group_by(Commitment.status)
                ).all()
                by_status = {status: count for status, count in status_rows}

                # Overdue (pending with past deadline)
                now = datetime.utcnow()
                overdue_count = session.execute(
                    select(func.count(Commitment.id))
                    .where(
                        Commitment.deadline.isnot(None),
                        Commitment.deadline < now,
                        Commitment.status.in_(["pending", "overdue"]),
                    )
                ).scalar() or 0

                # By responsible person (top 10)
                person_rows = session.execute(
                    select(
                        Commitment.responsible_person,
                        func.count(Commitment.id),
                    )
                    .where(Commitment.responsible_person.isnot(None))
                    .group_by(Commitment.responsible_person)
                    .order_by(func.count(Commitment.id).desc())
                    .limit(10)
                ).all()
                by_person = [
                    {"person": person, "count": count}
                    for person, count in person_rows
                ]

                # Upcoming deadlines (next 7 days)
                week_from_now = now + timedelta(days=7)
                upcoming = session.execute(
                    select(func.count(Commitment.id))
                    .where(
                        Commitment.deadline.isnot(None),
                        Commitment.deadline >= now,
                        Commitment.deadline <= week_from_now,
                        Commitment.status.in_(["pending"]),
                    )
                ).scalar() or 0

                return {
                    "total": total,
                    "pending": by_status.get("pending", 0),
                    "completed": by_status.get("completed", 0),
                    "overdue": overdue_count,
                    "uncertain": by_status.get("uncertain", 0),
                    "upcoming_deadlines_7d": upcoming,
                    "by_status": by_status,
                    "by_person": by_person,
                    "generated_at": now.isoformat(),
                }

        except Exception as exc:
            logger.error("Error generating commitment summary: %s", exc, exc_info=True)
            return {"total": 0, "error": str(exc)}

    def get_commitments_by_person(self, person: str) -> List[Dict[str, Any]]:
        """
        Get all commitments for a specific person.

        Args:
            person: Name or email of the responsible person.

        Returns:
            List of commitment dicts.
        """
        try:
            from sqlalchemy import select
            from src.database.models import Commitment

            person_norm = normalize_email(person) if "@" in person else person

            with self.db.get_session() as session:
                stmt = (
                    select(Commitment)
                    .where(Commitment.responsible_person.ilike(f"%{person_norm}%"))
                    .order_by(Commitment.deadline.asc().nullslast())
                )
                results = session.execute(stmt).scalars().all()
                return [self._commitment_to_dict(c) for c in results]

        except Exception as exc:
            logger.error("Error getting commitments for %s: %s", person, exc, exc_info=True)
            return []

    def get_upcoming_deadlines(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get commitments with deadlines in the next N days.

        Args:
            days: Number of days to look ahead.

        Returns:
            List of commitment dicts with approaching deadlines.
        """
        try:
            from sqlalchemy import select, and_
            from src.database.models import Commitment

            now = datetime.utcnow()
            future = now + timedelta(days=days)

            with self.db.get_session() as session:
                stmt = (
                    select(Commitment)
                    .where(
                        and_(
                            Commitment.deadline.isnot(None),
                            Commitment.deadline >= now,
                            Commitment.deadline <= future,
                            Commitment.status.in_(["pending"]),
                        )
                    )
                    .order_by(Commitment.deadline.asc())
                )
                results = session.execute(stmt).scalars().all()
                commitments = [self._commitment_to_dict(c) for c in results]

                # Add days-remaining info
                for c in commitments:
                    if c.get("deadline"):
                        dl = c["deadline"]
                        if isinstance(dl, str):
                            from src.utils.helpers import parse_date
                            dl = parse_date(dl)
                        if isinstance(dl, datetime):
                            c["days_remaining"] = max(0, (dl - now).days)

                return commitments

        except Exception as exc:
            logger.error("Error getting upcoming deadlines: %s", exc, exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_commitments(
        self, text: str, sender: str
    ) -> List[Dict[str, Any]]:
        """
        Detect commitments in text using external detector or built-in patterns.

        Args:
            text: Email body text.
            sender: Sender email address.

        Returns:
            List of raw commitment dicts.
        """
        # Try external detector first
        if self.detector:
            try:
                return self.detector.detect(text, sender)
            except Exception as exc:
                logger.warning("External commitment detector failed: %s", exc)

        # Built-in pattern matching
        commitments: List[Dict[str, Any]] = []

        for pattern in _COMMITMENT_PATTERNS:
            for match in pattern.finditer(text):
                full_match = match.group(0).strip()
                description = match.group(1).strip() if match.lastindex else full_match

                # Skip very short or noisy matches
                if len(description) < 10:
                    continue

                # Extract deadline if present
                deadline = self._extract_deadline(full_match)

                # Confidence based on pattern specificity
                confidence = 0.7
                if any(kw in full_match.lower() for kw in ["promise", "commit", "guarantee"]):
                    confidence = 0.9
                elif any(kw in full_match.lower() for kw in ["will", "'ll"]):
                    confidence = 0.7
                elif "going to" in full_match.lower():
                    confidence = 0.65

                commitments.append({
                    "description": truncate_string(description, 300),
                    "responsible_person": sender,
                    "deadline": deadline,
                    "source_text": truncate_string(full_match, 500),
                    "confidence": confidence,
                })

        return commitments

    def _extract_deadline(self, text: str) -> Optional[str]:
        """Extract deadline text from a commitment sentence."""
        for pattern in _DEADLINE_EXTRACTION_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        return None

    def _parse_deadline_text(
        self, deadline_text: str, reference_date: Any = None
    ) -> Optional[datetime]:
        """
        Parse deadline text into a datetime.

        Args:
            deadline_text: Human-readable deadline (e.g., "tomorrow", "Friday").
            reference_date: Reference date for relative deadlines.

        Returns:
            datetime or None.
        """
        if not deadline_text:
            return None

        # Get reference date
        if isinstance(reference_date, datetime):
            ref = reference_date
        elif isinstance(reference_date, str):
            from src.utils.helpers import parse_date
            ref = parse_date(reference_date) or datetime.utcnow()
        else:
            ref = datetime.utcnow()

        text_lower = deadline_text.lower().strip()

        # Absolute dates
        from src.utils.helpers import parse_date
        parsed = parse_date(deadline_text)
        if parsed:
            return parsed

        # Relative dates
        if text_lower in ("today", "tonight", "eod", "cob", "close of business", "end of day"):
            return ref.replace(hour=23, minute=59, second=59)
        elif text_lower == "tomorrow":
            return (ref + timedelta(days=1)).replace(hour=23, minute=59, second=59)
        elif text_lower in ("end of week", "this week"):
            days_until_friday = (4 - ref.weekday()) % 7
            if days_until_friday == 0:
                days_until_friday = 7
            return (ref + timedelta(days=days_until_friday)).replace(
                hour=23, minute=59, second=59
            )
        elif text_lower == "next week":
            days_until_monday = (7 - ref.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            return (ref + timedelta(days=days_until_monday + 4)).replace(
                hour=23, minute=59, second=59
            )
        elif text_lower == "end of month":
            if ref.month == 12:
                return datetime(ref.year + 1, 1, 1) - timedelta(seconds=1)
            else:
                return datetime(ref.year, ref.month + 1, 1) - timedelta(seconds=1)

        # Day-of-week matching
        days_of_week = {
            "monday": 0, "tuesday": 1, "wednesday": 2,
            "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
        }
        for day_name, day_num in days_of_week.items():
            if day_name in text_lower:
                days_ahead = (day_num - ref.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                return (ref + timedelta(days=days_ahead)).replace(
                    hour=23, minute=59, second=59
                )

        # "within X days/hours"
        within_match = re.search(r"(\d+)\s+(hours?|days?|weeks?)", text_lower)
        if within_match:
            amount = int(within_match.group(1))
            unit = within_match.group(2)
            if "hour" in unit:
                return ref + timedelta(hours=amount)
            elif "day" in unit:
                return ref + timedelta(days=amount)
            elif "week" in unit:
                return ref + timedelta(weeks=amount)

        return None

    def _commitment_to_dict(self, commitment) -> Dict[str, Any]:
        """Convert a Commitment ORM object to a dict."""
        return {
            "id": commitment.id,
            "email_id": commitment.email_id,
            "description": commitment.description,
            "responsible_person": commitment.responsible_person,
            "related_entity": commitment.related_entity,
            "deadline": commitment.deadline.isoformat() if commitment.deadline else None,
            "status": commitment.status,
            "source_text": commitment.source_text,
            "detected_at": commitment.detected_at.isoformat() if commitment.detected_at else None,
            "updated_at": commitment.updated_at.isoformat() if commitment.updated_at else None,
            "confidence": commitment.confidence,
        }
