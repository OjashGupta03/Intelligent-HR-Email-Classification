"""
Executive AI Chief of Staff - Database Manager
Provides session management and CRUD operations for all database tables.
Uses SQLAlchemy 2.0 patterns with proper error handling and logging.
"""

import json
import logging
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from sqlalchemy import create_engine, func, select, and_, or_, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DB_URL
from src.database.models import (
    Base,
    Email,
    EmailPriority,
    Thread,
    Commitment,
    Risk,
    Contact,
    DailyBriefing,
    Annotation,
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Central database manager for the Executive AI Chief of Staff.

    Handles all CRUD operations, session management, and provides
    convenience methods for querying emails, priorities, commitments,
    risks, contacts, briefings, and annotations.
    """

    def __init__(self, db_url: Optional[str] = None) -> None:
        """
        Initialize the database manager.

        Args:
            db_url: SQLAlchemy database URL. Defaults to config DB_URL.
        """
        self._db_url = db_url or DB_URL
        self._engine = create_engine(
            self._db_url,
            echo=False,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False} if "sqlite" in self._db_url else {},
        )
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)
        logger.info("DatabaseManager initialized with URL: %s", self._db_url)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """Create all tables in the database if they don't exist."""
        try:
            Base.metadata.create_all(self._engine)
            # Enable SQLite pragmas
            with self._engine.connect() as conn:
                if "sqlite" in self._db_url:
                    conn.execute(text("PRAGMA journal_mode=WAL;"))
                    conn.execute(text("PRAGMA foreign_keys=ON;"))
                    conn.commit()
            logger.info("Database tables created / verified successfully.")
        except SQLAlchemyError as exc:
            logger.error("Failed to initialize database: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Provide a transactional session scope.

        Yields:
            A SQLAlchemy Session. Commits on success, rolls back on error.
        """
        session: Session = self._session_factory()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as exc:
            session.rollback()
            logger.error("Session error (rolled back): %s", exc)
            raise
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Email CRUD
    # ------------------------------------------------------------------

    def add_email(self, email_data: Dict[str, Any]) -> Email:
        """
        Add a single email to the database.

        Args:
            email_data: Dict with keys matching Email model columns.
                        'recipients' and 'cc' may be lists (auto-serialised to JSON).

        Returns:
            The persisted Email object.
        """
        with self.get_session() as session:
            # Serialize list fields to JSON
            data = dict(email_data)
            if isinstance(data.get("recipients"), list):
                data["recipients"] = json.dumps(data["recipients"])
            if isinstance(data.get("cc"), list):
                data["cc"] = json.dumps(data["cc"])

            email = Email(**data)
            session.add(email)
            session.flush()
            logger.debug("Added email id=%s, message_id=%s", email.id, email.message_id)
            return email

    def get_email(self, email_id: int) -> Optional[Email]:
        """
        Retrieve a single email by primary key.

        Args:
            email_id: The database id of the email.

        Returns:
            Email object or None.
        """
        with self.get_session() as session:
            email = session.get(Email, email_id)
            return email

    def get_email_by_message_id(self, message_id: str) -> Optional[Email]:
        """
        Retrieve a single email by its RFC message-id header.

        Args:
            message_id: The message-id string.

        Returns:
            Email object or None.
        """
        with self.get_session() as session:
            stmt = select(Email).where(Email.message_id == message_id)
            return session.execute(stmt).scalar_one_or_none()

    def get_emails(
        self,
        filters: Optional[Dict[str, Any]] = None,
        *,
        sender: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        priority_level: Optional[str] = None,
        thread_id: Optional[str] = None,
        label_source: Optional[str] = None,
        limit: int = 10000,
        offset: int = 0,
    ) -> List[Email]:
        """Query emails with optional filters dict or keyword arguments."""
        if filters and isinstance(filters, dict):
            sender = filters.get("sender", sender)
            start_date = filters.get("start_date", start_date)
            end_date = filters.get("end_date", end_date)
            priority_level = filters.get("priority_level", priority_level)
            thread_id = filters.get("thread_id", thread_id)
            label_source = filters.get("label_source", label_source)
            limit = filters.get("limit", limit)
            offset = filters.get("offset", offset)
        with self.get_session() as session:
            stmt = select(Email)
            conditions = []

            if sender:
                conditions.append(Email.sender.ilike(f"%{sender}%"))
            if start_date:
                conditions.append(Email.date >= start_date)
            if end_date:
                conditions.append(Email.date <= end_date)
            if thread_id:
                conditions.append(Email.thread_id == thread_id)
            if label_source:
                conditions.append(Email.label_source == label_source)

            if conditions:
                stmt = stmt.where(and_(*conditions))

            if priority_level:
                stmt = stmt.join(EmailPriority, Email.id == EmailPriority.email_id).where(
                    EmailPriority.priority_level == priority_level.lower()
                )

            stmt = stmt.order_by(Email.date.desc()).limit(limit).offset(offset)
            result = session.execute(stmt).scalars().all()
            return list(result)

    def get_emails_in_date_range(
        self, start: datetime, end: datetime
    ) -> List[Email]:
        """
        Return all emails in a date range (inclusive).

        Args:
            start: Start datetime.
            end: End datetime.

        Returns:
            List of Email objects.
        """
        return self.get_emails(start_date=start, end_date=end, limit=10_000)

    # ------------------------------------------------------------------
    # Email Priority helpers
    # ------------------------------------------------------------------

    def get_email_priority(self, email_id: int) -> Optional[EmailPriority]:
        """Get priority record for a specific email."""
        if email_id is None:
            return None
        with self.get_session() as session:
            return session.execute(
                select(EmailPriority).where(EmailPriority.email_id == email_id)
            ).scalar_one_or_none()

    def get_email_priorities(self, limit: int = 500) -> List[EmailPriority]:
        """Get all email priority records."""
        with self.get_session() as session:
            stmt = select(EmailPriority).order_by(EmailPriority.priority_score.desc()).limit(limit)
            return list(session.execute(stmt).scalars().all())

    def bulk_add_emails(self, emails_list: List[Dict[str, Any]]) -> int:
        """
        Bulk-insert emails. Duplicates (by message_id) are skipped.

        Args:
            emails_list: List of email dicts.

        Returns:
            Number of emails actually inserted.
        """
        added = 0
        # Get valid column names for Email model
        valid_columns = {c.name for c in Email.__table__.columns}
        with self.get_session() as session:
            for email_data in emails_list:
                data = dict(email_data)
                if isinstance(data.get("recipients"), list):
                    data["recipients"] = json.dumps(data["recipients"])
                if isinstance(data.get("cc"), list):
                    data["cc"] = json.dumps(data["cc"])

                # Parse date string to datetime if needed
                if isinstance(data.get("date"), str):
                    try:
                        data["date"] = datetime.fromisoformat(data["date"])
                    except Exception:
                        data["date"] = None

                # Skip if message_id already exists
                msg_id = data.get("message_id")
                if msg_id:
                    existing = session.execute(
                        select(Email.id).where(Email.message_id == msg_id)
                    ).scalar_one_or_none()
                    if existing is not None:
                        # Store db_id for reference
                        email_data["db_id"] = existing
                        continue

                # Filter to only valid Email columns
                filtered_data = {k: v for k, v in data.items() if k in valid_columns}

                email = Email(**filtered_data)
                session.add(email)
                session.flush()
                # Store db_id back for later reference
                email_data["db_id"] = email.id
                added += 1

            session.flush()
        logger.info("Bulk added %d emails (out of %d provided).", added, len(emails_list))
        return added

    # ------------------------------------------------------------------
    # Priority CRUD
    # ------------------------------------------------------------------

    def add_priority(self, email_id: int, scores_dict: Dict[str, Any]) -> EmailPriority:
        """
        Add or update priority scores for an email.

        Args:
            email_id: The email id.
            scores_dict: Dict with score fields. Should include 'priority_level',
                         'priority_score', and individual dimension scores.

        Returns:
            The persisted EmailPriority object.
        """
        with self.get_session() as session:
            # Check if priority already exists
            existing = session.execute(
                select(EmailPriority).where(EmailPriority.email_id == email_id)
            ).scalar_one_or_none()

            data = dict(scores_dict)
            # Serialize score_explanation if it's a dict
            if isinstance(data.get("score_explanation"), dict):
                data["score_explanation"] = json.dumps(data["score_explanation"])

            if existing:
                for key, value in data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                session.flush()
                logger.debug("Updated priority for email_id=%d", email_id)
                return existing
            else:
                priority = EmailPriority(email_id=email_id, **data)
                session.add(priority)
                session.flush()
                logger.debug("Added priority for email_id=%d", email_id)
                return priority

    # ------------------------------------------------------------------
    # Commitment CRUD
    # ------------------------------------------------------------------

    def add_commitment(self, data: Dict[str, Any]) -> Commitment:
        """
        Add a commitment record.

        Args:
            data: Dict with Commitment fields.

        Returns:
            The persisted Commitment object.
        """
        with self.get_session() as session:
            valid_cols = {c.name for c in Commitment.__table__.columns}
            filtered = {k: v for k, v in data.items() if k in valid_cols}
            if isinstance(filtered.get("deadline"), datetime):
                filtered["deadline_date"] = filtered["deadline"]
                filtered["deadline"] = filtered["deadline"].strftime("%Y-%m-%d")
            commitment = Commitment(**filtered)
            session.add(commitment)
            session.flush()
            logger.debug("Added commitment id=%d for email_id=%s", commitment.id, data.get("email_id"))
            return commitment

    def get_commitments(
        self,
        status: Optional[str] = None,
        *,
        overdue: bool = False,
        limit: int = 500,
    ) -> List[Commitment]:
        """
        Query commitments with optional filters.

        Args:
            status: Filter by status (pending/completed/overdue/uncertain).
            overdue: If True, return only commitments past their deadline.
            limit: Max results.

        Returns:
            List of Commitment objects.
        """
        with self.get_session() as session:
            stmt = select(Commitment)
            conditions = []

            if status:
                conditions.append(Commitment.status == status)
            if overdue:
                now = datetime.utcnow()
                conditions.append(
                    and_(
                        Commitment.deadline.isnot(None),
                        Commitment.deadline < now,
                        Commitment.status != "completed",
                    )
                )

            if conditions:
                stmt = stmt.where(and_(*conditions))

            stmt = stmt.order_by(Commitment.deadline.asc().nullslast()).limit(limit)
            return list(session.execute(stmt).scalars().all())

    def update_commitment_status(self, commitment_id: int, status: str) -> None:
        """
        Update the status of a commitment.

        Args:
            commitment_id: The commitment's database id.
            status: New status string.
        """
        with self.get_session() as session:
            commitment = session.get(Commitment, commitment_id)
            if commitment is None:
                logger.warning("Commitment id=%d not found.", commitment_id)
                return
            commitment.status = status
            commitment.updated_at = datetime.utcnow()
            session.flush()
            logger.debug("Updated commitment id=%d to status='%s'", commitment_id, status)

    # ------------------------------------------------------------------
    # Risk CRUD
    # ------------------------------------------------------------------

    def add_risk(self, data: Dict[str, Any]) -> Risk:
        """
        Add a risk record.

        Args:
            data: Dict with Risk fields.

        Returns:
            The persisted Risk object.
        """
        with self.get_session() as session:
            risk = Risk(**data)
            session.add(risk)
            session.flush()
            logger.debug("Added risk id=%d for email_id=%s", risk.id, data.get("email_id"))
            return risk

    def get_risks(
        self,
        category: Optional[str] = None,
        severity: Optional[str] = None,
        *,
        limit: int = 500,
    ) -> List[Risk]:
        """
        Query risks with optional filters.

        Args:
            category: Filter by risk category.
            severity: Filter by severity level.
            limit: Max results.

        Returns:
            List of Risk objects.
        """
        with self.get_session() as session:
            stmt = select(Risk)
            conditions = []

            if category:
                conditions.append(Risk.category == category)
            if severity:
                conditions.append(Risk.severity == severity)

            if conditions:
                stmt = stmt.where(and_(*conditions))

            stmt = stmt.order_by(Risk.detected_at.desc()).limit(limit)
            return list(session.execute(stmt).scalars().all())

    # ------------------------------------------------------------------
    # Contact CRUD
    # ------------------------------------------------------------------

    def add_contact(self, data: Dict[str, Any]) -> Contact:
        """
        Add or update a contact.

        Args:
            data: Dict with Contact fields. Must include 'email_address'.

        Returns:
            The persisted Contact object.
        """
        with self.get_session() as session:
            email_addr = data.get("email_address", "").strip().lower()
            existing = session.execute(
                select(Contact).where(Contact.email_address == email_addr)
            ).scalar_one_or_none()

            if existing:
                for key, value in data.items():
                    if key != "email_address" and hasattr(existing, key):
                        setattr(existing, key, value)
                session.flush()
                logger.debug("Updated contact '%s'", email_addr)
                return existing
            else:
                data_copy = dict(data)
                data_copy["email_address"] = email_addr
                contact = Contact(**data_copy)
                session.add(contact)
                session.flush()
                logger.debug("Added contact '%s'", email_addr)
                return contact

    def get_contact(self, email_address: str) -> Optional[Contact]:
        """
        Retrieve a contact by email address.

        Args:
            email_address: The email address to look up.

        Returns:
            Contact object or None.
        """
        with self.get_session() as session:
            addr = email_address.strip().lower()
            stmt = select(Contact).where(Contact.email_address == addr)
            return session.execute(stmt).scalar_one_or_none()

    def update_contact_stats(
        self, email_address: str, *, sent: bool = False, received: bool = False
    ) -> None:
        """
        Increment send/receive counters for a contact, creating if needed.

        Args:
            email_address: Contact's email address.
            sent: Increment sent counter.
            received: Increment received counter.
        """
        with self.get_session() as session:
            addr = email_address.strip().lower()
            contact = session.execute(
                select(Contact).where(Contact.email_address == addr)
            ).scalar_one_or_none()

            now = datetime.utcnow()
            if contact is None:
                contact = Contact(
                    email_address=addr,
                    total_emails_sent=1 if sent else 0,
                    total_emails_received=1 if received else 0,
                    first_contact_date=now,
                    last_contact_date=now,
                )
                session.add(contact)
            else:
                if sent:
                    contact.total_emails_sent += 1
                if received:
                    contact.total_emails_received += 1
                contact.last_contact_date = now

            session.flush()

    def get_top_contacts(self, limit: int = 20) -> List[Contact]:
        """
        Return the top contacts sorted by importance score and total interactions.

        Args:
            limit: Number of contacts to return.

        Returns:
            List of Contact objects.
        """
        with self.get_session() as session:
            stmt = (
                select(Contact)
                .order_by(
                    Contact.importance_score.desc(),
                    (Contact.total_emails_sent + Contact.total_emails_received).desc(),
                )
                .limit(limit)
            )
            return list(session.execute(stmt).scalars().all())

    # ------------------------------------------------------------------
    # Thread operations
    # ------------------------------------------------------------------

    def get_thread(self, thread_id: str) -> Optional[Thread]:
        """
        Retrieve a thread by its thread_id, along with related emails.

        Args:
            thread_id: The thread identifier string.

        Returns:
            Thread object or None.
        """
        with self.get_session() as session:
            stmt = select(Thread).where(Thread.thread_id == thread_id)
            thread = session.execute(stmt).scalar_one_or_none()
            return thread

    def upsert_thread(self, data: Dict[str, Any]) -> Thread:
        """
        Insert or update a thread record.

        Args:
            data: Dict with Thread fields. Must include 'thread_id'.

        Returns:
            The persisted Thread object.
        """
        with self.get_session() as session:
            tid = data["thread_id"]
            existing = session.execute(
                select(Thread).where(Thread.thread_id == tid)
            ).scalar_one_or_none()

            # Serialize participants if it's a list
            data_copy = dict(data)
            if isinstance(data_copy.get("participants"), list):
                data_copy["participants"] = json.dumps(data_copy["participants"])

            if existing:
                for key, value in data_copy.items():
                    if key != "thread_id" and hasattr(existing, key):
                        setattr(existing, key, value)
                session.flush()
                return existing
            else:
                thread = Thread(**data_copy)
                session.add(thread)
                session.flush()
                return thread

    # ------------------------------------------------------------------
    # Daily Briefing
    # ------------------------------------------------------------------

    def save_briefing(self, date, content: Dict[str, Any]) -> DailyBriefing:
        """
        Save a daily briefing. Overwrites if one already exists for the date.

        Args:
            date: The briefing date (datetime or string like '2024-01-15').
            content: Dict of briefing content.

        Returns:
            The persisted DailyBriefing object.
        """
        with self.get_session() as session:
            # Normalize to datetime object for SQLite DateTime column
            if isinstance(date, str):
                try:
                    briefing_date = datetime.strptime(date[:10], "%Y-%m-%d")
                except Exception:
                    briefing_date = datetime.utcnow()
            elif isinstance(date, datetime):
                briefing_date = datetime(date.year, date.month, date.day)
            else:
                briefing_date = datetime.utcnow()

            existing = session.execute(
                select(DailyBriefing).where(DailyBriefing.date == briefing_date)
            ).scalar_one_or_none()

            content_json = json.dumps(content) if content else None

            if existing:
                existing.content = content_json
                existing.generated_at = datetime.utcnow()
                session.flush()
                logger.info("Updated briefing for date=%s", briefing_date)
                return existing
            else:
                briefing = DailyBriefing(
                    date=briefing_date,
                    content=content_json,
                    generated_at=datetime.utcnow(),
                )
                session.add(briefing)
                session.flush()
                logger.info("Saved briefing for date=%s", briefing_date)
                return briefing

    # ------------------------------------------------------------------
    # Annotation CRUD
    # ------------------------------------------------------------------

    def add_annotation(self, email_id: int, labels: Dict[str, Any]) -> Annotation:
        """
        Add an annotation to an email.

        Args:
            email_id: The email id.
            labels: Dict with annotation fields (priority_label, urgency_label, etc.).

        Returns:
            The persisted Annotation object.
        """
        with self.get_session() as session:
            annotation = Annotation(email_id=email_id, **labels)
            session.add(annotation)
            session.flush()
            logger.debug("Added annotation for email_id=%d", email_id)
            return annotation

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for the entire database.

        Returns:
            Dict with counts for each table and additional aggregate info.
        """
        with self.get_session() as session:
            email_count = session.execute(select(func.count(Email.id))).scalar() or 0
            priority_count = session.execute(select(func.count(EmailPriority.id))).scalar() or 0
            thread_count = session.execute(select(func.count(Thread.id))).scalar() or 0
            commitment_count = session.execute(select(func.count(Commitment.id))).scalar() or 0
            risk_count = session.execute(select(func.count(Risk.id))).scalar() or 0
            contact_count = session.execute(select(func.count(Contact.id))).scalar() or 0
            briefing_count = session.execute(select(func.count(DailyBriefing.id))).scalar() or 0
            annotation_count = session.execute(select(func.count(Annotation.id))).scalar() or 0

            # Priority level breakdown
            priority_breakdown: Dict[str, int] = {}
            if priority_count > 0:
                rows = session.execute(
                    select(EmailPriority.priority_level, func.count(EmailPriority.id))
                    .group_by(EmailPriority.priority_level)
                ).all()
                for level, cnt in rows:
                    priority_breakdown[level] = cnt

            # Commitment status breakdown
            commitment_breakdown: Dict[str, int] = {}
            if commitment_count > 0:
                rows = session.execute(
                    select(Commitment.status, func.count(Commitment.id))
                    .group_by(Commitment.status)
                ).all()
                for status, cnt in rows:
                    commitment_breakdown[status] = cnt

            # Risk severity breakdown
            risk_breakdown: Dict[str, int] = {}
            if risk_count > 0:
                rows = session.execute(
                    select(Risk.severity, func.count(Risk.id))
                    .group_by(Risk.severity)
                ).all()
                for severity, cnt in rows:
                    risk_breakdown[severity] = cnt

            # Annotation label_type breakdown
            annotation_breakdown: Dict[str, int] = {}
            if annotation_count > 0:
                rows = session.execute(
                    select(Annotation.label_type, func.count(Annotation.id))
                    .group_by(Annotation.label_type)
                ).all()
                for ltype, cnt in rows:
                    annotation_breakdown[ltype] = cnt

            # Date range
            min_date = session.execute(select(func.min(Email.date))).scalar()
            max_date = session.execute(select(func.max(Email.date))).scalar()

            return {
                "emails": email_count,
                "priorities": priority_count,
                "threads": thread_count,
                "commitments": commitment_count,
                "risks": risk_count,
                "contacts": contact_count,
                "briefings": briefing_count,
                "annotations": annotation_count,
                "priority_breakdown": priority_breakdown,
                "commitment_breakdown": commitment_breakdown,
                "risk_breakdown": risk_breakdown,
                "annotation_breakdown": annotation_breakdown,
                "date_range": {
                    "earliest": min_date.isoformat() if min_date else None,
                    "latest": max_date.isoformat() if max_date else None,
                },
            }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def drop_all(self) -> None:
        """Drop all tables. Use with caution - primarily for testing."""
        Base.metadata.drop_all(self._engine)
        logger.warning("All database tables dropped.")

    def get_recent_briefings(self, n: int = 7) -> List:
        """Return the N most recent daily briefings."""
        with self.get_session() as session:
            stmt = select(DailyBriefing).order_by(DailyBriefing.generated_at.desc()).limit(n)
            return list(session.execute(stmt).scalars().all())

    def session_scope(self):
        """Expose session context manager for external use."""
        return self.get_session()

    def dispose(self) -> None:
        """Dispose of the engine connection pool."""
        self._engine.dispose()
        logger.info("Database engine disposed.")
