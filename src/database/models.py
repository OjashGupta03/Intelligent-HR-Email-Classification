"""
Executive AI Chief of Staff - SQLAlchemy ORM Models
Database schema for email intelligence, priority scoring, commitments, risks, and contacts.
Uses SQLAlchemy 2.0 style with mapped_column.
"""

import json
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    Session,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class Email(Base):
    """Represents a parsed email message stored in the system."""

    __tablename__ = "emails"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[Optional[str]] = mapped_column(String(512), unique=True, nullable=True, index=True)
    subject: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    sender: Mapped[Optional[str]] = mapped_column(String(512), nullable=True, index=True)
    recipients: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON list
    cc: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON list
    date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, index=True)
    body: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    raw_headers: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    thread_id: Mapped[Optional[str]] = mapped_column(String(512), nullable=True, index=True)
    in_reply_to: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    references_field: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_reply: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    label_source: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True, default=None
    )  # gold / heuristic / weak

    # Relationships
    priority: Mapped[Optional["EmailPriority"]] = relationship(
        "EmailPriority", back_populates="email", uselist=False, cascade="all, delete-orphan"
    )
    commitments: Mapped[List["Commitment"]] = relationship(
        "Commitment", back_populates="email", cascade="all, delete-orphan"
    )
    risks: Mapped[List["Risk"]] = relationship(
        "Risk", back_populates="email", cascade="all, delete-orphan"
    )
    annotations: Mapped[List["Annotation"]] = relationship(
        "Annotation", back_populates="email", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_emails_date_sender", "date", "sender"),
        Index("ix_emails_thread_date", "thread_id", "date"),
    )

    def get_recipients(self) -> list:
        """Return recipients as a Python list."""
        if self.recipients:
            try:
                return json.loads(self.recipients)
            except (json.JSONDecodeError, TypeError):
                return []
        return []

    def set_recipients(self, recipients_list: list) -> None:
        """Set recipients from a Python list."""
        self.recipients = json.dumps(recipients_list) if recipients_list else None

    def get_cc(self) -> list:
        """Return CC as a Python list."""
        if self.cc:
            try:
                return json.loads(self.cc)
            except (json.JSONDecodeError, TypeError):
                return []
        return []

    def set_cc(self, cc_list: list) -> None:
        """Set CC from a Python list."""
        self.cc = json.dumps(cc_list) if cc_list else None

    def __repr__(self) -> str:
        return f"<Email(id={self.id}, subject='{self.subject[:50] if self.subject else ''}')>"


class EmailPriority(Base):
    """Priority scoring results for an email."""

    __tablename__ = "email_priorities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("emails.id", ondelete="CASCADE"), nullable=False, index=True
    )
    priority_level: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # critical / high / medium / low
    priority_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)  # 0-100
    urgency_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    importance_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    business_impact_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    sender_importance_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    deadline_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    financial_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    action_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    strategic_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    context_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    score_explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    ml_prediction: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    rule_prediction: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    # Relationships
    email: Mapped["Email"] = relationship("Email", back_populates="priority")

    __table_args__ = (
        Index("ix_priorities_level_score", "priority_level", "priority_score"),
    )

    def get_score_explanation(self) -> dict:
        """Return score explanation as a Python dict."""
        if self.score_explanation:
            try:
                return json.loads(self.score_explanation)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

    def set_score_explanation(self, explanation: dict) -> None:
        """Set score explanation from a Python dict."""
        self.score_explanation = json.dumps(explanation) if explanation else None

    def __repr__(self) -> str:
        return (
            f"<EmailPriority(id={self.id}, level='{self.priority_level}', "
            f"score={self.priority_score:.1f})>"
        )


class Thread(Base):
    """Represents an email conversation thread."""

    __tablename__ = "threads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(String(512), unique=True, nullable=False, index=True)
    subject: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    participants: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON list
    email_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    first_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True, default="active"
    )  # active / resolved / stale

    __table_args__ = (
        Index("ix_threads_last_date", "last_date"),
    )

    def get_participants(self) -> list:
        """Return participants as a Python list."""
        if self.participants:
            try:
                return json.loads(self.participants)
            except (json.JSONDecodeError, TypeError):
                return []
        return []

    def set_participants(self, participants_list: list) -> None:
        """Set participants from a Python list."""
        self.participants = json.dumps(participants_list) if participants_list else None

    def __repr__(self) -> str:
        return f"<Thread(id={self.id}, thread_id='{self.thread_id}', emails={self.email_count})>"


class Commitment(Base):
    """A commitment or promise detected in an email."""

    __tablename__ = "commitments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("emails.id", ondelete="CASCADE"), nullable=False, index=True
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    responsible_person: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    related_entity: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    deadline: Mapped[Optional[str]] = mapped_column(String(512), nullable=True, index=True)
    deadline_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending"
    )  # pending / completed / overdue / uncertain
    source_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Relationships
    email: Mapped["Email"] = relationship("Email", back_populates="commitments")

    __table_args__ = (
        Index("ix_commitments_status_deadline", "status", "deadline"),
    )

    def __repr__(self) -> str:
        return (
            f"<Commitment(id={self.id}, status='{self.status}', "
            f"deadline={self.deadline})>"
        )


class Risk(Base):
    """A risk identified in an email."""

    __tablename__ = "risks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("emails.id", ondelete="CASCADE"), nullable=False, index=True
    )
    category: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    severity: Mapped[str] = mapped_column(
        String(16), nullable=False, default="medium"
    )  # critical / high / medium / low
    explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    suggested_action: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    related_thread_id: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Relationships
    email: Mapped["Email"] = relationship("Email", back_populates="risks")

    __table_args__ = (
        Index("ix_risks_severity_category", "severity", "category"),
    )

    def __repr__(self) -> str:
        return (
            f"<Risk(id={self.id}, category='{self.category}', "
            f"severity='{self.severity}')>"
        )


class Contact(Base):
    """A contact (person or entity) derived from email interactions."""

    __tablename__ = "contacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email_address: Mapped[str] = mapped_column(
        String(512), unique=True, nullable=False, index=True
    )
    name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    organization: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    importance_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    is_vip: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    total_emails_sent: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_emails_received: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_contact_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    first_contact_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_contacts_importance", "importance_score"),
        Index("ix_contacts_vip", "is_vip"),
    )

    def __repr__(self) -> str:
        return (
            f"<Contact(id={self.id}, email='{self.email_address}', "
            f"name='{self.name}')>"
        )


class DailyBriefing(Base):
    """A generated daily briefing summary."""

    __tablename__ = "daily_briefings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False, unique=True, index=True)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    generated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    def get_content(self) -> dict:
        """Return content as a Python dict."""
        if self.content:
            try:
                return json.loads(self.content)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

    def set_content(self, content_dict: dict) -> None:
        """Set content from a Python dict."""
        self.content = json.dumps(content_dict) if content_dict else None

    def __repr__(self) -> str:
        return f"<DailyBriefing(id={self.id}, date={self.date})>"


class Annotation(Base):
    """Manual or heuristic annotations/labels for an email, used for training data."""

    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("emails.id", ondelete="CASCADE"), nullable=False, index=True
    )
    priority_label: Mapped[Optional[str]] = mapped_column(
        String(16), nullable=True
    )  # critical / high / medium / low
    urgency_label: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    importance_label: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    action_required: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    has_commitment: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    has_deadline: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    has_risk: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    risk_category: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    label_type: Mapped[str] = mapped_column(
        String(32), nullable=False, default="heuristic"
    )  # gold / heuristic / weak
    annotated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    # Relationships
    email: Mapped["Email"] = relationship("Email", back_populates="annotations")

    __table_args__ = (
        Index("ix_annotations_label_type", "label_type"),
        Index("ix_annotations_priority", "priority_label"),
    )

    def __repr__(self) -> str:
        return (
            f"<Annotation(id={self.id}, email_id={self.email_id}, "
            f"priority='{self.priority_label}', type='{self.label_type}')>"
        )


# ---------------------------------------------------------------------------
# SQLite-specific optimizations: enable WAL mode and foreign keys
# ---------------------------------------------------------------------------

@event.listens_for(Base.metadata, "after_create")
def _set_sqlite_pragma(target, connection, **kwargs):
    """Enable WAL mode and foreign keys for SQLite connections."""
    if connection.dialect.name == "sqlite":
        cursor = connection.connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()
