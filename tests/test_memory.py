"""Unit tests for ConversationMemory and thread reconstruction."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.database.db_manager import DatabaseManager
from src.intelligence.conversation_memory import ConversationMemory


def test_thread_reconstruction_headers():
    db = DatabaseManager()
    memory = ConversationMemory(db)

    emails = [
        {
            "message_id": "<msg-001@acme.com>",
            "subject": "Acquisition terms for Beta Corp",
            "body": "What are the latest terms for Beta Corp?",
            "sender": "ceo@acme.com",
            "in_reply_to": None,
            "references_field": "",
            "date": "2024-03-01T10:00:00",
        },
        {
            "message_id": "<msg-002@acme.com>",
            "subject": "Re: Acquisition terms for Beta Corp",
            "body": "Here are the terms: $15M cash plus equity. We agreed to move forward.",
            "sender": "cfo@acme.com",
            "in_reply_to": "<msg-001@acme.com>",
            "references_field": "<msg-001@acme.com>",
            "date": "2024-03-01T11:00:00",
        },
    ]

    threads = memory.reconstruct_threads(emails)
    assert len(threads) == 1
    thread_emails = list(threads.values())[0]
    assert len(thread_emails) == 2


def test_thread_reconstruction_subject_matching():
    db = DatabaseManager()
    memory = ConversationMemory(db)

    emails = [
        {
            "message_id": "<msg-101@acme.com>",
            "subject": "Board Meeting Preparation",
            "body": "Who will present the Q4 results?",
            "sender": "board@acme.com",
            "in_reply_to": None,
            "references_field": "",
            "date": "2024-03-02T10:00:00",
        },
        {
            "message_id": "<msg-102@acme.com>",
            "subject": "Re: Board Meeting Preparation",
            "body": "I will handle the slides.",
            "sender": "cfo@acme.com",
            "in_reply_to": None,
            "references_field": "",
            "date": "2024-03-02T12:00:00",
        },
    ]

    threads = memory.reconstruct_threads(emails)
    assert len(threads) == 1
    thread_emails = list(threads.values())[0]
    assert len(thread_emails) == 2


def test_decision_and_key_point_extraction():
    db = DatabaseManager()
    memory = ConversationMemory(db)

    text = (
        "After careful review of the budget, we decided to approve the $2M cloud expansion.\n"
        "We agreed to finalize the vendor contract by Friday."
    )
    decisions = memory._extract_decisions(text)
    assert len(decisions) >= 1
    assert any("decided" in d.lower() or "agreed" in d.lower() for d in decisions)

