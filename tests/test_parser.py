"""Unit tests for email parsing and preprocessing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.data.email_parser import parse_raw_email, clean_email_body, extract_reply_chain
from src.data.preprocessor import preprocess_email, tokenize, remove_stopwords, extract_email_metadata


def test_clean_email_body():
    raw = (
        "Here is the project update.\n\n"
        "-----Original Message-----\n"
        "From: boss@company.com\n"
        "To: me@company.com\n"
        "> Old quoted text\n"
        "> Another quoted line"
    )
    cleaned = clean_email_body(raw)
    assert "Here is the project update" in cleaned
    assert "Old quoted text" not in cleaned


def test_preprocess_email():
    text = "Please check https://example.com/report and let me know ASAP!  \n\n\nThanks."
    cleaned = preprocess_email(text)
    assert "https://example.com" not in cleaned
    assert "ASAP" in cleaned


def test_tokenize_and_stopwords():
    text = "The quarterly financial report was submitted yesterday"
    tokens = tokenize(text)
    assert "quarterly" in tokens
    assert "financial" in tokens
    filtered = remove_stopwords(tokens)
    assert "the" not in filtered
    assert "was" not in filtered
    assert "financial" in filtered


def test_extract_metadata():
    email_dict = {
        "subject": "URGENT: Contract Approval Needed",
        "body": "Please review the attached contract for $2.5M. Deadline is Friday.",
        "sender": "cfo@company.com",
        "recipients": ["ceo@company.com"],
        "date": "2024-03-15T09:30:00",
    }
    meta = extract_email_metadata(email_dict)
    assert meta["has_urgent_subject"] is True
    assert meta["has_attachment_mention"] is True
    assert meta["is_all_caps_subject"] is False
    assert meta["num_recipients"] == 1

