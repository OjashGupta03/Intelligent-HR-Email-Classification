"""
Executive AI Chief of Staff - Email Preprocessor
Text cleaning, normalisation, and metadata extraction.
"""
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_QUOTED_LINE_RE = re.compile(r"^>+.*$", re.MULTILINE)
_REPLY_HEADER_RE = re.compile(
    r"(On .{10,100}wrote:|From:.+\nTo:.+\nSubject:.+|-----Original Message-----)", re.IGNORECASE
)


def preprocess_email(text: str, max_len: int = 2000) -> str:
    """
    Clean and normalise raw email body text.

    Steps:
      1. Remove quoted reply chains (lines starting with >)
      2. Remove URLs
      3. Remove reply/forward headers
      4. Collapse whitespace
      5. Truncate to max_len
    """
    if not text:
        return ""
    text = _QUOTED_LINE_RE.sub("", text)
    text = _REPLY_HEADER_RE.split(text)[0]  # keep only the top-most reply
    text = _URL_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    text = text.strip()
    if len(text) > max_len:
        text = text[:max_len]
    return text


def tokenize(text: str) -> List[str]:
    """Simple whitespace+punctuation tokeniser."""
    return re.findall(r"\b[a-zA-Z]\w*\b", text.lower())


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove common English stop words."""
    try:
        from nltk.corpus import stopwords
        sw = set(stopwords.words("english"))
    except Exception:
        sw = {
            "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
            "they", "their", "this", "that", "is", "are", "was", "were", "be",
            "been", "have", "has", "do", "does", "did", "will", "would", "can",
            "could", "should", "may", "might", "shall", "a", "an", "the", "in",
            "on", "at", "to", "for", "of", "and", "or", "but", "if", "as",
            "with", "from", "by", "not", "no", "so", "up", "out", "about",
        }
    return [t for t in tokens if t not in sw]


def extract_email_metadata(email_dict: Dict) -> Dict:
    """
    Extract structured metadata from an email dict.
    Returns a dict of features useful for scoring and classification.
    """
    subject = email_dict.get("subject") or ""
    body = email_dict.get("body") or ""
    sender = email_dict.get("sender") or ""
    recipients = email_dict.get("recipients") or []
    cc = email_dict.get("cc") or []
    date_str = email_dict.get("date") or ""

    # Date parsing
    email_date: Optional[datetime] = None
    hour = 12
    day_of_week = 1
    is_weekend = False
    if date_str:
        try:
            if isinstance(date_str, datetime):
                email_date = date_str
            else:
                email_date = datetime.fromisoformat(str(date_str))
            hour = email_date.hour
            day_of_week = email_date.weekday()  # 0=Mon, 6=Sun
            is_weekend = day_of_week >= 5
        except Exception:
            pass

    subject_lower = subject.lower()
    body_lower = body.lower()
    full_text = f"{subject} {body}".lower()

    return {
        "char_count": len(body),
        "word_count": len(body.split()),
        "sentence_count": max(1, body.count(".") + body.count("!") + body.count("?")),
        "subject_length": len(subject),
        "num_recipients": len(recipients) if isinstance(recipients, list) else 1,
        "num_cc": len(cc) if isinstance(cc, list) else 0,
        "is_reply": bool(email_dict.get("is_reply") or subject_lower.startswith("re:")),
        "is_forward": "fwd:" in subject_lower or "forwarded" in body_lower[:200],
        "hour_of_day": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "has_urgent_subject": any(w in subject_lower for w in ["urgent", "asap", "critical", "immediate", "action required"]),
        "has_financial_subject": any(w in subject_lower for w in ["budget", "revenue", "cost", "$", "financial", "invoice", "payment"]),
        "is_all_caps_subject": subject.isupper() and len(subject) > 5,
        "sender_domain": sender.split("@")[-1] if "@" in sender else "",
        "has_attachment_mention": any(w in body_lower for w in ["attached", "attachment", "please find", "see attached", "enclosed"]),
    }


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip."""
    return _MULTI_SPACE_RE.sub(" ", text.lower()).strip()


def truncate_text(text: str, max_len: int = 500) -> str:
    """Truncate text to max_len characters, breaking at a word boundary."""
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    last_space = truncated.rfind(" ")
    if last_space > max_len - 50:
        truncated = truncated[:last_space]
    return truncated + "…"

