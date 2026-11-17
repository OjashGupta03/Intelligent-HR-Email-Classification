"""
Executive AI Chief of Staff - Email Parser
Parses raw RFC 2822 emails (Enron dataset format), extracts structured fields,
cleans email bodies, and handles reply chain extraction.
"""

import email
import email.utils
import email.header
import logging
import re
from datetime import datetime
from email.message import Message
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regex patterns for body cleaning
# ---------------------------------------------------------------------------

# Patterns to detect forwarded message markers
_FWD_PATTERNS = [
    re.compile(r"-{3,}\s*Forwarded\s+by\s+.+?-{3,}", re.IGNORECASE | re.DOTALL),
    re.compile(r"-{3,}\s*Original\s+Message\s*-{3,}", re.IGNORECASE),
    re.compile(r"Begin\s+forwarded\s+message:", re.IGNORECASE),
    re.compile(r">{2,}\s*-{3,}", re.IGNORECASE),
]

# Patterns to detect signature blocks
_SIG_PATTERNS = [
    re.compile(r"\n--\s*\n.*", re.DOTALL),  # standard signature delimiter
    re.compile(r"\n_{20,}.*", re.DOTALL),  # long underscore line
    re.compile(r"\nSent\s+from\s+my\s+(iPhone|iPad|BlackBerry|Android).*", re.IGNORECASE | re.DOTALL),
    re.compile(r"\nGet\s+Outlook\s+for\s+.*", re.IGNORECASE | re.DOTALL),
]

# Reply header patterns (for splitting reply chains)
_REPLY_HEADER_PATTERN = re.compile(
    r"(?:^|\n)(?:On\s+.+?wrote:|From:\s+.+?\n(?:Sent|Date):\s+.+?\nTo:\s+.+?\nSubject:\s+.+?\n)",
    re.IGNORECASE | re.DOTALL,
)

# Quoted text (lines starting with >)
_QUOTED_LINE_PATTERN = re.compile(r"^>+\s?.*$", re.MULTILINE)

# Email address extraction
_EMAIL_ADDR_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# Excessive whitespace
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def parse_raw_email(raw_text: str) -> Dict[str, Any]:
    """
    Parse a raw RFC 2822 email string into a structured dict.

    Args:
        raw_text: The raw email text (headers + body).

    Returns:
        Dict with keys: subject, sender, recipients, cc, date, body,
        message_id, in_reply_to, references, raw_headers, is_reply, thread_id.
    """
    try:
        msg: Message = email.message_from_string(raw_text)
    except Exception as exc:
        logger.error("Failed to parse raw email: %s", exc)
        return _empty_email_dict()

    return _extract_fields(msg, raw_text)


def parse_email_file(filepath: str) -> Dict[str, Any]:
    """
    Parse an email from a file path.

    Args:
        filepath: Path to the email file.

    Returns:
        Parsed email dict, or empty dict on failure.
    """
    fpath = Path(filepath)
    if not fpath.exists():
        logger.warning("Email file not found: %s", filepath)
        return _empty_email_dict()

    try:
        # Try common encodings
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                raw_text = fpath.read_text(encoding=enc)
                break
            except (UnicodeDecodeError, ValueError):
                continue
        else:
            raw_text = fpath.read_bytes().decode("utf-8", errors="replace")

        return parse_raw_email(raw_text)
    except Exception as exc:
        logger.error("Error reading email file '%s': %s", filepath, exc)
        return _empty_email_dict()


def clean_email_body(body: str) -> str:
    """
    Clean an email body by removing forwarded headers, signatures,
    and quoted text.

    Args:
        body: The raw email body text.

    Returns:
        Cleaned body text.
    """
    if not body:
        return ""

    cleaned = body

    # Remove forwarded message blocks
    for pattern in _FWD_PATTERNS:
        cleaned = pattern.split(cleaned)[0]

    # Remove signature blocks
    for pattern in _SIG_PATTERNS:
        cleaned = pattern.split(cleaned)[0]

    # Remove quoted lines (lines starting with >)
    cleaned = _QUOTED_LINE_PATTERN.sub("", cleaned)

    # Collapse excessive whitespace
    cleaned = _MULTI_NEWLINE.sub("\n\n", cleaned)
    cleaned = _MULTI_SPACE.sub(" ", cleaned)

    return cleaned.strip()


def extract_reply_chain(body: str) -> List[str]:
    """
    Extract individual messages from a reply chain.

    Args:
        body: Full email body that may contain reply chains.

    Returns:
        List of message segments, ordered from most recent to oldest.
    """
    if not body:
        return []

    # Split on reply headers
    segments = _REPLY_HEADER_PATTERN.split(body)

    result = []
    for segment in segments:
        cleaned = segment.strip()
        if cleaned and len(cleaned) > 5:
            # Remove leading/trailing quote markers
            cleaned = re.sub(r"^>+\s?", "", cleaned, flags=re.MULTILINE)
            cleaned = cleaned.strip()
            if cleaned:
                result.append(cleaned)

    return result if result else [body.strip()]


def normalize_email_address(addr: str) -> str:
    """
    Normalize an email address to lowercase, stripped form.

    Handles formats like:
    - "John Smith" <john@example.com>
    - john@example.com
    - <john@example.com>

    Args:
        addr: Raw email address string.

    Returns:
        Normalized email address (lowercase, no angle brackets or display name).
    """
    if not addr:
        return ""

    # Use email.utils to parse display name and address
    _, parsed_addr = email.utils.parseaddr(addr)
    if parsed_addr:
        return parsed_addr.strip().lower()

    # Fallback: extract with regex
    match = _EMAIL_ADDR_PATTERN.search(addr)
    if match:
        return match.group(0).strip().lower()

    return addr.strip().lower()


def extract_date(date_str: str) -> Optional[datetime]:
    """
    Parse an email date header string into a datetime object.

    Handles various RFC 2822 date formats and common malformed dates.

    Args:
        date_str: Date header string.

    Returns:
        datetime object or None if unparseable.
    """
    if not date_str:
        return None

    # Try email.utils first (handles RFC 2822)
    try:
        parsed = email.utils.parsedate_to_datetime(date_str)
        return parsed.replace(tzinfo=None)  # Store as naive UTC
    except (TypeError, ValueError, IndexError):
        pass

    # Try common fallback formats
    fallback_formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%d %b %Y %H:%M:%S",
        "%a, %d %b %Y %H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in fallback_formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except (ValueError, TypeError):
            continue

    # Last resort: try to extract a date-like substring
    date_match = re.search(
        r"(\d{1,2}\s+\w{3}\s+\d{4}\s+\d{1,2}:\d{2}:\d{2})", date_str
    )
    if date_match:
        try:
            return datetime.strptime(date_match.group(1), "%d %b %Y %H:%M:%S")
        except ValueError:
            pass

    logger.debug("Could not parse date string: '%s'", date_str)
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_fields(msg: Message, raw_text: str) -> Dict[str, Any]:
    """Extract structured fields from a parsed email.message.Message."""

    # --- Subject ---
    subject = _decode_header(msg.get("Subject", ""))

    # --- Sender ---
    sender_raw = msg.get("From", "")
    sender = normalize_email_address(sender_raw)

    # --- Recipients ---
    to_raw = msg.get("To", "")
    recipients = _parse_address_list(to_raw)

    # --- CC ---
    cc_raw = msg.get("Cc", "") or msg.get("CC", "")
    cc = _parse_address_list(cc_raw)

    # --- Date ---
    date_str = msg.get("Date", "")
    parsed_date = extract_date(date_str)

    # --- Body ---
    body = _extract_body(msg)

    # --- Message-ID ---
    message_id = msg.get("Message-ID", "") or msg.get("Message-Id", "")
    if message_id:
        message_id = message_id.strip().strip("<>")

    # --- In-Reply-To ---
    in_reply_to = msg.get("In-Reply-To", "")
    if in_reply_to:
        in_reply_to = in_reply_to.strip().strip("<>")

    # --- References ---
    references = msg.get("References", "")
    if references:
        references = references.strip()

    # --- Thread ID ---
    # Use the first reference or in_reply_to as thread_id, falling back to message_id
    thread_id = _derive_thread_id(message_id, in_reply_to, references)

    # --- Is Reply ---
    is_reply = bool(in_reply_to) or _subject_is_reply(subject)

    # --- Raw headers ---
    raw_headers = ""
    try:
        header_end = raw_text.find("\n\n")
        if header_end > 0:
            raw_headers = raw_text[:header_end]
    except Exception:
        pass

    return {
        "subject": subject,
        "sender": sender,
        "recipients": recipients,
        "cc": cc,
        "date": parsed_date,
        "body": body,
        "message_id": message_id or None,
        "in_reply_to": in_reply_to or None,
        "references_field": references or None,
        "raw_headers": raw_headers,
        "thread_id": thread_id,
        "is_reply": is_reply,
    }


def _decode_header(header_value: str) -> str:
    """Decode an email header that may have RFC 2047 encoded words."""
    if not header_value:
        return ""
    try:
        decoded_parts = email.header.decode_header(header_value)
        parts = []
        for part, charset in decoded_parts:
            if isinstance(part, bytes):
                parts.append(part.decode(charset or "utf-8", errors="replace"))
            else:
                parts.append(part)
        return " ".join(parts).strip()
    except Exception:
        return str(header_value).strip()


def _parse_address_list(raw: str) -> List[str]:
    """Parse a comma-separated address list into normalized addresses."""
    if not raw:
        return []
    addresses = []
    for addr in raw.split(","):
        normalized = normalize_email_address(addr.strip())
        if normalized and "@" in normalized:
            addresses.append(normalized)
    return addresses


def _extract_body(msg: Message) -> str:
    """Extract the text body from an email message (handles multipart)."""
    if msg.is_multipart():
        # Walk through parts and find text/plain
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        return payload.decode(charset, errors="replace")
                except Exception:
                    continue
        # Fallback: try text/html
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/html":
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        html_text = payload.decode(charset, errors="replace")
                        # Strip HTML tags as rough conversion
                        return re.sub(r"<[^>]+>", " ", html_text).strip()
                except Exception:
                    continue
        return ""
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                return payload.decode(charset, errors="replace")
            # If decode=True returns None, try without decoding
            payload = msg.get_payload()
            if isinstance(payload, str):
                return payload
        except Exception:
            pass
        return ""


def _derive_thread_id(
    message_id: str, in_reply_to: str, references: str
) -> Optional[str]:
    """Derive a thread ID from message headers."""
    # Use the first message in the References chain as the thread root
    if references:
        ref_ids = references.strip().split()
        if ref_ids:
            first_ref = ref_ids[0].strip("<>")
            if first_ref:
                return first_ref

    # Fall back to in_reply_to
    if in_reply_to:
        return in_reply_to.strip("<>")

    # If no threading info, thread_id is the message itself
    if message_id:
        return message_id

    return None


def _subject_is_reply(subject: str) -> bool:
    """Check if a subject line indicates a reply or forward."""
    if not subject:
        return False
    stripped = subject.strip().lower()
    return bool(re.match(r"^(re|fw|fwd)\s*[:|\[]", stripped))


def _empty_email_dict() -> Dict[str, Any]:
    """Return an empty email dict template."""
    return {
        "subject": "",
        "sender": "",
        "recipients": [],
        "cc": [],
        "date": None,
        "body": "",
        "message_id": None,
        "in_reply_to": None,
        "references_field": None,
        "raw_headers": "",
        "thread_id": None,
        "is_reply": False,
    }
