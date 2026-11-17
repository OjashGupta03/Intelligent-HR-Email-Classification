"""
Executive AI Chief of Staff - Utility Helper Functions
Common utility functions used across all modules.
"""

import functools
import hashlib
import json
import logging
import re
import time
from datetime import datetime
from html.parser import HTMLParser
from io import StringIO
from pathlib import Path
from typing import Any, Generator, Optional, TypeVar, Union

import numpy as np

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(name: str, level: str = "INFO") -> logging.Logger:
    """
    Configure and return a logger with a consistent format.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        Configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def safe_json_loads(text: str, default: Any = None) -> Any:
    """
    Parse a JSON string, returning *default* on any failure.

    Args:
        text: Raw JSON string.
        default: Fallback value when parsing fails.

    Returns:
        Parsed Python object, or *default*.
    """
    if not text:
        return default
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


def safe_json_dumps(obj: Any) -> str:
    """
    Serialise an object to a JSON string safely.

    Handles non-serialisable types by converting them to strings.

    Args:
        obj: Python object to serialise.

    Returns:
        JSON string.
    """
    def _default(o: Any) -> str:
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, set):
            return list(o)
        if isinstance(o, Path):
            return str(o)
        return str(o)

    try:
        return json.dumps(obj, default=_default, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(str(obj))


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

_DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%d-%b-%Y",
    "%d %b %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    # RFC 2822 (common in email headers)
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S",
    "%d %b %Y %H:%M:%S %z",
]


def format_date(dt: datetime, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """
    Format a ``datetime`` object to a human-readable string.

    Args:
        dt: The datetime to format.
        fmt: ``strftime`` format string.

    Returns:
        Formatted date string, or ``""`` if *dt* is ``None``.
    """
    if dt is None:
        return ""
    try:
        return dt.strftime(fmt)
    except (AttributeError, ValueError):
        return str(dt)


def parse_date(text: str) -> Optional[datetime]:
    """
    Parse a date string by trying multiple common formats.

    Args:
        text: Date string to parse.

    Returns:
        ``datetime`` object, or ``None`` if no format matched.
    """
    if not text or not isinstance(text, str):
        return None

    text = text.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except (ValueError, TypeError):
            continue
    return None


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------

def truncate_string(s: str, max_len: int = 200) -> str:
    """
    Truncate *s* to at most *max_len* characters, appending ``...`` if cut.

    Args:
        s: Input string.
        max_len: Maximum allowed length.

    Returns:
        Truncated string.
    """
    if not s:
        return ""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


class _HTMLStripper(HTMLParser):
    """Internal HTML tag stripper."""

    def __init__(self) -> None:
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self._fed: list[str] = []

    def handle_data(self, data: str) -> None:  # noqa: D401
        self._fed.append(data)

    def get_data(self) -> str:
        return "".join(self._fed)


def clean_html(text: str) -> str:
    """
    Strip HTML tags from *text* and return plain text.

    Args:
        text: Possibly-HTML string.

    Returns:
        Plain text without HTML tags.
    """
    if not text:
        return ""
    try:
        stripper = _HTMLStripper()
        stripper.feed(text)
        cleaned = stripper.get_data()
        # Collapse whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
    except Exception:
        # Fallback: simple regex strip
        return re.sub(r"<[^>]+>", "", text).strip()


# ---------------------------------------------------------------------------
# Math / vector helpers
# ---------------------------------------------------------------------------

def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity in [-1, 1], or ``0.0`` on error.
    """
    try:
        vec1 = np.asarray(vec1, dtype=np.float32).flatten()
        vec2 = np.asarray(vec2, dtype=np.float32).flatten()
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    except Exception:
        return 0.0


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Safely divide *a* by *b*, returning *default* if *b* is zero.

    Args:
        a: Numerator.
        b: Denominator.
        default: Value to return when *b* is zero.

    Returns:
        ``a / b`` or *default*.
    """
    try:
        if b == 0:
            return default
        return float(a) / float(b)
    except (TypeError, ValueError, ZeroDivisionError):
        return default


# ---------------------------------------------------------------------------
# Iteration helpers
# ---------------------------------------------------------------------------

def batch_iterator(items: list, batch_size: int) -> Generator[list, None, None]:
    """
    Yield successive batches of *batch_size* from *items*.

    Args:
        items: The full list of items.
        batch_size: Size of each batch.

    Yields:
        Lists of up to *batch_size* items.
    """
    if batch_size < 1:
        batch_size = 1
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# ---------------------------------------------------------------------------
# Decorator helpers
# ---------------------------------------------------------------------------

def timer_decorator(func):
    """
    Decorator that logs the execution time of the wrapped function.

    Uses the function's module-level logger when available, otherwise
    creates a logger named ``timer``.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__ or "timer")
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            logger.info(
                "%s.%s executed in %.3f seconds",
                func.__module__,
                func.__qualname__,
                elapsed,
            )
    return wrapper


# ---------------------------------------------------------------------------
# File / path helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it (and parents) if necessary.

    Args:
        path: Directory path.

    Returns:
        Resolved ``Path`` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


# ---------------------------------------------------------------------------
# Email helpers
# ---------------------------------------------------------------------------

def normalize_email(email_str: str) -> str:
    """
    Normalise an email address: lower-case, strip whitespace, remove display name.

    Args:
        email_str: Raw email string, possibly ``"Name <user@example.com>"``.

    Returns:
        Clean lower-case email address.
    """
    if not email_str:
        return ""
    email_str = email_str.strip()
    # Extract address from "Name <email>" format
    match = re.search(r"<([^>]+)>", email_str)
    if match:
        email_str = match.group(1)
    return email_str.lower().strip()


def extract_domain(email_str: str) -> str:
    """
    Extract the domain part from an email address.

    Args:
        email_str: Email address string.

    Returns:
        Domain string, or ``""`` if extraction fails.
    """
    normalised = normalize_email(email_str)
    if "@" in normalised:
        return normalised.split("@", 1)[1]
    return ""


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def hash_text(text: str) -> str:
    """
    Return a SHA-256 hex digest of *text* for deduplication.

    Args:
        text: Input text.

    Returns:
        64-character hex string.
    """
    if not text:
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
