"""
Executive AI Chief of Staff - Core Text Processing Module
Provides text extraction, NER, date parsing, sentiment analysis, and key phrase extraction
using NLTK, regex patterns, and lightweight heuristics. All processing runs locally.
"""

import logging
import math
import re
import string
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import nltk

# ---------------------------------------------------------------------------
# Ensure required NLTK data is available
# ---------------------------------------------------------------------------
_NLTK_RESOURCES = [
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker",
    "maxent_ne_chunker_tab",
    "words",
    "stopwords",
]

for _res in _NLTK_RESOURCES:
    try:
        nltk.data.find(f"tokenizers/{_res}" if "punkt" in _res else _res)
    except LookupError:
        try:
            nltk.download(_res, quiet=True)
        except Exception:
            pass

from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.tree import Tree

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentiment lexicons  (lightweight, no API)
# ---------------------------------------------------------------------------
_POSITIVE_WORDS = frozenset([
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "outstanding", "impressive", "pleased", "happy", "delighted", "thrilled",
    "successful", "positive", "strong", "improve", "improved", "growth",
    "profit", "gain", "opportunity", "congratulations", "well done", "terrific",
    "superb", "brilliant", "perfect", "love", "thankful", "grateful",
    "appreciate", "excited", "promising", "favorable", "encouraged", "confident",
    "optimistic", "achieve", "achievement", "award", "benefit", "boost",
    "celebrate", "commend", "effective", "efficient", "exceptional", "exceeded",
])

_NEGATIVE_WORDS = frozenset([
    "bad", "poor", "terrible", "horrible", "awful", "disappointing",
    "disappointed", "failure", "fail", "failed", "issue", "problem",
    "concern", "risk", "loss", "decline", "decrease", "drop",
    "threat", "danger", "crisis", "urgent", "critical", "severe",
    "delay", "overdue", "complaint", "unhappy", "dissatisfied", "frustrated",
    "angry", "unfortunately", "regret", "sorry", "mistake", "error",
    "penalty", "fine", "violation", "breach", "lawsuit", "lawsuit",
    "damage", "harm", "negative", "worst", "trouble", "worry",
    "anxious", "alarm", "deficit", "shortfall", "underperform",
])

# ---------------------------------------------------------------------------
# Urgency / action words
# ---------------------------------------------------------------------------
_ACTION_TRIGGERS = [
    r"please\s+(?:review|send|complete|submit|prepare|arrange|provide|update|confirm|approve|forward|share|check|handle|process|follow\s*up|coordinate|schedule|address|finalize|respond)",
    r"(?:need|needs)\s+(?:you\s+)?to\b",
    r"action\s+required",
    r"(?:could|can|would)\s+you\s+(?:please\s+)?(?:\w+)",
    r"(?:kindly|please)\s+(?:ensure|make\s+sure|let\s+(?:me|us)\s+know)",
    r"(?:must|shall|should)\s+(?:be\s+)?(?:completed|done|submitted|sent|reviewed|resolved)",
    r"your\s+(?:immediate\s+)?(?:attention|action|response|input|review|approval)\s+(?:is\s+)?(?:needed|required|requested)",
    r"(?:i|we)\s+(?:am|are)\s+(?:expecting|waiting\s+for|looking\s+forward\s+to)",
    r"deadline\s+(?:is|for)",
    r"(?:by|before)\s+(?:end\s+of\s+(?:day|week|month)|tomorrow|friday|monday|EOD|COB)",
    r"follow\s*[\-\s]*up\s+(?:on|with|regarding)",
    r"request(?:ing)?\s+(?:your|that\s+you)",
    r"assign(?:ed|ing)?\s+to\s+you",
]

_ACTION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _ACTION_TRIGGERS]

# ---------------------------------------------------------------------------
# Date patterns
# ---------------------------------------------------------------------------
_DAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]
_MONTH_ABBR = [m[:3] for m in _MONTH_NAMES]

_DATE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # ISO / slash dates: 2024-03-15, 03/15/2024, 15-Mar-2024
    (re.compile(r"\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b"), "explicit_date"),
    (re.compile(r"\b(\d{4}[/\-]\d{1,2}[/\-]\d{1,2})\b"), "explicit_date"),
    (re.compile(
        r"\b(\d{1,2})\s*(?:st|nd|rd|th)?\s+(?:of\s+)?("
        + "|".join(_MONTH_NAMES + _MONTH_ABBR)
        + r")[\s,]*(\d{2,4})?\b",
        re.IGNORECASE,
    ), "day_month_year"),
    (re.compile(
        r"\b(" + "|".join(_MONTH_NAMES + _MONTH_ABBR)
        + r")\s+(\d{1,2})(?:st|nd|rd|th)?[\s,]*(\d{2,4})?\b",
        re.IGNORECASE,
    ), "month_day_year"),
    # Relative: next Monday, this Friday, by Wednesday
    (re.compile(
        r"\b(?:next|this|coming|by|before)\s+(" + "|".join(_DAY_NAMES) + r")\b",
        re.IGNORECASE,
    ), "relative_day"),
    # Tomorrow, today, yesterday
    (re.compile(r"\b(today|tomorrow|yesterday)\b", re.IGNORECASE), "relative_simple"),
    # Within X hours/days/weeks
    (re.compile(r"\bwithin\s+(\d+)\s+(hour|day|week|month|business\s+day)s?\b", re.IGNORECASE), "within"),
    # End of period
    (re.compile(r"\b(?:by\s+)?(?:end\s+of|EOD|COB|EOM|EOW|EOQ|EOY)\s*(?:the\s+)?(\w*)\b", re.IGNORECASE), "end_of_period"),
    (re.compile(r"\b(EOD|COB|EOM|EOW|EOQ|EOY)\b", re.IGNORECASE), "end_of_period_abbrev"),
    # Q1, Q2, Q3, Q4
    (re.compile(r"\b(Q[1-4])\s*(\d{2,4})?\b", re.IGNORECASE), "quarter"),
    # In X days/weeks/months
    (re.compile(r"\bin\s+(\d+)\s+(day|week|month|hour)s?\b", re.IGNORECASE), "in_period"),
    # By + date-like
    (re.compile(r"\bby\s+(\d{1,2}[/\-]\d{1,2}(?:[/\-]\d{2,4})?)\b", re.IGNORECASE), "by_date"),
]

# ---------------------------------------------------------------------------
# Monetary patterns
# ---------------------------------------------------------------------------
_MONEY_PATTERNS = [
    re.compile(
        r"(?P<currency>[$€£¥₹])\s*(?P<amount>\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})?)"
        r"(?:\s*(?P<scale>million|billion|thousand|[MBKmk])\b)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<amount>\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})?)\s*"
        r"(?P<currency>USD|EUR|GBP|JPY|INR|CAD|AUD|CHF|dollars?|euros?|pounds?|rupees?)"
        r"(?:\s*(?P<scale>million|billion|thousand|[MBKmk])\b)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<currency>[$€£¥₹])\s*(?P<amount>\d+(?:\.\d+)?)\s*"
        r"(?P<scale>million|billion|thousand|[MBKmk])\b",
        re.IGNORECASE,
    ),
]

_CURRENCY_MAP = {
    "$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR",
    "usd": "USD", "eur": "EUR", "gbp": "GBP", "jpy": "JPY", "inr": "INR",
    "cad": "CAD", "aud": "AUD", "chf": "CHF",
    "dollar": "USD", "dollars": "USD", "euro": "EUR", "euros": "EUR",
    "pound": "GBP", "pounds": "GBP", "rupee": "INR", "rupees": "INR",
}

_SCALE_MAP = {
    "thousand": 1_000, "k": 1_000, "K": 1_000,
    "million": 1_000_000, "m": 1_000_000, "M": 1_000_000,
    "billion": 1_000_000_000, "b": 1_000_000_000, "B": 1_000_000_000,
}


class TextProcessor:
    """Core text processing engine for the Executive AI Chief of Staff.

    Uses NLTK for tokenization, POS tagging, and NER; regex patterns for
    dates, monetary values, action items; and a simple lexicon for sentiment.
    All operations run locally — no external API calls.
    """

    def __init__(self) -> None:
        """Initialise the TextProcessor, loading NLTK resources and stopwords."""
        try:
            self._stopwords: set = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            self._stopwords = set(stopwords.words("english"))
        logger.info("TextProcessor initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_sentences(self, text: str) -> List[str]:
        """Split *text* into a list of sentences using NLTK sent_tokenize.

        Args:
            text: Raw input text.

        Returns:
            List of sentence strings.
        """
        if not text or not text.strip():
            return []
        try:
            sentences = nltk.sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as exc:
            logger.warning("Sentence tokenization failed, falling back to regex: %s", exc)
            return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

    def extract_named_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities from *text* using NLTK NER and supplemental regex.

        Returns:
            List of dicts with keys ``entity`` and ``type`` (PERSON, ORGANIZATION,
            GPE, DATE, MONEY, EMAIL, PHONE, etc.).
        """
        if not text or not text.strip():
            return []

        entities: List[Dict[str, str]] = []
        seen: set = set()

        # --- NLTK chunker-based NER ---
        try:
            sentences = self.extract_sentences(text)
            for sent in sentences:
                tokens = word_tokenize(sent)
                tagged = pos_tag(tokens)
                tree = ne_chunk(tagged, binary=False)
                for subtree in tree:
                    if isinstance(subtree, Tree):
                        entity_text = " ".join(w for w, _ in subtree.leaves())
                        entity_type = subtree.label()
                        key = (entity_text.lower(), entity_type)
                        if key not in seen:
                            seen.add(key)
                            entities.append({"entity": entity_text, "type": entity_type})
        except Exception as exc:
            logger.warning("NLTK NER failed: %s", exc)

        # --- Regex-based supplemental entities ---
        # Email addresses
        for m in re.finditer(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text):
            val = m.group()
            key = (val.lower(), "EMAIL")
            if key not in seen:
                seen.add(key)
                entities.append({"entity": val, "type": "EMAIL"})

        # Phone numbers
        for m in re.finditer(r"\+?\d[\d\-\s().]{7,}\d", text):
            val = m.group().strip()
            if len(re.sub(r"\D", "", val)) >= 7:
                key = (val, "PHONE")
                if key not in seen:
                    seen.add(key)
                    entities.append({"entity": val, "type": "PHONE"})

        # Monetary values (tagged as MONEY)
        for mv in self.extract_monetary_values(text):
            val = mv.get("context", mv.get("amount", ""))
            key = (str(val).lower(), "MONEY")
            if key not in seen:
                seen.add(key)
                entities.append({"entity": str(val), "type": "MONEY"})

        return entities

    def extract_dates(self, text: str) -> List[Dict[str, Any]]:
        """Detect date/time references in *text* and attempt to parse them.

        Returns:
            List of dicts with ``date_text`` (matched span) and ``parsed_date``
            (a ``datetime`` or ``None`` if parsing was not possible).
        """
        if not text or not text.strip():
            return []

        results: List[Dict[str, Any]] = []
        seen_spans: set = set()
        now = datetime.now()

        for pattern, kind in _DATE_PATTERNS:
            for m in pattern.finditer(text):
                date_text = m.group().strip()
                if date_text.lower() in seen_spans:
                    continue
                seen_spans.add(date_text.lower())
                parsed = self._try_parse_date(m, kind, now)
                results.append({"date_text": date_text, "parsed_date": parsed})

        return results

    def extract_monetary_values(self, text: str) -> List[Dict[str, Any]]:
        """Extract monetary values from *text*.

        Returns:
            List of dicts with ``amount`` (float), ``currency`` (str),
            and ``context`` (surrounding text snippet).
        """
        if not text or not text.strip():
            return []

        results: List[Dict[str, Any]] = []
        seen: set = set()

        for pattern in _MONEY_PATTERNS:
            for m in pattern.finditer(text):
                raw_amount = m.group("amount").replace(",", "").replace(" ", "")
                try:
                    amount = float(raw_amount)
                except ValueError:
                    continue
                # Apply scale
                scale_str = m.groupdict().get("scale", None)
                if scale_str:
                    amount *= _SCALE_MAP.get(scale_str.lower(), _SCALE_MAP.get(scale_str, 1))
                currency_raw = m.group("currency").strip()
                currency = _CURRENCY_MAP.get(currency_raw.lower(), currency_raw.upper())
                # Context: ±40 chars around match
                start = max(0, m.start() - 40)
                end = min(len(text), m.end() + 40)
                context = text[start:end].strip()
                key = (amount, currency)
                if key not in seen:
                    seen.add(key)
                    results.append({
                        "amount": amount,
                        "currency": currency,
                        "context": context,
                    })

        return results

    def extract_action_items(self, text: str) -> List[Dict[str, Optional[str]]]:
        """Detect action-item phrases in *text*.

        Returns:
            List of dicts with ``action`` (the matched phrase / sentence),
            ``assignee`` (best-guess person, or ``None``), and ``deadline``
            (detected deadline text near the action, or ``None``).
        """
        if not text or not text.strip():
            return []

        sentences = self.extract_sentences(text)
        actions: List[Dict[str, Optional[str]]] = []
        seen: set = set()

        for sent in sentences:
            for pat in _ACTION_PATTERNS:
                m = pat.search(sent)
                if m and sent not in seen:
                    seen.add(sent)
                    assignee = self._guess_assignee(sent)
                    deadline = self._find_nearby_deadline(sent)
                    actions.append({
                        "action": sent.strip(),
                        "assignee": assignee,
                        "deadline": deadline,
                    })
                    break  # one match per sentence is enough

        return actions

    def extract_questions(self, text: str) -> List[str]:
        """Return all interrogative sentences from *text*.

        Detection uses ``?`` termination as well as sentences starting with
        interrogative words (who, what, when, where, why, how, …).
        """
        if not text or not text.strip():
            return []

        sentences = self.extract_sentences(text)
        questions: List[str] = []
        interrogatives = {"who", "what", "when", "where", "why", "how", "which",
                          "whom", "whose", "is", "are", "do", "does", "did",
                          "can", "could", "will", "would", "shall", "should",
                          "have", "has", "had", "may", "might"}

        for sent in sentences:
            sent_stripped = sent.strip()
            if sent_stripped.endswith("?"):
                questions.append(sent_stripped)
            elif sent_stripped:
                first_word = sent_stripped.split()[0].lower().rstrip(",:;")
                if first_word in interrogatives and len(sent_stripped.split()) > 2:
                    questions.append(sent_stripped)

        return questions

    def detect_sentiment(self, text: str) -> Dict[str, Any]:
        """Simple lexicon-based sentiment analysis.

        Returns:
            Dict with ``polarity`` (float, −1 to +1), ``label`` (``positive``,
            ``negative``, or ``neutral``), ``positive_count``, ``negative_count``.
        """
        if not text or not text.strip():
            return {"polarity": 0.0, "label": "neutral", "positive_count": 0, "negative_count": 0}

        words = [w.lower().strip(string.punctuation) for w in text.split()]
        pos_count = sum(1 for w in words if w in _POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in _NEGATIVE_WORDS)
        total = pos_count + neg_count

        if total == 0:
            polarity = 0.0
        else:
            polarity = (pos_count - neg_count) / total

        if polarity > 0.1:
            label = "positive"
        elif polarity < -0.1:
            label = "negative"
        else:
            label = "neutral"

        return {
            "polarity": round(polarity, 3),
            "label": label,
            "positive_count": pos_count,
            "negative_count": neg_count,
        }

    def extract_key_phrases(self, text: str, top_n: int = 5) -> List[str]:
        """Extract key phrases using TF-based scoring with POS filtering.

        Keeps noun-phrases and high-TF terms, filters stopwords and short tokens.

        Args:
            text: Input text.
            top_n: Maximum number of key phrases to return.

        Returns:
            List of key-phrase strings, ordered by relevance.
        """
        if not text or not text.strip():
            return []

        # Tokenize and POS-tag
        try:
            tokens = word_tokenize(text.lower())
            tagged = pos_tag(tokens)
        except Exception:
            tokens = text.lower().split()
            tagged = [(t, "NN") for t in tokens]

        # Keep nouns and adjectives
        valid_pos = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"}
        filtered = [
            w for w, tag in tagged
            if tag in valid_pos
            and w not in self._stopwords
            and len(w) > 2
            and w.isalpha()
        ]

        if not filtered:
            return []

        # Build bigrams for noun-phrases
        phrases: List[str] = list(filtered)
        for i in range(len(filtered) - 1):
            phrases.append(f"{filtered[i]} {filtered[i + 1]}")

        freq = Counter(phrases)
        # Score: TF * length-bonus
        scored: List[Tuple[str, float]] = []
        for phrase, count in freq.items():
            word_count = len(phrase.split())
            length_bonus = 1.0 + 0.5 * (word_count - 1)
            score = count * length_bonus
            scored.append((phrase, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # De-duplicate: remove single-word phrases already covered by bigram
        selected: List[str] = []
        selected_lower: set = set()
        for phrase, _ in scored:
            if phrase.lower() not in selected_lower:
                # Skip single word if it's part of an already-selected bigram
                skip = False
                if " " not in phrase:
                    for sel in selected:
                        if phrase in sel.split():
                            skip = True
                            break
                if not skip:
                    selected.append(phrase)
                    selected_lower.add(phrase.lower())
                if len(selected) >= top_n:
                    break

        return selected

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _try_parse_date(match: re.Match, kind: str, now: datetime) -> Optional[datetime]:
        """Attempt to resolve a regex match into a concrete datetime."""
        try:
            if kind == "explicit_date":
                text = match.group(1) if match.lastindex else match.group()
                for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y",
                            "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%y", "%d/%m/%y"):
                    try:
                        return datetime.strptime(text, fmt)
                    except ValueError:
                        continue

            elif kind in ("day_month_year", "month_day_year"):
                groups = match.groups()
                if kind == "day_month_year":
                    day_s, month_s = groups[0], groups[1]
                    year_s = groups[2] if len(groups) > 2 else None
                else:
                    month_s, day_s = groups[0], groups[1]
                    year_s = groups[2] if len(groups) > 2 else None
                month_lower = month_s.lower()[:3]
                month_idx = None
                for i, mn in enumerate(_MONTH_ABBR):
                    if mn == month_lower:
                        month_idx = i + 1
                        break
                if month_idx is None:
                    return None
                day = int(day_s)
                if year_s:
                    year = int(year_s)
                    if year < 100:
                        year += 2000
                else:
                    year = now.year
                return datetime(year, month_idx, day)

            elif kind == "relative_day":
                day_name = match.group(1).lower()
                target_idx = _DAY_NAMES.index(day_name)
                current_idx = now.weekday()
                delta = (target_idx - current_idx) % 7
                if delta == 0:
                    delta = 7
                return now + timedelta(days=delta)

            elif kind == "relative_simple":
                word = match.group(1).lower()
                if word == "today":
                    return now.replace(hour=23, minute=59, second=59)
                elif word == "tomorrow":
                    return (now + timedelta(days=1)).replace(hour=23, minute=59, second=59)
                elif word == "yesterday":
                    return (now - timedelta(days=1)).replace(hour=23, minute=59, second=59)

            elif kind == "within":
                amount = int(match.group(1))
                unit = match.group(2).lower()
                if "hour" in unit:
                    return now + timedelta(hours=amount)
                elif "day" in unit:
                    return now + timedelta(days=amount)
                elif "week" in unit:
                    return now + timedelta(weeks=amount)
                elif "month" in unit:
                    return now + timedelta(days=amount * 30)

            elif kind in ("end_of_period", "end_of_period_abbrev"):
                text_lower = match.group().lower()
                if "eod" in text_lower or "cob" in text_lower or "end of day" in text_lower:
                    return now.replace(hour=17, minute=0, second=0, microsecond=0)
                elif "eow" in text_lower or "end of week" in text_lower:
                    days_until_friday = (4 - now.weekday()) % 7
                    if days_until_friday == 0 and now.hour >= 17:
                        days_until_friday = 7
                    return (now + timedelta(days=days_until_friday)).replace(
                        hour=17, minute=0, second=0, microsecond=0
                    )
                elif "eom" in text_lower or "end of month" in text_lower:
                    if now.month == 12:
                        return datetime(now.year + 1, 1, 1) - timedelta(seconds=1)
                    else:
                        return datetime(now.year, now.month + 1, 1) - timedelta(seconds=1)
                elif "eoq" in text_lower or "end of quarter" in text_lower:
                    quarter_end_month = ((now.month - 1) // 3 + 1) * 3
                    if quarter_end_month == 12:
                        return datetime(now.year + 1, 1, 1) - timedelta(seconds=1)
                    return datetime(now.year, quarter_end_month + 1, 1) - timedelta(seconds=1)
                elif "eoy" in text_lower or "end of year" in text_lower:
                    return datetime(now.year + 1, 1, 1) - timedelta(seconds=1)

            elif kind == "quarter":
                q = int(match.group(1)[1])
                year_s = match.group(2) if match.lastindex and match.lastindex >= 2 else None
                year = int(year_s) if year_s else now.year
                if year < 100:
                    year += 2000
                quarter_end_month = q * 3
                if quarter_end_month == 12:
                    return datetime(year + 1, 1, 1) - timedelta(seconds=1)
                return datetime(year, quarter_end_month + 1, 1) - timedelta(seconds=1)

            elif kind == "in_period":
                amount = int(match.group(1))
                unit = match.group(2).lower()
                if "hour" in unit:
                    return now + timedelta(hours=amount)
                elif "day" in unit:
                    return now + timedelta(days=amount)
                elif "week" in unit:
                    return now + timedelta(weeks=amount)
                elif "month" in unit:
                    return now + timedelta(days=amount * 30)

            elif kind == "by_date":
                date_str = match.group(1)
                for fmt in ("%m/%d/%Y", "%m/%d/%y", "%m/%d", "%d/%m/%Y", "%d/%m"):
                    try:
                        parsed = datetime.strptime(date_str, fmt)
                        if parsed.year == 1900:
                            parsed = parsed.replace(year=now.year)
                        return parsed
                    except ValueError:
                        continue

        except Exception as exc:
            logger.debug("Date parsing failed for kind=%s: %s", kind, exc)

        return None

    @staticmethod
    def _guess_assignee(sentence: str) -> Optional[str]:
        """Best-effort extraction of an assignee from an action sentence."""
        # Pattern: <Name>, please / Dear <Name>
        m = re.search(r"(?:Dear|Hi|Hey|Hello)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", sentence)
        if m:
            return m.group(1)
        # @mention style
        m = re.search(r"@(\w+)", sentence)
        if m:
            return m.group(1)
        # "assigned to <Name>"
        m = re.search(r"assigned?\s+to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", sentence)
        if m:
            return m.group(1)
        # "<Name> to review/send/..."
        m = re.search(r"([A-Z][a-z]+)\s+(?:to|will|should|needs?\s+to)\s+\w+", sentence)
        if m:
            name = m.group(1)
            # Avoid common false positives
            if name.lower() not in {"the", "this", "that", "please", "also", "just",
                                     "but", "and", "for", "with", "from", "our", "your"}:
                return name
        return None

    def _find_nearby_deadline(self, sentence: str) -> Optional[str]:
        """Check if a deadline expression exists in the sentence."""
        dates = self.extract_dates(sentence)
        if dates:
            return dates[0]["date_text"]
        # Look for inline temporal cues
        m = re.search(
            r"(?:by|before|until|due)\s+(\w[\w\s,/\-]{2,20})",
            sentence,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).strip().rstrip(".,;")
        return None
