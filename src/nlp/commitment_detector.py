"""
Executive AI Chief of Staff - Commitment Detector
Detects implicit and explicit commitments in email text using rule-based
pattern matching. All detection is local - no external API calls.
"""
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 30+ commitment patterns
# ---------------------------------------------------------------------------
_COMMITMENT_PATTERNS: List[Dict] = [
    {"pattern": re.compile(r"I\s+will\s+send\s+(.{5,80}?)(?:\s+by\s+(.{3,40}))?[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.90},
    {"pattern": re.compile(r"I\s+will\s+provide\s+(.{5,80}?)(?:\s+by\s+(.{3,40}))?[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.88},
    {"pattern": re.compile(r"I\s+will\s+schedule\s+(.{5,80}?)(?:\s+by\s+(.{3,40}))?[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.85},
    {"pattern": re.compile(r"I\s+will\s+arrange\s+(.{5,80}?)(?:\s+by\s+(.{3,40}))?[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.85},
    {"pattern": re.compile(r"I\s+will\s+follow\s+up\s+(.{0,80}?)(?:\s+by\s+(.{3,40}))?[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.82},
    {"pattern": re.compile(r"I\s+will\s+review\s+(.{5,80}?)(?:\s+by\s+(.{3,40}))?[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.80},
    {"pattern": re.compile(r"I\s+will\s+prepare\s+(.{5,80}?)(?:\s+by\s+(.{3,40}))?[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.80},
    {"pattern": re.compile(r"I\s+will\s+draft\s+(.{5,80}?)(?:\s+by\s+(.{3,40}))?[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.80},
    {"pattern": re.compile(r"I\s+will\s+complete\s+(.{5,80}?)(?:\s+by\s+(.{3,40}))?[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.85},
    {"pattern": re.compile(r"I\s+will\s+discuss\s+(.{5,80}?)(?:\s+with\s+(.{3,40}))?[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.75},
    {"pattern": re.compile(r"I\s*'ll\s+send\s+(.{5,80}?)(?:\s+by\s+(.{3,40}))?[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.90},
    {"pattern": re.compile(r"I\s*'ll\s+have\s+(.{5,80}?)\s+(?:ready|done|completed)\s+by\s+(.{3,40})[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.92},
    {"pattern": re.compile(r"I\s*'ll\s+follow\s+up\s+(.{0,80}?)[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.80},
    {"pattern": re.compile(r"I\s*'ll\s+get\s+back\s+to\s+you\s+(.{0,60}?)[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.80},
    {"pattern": re.compile(r"I\s*'ll\s+look\s+into\s+(.{5,80}?)[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.75},
    {"pattern": re.compile(r"We\s+will\s+deliver\s+(.{5,80}?)(?:\s+by\s+(.{3,40}))?[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.90},
    {"pattern": re.compile(r"We\s+will\s+complete\s+(.{5,80}?)(?:\s+by\s+(.{3,40}))?[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.88},
    {"pattern": re.compile(r"We\s+commit\s+to\s+(.{5,80}?)(?:\s+by\s+(.{3,40}))?[.\n]", re.IGNORECASE), "type": "agreement", "confidence": 0.92},
    {"pattern": re.compile(r"We\s+promise\s+to\s+(.{5,80}?)[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.90},
    {"pattern": re.compile(r"we\s+should\s+complete\s+(.{5,80}?)(?:\s+by\s+(.{3,40}))?[.\n]", re.IGNORECASE), "type": "obligation", "confidence": 0.75},
    {"pattern": re.compile(r"please\s+arrange\s+(.{5,80}?)(?:\s+by\s+(.{3,40}))?[.\n]", re.IGNORECASE), "type": "request", "confidence": 0.82},
    {"pattern": re.compile(r"please\s+schedule\s+(.{5,80}?)(?:\s+for\s+(.{3,40}))?[.\n]", re.IGNORECASE), "type": "request", "confidence": 0.82},
    {"pattern": re.compile(r"please\s+ensure\s+(.{5,80}?)[.\n]", re.IGNORECASE), "type": "request", "confidence": 0.78},
    {"pattern": re.compile(r"please\s+confirm\s+(.{5,80}?)(?:\s+by\s+(.{3,40}))?[.\n]", re.IGNORECASE), "type": "request", "confidence": 0.80},
    {"pattern": re.compile(r"please\s+provide\s+(.{5,80}?)(?:\s+by\s+(.{3,40}))?[.\n]", re.IGNORECASE), "type": "request", "confidence": 0.80},
    {"pattern": re.compile(r"I\s+promise\s+to\s+(.{5,80}?)[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.95},
    {"pattern": re.compile(r"I\s+commit\s+to\s+(.{5,80}?)[.\n]", re.IGNORECASE), "type": "agreement", "confidence": 0.95},
    {"pattern": re.compile(r"we\s+agree(d)?\s+to\s+(.{5,80}?)(?:\s+by\s+(.{3,40}))?[.\n]", re.IGNORECASE), "type": "agreement", "confidence": 0.90},
    {"pattern": re.compile(r"I\s+plan\s+to\s+(.{5,80}?)(?:\s+by\s+(.{3,40}))?[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.70},
    {"pattern": re.compile(r"I\s+am\s+going\s+to\s+(.{5,80}?)(?:\s+by\s+(.{3,40}))?[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.72},
    {"pattern": re.compile(r"let\s+me\s+(?:know|check|look into|schedule|arrange)\s+(.{0,80}?)[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.68},
    {"pattern": re.compile(r"I\s+will\s+circle\s+back\s+(.{0,60}?)[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.75},
    {"pattern": re.compile(r"(?:it|this)\s+will\s+be\s+(?:ready|done|completed|delivered)\s+by\s+(.{3,40})[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.85},
    {"pattern": re.compile(r"I\s+will\s+call\s+(.{5,60}?)(?:\s+(?:on|at|by)\s+(.{3,40}))?[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.75},
    {"pattern": re.compile(r"I\s+will\s+meet\s+(.{5,60}?)(?:\s+(?:on|at|by)\s+(.{3,40}))?[.\n]", re.IGNORECASE), "type": "promise", "confidence": 0.75},
]

_DEADLINE_INDICATORS = re.compile(
    r"\b(?:by|before|on|no later than)\s+"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"today|tomorrow|eod|cob|end of (?:day|week|month)|"
    r"q[1-4]|next week|this (?:week|month)|\d{1,2}[/\-]\d{1,2})",
    re.IGNORECASE,
)

_PERSON_INDICATORS = re.compile(
    r"\b(?:John|Sarah|Robert|Emily|Michael|Jennifer|David|Amanda|"
    r"Christopher|Lisa|James|Patricia|Daniel|Nancy|Mark|Sandra|"
    r"Paul|Karen|Steven|Betty)\b",
    re.IGNORECASE,
)


class CommitmentDetector:
    """Detects commitments in email text using rule-based patterns."""

    def detect_commitments(self, text: str) -> List[Dict]:
        """
        Detect all commitments in the given text.
        Returns a list of commitment dicts.
        """
        if not text:
            return []

        # Split into sentences for context
        sentences = re.split(r"(?<=[.!?])\s+", text)
        seen_descriptions: set = set()
        commitments: List[Dict] = []

        for sent in sentences:
            for pattern_info in _COMMITMENT_PATTERNS:
                match = pattern_info["pattern"].search(sent)
                if not match:
                    continue

                # Extract description from first capture group
                groups = match.groups()
                description = groups[0].strip() if groups else sent.strip()
                deadline_text = groups[1].strip() if len(groups) > 1 and groups[1] else None

                # Try to find deadline in surrounding sentence
                if not deadline_text:
                    dl_match = _DEADLINE_INDICATORS.search(sent)
                    if dl_match:
                        deadline_text = dl_match.group(0)

                # Dedup based on description prefix
                key = description[:40].lower()
                if key in seen_descriptions:
                    continue
                seen_descriptions.add(key)

                responsible = self._extract_responsible_person(sent)
                confidence = self._assess_confidence(pattern_info, description, deadline_text)

                commitments.append({
                    "description": description,
                    "responsible_person": responsible,
                    "related_entity": None,
                    "deadline": deadline_text,
                    "commitment_type": pattern_info["type"],
                    "confidence": confidence,
                    "source_text": sent.strip(),
                })

        return commitments

    def _extract_responsible_person(self, sentence: str) -> str:
        """Try to identify who is responsible from the sentence."""
        # Check if sentence starts with I / We
        sentence_stripped = sentence.strip()
        if re.match(r"^I\s", sentence_stripped, re.IGNORECASE):
            return "Executive (self)"
        if re.match(r"^We\s", sentence_stripped, re.IGNORECASE):
            return "Team"
        # Named entity
        name_match = _PERSON_INDICATORS.search(sentence)
        if name_match:
            return name_match.group(0)
        if re.match(r"^Please\s", sentence_stripped, re.IGNORECASE):
            return "Recipient"
        return "Unknown"

    def _assess_confidence(self, pattern_info: Dict, description: str, deadline: Optional[str]) -> float:
        """Adjust confidence based on description quality and deadline presence."""
        confidence = pattern_info["confidence"]
        # Penalty for very short descriptions
        if len(description) < 10:
            confidence -= 0.15
        # Boost for having a deadline
        if deadline:
            confidence += 0.05
        return max(0.1, min(1.0, confidence))

    def classify_commitment_type(self, text: str) -> str:
        """Classify the type of commitment."""
        tl = text.lower()
        if any(w in tl for w in ["request", "please", "can you", "could you"]):
            return "request"
        if any(w in tl for w in ["agree", "agreed", "commitment", "commit"]):
            return "agreement"
        if any(w in tl for w in ["must", "should", "required", "obligated", "need to"]):
            return "obligation"
        return "promise"

    def check_completion_evidence(self, commitment_desc: str, newer_emails: List[Dict]) -> str:
        """
        Check if newer emails provide evidence that a commitment was fulfilled.
        Returns 'completed', 'pending', or 'uncertain'.
        """
        keywords = set(re.findall(r"\b\w{4,}\b", commitment_desc.lower()))
        for email in newer_emails:
            body = (email.get("body") or "").lower()
            # Look for completion signals
            if any(phrase in body for phrase in ["completed", "done", "finished", "delivered", "sent", "attached"]):
                # Check if it refers to same topic
                overlap = sum(1 for kw in keywords if kw in body)
                if overlap >= 2:
                    return "completed"
        return "pending"

