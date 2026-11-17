"""
Executive AI Chief of Staff - Feature Extractor
Builds feature vectors for ML classification combining structural,
linguistic, and content features.
"""
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.preprocessor import tokenize, remove_stopwords

logger = logging.getLogger(__name__)

_URGENCY_WORDS = [
    "urgent", "asap", "immediately", "critical", "emergency", "right away",
    "time-sensitive", "priority", "deadline", "today", "eod", "now",
]
_LEGAL_TERMS = [
    "lawsuit", "litigation", "legal", "compliance", "regulatory", "sec", "irs",
    "subpoena", "investigation", "breach", "violation", "audit",
]
_FINANCIAL_TERMS = [
    "revenue", "budget", "cost", "invoice", "payment", "financial", "profit",
    "loss", "quarterly", "annual", "million", "billion", "dollar",
]
_ACTION_TERMS = [
    "please review", "action required", "your approval", "your input",
    "need you to", "please confirm", "please respond", "please provide",
]
_MEETING_TERMS = ["meeting", "call", "conference", "agenda", "calendar", "schedule"]
_PROJECT_TERMS = ["project", "milestone", "deliverable", "timeline", "phase", "sprint"]
_CUSTOMER_TERMS = ["customer", "client", "account", "user", "subscriber", "partner"]


class FeatureExtractor:
    """
    Extracts a fixed-length feature vector from an email dict.
    Features are deterministic and interpretable.
    """

    def __init__(self):
        self._feature_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_structural_features(self, email_dict: Dict) -> Dict[str, float]:
        body = email_dict.get("body") or ""
        subject = email_dict.get("subject") or ""
        recipients = email_dict.get("recipients") or []
        cc = email_dict.get("cc") or []
        date_str = email_dict.get("date") or ""

        hour, dow, is_weekend = 12, 1, 0
        if date_str:
            try:
                dt = datetime.fromisoformat(str(date_str)) if not isinstance(date_str, datetime) else date_str
                hour = dt.hour
                dow = dt.weekday()
                is_weekend = int(dow >= 5)
            except Exception:
                pass

        word_count = len(body.split())
        n_recipients = len(recipients) if isinstance(recipients, list) else 1
        n_cc = len(cc) if isinstance(cc, list) else 0
        subject_lower = subject.lower()

        return {
            "word_count": min(word_count, 2000),
            "word_count_log": float(np.log1p(word_count)),
            "char_count_log": float(np.log1p(len(body))),
            "sentence_count": min(body.count(".") + body.count("!") + body.count("?"), 100),
            "num_recipients": min(n_recipients, 20),
            "num_cc": min(n_cc, 20),
            "has_many_recipients": float(n_recipients > 5),
            "is_reply": float(email_dict.get("is_reply") or subject_lower.startswith("re:")),
            "is_forward": float("fwd:" in subject_lower or "forwarded" in body[:200].lower()),
            "hour_of_day": hour,
            "day_of_week": dow,
            "is_weekend": is_weekend,
            "is_early_morning": float(hour < 7),
            "is_after_hours": float(hour > 19),
            "subject_length": len(subject),
            "has_attachment": float(any(w in body.lower() for w in ["attached", "attachment", "see attached", "please find"])),
        }

    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        words = text.split()
        total = max(len(words), 1)
        caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        q_count = text.count("?")
        excl_count = text.count("!")

        return {
            "question_count": min(q_count, 10),
            "has_question": float(q_count > 0),
            "exclamation_count": min(excl_count, 10),
            "caps_ratio": min(caps_words / total, 1.0),
            "urgency_word_count": sum(1 for w in _URGENCY_WORDS if w in text.lower()),
            "avg_word_length": float(np.mean([len(w) for w in words]) if words else 4),
        }

    def extract_content_features(self, text: str) -> Dict[str, float]:
        tl = text.lower()
        return {
            "has_monetary": float(bool(re.search(r"\$[\d,]+|million|billion", text, re.IGNORECASE))),
            "has_financial_terms": float(sum(1 for t in _FINANCIAL_TERMS if t in tl) >= 2),
            "has_deadline": float(bool(re.search(r"\b(by|before|due|deadline)\b.{0,60}(today|tomorrow|eod|friday|monday|\d{1,2}/\d{1,2})", tl))),
            "has_legal_terms": float(sum(1 for t in _LEGAL_TERMS if t in tl) >= 1),
            "has_action_request": float(any(t in tl for t in _ACTION_TERMS)),
            "has_meeting_reference": float(any(t in tl for t in _MEETING_TERMS)),
            "has_project_reference": float(any(t in tl for t in _PROJECT_TERMS)),
            "has_customer_mention": float(any(t in tl for t in _CUSTOMER_TERMS)),
            "has_risk_words": float(bool(re.search(r"\b(risk|threat|danger|concern|issue|problem|crisis)\b", tl))),
            "has_commitment": float(bool(re.search(r"\b(i will|i'll|we will|we'll|i promise|i commit)\b", tl))),
            "urgency_word_count_norm": min(sum(1 for w in _URGENCY_WORDS if w in tl) / 10.0, 1.0),
        }

    def extract_sender_features(self, sender: str, db_manager=None) -> Dict[str, float]:
        import sys, os
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        try:
            from config.settings import VIP_SENDERS, VIP_DOMAINS
        except Exception:
            VIP_SENDERS, VIP_DOMAINS = {}, {}

        sender_lower = (sender or "").lower()
        domain = sender_lower.split("@")[-1] if "@" in sender_lower else ""

        is_vip = float(sender_lower in VIP_SENDERS)
        vip_score = VIP_SENDERS.get(sender_lower, 0) / 100.0
        domain_boost = VIP_DOMAINS.get(domain, 0) / 100.0

        # Simple domain importance heuristic
        gov_domain = float(domain.endswith(".gov"))
        edu_domain = float(domain.endswith(".edu"))

        return {
            "is_vip": is_vip,
            "vip_score": vip_score,
            "domain_boost": domain_boost,
            "is_gov_domain": gov_domain,
            "is_edu_domain": edu_domain,
        }

    def build_feature_vector(self, email_dict: Dict, text_features: Optional[Dict] = None) -> np.ndarray:
        """Combine all features into a 1D numpy array."""
        subject = email_dict.get("subject") or ""
        body = email_dict.get("body") or ""
        text = f"{subject} {body}"

        structural = self.extract_structural_features(email_dict)
        linguistic = self.extract_linguistic_features(text)
        content = self.extract_content_features(text)
        sender = self.extract_sender_features(email_dict.get("sender", ""))

        all_features = {}
        all_features.update(structural)
        all_features.update(linguistic)
        all_features.update(content)
        all_features.update(sender)

        if text_features:
            all_features.update(text_features)

        # Deterministic ordering
        self._feature_names = sorted(all_features.keys())
        return np.array([all_features[k] for k in self._feature_names], dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        if self._feature_names is None:
            # Build with dummy email to populate names
            dummy = {"subject": "test", "body": "test", "sender": "test@test.com", "date": "2024-01-01"}
            self.build_feature_vector(dummy)
        return self._feature_names

