"""
Executive AI Chief of Staff - Urgency & Importance Detector
Treats urgency and importance as TWO SEPARATE DIMENSIONS (Eisenhower matrix).
"""
import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Urgency signal patterns (time pressure / immediacy)
# ---------------------------------------------------------------------------
_URGENCY_CRITICAL_PATTERNS = [
    re.compile(r"\b(asap|immediately|right away|right now|at once|this instant|urgent|emergency|critical priority|highest priority)\b", re.IGNORECASE),
    re.compile(r"\b(cannot wait|no delay|drop everything|stop everything|halt everything)\b", re.IGNORECASE),
    re.compile(r"\b(by (eod|cob|end of (day|business)|close of business)|due today|needed today|today is the deadline)\b", re.IGNORECASE),
    re.compile(r"\bdeadline\s+(is\s+)?today\b", re.IGNORECASE),
    re.compile(r"\bwithin\s+(the\s+next\s+)?(1\s*hour|2\s*hours|few\s+hours|next\s+hour)\b", re.IGNORECASE),
]
_URGENCY_HIGH_PATTERNS = [
    re.compile(r"\b(by\s+tomorrow|within\s+24\s+hours|within\s+48\s+hours|by\s+end\s+of\s+(week|tomorrow))\b", re.IGNORECASE),
    re.compile(r"\b(this\s+(morning|afternoon|evening)|before\s+(noon|lunch|end of day))\b", re.IGNORECASE),
    re.compile(r"\b(time.sensitive|time.critical|time is of the essence|time is running out)\b", re.IGNORECASE),
    re.compile(r"\b(needs? (your )?attention (immediately|now|today|urgently))\b", re.IGNORECASE),
    re.compile(r"\b(deadline\s+is\s+(tomorrow|this\s+(week|friday|monday)))\b", re.IGNORECASE),
    re.compile(r"\b(please\s+(respond|reply|confirm|approve)\s+(immediately|today|asap|urgently))\b", re.IGNORECASE),
]
_URGENCY_MEDIUM_PATTERNS = [
    re.compile(r"\b(by\s+(end\s+of\s+)?this\s+week|within\s+the\s+week|by\s+friday)\b", re.IGNORECASE),
    re.compile(r"\b(next\s+(business\s+day|day)|by\s+monday|early\s+next\s+week)\b", re.IGNORECASE),
    re.compile(r"\b(action\s+required|your\s+input\s+needed|follow.up\s+needed)\b", re.IGNORECASE),
    re.compile(r"\b(please\s+(review|provide|confirm)\s+(by|before|at your earliest))\b", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Importance signal patterns (business/strategic impact)
# ---------------------------------------------------------------------------
_IMPORTANCE_CRITICAL_PATTERNS = [
    re.compile(r"\b(board|shareholders?|investors?|acquisition|merger|ipo|sec|irs|lawsuit|litigation)\b", re.IGNORECASE),
    re.compile(r"\b(company.wide|organization.wide|enterprise.wide|strategic|transformation)\b", re.IGNORECASE),
    re.compile(r"\b(\$\d+\s*(million|billion|M|B)|[0-9]+\s*(million|billion)\s+(dollars?|revenue|deal|contract))\b", re.IGNORECASE),
    re.compile(r"\b(data\s+breach|security\s+incident|regulatory\s+(action|fine|investigation))\b", re.IGNORECASE),
]
_IMPORTANCE_HIGH_PATTERNS = [
    re.compile(r"\b(quarterly\s+(results|earnings|revenue|report)|annual\s+(report|results|guidance))\b", re.IGNORECASE),
    re.compile(r"\b(customer\s+(churn|loss|at\s+risk)|key\s+(account|customer|client)\s+(at\s+risk|threatening))\b", re.IGNORECASE),
    re.compile(r"\b(executive\s+(approval|decision|sign.off)|board\s+(approval|vote|meeting))\b", re.IGNORECASE),
    re.compile(r"\b(contract\s+(renewal|termination|breach)|legal\s+(action|notice|dispute))\b", re.IGNORECASE),
    re.compile(r"\b(critical\s+(system|service|infrastructure)\s+(failure|outage|down))\b", re.IGNORECASE),
    re.compile(r"\b(resignation|restructur|layoff|acquisition\s+(target|deal)|due\s+diligence)\b", re.IGNORECASE),
]
_IMPORTANCE_MEDIUM_PATTERNS = [
    re.compile(r"\b(project\s+(update|status|milestone)|product\s+(launch|release|update))\b", re.IGNORECASE),
    re.compile(r"\b(budget\s+(approval|request|review)|resource\s+(request|allocation))\b", re.IGNORECASE),
    re.compile(r"\b(partnership|vendor|supplier|customer\s+feedback)\b", re.IGNORECASE),
    re.compile(r"\b(performance\s+review|quarterly\s+planning|annual\s+planning)\b", re.IGNORECASE),
]


class UrgencyImportanceDetector:
    """
    Detects urgency and importance as independent dimensions.
    Maps them to the Eisenhower matrix quadrants.
    """

    def detect_urgency(self, text: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Score urgency (time pressure) on 0-100 scale.
        Returns score, level, signals, explanation.
        """
        if not text:
            return {"score": 0, "level": "low", "signals": [], "explanation": "No text provided."}

        score = 0
        signals: List[str] = []

        for pat in _URGENCY_CRITICAL_PATTERNS:
            m = pat.search(text)
            if m:
                score += 40
                signals.append(f"Critical urgency: '{m.group(0)}'")

        for pat in _URGENCY_HIGH_PATTERNS:
            m = pat.search(text)
            if m:
                score += 25
                signals.append(f"High urgency: '{m.group(0)}'")

        for pat in _URGENCY_MEDIUM_PATTERNS:
            m = pat.search(text)
            if m:
                score += 15
                signals.append(f"Medium urgency: '{m.group(0)}'")

        # Structural boosts from metadata
        if metadata:
            if metadata.get("is_early_morning") or metadata.get("is_after_hours"):
                score += 10
                signals.append("Sent outside business hours (may indicate urgency)")
            if metadata.get("has_urgent_subject"):
                score += 15
                signals.append("Urgent keywords in subject line")

        score = min(100, score)
        level = self._score_to_level(score)

        explanation = f"Urgency score {score}/100 ({level}). "
        if signals:
            explanation += "Key signals: " + "; ".join(signals[:3]) + "."
        else:
            explanation += "No strong urgency signals detected."

        return {"score": score, "level": level, "signals": signals[:5], "explanation": explanation}

    def detect_importance(self, text: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Score importance (business/strategic impact) on 0-100 scale.
        Returns score, level, signals, explanation.
        """
        if not text:
            return {"score": 0, "level": "low", "signals": [], "explanation": "No text provided."}

        score = 0
        signals: List[str] = []

        for pat in _IMPORTANCE_CRITICAL_PATTERNS:
            m = pat.search(text)
            if m:
                score += 40
                signals.append(f"Critical importance: '{m.group(0)}'")

        for pat in _IMPORTANCE_HIGH_PATTERNS:
            m = pat.search(text)
            if m:
                score += 25
                signals.append(f"High importance: '{m.group(0)}'")

        for pat in _IMPORTANCE_MEDIUM_PATTERNS:
            m = pat.search(text)
            if m:
                score += 12
                signals.append(f"Medium importance: '{m.group(0)}'")

        # Structural boosts
        if metadata:
            if metadata.get("is_vip") or metadata.get("vip_score", 0) > 0:
                score += 20
                signals.append("Email from VIP sender")
            if metadata.get("is_gov_domain"):
                score += 15
                signals.append("Government domain sender")
            if metadata.get("num_cc", 0) > 3:
                score += 5
                signals.append("Wide distribution (many CC recipients)")

        score = min(100, score)
        level = self._score_to_level(score)

        explanation = f"Importance score {score}/100 ({level}). "
        if signals:
            explanation += "Key signals: " + "; ".join(signals[:3]) + "."
        else:
            explanation += "No strong importance signals detected."

        return {"score": score, "level": level, "signals": signals[:5], "explanation": explanation}

    def get_urgency_importance_matrix(self, urgency_score: float, importance_score: float) -> str:
        """Return the Eisenhower matrix quadrant."""
        urgent = urgency_score >= 50
        important = importance_score >= 50
        if urgent and important:
            return "urgent+important"
        elif urgent and not important:
            return "urgent+not-important"
        elif not urgent and important:
            return "not-urgent+important"
        else:
            return "not-urgent+not-important"

    def explain_difference(self, urgency_result: Dict, importance_result: Dict) -> str:
        """Human-readable explanation of urgency vs importance distinction."""
        u_score = urgency_result["score"]
        i_score = importance_result["score"]
        quadrant = self.get_urgency_importance_matrix(u_score, i_score)

        explanations = {
            "urgent+important": (
                f"This email is BOTH urgent (score {u_score}/100) and important (score {i_score}/100). "
                "It requires your immediate attention — do it now."
            ),
            "urgent+not-important": (
                f"This email feels urgent (score {u_score}/100) but has limited strategic importance (score {i_score}/100). "
                "Consider delegating. The time pressure is real but the business impact is limited."
            ),
            "not-urgent+important": (
                f"This email is important (score {i_score}/100) but not immediately time-critical (urgency {u_score}/100). "
                "Schedule dedicated time to address it properly — don't let it slip through the cracks."
            ),
            "not-urgent+not-important": (
                f"This email has low urgency (score {u_score}/100) and low importance (score {i_score}/100). "
                "Consider delegating or filing for reference."
            ),
        }
        return explanations.get(quadrant, "Unable to determine urgency/importance relationship.")

    @staticmethod
    def _score_to_level(score: float) -> str:
        if score >= 75:
            return "critical"
        if score >= 50:
            return "high"
        if score >= 25:
            return "medium"
        return "low"

