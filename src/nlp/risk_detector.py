"""
Executive AI Chief of Staff - Risk Detector
Identifies business risks from email text using category-specific patterns.
Provides severity assessment, explanations, and suggested actions.
"""
import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Risk patterns per category — 15-20 patterns each
# ---------------------------------------------------------------------------
_RISK_DEFINITIONS: Dict[str, Dict] = {
    "financial": {
        "patterns": [
            r"\brevenue\s+(shortfall|decline|miss|drop|at risk)\b",
            r"\bbudget\s+(overrun|overage|exceeded|cut)\b",
            r"\bcash\s+flow\s+(concern|issue|problem|negative)\b",
            r"\bwrite.off\b",
            r"\bimpairment\b",
            r"\bfinancial\s+(risk|exposure|penalty|loss)\b",
            r"\baudit\s+(finding|issue|concern|failure)\b",
            r"\bunexpected\s+(cost|expense|charge)\b",
            r"\bprofit\s+(loss|decline|miss|warning)\b",
            r"\bdebt\s+(covenant|default|breach)\b",
            r"\bliquidity\s+(concern|risk|crisis)\b",
            r"\b(earning|revenue|margin)\s+miss\b",
            r"\bfine\s+of\s+\$[\d,]+\b",
            r"\bpenalt(y|ies)\s+(of|totaling)\s+\$[\d,]+\b",
            r"\bQ[1-4]\s+guidance\s+(miss|risk|below)\b",
            r"\bannual\s+(target|goal)\s+(risk|missed|below)\b",
        ],
        "severity_boost": {"write-off": 1, "audit finding": 1, "cash flow": 1},
        "default_severity": "high",
        "suggested_action": "Review financial reports, engage CFO, assess impact on annual guidance.",
    },
    "legal_compliance": {
        "patterns": [
            r"\blawsuit\b",
            r"\blitigation\b",
            r"\bsubpoena\b",
            r"\bregulatory\s+(investigation|action|notice|violation)\b",
            r"\bnon.compliance\b",
            r"\bviolation\s+of\s+(law|regulation|policy|contract)\b",
            r"\bbreach\s+of\s+(contract|fiduciary|warranty)\b",
            r"\bsec\s+(investigation|inquiry|notice)\b",
            r"\birs\s+(audit|notice|investigation)\b",
            r"\bgdpr\s+(violation|breach|fine)\b",
            r"\bccpa\s+(violation|breach)\b",
            r"\battorney.client\b",
            r"\blegal\s+action\b",
            r"\bclass\s+action\b",
            r"\bregulatory\s+(fine|penalty|sanction)\b",
            r"\bcompliance\s+(failure|breach|issue)\b",
            r"\bdisclosure\s+obligation\b",
        ],
        "severity_boost": {"lawsuit": 2, "subpoena": 2, "sec investigation": 2},
        "default_severity": "critical",
        "suggested_action": "Engage legal counsel immediately, preserve documents, notify board if required.",
    },
    "customer_dissatisfaction": {
        "patterns": [
            r"\bcustomer\s+(complaint|churn|threatening|dissatisfied|angry|upset)\b",
            r"\bthreatening\s+to\s+(cancel|leave|churn|sue)\b",
            r"\bcancel(l?ing)?\s+(their\s+)?(contract|subscription|account|service)\b",
            r"\bnegative\s+(feedback|review|publicity)\b",
            r"\bchurn\s+(risk|rate|issue)\b",
            r"\bescalat(ed|ing)\s+(by|from|to)\s+customer\b",
            r"\bclient\s+(at risk|threatening|dissatisfied)\b",
            r"\bsla\s+(breach|violation|penalty|missed)\b",
            r"\bcustomer\s+satisfaction\s+(decline|drop|low)\b",
            r"\brefund\s+demand(ed)?\b",
            r"\baccount\s+at\s+risk\b",
            r"\blargest\s+(customer|client|account)\s+(at risk|threatening)\b",
            r"\b(million|billion)\s+arr?\s+at\s+risk\b",
        ],
        "severity_boost": {"threatening to cancel": 1, "largest customer": 2},
        "default_severity": "high",
        "suggested_action": "Executive engagement with customer, emergency service review, retention strategy.",
    },
    "project_delay": {
        "patterns": [
            r"\bbehind\s+schedule\b",
            r"\bdelayed?\s+(by|due to|because)\b",
            r"\bslipping\s+(behind|schedule)\b",
            r"\bmissed\s+(milestone|deadline|delivery)\b",
            r"\bresource\s+constraint\b",
            r"\bblocker?\b",
            r"\bnot\s+on\s+track\b",
            r"\btimeline\s+(risk|slipping|at risk)\b",
            r"\bproject\s+(at risk|delayed|overdue)\b",
            r"\b(two|three|four|\d)\s+weeks?\s+delay\b",
            r"\bgo.live\s+(at risk|delayed|postponed)\b",
            r"\brelease\s+(delayed|postponed|at risk)\b",
            r"\bdeployment\s+(delayed|postponed)\b",
            r"\boverdue\s+(by|since)\b",
        ],
        "severity_boost": {"missed milestone": 1, "go-live at risk": 1},
        "default_severity": "medium",
        "suggested_action": "Review project plan, allocate additional resources, escalate blockers, update stakeholders.",
    },
    "contract": {
        "patterns": [
            r"\bcontract\s+(expir|terminat|breach|dispute|at risk)\b",
            r"\bagreement\s+(expir|terminat|breach)\b",
            r"\bauto.renew(s|al)?\b",
            r"\bpenalty\s+clause\b",
            r"\brenegotiat(e|ion)\b",
            r"\btermination\s+notice\b",
            r"\bcontract\s+(renewal|deadline|deadline)\b",
            r"\blease\s+(expir|terminat|renewal)\b",
            r"\bsupplier\s+(contract|agreement)\s+(issue|risk)\b",
            r"\bvendor\s+(contract|termination|dispute)\b",
            r"\bpartnership\s+(agreement|contract)\s+(issue|risk|terminat)\b",
            r"\bnotice\s+period\b",
        ],
        "severity_boost": {"auto-renew": 1, "termination notice": 1},
        "default_severity": "high",
        "suggested_action": "Legal review, negotiate terms, engage procurement, set calendar reminder for contract deadline.",
    },
    "security": {
        "patterns": [
            r"\bsecurity\s+(breach|incident|alert|vulnerability)\b",
            r"\bdata\s+(breach|leak|exposure|theft)\b",
            r"\bunauthorized\s+access\b",
            r"\bcompromise[d]?\b",
            r"\bransomware\b",
            r"\bphishing\b",
            r"\bmalware\b",
            r"\bintrusion\b",
            r"\bsuspicious\s+activity\b",
            r"\bcybersecurity\s+(incident|risk|threat)\b",
            r"\bpii\s+(exposed|at risk|compromised)\b",
            r"\bcredential\s+(stolen|compromised|exposed)\b",
            r"\bsystem\s+(compromised|hacked)\b",
        ],
        "severity_boost": {"data breach": 2, "ransomware": 2, "pii exposed": 2},
        "default_severity": "critical",
        "suggested_action": "Engage CISO, isolate affected systems, notify legal, assess GDPR/CCPA notification requirements.",
    },
    "escalation": {
        "patterns": [
            r"\bescalat(ed|ing|ion)\b",
            r"\bexecutive\s+(attention|review|intervention|escalation)\b",
            r"\bboard\s+(attention|notification|aware)\b",
            r"\bceo\s+(involvement|aware|escalation)\b",
            r"\bpersonal\s+intervention\b",
            r"\burgent\s+escalation\b",
            r"\bmajor\s+(issue|problem|concern)\s+escalat\b",
            r"\bseverity\s+1\b",
            r"\bp1\s+(incident|issue)\b",
            r"\bhigh.level\s+(escalation|attention)\b",
        ],
        "severity_boost": {"board notification": 1, "ceo involvement": 1},
        "default_severity": "high",
        "suggested_action": "Acknowledge escalation, assign ownership, provide ETA for resolution, communicate to stakeholders.",
    },
    "missed_deadline": {
        "patterns": [
            r"\boverdue\b",
            r"\bpast\s+due\b",
            r"\bmissed\s+(deadline|date|delivery)\b",
            r"\bfailed\s+to\s+(deliver|complete|submit|respond)\b",
            r"\blate\s+(delivery|submission|response|payment)\b",
            r"\bnot\s+delivered\s+(on time|by|as promised)\b",
            r"\bexpired\s+(deadline|contract|license)\b",
            r"\bno\s+response\s+(received|from).{0,30}deadline\b",
            r"\bstill\s+waiting\s+for\b",
            r"\bpromised\s+by.{0,40}not\s+(delivered|received|completed)\b",
        ],
        "severity_boost": {"missed deadline": 1, "failed to deliver": 1},
        "default_severity": "high",
        "suggested_action": "Identify root cause, communicate delay with new ETA, assess impact, escalate if critical.",
    },
    "operational": {
        "patterns": [
            r"\boutage\b",
            r"\bdowntime\b",
            r"\bsystem\s+(failure|down|unavailable|outage)\b",
            r"\bservice\s+(disruption|outage|failure|down)\b",
            r"\bbottleneck\b",
            r"\bcapacity\s+(issue|constraint|exceeded)\b",
            r"\bproduction\s+(issue|problem|halt|stoppage)\b",
            r"\boperational\s+(risk|issue|failure)\b",
            r"\bsupply\s+chain\s+(disruption|issue|risk)\b",
            r"\binfrastructure\s+(failure|issue|outage)\b",
            r"\bcritical\s+(system|service)\s+(down|failure)\b",
        ],
        "severity_boost": {"outage": 1, "production halt": 2},
        "default_severity": "high",
        "suggested_action": "Engage operations team, implement contingency plan, communicate impact to stakeholders.",
    },
}

# Pre-compile all patterns
for _category, _def in _RISK_DEFINITIONS.items():
    _def["compiled"] = [re.compile(p, re.IGNORECASE) for p in _def["patterns"]]


class RiskDetector:
    """Detects business risks in email text using category-specific patterns."""

    def detect_risks(self, text: str) -> List[Dict]:
        """Detect all risks in the given text. Returns list of risk dicts."""
        if not text:
            return []
        risks: List[Dict] = []
        for category, definition in _RISK_DEFINITIONS.items():
            matches = []
            for compiled_pat in definition["compiled"]:
                match = compiled_pat.search(text)
                if match:
                    matches.append(match.group(0))

            if not matches:
                continue

            severity = self._assess_severity(text, category, matches, definition)
            explanation = self._generate_explanation(category, matches)
            action = definition["suggested_action"]
            confidence = min(0.5 + len(matches) * 0.1, 0.95)

            # Find source sentence
            sentences = re.split(r"(?<=[.!?])\s+", text)
            source_text = next(
                (s for s in sentences if any(m.lower() in s.lower() for m in matches[:1])),
                matches[0] if matches else "",
            )

            risks.append({
                "category": category,
                "severity": severity,
                "explanation": explanation,
                "source_text": source_text.strip(),
                "suggested_action": action,
                "confidence": confidence,
                "matched_patterns": matches[:3],  # top 3 matches
            })

        return risks

    def _assess_severity(self, text: str, category: str, matches: List[str], definition: Dict) -> str:
        """Assess severity based on category defaults and boost patterns."""
        base = definition["default_severity"]
        severity_map = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        level = severity_map.get(base, 2)

        boosts = definition.get("severity_boost", {})
        for phrase, boost in boosts.items():
            if phrase.lower() in text.lower():
                level = min(level + boost, 3)

        # Additional boost for explicit keywords
        if re.search(r"\b(critical|severe|extreme|major|serious)\b", text, re.IGNORECASE):
            level = min(level + 1, 3)
        if re.search(r"\b(million|billion)\b", text, re.IGNORECASE):
            level = min(level + 1, 3)

        return ["low", "medium", "high", "critical"][level]

    def _generate_explanation(self, category: str, matches: List[str]) -> str:
        """Generate a human-readable explanation."""
        category_names = {
            "financial": "Financial Risk",
            "legal_compliance": "Legal & Compliance Risk",
            "customer_dissatisfaction": "Customer Dissatisfaction Risk",
            "project_delay": "Project Delay Risk",
            "contract": "Contract Risk",
            "security": "Security Risk",
            "escalation": "Escalation Risk",
            "missed_deadline": "Missed Deadline Risk",
            "operational": "Operational Risk",
        }
        name = category_names.get(category, category.title())
        matched_str = ", ".join(f'"{m}"' for m in matches[:3])
        return f"{name} detected. Key indicators: {matched_str}."

    def get_risk_summary(self, risks: List[Dict]) -> Dict:
        """Return aggregate stats for a list of risks."""
        summary: Dict = {"total": len(risks), "by_category": {}, "by_severity": {}}
        for r in risks:
            cat = r.get("category", "unknown")
            sev = r.get("severity", "low")
            summary["by_category"][cat] = summary["by_category"].get(cat, 0) + 1
            summary["by_severity"][sev] = summary["by_severity"].get(sev, 0) + 1
        return summary

