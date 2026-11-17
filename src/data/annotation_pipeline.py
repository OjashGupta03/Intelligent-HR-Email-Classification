"""
Executive AI Chief of Staff - Heuristic Annotation Pipeline
Generates weak/heuristic labels for training data.
All labels are explicitly marked as label_type='heuristic'.
"""
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern banks
# ---------------------------------------------------------------------------

_URGENCY_CRITICAL = re.compile(
    r"\b(urgent|asap|immediately|right away|critical|emergency|time.sensitive|"
    r"right now|at once|this instant|no delay|cannot wait|drop everything)\b",
    re.IGNORECASE,
)
_URGENCY_HIGH = re.compile(
    r"\b(by (eod|cob|end of day|close of business|today|tomorrow)|within (24|48) hours|"
    r"this (morning|afternoon|evening|week)|priority|needs? your attention|time is (of the essence|running out))\b",
    re.IGNORECASE,
)
_IMPORTANCE_HIGH = re.compile(
    r"\b(board|ceo|cfo|cto|chairman|investor|shareholder|acquisition|merger|"
    r"regulation|compliance|sec|irs|lawsuit|litigation|strategic|revenue|"
    r"million|billion|\$[0-9]|quarterly results|annual report|ipo|audit)\b",
    re.IGNORECASE,
)
_FINANCIAL = re.compile(
    r"(\$[\d,]+(\.\d+)?([KMBkmb])?|\d+[KMBkmb]?\s*(million|billion|thousand|dollars?|usd)|"
    r"\b(budget|revenue|cost|invoice|payment|refund|overdue|penalty|fine|write.off)\b)",
    re.IGNORECASE,
)
_ACTION_REQUIRED = re.compile(
    r"\b(please (review|approve|confirm|respond|reply|sign|complete|action|"
    r"provide|send|schedule|arrange|ensure)|action required|your (approval|input|feedback|signature|"
    r"decision|direction|response) (is |are )?(needed|required|requested)|"
    r"can you (please|kindly)?|could you (please|kindly)?|"
    r"i need (you|your)|we need (you|your)|do you have|let me know)\b",
    re.IGNORECASE,
)
_DEADLINE = re.compile(
    r"\b(by|before|due|deadline|no later than|not later than|must (be|have)|"
    r"submit(ted)? by|complete(d)? by|respond by|confirm(ed)? by)\b.{0,60}"
    r"(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"end of (day|week|month|quarter|year)|eod|eow|this (week|month)|next (week|month)|"
    r"q[1-4]|jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|"
    r"jul(y)?|aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?|\d{1,2}[/\-]\d{1,2})",
    re.IGNORECASE,
)
_COMMITMENT = re.compile(
    r"\b(i (will|shall|'ll|am going to|promise to|commit to|plan to)|"
    r"we (will|shall|'ll|are going to|commit to|plan to)|"
    r"i'll (send|provide|schedule|arrange|confirm|review|draft|prepare|follow up|"
    r"discuss|call|ensure|look into|check)|"
    r"we'll (send|provide|schedule|arrange|confirm|review|draft|prepare)|"
    r"please (arrange|schedule|ensure|confirm|provide|send|prepare)|"
    r"i (can|could) have .{0,40} (ready|done|completed) by|"
    r"let me (know|check|look into|schedule|arrange|draft)|"
    r"i (will|'ll) (get back|follow up|circle back))\b",
    re.IGNORECASE,
)
_RISK_PATTERNS = {
    "financial": re.compile(
        r"\b(revenue (shortfall|decline|miss)|budget (overrun|overage)|"
        r"cash flow|write.off|impairment|penalty|fine|unexpected cost|loss|"
        r"financial risk|audit finding)\b",
        re.IGNORECASE,
    ),
    "legal_compliance": re.compile(
        r"\b(lawsuit|litigation|subpoena|investigation|regulatory|compliance|"
        r"violation|non.compliance|breach of (contract|law|regulation)|sec|irs|"
        r"attorney|legal action|court|arbitration|gdpr|ccpa)\b",
        re.IGNORECASE,
    ),
    "customer_dissatisfaction": re.compile(
        r"\b(customer (complaint|churn|cancel|threatening|unhappy|dissatisfied|"
        r"escalation|at risk)|cancel(l?ing)? (their )?(contract|subscription|account)|"
        r"negative (feedback|review)|churn|threatening to leave)\b",
        re.IGNORECASE,
    ),
    "project_delay": re.compile(
        r"\b(behind schedule|delayed|slipping|missed (milestone|deadline)|"
        r"resource constraint|blocked|not on track|at risk of delay|"
        r"timeline (risk|slipping))\b",
        re.IGNORECASE,
    ),
    "contract": re.compile(
        r"\b(contract (expir|terminat|breach|dispute|penalty|renewal)|"
        r"agreement (expir|terminat)|auto.renew|penalty clause|"
        r"renegotiat|termination notice)\b",
        re.IGNORECASE,
    ),
    "security": re.compile(
        r"\b(security (breach|incident|vulnerability|alert)|data (breach|leak|exposure)|"
        r"unauthorized access|compromise(d)?|ransomware|phishing|malware|"
        r"intrusion|suspicious activity)\b",
        re.IGNORECASE,
    ),
    "escalation": re.compile(
        r"\b(escalat(ed|ing|ion)|executive (attention|review|intervention)|"
        r"board (attention|notification|aware)|ceo (involvement|aware)|"
        r"personal intervention|urgent escalation)\b",
        re.IGNORECASE,
    ),
    "missed_deadline": re.compile(
        r"\b(overdue|past due|missed (deadline|date)|failed to (deliver|complete|submit)|"
        r"late (delivery|submission)|not delivered (on time|by)|expired)\b",
        re.IGNORECASE,
    ),
    "operational": re.compile(
        r"\b(outage|downtime|system (failure|down|unavailable)|service disruption|"
        r"bottleneck|capacity (issue|constraint)|production (issue|problem|halt)|"
        r"operational risk)\b",
        re.IGNORECASE,
    ),
}


# ---------------------------------------------------------------------------
# Individual annotation functions
# ---------------------------------------------------------------------------

def annotate_priority(email_dict: Dict) -> Dict:
    """
    Heuristically assign priority and urgency/importance scores.
    Returns a dict with priority_label, urgency_score, importance_score.
    """
    # Use pre-computed hints if available (from synthetic generator)
    priority_hint = email_dict.get("_priority_hint")
    urgency_hint = email_dict.get("_urgency_hint")
    if priority_hint and urgency_hint:
        return {
            "priority_label": priority_hint,
            "urgency_score": {"critical": 95, "high": 75, "medium": 50, "low": 20}.get(urgency_hint, 50),
            "importance_score": {"critical": 90, "high": 70, "medium": 45, "low": 20}.get(priority_hint, 45),
        }

    subject = email_dict.get("subject", "")
    body = email_dict.get("body", "")
    text = f"{subject} {body}"

    urgency_score = 0
    importance_score = 0

    if _URGENCY_CRITICAL.search(text):
        urgency_score += 50
    if _URGENCY_HIGH.search(text):
        urgency_score += 30
    if _IMPORTANCE_HIGH.search(text):
        importance_score += 40
    if _FINANCIAL.search(text):
        importance_score += 20
    if _ACTION_REQUIRED.search(text):
        urgency_score += 15
    if _DEADLINE.search(text):
        urgency_score += 20
    if _RISK_PATTERNS["legal_compliance"].search(text):
        importance_score += 25
    if _RISK_PATTERNS["security"].search(text):
        importance_score += 20
        urgency_score += 20

    urgency_score = min(100, urgency_score)
    importance_score = min(100, importance_score)
    combined = (urgency_score * 0.5 + importance_score * 0.5)

    if combined >= 75:
        priority_label = "critical"
    elif combined >= 55:
        priority_label = "high"
    elif combined >= 30:
        priority_label = "medium"
    else:
        priority_label = "low"

    return {
        "priority_label": priority_label,
        "urgency_score": urgency_score,
        "importance_score": importance_score,
    }


def annotate_action_required(text: str) -> bool:
    return bool(_ACTION_REQUIRED.search(text))


def annotate_commitment(text: str) -> bool:
    return bool(_COMMITMENT.search(text))


def annotate_deadline(text: str) -> Dict:
    match = _DEADLINE.search(text)
    if match:
        return {"has_deadline": True, "deadline_text": match.group(0).strip()}
    return {"has_deadline": False, "deadline_text": None}


def annotate_risk(text: str) -> Dict:
    for category, pattern in _RISK_PATTERNS.items():
        if pattern.search(text):
            return {"has_risk": True, "risk_category": category}
    return {"has_risk": False, "risk_category": None}


def annotate_financial(text: str) -> bool:
    return bool(_FINANCIAL.search(text))


def run_annotation_pipeline(emails: List[Dict]) -> List[Dict]:
    """
    Run the full heuristic annotation pipeline on a list of email dicts.
    Adds an 'annotations' key to each email with heuristic labels.
    """
    annotated = []
    for email in emails:
        subject = email.get("subject", "")
        body = email.get("body", "")
        text = f"{subject} {body}"

        priority_ann = annotate_priority(email)
        deadline_ann = annotate_deadline(text)
        risk_ann = annotate_risk(text)

        annotations = {
            "priority_label": priority_ann["priority_label"],
            "urgency_label": _score_to_level(priority_ann["urgency_score"]),
            "importance_label": _score_to_level(priority_ann["importance_score"]),
            "urgency_score": priority_ann["urgency_score"],
            "importance_score": priority_ann["importance_score"],
            "action_required": annotate_action_required(text),
            "has_commitment": annotate_commitment(text) or email.get("_has_commitment", False),
            "has_deadline": deadline_ann["has_deadline"] or email.get("_has_deadline", False),
            "deadline_text": deadline_ann["deadline_text"],
            "has_risk": risk_ann["has_risk"] or email.get("_has_risk", False),
            "risk_category": risk_ann["risk_category"],
            "has_financial": annotate_financial(text),
            "label_type": "heuristic",
        }

        email = dict(email)
        email["annotations"] = annotations
        annotated.append(email)

    logger.info("Annotation pipeline complete: %d emails annotated", len(annotated))

    # Log distribution
    dist: Dict[str, int] = {}
    for e in annotated:
        lbl = e["annotations"]["priority_label"]
        dist[lbl] = dist.get(lbl, 0) + 1
    logger.info("Priority distribution: %s", dist)

    return annotated


def _score_to_level(score: float) -> str:
    if score >= 75:
        return "critical"
    if score >= 55:
        return "high"
    if score >= 30:
        return "medium"
    return "low"


def save_annotations(annotations: List[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, default=str)
    logger.info("Saved %d annotations to %s", len(annotations), path)


def load_annotations(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d annotations from %s", len(data), path)
    return data

