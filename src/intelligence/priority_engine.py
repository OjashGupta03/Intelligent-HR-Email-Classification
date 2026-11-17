"""
Executive AI Chief of Staff - Priority Engine
Hybrid scoring engine combining pattern-based heuristics and business rules
to compute executive-grade priority scores for emails.
"""

import json
import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    DEFAULT_SCORING_WEIGHTS,
    PRIORITY_LEVELS,
    VIP_DOMAINS,
    VIP_SENDERS,
    get_priority_level,
    load_scoring_weights,
)
from src.utils.helpers import (
    clean_html,
    extract_domain,
    normalize_email,
    safe_json_loads,
    truncate_string,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword / pattern banks used by individual scoring components
# ---------------------------------------------------------------------------

_URGENCY_KEYWORDS: Dict[str, float] = {
    "urgent": 25, "asap": 25, "immediately": 25, "critical": 20,
    "emergency": 30, "time-sensitive": 20, "deadline": 15,
    "overdue": 20, "past due": 20, "expires today": 25,
    "expiring": 18, "last chance": 18, "final notice": 22,
    "action required": 18, "respond immediately": 25,
    "escalation": 20, "escalated": 18, "p0": 25, "p1": 20,
    "sev1": 25, "sev-1": 25, "severity 1": 25,
    "blocker": 22, "showstopper": 22, "breaking": 18,
    "outage": 25, "downtime": 22, "incident": 18,
}

_BUSINESS_IMPACT_KEYWORDS: Dict[str, float] = {
    "revenue": 20, "profit": 18, "loss": 20, "growth": 15,
    "market share": 18, "competitive": 15, "acquisition": 22,
    "merger": 22, "partnership": 18, "investment": 20,
    "funding": 20, "budget": 15, "forecast": 12,
    "quarterly": 12, "annual report": 15, "board meeting": 22,
    "stakeholder": 15, "shareholder": 18, "investor": 20,
    "regulatory": 22, "compliance": 20, "audit": 18,
    "lawsuit": 25, "litigation": 25, "legal action": 25,
    "penalty": 22, "fine": 20, "violation": 22,
    "contract": 18, "agreement": 15, "SLA": 18,
    "customer churn": 22, "client loss": 22, "retention": 15,
}

_FINANCIAL_PATTERNS: List[re.Pattern] = [
    re.compile(r"\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|M|B|K|k))?", re.I),
    re.compile(r"(?:USD|EUR|GBP|INR)\s*[\d,]+(?:\.\d{2})?", re.I),
    re.compile(r"[\d,]+\s*(?:million|billion|crore|lakh)\s*(?:dollars?|USD|rupees?)?", re.I),
    re.compile(r"(?:revenue|cost|budget|investment|loss|profit)\s*(?:of|:)?\s*\$?[\d,.]+", re.I),
]

_DEADLINE_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?:by|before|due|deadline)\s+(?:end of day|EOD|COB|close of business)", re.I),
    re.compile(r"(?:by|before|due|deadline)\s+(\w+ \d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?)", re.I),
    re.compile(r"(?:by|before|due|deadline)\s+(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)", re.I),
    re.compile(r"(?:due|deadline|expires?)\s*(?::|on|is)?\s*(?:today|tomorrow|this week|next week|monday|tuesday|wednesday|thursday|friday)", re.I),
    re.compile(r"(?:need|required|expecting)\s+(?:by|before)\s+(\w+)", re.I),
    re.compile(r"(?:today|tonight|this morning|this afternoon|this evening)", re.I),
    re.compile(r"(?:within|in)\s+(\d+)\s+(?:hours?|days?|business days?)", re.I),
]

_ACTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?:please|kindly|could you|can you|would you)\s+(?:review|approve|sign|confirm|respond|reply|send|provide|submit|complete|update|schedule|arrange|prepare|finalize|forward)", re.I),
    re.compile(r"(?:action required|action needed|your action|awaiting your|needs your|requires your|pending your)", re.I),
    re.compile(r"(?:please respond|please reply|please confirm|please approve|please review|please sign)", re.I),
    re.compile(r"(?:need your|waiting for your|expecting your)\s+(?:approval|input|feedback|decision|response|sign-off|confirmation|review)", re.I),
    re.compile(r"(?:FYI|for your information|no action needed|no action required)", re.I),
]

_STRATEGIC_KEYWORDS: Dict[str, float] = {
    "strategy": 15, "strategic": 18, "roadmap": 15, "vision": 12,
    "transformation": 18, "restructuring": 22, "reorganization": 20,
    "executive committee": 22, "board": 20, "c-suite": 22,
    "CEO": 20, "CFO": 18, "CTO": 18, "COO": 18, "VP": 15,
    "director": 12, "leadership": 15, "governance": 15,
    "key initiative": 18, "critical project": 18, "flagship": 15,
    "expansion": 15, "new market": 18, "product launch": 18,
    "IPO": 25, "due diligence": 20, "valuation": 18,
}


class PriorityEngine:
    """
    Hybrid scoring engine combining pattern-based heuristics and business rules
    to produce executive-grade priority scores for emails.

    Each email is evaluated across eight dimensions:
    urgency, business_impact, sender_importance, deadline_proximity,
    financial_impact, action_required, strategic_importance, and
    conversation_context.  The weighted combination yields a 0-100 score
    mapped to CRITICAL / HIGH / MEDIUM / LOW.
    """

    def __init__(
        self,
        db_manager,
        classifier=None,
        embedding_manager=None,
    ) -> None:
        """
        Initialise the PriorityEngine.

        Args:
            db_manager: DatabaseManager instance for persistence.
            classifier: Optional zero-shot classifier (provides ml_prediction).
            embedding_manager: Optional embedding manager for context scoring.
        """
        self.db = db_manager
        self.classifier = classifier
        self.embedding_manager = embedding_manager
        self.weights = load_scoring_weights()
        # Ensure weights sum to 1.0
        total = sum(self.weights.values())
        if total > 0 and abs(total - 1.0) > 0.01:
            self.weights = {k: v / total for k, v in self.weights.items()}
        logger.info("PriorityEngine initialised with weights: %s", self.weights)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_email(self, email_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single email, returning complete priority analysis.

        Args:
            email_dict: Dict with keys: subject, body, sender, recipients,
                        date, message_id, in_reply_to, thread_id, etc.

        Returns:
            Comprehensive priority analysis dict.
        """
        try:
            text = self._get_full_text(email_dict)
            sender = email_dict.get("sender", "")
            recipients = email_dict.get("recipients", [])
            if isinstance(recipients, str):
                recipients = safe_json_loads(recipients, [])
            email_date = email_dict.get("date")
            if isinstance(email_date, str):
                from src.utils.helpers import parse_date
                email_date = parse_date(email_date)

            # Compute each dimension
            urgency_score, urgency_signals = self._compute_urgency_score(email_dict)
            impact_score, impact_signals = self._compute_business_impact(text)
            sender_score, sender_signals = self._compute_sender_importance(sender, recipients)
            deadline_score, deadline_signals = self._compute_deadline_score(text, email_date)
            financial_score, financial_signals = self._compute_financial_score(text)
            action_score, action_signals = self._compute_action_score(text)
            strategic_score, strategic_signals = self._compute_strategic_score(text)
            context_score, context_signals = self._compute_context_score(email_dict)

            component_scores = {
                "urgency": urgency_score,
                "business_impact": impact_score,
                "sender_importance": sender_score,
                "deadline_proximity": deadline_score,
                "financial_impact": financial_score,
                "action_required": action_score,
                "strategic_importance": strategic_score,
                "conversation_context": context_score,
            }

            all_signals = {
                "urgency": urgency_signals,
                "business_impact": impact_signals,
                "sender_importance": sender_signals,
                "deadline_proximity": deadline_signals,
                "financial_impact": financial_signals,
                "action_required": action_signals,
                "strategic_importance": strategic_signals,
                "conversation_context": context_signals,
            }

            # Combine into final score
            final_score = self._combine_scores(component_scores)

            # ML prediction if classifier available
            ml_prediction = self._get_ml_prediction(text)

            # Rule-based prediction
            rule_prediction = get_priority_level(final_score)

            # If ML says CRITICAL or HIGH but rules say LOW/MEDIUM, boost
            if ml_prediction and ml_prediction in ("CRITICAL", "HIGH"):
                if rule_prediction in ("LOW", "MEDIUM"):
                    boost = 15 if ml_prediction == "CRITICAL" else 10
                    final_score = min(100.0, final_score + boost)
                    rule_prediction = get_priority_level(final_score)

            # Determine urgency and importance levels
            urgency_level = self._score_to_level(urgency_score)
            importance_level = self._score_to_level(
                (impact_score + sender_score + strategic_score) / 3.0
            )

            # Generate explanation and evidence
            explanation = self._generate_explanation(component_scores, all_signals)
            evidence = self._extract_evidence(text, all_signals)

            result = {
                "priority_level": rule_prediction,
                "priority_score": round(final_score, 2),
                "urgency": {
                    "score": round(urgency_score, 2),
                    "level": urgency_level,
                    "signals": urgency_signals,
                },
                "importance": {
                    "score": round(
                        (impact_score + sender_score + strategic_score) / 3.0, 2
                    ),
                    "level": importance_level,
                    "signals": impact_signals + sender_signals + strategic_signals,
                },
                "component_scores": {
                    k: round(v, 2) for k, v in component_scores.items()
                },
                "explanation": explanation,
                "evidence": evidence,
                "ml_prediction": ml_prediction,
                "rule_prediction": rule_prediction,
            }

            # Persist to database if email has an id
            email_id = email_dict.get("id") or email_dict.get("email_id") or email_dict.get("db_id")
            if email_id:
                self._persist_priority(email_id, result)

            logger.debug(
                "Scored email '%s': %.1f (%s)",
                truncate_string(email_dict.get("subject", ""), 50),
                final_score,
                rule_prediction,
            )
            return result

        except Exception as exc:
            logger.error("Error scoring email: %s", exc, exc_info=True)
            return self._default_result()

    def rank_emails(self, email_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score and rank multiple emails by priority.

        Args:
            email_dicts: List of email dicts.

        Returns:
            List of dicts with email data merged with priority analysis,
            sorted by priority_score descending.
        """
        scored: List[Dict[str, Any]] = []
        for email_dict in email_dicts:
            try:
                analysis = self.score_email(email_dict)
                merged = {**email_dict, "priority_analysis": analysis}
                scored.append(merged)
            except Exception as exc:
                logger.warning("Failed to score email: %s", exc)
                scored.append({**email_dict, "priority_analysis": self._default_result()})

        scored.sort(
            key=lambda x: x["priority_analysis"]["priority_score"], reverse=True
        )
        return scored

    def get_top_emails(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the top N priority emails from the database.

        Args:
            n: Number of emails to return.

        Returns:
            List of email dicts with priority information.
        """
        try:
            from sqlalchemy import select
            from src.database.models import Email, EmailPriority

            with self.db.get_session() as session:
                stmt = (
                    select(Email, EmailPriority)
                    .join(EmailPriority, Email.id == EmailPriority.email_id)
                    .order_by(EmailPriority.priority_score.desc())
                    .limit(n)
                )
                rows = session.execute(stmt).all()

                results = []
                for email, priority in rows:
                    results.append({
                        "id": email.id,
                        "subject": email.subject,
                        "sender": email.sender,
                        "date": email.date.isoformat() if email.date else None,
                        "priority_level": priority.priority_level,
                        "priority_score": priority.priority_score,
                        "urgency_score": priority.urgency_score,
                        "importance_score": priority.importance_score,
                        "explanation": priority.get_score_explanation(),
                        "body_preview": truncate_string(email.body or "", 200),
                    })
                return results

        except Exception as exc:
            logger.error("Error getting top emails: %s", exc, exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Component scoring methods
    # ------------------------------------------------------------------

    def _compute_urgency_score(self, email_dict: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Compute urgency score using keyword matching, subject analysis,
        and temporal signals.

        Returns:
            (score 0-100, list of signal descriptions)
        """
        text = self._get_full_text(email_dict)
        subject = (email_dict.get("subject") or "").lower()
        signals: List[str] = []
        score = 0.0

        # Check urgency keywords
        text_lower = text.lower()
        for keyword, weight in _URGENCY_KEYWORDS.items():
            if keyword in text_lower:
                score += weight
                signals.append(f"Urgency keyword: '{keyword}'")

        # Subject line urgency markers
        urgency_subject_patterns = [
            (r"\burgent\b", 15, "Subject marked URGENT"),
            (r"\basap\b", 12, "Subject contains ASAP"),
            (r"\bcritical\b", 12, "Subject marked CRITICAL"),
            (r"\bimmediate\b", 10, "Subject requests immediate attention"),
            (r"\bemergency\b", 15, "Subject marked EMERGENCY"),
            (r"^re:\s*re:\s*re:", 5, "Multiple reply depth indicates active discussion"),
        ]
        for pattern, weight, desc in urgency_subject_patterns:
            if re.search(pattern, subject, re.I):
                score += weight
                signals.append(desc)

        # Exclamation marks in subject signal urgency
        exclamation_count = subject.count("!")
        if exclamation_count >= 2:
            score += min(exclamation_count * 3, 10)
            signals.append(f"Subject has {exclamation_count} exclamation marks")

        # ALL-CAPS words in subject
        words = (email_dict.get("subject") or "").split()
        caps_words = [w for w in words if w.isupper() and len(w) > 2]
        if len(caps_words) >= 2:
            score += 8
            signals.append(f"ALL-CAPS words in subject: {', '.join(caps_words[:3])}")

        return (min(score, 100.0), signals)

    def _compute_business_impact(self, text: str) -> Tuple[float, List[str]]:
        """
        Detect business impact keywords and contextual phrases.

        Returns:
            (score 0-100, list of signal descriptions)
        """
        signals: List[str] = []
        score = 0.0
        text_lower = text.lower()

        for keyword, weight in _BUSINESS_IMPACT_KEYWORDS.items():
            if keyword.lower() in text_lower:
                score += weight
                signals.append(f"Business impact: '{keyword}'")

        # Check for multi-word business contexts
        business_contexts = [
            (r"(?:impact|affect|risk)\s+(?:to|on)\s+(?:revenue|business|operations|customers?|clients?)", 15,
             "Direct business impact reference"),
            (r"(?:board|executive|leadership)\s+(?:meeting|review|decision|approval)", 12,
             "Executive-level activity"),
            (r"(?:customer|client)\s+(?:complaint|escalation|issue|concern)", 15,
             "Customer issue detected"),
            (r"(?:project|product)\s+(?:delay|failure|risk|blocked)", 12,
             "Project risk detected"),
        ]
        for pattern, weight, desc in business_contexts:
            if re.search(pattern, text, re.I):
                score += weight
                signals.append(desc)

        return (min(score, 100.0), signals)

    def _compute_sender_importance(
        self, sender: str, recipients: Any
    ) -> Tuple[float, List[str]]:
        """
        Determine sender importance from VIP lists, contact history,
        and domain importance.

        Returns:
            (score 0-100, list of signal descriptions)
        """
        signals: List[str] = []
        score = 0.0
        sender_norm = normalize_email(sender)
        sender_domain = extract_domain(sender_norm)

        # Check VIP senders
        if sender_norm in VIP_SENDERS:
            vip_score = VIP_SENDERS[sender_norm]
            score += vip_score
            signals.append(f"VIP sender: {sender_norm} (importance={vip_score})")

        # Check VIP domains
        if sender_domain in VIP_DOMAINS:
            domain_boost = VIP_DOMAINS[sender_domain]
            score += domain_boost
            signals.append(f"VIP domain: {sender_domain} (boost={domain_boost})")

        # Check contact database for historical importance
        try:
            contact = self.db.get_contact(sender_norm)
            if contact:
                if contact.is_vip:
                    score += 30
                    signals.append(f"Database VIP: {contact.name or sender_norm}")
                if contact.importance_score > 0:
                    score += min(contact.importance_score * 0.5, 25)
                    signals.append(
                        f"Contact importance: {contact.importance_score:.0f}"
                    )
                # High-volume senders who are not VIP get a small boost
                total_interactions = (
                    contact.total_emails_sent + contact.total_emails_received
                )
                if total_interactions > 50:
                    score += 5
                    signals.append(
                        f"Frequent contact ({total_interactions} interactions)"
                    )
        except Exception:
            pass  # Contact lookup is best-effort

        # Government / regulatory domains get a boost
        gov_domains = {"sec.gov", "irs.gov", "fda.gov", "ftc.gov", "doj.gov", "gov.in"}
        if sender_domain in gov_domains:
            score += 25
            signals.append(f"Government sender: {sender_domain}")

        # Direct email (few recipients) slightly higher priority
        if isinstance(recipients, list) and len(recipients) <= 2:
            score += 5
            signals.append("Direct email (≤2 recipients)")

        # C-suite title detection in sender display name
        sender_raw = sender or ""
        csuite_patterns = [
            (r"\bCEO\b", 25), (r"\bCFO\b", 22), (r"\bCTO\b", 22),
            (r"\bCOO\b", 22), (r"\bCIO\b", 20), (r"\bCMO\b", 18),
            (r"\bVP\b", 15), (r"\bPresident\b", 22), (r"\bDirector\b", 12),
            (r"\bGeneral Manager\b", 12), (r"\bPartner\b", 15),
        ]
        for pattern, weight in csuite_patterns:
            if re.search(pattern, sender_raw, re.I):
                score += weight
                signals.append(f"C-suite/exec sender title detected")
                break  # Only count one title

        return (min(score, 100.0), signals)

    def _compute_deadline_score(
        self, text: str, email_date: Optional[datetime]
    ) -> Tuple[float, List[str]]:
        """
        Detect deadlines and compute proximity score.

        Returns:
            (score 0-100, list of signal descriptions)
        """
        signals: List[str] = []
        score = 0.0

        for pattern in _DEADLINE_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                score += 15
                match_str = matches[0] if isinstance(matches[0], str) else str(matches[0])
                signals.append(f"Deadline reference: '{truncate_string(match_str, 50)}'")

        text_lower = text.lower()

        # Same-day deadlines
        if any(kw in text_lower for kw in ["today", "tonight", "eod", "end of day", "cob", "close of business"]):
            score += 30
            signals.append("Same-day deadline detected")
        # Tomorrow
        elif "tomorrow" in text_lower:
            score += 25
            signals.append("Tomorrow deadline detected")
        # This week
        elif "this week" in text_lower or "end of week" in text_lower:
            score += 15
            signals.append("This-week deadline detected")
        # Next week
        elif "next week" in text_lower:
            score += 8
            signals.append("Next-week deadline detected")

        # "Within X hours/days" patterns
        within_match = re.search(r"(?:within|in)\s+(\d+)\s+(hours?|days?|business\s*days?)", text, re.I)
        if within_match:
            amount = int(within_match.group(1))
            unit = within_match.group(2).lower()
            if "hour" in unit and amount <= 24:
                score += 25
                signals.append(f"Deadline within {amount} hour(s)")
            elif "day" in unit and amount <= 3:
                score += 20
                signals.append(f"Deadline within {amount} day(s)")
            elif "day" in unit and amount <= 7:
                score += 10
                signals.append(f"Deadline within {amount} day(s)")

        # If email is older and has a deadline, it might be overdue
        if email_date and score > 0:
            now = datetime.utcnow()
            age_days = (now - email_date).days if isinstance(email_date, datetime) else 0
            if age_days > 3:
                score += 10
                signals.append(f"Email is {age_days} days old with deadline references")

        return (min(score, 100.0), signals)

    def _compute_financial_score(self, text: str) -> Tuple[float, List[str]]:
        """
        Detect financial references and monetary amounts.

        Returns:
            (score 0-100, list of signal descriptions)
        """
        signals: List[str] = []
        score = 0.0

        for pattern in _FINANCIAL_PATTERNS:
            matches = pattern.findall(text)
            for match in matches[:3]:  # Limit to avoid runaway scoring
                score += 15
                signals.append(f"Financial amount: '{truncate_string(match, 40)}'")

        text_lower = text.lower()
        financial_keywords = [
            ("invoice", 10), ("payment", 12), ("billing", 10),
            ("purchase order", 12), ("PO number", 10),
            ("budget approval", 18), ("cost overrun", 20),
            ("expense", 8), ("refund", 12), ("write-off", 15),
            ("tax", 10), ("accounts payable", 10), ("accounts receivable", 10),
        ]
        for keyword, weight in financial_keywords:
            if keyword.lower() in text_lower:
                score += weight
                signals.append(f"Financial keyword: '{keyword}'")

        # Large-amount detection
        large_amount_pattern = re.compile(
            r"(?:\$|USD|EUR|GBP)\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?", re.I
        )
        for match in large_amount_pattern.finditer(text):
            try:
                amount_str = match.group(1).replace(",", "")
                amount = float(amount_str)
                full_match = match.group(0).lower()
                if "billion" in full_match or "b" in full_match.split()[-1:]:
                    amount *= 1_000_000_000
                elif "million" in full_match or "m" in full_match.split()[-1:]:
                    amount *= 1_000_000
                if amount >= 1_000_000:
                    score += 20
                    signals.append(f"Large financial amount (≥$1M)")
                elif amount >= 100_000:
                    score += 12
                    signals.append(f"Significant financial amount (≥$100K)")
            except (ValueError, IndexError):
                pass

        return (min(score, 100.0), signals)

    def _compute_action_score(self, text: str) -> Tuple[float, List[str]]:
        """
        Detect action requirements in the email.

        Returns:
            (score 0-100, list of signal descriptions)
        """
        signals: List[str] = []
        score = 0.0
        is_fyi = False

        for pattern in _ACTION_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                for match in matches[:3]:
                    match_str = match if isinstance(match, str) else str(match)
                    if re.search(r"FYI|for your information|no action needed|no action required", match_str, re.I):
                        is_fyi = True
                        signals.append("FYI / no action required")
                    else:
                        score += 15
                        signals.append(f"Action requested: '{truncate_string(match_str, 50)}'")

        # Question marks suggest action needed
        text_lower = text.lower()
        question_count = text.count("?")
        if question_count >= 3:
            score += 10
            signals.append(f"Multiple questions ({question_count}) suggest response needed")
        elif question_count >= 1:
            score += 5
            signals.append("Contains question(s) requiring response")

        # Decision-related phrases
        decision_phrases = [
            "need your decision", "approval needed", "sign-off required",
            "waiting for your go-ahead", "green light", "authorize",
            "your call", "decide by", "choose between",
        ]
        for phrase in decision_phrases:
            if phrase in text_lower:
                score += 15
                signals.append(f"Decision required: '{phrase}'")

        # FYI emails are lower priority
        if is_fyi and score > 0:
            score = max(0, score - 20)

        return (min(score, 100.0), signals)

    def _compute_strategic_score(self, text: str) -> Tuple[float, List[str]]:
        """
        Detect strategic importance markers.

        Returns:
            (score 0-100, list of signal descriptions)
        """
        signals: List[str] = []
        score = 0.0
        text_lower = text.lower()

        for keyword, weight in _STRATEGIC_KEYWORDS.items():
            if keyword.lower() in text_lower:
                score += weight
                signals.append(f"Strategic keyword: '{keyword}'")

        # Long-term planning references
        strategic_contexts = [
            (r"(?:Q[1-4]|quarter\s*[1-4])\s+(?:plan|goal|target|objective)", 12,
             "Quarterly planning reference"),
            (r"(?:annual|yearly|long.?term)\s+(?:plan|strategy|goal|budget|review)", 15,
             "Annual/long-term planning"),
            (r"(?:board|executive|leadership)\s+(?:directive|priority|initiative)", 18,
             "Leadership initiative"),
            (r"(?:competitive|market)\s+(?:analysis|intelligence|landscape|position)", 12,
             "Competitive intelligence"),
        ]
        for pattern, weight, desc in strategic_contexts:
            if re.search(pattern, text, re.I):
                score += weight
                signals.append(desc)

        return (min(score, 100.0), signals)

    def _compute_context_score(self, email_dict: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Check conversation history and thread context for scoring signals.

        Returns:
            (score 0-100, list of signal descriptions)
        """
        signals: List[str] = []
        score = 0.0

        # Thread-based context
        thread_id = email_dict.get("thread_id")
        if thread_id:
            try:
                thread = self.db.get_thread(thread_id)
                if thread:
                    # Active threads with many participants
                    if thread.email_count and thread.email_count > 5:
                        score += 10
                        signals.append(
                            f"Active thread ({thread.email_count} emails)"
                        )
                    participants = thread.get_participants() if hasattr(thread, 'get_participants') else []
                    if len(participants) > 5:
                        score += 8
                        signals.append(
                            f"Multi-party thread ({len(participants)} participants)"
                        )
            except Exception:
                pass

        # Reply chains indicate ongoing discussion
        if email_dict.get("is_reply") or email_dict.get("in_reply_to"):
            score += 5
            signals.append("Part of ongoing conversation")

        # If sender has previously sent high-priority emails
        sender = normalize_email(email_dict.get("sender", ""))
        if sender:
            try:
                from sqlalchemy import select, func
                from src.database.models import Email, EmailPriority

                with self.db.get_session() as session:
                    high_count = session.execute(
                        select(func.count(EmailPriority.id))
                        .join(Email, Email.id == EmailPriority.email_id)
                        .where(
                            Email.sender.ilike(f"%{sender}%"),
                            EmailPriority.priority_level.in_(["CRITICAL", "HIGH"]),
                        )
                    ).scalar() or 0

                    if high_count >= 3:
                        score += 10
                        signals.append(
                            f"Sender has {high_count} previous high-priority emails"
                        )
            except Exception:
                pass

        return (min(score, 100.0), signals)

    # ------------------------------------------------------------------
    # Score combination & explanation
    # ------------------------------------------------------------------

    def _combine_scores(self, component_scores: Dict[str, float]) -> float:
        """
        Weighted combination of component scores normalised to 0-100.

        Args:
            component_scores: Dict mapping dimension name to its raw score.

        Returns:
            Final combined priority score (0-100).
        """
        total = 0.0
        for dimension, raw_score in component_scores.items():
            weight = self.weights.get(dimension, 0.0)
            total += raw_score * weight

        # Apply non-linear scaling to spread scores
        # Slight sigmoid-like boost for high-scoring items
        if total > 70:
            total = 70 + (total - 70) * 1.2
        elif total < 20:
            total = total * 0.9

        return max(0.0, min(100.0, total))

    def _generate_explanation(
        self,
        component_scores: Dict[str, float],
        all_signals: Dict[str, List[str]],
    ) -> List[str]:
        """
        Generate human-readable explanation of the priority score.

        Args:
            component_scores: Individual dimension scores.
            all_signals: All signals grouped by dimension.

        Returns:
            List of explanation strings.
        """
        explanations: List[str] = []

        # Sort dimensions by score descending
        sorted_dims = sorted(
            component_scores.items(), key=lambda x: x[1], reverse=True
        )

        for dim_name, dim_score in sorted_dims:
            if dim_score < 5:
                continue
            level = self._score_to_level(dim_score)
            dim_signals = all_signals.get(dim_name, [])

            # Pretty dimension name
            pretty_name = dim_name.replace("_", " ").title()
            explanation = f"{pretty_name}: {level} ({dim_score:.0f}/100)"
            if dim_signals:
                top_signals = dim_signals[:2]
                explanation += f" — {'; '.join(top_signals)}"
            explanations.append(explanation)

        if not explanations:
            explanations.append("No significant priority signals detected.")

        return explanations

    def _extract_evidence(
        self, text: str, all_signals: Dict[str, List[str]]
    ) -> List[str]:
        """
        Extract direct quotes from the email that support the priority score.

        Args:
            text: Full email text.
            all_signals: All signals grouped by dimension.

        Returns:
            List of evidence quotes.
        """
        evidence: List[str] = []
        text_lower = text.lower()

        # Collect keywords from signals
        keywords_to_find: List[str] = []
        for signal_list in all_signals.values():
            for signal in signal_list:
                # Extract quoted keywords from signal descriptions
                quoted = re.findall(r"'([^']+)'", signal)
                keywords_to_find.extend(quoted)

        # Find sentences containing these keywords
        sentences = re.split(r"[.!?\n]+", text)
        seen: set = set()
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue
            sentence_lower = sentence.lower()
            for keyword in keywords_to_find:
                if keyword.lower() in sentence_lower and sentence not in seen:
                    evidence.append(f'"{truncate_string(sentence, 150)}"')
                    seen.add(sentence)
                    break
            if len(evidence) >= 5:
                break

        return evidence

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_full_text(self, email_dict: Dict[str, Any]) -> str:
        """Combine subject and body into searchable text."""
        subject = email_dict.get("subject") or ""
        body = email_dict.get("body") or ""
        body = clean_html(body)
        return f"{subject}\n{body}"

    def _score_to_level(self, score: float) -> str:
        """Map a 0-100 score to a human-readable level string."""
        if score >= 75:
            return "HIGH"
        elif score >= 40:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_ml_prediction(self, text: str) -> Optional[str]:
        """
        Get ML-based priority prediction if a classifier is available.

        Args:
            text: Email text.

        Returns:
            Priority level string or None.
        """
        if not self.classifier:
            return None
        try:
            candidate_labels = ["critical", "high", "medium", "low"]
            result = self.classifier(
                truncate_string(text, 512),
                candidate_labels,
                hypothesis_template="This email has {} priority.",
            )
            top_label = result["labels"][0].upper()
            return top_label
        except Exception as exc:
            logger.warning("ML prediction failed: %s", exc)
            return None

    def _persist_priority(self, email_id: int, result: Dict[str, Any]) -> None:
        """Save priority result to database."""
        try:
            scores = result.get("component_scores", {})
            self.db.add_priority(email_id, {
                "priority_level": result["priority_level"],
                "priority_score": result["priority_score"],
                "urgency_score": scores.get("urgency", 0),
                "importance_score": result["urgency"]["score"],
                "business_impact_score": scores.get("business_impact", 0),
                "sender_importance_score": scores.get("sender_importance", 0),
                "deadline_score": scores.get("deadline_proximity", 0),
                "financial_score": scores.get("financial_impact", 0),
                "action_score": scores.get("action_required", 0),
                "strategic_score": scores.get("strategic_importance", 0),
                "context_score": scores.get("conversation_context", 0),
                "score_explanation": {
                    "explanation": result["explanation"],
                    "evidence": result["evidence"],
                },
                "ml_prediction": result.get("ml_prediction"),
                "rule_prediction": result.get("rule_prediction"),
            })
        except Exception as exc:
            logger.warning("Failed to persist priority for email %s: %s", email_id, exc)

    def _default_result(self) -> Dict[str, Any]:
        """Return a default low-priority result."""
        return {
            "priority_level": "LOW",
            "priority_score": 0.0,
            "urgency": {"score": 0.0, "level": "LOW", "signals": []},
            "importance": {"score": 0.0, "level": "LOW", "signals": []},
            "component_scores": {
                "urgency": 0.0, "business_impact": 0.0,
                "sender_importance": 0.0, "deadline_proximity": 0.0,
                "financial_impact": 0.0, "action_required": 0.0,
                "strategic_importance": 0.0, "conversation_context": 0.0,
            },
            "explanation": ["Unable to score email."],
            "evidence": [],
            "ml_prediction": None,
            "rule_prediction": "LOW",
        }
