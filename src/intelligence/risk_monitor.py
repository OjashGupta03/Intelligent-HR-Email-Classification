"""
Executive AI Chief of Staff - Risk Monitor
Detects, categorises, and tracks business risks found in email
communications.  Provides dashboards, trend analysis, and
comprehensive risk reports.
"""

import json
import logging
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import RISK_CATEGORIES
from src.utils.helpers import (
    clean_html,
    normalize_email,
    safe_json_loads,
    truncate_string,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Risk detection patterns organised by category
# ---------------------------------------------------------------------------

_RISK_PATTERNS: Dict[str, List[Tuple[re.Pattern, float, str]]] = {
    "financial": [
        (re.compile(r"(?:revenue|profit)\s+(?:decline|drop|loss|miss)", re.I), 0.85, "Revenue/profit decline"),
        (re.compile(r"cost\s+overrun", re.I), 0.80, "Cost overrun"),
        (re.compile(r"budget\s+(?:shortfall|deficit|cut|exceeded)", re.I), 0.80, "Budget issue"),
        (re.compile(r"(?:cash\s+flow|liquidity)\s+(?:issue|problem|concern|crisis)", re.I), 0.90, "Cash flow concern"),
        (re.compile(r"write[\s-]?off", re.I), 0.75, "Write-off"),
        (re.compile(r"(?:unexpected|unplanned)\s+(?:cost|expense|charge)", re.I), 0.80, "Unexpected cost"),
        (re.compile(r"(?:invoice|payment)\s+(?:dispute|late|overdue|rejected)", re.I), 0.70, "Payment issue"),
    ],
    "legal_compliance": [
        (re.compile(r"(?:lawsuit|litigation|legal\s+action|court\s+order)", re.I), 0.95, "Legal action"),
        (re.compile(r"(?:subpoena|injunction|cease\s+and\s+desist)", re.I), 0.95, "Legal instrument"),
        (re.compile(r"(?:regulatory|compliance)\s+(?:violation|breach|issue|concern|fine|penalty)", re.I), 0.90, "Compliance violation"),
        (re.compile(r"(?:GDPR|HIPAA|SOX|PCI)\s+(?:violation|breach|non.?compliance)", re.I), 0.90, "Regulatory breach"),
        (re.compile(r"(?:audit|investigation)\s+(?:finding|concern|issue|failure)", re.I), 0.80, "Audit finding"),
        (re.compile(r"(?:data|privacy|security)\s+breach", re.I), 0.95, "Data/privacy breach"),
        (re.compile(r"(?:intellectual\s+property|IP|patent|trademark)\s+(?:infringement|violation|dispute)", re.I), 0.85, "IP issue"),
    ],
    "customer_dissatisfaction": [
        (re.compile(r"customer\s+(?:complaint|dissatisfaction|frustration|anger|outrage)", re.I), 0.80, "Customer complaint"),
        (re.compile(r"(?:churn|attrition|cancel|defect)\s*(?:rate|risk|threat)?", re.I), 0.80, "Churn risk"),
        (re.compile(r"(?:NPS|satisfaction)\s+(?:drop|decline|low|negative)", re.I), 0.75, "Satisfaction decline"),
        (re.compile(r"(?:refund|compensation)\s+(?:request|demand)", re.I), 0.70, "Refund request"),
        (re.compile(r"(?:escalat|escalaion)\w*\s+(?:to|from)\s+(?:management|executive|VP|director|CEO)", re.I), 0.85, "Customer escalation"),
        (re.compile(r"(?:social\s+media|public|press)\s+(?:complaint|backlash|negative)", re.I), 0.85, "Public complaint"),
    ],
    "project_delay": [
        (re.compile(r"(?:project|timeline|schedule|milestone)\s+(?:delay|slip|miss|behind|at\s+risk)", re.I), 0.80, "Project delay"),
        (re.compile(r"(?:deadline|delivery|launch)\s+(?:miss|slip|delay|postpone|push)", re.I), 0.80, "Deadline missed"),
        (re.compile(r"(?:scope\s+creep|feature\s+creep)", re.I), 0.70, "Scope creep"),
        (re.compile(r"(?:resource|staffing)\s+(?:shortage|gap|constraint|bottleneck)", re.I), 0.75, "Resource constraint"),
        (re.compile(r"(?:blocked|blocker|impediment|obstacle)", re.I), 0.70, "Blocker"),
        (re.compile(r"(?:behind\s+schedule|off\s+track|falling\s+behind)", re.I), 0.80, "Behind schedule"),
    ],
    "contract": [
        (re.compile(r"contract\s+(?:breach|violation|dispute|termination|cancellation)", re.I), 0.90, "Contract issue"),
        (re.compile(r"SLA\s+(?:breach|violation|miss|failure)", re.I), 0.85, "SLA breach"),
        (re.compile(r"(?:vendor|supplier)\s+(?:failure|issue|default|non.?delivery)", re.I), 0.80, "Vendor issue"),
        (re.compile(r"(?:penalty|liquidated\s+damages|indemnification)", re.I), 0.85, "Contractual penalty"),
        (re.compile(r"(?:renegotiat|renegociat)\w*", re.I), 0.65, "Contract renegotiation"),
    ],
    "security": [
        (re.compile(r"(?:security|cyber)\s+(?:breach|incident|attack|threat|vulnerability)", re.I), 0.95, "Security incident"),
        (re.compile(r"(?:phishing|malware|ransomware|DDoS|hack)", re.I), 0.90, "Cyber attack"),
        (re.compile(r"(?:unauthorized|unauthorised)\s+(?:access|entry|use)", re.I), 0.90, "Unauthorized access"),
        (re.compile(r"(?:data\s+(?:leak|exposure|exfiltration))", re.I), 0.95, "Data leak"),
        (re.compile(r"(?:system|server|network)\s+(?:down|outage|failure|compromised)", re.I), 0.85, "System outage"),
    ],
    "escalation": [
        (re.compile(r"(?:escalat)\w*\s+(?:to|from|by)", re.I), 0.80, "Escalation"),
        (re.compile(r"(?:involve|invoking|bringing\s+in)\s+(?:management|leadership|executive|legal)", re.I), 0.75, "Management involvement"),
        (re.compile(r"(?:formal|official)\s+(?:complaint|grievance|dispute)", re.I), 0.80, "Formal complaint"),
        (re.compile(r"(?:threat|threaten)\w*\s+(?:to|of)\s+(?:sue|legal|terminate|cancel|leave)", re.I), 0.85, "Threat detected"),
    ],
    "missed_deadline": [
        (re.compile(r"(?:missed|missing|passed|expired)\s+(?:deadline|due\s+date)", re.I), 0.85, "Missed deadline"),
        (re.compile(r"(?:overdue|past\s+due|late)", re.I), 0.75, "Overdue item"),
        (re.compile(r"(?:failed\s+to|didn't|did\s+not)\s+(?:deliver|submit|complete|meet)", re.I), 0.80, "Failed to deliver"),
    ],
    "operational": [
        (re.compile(r"(?:outage|downtime|service\s+disruption)", re.I), 0.85, "Operational outage"),
        (re.compile(r"(?:quality|QA|testing)\s+(?:issue|failure|defect|bug)", re.I), 0.70, "Quality issue"),
        (re.compile(r"(?:supply\s+chain|logistics)\s+(?:disruption|issue|delay)", re.I), 0.80, "Supply chain issue"),
        (re.compile(r"(?:employee|staff|team)\s+(?:turnover|resignation|departure|leaving)", re.I), 0.75, "Staff departure"),
        (re.compile(r"(?:process|workflow)\s+(?:failure|breakdown|error)", re.I), 0.70, "Process failure"),
    ],
}

# Severity classification thresholds
_SEVERITY_MAP = {
    (0.90, 1.00): "critical",
    (0.75, 0.90): "high",
    (0.55, 0.75): "medium",
    (0.00, 0.55): "low",
}

# Suggested actions per category
_CATEGORY_ACTIONS: Dict[str, str] = {
    "financial": "Review financial impact with CFO; assess mitigation options",
    "legal_compliance": "Engage legal counsel immediately; document all evidence",
    "customer_dissatisfaction": "Escalate to customer success lead; prepare retention response",
    "project_delay": "Convene project standup; identify resource reallocation options",
    "contract": "Review contract terms with legal; prepare negotiation position",
    "security": "Activate incident response team; assess data exposure scope",
    "escalation": "Prepare executive response; gather context from team",
    "missed_deadline": "Communicate revised timeline; identify root cause",
    "operational": "Assess operational impact; initiate contingency procedures",
}


class RiskMonitor:
    """
    Detects, categorises, and tracks business risks from email
    communications.  Provides dashboards, trend analysis, and
    comprehensive risk reports for executive decision-making.
    """

    def __init__(self, db_manager, risk_detector=None) -> None:
        """
        Initialise RiskMonitor.

        Args:
            db_manager: DatabaseManager instance.
            risk_detector: Optional external NLP risk detector.
                           If None, built-in pattern matching is used.
        """
        self.db = db_manager
        self.detector = risk_detector
        logger.info("RiskMonitor initialised (external_detector=%s)",
                     "yes" if risk_detector else "no")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_email_risks(self, email_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect and store risks from an email.

        Args:
            email_dict: Email dict with body, sender, date, id, etc.

        Returns:
            List of detected risk dicts persisted to the database.
        """
        try:
            body = clean_html(email_dict.get("body", ""))
            subject = email_dict.get("subject", "")
            text = f"{subject}\n{body}"
            email_id = email_dict.get("id") or email_dict.get("email_id")

            if not text.strip() or not email_id:
                return []

            # Detect risks
            raw_risks = self._detect_risks(text)

            # Deduplicate by category + description similarity
            unique_risks: List[Dict[str, Any]] = []
            seen: set = set()
            for risk in raw_risks:
                key = f"{risk['category']}:{risk['description'][:50]}"
                if key not in seen:
                    seen.add(key)
                    unique_risks.append(risk)

            # Persist
            persisted: List[Dict[str, Any]] = []
            for risk_data in unique_risks:
                record = {
                    "email_id": email_id,
                    "category": risk_data["category"],
                    "severity": risk_data["severity"],
                    "explanation": risk_data["description"],
                    "source_text": risk_data.get("source_text", ""),
                    "suggested_action": risk_data.get("suggested_action", ""),
                    "related_thread_id": email_dict.get("thread_id"),
                    "confidence": risk_data.get("confidence", 0.7),
                }
                try:
                    db_record = self.db.add_risk(record)
                    record["id"] = db_record.id
                    persisted.append(record)
                except Exception as exc:
                    logger.warning("Failed to persist risk: %s", exc)

            logger.info(
                "Detected %d risks in email %s", len(persisted), email_id
            )
            return persisted

        except Exception as exc:
            logger.error("Error processing risks: %s", exc, exc_info=True)
            return []

    def get_risk_dashboard(self) -> Dict[str, Any]:
        """
        Overview: risk counts by category, severity distribution, trends.

        Returns:
            Comprehensive risk dashboard dict.
        """
        try:
            from sqlalchemy import select, func
            from src.database.models import Risk

            with self.db.get_session() as session:
                # Total risks
                total = session.execute(
                    select(func.count(Risk.id))
                ).scalar() or 0

                # By category
                cat_rows = session.execute(
                    select(Risk.category, func.count(Risk.id))
                    .group_by(Risk.category)
                    .order_by(func.count(Risk.id).desc())
                ).all()
                by_category = {cat: cnt for cat, cnt in cat_rows}

                # By severity
                sev_rows = session.execute(
                    select(Risk.severity, func.count(Risk.id))
                    .group_by(Risk.severity)
                ).all()
                by_severity = {sev: cnt for sev, cnt in sev_rows}

                # Recent risks (last 7 days)
                week_ago = datetime.utcnow() - timedelta(days=7)
                recent = session.execute(
                    select(func.count(Risk.id))
                    .where(Risk.detected_at >= week_ago)
                ).scalar() or 0

                # Critical + high count
                critical_high = session.execute(
                    select(func.count(Risk.id))
                    .where(Risk.severity.in_(["critical", "high"]))
                ).scalar() or 0

                # Risk per day (last 30 days)
                month_ago = datetime.utcnow() - timedelta(days=30)
                daily_rows = session.execute(
                    select(
                        func.date(Risk.detected_at),
                        func.count(Risk.id),
                    )
                    .where(Risk.detected_at >= month_ago)
                    .group_by(func.date(Risk.detected_at))
                    .order_by(func.date(Risk.detected_at))
                ).all()
                daily_trend = [
                    {"date": str(d), "count": c} for d, c in daily_rows
                ]

                return {
                    "total_risks": total,
                    "critical_and_high": critical_high,
                    "recent_7d": recent,
                    "by_category": by_category,
                    "by_severity": by_severity,
                    "daily_trend": daily_trend,
                    "generated_at": datetime.utcnow().isoformat(),
                }

        except Exception as exc:
            logger.error("Error generating risk dashboard: %s", exc, exc_info=True)
            return {"total_risks": 0, "error": str(exc)}

    def get_critical_risks(self) -> List[Dict[str, Any]]:
        """
        Get all critical and high severity risks.

        Returns:
            List of risk dicts.
        """
        try:
            critical = self.db.get_risks(severity="critical")
            high = self.db.get_risks(severity="high")
            all_risks = critical + high
            return [self._risk_to_dict(r) for r in all_risks]
        except Exception as exc:
            logger.error("Error getting critical risks: %s", exc, exc_info=True)
            return []

    def get_risks_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get risks for a specific category.

        Args:
            category: Risk category string.

        Returns:
            List of risk dicts.
        """
        try:
            risks = self.db.get_risks(category=category)
            return [self._risk_to_dict(r) for r in risks]
        except Exception as exc:
            logger.error("Error getting risks for category %s: %s", category, exc, exc_info=True)
            return []

    def get_risk_trends(self, days: int = 30) -> Dict[str, Any]:
        """
        Risk trends over time.

        Args:
            days: Number of days to analyse.

        Returns:
            Trend analysis dict.
        """
        try:
            from sqlalchemy import select, func, and_
            from src.database.models import Risk

            cutoff = datetime.utcnow() - timedelta(days=days)

            with self.db.get_session() as session:
                # Overall daily trend
                daily_rows = session.execute(
                    select(
                        func.date(Risk.detected_at),
                        func.count(Risk.id),
                    )
                    .where(Risk.detected_at >= cutoff)
                    .group_by(func.date(Risk.detected_at))
                    .order_by(func.date(Risk.detected_at))
                ).all()

                # Category trends
                cat_rows = session.execute(
                    select(
                        Risk.category,
                        func.date(Risk.detected_at),
                        func.count(Risk.id),
                    )
                    .where(Risk.detected_at >= cutoff)
                    .group_by(Risk.category, func.date(Risk.detected_at))
                    .order_by(func.date(Risk.detected_at))
                ).all()

                category_trends: Dict[str, List[Dict]] = defaultdict(list)
                for cat, date_val, count in cat_rows:
                    category_trends[cat].append({
                        "date": str(date_val), "count": count
                    })

                # Severity trend
                sev_rows = session.execute(
                    select(
                        Risk.severity,
                        func.date(Risk.detected_at),
                        func.count(Risk.id),
                    )
                    .where(Risk.detected_at >= cutoff)
                    .group_by(Risk.severity, func.date(Risk.detected_at))
                    .order_by(func.date(Risk.detected_at))
                ).all()

                severity_trends: Dict[str, List[Dict]] = defaultdict(list)
                for sev, date_val, count in sev_rows:
                    severity_trends[sev].append({
                        "date": str(date_val), "count": count
                    })

                # Compute week-over-week change
                total_this_week = 0
                total_last_week = 0
                now = datetime.utcnow()
                for date_str, count in daily_rows:
                    try:
                        from src.utils.helpers import parse_date
                        d = parse_date(str(date_str))
                        if d:
                            days_ago = (now - d).days
                            if days_ago <= 7:
                                total_this_week += count
                            elif days_ago <= 14:
                                total_last_week += count
                    except Exception:
                        pass

                wow_change = total_this_week - total_last_week
                wow_pct = (
                    (wow_change / total_last_week * 100)
                    if total_last_week > 0
                    else 0.0
                )

                return {
                    "period_days": days,
                    "daily_trend": [
                        {"date": str(d), "count": c} for d, c in daily_rows
                    ],
                    "category_trends": dict(category_trends),
                    "severity_trends": dict(severity_trends),
                    "week_over_week": {
                        "this_week": total_this_week,
                        "last_week": total_last_week,
                        "change": wow_change,
                        "change_pct": round(wow_pct, 1),
                    },
                    "generated_at": now.isoformat(),
                }

        except Exception as exc:
            logger.error("Error computing risk trends: %s", exc, exc_info=True)
            return {"period_days": days, "error": str(exc)}

    def generate_risk_report(self) -> Dict[str, Any]:
        """
        Comprehensive risk report with explanations and recommended actions.

        Returns:
            Full risk report dict.
        """
        try:
            dashboard = self.get_risk_dashboard()
            critical_risks = self.get_critical_risks()
            trends = self.get_risk_trends(days=30)

            # Group critical risks by category for narrative
            by_category: Dict[str, List[Dict]] = defaultdict(list)
            for risk in critical_risks:
                by_category[risk.get("category", "unknown")].append(risk)

            # Build category summaries with actions
            category_summaries: List[Dict[str, Any]] = []
            for cat in RISK_CATEGORIES:
                cat_risks = by_category.get(cat, [])
                cat_count = dashboard.get("by_category", {}).get(cat, 0)
                if cat_count == 0:
                    continue

                category_summaries.append({
                    "category": cat,
                    "total_risks": cat_count,
                    "critical_risks": len([r for r in cat_risks if r.get("severity") == "critical"]),
                    "high_risks": len([r for r in cat_risks if r.get("severity") == "high"]),
                    "top_risks": cat_risks[:3],
                    "suggested_action": _CATEGORY_ACTIONS.get(cat, "Review and assess impact"),
                })

            # Executive summary
            severity_dist = dashboard.get("by_severity", {})
            exec_summary_parts: List[str] = []
            exec_summary_parts.append(
                f"Total active risks: {dashboard.get('total_risks', 0)}"
            )
            if severity_dist.get("critical", 0) > 0:
                exec_summary_parts.append(
                    f"⚠️ {severity_dist['critical']} CRITICAL risks require immediate attention"
                )
            if severity_dist.get("high", 0) > 0:
                exec_summary_parts.append(
                    f"{severity_dist['high']} HIGH severity risks to monitor"
                )

            wow = trends.get("week_over_week", {})
            if wow.get("change", 0) > 0:
                exec_summary_parts.append(
                    f"Risk volume increased {wow['change_pct']:.0f}% week-over-week"
                )
            elif wow.get("change", 0) < 0:
                exec_summary_parts.append(
                    f"Risk volume decreased {abs(wow['change_pct']):.0f}% week-over-week"
                )

            return {
                "executive_summary": ". ".join(exec_summary_parts),
                "dashboard": dashboard,
                "critical_risks": critical_risks[:10],
                "category_summaries": category_summaries,
                "trends": trends,
                "recommended_actions": self._generate_recommendations(
                    critical_risks, dashboard
                ),
                "generated_at": datetime.utcnow().isoformat(),
            }

        except Exception as exc:
            logger.error("Error generating risk report: %s", exc, exc_info=True)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_risks(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect risks in text using external detector or built-in patterns.

        Args:
            text: Combined subject + body text.

        Returns:
            List of raw risk dicts.
        """
        # Try external detector first
        if self.detector:
            try:
                return self.detector.detect(text)
            except Exception as exc:
                logger.warning("External risk detector failed: %s", exc)

        # Built-in pattern matching
        risks: List[Dict[str, Any]] = []

        for category, patterns in _RISK_PATTERNS.items():
            for pattern, base_confidence, description in patterns:
                matches = pattern.finditer(text)
                for match in matches:
                    source_text = match.group(0).strip()

                    # Determine severity from confidence
                    severity = self._confidence_to_severity(base_confidence)

                    # Context boost: more surrounding risk keywords → higher severity
                    context_window = text[
                        max(0, match.start() - 100): min(len(text), match.end() + 100)
                    ].lower()
                    risk_word_count = sum(
                        1 for w in ["critical", "urgent", "severe", "major", "significant"]
                        if w in context_window
                    )
                    if risk_word_count >= 2 and severity in ("medium", "low"):
                        severity = "high"
                        base_confidence = min(1.0, base_confidence + 0.1)

                    risks.append({
                        "category": category,
                        "severity": severity,
                        "description": description,
                        "source_text": truncate_string(source_text, 300),
                        "confidence": round(base_confidence, 2),
                        "suggested_action": _CATEGORY_ACTIONS.get(category, ""),
                    })

        return risks

    def _confidence_to_severity(self, confidence: float) -> str:
        """Map confidence score to severity level."""
        for (low, high), severity in _SEVERITY_MAP.items():
            if low <= confidence < high:
                return severity
        return "medium"

    def _risk_to_dict(self, risk) -> Dict[str, Any]:
        """Convert a Risk ORM object to a dict."""
        return {
            "id": risk.id,
            "email_id": risk.email_id,
            "category": risk.category,
            "severity": risk.severity,
            "explanation": risk.explanation,
            "source_text": risk.source_text,
            "suggested_action": risk.suggested_action,
            "related_thread_id": risk.related_thread_id,
            "detected_at": risk.detected_at.isoformat() if risk.detected_at else None,
            "confidence": risk.confidence,
        }

    def _generate_recommendations(
        self,
        critical_risks: List[Dict[str, Any]],
        dashboard: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate prioritised recommended actions from risk data."""
        recommendations: List[Dict[str, Any]] = []
        seen_categories: set = set()

        # High-priority recommendations from critical risks
        for risk in critical_risks:
            cat = risk.get("category", "unknown")
            if cat in seen_categories:
                continue
            seen_categories.add(cat)

            recommendations.append({
                "priority": "CRITICAL" if risk.get("severity") == "critical" else "HIGH",
                "category": cat,
                "action": risk.get("suggested_action") or _CATEGORY_ACTIONS.get(cat, "Review and assess"),
                "related_risk_count": dashboard.get("by_category", {}).get(cat, 0),
                "sample_risk": risk.get("explanation", ""),
            })

        # Sort: CRITICAL first, then HIGH
        recommendations.sort(
            key=lambda x: 0 if x["priority"] == "CRITICAL" else 1
        )

        return recommendations[:10]
