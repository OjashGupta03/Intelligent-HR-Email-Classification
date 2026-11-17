"""
Executive AI Chief of Staff - Daily Briefing Generator
Produces executive-grade daily briefings summarising priorities,
deadlines, risks, commitments, and recommended actions.
"""

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.helpers import (
    clean_html,
    normalize_email,
    safe_json_loads,
    truncate_string,
)

logger = logging.getLogger(__name__)


class DailyBriefingGenerator:
    """
    Generates executive daily briefings by integrating outputs from the
    PriorityEngine, CommitmentTracker, and RiskMonitor.

    Each briefing includes:
    1. Top emails requiring attention
    2. Key business developments
    3. Pending decisions
    4. Upcoming deadlines
    5. Overdue commitments
    6. Critical risks
    7. Recommended actions
    """

    def __init__(
        self,
        db_manager,
        priority_engine,
        commitment_tracker,
        risk_monitor,
    ) -> None:
        """
        Initialise the DailyBriefingGenerator.

        Args:
            db_manager: DatabaseManager instance.
            priority_engine: PriorityEngine instance.
            commitment_tracker: CommitmentTracker instance.
            risk_monitor: RiskMonitor instance.
        """
        self.db = db_manager
        self.priority_engine = priority_engine
        self.commitment_tracker = commitment_tracker
        self.risk_monitor = risk_monitor
        logger.info("DailyBriefingGenerator initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_briefing(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate a daily briefing.

        Args:
            date: Target date for the briefing.  Defaults to today.

        Returns:
            Comprehensive briefing dict with all sections.
        """
        try:
            if date is None:
                date = datetime.utcnow()
            briefing_date = datetime(date.year, date.month, date.day)

            logger.info("Generating daily briefing for %s", briefing_date.date())

            # ---- Section 1: Top emails ----
            top_emails = self._get_top_emails()

            # ---- Section 2: Key developments ----
            key_developments = self._get_key_developments(briefing_date)

            # ---- Section 3: Pending decisions ----
            pending_decisions = self._get_pending_decisions(briefing_date)

            # ---- Section 4: Upcoming deadlines ----
            upcoming_deadlines = self._get_upcoming_deadlines()

            # ---- Section 5: Overdue commitments ----
            overdue_commitments = self._get_overdue_commitments()

            # ---- Section 6: Critical risks ----
            critical_risks = self._get_critical_risks()

            # ---- Section 7: Recommended actions ----
            recommended_actions = self._generate_recommended_actions(
                top_emails=top_emails,
                pending_decisions=pending_decisions,
                upcoming_deadlines=upcoming_deadlines,
                overdue_commitments=overdue_commitments,
                critical_risks=critical_risks,
            )

            # Build the briefing
            briefing = {
                "date": briefing_date.isoformat(),
                "generated_at": datetime.utcnow().isoformat(),
                "executive_summary": self._build_executive_summary(
                    top_emails, key_developments, pending_decisions,
                    upcoming_deadlines, overdue_commitments, critical_risks,
                ),
                "sections": {
                    "top_emails": {
                        "title": "Top Emails Requiring Attention",
                        "count": len(top_emails),
                        "items": top_emails,
                    },
                    "key_developments": {
                        "title": "Key Business Developments",
                        "count": len(key_developments),
                        "items": key_developments,
                    },
                    "pending_decisions": {
                        "title": "Decisions Awaiting Your Action",
                        "count": len(pending_decisions),
                        "items": pending_decisions,
                    },
                    "upcoming_deadlines": {
                        "title": "Upcoming Deadlines (Next 7 Days)",
                        "count": len(upcoming_deadlines),
                        "items": upcoming_deadlines,
                    },
                    "overdue_commitments": {
                        "title": "Overdue Commitments",
                        "count": len(overdue_commitments),
                        "items": overdue_commitments,
                    },
                    "critical_risks": {
                        "title": "Active Critical Risks",
                        "count": len(critical_risks),
                        "items": critical_risks,
                    },
                    "recommended_actions": {
                        "title": "Recommended Next Steps",
                        "count": len(recommended_actions),
                        "items": recommended_actions,
                    },
                },
                "stats": self._get_briefing_stats(briefing_date),
            }

            # Persist the briefing
            try:
                self.db.save_briefing(briefing_date, briefing)
                logger.info("Briefing saved for %s", briefing_date.date())
            except Exception as exc:
                logger.warning("Failed to save briefing: %s", exc)

            return briefing

        except Exception as exc:
            logger.error("Error generating briefing: %s", exc, exc_info=True)
            return {
                "date": (date or datetime.utcnow()).isoformat(),
                "error": str(exc),
                "sections": {},
            }

    def get_previous_briefings(self, n: int = 7) -> List[Dict[str, Any]]:
        """
        Get the last N briefings from the database.

        Args:
            n: Number of briefings to retrieve.

        Returns:
            List of briefing dicts, most recent first.
        """
        try:
            from sqlalchemy import select
            from src.database.models import DailyBriefing

            with self.db.get_session() as session:
                stmt = (
                    select(DailyBriefing)
                    .order_by(DailyBriefing.date.desc())
                    .limit(n)
                )
                briefings = session.execute(stmt).scalars().all()
                return [
                    {
                        "date": b.date.isoformat() if b.date else None,
                        "generated_at": b.generated_at.isoformat() if b.generated_at else None,
                        "content": b.get_content(),
                    }
                    for b in briefings
                ]
        except Exception as exc:
            logger.error("Error getting previous briefings: %s", exc, exc_info=True)
            return []

    def compare_briefings(self, date1: datetime, date2: datetime) -> Dict[str, Any]:
        """
        Compare two briefings to show changes.

        Args:
            date1: First (earlier) date.
            date2: Second (later) date.

        Returns:
            Comparison dict highlighting changes between the two dates.
        """
        try:
            from sqlalchemy import select
            from src.database.models import DailyBriefing

            d1 = datetime(date1.year, date1.month, date1.day)
            d2 = datetime(date2.year, date2.month, date2.day)

            with self.db.get_session() as session:
                b1 = session.execute(
                    select(DailyBriefing).where(DailyBriefing.date == d1)
                ).scalar_one_or_none()
                b2 = session.execute(
                    select(DailyBriefing).where(DailyBriefing.date == d2)
                ).scalar_one_or_none()

            content1 = b1.get_content() if b1 else {}
            content2 = b2.get_content() if b2 else {}

            comparison: Dict[str, Any] = {
                "date1": d1.isoformat(),
                "date2": d2.isoformat(),
                "changes": {},
            }

            # Compare each section's item counts
            sections1 = content1.get("sections", {})
            sections2 = content2.get("sections", {})

            all_section_keys = set(list(sections1.keys()) + list(sections2.keys()))
            for section_key in all_section_keys:
                s1 = sections1.get(section_key, {})
                s2 = sections2.get(section_key, {})
                count1 = s1.get("count", 0)
                count2 = s2.get("count", 0)
                comparison["changes"][section_key] = {
                    "title": s2.get("title") or s1.get("title", section_key),
                    "count_before": count1,
                    "count_after": count2,
                    "change": count2 - count1,
                    "new_items": self._find_new_items(
                        s1.get("items", []), s2.get("items", [])
                    ),
                    "resolved_items": self._find_new_items(
                        s2.get("items", []), s1.get("items", [])
                    ),
                }

            # Overall trend
            total_before = sum(
                v.get("count_before", 0) for v in comparison["changes"].values()
            )
            total_after = sum(
                v.get("count_after", 0) for v in comparison["changes"].values()
            )
            comparison["overall"] = {
                "total_items_before": total_before,
                "total_items_after": total_after,
                "trend": "increasing" if total_after > total_before
                         else "decreasing" if total_after < total_before
                         else "stable",
            }

            return comparison

        except Exception as exc:
            logger.error("Error comparing briefings: %s", exc, exc_info=True)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _get_top_emails(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get top N priority emails."""
        try:
            return self.priority_engine.get_top_emails(n=n)
        except Exception as exc:
            logger.warning("Failed to get top emails: %s", exc)
            return []

    def _get_key_developments(self, briefing_date: datetime) -> List[Dict[str, Any]]:
        """
        Identify key business developments from recent emails.
        Looks for high-impact emails from the last 24 hours.
        """
        try:
            start = briefing_date - timedelta(days=1)
            end = briefing_date + timedelta(days=1)
            emails = self.db.get_emails(start_date=start, end_date=end, limit=200)

            developments: List[Dict[str, Any]] = []
            # Keywords that signal developments
            dev_keywords = [
                "announcement", "launched", "acquired", "partnership",
                "signed", "approved", "milestone", "achievement",
                "new client", "new customer", "won", "awarded",
                "expansion", "restructuring", "merger",
            ]

            for email in emails:
                body_lower = (clean_html(email.body or "") + " " + (email.subject or "")).lower()
                matched_keywords = [kw for kw in dev_keywords if kw in body_lower]

                if matched_keywords:
                    developments.append({
                        "email_id": email.id,
                        "subject": email.subject,
                        "sender": email.sender,
                        "date": email.date.isoformat() if email.date else None,
                        "preview": truncate_string(clean_html(email.body or ""), 200),
                        "development_type": matched_keywords[0],
                        "keywords": matched_keywords,
                    })

            # Sort by number of matched keywords (more = more significant)
            developments.sort(key=lambda x: len(x.get("keywords", [])), reverse=True)
            return developments[:10]

        except Exception as exc:
            logger.warning("Failed to get key developments: %s", exc)
            return []

    def _get_pending_decisions(self, briefing_date: datetime) -> List[Dict[str, Any]]:
        """
        Find emails requiring executive decisions.
        Looks for approval requests, decision prompts, and sign-off requests.
        """
        try:
            import re
            start = briefing_date - timedelta(days=7)
            end = briefing_date + timedelta(days=1)
            emails = self.db.get_emails(start_date=start, end_date=end, limit=300)

            decisions: List[Dict[str, Any]] = []
            decision_patterns = [
                re.compile(r"(?:need|require|awaiting|pending)\s+(?:your|executive)?\s*(?:approval|decision|sign.?off|authorization|go.?ahead)", re.I),
                re.compile(r"(?:please|kindly)\s+(?:approve|decide|authorize|sign|confirm)", re.I),
                re.compile(r"(?:waiting\s+for|expecting)\s+(?:your|the)\s+(?:decision|approval|response|go.?ahead)", re.I),
                re.compile(r"(?:decision|approval)\s+(?:required|needed|pending)", re.I),
            ]

            for email in emails:
                text = (email.subject or "") + " " + clean_html(email.body or "")
                for pattern in decision_patterns:
                    match = pattern.search(text)
                    if match:
                        decisions.append({
                            "email_id": email.id,
                            "subject": email.subject,
                            "sender": email.sender,
                            "date": email.date.isoformat() if email.date else None,
                            "decision_type": truncate_string(match.group(0), 80),
                            "preview": truncate_string(clean_html(email.body or ""), 200),
                            "age_days": (briefing_date - email.date).days if email.date else None,
                        })
                        break  # One match per email

            # Sort by age (oldest first — needs attention sooner)
            decisions.sort(key=lambda x: x.get("age_days", 0) or 0, reverse=True)
            return decisions[:10]

        except Exception as exc:
            logger.warning("Failed to get pending decisions: %s", exc)
            return []

    def _get_upcoming_deadlines(self) -> List[Dict[str, Any]]:
        """Get commitments with deadlines in the next 7 days."""
        try:
            return self.commitment_tracker.get_upcoming_deadlines(days=7)
        except Exception as exc:
            logger.warning("Failed to get upcoming deadlines: %s", exc)
            return []

    def _get_overdue_commitments(self) -> List[Dict[str, Any]]:
        """Get overdue commitments."""
        try:
            return self.commitment_tracker.get_overdue_commitments()
        except Exception as exc:
            logger.warning("Failed to get overdue commitments: %s", exc)
            return []

    def _get_critical_risks(self) -> List[Dict[str, Any]]:
        """Get critical and high-severity risks."""
        try:
            return self.risk_monitor.get_critical_risks()
        except Exception as exc:
            logger.warning("Failed to get critical risks: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Recommendations and summary
    # ------------------------------------------------------------------

    def _generate_recommended_actions(
        self,
        top_emails: List[Dict],
        pending_decisions: List[Dict],
        upcoming_deadlines: List[Dict],
        overdue_commitments: List[Dict],
        critical_risks: List[Dict],
    ) -> List[Dict[str, Any]]:
        """
        Generate prioritised recommended actions from all briefing sections.

        Returns:
            Ordered list of action dicts.
        """
        actions: List[Dict[str, Any]] = []
        priority_counter = 1

        # Critical risks need immediate attention
        for risk in critical_risks[:3]:
            if risk.get("severity") == "critical":
                actions.append({
                    "priority": priority_counter,
                    "action": f"Address critical risk: {risk.get('explanation', risk.get('category', 'unknown'))}",
                    "category": "risk",
                    "urgency": "IMMEDIATE",
                    "source": f"Risk #{risk.get('id', '')}",
                    "suggested_action": risk.get("suggested_action", "Review and mitigate"),
                })
                priority_counter += 1

        # Overdue commitments
        for c in overdue_commitments[:3]:
            actions.append({
                "priority": priority_counter,
                "action": f"Follow up on overdue commitment: {truncate_string(c.get('description', ''), 80)}",
                "category": "commitment",
                "urgency": "HIGH",
                "source": f"Commitment from {c.get('responsible_person', 'unknown')}",
                "deadline": c.get("deadline"),
            })
            priority_counter += 1

        # Pending decisions (oldest first)
        for d in pending_decisions[:3]:
            actions.append({
                "priority": priority_counter,
                "action": f"Make decision: {truncate_string(d.get('subject', ''), 80)}",
                "category": "decision",
                "urgency": "HIGH" if (d.get("age_days") or 0) > 3 else "MEDIUM",
                "source": f"From {d.get('sender', 'unknown')} ({d.get('age_days', '?')} days ago)",
            })
            priority_counter += 1

        # Today's deadlines
        for dl in upcoming_deadlines[:3]:
            days_rem = dl.get("days_remaining", 999)
            if days_rem <= 1:
                urgency = "IMMEDIATE"
            elif days_rem <= 3:
                urgency = "HIGH"
            else:
                urgency = "MEDIUM"

            actions.append({
                "priority": priority_counter,
                "action": f"Ensure completion: {truncate_string(dl.get('description', ''), 80)}",
                "category": "deadline",
                "urgency": urgency,
                "deadline": dl.get("deadline"),
                "days_remaining": days_rem,
            })
            priority_counter += 1

        # Top emails needing response
        for email in top_emails[:2]:
            actions.append({
                "priority": priority_counter,
                "action": f"Respond to: {truncate_string(email.get('subject', ''), 80)}",
                "category": "email",
                "urgency": "MEDIUM",
                "source": f"From {email.get('sender', 'unknown')}",
            })
            priority_counter += 1

        return actions

    def _build_executive_summary(
        self,
        top_emails: List,
        key_developments: List,
        pending_decisions: List,
        upcoming_deadlines: List,
        overdue_commitments: List,
        critical_risks: List,
    ) -> str:
        """Build a one-paragraph executive summary."""
        parts: List[str] = []

        # Email summary
        if top_emails:
            parts.append(f"{len(top_emails)} emails require your attention")

        # Critical items
        critical_count = len([
            r for r in critical_risks if r.get("severity") == "critical"
        ])
        if critical_count > 0:
            parts.append(f"⚠️ {critical_count} critical risk(s) detected")

        if overdue_commitments:
            parts.append(f"{len(overdue_commitments)} commitment(s) are overdue")

        if pending_decisions:
            parts.append(f"{len(pending_decisions)} decision(s) pending")

        if upcoming_deadlines:
            parts.append(
                f"{len(upcoming_deadlines)} deadline(s) approaching this week"
            )

        if key_developments:
            parts.append(
                f"{len(key_developments)} key development(s) to note"
            )

        if not parts:
            return "No significant items to report today."

        return ". ".join(parts) + "."

    def _get_briefing_stats(self, briefing_date: datetime) -> Dict[str, Any]:
        """Get statistical context for the briefing."""
        try:
            stats = self.db.get_stats()
            return {
                "total_emails": stats.get("emails", 0),
                "total_commitments": stats.get("commitments", 0),
                "total_risks": stats.get("risks", 0),
                "priority_breakdown": stats.get("priority_breakdown", {}),
                "commitment_breakdown": stats.get("commitment_breakdown", {}),
                "risk_breakdown": stats.get("risk_breakdown", {}),
            }
        except Exception as exc:
            logger.warning("Failed to get briefing stats: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_new_items(
        self,
        old_items: List[Dict],
        new_items: List[Dict],
    ) -> List[Dict[str, Any]]:
        """Find items in new_items that are not in old_items."""
        old_ids = set()
        for item in old_items:
            item_id = item.get("id") or item.get("email_id") or item.get("subject", "")
            old_ids.add(str(item_id))

        new_only: List[Dict[str, Any]] = []
        for item in new_items:
            item_id = item.get("id") or item.get("email_id") or item.get("subject", "")
            if str(item_id) not in old_ids:
                new_only.append(item)

        return new_only[:5]
