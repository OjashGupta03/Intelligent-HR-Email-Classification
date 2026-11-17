"""
Executive AI Chief of Staff - Dataset Loader
Loads the Enron email dataset (CSV format) or generates realistic
synthetic executive emails when the dataset is unavailable.
"""
import csv
import hashlib
import json
import logging
import random
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------

_EXECUTIVE_NAMES = [
    "John Mitchell", "Sarah Chen", "Robert Davis", "Emily Watson",
    "Michael Torres", "Jennifer Lee", "David Kim", "Amanda Foster",
    "Christopher Brown", "Lisa Anderson", "James Wilson", "Patricia Moore",
    "Daniel Martinez", "Nancy Taylor", "Mark Thompson", "Sandra Harris",
    "Paul Jackson", "Karen White", "Steven Clark", "Betty Lewis",
]

_ORGANIZATIONS = [
    "Acme Corp", "Global Ventures", "Pacific Holdings", "Atlantic Partners",
    "Summit Capital", "Meridian Group", "Apex Solutions", "Nexus International",
    "Pinnacle Industries", "Vanguard Enterprises",
]

_DOMAINS = [
    "acmecorp.com", "globalventures.com", "pacificholdings.com",
    "atlanticpartners.com", "summitcapital.com", "meridiangroup.com",
    "apexsolutions.com", "nexusintl.com", "pinnacle-ind.com", "vanguardent.com",
]

_TEMPLATES = [
    # (subject_template, body_template, priority_hint, urgency_hint, has_commitment, has_risk, has_deadline)
    (
        "URGENT: Q{q} Revenue Shortfall - Immediate Action Required",
        "John,\n\nI need to bring to your attention a critical issue with our Q{q} revenue. "
        "We are currently tracking $12.3M below our target, representing a 15% shortfall. "
        "The CFO has flagged this as a critical risk to our annual guidance.\n\n"
        "I will send a full analysis report by tomorrow morning. "
        "We need your approval on the revised revenue plan by Friday EOD.\n\n"
        "This is time-sensitive as the board meeting is scheduled for Monday.\n\n"
        "Best regards,\n{sender_name}",
        "critical", "critical", True, True, True,
    ),
    (
        "Board Meeting Agenda - {date_str}",
        "Dear {recipient_name},\n\nPlease find attached the agenda for the upcoming board meeting "
        "scheduled for {date_str} at 9:00 AM EST.\n\n"
        "Key items for discussion:\n"
        "1. Q{q} Financial Results\n"
        "2. Strategic acquisition update - {company}\n"
        "3. Executive compensation review\n"
        "4. Risk committee report\n\n"
        "Please review the materials and come prepared to discuss item 2 in detail. "
        "We need a decision on the acquisition by end of this quarter.\n\n"
        "Regards,\n{sender_name}\nBoard Secretary",
        "high", "medium", True, False, True,
    ),
    (
        "Re: Contract Renewal - {company} - Action Required",
        "Hi {recipient_name},\n\nFollowing up on our discussion about the {company} contract. "
        "The current agreement expires on {date_str} and we need to decide whether to renew, "
        "renegotiate, or terminate.\n\n"
        "Legal has flagged several clauses that need revision, including the indemnification "
        "section and the SLA penalties. If we don't respond by {date_str}, the contract "
        "auto-renews at current unfavorable terms.\n\n"
        "I will schedule a call with legal and procurement this week. "
        "Please confirm your availability.\n\n"
        "Best,\n{sender_name}",
        "high", "high", True, True, True,
    ),
    (
        "Customer Escalation: {company} - Threatening to Cancel",
        "Executive Team,\n\nI'm escalating an urgent customer situation. "
        "{company} is our 3rd largest account ($4.2M ARR) and their CEO "
        "has personally threatened to cancel unless we resolve their service issues by {date_str}.\n\n"
        "Issues:\n- 3 critical bugs open for 45+ days\n- SLA breached 7 times this quarter\n"
        "- Support response times averaging 48 hours\n\n"
        "I need your personal intervention. Can you join a call with their CEO tomorrow?\n\n"
        "Regards,\n{sender_name}",
        "critical", "critical", True, True, True,
    ),
    (
        "M&A Update: Due Diligence Phase 2 - Confidential",
        "Dear John,\n\nConfidential - For your eyes only.\n\n"
        "Phase 2 due diligence on the {company} acquisition is progressing. "
        "The financial audit has revealed some concerns:\n\n"
        "1. Revenue recognition issues in FY{year} ($8.5M at risk)\n"
        "2. Pending litigation not disclosed in initial review\n"
        "3. Key person dependency - CFO plans to leave post-acquisition\n\n"
        "I recommend we request a price adjustment of 12-15% before proceeding. "
        "Investment bankers will present revised terms next week.\n\n"
        "Should I proceed with renegotiation? Need your direction by {date_str}.\n\n"
        "Strictly confidential,\n{sender_name}",
        "critical", "high", True, True, True,
    ),
    (
        "Weekly Project Status - {project}",
        "Hi team,\n\nWeekly status update for {project}:\n\n"
        "Overall Status: Yellow\n"
        "Progress: 65% complete (target: 72%)\n"
        "Issues: Resource constraint on backend development\n\n"
        "Key highlights:\n- Completed user authentication module\n"
        "- API integration with {company} delayed by 2 weeks\n"
        "- Testing environment setup in progress\n\n"
        "Risks:\n- Timeline slipping due to resource shortage\n"
        "- Third-party API changes may require rework\n\n"
        "Next steps:\n- Requesting 2 additional developers from {company}\n"
        "- Risk review meeting scheduled for {date_str}\n\n"
        "Best,\n{sender_name}",
        "medium", "low", True, True, True,
    ),
    (
        "Investor Relations: Q{q} Earnings Call Preparation",
        "John,\n\nPreparing for the Q{q} earnings call on {date_str}. "
        "I need your sign-off on the prepared remarks and Q&A document by {date_str}.\n\n"
        "Key messages to emphasize:\n"
        "- Revenue growth of 18% YoY\n- New product launches driving engagement\n"
        "- International expansion ahead of plan\n\n"
        "Sensitive areas to watch:\n"
        "- Margin compression questions\n- Competitive pressure from {company}\n\n"
        "I'll send the draft tomorrow. Please review and provide feedback by EOD {date_str}.\n\n"
        "Regards,\n{sender_name}",
        "high", "medium", True, False, True,
    ),
    (
        "Legal Alert: Regulatory Investigation Notice",
        "CONFIDENTIAL - ATTORNEY-CLIENT PRIVILEGE\n\n"
        "John,\n\nWe have received a formal investigation notice from the SEC regarding "
        "our Q{q} disclosures. This is serious and requires immediate attention.\n\n"
        "Required actions:\n"
        "1. Preserve all documents related to Q{q} reporting (litigation hold effective immediately)\n"
        "2. Engage external counsel (I recommend {company})\n"
        "3. Board notification required within 48 hours\n\n"
        "We must respond to the SEC by {date_str}. I need to meet with you today.\n\n"
        "Privileged and Confidential,\n{sender_name}\nGeneral Counsel",
        "critical", "critical", True, True, True,
    ),
    (
        "HR Update: Key Executive Resignation",
        "John,\n\nI need to inform you of a sensitive personnel matter. "
        "Our Chief Revenue Officer, {name}, has submitted their resignation effective {date_str}. "
        "This is not public information yet.\n\n"
        "Immediate concerns:\n"
        "- Q{q} sales pipeline at risk\n- Key client relationships need transition\n"
        "- Succession planning required urgently\n\n"
        "I recommend we act quickly to:\n"
        "1. Negotiate retention or extend notice period\n"
        "2. Identify internal candidates\n"
        "3. Engage executive search firm\n\n"
        "Can we discuss today?\n\nConfidentially,\n{sender_name}",
        "high", "high", True, True, True,
    ),
    (
        "Budget Approval Needed: {project} - $2.4M",
        "Hi John,\n\nRequesting your approval for the {project} budget of $2,400,000.\n\n"
        "Breakdown:\n- Infrastructure: $850,000\n- Staffing: $1,100,000\n"
        "- Software licenses: $350,000\n- Contingency: $100,000\n\n"
        "Business case:\n- Expected ROI: 340% over 3 years\n"
        "- Payback period: 18 months\n- Risk-adjusted NPV: $4.2M\n\n"
        "This needs approval by {date_str} to maintain project timeline.\n\n"
        "Best,\n{sender_name}",
        "high", "medium", True, False, True,
    ),
    (
        "Security Incident Report: Potential Data Breach",
        "URGENT - Security Incident\n\nJohn,\n\n"
        "Our security team has detected suspicious activity on our systems. "
        "Potential data breach affecting customer PII.\n\n"
        "Current status:\n- Incident detected at 2:30 AM today\n"
        "- Security team actively investigating\n- Systems partially isolated\n\n"
        "Potential impact:\n- Up to 50,000 customer records at risk\n"
        "- Regulatory notification may be required within 72 hours (GDPR/CCPA)\n\n"
        "We need your immediate authorization to:\n"
        "1. Engage external forensics firm\n2. Notify legal and compliance\n"
        "3. Prepare customer communication\n\n"
        "Please call me immediately.\n\n{sender_name}\nCISO",
        "critical", "critical", True, True, True,
    ),
    (
        "Partnership Proposal: {company} Strategic Alliance",
        "Dear {recipient_name},\n\nI'm reaching out on behalf of {company} to explore "
        "a strategic partnership that could significantly benefit both organizations.\n\n"
        "We believe there is a compelling opportunity in:\n"
        "- Joint go-to-market in the {project} space\n"
        "- Technology integration\n- Co-development of new solutions\n\n"
        "Initial analysis suggests combined market opportunity of $500M+.\n\n"
        "Would you be open to an executive briefing next week?\n\n"
        "Best regards,\n{sender_name}",
        "medium", "low", False, False, False,
    ),
    (
        "Team Offsite Planning - {date_str}",
        "Hi everyone,\n\nWe're planning a leadership team offsite for {date_str}. "
        "Please mark your calendars.\n\nAgenda themes:\n"
        "- Annual strategy review\n- Team building\n- Q{q} planning\n\n"
        "Location: {company} Conference Center\n"
        "Please confirm attendance by {date_str}.\n\nThanks,\n{sender_name}",
        "low", "low", False, False, True,
    ),
    (
        "FYI: Industry News - {company} Acquisition",
        "Team,\n\nSharing this article about {company}'s acquisition of their competitor. "
        "This may impact our competitive positioning.\n\n"
        "Key highlights:\n- $3.2B deal announced this morning\n"
        "- Expected to close Q{q}\n- May affect pricing dynamics\n\n"
        "Worth monitoring. I'll add to our competitive intelligence briefing.\n\n"
        "FYI only,\n{sender_name}",
        "low", "low", False, False, False,
    ),
    (
        "Quarterly Business Review - Preparation Required",
        "Leadership Team,\n\nOur Q{q} Business Review is scheduled for {date_str}.\n\n"
        "Each function head is requested to prepare:\n"
        "1. Performance vs targets\n2. Key wins and misses\n3. Q{q+1} outlook\n4. Resource needs\n\n"
        "Please submit your decks by {date_str} for consolidation.\n\n"
        "Format: Max 10 slides per function. Executive summary first.\n\n"
        "Best,\n{sender_name}",
        "medium", "medium", True, False, True,
    ),
    (
        "Re: Supplier Issue - Delivery Delay Impacting Production",
        "John,\n\nFollowing up on the {company} supplier situation. "
        "They have confirmed a 3-week delay in delivery of critical components.\n\n"
        "Impact:\n- Production line shutdown from {date_str}\n"
        "- Revenue at risk: $6.8M\n- Customer commitments affected: 12 orders\n\n"
        "I have already contacted 2 alternative suppliers but lead times are 4+ weeks.\n\n"
        "Options:\n1. Accept delay and renegotiate customer commitments\n"
        "2. Air freight at premium cost ($280K)\n3. Partial fulfillment\n\n"
        "Need your decision today.\n\nBest,\n{sender_name}",
        "critical", "critical", True, True, True,
    ),
    (
        "Compliance Update: Annual Policy Review Due",
        "Dear Executive Team,\n\nAnnual compliance policy review is due by {date_str}.\n\n"
        "Policies requiring executive sign-off:\n"
        "- Code of Business Conduct\n- Data Privacy Policy\n"
        "- Anti-Bribery and Corruption Policy\n- Conflict of Interest Disclosure\n\n"
        "Please complete the online attestation at [internal link] by {date_str}.\n\n"
        "Non-completion will be reported to the Audit Committee.\n\n"
        "Compliance Team",
        "medium", "medium", True, False, True,
    ),
    (
        "Office Space Expansion - Decision Required",
        "Hi John,\n\nWe need to decide on office expansion by {date_str}.\n\n"
        "Options evaluated:\n"
        "1. Current location expansion: +5,000 sqft, $180K/yr\n"
        "2. New location in downtown: 15,000 sqft, $420K/yr\n"
        "3. Hybrid remote model: Reduce current space, save $95K/yr\n\n"
        "Recommendation: Option 3 based on post-pandemic work patterns.\n\n"
        "Lease expires {date_str} - decision needed by {date_str} to allow notice period.\n\n"
        "Best,\n{sender_name}",
        "medium", "medium", False, False, True,
    ),
    (
        "Thank You for the Meeting - Next Steps",
        "Hi {recipient_name},\n\nThank you for making time to meet yesterday.\n\n"
        "As discussed, here are the agreed next steps:\n"
        "1. {sender_name} to send proposal by {date_str}\n"
        "2. {recipient_name} to review and provide feedback\n"
        "3. Follow-up call scheduled for {date_str}\n\n"
        "Looking forward to our continued collaboration.\n\n"
        "Best regards,\n{sender_name}",
        "low", "low", True, False, True,
    ),
    (
        "Critical System Outage - Customer Impact",
        "INCIDENT ALERT - P1\n\nJohn,\n\nWe are experiencing a critical system outage. "
        "Our primary payment processing service has been down for 47 minutes.\n\n"
        "Impact:\n- {count} customer transactions failing per hour\n"
        "- Estimated revenue loss: $45K per hour\n"
        "- 3 enterprise customers affected\n\n"
        "Current status: Engineering team actively working on resolution.\n"
        "ETA: 2-3 hours.\n\n"
        "Should we issue a public status update? Need your approval.\n\n"
        "{sender_name}\nVP Engineering",
        "critical", "critical", True, True, True,
    ),
]


def _make_email_address(name: str, domain: str) -> str:
    parts = name.lower().split()
    return f"{parts[0]}.{parts[-1]}@{domain}"


def generate_synthetic_executive_emails(n: int = 3000) -> List[Dict]:
    """
    Generate realistic synthetic executive-style emails.

    Each email has varied topics: financial, legal, HR, operations,
    M&A, security, strategic, etc. Includes reply chains.

    Returns a list of email dicts.
    """
    random.seed(42)
    emails: List[Dict] = []
    executive_name = "John Mitchell"
    executive_email = "john.mitchell@acmecorp.com"
    base_date = datetime(2024, 1, 1)

    for i in range(n):
        template = random.choice(_TEMPLATES)
        (subj_tmpl, body_tmpl, priority_hint,
         urgency_hint, has_commitment, has_risk, has_deadline) = template

        # Random participants
        sender_name = random.choice(_EXECUTIVE_NAMES)
        while sender_name == executive_name:
            sender_name = random.choice(_EXECUTIVE_NAMES)

        domain = random.choice(_DOMAINS)
        sender_email = _make_email_address(sender_name, domain)

        # Recipient is either the executive or a third party
        recipient_name = executive_name if random.random() < 0.7 else random.choice(_EXECUTIVE_NAMES)
        recipient_email = executive_email if recipient_name == executive_name else _make_email_address(recipient_name, random.choice(_DOMAINS))

        # Random date
        offset = timedelta(days=random.randint(0, 365), hours=random.randint(6, 22), minutes=random.randint(0, 59))
        email_date = base_date + offset

        # Filler values
        q = random.randint(1, 4)
        company = random.choice(_ORGANIZATIONS)
        project = random.choice(["Project Atlas", "Project Titan", "Digital Transformation", "ERP Migration", "Platform 2.0"])
        name = random.choice(_EXECUTIVE_NAMES)
        year = random.randint(2022, 2024)
        count = random.randint(100, 5000)
        days_ahead = random.randint(1, 30)
        date_str = (email_date + timedelta(days=days_ahead)).strftime("%B %d, %Y")

        def _fmt(tmpl: str) -> str:
            try:
                return tmpl.format(
                    q=q, company=company, project=project, name=name, year=year,
                    count=count, date_str=date_str, sender_name=sender_name,
                    recipient_name=recipient_name,
                )
            except (KeyError, IndexError):
                return tmpl

        subject = _fmt(subj_tmpl)
        body = _fmt(body_tmpl)

        # Thread simulation: some emails are replies
        is_reply = random.random() < 0.35
        thread_base = hashlib.md5(subject.lower().replace("re: ", "").encode()).hexdigest()[:8]
        message_id = f"<{hashlib.md5(f'{i}{sender_email}{email_date}'.encode()).hexdigest()}@{domain}>"
        in_reply_to = f"<{thread_base}@{domain}>" if is_reply else None

        if is_reply:
            body = f"On {(email_date - timedelta(days=random.randint(1, 3))).strftime('%a, %b %d, %Y')}, {recipient_name} wrote:\n> " + body[:200].replace("\n", "\n> ") + "\n\n" + body

        email_dict = {
            "message_id": message_id,
            "subject": ("Re: " if is_reply else "") + subject,
            "sender": sender_email,
            "sender_name": sender_name,
            "recipients": [recipient_email],
            "cc": [_make_email_address(random.choice(_EXECUTIVE_NAMES), domain)] if random.random() < 0.3 else [],
            "date": email_date.isoformat(),
            "body": body,
            "in_reply_to": in_reply_to,
            "thread_id": thread_base,
            "is_reply": is_reply,
            "label_source": "heuristic",
            # Annotation hints (for training)
            "_priority_hint": priority_hint,
            "_urgency_hint": urgency_hint,
            "_has_commitment": has_commitment,
            "_has_risk": has_risk,
            "_has_deadline": has_deadline,
        }
        emails.append(email_dict)

    logger.info("Generated %d synthetic executive emails", len(emails))
    return emails


# ---------------------------------------------------------------------------
# Enron CSV loader
# ---------------------------------------------------------------------------

def load_enron_from_csv(path: str) -> List[Dict]:
    """
    Load emails from an Enron-style CSV file.
    Handles multiple possible column layouts.
    """
    emails = []
    path = Path(path)
    if not path.exists():
        logger.error("CSV path does not exist: %s", path)
        return emails

    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = [fn.lower().strip() for fn in (reader.fieldnames or [])]
        logger.info("CSV columns detected: %s", fieldnames)

        for i, row in enumerate(reader):
            # Normalize keys
            row = {k.lower().strip(): v for k, v in row.items()}
            try:
                email_dict = {
                    "message_id": row.get("message_id") or row.get("id") or f"csv_{i}",
                    "subject": row.get("subject", ""),
                    "sender": row.get("from") or row.get("sender", ""),
                    "recipients": [row.get("to") or row.get("recipients", "")] if (row.get("to") or row.get("recipients")) else [],
                    "cc": [row.get("cc", "")] if row.get("cc") else [],
                    "date": row.get("date", ""),
                    "body": row.get("body") or row.get("message") or row.get("content", ""),
                    "in_reply_to": row.get("in_reply_to", ""),
                    "thread_id": None,
                    "is_reply": False,
                    "label_source": "enron_csv",
                    "_spam": row.get("spam") or row.get("label", ""),
                }
                emails.append(email_dict)
            except Exception as e:
                logger.debug("Skipping malformed CSV row %d: %s", i, e)

    logger.info("Loaded %d emails from CSV: %s", len(emails), path)
    return emails


# ---------------------------------------------------------------------------
# Train / val / test splitting
# ---------------------------------------------------------------------------

def create_train_val_test_split(
    emails: List[Dict],
    strategy: str = "by_sender",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict[str, List[Dict]]:
    """
    Create leakage-safe splits.

    strategy='by_sender' — groups all emails from a sender into the same split,
    preventing sender-based leakage.
    strategy='by_thread' — groups by thread_id.
    strategy='random' — random split (use only as baseline).
    """
    if strategy == "by_sender":
        from collections import defaultdict
        sender_groups: Dict[str, List[Dict]] = defaultdict(list)
        for e in emails:
            key = e.get("sender", "unknown")
            sender_groups[key].append(e)
        keys = list(sender_groups.keys())
        random.seed(42)
        random.shuffle(keys)
        n = len(keys)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        train_keys = set(keys[:train_end])
        val_keys = set(keys[train_end:val_end])
        train, val, test = [], [], []
        for k, v in sender_groups.items():
            if k in train_keys:
                train.extend(v)
            elif k in val_keys:
                val.extend(v)
            else:
                test.extend(v)

    elif strategy == "by_thread":
        from collections import defaultdict
        thread_groups: Dict[str, List[Dict]] = defaultdict(list)
        for e in emails:
            key = e.get("thread_id") or e.get("message_id", "none")
            thread_groups[key].append(e)
        keys = list(thread_groups.keys())
        random.seed(42)
        random.shuffle(keys)
        n = len(keys)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        train_keys = set(keys[:train_end])
        val_keys = set(keys[train_end:val_end])
        train, val, test = [], [], []
        for k, v in thread_groups.items():
            if k in train_keys:
                train.extend(v)
            elif k in val_keys:
                val.extend(v)
            else:
                test.extend(v)
    else:
        # Random split
        random.seed(42)
        shuffled = emails[:]
        random.shuffle(shuffled)
        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        train = shuffled[:train_end]
        val = shuffled[train_end:val_end]
        test = shuffled[val_end:]

    logger.info("Split: train=%d  val=%d  test=%d  strategy=%s", len(train), len(val), len(test), strategy)
    return {"train": train, "val": val, "test": test}


def get_dataset_stats(emails: List[Dict]) -> Dict:
    """Return summary statistics for a list of email dicts."""
    if not emails:
        return {}
    senders = [e.get("sender", "") for e in emails]
    dates = []
    for e in emails:
        try:
            dates.append(datetime.fromisoformat(e["date"]) if e.get("date") else None)
        except Exception:
            dates.append(None)
    valid_dates = [d for d in dates if d is not None]
    return {
        "total": len(emails),
        "unique_senders": len(set(senders)),
        "replies": sum(1 for e in emails if e.get("is_reply")),
        "threads": len(set(e.get("thread_id", "") for e in emails if e.get("thread_id"))),
        "date_range": {
            "earliest": min(valid_dates).isoformat() if valid_dates else None,
            "latest": max(valid_dates).isoformat() if valid_dates else None,
        },
    }

