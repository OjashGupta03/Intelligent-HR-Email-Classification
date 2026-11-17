# Executive User Guide

Welcome to the **Executive AI Chief of Staff** dashboard. This guide walks you through every screen and feature designed to save you hours of inbox management each day.

---

## Navigation & Dark Executive Interface

The dashboard uses an **Executive Obsidian & Amber** aesthetic. The left-hand sidebar displays real-time telemetry:
- Total emails processed
- Critical & High priority inbox counts
- Active commitments & Overdue alerts
- High-severity business risks
- Direct navigation links to all 10 views

---

## 1. Executive Overview (`01_executive_overview.py`)
**Purpose:** Your morning mission control at a glance.

### Key Sections:
- **KPI Metric Banners:** Total emails, Critical items, Active business risks, Pending commitments, and VIP communications.
- **Priority Distribution Chart:** Interactive donut chart breaking down Critical, High, Medium, and Low priorities.
- **Eisenhower Matrix View:** 2D scatter of emails plotted by Urgency (X-axis) vs. Importance (Y-axis) into 4 actionable quadrants:
  - *Do First* (Urgent + Important)
  - *Schedule* (Not-Urgent + Important)
  - *Delegate* (Urgent + Not-Important)
  - *Eliminate/Defer* (Not-Urgent + Not-Important)
- **Immediate Attention Queue:** The top 5 highest-priority communications requiring executive sign-off today.

---

## 2. Priority Inbox (`02_priority_inbox.py`)
**Purpose:** Triage and process high-impact emails without getting bogged down by noise.

### Key Sections:
- **Filter Controls:** Filter by Priority Tier (Critical, High, Medium, Low), Category, or Search Query.
- **Executive Card View:** Clean cards showing sender, company domain, priority badge, urgency badge, and deadline notice.
- **One-Click Actions:**
  - *View Deep-Dive Analysis:* Opens full breakdown.
  - *Mark as Done:* Archive or acknowledge communication.
  - *Extract Commitments:* Instantly extract promises or deadlines to your task list.

---

## 3. Email Intelligence Deep Dive (`03_email_intelligence.py`)
**Purpose:** Explainable, transparent AI analysis of any individual email message.

### Key Sections:
- **8-Component Priority Breakdown:** Interactive radar/bar visual displaying raw and weighted scores for:
  - Urgency
  - Business Impact
  - Sender Importance
  - Deadline Proximity
  - Financial Impact
  - Action Required
  - Strategic Importance
  - Conversation Context
- **AI Explanation & Evidence:** Natural language explanation detailing exactly *why* an email received its score, citing specific triggers and dollar amounts.
- **Extractive Summary:** 2–3 sentence executive digest.

---

## 4. Conversation Memory (`04_conversation_memory.py`)
**Purpose:** Eliminate the pain of digging through 20-message email chains.

### Key Sections:
- **Thread Reconstructor:** Displays all related emails grouped chronologically into a single cohesive story.
- **Decision Tracker:** Automatically extracts confirmed decisions ("We agreed to acquire...", "Approved $2.5M budget").
- **Unresolved Questions:** Surfaces open questions asked earlier in the thread that remain unanswered by other parties.
- **Participant Flow:** Timeline of who said what and when.

---

## 5. Commitment Tracker (`05_commitment_tracker.py`)
**Purpose:** Never drop a promise made by you or to you.

### Key Sections:
- **Commitment Status Board:** Categorized by *Pending*, *Completed*, and *Overdue*.
- **Directional Ownership:**
  - *My Commitments:* Promises you made ("I will deliver the board slides by Friday").
  - *Waiting on Others:* Promises made to you by vendors, partners, or subordinates.
- **Deadline Calendar & Countdown:** Color-coded countdown badges highlighting imminent deadlines.

---

## 6. Risk Monitor (`06_risk_monitor.py`)
**Purpose:** Proactive alerting across 9 corporate business risk categories.

### Key Sections:
- **Risk Category Selector:**
  - Financial (shortfalls, budget overruns)
  - Legal & Compliance (subpoenas, GDPR, regulatory audits)
  - Customer Dissatisfaction (churn threats, tier-1 escalations)
  - Project Delay (blocked deliverables, milestone slippage)
  - Contract & Vendor (expirations, renewal disputes)
  - Security & Privacy (breaches, vulnerabilities)
  - Executive Escalation (VP/Board escalations)
  - Missed Deadlines (overdue compliance filings)
  - Operational (system downtime, server outages)
- **Severity Ranking:** Critical, High, Medium severity filters with recommended remediation actions.

---

## 7. Daily Executive Briefing (`07_daily_briefing.py`)
**Purpose:** Read this over your morning coffee in under 3 minutes.

### Key Sections:
- **Automated Morning Digest:** Generated daily summary synthesized from the past 24 hours.
- **Top 3 Priorities Today:** The single most critical decisions or approvals needed.
- **Upcoming Deadlines & Due Today:** Chronological list of commitments due before EOD.
- **Risk Highlights:** Newly detected corporate exposures.
- **Export & Print:** Print-friendly formatting for offline review or tablet reading.

---

## 8. "What Did I Miss?" (`08_what_did_i_miss.py`) — Flagship Feature
**Purpose:** Returning from vacation, an offsite, or a 4-hour board meeting? Instantly catch up.

### Key Sections:
- **Time Window Selector:** Select *Last 4 Hours*, *Today*, *Yesterday*, *Last 3 Days*, *Past Week*, or a *Custom Date Range*.
- **Executive Catch-Up Summary:** Synthesized overview of what transpired during your absence.
- **Critical Developments:** Key topics that emerged while you were away.
- **New Commitments & Risks:** Any promises made on your behalf or risks reported.
- **Top Communications Requiring Your Reply:** Filtered list of emails directly addressing you.

---

## 9. Relationship Intelligence (`09_relationship_intelligence.py`)
**Purpose:** Understand your communication network and protect key relationships.

### Key Sections:
- **VIP Contact Roster:** Scored contacts based on frequency, seniority, and strategic significance.
- **Interactive Network Graph:** NetworkX force-directed graph illustrating communication clusters, frequent threads, and bridge contacts.
- **Relationship Health & Cadence:** Flags VIP contacts you haven't responded to in more than 5 days.
- **Organization Breakdown:** Communication volume grouped by company domain (internal teams vs. investors vs. enterprise customers).

---

## 10. Model Performance & Analytics (`10_model_performance.py`)
**Purpose:** For CTOs, technical founders, and auditing teams to inspect underlying AI health.

### Key Sections:
- **Model Comparison Table:** Live benchmarks comparing Logistic Regression, LinearSVC, and XGBoost across Accuracy, Precision, Recall, and F1.
- **NDCG@5 Ranking Metrics:** Evaluates how effectively the models rank the top 5 most urgent emails.
- **Confusion Matrix:** Interactive heatmap displaying classification distributions.
- **Feature Importance:** Top positive and negative coefficients driving predictions.
- **Local Hardware Status:** Verifies NVIDIA CUDA device utilization and VRAM allocation.

