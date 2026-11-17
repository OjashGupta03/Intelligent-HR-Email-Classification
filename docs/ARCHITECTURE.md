# Architecture Documentation

## System Architecture

The Executive AI Chief of Staff follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                           │
│                   Streamlit Multi-Page Dashboard                    │
│              (10 executive-focused views + dark theme)               │
├─────────────────────────────────────────────────────────────────────┤
│                        INTELLIGENCE LAYER                           │
│  Priority Engine │ Conversation Memory │ Commitment Tracker         │
│  Risk Monitor    │ Daily Briefing      │ Relationship Intelligence  │
│                  │ "What Did I Miss?"  │                            │
├─────────────────────────────────────────────────────────────────────┤
│                          NLP LAYER                                  │
│  Feature Extractor │ Urgency/Importance Detector │ Summarizer       │
│  Commitment Detector (35+ patterns) │ Risk Detector (9 categories) │
├─────────────────────────────────────────────────────────────────────┤
│                           ML LAYER                                  │
│  Embeddings (all-MiniLM-L6-v2) │ FAISS Indexing │ TF-IDF Vectorizer│
│  Priority Classifier (LR, SVM, XGBoost) │ Cross-Validation         │
├─────────────────────────────────────────────────────────────────────┤
│                         DATA LAYER                                  │
│  Email Parser │ Preprocessor │ Annotation Pipeline │ Dataset Loader │
│               SQLite Database (SQLAlchemy 2.0 ORM)                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Priority Scoring Engine

### 8-Component Hybrid Scoring

The Priority Engine computes a 0–100 score as a weighted combination of eight dimensions:

| Component | Weight | Description |
|-----------|--------|-------------|
| `urgency` | 0.20 | Keyword matching (URGENT, ASAP, critical, etc.) + subject analysis |
| `business_impact` | 0.20 | Revenue, legal, compliance, customer, and project impact keywords |
| `sender_importance` | 0.15 | VIP lists, C-suite title detection, contact history, gov domains |
| `deadline_proximity` | 0.12 | Deadline references, same-day/tomorrow/this-week detection |
| `financial_impact` | 0.10 | Monetary amounts ($M, $B), financial keywords (invoice, write-off) |
| `action_required` | 0.10 | Action phrases, question marks, decision-required patterns |
| `strategic_importance` | 0.08 | Strategy, board, IPO, transformation, roadmap keywords |
| `conversation_context` | 0.05 | Thread activity, reply chains, sender history |

### Score Formula

```
final_score = Σ (component_score_i × weight_i)

# Non-linear scaling:
if final_score > 70:  final_score = 70 + (final_score - 70) × 1.2
if final_score < 20:  final_score = final_score × 0.9
```

### Priority Level Mapping

| Level | Minimum Score | Color |
|-------|:------------:|-------|
| CRITICAL | 85 | 🔴 Red |
| HIGH | 65 | 🟠 Orange |
| MEDIUM | 40 | 🟡 Amber |
| LOW | 0 | 🟢 Green |

### ML Boosting

When the ML classifier predicts CRITICAL or HIGH but rules say LOW/MEDIUM, a boost is applied:
- CRITICAL prediction → +15 points
- HIGH prediction → +10 points

---

## Eisenhower Matrix (2D Scoring)

Urgency and Importance are scored independently on 0–100 scales:

```
                    HIGH IMPORTANCE
                         │
    Urgent+Important     │    Not-Urgent+Important
    (Do First)           │    (Schedule)
                         │
   ──────────────────────┼──────────────────────
                         │
    Urgent+Not-Important │    Not-Urgent+Not-Important
    (Delegate)           │    (Eliminate/Defer)
                         │
                    LOW IMPORTANCE
```

**Urgency signals:** Time-sensitive keywords, deadline proximity, exclamation marks, ALL-CAPS
**Importance signals:** Financial amounts, board/executive references, strategic keywords, VIP senders

Threshold: score ≥ 50 → "urgent" or "important"

---

## Risk Detection Architecture

Nine business risk categories, each with 15–20 regex patterns:

| Category | Example Triggers | Default Severity |
|----------|-----------------|-----------------|
| `financial` | revenue shortfall, budget overrun, cash flow concern | High |
| `legal_compliance` | lawsuit, subpoena, SEC investigation, GDPR violation | High |
| `customer_dissatisfaction` | churn, complaint, cancel, dissatisfied | High |
| `project_delay` | delayed, behind schedule, slippage, blocked | High |
| `contract` | expires, breach, renegotiation, SLA violation | Medium |
| `security` | data breach, unauthorized access, vulnerability | Critical |
| `escalation` | escalated, VP attention, CEO intervention | High |
| `missed_deadline` | overdue, past due, deadline missed | High |
| `operational` | outage, system failure, production halt | High |

**Severity Assessment:**
```
base_severity = category_default
for each boost_keyword matched:
    severity += boost_value
confidence = min(0.5 + num_matches × 0.1, 0.95)
```

---

## Commitment Detection

35+ regex patterns organized by type:

| Type | Example Patterns | Confidence Range |
|------|-----------------|-----------------|
| **Promise** | "I will send…", "I'll follow up…", "I plan to…" | 0.68–0.95 |
| **Agreement** | "We commit to…", "We agreed to…", "I commit to…" | 0.90–0.95 |
| **Request** | "Please arrange…", "Please confirm…", "Please provide…" | 0.78–0.82 |
| **Obligation** | "We should complete…" | 0.75 |

**Responsible Person Extraction:**
- Sentence starts with "I" → "Executive (self)"
- Sentence starts with "We" → "Team"
- Contains named entity patterns → Extracted name
- Starts with "Please" → "Recipient"

---

## Conversation Memory

Three-pass thread reconstruction:

1. **Header-based linking** — `In-Reply-To` and `References` headers (Union-Find)
2. **Subject-line matching** — Normalized subjects (strip `Re:`, `Fwd:` prefixes)
3. **Semantic similarity** — Embedding cosine similarity (threshold: 0.75)

### Extracted Intelligence

- **Pending Questions** — Unanswered questions detected via `?` pattern matching
- **Decision Points** — Keywords: "decided", "approved", "agreed", "going with"
- **Unresolved Issues** — Issue keywords without matching resolution keywords
- **Key Points** — High-TF sentences from thread bodies

---

## SQLite Database Schema

```
┌────────────────┐     ┌─────────────────┐
│     emails     │────>│ email_priorities │
│  (3,000 rows)  │     │   (906 rows)    │
├────────────────┤     ├─────────────────┤
│ id (PK)        │     │ email_id (FK)   │
│ message_id     │     │ priority_level  │
│ subject        │     │ priority_score  │
│ sender         │     │ urgency_score   │
│ recipients     │     │ importance_score│
│ date           │     │ 6 sub-scores    │
│ body           │     │ explanation     │
│ thread_id      │     └─────────────────┘
│ in_reply_to    │
│ is_reply       │     ┌────────────────┐
│ label_source   │────>│  commitments   │
└────────────────┘     │  (124 rows)    │
                       ├────────────────┤
┌────────────────┐     │ email_id (FK)  │
│    threads     │     │ description    │
│  (329 rows)    │     │ responsible    │
├────────────────┤     │ deadline       │
│ thread_id      │     │ deadline_date  │
│ subject        │     │ status         │
│ participants   │     │ confidence     │
│ email_count    │     └────────────────┘
│ first/last_date│
│ summary        │     ┌────────────────┐
│ status         │     │     risks      │
└────────────────┘     │  (1,542 rows)  │
                       ├────────────────┤
┌────────────────┐     │ email_id (FK)  │
│   contacts     │     │ category       │
│  (190 rows)    │     │ severity       │
├────────────────┤     │ explanation    │
│ email_address  │     │ suggested_action│
│ name           │     │ confidence     │
│ importance     │     └────────────────┘
│ is_vip         │
│ total_sent     │     ┌──────────────────┐
│ total_received │     │ daily_briefings  │
└────────────────┘     │   (1 row)        │
                       ├──────────────────┤
┌────────────────┐     │ date             │
│  annotations   │     │ content (JSON)   │
│  (labelled)    │     │ generated_at     │
├────────────────┤     └──────────────────┘
│ email_id (FK)  │
│ priority_label │
│ urgency_label  │
│ action_required│
│ has_commitment │
│ has_risk       │
│ label_type     │
└────────────────┘
```

**Indexes:** Optimized for common queries — date+sender, thread+date, priority level+score, severity+category, VIP contacts.

**WAL Mode:** SQLite Write-Ahead Logging enabled for concurrent dashboard reads during writes.

---

## Data Flow

```
Raw Emails (RFC 2822 / Synthetic JSON)
    │
    ▼
[Email Parser] → Clean body, strip quotes, normalize headers
    │
    ▼
[Preprocessor] → URL removal, whitespace normalization, tokenization
    │
    ▼
[Annotation Pipeline] → Heuristic labels (priority, urgency, importance)
    │
    ├──► [Feature Extractor] → Structural + linguistic feature vectors
    │       │
    │       ▼
    │    [ML Classifier] → TF-IDF + LR/SVM/XGBoost predictions
    │
    ├──► [Embedding Manager] → all-MiniLM-L6-v2 (384-dim) + FAISS index
    │
    ├──► [Commitment Detector] → Extract promises, deadlines, responsible parties
    │
    ├──► [Risk Detector] → Categorize risks with severity and actions
    │
    ├──► [Urgency/Importance] → 2D Eisenhower scoring
    │
    └──► [Summarizer] → Extractive sentence scoring
            │
            ▼
    [Priority Engine] → Weighted 8-component score → SQLite persistence
            │
            ▼
    [Intelligence Modules] → Conversation Memory, Commitment Tracker,
                             Risk Monitor, Daily Briefing, What Did I Miss?,
                             Relationship Intelligence
            │
            ▼
    [Streamlit Dashboard] → 10-page executive interface
```

