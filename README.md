# Executive AI Chief of Staff

**An Intelligent Email Prioritization and Decision Intelligence System**

> 100% local processing • No external APIs • Built for business executives

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/downloads/)
[![CUDA Enabled](https://img.shields.io/badge/CUDA-Enabled-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Executive AI Chief of Staff is a fully local NLP-powered system that acts as an intelligent email triage and decision intelligence assistant for busy executives. It processes email communications to surface what matters most — critical decisions, financial risks, overdue commitments, and relationship insights — without sending a single byte to external servers.

### Key Capabilities

| Feature | Description |
|---------|-------------|
| **Priority Scoring** | 8-component hybrid engine (urgency, business impact, sender importance, deadlines, financial impact, action required, strategic importance, conversation context) |
| **Risk Detection** | 9 business risk categories with severity grading and actionable recommendations |
| **Commitment Tracking** | 35+ pattern detector for promises, agreements, requests, and obligations |
| **2D Urgency × Importance** | Independent Eisenhower Matrix scoring with quadrant mapping |
| **Conversation Memory** | Thread reconstruction via header linking, subject matching, and semantic similarity |
| **Relationship Intelligence** | Contact importance scoring, communication frequency analysis, and network graphs |
| **Daily Executive Briefing** | Auto-generated morning summary with top priorities, risks, and deadlines |
| **"What Did I Miss?"** | Flagship feature — catch up on any time period with intelligent summaries |
| **Extractive Summarization** | TF-IDF + position-weighted sentence scoring for emails and threads |
| **ML Classification** | Multi-baseline comparison (Logistic Regression, LinearSVC, XGBoost) with NDCG@5 |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard (10 Pages)                │
├─────────────────────────────────────────────────────────────────┤
│ Overview │ Inbox │ Intel │ Memory │ Commit │ Risk │ Brief │ ... │
├─────────────────────────────────────────────────────────────────┤
│                    Intelligence Engine                           │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │ Priority │ │Commitment │ │   Risk   │ │  Conversation    │  │
│  │  Engine  │ │  Tracker  │ │ Monitor  │ │     Memory       │  │
│  └──────────┘ └───────────┘ └──────────┘ └──────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    NLP Pipeline                                  │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │ Feature  │ │Urgency/Imp│ │Summarizer│ │    Embeddings    │  │
│  │Extractor │ │ Detector  │ │          │ │ (MiniLM-L6-v2)   │  │
│  └──────────┘ └───────────┘ └──────────┘ └──────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Data Layer                                    │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │  Parser  │ │Preprocessor│ │Annotation│ │   SQLite + ORM   │  │
│  │          │ │           │ │ Pipeline │ │  (SQLAlchemy 2.0) │  │
│  └──────────┘ └───────────┘ └──────────┘ └──────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│              ML Models (Scikit-learn + XGBoost)                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐    │
│  │ TF-IDF + LR  │ │TF-IDF + SVM  │ │ XGBoost (tree_hist)  │    │
│  └──────────────┘ └──────────────┘ └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- **OS:** Windows 10/11
- **Python:** 3.13+
- **GPU:** NVIDIA GPU with CUDA support (optional, falls back to CPU)
- **RAM:** 16 GB recommended

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd "NLP project"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"

# 4. Train models and populate database
python training/train_priority.py

# 5. Run evaluations
python evaluation/run_all_evaluations.py

# 6. Launch dashboard
streamlit run dashboard/app.py
```

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for detailed step-by-step instructions.

---

## Project Structure

```
NLP project/
├── config/
│   ├── settings.py              # Central configuration (device, weights, thresholds)
│   └── scoring_weights.yaml     # Tunable 8-component scoring weights
├── src/
│   ├── data/
│   │   ├── email_parser.py      # RFC 2822 email parser & quote stripper
│   │   ├── dataset_loader.py    # Synthetic generator (3,000 executive emails)
│   │   ├── preprocessor.py      # Text cleaning, tokenization, metadata extraction
│   │   └── annotation_pipeline.py # Heuristic annotation for training labels
│   ├── nlp/
│   │   ├── feature_extractor.py # Structural + linguistic feature vectors
│   │   ├── commitment_detector.py # 35+ pattern commitment/promise detector
│   │   ├── risk_detector.py     # 9-category business risk detector
│   │   ├── urgency_detector.py  # 2D urgency × importance scoring
│   │   └── summarizer.py       # Extractive TF/position summarization
│   ├── ml/
│   │   ├── embeddings.py       # all-MiniLM-L6-v2 encoder + FAISS indexing
│   │   ├── priority_classifier.py # Multi-baseline classifier (LR, SVM, XGBoost)
│   │   └── trainer.py          # Training orchestration & cross-validation
│   ├── intelligence/
│   │   ├── priority_engine.py   # 8-component hybrid priority scoring
│   │   ├── intelligence_modules.py # Core intelligence implementations
│   │   ├── conversation_memory.py  # Thread reconstruction & context tracking
│   │   ├── commitment_tracker.py   # Commitment lifecycle management
│   │   ├── risk_monitor.py         # Risk monitoring & alerting
│   │   └── daily_briefing.py       # Automated briefing generation
│   ├── database/
│   │   ├── models.py           # SQLAlchemy 2.0 ORM (8 tables)
│   │   └── db_manager.py       # Session management & query API
│   └── utils/
│       └── helpers.py          # Shared utilities (date parsing, similarity, etc.)
├── dashboard/
│   ├── app.py                  # Main Streamlit app with executive dark theme
│   ├── components/shared.py    # Reusable UI components
│   └── pages/                  # 10 dashboard views
│       ├── 01_executive_overview.py
│       ├── 02_priority_inbox.py
│       ├── 03_email_intelligence.py
│       ├── 04_conversation_memory.py
│       ├── 05_commitment_tracker.py
│       ├── 06_risk_monitor.py
│       ├── 07_daily_briefing.py
│       ├── 08_what_did_i_miss.py    # Flagship feature
│       ├── 09_relationship_intelligence.py
│       └── 10_model_performance.py
├── training/
│   └── train_priority.py       # End-to-end training pipeline
├── evaluation/
│   └── run_all_evaluations.py  # Benchmark suite (ML, Risk, ROUGE, Commitments)
├── tests/
│   ├── test_parser.py          # Email parsing & preprocessing tests
│   ├── test_scoring.py         # Urgency/importance & priority engine tests
│   ├── test_commitments.py     # Commitment detection tests
│   ├── test_risks.py           # Risk detection tests
│   └── test_memory.py          # Conversation memory & thread reconstruction tests
├── data/
│   ├── executive_ai.db         # SQLite database (3,000 emails, auto-generated)
│   ├── models/                 # Saved model artifacts
│   ├── raw/                    # Raw synthetic email data
│   └── processed/              # Processed annotations
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATASET_GUIDE.md
│   ├── MODEL_DOCUMENTATION.md
│   ├── EXPERIMENTS.md
│   ├── INSTALLATION.md
│   └── USER_GUIDE.md
└── requirements.txt
```

---

## Benchmark Results

| Component | Metric | Score |
|-----------|--------|-------|
| **ML Classifier (LR)** | Accuracy / Macro-F1 | 1.0000 |
| **ML Classifier (SVM)** | Accuracy / Macro-F1 | 1.0000 |
| **ML Classifier (XGBoost)** | Accuracy / Macro-F1 | 0.9914 / 0.9905 |
| **Risk Detection** | Macro-F1 (9 categories) | 0.8889 |
| **Commitment Detection** | F1-Score | 1.0000 |
| **Summarization** | ROUGE-L F-measure | 0.3535 |
| **Urgency/Importance** | Quadrant Accuracy | 0.5000 |

---

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, scoring formulas, SQLite schema |
| [DATASET_GUIDE.md](docs/DATASET_GUIDE.md) | Synthetic dataset generation & annotation taxonomy |
| [MODEL_DOCUMENTATION.md](docs/MODEL_DOCUMENTATION.md) | Model details, hyperparameters, memory footprint |
| [EXPERIMENTS.md](docs/EXPERIMENTS.md) | Baseline comparisons & evaluation tables |
| [INSTALLATION.md](docs/INSTALLATION.md) | Step-by-step Windows setup guide |
| [USER_GUIDE.md](docs/USER_GUIDE.md) | Dashboard walkthrough for all 10 pages |

---

## Technology Stack

- **Language:** Python 3.13
- **NLP:** NLTK, Sentence-Transformers (`all-MiniLM-L6-v2`)
- **ML:** Scikit-learn, XGBoost
- **Database:** SQLite + SQLAlchemy 2.0
- **Dashboard:** Streamlit + Plotly + NetworkX
- **Embeddings:** FAISS for similarity search
- **GPU:** CUDA-accelerated (RTX 2050 compatible)

---

## Privacy & Security

This system is designed with a **zero-trust, zero-cloud** architecture:

- ✅ All processing happens locally on your machine
- ✅ No data leaves your network
- ✅ No API keys, tokens, or cloud credentials required
- ✅ SQLite database stored locally
- ✅ Model weights are open-source and downloaded once
- ❌ No OpenAI, Claude, Gemini, or any paid API calls

---

## License

MIT License. See [LICENSE](LICENSE) for details.

#   I n t e l l i g e n t - H R - E m a i l - C l a s s i f i c a t i o n  
 