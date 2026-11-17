# Installation & Setup Guide

This guide provides instructions for setting up and running the **Executive AI Chief of Staff** system on a Windows workstation with Python 3.13, Anaconda or venv, and NVIDIA GPU acceleration (CUDA).

---

## 1. System Requirements

### Hardware
- **Operating System:** Windows 10 or 11 (64-bit)
- **RAM:** Minimum 8 GB (16 GB recommended)
- **GPU:** NVIDIA GPU with 4GB+ VRAM (e.g., RTX 2050, 3050, 4060) or CPU
- **Disk Space:** ~2 GB for SQLite database, embeddings, and dependencies

### Software
- **Python:** 3.13.x (or 3.10+)
- **NVIDIA Driver:** Driver version 520+ supporting CUDA 11.8 or 12.x
- **Git:** Git for Windows

---

## 2. Environment Preparation

### Option A: Using Existing Anaconda Environment (Recommended)

Open **PowerShell** or **Anaconda Prompt**:

```powershell
# Navigate to project directory
cd "c:\Users\Hitesh m\OneDrive\Desktop\NLP project"

# Confirm active Python is 3.13
python --version
```

### Option B: Using a Python Virtual Environment

```powershell
cd "c:\Users\Hitesh m\OneDrive\Desktop\NLP project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

## 3. Dependency Installation

Install all required packages from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

### Required Dependencies Overview:
- **NLP & Embeddings:** `sentence-transformers`, `nltk`, `rouge-score`, `faiss-cpu`
- **Machine Learning:** `scikit-learn`, `xgboost`, `scipy`, `numpy`
- **Database:** `sqlalchemy>=2.0.0`
- **Dashboard & Visualization:** `streamlit`, `plotly`, `networkx`
- **Configuration & Utilities:** `pyyaml`, `pytest`

### NLTK Tokenizer & Lexicon Downloads

Run this one-liner to download necessary tokenizers:

```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

---

## 4. Hardware & CUDA Verification

Verify PyTorch device detection and GPU availability:

```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected Output:
```text
CUDA Available: True
Device Name: NVIDIA GeForce RTX 2050
```

> [!NOTE]
> On Windows workstations with Intel MKL and PyTorch, an OpenMP conflict (`libiomp5md.dll`) may occur. The system already automatically handles this via `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` across all configuration files and entry points.

---

## 5. Data Generation & Model Training

Execute the end-to-end pipeline script to generate the synthetic corpus, extract features, train all ML baseline models, populate the SQLite database, and compute priority scores:

```powershell
python training/train_priority.py
```

### What This Script Does:
1. Generates 3,000 realistic executive emails in `data/raw/synthetic_emails.json`.
2. Computes 384-dimensional dense semantic embeddings using `all-MiniLM-L6-v2`.
3. Runs heuristic annotation for ground-truth priority and risk profiling.
4. Trains and validates `Logistic Regression`, `LinearSVC`, and `XGBoost`.
5. Saves serialized models into `data/models/priority_classifier.pkl`.
6. Populates `data/executive_ai.db` with emails, threads, risks, commitments, and contacts.

---

## 6. Running the Evaluation Suite

Verify system benchmarks and compute test-set metrics:

```powershell
python evaluation/run_all_evaluations.py
```

This will run evaluations for:
- Priority classifier (Accuracy, F1, NDCG@5)
- Business risk detection across 9 corporate categories
- ROUGE-1, ROUGE-2, and ROUGE-L extractive summarization scores
- Commitment and deadline extraction accuracy
- Eisenhower 2D matrix quadrant mapping

---

## 7. Running Unit Tests

Run the test suite with `pytest`:

```powershell
pytest tests/ -v
```

All 20 unit tests across parser, scoring, commitments, risks, and memory modules should pass.

---

## 8. Launching the Executive Dashboard

Launch the multi-page Streamlit application:

```powershell
streamlit run dashboard/app.py
```

The browser will open to:
```
http://localhost:8501
```

---

## 9. Troubleshooting

### Port In Use
If port 8501 is occupied, specify an alternative port:
```powershell
streamlit run dashboard/app.py --server.port 8502
```

### PyTorch Out of Memory
The system defaults to a batch size of 32 for sentence embeddings, which consumes only ~350MB of VRAM on RTX 2050. If running other GPU-heavy software, adjust `BATCH_SIZE = 16` in `config/settings.py`.

