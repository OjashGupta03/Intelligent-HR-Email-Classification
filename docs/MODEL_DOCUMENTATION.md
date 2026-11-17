# Model Documentation

## Overview

All models in the Executive AI Chief of Staff run **100% locally** without any external API calls. The system uses a combination of pre-trained embeddings and lightweight classifiers optimized for a 16 GB RAM / RTX 2050 (4 GB VRAM) environment.

---

## 1. Embedding Model: `all-MiniLM-L6-v2`

### Specifications

| Property | Value |
|----------|-------|
| **Architecture** | MiniLM (6-layer Transformer) |
| **Source** | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| **Parameters** | ~22.7 million |
| **Output Dimension** | 384 |
| **Max Sequence Length** | 256 tokens |
| **Model Size** | ~80 MB |
| **VRAM Usage** | ~350 MB (inference) |
| **Inference Speed** | ~500 emails/sec on RTX 2050 |

### Usage in the System

- **Semantic similarity** for thread reconstruction (cosine threshold: 0.75)
- **FAISS indexing** for fast nearest-neighbor search across all 3,000 email embeddings
- **Email embedding matrix** stored as `email_embeddings.npy` (3000 × 384, ~4.4 MB)

### Why This Model?

- Small enough to fit entirely in 4 GB VRAM alongside other models
- Strong performance on semantic textual similarity benchmarks
- No fine-tuning required — works well zero-shot on email text
- Widely used and well-tested in production systems

---

## 2. TF-IDF Vectorizer

### Configuration

```python
TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),      # Unigrams + Bigrams
    sublinear_tf=True,       # log(1 + tf) scaling
    min_df=2,                # Minimum document frequency
    strip_accents="unicode", # Normalize accents
)
```

### Feature Space

| Property | Value |
|----------|-------|
| Max vocabulary size | 8,000 features |
| N-gram range | (1, 2) — unigrams and bigrams |
| TF scaling | Sublinear (logarithmic) |
| Memory footprint | ~15 MB (sparse matrix for 3,000 docs) |

### Structural Features

In addition to TF-IDF, the classifier uses structural features extracted by `FeatureExtractor`:

- Email length (characters, words, sentences)
- Subject line length and word count
- Number of recipients / CC
- Has attachments, has URLs, has financial amounts
- Hour of day, day of week
- Reply depth, thread position
- Exclamation/question mark counts
- ALL-CAPS word count

These are scaled via `StandardScaler(with_mean=False)` and concatenated with TF-IDF features.

---

## 3. Logistic Regression (Best Model)

### Configuration

```python
LogisticRegression(
    C=1.0,
    max_iter=1000,
    class_weight="balanced",  # Computed per-class
    solver="lbfgs",
    multi_class="multinomial",
    random_state=42,
)
```

### Performance

| Metric | Score |
|--------|-------|
| Accuracy | 1.0000 |
| Macro Precision | 1.0000 |
| Macro Recall | 1.0000 |
| Macro F1 | 1.0000 |
| NDCG@5 | 0.9563 |

### Memory Footprint

| Component | Size |
|-----------|------|
| Model weights | ~2 MB |
| TF-IDF vocabulary | ~12 MB |
| Total (serialized) | ~15 MB |
| RAM at inference | ~50 MB |

### Top Features per Class

| Class | Top Features |
|-------|-------------|
| critical | shortfall (0.090), attention (0.076) |
| high | earnings (0.143), earnings call (0.143), customer (0.139) |
| medium | policy (0.135), compliance (0.100), due (0.092) |
| low | 000 (0.106), business review (0.105), function (0.105) |

---

## 4. LinearSVC

### Configuration

```python
LinearSVC(
    C=1.0,
    max_iter=2000,
    class_weight="balanced",
    random_state=42,
)
```

### Performance

| Metric | Score |
|--------|-------|
| Accuracy | 1.0000 |
| Macro F1 | 1.0000 |
| NDCG@5 | 1.0000 |

### Notes

- Does not provide `predict_proba` — confidence is set to 0.75 by default
- Linear kernel only (no RBF) — fast training and inference
- Trained with balanced class weights to handle priority distribution imbalance

---

## 5. XGBoost

### Configuration

```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
    tree_method="hist",  # Histogram-based (memory efficient)
    device="cpu",        # Safe fallback
)
```

### Performance

| Metric | Score |
|--------|-------|
| Accuracy | 0.9914 |
| Macro Precision | 0.9931 |
| Macro Recall | 0.9881 |
| Macro F1 | 0.9905 |
| NDCG@5 | 0.6667 |

### Confusion Matrix

```
              Predicted
              crit  high   low   med
Actual crit    73     0     0     2
       high     0    94     0     2
        low     0     0   155     0
        med     0     0     0   140
```

### Memory Footprint

| Component | Size |
|-----------|------|
| Ensemble (200 trees) | ~30 MB |
| Total (serialized) | ~35 MB |
| RAM at inference | ~80 MB |

---

## 6. Model Selection Strategy

The system trains all three baselines and selects the best by **macro-F1 on the validation set**:

```
1. Split data: 60% train / 20% validation / 20% test (by sender)
2. Train: LR, SVM, XGBoost on train set
3. Evaluate: macro-F1 on validation set for each model
4. Select: highest macro-F1 model as best_model
5. Final eval: full metrics on held-out test set
```

### Cross-Validation

The `ModelTrainer` supports optional **5-fold cross-validation** for more robust estimates. Folds are stratified to maintain priority class distribution.

---

## Hardware Requirements

### Minimum

| Component | Requirement |
|-----------|-------------|
| CPU | Any modern x86_64 (2+ cores) |
| RAM | 8 GB |
| GPU | None (CPU fallback) |
| Disk | 500 MB free |

### Recommended (Development Environment)

| Component | Specification |
|-----------|---------------|
| CPU | Intel i7/i9 or AMD Ryzen 7/9 |
| RAM | 16 GB |
| GPU | NVIDIA RTX 2050+ (4 GB VRAM, CUDA 11.8+) |
| Disk | 2 GB free (models + database) |

### CUDA Verification

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
```

