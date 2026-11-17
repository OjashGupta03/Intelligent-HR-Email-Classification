"""
Executive AI Chief of Staff - Configuration Settings
Central configuration for all modules.
"""
import os
import yaml
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
MODELS_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "executive_ai.db"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, ANNOTATIONS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Environment workaround for OpenMP on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ─── Device Configuration ───
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Config] Using device: {DEVICE}")

# ─── Model Configuration ───
# Small models that fit in 4GB VRAM / 16GB RAM
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # ~80MB, 384-dim
ZERO_SHOT_MODEL_NAME = "typeform/distilbert-base-uncased-mnli"  # ~250MB

# ─── Scoring Weights (default, overridable from YAML) ───
DEFAULT_SCORING_WEIGHTS = {
    "urgency": 0.20,
    "business_impact": 0.20,
    "sender_importance": 0.15,
    "deadline_proximity": 0.12,
    "financial_impact": 0.10,
    "action_required": 0.10,
    "strategic_importance": 0.08,
    "conversation_context": 0.05,
}

def load_scoring_weights() -> dict:
    """Load scoring weights from YAML config, falling back to defaults."""
    weights_path = CONFIG_DIR / "scoring_weights.yaml"
    if weights_path.exists():
        with open(weights_path, "r") as f:
            data = yaml.safe_load(f)
            if data and "weights" in data:
                return data["weights"]
    return DEFAULT_SCORING_WEIGHTS.copy()

# ─── Priority Levels ───
PRIORITY_LEVELS = {
    "CRITICAL": {"min_score": 85, "color": "#FF0000"},
    "HIGH":     {"min_score": 65, "color": "#FF6600"},
    "MEDIUM":   {"min_score": 40, "color": "#FFB800"},
    "LOW":      {"min_score": 0,  "color": "#28A745"},
}

def get_priority_level(score: float) -> str:
    """Map a 0-100 score to a priority level."""
    for level, cfg in PRIORITY_LEVELS.items():
        if score >= cfg["min_score"]:
            return level
    return "LOW"

# ─── VIP / Important Senders Configuration ───
VIP_SENDERS = {
    # Format: email_address -> importance_score (0-100)
    # Users can add their own VIP senders
}

VIP_DOMAINS = {
    # Format: domain -> importance_boost
    # e.g., "sec.gov": 30, "irs.gov": 25
}

# ─── NLP Configuration ───
MAX_EMAIL_LENGTH = 2000  # characters for processing
MIN_EMAIL_LENGTH = 10
BATCH_SIZE = 32
SIMILARITY_THRESHOLD = 0.75  # for thread reconstruction

# ─── Risk Categories ───
RISK_CATEGORIES = [
    "financial",
    "legal_compliance",
    "customer_dissatisfaction",
    "project_delay",
    "contract",
    "security",
    "escalation",
    "missed_deadline",
    "operational",
]

# ─── Commitment Statuses ───
COMMITMENT_STATUSES = ["pending", "completed", "overdue", "uncertain"]

# ─── Database Configuration ───
DB_URL = f"sqlite:///{DB_PATH}"

