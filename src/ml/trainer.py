"""
Executive AI Chief of Staff - Model Trainer
Orchestrates training, evaluation, and experiment tracking.
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import MODELS_DIR
from src.ml.priority_classifier import PriorityClassifier, LABEL_ORDER
from src.nlp.feature_extractor import FeatureExtractor
from src.ml.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


def _safe_label(label: str) -> str:
    return label if label in LABEL_ORDER else "medium"


class ModelTrainer:
    """Orchestrates the full training pipeline."""

    def __init__(self, db_manager, feature_extractor: FeatureExtractor, embedding_manager: EmbeddingManager):
        self.db = db_manager
        self.feature_extractor = feature_extractor
        self.embeddings = embedding_manager
        self.classifier = PriorityClassifier()
        self.results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _extract_texts_labels_features(self, emails: List[Dict]) -> Tuple[List[str], List[str], np.ndarray]:
        """Extract texts, labels, and structural feature vectors."""
        texts, labels, features = [], [], []
        for e in emails:
            text = f"{e.get('subject', '')} {e.get('body', '') or e.get('cleaned_body', '')}"
            label = _safe_label(e.get("annotations", {}).get("priority_label", "medium"))
            try:
                feat = self.feature_extractor.build_feature_vector(e)
            except Exception:
                feat = np.zeros(32, dtype=np.float32)
            texts.append(text[:2000])
            labels.append(label)
            features.append(feat)
        return texts, labels, np.vstack(features) if features else np.zeros((0, 32))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_priority_model_from_emails(
        self,
        train_emails: List[Dict],
        val_emails: List[Dict],
        test_emails: List[Dict],
    ) -> Dict:
        """Full training pipeline from email dicts."""
        logger.info("Preparing training data (%d emails)...", len(train_emails))
        train_texts, train_labels, train_feats = self._extract_texts_labels_features(train_emails)
        val_texts, val_labels, val_feats = self._extract_texts_labels_features(val_emails)

        if not train_texts:
            logger.error("No training data available.")
            return {}

        # Encode labels
        self.classifier.encode_labels(train_labels, fit=True)
        y_train = self.classifier.encode_labels(train_labels)
        y_val = self.classifier.encode_labels(val_labels)

        # Build feature matrices
        logger.info("Building TF-IDF + structural feature matrices...")
        X_train = self.classifier.prepare_data(train_texts, train_feats, fit=True)
        X_val = self.classifier.prepare_data(val_texts, val_feats)

        # Train all baselines
        comparison = self.classifier.train_all_baselines(X_train, y_train, X_val, y_val)
        self.results["comparison"] = comparison
        self.results["train_size"] = len(train_texts)
        self.results["val_size"] = len(val_texts)

        # Save classifier
        model_path = MODELS_DIR / "priority_classifier.pkl"
        self.classifier.save_model(str(model_path))

        return {"comparison": comparison, "best_model": self.classifier.best_model_name}

    def evaluate_all_models_from_emails(self, test_emails: List[Dict]) -> Dict:
        """Evaluate all trained models on test set."""
        if not test_emails:
            return {"error": "No test data"}
        test_texts, test_labels, test_feats = self._extract_texts_labels_features(test_emails)
        y_test = self.classifier.encode_labels(test_labels)
        X_test = self.classifier.prepare_data(test_texts, test_feats)

        eval_results = self.classifier.evaluate(X_test, y_test)
        self.results["test_evaluation"] = eval_results
        self.results["test_size"] = len(test_texts)

        # Find best model by test macro-F1
        best_model, best_f1 = "unknown", 0.0
        for name, res in eval_results.items():
            f1 = res.get("macro_f1", 0.0)
            if isinstance(f1, (int, float)) and f1 > best_f1:
                best_f1, best_model = f1, name

        # Feature importance for LR
        feature_importance = self.classifier.get_feature_importance("logistic_regression")

        summary = {
            "best_model": best_model,
            "best_f1": best_f1,
            "per_model": eval_results,
            "feature_importance": feature_importance,
        }
        self.results.update(summary)
        return summary

    def run_cross_validation(
        self, emails: List[Dict], model_name: str = "logistic_regression", k: int = 5
    ) -> Dict:
        """K-fold cross-validation for a specific model."""
        texts, labels, feats = self._extract_texts_labels_features(emails)
        if not texts:
            return {}
        self.classifier.encode_labels(labels, fit=True)
        y = self.classifier.encode_labels(labels)
        X = self.classifier.prepare_data(texts, feats, fit=True)

        model = self.classifier.models.get(model_name)
        if model is None:
            logger.warning("Model %s not found, using LR.", model_name)
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")

        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        cv_results = cross_validate(
            model, X, y, cv=cv,
            scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"],
            return_train_score=True,
        )
        summary = {
            "k": k,
            "model": model_name,
            "mean_accuracy": float(np.mean(cv_results["test_accuracy"])),
            "mean_f1": float(np.mean(cv_results["test_f1_macro"])),
            "mean_precision": float(np.mean(cv_results["test_precision_macro"])),
            "mean_recall": float(np.mean(cv_results["test_recall_macro"])),
            "std_f1": float(np.std(cv_results["test_f1_macro"])),
        }
        self.results["cross_validation"] = summary
        return summary

    def save_results(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info("Experiment results saved to %s", path)

    def load_results(self, path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            self.results = json.load(f)
        return self.results

    def generate_experiment_report(self) -> Dict:
        """Return a dict summarising the experiment."""
        return {
            "generated_at": datetime.now().isoformat(),
            "best_model": self.results.get("best_model", "unknown"),
            "best_f1": self.results.get("best_f1", 0.0),
            "comparison": self.results.get("comparison", {}),
            "test_evaluation": self.results.get("test_evaluation", {}),
            "cross_validation": self.results.get("cross_validation", {}),
        }

