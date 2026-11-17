"""
Executive AI Chief of Staff - ML Priority Classifier
Multi-baseline classifier: TF-IDF+LR, TF-IDF+SVM, XGBoost.
Provides training, evaluation, and prediction.
"""
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, ndcg_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from scipy.sparse import hstack, csr_matrix

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    logger.warning("XGBoost not available; XGB baseline will be skipped.")

LABEL_ORDER = ["low", "medium", "high", "critical"]


class PriorityClassifier:
    """
    Multi-baseline email priority classifier.
    Trains TF-IDF + LogReg, TF-IDF + LinearSVC, and XGBoost.
    Selects the best model based on macro-F1 on validation set.
    """

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.scaler: Optional[StandardScaler] = None
        self.best_model_name: Optional[str] = None
        self.evaluation_results: Dict[str, Any] = {}
        self._feature_dim: int = 0

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        texts: List[str],
        structural_features: Optional[np.ndarray] = None,
        fit: bool = False,
    ) -> np.ndarray:
        """
        Build combined feature matrix: TF-IDF + structural features.
        """
        if fit:
            self.vectorizer = TfidfVectorizer(
                max_features=8000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=2,
                strip_accents="unicode",
            )
            tfidf = self.vectorizer.fit_transform(texts)
        else:
            if self.vectorizer is None:
                raise RuntimeError("Vectorizer not fitted. Call prepare_data with fit=True first.")
            tfidf = self.vectorizer.transform(texts)

        if structural_features is not None:
            if fit:
                self.scaler = StandardScaler(with_mean=False)
                struct = self.scaler.fit_transform(csr_matrix(structural_features))
            else:
                struct = self.scaler.transform(csr_matrix(structural_features)) if self.scaler else csr_matrix(structural_features)
            X = hstack([tfidf, struct])
        else:
            X = tfidf

        return X

    def encode_labels(self, labels: List[str], fit: bool = False) -> np.ndarray:
        """Encode string labels to integers."""
        if fit:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(LABEL_ORDER)
        if self.label_encoder is None:
            raise RuntimeError("Label encoder not fitted.")
        # Map unknown labels to 'medium'
        safe_labels = [l if l in LABEL_ORDER else "medium" for l in labels]
        return self.label_encoder.transform(safe_labels)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_logistic_regression(self, X_train, y_train) -> Dict:
        logger.info("Training Logistic Regression...")
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        cw = dict(zip(classes, weights))
        model = LogisticRegression(
            C=1.0, max_iter=1000, class_weight=cw,
            solver="lbfgs", multi_class="multinomial", random_state=42,
        )
        model.fit(X_train, y_train)
        self.models["logistic_regression"] = model
        train_preds = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)
        logger.info("LR train accuracy: %.4f", train_acc)
        return {"model": model, "train_accuracy": train_acc}

    def train_svm(self, X_train, y_train) -> Dict:
        logger.info("Training LinearSVC...")
        model = LinearSVC(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
        model.fit(X_train, y_train)
        self.models["svm"] = model
        train_preds = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)
        logger.info("SVM train accuracy: %.4f", train_acc)
        return {"model": model, "train_accuracy": train_acc}

    def train_xgboost(self, X_train, y_train) -> Dict:
        if not _HAS_XGB:
            logger.warning("XGBoost not available, skipping.")
            return {}
        logger.info("Training XGBoost...")
        n_classes = len(np.unique(y_train))
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            tree_method="hist",
            device="cpu",  # Safe for all hardware
        )
        # XGBoost needs dense for small data
        X_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
        model.fit(X_dense, y_train)
        self.models["xgboost"] = model
        train_preds = model.predict(X_dense)
        train_acc = accuracy_score(y_train, train_preds)
        logger.info("XGB train accuracy: %.4f", train_acc)
        return {"model": model, "train_accuracy": train_acc}

    def train_all_baselines(
        self, X_train, y_train, X_val, y_val
    ) -> Dict:
        """Train all baselines and select best by macro-F1."""
        results: Dict[str, Any] = {}

        # Train each
        self.train_logistic_regression(X_train, y_train)
        self.train_svm(X_train, y_train)
        if _HAS_XGB:
            self.train_xgboost(X_train, y_train)

        # Evaluate on validation set
        best_f1 = -1.0
        for name, model in self.models.items():
            try:
                if name == "xgboost":
                    X_v = X_val.toarray() if hasattr(X_val, "toarray") else X_val
                else:
                    X_v = X_val
                preds = model.predict(X_v)
                acc = accuracy_score(y_val, preds)
                p, r, f1, _ = precision_recall_fscore_support(y_val, preds, average="macro", zero_division=0)
                results[name] = {"accuracy": acc, "precision": p, "recall": r, "f1": f1}
                logger.info("[%s] Val  acc=%.4f  macro-F1=%.4f", name, acc, f1)
                if f1 > best_f1:
                    best_f1 = f1
                    self.best_model_name = name
            except Exception as e:
                logger.error("Error evaluating %s: %s", name, e)
                results[name] = {"error": str(e)}

        logger.info("Best model: %s (F1=%.4f)", self.best_model_name, best_f1)
        self.evaluation_results["comparison"] = results
        return results

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, text: str, structural: Optional[np.ndarray] = None) -> Dict:
        """Predict priority for a single email text."""
        if not self.models or self.best_model_name not in self.models:
            return {"priority_level": "medium", "confidence": 0.5, "probabilities": {}}
        try:
            X = self.prepare_data([text], structural_features=structural.reshape(1, -1) if structural is not None else None)
            model = self.models[self.best_model_name]
            if self.best_model_name == "xgboost":
                X = X.toarray() if hasattr(X, "toarray") else X
            pred_int = model.predict(X)[0]
            label = self.label_encoder.inverse_transform([pred_int])[0]

            # Probabilities (not available for LinearSVC)
            probs = {}
            if hasattr(model, "predict_proba"):
                prob_arr = model.predict_proba(X)[0]
                probs = {self.label_encoder.classes_[i]: float(p) for i, p in enumerate(prob_arr)}
                confidence = float(max(prob_arr))
            else:
                confidence = 0.75  # SVM doesn't provide calibrated probabilities

            return {"priority_level": label, "confidence": confidence, "probabilities": probs}
        except Exception as e:
            logger.error("Prediction error: %s", e)
            return {"priority_level": "medium", "confidence": 0.5, "probabilities": {}}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X_test, y_test) -> Dict:
        """Full evaluation on test set for all models."""
        results: Dict[str, Any] = {}
        for name, model in self.models.items():
            try:
                X_t = X_test.toarray() if (name == "xgboost" and hasattr(X_test, "toarray")) else X_test
                preds = model.predict(X_t)
                acc = accuracy_score(y_test, preds)
                p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
                cm = confusion_matrix(y_test, preds).tolist()
                report = classification_report(
                    y_test, preds,
                    target_names=[self.label_encoder.classes_[i] for i in sorted(set(y_test))],
                    zero_division=0, output_dict=True,
                )
                # Ranking metrics
                ndcg = self._compute_ndcg(y_test, model, X_t)
                results[name] = {
                    "accuracy": round(acc, 4),
                    "macro_precision": round(p, 4),
                    "macro_recall": round(r, 4),
                    "macro_f1": round(f1, 4),
                    "confusion_matrix": cm,
                    "classification_report": report,
                    "ndcg_at_5": round(ndcg, 4),
                }
            except Exception as e:
                logger.error("Evaluation error for %s: %s", name, e)
                results[name] = {"error": str(e)}

        self.evaluation_results["test"] = results
        return results

    def _compute_ndcg(self, y_true, model, X) -> float:
        """Compute NDCG@5 treating priority as relevance score."""
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                # Use predicted prob of critical/high as relevance
                relevance = proba[:, -1] + proba[:, -2]  # critical + high prob
            else:
                preds = model.predict(X)
                relevance = preds / (len(LABEL_ORDER) - 1)

            true_relevance = y_true / (len(LABEL_ORDER) - 1)
            # ndcg_score expects 2D arrays
            return ndcg_score([true_relevance[:100]], [relevance[:100]], k=5)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "models": self.models,
            "vectorizer": self.vectorizer,
            "label_encoder": self.label_encoder,
            "scaler": self.scaler,
            "best_model_name": self.best_model_name,
            "evaluation_results": self.evaluation_results,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("Classifier saved to %s", path)

    def load_model(self, path: str) -> bool:
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            self.models = payload["models"]
            self.vectorizer = payload["vectorizer"]
            self.label_encoder = payload["label_encoder"]
            self.scaler = payload.get("scaler")
            self.best_model_name = payload["best_model_name"]
            self.evaluation_results = payload.get("evaluation_results", {})
            logger.info("Classifier loaded from %s (best: %s)", path, self.best_model_name)
            return True
        except Exception as e:
            logger.error("Failed to load classifier: %s", e)
            return False

    def get_feature_importance(self, model_name: Optional[str] = None) -> Dict:
        """Return top feature importances for LR model."""
        name = model_name or self.best_model_name
        model = self.models.get(name)
        if model is None or not hasattr(model, "coef_"):
            return {}
        if self.vectorizer is None:
            return {}
        feature_names = self.vectorizer.get_feature_names_out().tolist()
        coef = model.coef_
        result = {}
        for i, cls in enumerate(self.label_encoder.classes_):
            if i < len(coef):
                top_indices = np.argsort(coef[i])[-15:][::-1]
                result[cls] = [(feature_names[j], round(float(coef[i][j]), 4)) for j in top_indices if j < len(feature_names)]
        return result

