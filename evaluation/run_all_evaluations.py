"""
Executive AI Chief of Staff — Evaluation & Benchmark Suite
Runs all evaluations: ML classifier metrics, Risk detection F1,
Summarization ROUGE, and Priority Engine accuracy.
All processing is 100% local — no external API calls.
"""
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# OpenMP fix for Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from config.settings import MODELS_DIR, RISK_CATEGORIES
from src.database.db_manager import DatabaseManager
from src.nlp.commitment_detector import CommitmentDetector
from src.nlp.risk_detector import RiskDetector
from src.nlp.summarizer import EmailSummarizer
from src.nlp.urgency_detector import UrgencyImportanceDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. ML Classifier Metrics (from saved training results)
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_ml_classifiers() -> Dict[str, Any]:
    """Load and display saved ML experiment results."""
    results_path = MODELS_DIR / "training_summary.json"
    if not results_path.exists():
        logger.warning("No training_summary.json found. Run training first.")
        return {}

    with open(results_path, "r") as f:
        data = json.load(f)

    eval_data = data.get("evaluation_results", {})
    per_model = eval_data.get("per_model", {})

    print("\n" + "=" * 70)
    print("  ML CLASSIFIER BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Best Model: {eval_data.get('best_model', 'N/A')}")
    print(f"  Best Macro-F1: {eval_data.get('best_f1', 'N/A')}")
    print("-" * 70)
    print(f"  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'NDCG@5':>10}")
    print("-" * 70)

    for name, metrics in per_model.items():
        if "error" in metrics:
            print(f"  {name:<25} ERROR: {metrics['error']}")
            continue
        print(
            f"  {name:<25} "
            f"{metrics.get('accuracy', 0):>10.4f} "
            f"{metrics.get('macro_precision', 0):>10.4f} "
            f"{metrics.get('macro_recall', 0):>10.4f} "
            f"{metrics.get('macro_f1', 0):>10.4f} "
            f"{metrics.get('ndcg_at_5', 0):>10.4f}"
        )

    # Per-class breakdown for best model
    best = eval_data.get("best_model", "")
    if best and best in per_model:
        report = per_model[best].get("classification_report", {})
        print(f"\n  Per-Class Report ({best}):")
        print(f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("  " + "-" * 55)
        for cls in ["critical", "high", "medium", "low"]:
            if cls in report:
                r = report[cls]
                print(
                    f"  {cls:<15} "
                    f"{r.get('precision', 0):>10.4f} "
                    f"{r.get('recall', 0):>10.4f} "
                    f"{r.get('f1-score', 0):>10.4f} "
                    f"{r.get('support', 0):>10.0f}"
                )

    # Feature importance
    feat_imp = eval_data.get("feature_importance", {})
    if feat_imp:
        print("\n  Top Features by Class:")
        for cls, features in feat_imp.items():
            top = features[:4] if isinstance(features, list) else []
            top_str = ", ".join(f"{f[0]} ({f[1]:.3f})" for f in top)
            print(f"    {cls}: {top_str}")

    print()
    return per_model


# ──────────────────────────────────────────────────────────────────────────────
# 2. Risk Detection Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_risk_detection() -> Dict[str, Any]:
    """Evaluate risk detector on synthetic test cases with known labels."""
    detector = RiskDetector()

    # Gold-standard test cases with known risk categories
    test_cases = [
        ("Our Q4 revenue is showing a significant shortfall compared to guidance, with a budget overrun of $2M.",
         ["financial"]),
        ("We received a subpoena from the SEC regarding our accounting practices and potential litigation.",
         ["legal_compliance"]),
        ("Multiple enterprise customers are threatening to churn due to service quality issues.",
         ["customer_dissatisfaction"]),
        ("The product launch is delayed by 6 weeks due to unresolved engineering blockers.",
         ["project_delay"]),
        ("The vendor contract expires next week and we have not agreed on renewal terms yet.",
         ["contract"]),
        ("Our security team detected a data breach in the production database affecting user credentials.",
         ["security"]),
        ("This issue has been escalated to the VP level after three failed resolution attempts.",
         ["escalation"]),
        ("The compliance report deadline was missed and the regulatory body has issued a warning notice.",
         ["missed_deadline"]),
        ("The data center experienced a complete system outage affecting all production services.",
         ["operational"]),
        ("Hope you are doing well. Let us have lunch sometime next week.",
         []),
    ]

    # Track per-category TP/FP/FN
    category_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for text, expected_cats in test_cases:
        detected = detector.detect_risks(text)
        detected_cats = {r["category"] for r in detected}

        for cat in RISK_CATEGORIES:
            expected = cat in expected_cats
            detected_flag = cat in detected_cats
            if expected and detected_flag:
                category_stats[cat]["tp"] += 1
            elif not expected and detected_flag:
                category_stats[cat]["fp"] += 1
            elif expected and not detected_flag:
                category_stats[cat]["fn"] += 1

    print("\n" + "=" * 70)
    print("  RISK DETECTION EVALUATION")
    print("=" * 70)
    print(f"  {'Category':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 70)

    all_p, all_r, all_f1 = [], [], []
    for cat in RISK_CATEGORIES:
        stats = category_stats[cat]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0  # no expected → perfect recall
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        all_p.append(precision)
        all_r.append(recall)
        all_f1.append(f1)
        print(f"  {cat:<25} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")

    macro_p = np.mean(all_p)
    macro_r = np.mean(all_r)
    macro_f1 = np.mean(all_f1)
    print("-" * 70)
    print(f"  {'MACRO AVERAGE':<25} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f}")
    print()

    return {"macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f1}


# ──────────────────────────────────────────────────────────────────────────────
# 3. Summarization ROUGE Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_summarization() -> Dict[str, float]:
    """Evaluate extractive summariser using ROUGE scores on sample emails."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logger.warning("rouge-score not installed. Skipping ROUGE evaluation.")
        return {}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    summarizer = EmailSummarizer()

    # Sample emails with hand-written reference summaries
    test_pairs = [
        {
            "email": (
                "Dear Board Members,\n\n"
                "I am writing to inform you of a critical situation regarding our Q3 financial results. "
                "Revenue fell short of projections by $4.2 million, primarily due to delayed enterprise deals. "
                "The sales pipeline shows recovery potential in Q4 with $12M in committed deals. "
                "We need to schedule an emergency board meeting to discuss cost reduction measures. "
                "Our CFO has prepared a detailed analysis of the shortfall and proposed remediation steps. "
                "Please confirm your availability for Thursday at 2 PM EST.\n\n"
                "Best regards,\nJohn Smith, CEO"
            ),
            "reference": (
                "Q3 revenue fell short by $4.2 million due to delayed enterprise deals. "
                "Q4 pipeline shows $12M in committed deals. "
                "Emergency board meeting needed for cost reduction. CFO has prepared analysis."
            ),
        },
        {
            "email": (
                "Team,\n\n"
                "The product launch for Project Aurora has been delayed by two weeks. "
                "The engineering team identified a critical security vulnerability in the authentication module. "
                "QA testing revealed three additional edge cases that need to be addressed. "
                "Marketing has agreed to adjust the campaign schedule accordingly. "
                "We will hold daily standup meetings until the launch is back on track. "
                "The revised launch date is November 28th.\n\n"
                "Thanks,\nSarah"
            ),
            "reference": (
                "Project Aurora launch delayed two weeks due to security vulnerability. "
                "QA found three additional edge cases. Revised launch date: November 28th."
            ),
        },
    ]

    rouge_totals: Dict[str, Dict[str, float]] = defaultdict(lambda: {"precision": 0, "recall": 0, "fmeasure": 0})
    count = len(test_pairs)

    for pair in test_pairs:
        summary = summarizer.summarize_email(pair["email"], max_sentences=3)
        scores = scorer.score(pair["reference"], summary)
        for metric_name, metric_val in scores.items():
            rouge_totals[metric_name]["precision"] += metric_val.precision
            rouge_totals[metric_name]["recall"] += metric_val.recall
            rouge_totals[metric_name]["fmeasure"] += metric_val.fmeasure

    print("\n" + "=" * 70)
    print("  SUMMARIZATION ROUGE EVALUATION")
    print("=" * 70)
    print(f"  {'Metric':<15} {'Precision':>12} {'Recall':>12} {'F-measure':>12}")
    print("-" * 70)

    results = {}
    for metric in ["rouge1", "rouge2", "rougeL"]:
        p = rouge_totals[metric]["precision"] / count
        r = rouge_totals[metric]["recall"] / count
        f = rouge_totals[metric]["fmeasure"] / count
        results[f"{metric}_f"] = round(f, 4)
        print(f"  {metric:<15} {p:>12.4f} {r:>12.4f} {f:>12.4f}")

    print()
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 4. Commitment Detection Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_commitment_detection() -> Dict[str, float]:
    """Evaluate commitment detector on labelled examples."""
    detector = CommitmentDetector()

    test_cases = [
        ("I will send the updated financial projections by Friday afternoon.", True),
        ("We commit to delivering the final audit report by EOD Monday.", True),
        ("Please provide the Q3 revenue spreadsheet by tomorrow.", True),
        ("I will schedule a follow-up call with the board of directors.", True),
        ("I promise to complete the vendor evaluation by next week.", True),
        ("The weather is nice today and the office is quiet.", False),
        ("Thank you for the update on the project status.", False),
        ("Looking forward to the holiday party next month.", False),
    ]

    tp = fp = fn = tn = 0
    for text, has_commitment in test_cases:
        detected = detector.detect_commitments(text)
        found = len(detected) > 0
        if has_commitment and found:
            tp += 1
        elif has_commitment and not found:
            fn += 1
        elif not has_commitment and found:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(test_cases)

    print("\n" + "=" * 70)
    print("  COMMITMENT DETECTION EVALUATION")
    print("=" * 70)
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  (TP={tp}, FP={fp}, FN={fn}, TN={tn})")
    print()

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# ──────────────────────────────────────────────────────────────────────────────
# 5. Urgency/Importance 2D Scoring Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_urgency_importance() -> Dict[str, Any]:
    """Evaluate 2D scoring on test examples with expected quadrants."""
    detector = UrgencyImportanceDetector()

    test_cases = [
        ("URGENT: Server outage affecting all customers. Need immediate fix!", "urgent+important"),
        ("Please review the office supply order for next month when you get a chance.", "not-urgent+not-important"),
        ("The board meeting presentation needs final sign-off before Friday.", "urgent+important"),
        ("FYI: Updated the team wiki with new onboarding docs.", "not-urgent+not-important"),
    ]

    correct = 0
    for text, expected_quadrant in test_cases:
        urgency = detector.detect_urgency(text)
        importance = detector.detect_importance(text)
        quadrant = detector.get_urgency_importance_matrix(urgency["score"], importance["score"])
        if quadrant == expected_quadrant:
            correct += 1

    accuracy = correct / len(test_cases)

    print("\n" + "=" * 70)
    print("  URGENCY/IMPORTANCE 2D SCORING EVALUATION")
    print("=" * 70)
    print(f"  Quadrant Accuracy: {accuracy:.4f} ({correct}/{len(test_cases)})")
    print()

    return {"quadrant_accuracy": accuracy}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "#" * 70)
    print("#  Executive AI Chief of Staff — Full Evaluation Suite")
    print(f"#  Run at: {datetime.now().isoformat()}")
    print("#" * 70)

    all_results: Dict[str, Any] = {"timestamp": datetime.now().isoformat()}

    # 1. ML Classifiers
    all_results["ml_classifiers"] = evaluate_ml_classifiers()

    # 2. Risk Detection
    all_results["risk_detection"] = evaluate_risk_detection()

    # 3. Summarization ROUGE
    all_results["summarization"] = evaluate_summarization()

    # 4. Commitment Detection
    all_results["commitment_detection"] = evaluate_commitment_detection()

    # 5. Urgency/Importance
    all_results["urgency_importance"] = evaluate_urgency_importance()

    # Save results
    output_path = PROJECT_ROOT / "evaluation" / "evaluation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("=" * 70)
    print(f"  Results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
