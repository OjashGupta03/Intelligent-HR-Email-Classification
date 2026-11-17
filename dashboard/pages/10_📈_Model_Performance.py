"""
Page 10: Model Performance & Evaluation
"""
import os, sys, json
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

import streamlit as st
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dashboard.components.shared import (
    inject_custom_theme, render_hero, style_plotly, get_db, _val
)

st.set_page_config(page_title="Model Performance | Executive AI", page_icon="📈", layout="wide")
inject_custom_theme()

render_hero(
    title="Model Performance & Evaluation",
    subtitle="Offline benchmarking experiments, NDCG ranking evaluations, multi-class confusion matrix, and feature importances.",
    badge_text="Module 08 · ML Diagnostics"
)

# ── Load experiment results ──────────────────────────────────
from config.settings import MODELS_DIR

results_path = MODELS_DIR / "experiment_results.json"
summary_path = MODELS_DIR / "training_summary.json"

results = None
if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)
elif summary_path.exists():
    with open(summary_path) as f:
        raw = json.load(f)
        results = raw.get("evaluation_results")

if not results:
    st.warning("No experiment results found. Run the training pipeline first:\n```\npython training/train_priority.py\n```")
    st.stop()

# ── Telemetry & Local Hardware ───────────────────────────────
h1, h2, h3, h4 = st.columns(4)
try:
    import torch
    cuda_avail = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if cuda_avail else "Intel / AMD CPU"
    h1.metric("Compute Device", "CUDA GPU" if cuda_avail else "CPU")
    h2.metric("Hardware Accelerat.", dev_name[:18])
except Exception:
    h1.metric("Compute Device", "CPU")
    h2.metric("Hardware Accelerat.", "Local Machine")

best_m = results.get("best_model", "logistic_regression").replace("_", " ").title()
h3.metric("Champion Model", best_m)
h4.metric("Offline Security", "100% Air-Gapped")

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# ── Baseline Comparison Table ────────────────────────────────
st.markdown('<div class="exec-glass-card">', unsafe_allow_html=True)
st.markdown('<div style="font-weight:700; font-size:1.05rem; color:#F8FAFC; margin-bottom:12px;">🏆 Offline Benchmark Comparison</div>', unsafe_allow_html=True)

per_model = results.get("per_model", {})
if per_model:
    comparison_rows = []
    for model_name, metrics in per_model.items():
        if "error" in metrics:
            continue
        comparison_rows.append({
            "Algorithm": model_name.replace("_", " ").title(),
            "Accuracy": f"{metrics.get('accuracy', 0):.4f}",
            "Macro Precision": f"{metrics.get('macro_precision', 0):.4f}",
            "Macro Recall": f"{metrics.get('macro_recall', 0):.4f}",
            "Macro F1": f"{metrics.get('macro_f1', 0):.4f}",
            "NDCG@5 (Ranking)": f"{metrics.get('ndcg_at_5', 0):.4f}",
        })
    df_comp = pd.DataFrame(comparison_rows)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    # Highlight best model
    best = results.get("best_model", "logistic_regression")
    st.markdown(f"""
    <div style="background:rgba(16, 185, 129, 0.1); border:1px solid rgba(16, 185, 129, 0.25); border-radius:8px; padding:10px 14px; margin-top:12px; display:flex; align-items:center; gap:8px;">
        <span style="color:#34D399; font-size:1.1rem;">🏆</span>
        <span style="color:#D1FAE5; font-size:0.88rem; font-weight:600;">
            Production Model Selected: <strong>{best.replace('_', ' ').title()}</strong> &bull; Optimal balance of zero-latency inference and high ranking fidelity (Macro-F1: {results.get('best_f1', 1.0):.4f}).
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# ── Confusion Matrix & Per-Class Performance ─────────────────
col_cm, col_report = st.columns([1, 1])

with col_cm:
    st.markdown('<div class="exec-glass-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-weight:700; font-size:1.02rem; color:#F8FAFC; margin-bottom:12px;">🔍 Confusion Matrix (Production Model)</div>', unsafe_allow_html=True)
    best_model_name = results.get("best_model", "logistic_regression")
    best_metrics = per_model.get(best_model_name, {})
    cm = best_metrics.get("confusion_matrix")

    if cm:
        labels = ["Critical", "High", "Low", "Medium"]
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted Class", y="Actual Ground Truth"),
            x=labels, y=labels,
            color_continuous_scale=[[0, "#0F172A"], [0.5, "#1E40AF"], [1.0, "#3B82F6"]]
        )
        fig_cm = style_plotly(fig_cm, height=280)
        fig_cm.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("Confusion matrix not available.")
    st.markdown('</div>', unsafe_allow_html=True)

with col_report:
    st.markdown('<div class="exec-glass-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-weight:700; font-size:1.02rem; color:#F8FAFC; margin-bottom:12px;">📋 Class-by-Class Evaluation Metrics</div>', unsafe_allow_html=True)
    clf_report = best_metrics.get("classification_report", {})
    if clf_report:
        report_rows = []
        for cls in ["critical", "high", "medium", "low"]:
            if cls in clf_report:
                report_rows.append({
                    "Priority Tier": cls.upper(),
                    "Precision": f"{clf_report[cls].get('precision', 0):.3f}",
                    "Recall": f"{clf_report[cls].get('recall', 0):.3f}",
                    "F1-Score": f"{clf_report[cls].get('f1-score', 0):.3f}",
                    "Test Samples": int(clf_report[cls].get('support', 0)),
                })
        st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Per-class report not available.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# ── Feature Importance ───────────────────────────────────────
st.markdown('<div class="exec-glass-card">', unsafe_allow_html=True)
st.markdown('<div style="font-weight:700; font-size:1.05rem; color:#F8FAFC; margin-bottom:14px;">🔑 Top NLP & Structural Signal Drivers</div>', unsafe_allow_html=True)
feat_imp = results.get("feature_importance", {})

if feat_imp:
    cols = st.columns(4)
    tier_colors = {
        "critical": "#EF4444",
        "high": "#F97316",
        "medium": "#FBBF24",
        "low": "#10B981"
    }
    for i, cls in enumerate(["critical", "high", "medium", "low"]):
        with cols[i]:
            c_color = tier_colors.get(cls, "#94A3B8")
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:12px 14px;">
                <div style="color:{c_color}; font-weight:800; font-size:0.85rem; letter-spacing:0.05em; text-transform:uppercase; margin-bottom:8px;">
                    {cls}
                </div>
            """, unsafe_allow_html=True)
            features = feat_imp.get(cls, [])
            if features:
                for word, weight in features[:5]:
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.82rem; margin-bottom:5px;">
                        <span style="color:#CBD5E1; font-family:'JetBrains Mono', monospace;">{word}</span>
                        <span style="color:#60A5FA; font-weight:600;">{weight:+.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<div style="color:#94A3B8; font-size:0.8rem;">No features</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Feature importance data not available.")

st.markdown('</div>', unsafe_allow_html=True)
