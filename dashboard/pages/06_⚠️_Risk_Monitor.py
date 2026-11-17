"""
Page 6: Business Risk Monitor
Proactive intelligence across 9 corporate risk categories.
"""
import os, sys
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

import streamlit as st
import plotly.express as px
import pandas as pd
from dashboard.components.shared import (
    get_db, priority_badge, severity_badge, category_pill, render_hero,
    inject_custom_theme, style_plotly, no_data_message, _val
)

st.set_page_config(page_title="Risk Monitor", page_icon="⚠️", layout="wide")
inject_custom_theme()

render_hero(
    title="Business Risk Monitor & Radar",
    subtitle="Proactive multi-category detection covering financial variance, legal exposure, SLA breaches, and project delays.",
    badge_text="Enterprise Risk Governance"
)

db = get_db()

try:
    all_risks = db.get_risks()
    if not all_risks:
        no_data_message("risks")
        st.stop()
except Exception as e:
    st.error(f"Database error: {e}")
    st.stop()

# ── Summary metrics ──────────────────────────────────────────
sev_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
for r in all_risks:
    sev = (_val(r, "severity") or "medium").lower()
    if sev in sev_counts:
        sev_counts[sev] += 1

k1, k2, k3, k4 = st.columns(4)
k1.metric("🔴 Critical Threats", sev_counts["critical"], "Regulatory, Outage, Legal")
k2.metric("🟠 High Severity", sev_counts["high"], "Churn, Financial, Project")
k3.metric("🟡 Medium Concerns", sev_counts["medium"], "Operational, Compliance")
k4.metric("📊 Total Risks Detected", len(all_risks), "9 Taxonomies")

st.markdown("<div style='margin-bottom:18px;'></div>", unsafe_allow_html=True)

# ── Filters & Category Breakdown ────────────────────────────
col_filters, col_chart = st.columns([1, 1.2])

with col_filters:
    st.markdown("### 🔍 Filter Risk Signals")
    all_categories = sorted(list(set(_val(r, "category", "") for r in all_risks if _val(r, "category"))))
    sel_categories = st.multiselect("Risk Category", all_categories, default=all_categories[:4] if all_categories else [])
    sel_severities = st.multiselect("Severity Tier", ["critical", "high", "medium", "low"], default=["critical", "high"])

with col_chart:
    st.markdown("### 📊 Distribution Across Taxonomies")
    cat_counts = {}
    for r in all_risks:
        cat = _val(r, "category", "unknown").replace("_", " ").title()
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    df_cat = pd.DataFrame(list(cat_counts.items()), columns=["Category", "Count"]).sort_values(by="Count", ascending=True)
    fig = px.bar(
        df_cat, x="Count", y="Category", orientation="h",
        color="Count", color_continuous_scale="Reds"
    )
    fig = style_plotly(fig, height=220)
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

# ── Filtered Risks Feed ──────────────────────────────────────
filtered_risks = []
for r in all_risks:
    cat = _val(r, "category", "")
    sev = (_val(r, "severity") or "medium").lower()
    if sel_categories and cat not in sel_categories:
        continue
    if sel_severities and sev not in sel_severities:
        continue
    filtered_risks.append(r)

st.markdown(f"### ⚠️ Identified Business Risks ({len(filtered_risks):,} matching)")

for r in filtered_risks[:25]:
    cat = _val(r, "category", "general")
    sev = (_val(r, "severity") or "medium").lower()
    expl = _val(r, "explanation", "")
    act = _val(r, "suggested_action", "")
    src = _val(r, "source_text", "")
    conf = _val(r, "confidence", 0)

    accent_color = "#EF4444" if sev == "critical" else ("#F97316" if sev == "high" else "#FBBF24")
    
    st.markdown(f"""
    <div class="exec-glass-card" style="border-left:4px solid {accent_color} !important; padding:16px 20px; margin-bottom:12px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <div style="display:flex; align-items:center; gap:8px;">
                {category_pill(cat)}
                {severity_badge(sev)}
            </div>
            <span style="font-size:0.8rem; color:#94A3B8;">Confidence: <strong style="color:#60A5FA;">{conf:.0%}</strong></span>
        </div>
        <div style="color:#F8FAFC; font-size:0.95rem; font-weight:700; margin:6px 0;">{expl}</div>
        {f'<div style="color:#94A3B8; font-size:0.83rem; font-style:italic; margin-bottom:8px;">"{src}"</div>' if src else ''}
        <div style="background:rgba(59,130,246,0.08); border-left:3px solid #3B82F6; padding:8px 12px; border-radius:6px; margin-top:6px;">
            <span style="color:#93C5FD; font-size:0.83rem;">💡 <strong>Suggested Mitigation Action:</strong> {act}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
