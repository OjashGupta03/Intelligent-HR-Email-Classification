"""
Page 1: Executive Overview Dashboard
High-End Executive Mission Control
"""
import os, sys
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import datetime

from dashboard.components.shared import (
    get_db, get_priority_engine, priority_badge, severity_badge,
    category_pill, render_hero, inject_custom_theme, style_plotly, no_data_message, _val
)

st.set_page_config(page_title="Executive Overview", page_icon="📋", layout="wide")
inject_custom_theme()

render_hero(
    title="Executive Overview",
    subtitle="High-level mission control — immediate priorities, critical exposures, and operational commitments.",
    badge_text="Executive Mission Control"
)

db = get_db()

try:
    stats = db.get_stats()
except Exception as e:
    st.error(f"Database error: {e}")
    st.stop()

email_count = stats.get("email_count", 0)
if email_count == 0:
    no_data_message("emails")
    st.stop()

# ── KPI Metric Row ──────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📬 Total Indexed", f"{email_count:,}")
k2.metric("🔴 Critical Items", stats.get("critical_count", 0), "Sign-off required")
k3.metric("📌 Commitments", stats.get("commitment_count", 0), "Promises active")
k4.metric("⚠️ Business Risks", stats.get("risk_count", 0), "9 Categories")
k5.metric("🧵 Threads", stats.get("thread_count", 0), "Reconstructed")

st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)

# ── Main Two-Column Layout ──────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("### 🔴 Highest Priority Communications")
    try:
        pe = get_priority_engine()
        top_emails = pe.get_top_emails(n=5)
    except Exception:
        top_emails = db.get_emails({"priority_level": "critical"})[:5]

    if not top_emails:
        st.info("No high-priority emails found yet.")
    else:
        for e in top_emails:
            subject = _val(e, "subject", "(No Subject)")[:85]
            sender = _val(e, "sender", "Unknown Sender")
            priority = (_val(e, "priority_level") or "high").lower()
            score = _val(e, "priority_score", 0)
            date = str(_val(e, "date", ""))[:16]
            preview = _val(e, "body_preview", "") or _val(e, "body", "")[:180]
            
            accent_color = "#EF4444" if priority == "critical" else "#F97316"

            st.markdown(f"""
            <div class="exec-glass-card" style="border-left:4px solid {accent_color} !important; padding:16px 20px;">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:6px;">
                    <div style="font-weight:700; font-size:1.02rem; color:#F8FAFC;">
                        {subject}
                    </div>
                    <div>{priority_badge(priority)}</div>
                </div>
                <div style="font-size:0.82rem; color:#94A3B8; margin-bottom:8px;">
                    <strong>From:</strong> {sender} &nbsp;•&nbsp; <span>{date}</span> &nbsp;•&nbsp; Score: <strong style="color:#60A5FA;">{score:.1f}/100</strong>
                </div>
                <div style="font-size:0.86rem; color:#CBD5E1; line-height:1.5;">
                    {preview}…
                </div>
            </div>
            """, unsafe_allow_html=True)

with col_right:
    st.markdown("### 📊 Priority Tier Distribution")
    priority_counts = {
        "Critical": stats.get("critical_count", 0),
        "High": stats.get("high_count", 0),
        "Medium": stats.get("medium_count", 0),
        "Low": stats.get("low_count", 0),
    }
    df_p = pd.DataFrame(list(priority_counts.items()), columns=["Priority", "Count"])
    colors = ["#EF4444", "#F97316", "#FBBF24", "#10B981"]
    fig = px.pie(
        df_p, values="Count", names="Priority",
        color_discrete_sequence=colors, hole=0.55
    )
    fig.update_traces(
        textposition="outside",
        textinfo="label+percent",
        marker=dict(line=dict(color="#0B0F19", width=2))
    )
    fig = style_plotly(fig, height=260)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<hr style='border-color:rgba(255,255,255,0.06); margin:24px 0;'>", unsafe_allow_html=True)

# ── Recent Risks & Commitments ──────────────────────────────
col_risks, col_commit = st.columns(2)

with col_risks:
    st.markdown("### ⚠️ Active Business Risks")
    risks = db.get_risks()[:5]
    if not risks:
        st.info("No risks detected.")
    else:
        for r in risks:
            cat = _val(r, "category", "general")
            sev = _val(r, "severity", "medium").lower()
            expl = _val(r, "explanation", "")[:130]
            act = _val(r, "suggested_action", "")
            
            accent = "#EF4444" if sev == "critical" else "#F97316"
            st.markdown(f"""
            <div class="exec-glass-card" style="border-left:3px solid {accent} !important; padding:12px 16px; margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                    <div>{category_pill(cat)} &nbsp; {severity_badge(sev)}</div>
                </div>
                <div style="color:#E2E8F0; font-size:0.86rem; font-weight:600; margin-top:4px;">{expl}</div>
                {f'<div style="color:#60A5FA; font-size:0.8rem; margin-top:4px;">💡 <em>Action: {act[:90]}…</em></div>' if act else ''}
            </div>
            """, unsafe_allow_html=True)

with col_commit:
    st.markdown("### 📌 Imminent Commitments")
    commitments = db.get_commitments()[:5]
    if not commitments:
        st.info("No commitments tracked.")
    else:
        for c in commitments:
            desc = _val(c, "description", "")[:110]
            resp = _val(c, "responsible_person", "Unknown")
            dl = _val(c, "deadline", "Unspecified")
            st.markdown(f"""
            <div class="exec-glass-card" style="border-left:3px solid #FBBF24 !important; padding:12px 16px; margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                    <strong style="color:#F8FAFC; font-size:0.88rem;">{desc}</strong>
                    <span style="font-size:0.78rem; color:#FCA5A5; font-weight:600;">⏰ {dl}</span>
                </div>
                <div style="color:#94A3B8; font-size:0.8rem;">
                    Responsible: <strong style="color:#CBD5E1;">{resp}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
