"""
Page 2: Priority Action Inbox
"""
import os, sys
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

from dashboard.components.shared import (
    get_db, get_priority_engine, priority_badge, render_hero,
    inject_custom_theme, style_plotly, no_data_message, _val
)

st.set_page_config(page_title="Priority Inbox", page_icon="📥", layout="wide")
inject_custom_theme()

render_hero(
    title="Priority Action Inbox",
    subtitle="Communications triaged by hybrid 8-component business priority scoring and 2D urgency analysis.",
    badge_text="Real-Time Triage Engine"
)

db = get_db()

# ── Sleek Filter Bar ─────────────────────────────────────────
with st.expander("🔍 Filter & Triage Controls", expanded=True):
    fc1, fc2, fc3 = st.columns([1.5, 2, 1])
    with fc1:
        level_filter = st.multiselect(
            "Priority Tier",
            ["critical", "high", "medium", "low"],
            default=["critical", "high"]
        )
    with fc2:
        search_text = st.text_input("Search keyword (Subject, Sender, or Body)", "")
    with fc3:
        days_back = st.slider("Time Window (Days Back)", 1, 365, 45)

# ── Load & filter emails ────────────────────────────────────
try:
    all_emails = db.get_emails({})
    if not all_emails:
        no_data_message()
        st.stop()
except Exception as e:
    st.error(f"Error loading emails: {e}")
    st.stop()

cutoff = datetime.utcnow() - timedelta(days=days_back)
filtered = []
for e in all_emails:
    priority = (_val(e, "priority_level") or "low").lower()
    if level_filter and priority not in [l.lower() for l in level_filter]:
        continue
    date = _val(e, "date")
    if isinstance(date, str):
        try:
            date = datetime.fromisoformat(date)
        except Exception:
            date = None
    if date and date < cutoff:
        continue
    subject = _val(e, "subject", "")
    body = _val(e, "body", "")
    if search_text and search_text.lower() not in (subject + " " + _val(e, "sender", "") + " " + body).lower():
        continue
    filtered.append(e)

# Sort by priority score descending
_LEVEL_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}
filtered.sort(
    key=lambda x: (
        _LEVEL_ORDER.get((_val(x, "priority_level") or "low").lower(), 1),
        _val(x, "priority_score", 0)
    ),
    reverse=True
)

# ── Triage Summary & 2D Matrix View ─────────────────────────
col_count, col_matrix = st.columns([1, 1])

with col_count:
    st.markdown(f"""
    <div class="exec-glass-card" style="padding:14px 18px; margin-bottom:14px;">
        <div style="font-size:0.8rem; font-weight:700; color:#94A3B8; text-transform:uppercase;">Triage Matches</div>
        <div style="font-size:1.6rem; font-weight:800; color:#F8FAFC; margin:4px 0;">{len(filtered):,} <span style="font-size:0.9rem; font-weight:500; color:#94A3B8;">communications</span></div>
        <div style="font-size:0.82rem; color:#60A5FA;">Filtered by selected priority levels and date window.</div>
    </div>
    """, unsafe_allow_html=True)

with col_matrix:
    try:
        if filtered:
            sub = filtered[:120]
            u_scores = [_val(x, "urgency_score", 0) for x in sub]
            i_scores = [_val(x, "importance_score", 0) for x in sub]
            p_levels = [(_val(x, "priority_level") or "low").upper() for x in sub]
            sub_subjs = [_val(x, "subject", "")[:40] for x in sub]
            
            df_scatter = pd.DataFrame({
                "Urgency": u_scores,
                "Importance": i_scores,
                "Priority": p_levels,
                "Subject": sub_subjs
            })
            fig = px.scatter(
                df_scatter, x="Urgency", y="Importance", color="Priority",
                hover_data=["Subject"],
                color_discrete_map={
                    "CRITICAL": "#EF4444", "HIGH": "#F97316",
                    "MEDIUM": "#FBBF24", "LOW": "#10B981"
                }
            )
            fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.15)")
            fig.add_vline(x=50, line_dash="dash", line_color="rgba(255,255,255,0.15)")
            fig = style_plotly(fig, height=140)
            fig.update_layout(showlegend=False, margin=dict(t=5, b=5, l=25, r=10))
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

st.markdown("<hr style='border-color:rgba(255,255,255,0.06); margin:18px 0 22px 0;'>", unsafe_allow_html=True)

# ── Streamlined Email Feed ──────────────────────────────────
st.markdown(f"### 📬 Ranked Action Feed")

for i, e in enumerate(filtered[:40]):
    subject = _val(e, "subject", "(No Subject)")[:90]
    sender = _val(e, "sender", "Unknown Sender")
    priority = (_val(e, "priority_level") or "low").lower()
    score = _val(e, "priority_score", 0)
    date_str = str(_val(e, "date", ""))[:16]
    body = _val(e, "body") or ""
    body_preview = body[:210].strip()
    
    border_color = (
        "#EF4444" if priority == "critical" else
        ("#F97316" if priority == "high" else
        ("#FBBF24" if priority == "medium" else "#10B981"))
    )

    st.markdown(f"""
    <div class="exec-glass-card" style="border-left:4px solid {border_color} !important; padding:16px 20px; margin-bottom:12px;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:6px;">
            <div style="font-weight:700; font-size:1.02rem; color:#F8FAFC;">
                {subject}
            </div>
            <div>{priority_badge(priority)}</div>
        </div>
        <div style="font-size:0.82rem; color:#94A3B8; margin-bottom:8px;">
            <strong>From:</strong> {sender} &nbsp;•&nbsp; <span>{date_str}</span> &nbsp;•&nbsp; Priority Score: <strong style="color:#60A5FA;">{score:.1f}/100</strong>
        </div>
        <div style="font-size:0.87rem; color:#CBD5E1; line-height:1.5;">
            {body_preview}…
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"🔍 Inspect Full Body & Quick Actions (#{i+1})", expanded=False):
        st.text_area("Original Content", body, height=140, key=f"body_{i}", disabled=True)
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            if st.button("Mark as Acknowledged", key=f"read_{i}"):
                st.success("Acknowledged!")
        with col_b2:
            if st.button("Extract Commitments", key=f"comm_{i}"):
                from src.nlp.commitment_detector import CommitmentDetector
                cd = CommitmentDetector()
                found = cd.detect_commitments(body)
                if found:
                    for c in found:
                        st.markdown(f"• **{c['description']}** (Deadline: `{c.get('deadline','None')}`)")
                else:
                    st.info("No explicit commitments detected.")
        with col_b3:
            if st.button("Deep Dive Analysis", key=f"dd_{i}"):
                st.session_state["selected_email_id"] = _val(e, "id")
                st.info(f"Navigating to Email Intelligence analyzer for email ID #{_val(e, 'id')}")
