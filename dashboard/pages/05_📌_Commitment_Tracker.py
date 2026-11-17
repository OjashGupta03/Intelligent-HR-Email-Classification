"""
Page 5: Commitment & Deadline Radar
"""
import os, sys
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

import streamlit as st
import pandas as pd
from datetime import datetime
from dashboard.components.shared import (
    get_db, priority_badge, render_hero, inject_custom_theme,
    no_data_message, _val
)

st.set_page_config(page_title="Commitment Tracker", page_icon="📌", layout="wide")
inject_custom_theme()

render_hero(
    title="Commitment & Deadline Radar",
    subtitle="35+ pattern linguistic detector for promises, agreements, obligations, and target dates.",
    badge_text="Accountability Intelligence"
)

db = get_db()

try:
    all_commitments = db.get_commitments()
    if not all_commitments:
        no_data_message("commitments")
        st.stop()
except Exception as e:
    st.error(f"Database error: {e}")
    st.stop()

# ── Summary Metrics ──────────────────────────────────────────
status_counts = {"pending": 0, "overdue": 0, "completed": 0, "uncertain": 0}
for c in all_commitments:
    s = (_val(c, "status") or "pending").lower()
    if s in status_counts:
        status_counts[s] += 1

m1, m2, m3, m4 = st.columns(4)
m1.metric("🟡 Active Pending", status_counts["pending"], "Awaiting delivery")
m2.metric("🔴 Overdue Deadlines", status_counts["overdue"], "Requires escalation")
m3.metric("🟢 Completed", status_counts["completed"], "Verified resolved")
m4.metric("📊 Total Commitments", len(all_commitments), "Extracted from corpus")

st.markdown("<div style='margin-bottom:18px;'></div>", unsafe_allow_html=True)

# ── Filters Bar ──────────────────────────────────────────────
with st.expander("🔍 Filter Commitments", expanded=True):
    col1, col2 = st.columns([1, 2])
    with col1:
        status_filter = st.multiselect(
            "Filter Status",
            ["pending", "overdue", "completed", "uncertain"],
            default=["pending", "overdue"]
        )
    with col2:
        search_commitment = st.text_input("Search description or assignee", "")

filtered = []
for c in all_commitments:
    status = (_val(c, "status") or "pending").lower()
    if status_filter and status not in [s.lower() for s in status_filter]:
        continue
    desc = (_val(c, "description") or "").lower() + " " + (_val(c, "responsible_person") or "").lower()
    if search_commitment and search_commitment.lower() not in desc:
        continue
    filtered.append(c)

st.markdown(f"<div style='font-size:0.88rem; color:#94A3B8; margin-bottom:12px;'>Showing <strong>{len(filtered)} commitments</strong></div>", unsafe_allow_html=True)

# ── Commitments Grid View ────────────────────────────────────
STATUS_STYLING = {
    "pending": {"color": "#FBBF24", "bg": "rgba(251, 191, 36, 0.12)", "border": "rgba(251, 191, 36, 0.35)", "label": "PENDING"},
    "overdue": {"color": "#EF4444", "bg": "rgba(239, 68, 68, 0.12)", "border": "rgba(239, 68, 68, 0.35)", "label": "OVERDUE"},
    "completed": {"color": "#10B981", "bg": "rgba(16, 185, 129, 0.12)", "border": "rgba(16, 185, 129, 0.35)", "label": "COMPLETED"},
    "uncertain": {"color": "#94A3B8", "bg": "rgba(148, 163, 184, 0.12)", "border": "rgba(148, 163, 184, 0.35)", "label": "UNCERTAIN"},
}

for c in filtered[:25]:
    status = (_val(c, "status") or "pending").lower()
    desc = _val(c, "description", "")
    resp = _val(c, "responsible_person", "Unknown")
    dl = _val(c, "deadline", "Unspecified")
    conf = _val(c, "confidence", 0)
    source = _val(c, "source_text", "")
    
    st_cfg = STATUS_STYLING.get(status, STATUS_STYLING["pending"])

    st.markdown(f"""
    <div class="exec-glass-card" style="border-left:4px solid {st_cfg['color']} !important; padding:14px 18px; margin-bottom:10px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <div style="font-weight:700; font-size:0.98rem; color:#F8FAFC;">
                {desc}
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="background:{st_cfg['bg']}; border:1px solid {st_cfg['border']}; color:{st_cfg['color']}; padding:2px 8px; border-radius:12px; font-size:10px; font-weight:700; letter-spacing:0.04em;">
                    {st_cfg['label']}
                </span>
                <span style="font-size:0.8rem; color:#FCA5A5; font-weight:600;">⏰ {dl}</span>
            </div>
        </div>
        <div style="font-size:0.82rem; color:#94A3B8; margin-bottom:4px;">
            Assignee: <strong style="color:#CBD5E1;">{resp}</strong> &nbsp;•&nbsp; AI Confidence: <strong style="color:#60A5FA;">{conf:.0%}</strong>
        </div>
        {f'<div style="font-size:0.8rem; color:#64748B; font-style:italic; margin-top:4px;">"{source}"</div>' if source else ''}
    </div>
    """, unsafe_allow_html=True)
