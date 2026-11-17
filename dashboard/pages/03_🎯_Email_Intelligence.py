"""
Page 3: Email Intelligence Deep-Dive
Explainable 8-component priority radar and NLP extraction breakdown.
"""
import os, sys
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

import streamlit as st
import plotly.graph_objects as go
from dashboard.components.shared import (
    get_db, get_priority_engine, priority_badge, severity_badge,
    category_pill, render_hero, inject_custom_theme, style_plotly, no_data_message, _val
)

st.set_page_config(page_title="Email Intelligence", page_icon="🔬", layout="wide")
inject_custom_theme()

render_hero(
    title="Email Intelligence Analyzer",
    subtitle="Deep NLP feature extraction, 8-component priority radar breakdown, and evidence citation.",
    badge_text="Explainable AI Deep-Dive"
)

db = get_db()

search = st.text_input("🔍 Quick Search by Subject or Sender", "")

try:
    emails = db.get_emails({})
    if not emails:
        no_data_message()
        st.stop()
except Exception as e:
    st.error(f"Database error: {e}")
    st.stop()

# Filter by search
if search:
    emails = [e for e in emails if search.lower() in (_val(e,"subject","") + " " + _val(e,"sender","")).lower()]

options = {
    f"[{(_val(e,'priority_level') or 'low').upper()}] {_val(e,'subject','(No Subject)')[:65]} — {_val(e,'sender','')}"[:110]: e 
    for e in emails[:120]
}

if not options:
    st.info("No emails found matching your search.")
    st.stop()

# If pre-selected from another page
default_index = 0
selected_id = st.session_state.get("selected_email_id")
if selected_id:
    for idx, (label, em) in enumerate(options.items()):
        if _val(em, "id") == selected_id:
            default_index = idx
            break

selected_label = st.selectbox("Select Target Communication to Inspect", list(options.keys()), index=default_index)
selected = options[selected_label]

st.markdown("<hr style='border-color:rgba(255,255,255,0.06); margin:18px 0;'>", unsafe_allow_html=True)

c_body, c_radar = st.columns([1.3, 1])

with c_body:
    priority = (_val(selected, "priority_level") or "low").lower()
    subject = _val(selected, "subject", "(No Subject)")
    sender = _val(selected, "sender", "Unknown")
    date_str = str(_val(selected, "date", ""))[:16]
    body = _val(selected, "body", "")
    
    st.markdown(f"""
    <div class="exec-glass-card" style="margin-bottom:12px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <div style="font-size:1.15rem; font-weight:700; color:#F8FAFC;">{subject}</div>
            <div>{priority_badge(priority)}</div>
        </div>
        <div style="font-size:0.82rem; color:#94A3B8; margin-bottom:12px;">
            <strong>From:</strong> <span style="color:#CBD5E1;">{sender}</span> &nbsp;•&nbsp; <span>{date_str}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.text_area("Full Communication Text", body, height=260, disabled=True)

with c_radar:
    score = _val(selected, "priority_score", 0)
    st.markdown(f"""
    <div class="exec-glass-card" style="padding:14px 18px; margin-bottom:10px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase;">Overall Priority Score</span>
                <div style="font-size:1.9rem; font-weight:800; color:#60A5FA;">{score:.1f}<span style="font-size:1rem; color:#94A3B8;">/100</span></div>
            </div>
            <div>{priority_badge(priority)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 8-Component Radar Visualization
    try:
        ep = db.get_email_priority(_val(selected, "id"))
        if ep:
            dims = ["Urgency", "Impact", "Sender", "Deadline", "Financial", "Action", "Strategic", "Context"]
            vals = [
                getattr(ep, "urgency_score", 0) or 0,
                getattr(ep, "business_impact_score", 0) or 0,
                getattr(ep, "sender_importance_score", 0) or 0,
                getattr(ep, "deadline_score", 0) or 0,
                getattr(ep, "financial_score", 0) or 0,
                getattr(ep, "action_score", 0) or 0,
                getattr(ep, "strategic_score", 0) or 0,
                getattr(ep, "context_score", 0) or 0,
            ]
            fig = go.Figure(data=go.Scatterpolar(
                r=vals + [vals[0]],
                theta=dims + [dims[0]],
                fill='toself',
                fillcolor='rgba(59, 130, 246, 0.25)',
                line=dict(color='#60A5FA', width=2),
                marker=dict(size=5, color='#93C5FD')
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.08)", tickfont=dict(color="#64748B", size=9)),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.08)", tickfont=dict(color="#E2E8F0", size=11, family="Plus Jakarta Sans"))
                ),
                showlegend=False,
                height=260,
                margin=dict(t=20, b=20, l=40, r=40),
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

# ── Extracted Intelligence Modules ──────────────────────────
st.markdown("<hr style='border-color:rgba(255,255,255,0.06); margin:18px 0 20px 0;'>", unsafe_allow_html=True)
st.markdown("### 🧠 AI Extracted Decision Signals")

col_comm, col_risk = st.columns(2)

with col_comm:
    st.markdown("#### 📌 Promises & Commitments")
    try:
        from src.nlp.commitment_detector import CommitmentDetector
        cd = CommitmentDetector()
        comms = cd.detect_commitments(_val(selected, "body", ""))
        if comms:
            for c in comms:
                st.markdown(f"""
                <div class="exec-glass-card" style="border-left:3px solid #FBBF24 !important; padding:12px 14px; margin-bottom:8px;">
                    <div style="font-weight:700; color:#F8FAFC; font-size:0.9rem;">{c['description']}</div>
                    <div style="font-size:0.8rem; color:#94A3B8; margin-top:3px;">
                        Responsible: <strong style="color:#CBD5E1;">{c['responsible_person']}</strong> &nbsp;•&nbsp; 
                        Deadline: <span style="color:#FCA5A5;">⏰ {c.get('deadline','Unspecified')}</span> &nbsp;•&nbsp; 
                        Confidence: <strong>{c['confidence']:.0%}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No explicit commitments detected in this message.")
    except Exception as e:
        st.error(f"Commitment error: {e}")

with col_risk:
    st.markdown("#### ⚠️ Corporate Risk Signals")
    try:
        from src.nlp.risk_detector import RiskDetector
        rd = RiskDetector()
        risks = rd.detect_risks(_val(selected, "body", ""))
        if risks:
            for r in risks:
                sev = r.get("severity", "medium").lower()
                accent = "#EF4444" if sev == "critical" else "#F97316"
                st.markdown(f"""
                <div class="exec-glass-card" style="border-left:3px solid {accent} !important; padding:12px 14px; margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <div>{category_pill(r['category'])} &nbsp; {severity_badge(sev)}</div>
                    </div>
                    <div style="font-weight:600; color:#F1F5F9; font-size:0.88rem; margin-top:4px;">{r['explanation']}</div>
                    <div style="font-size:0.8rem; color:#60A5FA; margin-top:4px;">💡 <em>Action: {r.get('suggested_action','')}</em></div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No business risks identified in this message.")
    except Exception as e:
        st.error(f"Risk error: {e}")
