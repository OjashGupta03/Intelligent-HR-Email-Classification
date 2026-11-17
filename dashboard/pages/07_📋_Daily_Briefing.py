"""
Page 7: Daily Executive Briefing
"""
import os, sys, json
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

import streamlit as st
from datetime import datetime, date
from dashboard.components.shared import (
    inject_custom_theme, render_hero, get_db, get_priority_engine,
    priority_badge, category_pill, deadline_pill, no_data_message, _val
)

st.set_page_config(page_title="Daily Briefing | Executive AI", page_icon="📊", layout="wide")
inject_custom_theme()

render_hero(
    title="Daily Executive Briefing",
    subtitle="AI-synthesized morning intelligence brief summarizing critical developments, pending decisions, commitments, and threats.",
    badge_text="Module 05 · Executive Intelligence"
)

db = get_db()

# ── Control & Date Bar ─────────────────────────────────────────
st.markdown('<div class="exec-glass-card" style="margin-bottom:20px;">', unsafe_allow_html=True)
col_date, col_spacer, col_btn = st.columns([3, 4, 2])
with col_date:
    selected_date = st.date_input("Select Briefing Date", value=date.today())
with col_btn:
    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
    generate = st.button("⚡ Generate Briefing", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if generate:
    with st.spinner("Synthesizing daily executive intelligence..."):
        try:
            pe = get_priority_engine()
            from src.intelligence.intelligence_modules import CommitmentTracker, RiskMonitor, DailyBriefingGenerator
            from src.nlp.commitment_detector import CommitmentDetector
            from src.nlp.risk_detector import RiskDetector
            ct = CommitmentTracker(db, CommitmentDetector())
            rm = RiskMonitor(db, RiskDetector())
            gen = DailyBriefingGenerator(db, pe, ct, rm)
            briefing = gen.generate_briefing(datetime.combine(selected_date, datetime.min.time()))
            st.session_state["current_briefing"] = briefing
            st.success("Executive briefing generated successfully!")
        except Exception as e:
            st.error(f"Error generating briefing: {e}")

briefing = st.session_state.get("current_briefing")

if not briefing:
    # Try to load latest saved briefing
    try:
        recent = db.get_recent_briefings(1)
        if recent:
            b = recent[0]
            content_raw = getattr(b, "content", None)
            if content_raw:
                briefing = json.loads(content_raw) if isinstance(content_raw, str) else content_raw
                st.info(f"Displaying archived briefing from {getattr(b, 'date', 'recent')}")
    except Exception:
        pass

if not briefing:
    st.markdown("""
    <div class="exec-glass-card" style="text-align:center; padding:36px 20px;">
        <div style="font-size:2.2rem; margin-bottom:10px;">☕</div>
        <h3 style="color:#F8FAFC; margin-bottom:8px;">Ready for Today's Executive Briefing</h3>
        <p style="color:#94A3B8; font-size:0.92rem; max-width:550px; margin:0 auto 18px auto;">
            Click <strong>Generate Briefing</strong> above to synthesize all overnight emails, detect overdue commitments, and surface high-impact business decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Summary KPI Tiles ─────────────────────────────────────────
top_emails = briefing.get("top_emails", [])
devs = briefing.get("key_developments", [])
decisions = briefing.get("pending_decisions", [])
deadlines = briefing.get("upcoming_deadlines", [])
overdue = briefing.get("overdue_commitments", [])
risks = briefing.get("critical_risks", [])
actions = briefing.get("recommended_actions", [])

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Urgent Emails", len(top_emails))
k2.metric("Pending Decisions", len(decisions))
k3.metric("Deadlines", len(deadlines))
k4.metric("Overdue Items", len(overdue))
k5.metric("Active Risks", len(risks))

st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

# ── Briefing Date Banner ──────────────────────────────────────
b_date = briefing.get("date", str(selected_date))
st.markdown(f"""
<div style="background:linear-gradient(135deg, rgba(30, 58, 95, 0.45) 0%, rgba(17, 24, 39, 0.75) 100%);
            border:1px solid rgba(59, 130, 246, 0.3); border-radius:12px; padding:16px 22px; margin-bottom:24px;
            display:flex; justify-content:space-between; align-items:center;">
    <div>
        <span style="font-size:11px; font-weight:700; color:#60A5FA; letter-spacing:0.08em; text-transform:uppercase;">SYNTHESIZED REPORT</span>
        <h3 style="color:#F8FAFC; margin:4px 0 0 0; font-size:1.3rem;">Executive Dossier for {b_date}</h3>
    </div>
    <div style="background:rgba(59, 130, 246, 0.15); border:1px solid rgba(59, 130, 246, 0.35); padding:6px 14px; border-radius:20px; color:#93C5FD; font-size:12px; font-weight:700;">
        100% LOCAL SYNTHESIS
    </div>
</div>
""", unsafe_allow_html=True)

# ── 1. Top Emails Requiring Attention ────────────────────────
st.markdown("### 📧 1. Priority Emails Requiring Immediate Attention")
if not top_emails:
    st.markdown('<div class="exec-glass-card" style="color:#94A3B8;">✅ No critical or urgent emails requiring immediate action today.</div>', unsafe_allow_html=True)
else:
    for e in top_emails:
        priority = (_val(e, "priority") or "medium").lower()
        score = _val(e, "score", 0)
        subject = _val(e, "subject", "(no subject)")
        sender = _val(e, "sender", "Unknown")
        snippet = _val(e, "snippet") or _val(e, "summary") or _val(e, "body") or ""
        
        st.markdown(f"""
        <div class="exec-glass-card" style="padding:14px 18px; margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <div style="display:flex; align-items:center; gap:10px;">
                    {priority_badge(priority)}
                    <span style="color:#F8FAFC; font-weight:700; font-size:0.98rem;">{subject}</span>
                </div>
                <div style="background:rgba(99, 102, 241, 0.15); color:#818CF8; border:1px solid rgba(99, 102, 241, 0.3); font-weight:700; font-size:11px; padding:3px 9px; border-radius:8px;">
                    SCORE {score:.0f}
                </div>
            </div>
            <div style="color:#94A3B8; font-size:0.83rem;">
                From: <span style="color:#CBD5E1; font-weight:600;">{sender}</span>
            </div>
            {f'<div style="color:#94A3B8; font-size:0.83rem; margin-top:6px; font-style:italic;">"{snippet[:160]}..."</div>' if snippet else ''}
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

# ── 2. Key Business Developments ─────────────────────────────
st.markdown("### ⚡ 2. Strategic Business Developments")
if not devs:
    st.markdown('<div class="exec-glass-card" style="color:#94A3B8;">No major developments noted in the current window.</div>', unsafe_allow_html=True)
else:
    for d in devs:
        subj = d.get('subject', '')
        summ = d.get('summary', '')
        st.markdown(f"""
        <div class="exec-glass-card" style="border-left:3px solid #6366F1; padding:12px 18px; margin-bottom:8px;">
            <div style="color:#F1F5F9; font-weight:700; font-size:0.94rem; margin-bottom:4px;">{subj}</div>
            <div style="color:#94A3B8; font-size:0.86rem; line-height:1.4;">{summ}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

# ── 3. Pending Decisions ─────────────────────────────────────
st.markdown("### ⚖️ 3. Decisions Awaiting Executive Action")
if not decisions:
    st.markdown('<div class="exec-glass-card" style="color:#94A3B8;">✅ No critical decisions currently awaiting your approval.</div>', unsafe_allow_html=True)
else:
    for d in decisions:
        subj = d.get('subject', '')
        sender = d.get('sender', '')
        st.markdown(f"""
        <div class="exec-glass-card" style="border-left:3px solid #F59E0B; padding:12px 18px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="color:#F1F5F9; font-weight:700; font-size:0.94rem;">⏳ {subj}</div>
                <div style="color:#94A3B8; font-size:0.82rem; margin-top:2px;">Requested by: <span style="color:#CBD5E1;">{sender}</span></div>
            </div>
            <span style="background:rgba(245, 158, 11, 0.12); border:1px solid rgba(245, 158, 11, 0.3); color:#FBBF24; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:700;">
                DECISION PENDING
            </span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

# ── 4 & 5. Deadlines & Overdue ───────────────────────────────
c_col1, c_col2 = st.columns(2)
with c_col1:
    st.markdown("### ⏰ 4. Approaching Deadlines")
    if not deadlines:
        st.markdown('<div class="exec-glass-card" style="color:#94A3B8;">No approaching deadlines detected.</div>', unsafe_allow_html=True)
    else:
        for dl in deadlines:
            st.markdown(f"""
            <div class="exec-glass-card" style="border-left:3px solid #3B82F6; padding:12px 16px; margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="color:#93C5FD; font-size:11px; font-weight:700;">DUE {dl.get('deadline','')}</span>
                </div>
                <div style="color:#F1F5F9; font-size:0.88rem; font-weight:600;">{dl.get('description','')[:75]}</div>
            </div>
            """, unsafe_allow_html=True)

with c_col2:
    st.markdown("### 📌 5. Overdue Commitments")
    if not overdue:
        st.markdown('<div class="exec-glass-card" style="color:#94A3B8;">✅ Zero overdue commitments recorded.</div>', unsafe_allow_html=True)
    else:
        for ov in overdue:
            st.markdown(f"""
            <div class="exec-glass-card" style="border-left:3px solid #EF4444; padding:12px 16px; margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="color:#FCA5A5; font-size:11px; font-weight:700;">OVERDUE</span>
                    <span style="color:#94A3B8; font-size:11px;">Owner: {ov.get('responsible_person','Self')}</span>
                </div>
                <div style="color:#F1F5F9; font-size:0.88rem; font-weight:600;">{ov.get('description','')}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

# ── 6. Critical Risks ─────────────────────────────────────────
st.markdown("### ⚠️ 6. Critical Business Risks")
if not risks:
    st.markdown('<div class="exec-glass-card" style="color:#94A3B8;">✅ No high-severity business risks flagged.</div>', unsafe_allow_html=True)
else:
    for r in risks:
        cat = r.get('category','').upper()
        exp = r.get('explanation','')
        st.markdown(f"""
        <div class="exec-glass-card" style="border-left:3px solid #EF4444; padding:14px 18px; margin-bottom:8px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="background:rgba(239, 68, 68, 0.15); color:#FCA5A5; border:1px solid rgba(239, 68, 68, 0.3); padding:2px 8px; border-radius:6px; font-size:11px; font-weight:700;">
                    {cat} RISK
                </span>
            </div>
            <div style="color:#F8FAFC; font-size:0.9rem; font-weight:500; margin-top:6px;">{exp}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

# ── 7. Recommended Actions ───────────────────────────────────
st.markdown("### 🎯 7. Recommended Action Plan for Today")
if not actions:
    st.markdown('<div class="exec-glass-card" style="color:#94A3B8;">Standard inbox review recommended.</div>', unsafe_allow_html=True)
else:
    for i, a in enumerate(actions, 1):
        st.markdown(f"""
        <div class="exec-glass-card" style="border-left:3px solid #10B981; padding:12px 18px; margin-bottom:8px; display:flex; align-items:center; gap:12px;">
            <span style="background:rgba(16, 185, 129, 0.15); color:#34D399; width:26px; height:26px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; font-weight:700; font-size:12px; flex-shrink:0;">
                {i}
            </span>
            <span style="color:#F1F5F9; font-size:0.92rem; font-weight:600;">{a}</span>
        </div>
        """, unsafe_allow_html=True)
