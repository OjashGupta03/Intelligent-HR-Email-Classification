"""
Page 8: What Did I Miss? — Flagship Feature
"""
import os, sys
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

import streamlit as st
from datetime import datetime, date, timedelta
from dashboard.components.shared import (
    inject_custom_theme, render_hero, get_db, get_priority_engine,
    priority_badge, category_pill, deadline_pill, no_data_message, _val
)

st.set_page_config(page_title="What Did I Miss? | Executive AI", page_icon="🕵️", layout="wide")
inject_custom_theme()

render_hero(
    title="What Did I Miss?",
    subtitle="Flagship absence intelligence engine. Returning from leave or travel? Synthesizes key decisions, threats, and commitments across multi-day inbox surges.",
    badge_text="Flagship Feature · Module 06"
)

db = get_db()

# ── Absence Period Selector ──────────────────────────────────
st.markdown('<div class="exec-glass-card" style="margin-bottom:20px;">', unsafe_allow_html=True)
st.markdown('<div style="font-weight:700; font-size:0.95rem; color:#F8FAFC; margin-bottom:12px;">📅 Define Absence Time Window</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns([3, 3, 2])
with c1:
    start_date = st.date_input("Absence Departure Date", value=date.today() - timedelta(days=5))
with c2:
    end_date = st.date_input("Return Date", value=date.today())
with c3:
    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
    generate = st.button("🔍 Generate Intelligence Report", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if start_date > end_date:
    st.error("Error: Departure date must be prior to Return date.")
    st.stop()

if generate:
    with st.spinner(f"Analyzing inbox intelligence from {start_date} to {end_date}..."):
        try:
            pe = get_priority_engine()
            from src.intelligence.intelligence_modules import (
                CommitmentTracker, RiskMonitor, ConversationMemory, WhatDidIMiss
            )
            from src.nlp.commitment_detector import CommitmentDetector
            from src.nlp.risk_detector import RiskDetector
            ct = CommitmentTracker(db, CommitmentDetector())
            rm = RiskMonitor(db, RiskDetector())
            cm = ConversationMemory(db)
            wdim = WhatDidIMiss(db, pe, ct, rm, cm)
            report = wdim.generate_report(
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()),
            )
            st.session_state["wdim_report"] = report
        except Exception as e:
            st.error(f"Error generating absence report: {e}")

report = st.session_state.get("wdim_report")
if not report:
    st.markdown("""
    <div class="exec-glass-card" style="text-align:center; padding:40px 20px;">
        <div style="font-size:2.5rem; margin-bottom:12px;">🛫</div>
        <h3 style="color:#F8FAFC; margin-bottom:8px;">Zero Context Switching Upon Return</h3>
        <p style="color:#94A3B8; font-size:0.92rem; max-width:600px; margin:0 auto 20px auto;">
            Select your absence window above and generate an executive debriefing. The AI scans threads, reconstructs chronological developments, identifies promises made by your team, and flags emerging operational or customer risks.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Executive Summary Banner ─────────────────────────────────
period = report.get("period", {})
total_emails = period.get("total_emails", 0)
total_threads = period.get("total_threads", 0)
summary_text = report.get("executive_summary", "")

st.markdown(f"""
<div style="background: radial-gradient(circle at 10% 20%, rgba(30, 58, 95, 0.7) 0%, rgba(17, 24, 39, 0.9) 100%);
            border: 1px solid rgba(59, 130, 246, 0.35); border-radius: 14px; padding: 22px 26px; margin-bottom: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <span style="background:rgba(59, 130, 246, 0.2); color:#60A5FA; border:1px solid rgba(59, 130, 246, 0.4); font-size:11px; font-weight:700; padding:3px 10px; border-radius:20px; letter-spacing:0.06em; text-transform:uppercase;">
                EXECUTIVE DEBRIEF
            </span>
            <span style="color:#94A3B8; font-size:0.85rem;">Window: {period.get('start','')[:10]} → {period.get('end','')[:10]}</span>
        </div>
        <span style="color:#34D399; font-size:12px; font-weight:700;">● SYSTEM OPERATIONAL</span>
    </div>
    <div style="color:#F1F5F9; font-size:1.02rem; line-height:1.6; font-weight:500;">
        {summary_text or f"Analyzed {total_emails} incoming emails across {total_threads} reconstructed threads during your absence window."}
    </div>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Emails Received", total_emails)
m2.metric("Active Threads", total_threads)
m3.metric("Critical Developments", len(report.get("critical_developments", [])))
m4.metric("Emerging Risks", len(report.get("risks_emerged", [])))

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# ── Structured Tabs ──────────────────────────────────────────
def render_email_card(e):
    priority = (_val(e, "priority_level") or _val(e, "priority") or "medium").lower()
    subject = _val(e, "subject", "(no subject)")
    sender = _val(e, "sender", "Unknown")
    date_str = str(_val(e, "date", ""))[:16]
    score = _val(e, "priority_score") or _val(e, "score") or 0
    preview = (_val(e, "summary") or _val(e, "body") or "")[:200]
    
    st.markdown(f"""
    <div class="exec-glass-card" style="padding:14px 20px; margin-bottom:10px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <div style="display:flex; align-items:center; gap:10px;">
                {priority_badge(priority)}
                <span style="color:#F8FAFC; font-weight:700; font-size:0.96rem;">{subject}</span>
            </div>
            <span style="background:rgba(99, 102, 241, 0.15); color:#818CF8; border:1px solid rgba(99, 102, 241, 0.3); font-weight:700; font-size:11px; padding:3px 9px; border-radius:8px;">
                SCORE {score:.0f}
            </span>
        </div>
        <div style="color:#94A3B8; font-size:0.83rem; margin-bottom:6px;">
            From: <span style="color:#CBD5E1; font-weight:600;">{sender}</span> &bull; Received: <span>{date_str}</span>
        </div>
        {f'<div style="color:#94A3B8; font-size:0.85rem; line-height:1.45; border-left:2px solid rgba(255,255,255,0.1); padding-left:10px; margin-top:8px;">{preview}...</div>' if preview else ''}
    </div>
    """, unsafe_allow_html=True)

tabs = st.tabs([
    "🔴 Critical Developments",
    "📋 Needs Attention",
    "⚖️ Decisions",
    "📌 Commitments",
    "⏰ Missed Deadlines",
    "😤 Customer Issues",
    "⚠️ Risks",
    "🧵 Thread Updates",
    "🎯 Return Action Plan",
])

with tabs[0]:
    crit = report.get("critical_developments", [])
    if not crit:
        st.markdown('<div class="exec-glass-card" style="color:#94A3B8;">✅ No critical escalations occurred during this timeframe.</div>', unsafe_allow_html=True)
    else:
        for e in crit[:10]:
            render_email_card(e)

with tabs[1]:
    attn = report.get("attention_needed", [])
    if not attn:
        st.markdown('<div class="exec-glass-card" style="color:#94A3B8;">✅ No pending items requiring immediate executive attention.</div>', unsafe_allow_html=True)
    else:
        for e in attn[:10]:
            render_email_card(e)

with tabs[2]:
    decs = report.get("decisions_pending", [])
    if not decs:
        st.markdown('<div class="exec-glass-card" style="color:#94A3B8;">✅ No decisions pending executive signature or approval.</div>', unsafe_allow_html=True)
    else:
        for e in decs[:10]:
            render_email_card(e)

with tabs[3]:
    commitments = report.get("commitments_made", [])
    if not commitments:
        st.markdown('<div class="exec-glass-card" style="color:#94A3B8;">No new team commitments recorded during this period.</div>', unsafe_allow_html=True)
    else:
        for c in commitments[:12]:
            desc = (_val(c, 'description') or '')
            resp = _val(c, 'responsible_person', 'Team Member')
            dl = _val(c, 'deadline', 'Not specified')
            st.markdown(f"""
            <div class="exec-glass-card" style="border-left:3px solid #3B82F6; padding:12px 18px; margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="color:#93C5FD; font-size:11px; font-weight:700;">OWNER: {resp}</span>
                    <span style="color:#FCA5A5; font-size:11px; font-weight:600;">DUE: {dl}</span>
                </div>
                <div style="color:#F1F5F9; font-size:0.9rem; font-weight:600;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

with tabs[4]:
    overdue = report.get("missed_deadlines", [])
    if not overdue:
        st.markdown('<div class="exec-glass-card" style="color:#10B981;">✅ Excellent: Zero missed deadlines during your absence.</div>', unsafe_allow_html=True)
    else:
        for c in overdue[:8]:
            st.markdown(f"""
            <div class="exec-glass-card" style="border-left:3px solid #EF4444; padding:12px 18px; margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="color:#FCA5A5; font-size:11px; font-weight:700;">MISSED DEADLINE</span>
                    <span style="color:#94A3B8; font-size:11px;">Responsible: {_val(c,'responsible_person','Self')}</span>
                </div>
                <div style="color:#F1F5F9; font-size:0.9rem; font-weight:600;">{_val(c,'description','')}</div>
                <div style="color:#EF4444; font-size:0.82rem; margin-top:2px;">Target date was: {_val(c,'deadline','')}</div>
            </div>
            """, unsafe_allow_html=True)

with tabs[5]:
    cust = report.get("customer_issues", [])
    if not cust:
        st.markdown('<div class="exec-glass-card" style="color:#94A3B8;">✅ Zero client escalations or complaints detected.</div>', unsafe_allow_html=True)
    else:
        for e in cust[:8]:
            render_email_card(e)

with tabs[6]:
    risks = report.get("risks_emerged", [])
    if not risks:
        st.markdown('<div class="exec-glass-card" style="color:#94A3B8;">✅ No operational or financial risks emerged.</div>', unsafe_allow_html=True)
    else:
        for r in risks[:8]:
            cat = r.get('category','').upper()
            exp = r.get('explanation','')
            act = r.get('suggested_action','')
            sev = r.get('severity','medium')
            st.markdown(f"""
            <div class="exec-glass-card" style="border-left:3px solid #EF4444; padding:14px 18px; margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <span style="background:rgba(239, 68, 68, 0.15); color:#FCA5A5; border:1px solid rgba(239, 68, 68, 0.3); padding:2px 8px; border-radius:6px; font-size:11px; font-weight:700;">
                        {cat} RISK &bull; {sev.upper()}
                    </span>
                </div>
                <div style="color:#F8FAFC; font-size:0.9rem; font-weight:600; margin-bottom:4px;">{exp}</div>
                {f'<div style="color:#93C5FD; font-size:0.83rem; margin-top:6px;">👉 Recommended Action: {act}</div>' if act else ''}
            </div>
            """, unsafe_allow_html=True)

with tabs[7]:
    thread_updates = report.get("thread_updates", [])
    if not thread_updates:
        st.markdown('<div class="exec-glass-card" style="color:#94A3B8;">No significant multi-party thread updates.</div>', unsafe_allow_html=True)
    else:
        for t in thread_updates[:8]:
            subj = t.get('subject','')
            count = t.get('email_count', 0)
            sender = t.get('latest_sender','')
            summ = t.get('summary','')
            st.markdown(f"""
            <div class="exec-glass-card" style="padding:14px 18px; margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="color:#F8FAFC; font-weight:700; font-size:0.94rem;">🧵 {subj}</span>
                    <span style="background:rgba(99, 102, 241, 0.15); color:#818CF8; border:1px solid rgba(99, 102, 241, 0.3); padding:2px 8px; border-radius:10px; font-size:11px; font-weight:700;">
                        {count} EMAILS
                    </span>
                </div>
                <div style="color:#94A3B8; font-size:0.83rem; margin-bottom:6px;">Latest update from: <span style="color:#CBD5E1;">{sender}</span></div>
                <div style="color:#94A3B8; font-size:0.86rem; line-height:1.45;">{summ[:200]}</div>
            </div>
            """, unsafe_allow_html=True)

with tabs[8]:
    actions = report.get("recommended_actions", [])
    if not actions:
        st.markdown('<div class="exec-glass-card" style="color:#94A3B8;">Standard inbox review recommended.</div>', unsafe_allow_html=True)
    else:
        for i, a in enumerate(actions, 1):
            st.markdown(f"""
            <div class="exec-glass-card" style="border-left:3px solid #10B981; padding:14px 18px; margin-bottom:8px; display:flex; align-items:center; gap:14px;">
                <span style="background:rgba(16, 185, 129, 0.15); color:#34D399; width:28px; height:28px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; font-weight:700; font-size:12px; flex-shrink:0;">
                    {i}
                </span>
                <span style="color:#F1F5F9; font-size:0.93rem; font-weight:600;">{a}</span>
            </div>
            """, unsafe_allow_html=True)
