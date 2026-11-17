"""
Executive AI Chief of Staff — Main Application
High-End Executive Command Center Home
"""
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import streamlit as st
from dashboard.components.shared import (
    inject_custom_theme, get_db, render_hero, priority_badge, _val
)

# ── Page config must be first ──────────────────────────────
st.set_page_config(
    page_title="Executive AI Chief of Staff",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_theme()

db = get_db()

# ── Sidebar Styling & Telemetry ─────────────────────────────
with st.sidebar:
    try:
        stats = db.get_stats()
        email_count = stats.get("email_count", 0)
        crit_count = stats.get("critical_count", 0)
        risk_count = stats.get("risk_count", 0)
        comm_count = stats.get("commitment_count", 0)
        
        status_dot = "#10B981" if email_count > 0 else "#EF4444"
        status_label = "ONLINE (LOCAL)" if email_count > 0 else "NO DATA"
        
        st.markdown(f"""
        <div style="background:rgba(17,24,39,0.7); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:12px 14px; margin-bottom:14px; box-shadow:0 4px 14px rgba(0,0,0,0.3);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <span style="font-size:0.75rem; font-weight:800; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; font-family:'JetBrains Mono', monospace;">SYSTEM TELEMETRY</span>
                <span style="display:inline-flex; align-items:center; gap:4px; font-size:0.72rem; font-weight:700; color:{status_dot};">
                    <span style="width:6px; height:6px; border-radius:50%; background:{status_dot}; box-shadow:0 0 6px {status_dot};"></span>
                    {status_label}
                </span>
            </div>
            <div style="font-size:0.83rem; color:#CBD5E1; line-height:1.75;">
                • Indexed Emails: <strong style="color:#F8FAFC;">{email_count:,}</strong><br>
                • Critical Items: <strong style="color:#EF4444;">{crit_count:,}</strong><br>
                • Detected Risks: <strong style="color:#F97316;">{risk_count:,}</strong><br>
                • Active Commitments: <strong style="color:#FBBF24;">{comm_count:,}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"DB Error: {e}")

    st.markdown("""
    <div style="background:rgba(30,41,59,0.4); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:12px 14px; margin-bottom:14px;">
        <div style="font-size:0.72rem; font-weight:800; color:#64748B; text-transform:uppercase; margin-bottom:6px; letter-spacing:0.06em; font-family:'JetBrains Mono', monospace;">LOCAL HARDWARE</div>
        <div style="font-size:0.82rem; color:#94A3B8; line-height:1.6;">
            ⚡ <strong>NVIDIA RTX 2050</strong> (CUDA)<br>
            🔒 <strong>Zero External APIs</strong><br>
            💾 <strong>Local SQLite (WAL)</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align:center; padding:8px 0; color:#64748B; font-size:0.72rem;">
        Executive AI Chief of Staff &bull; Enterprise 2026<br>
        <span style="color:#3B82F6;">100% Air-Gapped Privacy</span>
    </div>
    """, unsafe_allow_html=True)

# ── Main Hero Section ───────────────────────────────────────
render_hero(
    title="Executive AI Chief of Staff",
    subtitle="Autonomous decision intelligence, communication triage, hidden commitment tracking, and enterprise risk radar — 100% on-premises.",
    badge_text="Autonomous Decision Intelligence"
)

# ── Top Real-Time KPI Row ────────────────────────────────────
try:
    stats = db.get_stats()
except Exception:
    stats = {}

m1, m2, m3, m4 = st.columns(4)
m1.metric("📬 Total Communications", f"{stats.get('email_count', 0):,}", "Indexed Locally")
m2.metric("🔴 Critical Decisions", stats.get("critical_count", 0), "Requires Sign-off")
m3.metric("⚠️ Business Risks", stats.get("risk_count", 0), "9 Categories")
m4.metric("📌 Active Commitments", stats.get("commitment_count", 0), "Promises Tracked")

st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)

# ── Featured Flagship Workspaces ─────────────────────────────
st.markdown("### ⚡ Primary Executive Workspaces")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="exec-glass-card" style="border-top:3px solid #3B82F6 !important; min-height:190px;">
        <div style="font-size:1.6rem; margin-bottom:8px;">📥</div>
        <h4 style="margin:0 0 6px 0; color:#F8FAFC; font-weight:700;">Priority Action Inbox</h4>
        <p style="font-size:0.86rem; color:#94A3B8; margin-bottom:14px; line-height:1.5;">
            Triage communications ranked by an 8-component business priority score and independent 2D Urgency × Importance matrix.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/02_priority_inbox.py", label="Open Priority Inbox →", icon="📥")

with c2:
    st.markdown("""
    <div class="exec-glass-card" style="border-top:3px solid #8B5CF6 !important; min-height:190px;">
        <div style="font-size:1.6rem; margin-bottom:8px;">🕵️</div>
        <h4 style="margin:0 0 6px 0; color:#F8FAFC; font-weight:700;">What Did I Miss?</h4>
        <p style="font-size:0.86rem; color:#94A3B8; margin-bottom:14px; line-height:1.5;">
            Returning from leave or meetings? Select your absence period for a synthesized intelligence report of all developments.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/08_what_did_i_miss.py", label="Launch Catch-Up Report →", icon="🕵️")

with c3:
    st.markdown("""
    <div class="exec-glass-card" style="border-top:3px solid #10B981 !important; min-height:190px;">
        <div style="font-size:1.6rem; margin-bottom:8px;">📊</div>
        <h4 style="margin:0 0 6px 0; color:#F8FAFC; font-weight:700;">Daily Executive Briefing</h4>
        <p style="font-size:0.86rem; color:#94A3B8; margin-bottom:14px; line-height:1.5;">
            Read your AI-generated morning digest in under 3 minutes: top approvals, due-today deliverables, and new exposures.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/07_daily_briefing.py", label="View Today's Briefing →", icon="📊")

st.markdown("<hr style='border-color:rgba(255,255,255,0.06); margin:30px 0 24px 0;'>", unsafe_allow_html=True)

# ── Full 10-Page System Navigation Directory ────────────────
st.markdown("### 🧭 Complete Intelligence Suite Directory")
st.caption("Access all specialized analysis modules, relationship graphs, and machine learning telemetry.")

col_a, col_b = st.columns(2)

suite_pages = [
    ("pages/01_executive_overview.py", "📋 Executive Overview", "Mission control: KPI distribution, Eisenhower 2D matrix, and urgent queue."),
    ("pages/02_priority_inbox.py", "📥 Priority Inbox", "Action-focused inbox with 8-component explainability and filters."),
    ("pages/03_email_intelligence.py", "🔬 Email Intelligence", "Deep-dive analyzer: radar visual of scoring signals and evidence."),
    ("pages/04_conversation_memory.py", "🧵 Conversation Memory", "Multi-pass thread reconstruction, decisions, and unanswered questions."),
    ("pages/05_commitment_tracker.py", "📌 Commitment Tracker", "35+ pattern detector for promises, agreements, and target deadlines."),
    ("pages/06_risk_monitor.py", "⚠️ Risk Monitor", "Enterprise risk radar covering financial, legal, security, and churn."),
    ("pages/07_daily_briefing.py", "📊 Daily Executive Briefing", "Factual extractive morning digest with recommended action plan."),
    ("pages/08_what_did_i_miss.py", "🕵️ What Did I Miss?", "Flagship absence catch-up report with time-window filtering."),
    ("pages/09_relationship_intelligence.py", "🤝 Relationship Intelligence", "VIP contact scoring, interaction cadence, and NetworkX graph."),
    ("pages/10_model_performance.py", "📈 Model Performance", "Empirical academic baselines: LR, SVM, XGBoost, NDCG@5, and F1."),
]

for idx, (path, title, desc) in enumerate(suite_pages):
    target_col = col_a if idx % 2 == 0 else col_b
    with target_col:
        st.markdown(f"""
        <div class="exec-glass-card" style="padding:14px 18px; margin-bottom:10px;">
            <div style="font-weight:700; font-size:0.98rem; color:#F1F5F9; margin-bottom:4px;">{title}</div>
            <div style="font-size:0.82rem; color:#94A3B8; margin-bottom:8px; line-height:1.4;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link(path, label=f"Navigate to {title} →")
        st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)

st.markdown("<hr style='border-color:rgba(255,255,255,0.06); margin:26px 0 20px 0;'>", unsafe_allow_html=True)

# ── Standalone CLI Demo Helper ──────────────────────────────
with st.expander("💻 Standalone Terminal Demonstration Mode", expanded=False):
    st.markdown("""
    Want to run a complete console-based demonstration of all 10 modules without using the browser?
    A self-contained script containing 6 sample corporate scenarios is ready:
    """)
    st.code("python demo_standalone.py", language="powershell")
    st.caption("Demonstrates priority scores, 2D matrix, commitment detection, 9 risk categories, conversation memory, and ML classification in terminal output.")

if __name__ == "__main__":
    pass
