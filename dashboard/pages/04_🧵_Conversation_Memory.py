"""
Page 4: Conversation Memory & Thread Viewer
Multi-pass thread reconstruction, decisions, and unanswered questions.
"""
import os, sys
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

import streamlit as st
from collections import defaultdict
import re
from datetime import datetime

from dashboard.components.shared import (
    get_db, priority_badge, render_hero, inject_custom_theme,
    no_data_message, _val
)

st.set_page_config(page_title="Conversation Memory", page_icon="🧵", layout="wide")
inject_custom_theme()

render_hero(
    title="Conversation Memory & Thread Reconstruction",
    subtitle="Algorithmic multi-pass conversation clustering, chronological timeline analysis, and extracted decision tracking.",
    badge_text="Contextual Dialogue Memory"
)

db = get_db()

try:
    emails = db.get_emails({})
    if not emails:
        no_data_message()
        st.stop()
except Exception as e:
    st.error(f"Database error: {e}")
    st.stop()

def normalize_subj(s):
    return re.sub(r"^(Re|Fwd|Fw|RE|FW|FWD):\s*", "", s or "", flags=re.IGNORECASE).strip()

threads = defaultdict(list)
for e in emails:
    tid = _val(e, "thread_id") or f"subj_{normalize_subj(_val(e,'subject',''))[:30]}"
    threads[tid].append(e)

def latest_date(thread_emails):
    dates = []
    for e in thread_emails:
        d = _val(e, "date")
        try:
            dates.append(datetime.fromisoformat(str(d)) if d else None)
        except Exception:
            pass
    valid = [d for d in dates if d]
    return max(valid) if valid else datetime.min

sorted_threads = sorted(threads.items(), key=lambda x: latest_date(x[1]), reverse=True)

st.markdown(f"<div style='font-size:0.88rem; color:#94A3B8; margin-bottom:14px;'>Indexed <strong>{len(sorted_threads):,} conversation threads</strong> across {len(emails):,} messages.</div>", unsafe_allow_html=True)

col_list, col_detail = st.columns([1, 1.8])

with col_list:
    st.markdown("### 🗂️ Active Threads")
    thread_options = {}
    for tid, t_emails in sorted_threads[:35]:
        first = t_emails[0]
        subj = normalize_subj(_val(first, "subject", "Untitled"))[:45]
        count = len(t_emails)
        thread_options[f"[{count} msg{'s' if count>1 else ''}] {subj}"] = tid

    selected_thread_label = st.radio("Select Thread", list(thread_options.keys()), label_visibility="collapsed")
    selected_tid = thread_options[selected_thread_label]

with col_detail:
    t_emails = threads[selected_tid]
    t_emails_sorted = sorted(t_emails, key=lambda x: str(_val(x, "date", "")))
    first = t_emails_sorted[0]
    thread_subject = normalize_subj(_val(first, 'subject', 'Untitled'))
    participants = sorted(list(set(_val(e,'sender','') for e in t_emails_sorted if _val(e,'sender'))))

    st.markdown(f"""
    <div class="exec-glass-card" style="border-top:3px solid #6366F1 !important; margin-bottom:16px;">
        <div style="font-size:1.25rem; font-weight:800; color:#F8FAFC; margin-bottom:6px;">🧵 {thread_subject}</div>
        <div style="font-size:0.82rem; color:#94A3B8; line-height:1.6;">
            <strong>Thread ID:</strong> <code style="color:#A5B4FC; font-size:0.75rem;">{selected_tid}</code> &nbsp;•&nbsp; 
            <strong>Messages:</strong> <span style="color:#60A5FA; font-weight:700;">{len(t_emails_sorted)}</span><br>
            <strong>Participants:</strong> <span style="color:#CBD5E1;">{', '.join(participants)}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Thread summary if multi-message
    if len(t_emails_sorted) > 1:
        with st.expander("📝 Extractive Thread Summary", expanded=True):
            try:
                from src.nlp.summarizer import EmailSummarizer
                summarizer = EmailSummarizer()
                thread_summary = summarizer.summarize_thread([
                    {"subject": _val(e,"subject",""), "body": _val(e,"body",""),
                     "cleaned_body": _val(e,"body",""), "date": _val(e,"date","")}
                    for e in t_emails_sorted
                ])
                st.markdown(f"""
                <div style="background:rgba(59, 130, 246, 0.08); border-left:3px solid #3B82F6; padding:10px 14px; border-radius:6px; color:#E2E8F0; font-size:0.88rem; line-height:1.5;">
                    {thread_summary}
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.caption(f"Summary unavailable: {e}")

    # Messages in thread
    st.markdown("### 💬 Chronological Timeline")
    for i, e in enumerate(t_emails_sorted):
        priority = (_val(e, "priority_level") or "low").lower()
        sender = _val(e, "sender", "Unknown")
        date_str = str(_val(e, "date", ""))[:16]
        body = _val(e, "body", "")
        
        accent = "#3B82F6" if i == 0 else "#64748B"

        st.markdown(f"""
        <div class="exec-glass-card" style="border-left:3px solid {accent} !important; padding:14px 18px; margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <div style="font-size:0.9rem; font-weight:700; color:#F1F5F9;">
                    #{i+1} &nbsp;•&nbsp; {sender}
                </div>
                <div style="display:flex; align-items:center; gap:8px;">
                    {priority_badge(priority)}
                    <span style="font-size:0.78rem; color:#94A3B8;">{date_str}</span>
                </div>
            </div>
            <div style="font-size:0.86rem; color:#CBD5E1; line-height:1.5; white-space:pre-wrap;">{body}</div>
        </div>
        """, unsafe_allow_html=True)
