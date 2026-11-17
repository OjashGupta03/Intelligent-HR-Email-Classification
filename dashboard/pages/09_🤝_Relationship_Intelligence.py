"""
Page 9: Relationship Intelligence
"""
import os, sys
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dashboard.components.shared import (
    inject_custom_theme, render_hero, style_plotly, get_db, no_data_message, _val
)

st.set_page_config(page_title="Relationship Intelligence | Executive AI", page_icon="🤝", layout="wide")
inject_custom_theme()

render_hero(
    title="Relationship Intelligence",
    subtitle="Communication graph analysis, VIP stakeholder mapping, interaction density, and executive network topology.",
    badge_text="Module 07 · Network Science"
)

db = get_db()

try:
    contacts = db.get_top_contacts(limit=50)
    if not contacts:
        no_data_message("contacts")
        st.stop()
except Exception as e:
    st.error(f"Database error: {e}")
    st.stop()

# ── Metric Tiles ─────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Tracked Contacts", len(contacts))
vip_count = sum(1 for c in contacts if getattr(c, "is_vip", False))
m2.metric("VIP Stakeholders", vip_count)
total_emails = sum(getattr(c, "total_emails_sent", 0) + getattr(c, "total_emails_received", 0) for c in contacts)
m3.metric("Total Exchanges", total_emails)
avg_imp = sum(getattr(c, "importance_score", 50) for c in contacts) / max(len(contacts), 1)
m4.metric("Avg Importance Index", f"{avg_imp:.1f}/100")

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# ── Top Communicators Table & Histogram ──────────────────────
col_table, col_chart = st.columns([13, 7])

with col_table:
    st.markdown('<div class="exec-glass-card" style="margin-bottom:0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-weight:700; font-size:1.05rem; color:#F8FAFC; margin-bottom:14px;">👥 High-Value Communicators</div>', unsafe_allow_html=True)
    df = pd.DataFrame([{
        "VIP": "⭐" if getattr(c, "is_vip", False) else "—",
        "Name": getattr(c, "name", "") or getattr(c, "email_address", ""),
        "Email": getattr(c, "email_address", ""),
        "Organization": getattr(c, "organization", "") or "Internal",
        "Sent": getattr(c, "total_emails_sent", 0),
        "Recv": getattr(c, "total_emails_received", 0),
        "Importance": f"{getattr(c, 'importance_score', 50):.0f}",
    } for c in contacts[:20]])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_chart:
    st.markdown('<div class="exec-glass-card" style="margin-bottom:0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-weight:700; font-size:1.05rem; color:#F8FAFC; margin-bottom:14px;">📊 Stakeholder Score Spread</div>', unsafe_allow_html=True)
    scores = [getattr(c, "importance_score", 50) for c in contacts]
    fig = px.histogram(
        x=scores,
        nbins=12,
        color_discrete_sequence=["#3B82F6"],
        labels={"x": "Importance Score (0–100)", "y": "Contacts Count"}
    )
    fig.update_traces(marker_line_width=1, marker_line_color="#1E3A8A")
    fig = style_plotly(fig, height=270)
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

# ── Network Graph ────────────────────────────────────────────
st.markdown('<div class="exec-glass-card">', unsafe_allow_html=True)
st.markdown('<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">'
            '<span style="font-weight:700; font-size:1.05rem; color:#F8FAFC;">🕸️ Executive Interaction Topology</span>'
            '<span style="font-size:11px; color:#94A3B8;">Built with NetworkX & Plotly Engine</span></div>', unsafe_allow_html=True)

try:
    import networkx as nx
    G = nx.Graph()

    # Executive center node
    G.add_node("Executive (You)", size=32, color="#6366F1", role="Self")

    # Add top contacts as nodes
    for c in contacts[:16]:
        name = getattr(c, "name", None) or getattr(c, "email_address", "unknown")
        total = getattr(c, "total_emails_sent", 0) + getattr(c, "total_emails_received", 0)
        is_vip = getattr(c, "is_vip", False)
        short_name = name.split("@")[0].replace(".", " ").title()[:14]
        G.add_node(short_name, size=22 if is_vip else 14, color="#F59E0B" if is_vip else "#38BDF8", role="VIP" if is_vip else "Stakeholder")
        G.add_edge("Executive (You)", short_name, weight=max(total, 1))

    pos = nx.spring_layout(G, seed=42, k=0.45)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.2, color="rgba(255, 255, 255, 0.12)"),
        hoverinfo="none",
        mode="lines"
    )

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node} ({G.nodes[node].get('role', '')})")
        node_color.append(G.nodes[node].get("color", "#38BDF8"))
        node_size.append(G.nodes[node].get("size", 16))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=[n for n in G.nodes()],
        textposition="top center",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color="rgba(255, 255, 255, 0.4)")
        ),
        textfont=dict(color="#E2E8F0", size=10, family="Plus Jakarta Sans"),
        hoverinfo="text",
        hovertext=node_text
    )

    fig_net = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=10, l=10, r=10, t=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380
        )
    )
    st.plotly_chart(fig_net, use_container_width=True)
except Exception as e:
    st.info(f"Communication graph currently visualizing basic topology: {e}")

st.markdown('</div>', unsafe_allow_html=True)
