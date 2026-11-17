"""
Dashboard shared utilities, themes, and executive visual components.
"""
import os, sys
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

import streamlit as st

PRIORITY_CONFIG = {
    "critical": {
        "color": "#EF4444",
        "bg": "rgba(239, 68, 68, 0.12)",
        "border": "rgba(239, 68, 68, 0.35)",
        "dot": "#EF4444",
        "label": "CRITICAL"
    },
    "high": {
        "color": "#F97316",
        "bg": "rgba(249, 115, 22, 0.12)",
        "border": "rgba(249, 115, 22, 0.35)",
        "dot": "#F97316",
        "label": "HIGH"
    },
    "medium": {
        "color": "#FBBF24",
        "bg": "rgba(251, 191, 36, 0.12)",
        "border": "rgba(251, 191, 36, 0.35)",
        "dot": "#FBBF24",
        "label": "MEDIUM"
    },
    "low": {
        "color": "#10B981",
        "bg": "rgba(16, 185, 129, 0.12)",
        "border": "rgba(16, 185, 129, 0.35)",
        "dot": "#10B981",
        "label": "LOW"
    },
}

def inject_custom_theme():
    """Injects high-end Obsidian/Enterprise dark mode CSS and sidebar styling."""
    try:
        logo_path = Path(__file__).parent.parent / "assets" / "logo.png"
        if logo_path.exists():
            st.logo(str(logo_path), size="large")
    except Exception:
        pass

    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        /* Global Typography & Background */
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            color: #E2E8F0;
        }

        .stApp {
            background: radial-gradient(circle at 15% 10%, #111827 0%, #0B0F19 60%, #06090F 100%) !important;
        }

        /* ── Modern Obsidian Sidebar ───────────────────────────── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #090D16 0%, #05080E 100%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
            box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4) !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #94A3B8;
        }

        /* Sidebar Logo Container */
        [data-testid="stSidebarHeader"] {
            padding-top: 18px !important;
            padding-bottom: 8px !important;
        }
        [data-testid="stSidebarHeader"] img {
            max-height: 48px !important;
            border-radius: 8px !important;
        }

        /* Sidebar Navigation Container */
        [data-testid="stSidebarNav"] {
            padding-top: 12px !important;
            padding-bottom: 14px !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
            margin-bottom: 16px !important;
        }

        /* Section Header above Navigation */
        [data-testid="stSidebarNav"]::before {
            content: "EXECUTIVE SUITE";
            display: block;
            font-size: 10px;
            font-weight: 800;
            letter-spacing: 0.12em;
            color: #64748B;
            padding: 4px 14px 10px 14px;
            text-transform: uppercase;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Style the default "app" link to read as Command Center */
        [data-testid="stSidebarNav"] li:first-child a span {
            display: none !important;
        }
        [data-testid="stSidebarNav"] li:first-child a::after {
            content: "⚡  Command Center" !important;
            font-weight: 700 !important;
            font-size: 0.88rem !important;
            color: inherit !important;
            display: inline-block !important;
        }

        /* Individual Navigation Links */
        [data-testid="stSidebarNav"] ul {
            padding: 0 4px !important;
            gap: 3px !important;
        }
        [data-testid="stSidebarNav"] li {
            margin-bottom: 3px !important;
        }
        [data-testid="stSidebarNavLink"] {
            border-radius: 8px !important;
            padding: 8px 12px !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            letter-spacing: 0.01em !important;
            color: #94A3B8 !important;
            background: transparent !important;
            border: 1px solid transparent !important;
            transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1) !important;
            display: flex !important;
            align-items: center !important;
        }

        /* Hover Navigation Link */
        [data-testid="stSidebarNavLink"]:hover {
            background: rgba(255, 255, 255, 0.04) !important;
            border-color: rgba(255, 255, 255, 0.08) !important;
            color: #F8FAFC !important;
            transform: translateX(4px) !important;
        }

        /* Active Selected Page */
        [data-testid="stSidebarNavLink"][aria-current="page"] {
            background: linear-gradient(90deg, rgba(59, 130, 246, 0.18) 0%, rgba(59, 130, 246, 0.04) 100%) !important;
            border: 1px solid rgba(59, 130, 246, 0.35) !important;
            border-left: 3px solid #3B82F6 !important;
            color: #60A5FA !important;
            box-shadow: 0 4px 14px rgba(59, 130, 246, 0.16) !important;
        }

        [data-testid="stSidebarNavLink"] span {
            font-size: 0.88rem !important;
            font-weight: 600 !important;
            color: inherit !important;
        }

        /* Glassmorphic Cards */
        .exec-glass-card {
            background: rgba(17, 24, 39, 0.7) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 14px !important;
            padding: 18px 22px !important;
            margin-bottom: 14px !important;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
            box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4) !important;
        }
        .exec-glass-card:hover {
            transform: translateY(-2px);
            border-color: rgba(99, 102, 241, 0.35) !important;
            box-shadow: 0 8px 30px -4px rgba(99, 102, 241, 0.15) !important;
        }

        /* Metric Tiles Override */
        [data-testid="stMetric"] {
            background: rgba(17, 24, 39, 0.75) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 12px !important;
            padding: 14px 18px !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
            transition: border-color 0.2s ease;
        }
        [data-testid="stMetric"]:hover {
            border-color: rgba(96, 165, 250, 0.35) !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.8rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.06em !important;
            color: #94A3B8 !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.85rem !important;
            font-weight: 800 !important;
            color: #F8FAFC !important;
            font-feature-settings: "tnum" on, "zero" on;
        }

        /* Custom Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background-color: transparent;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            padding: 0 18px;
            border-radius: 8px 8px 0 0;
            color: #94A3B8;
            font-weight: 600;
            font-size: 0.88rem;
            border: none !important;
            background: transparent;
            transition: all 0.15s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #F8FAFC;
            background: rgba(255, 255, 255, 0.04);
        }
        .stTabs [aria-selected="true"] {
            color: #60A5FA !important;
            border-bottom: 2px solid #3B82F6 !important;
            background: rgba(59, 130, 246, 0.06) !important;
        }

        /* Streamlit Expander */
        .streamlit-expanderHeader {
            background-color: rgba(17, 24, 39, 0.6) !important;
            border: 1px solid rgba(255, 255, 255, 0.06) !important;
            border-radius: 10px !important;
            color: #E2E8F0 !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
        }
        .streamlit-expanderContent {
            border: 1px solid rgba(255, 255, 255, 0.06) !important;
            border-top: none !important;
            border-radius: 0 0 10px 10px !important;
            background-color: rgba(11, 15, 25, 0.4) !important;
        }

        /* Primary Button */
        button[kind="primary"] {
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            color: white !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
            transition: all 0.2s ease !important;
        }
        button[kind="primary"]:hover {
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5) !important;
            transform: translateY(-1px);
        }

        /* Form Inputs */
        input, select, textarea {
            background-color: #111827 !important;
            color: #F1F5F9 !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 8px !important;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 7px;
            height: 7px;
        }
        ::-webkit-scrollbar-track {
            background: #0B0F19;
        }
        ::-webkit-scrollbar-thumb {
            background: #1F2937;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #374151;
        }
    </style>
    """, unsafe_allow_html=True)

def priority_badge(level: str) -> str:
    """Returns an ultra-clean, glowing priority badge."""
    lvl = (level or "low").lower()
    cfg = PRIORITY_CONFIG.get(lvl, PRIORITY_CONFIG["low"])
    return (
        f'<span style="display:inline-flex; align-items:center; gap:6px; '
        f'background:{cfg["bg"]}; border:1px solid {cfg["border"]}; color:{cfg["color"]}; '
        f'padding:3px 10px; border-radius:20px; font-weight:700; font-size:11px; '
        f'letter-spacing:0.04em; text-transform:uppercase; box-shadow:0 0 10px {cfg["bg"]};">'
        f'<span style="width:6px; height:6px; border-radius:50%; background:{cfg["dot"]}; '
        f'box-shadow:0 0 6px {cfg["dot"]};"></span>'
        f'{cfg["label"]}</span>'
    )

def severity_badge(level: str) -> str:
    return priority_badge(level)

def category_pill(cat: str) -> str:
    name = (cat or "general").replace("_", " ").title()
    return (
        f'<span style="background:rgba(59, 130, 246, 0.1); border:1px solid rgba(59, 130, 246, 0.25); '
        f'color:#93C5FD; padding:2px 8px; border-radius:6px; font-size:11px; font-weight:600;">'
        f'{name}</span>'
    )

def deadline_pill(deadline: str) -> str:
    if not deadline:
        return ""
    return (
        f'<span style="background:rgba(239, 68, 68, 0.1); border:1px solid rgba(239, 68, 68, 0.25); '
        f'color:#FCA5A5; padding:2px 8px; border-radius:6px; font-size:11px; font-weight:600;">'
        f'⏰ {deadline}</span>'
    )

def render_hero(title: str, subtitle: str, badge_text: str = None):
    """Renders a sleek luxury gradient page header."""
    badge_html = f'<div style="display:inline-block; margin-bottom:8px; padding:3px 10px; background:rgba(99, 102, 241, 0.15); border:1px solid rgba(99, 102, 241, 0.3); border-radius:20px; font-size:11px; font-weight:700; color:#A5B4FC; letter-spacing:0.08em; text-transform:uppercase;">{badge_text}</div>' if badge_text else ''
    st.markdown(f"""
    <div style="margin-bottom:24px; padding-bottom:14px; border-bottom:1px solid rgba(255,255,255,0.06);">
        {badge_html}
        <h1 style="font-size:2.1rem; font-weight:800; margin:0 0 6px 0; background:linear-gradient(135deg, #FFFFFF 20%, #94A3B8 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            {title}
        </h1>
        <p style="font-size:0.95rem; color:#94A3B8; margin:0; font-weight:400; line-height:1.5;">
            {subtitle}
        </p>
    </div>
    """, unsafe_allow_html=True)

def style_plotly(fig, height: int = 280):
    """Enforces dark luxury theme styling on Plotly charts."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(t=25, b=20, l=35, r=20),
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#94A3B8", size=12),
        legend=dict(
            bgcolor="rgba(17,24,39,0.7)",
            bordercolor="rgba(255,255,255,0.08)",
            font=dict(color="#E2E8F0")
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.08)",
            tickfont=dict(color="#94A3B8")
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.08)",
            tickfont=dict(color="#94A3B8")
        )
    )
    return fig

@st.cache_resource(show_spinner="Connecting to local database…")
def get_db():
    from src.database.db_manager import DatabaseManager
    db = DatabaseManager()
    db.init_db()
    return db

@st.cache_resource(show_spinner="Loading priority engine…")
def get_priority_engine():
    db = get_db()
    from src.intelligence.priority_engine import PriorityEngine
    return PriorityEngine(db)

def _val(obj, key, default=""):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def no_data_message(entity: str = "emails"):
    st.markdown(f"""
    <div class="exec-glass-card" style="text-align:center; padding:30px 20px;">
        <div style="font-size:2rem; margin-bottom:8px;">📁</div>
        <h3 style="color:#F1F5F9; margin-bottom:6px;">No {entity.title()} Found</h3>
        <p style="color:#94A3B8; font-size:0.9rem; margin-bottom:16px;">The database is currently empty. Initialize and train the local intelligence models to populate.</p>
        <code style="background:#111827; padding:6px 12px; border-radius:6px; color:#60A5FA;">python training/train_priority.py</code>
    </div>
    """, unsafe_allow_html=True)
