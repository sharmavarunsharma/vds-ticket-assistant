"""
app.py — Main Streamlit UI for VDS Jira Ticket Assistant
Tabs: Analyze Ticket | Chatbot | Reports
Author: Built for Ahead.com / EWS & DevOps ServiceDesk / VDS Project
"""

import os
import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─── Import project modules ────────────────────────────────────────────────────
from utils import (
    load_tickets_from_csv,
    build_faiss_index,
    search_similar_tickets,
    extract_text_from_image,
    generate_ai_response,
    chat_with_tickets,
)
from rules import apply_rules, get_priority_badge, generate_routing_comment
from report import generate_weekly_report, format_report_for_display
from scheduler import start_scheduler, stop_scheduler, get_scheduler_status, get_schedule_info, run_now


# ─── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VDS Jira Ticket Assistant",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Import Fonts */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@300;400;500;600;700&display=swap');

    /* Root variables */
    :root {
        --bg-dark: #0d1117;
        --bg-card: #161b22;
        --bg-card2: #1c2128;
        --border: #30363d;
        --accent-blue: #58a6ff;
        --accent-green: #3fb950;
        --accent-yellow: #d29922;
        --accent-red: #f85149;
        --accent-purple: #bc8cff;
        --text-primary: #e6edf3;
        --text-secondary: #8b949e;
        --text-muted: #6e7681;
        --font-mono: 'JetBrains Mono', monospace;
        --font-sans: 'DM Sans', sans-serif;
    }

    /* Global */
    html, body, [class*="css"] {
        font-family: var(--font-sans);
        background-color: var(--bg-dark);
        color: var(--text-primary);
    }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Main container */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Header banner */
    .app-header {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1c2128 100%);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        border-left: 4px solid var(--accent-blue);
    }
    .app-header h1 {
        font-family: var(--font-mono);
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--accent-blue);
        margin: 0;
        letter-spacing: -0.02em;
    }
    .app-header p {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin: 0;
    }

    /* Cards */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-family: var(--font-mono);
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-secondary);
        margin-bottom: 0.75rem;
    }

    /* Similar ticket cards */
    .ticket-card {
        background: var(--bg-card2);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        border-left: 3px solid var(--accent-blue);
        transition: border-color 0.2s;
    }
    .ticket-card:hover {
        border-left-color: var(--accent-purple);
    }
    .ticket-id {
        font-family: var(--font-mono);
        font-size: 0.78rem;
        color: var(--accent-blue);
        font-weight: 600;
    }
    .ticket-summary {
        font-size: 0.9rem;
        font-weight: 500;
        color: var(--text-primary);
        margin: 0.3rem 0;
    }
    .ticket-resolution {
        font-size: 0.82rem;
        color: var(--text-secondary);
        margin-top: 0.4rem;
    }
    .similarity-badge {
        float: right;
        font-family: var(--font-mono);
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 600;
    }

    /* Routing card */
    .routing-card {
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .routing-team {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .routing-action {
        font-size: 0.88rem;
        opacity: 0.85;
    }

    /* Priority badges */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: var(--font-mono);
        letter-spacing: 0.05em;
    }
    .badge-red { background: rgba(248,81,73,0.15); color: #f85149; border: 1px solid rgba(248,81,73,0.3); }
    .badge-orange { background: rgba(249,115,22,0.15); color: #fb923c; border: 1px solid rgba(249,115,22,0.3); }
    .badge-yellow { background: rgba(210,153,34,0.15); color: #d29922; border: 1px solid rgba(210,153,34,0.3); }
    .badge-green { background: rgba(63,185,80,0.15); color: #3fb950; border: 1px solid rgba(63,185,80,0.3); }
    .badge-gray { background: rgba(110,118,129,0.15); color: #8b949e; border: 1px solid rgba(110,118,129,0.3); }

    /* AI Response sections */
    .ai-section {
        background: var(--bg-card2);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
    }
    .ai-section-title {
        font-family: var(--font-mono);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
    }
    .ai-section-content {
        font-size: 0.9rem;
        color: var(--text-primary);
        line-height: 1.6;
    }

    /* Step list */
    .step-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    .step-list li {
        padding: 0.4rem 0;
        font-size: 0.88rem;
        color: var(--text-primary);
        display: flex;
        gap: 0.5rem;
        border-bottom: 1px solid var(--border);
    }
    .step-list li:last-child { border-bottom: none; }
    .step-num {
        color: var(--accent-blue);
        font-family: var(--font-mono);
        font-size: 0.78rem;
        font-weight: 700;
        min-width: 20px;
    }

    /* Copy button area */
    .jira-comment-box {
        background: #0d1117;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
        font-family: var(--font-mono);
        font-size: 0.82rem;
        color: var(--text-secondary);
        white-space: pre-wrap;
        line-height: 1.6;
    }

    /* Chat bubbles */
    .chat-user {
        background: rgba(88,166,255,0.1);
        border: 1px solid rgba(88,166,255,0.2);
        border-radius: 12px 12px 2px 12px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        margin-left: 15%;
        font-size: 0.9rem;
    }
    .chat-bot {
        background: var(--bg-card2);
        border: 1px solid var(--border);
        border-radius: 12px 12px 12px 2px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        margin-right: 5%;
        font-size: 0.9rem;
    }
    .chat-label {
        font-family: var(--font-mono);
        font-size: 0.7rem;
        color: var(--text-muted);
        margin-bottom: 0.3rem;
    }

    /* Stats cards */
    .stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .stat-value {
        font-family: var(--font-mono);
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent-blue);
    }
    .stat-label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.3rem;
    }

    /* Tabs override */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card);
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 6px;
        color: var(--text-secondary);
        font-family: var(--font-mono);
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        padding: 0.4rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: var(--bg-dark);
        color: var(--accent-blue) !important;
    }

    /* Input fields */
    .stTextArea textarea, .stTextInput input {
        background: var(--bg-card2) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
        font-family: var(--font-sans) !important;
    }

    /* Buttons */
    .stButton > button {
        background: var(--bg-card2);
        border: 1px solid var(--border);
        color: var(--text-primary);
        border-radius: 8px;
        font-family: var(--font-mono);
        font-size: 0.82rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        border-color: var(--accent-blue);
        color: var(--accent-blue);
        background: rgba(88,166,255,0.05);
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: var(--accent-blue);
        color: #0d1117;
        border-color: var(--accent-blue);
    }
    .stButton > button[kind="primary"]:hover {
        background: #79c0ff;
        color: #0d1117;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--bg-card);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }

    /* Divider */
    hr { border-color: var(--border); margin: 1.5rem 0; }

    /* File uploader */
    .stFileUploader {
        background: var(--bg-card2);
        border: 1px dashed var(--border);
        border-radius: 8px;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg-card2);
        border-radius: 8px;
        font-family: var(--font-mono);
        font-size: 0.82rem;
    }

    /* Success/Warning/Error */
    .stSuccess { border-radius: 8px; }
    .stWarning { border-radius: 8px; }
    .stError { border-radius: 8px; }
    .stInfo { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ────────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "df": None,
        "faiss_index": None,
        "texts": None,
        "chat_history": [],
        "last_report": None,
        "scheduler_started": False,
        "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace; font-size:1rem; font-weight:700; 
                color:#58a6ff; margin-bottom:0.25rem;">🎫 VDS Ticket Assistant</div>
    <div style="font-size:0.75rem; color:#8b949e; margin-bottom:1.5rem;">
        Ahead.com · EWS & DevOps ServiceDesk
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### ⚙️ Settings")

    # API Key input
    api_key_input = st.text_input(
        "Anthropic API Key",
        value=st.session_state.api_key,
        type="password",
        help="Get your key from console.anthropic.com"
    )
    if api_key_input:
        st.session_state.api_key = api_key_input

    if st.session_state.api_key:
        st.success("✅ API Key set")
    else:
        st.warning("⚠️ No API key — AI features disabled")

    st.divider()

    # CSV Upload (in sidebar)
    st.markdown("#### 📂 Knowledge Base")
    uploaded_csv = st.file_uploader(
        "Upload Tickets CSV",
        type=["csv"],
        help="Upload your Jira CSV export. Needs Summary, Description, Resolution columns."
    )

    if uploaded_csv:
        with st.spinner("Building FAISS index..."):
            try:
                df = load_tickets_from_csv(uploaded_csv)
                faiss_index, texts, df = build_faiss_index(df)
                st.session_state.df = df
                st.session_state.faiss_index = faiss_index
                st.session_state.texts = texts
                st.success(f"✅ {len(df)} tickets indexed!")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown(f"""
        <div style="background:#1c2128;border:1px solid #30363d;border-radius:8px;padding:0.75rem;font-size:0.8rem;">
        📊 <b style="color:#58a6ff;">{len(df)}</b> tickets in knowledge base<br>
        📁 Columns: <span style="color:#8b949e;">{', '.join(df.columns[:5].tolist())}</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Scheduler status
    st.markdown("#### ⏰ Scheduler")
    sched_info = get_schedule_info()

    if not sched_info["available"]:
        st.info("APScheduler not installed. Install it for auto-reports.")
    else:
        sched_status = get_scheduler_status()
        if sched_status["running"]:
            st.success(f"🟢 Active · {sched_info['schedule']}")
            if sched_status.get("next_run"):
                st.caption(f"Next: {sched_status['next_run'][:19]}")
            if st.button("Stop Scheduler", use_container_width=True):
                stop_scheduler()
                st.session_state.scheduler_started = False
                st.rerun()
        else:
            st.info(f"⚪ Inactive · {sched_info['schedule']}")
            if st.button("▶ Start Auto-Reports", use_container_width=True):
                if st.session_state.df is not None:
                    result = start_scheduler(
                        report_callback=generate_weekly_report,
                        report_kwargs={"df": st.session_state.df}
                    )
                    st.session_state.scheduler_started = True
                    st.success(result["message"])
                    st.rerun()
                else:
                    st.warning("Upload CSV first to enable scheduler")

    st.divider()
    st.markdown("""
    <div style="font-size:0.73rem;color:#6e7681;line-height:1.7;">
    🔧 <b>Stack:</b> Streamlit · FAISS<br>
    🤖 sentence-transformers<br>
    👁️ pytesseract OCR<br>
    🧠 Anthropic Claude API<br><br>
    <span style="color:#3fb950;">v1.0.0</span> · Ahead.com
    </div>
    """, unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <div style="font-size:2rem;">🎫</div>
    <div>
        <h1>VDS Jira Ticket Assistant</h1>
        <p>AI-powered ticket analysis · Rule-based routing · Weekly reporting · RAG Chatbot</p>
    </div>
    <div style="margin-left:auto; text-align:right;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#58a6ff;">EWS & DevOps ServiceDesk</div>
        <div style="font-size:0.72rem; color:#8b949e;">Stellantis VDS Project · Ahead.com</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔍  Analyze Ticket", "💬  Chatbot", "📊  Reports"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYZE TICKET
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("#### Enter a New Ticket")

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        # Input method selector
        input_method = st.radio(
            "Input Method",
            ["✏️ Text / Paste", "🖼️ Screenshot (OCR)"],
            horizontal=True,
            label_visibility="collapsed"
        )

        ticket_text = ""

        if input_method == "✏️ Text / Paste":
            ticket_text = st.text_area(
                "Paste ticket description or summary",
                height=180,
                placeholder="e.g. User cannot SSH into EC2 instance. Getting permission denied error. Tried restarting but still failing...",
                label_visibility="collapsed"
            )

        else:
            # Image upload with preview
            img_file = st.file_uploader(
                "Upload screenshot",
                type=["png", "jpg", "jpeg", "bmp", "tiff"],
                label_visibility="collapsed"
            )
            if img_file:
                st.image(img_file, caption="Uploaded Screenshot", use_column_width=True)
                with st.spinner("🔍 Running OCR..."):
                    ticket_text = extract_text_from_image(img_file)

                if ticket_text and not ticket_text.startswith("❌") and not ticket_text.startswith("⚠️"):
                    st.success("✅ Text extracted successfully")
                    st.text_area("Extracted Text (editable)", value=ticket_text, height=120, key="ocr_text")
                    ticket_text = st.session_state.get("ocr_text", ticket_text)
                else:
                    st.error(ticket_text)

        # Top-k slider
        top_k = st.slider("Similar tickets to find", min_value=1, max_value=10, value=5)

        analyze_btn = st.button(
            "⚡ Analyze Ticket",
            use_container_width=True,
            type="primary",
            disabled=not bool(ticket_text.strip())
        )

    # ── Results Column ─────────────────────────────────────────────────────────
    with col_result:
        if analyze_btn and ticket_text.strip():

            # 1. Rule Engine
            routing = apply_rules(ticket_text)
            priority_colors = {
                "Highest": ("#ef4444", "badge-red"),
                "High": ("#f97316", "badge-orange"),
                "Medium": ("#d29922", "badge-yellow"),
                "Low": ("#3fb950", "badge-green"),
            }
            p_color, p_badge = priority_colors.get(routing["priority"], ("#6b7280", "badge-gray"))

            st.markdown(f"""
            <div class="routing-card" style="background: {routing['color']}18; border: 1px solid {routing['color']}44;">
                <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.3rem;">
                    <span style="font-size:1.3rem;">{routing['icon']}</span>
                    <span class="routing-team" style="color:{routing['color']};">{routing['team']}</span>
                    <span class="badge {p_badge}" style="margin-left:auto;">{routing['priority']}</span>
                </div>
                <div class="routing-action" style="color:{routing['color']}cc;">{routing['action']}</div>
                <div style="margin-top:0.5rem; font-size:0.77rem; color:#8b949e;">
                    Queue: <code style="color:#58a6ff;">{routing['queue']}</code> &nbsp;|&nbsp; 
                    SLA: <code style="color:#3fb950;">{routing['sla']}</code>
                    {f" &nbsp;|&nbsp; Keywords: <code>{', '.join(routing['matched_keywords'])}</code>" if routing['matched_keywords'] else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 2. Similar Tickets
            if st.session_state.faiss_index is not None:
                similar = search_similar_tickets(
                    ticket_text,
                    st.session_state.faiss_index,
                    st.session_state.texts,
                    st.session_state.df,
                    top_k=top_k
                )

                if similar:
                    st.markdown(f"<div class='card-title'>🔎 Top {len(similar)} Similar Tickets</div>", unsafe_allow_html=True)
                    for t in similar:
                        sim_color = "#3fb950" if t["similarity"] > 75 else "#d29922" if t["similarity"] > 50 else "#8b949e"
                        st.markdown(f"""
                        <div class="ticket-card">
                            <div>
                                <span class="ticket-id">{t['ticket_id']}</span>
                                <span class="similarity-badge" style="background:{sim_color}22;color:{sim_color};border:1px solid {sim_color}44;">
                                    {t['similarity']}%
                                </span>
                            </div>
                            <div class="ticket-summary">{t['summary'][:100]}</div>
                            <div class="ticket-resolution">💡 {t['resolution'][:200] if t['resolution'] else 'No resolution recorded'}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No similar tickets found in knowledge base.")
            else:
                similar = []
                st.info("💡 Upload a CSV to enable similarity search.")

            # 3. AI Resolution Guide
            if st.session_state.api_key:
                with st.spinner("🧠 Generating AI resolution guide..."):
                    ai_result = generate_ai_response(ticket_text, similar, st.session_state.api_key)

                if "error" not in ai_result:
                    st.markdown("---")

                    # Confidence + Category
                    conf = ai_result.get("confidence", 0)
                    conf_color = "#3fb950" if conf > 70 else "#d29922" if conf > 40 else "#f85149"
                    st.markdown(f"""
                    <div style="display:flex;gap:0.75rem;margin-bottom:1rem;flex-wrap:wrap;">
                        <span class="badge" style="background:{conf_color}22;color:{conf_color};border:1px solid {conf_color}44;">
                            🎯 Confidence: {conf}%
                        </span>
                        <span class="badge badge-gray">📂 {ai_result.get('category','Unknown')}</span>
                        <span class="badge badge-gray">⏱️ {ai_result.get('estimated_resolution_time','TBD')}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Root Cause
                    st.markdown(f"""
                    <div class="ai-section">
                        <div class="ai-section-title">🔍 Root Cause</div>
                        <div class="ai-section-content">{ai_result.get('root_cause','')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Solution Steps
                    steps_html = "".join([
                        f'<li><span class="step-num">{i+1:02d}</span><span>{step}</span></li>'
                        for i, step in enumerate(ai_result.get("solution_steps", []))
                    ])
                    st.markdown(f"""
                    <div class="ai-section">
                        <div class="ai-section-title">✅ Step-by-Step Solution</div>
                        <ul class="step-list">{steps_html}</ul>
                    </div>
                    """, unsafe_allow_html=True)

                    # Troubleshooting
                    trouble_html = "".join([
                        f'<li><span class="step-num">→</span><span>{step}</span></li>'
                        for step in ai_result.get("troubleshooting_steps", [])
                    ])
                    st.markdown(f"""
                    <div class="ai-section">
                        <div class="ai-section-title">🛠️ Troubleshooting Checklist</div>
                        <ul class="step-list">{trouble_html}</ul>
                    </div>
                    """, unsafe_allow_html=True)

                    # Preventive Measures
                    prev_html = "".join([
                        f'<li><span class="step-num">🛡️</span><span>{m}</span></li>'
                        for m in ai_result.get("preventive_measures", [])
                    ])
                    st.markdown(f"""
                    <div class="ai-section">
                        <div class="ai-section-title">🛡️ Preventive Measures</div>
                        <ul class="step-list">{prev_html}</ul>
                    </div>
                    """, unsafe_allow_html=True)

                    # Jira Comment (copyable)
                    with st.expander("📋 Copy-Paste Jira Comment", expanded=True):
                        jira_comment = ai_result.get("jira_comment", "")
                        routing_comment = generate_routing_comment(routing, ticket_text[:100])
                        full_comment = f"{jira_comment}\n\n---\n{routing_comment}"
                        st.code(full_comment, language=None)
                else:
                    st.error(f"AI Error: {ai_result.get('root_cause','')}")
            else:
                st.warning("⚠️ Set your Anthropic API key in the sidebar to enable AI analysis.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("#### 💬 Ask anything about your tickets")

    if st.session_state.faiss_index is None:
        st.info("📂 Upload a CSV in the sidebar to enable the chatbot knowledge base.")

    # Chat history display
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align:center;padding:2rem;color:#6e7681;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">💬</div>
                <div style="font-size:0.9rem;">Ask me about your ticket patterns, resolutions, or anything support-related.</div>
                <div style="font-size:0.8rem;margin-top:0.5rem;">
                    Try: "What are the most common VPN issues?" or "How do I resolve SSH permission denied?"
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-user">
                        <div class="chat-label">YOU</div>
                        {msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-bot">
                        <div class="chat-label">🤖 VDS ASSISTANT</div>
                        {msg["content"].replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("---")

    # Chat input
    col_chat, col_clear = st.columns([5, 1])
    with col_chat:
        user_msg = st.text_input(
            "Message",
            placeholder="Ask about ticket patterns, resolutions, team routing...",
            label_visibility="collapsed",
            key="chat_input"
        )
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    send_btn = st.button("Send Message ↑", type="primary", use_container_width=True)

    if send_btn and user_msg.strip():
        if not st.session_state.api_key:
            st.error("Set API key in sidebar first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})

            with st.spinner("Thinking..."):
                if st.session_state.faiss_index is not None:
                    reply = chat_with_tickets(
                        user_msg,
                        st.session_state.chat_history,
                        st.session_state.faiss_index,
                        st.session_state.texts,
                        st.session_state.df,
                        st.session_state.api_key
                    )
                else:
                    # Fallback: answer from general knowledge
                    import anthropic
                    client = anthropic.Anthropic(api_key=st.session_state.api_key)
                    response = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=600,
                        system="You are an expert IT support chatbot for an enterprise DevOps team (Stellantis VDS project). Answer questions helpfully and concisely.",
                        messages=[{"role": "user", "content": user_msg}]
                    )
                    reply = response.content[0].text

            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

    # Suggested questions
    st.markdown("**💡 Quick questions:**")
    q_cols = st.columns(3)
    suggestions = [
        "What are common VPN issues?",
        "How to fix SSH permission denied?",
        "What tickets need escalation?",
        "Show top 3 categories this week",
        "How to handle P1 tickets?",
        "What's the SLA for Highest priority?"
    ]
    for i, q in enumerate(suggestions):
        with q_cols[i % 3]:
            if st.button(q, key=f"sugg_{i}", use_container_width=True):
                if st.session_state.api_key:
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    with st.spinner("Thinking..."):
                        if st.session_state.faiss_index is not None:
                            reply = chat_with_tickets(
                                q, st.session_state.chat_history,
                                st.session_state.faiss_index,
                                st.session_state.texts, st.session_state.df,
                                st.session_state.api_key
                            )
                        else:
                            import anthropic
                            client = anthropic.Anthropic(api_key=st.session_state.api_key)
                            res = client.messages.create(
                                model="claude-sonnet-4-20250514", max_tokens=400,
                                messages=[{"role": "user", "content": q}]
                            )
                            reply = res.content[0].text
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — REPORTS
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("#### 📊 Weekly Ticket Report")

    if st.session_state.df is None:
        st.info("📂 Upload a CSV file in the sidebar to generate reports.")
    else:
        df = st.session_state.df

        col_r1, col_r2 = st.columns([1, 2], gap="large")

        with col_r1:
            st.markdown("**Generate Report**")

            report_file = st.text_input("Output filename", value="weekly_report.txt")

            gen_btn = st.button("📊 Generate Now", type="primary", use_container_width=True)

            # Scheduler section
            st.markdown("---")
            sched_info = get_schedule_info()
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Auto Schedule</div>
                <div style="font-size:0.85rem;">📅 {sched_info['schedule']}</div>
                <div style="font-size:0.8rem;color:#8b949e;margin-top:0.3rem;">
                    Output: <code>{sched_info['output_file']}</code>
                </div>
                <div style="font-size:0.8rem;color:#8b949e;">{sched_info['timezone']}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_r2:
            if gen_btn or st.session_state.last_report:

                if gen_btn:
                    with st.spinner("Generating report..."):
                        report = generate_weekly_report(df, output_path=report_file)
                        st.session_state.last_report = report
                        st.success(f"✅ Report saved to `{report_file}`")

                if st.session_state.last_report:
                    report = st.session_state.last_report

                    # Stats row
                    s1, s2, s3, s4 = st.columns(4)
                    with s1:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-value">{report['total_tickets']}</div>
                            <div class="stat-label">Total</div>
                        </div>""", unsafe_allow_html=True)
                    with s2:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-value" style="color:#f85149;">{report['open_tickets']}</div>
                            <div class="stat-label">Open</div>
                        </div>""", unsafe_allow_html=True)
                    with s3:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-value" style="color:#3fb950;">{report['closed_tickets']}</div>
                            <div class="stat-label">Closed</div>
                        </div>""", unsafe_allow_html=True)
                    with s4:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-value" style="color:#d29922;">{report['resolution_rate']}%</div>
                            <div class="stat-label">Resolved</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Top issues chart
                    if report.get("top_issues"):
                        st.markdown("**🔍 Top Issue Categories**")
                        issues_df = pd.DataFrame(
                            list(report["top_issues"].items()),
                            columns=["Category", "Count"]
                        ).head(8)
                        st.bar_chart(issues_df.set_index("Category"))

                    # Priority breakdown
                    if report.get("priority_breakdown"):
                        st.markdown("**🏷️ Priority Breakdown**")
                        prio_df = pd.DataFrame(
                            list(report["priority_breakdown"].items()),
                            columns=["Priority", "Count"]
                        )
                        st.dataframe(prio_df, hide_index=True, use_container_width=True)

                    # Suggested actions
                    st.markdown("**💡 Suggested Actions**")
                    for action in report["suggested_actions"]:
                        st.markdown(f"> {action}")

                    # Download button
                    try:
                        with open(report_file, "r") as f:
                            report_content = f.read()
                        st.download_button(
                            "⬇️ Download Report (.txt)",
                            data=report_content,
                            file_name=report_file,
                            mime="text/plain",
                            use_container_width=True
                        )
                    except FileNotFoundError:
                        pass

        # Raw ticket table
        with st.expander("📋 View Raw Ticket Data", expanded=False):
            display_cols = [c for c in ["ticket_id", "summary", "status", "priority", "assignee"] if c in df.columns]
            if display_cols:
                st.dataframe(df[display_cols], use_container_width=True, height=300)
            else:
                st.dataframe(df.head(50), use_container_width=True, height=300)
