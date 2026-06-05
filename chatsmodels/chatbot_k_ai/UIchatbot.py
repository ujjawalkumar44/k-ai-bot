import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="K AI Bot", page_icon="🤖", layout="centered")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #08090c;
    color: #ddd9d0;
}
#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 2.2rem 2rem 7rem 2rem;
    max-width: 800px;
}

/* ── Brand header ── */
.brand {
    display: flex;
    align-items: baseline;
    gap: 0.4rem;
    margin-bottom: 0.15rem;
}
.brand-k {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: #ffffff;
    line-height: 1;
}
.brand-ai {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: #e05aff;
    line-height: 1;
}
.brand-bot {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: #ffffff;
    line-height: 1;
}
.brand-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #3e3d45;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1.6rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #141419;
}

/* ── Mode selector ── */
.mode-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #4a4855;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* Radio pills */
div[data-testid="stRadio"] > div {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
}
div[data-testid="stRadio"] label {
    background: #111118 !important;
    border: 1px solid #222230 !important;
    border-radius: 999px !important;
    padding: 0.3rem 0.85rem !important;
    font-size: 0.8rem !important;
    font-family: 'Syne', sans-serif !important;
    color: #6e6b80 !important;
    cursor: pointer;
    transition: all 0.15s;
}
div[data-testid="stRadio"] label:hover {
    border-color: #e05aff !important;
    color: #e05aff !important;
}
div[data-testid="stRadio"] [aria-checked="true"] + div label,
div[data-testid="stRadio"] label[data-checked="true"] {
    background: #1e0f26 !important;
    border-color: #e05aff !important;
    color: #e05aff !important;
}
/* Hide the actual radio dot */
div[data-testid="stRadio"] [type="radio"] { display: none !important; }

/* ── Chat area ── */
.chat-area {
    max-height: 52vh;
    overflow-y: auto;
    padding-right: 0.2rem;
    margin: 1.4rem 0 1.2rem 0;
}
.chat-area::-webkit-scrollbar { width: 3px; }
.chat-area::-webkit-scrollbar-thumb { background: #222230; border-radius: 3px; }

.chat-row {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    margin-bottom: 1.2rem;
}
.chat-row.user { flex-direction: row-reverse; }

.avatar {
    width: 32px; height: 32px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 700;
    flex-shrink: 0;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.02em;
}
.avatar.bot  { background: #180e22; color: #e05aff; border: 1px solid #2e1c3e; }
.avatar.user { background: #0e1420; color: #5ab4ff; border: 1px solid #1a2535; }

.bubble {
    padding: 0.7rem 1rem;
    border-radius: 14px;
    max-width: 76%;
    line-height: 1.65;
    font-size: 0.9rem;
}
.bubble.bot {
    background: #0f0f16;
    border: 1px solid #1e1e2e;
    color: #bfbbce;
    border-top-left-radius: 3px;
    font-family: 'DM Mono', monospace;
    font-size: 0.84rem;
}
.bubble.user {
    background: #0c1420;
    border: 1px solid #192030;
    color: #c8dff5;
    border-top-right-radius: 3px;
}

/* Mode badge on bubble */
.mode-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
    margin-bottom: 0.4rem;
    background: #1e0f26;
    color: #e05aff;
    border: 1px solid #2e1c3e;
}

/* ── Input ── */
.stTextInput > div > div > input {
    background: #0d0d14 !important;
    border: 1px solid #222230 !important;
    border-radius: 12px !important;
    color: #ddd9d0 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.7rem 1.1rem !important;
    caret-color: #e05aff;
}
.stTextInput > div > div > input:focus {
    border-color: #e05aff !important;
    box-shadow: 0 0 0 2px rgba(224,90,255,0.12) !important;
}
.stTextInput > div > div > input::placeholder { color: #333340 !important; }

/* ── Send button ── */
.stButton > button {
    background: linear-gradient(135deg, #c030f0, #8020d0) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    padding: 0.7rem 1.4rem !important;
    letter-spacing: 0.02em;
    transition: opacity 0.15s;
    width: 100%;
}
.stButton > button:hover { opacity: 0.82 !important; }

/* ── Clear button ── */
.clear-wrap > div > button {
    background: transparent !important;
    border: 1px solid #1e1e28 !important;
    color: #383840 !important;
    font-size: 0.75rem !important;
    padding: 0.28rem 0.75rem !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
}
.clear-wrap > div > button:hover {
    border-color: #e05aff !important;
    color: #e05aff !important;
}

/* ── Divider under mode picker ── */
.mode-divider {
    border: none;
    border-top: 1px solid #141419;
    margin: 1rem 0 0 0;
}

/* ── Session ended banner ── */
.ended-banner {
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #e05aff;
    letter-spacing: 0.06em;
    padding: 0.6rem;
    border: 1px solid #2e1c3e;
    border-radius: 8px;
    background: #180e22;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Model (cached) ────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return ChatMistralAI(model="mistral-small-2603", temperature=0.9)

model = get_model()

# ── Mode map ──────────────────────────────────────────────────────────────────
MODES = {
    "😤 Angry":        "You are an angry AI Agent, you answer questions in an angry manner.",
    "😊 Friendly":     "You are a friendly AI Agent, you answer questions in a friendly manner.",
    "💼 Professional": "You are a professional AI Agent, you answer questions in a professional manner.",
    "😂 Funny":        "You are a funny AI Agent, you answer questions in a funny manner.",
    "😄 Happy":        "You are a happy AI Agent, you answer questions in a happy manner.",
}

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_mode" not in st.session_state:
    st.session_state.current_mode = None
if "ended" not in st.session_state:
    st.session_state.ended = False

# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_clear = st.columns([5, 1])
with col_title:
    st.markdown("""
    <div class="brand">
        <span class="brand-k">K</span>
        <span class="brand-ai"> AI</span>
        <span class="brand-bot"> Bot</span>
    </div>
    <p class="brand-sub">choose your mode below</p>
    """, unsafe_allow_html=True)
with col_clear:
    st.markdown('<div class="clear-wrap">', unsafe_allow_html=True)
    if st.button("clear"):
        st.session_state.messages = []
        st.session_state.current_mode = None
        st.session_state.ended = False
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ── Mode selector ─────────────────────────────────────────────────────────────
st.markdown('<p class="mode-label">Agent Mode</p>', unsafe_allow_html=True)
selected_mode = st.radio(
    label="mode",
    options=list(MODES.keys()),
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown('<hr class="mode-divider">', unsafe_allow_html=True)

# Reset history if mode changed
if selected_mode != st.session_state.current_mode:
    st.session_state.messages = [SystemMessage(content=MODES[selected_mode])]
    st.session_state.current_mode = selected_mode
    st.session_state.ended = False

# ── Chat display ──────────────────────────────────────────────────────────────
chat_html = '<div class="chat-area">'
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        chat_html += f"""
        <div class="chat-row user">
            <div class="avatar user">you</div>
            <div class="bubble user">{msg.content}</div>
        </div>"""
    elif isinstance(msg, AIMessage):
        chat_html += f"""
        <div class="chat-row">
            <div class="avatar bot">K·AI</div>
            <div class="bubble bot">
                <span class="mode-badge">{selected_mode}</span><br>
                {msg.content}
            </div>
        </div>"""
chat_html += "</div>"
st.markdown(chat_html, unsafe_allow_html=True)

# ── Session ended state ───────────────────────────────────────────────────────
if st.session_state.ended:
    st.markdown('<div class="ended-banner">SESSION ENDED · REFRESH OR CLEAR TO RESTART</div>', unsafe_allow_html=True)
    st.stop()

# ── Input row (form enables Enter key to submit) ──────────────────────────────
with st.form(key="chat_form", clear_on_submit=True):
    col_input, col_btn = st.columns([6, 1])
    with col_input:
        prompt = st.text_input(
            label="msg",
            label_visibility="collapsed",
            placeholder="Type a message… (0 to exit)",
            key="user_input",
        )
    with col_btn:
        send = st.form_submit_button("Send")

# ── Logic ─────────────────────────────────────────────────────────────────────
if send and prompt:
    if prompt.strip() == "0":
        st.session_state.ended = True
        st.rerun()

    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.spinner("K AI Bot is thinking…"):
        response = model.invoke(st.session_state.messages)

    st.session_state.messages.append(AIMessage(content=response.content))
    st.rerun()
    
    
    
    
    