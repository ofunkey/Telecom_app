import os
import streamlit as st
from dotenv import load_dotenv

# 1. MUST be the first Streamlit command executed
st.set_page_config(
    page_title="Telecom Support Chat",
    page_icon="📡",
    layout="centered",
)

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
load_dotenv()

# 2. Automatically sync GROQ_API_KEY from Streamlit Secrets to environment variables
if "GROQ_API_KEY" not in os.environ and "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from rag_chain import build_chain

SAMPLE_QUESTIONS = [
    "Why is my mobile internet so slow?",
    "My calls keep dropping — what should I do?",
    "How do I activate international roaming?",
    "Why is my bill higher than usual this month?",
    "My phone shows SIM not detected after a restart",
    "How do I enable Wi-Fi calling?",
    "I was charged for roaming but had a bundle active",
    "How do I unlock my phone for another network?",
]


@st.cache_resource
def get_chain():
    return build_chain()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 Telecom Support")
    st.caption("Powered by RAG · Qwen3.6-27B on Groq")
    st.divider()

    st.markdown("**Sample questions**")
    st.caption("Click one to send it instantly.")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []

# ── Main ─────────────────────────────────────────────────────────────────────
st.title("Customer Care Assistant")
st.caption(
    "Ask me anything about your mobile service — connectivity, billing, SIM, roaming, and more."
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Resolve question from chat input or sidebar button click
question = st.chat_input("Describe your issue…")
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    # append user turn to session history
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # build a simple chat_history transcript to pass to the RAG chain
    transcript_lines = []
    for m in st.session_state.messages:
        prefix = "User:" if m["role"] == "user" else "Assistant:"
        transcript_lines.append(f"{prefix} {m['content']}")
    chat_history = "\n".join(transcript_lines)

    # invoke chain with question + chat_history
    chain = get_chain()
    with st.spinner("Thinking..."):
        response = chain.invoke({"question": question, "chat_history": chat_history})

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
