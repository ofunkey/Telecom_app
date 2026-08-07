import os
import streamlit as st
from dotenv import load_dotenv

# 1. Page config MUST be the first Streamlit command
st.set_page_config(
    page_title="Telecom Support Chat",
    page_icon="📡",
    layout="centered",
)

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
load_dotenv()

# Check secrets / env first
if "GROQ_API_KEY" not in os.environ and "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

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

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 Telecom Support")
    st.caption("Powered by RAG · Qwen3.6-27B on Groq")
    st.divider()

    # Always render the key input in the sidebar if missing
    if not os.environ.get("GROQ_API_KEY"):
        user_key = st.text_input("Enter Groq API Key", type="password")
        if user_key:
            os.environ["GROQ_API_KEY"] = user_key
            st.success("Key applied!")
            st.rerun()
        else:
            st.warning("⚠️ Please enter a Groq API key to activate the bot.")

    st.markdown("**Sample questions**")
    st.caption("Click one to send it instantly.")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []


# Safe chain loader function
def get_chain():
    if not os.environ.get("GROQ_API_KEY"):
        st.error("🔑 Groq API Key is missing! Please enter your key in the sidebar.")
        st.stop()

    # Deferred import prevents initializing Groq before key exists
    from rag_chain import build_chain

    return build_chain()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

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
    # Ensure key exists BEFORE running any query logic
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Please enter a Groq API Key in the sidebar before asking questions.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    transcript_lines = []
    for m in st.session_state.messages:
        prefix = "User:" if m["role"] == "user" else "Assistant:"
        transcript_lines.append(f"{prefix} {m['content']}")
    chat_history = "\n".join(transcript_lines)

    chain = get_chain()
    with st.spinner("Thinking..."):
        response = chain.invoke({"question": question, "chat_history": chat_history})

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
