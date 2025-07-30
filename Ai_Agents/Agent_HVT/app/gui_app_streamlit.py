import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
from matplotlib.figure import Figure
from plotly.tools import mpl_to_plotly
from Ai_Agents.Agent_HVT.agent_runner import ask_agent

st.set_page_config(page_title="Agent - HVT Data", layout="wide")
st.markdown("""
<style>
body {background-color:#202020;color:white;}
.block-container {background-color:#202020;color:#fff;min-height:100vh;padding-bottom:40px;}
.chat-history {padding:10px;border-radius:8px;}
[data-testid="stSidebar"] {background-color:#181818;}
.stTextInput input {background-color:#2d2d2d;color:white;border-radius:8px;border:1px solid #666;padding:8px;}
.stButton>button {font-size:16px;font-weight:bold;border-radius:6px;padding:8px 16px;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Agent - HVT Data")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "data" not in st.session_state:
    st.session_state.data = None
if "last_sent_question" not in st.session_state:
    st.session_state.last_sent_question = ""
if "reset_user_question" not in st.session_state:
    st.session_state.reset_user_question = False

# איפוס השדה בתחילת ריצה לפי דגל (רק פעם אחת אחרי שליחה)
if st.session_state.reset_user_question:
    st.session_state.user_question = ""
    st.session_state.reset_user_question = False

# Sidebar
with st.sidebar:
    st.header("📂 Add files")
    uploaded_file = st.file_uploader("Upload ZIP or CSV", type=["zip", "csv"])
    if uploaded_file:
        st.info(f"Processing file: {uploaded_file.name}")
        with st.spinner("Loading file..."):
            try:
                resp = ask_agent(f"Load file {uploaded_file.name}")
                st.session_state.data = resp if isinstance(resp, pd.DataFrame) else None
                st.success("✅ Loaded successfully")
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")

    st.markdown("---")
    st.subheader("💬 Ask the agent")
    user_question = st.text_input(
        "Type your question and press Send",
        value=st.session_state.get("user_question", ""),
        key="user_question"
    )
    col_send, col_clear = st.columns(2)
    send = col_send.button("Send")
    clear = col_clear.button("Clear history")

# Clear history button
if clear:
    st.session_state.chat_history.clear()
    st.session_state.last_sent_question = ""
    st.session_state.user_question = ""
    st.rerun()

# Send question (handles both button and Enter)
if send or (user_question and st.session_state.last_sent_question != user_question):
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        st.session_state.chat_history.append(("You", user_question))
        with st.spinner("Agent is thinking..."):
            try:
                answer = ask_agent(user_question, st.session_state.data)
            except Exception as e:
                answer = f"Error: {e}"
        st.session_state.chat_history.append(("Agent", answer))
        st.session_state.last_sent_question = user_question
        st.session_state.reset_user_question = True  # הדגל לאיפוס
        st.rerun()

# ---------- כאן מתחיל הקטע שמארגן את ההודעות כ"זוגות" ----------------

def get_message_pairs(chat_history):
    pairs = []
    temp = []
    for item in chat_history:
        temp.append(item)
        if len(temp) == 2:
            pairs.append(temp)
            temp = []
    if temp:
        pairs.append(temp)
    return list(reversed(pairs))

st.markdown("<div class='chat-history'>", unsafe_allow_html=True)
for pair in get_message_pairs(st.session_state.chat_history):
    for role, content in pair:
        st.markdown(f"**{role}:**")
        if isinstance(content, pd.DataFrame):
            st.dataframe(content)
        elif isinstance(content, Figure):
            try:
                st.plotly_chart(mpl_to_plotly(content), use_container_width=True)
            except Exception:
                st.pyplot(content)
        else:
            st.markdown(f"{content}")
    st.markdown("---")
st.markdown("</div>", unsafe_allow_html=True)

