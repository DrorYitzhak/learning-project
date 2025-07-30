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

if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.title("🤖 Agent - HVT Data (Chat Mode)")

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

# הצגת כל ההודעות
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], pd.DataFrame):
            st.dataframe(msg["content"])
        elif isinstance(msg["content"], Figure):
            try:
                st.plotly_chart(mpl_to_plotly(msg["content"]), use_container_width=True)
            except Exception:
                st.pyplot(msg["content"])
        else:
            st.markdown(msg["content"])

# קלט מהמשתמש (קלט חדש של Streamlit)
user_prompt = st.chat_input("Type your question and press Enter...")

if user_prompt:
    # 1. מציגים מיידית את הודעת המשתמש (יופיע מייד בצ'אט!)
    st.session_state["messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. ממתינים לסוכן (הודעת "הסוכן חושב" תוצג בינתיים)
    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            answer = ask_agent(user_prompt, st.session_state.get("data", None))
            # תשובת הסוכן - מציגים בכל פורמט
            if isinstance(answer, pd.DataFrame):
                st.dataframe(answer)
            elif isinstance(answer, Figure):
                try:
                    st.plotly_chart(mpl_to_plotly(answer), use_container_width=True)
                except Exception:
                    st.pyplot(answer)
            else:
                st.markdown(answer)
    # מוסיפים את תשובת הסוכן להיסטוריה
    st.session_state["messages"].append({"role": "assistant", "content": answer})

