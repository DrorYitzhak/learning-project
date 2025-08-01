import sys
import os
import tempfile
import shutil

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
from matplotlib.figure import Figure
from plotly.tools import mpl_to_plotly
import plotly.graph_objs as go  # plotly
from Ai_Agents.Agent_HVT.agent_runner import ask_agent

st.set_page_config(page_title="Agent - HVT Data", layout="wide")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.title("🤖 Agent - HVT Data (Chat Mode)")

def safe_remove(path):
    """מחיקה בטוחה של קובץ או תיקיה זמנית"""
    try:
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    except Exception as e:
        st.warning(f"⚠️ Could not delete temporary file: {e}")

def display_content(content):
    """פונקציה אוניברסלית להצגת כל סוג תשובה בצ'אט"""
    # (1) אם זו רשימה – מציג כל פריט בנפרד
    if isinstance(content, list):
        for item in content:
            display_content(item)
        return
    # (2) dict עם output (שכיח ב־LangChain tools)
    if isinstance(content, dict) and "output" in content:
        display_content(content["output"])
        return
    # (3) plotly
    if isinstance(content, go.Figure):
        st.plotly_chart(content, use_container_width=True)
        return
    # (4) matplotlib
    if isinstance(content, Figure):
        try:
            st.plotly_chart(mpl_to_plotly(content), use_container_width=True)
        except Exception:
            st.pyplot(content)
        return
    # (5) DataFrame
    if isinstance(content, pd.DataFrame):
        st.dataframe(content)
        return
    # (6) טקסט
    st.markdown(str(content))

with st.sidebar:
    st.header("📂 Add files")
    uploaded_file = st.file_uploader("Upload ZIP or CSV", type=["zip", "csv"])
    if uploaded_file:
        st.info(f"Processing file: {uploaded_file.name}")
        with st.spinner("Loading file..."):
            tmp_path = None
            try:
                ext = os.path.splitext(uploaded_file.name)[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name
                # שולח לסוכן רק את הנתיב הזמני (ללא תלות בשם המקורי)
                resp = ask_agent(f"Load file {tmp_path}")
                st.session_state.data = resp if isinstance(resp, pd.DataFrame) else None
                st.success("✅ Loaded successfully")
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    safe_remove(tmp_path)

# הצגת כל ההודעות מההיסטוריה (כולל גרפים)
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        display_content(msg["content"])

# קלט מהמשתמש (צ'אט)
user_prompt = st.chat_input("Type your question and press Enter...")

if user_prompt:
    # שלב 1: הצגת הודעת משתמש מיידית
    st.session_state["messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # שלב 2: קבלת תשובה מהסוכן + debug print
    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            answer = ask_agent(user_prompt, st.session_state.get("data", None))
            # DEBUG – חשוב מאוד!
            print("=== ANSWER TYPE:", type(answer), "\n=== ANSWER VALUE:", answer)
            display_content(answer)
    # שלב 3: שמירה להיסטוריה – לשימור כל סוג פלט (כולל dict/figure!)
    st.session_state["messages"].append({"role": "assistant", "content": answer})

