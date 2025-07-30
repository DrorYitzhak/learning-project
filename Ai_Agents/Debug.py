import streamlit as st
import pandas as pd
import plotly.express as px
from Ai_Agents.Agent_HVT.agent_runner import ask_agent

# כותרת ראשית
st.set_page_config(page_title="Agent - HVT Data", layout="wide")
st.title("🤖 Agent - HVT Data Analysis")

# משתנים גלובליים
if 'data' not in st.session_state:
    st.session_state['data'] = None

# אזור העלאת קובץ
st.sidebar.header("📂 העלאת קובץ")
uploaded_file = st.sidebar.file_uploader("בחר קובץ ZIP או CSV", type=["zip", "csv"])

if uploaded_file is not None:
    st.write(f"**קובץ נטען:** {uploaded_file.name}")
    # שמור או טפל בקובץ
    ask_agent(f"טען את הקובץ {uploaded_file.name}")
    st.success("✅ הקובץ נטען בהצלחה")

# שדה קלט לשאלה
st.subheader("שאל את הסוכן")
user_question = st.text_input("הקלד שאלה:")

if st.button("שלח שאלה"):
    if not user_question.strip():
        st.warning("❗ כתוב שאלה לפני שליחה")
    else:
        st.write(f"**אתה:** {user_question}")
        with st.spinner("הסוכן חושב..."):
            answer = ask_agent(user_question, st.session_state['data'])
        st.success("תשובת הסוכן:")

        # אם התשובה DataFrame - נציג טבלה
        if isinstance(answer, pd.DataFrame):
            st.dataframe(answer)

            # תן אופציה להציג גרף אם יש עמודות מתאימות
            numeric_cols = answer.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) >= 1:
                st.subheader("📊 גרף אינטראקטיבי")
                x_axis = st.selectbox("בחר עמודה ל-X", answer.columns)
                y_axis = st.selectbox("בחר עמודה ל-Y", numeric_cols)
                fig = px.scatter(answer, x=x_axis, y=y_axis, color=x_axis, title="גרף אינטראקטיבי")
                st.plotly_chart(fig, use_container_width=True)

        # אם התשובה טקסט - נציג כטקסט
        else:
            st.write(answer)
