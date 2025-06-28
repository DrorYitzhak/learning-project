import pandas as pd
import os
from llm.llm_model import get_llm

def ask_smart_question(df: pd.DataFrame, question: str) -> str:
    """
    מקבל DataFrame ושאלה, ובונה קונטקסט חכם כדי לשאול את המודל ולהחזיר תשובה טקסטואלית.
    """
    df_head = df.head(15).to_string(index=False)
    columns = df.columns.tolist()
    types = df.dtypes.astype(str).to_dict()
    stats = df.describe().to_string()

    prompt = f"""
    אתה עוזר חכם שמנתח טבלאות בדיקות ייצור.
    טבלה לדוגמה (15 שורות ראשונות):
    {df_head}

    שמות העמודות:
    {columns}

    סוגי העמודות:
    {types}

    סטטיסטיקה כללית:
    {stats}

    כעת ענה בצורה מקצועית, מדויקת וברורה על השאלה הבאה:
    {question}
    """

    return get_llm().invoke(prompt)

if __name__ == "__main__":
    # מחשב נתיב מוחלט אל קובץ ה-CSV מתוך תיקיית Agent_HVT/data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "Agent_HVT", "data", "HVT_Production_Results_4432102105000898_2025-05-11--19-09-55.csv")

    # טען את הקובץ
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"הקובץ לא נמצא בנתיב: {csv_path}")

    df = pd.read_csv(csv_path)

    # שאלת דוגמה (אפשר להחליף)
    question = "באיזה מבחנים יש נפילות ומה ערך הנפילה ?"
    answer = ask_smart_question(df, question)

    print("\n🔎 תשובת המנוע החכם:")
    print(answer)
