# --- 🧠 סוכן להשוואת קבצי אפליקציה והפקת סיכום שינויים ---
# שלב ראשון: קריאת קובץ Word והשוואת תוכן בין גרסאות, שלב שני: סיכום ההבדלים בעזרת GPT

from docx import Document
import difflib
import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# ✅ טעינת משתני סביבה מתוך קובץ .env
load_dotenv()

# === שלב 1: קריאת תוכן מקובץ Word ===
def read_docx_text(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

# === שלב 2: השוואת טקסטים ===
def compare_texts(text1, text2):
    diff = difflib.unified_diff(
        text1.splitlines(),
        text2.splitlines(),
        lineterm="",
        fromfile="old_version",
        tofile="new_version"
    )
    return "\n".join(diff)

# === שלב 3: הפקת סיכום בעזרת GPT ===
def summarize_with_gpt(diff_text):
    prompt = PromptTemplate.from_template("""
    אתה מקבל רשימת הבדלים בין גרסה ישנה לחדשה של מפרט אפליקציה. 
    אנא סכם את השינויים העיקריים שנעשו במסמך, והסבר בקצרה איך הם משפרים את חוויית המשתמש או את הפונקציונליות של האפליקציה.

    הבדלים:
    {diff}
    """)
    llm = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), temperature=0)
    full_prompt = prompt.format(diff=diff_text)
    return llm.invoke(full_prompt)

# === שלב 4: שמירת סיכום לקובץ חדש ===
def save_summary(summary_text):
    today = datetime.date.today().isoformat()
    filename = f"summary_{today}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"✅ הסיכום נשמר בקובץ: {filename}")

# === שלב 5: הפעלת כל התהליך ===
if __name__ == "__main__":
    old_path = r"C:\Users\drory\OneDrive - Mobileye\Desktop\for Dror\app_docs\מפרט אפליקציית חיבור לקוחות-ישן.docx"
    new_path = r"C:\Users\drory\OneDrive - Mobileye\Desktop\for Dror\app_docs\מפרט אפליקציית חיבור לקוחות-חדש.docx"

    print("📅 טוען גרסה ישנה...")
    old_text = read_docx_text(old_path)

    print("📅 טוען גרסה חדשה...")
    new_text = read_docx_text(new_path)

    print("🔍 משווה בין הגרסאות...")
    diff_text = compare_texts(old_text, new_text)

    if not diff_text.strip():
        print("✅ לא זוהו שינויים בין הקבצים.")
    else:
        print("🧐 הבדלים זוהו – שולח ל-GPT לסיכום...")
        with open("diff.txt", "w", encoding="utf-8") as f:
            f.write(diff_text)
        print("📂 נשמר קובץ diff.txt עם ההבדלים.")

        summary = summarize_with_gpt(diff_text)
        save_summary(summary)
