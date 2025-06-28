import os
from dotenv import load_dotenv
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
    )

if __name__ == "__main__":
    llm = get_llm()

    # טמפלט פשוט עם משתנה אחד {question}
    simple_template = PromptTemplate.from_template(
        "ענה בעברית בקצרה: {question}"
    )
    prompt_text = simple_template.format(question="מה זה GPT?")
    response = llm.invoke(prompt_text)
    print("🔁 תשובה מטמפלט פשוט:")
    print(response)

    print("\n" + "-"*40 + "\n")

    # טמפלט מפורט יותר עם הוראות מפורטות ושאלת המשתמש
    detailed_template = PromptTemplate.from_template(
        "אתה עוזר טכני.\n"
        "ענה בעברית בצורה ברורה ומסודרת.\n"
        "שאלה: {question}\n"
        "אם אפשר, הוסף דוגמאות קוד."
    )
    prompt_text_detailed = detailed_template.format(question="איך עובד למידת מכונה?")
    response_detailed = llm.invoke(prompt_text_detailed)
    print("🔁 תשובה מטמפלט מפורט:")
    print(response_detailed)
