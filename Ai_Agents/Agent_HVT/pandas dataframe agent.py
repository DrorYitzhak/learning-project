"""
LangChain Pandas Agent using Google Gemini
-------------------------------------------
Features:
1. Loads a CSV file into a Pandas DataFrame.
2. Cleans important columns for consistent analysis.
3. Creates an agent that answers questions about the data in natural language.
4. Uses Google Gemini (via langchain-google-genai).
"""

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY is missing. Please set it in your .env file or environment variables.")

# ✅ Step 1: Load CSV
csv_path = r"C:\Users\drory\Downloads\agent_data.csv"
df = pd.read_csv(csv_path)
print("✅ CSV Loaded Successfully!")
print(f"Shape: {df.shape}")
print("Columns:", df.columns.tolist())

# ✅ Step 2: Create LLM (Google Gemini)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)

# ✅ Step 3: Create Pandas Agent
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    allow_dangerous_code=True,  # Allows Python execution
    include_df_in_prompt=True,  # Helps the model understand data
    number_of_head_rows=10      # Sends first 10 rows for context
)

# ✅ Step 4: Interactive Q&A
print("\n🤖 Agent is ready! Ask questions about your data (type 'exit' to quit):\n")

while True:
    query = input("Your question: ")
    if query.lower() == "exit":
        print("👋 Goodbye!")
        break
    try:
        result = agent.invoke({"input": query})
        print(f"\n🔍 Answer:\n{result['output']}\n")
    except Exception as e:
        print(f"⚠️ Error: {e}")
