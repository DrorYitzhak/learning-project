# agent_gemini_csv.py

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
from dotenv import load_dotenv
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY is missing. Please set it in your .env file or environment variables.")

# ✅ Step 1: Load CSV
csv_path = r"C:\Users\drory\Downloads\HVT_BPU_BSRc_fail.csv"
df = pd.read_csv(csv_path)
# Mapping
column_types = {
    "DUT_SN": "int",
    "Sys_Type": "string",
    "Test_Group": "string",
    "Test_Name": "string",
    "Board_SN": "int",
    "BPU_SN": "int",
    "Chip_Type": "string",
    "Chip_Num": "int",
    "Channel": "int",
    "PA": "int",
    "Rx_Num": "int",
    "Tx_Num": "int",
    "Result": "float",
    "Units": "string",
    "Min_Limit_ATE": "float",
    "Max_Limit_ATE": "float",
    "Verdict_ATE": "int",
    "Error_Msg": "string",
    "LOM_Freq_Config_MHz": "float"
}

# Cleaning
for col, col_type in column_types.items():
    if col in df.columns:
        if col_type == "string":
            df[col] = df[col].astype(str).str.strip().str.upper()
        elif col_type == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif col_type == "int":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
print("✅ CSV Loaded Successfully!")
print(f"Shape: {df.shape}")
print("Columns:", df.columns.tolist())


# ✅ Step 3: Create LLM (Google Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY)

# ✅ Step 4: Create Pandas Agent
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True  # Required for executing Python
)

# ✅ Step 5: Interactive Q&A Loop
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
