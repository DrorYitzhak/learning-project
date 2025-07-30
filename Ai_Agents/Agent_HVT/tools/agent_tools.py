import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from langchain_core.tools import BaseTool
from typing import ClassVar
import uuid
import logging
import json


# 🔧 כלי חדש: סמול-טוק מבוסס LLM
from Ai_Agents.Agent_HVT.llm.llm_model import get_llm
from langchain_core.tools import BaseTool
from typing import ClassVar

# הגדרת לוג בסיסי
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# 🗂️ משתנים גלובליים
GLOBAL_LOADED_DATA = None
GLOBAL_SOURCE_SUMMARY = None
GLOBAL_DF_CACHE = {}

# פונקציות עזר לגלובלים
def get_loaded_data():
    return GLOBAL_LOADED_DATA

def get_data_summary():
    return GLOBAL_SOURCE_SUMMARY

def add_to_cache(df):
    df_id = str(uuid.uuid4())
    GLOBAL_DF_CACHE[df_id] = df
    return df_id

def get_from_cache(df_id):
    return GLOBAL_DF_CACHE.get(df_id)

# 🔧 כלי 1 – טעינה
class DataLoaderTool(BaseTool):
    name: ClassVar[str] = "data_loader_tool"
    description: ClassVar[str] = "Loads a CSV file or a ZIP file containing CSVs into memory. Supports nested folders inside ZIP."

    def _run(self, file_path: str) -> str:
        global GLOBAL_LOADED_DATA, GLOBAL_SOURCE_SUMMARY, GLOBAL_DF_CACHE
        dfs_loaded = []
        file_path = file_path.strip()
        if not os.path.exists(file_path):
            logging.error(f"נתיב לא קיים: {file_path}")
            return f"❌ הנתיב לא קיים: {file_path}"
        if file_path.lower().endswith(".csv"):
            try:
                df = pd.read_csv(file_path)
                GLOBAL_LOADED_DATA = df
                GLOBAL_SOURCE_SUMMARY = [(os.path.basename(file_path), len(df))]
                GLOBAL_DF_CACHE.clear()
                logging.info(f"{os.path.basename(file_path)} loaded successfully.")
                return f"✅ {os.path.basename(file_path)} loaded successfully."
            except Exception as e:
                logging.exception("Failed to load CSV")
                return f"❌ Failed to load CSV: {str(e)}"
        elif file_path.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    csv_files = [f for f in zip_ref.namelist() if f.lower().endswith(".csv")]
                    if not csv_files:
                        logging.warning("לא נמצאו קובצי CSV בתוך קובץ ה-ZIP.")
                        return "⚠️ לא נמצאו קובצי CSV בתוך קובץ ה-ZIP."
                    combined_df = pd.DataFrame()
                    summary = []
                    for csv_name in csv_files:
                        with zip_ref.open(csv_name) as f:
                            try:
                                df = pd.read_csv(f)
                                df["__source_file__"] = os.path.basename(csv_name)
                                combined_df = pd.concat([combined_df, df], ignore_index=True)
                                summary.append((os.path.basename(csv_name), len(df)))
                            except Exception as e:
                                logging.exception(f"שגיאה בטעינת {csv_name}")
                                return f"❌ שגיאה בטעינת {csv_name}: {str(e)}"
                    GLOBAL_LOADED_DATA = combined_df
                    GLOBAL_SOURCE_SUMMARY = summary
                    GLOBAL_DF_CACHE.clear()
                    logging.info(f"Loaded {len(summary)} CSV files from ZIP.")
                    return f"✅ Loaded {len(summary)} CSV files from ZIP."
            except Exception as e:
                logging.exception("שגיאה בפתיחת קובץ ZIP")
                return f"❌ שגיאה בפתיחת קובץ ZIP: {str(e)}"
        else:
            logging.error("פורמט לא נתמך")
            return "❌ פורמט לא נתמך. יש לספק קובץ CSV או ZIP."

# 🔧 כלי חדש: DataFrame Filter Tool
class DataFrameFilterTool(BaseTool):
    name: ClassVar[str] = "dataframe_filter_tool"
    description: ClassVar[str] = (
        "Filters the loaded DataFrame by flexible conditions (JSON format or DICT). "
        "Returns a short preview (head(5)) and a DataFrame-ID for next steps. "
        "Supports operators ==, !=, <, >, <=, >=, in, notin, contains, AND/OR, choose columns."
    )

    def _run(self, filter_json) -> dict:
        df = get_loaded_data()
        if df is None or len(df) == 0:
            logging.error("No data loaded yet.")
            return {"error": "No data loaded yet."}
        try:
            # אם קיבלנו string – ננסה ל-parse ל-dict
            if isinstance(filter_json, str):
                try:
                    # ננקה קידוד מיותר של json/code
                    clean = filter_json.strip()
                    if clean.startswith("```json"):
                        clean = clean.replace("```json", "").replace("```", "").strip()
                    elif clean.startswith("```python"):
                        clean = clean.replace("```python", "").replace("```", "").strip()
                    filter_json = json.loads(clean)
                except Exception as e:
                    logging.error(f"Could not parse JSON string: {filter_json}")
                    return {"error": f"Could not parse JSON string: {str(e)}"}
            if not isinstance(filter_json, dict):
                logging.error(f"filter_json is not a dict: {type(filter_json)}")
                return {"error": f"Input filter_json must be a dict, got {type(filter_json)}."}
            # Parse filter_json
            conditions = filter_json.get("conditions", [])
            logical = filter_json.get("logical", "AND").upper()
            columns_keep = filter_json.get("columns_keep")
            mask = pd.Series([True]*len(df))
            for cond in conditions:
                if not isinstance(cond, dict) or not all(k in cond for k in ["col", "op", "val"]):
                    logging.error(f"Bad condition: {cond}")
                    return {"error": f"Bad condition structure: {cond}"}
                col, op, val = cond["col"], cond["op"], cond["val"]
                if col not in df.columns:
                    logging.error(f"Missing column in DataFrame: {col}")
                    return {"error": f"Column not found: {col}"}
                if op == "==":
                    m = df[col] == val
                elif op == "!=":
                    m = df[col] != val
                elif op == "<":
                    m = df[col] < val
                elif op == ">":
                    m = df[col] > val
                elif op == "<=":
                    m = df[col] <= val
                elif op == ">=":
                    m = df[col] >= val
                elif op == "in":
                    m = df[col].isin(val)
                elif op == "notin":
                    m = ~df[col].isin(val)
                elif op == "contains":
                    m = df[col].astype(str).str.contains(str(val))
                else:
                    logging.error(f"Unsupported operator: {op}")
                    return {"error": f"Unsupported operator: {op}"}
                if logical == "AND":
                    mask = mask & m
                elif logical == "OR":
                    mask = mask | m
            filtered = df[mask]
            if columns_keep:
                missing = [c for c in columns_keep if c not in filtered.columns]
                if missing:
                    logging.warning(f"Requested columns not found: {missing}")
                    return {"error": f"Requested columns not found: {missing}"}
                filtered = filtered[columns_keep]
            df_id = add_to_cache(filtered)
            preview = filtered.head(5).to_dict(orient="records")
            info = {
                "num_rows": len(filtered),
                "columns": list(filtered.columns),
                "DataFrame_ID": df_id,
                "preview": preview
            }
            if len(filtered) == 0:
                logging.warning("Result DataFrame is empty. Check your filter.")
                info["warning"] = "Result DataFrame is empty. Check your filter."
            return info
        except Exception as e:
            logging.exception("Error in DataFrameFilterTool")
            return {"error": str(e)}

# 🔧 כלי 2 – סיכום כשלונות לפי DUT_SN (ללא שינוי)
class FailureCountPerUnitTool(BaseTool):
    name: ClassVar[str] = "failure_count_per_unit_tool"
    description: ClassVar[str] = (
        "סופר כמות כשלונות (Verdict_ATE == 0) לכל DUT_SN בכל הקבצים שטעונים ( תומך גם ב-ZIP )."
    )
    def _run(self, query: str) -> str:
        df = get_loaded_data()
        if df is None or len(df) == 0:
            return "👭 לא נטען עדיין קובץ נתונים."
        if "Verdict_ATE" not in df.columns or "DUT_SN" not in df.columns:
            return "⚠️ חסרות עמודות נדרשות ('Verdict_ATE', 'DUT_SN')."
        failed_counts = df[df["Verdict_ATE"] == 0].groupby("DUT_SN").size().reset_index(name="Failures")
        if failed_counts.empty:
            return "✅ לא נמצאו כשלונות."
        return f"📊 סיכום כשלונות לפי DUT_SN:\n\n{failed_counts.to_string(index=False)}"

# 🔧 כלי 3 – קובץ על שדות שנכשלו (ללא שינוי)
class FailureQueryTool(BaseTool):
    name: ClassVar[str] = "failure_query_tool"
    description: ClassVar[str] = "מחזיר ערכים מתוך השורות שנכשלו לפי שמות עמודות שהוזכרו בשאלה."
    def _run(self, query: str) -> str:
        df = get_loaded_data()
        if df is None:
            return "📬 לא נטען עדיין קובץ נתונים."
        if "Verdict_ATE" not in df.columns:
            return "❌ העמודה 'Verdict_ATE' לא קיימת."
        failed_df = df[df["Verdict_ATE"] == 0]
        if failed_df.empty:
            return "✅ אין שורות שנכשלו."
        requested_cols = [col for col in df.columns if col.lower() in query.lower()]
        if not requested_cols:
            return "⚠️ לא נמצאו עמודות תואמות בשאלה שלך."
        preview = failed_df[requested_cols].head(10).to_string(index=False)
        return f"📋 שורות שנכשלו ( ראשונות ):\n\n{preview}"

# 🔧 כלי 4 – סיכום כשלים (ללא שינוי)
class FailureSummaryTool(BaseTool):
    name: ClassVar[str] = "failure_summary_tool"
    description: ClassVar[str] = "מסכם את כל השורות שנכשלו כולל תדר, בדיקה, גבולות, ערך, שגיאה, ערוץ, צ'יפ ו‏PA."
    def _run(self, query: str) -> str:
        df = get_loaded_data()
        if df is None:
            return "📬 לא נטען עדיין קובץ נתונים."
        if "Verdict_ATE" not in df.columns:
            return "❌ העמודה 'Verdict_ATE' לא קיימת."
        failed_df = df[df["Verdict_ATE"] == 0]
        if failed_df.empty:
            return "✅ אין שורות שנכשלו."
        summaries = []
        for _, row in failed_df.iterrows():
            summary = (
                f"תדר: {row.get('LOM_Freq_Config_MHz', 'N/A')} MHz, "
                f"קבוצה: {row.get('Test_Group', 'N/A')}, בדיקה: {row.get('Test_Name', 'N/A')}, "
                f"Chip: {row.get('Chip_Type', 'N/A')}, Chip_Num: {row.get('Chip_Num', 'N/A')}, "
                f"Channel: {row.get('Channel', 'N/A')}, PA: {row.get('PA', 'N/A')}\n"
                f"תוצאה: {row.get('Result', 'N/A')} (גבולות: {row.get('Min_Limit_ATE', 'N/A')} – {row.get('Max_Limit_ATE', 'N/A')}), "
                f"שגיאה: {row.get('Error_Msg', 'אין')}\n"
            )
            summaries.append(summary)
        return f"נמצאו {len(failed_df)} שורות שנכשלו:\n\n" + "\n".join(summaries)

# 🔧 כלי בסיס לגרפים (ללא שינוי)
class BaseChartTool(BaseTool):
    name: ClassVar[str] = "chart_base_tool"
    description: ClassVar[str] = "Generates a chart and returns a matplotlib Figure object."
    def _generate_chart(self, query: str) -> plt.Figure:
        raise NotImplementedError("Subclasses must implement _generate_chart method.")
    def _run(self, query: str) -> dict:
        fig = self._generate_chart(query)
        return {"output": fig}

# 🔧 גרף לדוגמה (ללא שינוי)
class DemoChartTool(BaseChartTool):
    name: ClassVar[str] = "demo_chart_tool"
    description: ClassVar[str] = "Creates a simple demo line chart."
    def _generate_chart(self, query: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([1, 2, 3, 4], [10, 5, 8, 12], marker='o')
        ax.set_title("Demo Chart")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        return fig

# 🔧 גרף פארטו של כשלים לפי שם בדיקה (ללא שינוי)
class FailureParetoChartTool(BaseChartTool):
    name: ClassVar[str] = "failure_pareto_chart_tool"
    description: ClassVar[str] = "יוצר גרף פארטו של כשלים לפי שם הבדיקה (Test_Name)."
    def _generate_chart(self, query: str) -> plt.Figure:
        df = get_loaded_data()
        if df is None or "Verdict_ATE" not in df.columns or "Test_Name" not in df.columns:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "📬 Data not available or missing columns", ha='center', va='center')
            return fig
        df_failed = df[df["Verdict_ATE"] == 0]
        failures_by_test = df_failed.groupby("Test_Name").size().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        failures_by_test.plot(kind="bar", ax=ax)
        ax.set_title("Pareto Chart of Failures by Test")
        ax.set_xlabel("Test Name")
        ax.set_ylabel("Number of Failures")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        return fig

# 🔧 כלי לשאלות כלליות (ללא שינוי)
class GeneralResponseTool(BaseTool):
    name: ClassVar[str] = "general_response_tool"
    description: ClassVar[str] = "כלי לשאלות כלליות כמו 'מי אתה' או 'שלום'."
    def _run(self, query: str) -> str:
        return query


class SmallTalkLLMTool(BaseTool):
    name: ClassVar[str] = "small_talk_llm"
    description: ClassVar[str] = (
        "Answer casual, open-ended, and small-talk questions using a general LLM. "
        "This tool is only for greetings, ice-breakers, jokes, small talk, and friendly communication unrelated to HVT data."
    )
    def _run(self, query: str, chat_history=None) -> str:
        llm = get_llm()
        # אם אין היסטוריה פשוט הגדר כ-ריק
        if chat_history is None:
            chat_history = ""
        prompt = (
            "Always reply in the same language as the user's message. "
            "If there is no previous conversation, or this is the first message, "
            "introduce yourself briefly as an assistant for radar test data analysis. "
            "When introducing yourself in the first message, do not ask for files or data, just say you are the radar test data analysis assistant and the user is welcome to ask or upload anything. "
            "Do not expand or explain more than that. "
            "If there is previous conversation, always take into account the chat history "
            "and reply concisely and contextually – answer based on the ongoing conversation. "
            "If the user's message is unrelated to radar test data, CSV files, or analysis, "
            "gently encourage them to ask about radar data or test results. "
            "Do not give long or off-topic answers. "
            "If the user is confused, help them get started by explaining that you assist with radar test data. "
            "Be polite and positive. "
            "\n\nPrevious conversation: {chat_history}\n"
            "User: {query}\n"
            "AI:"
        )

        response = llm.invoke(prompt)
        return response



# 🔧 הרשימה הכוללת של הכלים
TOOLS = [
    DataLoaderTool(),
    DataFrameFilterTool(),
    FailureCountPerUnitTool(),
    FailureQueryTool(),
    FailureSummaryTool(),
    FailureParetoChartTool(),
    DemoChartTool(),
    GeneralResponseTool(),
    SmallTalkLLMTool(),  # <---- הוספת הכלי החדש
]
