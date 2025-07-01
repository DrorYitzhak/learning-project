import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from langchain_core.tools import BaseTool
from typing import ClassVar

# 🗂️ משתנים גלובליים
GLOBAL_LOADED_DATA = None
GLOBAL_SOURCE_SUMMARY = None

# פונקציות עזר לגלובלים
def get_loaded_data():
    return GLOBAL_LOADED_DATA

def get_data_summary():
    return GLOBAL_SOURCE_SUMMARY

# 🔧 כלי 1 – טעינה
class DataLoaderTool(BaseTool):
    name: ClassVar[str] = "data_loader_tool"
    description: ClassVar[str] = "Loads a CSV file or a ZIP file containing CSVs into memory. Supports nested folders inside ZIP."

    def _run(self, file_path: str) -> str:
        global GLOBAL_LOADED_DATA, GLOBAL_SOURCE_SUMMARY
        dfs_loaded = []

        file_path = file_path.strip()

        if not os.path.exists(file_path):
            return f"❌ הנתיב לא קיים: {file_path}"

        if file_path.lower().endswith(".csv"):
            try:
                df = pd.read_csv(file_path)
                GLOBAL_LOADED_DATA = df
                GLOBAL_SOURCE_SUMMARY = [(os.path.basename(file_path), len(df))]
                return f"✅ {os.path.basename(file_path)} loaded successfully."
            except Exception as e:
                return f"❌ Failed to load CSV: {str(e)}"

        elif file_path.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    csv_files = [f for f in zip_ref.namelist() if f.lower().endswith(".csv")]
                    if not csv_files:
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
                                return f"❌ שגיאה בטעינת {csv_name}: {str(e)}"

                    GLOBAL_LOADED_DATA = combined_df
                    GLOBAL_SOURCE_SUMMARY = summary
                    return f"✅ Loaded {len(summary)} CSV files from ZIP."

            except Exception as e:
                return f"❌ שגיאה בפתיחת קובץ ZIP: {str(e)}"

        else:
            return "❌ פורמט לא נתמך. יש לספק קובץ CSV או ZIP."

# 🔧 כלי 2 – סיכום כשלונות לפי DUT_SN
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

# 🔧 כלי 3 – קובץ על שדות שנכשלו
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

# 🔧 כלי 4 – סיכום כשלים
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

# 🔧 כלי בסיס לגרפים
class BaseChartTool(BaseTool):
    name: ClassVar[str] = "chart_base_tool"
    description: ClassVar[str] = "Generates a chart and returns a matplotlib Figure object."

    def _generate_chart(self, query: str) -> plt.Figure:
        raise NotImplementedError("Subclasses must implement _generate_chart method.")

    def _run(self, query: str) -> dict:
        fig = self._generate_chart(query)
        return {"output": fig}

# 🔧 גרף לדוגמה
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

# 🔧 גרף פארטו של כשלים לפי שם בדיקה
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

# 🔧 כלי לשאלות כלליות
class GeneralResponseTool(BaseTool):
    name: ClassVar[str] = "general_response_tool"
    description: ClassVar[str] = "כלי לשאלות כלליות כמו 'מי אתה' או 'שלום'."

    def _run(self, query: str) -> str:
        return query

# 🔧 הרשימה הכוללת של הכלים
TOOLS = [
    DataLoaderTool(),
    FailureCountPerUnitTool(),
    FailureQueryTool(),
    FailureSummaryTool(),
    FailureParetoChartTool(),
    DemoChartTool(),
    GeneralResponseTool(),
]
