import sys
import zipfile
import os
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPlainTextEdit, QPushButton, QLabel,
    QHBoxLayout, QSizePolicy, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QKeyEvent, QTextOption

# ✅ נוספה קריאה לסוכן בזמן טעינה
from agent_runner import ask_agent

def summarize_failures(data):
    if data is None:
        return "No data loaded. Please load a file with CSV(s) first."

    summary = []
    required_columns = {'DUT_SN', 'Sys_Type', 'Freq', 'Verdict', 'Test_Name'}
    if not required_columns.issubset(data.columns):
        return "Missing required columns in data."

    grouped = data[data['Verdict'] != 1].groupby(['DUT_SN', 'Sys_Type'])

    for (dut, system), group in grouped:
        tests = group['Test_Name'].unique().tolist()
        freqs = group['Freq'].unique().tolist()
        date = group['source_file'].iloc[0] if 'source_file' in group else 'unknown'
        summary.append({
            'Unit': dut,
            'System': system,
            'Failed_Tests': tests,
            'Failed_Freqs': freqs,
            'Source_File': date
        })

    if not summary:
        return "No failures found in the dataset."

    df_summary = pd.DataFrame(summary)
    return df_summary.to_markdown(index=False)


class AgentThread(QThread):
    result_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, question, data=None):
        super().__init__()
        self.question = question
        self.data = data

    def run(self):
        try:
            if "summarize failures" in self.question.lower():
                answer = summarize_failures(self.data)
            else:
                answer = ask_agent(self.question, self.data)
            self.result_ready.emit(str(answer))
        except Exception as e:
            self.error_occurred.emit(str(e))


class AgentChatGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Agent Chat")
        self.resize(1000, 720)
        self.loaded_data = None
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("background-color: #202020; color: #ffffff;")

        layout = QVBoxLayout()

        title = QLabel("🤖 Agent - HVT Data")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setStyleSheet("color: #ffffff; margin-bottom: 12px;")
        layout.addWidget(title)

        self.chat_display = QPlainTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Segoe UI", 10))
        self.chat_display.setStyleSheet("background-color: #282828; color: #ffffff; padding: 12px; border-radius: 8px; border: 1px solid #333;")
        layout.addWidget(self.chat_display)

        input_layout = QHBoxLayout()

        self.input_field = QPlainTextEdit()
        self.input_field.setPlaceholderText("Type your question here...")
        self.input_field.setFont(QFont("Segoe UI", 10))
        self.input_field.setStyleSheet("background-color: #2d2d2d; color: white; padding: 8px; border-radius: 6px; border: 1px solid #666;")
        self.input_field.setFixedHeight(80)
        self.input_field.installEventFilter(self)
        self.input_field.setLayoutDirection(Qt.LeftToRight)
        self.input_field.document().setDefaultTextOption(QTextOption(Qt.AlignLeft))

        input_button_layout = QVBoxLayout()

        self.send_button = QPushButton("Send")
        self.send_button.setFont(QFont("Segoe UI", 10))
        self.send_button.setStyleSheet("background-color: #26c6da; color: white; padding: 8px 16px; border-radius: 5px;")
        self.send_button.clicked.connect(self.handle_query)

        self.clear_button = QPushButton("Clear")
        self.clear_button.setFont(QFont("Segoe UI", 10))
        self.clear_button.setStyleSheet("background-color: #666; color: white; padding: 8px 16px; border-radius: 5px;")
        self.clear_button.clicked.connect(self.chat_display.clear)

        button_row_layout = QHBoxLayout()
        button_row_layout.addWidget(self.send_button)
        button_row_layout.addWidget(self.clear_button)

        self.load_file_button = QPushButton("Add Files")
        self.load_file_button.setFont(QFont("Segoe UI", 10))
        self.load_file_button.setStyleSheet("background-color: #8bc34a; color: white; padding: 8px 16px; border-radius: 5px;")
        self.load_file_button.setFixedSize(172, 36)
        self.load_file_button.clicked.connect(self.load_data_dialog)

        input_button_layout.addLayout(button_row_layout)
        input_button_layout.addWidget(self.load_file_button)
        input_layout.addWidget(self.input_field)
        input_layout.addLayout(input_button_layout)

        layout.addLayout(input_layout)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #aaaaaa; padding: 4px;")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def handle_query(self):
        question = self.input_field.toPlainText().strip()
        if not question:
            return

        self.chat_display.appendPlainText(f"You: {question}\n")
        self.status_label.setText("Thinking...")
        self.input_field.clear()

        self.thread = AgentThread(question, data=self.loaded_data)
        self.thread.result_ready.connect(self.display_answer)
        self.thread.error_occurred.connect(self.display_error)
        self.thread.start()

    def display_answer(self, answer):
        self.chat_display.appendPlainText(f"Agent: {answer}\n")
        self.status_label.setText("Ready")

    def display_error(self, error):
        self.chat_display.appendPlainText(f"⚠️ Error: {error}\n")
        self.status_label.setText("Ready")

    def load_data_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select ZIP or CSV File", "", "ZIP or CSV Files (*.zip *.csv)")

        if not path:
            QMessageBox.information(self, "Info", "No file selected.")
            return

        try:
            # שולח בקשת טעינה לסוכן
            self.chat_display.appendPlainText(f"You: טען את הקובץ {path}\n")
            self.status_label.setText("Loading...")

            answer = ask_agent(f"טען את הקובץ {path}")
            self.chat_display.appendPlainText(f"Agent: {answer}\n")
            self.status_label.setText("Ready")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while loading file:\n{str(e)}")

    def eventFilter(self, source, event):
        if source == self.input_field and event.type() == event.KeyPress:
            if event.key() == Qt.Key_Return and not (event.modifiers() & Qt.ShiftModifier):
                self.handle_query()
                return True
        return super().eventFilter(source, event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AgentChatGUI()
    window.show()
    sys.exit(app.exec_())
