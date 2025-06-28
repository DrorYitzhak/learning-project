from langchain.memory import ConversationBufferMemory
import os
def get_memory():
    """
    מחזיר אובייקט memory שמאפשר לסוכן לזכור שיחות קודמות.
    זה מאפשר לשאול שאלה אחת (למשל שם קובץ) ואז להמשיך לשאול שאלות על הקובץ בלי לחזור על המידע.
    """
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        input_key="input"
    )
    return memory
