import os
import json
import ollama

from typing import List, Dict
from prompts import SUMMARY_PROMPT

from utils import (
    count_tokens,
    split_text_into_chunks,
    new_id,
    now,
    MAX_TOKENS
)


CHAT_DIR = "chats"
os.makedirs(CHAT_DIR, exist_ok=True)


class Chat:
    def __init__(self, chat_id: str | None = None):
        self.chat_id = chat_id or new_id("chat")
        self.file_path = os.path.join(CHAT_DIR, f"{self.chat_id}.json")

        self.messages: List[Dict] = []
        self.highlights: str = ""
        self.summary_tokens: int = 0
        self.last_summarized: int = 0
        self.total_tokens = 0
        self.title = "New chat"   # ✅ NEW

        if os.path.exists(self.file_path):
            self._load()
        else:
            self._persist()

    # ---------- persistence ----------

    def _persist(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "chat_id": self.chat_id,
                    "title": self.title,   # ✅ NEW
                    "messages": self.messages,
                    "highlights": self.highlights,
                    "last_summarized": self.last_summarized,
                    "summary_tokens": self.summary_tokens,
                    "total_tokens": self.total_tokens,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _load(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.messages = data["messages"]
        self.highlights = data.get("highlights", "")
        self.last_summarized = data.get("last_summarized, 0")
        self.summary_tokens = data.get("summary_tokens", 0)
        self.total_tokens = data.get("total_tokens", 0)
        self.title = data.get("title", "New chat")  # ✅ NEW

    def set_title(self, new_title: str):   # ✅ NEW
        self.title = new_title.strip() if new_title else "New chat"
        self._persist()

    # ---------- summarization ----------

    def summarize_messages(self, messages: List[Dict]) -> str:
        summary_lines = []
        for m in messages:
            role = m["role"]
            text = m["content"]
            summary_lines.append(f"{role}: {text}")
        
        history = "\n".join(summary_lines)

        if self.highlights:
            history = "Previous Conversations Summary:\n" + self.highlights + "-"*30 + "\nNew Messages:"
            history += "\n".join(summary_lines)

        summary = ollama.chat(model='gemma3:4b',
                            messages=[{'role': 'system', 'content': SUMMARY_PROMPT},
                                    {'role': 'user', 'content': history}]
                            ).message.content

        return summary

    # ---------- message handling ----------

    def add_message(
        self,
        role: str,
        content: str,
        references: List[Dict] | None = None,
    ):
        references = references or []

        chunks = split_text_into_chunks(content)
        for i, chunk in enumerate(chunks):
            msg = {
                "id": new_id("msg"),
                "role": role,
                "content": chunk,
                "chunk_index": i,
                "n_chunks": len(chunks),
                "references": references if role == "assistant" else [],
                "created_at": now(),
                "tokens": count_tokens(chunk),
            }

            self.messages.append(msg)
            self.total_tokens += msg["tokens"]
            self.summary_tokens += msg["tokens"]

        if self.summary_tokens > MAX_TOKENS:
            self.highlights = self.summarize_messages(self.messages[self.last_summarized:])
            self.last_summarized = len(self.messages)
            self.summary_tokens = count_tokens(self.highlights)

        self._persist()

    # ---------- context for LLM ----------

    def context_for_llm(self) -> str:
        if self.messages:
            history =  "\n".join(
                f"{m['role']}: {m['content']}" for m in self.messages
            )
            return "Conversation history:\n" + history
        else:
            return f"Conversation summary:\n{self.highlights}"


# ✅ NEW: load all saved chats
def load_all_chats():
    chats = {}

    for file in os.listdir(CHAT_DIR):
        if file.endswith(".json"):
            chat_id = file.replace(".json", "")
            chat = Chat(chat_id)
            chats[chat_id] = chat

    return chats