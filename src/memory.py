"""
CS4241 — Introduction to Artificial Intelligence
Student: Farima Konaré | Index: 10012200004
Module: memory.py — Conversation memory for multi-turn RAG (Part G).
"""

from typing import List, Dict
from dataclasses import dataclass


MAX_TURNS = 5


@dataclass
class Turn:
    user: str
    assistant: str


class ConversationMemory:
    """Stores the last MAX_TURNS Q&A pairs; evicts oldest when at capacity."""

    def __init__(self, max_turns: int = MAX_TURNS):
        self.max_turns = max_turns
        self.history: List[Turn] = []

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        self.history.append(Turn(user=user_message, assistant=assistant_message))
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def format_for_prompt(self) -> str:
        if not self.history:
            return ""
        lines = []
        for i, turn in enumerate(self.history, start=1):
            lines.append(f"Q{i}: {turn.user}")
            lines.append(f"A{i}: {turn.assistant}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.history = []

    def __len__(self) -> int:
        return len(self.history)

    def to_list(self) -> List[Dict[str, str]]:
        return [{"user": t.user, "assistant": t.assistant} for t in self.history]
