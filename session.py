"""
会话管理模块。
对应 OpenClaw: src/config/sessions/store.ts
"""

import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Message:
    role: str          # user | assistant | system | tool
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict:
        d: dict = {"role": self.role, "content": self.content, "timestamp": self.timestamp}
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d

    @staticmethod
    def from_dict(d: dict) -> "Message":
        return Message(
            role=d["role"], content=d["content"],
            timestamp=d.get("timestamp", 0),
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
        )


@dataclass
class Session:
    session_id: str
    messages: list[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(d: dict) -> "Session":
        return Session(
            session_id=d["session_id"],
            messages=[Message.from_dict(m) for m in d.get("messages", [])],
            created_at=d.get("created_at", 0),
            updated_at=d.get("updated_at", 0),
        )


class SessionStore:
    """会话持久化存储。"""

    def __init__(self, store_path: str):
        self.store_path = Path(store_path).expanduser()
        self.sessions: dict[str, Session] = {}
        self._load()

    def _load(self):
        if self.store_path.exists():
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for key, data in raw.items():
                    self.sessions[key] = Session.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                self.sessions = {}

    def _save(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: s.to_dict() for k, s in self.sessions.items()}
        # 写入临时文件后原子替换，避免写入中途崩溃导致数据损坏
        tmp_path = self.store_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp_path.replace(self.store_path)

    def get_or_create(self, session_id: str) -> Session:
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(session_id=session_id)
        return self.sessions[session_id]

    def add_message(self, session_id: str, message: Message):
        session = self.get_or_create(session_id)
        session.messages.append(message)
        session.updated_at = time.time()
        self._save()

    def get_history(self, session_id: str, limit: int = 50) -> list[Message]:
        session = self.get_or_create(session_id)
        if limit > 0 and len(session.messages) > limit:
            return list(session.messages[-limit:])
        return list(session.messages)

    def reset_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            self._save()

    def list_sessions(self) -> list[str]:
        return list(self.sessions.keys())
