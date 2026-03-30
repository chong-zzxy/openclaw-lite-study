"""
长期记忆系统。

跨会话持久化的记忆，包括：
- 用户画像（偏好、习惯、身份信息）
- 知识记忆（重要事实、决策、文件操作记录）

生命周期独立于 session，reset 会话不会丢失长期记忆。
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from providers.base import LLMProvider


@dataclass
class MemoryEntry:
    """一条知识记忆"""
    content: str
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    source_session: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "MemoryEntry":
        return MemoryEntry(
            content=d["content"],
            tags=d.get("tags", []),
            created_at=d.get("created_at", 0),
            source_session=d.get("source_session", ""),
        )


@dataclass
class UserProfile:
    """用户画像"""
    preferences: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "UserProfile":
        return UserProfile(
            preferences=d.get("preferences", []),
            facts=d.get("facts", []),
            updated_at=d.get("updated_at", 0),
        )


class LongTermMemory:
    """
    长期记忆管理器。

    存储路径与 session store 同级，但独立于会话：
    - ~/.openclaw-lite/user_profile.json  — 用户画像
    - ~/.openclaw-lite/memories.json      — 知识记忆
    """

    MAX_MEMORIES = 100  # 最多保留的知识记忆条数
    MAX_PROFILE_ITEMS = 20  # 画像中每类最多条目数

    def __init__(self, store_dir: str):
        self._dir = Path(store_dir).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._profile_path = self._dir / "user_profile.json"
        self._memories_path = self._dir / "memories.json"
        self.profile = self._load_profile()
        self.memories: list[MemoryEntry] = self._load_memories()

    def _load_profile(self) -> UserProfile:
        if self._profile_path.exists():
            try:
                with open(self._profile_path, "r", encoding="utf-8") as f:
                    return UserProfile.from_dict(json.load(f))
            except (json.JSONDecodeError, KeyError):
                pass
        return UserProfile()

    def _load_memories(self) -> list[MemoryEntry]:
        if self._memories_path.exists():
            try:
                with open(self._memories_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                return [MemoryEntry.from_dict(m) for m in raw]
            except (json.JSONDecodeError, KeyError):
                pass
        return []

    def _save_profile(self):
        tmp = self._profile_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.profile.to_dict(), f, ensure_ascii=False, indent=2)
        tmp.replace(self._profile_path)

    def _save_memories(self):
        tmp = self._memories_path.with_suffix(".tmp")
        data = [m.to_dict() for m in self.memories]
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(self._memories_path)

    def add_memory(self, content: str, tags: list[str] = None, session_id: str = ""):
        """添加一条知识记忆"""
        entry = MemoryEntry(
            content=content,
            tags=tags or [],
            source_session=session_id,
        )
        self.memories.append(entry)
        # 超过上限时淘汰最旧的
        if len(self.memories) > self.MAX_MEMORIES:
            self.memories = self.memories[-self.MAX_MEMORIES:]
        self._save_memories()

    def update_profile(self, preferences: list[str] = None, facts: list[str] = None):
        """更新用户画像（合并去重，保留最新）"""
        if preferences:
            existing = set(self.profile.preferences)
            for p in preferences:
                if p not in existing:
                    self.profile.preferences.append(p)
            # 保留最新的 N 条
            self.profile.preferences = self.profile.preferences[-self.MAX_PROFILE_ITEMS:]
        if facts:
            existing = set(self.profile.facts)
            for f in facts:
                if f not in existing:
                    self.profile.facts.append(f)
            self.profile.facts = self.profile.facts[-self.MAX_PROFILE_ITEMS:]
        self.profile.updated_at = time.time()
        self._save_profile()

    def get_relevant_memories(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        """
        获取与查询相关的记忆。
        简化实现：关键词匹配 + 时间衰减。
        生产环境应替换为向量相似度检索。
        """
        scored = []
        now = time.time()
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for mem in self.memories:
            content_lower = mem.content.lower()
            # 关键词匹配得分
            word_score = sum(1 for w in query_words if w in content_lower)
            # 标签匹配得分
            tag_score = sum(2 for t in mem.tags if t.lower() in query_lower)
            # 时间衰减（越新越重要，半衰期 7 天）
            age_days = (now - mem.created_at) / 86400
            time_score = 1.0 / (1.0 + age_days / 7.0)

            total = word_score + tag_score + time_score
            if total > 0.1:
                scored.append((total, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:limit]]

    def format_for_prompt(self, query: str = "") -> str:
        """将长期记忆格式化为可注入 system prompt 的文本"""
        sections = []

        # 用户画像
        if self.profile.preferences or self.profile.facts:
            lines = ["## 用户画像（长期记忆）"]
            if self.profile.preferences:
                lines.append("偏好：")
                for p in self.profile.preferences:
                    lines.append(f"- {p}")
            if self.profile.facts:
                lines.append("已知信息：")
                for f in self.profile.facts:
                    lines.append(f"- {f}")
            sections.append("\n".join(lines))

        # 相关知识记忆
        relevant = self.get_relevant_memories(query, limit=5)
        if relevant:
            lines = ["## 相关历史记忆"]
            for mem in relevant:
                lines.append(f"- {mem.content}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)


def extract_memories_from_conversation(
    messages: list,
    provider: LLMProvider,
    ltm: LongTermMemory,
    session_id: str = "",
):
    """
    从对话中提取长期记忆。
    在会话结束或 reset 时调用。
    """
    if len(messages) < 4:
        return  # 对话太短，没什么可提取的

    # 格式化最近的对话
    lines = []
    for msg in messages[-20:]:  # 最多看最近 20 条
        role = {"user": "用户", "assistant": "助手"}.get(msg.role, msg.role)
        if msg.role in ("user", "assistant"):
            content = msg.content[:500] if len(msg.content) > 500 else msg.content
            lines.append(f"[{role}]: {content}")

    if not lines:
        return

    conversation_text = "\n".join(lines)

    prompt = [
        {
            "role": "system",
            "content": (
                "你是一个记忆提取助手。从对话中提取值得长期记住的信息。\n"
                "返回 JSON 格式，包含三个字段：\n"
                '- "preferences": 用户偏好列表（如回复风格、语言偏好等）\n'
                '- "facts": 用户相关事实列表（如技术栈、项目信息等）\n'
                '- "memories": 重要事件列表，每项包含 "content" 和 "tags"\n'
                "如果没有值得记住的内容，返回空列表。\n"
                "只返回 JSON，不要其他文字。"
            ),
        },
        {
            "role": "user",
            "content": f"从以下对话中提取长期记忆：\n\n{conversation_text}",
        },
    ]

    try:
        response = provider.chat(messages=prompt, temperature=0.2)
        text = response.text or ""
        # 尝试提取 JSON（兼容 markdown code block）
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())

        # 更新用户画像
        prefs = data.get("preferences", [])
        facts = data.get("facts", [])
        if prefs or facts:
            ltm.update_profile(preferences=prefs, facts=facts)
            print(f"  🧠 更新用户画像: {len(prefs)} 条偏好, {len(facts)} 条事实")

        # 添加知识记忆
        mems = data.get("memories", [])
        for m in mems:
            if isinstance(m, dict) and "content" in m:
                ltm.add_memory(
                    content=m["content"],
                    tags=m.get("tags", []),
                    session_id=session_id,
                )
        if mems:
            print(f"  🧠 新增 {len(mems)} 条知识记忆")

    except (json.JSONDecodeError, Exception) as e:
        print(f"  ⚠️ 记忆提取失败: {e}")
