"""
长期记忆系统（三层策略）。

跨会话持久化的记忆，包括：
- 用户画像（偏好、习惯、身份信息）
- 知识记忆（重要事实、决策、文件操作记录）

记忆写入的三种触发方式：
1. 显式记忆：用户通过 /remember 命令主动存入，零成本零歧义
2. 实时检测：规则引擎扫描每条用户消息，命中高价值模式时直接存入（不调 LLM）
3. 批量提取：每 N 轮对话或会话 reset 时，调 LLM 批量提取（摊薄成本）

生命周期独立于 session，reset 会话不会丢失长期记忆。
"""

import re
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
    source: str = ""  # "explicit" | "detected" | "extracted"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "MemoryEntry":
        return MemoryEntry(
            content=d["content"],
            tags=d.get("tags", []),
            created_at=d.get("created_at", 0),
            source=d.get("source", ""),
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


# === 实时检测规则 ===

# 每条规则: (正则模式, 分类, 标签列表)
# 分类: "fact" 存入 profile.facts, "preference" 存入 profile.preferences
DETECTION_RULES: list[tuple[str, str, list[str]]] = [
    # 自我介绍
    (r"我(?:叫|是|名字是|的名字叫)\s*(.+)", "fact", ["身份"]),
    (r"我的(?:名字|昵称|网名)(?:是|叫)\s*(.+)", "fact", ["身份"]),
    # 职业/角色
    (r"我是(?:一[个名位])?(.+?(?:工程师|开发者|程序员|设计师|学生|老师|产品经理))", "fact", ["职业"]),
    # 技术栈
    (r"我(?:用|使用|在用|主要用)\s*(.+?)(?:开发|编程|写代码|$)", "fact", ["技术栈"]),
    (r"我(?:的)?(?:技术栈|主力语言)(?:是|有)\s*(.+)", "fact", ["技术栈"]),
    # 偏好表达
    (r"(?:请|以后)?(?:用|使用)(.+?)(?:回复|回答|风格|语气)", "preference", ["风格"]),
    (r"我(?:喜欢|偏好|希望|想要)(.+?)(?:的)?(?:回复|风格|方式|语气)", "preference", ["风格"]),
    (r"(?:不要|别|禁止)(.+?)(?:回复|回答|输出)", "preference", ["风格"]),
    # 授权/操作偏好
    (r"(全权|自主|自动|不用问我|不需要确认).*(?:操作|执行|运行|处理)", "preference", ["授权"]),
    (r"(?:非|除了|不要)(.+?)(?:之类的|等)(?:敏感|危险)?操作", "preference", ["授权"]),
    # 语言偏好
    (r"(?:请)?用(中文|英文|日文|韩文)(?:回复|回答|交流)?", "preference", ["语言"]),
    # 项目信息
    (r"(?:这个|我的)项目(?:叫|是|名字是)\s*(.+)", "fact", ["项目"]),
]


def detect_memory_from_message(text: str) -> list[tuple[str, str, list[str]]]:
    """
    用规则引擎扫描用户消息，检测高价值信息。
    返回: [(提取的内容, 分类, 标签列表), ...]
    不调用 LLM，纯正则匹配，零成本。
    """
    results = []
    for pattern, category, tags in DETECTION_RULES:
        match = re.search(pattern, text)
        if match:
            captured = match.group(1).strip().rstrip("。，,.!！?？")
            if captured and len(captured) < 100:
                results.append((captured, category, tags))
    return results


class LongTermMemory:
    """
    长期记忆管理器。

    存储路径与 session store 同级，但独立于会话：
    - ~/.openclaw-lite/user_profile.json  — 用户画像
    - ~/.openclaw-lite/memories.json      — 知识记忆
    """

    MAX_MEMORIES = 100  # 记忆条目上限，超出后丢弃最早的
    MAX_PROFILE_ITEMS = 20  # 用户画像每类（偏好/事实）最多保留条数
    BATCH_EXTRACT_INTERVAL = 10  # 每 N 轮对话触发一次 LLM 批量提取，摊薄调用成本

    def __init__(self, store_dir: str):
        self._dir = Path(store_dir).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._profile_path = self._dir / "user_profile.json"
        self._memories_path = self._dir / "memories.json"
        self.profile = self._load_profile()
        self.memories: list[MemoryEntry] = self._load_memories()
        self._turns_since_extract = 0  # 距上次批量提取的轮数

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

    def add_memory(self, content: str, tags: list[str] = None, source: str = ""):
        """添加一条知识记忆"""
        # 去重：内容相同的不重复添加
        for existing in self.memories:
            if existing.content == content:
                return
        entry = MemoryEntry(content=content, tags=tags or [], source=source)
        self.memories.append(entry)
        if len(self.memories) > self.MAX_MEMORIES:
            self.memories = self.memories[-self.MAX_MEMORIES:]
        self._save_memories()

    def update_profile(self, preferences: list[str] = None, facts: list[str] = None):
        """更新用户画像（合并去重，保留最新）"""
        changed = False
        if preferences:
            existing = set(self.profile.preferences)
            for p in preferences:
                if p not in existing:
                    self.profile.preferences.append(p)
                    changed = True
            self.profile.preferences = self.profile.preferences[-self.MAX_PROFILE_ITEMS:]
        if facts:
            existing = set(self.profile.facts)
            for f in facts:
                if f not in existing:
                    self.profile.facts.append(f)
                    changed = True
            self.profile.facts = self.profile.facts[-self.MAX_PROFILE_ITEMS:]
        if changed:
            self.profile.updated_at = time.time()
            self._save_profile()

    # === 层级 1: 显式记忆 ===

    def remember_explicit(self, text: str):
        """用户通过 /remember 命令主动存入的记忆"""
        self.add_memory(content=text, tags=["显式记忆"], source="explicit")
        # 同时尝试分类到画像
        detections = detect_memory_from_message(text)
        for content, category, tags in detections:
            if category == "fact":
                self.update_profile(facts=[content])
            elif category == "preference":
                self.update_profile(preferences=[content])
        # 如果规则没命中，作为通用事实存入画像
        if not detections:
            self.update_profile(facts=[text])
        print(f"  🧠 已记住: {text}")

    # === 层级 2: 实时检测 ===

    def detect_and_store(self, user_message: str) -> bool:
        """
        扫描用户消息，检测高价值信息并存入。
        返回 True 表示检测到"记住"等强信号词，建议立即触发批量提取。
        """
        detections = detect_memory_from_message(user_message)
        for content, category, tags in detections:
            if category == "fact":
                self.update_profile(facts=[content])
            elif category == "preference":
                self.update_profile(preferences=[content])
            self.add_memory(content=content, tags=tags, source="detected")
            print(f"  🧠 检测到记忆: {content}")

        # "记住"是强信号——用户明确要求 Agent 记住某些事
        urgent_keywords = ["记住", "别忘了", "牢记", "务必记得"]
        needs_urgent_extract = any(kw in user_message for kw in urgent_keywords)
        return needs_urgent_extract

    # === 层级 3: 批量提取 ===

    def should_batch_extract(self) -> bool:
        """判断是否该触发批量提取"""
        self._turns_since_extract += 1
        return self._turns_since_extract >= self.BATCH_EXTRACT_INTERVAL

    def reset_extract_counter(self):
        """重置提取计数器"""
        self._turns_since_extract = 0

    def get_relevant_memories(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        """获取与查询相关的记忆（关键词匹配 + 时间衰减）"""
        scored = []
        now = time.time()
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for mem in self.memories:
            content_lower = mem.content.lower()
            word_score = sum(1 for w in query_words if w in content_lower)
            tag_score = sum(2 for t in mem.tags if t.lower() in query_lower)
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
    层级 3: 从对话中批量提取长期记忆（调用 LLM）。
    仅在以下时机调用：
    - 每 N 轮对话（由 should_batch_extract 控制）
    - 会话 reset 时
    """
    if len(messages) < 2:
        return

    lines = []
    for msg in messages[-20:]:
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
                '- "facts": 用户相关事实列表（如姓名、技术栈、项目信息等）\n'
                '- "memories": 重要事件列表，每项含 "content" 和 "tags"\n'
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
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())

        prefs = data.get("preferences", [])
        facts = data.get("facts", [])
        if prefs or facts:
            ltm.update_profile(preferences=prefs, facts=facts)
            print(f"  🧠 批量提取画像: {len(prefs)} 条偏好, {len(facts)} 条事实")

        mems = data.get("memories", [])
        for m in mems:
            if isinstance(m, dict) and "content" in m:
                ltm.add_memory(
                    content=m["content"],
                    tags=m.get("tags", []),
                    source="extracted",
                )
        if mems:
            print(f"  🧠 批量提取记忆: {len(mems)} 条")

        ltm.reset_extract_counter()

    except (json.JSONDecodeError, Exception) as e:
        print(f"  ⚠️ 记忆提取失败: {e}")
