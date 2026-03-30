# 记忆系统深度解析

本文档详细拆解 OpenClaw Lite 中"记忆"的实现思路，涵盖存储、读取、压缩的完整链路。

## 为什么 Agent 需要记忆

LLM 本身是无状态的。每次调用 API，它只能看到你传入的 messages 列表，对之前的对话一无所知。
所谓"记忆"，就是我们在每次调用前，把历史消息塞回 messages 列表里，让 LLM 以为它"记得"之前的对话。

这带来一个核心矛盾：历史越长，上下文越丰富，但 token 消耗越大，最终会撞上 context window 上限。
整个记忆系统的设计，就是在"记得多"和"装得下"之间找平衡。

## 记忆的三层架构

```
┌─────────────────────────────────────────────┐
│  Layer 3: System Prompt（身份 + 工作区上下文）│  ← 每次都注入，永不压缩
├─────────────────────────────────────────────┤
│  Layer 2: Compacted History（压缩摘要）      │  ← 旧消息的 LLM 生成摘要
├─────────────────────────────────────────────┤
│  Layer 1: Recent Messages（最近的原始消息）   │  ← 最近 N 条完整保留
└─────────────────────────────────────────────┘
```

三层从上到下，优先级递减，但都会被拼进发给 LLM 的 messages 列表。


## Layer 1: 原始消息存储 (`session.py`)

### 数据模型

每条消息是一个 `Message` dataclass：

```python
@dataclass
class Message:
    role: str          # "user" | "assistant" | "system" | "tool"
    content: str       # 消息正文
    timestamp: float   # Unix 时间戳，自动生成
    tool_call_id: str  # 工具调用关联 ID（仅 tool 消息）
    name: str          # 工具名称（仅 tool 消息）
```

四种 role 的含义：
- `user`：用户输入
- `assistant`：LLM 的回复
- `tool`：工具执行结果（通过 `tool_call_id` 关联到 LLM 请求的工具调用）
- `system`：系统消息（system prompt，不存入会话历史）

### 会话结构

多条消息组成一个 `Session`：

```python
@dataclass
class Session:
    session_id: str           # 会话标识，默认 "default"
    messages: list[Message]   # 消息列表（按时间顺序）
    created_at: float         # 创建时间
    updated_at: float         # 最后更新时间
```


### 持久化机制

`SessionStore` 负责将会话序列化为 JSON 文件（默认路径 `~/.openclaw-lite/sessions.json`）：

```json
{
  "default": {
    "session_id": "default",
    "messages": [
      {"role": "user", "content": "读一下 config.json", "timestamp": 1711234567.0},
      {"role": "assistant", "content": "好的...", "timestamp": 1711234568.0}
    ],
    "created_at": 1711234560.0,
    "updated_at": 1711234568.0
  }
}
```

写入策略：每次 `add_message()` 都触发全量写入。为了防止写入中途崩溃导致文件损坏，
采用"写临时文件 → 原子替换"的方式：

```python
def _save(self):
    tmp_path = self.store_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(self.store_path)  # 原子操作
```

这样即使程序在写入过程中崩溃，原文件也不会被破坏。


### 消息如何流入 LLM

在 `agent.py` 的 `build_messages_for_llm()` 中，历史消息被转换为 OpenAI API 格式：

```python
def build_messages_for_llm(system_prompt, history):
    messages = [{"role": "system", "content": system_prompt}]  # 永远在最前面
    for msg in history:
        if msg.role == "tool":
            messages.append({
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id,  # 必须关联到对应的工具调用
            })
        else:
            messages.append({"role": msg.role, "content": msg.content})
    return messages
```

注意 `tool` 消息必须带 `tool_call_id`，这是 OpenAI API 的要求——
每个工具结果必须关联到 LLM 之前请求的那个工具调用，否则 API 会报错。

### 历史裁剪

加载历史时有一个 `history_limit`（默认 50）：

```python
def get_history(self, session_id, limit=50):
    if limit > 0 and len(session.messages) > limit:
        return list(session.messages[-limit:])  # 只取最近 N 条
    return list(session.messages)
```

这是最粗暴的记忆管理——直接丢弃最早的消息。但它只是第一道防线，真正精细的处理在 compaction。


## Layer 2: 历史压缩 (`compaction.py`)

### 触发条件

```python
def should_compact(messages, threshold=40):
    conversation_count = sum(1 for m in messages if m.role in ("user", "assistant"))
    return conversation_count > threshold
```

只计算 user 和 assistant 消息数（tool 消息跟随 assistant，不单独计数）。
默认阈值 40，意味着大约 20 轮对话后触发压缩。

OpenClaw 原版基于 token 计数（接近 context window 的 80% 时触发），
这里简化为消息数量，牺牲精度换取实现简单。


### 压缩流程

`compact_history()` 的完整流程：

```
压缩前（45 条消息）：
[msg1, msg2, ..., msg35] [msg36, msg37, ..., msg45]
 ←——— 旧消息（35条）——→  ←—— 最近消息（10条）——→

         ↓ LLM 生成摘要

压缩后（11 条消息）：
[摘要消息] [msg36, msg37, ..., msg45]
```

步骤：
1. 将消息列表分割为"旧消息"和"最近消息"（保留最近 10 条）
2. 将旧消息格式化为文本，发给 LLM 生成摘要
3. 摘要作为一条 assistant 消息替换所有旧消息
4. 更新 SessionStore 中的消息列表并持久化


### 摘要生成的 Prompt

```python
summary_prompt = [
    {
        "role": "system",
        "content": (
            "你是一个对话摘要助手。请将以下对话历史压缩为简洁的摘要。\n"
            "保留关键信息：\n"
            "- 用户的主要请求和偏好\n"
            "- 重要的决策和结论\n"
            "- 修改过的文件路径和关键命令\n"
            "- 遇到的错误和解决方案\n"
            "用 2-5 段话概括，不要遗漏重要细节。"
        ),
    },
    {
        "role": "user",
        "content": f"请压缩以下对话历史：\n\n{old_text}",
    },
]
```

关键设计决策：
- 要求保留文件路径和命令——这些是 Agent 工作的"锚点"，丢了就不知道之前改了什么
- 要求保留错误和解决方案——避免 Agent 重复犯同样的错误
- 用低温度（0.3）生成——摘要需要准确，不需要创造性


### 容错设计

压缩可能失败（LLM API 超时、返回空内容等），所以有兜底：

```python
try:
    response = provider.chat(messages=summary_prompt, temperature=0.3)
    summary_text = response.text or "（压缩失败，无摘要）"
except Exception as e:
    print(f"  ⚠️ 历史压缩失败: {e}，保留原始历史")
    return messages  # 失败时返回原始消息，不丢数据
```

在 `agent.py` 的调用侧也有 try/except 包裹，确保压缩失败不会中断正常对话。

## Layer 3: System Prompt 中的持久上下文

System prompt 是每次对话都会注入的"永久记忆"，不受压缩影响。
它包含身份信息、工具描述、安全规则，以及工作区上下文文件。

工作区上下文文件（`SOUL.md`、`AGENTS.md`）相当于"长期记忆"——
它们不是从对话中学来的，而是人工编写的项目知识，每次对话都会被读取并注入。


## 完整的记忆生命周期

一次对话中，记忆的完整流转：

```
1. 用户输入 "帮我改一下 config.json"
   ↓
2. SessionStore.get_history() 加载历史（最多 50 条）
   ↓
3. should_compact() 检查是否需要压缩
   ├── 是 → compact_history() 压缩旧消息，更新 SessionStore
   └── 否 → 跳过
   ↓
4. 用户消息存入 SessionStore（立即持久化到 JSON 文件）
   ↓
5. build_messages_for_llm() 拼装完整消息列表：
   [system_prompt, ...历史消息..., 用户消息]
   ↓
6. 发送给 LLM → LLM 返回工具调用或文本回复
   ├── 工具调用 → 执行工具 → 结果加入 turn_messages → 回到步骤 6
   └── 文本回复 → assistant 消息存入 SessionStore → 结束
```

注意步骤 6 中的 `turn_messages`（当前轮次的工具调用消息）没有存入 SessionStore，
只有最终的 assistant 回复会被持久化。这意味着如果中途崩溃，工具调用的中间状态会丢失。
这是一个已知的简化——OpenClaw 原版会持久化完整的工具调用链。


## 与 OpenClaw 原版的差异

| 维度 | 本项目 | OpenClaw 原版 |
|------|--------|--------------|
| 压缩触发 | 消息数量 > 40 | Token 计数接近 context window 80% |
| 压缩粒度 | 整体摘要 | 保留工具调用标识符的精细摘要 |
| 中间状态 | 不持久化 tool 调用链 | 完整持久化 |
| 并发安全 | 原子文件替换 | 数据库级别的事务 |
| 摘要模型 | 复用主模型 | 可配置独立的廉价模型 |

## 可改进方向

1. 基于 token 计数触发压缩（用 tiktoken 库计算）
2. 持久化工具调用的中间状态，支持断点恢复
3. 分层压缩（最近 → 摘要 → 超级摘要），支持更长的对话
4. 向量数据库存储历史，按相关性检索而非按时间裁剪
5. 用更便宜的模型做压缩，节省 token 成本
