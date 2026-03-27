"""
会话历史压缩（Compaction）。

对应 OpenClaw: src/agents/compaction.ts

OpenClaw 的 compaction 机制：
当会话历史接近模型的 context window 上限时，自动压缩旧消息。

压缩策略：
1. 保留 system prompt（不压缩）
2. 保留最近 N 条消息（不压缩）
3. 将较早的消息用 LLM 生成摘要替代
4. 摘要保留关键信息：决策、文件修改、重要上下文

OpenClaw 的实现细节：
- 使用独立的 LLM 调用生成摘要（可以用更便宜的模型）
- 支持重试（compaction 失败时保留原始历史）
- 保留工具调用的标识符（文件路径、命令等）
- 有 token 计数器精确控制压缩阈值
- 支持 compaction 安全超时

这里简化为：当消息数超过阈值时，用 LLM 生成摘要替换旧消息。
"""

from session import Message, SessionStore
from providers.base import LLMProvider


# 默认保留最近的消息数
DEFAULT_KEEP_RECENT = 10
# 触发 compaction 的消息数阈值
DEFAULT_COMPACTION_THRESHOLD = 40


def should_compact(
    messages: list[Message],
    threshold: int = DEFAULT_COMPACTION_THRESHOLD,
) -> bool:
    """
    判断是否需要压缩。

    对应 OpenClaw: compaction 触发条件
    OpenClaw 基于 token 计数判断（接近 context window 的 80%），
    这里简化为基于消息数量。
    """
    # 只计算 user 和 assistant 消息（tool 消息跟随 assistant）
    conversation_count = sum(1 for m in messages if m.role in ("user", "assistant"))
    return conversation_count > threshold


def compact_history(
    messages: list[Message],
    provider: LLMProvider,
    keep_recent: int = DEFAULT_KEEP_RECENT,
) -> list[Message]:
    """
    压缩会话历史。

    对应 OpenClaw: compactEmbeddedPiSession()

    流程：
    1. 将旧消息（除最近 keep_recent 条外）提取出来
    2. 用 LLM 生成摘要
    3. 用一条 system 消息替换旧消息
    4. 保留最近的消息不变

    OpenClaw 的压缩更精细：
    - 保留工具调用中的文件路径和命令
    - 保留用户的明确指令和偏好
    - 保留错误信息和修复记录
    - 使用 identifier preservation 策略
    """
    if len(messages) <= keep_recent:
        return messages

    # 分割：旧消息 + 最近消息
    old_messages = messages[:-keep_recent]
    recent_messages = messages[-keep_recent:]

    # 构建摘要请求
    old_text = _format_messages_for_summary(old_messages)

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

    try:
        response = provider.chat(messages=summary_prompt, temperature=0.3)
        summary_text = response.text or "（压缩失败，无摘要）"
    except Exception as e:
        # Compaction 失败时保留原始历史（对应 OpenClaw 的 compaction retry 逻辑）
        print(f"  ⚠️ 历史压缩失败: {e}，保留原始历史")
        return messages

    # 用摘要消息替换旧消息（用 assistant role 避免多条 system 消息的兼容性问题）
    summary_message = Message(
        role="assistant",
        content=f"[对话历史摘要 — 以下是之前 {len(old_messages)} 条消息的压缩]\n\n{summary_text}",
    )

    return [summary_message] + recent_messages


def _format_messages_for_summary(messages: list[Message]) -> str:
    """将消息列表格式化为文本，用于生成摘要"""
    lines = []
    for msg in messages:
        role_label = {
            "user": "用户",
            "assistant": "助手",
            "system": "系统",
            "tool": "工具结果",
        }.get(msg.role, msg.role)

        content = msg.content
        # 截断过长的单条消息
        if len(content) > 2000:
            content = content[:2000] + "...[截断]"

        if msg.name:
            lines.append(f"[{role_label} - {msg.name}]: {content}")
        else:
            lines.append(f"[{role_label}]: {content}")

    return "\n\n".join(lines)
