"""
多层上下文压缩。
v2: 从消息数驱动改为 token 数驱动，新增工具结果截断层。

对标 Claude Code 的四层压缩策略（简化版）：
1. 工具结果截断：长工具输出先截短（最轻量）
2. 历史摘要压缩：旧消息用 LLM 生成摘要替代（重量级）

触发条件基于 token 估算，而非消息数量。
"""

from session import Message
from providers.base import LLMProvider
from token_counter import estimate_tokens, estimate_messages_tokens

DEFAULT_KEEP_RECENT = 10
# 单条工具结果的截断阈值（字符数）
TOOL_RESULT_TRUNCATE_CHARS = 8000
# 工具结果截断后的保留字符数（头+尾）
TOOL_RESULT_KEEP_HEAD = 3000
TOOL_RESULT_KEEP_TAIL = 2000


def should_compact(messages: list[Message], token_threshold: int = 80000) -> bool:
    """
    基于 token 估算判断是否需要压缩。
    对标 Claude Code 基于 API 返回的 input_tokens 判断。
    """
    return estimate_messages_tokens(messages) > token_threshold


def truncate_tool_results(messages: list[Message]) -> tuple[list[Message], int]:
    """
    第一层压缩：截断过长的工具结果。
    对标 Claude Code 的 applyToolResultBudget()。

    只处理 tool role 的消息，保留头部和尾部，中间用省略标记替代。
    返回 (处理后的消息列表, 节省的估算 token 数)。
    """
    result = []
    tokens_saved = 0
    for msg in messages:
        if msg.role == "tool" and len(msg.content) > TOOL_RESULT_TRUNCATE_CHARS:
            original_tokens = estimate_tokens(msg.content)
            truncated = (
                msg.content[:TOOL_RESULT_KEEP_HEAD]
                + f"\n\n[... 截断 {len(msg.content) - TOOL_RESULT_KEEP_HEAD - TOOL_RESULT_KEEP_TAIL} 字符 ...]\n\n"
                + msg.content[-TOOL_RESULT_KEEP_TAIL:]
            )
            new_msg = Message(
                role=msg.role, content=truncated,
                timestamp=msg.timestamp,
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            )
            result.append(new_msg)
            tokens_saved += original_tokens - estimate_tokens(truncated)
        else:
            result.append(msg)
    return result, tokens_saved


def compact_history(
    messages: list[Message],
    provider: LLMProvider,
    keep_recent: int = DEFAULT_KEEP_RECENT,
    token_threshold: int = 80000,
) -> list[Message]:
    """
    完整压缩流程：
    1. 先截断工具结果（轻量级）
    2. 如果仍超阈值，用 LLM 摘要压缩旧消息（重量级）

    对标 Claude Code 的 autocompact + microcompact 组合。
    """
    # 第一层：工具结果截断
    messages, saved = truncate_tool_results(messages)
    if saved > 0:
        print(f"  📦 工具结果截断，节省约 {saved} tokens")

    # 检查截断后是否仍需压缩
    if not should_compact(messages, token_threshold):
        return messages

    # 第二层：LLM 摘要压缩
    if len(messages) <= keep_recent:
        return messages

    old_messages = messages[:-keep_recent]
    recent_messages = messages[-keep_recent:]

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
                "- 工具调用的关键结果\n"
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
        print(f"  ⚠️ 历史压缩失败: {e}，保留原始历史")
        return messages

    summary_message = Message(
        role="assistant",
        content=f"[对话历史摘要 — 以下是之前 {len(old_messages)} 条消息的压缩]\n\n{summary_text}",
    )

    result = [summary_message] + recent_messages
    old_tokens = estimate_messages_tokens(old_messages)
    new_tokens = estimate_tokens(summary_text)
    print(f"  📦 摘要压缩完成: {old_tokens} → {new_tokens} tokens (节省 {old_tokens - new_tokens})")
    return result


def _format_messages_for_summary(messages: list[Message]) -> str:
    lines = []
    for msg in messages:
        role_label = {
            "user": "用户", "assistant": "助手",
            "system": "系统", "tool": "工具结果",
        }.get(msg.role, msg.role)
        content = msg.content
        if len(content) > 2000:
            content = content[:2000] + "...[截断]"
        if msg.name:
            lines.append(f"[{role_label} - {msg.name}]: {content}")
        else:
            lines.append(f"[{role_label}]: {content}")
    return "\n\n".join(lines)