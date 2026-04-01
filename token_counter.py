"""
Token 计数。对标 Claude Code 的 tokenCountWithEstimation()。
优先使用 API 返回的 usage，fallback 到字符估算。
"""

from __future__ import annotations

CHARS_PER_TOKEN_CJK = 1.5  # 实际 DashScope/OpenAI tokenizer 约 1.3-1.5 字符/token
CHARS_PER_TOKEN_EN = 4.0


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff'
              or '\u3000' <= c <= '\u303f'
              or '\uff00' <= c <= '\uffef')
    en = len(text) - cjk
    return int(cjk / CHARS_PER_TOKEN_CJK + en / CHARS_PER_TOKEN_EN)


def estimate_messages_tokens(messages: list) -> int:
    """估算消息列表（Message 对象或 dict）的总 token 数"""
    total = 0
    for msg in messages:
        if hasattr(msg, 'content'):
            content = msg.content or ""
        elif isinstance(msg, dict):
            content = msg.get("content", "") or ""
        else:
            content = str(msg)
        if isinstance(content, str):
            total += estimate_tokens(content)
        total += 4  # 每条消息固定开销
    return total


class TokenTracker:
    """追踪 token 用量，优先用 API 精确值"""

    def __init__(self):
        self._last_input: int | None = None
        self._cumulative_input: int = 0
        self._cumulative_output: int = 0

    def record_usage(self, usage: dict | None):
        if not usage:
            return
        inp = usage.get("prompt_tokens", 0)
        out = usage.get("completion_tokens", 0)
        self._last_input = inp
        self._cumulative_input += inp
        self._cumulative_output += out

    @property
    def last_input_tokens(self) -> int:
        return self._last_input or 0

    @property
    def cumulative_tokens(self) -> int:
        return self._cumulative_input + self._cumulative_output

    def estimate_context(self, messages: list) -> int:
        if self._last_input is not None:
            return self._last_input
        return estimate_messages_tokens(messages)

    def reset(self):
        self._last_input = None
        self._cumulative_input = 0
        self._cumulative_output = 0
