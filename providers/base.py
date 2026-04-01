"""
LLM Provider 抽象基类。
v2: 新增流式输出支持（stream_chat 方法）。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class ToolCall:
    """LLM 返回的工具调用请求"""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """完整的 LLM 响应"""
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict | None = None


@dataclass
class StreamChunk:
    """
    流式输出的单个 chunk。
    对标 Claude Code 的 StreamEvent。

    type:
    - "text_delta": 文本增量
    - "tool_call_start": 工具调用开始（有 tool_call_id, tool_name）
    - "tool_call_delta": 工具调用参数增量
    - "tool_call_done": 工具调用参数解析完成
    - "done": 流结束
    - "usage": token 用量
    """
    type: str
    text: str = ""
    tool_call_id: str = ""
    tool_name: str = ""
    tool_args_delta: str = ""
    tool_call: ToolCall | None = None
    usage: dict | None = None
    finish_reason: str = ""


class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float = 0.7) -> LLMResponse: ...

    @abstractmethod
    def stream_chat(self, messages: list[dict], tools: list[dict] | None = None,
                    temperature: float = 0.7) -> Generator[StreamChunk, None, None]:
        """流式输出。yield StreamChunk 对象。"""
        ...

    @abstractmethod
    def name(self) -> str: ...
