"""
LLM Provider 抽象基类。
对应 OpenClaw: src/agents/pi-embedded-runner/run.ts 中的 resolveModel()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """LLM 返回的工具调用请求。"""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """LLM 响应。"""
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict | None = None


class LLMProvider(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> LLMResponse: ...

    @abstractmethod
    def name(self) -> str: ...
