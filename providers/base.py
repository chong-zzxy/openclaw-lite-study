"""
LLM Provider 抽象基类。
对应 OpenClaw: src/agents/pi-embedded-runner/run.ts 中的 resolveModel()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """LLM 返回的工具调用请求。id 用于关联后续的工具结果消息。"""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """
    LLM 响应。包含文本回复和/或工具调用请求。
    - 只有 text → 最终回复，Agent 循环结束
    - 有 tool_calls → LLM 请求调用工具，Agent 需要执行后回传结果
    """
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict | None = None  # token 用量统计


class LLMProvider(ABC):
    """
    LLM Provider 抽象基类。
    所有模型提供商（OpenAI、dashscope 等）都实现这个接口。
    只需两个方法：chat() 发送对话，name() 返回标识。
    """
    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> LLMResponse: ...

    @abstractmethod
    def name(self) -> str: ...
