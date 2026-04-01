"""
OpenAI Provider 实现。
对应 OpenClaw 通过 pi-ai 库调用 OpenAI Chat Completions API。
"""

import json
from providers.base import LLMProvider, LLMResponse, ToolCall


class OpenAIProvider(LLMProvider):
    """
    OpenAI 兼容 API 的 Provider 实现。
    通过 base_url 参数支持 dashscope、deepseek 等兼容 OpenAI 格式的服务。
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def name(self) -> str:
        return f"openai/{self._model}"

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        调用 OpenAI Chat Completions API。
        将返回的 tool_calls 解析为 ToolCall 对象列表，
        将 usage 统计提取为字典。
        """
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "frequency_penalty": 1.05,  # 抑制重复 token 生成
            "presence_penalty": 1.05,   # 惩罚已出现过的 token，抑制段落级重复
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    print(f"  ⚠️ 工具参数 JSON 解析失败: {tc.function.name}，使用空参数")
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc.id, name=tc.function.name, arguments=args,
                ))

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            text=msg.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )
