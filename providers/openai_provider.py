"""
OpenAI Provider — 支持流式和非流式。
v2: 新增 stream_chat() 实现。
"""

import json
from providers.base import LLMProvider, LLMResponse, ToolCall, StreamChunk


class OpenAIProvider(LLMProvider):

    def __init__(self, api_key: str, model: str = "gpt-4o-mini",
                 base_url: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def name(self) -> str:
        return f"openai/{self._model}"

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float = 0.7) -> LLMResponse:
        kwargs: dict = {"model": self._model, "messages": messages,
                        "temperature": temperature,
                        "frequency_penalty": 1.05,  # 抑制重复 token 生成
                        "presence_penalty": 1.05}    # 惩罚已出现过的 token，抑制段落级重复
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
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name,
                                           arguments=args))

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(text=msg.content, tool_calls=tool_calls,
                           finish_reason=choice.finish_reason or "stop",
                           usage=usage)

    def stream_chat(self, messages: list[dict], tools: list[dict] | None = None,
                    temperature: float = 0.7):
        """
        流式输出。解析 OpenAI SSE 流，yield StreamChunk。

        关键逻辑：
        - text delta → 直接 yield
        - tool_call 分多个 chunk 到达：先 start（有 id+name），
          然后多个 delta（参数 JSON 片段），最后在 finish_reason=tool_calls 时
          拼接完整参数并 yield tool_call_done
        """
        kwargs: dict = {"model": self._model, "messages": messages,
                        "temperature": temperature, "stream": True,
                        "stream_options": {"include_usage": True},
                        "frequency_penalty": 1.05,
                        "presence_penalty": 1.05}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = self._client.chat.completions.create(**kwargs)

        # 追踪正在构建中的 tool calls
        building: dict[int, dict] = {}  # index -> {id, name, args_buffer}

        for chunk in stream:
            # usage chunk（流结束时）
            if chunk.usage:
                yield StreamChunk(
                    type="usage",
                    usage={
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    },
                )

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            finish = chunk.choices[0].finish_reason

            # 文本增量
            if delta and delta.content:
                yield StreamChunk(type="text_delta", text=delta.content)

            # 工具调用增量
            if delta and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in building:
                        building[idx] = {
                            "id": tc_delta.id or "",
                            "name": (tc_delta.function.name
                                     if tc_delta.function else ""),
                            "args_buffer": "",
                        }
                        if building[idx]["id"]:
                            yield StreamChunk(
                                type="tool_call_start",
                                tool_call_id=building[idx]["id"],
                                tool_name=building[idx]["name"],
                            )
                    else:
                        if tc_delta.id:
                            building[idx]["id"] = tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            building[idx]["name"] = tc_delta.function.name

                    if tc_delta.function and tc_delta.function.arguments:
                        building[idx]["args_buffer"] += tc_delta.function.arguments
                        yield StreamChunk(
                            type="tool_call_delta",
                            tool_call_id=building[idx]["id"],
                            tool_args_delta=tc_delta.function.arguments,
                        )

            # 流结束
            if finish:
                # 完成所有正在构建的 tool calls
                if finish in ("tool_calls", "stop"):
                    for idx, info in building.items():
                        try:
                            args = json.loads(info["args_buffer"]) if info["args_buffer"] else {}
                        except json.JSONDecodeError:
                            args = {}
                        yield StreamChunk(
                            type="tool_call_done",
                            tool_call_id=info["id"],
                            tool_name=info["name"],
                            tool_call=ToolCall(
                                id=info["id"],
                                name=info["name"],
                                arguments=args,
                            ),
                        )
                    building.clear()

                yield StreamChunk(type="done", finish_reason=finish)
