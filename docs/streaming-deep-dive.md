# 流式输出与流式工具执行 深度解析

本文档详细拆解 OpenClaw Lite v2 中流式输出和流式工具执行的实现，对标 Claude Code 的 `StreamingToolExecutor` 和 `queryLoop()` 中的流式处理逻辑。

## 为什么需要流式输出

非流式模式下，用户发送消息后要等 LLM 生成完整响应才能看到任何输出。对于长回复（几百 token），这意味着 5-30 秒的空白等待。流式输出让模型边生成边显示，首字延迟从"整个响应时间"降到"首 token 生成时间"（通常 < 1 秒）。

Claude Code 的做法：`queryLoop()` 中通过 `for await (const message of deps.callModel(...))` 逐条接收流式事件，每收到一个 text block 就立即 yield 给上层渲染。

## 流式协议设计

### StreamChunk 数据结构

对标 Claude Code 的 `StreamEvent`，我们定义了 `StreamChunk`：

```python
@dataclass
class StreamChunk:
    type: str              # 事件类型
    text: str = ""         # text_delta 的文本增量
    tool_call_id: str = "" # 工具调用 ID
    tool_name: str = ""    # 工具名称
    tool_args_delta: str = ""  # 工具参数 JSON 片段
    tool_call: ToolCall | None = None  # 完整的工具调用（tool_call_done 时）
    usage: dict | None = None  # token 用量
    finish_reason: str = ""    # 结束原因
```

事件类型流转：

```
模型开始生成
  ├── text_delta × N          → 文本增量，逐字到达
  ├── tool_call_start         → 工具调用开始（有 id + name）
  ├── tool_call_delta × N     → 工具参数 JSON 片段
  ├── tool_call_done          → 参数拼接完成，可以执行了
  ├── usage                   → token 用量统计
  └── done                    → 流结束（finish_reason: stop/tool_calls/length）
```

### OpenAI SSE 流解析

OpenAI 的流式响应是 Server-Sent Events 格式。工具调用的参数分多个 chunk 到达，需要在客户端拼接：

```python
# 追踪正在构建中的 tool calls
building: dict[int, dict] = {}  # index -> {id, name, args_buffer}

for chunk in stream:
    if delta.tool_calls:
        for tc_delta in delta.tool_calls:
            idx = tc_delta.index
            if idx not in building:
                # 新工具调用开始
                building[idx] = {"id": tc_delta.id, "name": ..., "args_buffer": ""}
                yield StreamChunk(type="tool_call_start", ...)
            # 累积参数片段
            building[idx]["args_buffer"] += tc_delta.function.arguments
            yield StreamChunk(type="tool_call_delta", ...)

    if finish_reason:
        # 流结束，拼接完整参数
        for info in building.values():
            args = json.loads(info["args_buffer"])
            yield StreamChunk(type="tool_call_done", tool_call=ToolCall(...))
```

关键点：`tool_call_delta` 中的 `arguments` 是 JSON 字符串的片段（如 `{"file_`、`path": "test`、`.py"}`），必须全部拼接后才能 `json.loads()`。


## 流式工具执行器（StreamingToolExecutor）

### 设计动机

传统流程：等模型响应完 → 逐个执行工具 → 收集结果 → 再调模型。
如果模型返回 3 个工具调用，每个执行 2 秒，总等待 6 秒。

Claude Code 的优化：模型还在流式输出时，已解析完的 tool_call 立即开始执行。如果 3 个工具调用分别在第 1、3、5 秒解析完，它们可以并行执行，总等待时间接近最慢的那个（2 秒），而不是串行的 6 秒。

### 并行策略

```
只读工具（read, glob, grep, web_search）→ 提交到 ThreadPoolExecutor 并行
写工具（write, edit, exec）→ 等前面的都完成后再执行（串行保证）
```

为什么写工具要串行？因为写操作可能有依赖关系（先写文件 A，再编辑文件 A），并行会导致竞态条件。只读工具之间没有副作用，可以安全并行。

### 核心 API

```python
class StreamingToolExecutor:
    def add_tool(self, tool_call_id, tool_name, arguments, is_read_only=False):
        """提交工具执行。只读工具立即并行，写工具等前面完成。"""

    def get_completed_results(self) -> list[ToolExecResult]:
        """非阻塞获取已完成的结果。在流式接收过程中调用。"""

    def get_remaining_results(self) -> list[ToolExecResult]:
        """阻塞等待所有剩余结果。流结束后调用。"""

    def discard(self):
        """丢弃所有待执行结果。模型 fallback 或中断时调用。"""
```

### 在 Agent 循环中的集成

```python
executor = StreamingToolExecutor(registry=tool_registry, execute_fn=...)

for chunk in provider.stream_chat(messages, tools):
    if chunk.type == "text_delta":
        sys.stdout.write(chunk.text)  # 实时打印

    elif chunk.type == "tool_call_done":
        # 权限检查通过后，立即提交执行
        executor.add_tool(tc.id, tc.name, tc.arguments, is_read_only=...)

# 流结束后，收集所有工具结果
completed = executor.get_completed_results()   # 已完成的
remaining = executor.get_remaining_results()   # 等待剩余的
all_results = completed + remaining
```

### 与 Claude Code 的对比

| 特性 | Claude Code | OpenClaw Lite v2 |
|------|------------|-----------------|
| 执行时机 | 流式输出中立即执行 | 同上 |
| 并行策略 | 基于 `isConcurrencySafe()` | 基于 `read_only` 标记 |
| 线程模型 | Node.js 异步 | Python ThreadPoolExecutor |
| 中断处理 | `abortController.signal` | `executor.discard()` |
| Fallback 清理 | `streamingToolExecutor.discard()` | 同上 |

## 流式模式下的输出体验

```
你> 帮我看看 config.json 的内容

  🔄 调用 openai/qwen3-max（迭代 1/25）...
🦞 好的，我来读取 config.json 的内容。     ← 文本边生成边显示
  🔧 read(file_path=config.json)           ← tool_call 解析完立即执行
  ✅ read: OK
  🔄 调用 openai/qwen3-max（迭代 2/25）...
🦞 config.json 的内容如下：...              ← 第二轮流式输出
  📊 Token: 1234
```

首字延迟从 5-30 秒降到 < 1 秒，工具执行与模型输出重叠，整体响应时间显著缩短。