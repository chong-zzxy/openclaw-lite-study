# 核心对话循环与容错机制 深度解析

本文档拆解 OpenClaw Lite v2 的 Agent 循环设计，对标 Claude Code `src/query.ts` 中的 `queryLoop()`。

## Agent 循环的本质

所有 AI Agent 的核心都是同一个循环：

```python
while True:
    response = llm.chat(messages, tools)
    if response.has_tool_calls:
        for tc in response.tool_calls:
            result = execute_tool(tc)
            messages.append(tool_result)
        continue
    else:
        return response.text
```

v2 在这个骨架上叠加了 5 层容错和优化机制。

## 完整流程

```
run_agent_turn()
  │
  ├── 1. 准备阶段
  │     ├── 加载长期记忆 → 注入 system prompt
  │     ├── 加载会话历史
  │     └── 构建工具 schema
  │
  ├── 2. Compaction（token 驱动）
  │     ├── estimate_messages_tokens(history) > threshold?
  │     ├── 第一层: truncate_tool_results() — 截断长工具输出
  │     └── 第二层: compact_history() — LLM 摘要压缩旧消息
  │
  ├── 3. Agent 循环（_run_streaming_tool_loop）
  │     │
  │     │  while iteration < max_iterations:
  │     │    ├── 构建 LLM 消息列表
  │     │    ├── stream_chat() 流式调用
  │     │    │     ├── text_delta → 实时打印
  │     │    │     ├── tool_call_done → 权限检查 → 提交执行器
  │     │    │     └── done → 记录 finish_reason
  │     │    │
  │     │    ├── 无工具调用?
  │     │    │     ├── finish_reason == "length"?
  │     │    │     │     └── 注入恢复指令 → continue
  │     │    │     └── 返回最终文本
  │     │    │
  │     │    └── 有工具调用?
  │     │          ├── 收集执行器结果
  │     │          ├── 构建 tool messages
  │     │          └── continue（带结果再问 LLM）
  │     │
  │     └── 达到 max_iterations → 返回错误
  │
  ├── 4. Failover 包裹
  │     ├── 主模型失败 → FailoverError
  │     └── 切换备用模型重试整个循环
  │
  └── 5. 收尾
        ├── 保存 assistant 回复到 SessionStore
        └── 按需提取长期记忆
```

## 容错机制 1: Token 驱动的多层压缩

### 为什么用 token 而不是消息数

v1 用消息数量判断（> 40 条就压缩），但 10 条消息可能是 500 token 也可能是 50000 token。一次 `ls -la` 大目录的工具结果就可能占 10000 token。

v2 用 `estimate_messages_tokens()` 估算实际 token 数，阈值默认 80000。

### 第一层: 工具结果截断

对标 Claude Code 的 `applyToolResultBudget()`。很多工具输出信息密度低（如完整的文件列表、大段日志），截断后不影响 Agent 理解。

```python
def truncate_tool_results(messages):
    for msg in messages:
        if msg.role == "tool" and len(msg.content) > 8000:
            # 保留头 3000 + 尾 2000 字符，中间省略
            msg.content = head + "[... 截断 ...]" + tail
```

### 第二层: LLM 摘要压缩

保留最近 10 条消息，将更早的消息用 LLM 生成摘要替代。摘要保留关键信息：用户请求、决策结论、文件路径、错误和修复记录。

两层组合的效果：先截断工具结果（轻量，不调 LLM），如果仍超阈值再做摘要压缩（重量级）。大多数情况下第一层就够了。


## 容错机制 2: max_output_tokens 恢复

对标 Claude Code 的 `isWithheldMaxOutputTokens` + recovery 逻辑。

当模型输出被截断时（`finish_reason == "length"`），不是直接返回不完整的文本，而是注入一条恢复指令让模型继续：

```python
if finish_reason == "length" and recovery_count < MAX_RECOVERY:
    turn_messages.append({"role": "assistant", "content": text_buffer})
    turn_messages.append({
        "role": "user",
        "content": "Output token limit hit. Resume directly — "
                   "no apology, no recap. Pick up mid-thought."
    })
    continue  # 再调一次 LLM
```

最多恢复 2 次（`MAX_OUTPUT_RECOVERY = 2`），防止无限循环。恢复指令要求模型"直接继续，不要道歉，不要复述"，避免浪费 token。

Claude Code 还有一个更激进的优化：先用 8k 的 max_output_tokens 上限，如果被截断就升级到 64k 重试。我们简化为直接用恢复指令。

## 容错机制 3: Failover

主模型 API 调用失败时（网络错误、速率限制、认证失败等），自动切换到备用模型重试。

```python
def agent_loop(provider):
    return _run_streaming_tool_loop(provider, ...)

if fallback_refs:
    fallbacks = [create_provider(fb) for fb in fallback_refs]
    result, _ = run_with_failover(primary, fallbacks, agent_loop)
```

关键：failover 重试的是整个 `agent_loop`，不是单次 API 调用。因为切换模型后，之前的工具调用结果可能需要重新解释。

`FailoverError` 区分可重试和不可重试错误：
- 可重试：`rate_limit`, `api_error`, `overloaded`
- 不可重试：`auth_failed`（API key 错误）

## 容错机制 4: 中断处理

对标 Claude Code 的 `abortController.signal` 机制。

```python
# main.py: SIGINT handler
def handler(signum, frame):
    if store.get().is_running:
        store.request_abort()  # 设置 abort 标志

# agent.py: 循环中检查
for chunk in provider.stream_chat(...):
    if store.is_aborted:
        executor.discard()  # 丢弃待执行的工具
        return "[用户中断]"
```

中断是协作式的：设置标志 → 循环检查 → 优雅退出。不是暴力 kill 进程。

## 容错机制 5: 工具调用上限

`max_tool_iterations`（默认 25）防止 Agent 陷入无限工具调用循环。达到上限后返回错误消息，不会无限消耗 token。

## 与 Claude Code queryLoop 的对比

| 特性 | Claude Code | OpenClaw Lite v2 |
|------|------------|-----------------|
| 循环结构 | AsyncGenerator + while(true) | while + iteration counter |
| 压缩层数 | 4 层（snip + micro + collapse + auto） | 2 层（工具截断 + LLM 摘要） |
| 压缩触发 | API 返回的 input_tokens | 字符估算 + API usage |
| 输出恢复 | 8k→64k 升级 + 恢复指令 | 恢复指令（最多 2 次） |
| Failover | 模型 + auth profile 轮换 | 模型列表轮换 |
| 中断 | AbortController | StateStore.abort_requested |
| 投机执行 | SpeculationState | 未实现 |
| Stop hooks | handleStopHooks() | 未实现 |
| 响应式压缩 | 413 → 紧急压缩重试 | 未实现 |

Claude Code 的 `queryLoop` 有约 1500 行代码，处理了大量边界情况（缓存编辑、上下文折叠、token 预算、任务摘要等）。我们提取了最核心的 5 个容错机制，用约 200 行代码实现。