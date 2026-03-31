# OpenClaw Lite

> 🎓 学习项目 — 仅供学习和参考，不建议用于生产环境。

OpenClaw 简化版 — 用 Python 实现的 AI Agent 学习项目。

聚焦核心概念：Agent 循环、工具调用、钩子系统、模型 Failover、历史压缩、身份/风格。

## 项目背景

本项目是 [OpenClaw](https://github.com/anthropics/openclaw)（TypeScript 实现的 AI Agent 框架）的 Python 简化复刻版。目标不是做一个生产级框架，而是用最少的代码把 AI Agent 的核心机制讲清楚。

如果你想理解"一个 AI Agent 到底是怎么运转的"，这个项目就是答案。

## 核心架构

整个项目围绕一个核心循环展开：

```
用户输入 → main.py → agent.py: run_agent_turn()
  ├── 1. 加载配置、构建 system prompt
  ├── 2. Compaction（历史过长时压缩）
  ├── 3. Agent 循环（带 failover）:
  │     ├── 调用 LLM
  │     ├── 如果有工具调用:
  │     │   ├── before_tool_call 钩子（可拦截）
  │     │   ├── 执行工具
  │     │   ├── after_tool_call 钩子（可修改结果）
  │     │   └── 结果回传 LLM → 继续循环
  │     └── 如果是文本回复 → 结束循环
  ├── 4. 保存会话历史
  └── 5. 返回回复
```

用伪代码表示 Agent 循环的本质：

```python
while True:
    response = llm.chat(messages, tools)
    if response.has_tool_calls:
        for tool_call in response.tool_calls:
            result = execute_tool(tool_call)
            messages.append(tool_result)
        continue  # 带着工具结果再问一次 LLM
    else:
        return response.text  # LLM 给出了最终回复
```

这就是所有 AI Agent 的核心——一个"调用 LLM → 执行工具 → 回传结果"的循环。本项目在此基础上叠加了三个增强机制：Hooks、Failover、Compaction。


## 项目结构

```
openclaw-lite/
├── main.py              # CLI 入口（REPL 交互循环）
├── agent.py             # Agent 循环核心
├── config.py            # 配置加载（dataclass + JSON + .env）
├── config.json          # 配置文件
├── session.py           # 会话管理（历史持久化，原子写入）
├── identity.py          # 身份/风格系统
├── system_prompt.py     # System Prompt 构建
├── hooks.py             # 工具调用钩子（before/after）
├── failover.py          # 模型故障转移
├── compaction.py        # 历史压缩
├── memory.py            # 长期记忆（三层策略：显式/检测/批量提取）
├── tools/
│   ├── registry.py      # 工具注册表和调度
│   ├── read_tool.py     # 文件读取
│   ├── write_tool.py    # 文件写入
│   ├── edit_tool.py     # 文件编辑（查找替换）
│   ├── exec_tool.py     # Shell 命令执行
│   └── web_search.py    # Web 搜索（DuckDuckGo）
├── providers/
│   ├── base.py          # LLM Provider 抽象基类
│   └── openai_provider.py  # OpenAI 兼容 API 实现
├── .env.example         # 环境变量模板
└── workspace-example/
    ├── SOUL.md           # Agent 人格定义示例
    └── AGENTS.md         # 项目指引示例
```


## 快速开始

```bash
# 1. 安装依赖
pip install openai python-dotenv requests beautifulsoup4

# 2. 配置密钥
cp .env.example .env
# 编辑 .env，填入你的 DASHSCOPE_API_KEY

# 3. 运行
python main.py
```

进入 REPL 后可以直接对话，也可以用内置命令：

| 命令 | 说明 |
|------|------|
| `/new` `/reset` | 重置会话 |
| `/remember <文本>` | 主动记住一条信息（存入长期记忆） |
| `/memory` | 查看当前长期记忆 |
| `/model dashscope/qwen-max` | 运行时切换模型 |
| `/sessions` | 查看会话状态 |
| `/quit` `/exit` | 退出 |


## 六大核心模块详解

### 1. Agent 循环 (`agent.py`)

这是整个项目的心脏。`run_agent_turn()` 函数执行一个完整的对话轮次：

1. 构建 system prompt（身份 + 工具描述 + 工作区上下文）
2. 检查是否需要历史压缩
3. 进入工具调用循环（`_run_tool_loop`）
4. 保存会话历史并返回回复

工具调用循环的关键在于：LLM 每次返回时，要么给出文本回复（循环结束），要么请求调用工具（执行工具后把结果塞回消息列表，再次调用 LLM）。这个循环最多执行 `max_tool_iterations` 次（默认 20 次），防止无限循环。

`execute_tool_with_hooks()` 是工具执行的入口，它在实际执行前后分别运行 before/after 钩子，实现了拦截、参数修改、结果处理等能力。


### 2. 钩子系统 (`hooks.py`)

钩子是工具调用的拦截器，分两个时机：

- `before_tool_call`：工具执行前。可以放行、拦截（`proceed=False`）、或修改参数
- `after_tool_call`：工具执行后。可以拿到结果并修改它

`HookRunner` 维护两个钩子列表，按注册顺序依次执行。做了错误隔离——单个钩子抛异常只打印警告，不影响其他钩子。

内置三个钩子：

| 钩子 | 类型 | 作用 |
|------|------|------|
| 日志钩子 | before + after | 记录每次工具调用的耗时 |
| 危险命令拦截 | before | 拦截包含 `rm -rf`、`sudo` 等关键词的 shell 命令 |
| 输出截断 | after | 超过阈值的工具输出自动截断，防止撑爆 context window |


### 3. 模型 Failover (`failover.py`)

当主模型（如 qwen-max）调用失败时，自动切换到备用模型（如 qwen-plus）重试。

`run_with_failover()` 的逻辑：
1. 用主 provider 执行 `run_fn`
2. 如果抛出 `FailoverError` 且 `retryable=True`，尝试下一个 provider
3. 所有 provider 都失败则抛出最后一个错误

`FailoverError` 包含 `reason` 字段区分错误类型（`rate_limit`、`auth_failed`、`overloaded` 等），以及 `retryable` 标记是否可重试。不可重试的错误会直接抛出，不触发 failover。


### 4. 历史压缩 (`compaction.py`)

当会话消息数超过阈值（默认 40 条）时，自动压缩旧消息：

1. `should_compact()` 检查是否需要压缩（只计算 user/assistant 消息数）
2. `compact_history()` 执行压缩：
   - 保留最近 10 条消息不动
   - 将更早的消息用 LLM 生成一段摘要
   - 摘要保留关键信息：用户请求、决策结论、文件路径、错误和修复记录
   - 用一条 assistant 消息替换所有旧消息

压缩失败时保留原始历史，不会丢数据。


### 5. 工具系统 (`tools/`)

工具系统由两部分组成：

`ToolRegistry`（注册表）管理所有工具的注册、查找和调度。支持 allow/deny 列表过滤，`get_tools_for_llm()` 生成 OpenAI function calling 格式的 JSON Schema。

五个内置工具：

| 工具 | 功能 | 关键细节 |
|------|------|----------|
| `read` | 读取文件 | 支持行范围，超过 100K 字符自动截断 |
| `write` | 写入文件 | 自动创建父目录 |
| `edit` | 查找替换 | 要求 old_string 在文件中唯一匹配，防止误改 |
| `exec` | 执行命令 | 带超时（默认 30s），输出超 50K 截断 |
| `web_search` | 网页搜索 | DuckDuckGo HTML 版，纯 HTTP 请求 |


### 6. System Prompt 构建 (`system_prompt.py`)

`build_system_prompt()` 动态拼装 system prompt，包含以下部分：

1. 身份声明（名字、主题、emoji）
2. 可用工具列表及描述
3. 工具调用风格指引（默认不叙述，直接调用）
4. 安全规则（优先安全和人类监督）
5. 工作区路径
6. 当前时间
7. 工作区上下文文件（自动加载 `SOUL.md`、`AGENTS.md`、`TOOLS.md`）
8. 用户自定义的额外 prompt

其中 `SOUL.md` 定义 Agent 人格，`AGENTS.md` 定义项目编码规范，这些文件放在工作区目录下会被自动读取并注入 system prompt。


## 配置说明

所有配置在 `config.json` 中，支持以下配置段：

```jsonc
{
  "identity": {          // Agent 身份
    "name": "Clawd",     // 名字
    "theme": "helpful coding assistant",  // 主题描述
    "emoji": "🦞",       // 前缀 emoji
    "system_prompt_extra": ""  // 额外注入 system prompt 的内容
  },
  "agent": {
    "workspace": "~/.openclaw-lite/workspace",  // 工作区目录
    "model": { "provider": "dashscope", "model": "qwen3-max" },
    "fallback_models": [{ "provider": "dashscope", "model": "qwen3.5-plus" }],
    "timeout_seconds": 300,
    "max_tool_iterations": 20,  // 单轮最大工具调用次数
    "compaction_threshold": 40  // 触发历史压缩的消息数
  },
  "hooks": {
    "enable_logging": true,
    "enable_dangerous_command_guard": true,
    "output_truncation_chars": 50000
  },
  "tools": {
    "allow": ["exec", "read", "write", "edit", "web_search"],
    "deny": []
  },
  "session": {
    "store_path": "~/.openclaw-lite/sessions.json",
    "history_limit": 50,
    "reset_triggers": ["/new", "/reset"]
  }
}
```

密钥通过 `.env` 文件管理（基于 python-dotenv），不要提交到版本控制。


## 对应 OpenClaw 源码映射

| 本项目 | OpenClaw 原版 |
|--------|--------------|
| `agent.py` | `src/agents/agent-command.ts` + `src/agents/pi-embedded-runner/run.ts` |
| `hooks.py` | `src/agents/pi-tools.before-tool-call.ts` + `src/hooks/` |
| `failover.py` | `src/agents/model-fallback.ts` |
| `compaction.py` | `src/agents/compaction.ts` |
| `tools/registry.py` | `src/agents/tool-catalog.ts` + `src/agents/pi-tools.ts` |
| `system_prompt.py` | `src/agents/system-prompt.ts` |
| `identity.py` | `src/agents/identity.ts` |
| `session.py` | `src/config/sessions/store.ts` |
| `config.py` | `src/config/config.ts` |


## 深度解析文档

- [记忆系统深度解析](docs/memory-deep-dive.md) — 详细拆解消息存储、历史压缩、记忆生命周期的完整实现思路
