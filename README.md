# OpenClaw Lite

> 🎓 学习项目 — 仅供学习和参考，不建议用于生产环境。

OpenClaw Lite 是一个用 Python 实现的 AI Agent 学习项目，通过逆向分析 [Claude Code](https://github.com/anthropics/claude-code) 源码，提炼并复现了其核心架构设计。

v2 版本受 Claude Code 源码启发，新增了流式输出、流式工具执行、四级权限系统、Token 驱动的多层压缩、全局状态管理等机制。

## 核心架构

v2 的核心循环（对标 Claude Code 的 `queryLoop()`）：

```
用户输入 → main.py → agent.py: run_agent_turn()
  ├── 1. 加载配置、构建 system prompt、注入长期记忆
  ├── 2. Token 驱动的 Compaction（工具结果截断 + LLM 摘要）
  ├── 3. 流式 Agent 循环（带 failover）:
  │     ├── stream_chat() 流式调用 LLM
  │     ├── text_delta → 实时打印到终端
  │     ├── tool_call_done → 权限检查 → 提交流式执行器
  │     │   ├── 只读工具并行执行（ThreadPoolExecutor）
  │     │   └── 写工具串行执行（防竞态）
  │     ├── 收集工具结果 → 回传 LLM → 继续循环
  │     ├── finish_reason=length → 注入恢复指令（最多 2 次）
  │     └── 无工具调用 → 结束循环
  ├── 4. 保存会话历史
  └── 5. 按需提取长期记忆
```

## 项目结构

```
openclaw-lite/
├── main.py              # CLI 入口（REPL + 命令系统）
├── agent.py             # Agent 循环核心（流式 + 非流式双模式）
├── config.py            # 配置加载（含权限、流式、token 阈值）
├── config.json          # 配置文件
├── state.py             # 全局状态管理（AppState + StateStore）
├── token_counter.py     # Token 计数（估算 + API 精确值）
├── permissions.py       # 四级权限系统
├── streaming.py         # 流式工具执行器（StreamingToolExecutor）
├── session.py           # 会话管理（历史持久化，原子写入）
├── identity.py          # 身份/风格系统
├── system_prompt.py     # System Prompt 构建
├── hooks.py             # 工具调用钩子（before/after）
├── failover.py          # 模型故障转移
├── compaction.py        # 多层上下文压缩（token 驱动）
├── memory.py            # 长期记忆（三层策略）
├── tools/
│   ├── registry.py      # 工具注册表（含 read_only/destructive 标记）
│   ├── sandbox.py       # 路径沙箱校验
│   ├── read_tool.py     # 文件读取（read_only）
│   ├── write_tool.py    # 文件写入
│   ├── edit_tool.py     # 文件编辑（查找替换）
│   ├── exec_tool.py     # Shell 命令执行（destructive）
│   ├── glob_tool.py     # 文件搜索（read_only）
│   ├── grep_tool.py     # 内容搜索（read_only）
│   └── web_search.py    # Web 搜索（read_only）
├── providers/
│   ├── base.py          # LLM Provider 抽象（含 StreamChunk）
│   └── openai_provider.py  # OpenAI 兼容实现（含 stream_chat）
├── docs/
│   ├── streaming-deep-dive.md      # 流式输出与流式工具执行
│   ├── permissions-deep-dive.md    # 权限系统设计
│   ├── query-loop-deep-dive.md     # 核心对话循环与容错机制
│   ├── tool-system-deep-dive.md    # 工具系统设计
│   ├── state-management-deep-dive.md  # 状态管理设计
│   └── memory-deep-dive.md         # 记忆系统（已有，已更新）
├── .env.example
└── workspace-example/
    ├── SOUL.md
    └── AGENTS.md
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

## 命令列表

| 命令 | 说明 |
|------|------|
| `/new` `/reset` | 重置会话（会先提取长期记忆） |
| `/mode [模式]` | 查看/切换权限模式（default/acceptEdits/plan/bypassPermissions） |
| `/stats` | 查看累计 Token、压缩次数等统计 |
| `/compact` | 手动压缩历史 |
| `/remember <文本>` | 主动记住一条信息 |
| `/memory` | 查看长期记忆 |
| `/model <provider/model>` | 运行时切换模型 |
| `/sessions` | 查看会话状态 |
| `/help` | 显示帮助 |
| `/quit` `/exit` | 退出 |


## v2 新增特性

### 1. 流式输出
模型边生成边打印到终端，不再等完整响应。Provider 新增 `stream_chat()` 方法，yield `StreamChunk` 对象（text_delta / tool_call_start / tool_call_delta / tool_call_done / usage / done）。详见 [流式输出深度解析](docs/streaming-deep-dive.md)。

### 2. 流式工具执行
`StreamingToolExecutor` 在模型还在流式输出时，已解析完的 tool_call 立即提交执行。只读工具之间并行（ThreadPoolExecutor），写工具串行。详见 [流式输出深度解析](docs/streaming-deep-dive.md)。

### 3. 四级权限系统
- `default`：写操作需用户确认，只读工具和安全命令自动放行
- `acceptEdits`：文件编辑自动放行，shell 命令仍需确认
- `plan`：只规划不执行，所有写操作被拒绝
- `bypassPermissions`：跳过所有权限检查

工具标记 `read_only` / `destructive`，影响权限判断。详见 [权限系统深度解析](docs/permissions-deep-dive.md)。

### 4. 核心循环容错机制
- Token 驱动的多层压缩（工具结果截断 + LLM 摘要）
- max_output_tokens 恢复（输出截断时自动注入继续指令）
- Failover（主模型失败切换备用）
- Ctrl+C 优雅中断

详见 [核心对话循环深度解析](docs/query-loop-deep-dive.md)。

### 5. 工具系统增强
- 工具标记 `read_only` / `destructive`，驱动权限判断和并行策略
- 新增 `glob`（文件搜索）和 `grep`（内容搜索）工具
- 7 个内置工具

详见 [工具系统深度解析](docs/tool-system-deep-dive.md)。

### 6. 全局状态管理
`StateStore` 集中管理运行时状态（权限模式、token 统计、中断信号等），线程安全，支持监听器。详见 [状态管理深度解析](docs/state-management-deep-dive.md)。


## 配置说明

```jsonc
{
  "identity": {
    "name": "Clawd",
    "theme": "helpful coding assistant",
    "emoji": "🦞"
  },
  "agent": {
    "workspace": "~/.openclaw-lite/workspace",
    "model": { "provider": "dashscope", "model": "qwen3-max" },
    "fallback_models": [{ "provider": "dashscope", "model": "qwen3.5-plus" }],
    "max_tool_iterations": 25,
    "compaction_token_threshold": 80000,  // token 数触发压缩
    "compaction_keep_recent": 10,
    "stream": true                        // 流式输出开关
  },
  "permissions": {
    "mode": "default",                    // 权限模式
    "auto_allow_read_tools": true
  },
  "hooks": {
    "enable_logging": true,
    "enable_dangerous_command_guard": true,
    "output_truncation_chars": 50000
  },
  "tools": {
    "allow": ["exec", "read", "write", "edit", "glob", "grep", "web_search"],
    "deny": []
  },
  "session": {
    "store_path": "~/.openclaw-lite/sessions.json",
    "history_limit": 100
  }
}
```


## 对标 Claude Code 源码映射

| 本项目 | Claude Code 原版 |
|--------|-----------------|
| `agent.py` `_run_streaming_tool_loop` | `src/query.ts` `queryLoop()` |
| `streaming.py` `StreamingToolExecutor` | `StreamingToolExecutor` class |
| `permissions.py` | `src/utils/permissions/PermissionMode.ts` |
| `state.py` `StateStore` | `src/state/AppStateStore.ts` |
| `token_counter.py` | `tokenCountWithEstimation()` |
| `compaction.py` | `autocompact` + `applyToolResultBudget` |
| `tools/registry.py` `read_only/destructive` | `Tool.isReadOnly()` / `Tool.isDestructive()` |
| `providers/base.py` `StreamChunk` | `StreamEvent` type |
| `hooks.py` | `src/utils/hooks/` |
| `failover.py` | `FallbackTriggeredError` handling in queryLoop |
| `system_prompt.py` | `src/agents/system-prompt.ts` |
| `memory.py` | 无直接对标（Claude Code 用 CLAUDE.md） |


## 深度解析文档

- [流式输出与流式工具执行](docs/streaming-deep-dive.md)
- [权限系统设计](docs/permissions-deep-dive.md)
- [核心对话循环与容错机制](docs/query-loop-deep-dive.md)
- [工具系统设计](docs/tool-system-deep-dive.md)
- [状态管理设计](docs/state-management-deep-dive.md)
- [记忆系统深度解析](docs/memory-deep-dive.md)
