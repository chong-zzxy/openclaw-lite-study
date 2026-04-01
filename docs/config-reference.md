# config.json 配置说明

## identity — 机器人身份

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | string | 机器人显示名称 |
| `theme` | string | 角色主题，用于生成系统提示词 |
| `emoji` | string | 标识表情符号 |
| `system_prompt_extra` | string | 追加到系统提示词末尾的自定义内容，留空则不追加 |

## agent — 代理核心配置

| 字段 | 类型 | 说明 |
|------|------|------|
| `workspace` | string | 工作目录路径，工具操作的根目录 |
| `model.provider` | string | 模型服务商名称，用于匹配 API Key 和请求端点 |
| `model.model` | string | 主模型名称 |
| `fallback_models` | array | 备用模型列表，主模型请求失败时按顺序尝试，每项包含 `provider` 和 `model` |
| `timeout_seconds` | int | 单次模型请求超时时间（秒） |
| `max_tool_iterations` | int | 单轮对话中最大工具调用次数，防止无限循环 |
| `compaction_threshold` | int | 对话消息数达到此阈值时触发上下文压缩 |

## hooks — 钩子配置

| 字段 | 类型 | 说明 |
|------|------|------|
| `enable_logging` | bool | 是否启用工具调用日志输出 |
| `enable_dangerous_command_guard` | bool | 是否拦截危险命令（如 `rm -rf`）并要求确认 |
| `output_truncation_chars` | int | 工具输出超过此字符数时自动截断 |

## tools — 工具权限控制

| 字段 | 类型 | 说明 |
|------|------|------|
| `allow` | array | 允许使用的工具列表 |
| `deny` | array | 禁止使用的工具列表，优先级高于 `allow` |

## session — 会话管理

| 字段 | 类型 | 说明 |
|------|------|------|
| `store_path` | string | 会话持久化存储文件路径 |
| `history_limit` | int | 保留的历史会话最大条数 |
| `reset_triggers` | array | 触发会话重置的命令列表 |
