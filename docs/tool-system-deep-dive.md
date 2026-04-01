# 工具系统设计 深度解析

本文档拆解 OpenClaw Lite v2 的工具系统，对标 Claude Code 的 `Tool` 接口和 `ToolRegistry`。

## 工具系统的角色

LLM 本身只能生成文本。工具系统赋予它"行动能力"——读文件、写文件、执行命令、搜索网页。

工作流程：
1. 注册阶段：每个工具提供 name、description、parameters（JSON Schema）
2. 调用阶段：LLM 根据 description 决定调用哪个工具，根据 parameters 生成参数
3. 执行阶段：ToolRegistry 根据 name 找到 handler，传入参数执行
4. 回传阶段：执行结果作为 tool message 回传给 LLM

## ToolDefinition 设计

```python
@dataclass
class ToolDefinition:
    name: str                    # LLM 通过这个名字调用
    description: str             # LLM 根据描述决定何时使用
    parameters: dict[str, Any]   # JSON Schema，LLM 据此生成参数
    handler: Callable[..., ToolResult]  # 实际执行函数
    read_only: bool = False      # v2: 只读标记
    destructive: bool = False    # v2: 破坏性标记
```

v2 新增的 `read_only` 和 `destructive` 标记有两个作用：
1. 驱动权限系统：read_only 工具在所有模式下自动放行
2. 驱动并行策略：read_only 工具可以在 StreamingToolExecutor 中并行执行

## 7 个内置工具

### read（只读）
读取文件内容，支持行范围。路径经过沙箱校验，超 100K 字符自动截断。

```python
read(file_path="src/main.py", start_line=10, end_line=20)
```

### write
创建或覆盖文件，自动创建父目录。

```python
write(file_path="output.txt", content="hello world")
```

### edit
查找替换编辑。要求 old_string 在文件中唯一匹配，防止误改多处。

```python
edit(file_path="config.json", old_string='"name": "old"', new_string='"name": "new"')
```

### exec（破坏性）
执行 shell 命令，cwd 设为 workspace。带超时（默认 30s），输出超 50K 截断。

```python
exec(command="git status", timeout=30)
```

### glob（只读，v2 新增）
按 glob 模式搜索文件，支持 `**` 递归。结果限制 200 条。

```python
glob(pattern="**/*.py", path="src")
```

### grep（只读，v2 新增）
按正则搜索文件内容。跳过二进制和大文件（> 1MB），结果限制 100 条。

```python
grep(pattern="def main", glob="**/*.py")
```

### web_search（只读）
DuckDuckGo HTML 版搜索，纯 HTTP 请求，不依赖 JS。优先用 BeautifulSoup 解析，bs4 不可用时正则兜底。

```python
web_search(query="Python asyncio tutorial", max_results=5)
```


## ToolRegistry 设计

```python
class ToolRegistry:
    def register(self, tool: ToolDefinition)
    def get_tool(self, name: str) -> ToolDefinition | None
    def get_available_tools(self, allow, deny) -> list[ToolDefinition]
    def get_tools_for_llm(self, allow, deny) -> list[dict]  # OpenAI schema
    def execute(self, name: str, arguments: dict) -> ToolResult
    def is_read_only(self, name: str) -> bool   # v2
    def is_destructive(self, name: str) -> bool  # v2
```

`get_tools_for_llm()` 生成 OpenAI function calling 格式的 JSON Schema，直接传给 API：

```json
{
  "type": "function",
  "function": {
    "name": "read",
    "description": "Read file contents within the workspace...",
    "parameters": {
      "type": "object",
      "properties": {
        "file_path": {"type": "string", "description": "File path to read"}
      },
      "required": ["file_path"]
    }
  }
}
```

## 沙箱安全（sandbox.py）

所有文件工具的路径都经过 `resolve_safe_path()` 校验：

```python
def resolve_safe_path(file_path: str) -> Path:
    ws = get_workspace()
    resolved = (ws / file_path).resolve()
    resolved.relative_to(ws)  # 不在 workspace 内则抛 ValueError
    return resolved
```

防止通过 `../../etc/passwd` 等路径逃逸出 workspace。exec 工具的 cwd 也固定为 workspace。

## 工具执行带钩子

工具不是直接执行的，而是经过 hooks 管道：

```
LLM 请求调用工具
  → before_tool_call hooks（可拦截/修改参数）
  → 实际执行 handler
  → after_tool_call hooks（可修改结果）
  → 结果回传 LLM
```

## 与 Claude Code 的对比

| 特性 | Claude Code | OpenClaw Lite v2 |
|------|------------|-----------------|
| 工具数量 | 30+ | 7 |
| 工具接口 | 完整的 Tool class（prompt, render, validate...） | ToolDefinition dataclass |
| 分类标记 | isReadOnly() / isDestructive() / isConcurrencySafe() | read_only / destructive |
| 参数校验 | Zod schema + validateInput() | JSON Schema（LLM 端校验） |
| 结果渲染 | React 组件（Ink） | 纯文本 print |
| MCP 集成 | 完整 MCP 客户端 | 未实现 |
| 动态注册 | 插件系统 + MCP 动态发现 | 启动时静态注册 |

Claude Code 的工具接口非常丰富（prompt 生成、UI 渲染、权限检查、输入校验、活动描述等），我们只保留了最核心的 name/description/parameters/handler 四要素加 read_only/destructive 两个标记。