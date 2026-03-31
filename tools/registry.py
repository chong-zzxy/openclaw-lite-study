"""
工具注册表和调度系统。
对应 OpenClaw: src/agents/tool-catalog.ts + src/agents/pi-tools.ts
"""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolResult:
    """工具执行结果。每个工具执行后都返回这个结构。"""
    success: bool            # 是否成功
    output: str              # 输出内容（成功时为结果，失败时可能为空）
    error: str | None = None # 错误信息（成功时为 None）


@dataclass
class ToolDefinition:
    """
    工具定义。每个工具注册时需要提供：
    - name: 工具名，LLM 通过这个名字调用工具
    - description: 工具描述，LLM 根据描述决定何时使用
    - parameters: JSON Schema 格式的参数定义，LLM 据此生成调用参数
    - handler: 实际执行函数，接收 LLM 传来的参数，返回 ToolResult
    """
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema，定义工具接受的参数
    handler: Callable[..., ToolResult]


class ToolRegistry:
    """
    工具注册表。

    职责：
    1. 管理所有工具的注册和查找
    2. 根据 allow/deny 列表过滤可用工具
    3. 生成 OpenAI function calling 格式的 schema（发给 LLM）
    4. 根据工具名分发调用并返回结果

    LLM 不直接调用工具，而是返回"我想调用 xxx 工具，参数是 yyy"，
    由 ToolRegistry.execute() 负责实际执行并返回结果。
    """
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition):
        """注册一个工具。同名工具会被覆盖。"""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> ToolDefinition | None:
        """按名称查找工具"""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """列出所有已注册的工具名"""
        return list(self._tools.keys())

    def get_available_tools(
        self, allow: list[str], deny: list[str],
    ) -> list[ToolDefinition]:
        """
        根据 allow/deny 列表过滤工具。
        - allow 非空时，只返回在 allow 中的工具
        - deny 中的工具会被排除
        - 两者同时存在时，deny 优先级更高
        """
        result = []
        for name, tool in self._tools.items():
            if deny and name in deny:
                continue
            if allow and name not in allow:
                continue
            result.append(tool)
        return result

    def get_tools_for_llm(
        self, allow: list[str], deny: list[str],
    ) -> list[dict]:
        """
        生成 OpenAI function calling 格式的工具 schema。
        这个列表会作为 tools 参数传给 LLM API，
        LLM 根据 schema 中的 name/description/parameters 决定调用哪个工具。
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self.get_available_tools(allow, deny)
        ]

    def execute(self, name: str, arguments: dict) -> ToolResult:
        """
        执行工具。根据 name 找到对应的 handler，传入 arguments 调用。
        arguments 由 LLM 生成，格式与 parameters JSON Schema 对应。
        工具不存在或执行异常时返回失败的 ToolResult。
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(False, "", f"Unknown tool: {name}")
        try:
            return tool.handler(**arguments)
        except Exception as e:
            return ToolResult(False, "", str(e))


def create_default_registry() -> ToolRegistry:
    """
    创建包含所有内置工具的注册表。
    延迟导入各工具模块，避免循环依赖。
    """
    from tools.read_tool import create_read_tool
    from tools.write_tool import create_write_tool
    from tools.edit_tool import create_edit_tool
    from tools.exec_tool import create_exec_tool
    from tools.web_search import create_web_search_tool

    reg = ToolRegistry()
    reg.register(create_read_tool())
    reg.register(create_write_tool())
    reg.register(create_edit_tool())
    reg.register(create_exec_tool())
    reg.register(create_web_search_tool())
    return reg
