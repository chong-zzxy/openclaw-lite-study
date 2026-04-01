"""
工具注册表。v2: 新增 read_only/destructive 标记，对标 Claude Code 的 Tool 接口。
"""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolResult:
    success: bool
    output: str
    error: str | None = None


@dataclass
class ToolDefinition:
    """
    工具定义。v2 新增：
    - read_only: 只读工具，在所有权限模式下自动放行，可并行执行
    - destructive: 破坏性工具，即使在 acceptEdits 模式下也需确认
    """
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., ToolResult]
    read_only: bool = False
    destructive: bool = False


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition):
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_available_tools(self, allow: list[str], deny: list[str]) -> list[ToolDefinition]:
        result = []
        for name, tool in self._tools.items():
            if deny and name in deny:
                continue
            if allow and name not in allow:
                continue
            result.append(tool)
        return result

    def get_tools_for_llm(self, allow: list[str], deny: list[str]) -> list[dict]:
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
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(False, "", f"Unknown tool: {name}")
        try:
            return tool.handler(**arguments)
        except Exception as e:
            return ToolResult(False, "", str(e))

    def is_read_only(self, name: str) -> bool:
        tool = self._tools.get(name)
        return tool.read_only if tool else False

    def is_destructive(self, name: str) -> bool:
        tool = self._tools.get(name)
        return tool.destructive if tool else False


def create_default_registry() -> ToolRegistry:
    from tools.read_tool import create_read_tool
    from tools.write_tool import create_write_tool
    from tools.edit_tool import create_edit_tool
    from tools.exec_tool import create_exec_tool
    from tools.glob_tool import create_glob_tool
    from tools.grep_tool import create_grep_tool
    from tools.web_search import create_web_search_tool

    reg = ToolRegistry()
    reg.register(create_read_tool())
    reg.register(create_write_tool())
    reg.register(create_edit_tool())
    reg.register(create_exec_tool())
    reg.register(create_glob_tool())
    reg.register(create_grep_tool())
    reg.register(create_web_search_tool())
    return reg
