"""文件编辑工具（查找替换）。对应 OpenClaw: src/agents/pi-tools.ts edit"""

from pathlib import Path
from tools.registry import ToolDefinition, ToolResult
from tools.sandbox import resolve_safe_path


def edit_file(file_path: str, old_string: str, new_string: str) -> ToolResult:
    """
    查找替换编辑文件。路径经过沙箱校验。
    old_string 必须在文件中唯一匹配（防止误改多处），否则报错。
    """
    try:
        p = resolve_safe_path(file_path)
        if not p.exists():
            return ToolResult(False, "", f"File not found: {file_path}")

        content = p.read_text(encoding="utf-8")
        count = content.count(old_string)
        if count == 0:
            return ToolResult(False, "", "old_string not found in file")
        if count > 1:
            return ToolResult(False, "", f"old_string matches {count} locations; must be unique")

        p.write_text(content.replace(old_string, new_string, 1), encoding="utf-8")
        return ToolResult(True, f"Edited {file_path}")
    except ValueError as e:
        return ToolResult(False, "", str(e))
    except Exception as e:
        return ToolResult(False, "", str(e))


def create_edit_tool() -> ToolDefinition:
    return ToolDefinition(
        name="edit",
        description="Find and replace a unique string in a file within the workspace. Paths are relative to workspace root.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "File path"},
                "old_string": {"type": "string", "description": "Exact string to find (must be unique)"},
                "new_string": {"type": "string", "description": "Replacement string"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
        handler=edit_file,
    )
