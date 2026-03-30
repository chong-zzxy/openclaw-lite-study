"""文件写入工具。对应 OpenClaw: src/agents/pi-tools.ts write"""

from pathlib import Path
from tools.registry import ToolDefinition, ToolResult
from tools.sandbox import resolve_safe_path


def write_file(file_path: str, content: str) -> ToolResult:
    try:
        p = resolve_safe_path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return ToolResult(True, f"Wrote {file_path} ({len(content)} chars)")
    except ValueError as e:
        return ToolResult(False, "", str(e))
    except Exception as e:
        return ToolResult(False, "", str(e))


def create_write_tool() -> ToolDefinition:
    return ToolDefinition(
        name="write",
        description="Create or overwrite a file within the workspace. Paths are relative to workspace root. Parent dirs created automatically.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "File path"},
                "content": {"type": "string", "description": "File content"},
            },
            "required": ["file_path", "content"],
        },
        handler=write_file,
    )
