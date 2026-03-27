"""文件读取工具。对应 OpenClaw: src/agents/pi-tools.read.ts"""

from pathlib import Path
from tools.registry import ToolDefinition, ToolResult

MAX_CHARS = 100_000


def read_file(file_path: str, start_line: int = 0, end_line: int = 0) -> ToolResult:
    try:
        p = Path(file_path).expanduser()
        if not p.exists():
            return ToolResult(False, "", f"File not found: {file_path}")
        if not p.is_file():
            return ToolResult(False, "", f"Not a file: {file_path}")

        content = p.read_text(encoding="utf-8")
        if start_line > 0 or end_line > 0:
            lines = content.splitlines(keepends=True)
            s = max(0, start_line - 1)
            e = end_line if end_line > 0 else len(lines)
            content = "".join(lines[s:e])

        if len(content) > MAX_CHARS:
            content = content[:MAX_CHARS] + f"\n\n[truncated at {MAX_CHARS} chars]"

        return ToolResult(True, content)
    except UnicodeDecodeError:
        return ToolResult(False, "", f"Binary file: {file_path}")
    except Exception as e:
        return ToolResult(False, "", str(e))


def create_read_tool() -> ToolDefinition:
    return ToolDefinition(
        name="read",
        description="Read file contents. Optionally specify line range.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "File path to read"},
                "start_line": {"type": "integer", "description": "Start line (1-indexed)", "default": 0},
                "end_line": {"type": "integer", "description": "End line (0=EOF)", "default": 0},
            },
            "required": ["file_path"],
        },
        handler=read_file,
    )
