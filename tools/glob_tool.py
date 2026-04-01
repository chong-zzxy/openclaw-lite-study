"""文件搜索工具（glob 模式匹配）。对标 Claude Code: GlobTool"""

import glob as globlib
from pathlib import Path
from tools.registry import ToolDefinition, ToolResult
from tools.sandbox import get_workspace


def glob_search(pattern: str, path: str = "") -> ToolResult:
    """在 workspace 内按 glob 模式搜索文件"""
    try:
        ws = get_workspace()
        base = ws / path if path else ws
        if not str(base.resolve()).startswith(str(ws)):
            return ToolResult(False, "", "路径不在工作区内")

        matches = sorted(globlib.glob(str(base / pattern), recursive=True))
        # 转为相对路径
        results = []
        for m in matches[:200]:  # 限制结果数
            try:
                rel = str(Path(m).relative_to(ws))
                results.append(rel)
            except ValueError:
                pass

        if not results:
            return ToolResult(True, f"No files matching '{pattern}'")
        return ToolResult(True, "\n".join(results))
    except Exception as e:
        return ToolResult(False, "", str(e))


def create_glob_tool() -> ToolDefinition:
    return ToolDefinition(
        name="glob",
        description="Search for files matching a glob pattern within the workspace. Use ** for recursive search.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern (e.g. '**/*.py')"},
                "path": {"type": "string", "description": "Subdirectory to search in", "default": ""},
            },
            "required": ["pattern"],
        },
        handler=glob_search,
        read_only=True,
    )
