"""内容搜索工具。对标 Claude Code: GrepTool"""

import re
from pathlib import Path
from tools.registry import ToolDefinition, ToolResult
from tools.sandbox import get_workspace

MAX_RESULTS = 100
MAX_LINE_LEN = 500


def grep_search(pattern: str, path: str = "", glob: str = "") -> ToolResult:
    """在 workspace 内搜索文件内容"""
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return ToolResult(False, "", f"Invalid regex: {e}")

    ws = get_workspace()
    base = ws / path if path else ws
    if not str(base.resolve()).startswith(str(ws)):
        return ToolResult(False, "", "路径不在工作区内")

    # 收集要搜索的文件
    if glob:
        import glob as globlib
        files = [Path(p) for p in globlib.glob(str(base / glob), recursive=True)]
    else:
        files = [p for p in base.rglob("*") if p.is_file()]

    results = []
    for fp in files:
        if not fp.is_file():
            continue
        # 跳过二进制和大文件
        if fp.stat().st_size > 1_000_000:
            continue
        try:
            rel = str(fp.relative_to(ws))
        except ValueError:
            continue
        try:
            lines = fp.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            if regex.search(line):
                display = line[:MAX_LINE_LEN]
                results.append(f"{rel}:{i}: {display}")
                if len(results) >= MAX_RESULTS:
                    results.append(f"[truncated at {MAX_RESULTS} matches]")
                    return ToolResult(True, "\n".join(results))

    if not results:
        return ToolResult(True, f"No matches for '{pattern}'")
    return ToolResult(True, "\n".join(results))


def create_grep_tool() -> ToolDefinition:
    return ToolDefinition(
        name="grep",
        description="Search file contents using regex within the workspace.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Subdirectory to search in", "default": ""},
                "glob": {"type": "string", "description": "File glob filter (e.g. '**/*.py')", "default": ""},
            },
            "required": ["pattern"],
        },
        handler=grep_search,
        read_only=True,
    )
