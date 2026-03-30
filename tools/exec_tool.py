"""Shell 命令执行工具。对应 OpenClaw: src/agents/bash-tools.exec.ts"""

import subprocess
from tools.registry import ToolDefinition, ToolResult
from tools.sandbox import get_workspace

MAX_OUTPUT = 50_000
DEFAULT_TIMEOUT = 30


def exec_command(command: str, timeout: int = DEFAULT_TIMEOUT) -> ToolResult:
    try:
        ws = get_workspace()
        r = subprocess.run(
            command, shell=True, capture_output=True,
            text=True, timeout=timeout, cwd=str(ws),
        )
        output = r.stdout or ""
        if r.stderr:
            output += f"\n[stderr]\n{r.stderr}" if output else r.stderr
        if len(output) > MAX_OUTPUT:
            output = output[:MAX_OUTPUT] + f"\n[truncated at {MAX_OUTPUT} chars]"

        if r.returncode == 0:
            return ToolResult(True, output)
        return ToolResult(False, output, f"exit code {r.returncode}")
    except subprocess.TimeoutExpired:
        return ToolResult(False, "", f"timeout ({timeout}s)")
    except Exception as e:
        return ToolResult(False, "", str(e))


def create_exec_tool() -> ToolDefinition:
    return ToolDefinition(
        name="exec",
        description="Run a shell command within the workspace directory and return output.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command"},
                "timeout": {"type": "integer", "description": "Timeout in seconds", "default": DEFAULT_TIMEOUT},
            },
            "required": ["command"],
        },
        handler=exec_command,
    )
