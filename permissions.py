"""
权限系统。对标 Claude Code: src/utils/permissions/PermissionMode.ts

四级权限：
- default:           写操作需用户确认
- acceptEdits:       文件编辑自动放行，shell 命令仍需确认
- plan:              只规划不执行（写操作被拦截）
- bypassPermissions: 跳过所有权限检查

工具分类影响权限判断：
- read_only 工具在所有模式下自动放行
- destructive 工具即使在 acceptEdits 下也需确认
"""

from __future__ import annotations
import re
from enum import Enum


class PermissionMode(Enum):
    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    PLAN = "plan"
    BYPASS = "bypassPermissions"

    @staticmethod
    def from_string(s: str) -> "PermissionMode":
        for m in PermissionMode:
            if m.value == s:
                return m
        return PermissionMode.DEFAULT


class Decision(Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


# 危险命令正则
DANGEROUS_PATTERNS = [
    r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|--recursive)\b",
    r"\bsudo\b", r"\bmkfs\b", r"\bdd\s+if=",
    r">\s*/dev/", r"\bchmod\s+777\b",
    r"\bcurl\b.*\|\s*(ba)?sh", r"\bwget\b.*\|\s*(ba)?sh",
]

SAFE_READ_COMMANDS = [
    "ls", "cat", "head", "tail", "wc", "find", "grep", "rg",
    "git status", "git log", "git diff", "git branch",
    "pwd", "echo", "date", "which", "file",
    "python --version", "node --version",
]


def is_dangerous_command(command: str) -> bool:
    for pat in DANGEROUS_PATTERNS:
        if re.search(pat, command):
            return True
    return False


def is_safe_read_command(command: str) -> bool:
    cmd = command.strip()
    for prefix in SAFE_READ_COMMANDS:
        if cmd == prefix or cmd.startswith(prefix + " "):
            return True
    return False


def check_permission(
    mode: PermissionMode,
    tool_name: str,
    arguments: dict,
    is_read_only: bool = False,
    is_destructive: bool = False,
) -> Decision:
    """
    核心权限判断。对标 Claude Code 的 checkPermissions()。
    返回 ALLOW/ASK/DENY。
    """
    # 只读工具在所有模式下放行
    if is_read_only:
        return Decision.ALLOW

    # bypass 模式全部放行
    if mode == PermissionMode.BYPASS:
        return Decision.ALLOW

    # plan 模式拒绝所有写操作
    if mode == PermissionMode.PLAN:
        return Decision.DENY

    # acceptEdits 模式
    if mode == PermissionMode.ACCEPT_EDITS:
        # 文件编辑工具自动放行
        if tool_name in ("write", "edit"):
            if is_destructive:
                return Decision.ASK
            return Decision.ALLOW
        # exec 命令需要进一步判断
        if tool_name == "exec":
            command = arguments.get("command", "")
            if is_dangerous_command(command):
                return Decision.ASK
            if is_safe_read_command(command):
                return Decision.ALLOW
            return Decision.ASK
        return Decision.ASK

    # default 模式：exec 中的安全只读命令放行，其余都问
    if tool_name == "exec":
        command = arguments.get("command", "")
        if is_safe_read_command(command):
            return Decision.ALLOW
    if is_read_only:
        return Decision.ALLOW
    return Decision.ASK


def ask_user_permission(tool_name: str, arguments: dict) -> bool:
    """交互式询问用户是否允许执行"""
    print(f"\n  🔒 权限确认: {tool_name}")
    if tool_name == "exec":
        print(f"     命令: {arguments.get('command', '')}")
    elif tool_name in ("write", "edit"):
        print(f"     文件: {arguments.get('file_path', '')}")
    else:
        keys = list(arguments.keys())[:3]
        print(f"     参数: {', '.join(keys)}")

    try:
        answer = input("     允许执行? [y/N/a(本次全部允许)] ").strip().lower()
        if answer == "a":
            return True  # 调用方需处理 "a" 切换到 acceptEdits
        return answer in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


MODE_DISPLAY = {
    PermissionMode.DEFAULT: ("Default", "写操作需确认"),
    PermissionMode.ACCEPT_EDITS: ("Accept Edits", "文件编辑自动放行"),
    PermissionMode.PLAN: ("Plan", "只规划不执行"),
    PermissionMode.BYPASS: ("Bypass", "跳过所有权限"),
}


def format_mode(mode: PermissionMode) -> str:
    name, desc = MODE_DISPLAY.get(mode, ("Unknown", ""))
    return f"{name} ({desc})"
