"""
工具调用钩子系统。

对应 OpenClaw:
- src/agents/pi-tools.before-tool-call.ts
- src/agents/pi-tool-definition-adapter.ts (after-tool-call)
- src/hooks/bundled/ — 内置钩子

OpenClaw 的钩子系统支持：
- before_tool_call: 工具执行前拦截，可修改参数、拒绝执行
- after_tool_call: 工具执行后处理，可修改结果
- before_agent_start: agent 启动前，可覆盖模型选择
- before_model_resolve: 模型解析前，可动态切换模型
- gateway_start / gateway_stop: 网关生命周期钩子

钩子可以来自：
1. 插件（plugin hooks）
2. 配置文件（hooks.mappings）
3. 工作区文件（TOOLS.md 中的指引）

这里实现 before_tool_call 和 after_tool_call 两个核心钩子点。
"""

from dataclasses import dataclass
from typing import Callable, Any
from tools.registry import ToolResult


@dataclass
class BeforeToolCallContext:
    """传递给 before_tool_call 钩子的上下文"""
    tool_name: str
    arguments: dict
    session_id: str


@dataclass
class AfterToolCallContext:
    """传递给 after_tool_call 钩子的上下文"""
    tool_name: str
    arguments: dict
    result: ToolResult
    session_id: str


@dataclass
class BeforeToolCallResult:
    """
    before_tool_call 钩子的返回值。

    对应 OpenClaw: before_tool_call 钩子可以：
    - 放行（proceed=True）
    - 拦截（proceed=False, reason="..."）
    - 修改参数（modified_arguments={...}）
    """
    proceed: bool = True
    reason: str | None = None
    modified_arguments: dict | None = None


# 钩子函数类型
BeforeToolCallHook = Callable[[BeforeToolCallContext], BeforeToolCallResult]
AfterToolCallHook = Callable[[AfterToolCallContext], ToolResult | None]


class HookRunner:
    """
    钩子运行器。

    对应 OpenClaw: getGlobalHookRunner() 返回的 hook runner。
    OpenClaw 的 hook runner 支持异步执行、超时控制、
    错误隔离（一个钩子失败不影响其他钩子）。
    """

    def __init__(self):
        self._before_tool_call: list[BeforeToolCallHook] = []
        self._after_tool_call: list[AfterToolCallHook] = []

    def register_before_tool_call(self, hook: BeforeToolCallHook):
        """注册 before_tool_call 钩子"""
        self._before_tool_call.append(hook)

    def register_after_tool_call(self, hook: AfterToolCallHook):
        """注册 after_tool_call 钩子"""
        self._after_tool_call.append(hook)

    def run_before_tool_call(self, ctx: BeforeToolCallContext) -> BeforeToolCallResult:
        """
        执行所有 before_tool_call 钩子。

        任何一个钩子返回 proceed=False 就拦截执行。
        最后一个返回 modified_arguments 的钩子生效。
        """
        result = BeforeToolCallResult()
        for hook in self._before_tool_call:
            try:
                hook_result = hook(ctx)
                if not hook_result.proceed:
                    return hook_result  # 拦截
                if hook_result.modified_arguments is not None:
                    result.modified_arguments = hook_result.modified_arguments
            except Exception as e:
                # 钩子失败不阻塞执行，只打印警告
                print(f"  ⚠️ before_tool_call 钩子异常: {e}")
        return result

    def run_after_tool_call(self, ctx: AfterToolCallContext) -> ToolResult:
        """
        执行所有 after_tool_call 钩子。

        钩子可以返回修改后的 ToolResult，或返回 None 表示不修改。
        """
        current_result = ctx.result
        for hook in self._after_tool_call:
            try:
                modified = hook(AfterToolCallContext(
                    tool_name=ctx.tool_name,
                    arguments=ctx.arguments,
                    result=current_result,
                    session_id=ctx.session_id,
                ))
                if modified is not None:
                    current_result = modified
            except Exception as e:
                print(f"  ⚠️ after_tool_call 钩子异常: {e}")
        return current_result


# === 内置钩子示例 ===

def create_logging_hook() -> tuple[BeforeToolCallHook, AfterToolCallHook]:
    """
    日志钩子：记录所有工具调用耗时。
    对应 OpenClaw 的工具调用日志（通过 subsystem logger）。
    """
    import time

    # 用 (session_id, tool_name, timestamp) 做 FIFO 匹配
    _pending: list[tuple[str, str, float]] = []

    def before(ctx: BeforeToolCallContext) -> BeforeToolCallResult:
        _pending.append((ctx.session_id, ctx.tool_name, time.time()))
        return BeforeToolCallResult()

    def after(ctx: AfterToolCallContext) -> ToolResult | None:
        for i, (sid, name, start) in enumerate(_pending):
            if sid == ctx.session_id and name == ctx.tool_name:
                _pending.pop(i)
                duration_ms = (time.time() - start) * 1000
                print(f"  ⏱️ {ctx.tool_name} 耗时 {duration_ms:.0f}ms")
                break
        return None

    return before, after



def create_dangerous_command_guard() -> BeforeToolCallHook:
    """
    危险命令拦截钩子。

    对应 OpenClaw: tools.elevated 机制。
    OpenClaw 的 elevated 模式支持：
    - "off": 禁止所有危险操作
    - "ask": 需要用户审批（/approve 命令）
    - "on": 允许但记录
    - "full": 完全允许

    这里简化为拦截包含危险关键词的 exec 命令。
    """
    import re
    DANGEROUS_PATTERNS = [
        r"\brm\s+.*-[a-zA-Z]*r[a-zA-Z]*f",  # rm -rf, rm -fr, rm --recursive -f 等
        r"\brm\s+-[a-zA-Z]*f[a-zA-Z]*r",     # rm -fr
        r"\brm\s+(-rf?|--force)\s+[/~.]",    # rm -rf /, rm -f ~, rm -rf .
        r"\bsudo\b",                           # sudo 任何形式
        r"\bmkfs\b",
        r"\bdd\s+if=",
        r">\s*/dev/",
        r"\bchmod\s+777\b",
        r"\bcurl\b.*\|\s*(ba)?sh",            # curl | sh
        r"\bwget\b.*\|\s*(ba)?sh",            # wget | sh
        r"\b:(){ :\|:& };:",                  # fork bomb
    ]

    def hook(ctx: BeforeToolCallContext) -> BeforeToolCallResult:
        if ctx.tool_name != "exec":
            return BeforeToolCallResult()

        command = ctx.arguments.get("command", "")
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, command):
                return BeforeToolCallResult(
                    proceed=False,
                    reason=f"拦截危险命令（匹配 '{pattern}'）。如需执行，请确认后重试。",
                )
        return BeforeToolCallResult()

    return hook


def create_output_truncation_hook(max_chars: int = 50000) -> AfterToolCallHook:
    """
    输出截断钩子。

    对应 OpenClaw: tool result 截断逻辑
    OpenClaw 会根据模型的 context window 动态计算截断阈值。
    """
    def hook(ctx: AfterToolCallContext) -> ToolResult | None:
        if len(ctx.result.output) <= max_chars:
            return None
        truncated = ctx.result.output[:max_chars]
        truncated += f"\n\n[输出已截断：原始 {len(ctx.result.output)} 字符，显示前 {max_chars} 字符]"
        return ToolResult(
            success=ctx.result.success,
            output=truncated,
            error=ctx.result.error,
        )

    return hook
