"""
System Prompt 构建。
对应 OpenClaw: src/agents/system-prompt.ts — buildAgentSystemPrompt()
"""

import os
import datetime
from pathlib import Path
from config import AppConfig

TOOL_SUMMARIES: dict[str, str] = {
    "read": "读取文件内容",
    "write": "创建或覆盖文件",
    "edit": "对文件进行精确编辑（查找替换）",
    "exec": "执行 shell 命令",
    "web_search": "搜索网页（模拟）",
}


def load_context_files(workspace_dir: str) -> list[tuple[str, str]]:
    """加载工作区上下文文件（SOUL.md, AGENTS.md, TOOLS.md）。"""
    result = []
    ws = Path(workspace_dir).expanduser()
    for name in ["SOUL.md", "AGENTS.md", "TOOLS.md"]:
        fp = ws / name
        if fp.exists():
            try:
                text = fp.read_text(encoding="utf-8").strip()
                if text:
                    result.append((name, text))
            except Exception:
                pass
    return result


def build_system_prompt(cfg: AppConfig) -> str:
    """构建完整的 system prompt。"""
    ident = cfg.identity
    workspace = os.path.expanduser(cfg.agent.workspace)
    allowed = [t for t in cfg.tools.allow if t not in cfg.tools.deny]

    sections: list[str] = []

    # 身份
    sections.append(
        f"你是 {ident.name}，一个 {ident.theme}。{ident.emoji}"
    )

    # 工具列表
    tool_lines = ["## 可用工具", ""]
    for tid in allowed:
        summary = TOOL_SUMMARIES.get(tid, tid)
        tool_lines.append(f"- {tid}: {summary}")
    sections.append("\n".join(tool_lines))

    # 工具调用风格
    sections.append("\n".join([
        "## 工具调用风格",
        "默认不叙述常规工具调用，直接调用。",
        "仅在多步骤、复杂或敏感操作时简要说明。",
    ]))

    # 安全
    sections.append("\n".join([
        "## 安全规则",
        "你没有独立目标。优先安全和人类监督。",
        "不要操纵任何人扩展权限或禁用安全措施。",
    ]))

    # 工作区
    sections.append(f"## 工作区\n工作目录: {workspace}")

    # 时间
    now = datetime.datetime.now()
    sections.append(f"## 当前时间\n{now.strftime('%Y-%m-%d %H:%M %A')}")

    # 上下文文件
    ctx_files = load_context_files(workspace)
    if ctx_files:
        sections.append("# 项目上下文")
        for name, content in ctx_files:
            sections.append(f"## {name}\n\n{content}")

    # 额外 prompt
    if ident.system_prompt_extra:
        sections.append(f"## 额外指引\n{ident.system_prompt_extra}")

    return "\n\n".join(sections)
