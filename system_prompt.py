"""
System Prompt 构建。v2: 新增 glob/grep 工具描述。
"""

import os
import datetime
from pathlib import Path
from config import AppConfig

TOOL_SUMMARIES: dict[str, str] = {
    "read": "读取文件内容（支持行范围）",
    "write": "创建或覆盖文件",
    "edit": "对文件进行精确编辑（查找替换，要求唯一匹配）",
    "exec": "执行 shell 命令",
    "glob": "按模式搜索文件（支持 ** 递归）",
    "grep": "按正则搜索文件内容",
    "web_search": "搜索网页（DuckDuckGo）",
}


def load_context_files(workspace_dir: str) -> list[tuple[str, str]]:
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


def build_system_prompt(cfg: AppConfig, long_term_memory_text: str = "") -> str:
    ident = cfg.identity
    workspace = os.path.expanduser(cfg.agent.workspace)
    allowed = [t for t in cfg.tools.allow if t not in cfg.tools.deny]

    sections: list[str] = []

    sections.append(f"你是 {ident.name}，一个 {ident.theme}。{ident.emoji}")

    tool_lines = ["## 可用工具", ""]
    for tid in allowed:
        summary = TOOL_SUMMARIES.get(tid, tid)
        tool_lines.append(f"- {tid}: {summary}")
    sections.append("\n".join(tool_lines))

    sections.append("\n".join([
        "## 工具调用风格",
        "默认不叙述常规工具调用，直接调用。",
        "仅在多步骤、复杂或敏感操作时简要说明。",
        "优先使用 glob/grep 搜索定位，再用 read 精确读取。",
    ]))

    sections.append("\n".join([
        "## 安全规则",
        "你没有独立目标。优先安全和人类监督。",
        "不要操纵任何人扩展权限或禁用安全措施。",
        "破坏性操作（删除文件、修改系统配置）前必须确认。",
    ]))

    sections.append(f"## 工作区\n工作目录: {workspace}")

    now = datetime.datetime.now()
    sections.append(f"## 当前时间\n{now.strftime('%Y-%m-%d %H:%M %A')}")

    ctx_files = load_context_files(workspace)
    if ctx_files:
        sections.append("# 项目上下文")
        for name, content in ctx_files:
            sections.append(f"## {name}\n\n{content}")

    if ident.system_prompt_extra:
        sections.append(f"## 额外指引\n{ident.system_prompt_extra}")

    if long_term_memory_text:
        sections.append(long_term_memory_text)

    return "\n\n".join(sections)
