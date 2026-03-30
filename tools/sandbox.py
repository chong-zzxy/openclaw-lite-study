"""
路径沙箱校验。
确保文件工具只能操作 workspace 目录内的文件，防止越权访问。
"""

from pathlib import Path

# 全局 workspace 路径，由 main.py 启动时设置
_workspace: Path | None = None


def set_workspace(workspace_path: str):
    """设置沙箱根目录"""
    global _workspace
    _workspace = Path(workspace_path).expanduser().resolve()


def get_workspace() -> Path:
    """获取沙箱根目录"""
    if _workspace is None:
        raise RuntimeError("workspace 未初始化，请先调用 set_workspace()")
    return _workspace


def resolve_safe_path(file_path: str) -> Path:
    """
    将用户传入的路径解析为安全的绝对路径。

    规则：
    - 相对路径：相对于 workspace 解析
    - 绝对路径：必须在 workspace 内
    - 禁止通过 .. 逃逸出 workspace

    返回解析后的绝对路径，不安全时抛出 ValueError。
    """
    ws = get_workspace()
    p = Path(file_path).expanduser()

    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = (ws / p).resolve()

    # 检查是否在 workspace 内
    try:
        resolved.relative_to(ws)
    except ValueError:
        raise ValueError(
            f"路径 '{file_path}' 不在工作区内。"
            f"只允许访问 {ws} 及其子目录。"
        )

    return resolved
