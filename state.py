"""
全局状态管理。
对标 Claude Code: src/state/AppStateStore.ts
集中管理运行时状态，所有模块通过 get_store() 访问。
"""

from __future__ import annotations
import threading
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class TurnStats:
    """单轮对话统计"""
    turn_number: int = 0
    tool_calls: int = 0
    api_calls: int = 0
    total_tokens: int = 0
    start_time: float = field(default_factory=time.time)
    errors: list[str] = field(default_factory=list)


@dataclass
class AppState:
    """
    全局应用状态，对标 Claude Code 的 AppState（大幅简化）。
    所有运行时可变状态集中在这里，通过 StateStore 访问。
    """
    # 权限模式: default | acceptEdits | plan | bypassPermissions
    permission_mode: str = "default"
    # 当前模型
    current_model_provider: str = ""
    current_model_name: str = ""
    # 会话
    session_id: str = "default"
    # 累计 token
    session_total_tokens: int = 0
    # 当前轮次统计
    current_turn: TurnStats = field(default_factory=TurnStats)
    # 执行控制
    is_running: bool = False
    abort_requested: bool = False
    # 流式输出
    stream_enabled: bool = True
    # 压缩追踪
    last_compaction_tokens: int = 0
    compaction_count: int = 0


class StateStore:
    """
    线程安全的状态存储，对标 Claude Code 的 zustand-like store。
    """

    def __init__(self):
        self._state = AppState()
        self._lock = threading.Lock()
        self._listeners: list[Callable[[AppState], None]] = []

    def get(self) -> AppState:
        with self._lock:
            return self._state

    def update(self, fn: Callable[[AppState], None]):
        with self._lock:
            fn(self._state)
            snapshot = self._state
            listeners = list(self._listeners)  # 遍历副本，防止迭代中修改
        for listener in listeners:
            try:
                listener(snapshot)
            except Exception:
                pass

    def subscribe(self, listener: Callable[[AppState], None]):
        with self._lock:
            self._listeners.append(listener)

    def reset_turn(self):
        with self._lock:
            self._state.current_turn = TurnStats()
            self._state.abort_requested = False

    def request_abort(self):
        with self._lock:
            self._state.abort_requested = True

    @property
    def is_aborted(self) -> bool:
        with self._lock:
            return self._state.abort_requested


_global_store: StateStore | None = None


def get_store() -> StateStore:
    global _global_store
    if _global_store is None:
        _global_store = StateStore()
    return _global_store
