"""
流式工具执行器。
对标 Claude Code: StreamingToolExecutor

核心思想：模型还在流式输出时，已完成解析的 tool_use block 立即开始执行，
不用等整个响应结束。多个工具可以并行执行（只读工具之间并行，写工具串行）。

工作流程：
1. 模型流式输出 → 解析出完整的 tool_call → addTool()
2. 后台线程立即开始执行 → 结果存入队列
3. 主循环通过 getCompletedResults() 消费已完成的结果
4. 所有工具完成后 getRemainingResults() 返回剩余结果
"""

from __future__ import annotations
import json
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Generator
from tools.registry import ToolRegistry, ToolResult


@dataclass
class ToolExecResult:
    """一个工具的执行结果"""
    tool_call_id: str
    tool_name: str
    arguments: dict
    result: ToolResult
    # 构建为 API 格式的 tool message
    def to_message(self) -> dict:
        content = self.result.output
        if self.result.error:
            content = (f"Error: {self.result.error}\n{self.result.output}"
                       if self.result.output else f"Error: {self.result.error}")
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": self.tool_call_id,
        }


class StreamingToolExecutor:
    """
    流式工具执行器。

    对标 Claude Code 的 StreamingToolExecutor：
    - 模型流式输出时，已解析完的 tool_call 立即提交执行
    - 只读工具之间可并行（ThreadPoolExecutor）
    - 写工具串行执行（防止竞态）
    - 支持中断（检查 abort 信号）
    """

    def __init__(
        self,
        registry: ToolRegistry,
        execute_fn=None,
        max_workers: int = 4,
    ):
        self._registry = registry
        self._execute_fn = execute_fn  # 可注入自定义执行函数（带 hooks）
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: list[tuple[str, str, dict, Future]] = []
        self._completed: list[ToolExecResult] = []
        self._lock = threading.Lock()
        self._discarded = False

    def add_tool(self, tool_call_id: str, tool_name: str, arguments: dict,
                 is_read_only: bool = False):
        """
        提交一个工具执行。模型流式输出时，每解析完一个 tool_call 就调用一次。
        只读工具提交到线程池并行执行，写工具等前面的都完成后再执行。
        """
        if self._discarded:
            return

        # 写工具需要等前面的都完成（简化版串行保证）
        if not is_read_only:
            self._wait_all_pending()

        future = self._pool.submit(self._execute_one, tool_call_id, tool_name, arguments)
        with self._lock:
            self._futures.append((tool_call_id, tool_name, arguments, future))

    def _execute_one(self, tool_call_id: str, tool_name: str, arguments: dict) -> ToolExecResult:
        """执行单个工具"""
        if self._execute_fn:
            result = self._execute_fn(tool_name, arguments)
        else:
            result = self._registry.execute(tool_name, arguments)
        return ToolExecResult(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
        )

    def _wait_all_pending(self):
        """等待所有已提交的 future 完成"""
        with self._lock:
            pending = list(self._futures)
        for _, _, _, future in pending:
            try:
                res = future.result(timeout=300)
                with self._lock:
                    self._completed.append(res)
            except Exception as e:
                pass

    def get_completed_results(self) -> list[ToolExecResult]:
        """
        获取已完成的结果（非阻塞）。
        对标 Claude Code 的 streamingToolExecutor.getCompletedResults()
        """
        newly_completed = []
        with self._lock:
            remaining = []
            for item in self._futures:
                tc_id, name, args, future = item
                if future.done():
                    try:
                        res = future.result(timeout=0)
                        newly_completed.append(res)
                    except Exception as e:
                        newly_completed.append(ToolExecResult(
                            tool_call_id=tc_id, tool_name=name,
                            arguments=args,
                            result=ToolResult(False, "", str(e)),
                        ))
                else:
                    remaining.append(item)
            self._futures = remaining
        return newly_completed

    def get_remaining_results(self) -> list[ToolExecResult]:
        """
        等待并返回所有剩余结果。
        对标 Claude Code 的 streamingToolExecutor.getRemainingResults()
        """
        results = []
        with self._lock:
            pending = list(self._futures)
            self._futures = []
        for tc_id, name, args, future in pending:
            try:
                res = future.result(timeout=300)
                results.append(res)
            except Exception as e:
                results.append(ToolExecResult(
                    tool_call_id=tc_id, tool_name=name,
                    arguments=args,
                    result=ToolResult(False, "", str(e)),
                ))
        return results

    def discard(self):
        """丢弃所有待执行的结果（模型 fallback 时使用）"""
        self._discarded = True
        with self._lock:
            for _, _, _, future in self._futures:
                future.cancel()
            self._futures = []

    def shutdown(self):
        self._pool.shutdown(wait=False)
