"""
模型 Failover（故障转移）机制。

对应 OpenClaw: src/agents/model-fallback.ts — runWithModelFallback()

OpenClaw 的 failover 机制：
1. 先用主模型（primary）尝试
2. 如果失败（rate limit、auth 错误、服务不可用等），
   按 fallbacks 列表依次尝试备用模型
3. 每个模型有独立的 auth profile 轮换
4. 支持 FailoverError 类型区分可重试/不可重试错误
5. 支持 cooldown（某个 profile 失败后暂时跳过）

这里简化为：主模型失败 → 依次尝试 fallback 列表。
"""

from dataclasses import dataclass
from typing import Callable, TypeVar
from providers.base import LLMProvider

T = TypeVar("T")


@dataclass
class FailoverConfig:
    """Failover 配置"""
    fallback_models: list[dict]  # [{"provider": "openai", "model": "gpt-4o-mini"}, ...]
    max_retries_per_model: int = 1


class FailoverError(Exception):
    """
    可触发 failover 的错误。

    对应 OpenClaw: src/agents/failover-error.ts
    OpenClaw 的 FailoverError 包含 reason 字段：
    - "rate_limit" — 速率限制
    - "auth_failed" — 认证失败
    - "model_not_found" — 模型不存在
    - "billing" — 计费问题
    - "overloaded" — 服务过载
    - "unknown" — 未知错误
    """

    def __init__(self, message: str, reason: str = "unknown", retryable: bool = True):
        super().__init__(message)
        self.reason = reason
        self.retryable = retryable


def run_with_failover(
    primary_provider: LLMProvider,
    fallback_providers: list[LLMProvider],
    run_fn: Callable[[LLMProvider], T],
) -> tuple[T, LLMProvider]:
    """
    带 failover 的执行。

    对应 OpenClaw: runWithModelFallback()

    流程：
    1. 用 primary_provider 执行 run_fn
    2. 如果抛出 FailoverError 且 retryable=True，尝试下一个 provider
    3. 返回 (结果, 实际使用的 provider)

    OpenClaw 的实现更复杂：
    - 支持 auth profile 轮换（同一个 provider 的多个 API key）
    - 支持 thinking level 降级（xhigh → high）
    - 支持 cooldown 追踪（失败的 profile 暂时跳过）
    - 支持 probe（预检测模型是否可用）
    """
    providers = [primary_provider] + fallback_providers

    last_error: Exception | None = None
    for i, provider in enumerate(providers):
        try:
            result = run_fn(provider)
            if i > 0:
                print(f"  ⚡ 已切换到备用模型: {provider.name()}")
            return result, provider
        except FailoverError as e:
            last_error = e
            if not e.retryable:
                raise
            print(f"  ⚠️ {provider.name()} 失败 ({e.reason}): {e}")
            if i < len(providers) - 1:
                print(f"  🔄 尝试下一个模型...")
            continue
        except Exception as e:
            # 非 FailoverError 不触发 failover
            raise

    raise last_error or RuntimeError("All providers failed")
