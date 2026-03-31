"""
身份/风格系统。
对应 OpenClaw: src/agents/identity.ts
"""

from config import AppConfig, IdentityConfig


def resolve_identity(cfg: AppConfig) -> IdentityConfig:
    """解析 agent 身份配置"""
    return cfg.identity


def format_identity_prefix(cfg: AppConfig) -> str:
    """生成消息前缀，如 "[Clawd]"，用于终端显示"""
    name = cfg.identity.name.strip()
    return f"[{name}]" if name else "[openclaw-lite]"


def format_greeting(cfg: AppConfig) -> str:
    """生成启动时的欢迎消息，包含 emoji、名字和主题"""
    identity = resolve_identity(cfg)
    emoji = identity.emoji or "🤖"
    name = identity.name or "Assistant"
    theme = identity.theme or "helpful assistant"
    return f"{emoji} {name} — {theme}"
