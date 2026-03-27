"""
配置加载模块。
对应 OpenClaw: src/config/config.ts
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class IdentityConfig:
    name: str = "Clawd"
    theme: str = "helpful assistant"
    emoji: str = "🦞"
    system_prompt_extra: str = ""


@dataclass
class ModelRef:
    provider: str = "dashscope"
    model: str = "qwen-max"


@dataclass
class AgentConfig:
    workspace: str = "~/.openclaw-lite/workspace"
    model: ModelRef = field(default_factory=ModelRef)
    fallback_models: list[ModelRef] = field(default_factory=list)
    timeout_seconds: int = 300
    max_tool_iterations: int = 20
    compaction_threshold: int = 40


@dataclass
class HooksConfig:
    """钩子配置。对应 OpenClaw: hooks 配置段"""
    enable_logging: bool = True
    enable_dangerous_command_guard: bool = True
    output_truncation_chars: int = 50000


@dataclass
class ToolsConfig:
    allow: list[str] = field(default_factory=lambda: [
        "exec", "read", "write", "edit", "web_search",
    ])
    deny: list[str] = field(default_factory=list)


@dataclass
class SessionConfig:
    store_path: str = "~/.openclaw-lite/sessions.json"
    history_limit: int = 50
    reset_triggers: list[str] = field(default_factory=lambda: ["/new", "/reset"])


@dataclass
class AppConfig:
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    hooks: HooksConfig = field(default_factory=HooksConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    session: SessionConfig = field(default_factory=SessionConfig)


def load_config(config_path: str = "config.json") -> AppConfig:
    """加载配置文件，缺失字段用默认值填充。"""
    path = Path(config_path)
    if not path.exists():
        return AppConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cfg = AppConfig()

    if "identity" in raw:
        d = raw["identity"]
        cfg.identity = IdentityConfig(
            name=d.get("name", "Clawd"),
            theme=d.get("theme", "helpful assistant"),
            emoji=d.get("emoji", "🦞"),
            system_prompt_extra=d.get("system_prompt_extra", ""),
        )

    if "agent" in raw:
        d = raw["agent"]
        m = d.get("model", {})
        fallbacks = []
        for fb in d.get("fallback_models", []):
            fallbacks.append(ModelRef(
                provider=fb.get("provider", "dashscope"),
                model=fb.get("model", "qwen-max"),
            ))
        cfg.agent = AgentConfig(
            workspace=d.get("workspace", cfg.agent.workspace),
            model=ModelRef(
                provider=m.get("provider", "dashscope"),
                model=m.get("model", "qwen-max"),
            ),
            fallback_models=fallbacks,
            timeout_seconds=d.get("timeout_seconds", 300),
            max_tool_iterations=d.get("max_tool_iterations", 20),
            compaction_threshold=d.get("compaction_threshold", 40),
        )

    if "hooks" in raw:
        d = raw["hooks"]
        cfg.hooks = HooksConfig(
            enable_logging=d.get("enable_logging", True),
            enable_dangerous_command_guard=d.get("enable_dangerous_command_guard", True),
            output_truncation_chars=d.get("output_truncation_chars", 50000),
        )

    if "tools" in raw:
        d = raw["tools"]
        cfg.tools = ToolsConfig(
            allow=d.get("allow", cfg.tools.allow),
            deny=d.get("deny", []),
        )

    if "session" in raw:
        d = raw["session"]
        cfg.session = SessionConfig(
            store_path=d.get("store_path", cfg.session.store_path),
            history_limit=d.get("history_limit", 50),
            reset_triggers=d.get("reset_triggers", ["/new", "/reset"]),
        )

    return cfg


def resolve_api_key(provider: str) -> Optional[str]:
    """从环境变量读取 API Key。"""
    env_map = {
        "dashscope": "DASHSCOPE_API_KEY",
    }
    return os.environ.get(env_map.get(provider, f"{provider.upper()}_API_KEY"))
