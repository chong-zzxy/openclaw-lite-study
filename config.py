"""
配置加载模块。v2: 新增权限配置、流式输出、token 阈值。
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
    emoji: str = "🐯"
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
    max_tool_iterations: int = 25
    # v2: token-based compaction
    compaction_token_threshold: int = 80000
    compaction_keep_recent: int = 10
    # v2: 流式输出
    stream: bool = True


@dataclass
class PermissionConfig:
    """v2: 权限配置"""
    mode: str = "default"  # default | acceptEdits | plan | bypassPermissions
    auto_allow_read_tools: bool = True


@dataclass
class HooksConfig:
    enable_logging: bool = True
    enable_dangerous_command_guard: bool = True
    output_truncation_chars: int = 50000


@dataclass
class ToolsConfig:
    allow: list[str] = field(default_factory=lambda: [
        "exec", "read", "write", "edit", "glob", "grep", "web_search",
    ])
    deny: list[str] = field(default_factory=list)


@dataclass
class SessionConfig:
    store_path: str = "~/.openclaw-lite/sessions.json"
    history_limit: int = 100
    reset_triggers: list[str] = field(default_factory=lambda: ["/new", "/reset"])


@dataclass
class AppConfig:
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    permissions: PermissionConfig = field(default_factory=PermissionConfig)
    hooks: HooksConfig = field(default_factory=HooksConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    session: SessionConfig = field(default_factory=SessionConfig)


def _parse_model_ref(d: dict, dp="dashscope", dm="qwen-max") -> ModelRef:
    return ModelRef(provider=d.get("provider", dp), model=d.get("model", dm))


def load_config(config_path: str = "config.json") -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        script_dir = Path(__file__).parent
        alt_path = script_dir / "config.json"
        if alt_path.exists():
            config_path = str(alt_path)
            path = alt_path
        else:
            return AppConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cfg = AppConfig()

    if "identity" in raw:
        d = raw["identity"]
        cfg.identity = IdentityConfig(
            name=d.get("name", "Clawd"),
            theme=d.get("theme", "helpful assistant"),
            emoji=d.get("emoji", "🐯"),
            system_prompt_extra=d.get("system_prompt_extra", ""),
        )

    if "agent" in raw:
        d = raw["agent"]
        fallbacks = [_parse_model_ref(fb) for fb in d.get("fallback_models", [])]
        cfg.agent = AgentConfig(
            workspace=d.get("workspace", cfg.agent.workspace),
            model=_parse_model_ref(d.get("model", {})),
            fallback_models=fallbacks,
            timeout_seconds=d.get("timeout_seconds", 300),
            max_tool_iterations=d.get("max_tool_iterations", 25),
            compaction_token_threshold=d.get("compaction_token_threshold", 80000),
            compaction_keep_recent=d.get("compaction_keep_recent", 10),
            stream=d.get("stream", True),
        )

    if "permissions" in raw:
        d = raw["permissions"]
        cfg.permissions = PermissionConfig(
            mode=d.get("mode", "default"),
            auto_allow_read_tools=d.get("auto_allow_read_tools", True),
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
            history_limit=d.get("history_limit", 100),
            reset_triggers=d.get("reset_triggers", ["/new", "/reset"]),
        )

    return cfg


def resolve_api_key(provider: str) -> Optional[str]:
    env_map = {
        "dashscope": "DASHSCOPE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    return os.environ.get(env_map.get(provider, f"{provider.upper()}_API_KEY"))
