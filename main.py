#!/usr/bin/env python3
"""
OpenClaw Lite — CLI 入口。
对应 OpenClaw: src/entry.ts → src/cli/run-main.ts
"""

import os
import sys
from pathlib import Path

from config import load_config
from session import SessionStore
from identity import format_greeting
from agent import run_agent_turn, create_hook_runner
from tools import create_default_registry
from tools.sandbox import set_workspace
from memory import LongTermMemory


def _init_workspace_examples(workspace: Path):
    """首次启动时将 workspace-example/ 下的示例文件复制到工作区"""
    example_dir = Path(__file__).parent / "workspace-example"
    if not example_dir.exists():
        return
    for src in example_dir.iterdir():
        if src.is_file() and src.suffix == ".md":
            dst = workspace / src.name
            if not dst.exists():
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                print(f"  📄 已复制示例文件: {src.name} → {dst}")


def main():
    # 加载配置
    config_path = os.environ.get("OPENCLAW_LITE_CONFIG", "config.json")
    if not Path(config_path).exists():
        script_dir = Path(__file__).parent
        alt_path = script_dir / "config.json"
        if alt_path.exists():
            config_path = str(alt_path)

    cfg = load_config(config_path)

    # 确保工作区目录存在
    workspace = Path(cfg.agent.workspace).expanduser()
    workspace.mkdir(parents=True, exist_ok=True)

    # 初始化沙箱
    set_workspace(cfg.agent.workspace)

    # 首次启动时复制示例文件到工作区
    _init_workspace_examples(workspace)

    # 初始化
    session_store = SessionStore(cfg.session.store_path)
    session_id = "default"
    tool_registry = create_default_registry()
    hook_runner = create_hook_runner(cfg)
    # 长期记忆存储在 session store 同级目录
    ltm_dir = str(Path(cfg.session.store_path).expanduser().parent)
    long_term_memory = LongTermMemory(ltm_dir)

    # 欢迎信息
    print()
    print(format_greeting(cfg))
    print(f"模型: {cfg.agent.model.provider}/{cfg.agent.model.model}")
    if cfg.agent.fallback_models:
        fb_names = [f"{fb.provider}/{fb.model}" for fb in cfg.agent.fallback_models]
        print(f"备用: {', '.join(fb_names)}")
    print(f"工具: {', '.join(cfg.tools.allow)}")
    print(f"钩子: 日志={'开' if cfg.hooks.enable_logging else '关'}"
          f" 危险命令拦截={'开' if cfg.hooks.enable_dangerous_command_guard else '关'}")
    print(f"工作区: {workspace}")
    print()
    print("输入消息开始对话。特殊命令：")
    print("  /new, /reset  — 重置会话")
    print("  /sessions     — 列出会话")
    print("  /model <p/m>  — 切换模型（如 /model openai/gpt-4o）")
    print("  /quit, /exit  — 退出")
    print()

    # 交互循环
    while True:
        try:
            user_input = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # 特殊命令
        if user_input in cfg.session.reset_triggers:
            # reset 前提取长期记忆
            history = session_store.get_history(session_id)
            if history and long_term_memory:
                print("  🧠 提取长期记忆...")
                try:
                    from agent import create_provider
                    provider = create_provider(cfg.agent.model.provider, cfg.agent.model.model)
                    from memory import extract_memories_from_conversation
                    extract_memories_from_conversation(history, provider, long_term_memory, session_id)
                except Exception as e:
                    print(f"  ⚠️ 记忆提取失败: {e}")
            session_store.reset_session(session_id)
            print("✨ 会话已重置\n")
            continue

        if user_input in ("/quit", "/exit"):
            print("再见！")
            break

        if user_input == "/sessions":
            sessions = session_store.list_sessions()
            if sessions:
                for sid in sessions:
                    s = session_store.sessions[sid]
                    print(f"  {sid}: {len(s.messages)} 条消息")
            else:
                print("  没有活跃会话")
            print()
            continue

        if user_input.startswith("/model "):
            # 运行时切换模型
            # 对应 OpenClaw: /model 命令 → session model override
            model_spec = user_input[7:].strip()
            if "/" in model_spec:
                provider, model = model_spec.split("/", 1)
                cfg.agent.model.provider = provider
                cfg.agent.model.model = model
            else:
                cfg.agent.model.model = model_spec
            print(f"✅ 模型已切换为: {cfg.agent.model.provider}/{cfg.agent.model.model}\n")
            continue

        # 执行 agent turn
        try:
            print()
            reply = run_agent_turn(
                user_message=user_input,
                cfg=cfg,
                session_store=session_store,
                session_id=session_id,
                tool_registry=tool_registry,
                hook_runner=hook_runner,
                long_term_memory=long_term_memory,
            )
            print()
            print(f"{cfg.identity.emoji} {cfg.identity.name}> {reply}")
            print()
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


if __name__ == "__main__":
    main()
