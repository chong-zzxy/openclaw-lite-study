#!/usr/bin/env python3
"""
OpenClaw Lite v2 — CLI 入口。
新增：/mode 权限切换、/stats 统计、/compact 手动压缩、Ctrl+C 中断。
"""

import os
import sys
import signal
from pathlib import Path

from config import load_config
from session import SessionStore
from identity import format_greeting
from agent import run_agent_turn, create_hook_runner, create_provider
from tools import create_default_registry
from tools.sandbox import set_workspace
from memory import LongTermMemory
from state import get_store
from permissions import PermissionMode, format_mode


def _init_workspace_examples(workspace: Path):
    """首次启动时复制示例文件到工作区"""
    example_dir = Path(__file__).parent / "workspace-example"
    if not example_dir.exists():
        return
    for src in example_dir.iterdir():
        if src.is_file() and src.suffix == ".md":
            dst = workspace / src.name
            if not dst.exists():
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                print(f"  📄 已复制示例文件: {src.name} → {dst}")


def _setup_interrupt_handler():
    """设置 Ctrl+C 中断处理"""
    store = get_store()

    def handler(signum, frame):
        if store.get().is_running:
            store.request_abort()
            print("\n  ⏹️ 正在中断...")
        else:
            print("\n再见！")
            sys.exit(0)

    signal.signal(signal.SIGINT, handler)


def _print_help():
    """打印帮助信息"""
    print("特殊命令：")
    print("  /new, /reset    — 重置会话")
    print("  /mode [模式]    — 查看/切换权限模式")
    print("                    可选: default, acceptEdits, plan, bypassPermissions")
    print("  /stats          — 查看本次会话统计")
    print("  /compact        — 手动压缩历史")
    print("  /remember <文本> — 主动记住一条信息")
    print("  /memory         — 查看长期记忆")
    print("  /sessions       — 列出会话")
    print("  /model <p/m>    — 切换模型（如 /model dashscope/qwen-max）")
    print("  /help           — 显示此帮助")
    print("  /quit, /exit    — 退出")


def main():
    """CLI 主循环"""
    # 加载配置
    config_path = os.environ.get("OPENCLAW_LITE_CONFIG", "config.json")
    if not Path(config_path).exists():
        script_dir = Path(__file__).parent
        alt_path = script_dir / "config.json"
        if alt_path.exists():
            config_path = str(alt_path)

    cfg = load_config(config_path)

    # 初始化状态
    store = get_store()
    store.update(lambda s: setattr(s, 'permission_mode', cfg.permissions.mode))
    store.update(lambda s: setattr(s, 'stream_enabled', cfg.agent.stream))
    store.update(lambda s: (
        setattr(s, 'current_model_provider', cfg.agent.model.provider),
        setattr(s, 'current_model_name', cfg.agent.model.model),
    ))

    # 确保工作区目录存在
    workspace = Path(cfg.agent.workspace).expanduser()
    workspace.mkdir(parents=True, exist_ok=True)
    set_workspace(cfg.agent.workspace)
    _init_workspace_examples(workspace)

    # 设置中断处理
    _setup_interrupt_handler()

    # 初始化
    session_store = SessionStore(cfg.session.store_path)
    session_id = "default"
    tool_registry = create_default_registry()
    hook_runner = create_hook_runner(cfg)
    ltm_dir = str(Path(cfg.session.store_path).expanduser().parent)
    long_term_memory = LongTermMemory(ltm_dir)

    # 欢迎信息
    mode = PermissionMode.from_string(store.get().permission_mode)
    print()
    print(format_greeting(cfg))
    print(f"模型: {cfg.agent.model.provider}/{cfg.agent.model.model}")
    if cfg.agent.fallback_models:
        fb_names = [f"{fb.provider}/{fb.model}" for fb in cfg.agent.fallback_models]
        print(f"备用: {', '.join(fb_names)}")
    print(f"工具: {', '.join(cfg.tools.allow)}")
    print(f"权限: {format_mode(mode)}")
    print(f"流式: {'开' if cfg.agent.stream else '关'}")
    print(f"工作区: {workspace}")
    print()
    print("输入 /help 查看所有命令")
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

        # === 特殊命令 ===
        if user_input in cfg.session.reset_triggers:
            history = session_store.get_history(session_id)
            if history and long_term_memory:
                print("  🧠 提取长期记忆...")
                try:
                    from memory import extract_memories_from_conversation
                    provider = create_provider(
                        cfg.agent.model.provider, cfg.agent.model.model)
                    extract_memories_from_conversation(
                        history, provider, long_term_memory, session_id)
                except Exception as e:
                    print(f"  ⚠️ 记忆提取失败: {e}")
            session_store.reset_session(session_id)
            print("✨ 会话已重置\n")
            continue

        if user_input in ("/quit", "/exit"):
            print("再见！")
            break

        if user_input == "/help":
            _print_help()
            print()
            continue

        if user_input.startswith("/mode"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 1:
                # 显示当前模式
                m = PermissionMode.from_string(store.get().permission_mode)
                print(f"  当前权限模式: {format_mode(m)}")
                print("  可选: default, acceptEdits, plan, bypassPermissions")
            else:
                new_mode = PermissionMode.from_string(parts[1].strip())
                store.update(lambda s: setattr(s, 'permission_mode', new_mode.value))
                print(f"  ✅ 权限模式切换为: {format_mode(new_mode)}")
            print()
            continue

        if user_input == "/stats":
            s = store.get()
            print(f"  累计 Token: {s.session_total_tokens}")
            print(f"  压缩次数: {s.compaction_count}")
            print(f"  权限模式: {format_mode(PermissionMode.from_string(s.permission_mode))}")
            print(f"  流式输出: {'开' if s.stream_enabled else '关'}")
            print()
            continue

        if user_input == "/compact":
            history = session_store.get_history(session_id)
            if len(history) < 5:
                print("  历史太短，无需压缩")
            else:
                print("  📦 手动压缩...")
                try:
                    from compaction import compact_history
                    provider = create_provider(
                        cfg.agent.model.provider, cfg.agent.model.model)
                    history = compact_history(
                        history, provider,
                        keep_recent=cfg.agent.compaction_keep_recent,
                        token_threshold=0,  # 强制压缩
                    )
                    session = session_store.get_or_create(session_id)
                    session.messages = list(history)
                    session_store._save()
                    store.update(lambda s: setattr(
                        s, 'compaction_count', s.compaction_count + 1))
                    print("  ✅ 压缩完成")
                except Exception as e:
                    print(f"  ❌ 压缩失败: {e}")
            print()
            continue

        if user_input.startswith("/remember "):
            text = user_input[10:].strip()
            if text:
                long_term_memory.remember_explicit(text)
            else:
                print("  用法: /remember 要记住的内容")
            print()
            continue

        if user_input == "/memory":
            p = long_term_memory.profile
            if p.preferences:
                print("  偏好:")
                for item in p.preferences:
                    print(f"    - {item}")
            if p.facts:
                print("  已知信息:")
                for item in p.facts:
                    print(f"    - {item}")
            if long_term_memory.memories:
                print(f"  知识记忆: {len(long_term_memory.memories)} 条")
            if not p.preferences and not p.facts and not long_term_memory.memories:
                print("  暂无长期记忆")
            print()
            continue

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
            model_spec = user_input[7:].strip()
            if "/" in model_spec:
                provider, model = model_spec.split("/", 1)
                cfg.agent.model.provider = provider
                cfg.agent.model.model = model
            else:
                cfg.agent.model.model = model_spec
            store.update(lambda s: (
                setattr(s, 'current_model_provider', cfg.agent.model.provider),
                setattr(s, 'current_model_name', cfg.agent.model.model),
            ))
            print(f"✅ 模型已切换为: {cfg.agent.model.provider}/{cfg.agent.model.model}\n")
            continue

        if user_input.startswith("/"):
            print(f"  未知命令: {user_input}，输入 /help 查看帮助\n")
            continue

        # === 执行 agent turn ===
        try:
            print()
            store.update(lambda s: setattr(s, 'is_running', True))
            reply = run_agent_turn(
                user_message=user_input,
                cfg=cfg,
                session_store=session_store,
                session_id=session_id,
                tool_registry=tool_registry,
                hook_runner=hook_runner,
                long_term_memory=long_term_memory,
            )
            store.update(lambda s: setattr(s, 'is_running', False))

            # 流式模式下文本已经实时打印了
            # 非流式模式下需要完整打印
            if not cfg.agent.stream:
                print()
                print(f"{cfg.identity.emoji} {cfg.identity.name}> {reply}")
            else:
                # 流式模式：文本已在流中打印，在前面补上身份前缀
                if reply and reply not in ("[用户中断]", "[达到最大工具调用迭代次数]"):
                    # 回溯打印前缀（流式文本已输出，这里只做标记）
                    print(f"  [{cfg.identity.name}]")
            print()
        except KeyboardInterrupt:
            store.update(lambda s: setattr(s, 'is_running', False))
            print("\n  ⏹️ 已中断\n")
        except Exception as e:
            store.update(lambda s: setattr(s, 'is_running', False))
            print(f"\n❌ 错误: {e}\n")


if __name__ == "__main__":
    main()
