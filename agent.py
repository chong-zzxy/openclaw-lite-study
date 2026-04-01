"""
Agent 循环核心 v2 — 集成流式输出、流式工具执行、权限系统、容错机制。

对标 Claude Code: src/query.ts 的 queryLoop()

核心循环：
    while True:
        response = llm.stream_chat(messages, tools)  # 流式输出
        for chunk in response:
            if text_delta → 实时打印
            if tool_call_done → 权限检查 → 提交流式执行器
        if no tool_calls → 执行 stop hooks → break
        收集工具结果 → 回传 → 继续循环

容错机制：
- Failover: 主模型失败时切换备用模型
- Compaction: token 超阈值时自动压缩（工具结果截断 + LLM 摘要）
- max_output_tokens 恢复: 输出被截断时注入"继续"指令
- 中断处理: Ctrl+C 优雅退出
"""

import json
import sys
import time
from config import AppConfig, resolve_api_key
from session import SessionStore, Message
from system_prompt import build_system_prompt
from tools.registry import ToolRegistry, ToolResult, create_default_registry
from providers.base import LLMProvider, LLMResponse, StreamChunk
from providers.openai_provider import OpenAIProvider
from hooks import (
    HookRunner, BeforeToolCallContext, AfterToolCallContext,
    create_logging_hook, create_dangerous_command_guard,
    create_output_truncation_hook,
)
from failover import run_with_failover, FailoverError
from compaction import should_compact, compact_history
from memory import LongTermMemory, extract_memories_from_conversation
from permissions import (
    PermissionMode, Decision, check_permission,
    ask_user_permission, format_mode,
)
from streaming import StreamingToolExecutor, ToolExecResult
from token_counter import TokenTracker
from state import get_store


def create_provider(provider_name: str, model_name: str) -> LLMProvider:
    """创建 LLM Provider"""
    api_key = resolve_api_key(provider_name)
    if not api_key:
        raise FailoverError(
            f"未找到 {provider_name} 的 API Key。"
            f"请设置环境变量: {provider_name.upper()}_API_KEY",
            reason="auth_failed", retryable=True,
        )
    base_url_map = {
        "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }
    return OpenAIProvider(
        api_key=api_key, model=model_name,
        base_url=base_url_map.get(provider_name),
    )


def create_hook_runner(cfg: AppConfig) -> HookRunner:
    """根据配置创建钩子运行器"""
    runner = HookRunner()
    if cfg.hooks.enable_logging:
        before_log, after_log = create_logging_hook()
        runner.register_before_tool_call(before_log)
        runner.register_after_tool_call(after_log)
    if cfg.hooks.enable_dangerous_command_guard:
        runner.register_before_tool_call(create_dangerous_command_guard())
    if cfg.hooks.output_truncation_chars > 0:
        runner.register_after_tool_call(
            create_output_truncation_hook(cfg.hooks.output_truncation_chars)
        )
    return runner


def build_messages_for_llm(
    system_prompt: str, history: list[Message],
) -> list[dict]:
    """构建发送给 LLM 的消息列表"""
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        if msg.role == "tool":
            messages.append({
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id or "",
            })
        else:
            messages.append({"role": msg.role, "content": msg.content})
    return messages


def execute_tool_with_hooks(
    tool_registry: ToolRegistry,
    hook_runner: HookRunner,
    tool_name: str,
    arguments: dict,
    session_id: str,
) -> ToolResult:
    """带钩子的工具执行"""
    before_ctx = BeforeToolCallContext(
        tool_name=tool_name, arguments=arguments, session_id=session_id,
    )
    before_result = hook_runner.run_before_tool_call(before_ctx)
    if not before_result.proceed:
        return ToolResult(
            success=False, output="",
            error=f"工具调用被拦截: {before_result.reason or '未知原因'}",
        )
    final_args = before_result.modified_arguments or arguments
    result = tool_registry.execute(tool_name, final_args)
    after_ctx = AfterToolCallContext(
        tool_name=tool_name, arguments=final_args,
        result=result, session_id=session_id,
    )
    result = hook_runner.run_after_tool_call(after_ctx)
    return result


def _check_tool_permission(
    cfg: AppConfig | None,
    tool_registry: ToolRegistry,
    tool_name: str,
    arguments: dict,
) -> bool:
    """
    权限检查。对标 Claude Code 的 checkPermissions() + canUseTool()。
    返回 True 表示允许执行，False 表示拒绝。
    """
    store = get_store()
    mode = PermissionMode.from_string(store.get().permission_mode)
    is_ro = tool_registry.is_read_only(tool_name)
    is_dest = tool_registry.is_destructive(tool_name)

    decision = check_permission(
        mode=mode, tool_name=tool_name, arguments=arguments,
        is_read_only=is_ro, is_destructive=is_dest,
    )

    if decision == Decision.ALLOW:
        return True
    if decision == Decision.DENY:
        print(f"  🚫 Plan 模式下不允许执行 {tool_name}")
        return False
    # ASK
    allowed = ask_user_permission(tool_name, arguments)
    if not allowed:
        print(f"  ⏭️ 用户拒绝执行 {tool_name}")
    return allowed


def run_agent_turn(
    user_message: str,
    cfg: AppConfig,
    session_store: SessionStore,
    session_id: str = "default",
    tool_registry: ToolRegistry | None = None,
    hook_runner: HookRunner | None = None,
    long_term_memory: LongTermMemory | None = None,
) -> str:
    """
    执行一个完整的 agent turn。
    对标 Claude Code: agentCommandInternal() + queryLoop()

    完整流程：
    1. 准备：创建 provider、构建 system prompt、加载历史
    2. Compaction：token 超阈值时压缩
    3. Agent 循环：流式 LLM 调用 → 权限检查 → 流式工具执行 → 迭代
    4. Failover：主模型失败时切换备用
    5. 收尾：保存历史、提取记忆
    """
    if tool_registry is None:
        tool_registry = create_default_registry()
    if hook_runner is None:
        hook_runner = create_hook_runner(cfg)

    store = get_store()
    store.reset_turn()
    token_tracker = TokenTracker()

    # === 1. 准备 ===
    ltm_text = ""
    if long_term_memory:
        ltm_text = long_term_memory.format_for_prompt(query=user_message)
    system_prompt = build_system_prompt(cfg, long_term_memory_text=ltm_text)
    tools_schema = tool_registry.get_tools_for_llm(
        allow=cfg.tools.allow, deny=cfg.tools.deny,
    )

    history = session_store.get_history(session_id, limit=cfg.session.history_limit)

    # === 2. Compaction ===
    if should_compact(history, token_threshold=cfg.agent.compaction_token_threshold):
        print("  📦 上下文过长，正在压缩...")
        try:
            comp_provider = create_provider(
                cfg.agent.model.provider, cfg.agent.model.model,
            )
            history = compact_history(
                history, comp_provider,
                keep_recent=cfg.agent.compaction_keep_recent,
                token_threshold=cfg.agent.compaction_token_threshold,
            )
            session = session_store.get_or_create(session_id)
            session.messages = list(history)
            session_store._save()
            store.update(lambda s: setattr(s, 'compaction_count',
                                           s.compaction_count + 1))
        except Exception as e:
            print(f"  ⚠️ 压缩失败: {e}")

    # 添加用户消息
    user_msg = Message(role="user", content=user_message)
    session_store.add_message(session_id, user_msg)
    history.append(user_msg)

    # 实时检测记忆
    needs_urgent_extract = False
    if long_term_memory:
        needs_urgent_extract = long_term_memory.detect_and_store(user_message)

    # === 3. 构建 provider ===
    primary = create_provider(cfg.agent.model.provider, cfg.agent.model.model)
    fallback_refs = cfg.agent.fallback_models

    # === 4. 带 failover 的 agent 循环 ===
    use_stream = cfg.agent.stream

    def agent_loop(provider: LLMProvider) -> str:
        if use_stream:
            return _run_streaming_tool_loop(
                provider=provider,
                system_prompt=system_prompt,
                history=history,
                tools_schema=tools_schema,
                tool_registry=tool_registry,
                hook_runner=hook_runner,
                session_store=session_store,
                session_id=session_id,
                cfg=cfg,
                token_tracker=token_tracker,
            )
        else:
            return _run_tool_loop(
                provider=provider,
                system_prompt=system_prompt,
                history=history,
                tools_schema=tools_schema,
                tool_registry=tool_registry,
                hook_runner=hook_runner,
                session_store=session_store,
                session_id=session_id,
                max_iterations=cfg.agent.max_tool_iterations,
                token_tracker=token_tracker,
            )

    if fallback_refs:
        fallbacks = [create_provider(fb.provider, fb.model) for fb in fallback_refs]
        result_text, _ = run_with_failover(
            primary_provider=primary,
            fallback_providers=fallbacks,
            run_fn=agent_loop,
        )
    else:
        result_text = agent_loop(primary)

    # === 5. 记忆提取 ===
    if long_term_memory and (needs_urgent_extract
                             or long_term_memory.should_batch_extract()):
        try:
            extract_memories_from_conversation(
                messages=history, provider=primary,
                ltm=long_term_memory, session_id=session_id,
            )
        except Exception as e:
            print(f"  ⚠️ 记忆提取失败: {e}")

    return result_text


# 最大输出截断恢复次数（对标 Claude Code 的 MAX_OUTPUT_TOKENS_RECOVERY_LIMIT）
MAX_OUTPUT_RECOVERY = 2


def _run_streaming_tool_loop(
    provider: LLMProvider,
    system_prompt: str,
    history: list[Message],
    tools_schema: list[dict],
    tool_registry: ToolRegistry,
    hook_runner: HookRunner,
    session_store: SessionStore,
    session_id: str,
    cfg: AppConfig,
    token_tracker: TokenTracker,
) -> str:
    """
    流式工具调用循环。对标 Claude Code 的 queryLoop() 核心。

    关键改进：
    1. 流式输出：模型边生成边打印
    2. 流式工具执行：tool_call 解析完立即提交执行
    3. 权限检查：执行前检查权限
    4. max_output_tokens 恢复：输出被截断时注入继续指令
    5. 中断处理：检查 abort 信号
    """
    store = get_store()
    iteration = 0
    max_iterations = cfg.agent.max_tool_iterations
    turn_messages: list[dict] = []
    output_recovery_count = 0

    while iteration < max_iterations:
        iteration += 1
        store.update(lambda s: setattr(s, 'current_turn',
                     type(s.current_turn)(turn_number=iteration)))

        # 检查中断
        if store.is_aborted:
            return "[用户中断]"

        llm_messages = build_messages_for_llm(system_prompt, history)
        llm_messages.extend(turn_messages)

        print(f"  🔄 调用 {provider.name()}（迭代 {iteration}/{max_iterations}）...")

        # 创建流式工具执行器
        executor = StreamingToolExecutor(
            registry=tool_registry,
            execute_fn=lambda name, args: execute_tool_with_hooks(
                tool_registry, hook_runner, name, args, session_id,
            ),
        )

        try:
            text_buffer = ""
            tool_calls_collected = []
            finish_reason = ""
            usage = None
            text_started = False  # 是否已开始输出文本

            # === 流式接收 ===
            for chunk in provider.stream_chat(
                messages=llm_messages,
                tools=tools_schema if tools_schema else None,
            ):
                # 中断检查
                if store.is_aborted:
                    executor.discard()
                    return "[用户中断]"

                if chunk.type == "text_delta":
                    if not text_started:
                        # 首次输出文本时打印身份前缀
                        sys.stdout.write(f"{cfg.identity.emoji} ")
                        text_started = True
                    text_buffer += chunk.text
                    sys.stdout.write(chunk.text)
                    sys.stdout.flush()

                elif chunk.type == "tool_call_start":
                    pass  # 等 tool_call_done 再处理

                elif chunk.type == "tool_call_done" and chunk.tool_call:
                    tc = chunk.tool_call
                    tool_calls_collected.append(tc)

                    # 权限检查
                    if _check_tool_permission(
                        cfg, tool_registry, tc.name, tc.arguments,
                    ):
                        is_ro = tool_registry.is_read_only(tc.name)
                        print(f"\n  🔧 {tc.name}({_summarize_args(tc.arguments)})")
                        # 提交到流式执行器（边流边执行）
                        executor.add_tool(
                            tc.id, tc.name, tc.arguments, is_read_only=is_ro,
                        )
                    else:
                        # 权限拒绝，生成拒绝结果
                        tool_calls_collected[-1] = None  # 标记为已拒绝
                        turn_messages.append({
                            "role": "tool",
                            "content": f"Permission denied: {tc.name} 被用户拒绝执行",
                            "tool_call_id": tc.id,
                        })

                elif chunk.type == "usage":
                    usage = chunk.usage
                    token_tracker.record_usage(usage)

                elif chunk.type == "done":
                    finish_reason = chunk.finish_reason

            # 流结束后换行
            if text_buffer:
                sys.stdout.write("\n")
                sys.stdout.flush()

        except Exception as e:
            executor.discard()
            raise FailoverError(str(e), reason="api_error", retryable=True)

        # 记录 token
        if usage:
            total = usage.get("total_tokens", 0)
            store.update(lambda s: setattr(
                s, 'session_total_tokens', s.session_total_tokens + total))
            print(f"  📊 Token: {total}")

        # === 处理结果 ===
        valid_tool_calls = [tc for tc in tool_calls_collected if tc is not None]

        # 没有工具调用 → 最终回复
        if not valid_tool_calls:
            # max_output_tokens 恢复
            # 对标 Claude Code 的 isWithheldMaxOutputTokens + recovery
            if (finish_reason == "length"
                    and output_recovery_count < MAX_OUTPUT_RECOVERY):
                output_recovery_count += 1
                print(f"  ⚠️ 输出被截断，自动恢复（{output_recovery_count}/{MAX_OUTPUT_RECOVERY}）...")
                # 保存已有文本
                if text_buffer:
                    turn_messages.append({
                        "role": "assistant", "content": text_buffer,
                    })
                # 注入恢复指令
                turn_messages.append({
                    "role": "user",
                    "content": (
                        "Output token limit hit. Resume directly — "
                        "no apology, no recap. Pick up mid-thought. "
                        "Break remaining work into smaller pieces."
                    ),
                })
                continue

            final_text = text_buffer or ""
            assistant_msg = Message(role="assistant", content=final_text)
            session_store.add_message(session_id, assistant_msg)
            executor.shutdown()
            return final_text

        # 有工具调用 → 构建 assistant 消息并收集结果
        assistant_tool_msg = {
            "role": "assistant",
            "content": text_buffer or None,
            "tool_calls": [
                {
                    "id": tc.id, "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in valid_tool_calls
            ],
        }
        turn_messages.append(assistant_tool_msg)

        # 收集流式执行器中已完成的结果
        completed = executor.get_completed_results()
        # 等待剩余结果
        remaining = executor.get_remaining_results()
        all_results = completed + remaining

        for res in all_results:
            status = "✅" if res.result.success else "❌"
            detail = res.result.error if res.result.error else "OK"
            print(f"  {status} {res.tool_name}: {detail}")
            turn_messages.append(res.to_message())

        executor.shutdown()
        output_recovery_count = 0  # 工具调用成功后重置

    return "[达到最大工具调用迭代次数]"


def _run_tool_loop(
    provider: LLMProvider,
    system_prompt: str,
    history: list[Message],
    tools_schema: list[dict],
    tool_registry: ToolRegistry,
    hook_runner: HookRunner,
    session_store: SessionStore,
    session_id: str,
    max_iterations: int,
    token_tracker: TokenTracker,
) -> str:
    """
    非流式工具调用循环（fallback 模式）。
    保留 v1 的同步逻辑，但加入权限检查和 token 追踪。
    """
    store = get_store()
    iteration = 0
    turn_messages: list[dict] = []

    while iteration < max_iterations:
        iteration += 1

        if store.is_aborted:
            return "[用户中断]"

        llm_messages = build_messages_for_llm(system_prompt, history)
        llm_messages.extend(turn_messages)

        print(f"  🔄 调用 {provider.name()}（迭代 {iteration}/{max_iterations}）...")

        try:
            response = provider.chat(
                messages=llm_messages,
                tools=tools_schema if tools_schema else None,
            )
        except Exception as e:
            raise FailoverError(str(e), reason="api_error", retryable=True)

        if response.usage:
            token_tracker.record_usage(response.usage)
            total = response.usage.get("total_tokens", "?")
            total_int = response.usage.get("total_tokens", 0)
            store.update(lambda s: setattr(
                s, 'session_total_tokens',
                s.session_total_tokens + total_int))
            print(f"  📊 Token: {total}")

        if not response.tool_calls:
            final_text = response.text or ""
            assistant_msg = Message(role="assistant", content=final_text)
            session_store.add_message(session_id, assistant_msg)
            return final_text

        assistant_tool_msg = {
            "role": "assistant",
            "content": response.text or None,
            "tool_calls": [
                {
                    "id": tc.id, "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in response.tool_calls
            ],
        }
        turn_messages.append(assistant_tool_msg)

        for tool_call in response.tool_calls:
            # 权限检查
            if not _check_tool_permission(
                cfg=None, tool_registry=tool_registry,
                tool_name=tool_call.name, arguments=tool_call.arguments,
            ):
                turn_messages.append({
                    "role": "tool",
                    "content": f"Permission denied: {tool_call.name} 被拒绝",
                    "tool_call_id": tool_call.id,
                })
                continue

            print(f"  🔧 {tool_call.name}({_summarize_args(tool_call.arguments)})")
            result = execute_tool_with_hooks(
                tool_registry, hook_runner,
                tool_call.name, tool_call.arguments, session_id,
            )
            status = "✅" if result.success else "❌"
            detail = result.error if result.error else "OK"
            print(f"  {status} {tool_call.name}: {detail}")

            result_content = result.output
            if result.error:
                result_content = (
                    f"Error: {result.error}\n{result.output}"
                    if result.output else f"Error: {result.error}"
                )
            turn_messages.append({
                "role": "tool",
                "content": result_content,
                "tool_call_id": tool_call.id,
            })

    return "[达到最大工具调用迭代次数]"


def _summarize_args(args: dict) -> str:
    parts = []
    for key, value in args.items():
        v = str(value)
        if len(v) > 50:
            v = v[:50] + "..."
        parts.append(f"{key}={v}")
    return ", ".join(parts) if parts else ""
