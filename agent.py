"""
Agent 循环核心 — 集成 failover、hooks、compaction。

对应 OpenClaw:
- src/agents/agent-command.ts — agentCommandInternal()
- src/agents/pi-embedded-runner/run.ts — runEmbeddedPiAgent()
- src/agents/pi-embedded-subscribe.ts — subscribeEmbeddedPiSession()

Agent 循环的本质：
    while True:
        response = llm.chat(messages, tools)
        if response.has_tool_calls:
            for tool_call in response.tool_calls:
                result = execute_tool(tool_call)  # ← hooks 在这里介入
                messages.append(tool_result)
            continue
        else:
            break

增强功能：
- Failover: 主模型失败时自动切换备用模型
- Hooks: before/after tool call 钩子（拦截、日志、截断）
- Compaction: 历史过长时自动压缩
"""

import json
from config import AppConfig, resolve_api_key
from session import SessionStore, Message
from system_prompt import build_system_prompt
from tools import ToolRegistry, ToolResult, create_default_registry
from providers.base import LLMProvider, LLMResponse
from providers.openai_provider import OpenAIProvider
from hooks import (
    HookRunner, BeforeToolCallContext, AfterToolCallContext,
    create_logging_hook, create_dangerous_command_guard,
    create_output_truncation_hook,
)
from failover import run_with_failover, FailoverError
from compaction import should_compact, compact_history
from memory import LongTermMemory, extract_memories_from_conversation


def create_provider(provider_name: str, model_name: str) -> LLMProvider:
    """
    创建 LLM Provider。
    对应 OpenClaw: resolveModel() + resolveModelAsync()
    """
    api_key = resolve_api_key(provider_name)
    if not api_key:
        raise FailoverError(
            f"未找到 {provider_name} 的 API Key。"
            f"请设置环境变量: {provider_name.upper()}_API_KEY",
            reason="auth_failed",
            retryable=True,
        )

    # 兼容 OpenAI 格式的第三方 API（可在 config 中扩展）
    base_url_map = {
        "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }
    base_url = base_url_map.get(provider_name)

    return OpenAIProvider(api_key=api_key, model=model_name, base_url=base_url)


def create_hook_runner(cfg: AppConfig) -> HookRunner:
    """
    根据配置创建钩子运行器。
    对应 OpenClaw: getGlobalHookRunner() + plugin hook 注册
    """
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
    system_prompt: str,
    history: list[Message],
) -> list[dict]:
    """
    构建发送给 LLM 的消息列表。
    对应 OpenClaw: pi-embedded-runner/run.ts 中的消息构建逻辑。
    """
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
    """
    带钩子的工具执行。

    对应 OpenClaw: pi-embedded-subscribe.handlers.tools.ts 中的完整流程：
    1. before_tool_call 钩子 → 可拦截/修改参数
    2. 实际执行工具
    3. after_tool_call 钩子 → 可修改结果
    """
    # 1. before_tool_call
    before_ctx = BeforeToolCallContext(
        tool_name=tool_name,
        arguments=arguments,
        session_id=session_id,
    )
    before_result = hook_runner.run_before_tool_call(before_ctx)

    if not before_result.proceed:
        return ToolResult(
            success=False,
            output="",
            error=f"工具调用被拦截: {before_result.reason or '未知原因'}",
        )

    # 使用可能被钩子修改的参数
    final_args = before_result.modified_arguments or arguments

    # 2. 执行工具
    result = tool_registry.execute(tool_name, final_args)

    # 3. after_tool_call
    after_ctx = AfterToolCallContext(
        tool_name=tool_name,
        arguments=final_args,
        result=result,
        session_id=session_id,
    )
    result = hook_runner.run_after_tool_call(after_ctx)

    return result


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
    执行一个完整的 agent turn（集成 failover + hooks + compaction）。

    对应 OpenClaw: agentCommandInternal() + runEmbeddedPiAgent()

    完整流程：
    1. 准备：创建 provider（含 failover 列表）、构建 system prompt、加载历史
    2. Compaction：历史过长时压缩
    3. Agent 循环：LLM 调用 → 工具执行（带 hooks）→ 结果回传 → 迭代
    4. Failover：主模型失败时切换备用模型重试
    5. 收尾：保存历史、返回回复
    """
    if tool_registry is None:
        tool_registry = create_default_registry()
    if hook_runner is None:
        hook_runner = create_hook_runner(cfg)

    # === 1. 准备阶段 ===
    ltm_text = ""
    if long_term_memory:
        ltm_text = long_term_memory.format_for_prompt(query=user_message)
    system_prompt = build_system_prompt(cfg, long_term_memory_text=ltm_text)
    tools_schema = tool_registry.get_tools_for_llm(
        allow=cfg.tools.allow,
        deny=cfg.tools.deny,
    )

    # 加载会话历史
    history = session_store.get_history(session_id, limit=cfg.session.history_limit)

    # === 2. Compaction ===
    # 对应 OpenClaw: compactEmbeddedPiSession()
    if should_compact(history, threshold=cfg.agent.compaction_threshold):
        print("  📦 历史过长，正在压缩...")
        try:
            compaction_provider = create_provider(
                cfg.agent.model.provider, cfg.agent.model.model
            )
            history = compact_history(history, compaction_provider)
            # 更新 session store 中的历史（替换整个消息列表）
            session = session_store.get_or_create(session_id)
            session.messages = list(history)
            session_store._save()
            print("  📦 压缩完成")
        except Exception as e:
            print(f"  ⚠️ 压缩失败，继续使用原始历史: {e}")

    # 添加用户消息
    user_msg = Message(role="user", content=user_message)
    session_store.add_message(session_id, user_msg)
    history.append(user_msg)

    # === 3. 构建 provider（延迟创建 fallback，避免缺少 API key 时提前报错）===
    primary = create_provider(cfg.agent.model.provider, cfg.agent.model.model)

    def make_fallback(fb_ref):
        """延迟创建 fallback provider，仅在实际 failover 时才触发"""
        return create_provider(fb_ref.provider, fb_ref.model)

    fallback_refs = cfg.agent.fallback_models

    # === 4. 带 failover 的 agent 循环 ===
    def agent_loop(provider: LLMProvider) -> str:
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
        )

    if fallback_refs:
        # 延迟创建 fallback providers
        fallbacks = [make_fallback(fb) for fb in fallback_refs]
        result_text, used_provider = run_with_failover(
            primary_provider=primary,
            fallback_providers=fallbacks,
            run_fn=agent_loop,
        )
    else:
        result_text = agent_loop(primary)

    # === 5. 提取长期记忆 ===
    if long_term_memory:
        try:
            extract_memories_from_conversation(
                messages=history,
                provider=primary,
                ltm=long_term_memory,
                session_id=session_id,
            )
        except Exception as e:
            print(f"  ⚠️ 长期记忆提取失败: {e}")

    return result_text


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
) -> str:
    """
    内部工具调用循环。

    对应 OpenClaw: pi-ai Agent 内部的 tool-use 循环。
    分离出来是为了让 failover 可以用不同 provider 重试整个循环。
    """
    iteration = 0
    turn_messages: list[dict] = []

    while iteration < max_iterations:
        iteration += 1

        llm_messages = build_messages_for_llm(system_prompt, history)
        llm_messages.extend(turn_messages)

        print(f"  🔄 调用 {provider.name()}（迭代 {iteration}/{max_iterations}）...")

        try:
            response = provider.chat(
                messages=llm_messages,
                tools=tools_schema if tools_schema else None,
            )
        except Exception as e:
            # 将 API 错误包装为 FailoverError 以触发 failover
            raise FailoverError(str(e), reason="api_error", retryable=True)

        if response.usage:
            tokens = response.usage.get("total_tokens", "?")
            print(f"  📊 Token: {tokens}")

        # 没有工具调用 → 最终回复
        if not response.tool_calls:
            final_text = response.text or ""
            assistant_msg = Message(role="assistant", content=final_text)
            session_store.add_message(session_id, assistant_msg)
            return final_text

        # 处理工具调用
        assistant_tool_msg = {
            "role": "assistant",
            "content": response.text or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
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
            print(f"  🔧 {tool_call.name}({_summarize_args(tool_call.arguments)})")

            # 带钩子的工具执行
            result = execute_tool_with_hooks(
                tool_registry=tool_registry,
                hook_runner=hook_runner,
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                session_id=session_id,
            )

            status = "✅" if result.success else "❌"
            detail = result.error if result.error else "OK"
            print(f"  {status} {tool_call.name}: {detail}")

            result_content = result.output
            if result.error:
                result_content = (
                    f"Error: {result.error}\n{result.output}"
                    if result.output
                    else f"Error: {result.error}"
                )

            turn_messages.append({
                "role": "tool",
                "content": result_content,
                "tool_call_id": tool_call.id,
            })

    return "[达到最大工具调用迭代次数]"


def _summarize_args(args: dict) -> str:
    """简要显示工具参数"""
    parts = []
    for key, value in args.items():
        v = str(value)
        if len(v) > 50:
            v = v[:50] + "..."
        parts.append(f"{key}={v}")
    return ", ".join(parts) if parts else ""
