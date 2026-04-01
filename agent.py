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
    # 如果调用方没有传入工具注册表或钩子运行器，则使用默认配置创建
    # 这使得函数既可以在 main.py 中被完整配置调用，也可以单独测试
    if tool_registry is None:
        tool_registry = create_default_registry()
    if hook_runner is None:
        hook_runner = create_hook_runner(cfg)

    # ========== 1. 准备阶段 ==========
    # 从长期记忆中检索与当前用户消息相关的记忆条目，格式化为文本片段
    # 这段文本会被注入到 system prompt 中，让 LLM 在回复时能参考历史记忆
    # 例如用户之前说过"我是前端工程师"，这里就会把这条 fact 带上
    ltm_text = ""
    if long_term_memory:
        ltm_text = long_term_memory.format_for_prompt(query=user_message)

    # 构建完整的 system prompt：身份设定 + 工具说明 + 长期记忆
    system_prompt = build_system_prompt(cfg, long_term_memory_text=ltm_text)

    # 获取工具的 JSON Schema 定义，供 LLM 的 function calling 使用
    # allow/deny 控制哪些工具对 LLM 可见，deny 优先级高于 allow
    tools_schema = tool_registry.get_tools_for_llm(
        allow=cfg.tools.allow,
        deny=cfg.tools.deny,
    )

    # 从 session store 加载历史对话记录
    # history_limit 控制最多加载多少条，避免 token 爆炸
    history = session_store.get_history(session_id, limit=cfg.session.history_limit)

    # ========== 2. Compaction（上下文压缩） ==========
    # 当历史消息数超过 compaction_threshold（默认 40 条）时触发压缩
    # 压缩原理：调 LLM 把冗长的历史对话总结成一条精简摘要，替换原始历史
    # 目的：控制后续 LLM 调用的 token 消耗，避免超出上下文窗口
    # 对应 OpenClaw: compactEmbeddedPiSession()
    if should_compact(history, threshold=cfg.agent.compaction_threshold):
        print("  📦 历史过长，正在压缩...")
        try:
            # 用当前主模型来做压缩（也可以用更便宜的模型，但这里复用主模型）
            compaction_provider = create_provider(
                cfg.agent.model.provider, cfg.agent.model.model
            )
            history = compact_history(history, compaction_provider)
            # 压缩后的历史写回 session store，持久化保存
            # 这样下次加载时直接用压缩后的版本，不会重复压缩
            session = session_store.get_or_create(session_id)
            session.messages = list(history)
            session_store._save()
            print("  📦 压缩完成")
        except Exception as e:
            # 压缩失败不是致命错误，降级为使用原始历史继续运行
            print(f"  ⚠️ 压缩失败，继续使用原始历史: {e}")

    # 将本轮用户消息追加到历史中，同时持久化到 session store
    user_msg = Message(role="user", content=user_message)
    session_store.add_message(session_id, user_msg)
    history.append(user_msg)

    # ========== 2.5 实时记忆检测（记忆系统第二层） ==========
    # 用正则规则扫描用户消息，检测高价值信息（身份、偏好、技术栈等）
    # 命中则立刻写入长期记忆，不调 LLM，零成本
    # needs_urgent_extract 标记：如果检测到了强信号，后面会提前触发 LLM 批量提取
    # 这样即使对话轮数没攒够 BATCH_EXTRACT_INTERVAL，重要信息也不会丢
    needs_urgent_extract = False
    if long_term_memory:
        needs_urgent_extract = long_term_memory.detect_and_store(user_message)

    # ========== 3. 构建 LLM Provider ==========
    # 创建主模型的 provider 实例（包含 API Key、base_url、模型名等）
    primary = create_provider(cfg.agent.model.provider, cfg.agent.model.model)

    # fallback provider 采用延迟创建策略：
    # 只在主模型实际失败触发 failover 时才实例化备用 provider
    # 好处：如果用户没配备用模型的 API Key，只要主模型正常就不会报错
    def make_fallback(fb_ref):
        return create_provider(fb_ref.provider, fb_ref.model)

    fallback_refs = cfg.agent.fallback_models

    # ========== 4. 带 Failover 的 Agent 循环（核心） ==========
    # agent_loop 封装了内层的工具调用循环 _run_tool_loop
    # 它接收一个 provider 参数，这样 failover 机制可以用不同的 provider 重试整个循环
    # 流程：LLM 生成回复 → 如果包含工具调用 → 执行工具 → 结果喂回 LLM → 循环
    #       如果 LLM 返回纯文本（无工具调用）→ 循环结束，返回最终回复
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
        # 有备用模型时，用 run_with_failover 包装：
        # 主模型抛出 FailoverError → 自动切换到 fallback 列表中的下一个模型重试
        fallbacks = [make_fallback(fb) for fb in fallback_refs]
        result_text, used_provider = run_with_failover(
            primary_provider=primary,
            fallback_providers=fallbacks,
            run_fn=agent_loop,
        )
    else:
        # 没有备用模型，直接用主模型跑，失败就失败
        result_text = agent_loop(primary)

    # ========== 5. 批量记忆提取（记忆系统第三层） ==========
    # 触发条件（满足任一即可）：
    #   a) needs_urgent_extract=True：第 2.5 步正则检测到了强信号词
    #   b) should_batch_extract()=True：对话轮数攒够了 BATCH_EXTRACT_INTERVAL（默认 10 轮）
    # 提取方式：调 LLM 分析整段对话历史，提取正则抓不到的隐含信息
    # 例如从技术讨论中推断用户熟悉 React，或从问题风格推断用户是初学者
    if long_term_memory and (needs_urgent_extract or long_term_memory.should_batch_extract()):
        try:
            extract_memories_from_conversation(
                messages=history,
                provider=primary,
                ltm=long_term_memory,
                session_id=session_id,
            )
        except Exception as e:
            # 记忆提取失败不影响本轮回复，静默降级
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
    内部工具调用循环（ReAct 模式的核心实现）。

    对应 OpenClaw: pi-ai Agent 内部的 tool-use 循环。
    单独抽出来是为了让 failover 可以用不同 provider 重试整个循环。

    循环逻辑：
        发消息给 LLM → LLM 返回工具调用 → 执行工具 → 结果追加到消息 → 再发给 LLM
        直到 LLM 返回纯文本（不再调用工具），循环结束
    """
    iteration = 0
    # turn_messages 存放本轮 agent 循环中产生的中间消息（assistant 的工具调用 + tool 的执行结果）
    # 这些消息不直接写入 session store，而是在每次调 LLM 时临时拼接到历史后面
    # 只有最终的纯文本回复才会持久化到 session store
    turn_messages: list[dict] = []

    while iteration < max_iterations:
        iteration += 1

        # 每次迭代都重新构建完整的消息列表：system prompt + 历史 + 本轮中间消息
        # 这样 LLM 能看到之前工具调用的结果，决定是否继续调用工具
        llm_messages = build_messages_for_llm(system_prompt, history)
        llm_messages.extend(turn_messages)

        print(f"  🔄 调用 {provider.name()}（迭代 {iteration}/{max_iterations}）...")

        try:
            response = provider.chat(
                messages=llm_messages,
                tools=tools_schema if tools_schema else None,
            )
        except Exception as e:
            # API 调用失败时包装为 FailoverError，外层的 run_with_failover 会捕获并切换备用模型
            raise FailoverError(str(e), reason="api_error", retryable=True)

        # 打印 token 用量，方便监控成本
        if response.usage:
            tokens = response.usage.get("total_tokens", "?")
            print(f"  📊 Token: {tokens}")

        # === 判断是否结束循环 ===
        # LLM 没有返回工具调用 → 说明它认为已经收集到足够信息，给出了最终回复
        if not response.tool_calls:
            final_text = response.text or ""
            # 最终回复持久化到 session store，下次对话时作为历史加载
            assistant_msg = Message(role="assistant", content=final_text)
            session_store.add_message(session_id, assistant_msg)
            return final_text

        # === 处理工具调用 ===
        # LLM 返回了一个或多个工具调用请求，需要逐个执行并把结果喂回去

        # 先把 LLM 的工具调用请求构造成 OpenAI 格式的 assistant 消息
        # 这条消息会在下次迭代时发给 LLM，让它知道自己之前请求了哪些工具
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

        # 逐个执行工具调用
        for tool_call in response.tool_calls:
            print(f"  🔧 {tool_call.name}({_summarize_args(tool_call.arguments)})")

            # 通过 hooks 执行工具：before 钩子可拦截/修改参数，after 钩子可修改结果
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

            # 构造工具执行结果消息，格式遵循 OpenAI 的 tool message 规范
            # tool_call_id 用于将结果与对应的工具调用请求关联
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

        # 本轮工具全部执行完毕，回到循环顶部，把结果发给 LLM 让它决定下一步

    # 超过最大迭代次数，强制终止，防止 LLM 陷入无限工具调用循环
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
