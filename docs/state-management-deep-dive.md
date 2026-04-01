# 状态管理设计 深度解析

本文档拆解 OpenClaw Lite v2 的全局状态管理，对标 Claude Code 的 `AppState` + `AppStateStore`。

## 为什么需要集中状态管理

v1 的状态散落在各处：权限模式在 config 里，token 统计在局部变量里，中断信号靠 KeyboardInterrupt。这导致：
- 模块间共享状态困难（权限检查需要知道当前模式，但模式可能被 `/mode` 命令改了）
- 统计数据无法跨函数累积
- 中断信号无法优雅传递

Claude Code 的解决方案：一个集中的 `AppState` 对象，所有模块通过 store 读写，状态变更可被监听。

## AppState 数据结构

```python
@dataclass
class AppState:
    # 权限模式
    permission_mode: str = "default"
    # 当前模型
    current_model_provider: str = ""
    current_model_name: str = ""
    # 会话
    session_id: str = "default"
    # 累计 token
    session_total_tokens: int = 0
    # 当前轮次统计
    current_turn: TurnStats = field(default_factory=TurnStats)
    # 执行控制
    is_running: bool = False
    abort_requested: bool = False
    # 流式输出
    stream_enabled: bool = True
    # 压缩追踪
    last_compaction_tokens: int = 0
    compaction_count: int = 0
```

每个字段都有明确的职责，不存储可以从其他字段推导的值。

## StateStore 设计

```python
class StateStore:
    def get(self) -> AppState          # 读取当前状态
    def update(self, fn)               # 原子更新
    def subscribe(self, listener)      # 监听变更
    def reset_turn(self)               # 重置轮次统计
    def request_abort(self)            # 请求中断
    def is_aborted(self) -> bool       # 检查中断
```

### 线程安全

`StateStore` 内部用 `threading.Lock` 保护所有读写。这在流式工具执行器（多线程）中很重要——工具在后台线程执行，主线程在接收流式输出，两者都可能读写状态。

### 原子更新

```python
store.update(lambda s: setattr(s, 'permission_mode', 'acceptEdits'))
```

`update()` 接收一个函数，在锁内执行。这保证了"读-改-写"的原子性。不会出现两个线程同时读到旧值、各自修改、互相覆盖的问题。

### 监听器

```python
store.subscribe(lambda state: print(f"Token: {state.session_total_tokens}"))
```

状态变更后自动通知所有监听器。当前主要用于调试，未来可以用于 UI 更新。

## 全局单例

```python
_global_store: StateStore | None = None

def get_store() -> StateStore:
    global _global_store
    if _global_store is None:
        _global_store = StateStore()
    return _global_store
```

所有模块通过 `get_store()` 获取同一个实例。这是最简单的依赖注入方式。

## 状态流转

### 启动时初始化

```python
# main.py
store = get_store()
store.update(lambda s: setattr(s, 'permission_mode', cfg.permissions.mode))
store.update(lambda s: setattr(s, 'stream_enabled', cfg.agent.stream))
```

### Agent 循环中更新

```python
# agent.py: 每轮开始
store.reset_turn()

# agent.py: 记录 token
store.update(lambda s: setattr(s, 'session_total_tokens',
                                s.session_total_tokens + total))

# agent.py: 压缩后
store.update(lambda s: setattr(s, 'compaction_count', s.compaction_count + 1))
```

### 中断处理

```python
# main.py: SIGINT handler
store.request_abort()

# agent.py: 循环中检查
if store.is_aborted:
    executor.discard()
    return "[用户中断]"

# agent.py: 轮次结束后重置
store.reset_turn()  # 同时重置 abort_requested
```

### 用户命令修改

```python
# /mode 命令
store.update(lambda s: setattr(s, 'permission_mode', new_mode.value))

# /stats 命令
s = store.get()
print(f"累计 Token: {s.session_total_tokens}")
```

## TurnStats 轮次统计

```python
@dataclass
class TurnStats:
    turn_number: int = 0
    tool_calls: int = 0
    api_calls: int = 0
    total_tokens: int = 0
    start_time: float = field(default_factory=time.time)
    errors: list[str] = field(default_factory=list)
```

每轮对话开始时 `reset_turn()` 重置，循环中累积。用于 `/stats` 命令展示和调试。

## 与 Claude Code 的对比

| 特性 | Claude Code | OpenClaw Lite v2 |
|------|------------|-----------------|
| 状态库 | 类 zustand 的 React store | 自实现 StateStore |
| 不可变性 | `DeepImmutable<AppState>` | 可变 dataclass |
| 状态字段 | 100+ 字段 | 12 字段 |
| 监听机制 | React useSyncExternalStore | subscribe() 回调 |
| 线程安全 | Node.js 单线程（无需锁） | threading.Lock |
| 持久化 | 部分字段持久化到 globalConfig | 仅运行时，不持久化 |

Claude Code 的 AppState 非常庞大（MCP 连接、插件状态、Bridge 状态、tmux 状态等），因为它支持 IDE 集成、远程控制、多代理协作等复杂场景。我们只保留了 Agent 循环必需的核心状态。

## TokenTracker 补充

`token_counter.py` 中的 `TokenTracker` 是状态管理的延伸，专门追踪 token 用量：

```python
class TokenTracker:
    def record_usage(self, usage: dict)     # 记录 API 返回的精确值
    def estimate_context(self, messages)     # 估算上下文 token 数
    def cumulative_tokens -> int             # 累计消耗
```

优先使用 API 返回的 `prompt_tokens`（精确），fallback 到字符估算（中文 3.5 字符/token，英文 4 字符/token）。这个估算是保守的，宁可高估触发压缩，也不要低估导致 API 报错。
