# 权限系统设计 深度解析

本文档详细拆解 OpenClaw Lite v2 的四级权限系统，对标 Claude Code 的 `PermissionMode` 和 `checkPermissions()` 机制。

## 设计动机

AI Agent 能执行 shell 命令、写文件、删文件——这些操作有真实的副作用。没有权限控制的 Agent 就像一个拥有 root 权限的实习生，随时可能 `rm -rf /`。

Claude Code 的解决方案是分级权限：让用户在"安全"和"效率"之间自由选择。想要安全？每次操作都确认。想要效率？自动放行编辑操作。想要极致效率？跳过所有权限。

## 四级权限模式

```
安全 ←————————————————————————————→ 效率
plan    default    acceptEdits    bypassPermissions
```

### plan（只规划不执行）
所有写操作直接拒绝，Agent 只能读取信息和生成计划。适合先让 Agent 分析问题、制定方案，确认后再切换到其他模式执行。

### default（默认模式）
- 只读工具（read, glob, grep, web_search）：自动放行
- 安全的只读命令（ls, git status, cat 等）：自动放行
- 其他写操作（write, edit, exec）：需要用户确认

### acceptEdits（接受编辑）
- 只读工具：自动放行
- 文件编辑（write, edit）：自动放行
- 安全只读命令：自动放行
- 危险命令（rm -rf, sudo 等）：仍需确认
- 其他 shell 命令：需要确认

### bypassPermissions（跳过权限）
所有操作直接放行，不做任何检查。仅在完全信任 Agent 时使用。

## 工具分类

每个工具在注册时标记两个属性：

| 工具 | read_only | destructive | 说明 |
|------|-----------|-------------|------|
| read | ✅ | ❌ | 只读，所有模式放行 |
| glob | ✅ | ❌ | 只读，所有模式放行 |
| grep | ✅ | ❌ | 只读，所有模式放行 |
| web_search | ✅ | ❌ | 只读，所有模式放行 |
| write | ❌ | ❌ | 写文件，acceptEdits 放行 |
| edit | ❌ | ❌ | 编辑文件，acceptEdits 放行 |
| exec | ❌ | ✅ | 执行命令，需进一步分析命令内容 |

`read_only` 决定是否在所有模式下自动放行。
`destructive` 决定是否在 acceptEdits 模式下仍需确认。

## 命令安全分析

exec 工具比较特殊——同一个工具，`ls` 和 `rm -rf /` 的风险完全不同。所以对 exec 有额外的命令级分析：

### 危险命令检测（正则匹配）

```python
DANGEROUS_PATTERNS = [
    r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|--recursive)\b",  # rm -rf
    r"\bsudo\b",           # sudo
    r"\bmkfs\b",           # 格式化
    r"\bdd\s+if=",         # dd
    r">\s*/dev/",          # 写设备
    r"\bchmod\s+777\b",    # 开放权限
    r"\bcurl\b.*\|\s*(ba)?sh",  # curl | sh
]
```

### 安全只读命令白名单

```python
SAFE_READ_COMMANDS = [
    "ls", "cat", "head", "tail", "wc", "find", "grep", "rg",
    "git status", "git log", "git diff", "git branch",
    "pwd", "echo", "date", "which", "file",
    "python --version", "node --version",
]
```


## 权限判断流程

```python
def check_permission(mode, tool_name, arguments, is_read_only, is_destructive):
    # 1. 只读工具 → 所有模式放行
    if is_read_only:
        return ALLOW

    # 2. bypass → 全部放行
    if mode == BYPASS:
        return ALLOW

    # 3. plan → 拒绝所有写操作
    if mode == PLAN:
        return DENY

    # 4. acceptEdits
    if mode == ACCEPT_EDITS:
        if tool_name in ("write", "edit"):
            return ALLOW if not is_destructive else ASK
        if tool_name == "exec":
            if is_dangerous_command(command):
                return ASK
            if is_safe_read_command(command):
                return ALLOW
            return ASK
        return ASK

    # 5. default
    if tool_name == "exec" and is_safe_read_command(command):
        return ALLOW
    return ASK
```

## 用户交互

当权限判断返回 ASK 时，终端显示确认提示：

```
  🔒 权限确认: exec
     命令: npm install express
     允许执行? [y/N/a(本次全部允许)]
```

三个选项：
- `y` — 允许本次执行
- `N`（默认）— 拒绝，Agent 收到 "Permission denied" 错误
- `a` — 允许，并将权限模式切换为 acceptEdits（本次会话内）

## 运行时切换

通过 `/mode` 命令随时切换：

```
你> /mode
  当前权限模式: Default (写操作需确认)
  可选: default, acceptEdits, plan, bypassPermissions

你> /mode acceptEdits
  ✅ 权限模式切换为: Accept Edits (文件编辑自动放行)
```

权限模式存储在 `StateStore` 中，所有模块通过 `get_store().get().permission_mode` 读取。

## 与 Claude Code 的对比

| 特性 | Claude Code | OpenClaw Lite v2 |
|------|------------|-----------------|
| 模式数量 | 6 个（含 auto, bubble） | 4 个（核心模式） |
| 工具分类 | `isReadOnly()` / `isDestructive()` | `read_only` / `destructive` 字段 |
| 命令分析 | bashClassifier（ML 分类器） | 正则匹配 + 白名单 |
| 用户确认 | 终端 UI 组件 | input() 交互 |
| 持久化 | 全局配置 + 会话级覆盖 | config.json + 运行时 StateStore |
| 自动模式 | auto（AI 自主判断） | 未实现 |

Claude Code 的 `auto` 模式使用 ML 分类器判断命令安全性，这是一个更高级的方案。我们用正则 + 白名单作为简化替代，覆盖了最常见的场景。