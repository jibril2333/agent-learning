# Plan-and-Execute Agent

## 架构概述

Plan-and-Execute 是一种带 Replanner 的 Agent 架构，核心流程为：

```
Plan → Execute → Replan → Execute → ... → Synthesize
```

1. **Planner** - 根据用户请求生成多步计划（1-5 步）
2. **Executor** - 逐步执行计划，内部支持多轮工具调用（类 ReAct 循环）
3. **Replanner** - 每步执行后审视剩余计划，决定继续或调整
4. **Synthesizer** - 整合所有步骤结果，生成最终回答

## 信息流

```
Planner（只看用户输入）
   ↓ steps
Executor（调用工具，内部消化结果，输出文本总结）
   ↓ 文本总结 → context
Replanner（看用户输入 + context + 剩余步骤）
   ↓ 调整后的 steps
Executor ...
   ↓
Synthesizer（看用户输入 + 所有 context）
```

关键点：
- Planner 只在开头调用一次，后续计划调整全由 Replanner 负责
- 工具原始返回值被 Executor 内部的 LLM "消化"为自然语言总结后，才向上传递
- Replanner 看到的是文本总结，不是原始工具输出

## Executor 内部的多轮工具调用

`execute_step` 内部有一个循环（最多 `max_tool_calls` 轮），本质是一个单步内的 ReAct 循环：

1. 请求 LLM，如果不需要调工具则直接返回文本
2. 如果需要调工具，执行工具调用，把结果追加到对话历史
3. 重复 1-2，直到 LLM 输出文本回答或达到上限
4. 循环结束后额外请求一次 LLM，确保最后一轮工具调用成功后也能输出文本

这使得 Executor 能在一步内自我纠错（例如参数格式错误后自动修正重试）。

## 发现的问题与修复

### 问题：Executor 失败时丢失错误上下文

原始代码中，`execute_step` 达到最大迭代次数后返回硬编码的：

```python
return "Failed to complete step within max iterations."
```

Replanner 只能看到"失败了"，但不知道**为什么**失败。例如 `weather_forecast('Shanghai')` 返回"错误：请用中文搜索"，这个关键信息被完全丢弃，导致 Replanner 无法对症下药，陷入无限重试的死循环。

**修复**：记录最后一次工具调用的结果，失败时一并返回：

```python
last_tool_result = ""
max_tool_calls = 5
for i in range(max_tool_calls):
    # ... 工具调用逻辑 ...
    result = available_tools[func_name](**func_args)
    last_tool_result = f"{func_name}({func_args}) -> {result}"
    # ...

# 循环结束后再请求一次 LLM 生成文本总结
response = openai.chat.completions.create(...)
msg = response.choices[0].message
if not msg.tool_calls:
    return msg.content

return f"Failed after {max_tool_calls} iterations. Last tool result: {last_tool_result}"
```

修复后 Replanner 能看到完整的错误信息，例如：

```
Failed after 5 iterations. Last tool result: weather_forecast({'city': 'Shanghai'}) -> 错误：请用中文搜索
```

从而正确调整后续计划（改用中文参数或换用其他工具）。

### 问题：循环结束后缺少最终 LLM 调用

原始代码中，如果工具调用恰好在最后一轮循环成功，没有机会再进入下一轮让 LLM 生成文本回答，直接走到末尾返回"Failed"——明明成功了却被误报为失败。

**修复**：在 `for` 循环结束后额外做一次 LLM 调用，让它有机会根据最后一次工具结果输出文本总结。

## 测试用例

| 用例 | 覆盖场景 |
|------|---------|
| `北京今天天气怎么样？` | 单工具、单步骤 |
| `比较上海和东京的天气，哪个更适合出行？` | 单工具、多步骤 |
| `我周末打算去杭州露营，帮我看看天气，再搜索推荐的露营地点` | 多工具协作 |
| `2026年有哪些值得关注的科技大会？` | 纯搜索 |
| `1+1等于几？` | 纯推理，不调用工具 |
| `帮我查深圳天气，天气不好搜室内活动，天气好搜户外活动` | 触发 Replan |
| `下周去日本，查东京大阪天气，搜热门景点和交通攻略` | 复杂多步（接近上限） |
| `asdfghjkl` | 边界/异常输入 |

## 备注

- Replanner 的决策是 LLM 的概率性行为，同样的错误信息可能导致不同的调整策略（修正参数重试 vs 换用其他工具），这取决于上下文措辞和模型推理
- 如需更可控的 Replanner 行为，可在 `REPLANNER_PROMPT` 中添加具体规则引导
