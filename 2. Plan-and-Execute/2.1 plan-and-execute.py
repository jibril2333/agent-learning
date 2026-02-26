import openai
from dotenv import load_dotenv
import os
import json

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def weather_forecast(city: str) -> str:
    return f"sunny, 22°C"

def web_search(query: str) -> str:
    return f"Precipitation: 10mm, Wind: 10km/h"

tools = [
    {
        "type": "function",
        "function": {
            "name": "weather_forecast",
            "description": "Get the weather forecast for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city":{"type": "string", "description": "The city name"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters":{
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    }
]

available_tools = {
    "weather_forecast": weather_forecast,
    "web_search": web_search,
}

# ============================================================
# Prompts
# ============================================================

def get_tool_descriptions() -> str:
    return "\n".join(
        f"- {t['function']['name']}: {t['function']['description']}"
        for t in tools
    )

PLANNER_PROMPT = """\
You are a planner.
Given a user request, break it down into a list of steps.

Rules:
- Each step should be a clear, actionable task.
- Keep the plan concise (1-5 steps).
- Only plan steps that can be done with the available tools or by direct reasoning.
- Do NOT plan meta-steps like "choose data source" or "confirm location".

Available tools:
{tool_descriptions}

Output format:
Return a JSON object with a "steps" key containing an array of step strings."""

EXECUTOR_PROMPT = """\
You are an executor.
Complete the given task using the available tools or by direct reasoning.

{context_str}"""

REPLANNER_PROMPT = """\
You are a replanner.
Given the original task, completed steps with results, and remaining steps,
decide if the remaining plan needs adjustment.

Rules:
- Prefer "continue" unless results clearly show the plan needs change.
- Keep any revised plan minimal (no more than 5 remaining steps).
- Prefer removing unnecessary steps over adding new ones.

Available tools:
{tool_descriptions}

Output format:
Return a JSON object with:
- "action": "continue" or "replan"
- "steps": the remaining steps (only required if action is "replan")"""

SYNTHESIZER_PROMPT = """\
Based on the following completed steps, provide a final comprehensive answer to the user."""

# ============================================================
# Functions
# ============================================================

def plan(user_input: str) -> list:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": PLANNER_PROMPT.format(
                    tool_descriptions=get_tool_descriptions()
                ),
            },
            {"role": "user", "content": user_input},
        ],
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)

    if isinstance(result, dict):
        steps = result.get("steps", list(result.values())[0])
    elif isinstance(result, list):
        steps = result
    else:
        steps = [str(result)]

    if isinstance(steps, str):
        steps = [steps]

    return steps

def execute_step(step: str, context: list) -> str:
    context_str = ""
    if context:
        context_str = "Previous steps results:\n"
        for i, c in enumerate(context):
            context_str += f"  Step {i+1}: {c}\n"

    messages = [
        {
            "role": "system",
            "content": EXECUTOR_PROMPT.format(context_str=context_str),
        },
        {"role": "user", "content": step},
    ]

    max_iterations = 5
    for i in range(max_iterations):
        response = openai.chat.completions.create(
            model = "gpt-4o",
            messages = messages,
            tools = tools,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content

        messages.append(msg)
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            print(f"    [工具调用] {func_name}({func_args})")

            result = available_tools[func_name](**func_args)
            print(f"    [工具结果] {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })

    return "Failed to complete step within max iterations."

def replan(user_input: str, original_steps: list, completed: list, remaining: list) -> list:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": REPLANNER_PROMPT.format(
                    tool_descriptions=get_tool_descriptions()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original task: {user_input}\n\n"
                    f"Completed steps:\n{json.dumps(completed, ensure_ascii=False, indent=2)}\n\n"
                    f"Remaining steps:\n{json.dumps(remaining, ensure_ascii=False, indent=2)}\n\n"
                    f"Should we continue with the remaining plan or adjust it?"
                ),
            },
        ],
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)

    if result.get("action") == "replan":
        return result.get("steps", remaining)
    return remaining

def run():
    print("Plan-and-Execute Agent（输入 q 退出）")
    print("=" * 50)

    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == "q":
            print("再见！")
            break
        
        print("\n 正在规划...")
        steps = plan(user_input)
        print(f"计划（共 {len(steps)} 步）:")
        for i, step in enumerate(steps):
            print(f"  {i+1}. {step}")

        # Step 2: Executor 逐步执行，每步后 replan
        context = []
        remaining = steps[:]
        step_num = 0

        while remaining:
            step = remaining.pop(0)
            step_num += 1
            print(f"\n▶ 执行第 {step_num} 步: {step}")
            result = execute_step(step, context)
            context.append(f"{step} → {result}")
            print(f"  结果: {result}")

            # Replan: 检查剩余计划是否需要调整
            if remaining:
                print("\n  🔄 检查是否需要调整计划...")
                new_remaining = replan(user_input, steps, context, remaining)
                if new_remaining != remaining:
                    print("  📋 计划已调整:")
                    for i, s in enumerate(new_remaining):
                        print(f"    {step_num + i + 1}. {s}")
                    remaining = new_remaining
                else:
                    print("  ✅ 计划不变，继续执行")

        # Step 3: 最终整合
        print("\n📝 整合最终回答...")
        final = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": SYNTHESIZER_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"Original request: {user_input}\n\n"
                        f"Completed steps:\n" + "\n".join(context)
                    ),
                },
            ],
        )
        print(f"\n最终回答: {final.choices[0].message.content}")

if __name__ == "__main__":
    run()