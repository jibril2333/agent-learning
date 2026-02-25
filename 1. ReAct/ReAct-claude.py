import openai
from dotenv import load_dotenv
import os
import json

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# ============================================================
# Tools
# ============================================================

def weather_forecast(city: str) -> str:
    return f"The weather in {city} is sunny, 22°C."


def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def search(query: str) -> str:
    # 模拟搜索结果，实际可接入搜索 API
    return f"Search results for '{query}': [模拟结果] ..."


tools = [
    {
        "type": "function",
        "function": {
            "name": "weather_forecast",
            "description": "Get the weather forecast for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to get the weather forecast for",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '2 + 3 * 4'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

available_tools = {
    "weather_forecast": weather_forecast,
    "calculator": calculator,
    "search": search,
}

# ============================================================
# ReAct Agent
# ============================================================

MAX_ITERATIONS = 10  # 防止无限循环


def run_conversation():
    system_prompt = (
        "You are a helpful assistant. "
        "Think step by step. "
        "Use tools when needed to gather information before answering. "
        "You can call multiple tools in sequence to solve complex problems."
    )

    messages = [{"role": "system", "content": system_prompt}]

    print("ReAct Agent（输入 q 退出）")
    print("=" * 50)

    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == "q":
            print("再见！")
            break

        messages.append({"role": "user", "content": user_input})

        # --------------------------------------------------
        # ReAct 循环: Thought → Action → Observation → ...
        # LLM 自行决定何时停止调用工具
        # --------------------------------------------------
        iteration = 0

        while iteration < MAX_ITERATIONS:
            iteration += 1

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
            )

            assistant_message = response.choices[0].message

            # 如果 LLM 没有调用工具 → 任务完成，跳出循环
            if not assistant_message.tool_calls:
                final_answer = assistant_message.content
                break

            # LLM 发出了 tool_calls → 执行工具，继续循环
            if assistant_message.content:
                print(f"  💭 思考: {assistant_message.content}")

            messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                print(f"  🔧 调用工具: {func_name}({func_args})")

                if func_name in available_tools:
                    result = available_tools[func_name](**func_args)
                else:
                    result = f"Error: unknown tool '{func_name}'"

                print(f"  📋 工具结果: {result}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )
        else:
            # 达到最大迭代次数仍未结束
            final_answer = "抱歉，我尝试了多次但未能完成任务。"

        messages.append({"role": "assistant", "content": final_answer})
        print(f"\n助手: {final_answer}")


if __name__ == "__main__":
    run_conversation()