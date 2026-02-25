import openai
from dotenv import load_dotenv
import os
import json

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def weather_forecast(city: str) -> str:
    return f"The weather in {city} is sunny."

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
                        "description": "The city to get the weather forecast for"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

available_tools = {
    "weather_forecast": weather_forecast
}

def run_conversation():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]

    print("ReAct 多轮对话测试（输入 q 退出）")
    print("=" * 40)
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == "q":
            print("再见！")
            break

        messages.append({"role": "user", "content": user_input})

        # ReAct 循环：推理 → 行动 → 观察 → 重复，直到 LLM 自己决定停止
        max_iterations = 10
        for i in range(max_iterations):
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
            )
            assistant_message = response.choices[0].message

            # LLM 没有调用工具 → 任务完成，跳出循环
            if not assistant_message.tool_calls:
                final_answer = assistant_message.content
                break

            # LLM 决定调用工具 → 执行并把结果喂回去
            if assistant_message.content:
                print(f"助手（思考中）: {assistant_message.content}")

            messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                print(f"  [调用工具] {func_name}({func_args})")

                result = available_tools[func_name](**func_args)
                print(f"  [工具结果] {result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })
        else:
            final_answer = "抱歉，达到最大迭代次数，无法完成任务。"

        messages.append({"role": "assistant", "content": final_answer})
        print(f"助手: {final_answer}")


if __name__ == "__main__":
    run_conversation()
    print("=" * 50)