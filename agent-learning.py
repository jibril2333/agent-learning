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


    print("多轮对话测试（输入 q 退出）")
    print("=" * 40)
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == "q":
            print("再见！")
            break
        
        messages.append({"role": "user", "content": user_input})
        # 第一次请求：让 LLM 决定是否调用工具
        response = openai.chat.completions.create(
            model="gpt-5",
            messages=messages,
            tools=tools,
        )

        # 获取助手的消息
        assistant_message = response.choices[0].message

        # 如果 LLM 决定调用工具
        if assistant_message.tool_calls:
            if assistant_message.content:
                print(f"助手（思考中）: {assistant_message.content}")

            # 把助手的消息（包含 tool_calls）加入对话历史
            messages.append(assistant_message)

            # 逐个执行工具调用
            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                print(f"[调用工具] {func_name}({func_args})")

                # 执行函数
                result = available_tools[func_name](**func_args)

                print(f"[工具结果] {result}\n")

                # 把工具结果加入对话历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })

            # 第二次请求：让 LLM 根据工具结果生成最终回复
            final_response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
            )
            final_answer = final_response.choices[0].message.content
        else:
            # LLM 不需要调用工具，直接回复
            final_answer = assistant_message.content

        messages.append({"role": "assistant", "content": final_answer})
        print(f"助手: {final_answer}")


if __name__ == "__main__":
    run_conversation()
    print("=" * 50)