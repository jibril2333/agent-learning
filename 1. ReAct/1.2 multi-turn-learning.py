import openai
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def chat():
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

        response = openai.chat.completions.create(
            model = "gpt-5",
            messages = messages,
        )

        assistant_content = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_content})

        print(f"\nassistant: {assistant_content}")

if __name__ == "__main__":
    chat()