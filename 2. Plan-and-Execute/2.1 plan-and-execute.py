import openai
from dotenv import load_dotenv
import os
import json

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def weather_forecast(city: str) -> str:
    return f"sunny, 22°C"

def web_search(query: str) -> str:
    return f"[Result 1, Result 2, Result 3]"

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

availiable_tools = {
    "weather_forecast": weather_forecast,
    "web_search": web_search,
}

def plan(user_input: str) -> list:
    response = openai.chat.completions.create(
        model = "gpt-4o",
        messages = [
            {
                "role": "system",
                "content": """You are a planner. Given a user request, break it down into a list of steps.
Each step should be a clear, actionable task.
Return ONLY a JSON array of strings, no other text.

Example:
User: "Compare the weather in Tokyo and Helsinki"
["Search for current weather in Tokyo", "Search for current weather in Helsinki", "Compare the two results and determine which city is colder"]
"""
            },
            {"role": "user", "content": user_input}
        ],
        response_format = {"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)

    if isinstance(result, list):
        return result
    elif "steps" in result:
        return result["steps"]
    else:
        return list(result.values())[0]