import openai
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def weather_forecast(city: str) -> str:
    if city == "Tokyo":
        return "sunny, 22C"
    elif city == "Osaka":
        return "cloudy, -5C"
    elif city == "Kyoto":
        return "rainy, 20C"
    elif city == "Fukuoka":
        return "sunny, 22C"
    else:
        return "Error: city not found"

def web_search(query: str) -> str:
    return "Precipitation: 10mm, Wind: 10km/h"

available_tools = {
    "weather_forecast": weather_forecast,
    "web_search": web_search,
}

# ============================================================
# Prompts
# ============================================================

PLANNER_PROMPT = """\
You are a planner that creates a step-by-step plan using available tools.

For each step, output a line of thought (Plan:) followed by a tool call
or direct reasoning step. Use #E1, #E2, ... as variable placeholders to
reference results from previous steps.

Available tools:
- weather_forecast(city): Get the weather forecast for a city
- web_search(query): Search the web for information

Output format (strict, one plan-evidence pair per step):
Plan: <reasoning about what to do>
#E1 = tool_name(arg or #Ex reference)
Plan: <next reasoning>
#E2 = tool_name(arg or #Ex reference)
...

Rules:
- Each #E variable must be assigned exactly once.
- Arguments can reference previous #E variables (e.g. #E1).
- Keep the plan concise (1-5 steps).
- Do NOT output anything else besides Plan/Evidence lines."""

SOLVER_PROMPT = """\
You are a solver. Given the original question and a set of evidence
from executed plan steps, provide a final comprehensive answer.

Respond directly to the user's question using the evidence provided."""

# ============================================================
# Functions
# ============================================================

PLAN_PATTERN = re.compile(
    r"Plan:\s*(.+?)\s*\n\s*#E(\d+)\s*=\s*(\w+)\s*\(([^)]*)\)",
    re.DOTALL,
)


def plan(user_input: str) -> list:
    """Generate all steps at once with #E variable placeholders."""
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": user_input},
        ],
    )

    raw_plan = response.choices[0].message.content
    print(f"\n  Raw plan:\n{raw_plan}")

    steps = []
    for match in PLAN_PATTERN.finditer(raw_plan):
        thought, eid, tool_name, raw_arg = match.groups()
        steps.append({
            "id": f"#E{eid}",
            "thought": thought.strip(),
            "tool": tool_name.strip(),
            "arg": raw_arg.strip().strip('"').strip("'"),
        })

    return steps


def worker(steps: list) -> dict:
    """Execute each step, substituting #E references with real results."""
    evidence = {}

    for step in steps:
        arg = step["arg"]
        for var, val in evidence.items():
            arg = arg.replace(var, val)

        tool_name = step["tool"]
        if tool_name in available_tools:
            result = available_tools[tool_name](arg)
            print(f"    [Tool Call] {tool_name}({arg})")
            print(f"    [Tool Result] {result}")
        else:
            result = f"Unknown tool: {tool_name}"
            print(f"    [Error] {result}")

        evidence[step["id"]] = result

    return evidence


def solver(user_input: str, steps: list, evidence: dict) -> str:
    """Combine all evidence to produce the final answer."""
    evidence_str = ""
    for step in steps:
        eid = step["id"]
        evidence_str += f"{eid} (Plan: {step['thought']})\n"
        evidence_str += f"  Result: {evidence.get(eid, 'N/A')}\n\n"

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SOLVER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {user_input}\n\n"
                    f"Evidence:\n{evidence_str}"
                ),
            },
        ],
    )

    return response.choices[0].message.content


def run():
    print("ReWOO Agent (press q to quit)")
    print("=" * 50)

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == "q":
            print("Goodbye!")
            break

        # Step 1: Planner - generate full plan at once
        print("\n Planning (all steps at once)...")
        steps = plan(user_input)

        if not steps:
            print("  Failed to parse plan. Please try again.")
            continue

        print(f"\n  Parsed plan ({len(steps)} steps):")
        for s in steps:
            print(f"    {s['id']}: {s['tool']}({s['arg']})  -- {s['thought']}")

        # Step 2: Worker - execute all steps sequentially
        print("\n Executing all steps...")
        evidence = worker(steps)

        # Step 3: Solver - synthesize final answer
        print("\n Solving...")
        answer = solver(user_input, steps, evidence)
        print(f"\nFinal Answer: {answer}")


if __name__ == "__main__":
    run()
