import json
import time
import os
from typing import Any
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class StepStatus(Enum):
    SUCCESS = "success"
    TOOL_ERROR = "tool_error"
    PARSE_ERROR = "parse_error"
    MAX_RETRIES = "max_retries"
    LOOP_DETECTED = "loop_detected"

@dataclass
class AgentStep:
    thought: str
    action: str | None
    action_input: dict | None
    observation: str | None
    status: StepStatus

@dataclass
class AgentState:
    query: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str | None = None
    total_tokens: int = 0

# A tool that occasionally fails to simulate real-world behavior
def web_search(query: str) -> str:
    import random
    if random.random() < 0.3:  # 30% chance to fail
        raise ConnectionError(f"Search API timeout for query: {query}")
    return f"Search results for '{query}': [mock result 1, mock result 2]"

def execute_tool_with_retry(
    tool_name: str, # The name of the tool to execute
    tool_input: dict, # The input to the tool
    tool_registry: dict, # The registry of tools
    max_retries: int = 3, # The maximum number of retries
    backoff_base: float = 1.5 # The base for exponential backoff
) -> tuple[str, StepStatus]:
    """
    Tool executor with exponential backoff.
    Returns (result_str, status).
    """
    if tool_name not in tool_registry:
        # Tool does not exist: hallucinated function call
        return (
            f"Error: tool '{tool_name}' does not exist. "
            f"Available tools: {list(tool_registry.keys())}",
            StepStatus.TOOL_ERROR
        )

    last_error = None
    for attempt in range(max_retries):
        try:
            result = tool_registry[tool_name](**tool_input)
            return str(result), StepStatus.SUCCESS

        except TypeError as e:
            # Wrong arguments -- not worth retrying
            return f"Error: wrong arguments for '{tool_name}': {e}", StepStatus.TOOL_ERROR

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = backoff_base ** attempt
                print(f"  [retry] attempt {attempt + 1} failed: {e}. waiting {wait:.1f}s")
                time.sleep(wait)

    return f"Tool '{tool_name}' failed after {max_retries} retries: {last_error}", StepStatus.MAX_RETRIES


@dataclass
class ParsedAction:
    thought: str
    action: str | None      # None means LLM gave a final answer directly
    action_input: dict | None
    final_answer: str | None

def parse_llm_response(response_text: str) -> ParsedAction:
    """
    Parse structured actions from LLM text output.
    Uses a simplified JSON block format.
    """
    import re

    # Try to find a JSON block
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)

    if not json_match:
        # No JSON: could be a direct answer or a format error
        if "Final Answer:" in response_text:
            answer = response_text.split("Final Answer:")[-1].strip()
            return ParsedAction(
                thought=response_text,
                action=None,
                action_input=None,
                final_answer=answer
            )
        raise ValueError(f"Cannot parse LLM output: no JSON block or final answer found")

    try:
        data = json.loads(json_match.group(1))
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in LLM output: {e}")

    # Validate required fields
    if "action" not in data:
        raise ValueError("Parsed JSON missing 'action' field")

    return ParsedAction(
        thought=data.get("thought", ""),
        action=data.get("action"),
        action_input=data.get("action_input", {}),
        final_answer=data.get("final_answer")
    )

class CircuitBreaker:
    """
    Prevents the agent from entering infinite loops or exceeding resource limits.
    """
    def __init__(
        self,
        max_steps: int = 10,
        max_consecutive_errors: int = 3,
        max_tokens: int = 50_000
    ):
        self.max_steps = max_steps
        self.max_consecutive_errors = max_consecutive_errors
        self.max_tokens = max_tokens

        self._steps = 0
        self._consecutive_errors = 0
        self._total_tokens = 0

    def record_step(self, status: StepStatus, tokens_used: int = 0) -> None:
        self._steps += 1
        self._total_tokens += tokens_used

        if status == StepStatus.SUCCESS:
            self._consecutive_errors = 0
        else:
            self._consecutive_errors += 1

    def should_stop(self) -> tuple[bool, str]:
        if self._steps >= self.max_steps:
            return True, f"Max steps reached ({self.max_steps})"
        if self._consecutive_errors >= self.max_consecutive_errors:
            return True, f"Too many consecutive errors ({self._consecutive_errors})"
        if self._total_tokens >= self.max_tokens:
            return True, f"Token budget exceeded ({self._total_tokens})"
        return False, ""

SYSTEM_PROMPT = """You are a helpful agent. At each step, respond with a JSON block:
```json
{
  "thought": "your reasoning",
  "action": "tool_name",
  "action_input": {"param": "value"}
}
```

When you have enough information, respond with:
```json
{
  "thought": "I have the answer",
  "action": "finish",
  "action_input": {},
  "final_answer": "your final answer here"
}
```
"""

def run_agent(query: str, tool_registry: dict) -> AgentState:
    state = AgentState(query=query)
    circuit_breaker = CircuitBreaker(max_steps=8, max_consecutive_errors=3)
    messages = [{"role": "user", "content": query}]

    print(f"\n[Agent] Query: {query}\n{'='*50}")

    while True:
        # 1. Circuit breaker check
        should_stop, reason = circuit_breaker.should_stop()
        if should_stop:
            print(f"[CircuitBreaker] Stopping: {reason}")
            state.final_answer = f"Agent stopped: {reason}"
            break

        # 2. Call LLM
        response = openai.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *messages,
            ],
        )
        response_text = response.choices[0].message.content
        tokens_used = response.usage.prompt_tokens + response.usage.completion_tokens

        print(f"\n[LLM output]\n{response_text}")

        # 3. Parse LLM output (with error handling)
        try:
            parsed = parse_llm_response(response_text)
        except ValueError as e:
            print(f"[ParseError] {e}")
            step = AgentStep(
                thought=response_text,
                action=None,
                action_input=None,
                observation=f"Parse error: {e}. Please respond with valid JSON.",
                status=StepStatus.PARSE_ERROR
            )
            state.steps.append(step)
            circuit_breaker.record_step(StepStatus.PARSE_ERROR, tokens_used)

            # Feed error back to LLM so it can correct itself
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": step.observation})
            continue

        # 4. Finish condition
        if parsed.action == "finish" or parsed.final_answer:
            state.final_answer = parsed.final_answer
            print(f"\n[Final Answer] {state.final_answer}")
            break

        # 5. Execute tool (with retry)
        observation, status = execute_tool_with_retry(
            tool_name=parsed.action,
            tool_input=parsed.action_input,
            tool_registry=tool_registry
        )

        step = AgentStep(
            thought=parsed.thought,
            action=parsed.action,
            action_input=parsed.action_input,
            observation=observation,
            status=status
        )
        state.steps.append(step)
        circuit_breaker.record_step(status, tokens_used)

        print(f"[Tool: {parsed.action}] -> {observation[:100]}")

        # 6. Update conversation history
        messages.append({"role": "assistant", "content": response_text})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return state


# Example usage
if __name__ == "__main__":
    tools = {
        "web_search": web_search,
        "calculator": lambda expression: str(eval(expression))  # demo only, do not use in production
    }

    result = run_agent(
        query="Search for the latest F1 2026 season regulations and summarise the key changes.",
        tool_registry=tools
    )

    print(f"\n{'='*50}")
    print(f"Steps taken: {len(result.steps)}")
    for i, step in enumerate(result.steps):
        print(f"  Step {i+1}: {step.action} -> {step.status.value}")