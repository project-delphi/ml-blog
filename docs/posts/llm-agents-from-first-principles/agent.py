"""A complete LLM agent in one file, using nothing but the standard library.

The point of this module is to make the boundary visible. Exactly one function
here is stochastic -- `GroqBackend.create`, which posts to an inference API and
gets back a sample. Everything else is ordinary deterministic Python: string
assembly, a regex, a dict lookup, a `while` loop with two counters.

Run it directly to execute both worked examples:

    GROQ_API_KEY=... python agent.py
"""

from __future__ import annotations

import ast
import json
import operator
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

# --------------------------------------------------------------------------
# The stochastic part: one HTTP call to a hosted model.
# --------------------------------------------------------------------------

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


class Backend(Protocol):
    """Anything that maps a message list to a string of model output."""

    def create(
        self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.0
    ) -> str: ...


@dataclass
class GroqBackend:
    """OpenAI-compatible chat completions against Groq's free tier.

    The API key is read from the environment, never passed in and never
    written down. This is the only object in the module that touches the
    network, and the only source of randomness in the whole agent.
    """

    model: str = DEFAULT_MODEL
    calls: int = 0  # how many times the policy was sampled

    def create(
        self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.0
    ) -> str:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com/keys and export it before running."
            )
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        ).encode()
        request = urllib.request.Request(
            GROQ_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                # Without this the edge rejects the request with a bare 403
                # (Cloudflare 1010): the stdlib's default `Python-urllib/3.x`
                # agent string is on a blocklist. `requests` gets away with it
                # only because it sends a UA the blocklist doesn't cover.
                "User-Agent": "llm-agents-from-first-principles/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:  # surface the API's own message
            raise RuntimeError(f"Groq {exc.code}: {exc.read().decode()[:400]}") from exc
        self.calls += 1
        return body["choices"][0]["message"]["content"].strip()


# --------------------------------------------------------------------------
# Tools: ordinary Python functions, plus a description the model can read.
# --------------------------------------------------------------------------

# Arithmetic only. `eval` would also run `__import__("os").system(...)`, so the
# expression is parsed to an AST and every node is checked against a whitelist.
_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


# A syntax whitelist stops arbitrary code but not arbitrary *cost*: `9**9**9`
# is four legal nodes that hang the process for hours building an integer with
# billions of digits. Bound the exponent too.
_MAX_EXPONENT = 256


def _eval_node(node: ast.AST) -> float:
    """Evaluate a whitelisted arithmetic AST node."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"exponent {right} exceeds the limit of {_MAX_EXPONENT}")
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


def calculator(expr: str) -> str:
    """Evaluate an arithmetic expression exactly, without `eval`."""
    # `^` means xor in Python but exponentiation in maths notation, and models
    # write the maths one. Translate it rather than failing on it.
    cleaned = expr.replace("^", "**")
    value = _eval_node(ast.parse(cleaned, mode="eval"))
    # `/` always yields a float, so an exact result reads as "2341.0" and the
    # model then feeds that string to a tool wanting an integer. Narrow it.
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{expr} = {value}"


def is_prime(n: int) -> str:
    """Trial-divide to decide primality."""
    n = int(n)
    if n < 2:
        return f"{n} is not prime"
    factor = next((d for d in range(2, int(n**0.5) + 1) if n % d == 0), None)
    if factor is None:
        return f"{n} is prime"
    return f"{n} is not prime (divisible by {factor})"


@dataclass
class Tool:
    """A callable the model may invoke, plus the text describing it."""

    name: str
    description: str
    args: dict[str, str]  # arg name -> type, purely for the prompt
    fn: Callable[..., str]

    def spec(self) -> str:
        signature = ", ".join(f"{k}: {v}" for k, v in self.args.items())
        return f"  {self.name}({signature}) -- {self.description}"


@dataclass
class ToolRegistry:
    """Holds the tools and runs them. Never raises into the agent loop.

    Every failure mode -- unknown tool name, wrong argument names, an
    exception inside the tool itself -- comes back as an observation string.
    A crashed tool is information the policy can act on, not a bug in the
    scaffold, so the loop keeps running and the model gets told what broke.
    """

    tools: dict[str, Tool] = field(default_factory=dict)

    def add(self, tool: Tool) -> "ToolRegistry":
        self.tools[tool.name] = tool
        return self

    def specs(self) -> str:
        if not self.tools:
            return "  (none)"
        return "\n".join(t.spec() for t in self.tools.values())

    def execute(self, name: str, args: dict) -> str:
        tool = self.tools.get(name)
        if tool is None:
            known = ", ".join(self.tools) or "none"
            return f"ERROR: no tool named {name!r}. Available tools: {known}."
        try:
            return str(tool.fn(**args))
        except TypeError as exc:
            expected = ", ".join(tool.args) or "no arguments"
            return f"ERROR: bad arguments for {name} (expects {expected}): {exc}"
        except Exception as exc:
            return f"ERROR: {name} raised {type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# State: what the model conditions on, and what only the scaffold knows.
# --------------------------------------------------------------------------


@dataclass
class Transcript:
    """Dialogue state -- the exact message list handed to the model."""

    messages: list[dict] = field(default_factory=list)

    def user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def observation(self, content: str) -> None:
        # Tool results re-enter as user turns: the chat schema has no separate
        # "environment" role, so the observation is framed as one.
        self.messages.append({"role": "user", "content": f"OBSERVATION: {content}"})

    def as_messages(self) -> list[dict]:
        return list(self.messages)


@dataclass
class AgentState:
    """Scaffold state. The model never sees this except as a budget line."""

    turn: int = 0
    max_turns: int = 6
    tool_calls: int = 0
    max_tool_calls: int = 4
    memory: dict[str, Any] = field(default_factory=dict)

    def budget_line(self) -> str:
        left = self.max_turns - self.turn
        return f"You have {left} turn(s) left. Finish before they run out."


# --------------------------------------------------------------------------
# Parsing: turning a string of model output into a typed action.
# --------------------------------------------------------------------------


@dataclass
class ToolCall:
    name: str
    args: dict
    thought: str = ""


@dataclass
class Final:
    answer: str
    thought: str = ""


@dataclass
class ParseError:
    reason: str


Action = ToolCall | Final | ParseError

_THOUGHT_RE = re.compile(r"^THOUGHT:\s*(.*)$", re.M)
_ACTION_RE = re.compile(r"^ACTION:\s*(\S+)\s*$", re.M)
_INPUT_RE = re.compile(r"^ACTION_INPUT:\s*(\{.*?\})\s*$", re.M | re.S)
_FINAL_RE = re.compile(r"^FINAL:\s*(.*)", re.M | re.S)

# Observed behaviour, not paranoia: told to emit "exactly one of two forms",
# the model frequently emits both, filling the unused ACTION slot with a null
# word instead of dropping the line. Read those as "no action".
_NULL_ACTIONS = {"none", "null", "n/a", "na", "-", "nothing"}


def parse(text: str) -> Action:
    """Extract a tool call or a final answer from raw model output.

    This is where the contract with the model is actually enforced. The model
    was *asked* for a format; nothing guarantees it complied. A violation is
    an expected event, so it returns a ParseError value rather than raising --
    the loop feeds it back and lets the policy correct itself.
    """
    thought_match = _THOUGHT_RE.search(text)
    thought = thought_match.group(1).strip() if thought_match else ""

    action_match = _ACTION_RE.search(text)
    if action_match and action_match.group(1).strip().lower() in _NULL_ACTIONS:
        action_match = None

    if action_match:
        input_match = _INPUT_RE.search(text)
        if not input_match:
            return ParseError(
                "ACTION was given without a valid ACTION_INPUT line. "
                "ACTION_INPUT must be a single-line JSON object."
            )
        try:
            args = json.loads(input_match.group(1))
        except json.JSONDecodeError as exc:
            return ParseError(f"ACTION_INPUT was not valid JSON: {exc}")
        if not isinstance(args, dict):
            return ParseError("ACTION_INPUT must be a JSON object, not a scalar.")
        return ToolCall(name=action_match.group(1), args=args, thought=thought)

    final_match = _FINAL_RE.search(text)
    if final_match:
        return Final(answer=final_match.group(1).strip(), thought=thought)

    return ParseError(
        "Output matched neither form. Emit either ACTION plus ACTION_INPUT, "
        "or FINAL followed by your answer."
    )


# --------------------------------------------------------------------------
# Prompt assembly: a pure function of state. No I/O, no randomness.
# --------------------------------------------------------------------------

PROTOCOL = """You are an agent that acts one step at a time.

Available tools:
{tools}

Reply in EXACTLY one of these two forms, and nothing else. No markdown, no
code fences, no commentary outside the fields.

To use a tool:
THOUGHT: <one line of reasoning>
ACTION: <tool name>
ACTION_INPUT: <a single-line JSON object of arguments>

To give your final answer:
THOUGHT: <one line of reasoning>
FINAL: <your answer to the user>

When you are finishing, omit the ACTION and ACTION_INPUT lines entirely --
do not write "ACTION: None". Emit each field at most once.

Call a tool only when you need it. {budget}"""


def render(
    transcript: Transcript, state: AgentState, registry: ToolRegistry
) -> list[dict]:
    """Assemble the full message list from transcript + state + tool specs."""
    system = PROTOCOL.format(tools=registry.specs(), budget=state.budget_line())
    return [{"role": "system", "content": system}, *transcript.as_messages()]


# --------------------------------------------------------------------------
# The loop.
# --------------------------------------------------------------------------


@dataclass
class Step:
    """One pass through the loop, recorded so a trajectory can be printed."""

    turn: int
    raw: str
    action: Action
    observation: str | None = None


@dataclass
class Trace:
    question: str
    steps: list[Step] = field(default_factory=list)
    answer: str | None = None
    stop_reason: str = "answered"

    def show(self) -> None:
        """Print the trajectory: every model output and every observation."""
        print(f"QUESTION: {self.question}\n")
        for step in self.steps:
            print(f"--- turn {step.turn} | model output ---")
            print(step.raw)
            if step.observation is not None:
                print(f"--- turn {step.turn} | observation ---")
                print(step.observation)
            print()
        print(f"ANSWER ({self.stop_reason}): {self.answer}")


@dataclass
class Agent:
    """Deterministic control flow around one stochastic call."""

    backend: Backend
    registry: ToolRegistry = field(default_factory=ToolRegistry)
    max_turns: int = 6
    max_tool_calls: int = 4
    temperature: float = 0.0

    def run(self, question: str) -> Trace:
        state = AgentState(max_turns=self.max_turns, max_tool_calls=self.max_tool_calls)
        transcript = Transcript()
        transcript.user(question)
        trace = Trace(question=question)

        while state.turn < state.max_turns:
            state.turn += 1

            # The one stochastic line in this function.
            raw = self.backend.create(
                render(transcript, state, self.registry),
                temperature=self.temperature,
            )
            transcript.assistant(raw)
            action = parse(raw)
            step = Step(turn=state.turn, raw=raw, action=action)
            trace.steps.append(step)

            if isinstance(action, Final):
                trace.answer = action.answer
                trace.stop_reason = "answered"
                return trace

            if isinstance(action, ParseError):
                step.observation = f"ERROR: {action.reason}"
                transcript.observation(step.observation)
                continue

            if state.tool_calls >= state.max_tool_calls:
                step.observation = (
                    "ERROR: tool call budget exhausted. Answer with FINAL now."
                )
                transcript.observation(step.observation)
                continue

            state.tool_calls += 1
            step.observation = self.registry.execute(action.name, action.args)
            state.memory[f"turn{state.turn}:{action.name}"] = step.observation
            transcript.observation(step.observation)

        trace.stop_reason = "turn budget exhausted"
        return trace


def default_registry() -> ToolRegistry:
    """The two tools the blog post's examples use."""
    return (
        ToolRegistry()
        .add(
            Tool(
                name="calculator",
                description="evaluate an arithmetic expression exactly",
                args={"expr": "string"},
                fn=calculator,
            )
        )
        .add(
            Tool(
                name="is_prime",
                description="test whether a whole number is prime",
                args={"n": "integer"},
                fn=is_prime,
            )
        )
    )


if __name__ == "__main__":
    agent = Agent(backend=GroqBackend(), registry=default_registry())

    print("=" * 70, "\nEXAMPLE 1 -- no tool needed\n", "=" * 70, sep="")
    Agent(backend=GroqBackend()).run("What is (37 * 4 - 9) / 5?").show()

    print("\n" + "=" * 70, "\nEXAMPLE 2 -- tool round trip\n", "=" * 70, sep="")
    agent.run(
        "Use the calculator tool to compute (128^2 + 3) / 7, "
        "then tell me whether the result is prime."
    ).show()
