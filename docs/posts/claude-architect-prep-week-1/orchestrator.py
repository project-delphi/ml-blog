"""An orchestrator-worker agent loop against the Claude Messages API.

The loop is deliberately explicit about the two things that decide whether an
agent survives production:

1. **Every `stop_reason` is handled by name.** A loop written as
   `while response.stop_reason == "tool_use"` treats `max_tokens`, `refusal`
   and `pause_turn` as "finished" and returns a truncated answer with no error.
   Here each terminal reason maps to a distinct `Outcome`, and an unrecognised
   reason stops the loop rather than falling through.

2. **A failing tool is a message, not an exception.** Anything a handler raises
   is converted into a `tool_result` with `is_error: True` so the model can
   read the failure and change approach. Dropping the block instead leaves an
   unanswered `tool_use` id and the next request is rejected outright.

Two structures sit on top of that: a `ToolRegistry` that validates input before
a handler ever runs, and `orchestrate()`, which exposes worker loops to a
planner model as a single `delegate` tool.

Requires `anthropic>=1.0` and an `ANTHROPIC_API_KEY` (or an `ant auth login`
profile, which the zero-argument client picks up on its own).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import anthropic
from anthropic.types import Message, MessageParam, ToolParam

log = logging.getLogger(__name__)

MODEL = "claude-opus-5"
WORKER_MAX_TOKENS = 8_000
ORCHESTRATOR_MAX_TOKENS = 16_000


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


def _call_with_deadline(fn: Callable[[], Any], timeout_s: float) -> Any:
    """Run `fn`, giving up on it after `timeout_s` seconds.

    A Python thread cannot be killed, so an abandoned handler runs to
    completion in the background. Two details keep that from mattering:

    * the thread is a **daemon**, so a handler blocked forever on a socket
      cannot hold the interpreter open at exit;
    * every call gets its **own** thread rather than a slot in a shared pool.
      A bounded pool would lose a worker to each abandoned handler until it
      had none left, and then time out every later tool without running it --
      which reads as "every tool is broken".
    """
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise TimeoutError
    if "error" in box:
        raise box["error"]
    return box["value"]


class ToolFailure(Exception):
    """A tool failed in a way the model can reasonably recover from.

    The message is handed back verbatim inside an `is_error` tool_result, so
    write it for the model: say what went wrong and what a valid call needs.
    """


class ToolAbort(Exception):
    """A tool failed in a way no retry can fix: revoked credentials, a missing
    binary, a tripped circuit breaker. Ends the run instead of inviting the
    model to try again against a broken dependency.
    """


@dataclass(frozen=True)
class Tool:
    """A callable paired with the JSON Schema the model must satisfy."""

    name: str
    description: str
    input_schema: Mapping[str, Any]
    handler: Callable[..., str]
    #: Wall-clock ceiling. The handler is abandoned once it passes this.
    timeout_s: float = 30.0
    #: `strict` makes the API validate input against the schema before the
    #: handler sees it, but it requires a closed schema. Set False for a tool
    #: that genuinely takes open-ended input.
    strict: bool = True

    def to_param(self) -> ToolParam:
        # `strict` makes the API reject malformed input server-side, before a
        # token of it reaches the handler. It requires `additionalProperties:
        # false` and a `required` list on the schema.
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "strict": self.strict,
        }


class ToolRegistry:
    """Dispatches `tool_use` blocks and guarantees a result block for each.

    The registry never raises on a handler failure. Every path -- unknown tool,
    schema violation, handler exception, overrun -- produces a `tool_result`
    carrying the matching `tool_use_id`, because the API rejects the next
    request if any id from the assistant turn goes unanswered.
    """

    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def dispatch(
        self,
        block: Any,
        granted: Collection[str] | None = None,
    ) -> dict[str, Any]:
        """Run one `tool_use` block and return its `tool_result`.

        `granted` is the set of names the caller actually offered the model.
        Narrowing the *definitions* sent in a request is not access control --
        a model can name a tool it was never shown, whether by hallucinating a
        name it saw in an earlier result or because injected text told it to.
        The registry is the only place that can refuse, so it does.
        """
        # `block.input` is already-parsed JSON. Never string-match the
        # serialised form: escaping of Unicode and forward slashes varies.
        visible = self._tools.keys() if granted is None else granted
        tool = self._tools.get(block.name)
        if tool is None or block.name not in visible:
            # Report only the names this caller was granted; listing the rest
            # hands the model a directory of tools it must not reach.
            return _error_result(
                block.id,
                f"No tool named {block.name!r}. Available: "
                f"{', '.join(sorted(visible))}.",
            )

        try:
            self._validate(tool, block.input)
        except ToolFailure as exc:
            # Schema failures are the single most common tool error in
            # production, and the most recoverable: the model usually fixes
            # the call on the next turn if told which field was wrong.
            log.warning("schema rejected call to %s: %s", tool.name, exc)
            return _error_result(block.id, f"Invalid input for {tool.name}: {exc}")

        started = time.monotonic()
        try:
            output = _call_with_deadline(
                lambda: tool.handler(**block.input),
                tool.timeout_s,
            )
        except TimeoutError:
            log.warning("%s exceeded %gs; abandoned", tool.name, tool.timeout_s)
            return _error_result(
                block.id,
                f"{tool.name} did not finish within {tool.timeout_s:g}s and was "
                "abandoned. Narrow the request and try again.",
            )
        except ToolFailure as exc:
            log.warning("%s failed: %s", tool.name, exc)
            return _error_result(block.id, str(exc))
        except ToolAbort:
            raise
        except Exception as exc:  # noqa: BLE001 - a handler bug must not kill the run
            log.exception("%s raised", tool.name)
            return _error_result(
                block.id,
                f"{tool.name} failed with {type(exc).__name__}: {exc}",
            )

        log.debug("%s ran in %.2fs", tool.name, time.monotonic() - started)

        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output if isinstance(output, str) else json.dumps(output),
        }

    @staticmethod
    def _validate(tool: Tool, supplied: Mapping[str, Any]) -> None:
        """Check required keys and reject unknown ones.

        `strict: True` already enforces this server-side. Repeating it here
        keeps the guarantee when the schema is relaxed for a tool that needs
        open-ended input, and keeps the handler's signature honest.
        """
        properties = tool.input_schema.get("properties", {})
        required = set(tool.input_schema.get("required", ()))
        missing = required - set(supplied)
        if missing:
            raise ToolFailure(
                f"missing required field(s): {', '.join(sorted(missing))}",
            )
        unknown = set(supplied) - set(properties)
        # Absent means "extras allowed" in JSON Schema. A `strict` tool has to
        # declare `additionalProperties: false` anyway, so honouring the real
        # default here is what lets an open schema stay open.
        if unknown and not tool.input_schema.get("additionalProperties", True):
            raise ToolFailure(f"unknown field(s): {', '.join(sorted(unknown))}")


def _error_result(tool_use_id: str, message: str) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": message,
        "is_error": True,
    }


# --------------------------------------------------------------------------- #
# Loop control
# --------------------------------------------------------------------------- #


class Outcome(str, Enum):
    """How a loop ended. Every value maps to one branch in `run_loop`."""

    COMPLETED = "completed"  # stop_reason end_turn / stop_sequence
    TRUNCATED = "truncated"  # stop_reason max_tokens, retry exhausted
    REFUSED = "refused"  # stop_reason refusal
    TURN_LIMIT = "turn_limit"  # our own ceiling, not the model's
    STALLED = "stalled"  # same tool call repeated, no progress
    ABORTED = "aborted"  # ToolAbort from a handler
    UNKNOWN_STOP = "unknown_stop"  # a stop_reason this code has never seen


@dataclass(frozen=True)
class Budget:
    """Ceilings the model cannot see and therefore cannot talk its way past."""

    max_turns: int = 12
    max_tool_calls: int = 40
    #: `pause_turn` means a server-side tool loop hit its own iteration cap.
    #: Resuming is correct; resuming without a bound is an infinite loop.
    max_pause_resumes: int = 5
    #: One retry at a larger `max_tokens` before declaring truncation.
    max_token_retries: int = 1
    #: Identical (name, input) calls tolerated before the run is called stalled.
    max_repeats: int = 2


@dataclass
class RunResult:
    outcome: Outcome
    text: str
    messages: list[MessageParam]
    turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.outcome is Outcome.COMPLETED


def run_loop(
    client: anthropic.Anthropic,
    *,
    system: str,
    messages: list[MessageParam],
    tools: Sequence[ToolParam],
    registry: ToolRegistry,
    budget: Budget = Budget(),
    model: str = MODEL,
    max_tokens: int = WORKER_MAX_TOKENS,
) -> RunResult:
    """Drive one agent loop to a terminal state.

    `messages` is mutated in place and returned on the result, so a caller can
    inspect, persist, or resume the exact history the model saw.
    """
    result = RunResult(outcome=Outcome.TURN_LIMIT, text="", messages=messages)
    seen_calls: dict[tuple[str, str], int] = {}
    pause_resumes = 0
    token_retries = 0
    turn_max_tokens = max_tokens

    while result.turns < budget.max_turns:
        result.turns += 1
        response = client.messages.create(
            model=model,
            max_tokens=turn_max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            thinking={"type": "adaptive"},
            tools=list(tools),
            messages=messages,
        )
        result.input_tokens += response.usage.input_tokens
        result.output_tokens += response.usage.output_tokens

        # Append the whole content list, never just the text. Dropping the
        # thinking and tool_use blocks breaks the next request.
        messages.append({"role": "assistant", "content": response.content})
        # Keep the most recent turn that actually said something. Assigning
        # unconditionally would wipe it: the terminal turn of a stalled or
        # turn-limited run is a tool_use turn, which usually has no text at all,
        # and those are precisely the outcomes this exists to preserve.
        previous_text = result.text
        if turn_text := _text_of(response):
            result.text = turn_text

        stop = response.stop_reason
        log.debug("turn %d stop_reason=%s", result.turns, stop)

        if stop in ("end_turn", "stop_sequence"):
            result.outcome = Outcome.COMPLETED
            return result

        if stop == "refusal":
            # `stop_details` is populated only for refusals; guard before use.
            details = response.stop_details
            result.outcome = Outcome.REFUSED
            result.detail = getattr(details, "explanation", None) or "no explanation"
            log.error(
                "refused (%s): %s",
                getattr(details, "category", None),
                result.detail,
            )
            return result

        if stop == "max_tokens":
            # Output was cut mid-sentence. Continuing as if it were complete is
            # how a half-written tool argument reaches a database.
            if token_retries < budget.max_token_retries:
                token_retries += 1
                turn_max_tokens *= 2
                messages.pop()  # discard the truncated turn before retrying
                result.text = previous_text  # ... and the text it contributed
                log.warning("truncated; retrying at max_tokens=%d", turn_max_tokens)
                continue
            result.outcome = Outcome.TRUNCATED
            result.detail = f"still truncated after {token_retries} retry(ies)"
            return result

        if stop == "pause_turn":
            # A server-side tool (web search, code execution) hit its own
            # iteration limit. Re-send the history unchanged -- the API sees the
            # trailing server_tool_use block and resumes; an added "continue"
            # message confuses it.
            pause_resumes += 1
            if pause_resumes > budget.max_pause_resumes:
                result.outcome = Outcome.TURN_LIMIT
                result.detail = f"paused {pause_resumes} times without finishing"
                return result
            continue

        if stop != "tool_use":
            # A reason added to the API after this code was written. Stopping
            # is the honest response; falling through would invent an answer.
            result.outcome = Outcome.UNKNOWN_STOP
            result.detail = f"unhandled stop_reason {stop!r}"
            log.error(result.detail)
            return result

        # --- stop_reason == "tool_use" -------------------------------------
        calls = [b for b in response.content if b.type == "tool_use"]
        granted = {t.get("name") for t in tools}
        tool_results: list[dict[str, Any]] = []
        terminal: tuple[Outcome, str] | None = None
        for block in calls:
            if result.tool_calls >= budget.max_tool_calls:
                terminal = (
                    Outcome.TURN_LIMIT,
                    f"exceeded {budget.max_tool_calls} tool calls",
                )
                break

            # An agent that repeats a call verbatim is not making progress; it
            # is usually re-reading a resource whose result it misread. Say so
            # in-band before the turn ceiling burns the whole budget.
            signature = (block.name, json.dumps(block.input, sort_keys=True))
            seen_calls[signature] = seen_calls.get(signature, 0) + 1
            if seen_calls[signature] > budget.max_repeats:
                # Answer the block anyway -- every tool_use id in the turn needs
                # a result or the history cannot be resent -- then stop.
                tool_results.append(
                    _error_result(
                        block.id,
                        f"{block.name} has already been called with these exact "
                        "arguments and returned the same result. Use a different "
                        "approach or state what is blocking you.",
                    ),
                )
                terminal = (
                    Outcome.STALLED,
                    f"{block.name} repeated {seen_calls[signature]} times",
                )
                break

            try:
                result.tool_calls += 1  # counted only when it actually runs
                tool_results.append(registry.dispatch(block, granted))
            except ToolAbort as exc:
                terminal = (Outcome.ABORTED, str(exc))
                log.error("aborted by %s: %s", block.name, exc)
                break

        # However the turn ended, every tool_use id in it needs a result or
        # the history cannot be resent -- so answer the blocks that never ran
        # before handing it back. This is why the loop breaks rather than
        # returning from inside the loop.
        answered = {r["tool_use_id"] for r in tool_results}
        tool_results.extend(
            _error_result(b.id, "Not run: the loop stopped part-way through this turn.")
            for b in calls
            if b.id not in answered
        )

        # All results go back in one user message. Splitting them across
        # several messages trains the model out of parallel tool calls.
        messages.append({"role": "user", "content": tool_results})

        if terminal is not None:
            result.outcome, result.detail = terminal
            return result

    result.detail = f"hit the {budget.max_turns}-turn ceiling"
    return result


def _text_of(response: Message) -> str:
    return "\n".join(b.text for b in response.content if b.type == "text").strip()


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

DELEGATE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "description": (
                "A self-contained instruction for the worker. It sees this "
                "string and nothing else from the conversation, so restate "
                "every fact it needs."
            ),
        },
        "tools": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Names of the tools this worker is allowed to call.",
        },
    },
    "required": ["task", "tools"],
    "additionalProperties": False,
}

ORCHESTRATOR_SYSTEM = """You plan and delegate; you do not call domain tools yourself.

Break the request into independent sub-tasks and issue one `delegate` call per
sub-task, in parallel where they do not depend on each other. Each worker starts
with no memory of this conversation, so restate the facts it needs. Grant each
worker only the tools its sub-task requires.

A worker result may report an error. Do not re-delegate an identical task after a
failure -- either change the task or report the blocker to the user."""


def orchestrate(
    client: anthropic.Anthropic,
    request: str,
    registry: ToolRegistry,
    *,
    worker_system: str = "You are a worker agent. Complete the task with the "
    "tools provided and report the result concisely.",
    orchestrator_budget: Budget = Budget(max_turns=8, max_tool_calls=12),
    worker_budget: Budget = Budget(),
    max_workers: int = 6,
    model: str = MODEL,
) -> RunResult:
    """Run a planner whose only tool spawns bounded worker loops.

    Delegation is a tool call like any other, so it inherits the whole error
    path above: a worker that stalls, truncates or is refused comes back as an
    `is_error` result the planner can read and route around, rather than as an
    exception that unwinds the run.
    """
    spend = {"workers": 0}

    def delegate(task: str, tools: list[str]) -> str:
        spend["workers"] += 1
        if spend["workers"] > max_workers:
            # Its own ceiling, deliberately below the planner's tool-call cap.
            # Sharing that cap made this unreachable: run_loop stops before it
            # dispatches the call that would exceed it.
            raise ToolAbort(f"worker budget exhausted after {max_workers} workers")

        unknown = [name for name in tools if name not in registry.names]
        if unknown:
            raise ToolFailure(f"no such tool(s): {', '.join(unknown)}")

        worker = run_loop(
            client,
            system=worker_system,
            messages=[{"role": "user", "content": task}],
            tools=registry.params(tools),  # run_loop enforces this grant
            registry=registry,
            budget=worker_budget,
            model=model,
        )
        if not worker.ok:
            # Surfaced to the planner as a tool failure, with the outcome name
            # so it can distinguish "ran out of turns" from "was refused".
            raise ToolFailure(
                f"worker ended as {worker.outcome.value} ({worker.detail}). "
                f"Partial output: {worker.text[:500] or '(none)'}",
            )
        return worker.text

    planner_tools = ToolRegistry(
        [
            Tool(
                name="delegate",
                description=(
                    "Hand one self-contained sub-task to a worker agent and "
                    "return its final report."
                ),
                input_schema=DELEGATE_SCHEMA,
                handler=delegate,
                timeout_s=300.0,
            ),
        ],
    )

    return run_loop(
        client,
        system=ORCHESTRATOR_SYSTEM,
        messages=[{"role": "user", "content": request}],
        tools=planner_tools.params(),
        registry=planner_tools,
        budget=orchestrator_budget,
        model=model,
        max_tokens=ORCHESTRATOR_MAX_TOKENS,
    )


# --------------------------------------------------------------------------- #
# Example wiring
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    PRICES = {"widget": 9.99, "sprocket": 24.50}

    def lookup_price(sku: str) -> str:
        if sku not in PRICES:
            raise ToolFailure(
                f"unknown sku {sku!r}; known: {', '.join(sorted(PRICES))}",
            )
        return f"{sku}: ${PRICES[sku]:.2f}"

    tools = ToolRegistry(
        [
            Tool(
                name="lookup_price",
                description="Return the catalogue price for one SKU.",
                input_schema={
                    "type": "object",
                    "properties": {"sku": {"type": "string"}},
                    "required": ["sku"],
                    "additionalProperties": False,
                },
                handler=lookup_price,
            ),
        ],
    )

    run = orchestrate(
        anthropic.Anthropic(),
        "Price a bundle of one widget and two sprockets, and flag anything unavailable.",
        tools,
    )
    print(f"[{run.outcome.value}] {run.detail or ''}")
    print(run.text)
    print(
        f"{run.turns} turns, {run.tool_calls} tool calls, "
        f"{run.input_tokens} in / {run.output_tokens} out",
    )
