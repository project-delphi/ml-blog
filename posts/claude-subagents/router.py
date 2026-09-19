"""Select one specialist, then call it using the official Anthropic SDK."""

import argparse
import json
import math
import sys
from pathlib import Path

import anthropic
import yaml

ROLES = {
    "python-scripter": "Python backend logic and general programming",
    "data-engineer": "SQL, ETL pipelines, and Pandas transformations",
    "front-end-specialist": "Accessible React components and Tailwind CSS",
}


def load_agents(path):
    """Validate the three specialist definitions before making API calls."""
    config = yaml.safe_load(Path(path).read_text())
    entries = config.get("agents") if isinstance(config, dict) else None
    if not isinstance(entries, list):
        raise ValueError("Configuration must contain an agents list")
    agents = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Every agent must be a mapping")
        for field in ("name", "model", "system_prompt"):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                raise ValueError(f"Every agent needs a nonempty {field}")
        name = entry["name"]
        if name in agents or name not in ROLES:
            raise ValueError(f"Duplicate or unknown agent: {name}")
        temperature = entry.get("temperature")
        if (
            type(temperature) not in (int, float)
            or not math.isfinite(temperature)
            or not 0 <= temperature <= 1
        ):
            raise ValueError(f"{name}: temperature must be a number from 0 to 1")
        if entry.get("tools"):
            raise ValueError("This text-only router does not implement tools")
        agents[name] = entry
    if set(agents) != set(ROLES):
        raise ValueError("Configuration must define all three specialist roles")
    return agents


def response_text(message):
    """Reject incomplete responses and extract text blocks."""
    if message.stop_reason != "end_turn":
        raise ValueError(f"API response did not finish: {message.stop_reason}")
    text = "\n".join(block.text for block in message.content if block.type == "text")
    if not text.strip():
        raise ValueError("API response contained no text")
    return text.strip()


def log_usage(label, message):
    """Log reported tokens to stderr, leaving stdout for generated content."""
    usage = message.usage
    inputs = usage.input_tokens
    outputs = usage.output_tokens
    print(
        f"{label}: input={inputs} output={outputs} total={inputs + outputs}",
        file=sys.stderr,
    )
    return inputs, outputs


def orchestrate(query, agents, client, max_tokens=2048):
    """Pay for one routing call and one specialist call; return specialist text."""
    if not query.strip():
        raise ValueError("Query must not be empty")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    router = client.messages.create(
        model=agents["data-engineer"]["model"],
        temperature=0,
        max_tokens=64,
        system=(
            "Classify the user's request. Return ONLY one exact agent name from "
            f"this role map: {json.dumps(ROLES)}. Choose the main task for mixed "
            "requests; use python-scripter if no role is a clear match."
        ),
        messages=[{"role": "user", "content": query}],
    )
    routing_input, routing_output = log_usage("router (added routing cost)", router)
    selected = response_text(router)
    if selected not in agents:
        raise ValueError(f"Router returned an unknown agent: {selected!r}")
    agent = agents[selected]
    result = client.messages.create(
        model=agent["model"],
        temperature=agent["temperature"],
        max_tokens=max_tokens,
        system=agent["system_prompt"],
        messages=[{"role": "user", "content": query}],
    )
    worker_input, worker_output = log_usage(selected, result)
    inputs = routing_input + worker_input
    outputs = routing_output + worker_output
    print(
        f"total: input={inputs} output={outputs} total={inputs + outputs}",
        file=sys.stderr,
    )
    return response_text(result)


def main():
    """Read configuration and print the selected specialist's response."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("agents.yaml")
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    try:
        agents = load_agents(args.config)
        with anthropic.Anthropic() as client:
            print(orchestrate(args.query, agents, client, args.max_tokens))
    except (OSError, ValueError, yaml.YAMLError, anthropic.AnthropicError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
