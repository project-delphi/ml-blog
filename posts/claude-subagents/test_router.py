"""Offline contract tests; never send requests to Anthropic."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import anthropic
import httpx
import pytest
import yaml
from router import load_agents, orchestrate


@pytest.fixture
def agents():
    return load_agents(Path(__file__).with_name("agents.yaml"))


def response(text, inputs, outputs, stop="end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=inputs, output_tokens=outputs),
        stop_reason=stop,
    )


@pytest.mark.parametrize(
    "name", ["python-scripter", "data-engineer", "front-end-specialist"]
)
def test_routing_passes_configuration_and_counts_both_calls(agents, name, capsys):
    client = Mock()
    client.messages.create.side_effect = [
        response(name, 100, 8),
        response("result", 200, 40),
    ]
    assert orchestrate("Build something", agents, client) == "result"
    first, second = client.messages.create.call_args_list
    assert first.kwargs["model"] == agents["data-engineer"]["model"]
    for field in ("model", "temperature"):
        assert second.kwargs[field] == agents[name][field]
    assert second.kwargs["system"] == agents[name]["system_prompt"]
    assert second.kwargs["messages"] == [{"role": "user", "content": "Build something"}]
    captured = capsys.readouterr()
    assert "router (added routing cost): input=100 output=8 total=108" in captured.err
    assert "total: input=300 output=48 total=348" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    "text, stop",
    [
        ("unknown", "end_turn"),
        ("python-scripter", "max_tokens"),
        ("", "end_turn"),
        ("no", "refusal"),
    ],
)
def test_invalid_routing_stops_before_worker(agents, text, stop):
    client = Mock()
    client.messages.create.return_value = response(text, 10, 2, stop)
    with pytest.raises(ValueError):
        orchestrate("Build something", agents, client)
    assert client.messages.create.call_count == 1


def test_truncated_worker_is_not_reported_as_success(agents):
    client = Mock()
    client.messages.create.side_effect = [
        response("python-scripter", 10, 2),
        response("partial", 20, 2048, "max_tokens"),
    ]
    with pytest.raises(ValueError, match="max_tokens"):
        orchestrate("Build something", agents, client)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "temperature", "tools"])
def test_invalid_configuration(tmp_path, agents, mutation):
    entries = list(agents.values())
    if mutation == "duplicate":
        entries.append(entries[0])
    elif mutation == "missing":
        entries.pop()
    elif mutation == "temperature":
        entries[0]["temperature"] = float("nan")
    else:
        entries[0]["tools"] = ["bash_terminal"]
    config = tmp_path / "agents.yaml"
    config.write_text(yaml.safe_dump({"agents": entries}))
    with pytest.raises(ValueError):
        load_agents(config)


def test_empty_query_makes_no_api_call(agents):
    client = Mock()
    with pytest.raises(ValueError, match="empty"):
        orchestrate(" ", agents, client)
    client.messages.create.assert_not_called()


def test_official_sdk_request_and_response_contract(agents):
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        text = "data-engineer" if len(requests) == 1 else "SELECT 1;"
        return httpx.Response(
            200,
            json={
                "id": f"msg_test_{len(requests)}",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    with anthropic.Anthropic(
        api_key="offline-test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        max_retries=0,
    ) as client:
        assert orchestrate("Write SQL", agents, client) == "SELECT 1;"
    assert len(requests) == 2
    assert requests[1]["system"] == agents["data-engineer"]["system_prompt"]
    assert requests[1]["temperature"] == 0.0
