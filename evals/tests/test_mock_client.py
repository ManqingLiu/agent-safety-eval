"""Tests for MockAnthropicClient -- response replay."""

import json
import tempfile
from pathlib import Path

import pytest

from evals.harness.mock_client import MockAnthropicClient, MockContentBlock, MockResponse


SAMPLE_RESPONSES = [
    {
        "content": [
            {"type": "text", "text": "I'll read that file for you."},
            {"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {"path": "README.md"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 100, "output_tokens": 50},
    },
    {
        "content": [
            {"type": "text", "text": "Here is the project summary."},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 200, "output_tokens": 80},
    },
]


class TestMockClient:
    def test_replays_responses_in_order(self):
        client = MockAnthropicClient(responses=SAMPLE_RESPONSES)

        r1 = client.messages.create(model="test", messages=[])
        assert r1.stop_reason == "tool_use"
        assert len(r1.content) == 2
        assert r1.content[0].type == "text"
        assert r1.content[1].type == "tool_use"
        assert r1.content[1].name == "read_file"

        r2 = client.messages.create(model="test", messages=[])
        assert r2.stop_reason == "end_turn"
        assert r2.content[0].text == "Here is the project summary."

    def test_returns_stop_when_exhausted(self):
        client = MockAnthropicClient(responses=[SAMPLE_RESPONSES[0]])
        client.messages.create(model="test", messages=[])  # consume the one response
        r = client.messages.create(model="test", messages=[])
        assert r.stop_reason == "end_turn"
        assert "exhausted" in r.content[0].text.lower()

    def test_logs_calls(self):
        client = MockAnthropicClient(responses=SAMPLE_RESPONSES)
        client.messages.create(model="m1", messages=[{"role": "user", "content": "hi"}])
        assert len(client.messages.call_log) == 1
        assert client.messages.call_log[0]["model"] == "m1"

    def test_from_recording_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(SAMPLE_RESPONSES, f)
            f.flush()
            client = MockAnthropicClient.from_recording(Path(f.name))

        r = client.messages.create(model="test", messages=[])
        assert r.stop_reason == "tool_use"

    def test_empty_client(self):
        client = MockAnthropicClient()
        r = client.messages.create(model="test", messages=[])
        assert r.stop_reason == "end_turn"


class TestMockResponse:
    def test_from_dict(self):
        r = MockResponse.from_dict(SAMPLE_RESPONSES[0])
        assert r.stop_reason == "tool_use"
        assert r.content[1].input == {"path": "README.md"}

    def test_content_block_types(self):
        block = MockContentBlock.from_dict({"type": "tool_use", "id": "tu_1", "name": "bash", "input": {"command": "ls"}})
        assert block.type == "tool_use"
        assert block.name == "bash"
        assert block.input == {"command": "ls"}
