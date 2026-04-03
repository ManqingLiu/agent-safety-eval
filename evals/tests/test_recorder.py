"""Tests for RecordingClient -- capture live responses for replay."""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import pytest

from evals.harness.recorder import RecordingClient, _response_to_dict
from evals.harness.mock_client import MockAnthropicClient


# -- Fake Anthropic response objects (mimic real SDK structure) --

@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class FakeTextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class FakeToolUseBlock:
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = None

    def __post_init__(self):
        if self.input is None:
            self.input = {}


@dataclass
class FakeResponse:
    content: list = None
    stop_reason: str = "end_turn"
    usage: FakeUsage = None

    def __post_init__(self):
        if self.content is None:
            self.content = []
        if self.usage is None:
            self.usage = FakeUsage()


class FakeMessages:
    """Mimics anthropic.Anthropic().messages."""
    def __init__(self, responses: list[FakeResponse]):
        self._responses = responses
        self._index = 0

    def create(self, **kwargs: Any) -> FakeResponse:
        if self._index >= len(self._responses):
            return FakeResponse(content=[FakeTextBlock(text="exhausted")])
        r = self._responses[self._index]
        self._index += 1
        return r


class FakeAnthropicClient:
    def __init__(self, responses: list[FakeResponse]):
        self.messages = FakeMessages(responses)


class TestRecordingClient:
    def test_records_text_response(self):
        fake = FakeAnthropicClient([
            FakeResponse(content=[FakeTextBlock(text="Hello world")], stop_reason="end_turn"),
        ])
        recorder = RecordingClient(fake)

        r = recorder.messages.create(model="test", messages=[])
        assert r.content[0].text == "Hello world"  # passthrough works
        assert len(recorder.recorded_responses) == 1
        assert recorder.recorded_responses[0]["content"][0]["text"] == "Hello world"
        assert recorder.recorded_responses[0]["stop_reason"] == "end_turn"

    def test_records_tool_use_response(self):
        fake = FakeAnthropicClient([
            FakeResponse(
                content=[
                    FakeTextBlock(text="Let me read that."),
                    FakeToolUseBlock(id="tu_1", name="read_file", input={"path": "x.txt"}),
                ],
                stop_reason="tool_use",
            ),
        ])
        recorder = RecordingClient(fake)
        recorder.messages.create(model="test", messages=[])

        recorded = recorder.recorded_responses[0]
        assert recorded["stop_reason"] == "tool_use"
        assert recorded["content"][0]["type"] == "text"
        assert recorded["content"][1]["type"] == "tool_use"
        assert recorded["content"][1]["name"] == "read_file"
        assert recorded["content"][1]["input"] == {"path": "x.txt"}

    def test_records_multiple_responses(self):
        fake = FakeAnthropicClient([
            FakeResponse(content=[FakeTextBlock(text="Step 1")]),
            FakeResponse(content=[FakeTextBlock(text="Step 2")]),
        ])
        recorder = RecordingClient(fake)
        recorder.messages.create(model="m1", messages=[])
        recorder.messages.create(model="m1", messages=[])

        assert len(recorder.recorded_responses) == 2

    def test_save_and_reload(self, tmp_path):
        fake = FakeAnthropicClient([
            FakeResponse(
                content=[
                    FakeTextBlock(text="Hello"),
                    FakeToolUseBlock(id="tu_1", name="bash", input={"command": "ls"}),
                ],
                stop_reason="tool_use",
            ),
            FakeResponse(content=[FakeTextBlock(text="Done.")], stop_reason="end_turn"),
        ])
        recorder = RecordingClient(fake)
        recorder.messages.create(model="test", messages=[])
        recorder.messages.create(model="test", messages=[])

        # Save
        out_path = tmp_path / "recording.json"
        recorder.save(out_path)
        assert out_path.exists()

        # Reload as mock client
        mock = MockAnthropicClient.from_recording(out_path)
        r1 = mock.messages.create(model="test", messages=[])
        assert r1.stop_reason == "tool_use"
        assert r1.content[0].text == "Hello"
        assert r1.content[1].name == "bash"

        r2 = mock.messages.create(model="test", messages=[])
        assert r2.stop_reason == "end_turn"
        assert r2.content[0].text == "Done."

    def test_save_creates_parent_dirs(self, tmp_path):
        fake = FakeAnthropicClient([
            FakeResponse(content=[FakeTextBlock(text="hi")]),
        ])
        recorder = RecordingClient(fake)
        recorder.messages.create(model="test", messages=[])

        deep_path = tmp_path / "a" / "b" / "c" / "recording.json"
        recorder.save(deep_path)
        assert deep_path.exists()

    def test_usage_captured(self):
        fake = FakeAnthropicClient([
            FakeResponse(
                content=[FakeTextBlock(text="hi")],
                usage=FakeUsage(input_tokens=42, output_tokens=17),
            ),
        ])
        recorder = RecordingClient(fake)
        recorder.messages.create(model="test", messages=[])

        assert recorder.recorded_responses[0]["usage"]["input_tokens"] == 42
        assert recorder.recorded_responses[0]["usage"]["output_tokens"] == 17


class TestResponseToDict:
    def test_text_block(self):
        resp = FakeResponse(content=[FakeTextBlock(text="hello")])
        d = _response_to_dict(resp)
        assert d["content"] == [{"type": "text", "text": "hello"}]

    def test_tool_use_block(self):
        resp = FakeResponse(content=[
            FakeToolUseBlock(id="t1", name="bash", input={"command": "ls"}),
        ])
        d = _response_to_dict(resp)
        assert d["content"][0]["name"] == "bash"
        assert d["content"][0]["id"] == "t1"
