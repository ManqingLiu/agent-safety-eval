"""MockAnthropicClient -- replays recorded LLM responses for fast, deterministic testing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MockContentBlock:
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> MockContentBlock:
        return cls(
            type=d["type"],
            text=d.get("text", ""),
            id=d.get("id", ""),
            name=d.get("name", ""),
            input=d.get("input"),
        )


@dataclass
class MockResponse:
    content: list[MockContentBlock]
    stop_reason: str
    usage: dict[str, int]

    @classmethod
    def from_dict(cls, d: dict) -> MockResponse:
        return cls(
            content=[MockContentBlock.from_dict(b) for b in d["content"]],
            stop_reason=d["stop_reason"],
            usage=d.get("usage", {"input_tokens": 0, "output_tokens": 0}),
        )


class MockMessages:
    """Drop-in replacement for client.messages with recorded responses."""

    def __init__(self, responses: list[dict]):
        self.responses = responses
        self.call_index = 0
        self.call_log: list[dict] = []

    def create(self, **kwargs: Any) -> MockResponse:
        self.call_log.append(kwargs)
        if self.call_index >= len(self.responses):
            # Return a stop response if recordings exhausted
            return MockResponse(
                content=[MockContentBlock(type="text", text="[Mock responses exhausted]")],
                stop_reason="end_turn",
                usage={"input_tokens": 0, "output_tokens": 0},
            )
        resp = self.responses[self.call_index]
        self.call_index += 1
        return MockResponse.from_dict(resp)


class MockAnthropicClient:
    """Drop-in replacement for anthropic.Anthropic that replays recorded responses.

    Usage:
        client = MockAnthropicClient.from_recording("path/to/recording.json")
        # or
        client = MockAnthropicClient(responses=[...])
    """

    def __init__(self, responses: list[dict] | None = None):
        self.messages = MockMessages(responses or [])

    @classmethod
    def from_recording(cls, path: Path) -> MockAnthropicClient:
        data = json.loads(path.read_text())
        return cls(responses=data if isinstance(data, list) else data.get("responses", []))

    @classmethod
    def from_responses(cls, responses: list[dict]) -> MockAnthropicClient:
        return cls(responses=responses)
