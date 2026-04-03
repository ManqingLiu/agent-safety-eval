"""RecordingClient -- wraps a live Anthropic client to capture responses for replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RecordingMessages:
    """Wraps client.messages to record every create() response."""

    def __init__(self, real_messages: Any):
        self._real = real_messages
        self.recorded: list[dict] = []

    def create(self, **kwargs: Any) -> Any:
        response = self._real.create(**kwargs)
        self.recorded.append(_response_to_dict(response))
        return response


class RecordingClient:
    """Drop-in wrapper for anthropic.Anthropic that records all LLM responses.

    Usage:
        from anthropic import Anthropic
        real_client = Anthropic()
        recorder = RecordingClient(real_client)

        # Use recorder as if it were the real client
        # ... run agent ...

        # Save recorded responses
        recorder.save("recordings/scenario_name.json")

        # Later, replay with MockAnthropicClient:
        mock = MockAnthropicClient.from_recording("recordings/scenario_name.json")
    """

    def __init__(self, real_client: Any):
        self._real = real_client
        self.messages = RecordingMessages(real_client.messages)

    def save(self, path: Path | str) -> None:
        """Save all recorded responses to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.messages.recorded, indent=2))

    @property
    def recorded_responses(self) -> list[dict]:
        return self.messages.recorded


def _response_to_dict(response: Any) -> dict:
    """Convert an Anthropic API response to a serializable dict."""
    content = []
    for block in response.content:
        block_dict: dict[str, Any] = {"type": block.type}
        if block.type == "text":
            block_dict["text"] = block.text
        elif block.type == "tool_use":
            block_dict["id"] = block.id
            block_dict["name"] = block.name
            block_dict["input"] = block.input
        content.append(block_dict)

    return {
        "content": content,
        "stop_reason": response.stop_reason,
        "usage": {
            "input_tokens": getattr(response.usage, "input_tokens", 0),
            "output_tokens": getattr(response.usage, "output_tokens", 0),
        },
    }
