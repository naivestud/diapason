"""Tests for the HTTP adapter's request and response handling.

These cover the pure helpers that build requests and parse responses; no
network access is needed.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from diapason.adapters.base import AdapterError
from diapason.adapters.http import HttpAdapter, build_chat_request, extract_answer
from diapason.datasets.manifest import Example


def test_build_request_includes_audio_and_prompt() -> None:
    example = Example(id="1", audio="clip.wav", prompt="What is this sound?")
    request = build_chat_request(example, model="audio-model")
    assert request["model"] == "audio-model"
    message = request["messages"][0]
    assert message["role"] == "user"
    assert message["content"][0]["type"] == "input_audio"
    assert message["content"][1]["text"] == "What is this sound?"


def test_build_request_uses_default_prompt() -> None:
    example = Example(id="1", audio="clip.wav")
    request = build_chat_request(example, model="m")
    assert request["messages"][0]["content"][1]["text"] == "Describe the audio."


def test_build_request_without_audio() -> None:
    example = Example(id="1", prompt="A question?")
    request = build_chat_request(example, model="m")
    content = request["messages"][0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"


def test_build_request_prepends_system_prompt() -> None:
    example = Example(id="1", prompt="What is this sound?")
    request = build_chat_request(example, model="m", system_prompt="Answer concisely.")
    assert request["messages"][0] == {"role": "system", "content": "Answer concisely."}
    assert request["messages"][1]["role"] == "user"


def test_build_request_omits_system_message_by_default() -> None:
    example = Example(id="1", prompt="What is this sound?")
    request = build_chat_request(example, model="m")
    assert len(request["messages"]) == 1
    assert request["messages"][0]["role"] == "user"


def test_adapter_passes_system_prompt() -> None:
    adapter = HttpAdapter(base_url="https://api.example.com", system_prompt="Be brief.")
    assert adapter.system_prompt == "Be brief."


def test_extract_answer_reads_content() -> None:
    response = {"choices": [{"message": {"content": "a dog barking"}}]}
    assert extract_answer(response) == "a dog barking"


def test_extract_answer_rejects_malformed_response() -> None:
    with pytest.raises(ValueError, match="unexpected chat completion response"):
        extract_answer({"unexpected": True})


def test_adapter_requires_base_url() -> None:
    with pytest.raises(ValueError, match="base_url"):
        HttpAdapter(base_url="")


def test_adapter_strips_trailing_slash() -> None:
    adapter = HttpAdapter(base_url="https://api.example.com/v1/")
    assert adapter.base_url == "https://api.example.com/v1"


def test_backoff_is_exponential() -> None:
    assert HttpAdapter._backoff_delay(0) == 1.0
    assert HttpAdapter._backoff_delay(2) == 4.0


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def _chat_response(answer: str) -> dict[str, object]:
    return {"choices": [{"message": {"content": answer}}]}


def test_predict_posts_and_parses_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _FakeResponse:
        seen.append(request.get_full_url())
        return _FakeResponse(_chat_response("a dog"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    adapter = HttpAdapter(base_url="https://api.example.com/v1", model="m")
    assert adapter.predict(Example(id="1", prompt="?")) == "a dog"
    assert seen == ["https://api.example.com/v1/chat/completions"]


def test_predict_retries_on_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def flaky_urlopen(request: urllib.request.Request, timeout: float) -> _FakeResponse:
        calls["count"] += 1
        if calls["count"] < 3:
            raise urllib.error.URLError("boom")
        return _FakeResponse(_chat_response("a cat"))

    monkeypatch.setattr(urllib.request, "urlopen", flaky_urlopen)
    monkeypatch.setattr("diapason.adapters.http.time.sleep", lambda _seconds: None)
    adapter = HttpAdapter(base_url="https://api.example.com", max_retries=3)
    assert adapter.predict(Example(id="1")) == "a cat"
    assert calls["count"] == 3


def test_predict_raises_adapter_error_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_urlopen(request: urllib.request.Request, timeout: float) -> _FakeResponse:
        raise urllib.error.URLError("down")

    monkeypatch.setattr(urllib.request, "urlopen", failing_urlopen)
    monkeypatch.setattr("diapason.adapters.http.time.sleep", lambda _seconds: None)
    adapter = HttpAdapter(base_url="https://api.example.com", max_retries=2)
    with pytest.raises(AdapterError, match="failed after 2 attempts"):
        adapter.predict(Example(id="1"))
