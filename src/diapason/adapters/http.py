"""Adapter that calls an OpenAI-compatible chat endpoint.

This adapter POSTs each example to a chat-completions endpoint and returns the
assistant's text answer. It is deliberately dependency-free: requests are made
with :mod:`urllib` from the standard library, so scoring a hosted model does
not require installing a separate HTTP client.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

from diapason.adapters.base import ModelAdapter
from diapason.registry import adapters

if TYPE_CHECKING:
    from diapason.datasets.manifest import Example


def build_chat_request(
    example: Example, *, model: str, max_tokens: int = 256, temperature: float = 0.0
) -> dict[str, Any]:
    """Build an OpenAI-style chat completion body for one example.

    Args:
        example: The example to turn into a request.
        model: Model identifier to pass to the endpoint.
        max_tokens: Cap on the generated answer length.
        temperature: Sampling temperature; ``0.0`` for deterministic output.

    Returns:
        A JSON-serializable request body.
    """
    content: list[dict[str, Any]] = []
    if example.audio:
        content.append({"type": "input_audio", "input_audio": {"path": example.audio}})
    content.append({"type": "text", "text": example.prompt or "Describe the audio."})
    return {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": content}],
    }


def extract_answer(response: dict[str, Any]) -> str:
    """Pull the assistant's text answer out of a chat completion response.

    Args:
        response: The decoded JSON response body.

    Returns:
        The assistant message content.

    Raises:
        ValueError: If the response does not look like a chat completion.
    """
    try:
        return str(response["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"unexpected chat completion response: {response!r}") from exc


@adapters.register("http")
class HttpAdapter(ModelAdapter):
    """Call an OpenAI-compatible chat endpoint to produce predictions."""

    name = "http"

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = "",
        max_tokens: int = 256,
        temperature: float = 0.0,
        timeout: float = 30.0,
        max_retries: int = 3,
        **options: Any,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            **options,
        )
        if not base_url:
            raise ValueError("http adapter requires a base_url")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    def predict(self, example: Example) -> str:
        """Send the example to the endpoint and return the assistant answer."""
        payload = build_chat_request(
            example,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        response = self._post(payload)
        return extract_answer(response)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST ``payload`` to the chat completions endpoint with retries."""
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                request = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self._backoff_delay(attempt))
        attempts = self.max_retries
        raise RuntimeError(f"chat completions request failed after {attempts} attempts") from last_error

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        """Exponential backoff delay in seconds for a zero-based attempt."""
        return 2.0**attempt
