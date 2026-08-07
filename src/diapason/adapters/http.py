"""Adapter that calls an OpenAI-compatible chat endpoint.

This adapter POSTs each example to a chat-completions endpoint and returns the
assistant's text answer. It is deliberately dependency-free: requests are made
with :mod:`urllib` from the standard library, so scoring a hosted model does
not require installing a separate HTTP client.
"""

from __future__ import annotations

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
        raise NotImplementedError
