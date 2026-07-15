from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class OllamaError(RuntimeError):
    pass

# Yegeon: Increase the timeout because local Ollama models can respond slowly,
# especially when running without a GPU.
class OllamaClient:
    def __init__(self, model: str, host: str = "http://localhost:11434", timeout: int = 300):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        format_json: bool = False,
    ) -> str:
        # Yegeon: Allow longer JSON feedback for evaluator output
        # while keeping patient dialogue responses short.
        max_tokens = 3000 if format_json else 160

        # Yegeon: Disable thinking output and reduce repeated responses
        # for the small local model.
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "repeat_penalty": 1.2,
            },
        }

        if format_json:
            payload["format"] = "json"

        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise OllamaError(
                "Could not reach Ollama. Start it with `ollama serve` and make sure the model is installed."
            ) from exc

        try:
            data = json.loads(raw)
            return data["message"]["content"].strip()
        except (json.JSONDecodeError, KeyError) as exc:
            raise OllamaError(f"Unexpected Ollama response: {raw[:500]}") from exc