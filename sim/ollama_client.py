from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, model: str, host: str = "http://localhost:11434", timeout: int = 120):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def warmup(self) -> None:
        self.chat(
            [
                {"role": "system", "content": "Reply with OK."},
                {"role": "user", "content": "OK"},
            ],
            temperature=0.0,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        format_json: bool = False,
        max_tokens: int | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        stream = on_chunk is not None
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": temperature},
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
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
                if not stream:
                    raw = response.read().decode("utf-8")
                    return self._parse_chat_response(raw)

                chunks: list[str] = []
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise OllamaError(f"Unexpected Ollama stream response: {line[:500]}") from exc

                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        chunks.append(chunk)
                        on_chunk(chunk)

                return "".join(chunks).strip()
        except urllib.error.URLError as exc:
            raise OllamaError(
                "Could not reach Ollama. Start it with `ollama serve` and make sure the model is installed."
            ) from exc

    def _parse_chat_response(self, raw: str) -> str:
        try:
            data = json.loads(raw)
            return data["message"]["content"].strip()
        except (json.JSONDecodeError, KeyError) as exc:
            raise OllamaError(f"Unexpected Ollama response: {raw[:500]}") from exc
