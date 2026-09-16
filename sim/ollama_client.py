from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1"
HEALTH_CHECK_TIMEOUT_SECONDS = 5


class OllamaError(RuntimeError):
    pass


class OllamaConnectionError(OllamaError):
    pass


class OllamaAPIError(OllamaError):
    pass


class OllamaModelUnavailableError(OllamaError):
    pass


class OllamaResponseError(OllamaError):
    pass


def configured_base_url(environment: Mapping[str, str] | None = None) -> str:
    values = os.environ if environment is None else environment
    return values.get("OLLAMA_BASE_URL", "").strip() or DEFAULT_OLLAMA_BASE_URL


def configured_model(environment: Mapping[str, str] | None = None) -> str:
    values = os.environ if environment is None else environment
    return values.get("OLLAMA_MODEL", "").strip() or DEFAULT_OLLAMA_MODEL


class OllamaClient:
    def __init__(self, model: str, host: str = DEFAULT_OLLAMA_BASE_URL, timeout: int = 120):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def health_check(self) -> dict[str, Any]:
        version_data = self._get_json("/api/version", timeout=HEALTH_CHECK_TIMEOUT_SECONDS)
        version = version_data.get("version")
        if not isinstance(version, str) or not version.strip():
            raise OllamaResponseError(
                f"Ollama at {self.host} returned a malformed response from /api/version."
            )

        tags_data = self._get_json("/api/tags", timeout=HEALTH_CHECK_TIMEOUT_SECONDS)
        models = tags_data.get("models")
        if not isinstance(models, list):
            raise OllamaResponseError(
                f"Ollama at {self.host} returned a malformed response from /api/tags."
            )

        available_models = sorted(
            {
                name
                for item in models
                if isinstance(item, dict)
                for name in (item.get("name"), item.get("model"))
                if isinstance(name, str) and name
            }
        )
        if not any(_model_names_match(self.model, available) for available in available_models):
            available = ", ".join(available_models) or "none"
            raise OllamaModelUnavailableError(
                f"Configured Ollama model '{self.model}' is not available at {self.host}. "
                f"Available models: {available}. Install it on the Ollama server with "
                f"`ollama pull {self.model}` or select another model with --model/OLLAMA_MODEL."
            )

        return {
            "base_url": self.host,
            "version": version,
            "model": self.model,
            "available_models": available_models,
        }

    def warmup(self) -> str:
        return self.chat(
            [
                {"role": "system", "content": "Reply with OK."},
                {"role": "user", "content": "OK"},
            ],
            temperature=0.0,
            max_tokens=8,
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
                    data = self._parse_json(line, endpoint="/api/chat stream")
                    self._raise_response_error(data)
                    message = data.get("message")
                    if not isinstance(message, dict):
                        raise OllamaResponseError(
                            f"Ollama at {self.host} returned a malformed streaming chat response."
                        )
                    chunk = message.get("content", "")
                    if not isinstance(chunk, str):
                        raise OllamaResponseError(
                            f"Ollama at {self.host} returned non-text streaming content."
                        )
                    if chunk:
                        chunks.append(chunk)
                        on_chunk(chunk)
                return "".join(chunks).strip()
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc) from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise self._connection_error(exc) from exc

    def _get_json(self, endpoint: str, *, timeout: int) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.host}{endpoint}", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc) from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise self._connection_error(exc) from exc

        data = self._parse_json(raw, endpoint=endpoint)
        self._raise_response_error(data)
        return data

    def _parse_chat_response(self, raw: str) -> str:
        data = self._parse_json(raw, endpoint="/api/chat")
        self._raise_response_error(data)
        message = data.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaResponseError(
                f"Ollama at {self.host} returned a malformed response from /api/chat."
            )
        return message["content"].strip()

    def _parse_json(self, raw: str, *, endpoint: str) -> dict[str, Any]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaResponseError(
                f"Ollama at {self.host} returned malformed JSON from {endpoint}."
            ) from exc
        if not isinstance(data, dict):
            raise OllamaResponseError(
                f"Ollama at {self.host} returned an unexpected response from {endpoint}."
            )
        return data

    def _raise_response_error(self, data: dict[str, Any]) -> None:
        error = data.get("error")
        if error:
            raise OllamaAPIError(f"Ollama API error at {self.host}: {error}")

    def _connection_error(self, exc: BaseException) -> OllamaConnectionError:
        reason = getattr(exc, "reason", exc)
        return OllamaConnectionError(
            f"Could not reach Ollama at {self.host}: {reason}. Verify that Ollama is running "
            "and, when using a remote server, that the SSH tunnel is active and forwarding "
            "to the configured address."
        )

    def _http_error(self, exc: urllib.error.HTTPError) -> OllamaAPIError:
        detail = ""
        try:
            body = exc.read().decode("utf-8")
            data = json.loads(body)
            if isinstance(data, dict) and data.get("error"):
                detail = f": {data['error']}"
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        return OllamaAPIError(
            f"Ollama API request to {exc.url} failed with HTTP {exc.code}{detail}."
        )


def _model_names_match(configured: str, available: str) -> bool:
    return configured == available or f"{configured}:latest" == available or configured == f"{available}:latest"
