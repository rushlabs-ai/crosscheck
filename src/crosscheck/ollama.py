"""A minimal Ollama client: chat with an image and a JSON schema. Standard library only."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

DEFAULT_URL = "http://127.0.0.1:11434"


class OllamaError(Exception):
    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


@dataclass
class Ollama:
    url: str = DEFAULT_URL
    timeout: float = 600.0

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url.rstrip("/") + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read() or b"null")
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "replace")
            try:
                text = json.loads(text).get("error") or text
            except (ValueError, AttributeError):
                pass
            raise OllamaError(str(text)[:300], e.code) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise OllamaError(f"cannot reach Ollama at {self.url} ({getattr(e, 'reason', e)})") from None

    def installed(self) -> set[str]:
        """Model names Ollama has, with and without the ':latest' tag."""
        names: set[str] = set()
        for m in (self._call("GET", "/api/tags") or {}).get("models") or []:
            name = m.get("name") or m.get("model") or ""
            names.add(name)
            if name.endswith(":latest"):
                names.add(name[: -len(":latest")])
        return names

    def chat_json(self, model: str, prompt: str, image: bytes, schema: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Ask about an image; returns (parsed JSON answer, timing info)."""
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(image).decode("ascii")]}],
            "format": schema,  # Ollama constrains the output to this JSON schema
            "stream": False,
            "think": False,
            "options": {"temperature": 0},
        }
        last: Exception | None = None
        for _ in range(2):  # one retry if the answer does not parse
            resp = self._call("POST", "/api/chat", body)
            content = ((resp or {}).get("message") or {}).get("content") or ""
            try:
                parsed = json.loads(content)
            except ValueError as e:
                last = e
                continue
            timing = {
                "seconds": round((resp.get("total_duration") or 0) / 1e9, 2),
                "load_seconds": round((resp.get("load_duration") or 0) / 1e9, 2),
                "output_tokens": resp.get("eval_count"),
            }
            return parsed, timing
        raise OllamaError(f"{model} did not return valid JSON ({last})")
