"""A fake Ollama for tests: /api/tags and /api/chat with structured output.

Model "a" and model "b" describe the same photo but read the dose differently;
model "bad" always answers with invalid JSON.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16

DESCRIPTIONS = {
    "a": {"summary": "A red mug of pills on a wooden table.",
          "facts": ["A red mug on a wooden table.", 'The label says "Ibuprofen 200 mg".', "There are 2 spoons."]},
    "b": {"summary": "A red mug and a pill bottle.",
          "facts": ["A red mug sits on a wooden table.", 'The label says "Ibuprofen 400 mg".', "A window is behind the mug."]},
}


ANSWERS = {"a": "Ibuprofen 200 mg", "b": "Ibuprofen 400 mg", "c": "ibuprofen 200 mg."}


def judge(model: str, statement: str) -> str:
    if "400" in statement and model == "a":
        return "no"
    if "200" in statement and model == "b":
        return "no"
    if "window" in statement.lower():
        return "unsure"
    return "yes"


class State:
    def __init__(self, installed=("a:latest", "b:latest", "c:latest", "bad:latest")):
        self.installed = list(installed)
        self.calls: list[dict] = []


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, status, obj):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/tags":
                return self._json(200, {"models": [{"name": n, "model": n} for n in state.installed]})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.calls.append(body)
            model = body["model"]
            msg = body["messages"][0]
            if not msg.get("images") or body.get("format") is None or body.get("stream") is not False:
                return self._json(400, {"error": "bad request from client"})
            if model == "bad":
                return self._json(200, {"message": {"role": "assistant", "content": "{not json"}})
            if model not in DESCRIPTIONS and model != "c":
                return self._json(404, {"error": f"model '{model}' not found"})
            prompt = msg["content"]
            if "numbered statements" in prompt:
                stmts = re.findall(r"^(\d+)\. (.+)$", prompt, re.MULTILINE)
                out = {"answers": [{"n": int(n), "answer": judge(model, s)} for n, s in stmts]}
            else:
                out = dict(DESCRIPTIONS.get(model) or DESCRIPTIONS["a"])
                if "answer" in (body["format"].get("properties") or {}):
                    out["answer"] = ANSWERS[model]
            self._json(200, {"model": model, "message": {"role": "assistant", "content": json.dumps(out)},
                             "total_duration": 2_500_000_000, "load_duration": 500_000_000, "eval_count": 40})

    return Handler


class FakeOllama:
    def __init__(self, **kw):
        self.state = State(**kw)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.state))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
