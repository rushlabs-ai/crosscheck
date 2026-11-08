"""The local web app: upload a photo, hear where the models agree.

Runs on 127.0.0.1 only. Requests must carry a local Host header (no DNS rebinding)
and, for POSTs, a same-origin Origin (no cross-site requests). Photos stay in
memory and go only to your Ollama.
"""

from __future__ import annotations

import base64
import binascii
import json
import secrets
import threading
import time
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from queue import Queue
from typing import Any
from urllib.parse import urlsplit

from . import __version__
from .images import ImageError, check_image
from .ollama import Ollama
from .pipeline import run

MAX_BODY = 25 * 1024 * 1024
MAX_JOBS = 20
JOB_TTL = 30 * 60
STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/app.css": ("app.css", "text/css; charset=utf-8")}
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; connect-src 'self'; "
       "form-action 'self'; base-uri 'none'; frame-ancestors 'none'")
LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")


class Jobs:
    """Photos are checked one at a time (the GPU is the bottleneck)."""

    def __init__(self, client: Ollama, models: list[str]):
        self.client, self.models = client, models
        self.jobs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.lock = threading.Lock()
        self.queue: Queue[str] = Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, image: bytes, question: str) -> str | None:
        with self.lock:
            self._expire()
            if sum(1 for j in self.jobs.values() if j["state"] in ("queued", "running")) >= MAX_JOBS:
                return None
            job_id = secrets.token_urlsafe(12)
            self.jobs[job_id] = {"state": "queued", "message": "Waiting for the previous photo to finish.",
                                 "created": time.time(), "image": image, "question": question}
        self.queue.put(job_id)
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            out = {k: v for k, v in job.items() if k not in ("image", "question", "created")}
            if job["state"] == "queued":
                ahead = 0
                for jid, j in self.jobs.items():
                    if jid == job_id:
                        break
                    if j["state"] in ("queued", "running"):
                        ahead += 1
                out["ahead"] = ahead
            return out

    def _expire(self) -> None:
        now = time.time()
        for jid in [j for j, v in self.jobs.items() if now - v["created"] > JOB_TTL and v["state"] in ("done", "error")]:
            del self.jobs[jid]

    def _set(self, job_id: str, **kw: Any) -> None:
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(kw)

    def _worker(self) -> None:
        while True:
            job_id = self.queue.get()
            with self.lock:
                job = self.jobs.get(job_id)
                if not job:
                    continue
                image, question = job.pop("image"), job.pop("question")
            self._set(job_id, state="running", message="Starting.")
            try:
                result = run(self.client, image, self.models, question,
                             progress=lambda msg: self._set(job_id, message=msg))
                self._set(job_id, state="done", message="Done.", result=result.to_dict())
            except Exception as e:  # report, never crash the worker
                self._set(job_id, state="error", message=f"Something went wrong: {e}")


def make_handler(jobs: Jobs, port: int):
    allowed_hosts = {f"{h}:{port}" for h in LOCAL_HOSTS}
    allowed_origins = {f"http://{h}" for h in allowed_hosts}

    class Handler(BaseHTTPRequestHandler):
        server_version = f"crosscheck/{__version__}"
        sys_version = ""

        def log_message(self, fmt: str, *args: Any) -> None:  # no photo data or questions in logs
            pass

        def _send(self, status: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj: Any) -> None:
            self._send(status, json.dumps(obj).encode())

        def _host_ok(self) -> bool:
            if self.headers.get("Host", "") not in allowed_hosts:
                self._json(HTTPStatus.FORBIDDEN, {"error": "host not allowed"})
                return False
            return True

        def do_GET(self) -> None:
            if not self._host_ok():
                return
            path = urlsplit(self.path).path
            if path in STATIC:
                name, ctype = STATIC[path]
                self._send(HTTPStatus.OK, (resources.files("crosscheck") / "static" / name).read_bytes(), ctype)
            elif path.startswith("/api/jobs/"):
                job = jobs.get(path.rsplit("/", 1)[-1])
                self._json(HTTPStatus.OK if job else HTTPStatus.NOT_FOUND, job or {"error": "unknown job"})
            elif path == "/api/info":
                self._json(HTTPStatus.OK, {"models": jobs.models, "version": __version__})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._host_ok():
                return
            origin = self.headers.get("Origin")
            if origin is not None and origin not in allowed_origins:
                self._json(HTTPStatus.FORBIDDEN, {"error": "cross-origin request refused"})
                return
            if urlsplit(self.path).path != "/api/check":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "send JSON"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length <= 0 or length > MAX_BODY:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "the photo is too large (25 MB at most)"})
                return
            try:
                body = json.loads(self.rfile.read(length))
                data = str(body.get("image") or "")
                if data.startswith("data:"):
                    data = data.split(",", 1)[-1]
                image = check_image(base64.b64decode(data, validate=True))
                question = str(body.get("question") or "")[:500]
            except ImageError as e:  # a ValueError too, so it comes first
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
                return
            except (ValueError, AttributeError, binascii.Error):
                self._json(HTTPStatus.BAD_REQUEST, {"error": "expected JSON with a base64 image"})
                return
            job_id = jobs.submit(image, question)
            if job_id is None:
                self._json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "too many photos waiting; try again shortly"})
                return
            self._json(HTTPStatus.ACCEPTED, {"job": job_id})

    return Handler


def serve(client: Ollama, models: list[str], port: int = 8765) -> ThreadingHTTPServer:
    jobs = Jobs(client, models)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(jobs, port))
    if port == 0:  # tests: bind first, then learn the port
        httpd.RequestHandlerClass = make_handler(jobs, httpd.server_address[1])
    return httpd
