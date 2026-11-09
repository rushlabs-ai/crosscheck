"""The local web app: upload a photo, hear where the models agree.

By default it runs on 127.0.0.1 only. With `lan=True` (crosscheck serve --lan) it also listens on
your network so a phone can use it; every device except this computer must then pair with a
6-digit code shown on this computer. Requests must carry a localhost or IP-address Host header (no
DNS rebinding) and, for POSTs, a same-origin Origin (no cross-site requests). Photos stay in memory
and go only to your Ollama.
"""

from __future__ import annotations

import base64
import binascii
import ipaddress
import json
import secrets
import ssl
import threading
import time
from collections import OrderedDict
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from queue import Queue
from typing import Any, Callable
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
SESSION_COOKIE = "cc_session"
SESSION_DAYS = 30


def host_allowed(host_header: str, port: int, lan: bool) -> bool:
    """A Host header naming this port on localhost or, on the network, an IP address.

    Domain names are refused: a page on another site that points its own name at this
    machine (DNS rebinding) never gets an answer.
    """
    host, sep, host_port = host_header.rpartition(":")
    if not sep or host_port != str(port):
        return False
    if host in LOCAL_HOSTS:
        return True
    if not lan:
        return False
    try:
        ipaddress.ip_address(host.removeprefix("[").removesuffix("]"))
    except ValueError:
        return False
    return True


class Pairing:
    """A 6-digit code shown on this computer; devices that enter it get a session cookie.

    After MAX_FAILURES wrong codes the code is replaced and pairing pauses for LOCKOUT seconds,
    so guessing all million codes would take months, and each new code shows up on the computer.
    """

    MAX_FAILURES = 5
    LOCKOUT = 60.0

    def __init__(self, on_new_code: Callable[[str], None] | None = None):
        self._lock = threading.Lock()
        self._sessions: set[str] = set()
        self._failures = 0
        self._locked_until = 0.0
        self._on_new_code = on_new_code
        self.code = f"{secrets.randbelow(10**6):06d}"

    def _fresh(self) -> str:
        code = f"{secrets.randbelow(10**6):06d}"
        if self._on_new_code:
            self._on_new_code(code)
        return code

    def pair(self, code: str) -> tuple[str | None, int]:
        """Returns (session token or None, attempts left with the current code; -1 while paused)."""
        with self._lock:
            if time.monotonic() < self._locked_until:
                return None, -1
            if secrets.compare_digest(code.strip().encode(), self.code.encode()):
                token = secrets.token_urlsafe(32)
                self._sessions.add(token)
                return token, self.MAX_FAILURES - self._failures
            self._failures += 1
            if self._failures >= self.MAX_FAILURES:
                self._failures = 0
                self._locked_until = time.monotonic() + self.LOCKOUT
                self.code = self._fresh()
                return None, 0
            return None, self.MAX_FAILURES - self._failures

    def valid(self, token: str) -> bool:
        with self._lock:
            return token in self._sessions


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


def make_handler(jobs: Jobs, port: int, pairing: Pairing | None = None, scheme: str = "http",
                 trust_loopback: bool = True):
    lan = pairing is not None

    class Handler(BaseHTTPRequestHandler):
        server_version = f"crosscheck/{__version__}"
        sys_version = ""

        def setup(self) -> None:
            if isinstance(self.request, ssl.SSLSocket):
                self.request.settimeout(15)
                self.request.do_handshake()  # here, in the connection's own thread
            super().setup()

        def log_message(self, fmt: str, *args: Any) -> None:  # no photo data or questions in logs
            pass

        def _send(self, status: int, body: bytes, ctype: str = "application/json",
                  extra: dict[str, str] | None = None) -> None:
            self.send_response(status)
            for name, value in (extra or {}).items():
                self.send_header(name, value)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj: Any, extra: dict[str, str] | None = None) -> None:
            self._send(status, json.dumps(obj).encode(), extra=extra)

        def _host_ok(self) -> bool:
            if not host_allowed(self.headers.get("Host", ""), port, lan):
                self._json(HTTPStatus.FORBIDDEN, {"error": "host not allowed"})
                return False
            return True

        def _paired(self) -> bool:
            if pairing is None:
                return True
            if trust_loopback and ipaddress.ip_address(self.client_address[0]).is_loopback:
                return True  # this computer itself
            try:
                morsel = SimpleCookie(self.headers.get("Cookie", "")).get(SESSION_COOKIE)
            except CookieError:
                return False
            return bool(morsel) and pairing.valid(morsel.value)

        def _require_pairing(self) -> bool:
            if self._paired():
                return True
            self._json(HTTPStatus.UNAUTHORIZED, {"error": "enter the pairing code shown on the computer first"})
            return False

        def _read_json(self, limit: int) -> dict | None:
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "send JSON"})
                return None
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length <= 0 or length > limit:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "the photo is too large (25 MB at most)"})
                return None
            try:
                body = json.loads(self.rfile.read(length))
            except ValueError:
                body = None
            if not isinstance(body, dict):
                self._json(HTTPStatus.BAD_REQUEST, {"error": "expected a JSON object"})
                return None
            return body

        def do_GET(self) -> None:
            if not self._host_ok():
                return
            path = urlsplit(self.path).path
            if path in STATIC:
                name, ctype = STATIC[path]
                self._send(HTTPStatus.OK, (resources.files("crosscheck") / "static" / name).read_bytes(), ctype)
            elif path == "/api/session":
                self._json(HTTPStatus.OK, {"pairing": lan, "paired": self._paired()})
            elif path.startswith("/api/jobs/"):
                if self._require_pairing():
                    job = jobs.get(path.rsplit("/", 1)[-1])
                    self._json(HTTPStatus.OK if job else HTTPStatus.NOT_FOUND, job or {"error": "unknown job"})
            elif path == "/api/info":
                if self._require_pairing():
                    self._json(HTTPStatus.OK, {"models": jobs.models, "version": __version__})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._host_ok():
                return
            origin = self.headers.get("Origin")
            if origin is not None and origin != f"{scheme}://{self.headers.get('Host')}":
                self._json(HTTPStatus.FORBIDDEN, {"error": "cross-origin request refused"})
                return
            path = urlsplit(self.path).path
            if path == "/api/pair" and pairing is not None:
                self._pair()
                return
            if path != "/api/check":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if not self._require_pairing():
                return
            body = self._read_json(MAX_BODY)
            if body is None:
                return
            try:
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

        def _pair(self) -> None:
            body = self._read_json(1024)
            if body is None:
                return
            token, left = pairing.pair(str(body.get("code") or ""))
            if token is None:
                if left < 0:
                    self._json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "Too many wrong codes. Wait a minute, then "
                                                              "enter the new code shown on the computer.", "attempts_left": 0})
                    return
                msg = ("Wrong code. A new code is now shown on the computer; wait a minute before trying it." if left == 0
                       else f"Wrong code. {left} attempt(s) left before the code changes.")
                self._json(HTTPStatus.FORBIDDEN, {"error": msg, "attempts_left": left})
                return
            cookie = (f"{SESSION_COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict; "
                      f"Max-Age={SESSION_DAYS * 86400}" + ("; Secure" if scheme == "https" else ""))
            self._json(HTTPStatus.OK, {"paired": True}, extra={"Set-Cookie": cookie})

    return Handler


def lan_addresses() -> list[str]:
    """This computer's IPv4 addresses on the local network (no packets are sent)."""
    import socket

    found: set[str] = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # UDP connect only picks a route
            found.add(s.getsockname()[0])
    except OSError:
        pass
    try:
        found.update(str(info[4][0]) for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    return sorted(ip for ip in found if not ip.startswith("127."))


class Server(ThreadingHTTPServer):
    daemon_threads = True
    pairing: Pairing | None = None

    def handle_error(self, request: Any, client_address: Any) -> None:
        import sys

        if isinstance(sys.exc_info()[1], (ssl.SSLError, ConnectionError, TimeoutError)):
            return  # failed TLS handshakes and dropped connections are not worth a traceback
        super().handle_error(request, client_address)


def serve(client: Ollama, models: list[str], port: int = 8765, lan: bool = False,
          tls: tuple[str, str] | None = None, on_new_code: Callable[[str], None] | None = None,
          trust_loopback: bool = True) -> Server:
    """The web app. `lan` listens on every network interface and requires pairing; `tls` is a
    (certificate, key) pair of PEM files for HTTPS."""
    jobs = Jobs(client, models)
    pairing = Pairing(on_new_code) if lan else None
    scheme = "https" if tls else "http"

    def handler(p: int):
        return make_handler(jobs, p, pairing, scheme, trust_loopback)

    httpd = Server(("0.0.0.0" if lan else "127.0.0.1", port), handler(port))
    if port == 0:  # tests: bind first, then learn the port
        httpd.RequestHandlerClass = handler(httpd.server_address[1])
    if tls:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_cert_chain(*tls)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True, do_handshake_on_connect=False)
    httpd.pairing = pairing
    return httpd
