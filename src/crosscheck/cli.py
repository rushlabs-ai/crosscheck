"""Command line: `crosscheck serve` (the web app) and `crosscheck describe PHOTO`."""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from pathlib import Path

from . import __version__
from .images import ImageError, check_image
from .ollama import DEFAULT_URL, Ollama, OllamaError
from .pipeline import run

# Three model families, so their mistakes are less likely to be shared. All run on a 12-16 GB GPU
# one at a time (Ollama swaps them), or on a CPU, slowly.
DEFAULT_MODELS = ["qwen3-vl:4b-instruct", "gemma4:e4b", "ministral-3:3b"]


def _models(arg: str | None) -> list[str]:
    raw = arg or os.environ.get("CROSSCHECK_MODELS") or ",".join(DEFAULT_MODELS)
    models = [m.strip() for m in raw.split(",") if m.strip()]
    return list(dict.fromkeys(models))


def _check_installed(client: Ollama, models: list[str]) -> int:
    try:
        have = client.installed()
    except OllamaError as e:
        print(f"crosscheck: {e}. Start Ollama first (https://ollama.com).", file=sys.stderr)
        return 2
    missing = [m for m in models if m not in have]
    if missing:
        print("crosscheck: these models are not in Ollama yet:", file=sys.stderr)
        for m in missing:
            print(f"  ollama pull {m}", file=sys.stderr)
        return 2
    if len(models) < 2:
        print("crosscheck: note: with one model nothing can be cross-checked.", file=sys.stderr)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import lan_addresses, serve

    if bool(args.tls_cert) != bool(args.tls_key):
        print("crosscheck: --tls-cert and --tls-key go together", file=sys.stderr)
        return 2
    client = Ollama(args.ollama)
    models = _models(args.models)
    if (code := _check_installed(client, models)) != 0:
        return code
    tls = (args.tls_cert, args.tls_key) if args.tls_cert else None
    scheme = "https" if tls else "http"

    def announce(new_code: str) -> None:
        print(f"Too many wrong codes. New pairing code: {' '.join(new_code)}", flush=True)

    try:
        httpd = serve(client, models, args.port, lan=args.lan, tls=tls, on_new_code=announce)
    except (OSError, ValueError) as e:
        print(f"crosscheck: could not start the web app: {e}", file=sys.stderr)
        return 1
    port = httpd.server_address[1]
    url = f"{scheme}://127.0.0.1:{port}/"
    print(f"Crosscheck is running at {url} with {', '.join(models)}. Press Ctrl+C to stop.")
    if args.lan and httpd.pairing:
        addresses = lan_addresses() or ["<this computer's IP address>"]
        print("On your phone, on the same Wi-Fi, open: " + "  or  ".join(f"{scheme}://{ip}:{port}/" for ip in addresses))
        print(f"Pairing code: {' '.join(httpd.pairing.code)}")
        print("Other devices on this network can reach the page, but nothing works without the code.")
        if not tls:
            print("Photos travel over your network unencrypted: use --lan only on a network you trust, "
                  "or add --tls-cert and --tls-key.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    client = Ollama(args.ollama)
    models = _models(args.models)
    try:
        image = check_image(Path(args.photo).read_bytes())
    except (OSError, ImageError) as e:
        print(f"crosscheck: {e}", file=sys.stderr)
        return 2
    if (code := _check_installed(client, models)) != 0:
        return code
    progress = None if args.json else (lambda msg: print(msg, file=sys.stderr))
    result = run(client, image, models, args.question or "", progress=progress)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0
    data = result.to_dict()
    print(result.spoken)
    for key, title in (("disputed", "Where the models disagree"), ("agreed", "What every model agrees on"),
                       ("unconfirmed", "Details not every model could judge")):
        if data[key]:
            print(f"\n{title}:")
            for f in data[key]:
                print(f"- {f['text']}")
                if key == "disputed":
                    print("  " + "; ".join(f"{m}: {j}" for m, j in f["judgements"].items()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="crosscheck", description="Several local vision models describe a photo; "
                                "hear where they agree and where they disagree.")
    p.add_argument("--version", action="version", version=f"crosscheck {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ollama", default=os.environ.get("CROSSCHECK_OLLAMA", DEFAULT_URL), help="Ollama URL")
    common.add_argument("--models", help=f"comma-separated Ollama models (default: {','.join(DEFAULT_MODELS)})")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve", parents=[common], help="Start the web app on 127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--lan", action="store_true",
                   help="also listen on your network so a phone can use it (devices pair with a code)")
    s.add_argument("--tls-cert", help="PEM certificate for HTTPS (with --tls-key)")
    s.add_argument("--tls-key", help="PEM private key for HTTPS")
    s.add_argument("--no-browser", action="store_true", help="Do not open a browser")
    s.set_defaults(func=cmd_serve)
    d = sub.add_parser("describe", parents=[common], help="Check one photo in the terminal")
    d.add_argument("photo")
    d.add_argument("--question", help="What you want to know, e.g. 'What does the label say?'")
    d.add_argument("--json", action="store_true", help="Print the full result as JSON")
    d.set_defaults(func=cmd_describe)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except OllamaError as e:
        print(f"crosscheck: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
