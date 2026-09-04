"""Serve the OPBench demo UI and a small read-only JSON API.

The demo deliberately uses Python's standard library so it can be started from
the public repository without installing another web framework. The API reads
a curated set of appendix QA cases and a compact, paper-derived JSON file for
the reported experimental results.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = PROJECT_ROOT / "web"
FRONTEND_ROOT = WEB_ROOT / "frontend"
RESULTS_PATH = WEB_ROOT / "data" / "research_results.json"
QA_CASES_PATH = WEB_ROOT / "data" / "qa_cases.json"


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


RESEARCH_DATA = _load_json(RESULTS_PATH)
QA_CASES = _load_json(QA_CASES_PATH)


def build_examples() -> dict[str, Any]:
    """Return the paper-grounded QA cases prepared for the interactive demo."""

    categories = QA_CASES.get("categories", {})
    if not isinstance(categories, dict) or not categories:
        raise ValueError(f"No QA categories found in {QA_CASES_PATH}")
    return categories


EXAMPLES = build_examples()


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class DemoHandler(SimpleHTTPRequestHandler):
    """Static-file handler with JSON API routes for the demo."""

    server_version = "OPBenchDemo/1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(FRONTEND_ROOT), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api(parsed.path, parse_qs(parsed.query))
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def _handle_api(self, path: str, query: dict[str, list[str]]) -> None:
        if path == "/api/health":
            self._send_json({"status": "ok", "service": "opbench-demo"})
            return
        if path == "/api/summary":
            self._send_json(
                {
                    "meta": RESEARCH_DATA["meta"],
                    "overview": RESEARCH_DATA["overview"],
                    "workflow": RESEARCH_DATA["workflow"],
                }
            )
            return
        if path == "/api/examples":
            category = query.get("category", ["irrelevance"])[0]
            if category not in EXAMPLES:
                self._send_json({"error": f"Unknown category: {category}"}, status=404)
                return
            self._send_json({"meta": QA_CASES.get("meta", {}), **EXAMPLES[category]})
            return
        if path == "/api/results":
            self._send_json(
                {
                    "meta": RESEARCH_DATA["meta"],
                    "rq1": RESEARCH_DATA["rq1"],
                    "rq2": RESEARCH_DATA["rq2"],
                    "rq3": RESEARCH_DATA["rq3"],
                    "rq4": RESEARCH_DATA["rq4"],
                }
            )
            return
        self._send_json({"error": "Not found"}, status=404)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[opbench-demo] {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OPBench demo website")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    args = parser.parse_args()

    mimetypes.add_type("text/javascript", ".js")
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"OPBench demo running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping OPBench demo")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
