#!/usr/bin/env python3
"""Tiny web server that connects web/index.html to the Python engine.

    python3 server.py              # http://127.0.0.1:8000
    python3 server.py --port 9000

Uses only the standard library (no pip install needed).

Endpoints
---------
    GET  /                 the web UI (web/index.html)
    GET  /api/catalogue    {"songs": [...], "query": "..."}  starting data
    POST /api/attribute    {"query": str, "songs": [{id,title,lyrics}], "k": 5}
                           -> the engine's AttributionReport as JSON
"""

import argparse
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from engine import AttributionEngine

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "web")
CATALOGUE = os.path.join(HERE, "data", "catalogue.json")
QUERY_FILE = os.path.join(HERE, "data", "query_song.txt")
MAX_BODY = 2 * 1024 * 1024


def default_payload():
    with open(CATALOGUE, "r", encoding="utf-8") as fh:
        songs = json.load(fh)
    query = ""
    if os.path.exists(QUERY_FILE):
        with open(QUERY_FILE, "r", encoding="utf-8") as fh:
            # first line of the file is a title, the rest is the lyrics
            lines = fh.read().splitlines()
            query = "\n".join(lines[1:]).strip() if len(lines) > 1 else "\n".join(lines)
    return {"songs": songs, "query": query}


def run_attribution(body):
    query = str(body.get("query", ""))
    songs = body.get("songs", [])
    k = max(1, min(int(body.get("k", 5)), 20))
    records = [
        {"id": str(s["id"]), "title": str(s.get("title", "")),
         "artist": str(s.get("artist", "")), "lyrics": str(s.get("lyrics", ""))}
        for s in songs
    ]
    if not records:
        return {"query": "query", "query_stats": {}, "matches": []}
    engine = AttributionEngine(k=k, top_k=len(records))
    engine.ingest_records(records)
    return engine.attribute(query).to_dict()


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, payload, content_type="application/json"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/catalogue":
            return self._send(200, default_payload())
        if path == "/":
            path = "/index.html"
        full = os.path.normpath(os.path.join(WEB_DIR, path.lstrip("/")))
        if not full.startswith(WEB_DIR) or not os.path.isfile(full):
            return self._send(404, {"error": "not found"})
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        with open(full, "rb") as fh:
            self._send(200, fh.read(), ctype + ("; charset=utf-8" if ctype.startswith("text") else ""))

    def do_POST(self):
        if self.path != "/api/attribute":
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY:
                return self._send(413, {"error": "request too large"})
            body = json.loads(self.rfile.read(length) or b"{}")
            self._send(200, run_attribution(body))
        except (ValueError, KeyError, TypeError) as exc:
            self._send(400, {"error": str(exc)})

    def log_message(self, fmt, *args):
        print("  " + fmt % args)


def main():
    ap = argparse.ArgumentParser(description="Serve the lyrics engine web UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Lyrics engine UI running at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
