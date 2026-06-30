#!/usr/bin/env python3
"""Simple HTTP server to receive experiment results from the cluster.

Run on the EC2 instance:
    python3 receive_results.py --port 8432

The SLURM job can POST results here instead of using scp (useful if direct
SSH from cluster to EC2 is awkward). Alternatively you can just use scp/rsync
directly — this server is a convenience.

Endpoints:
    POST /upload    — Upload a tar.gz file. Extracts into experiments/out/
    GET  /status    — Health check
    GET  /results   — List available results
"""

import http.server
import json
import os
import shutil
import tarfile
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "experiments" / "out"
PLOTS_DIR = Path(__file__).parent / "gen_imgs" / "attention_evolution"


class ResultsHandler(http.server.BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/status":
            self._respond_json({"status": "ok", "time": datetime.now().isoformat()})
        elif self.path == "/results":
            results = []
            if RESULTS_DIR.exists():
                for d in sorted(RESULTS_DIR.iterdir()):
                    if d.is_dir() and (d / "attn_evolution.pt").exists():
                        meta_path = d / "meta.json"
                        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                        results.append({"name": d.name, "meta": meta})
            self._respond_json({"results": results})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/upload":
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_error(400, "Empty body")
                return
            body = self.rfile.read(content_length)
            try:
                self._handle_upload(body)
                self._respond_json({"status": "uploaded", "size_mb": len(body) / 1e6})
            except Exception as e:
                self.send_error(500, str(e))
        else:
            self.send_error(404)

    def _handle_upload(self, data):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            f.write(data)
            tmp_path = f.name
        try:
            with tarfile.open(tmp_path, "r:gz") as tar:
                # Security: filter out absolute paths and parent traversals
                safe_members = []
                for member in tar.getmembers():
                    if member.name.startswith("/") or ".." in member.name:
                        continue
                    safe_members.append(member)
                tar.extractall(path=str(Path(__file__).parent), members=safe_members)
            print(f"[{datetime.now().isoformat()}] Received and extracted {len(data)} bytes")
        finally:
            os.unlink(tmp_path)

    def _respond_json(self, obj):
        body = json.dumps(obj, indent=2).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Results receiver server")
    parser.add_argument("--port", type=int, default=8432)
    parser.add_argument("--bind", default="0.0.0.0")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    server = http.server.HTTPServer((args.bind, args.port), ResultsHandler)
    print(f"Results receiver listening on {args.bind}:{args.port}")
    print(f"  POST /upload  — upload tar.gz of results")
    print(f"  GET  /status  — health check")
    print(f"  GET  /results — list received results")
    print(f"  Results dir: {RESULTS_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
