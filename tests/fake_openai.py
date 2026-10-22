# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""A scripted OpenAI-compatible server for the tests: /v1/models and a streamed /v1/chat/completions.

Each call to chat/completions takes the next scripted turn. A turn is a dict:
    {"reasoning": str, "content": str, "tool_calls": [{"name": str, "arguments": str}], "finish_reason": "stop"|"tool_calls"|"length",
     "usage": {"prompt_tokens": int, "completion_tokens": int}, "delay_s": float, "status": int}
Text is sent in pieces of `piece` characters so a test sees a real stream. `requests` keeps every request body.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeOpenAI:
    def __init__(self, turns=(), model="arex-2-test", piece=7):
        self.turns = list(turns)
        self.model = model
        self.piece = piece
        self.models_status = 200
        self.requests: list[dict] = []
        self.headers_seen: list[dict] = []
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.rstrip("/") == "/v1/models" and outer.models_status != 200:
                    self.send_response(outer.models_status)
                    self.end_headers()
                    self.wfile.write(b'{"error": "models unavailable"}')
                elif self.path.rstrip("/") == "/v1/models":
                    body = json.dumps({"object": "list", "data": [{"id": outer.model, "object": "model"}]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                with outer._lock:
                    outer.requests.append(body)
                    outer.headers_seen.append(dict(self.headers))
                    turn = outer.turns.pop(0) if outer.turns else None
                if turn is None:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b'{"error": "script exhausted"}')
                    return
                if turn.get("delay_s"):
                    time.sleep(turn["delay_s"])
                if turn.get("status", 200) != 200:
                    self.send_response(turn["status"])
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": turn.get("error", "failed")}).encode())
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    outer._stream(self.wfile, turn)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def _send(self, wfile, delta, finish=None, usage=None):
        chunk = {"id": "c1", "object": "chat.completion.chunk", "model": self.model,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}] if delta is not None or finish else []}
        if usage:
            chunk["usage"] = usage
        wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        wfile.flush()

    def _pieces(self, text):
        return [text[i:i + self.piece] for i in range(0, len(text), self.piece)] if text else []

    def _stream(self, wfile, turn):
        self._send(wfile, {"role": "assistant", "content": ""})
        for p in self._pieces(turn.get("reasoning", "")):
            self._send(wfile, {"reasoning_content": p})
        for p in self._pieces(turn.get("content", "")):
            self._send(wfile, {"content": p})
        for i, tc in enumerate(turn.get("tool_calls", [])):
            args = tc.get("arguments", "{}")
            first = {"tool_calls": [{"index": i, "id": tc.get("id", f"call_{i}"), "type": "function",
                                     "function": {"name": tc["name"], "arguments": ""}}]}
            self._send(wfile, first)
            for p in self._pieces(args):
                self._send(wfile, {"tool_calls": [{"index": i, "function": {"arguments": p}}]})
        finish = turn.get("finish_reason") or ("tool_calls" if turn.get("tool_calls") else "stop")
        self._send(wfile, {}, finish=finish)
        if turn.get("usage"):
            self._send(wfile, None, usage=turn["usage"])
        wfile.write(b"data: [DONE]\n\n")
        wfile.flush()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.server.shutdown()
        self.server.server_close()
