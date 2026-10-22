# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""A small streaming client for an OpenAI-compatible chat server (the AREX-2 bundle's vLLM).

The bundle serves with `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning_parser qwen3`, so tools go in the
standard `tools` field and the model's thinking arrives as `reasoning_content`. A reply is read as a stream so the page can show
the thinking as it grows: at about 14 tokens a second per request a long thought takes minutes, and a still page looks stuck.
Standard library only.
"""
from __future__ import annotations

import http.client
import json
import socket
import time
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import urlsplit

DEFAULT_URL = "http://127.0.0.1:8000/v1"


class ChatError(Exception):
    """Base class: every message is plain text a person can read."""


class ServerDown(ChatError):
    """The server cannot be reached or stopped answering."""


class ServerError(ChatError):
    def __init__(self, status: int, text: str):
        super().__init__(f"The server answered HTTP {status}: {text}")
        self.status = status


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str          # the raw JSON text the model wrote; the caller parses it and handles bad JSON


@dataclass
class Reply:
    content: str = ""
    reasoning: str = ""
    tool_calls: list = field(default_factory=list)
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    seconds: float = 0.0


class ChatClient:
    def __init__(self, base_url: str = DEFAULT_URL, model: Optional[str] = None, api_key: Optional[str] = None,
                 connect_timeout: float = 10.0, read_timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        parts = urlsplit(self.base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"not a server address: {base_url!r} (write it like http://127.0.0.1:8000/v1)")
        self._parts = parts

    # -- plumbing ----------------------------------------------------------------------------------------------------
    def _where(self) -> str:
        return f"{self._parts.hostname}:{self._parts.port or (443 if self._parts.scheme == 'https' else 80)}"

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _connect(self):
        cls = http.client.HTTPSConnection if self._parts.scheme == "https" else http.client.HTTPConnection
        conn = cls(self._parts.hostname, self._parts.port, timeout=self.connect_timeout)
        try:
            conn.connect()
        except (OSError, socket.timeout) as e:
            raise ServerDown(f"Cannot reach the model server at {self._where()}: {e}. Is it running? "
                             "(python serve/serve_arex2.py boots it on a leased board.)") from None
        conn.sock.settimeout(self.read_timeout)
        return conn

    def _path(self, tail: str) -> str:
        return self._parts.path.rstrip("/") + tail

    # -- calls -------------------------------------------------------------------------------------------------------
    def models(self) -> list:
        conn = self._connect()
        try:
            conn.request("GET", self._path("/models"), headers=self._headers())
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status != 200:
                raise ServerError(resp.status, raw.decode("utf-8", "replace")[:300])
            return [m["id"] for m in json.loads(raw).get("data", []) if isinstance(m, dict) and "id" in m]
        except (OSError, socket.timeout) as e:
            raise ServerDown(f"The model server at {self._where()} stopped answering: {e}") from None
        finally:
            conn.close()

    def complete(self, messages, tools=None, max_tokens: int = 4096, temperature: float = 0.6, top_p: float = 0.95,
                 on_delta: Optional[Callable] = None, cancel: Optional[Callable] = None, extra: Optional[dict] = None) -> Reply:
        """One chat turn. `on_delta(kind, text)` gets ("reasoning"|"content", piece); `cancel()` returning True ends the read
        early and the reply says finish_reason "cancelled". Raises ServerDown or ServerError."""
        if self.model is None:
            names = self.models()
            if not names:
                raise ServerError(200, "the server lists no models")
            self.model = names[0]
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "top_p": top_p, "stream": True, "stream_options": {"include_usage": True}}
        if tools:
            body["tools"] = tools
        if extra:
            body.update(extra)
        started = time.monotonic()
        conn = self._connect()
        reply = Reply()
        calls: dict = {}
        try:
            try:
                conn.request("POST", self._path("/chat/completions"), body=json.dumps(body).encode(), headers=self._headers())
                resp = conn.getresponse()
            except (OSError, socket.timeout) as e:
                raise self._timeout_or_down(e) from None
            if resp.status != 200:
                raw = resp.read().decode("utf-8", "replace")
                raise ServerError(resp.status, _error_text(raw))
            try:
                for line in _lines(resp):
                    if cancel is not None and cancel():
                        reply.finish_reason = "cancelled"
                        break
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    self._absorb(chunk, reply, calls, on_delta)
            except (OSError, socket.timeout) as e:
                raise self._timeout_or_down(e) from None
        finally:
            try:
                conn.close()
            except OSError:
                pass
        reply.tool_calls = [ToolCall(c["id"] or f"call_{i}", c["name"], c["arguments"]) for i, c in sorted(calls.items())]
        if reply.finish_reason is None:
            reply.finish_reason = "incomplete"       # the stream ended with no finish_reason: the server or the link dropped
        reply.seconds = time.monotonic() - started
        return reply

    def _timeout_or_down(self, e) -> ServerDown:
        if isinstance(e, (socket.timeout, TimeoutError)):
            return ServerDown(f"The model server at {self._where()} gave no answer for {self.read_timeout:g} s.")
        return ServerDown(f"The model server at {self._where()} stopped answering: {e}")

    @staticmethod
    def _absorb(chunk: dict, reply: Reply, calls: dict, on_delta) -> None:
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            reply.prompt_tokens = usage.get("prompt_tokens", reply.prompt_tokens)
            reply.completion_tokens = usage.get("completion_tokens", reply.completion_tokens)
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            r = delta.get("reasoning_content") or delta.get("reasoning")
            if r:
                reply.reasoning += r
                if on_delta:
                    on_delta("reasoning", r)
            c = delta.get("content")
            if c:
                reply.content += c
                if on_delta:
                    on_delta("content", c)
            for tc in delta.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
            if choice.get("finish_reason"):
                reply.finish_reason = choice["finish_reason"]


def _lines(resp):
    """Server-sent event lines, decoded, as they arrive."""
    buf = b""
    while True:
        part = resp.read1(4096) if hasattr(resp, "read1") else resp.read(4096)
        if not part:
            break
        buf += part
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            yield line.decode("utf-8", "replace").rstrip("\r")
    if buf.strip():
        yield buf.decode("utf-8", "replace").rstrip("\r")


def _error_text(raw: str) -> str:
    try:
        data = json.loads(raw)
        err = data.get("error", data)
        return (err.get("message") if isinstance(err, dict) else str(err))[:300]
    except (ValueError, AttributeError, TypeError):
        return raw[:300]
