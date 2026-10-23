# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""What "Download this run" writes: one JSON file holding the mission, the settings, every step with its reasoning, tool calls, results, tokens and
seconds, the answer, the evidence, the claim check, the lab rounds and the scorecard.

A saved run is shared and pasted into reports, so nothing private goes in. Every string in the file passes through `redact` (credentials, bearer
tokens, key names with their values, and home directories), the server address loses any user name and password, and a model name that is a path is
cut to its last part. The snapshot dict from `Session.snapshot` is the only input; nothing here talks to a server.
"""
from __future__ import annotations

import dataclasses
import json
import re
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

FORMAT, VERSION = "arex2-research-lab-run", 1
KEEP = 20                                         # exports kept in the folder; older ones are removed so it does not grow without end

# Each pattern removes the whole credential, including the word that introduces it, so neither the secret nor its label stays in the file.
_SECRET_PATTERNS = [
    re.compile(r"(?i)\bauthorization\s*[:=]\s*(?:bearer\s+|basic\s+|token\s+)?[^\s\"',;]+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=\-]+"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)\b[A-Z0-9_]*(?:api[_-]?key|secret|passwd|password|token)[A-Z0-9_]*\s*[=:]\s*[^\s\"',;]+"),
]
_HOME = re.compile(r"(?:/home|/Users)/[^/\s\"'<>]+")
_REDACTED = "<redacted>"


def redact(text: str) -> str:
    """The text with credentials and home directories removed."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(_REDACTED, text)
    return _HOME.sub("<home>", text)


def _plain(value: Any) -> Any:
    """JSON-safe data: dataclasses become dicts, sets and tuples lists, other objects text, and every string is redacted."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    if isinstance(value, str):
        return redact(value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    return redact(str(value))


def _server(url: str) -> str:
    """scheme://host:port of the model server, with no user name, password, path or query."""
    try:
        parts = urlsplit((url or "").strip())
        host = parts.hostname or ""
        if not parts.scheme or not host:
            return ""
        return f"{parts.scheme}://{host}" + (f":{parts.port}" if parts.port else "")
    except ValueError:
        return ""


def _settings(s: Any) -> dict:
    model = str(getattr(s, "model", "") or "")
    return {"server": _server(getattr(s, "base_url", "")), "model": model.rstrip("/").rsplit("/", 1)[-1] if "/" in model else model,
            "mode": s.mode, "max_steps": s.max_steps, "max_tokens": s.max_tokens, "max_seconds": s.max_seconds, "step_tokens": s.step_tokens,
            "temperature": s.temperature, "top_p": s.top_p, "compare": bool(s.compare)}


def export_run(snap: Optional[dict]) -> Optional[dict]:
    """The document for one session snapshot, or None when there is no session."""
    if not snap:
        return None
    m = snap.get("mission") or {}
    doc = {"format": FORMAT, "version": VERSION,
           "mission": {k: m.get(k) for k in ("id", "title", "kind", "prompt")},
           "settings": _settings(snap["settings"]),
           "running": bool(snap.get("running")),
           "final": snap.get("final"),
           "answer": snap.get("answer") or "",
           "steps": snap.get("steps") or [],
           "sources": snap.get("sources") or [],
           "claims": snap.get("claims") or [],
           "cited": snap.get("cited") or {},
           "rounds": [{"n": r.number, "code": r.code, "report": r.report} for r in (snap.get("rounds") or [])],
           "scorecard": {"agent": snap.get("agent"), "plain": snap.get("plain")}}
    return _plain(doc)


def write_export(snap: Optional[dict], folder: Path) -> Optional[Path]:
    """Write the export of a snapshot into `folder` and return its path (None when there is nothing to write)."""
    doc = export_run(snap)
    if doc is None:
        return None
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    mission_id = re.sub(r"[^A-Za-z0-9_\-]+", "-", str(doc["mission"].get("id") or "run")).strip("-") or "run"
    path = folder / f"arex2-run-{mission_id}-{time.strftime('%Y%m%d-%H%M%S')}-{time.monotonic_ns() % 10**6:06d}.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    old = sorted(folder.glob("arex2-run-*.json"), key=lambda p: (p.stat().st_mtime, p.name))
    for p in old[:-KEEP]:
        try:
            p.unlink()
        except OSError:
            pass
    return path
