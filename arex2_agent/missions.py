# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""The preset missions and the offline corpus.

The research and compute missions come from data/missions.json, built by tools/build_missions.py, which reads each fact from Wikipedia or arXiv and
keeps the quote and the date. The corpus in data/corpus.json is trimmed text of the same pages (Wikipedia text is CC BY-SA 4.0). The lab missions
are the tasks in lab.py.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from . import lab
from .tools import OfflineCorpus, Toolbox

DATA = Path(__file__).resolve().parent / "data"


@lru_cache(maxsize=1)
def _read(name: str):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def load_corpus() -> list:
    return [dict(d) for d in _read("corpus.json")]


def all_missions() -> list:
    out = [dict(m) for m in _read("missions.json")]
    for t in lab.TASKS.values():
        out.append({"id": f"lab-{t.id}", "title": f"Improvement lab: {t.title}", "kind": "lab", "task": t.id, "prompt": lab.mission_text(t),
                    "expected": f"A solution scored by the lab ({t.n_small + 1} tests; speed is timed on a large input).",
                    "facts": [t.function], "needs": ["submit_solution", "run_python"],
                    "plain_chat_risk": "A single reply cannot see its own test results or runtime.", "verified": {"read": "", "quotes": []}})
    return out


def by_kind(kind: str) -> list:
    return [m for m in all_missions() if m["kind"] == kind]


def get(mission_id: str) -> dict:
    for m in all_missions():
        if m["id"] == mission_id:
            return m
    raise KeyError(mission_id)


def offline_toolbox(**kw) -> Toolbox:
    return Toolbox(mode="offline", corpus=OfflineCorpus(load_corpus()), env={}, **kw)
