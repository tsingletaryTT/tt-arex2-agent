# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""The scripted stand-in model (tools/demo_server.py): it must keep working end to end, because the README screenshots are made with it."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import demo_server as ds  # noqa: E402

from arex2_agent import chat, code_sandbox as sb, missions, session  # noqa: E402


def run(mission_id, compare=False, mode="offline"):
    srv = ds.make_server(delay_s=0.0)
    with srv:
        sess = session.Session(missions.get(mission_id), session.Settings(base_url=srv.url, model="", mode=mode, compare=compare, max_steps=8))
        return list(sess.events())[-1]


def test_the_model_name_says_it_is_not_the_real_model():
    srv = ds.make_server()
    with srv:
        assert chat.ChatClient(srv.url).models() == ["SCRIPTED-FAKE-MODEL (not AREX-2)"]


def test_the_scripted_research_mission_runs_to_an_answer_with_every_claim_backed_or_computed_and_both_sources_cited_and_opened():
    last = run("opera-utzon", compare=True)
    assert last["final"]["stop_reason"] == "answered" and "55 years" in last["answer"]
    assert last["claims"] and {c.status for c in last["claims"]} <= {"backed", "computed"}
    assert last["cited"] == {"opened": 2, "never_opened": []}
    assert last["agent"]["steps"] == 4 and last["agent"]["tool_calls"] == 4 and len(last["sources"]) >= 2
    assert "I did not check" in last["plain"]["answer"] and last["plain"]["sources"] == 0


def test_the_script_is_stateless_so_two_runs_in_a_row_play_the_same_way():
    a, b = run("opera-utzon"), run("opera-utzon")
    assert a["answer"] == b["answer"] and a["agent"]["steps"] == b["agent"]["steps"]


def test_an_unscripted_mission_gets_a_plain_message_saying_so():
    srv = ds.make_server(delay_s=0.0)
    with srv:
        custom = {"id": "custom", "title": "t", "kind": "research", "prompt": "Something unscripted entirely", "expected": "", "facts": [], "needs": [], "verified": {}}
        last = list(session.Session(custom, session.Settings(base_url=srv.url, mode="offline")).events())[-1]
    assert "only knows" in last["answer"]


@pytest.mark.skipif(not sb.self_check()["ok"], reason="this kernel cannot enforce the sandbox")
def test_the_scripted_lab_run_scores_greedy_then_memoised_then_bottom_up_in_the_real_sandbox():
    last = run("lab-min_coins", compare=True)
    scores = [r.report.score for r in last["rounds"]]
    assert len(scores) == 3 and scores[0] < 0.7 and scores[1] < 0.95 and scores[2] >= 0.99, scores
    assert last["final"]["stop_reason"] == "answered" and last["plain"]["lab_score"] is not None and last["plain"]["lab_score"] < 0.7
