# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""One run of a mission (arex2_agent/session.py): the agent, the optional plain-chat comparison, snapshots for the page, stop and resume."""
from __future__ import annotations

import json
import socket
import threading

import pytest

from arex2_agent import code_sandbox as sb
from arex2_agent import lab, missions, session, tools
from tests.fake_openai import FakeOpenAI

OPERA = {"query": {"pages": {"1": {"pageid": 1, "title": "Sydney Opera House",
                                    "extract": "The Sydney Opera House is a performing arts centre. It was formally opened by Queen Elizabeth II on 20 October 1973."}}}}
UTZON = {"query": {"pages": {"1": {"pageid": 1, "title": "Jørn Utzon", "extract": "Jørn Utzon was a Danish architect, born on 9 April 1918 in Copenhagen."}}}}


def call(name, **args):
    return {"name": name, "arguments": json.dumps(args)}


def research(prompt="How old was the architect when it opened?", **kw):
    return {"id": "custom", "title": "t", "kind": "research", "prompt": prompt, "expected": "55", "facts": ["1918", "1973", "55"], "derived": ["55"], "needs": [], **kw}


def settings(f, **kw):
    return session.Settings(base_url=f.url, model="m", mode="online", **kw)


def fake_http(url, params=None, headers=None, data=None):
    title = (params or {}).get("titles", "")
    return json.dumps(UTZON if "Utzon" in title else OPERA)


def run(sess):
    return list(sess.events())


# ---- the agent run ------------------------------------------------------------------------------------------------------------

def test_a_research_run_yields_snapshots_and_ends_with_a_final_answer_stats_and_a_claim_check():
    turns = [{"reasoning": "Look up both.", "tool_calls": [call("wikipedia_page", title="Sydney Opera House"), call("wikipedia_page", title="Jørn Utzon")],
              "usage": {"prompt_tokens": 50, "completion_tokens": 30}},
             {"tool_calls": [call("calculator", expression="1973 - 1918")], "usage": {"prompt_tokens": 90, "completion_tokens": 12}},
             {"content": "Answer: 55 years.\nJørn Utzon was born on 9 April 1918 in Copenhagen. The Sydney Opera House was formally opened on 20 October 1973.\n\nSources:\n- https://en.wikipedia.org/wiki/Sydney_Opera_House",
              "usage": {"prompt_tokens": 140, "completion_tokens": 60}}]
    with FakeOpenAI(turns) as f:
        sess = session.Session(research(), settings(f), toolbox=tools.Toolbox(env={}, http=fake_http))
        snaps = run(sess)
    last = snaps[-1]
    assert last["final"]["stop_reason"] == "answered" and not last["running"]
    assert "55 years" in last["answer"]
    assert [s.url.rsplit("/", 1)[-1] for s in last["sources"]] == ["Sydney_Opera_House", "J%C3%B8rn_Utzon"]
    statuses = {c.status for c in last["claims"]}
    assert "backed" in statuses and "unbacked" not in statuses
    assert last["cited"] == {"opened": 1, "never_opened": []}
    a = last["agent"]
    assert a["steps"] == 3 and a["tool_calls"] == 3 and a["sources"] == 2 and a["completion_tokens"] == 102 and a["stop_reason"] == "answered"
    assert last["meters"]["steps"] == 3 and last["meters"]["tokens"] == 102


def test_the_first_snapshot_has_no_steps_and_a_middle_one_shows_a_live_step_with_its_reasoning_so_far():
    with FakeOpenAI([{"reasoning": "thinking hard about this", "content": "done"}], piece=3) as f:
        snaps = run(session.Session(research(), settings(f), toolbox=tools.Toolbox(env={})))
    assert snaps[0]["steps"] == [] and snaps[0]["running"]
    live = [s for s in snaps if s["steps"] and s["steps"][-1].get("live")]
    assert live and any("thinking" in s["steps"][-1]["reasoning"] for s in live)
    assert snaps[-1]["steps"][-1]["live"] is False


def test_the_mission_text_is_the_user_message_and_the_settings_reach_the_request():
    with FakeOpenAI([{"content": "x"}]) as f:
        run(session.Session(research(prompt="MISSION TEXT"), settings(f, step_tokens=777, temperature=0.2, top_p=0.5), toolbox=tools.Toolbox(env={})))
        body = f.requests[0]
    assert body["messages"][1]["content"] == "MISSION TEXT" and body["max_tokens"] == 777 and body["temperature"] == 0.2 and body["top_p"] == 0.5


def test_numbers_from_the_calculator_make_a_claim_computed_when_the_rest_is_backed():
    turns = [{"tool_calls": [call("wikipedia_page", title="Sydney Opera House"), call("wikipedia_page", title="Jørn Utzon")]},
             {"tool_calls": [call("calculator", expression="1973 - 1918")]},
             {"content": "Jørn Utzon was born in 1918, which is 55 years before the Sydney Opera House was formally opened in 1973."}]
    with FakeOpenAI(turns) as f:
        last = run(session.Session(research(), settings(f), toolbox=tools.Toolbox(env={}, http=fake_http)))[-1]
    assert [c.status for c in last["claims"]] == ["computed"]


def test_a_cited_address_the_agent_never_opened_shows_up_in_the_final_snapshot():
    with FakeOpenAI([{"content": "Answer: x.\n\nSources:\n- https://example.com/made-up"}]) as f:
        last = run(session.Session(research(), settings(f), toolbox=tools.Toolbox(env={})))[-1]
    assert last["cited"]["never_opened"] == ["https://example.com/made-up"]


def test_the_stats_count_only_tool_calls_that_ran():
    same = {"tool_calls": [call("calculator", expression="1+1")]}
    with FakeOpenAI([same, same, {"content": "ok"}]) as f:
        last = run(session.Session(research(), settings(f), toolbox=tools.Toolbox(env={})))[-1]
    assert last["agent"]["tool_calls"] == 1


def test_snapshots_are_throttled_so_a_long_stream_does_not_flood_the_page():
    with FakeOpenAI([{"reasoning": "r" * 4000, "content": "c" * 4000}], piece=2) as f:
        snaps = run(session.Session(research(), settings(f), toolbox=tools.Toolbox(env={})))
    assert len(snaps) < 60


# ---- the plain chat comparison --------------------------------------------------------------------------------------------------

def test_with_compare_on_the_same_question_is_asked_once_with_no_tools_and_the_result_lands_in_the_final_snapshot():
    with FakeOpenAI([{"content": "agent answer"}], plain_turns=[{"reasoning": "hm", "content": "plain guess: about 55 years", "usage": {"prompt_tokens": 9, "completion_tokens": 40}}]) as f:
        snaps = run(session.Session(research(prompt="THE QUESTION"), settings(f, compare=True), toolbox=tools.Toolbox(env={})))
        plain_req = next(r for r in f.requests if not r.get("tools"))
    last = snaps[-1]
    assert plain_req["messages"] == [{"role": "user", "content": "THE QUESTION"}]
    assert last["plain"]["answer"] == "plain guess: about 55 years" and last["plain"]["completion_tokens"] == 40 and last["plain"]["steps"] == 1
    assert last["plain"]["sources"] == 0 and last["plain"]["stop_reason"] == "answered"
    assert last["agent"]["answer"] == "agent answer"


def test_plain_chat_streams_into_the_snapshots_while_it_runs():
    with FakeOpenAI([{"content": "agent"}], plain_turns=[{"reasoning": "plain thought streaming", "content": "plain text", "delay_s": 0.0}], piece=2) as f:
        snaps = run(session.Session(research(), settings(f, compare=True), toolbox=tools.Toolbox(env={})))
    assert any(s["plain_live"]["reasoning"] or s["plain_live"]["content"] for s in snaps[:-1]) or snaps[-1]["plain"]["answer"]


def test_a_failure_of_the_plain_chat_does_not_stop_the_agent():
    with FakeOpenAI([{"content": "agent answer"}], plain_turns=[{"status": 503, "error": "busy"}]) as f:
        last = run(session.Session(research(), settings(f, compare=True), toolbox=tools.Toolbox(env={})))[-1]
    assert last["agent"]["stop_reason"] == "answered" and last["plain"]["stop_reason"] == "server_down"


def test_compare_off_leaves_plain_empty():
    with FakeOpenAI([{"content": "agent"}]) as f:
        last = run(session.Session(research(), settings(f), toolbox=tools.Toolbox(env={})))[-1]
    assert last["plain"] is None and len(f.requests) == 1


# ---- offline mode, a dead server, stop and resume -------------------------------------------------------------------------------------

def test_offline_mode_serves_the_bundled_corpus_and_never_calls_the_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network used")
    monkeypatch.setattr(tools, "default_http", boom)
    turns = [{"tool_calls": [call("wikipedia_page", title="Jørn Utzon")]}, {"content": "Born in 1918."}]
    with FakeOpenAI(turns) as f:
        last = run(session.Session(research(), session.Settings(base_url=f.url, model="m", mode="offline")))[-1]
    assert last["final"]["stop_reason"] == "answered" and last["sources"][0].title == "Jørn Utzon" and "1918" in last["sources"][0].text


def test_an_unreachable_server_ends_the_run_with_a_plain_message_and_no_traceback():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    sess = session.Session(research(), session.Settings(base_url=f"http://127.0.0.1:{port}/v1", model="m", mode="offline", compare=True))
    last = run(sess)[-1]
    assert last["final"]["stop_reason"] == "server_down" and f"127.0.0.1:{port}" in last["final"]["detail"] and not last["running"]
    assert last["plain"]["stop_reason"] == "server_down"


def test_an_empty_model_name_is_looked_up_from_the_server():
    with FakeOpenAI([{"content": "x"}], model="auto-found") as f:
        run(session.Session(research(), session.Settings(base_url=f.url, model="", mode="offline")))
        assert f.requests[0]["model"] == "auto-found"


def test_the_stop_button_ends_both_runs():
    stop = threading.Event()
    long = {"reasoning": "x" * 6000, "content": "y" * 100}
    with FakeOpenAI([long], plain_turns=[long], piece=2) as f:
        sess = session.Session(research(), settings(f, compare=True), cancel=stop, toolbox=tools.Toolbox(env={}))
        out = []
        for snap in sess.events():
            out.append(snap)
            if len(out) == 8:
                stop.set()
    last = out[-1]
    assert last["final"]["stop_reason"] == "cancelled" and last["plain"]["stop_reason"] == "cancelled" and not last["running"]


def test_a_cut_off_run_can_be_resumed_with_a_larger_limit_and_goes_on_to_an_answer():
    turns = [{"reasoning": "long", "finish_reason": "length"}, {"content": "now answered"}]
    with FakeOpenAI(turns) as f:
        sess = session.Session(research(), settings(f, step_tokens=500), toolbox=tools.Toolbox(env={}))
        first = run(sess)[-1]
        assert first["final"]["stop_reason"] == "length" and first["can_resume"]
        second = list(sess.resume(step_tokens=3000))[-1]
        assert f.requests[1]["max_tokens"] == 3000
    assert second["final"]["stop_reason"] == "answered" and second["answer"] == "now answered" and not second["can_resume"]


def test_resume_with_more_steps_continues_a_run_that_hit_the_step_limit():
    turns = [{"tool_calls": [call("calculator", expression="1+1")]}, {"tool_calls": [call("calculator", expression="2+2")]}, {"content": "done"}]
    with FakeOpenAI(turns) as f:
        sess = session.Session(research(), settings(f, max_steps=2), toolbox=tools.Toolbox(env={}))
        assert run(sess)[-1]["final"]["stop_reason"] == "step_cap"
        last = list(sess.resume(extra_steps=3))[-1]
    assert last["final"]["stop_reason"] == "answered" and last["agent"]["steps"] == 3


# ---- settings ----------------------------------------------------------------------------------------------------------------------

def test_settings_are_clamped_to_sane_ranges_and_a_bad_mode_becomes_online():
    s = session.Settings(base_url=" http://x:1/v1 ", model=" m ", mode="weird", max_steps=0, max_tokens=-5, max_seconds=0, step_tokens=10**9, temperature=9, top_p=0).clean()
    assert s.base_url == "http://x:1/v1" and s.model == "m" and s.mode == "online"
    assert 1 <= s.max_steps <= 40 and s.max_tokens >= 1000 and s.max_seconds >= 30 and 256 <= s.step_tokens <= 32768 and 0.0 <= s.temperature <= 2.0 and 0.05 <= s.top_p <= 1.0


def test_a_bad_server_address_is_reported_in_plain_words_not_raised():
    last = run(session.Session(research(), session.Settings(base_url="not a url", model="m", mode="offline")))[-1]
    assert last["final"]["stop_reason"] == "server_down" and "server address" in last["final"]["detail"]


# ---- the lab ---------------------------------------------------------------------------------------------------------------------------

REF = ("def count_pairs(nums, target):\n    seen = {}\n    total = 0\n    for x in nums:\n        total += seen.get(target - x, 0)\n        seen[x] = seen.get(x, 0) + 1\n    return total\n")
WRONG = "def count_pairs(nums, target):\n    return 0\n"       # fails most tests quickly


@pytest.mark.skipif(not sb.self_check()["ok"], reason="this kernel cannot enforce the sandbox")
def test_a_lab_run_records_each_submission_as_a_round_and_ends_with_the_best_score():
    mission = missions.get("lab-count_pairs")
    turns = [{"reasoning": "first a simple version", "tool_calls": [call("submit_solution", code=WRONG)]},
             {"reasoning": "it timed out; use a dictionary", "tool_calls": [call("submit_solution", code=REF)]},
             {"content": "Best code is round 2 with a hash map."}]
    with FakeOpenAI(turns) as f:
        snaps = run(session.Session(mission, settings(f, max_steps=6)))
    last = snaps[-1]
    assert [r.number for r in last["rounds"]] == [1, 2]
    assert last["rounds"][0].report.score < 0.7 and last["rounds"][1].report.score >= 0.99
    assert last["agent"]["lab_score"] >= 0.99 and last["final"]["stop_reason"] == "answered"
    assert any(len(s["rounds"]) == 1 for s in snaps)           # the page sees round 1 before round 2


def test_the_lab_uses_the_lab_system_prompt_and_no_web_tools():
    mission = missions.get("lab-count_pairs")
    with FakeOpenAI([{"content": "nothing to submit"}]) as f:
        run(session.Session(mission, settings(f)))
        body = f.requests[0]
    assert body["messages"][0]["content"] == lab.LAB_SYSTEM_PROMPT
    assert {t["function"]["name"] for t in body["tools"]} == {"submit_solution", "run_python", "calculator"}


@pytest.mark.skipif(not sb.self_check()["ok"], reason="this kernel cannot enforce the sandbox")
def test_in_the_lab_the_plain_chat_one_shot_code_is_scored_by_the_same_harness():
    mission = missions.get("lab-count_pairs")
    with FakeOpenAI([{"content": "agent final"}], plain_turns=[{"content": "```python\n" + WRONG + "```"}]) as f:
        last = run(session.Session(mission, settings(f, compare=True)))[-1]
    assert last["plain"]["lab_score"] is not None and last["plain"]["lab_score"] < 0.7
