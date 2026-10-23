# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""The page (demo/app.py): the helpers that decide what runs and what is shown, and the whole app driven through gradio_client."""
from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gradio as gr  # noqa: E402
from demo import app  # noqa: E402

from arex2_agent import code_sandbox as sb  # noqa: E402
from arex2_agent import evidence, lab, missions, session, tools  # noqa: E402
from tests.fake_openai import FakeOpenAI  # noqa: E402

PRESET = "opera-utzon"


def call(name, **args):
    return {"name": name, "arguments": json.dumps(args)}


# ---- choosing the mission ----------------------------------------------------------------------------------------------------------

def test_the_research_choices_hold_the_research_and_compute_presets_and_end_with_write_your_own():
    ids = [v for _, v in app.preset_choices(("research", "compute"))]
    assert "opera-utzon" in ids and "digits-2-1000" in ids and ids[-1] == app.CUSTOM and not any(i.startswith("lab-") for i in ids)


def test_the_lab_choices_hold_only_the_lab_tasks():
    ids = [v for _, v in app.preset_choices(("lab",))][:-1]
    assert ids and all(i.startswith("lab-") for i in ids) and len(ids) == len(lab.TASKS)


def test_an_unedited_preset_runs_as_that_preset_with_its_expected_facts():
    m = missions.get(PRESET)
    assert app.mission_from_inputs(PRESET, m["prompt"]) is not None and app.mission_from_inputs(PRESET, "  " + m["prompt"] + "  ")["id"] == PRESET


def test_an_edited_preset_runs_as_a_custom_mission_without_the_presets_facts():
    got = app.mission_from_inputs(PRESET, missions.get(PRESET)["prompt"] + " Also say the city.")
    assert got["id"] == "custom" and got["facts"] == [] and got["expected"] == "" and "Also say the city." in got["prompt"]


def test_write_your_own_runs_the_text_as_a_custom_research_mission():
    got = app.mission_from_inputs(app.CUSTOM, "Who wrote the Transformer paper?")
    assert got["id"] == "custom" and got["kind"] == "research" and got["prompt"] == "Who wrote the Transformer paper?"


def test_an_empty_or_too_long_mission_is_refused_with_a_plain_message():
    with pytest.raises(gr.Error) as e:
        app.mission_from_inputs(PRESET, "   ")
    assert "mission" in str(e.value).lower()
    with pytest.raises(gr.Error) as e2:
        app.mission_from_inputs(app.CUSTOM, "x" * (app.MAX_PROMPT + 1))
    assert "too long" in str(e2.value)


def test_a_lab_preset_is_not_accepted_on_the_research_page_and_a_research_preset_is_not_a_lab_task():
    lab_prompt = missions.get("lab-count_pairs")["prompt"]
    assert app.mission_from_inputs("lab-count_pairs", lab_prompt, ("research", "compute"))["id"] == "custom"
    assert app.mission_from_inputs("lab-count_pairs", lab_prompt, ("lab",))["id"] == "lab-count_pairs"
    assert app.mission_from_inputs(PRESET, missions.get(PRESET)["prompt"], ("lab",))["id"] == "custom"


def test_an_unknown_preset_id_falls_back_to_a_custom_mission():
    assert app.mission_from_inputs("not-a-preset", "Some question here")["id"] == "custom"


def test_the_settings_are_read_from_the_widgets_and_clamped():
    s = app.settings_from_inputs("http://h:1/v1", "m", "Offline corpus (no network)", 7, 50000, 10, 2048, 0.3, 0.8, True)
    assert (s.base_url, s.model, s.mode, s.max_steps, s.max_tokens, s.max_seconds, s.step_tokens, s.temperature, s.top_p, s.compare) == \
        ("http://h:1/v1", "m", "offline", 7, 50000, 600.0, 2048, 0.3, 0.8, True)
    s2 = app.settings_from_inputs("", None, "Live web tools", None, None, None, None, None, None, False)
    assert s2.mode == "online" and s2.base_url and s2.max_steps >= 1 and s2.compare is False


# ---- what is shown -------------------------------------------------------------------------------------------------------------------

def snap(**kw):
    base = {"mission": missions.get(PRESET), "settings": session.Settings(compare=True), "steps": [], "meters": {"budget": {"max_steps": 12, "max_tokens": 60000, "max_seconds": 1800.0, "step_tokens": 4096}, "steps": 0, "tokens": 0, "seconds": 0.0},
            "sources": [], "claims": [], "cited": {"opened": 0, "never_opened": []}, "final": None, "answer": "", "agent": None, "plain": None,
            "plain_live": {"reasoning": "", "content": ""}, "rounds": [], "can_resume": False, "running": True}
    base.update(kw)
    return base


def test_a_running_snapshot_says_which_step_is_running_and_hides_the_scorecard_and_the_verified_answer():
    r = app.render(snap(steps=[{"n": 2, "reasoning": "", "content": "", "tools": [], "live": True}]))
    assert "Step 2 is running" in r["notice"] and r["score"] == "" and "verified answer" not in r["answer"]
    assert "Step 2" in r["timeline"] and "1 / 12" not in r["meters"]


def test_a_finished_research_snapshot_shows_the_answer_the_verified_reveal_the_scorecard_and_no_continue():
    final = {"stop_reason": "answered", "detail": "", "answer": "55 years.", "steps": 3, "completion_tokens": 100, "seconds": 20.0, "sources": 1}
    agent = {"steps": 3, "completion_tokens": 100, "seconds": 20.0, "sources": 1, "tool_calls": 2, "answer": "55 years.", "claims": None, "stop_reason": "answered"}
    plain = {"answer": "about 55", "reasoning": "", "completion_tokens": 10, "seconds": 3.0, "steps": 1, "sources": 0, "tool_calls": 0, "claims": None, "stop_reason": "answered"}
    r = app.render(snap(running=False, final=final, answer="55 years.", agent=agent, plain=plain))
    assert "55 years." in r["answer"] and "Show the verified answer" in r["answer"] and "Sources read" in r["score"] and "about 55" in r["plain"] and r["continue_label"] == ""
    assert "Finished with an answer" in r["notice"]


def test_a_custom_mission_has_no_verified_reveal_and_no_expected_facts_row():
    custom = {"id": "custom", "title": "Your mission", "kind": "research", "prompt": "q", "expected": "", "facts": [], "verified": {}}
    final = {"stop_reason": "answered", "detail": "", "answer": "x", "steps": 1, "completion_tokens": 1, "seconds": 1.0, "sources": 0}
    agent = {"steps": 1, "completion_tokens": 1, "seconds": 1.0, "sources": 0, "tool_calls": 0, "answer": "x", "claims": None, "stop_reason": "answered"}
    r = app.render(snap(mission=custom, running=False, final=final, answer="x", agent=agent))
    assert "verified answer" not in r["answer"] and "expected facts" not in r["score"].lower()


def test_a_cut_off_snapshot_offers_continue_with_a_label_saying_what_it_will_do():
    final = {"stop_reason": "length", "detail": "", "answer": "", "steps": 1, "completion_tokens": 4096, "seconds": 200.0, "sources": 0}
    agent = {"steps": 1, "completion_tokens": 4096, "seconds": 200.0, "sources": 0, "tool_calls": 0, "answer": "", "claims": None, "stop_reason": "length"}
    r = app.render(snap(running=False, final=final, agent=agent, can_resume=True))
    assert "8,192" in r["continue_label"] and "thinking budget ran out" in r["notice"].lower()


def test_the_lab_page_shows_the_score_chart_in_the_side_panel_and_no_verified_reveal():
    rnd = lab.Round(1, "def f(): pass", lab.ScoreReport(score=0.6, correctness=0.9, speed=0.0, passed=9, total=10, large_seconds=5.0, text="Score 0.60"))
    r = app.render(snap(mission=missions.get("lab-count_pairs"), rounds=[rnd], running=False, final={"stop_reason": "answered", "detail": "", "answer": "best code", "steps": 2,
                                                                                         "completion_tokens": 10, "seconds": 5.0, "sources": 0}, answer="best code",
                        agent={"steps": 2, "completion_tokens": 10, "seconds": 5.0, "sources": 0, "tool_calls": 1, "answer": "best code", "claims": None, "stop_reason": "answered", "lab_score": 0.6}), lab_page=True)
    assert "<svg" in r["side"] and "Round 1" in r["side"] and "verified answer" not in r["answer"] and "Lab score" in r["score"]


def test_the_plain_panel_streams_while_the_comparison_runs_and_is_empty_when_there_is_no_comparison():
    live = {"reasoning": "plain thinking", "content": ""}
    assert "plain thinking" in app.render(snap(plain_live=live))["plain"]
    assert app.render(snap(settings=session.Settings(compare=False)))["plain"] == ""
    waiting = app.render(snap())["plain"]
    assert "answering" in waiting and "Plain chat" in waiting


def test_the_server_badge_names_the_model_when_the_server_answers_and_says_not_reachable_otherwise():
    with FakeOpenAI(model="AREX-2-served") as f:
        assert "AREX-2-served" in app.check_server(f.url, "Live web tools")
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    assert "not reachable" in app.check_server(f"http://127.0.0.1:{port}/v1", "Offline corpus (no network)").lower()
    assert "offline corpus" in app.check_server(f"http://127.0.0.1:{port}/v1", "Offline corpus (no network)").lower()


def test_a_server_address_that_is_not_an_address_gives_the_not_reachable_badge_not_an_exception():
    assert "not reachable" in app.check_server("not a url", "Live web tools").lower()


# ---- the whole app through gradio_client -----------------------------------------------------------------------------------------------

@pytest.fixture
def served_app():
    from gradio_client import Client
    demo = app.build_app("http://127.0.0.1:1/v1").queue(max_size=8, default_concurrency_limit=3)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True, css=app.views.CSS)
    client = Client(f"http://127.0.0.1:{port}/", verbose=False)
    yield client
    demo.close()


def run_args(prompt, url, mode="Offline corpus (no network)", compare=False, preset=app.CUSTOM, steps=8):
    return (preset, prompt, url, "", mode, steps, 60000, 30, 2048, 0.6, 0.95, compare)


def test_a_research_run_through_the_real_app_returns_the_timeline_the_answer_the_evidence_and_the_scorecard(served_app):
    turns = [{"reasoning": "Look up the architect.", "tool_calls": [call("wikipedia_page", title="Jørn Utzon"), call("wikipedia_page", title="Sydney Opera House")]},
             {"tool_calls": [call("calculator", expression="1973 - 1918")]},
             {"content": "Answer: 55 years.\nJørn Utzon was born on 9 April 1918 in Copenhagen. The Sydney Opera House was formally opened on 20 October 1973.\n\nSources:\n- https://en.wikipedia.org/wiki/Jørn_Utzon\n- https://en.wikipedia.org/wiki/Sydney_Opera_House"}]
    with FakeOpenAI(turns, plain_turns=[{"content": "About fifty years, I think."}]) as f:
        job = served_app.submit(*run_args(missions.get(PRESET)["prompt"], f.url, compare=True, preset=PRESET), api_name="/run_research")
        out = job.result(timeout=120)
    notice, meters, timeline, answer, side, score, plain = out            # the client leaves the Continue button out: it carries no data
    assert "Finished with an answer" in notice
    assert "wikipedia_page" in timeline and "calculator" in timeline and "Step 3" in timeline
    assert "55 years." in answer and "Show the verified answer" in answer
    assert "Jørn Utzon" in side and "backed by" in side and "Sydney Opera House" in side
    assert "Sources read" in score and "3 of 3" in score
    assert "About fifty years" in plain


def test_the_real_app_escapes_hostile_text_that_a_tool_result_and_the_model_put_in(served_app):
    turns = [{"reasoning": "<script>alert(1)</script>", "tool_calls": [call("wikipedia_page", title="Jørn Utzon")]},
             {"content": "<img src=x onerror=alert(2)> Utzon was born in 1918."}]
    with FakeOpenAI(turns) as f:
        out = served_app.submit(*run_args("Hostile text check question here", f.url), api_name="/run_research").result(timeout=120)
    html_all = " ".join(str(x) for x in out)
    assert "<script>" not in html_all and "<img src=x" not in html_all and "&lt;script&gt;" in html_all


def test_an_unreachable_server_shows_a_plain_notice_through_the_app(served_app):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    out = served_app.submit(*run_args("A question that needs a server", f"http://127.0.0.1:{port}/v1"), api_name="/run_research").result(timeout=60)
    assert "could not be reached" in out[0] and f"127.0.0.1:{port}" in out[0] and "Traceback" not in " ".join(str(x) for x in out)


def test_a_cut_off_run_shows_continue_and_resuming_it_finishes(served_app):
    turns = [{"reasoning": "long thought", "finish_reason": "length"}, {"content": "Answer after more room."}]
    with FakeOpenAI(turns) as f:
        first = served_app.submit(*run_args("A question with a long thought needed", f.url), api_name="/run_research").result(timeout=60)
        assert "thinking budget ran out" in first[0].lower()
        second = served_app.submit(api_name="/resume_research").result(timeout=60)
    assert "Answer after more room." in second[3]


@pytest.mark.skipif(not sb.self_check()["ok"], reason="this kernel cannot enforce the sandbox")
def test_a_lab_run_through_the_real_app_shows_the_rounds(served_app):
    ref = ("def count_pairs(nums, target):\n    seen = {}\n    total = 0\n    for x in nums:\n        total += seen.get(target - x, 0)\n        seen[x] = seen.get(x, 0) + 1\n    return total\n")
    turns = [{"reasoning": "dictionary approach", "tool_calls": [call("submit_solution", code=ref)]}, {"content": "Round 1 scored full marks."}]
    with FakeOpenAI(turns) as f:
        lab_prompt = missions.get("lab-count_pairs")["prompt"]
        out = served_app.submit("lab-count_pairs", lab_prompt, f.url, "", "Offline corpus (no network)", 6, 60000, 30, 2048, 0.6, 0.95, False, api_name="/run_lab").result(timeout=180)
    assert "<svg" in out[4] and "Round 1" in out[4] and "submit_solution" in out[2]


def test_the_page_has_the_three_tabs_the_api_names_and_the_how_to_read_text():
    demo = app.build_app()
    names = {fn.api_name for fn in demo.fns.values() if fn.api_name}
    assert {"run_research", "run_lab", "resume_research", "resume_lab"} <= names
    assert "not a chat box" in app.HEADER and "plain chat" in app.ABOUT.lower() and "string matching" in app.ABOUT.lower()


def test_a_scorecard_left_over_from_before_a_resume_is_not_shown_while_the_run_is_running_again():
    agent = {"steps": 1, "completion_tokens": 4096, "seconds": 200.0, "sources": 0, "tool_calls": 0, "answer": "", "claims": None, "stop_reason": "length"}
    r = app.render(snap(running=True, agent=agent, final=None))
    assert r["score"] == "" and r["continue_label"] == ""


def test_continue_is_not_offered_while_running_even_if_an_old_final_is_still_in_the_snapshot():
    final = {"stop_reason": "length", "detail": "", "answer": "", "steps": 1, "completion_tokens": 1, "seconds": 1.0, "sources": 0}
    assert app.render(snap(running=True, final=final, can_resume=True))["continue_label"] == ""


def test_continue_is_not_offered_when_the_run_cannot_be_resumed():
    final = {"stop_reason": "step_cap", "detail": "", "answer": "", "steps": 12, "completion_tokens": 1, "seconds": 1.0, "sources": 0}
    agent = {"steps": 12, "completion_tokens": 1, "seconds": 1.0, "sources": 0, "tool_calls": 0, "answer": "", "claims": None, "stop_reason": "step_cap"}
    assert app.render(snap(running=False, final=final, agent=agent, can_resume=False))["continue_label"] == ""


def test_the_lab_scorecard_has_no_expected_facts_row():
    final = {"stop_reason": "answered", "detail": "", "answer": "code", "steps": 2, "completion_tokens": 1, "seconds": 1.0, "sources": 0}
    agent = {"steps": 2, "completion_tokens": 1, "seconds": 1.0, "sources": 0, "tool_calls": 1, "answer": "code", "claims": None, "stop_reason": "answered", "lab_score": 0.9}
    r = app.render(snap(mission=missions.get("lab-count_pairs"), running=False, final=final, answer="code", agent=agent), lab_page=True)
    assert "expected facts" not in r["score"].lower() and "Lab score" in r["score"]


def test_the_tools_radio_refreshes_the_header_badge_when_it_changes():
    """Found on the first real-model run: the badge said "live web tools" while the Offline corpus radio was selected, because only the
    page load and the Check button refreshed it. Choosing a mode must refresh it too."""
    demo = app.build_app("http://127.0.0.1:1/v1")
    cfg = demo.get_config_file()
    components = {c["id"]: c for c in cfg["components"]}
    radios = [cid for cid, c in components.items() if c["type"] == "radio" and "Offline corpus (no network)" in str(c["props"].get("choices"))]
    assert len(radios) == 1
    refreshed = []
    for dep in cfg["dependencies"]:
        triggers = [(t[0], t[1]) for t in dep.get("targets", [])]
        if (radios[0], "change") in triggers or (radios[0], "select") in triggers:
            fn = demo.fns[dep["id"]].fn
            refreshed.append(getattr(fn, "__name__", ""))
    assert "check_server" in refreshed, refreshed
