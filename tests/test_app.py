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


def test_the_page_has_the_api_names_the_header_and_the_how_to_read_text():
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


# ---- the layout: a control sits in the container whose scope it has (design review of 2026-10-23) --------------------------------------------

def _cfg():
    demo = app.build_app("http://127.0.0.1:1/v1")
    return demo, demo.get_config_file()


def _components(cfg):
    return {c["id"]: c for c in cfg["components"]}


def _tree(node, parents=()):
    yield node, parents
    for ch in node.get("children", []):
        yield from _tree(ch, parents + (node["id"],))


def _by(cfg, ctype, **props):
    return [c for c in cfg["components"] if c["type"] == ctype and all(c["props"].get(k) == v for k, v in props.items())]


def _text(c):
    p = c["props"]
    return str(p.get("value") or p.get("label") or "")


def _ancestors(cfg, cid):
    for node, parents in _tree(cfg["layout"]):
        if node["id"] == cid:
            return list(parents)
    raise AssertionError(f"component {cid} is not in the layout")


def _fn_name(demo, dep):
    return getattr(demo.fns[dep["id"]].fn, "__name__", "")


def test_check_the_server_sits_in_the_same_row_as_the_server_address():
    _, cfg = _cfg()
    comps = _components(cfg)
    check = [c for c in cfg["components"] if c["type"] == "button" and c["props"].get("value") == "Check the server"]
    addr = [c for c in cfg["components"] if c["type"] == "textbox" and c["props"].get("label") == "Model server address"]
    assert len(check) == 1 and len(addr) == 1
    rows = lambda cid: [a for a in _ancestors(cfg, cid) if comps.get(a, {}).get("type") == "row"]
    assert rows(check[0]["id"]) and rows(check[0]["id"])[-1] == rows(addr[0]["id"])[-1]


def test_the_settings_accordion_is_named_for_every_run_and_holds_connection_limits_and_sampling_groups():
    _, cfg = _cfg()
    comps = _components(cfg)
    acc = [c for c in cfg["components"] if c["type"] == "accordion" and c["props"].get("label") == "Settings for every run"]
    assert len(acc) == 1 and acc[0]["props"].get("open") is False
    assert not [c for c in cfg["components"] if c["type"] == "accordion" and c["props"].get("label") == "Model server and budgets"]
    group_of = {}
    for node, parents in _tree(cfg["layout"]):
        c = comps.get(node["id"], {})
        if c.get("type") == "markdown" and _text(c).strip("# *").strip() in ("Connection", "Limits", "Sampling"):
            group_of[_text(c).strip("# *").strip()] = (node["id"], parents[-1])        # the heading and the container it opens
    assert set(group_of) == {"Connection", "Limits", "Sampling"} and acc[0]["id"] in _ancestors(cfg, group_of["Connection"][0])

    def members(group):
        container = group_of[group][1]
        found = set()
        for node, parents in _tree(cfg["layout"]):
            if container in parents and node["id"] in comps:
                p = comps[node["id"]]["props"]
                found.add(p.get("label") or p.get("value"))
        return found
    assert {"Model server address", "Model name (blank: ask the server)", "Tools", "Check the server"} <= members("Connection")
    assert {"Steps", "Tokens generated in total", "Minutes"} <= members("Limits") and not ({"Temperature", "Top-p"} & members("Limits"))
    assert {"Tokens per reply, thinking included", "Temperature", "Top-p"} <= members("Sampling") and not ({"Steps", "Minutes"} & members("Sampling"))


def test_the_tools_mode_is_said_in_words_beside_run_and_follows_the_radio():
    assert app.tools_label("Live web tools") == "Tools: live web tools"
    assert app.tools_label("Offline corpus (no network)") == "Tools: offline corpus (no network)"
    assert app.tools_label(None) == "Tools: live web tools"
    demo, cfg = _cfg()
    comps = _components(cfg)
    radio = [c for c in cfg["components"] if c["type"] == "radio" and "Offline corpus (no network)" in str(c["props"].get("choices"))][0]["id"]
    labels = [c["id"] for c in cfg["components"] if c["type"] in ("html", "markdown") and _text(c).startswith("Tools: ")]
    assert len(labels) == 2                                                  # one beside each tab's Run button
    for lid in labels:                                                       # beside Run: in the row that holds a Run button
        row = [a for a in _ancestors(cfg, lid) if comps.get(a, {}).get("type") == "row"][-1]
        assert any(c["props"].get("value") == "Run" and row in _ancestors(cfg, c["id"]) for c in cfg["components"] if c["type"] == "button")
    followed = set()
    for dep in cfg["dependencies"]:
        if (radio, "change") in [(t[0], t[1]) for t in dep.get("targets", [])] and _fn_name(demo, dep) == "tools_label":
            followed |= set(dep["outputs"])
    assert set(labels) <= followed
    assert any((None, "load") != (0, "x") and _fn_name(demo, d) == "tools_label" and any(t[1] == "load" for t in d["targets"]) for d in cfg["dependencies"])


def test_each_tab_has_download_this_run_and_copy_answer_beside_the_status_and_they_wait_for_a_finished_run():
    demo, cfg = _cfg()
    comps = _components(cfg)
    downloads = [c for c in cfg["components"] if c["type"] == "downloadbutton"]
    copies = [c for c in cfg["components"] if c["type"] == "button" and c["props"].get("value") == "Copy answer"]
    assert len(downloads) == 2 and len(copies) == 2
    assert all(d["props"].get("value") is None or True for d in downloads)
    assert all(d["props"]["label"] == "Download this run" or d["props"].get("value") == "Download this run" for d in downloads)
    assert all(not d["props"].get("interactive", True) for d in downloads) and all(not c["props"].get("interactive", True) for c in copies)
    chained = [d for d in cfg["dependencies"] if _fn_name(demo, d) == "prepare_export"]
    assert len(chained) >= 4                                                  # after a run and after a resume, on each tab
    assert all(d.get("trigger_after") is not None for d in chained)


def test_prepare_export_enables_download_and_copy_for_a_finished_run_and_disables_them_for_none_or_a_running_one(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "EXPORT_DIR", tmp_path)
    from tests.test_export import scripted_run
    sess = type("S", (), {})()
    snap = scripted_run()
    sess.snapshot = lambda: snap
    dl, cp, text = app.prepare_export(sess)
    assert dl["interactive"] is True and dl["value"].endswith(".json") and Path(dl["value"]).exists() and cp["interactive"] is True and "55 years." in text
    assert json.loads(Path(dl["value"]).read_text(encoding="utf-8"))["answer"].startswith("Answer: 55 years.")
    dl0, cp0, text0 = app.prepare_export(None)
    assert dl0["interactive"] is False and cp0["interactive"] is False and text0 == ""
    snap2 = dict(snap, running=True, final=None, answer="")
    sess.snapshot = lambda: snap2
    dl1, cp1, _ = app.prepare_export(sess)
    assert dl1["interactive"] is False and cp1["interactive"] is False


def test_the_how_to_read_text_is_a_collapsed_accordion_at_the_bottom_and_not_a_tab():
    _, cfg = _cfg()
    comps = _components(cfg)
    tabs = [c["props"].get("label") for c in cfg["components"] if c["type"] == "tabitem"]
    assert tabs == ["Research missions", "Improvement lab"]
    acc = [c for c in cfg["components"] if c["type"] == "accordion" and c["props"].get("label") == "How to read this page"]
    assert len(acc) == 1 and acc[0]["props"].get("open") is False
    top = [n["id"] for n in cfg["layout"]["children"]]
    tabs_id = [c["id"] for c in cfg["components"] if c["type"] == "tabs"][0]
    assert top.index(acc[0]["id"]) > top.index(tabs_id) and top[-1] == acc[0]["id"]
    assert any("What you are looking at" in _text(c) for c in cfg["components"] if c["type"] == "markdown" and acc[0]["id"] in _ancestors(cfg, c["id"]))


def test_stop_starts_disabled_run_is_disabled_while_a_run_is_going_and_both_come_back_after_it_ends():
    demo, cfg = _cfg()
    stops = [c for c in cfg["components"] if c["type"] == "button" and c["props"].get("value") == "Stop"]
    assert len(stops) == 2 and all(c["props"].get("interactive") is False for c in stops)
    runs = [c for c in cfg["components"] if c["type"] == "button" and c["props"].get("value") == "Run"]
    lock, unlock = app.lock_run(), app.unlock_run()
    assert lock[0]["interactive"] is False and lock[1]["interactive"] is True and unlock[0]["interactive"] is True and unlock[1]["interactive"] is False
    deps = {d["id"]: d for d in cfg["dependencies"]}
    for api in ("run_research", "run_lab", "resume_research", "resume_lab"):
        main = [d for d in cfg["dependencies"] if d.get("api_name") == api]
        assert len(main) == 1, api
        before = deps[main[0]["trigger_after"]]
        assert _fn_name(demo, before) == "lock_run" and any(t[1] == "click" for t in before["targets"])
        after = [d for d in cfg["dependencies"] if d.get("trigger_after") == main[0]["id"] and _fn_name(demo, d) == "unlock_run"]
        assert len(after) == 1 and not after[0].get("trigger_only_on_success")          # a refused or failed run must free the buttons too
    assert {t[0] for d in cfg["dependencies"] if _fn_name(demo, d) == "lock_run" for t in d["targets"]} >= {c["id"] for c in runs}


def test_the_lab_dropdown_drops_the_repeated_prefix_in_its_labels_and_keeps_the_ids():
    labs = app.preset_choices(("lab",), strip_prefix="Improvement lab: ")
    assert labs and all(not label.startswith("Improvement lab:") for label, _ in labs[:-1]) and [v for _, v in labs] == [v for _, v in app.preset_choices(("lab",))]
    assert ("Fewest coins for an amount", "lab-min_coins") in labs
    assert any(label.startswith("Improvement lab:") for label, _ in app.preset_choices(("lab",)))     # the default keeps the full titles
    _, cfg = _cfg()
    dropdowns = [c for c in cfg["components"] if c["type"] == "dropdown" and c["props"].get("label") == "Mission"]
    lab_dd = [d for d in dropdowns if any(str(ch[1]).startswith("lab-") for ch in d["props"]["choices"])]
    res_dd = [d for d in dropdowns if d not in lab_dd]
    assert len(lab_dd) == 1 and all(not str(ch[0]).startswith("Improvement lab:") for ch in lab_dd[0]["props"]["choices"])
    assert len(res_dd) == 1 and any("architect" in str(ch[0]) for ch in res_dd[0]["props"]["choices"])



def test_copy_answer_works_on_a_page_served_over_plain_http_too_and_says_it_copied():
    """navigator.clipboard exists only on https and localhost; a page opened over the LAN falls back to a selected textarea."""
    assert "navigator.clipboard" in app.COPY_JS and "isSecureContext" in app.COPY_JS and "execCommand('copy')" in app.COPY_JS
    assert "return [text]" in app.COPY_JS
    import inspect
    assert "gr.Info" in inspect.getsource(app.copied)


def test_every_class_the_layout_names_has_a_rule_so_the_check_button_lines_up_with_its_input():
    _, cfg = _cfg()
    used = set()
    for c in cfg["components"]:
        for cls in str(c["props"].get("elem_classes") or "").replace("[", " ").replace("]", " ").replace("'", " ").replace(",", " ").split():
            used.add(cls)
    assert {"inline-action", "tools-note", "settings-group"} <= used
    for cls in ("inline-action", "tools-note", "settings-group"):
        assert f".{cls}" in app.views.CSS, cls
    assert "align-self:flex-end" in app.views.CSS.split(".inline-action", 1)[1].split("}", 1)[0]


def test_download_and_copy_are_compact_buttons_and_the_status_banner_takes_the_rest_of_the_row():
    _, cfg = _cfg()
    comps = _components(cfg)
    for label in ("Download this run", "Copy answer"):
        btn = [c for c in cfg["components"] if c["type"] in ("button", "downloadbutton") and (c["props"].get("value") == label or c["props"].get("label") == label)]
        assert len(btn) == 2 and all(b["props"].get("scale") == 0 for b in btn), label
    notices = [c for c in cfg["components"] if c["type"] == "html" and c["props"].get("scale") == 1]
    assert len(notices) == 2


def test_starting_a_run_locks_run_and_clears_the_files_of_the_last_run_on_that_tab():
    demo, cfg = _cfg()
    comps = _components(cfg)
    for api in ("run_research", "run_lab", "resume_research", "resume_lab"):
        main = [d for d in cfg["dependencies"] if d.get("api_name") == api][0]
        lock = {d["id"]: d for d in cfg["dependencies"]}[main["trigger_after"]]
        types = sorted(comps[o]["type"] for o in lock["outputs"])
        assert types == ["button", "button", "button", "downloadbutton"], (api, types)      # Run, Stop, Copy answer and Download this run


def test_a_finished_run_with_no_answer_offers_the_download_but_not_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "EXPORT_DIR", tmp_path)
    from tests.test_export import scripted_run
    snap = dict(scripted_run(), answer="", final={"stop_reason": "step_cap", "detail": "", "answer": "", "steps": 12, "completion_tokens": 1, "seconds": 1.0, "sources": 0})
    sess = type("S", (), {"snapshot": lambda self: snap})()
    dl, cp, text = app.prepare_export(sess)
    assert dl["interactive"] is True and cp["interactive"] is False and text == ""


def test_nothing_is_offered_while_a_run_is_going_even_when_an_older_final_is_still_in_the_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "EXPORT_DIR", tmp_path)
    from tests.test_export import scripted_run
    snap = dict(scripted_run(), running=True)                  # the earlier run's final is still there: the resumed run has not replaced it yet
    assert snap["final"]
    sess = type("S", (), {"snapshot": lambda self: snap})()
    dl, cp, text = app.prepare_export(sess)
    assert dl["interactive"] is False and cp["interactive"] is False and text == "" and not list(tmp_path.iterdir())


def test_the_exports_go_to_their_own_folder_and_not_the_bare_temporary_directory():
    import tempfile
    assert app.EXPORT_DIR != Path(tempfile.gettempdir()) and app.EXPORT_DIR.parent == Path(tempfile.gettempdir()) and "arex2" in app.EXPORT_DIR.name
