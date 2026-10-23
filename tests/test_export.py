# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""The run export (arex2_agent/export.py): what "Download this run" writes. A run is worth keeping when it holds every step and nothing private."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from arex2_agent import export, lab, missions, session, tools  # noqa: E402
from tests.fake_openai import FakeOpenAI  # noqa: E402
from tests.test_session import OPERA, UTZON, call, fake_http, research, run, settings  # noqa: E402,F401


def scripted_run(**settings_kw):
    turns = [{"reasoning": "Look up the architect.", "tool_calls": [call("wikipedia_page", title="Jørn Utzon"), call("wikipedia_page", title="Sydney Opera House")]},
             {"tool_calls": [call("calculator", expression="1973 - 1918")]},
             {"content": "Answer: 55 years.\nJørn Utzon was born on 9 April 1918 in Copenhagen. The Sydney Opera House was formally opened on 20 October 1973.\n\nSources:\n- https://en.wikipedia.org/wiki/Sydney_Opera_House"}]
    with FakeOpenAI(turns, plain_turns=[{"content": "About fifty years."}]) as f:
        s = settings(f, compare=True, **settings_kw)
        sess = session.Session(research(), s, toolbox=tools.Toolbox(env={}, http=fake_http))
        return run(sess)[-1]


def test_a_finished_research_run_exports_the_mission_every_step_the_answer_the_evidence_and_the_scorecard():
    doc = export.export_run(scripted_run())
    assert doc["format"] == "arex2-research-lab-run" and doc["version"] == 1
    assert doc["mission"]["prompt"].startswith("How old was the architect") and doc["mission"]["kind"] == "research"
    assert [s["n"] for s in doc["steps"]] == [1, 2, 3]
    first = doc["steps"][0]
    assert first["reasoning"] == "Look up the architect." and [t["name"] for t in first["tools"]] == ["wikipedia_page", "wikipedia_page"]
    t0 = first["tools"][0]
    assert json.loads(t0["arguments"]) == {"title": "Jørn Utzon"} and "Danish architect" in t0["text"] and t0["ok"] is True and t0["seconds"] >= 0
    assert first["seconds"] >= 0 and "completion_tokens" in first
    assert "55 years." in doc["answer"] and doc["final"]["stop_reason"] == "answered"
    assert doc["sources"] and any("Utzon" in json.dumps(x, ensure_ascii=False) for x in doc["sources"])
    assert doc["claims"] and doc["scorecard"]["agent"]["steps"] == 3 and doc["scorecard"]["plain"]["answer"] == "About fifty years."


def test_the_settings_are_exported_without_a_credential_in_the_address_and_without_a_model_path():
    sn = scripted_run()
    sn["settings"].base_url = "http://user:hunter2@10.1.2.3:8103/v1"
    sn["settings"].model = "/home/someone/runs/model-dir"
    doc = export.export_run(sn)
    text = json.dumps(doc)
    assert "hunter2" not in text and "user:" not in text and "someone" not in text and "/home/" not in text
    s = doc["settings"]
    assert s["server"] == "http://10.1.2.3:8103" and s["model"] == "model-dir"
    assert (s["max_steps"], s["temperature"], s["top_p"], s["mode"], s["compare"]) == (12, 0.6, 0.95, "online", True)


def test_secrets_and_private_paths_in_text_from_the_model_or_a_tool_never_reach_the_file():
    hostile = ("Authorization: Bearer abc123SECRET and again Bearer zzz999TOKEN. hf_" + "A" * 30 + " sk-" + "b" * 24 + " ghp_" + "c" * 30 +
               " SERPER_API_KEY=serper-secret-value JINA_API_KEY: jina-secret-value password=hunter2 at /home/alice/work/run.py and /Users/bob/x")
    sn = scripted_run()
    sn["steps"][0]["tools"][0]["text"] = hostile
    sn["steps"][0]["reasoning"] = hostile
    sn["answer"] = hostile
    text = json.dumps(export.export_run(sn), ensure_ascii=False)
    for bad in ("Authorization", "abc123SECRET", "zzz999TOKEN", "hf_" + "A" * 30, "b" * 24, "c" * 30, "serper-secret-value", "jina-secret-value", "hunter2", "alice", "bob", "/home/", "/Users/"):
        assert bad not in text, bad


def test_ordinary_text_survives_the_redaction_unchanged():
    sn = scripted_run()
    doc = export.export_run(sn)
    assert "Danish architect" in json.dumps(doc, ensure_ascii=False) and "9 April 1918" in json.dumps(doc, ensure_ascii=False)
    assert export.redact("The token budget is 4096 tokens and the key idea is dynamic programming.") == "The token budget is 4096 tokens and the key idea is dynamic programming."


def test_a_lab_run_exports_the_rounds_with_code_and_score():
    rnd = lab.Round(1, "def f(): pass", lab.ScoreReport(score=0.6, correctness=0.9, speed=0.0, passed=9, total=10, large_seconds=5.0, text="Score 0.60"))
    sn = scripted_run()
    sn["mission"] = missions.get("lab-count_pairs")
    sn["rounds"] = [rnd]
    doc = export.export_run(sn)
    r = doc["rounds"][0]
    assert r["n"] == 1 and r["code"] == "def f(): pass" and r["report"]["score"] == 0.6 and r["report"]["passed"] == 9 and doc["mission"]["kind"] == "lab"


def test_a_run_still_going_or_not_started_exports_what_exists_and_says_so():
    sn = scripted_run()
    sn["running"], sn["final"] = True, None
    doc = export.export_run(sn)
    assert doc["running"] is True and doc["final"] is None
    assert export.export_run(None) is None


def test_the_file_is_written_as_utf8_json_named_for_the_mission(tmp_path):
    path = export.write_export(scripted_run(), tmp_path)
    assert path.parent == tmp_path and path.suffix == ".json" and path.name.startswith("arex2-run-custom-")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["mission"]["id"] == "custom" and "Jørn Utzon" in path.read_text(encoding="utf-8")
    assert export.write_export(None, tmp_path) is None



def test_old_exports_are_pruned_oldest_first_and_exactly_the_newest_few_stay(tmp_path):
    import os
    for i in range(export.KEEP + 5):
        p = tmp_path / f"arex2-run-custom-{i:04d}.json"
        p.write_text("{}")
        os.utime(p, (1_000_000 + i, 1_000_000 + i))
    other = tmp_path / "notes.txt"
    other.write_text("not an export")
    new = export.write_export(scripted_run(), tmp_path)
    left = sorted(q.name for q in tmp_path.glob("arex2-run-*.json"))
    assert len(left) == export.KEEP and new.name in left
    assert "arex2-run-custom-0000.json" not in left and f"arex2-run-custom-{export.KEEP + 4:04d}.json" in left     # the oldest went, the newest old ones stayed
    assert other.exists()                                                                                           # nothing else in the folder is touched


def test_an_object_that_is_not_plain_data_is_written_as_redacted_text():
    sn = scripted_run()
    sn["steps"][0]["tools"][0]["text"] = ValueError("Authorization: Bearer leaked-in-an-exception at /home/carol/x.py")
    sn["final"] = dict(sn["final"], detail=Path("/home/dave/run/out.txt"))
    text = json.dumps(export.export_run(sn), ensure_ascii=False)
    assert "leaked-in-an-exception" not in text and "carol" not in text and "dave" not in text and "Authorization" not in text


def test_a_number_that_is_not_a_number_is_written_as_null_so_the_file_is_valid_json():
    sn = scripted_run()
    sn["steps"][0]["seconds"] = float("nan")
    sn["steps"][1]["seconds"] = float("inf")
    doc = export.export_run(sn)
    assert doc["steps"][0]["seconds"] is None and doc["steps"][1]["seconds"] is None
    json.dumps(doc, allow_nan=False)                                                                               # a strict parser would refuse NaN and Infinity
