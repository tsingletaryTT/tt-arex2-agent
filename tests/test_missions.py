# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""The preset missions (arex2_agent/missions.py and data/): verified facts, an offline corpus that can answer them, and the lab tasks."""
from __future__ import annotations

import json
import re

from arex2_agent import lab, missions, tools


def test_the_data_loads_and_every_mission_has_the_required_fields():
    ms = missions.all_missions()
    assert len(ms) >= 6
    ids = [m["id"] for m in ms]
    assert len(ids) == len(set(ids))
    for m in ms:
        assert m["kind"] in ("research", "compute", "lab")
        assert m["title"].strip() and m["prompt"].strip() and m["expected"].strip()
        assert isinstance(m["facts"], list) and m["facts"] and all(isinstance(f, str) and f for f in m["facts"])
        assert isinstance(m["needs"], list) and m["needs"]


def test_every_research_and_compute_mission_carries_the_quotes_it_was_verified_against_and_the_date():
    for m in missions.all_missions():
        if m["kind"] == "lab":
            continue
        v = m["verified"]
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", v["read"])
        assert v["quotes"] and all(q["quote"].strip() for q in v["quotes"])
        if m["kind"] == "research":
            assert all(q["url"].startswith("https://") for q in v["quotes"])


def test_each_verified_quote_supports_a_fact_the_mission_expects():
    for m in missions.all_missions():
        if m["kind"] != "research":
            continue
        blob = " ".join(q["quote"] for q in m["verified"]["quotes"]).replace(",", "")
        assert sum(1 for f in m["facts"] if f.replace(",", "") in blob) >= 1, m["id"]


def test_a_prompt_never_contains_the_answer_it_is_checked_against():
    for m in missions.all_missions():
        if m["kind"] == "lab":
            continue
        for f in m["facts"]:
            assert not re.search(rf"\b{re.escape(f)}\b", m["prompt"]), (m["id"], f)


def test_the_offline_corpus_holds_every_fact_a_research_mission_expects_that_is_read_and_not_derived():
    text = " ".join(d["text"] for d in missions.load_corpus()).replace(",", "")
    for m in missions.all_missions():
        if m["kind"] != "research":
            continue
        assert set(m["derived"]) <= set(m["facts"])
        for f in m["facts"]:
            if f not in m["derived"]:
                assert f.replace(",", "") in text, (m["id"], f)


def test_derived_facts_are_the_ones_the_agent_has_to_work_out():
    for m in missions.all_missions():
        if m["kind"] == "research":
            assert m["derived"], m["id"]
            assert "calculator" in m["needs"] or m["id"] == "attention-authors"


def test_the_corpus_entries_have_an_address_a_kind_a_licence_and_a_reasonable_size():
    corpus = missions.load_corpus()
    assert 8 <= len(corpus) <= 30
    for d in corpus:
        assert d["url"].startswith("https://") and d["kind"] in ("wikipedia", "arxiv") and d["license"] and d["text"].strip() and d["title"].strip()
        assert len(d["text"]) <= 3000
    assert sum(len(d["text"]) for d in corpus) < 60000
    assert len({d["id"] for d in corpus}) == len(corpus)


def test_the_offline_toolbox_answers_a_mission_from_the_corpus_without_the_network():
    def boom(*a, **k):
        raise AssertionError("network used")
    tb = missions.offline_toolbox(http=boom)
    r = tb.run("wikipedia_page", json.dumps({"title": "Jørn Utzon"}))
    assert r.ok and "1918" in r.text
    s = tb.run("wikipedia_search", json.dumps({"query": "Sydney Opera House opened"}))
    assert s.ok and "Sydney Opera House" in s.text
    a = tb.run("arxiv_search", json.dumps({"query": "attention is all you need"}))
    assert a.ok and "Polosukhin" in a.text and "2017" in a.text


def test_the_compute_mission_expected_answer_is_the_real_digit_sum():
    m = next(m for m in missions.all_missions() if m["id"] == "digits-2-1000")
    assert str(sum(int(c) for c in str(2 ** 1000))) in m["expected"] and "run_python" in m["needs"]


def test_the_lab_tasks_are_offered_as_missions_with_the_lab_text_and_their_task_id():
    labs = [m for m in missions.all_missions() if m["kind"] == "lab"]
    assert {m["task"] for m in labs} == set(lab.TASKS)
    for m in labs:
        assert m["prompt"] == lab.mission_text(lab.TASKS[m["task"]]) and "submit_solution" in m["needs"]


def test_missions_of_each_kind_can_be_listed_for_the_pickers():
    assert {m["kind"] for m in missions.by_kind("research")} == {"research"} and len(missions.by_kind("research")) >= 4
    assert len(missions.by_kind("lab")) == len(lab.TASKS)


def test_the_expected_everest_answer_states_the_date_the_heights_were_read():
    m = next(m for m in missions.all_missions() if m["id"] == "everest-k2")
    assert "as Wikipedia states them on" in m["expected"] and m["verified"].get("note")


def test_get_returns_a_mission_by_id_and_raises_for_an_unknown_one():
    import pytest
    assert missions.get("opera-utzon")["kind"] == "research"
    with pytest.raises(KeyError):
        missions.get("nope")


def test_loading_returns_copies_so_a_caller_cannot_change_the_data_for_the_next_one():
    a = missions.all_missions()
    a[0]["prompt"] = "changed"
    assert missions.all_missions()[0]["prompt"] != "changed"
    c = missions.load_corpus()
    c[0]["text"] = "changed"
    assert missions.load_corpus()[0]["text"] != "changed"


def test_the_offline_toolbox_is_in_offline_mode_and_ignores_search_keys():
    tb = missions.offline_toolbox()
    assert tb.mode == "offline" and "web_search" not in tb.names()
