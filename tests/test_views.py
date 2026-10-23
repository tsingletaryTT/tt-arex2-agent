# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""The page fragments (arex2_agent/views.py): everything that came from the model or from a tool is escaped, and nothing claims more than was measured."""
from __future__ import annotations

import re

import pytest

from arex2_agent import evidence, lab, views
from arex2_agent.code_sandbox import Outcome
from arex2_agent.tools import Source

BAD = '<script>alert(1)</script><img src=x onerror=alert(2)>"\'&'


def step(**kw):
    base = {"n": 1, "reasoning": "", "content": "", "tools": [], "finish_reason": "stop", "seconds": 1.0, "completion_tokens": 10, "live": False}
    base.update(kw)
    return base


def tool(**kw):
    base = {"name": "calculator", "arguments": '{"expression": "1+1"}', "ok": True, "text": "1+1 = 2", "summary": "1+1", "seconds": 0.01, "executed": True, "sources": []}
    base.update(kw)
    return base


# ---- escaping ------------------------------------------------------------------------------------------------------------------

def test_reasoning_content_and_tool_text_from_outside_are_escaped_in_the_timeline():
    out = views.timeline_html([step(reasoning=BAD, content=BAD, tools=[tool(text=BAD, name="fetch_url", arguments=BAD)])])
    assert "<script" not in out and "<img" not in out and "&lt;script&gt;" in out
    assert "onerror=alert(2)" not in out.replace("&lt;img src=x onerror=alert(2)&gt;", "")


def test_a_tool_name_the_model_made_up_is_escaped():
    out = views.timeline_html([step(tools=[tool(name='"><script>x</script>')])])
    assert "<script" not in out


def test_only_http_and_https_addresses_become_links_and_they_open_safely():
    assert 'href="https://example.org/a?b=1&amp;c=2"' in views.safe_link("https://example.org/a?b=1&c=2", "label")
    assert 'rel="noopener noreferrer"' in views.safe_link("https://example.org/", "x") and 'target="_blank"' in views.safe_link("https://example.org/", "x")
    for bad in ["javascript:alert(1)", "data:text/html,<script>1</script>", "vbscript:x", "file:///etc/passwd", "//evil.example/x", "JaVaScRiPt:alert(1)", " javascript:alert(1)"]:
        out = views.safe_link(bad, "label")
        assert "<a " not in out and "href" not in out, bad


def test_a_link_label_is_escaped():
    assert "<script" not in views.safe_link("https://example.org/", BAD)


def test_the_evidence_board_escapes_titles_quotes_and_addresses():
    src = Source('https://example.org/"><script>x</script>', BAD, "page", BAD)
    out = views.evidence_html([src], [], {"opened": 0, "never_opened": [BAD]})
    assert "<script" not in out and "<img" not in out


def test_claim_text_and_evidence_are_escaped():
    c = evidence.Claim(BAD, "unbacked", missing_numbers=[BAD])
    assert "<script" not in views.evidence_html([], [c], {"opened": 0, "never_opened": []})


def test_the_final_answer_is_shown_as_text_with_line_breaks_not_markup():
    out = views.answer_html("Line one\n<b>bold</b>\nhttps://example.org/page", [])
    assert "<b>" not in out and "&lt;b&gt;" in out and "<br" in out


def test_error_and_status_texts_are_escaped():
    assert "<script" not in views.notice_html("error", BAD)
    assert "<script" not in views.server_badge_html(BAD, BAD, "online")


# ---- the timeline ------------------------------------------------------------------------------------------------------------------

def test_each_step_shows_its_number_time_and_tokens_and_the_tool_it_called():
    out = views.timeline_html([step(n=1, reasoning="think", seconds=41.2, completion_tokens=612, tools=[tool()]), step(n=2, content="Done.", seconds=3.0, completion_tokens=40)])
    assert "Step 1" in out and "Step 2" in out and "41.2 s" in out and "612 tokens" in out and "calculator" in out and "1+1 = 2" in out


def test_a_live_step_shows_it_is_still_thinking_and_keeps_only_the_tail_of_a_long_thought_open():
    long = "word " * 2000
    out = views.timeline_html([step(n=1, reasoning=long, live=True)])
    assert "thinking" in out.lower() and len(out) < 6000


def test_a_failed_tool_call_is_marked_and_a_repeated_call_is_marked_as_not_run():
    out = views.timeline_html([step(tools=[tool(ok=False, text="calculator: division by zero.", summary="calculator refused"),
                                           tool(ok=False, executed=False, text="You already made this exact call")])])
    assert "failed" in out.lower() and "not run" in out.lower()


def test_tool_results_are_cut_to_a_snippet_in_the_timeline_with_the_full_length_stated():
    out = views.timeline_html([step(tools=[tool(text="x" * 5000)])])
    assert "5000 characters" in out and len(out) < 4000


def test_arguments_are_shown_as_readable_json():
    out = views.timeline_html([step(tools=[tool(arguments='{"query":"Ada Lovelace","limit":3}')])])
    assert "Ada Lovelace" in out and "limit" in out


def test_an_empty_timeline_says_nothing_has_run_yet():
    assert "no steps yet" in views.timeline_html([]).lower()


# ---- budgets ------------------------------------------------------------------------------------------------------------------------

def test_meters_show_used_over_limit_and_a_percentage_and_turn_warning_near_the_limit():
    out = views.meters_html({"max_steps": 12, "max_tokens": 60000, "max_seconds": 1800.0, "step_tokens": 4096}, steps=3, tokens=45000, seconds=60.0)
    assert "3 / 12" in out and "45,000 / 60,000" in out and "75%" in out and "warn" in out
    assert "60 s / 1,800 s" in out or "1 min / 30 min" in out


def test_a_meter_over_its_limit_is_capped_at_full_width_and_marked_over():
    out = views.meters_html({"max_steps": 2, "max_tokens": 10, "max_seconds": 5.0, "step_tokens": 5}, steps=5, tokens=500, seconds=50.0)
    assert "width:100%" in out and "over" in out
    assert "width:250" not in out


# ---- the evidence board --------------------------------------------------------------------------------------------------------------

OPERA = Source("https://en.wikipedia.org/wiki/Sydney_Opera_House", "Sydney Opera House", "page", "It was formally opened on 20 October 1973 by Queen Elizabeth II. " * 5)


def test_the_board_lists_each_source_with_its_kind_and_a_quoted_snippet():
    snip = Source("https://example.org/s", "A search hit", "snippet", "short hit text")
    out = views.evidence_html([OPERA, snip], [], {"opened": 0, "never_opened": []})
    assert "Sydney Opera House" in out and "read in full" in out and "search snippet only" in out and "formally opened on 20 October 1973" in out


def test_claims_are_marked_backed_computed_or_not_backed_with_the_source_named():
    claims = [evidence.Claim("The opera house opened in 1973.", "backed", OPERA.url, OPERA.title, "page", "formally opened on 20 October 1973"),
              evidence.Claim("That is 55 years.", "computed", OPERA.url, OPERA.title, "page", "x"),
              evidence.Claim("It cost a billion.", "unbacked", missing_numbers=["1"])]
    out = views.evidence_html([OPERA], claims, {"opened": 1, "never_opened": []})
    assert "backed by" in out and "computed" in out.lower() and "not backed" in out.lower() and "Sydney Opera House" in out
    assert "1 of 3" in out or "1 backed" in out


def test_a_cited_address_the_agent_never_opened_is_called_out():
    out = views.evidence_html([], [], {"opened": 0, "never_opened": ["https://example.com/made-up"]})
    assert "never opened" in out.lower() and "https://example.com/made-up" in out


def test_the_board_says_what_the_check_is_and_is_not():
    out = views.evidence_html([OPERA], [], {"opened": 0, "never_opened": []})
    assert "matches words and numbers" in out and "does not" in out


def test_an_empty_board_says_the_agent_has_read_nothing_yet():
    assert "read nothing" in views.evidence_html([], [], {"opened": 0, "never_opened": []}).lower()


# ---- the scorecard -------------------------------------------------------------------------------------------------------------------

AGENT = {"stop_reason": "answered", "steps": 6, "completion_tokens": 1800, "seconds": 190.0, "sources": 3, "tool_calls": 5, "answer": "Born 1918, opened 1973: 55 years.",
         "claims": {"total": 2, "backed": 1, "computed": 1, "unbacked": 0}}
PLAIN = {"stop_reason": "answered", "steps": 1, "completion_tokens": 400, "seconds": 30.0, "sources": 0, "tool_calls": 0, "answer": "About 55 years.", "claims": None}


def test_the_scorecard_compares_sources_steps_tokens_and_time_in_plain_words():
    out = views.scorecard_html(AGENT, PLAIN, ["1918", "1973", "55"])
    for needle in ("Sources read", "Claims backed", "Steps", "Tokens", "Time", "190 s", "30 s", "Agent", "Plain chat"):
        assert needle in out


def test_the_scorecard_counts_expected_facts_found_in_each_answer_as_a_string_match():
    out = views.scorecard_html(AGENT, PLAIN, ["1918", "1973", "55"])
    assert "3 of 3" in out and "1 of 3" in out and "string match" in out.lower()


def test_without_expected_facts_the_row_is_left_out():
    assert "expected facts" not in views.scorecard_html(AGENT, PLAIN, None).lower()


def test_plain_chat_is_shown_as_having_no_sources_to_check_not_as_wrong():
    out = views.scorecard_html(AGENT, PLAIN, None)
    assert "no sources" in out.lower() and "wrong" not in out.lower() and "worse" not in out.lower() and "better" not in out.lower()


def test_a_cut_off_plain_chat_or_agent_shows_its_stop_reason_in_words():
    cut = dict(PLAIN, stop_reason="length", answer="")
    out = views.scorecard_html(AGENT, cut, None)
    assert "thinking budget ran out" in out.lower()


def test_the_scorecard_works_with_only_the_agent():
    out = views.scorecard_html(AGENT, None, ["1918"])
    assert "Agent" in out and "Plain chat" not in out


def test_the_headline_states_counts_not_quality():
    out = views.scorecard_html(AGENT, PLAIN, None)
    assert "opened 3 sources" in out and "5 tool calls" in out


# ---- the lab ----------------------------------------------------------------------------------------------------------------------------

def make_round(n, score, large=1.2, passed=9, total=9):
    r = lab.ScoreReport(score=score, correctness=passed / total, speed=0.4, passed=passed, total=total, large_seconds=large, text=f"Score {score:.2f}")
    return lab.Round(n, f"def f():\n    return {n}\n", r)


def test_the_rounds_table_lists_each_round_with_score_tests_and_large_runtime():
    out = views.rounds_html([make_round(1, 0.62, large=7.5, passed=8), make_round(2, 0.97, large=0.6)])
    assert "Round 1" in out and "Round 2" in out and "0.62" in out and "0.97" in out and "8 of 9" in out and "7.5 s" in out


def test_the_score_chart_is_an_svg_with_a_point_per_round_and_marks_the_best():
    out = views.rounds_html([make_round(1, 0.5), make_round(2, 0.9), make_round(3, 0.7)])
    assert out.count("<circle") == 3 and "<svg" in out and "best" in out.lower()


def test_a_single_round_still_draws_a_chart_and_zero_rounds_says_none_yet():
    assert "<svg" in views.rounds_html([make_round(1, 0.5)])
    assert "no submissions yet" in views.rounds_html([]).lower()


def test_the_code_of_the_best_round_is_shown_escaped():
    r = make_round(1, 0.9)
    r.code = BAD
    out = views.rounds_html([r])
    assert "<script" not in out and "&lt;script&gt;" in out


def test_the_lab_notes_say_the_score_is_defined_by_the_lab_and_not_a_quality_rating():
    out = views.rounds_html([make_round(1, 0.5)])
    assert "0.7" in out and "0.3" in out


# ---- notices and badges --------------------------------------------------------------------------------------------------------------

def test_the_server_badge_says_online_or_offline_corpus_and_names_the_model():
    on = views.server_badge_html("http://127.0.0.1:8000/v1", "AREX-2", "online")
    off = views.server_badge_html("http://127.0.0.1:8000/v1", "AREX-2", "offline")
    assert "AREX-2" in on and "offline corpus" not in on.lower() and "offline corpus" in off.lower()


def test_a_down_server_badge_says_so_plainly():
    assert "not reachable" in views.server_badge_html("http://127.0.0.1:8000/v1", None, "online").lower()


def test_each_stop_reason_has_a_plain_sentence_and_the_cut_off_one_offers_to_continue():
    for reason in ("answered", "step_cap", "token_cap", "time_cap", "length", "empty", "repeated_call", "bad_calls", "server_down", "cancelled"):
        assert views.STOP_TEXT[reason]
    assert "larger" in views.STOP_TEXT["length"].lower() and "thinking budget" in views.STOP_TEXT["length"].lower()


def test_the_final_notice_uses_the_detail_the_run_gave():
    out = views.final_notice_html({"stop_reason": "server_down", "detail": "Cannot reach the model server at 127.0.0.1:8000"})
    assert "127.0.0.1:8000" in out


def test_a_source_kind_with_a_quote_in_it_cannot_break_out_of_the_class_attribute():
    src = Source("https://example.org/", "T", 'page" onmouseover="alert(1)', "text")
    out = views.evidence_html([src], [], {"opened": 0, "never_opened": []})
    assert 'onmouseover="alert(1)"' not in out and "&quot;" in out


def test_a_claim_status_with_a_quote_in_it_cannot_break_out_of_the_class_attribute():
    c = evidence.Claim("text of the claim here", 'backed" onmouseover="alert(1)')
    out = views.evidence_html([], [c], {"opened": 0, "never_opened": []})
    assert 'onmouseover="alert(1)"' not in out


def test_a_notice_kind_with_markup_is_escaped():
    assert "<script" not in views.notice_html('x"><script>1</script>', "text")


def test_a_long_step_token_count_is_shown_with_a_thousands_separator():
    assert "1,800 tokens" in views.timeline_html([step(completion_tokens=1800)])


# ---- guards found by mutation ------------------------------------------------------------------------------------------------------

def test_an_uppercase_scheme_still_links_and_a_quote_in_the_address_does_not():
    assert "<a " in views.safe_link("HTTPS://EXAMPLE.ORG/x", "label")
    assert "<a " not in views.safe_link('https://x.example/"onmouseover="y', "label")
    assert "<a " not in views.safe_link("https://x.example/a b", "label")


def test_a_finished_step_with_a_long_thought_says_how_much_is_shown():
    out = views.timeline_html([step(reasoning="w " * 4000)])
    assert "showing the last 1,500 of 8,000 characters" in out


def test_a_repeated_call_gets_its_own_class_and_a_live_step_opens_its_reasoning():
    out = views.timeline_html([step(tools=[tool(ok=False, executed=False, text="again")])])
    assert 'class="tool skipped"' in out
    live = views.timeline_html([step(reasoning="thinking about it", live=True)])
    assert "<details open>" in live
    assert "<details open>" not in views.timeline_html([step(reasoning="thinking about it")])


def test_arguments_that_are_not_json_are_flagged_and_still_shown_escaped():
    out = views.timeline_html([step(tools=[tool(arguments='{"expression": <b>oops')])])
    assert "not valid JSON" in out and "&lt;b&gt;oops" in out


def test_a_meter_just_over_its_limit_is_marked_over_and_one_under_is_not():
    over = views.meters_html({"max_steps": 2, "max_tokens": 100, "max_seconds": 100.0, "step_tokens": 5}, steps=3, tokens=5, seconds=1.0)
    assert 'class="meter over"' in over and over.count("meter over") == 1
    under = views.meters_html({"max_steps": 2, "max_tokens": 100, "max_seconds": 100.0, "step_tokens": 5}, steps=2, tokens=5, seconds=1.0)
    assert "meter over" not in under


def test_a_long_source_text_is_quoted_only_in_part():
    src = Source("https://example.org/", "Long", "page", "word " * 2000)
    out = views.evidence_html([src], [], {"opened": 0, "never_opened": []})
    assert len(out) < 2500


def test_an_unbacked_claim_names_the_numbers_that_were_not_found():
    c = evidence.Claim("Opened in 1975 after long work.", "unbacked", missing_numbers=["1975"])
    assert "no 1975" in views.evidence_html([], [c], {"opened": 0, "never_opened": []})


def test_thousands_separators_in_an_answer_do_not_hide_an_expected_number():
    plain = dict(PLAIN, answer="It has 1,234 steps.")
    out = views.scorecard_html(dict(AGENT, answer="nothing"), plain, ["1234"])
    assert "1 of 1" in out and "0 of 1" in out


def test_the_plain_chat_column_says_it_has_no_sources_in_the_sources_row_and_no_sources_to_check_in_the_claims_row():
    out = views.scorecard_html(AGENT, PLAIN, None)
    assert "no sources (no tools)" in out and "no sources to check against" in out


def test_computed_claims_count_toward_the_backed_total_in_the_scorecard():
    assert "2 of 2" in views.scorecard_html(AGENT, PLAIN, None)


def test_the_expected_facts_note_says_it_is_not_a_grade():
    assert "not a grade" in views.scorecard_html(AGENT, PLAIN, ["1918"]).lower()


def test_the_chart_names_the_best_round_and_its_score():
    out = views.rounds_html([make_round(1, 0.5), make_round(2, 0.9), make_round(3, 0.7)])
    assert "best: round 2, 0.90" in out and out.count('class="best"') == 1


def test_the_lab_note_says_the_score_is_about_this_task():
    assert "measures this task" in views.rounds_html([make_round(1, 0.5)])


def test_a_finished_run_gets_the_good_notice_and_a_failed_one_the_error_notice():
    assert 'class="notice good"' in views.final_notice_html({"stop_reason": "answered"})
    assert 'class="notice error"' in views.final_notice_html({"stop_reason": "server_down", "detail": "x"})
    assert 'class="notice warn"' in views.final_notice_html({"stop_reason": "length"})


def test_a_claim_backed_by_two_sources_names_both():
    c = evidence.Claim("Utzon was born in 1918, 55 years before the opening in 1973.", "computed", "https://a/", "Sydney Opera House", "page", "x",
                       also_from=[("https://b/", "Jørn Utzon")])
    out = views.evidence_html([], [c], {"opened": 0, "never_opened": []})
    assert "Sydney Opera House and Jørn Utzon" in out


def test_the_second_source_name_is_escaped():
    c = evidence.Claim("A claim that has enough words in it to count.", "backed", "https://a/", "A", "page", "x", also_from=[("https://b/", BAD)])
    assert "<script" not in views.evidence_html([], [c], {"opened": 0, "never_opened": []})


# ---- the plain chat panel, the verified answer, the lab score and Continue --------------------------------------------------------------------

def test_the_plain_chat_panel_says_what_it_is_and_streams_the_live_text_escaped():
    out = views.plain_html(None, {"reasoning": "thinking <b>x</b>", "content": ""}, running=True)
    assert "same model, no tools, one shot" in out.lower() and "<b>x</b>" not in out and "&lt;b&gt;x" in out and "answering" in out.lower()
    streaming = views.plain_html(None, {"reasoning": "", "content": BAD}, running=True)
    assert "<script" not in streaming and "<img" not in streaming and "&lt;script&gt;" in streaming


def test_a_finished_plain_chat_shows_its_answer_and_how_it_ended():
    out = views.plain_html({"answer": "About 55 years.", "reasoning": "hmm", "stop_reason": "answered", "completion_tokens": 40, "seconds": 12.0}, {"reasoning": "", "content": ""}, running=False)
    assert "About 55 years." in out and "Finished with an answer" in out


def test_a_cut_off_plain_chat_says_the_thinking_budget_ran_out_and_shows_no_answer():
    out = views.plain_html({"answer": "", "reasoning": "long thought", "stop_reason": "length", "completion_tokens": 4096, "seconds": 300.0}, {"reasoning": "", "content": ""}, running=False)
    assert "thinking budget ran out" in out.lower()


def test_no_plain_chat_and_not_running_shows_nothing():
    assert views.plain_html(None, {"reasoning": "", "content": ""}, running=False) == ""


def test_the_verified_answer_is_collapsed_and_shows_the_quotes_the_date_and_links():
    m = {"expected": "55 years (born 1918, opened 1973)", "verified": {"read": "2026-10-22", "quotes": [{"url": "https://en.wikipedia.org/wiki/Sydney_Opera_House", "quote": "formally opened ... 1973"}]}}
    out = views.expected_html(m)
    assert "<details>" in out and "<details open" not in out and "55 years" in out and "2026-10-22" in out and 'href="https://en.wikipedia.org/wiki/Sydney_Opera_House"' in out
    assert "formally opened" in out


def test_the_verified_answer_escapes_everything_and_a_note_is_shown():
    m = {"expected": BAD, "verified": {"read": BAD, "quotes": [{"url": "javascript:alert(1)", "quote": BAD}], "note": BAD}}
    out = views.expected_html(m)
    assert "<script" not in out and "<img" not in out and "href=\"javascript" not in out


def test_a_mission_with_no_verification_shows_no_reveal_for_a_custom_mission():
    assert views.expected_html({"id": "custom", "expected": "", "verified": {}}) == ""


def test_the_scorecard_gets_a_lab_score_row_when_there_is_one():
    out = views.scorecard_html(dict(AGENT, lab_score=0.97), dict(PLAIN, lab_score=0.31), None)
    assert "Lab score" in out and "0.97" in out and "0.31" in out
    assert "Lab score" not in views.scorecard_html(AGENT, PLAIN, None)


def test_a_plain_chat_with_no_code_to_score_shows_a_dash_for_its_lab_score():
    out = views.scorecard_html(dict(AGENT, lab_score=0.9), dict(PLAIN, lab_score=None), None)
    assert "Lab score" in out and "no code to score" in out


def test_continue_picks_what_to_extend_for_each_stop_reason():
    s = {"step_tokens": 4096, "max_steps": 12, "max_tokens": 60000, "max_seconds": 1800.0}
    assert views.continue_kwargs("length", s) == {"step_tokens": 8192}
    assert views.continue_kwargs("step_cap", s) == {"extra_steps": 8}
    assert views.continue_kwargs("token_cap", s) == {"extra_tokens": 30000}
    assert views.continue_kwargs("time_cap", s) == {"extra_seconds": 900.0}
    assert views.continue_kwargs("answered", s) is None and views.continue_kwargs("server_down", s) is None and views.continue_kwargs("cancelled", s) is None


def test_the_larger_limit_is_capped_and_a_limit_already_at_the_cap_offers_nothing():
    assert views.continue_kwargs("length", {"step_tokens": 20000}) == {"step_tokens": 32768}
    assert views.continue_kwargs("length", {"step_tokens": 32768}) is None


def test_the_continue_label_says_what_it_will_do():
    assert "larger limit" in views.continue_label("length", {"step_tokens": 4096}).lower() and "8,192" in views.continue_label("length", {"step_tokens": 4096})
    assert "8 more steps" in views.continue_label("step_cap", {})
    assert views.continue_label("answered", {}) == ""


def test_a_finished_plain_answer_with_markup_is_escaped():
    out = views.plain_html({"answer": BAD, "reasoning": BAD, "stop_reason": "answered", "completion_tokens": 1, "seconds": 1.0}, {"reasoning": "", "content": ""}, running=False)
    assert "<script" not in out and "<img" not in out and "&lt;script&gt;" in out


def test_an_expected_answer_with_no_quotes_behind_it_is_not_offered():
    assert views.expected_html({"expected": "55", "verified": {"read": "2026-10-22", "quotes": []}}) == ""


def test_the_verification_note_is_shown_when_there_is_one():
    m = {"expected": "x", "verified": {"read": "2026-10-22", "quotes": [{"url": "", "quote": "q"}], "note": "Wikipedia figures can change."}}
    assert "Wikipedia figures can change." in views.expected_html(m)
