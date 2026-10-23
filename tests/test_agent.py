# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""The agent loop (arex2_agent/agent.py): think, call a tool, read the result, repeat, answer. Run against a scripted fake server."""
from __future__ import annotations

import json
import threading
import time

import pytest

from arex2_agent import agent, chat, tools
from tests.fake_openai import FakeOpenAI


def call(name, **args):
    return {"name": name, "arguments": json.dumps(args)}


def make(f, toolbox=None, budget=None, **kw):
    tb = toolbox or tools.Toolbox(env={}, http=lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    client = chat.ChatClient(f.url, model="m")
    return agent.AgentRun(client, tb, mission=kw.pop("mission", "What is 17*23?"), budget=budget or agent.Budget(), **kw)


def drain(run, **kw):
    return list(run.events(**kw))


def kinds(events):
    return [e["type"] for e in events]


# ---- the happy path ---------------------------------------------------------------------------------------------------------

def test_a_tool_call_then_an_answer_gives_two_steps_and_the_events_in_order():
    turns = [{"reasoning": "I should compute this.", "tool_calls": [call("calculator", expression="17 * 23")], "usage": {"prompt_tokens": 100, "completion_tokens": 20}},
             {"reasoning": "391.", "content": "17 * 23 = 391.", "usage": {"prompt_tokens": 140, "completion_tokens": 10}}]
    with FakeOpenAI(turns) as f:
        run = make(f)
        ev = drain(run)
    assert kinds(ev)[0] == "start" and kinds(ev)[-1] == "final"
    assert kinds([e for e in ev if e["type"] != "delta"]) == ["start", "step", "model", "tool", "step", "model", "final"]
    final = ev[-1]
    assert final["stop_reason"] == "answered" and final["answer"] == "17 * 23 = 391."
    tool_ev = next(e for e in ev if e["type"] == "tool")
    assert tool_ev["name"] == "calculator" and tool_ev["ok"] and "391" in tool_ev["text"] and tool_ev["n"] == 1
    assert run.completion_tokens == 30 and run.step_count == 2


def test_the_second_request_carries_the_tool_call_and_its_result_but_not_the_reasoning():
    turns = [{"reasoning": "SECRET THOUGHT", "tool_calls": [call("calculator", expression="2+2")]}, {"content": "4"}]
    with FakeOpenAI(turns) as f:
        drain(make(f))
        second = f.requests[1]["messages"]
    roles = [m["role"] for m in second]
    assert roles == ["system", "user", "assistant", "tool"]
    asst, tool = second[2], second[3]
    assert asst["tool_calls"][0]["function"]["name"] == "calculator" and tool["tool_call_id"] == asst["tool_calls"][0]["id"]
    assert "2+2 = 4" in tool["content"]
    assert "SECRET THOUGHT" not in json.dumps(second)


def test_the_request_has_the_system_prompt_the_mission_and_the_tools():
    with FakeOpenAI([{"content": "done"}]) as f:
        drain(make(f, mission="Find the thing."))
        body = f.requests[0]
    assert body["messages"][0]["role"] == "system" and "cite" in body["messages"][0]["content"].lower()
    assert body["messages"][1] == {"role": "user", "content": "Find the thing."}
    assert {t["function"]["name"] for t in body["tools"]} >= {"calculator", "wikipedia_search"}


def test_an_answer_with_no_tool_use_finishes_in_one_step():
    with FakeOpenAI([{"content": "Paris."}]) as f:
        run = make(f)
        ev = drain(run)
    assert ev[-1]["stop_reason"] == "answered" and run.step_count == 1


def test_deltas_stream_reasoning_and_content_with_the_step_number():
    with FakeOpenAI([{"reasoning": "abcdefgh", "content": "xyz"}], piece=3) as f:
        ev = drain(make(f))
    d = [e for e in ev if e["type"] == "delta"]
    assert "".join(e["text"] for e in d if e["kind"] == "reasoning") == "abcdefgh"
    assert "".join(e["text"] for e in d if e["kind"] == "content") == "xyz"
    assert all(e["n"] == 1 for e in d)


def test_the_model_event_carries_tokens_seconds_and_finish_reason():
    with FakeOpenAI([{"reasoning": "r", "content": "c", "usage": {"prompt_tokens": 5, "completion_tokens": 7}}]) as f:
        ev = drain(make(f))
    m = next(e for e in ev if e["type"] == "model")
    assert m["prompt_tokens"] == 5 and m["completion_tokens"] == 7 and m["finish_reason"] == "stop" and m["seconds"] >= 0
    assert m["reasoning"] == "r" and m["content"] == "c"


def test_two_tool_calls_in_one_reply_both_run_in_order_and_both_results_go_back():
    turns = [{"tool_calls": [call("calculator", expression="1+1"), call("calculator", expression="2+2")]}, {"content": "ok"}]
    with FakeOpenAI(turns) as f:
        ev = drain(make(f))
        second = f.requests[1]["messages"]
    t = [e for e in ev if e["type"] == "tool"]
    assert [e["text"] for e in t] == ["1+1 = 2", "2+2 = 4"]
    assert [m["role"] for m in second[2:]] == ["assistant", "tool", "tool"]
    assert len(second[2]["tool_calls"]) == 2


def test_a_reply_with_content_and_a_tool_call_runs_the_tool_and_goes_on():
    turns = [{"content": "Let me check.", "tool_calls": [call("calculator", expression="3*3")]}, {"content": "9"}]
    with FakeOpenAI(turns) as f:
        ev = drain(make(f))
    assert ev[-1]["answer"] == "9" and any(e["type"] == "tool" for e in ev)


# ---- sources ----------------------------------------------------------------------------------------------------------------

def test_sources_from_tools_collect_on_the_run_without_duplicates_and_ride_on_the_tool_event():
    wiki = json.dumps({"query": {"pages": {"1": {"pageid": 1, "title": "Ada Lovelace", "extract": "Ada was born in 1815."}}}})
    tb = tools.Toolbox(env={}, http=lambda url, params=None, headers=None, data=None: wiki)
    turns = [{"tool_calls": [call("wikipedia_page", title="Ada Lovelace")]}, {"tool_calls": [call("wikipedia_page", title="Ada Lovelace  ")]}, {"content": "1815"}]
    with FakeOpenAI(turns) as f:
        run = make(f, toolbox=tb)
        ev = drain(run)
    assert [s.url for s in run.sources.values()] == ["https://en.wikipedia.org/wiki/Ada_Lovelace"]
    assert next(e for e in ev if e["type"] == "tool")["sources"][0]["url"].endswith("Ada_Lovelace")


# ---- bad calls and loops ----------------------------------------------------------------------------------------------------

def test_a_tool_call_with_bad_json_is_answered_with_an_error_and_the_run_goes_on():
    turns = [{"tool_calls": [{"name": "calculator", "arguments": '{"expression": "1+1"'}]}, {"tool_calls": [call("calculator", expression="1+1")]}, {"content": "2"}]
    with FakeOpenAI(turns) as f:
        ev = drain(make(f))
        second = f.requests[1]["messages"]
    bad = [e for e in ev if e["type"] == "tool"][0]
    assert not bad["ok"] and "not valid JSON" in bad["text"]
    assert "not valid JSON" in second[-1]["content"]
    assert ev[-1]["stop_reason"] == "answered"


def test_three_bad_calls_in_a_row_stop_the_run_and_say_so():
    turns = [{"tool_calls": [{"name": "calculator", "arguments": "{oops"}]}] * 3 + [{"content": "never reached"}]
    with FakeOpenAI(turns) as f:
        ev = drain(make(f))
    assert ev[-1]["stop_reason"] == "bad_calls" and "three" in ev[-1]["detail"].lower()


def test_an_unknown_tool_is_reported_to_the_model_and_the_run_goes_on():
    turns = [{"tool_calls": [call("delete_everything", path="/")]}, {"content": "I cannot do that."}]
    with FakeOpenAI(turns) as f:
        ev = drain(make(f))
    t = next(e for e in ev if e["type"] == "tool")
    assert not t["ok"] and "unknown tool" in t["text"].lower() and ev[-1]["stop_reason"] == "answered"


def test_the_same_call_twice_is_answered_with_a_note_and_a_third_time_stops_the_run():
    same = {"tool_calls": [call("calculator", expression="1+1")]}
    with FakeOpenAI([same, same, same, {"content": "x"}]) as f:
        run = make(f)
        ev = drain(run)
    tools_ev = [e for e in ev if e["type"] == "tool"]
    assert tools_ev[0]["ok"] and "already" in tools_ev[1]["text"].lower() and not tools_ev[1].get("executed", True)
    assert ev[-1]["stop_reason"] == "repeated_call" and "calculator" in ev[-1]["detail"]
    assert len(f.requests) == 3


def test_a_repeat_with_different_spacing_in_the_arguments_still_counts_as_a_repeat():
    a = {"tool_calls": [{"name": "calculator", "arguments": '{"expression":"1+1"}'}]}
    b = {"tool_calls": [{"name": "calculator", "arguments": '{ "expression" : "1+1" }'}]}
    with FakeOpenAI([a, b, a, {"content": "x"}]) as f:
        ev = drain(make(f))
    assert ev[-1]["stop_reason"] == "repeated_call"


# ---- budgets ----------------------------------------------------------------------------------------------------------------

def test_the_step_cap_stops_a_run_that_keeps_calling_tools():
    turns = [{"tool_calls": [call("calculator", expression=f"{i}+1")]} for i in range(10)]
    with FakeOpenAI(turns) as f:
        run = make(f, budget=agent.Budget(max_steps=3))
        ev = drain(run)
    assert ev[-1]["stop_reason"] == "step_cap" and run.step_count == 3 and len(f.requests) == 3


def test_the_token_cap_stops_the_run_after_the_step_that_crossed_it():
    turns = [{"tool_calls": [call("calculator", expression=f"{i}+1")], "usage": {"prompt_tokens": 10, "completion_tokens": 600}} for i in range(5)]
    with FakeOpenAI(turns) as f:
        run = make(f, budget=agent.Budget(max_tokens=1000))
        ev = drain(run)
    assert ev[-1]["stop_reason"] == "token_cap" and run.completion_tokens == 1200 and len(f.requests) == 2


def test_the_time_cap_uses_the_clock_and_stops_before_the_next_step():
    now = [0.0]
    turns = [{"tool_calls": [call("calculator", expression="1+1")]}, {"tool_calls": [call("calculator", expression="2+2")]}, {"content": "x"}]
    with FakeOpenAI(turns) as f:
        run = make(f, budget=agent.Budget(max_seconds=100), clock=lambda: now[0])
        gen = run.events()
        out = []
        for e in gen:
            out.append(e)
            if e["type"] == "tool":
                now[0] += 150
    assert out[-1]["stop_reason"] == "time_cap" and len(f.requests) == 1


def test_the_per_step_token_limit_is_sent_as_max_tokens():
    with FakeOpenAI([{"content": "x"}]) as f:
        drain(make(f, budget=agent.Budget(step_tokens=1234)))
        assert f.requests[0]["max_tokens"] == 1234


# ---- a cut-off or empty reply ------------------------------------------------------------------------------------------------

def test_a_reply_cut_off_at_the_token_limit_stops_the_run_and_says_the_thinking_budget_ran_out():
    with FakeOpenAI([{"reasoning": "a very long thought", "finish_reason": "length"}]) as f:
        run = make(f, budget=agent.Budget(step_tokens=500))
        ev = drain(run)
    fin = ev[-1]
    assert fin["stop_reason"] == "length" and "thinking budget" in fin["detail"].lower() and "500" in fin["detail"]
    assert run.can_resume


def test_resuming_after_a_cut_off_asks_the_same_step_again_with_a_bigger_limit():
    turns = [{"reasoning": "long", "finish_reason": "length"}, {"content": "answer after more room"}]
    with FakeOpenAI(turns) as f:
        run = make(f, budget=agent.Budget(step_tokens=500))
        drain(run)
        ev = drain(run, step_tokens=2000)
        assert f.requests[1]["max_tokens"] == 2000
        assert f.requests[1]["messages"] == f.requests[0]["messages"]
    assert ev[-1]["stop_reason"] == "answered" and ev[-1]["answer"] == "answer after more room"


def test_an_empty_reply_gets_one_nudge_and_a_second_empty_reply_stops_the_run():
    with FakeOpenAI([{}, {}]) as f:
        ev = drain(make(f))
        nudge = f.requests[1]["messages"][-1]
    assert nudge["role"] == "user" and "continue" in nudge["content"].lower()
    assert ev[-1]["stop_reason"] == "empty" and "stopped" in ev[-1]["detail"].lower()


def test_an_empty_reply_followed_by_an_answer_recovers():
    with FakeOpenAI([{"reasoning": "hm"}, {"content": "fine"}]) as f:
        ev = drain(make(f))
    assert ev[-1]["stop_reason"] == "answered" and ev[-1]["answer"] == "fine"


# ---- the server and the stop button ---------------------------------------------------------------------------------------------

def test_a_server_error_in_the_middle_stops_the_run_with_a_plain_message_and_keeps_the_steps_so_far():
    turns = [{"tool_calls": [call("calculator", expression="1+1")]}, {"status": 503, "error": "engine restarting"}]
    with FakeOpenAI(turns) as f:
        run = make(f)
        ev = drain(run)
    assert ev[-1]["stop_reason"] == "server_down" and "engine restarting" in ev[-1]["detail"]
    assert run.step_count == 2 and any(e["type"] == "tool" for e in ev)


def test_an_unreachable_server_gives_a_plain_message():
    run = agent.AgentRun(chat.ChatClient("http://127.0.0.1:9/v1", model="m", connect_timeout=1), tools.Toolbox(env={}), mission="x", budget=agent.Budget())
    ev = drain(run)
    assert ev[-1]["stop_reason"] == "server_down" and "127.0.0.1:9" in ev[-1]["detail"]


def test_the_stop_button_ends_the_run_mid_stream():
    stop = threading.Event()
    long = {"reasoning": "x" * 3000, "content": "y" * 3000}
    with FakeOpenAI([long], piece=2) as f:
        run = make(f, cancel=stop.is_set)
        out = []
        for e in run.events():
            out.append(e)
            if e["type"] == "delta" and len(out) > 6:
                stop.set()
    assert out[-1]["stop_reason"] == "cancelled"


def test_a_stop_pressed_before_a_tool_runs_does_not_run_it():
    stop = threading.Event()
    ran = []
    class Spy(tools.Toolbox):
        def run(self, name, arguments):
            ran.append(name); return super().run(name, arguments)
    turns = [{"tool_calls": [call("calculator", expression="1+1")]}, {"content": "x"}]
    with FakeOpenAI(turns) as f:
        run = make(f, toolbox=Spy(env={}), cancel=stop.is_set)
        for e in run.events():
            if e["type"] == "model":
                stop.set()
    assert ran == [] and e["stop_reason"] == "cancelled"


def test_a_tool_that_raises_is_reported_as_an_error_not_a_crash():
    class Broken(tools.Toolbox):
        def run(self, name, arguments):
            raise RuntimeError("tool exploded")
    with FakeOpenAI([{"tool_calls": [call("calculator", expression="1")]}, {"content": "ok"}]) as f:
        ev = drain(make(f, toolbox=Broken(env={})))
    t = next(e for e in ev if e["type"] == "tool")
    assert not t["ok"] and "tool exploded" in t["text"] and ev[-1]["stop_reason"] == "answered"


# ---- plain chat (the comparison run) ---------------------------------------------------------------------------------------------

def test_plain_chat_sends_the_same_question_with_no_tools_in_one_shot():
    with FakeOpenAI([{"reasoning": "think", "content": "my guess is 42", "usage": {"prompt_tokens": 9, "completion_tokens": 4}}]) as f:
        res = agent.plain_chat(chat.ChatClient(f.url, model="m"), "What is 17*23?", step_tokens=900)
        body = f.requests[0]
    assert "tools" not in body and body["max_tokens"] == 900
    assert [m["role"] for m in body["messages"]] == ["user"] and body["messages"][0]["content"] == "What is 17*23?"
    assert res.answer == "my guess is 42" and res.completion_tokens == 4 and res.stop_reason == "answered"


def test_plain_chat_reports_a_cut_off_and_a_dead_server():
    with FakeOpenAI([{"reasoning": "long", "finish_reason": "length"}]) as f:
        assert agent.plain_chat(chat.ChatClient(f.url, model="m"), "q").stop_reason == "length"
    assert agent.plain_chat(chat.ChatClient("http://127.0.0.1:9/v1", model="m", connect_timeout=1), "q").stop_reason == "server_down"


# ---- resuming ----------------------------------------------------------------------------------------------------------------

def test_a_run_stopped_at_the_step_cap_continues_with_more_steps():
    turns = [{"tool_calls": [call("calculator", expression=f"{i}+1")]} for i in range(2)] + [{"content": "done"}]
    with FakeOpenAI(turns) as f:
        run = make(f, budget=agent.Budget(max_steps=2))
        first = drain(run)
        assert first[-1]["stop_reason"] == "step_cap" and run.can_resume
        more = list(run.resume(extra_steps=3))
    assert more[0]["type"] == "resume" and more[-1]["stop_reason"] == "answered" and run.step_count == 3


def test_a_run_stopped_at_the_token_cap_continues_with_more_tokens():
    turns = [{"tool_calls": [call("calculator", expression="1+1")], "usage": {"prompt_tokens": 1, "completion_tokens": 900}}, {"content": "ok"}]
    with FakeOpenAI(turns) as f:
        run = make(f, budget=agent.Budget(max_tokens=500))
        assert drain(run)[-1]["stop_reason"] == "token_cap"
        assert list(run.resume(extra_tokens=5000))[-1]["stop_reason"] == "answered"


def test_idle_time_between_a_stop_and_a_resume_does_not_use_up_the_time_budget():
    now = [0.0]
    turns = [{"tool_calls": [call("calculator", expression="1+1")]}, {"content": "ok"}]
    with FakeOpenAI(turns) as f:
        run = make(f, budget=agent.Budget(max_steps=1, max_seconds=100), clock=lambda: now[0])
        assert drain(run)[-1]["stop_reason"] == "step_cap"
        now[0] += 10_000                      # the person went to lunch
        out = list(run.resume(extra_steps=2))
    assert out[-1]["stop_reason"] == "answered" and out[-1]["seconds"] < 100


def test_a_run_stopped_at_the_time_cap_continues_with_more_time():
    now = [0.0]
    turns = [{"tool_calls": [call("calculator", expression="1+1")]}, {"content": "ok"}]
    with FakeOpenAI(turns) as f:
        run = make(f, budget=agent.Budget(max_seconds=100), clock=lambda: now[0])
        out = []
        for e in run.events():
            out.append(e)
            if e["type"] == "tool":
                now[0] += 150
        assert out[-1]["stop_reason"] == "time_cap"
        assert list(run.resume(extra_seconds=500))[-1]["stop_reason"] == "answered"


# ---- more guards -------------------------------------------------------------------------------------------------------------

def test_a_stop_pressed_during_the_last_tool_ends_the_run_before_another_model_call():
    stop = threading.Event()
    turns = [{"tool_calls": [call("calculator", expression="1+1")]}, {"content": "should not be asked"}]
    with FakeOpenAI(turns) as f:
        run = make(f, cancel=stop.is_set)
        out = []
        for e in run.events():
            out.append(e)
            if e["type"] == "tool":
                stop.set()
    assert out[-1]["stop_reason"] == "cancelled" and len(f.requests) == 1


def test_bad_calls_that_are_not_in_a_row_do_not_stop_the_run():
    bad = {"tool_calls": [{"name": "calculator", "arguments": "{oops"}]}
    good = lambda i: {"tool_calls": [call("calculator", expression=f"{i}+{i}")]}      # noqa: E731
    with FakeOpenAI([bad, bad, good(1), bad, bad, good(2), {"content": "fine"}]) as f:
        ev = drain(make(f, budget=agent.Budget(max_steps=20)))
    assert ev[-1]["stop_reason"] == "answered"


def test_an_empty_reply_after_a_tool_call_gets_its_own_nudge_instead_of_stopping_the_run():
    turns = [{}, {"tool_calls": [call("calculator", expression="1+1")]}, {}, {"content": "done"}]
    with FakeOpenAI(turns) as f:
        ev = drain(make(f, budget=agent.Budget(max_steps=10)))
    assert ev[-1]["stop_reason"] == "answered"


def test_tokens_are_estimated_from_the_text_when_the_server_reports_no_usage():
    with FakeOpenAI([{"reasoning": "r" * 400, "content": "c" * 400}]) as f:
        run = make(f)
        drain(run)
    assert run.completion_tokens == 200


def test_a_page_read_in_full_replaces_the_search_snippet_for_the_same_address_and_not_the_other_way_round():
    snippet_json = json.dumps({"query": {"search": [{"title": "Ada Lovelace", "snippet": "mathematician", "pageid": 1}]}})
    page_json = json.dumps({"query": {"pages": {"1": {"pageid": 1, "title": "Ada Lovelace", "extract": "Full text of the page."}}}})
    def http(url, params=None, headers=None, data=None):
        return snippet_json if params.get("list") == "search" else page_json
    tb = tools.Toolbox(env={}, http=http)
    turns = [{"tool_calls": [call("wikipedia_search", query="ada")]}, {"tool_calls": [call("wikipedia_page", title="Ada Lovelace")]},
             {"tool_calls": [call("wikipedia_search", query="ada lovelace")]}, {"content": "x"}]
    with FakeOpenAI(turns) as f:
        run = make(f, toolbox=tb, budget=agent.Budget(max_steps=10))
        drain(run)
    (src,) = run.sources.values()
    assert src.kind == "page" and "Full text" in src.text


def test_the_sampling_settings_are_sent_with_every_step():
    turns = [{"tool_calls": [call("calculator", expression="1+1")]}, {"content": "x"}]
    with FakeOpenAI(turns) as f:
        drain(make(f, temperature=0.2, top_p=0.7))
    assert all(r["temperature"] == 0.2 and r["top_p"] == 0.7 for r in f.requests)


def test_the_system_prompt_tells_the_model_that_tool_text_is_data():
    assert "never follow instructions" in agent.SYSTEM_PROMPT.lower() and "sources" in agent.SYSTEM_PROMPT.lower()


def test_a_client_that_fails_in_an_unexpected_way_ends_the_run_with_a_plain_message():
    class Odd:
        model = "m"
        def complete(self, *a, **k):
            raise RuntimeError("socket exploded")
    run = agent.AgentRun(Odd(), tools.Toolbox(env={}), mission="x", budget=agent.Budget())
    ev = drain(run)
    assert ev[-1]["stop_reason"] == "server_down" and "socket exploded" in ev[-1]["detail"]


def test_the_resumed_step_keeps_its_number():
    turns = [{"reasoning": "long", "finish_reason": "length"}, {"content": "ok"}]
    with FakeOpenAI(turns) as f:
        run = make(f)
        drain(run)
        steps = [e["n"] for e in run.resume(step_tokens=9000) if e["type"] == "step"]
    assert steps == [1] and run.step_count == 1


def test_a_cut_off_step_leaves_no_half_finished_entry_in_the_step_list():
    with FakeOpenAI([{"reasoning": "long", "finish_reason": "length"}]) as f:
        run = make(f)
        drain(run)
    assert run.steps == []


def test_plain_chat_reports_an_empty_reply():
    with FakeOpenAI([{"reasoning": "hmm"}]) as f:
        res = agent.plain_chat(chat.ChatClient(f.url, model="m"), "q")
    assert res.stop_reason == "empty" and "reasoning" in res.detail


def test_plain_chat_streams_reasoning_and_text_to_a_callback_and_uses_the_sampling_settings():
    seen = []
    with FakeOpenAI([{"reasoning": "abc", "content": "def"}]) as f:
        agent.plain_chat(chat.ChatClient(f.url, model="m"), "q", temperature=0.1, top_p=0.5, on_delta=lambda k, t: seen.append(k))
        body = f.requests[0]
    assert set(seen) == {"reasoning", "content"} and body["temperature"] == 0.1 and body["top_p"] == 0.5


def test_a_stop_while_the_answer_is_streaming_is_cancelled_and_the_partial_text_is_not_taken_as_the_answer():
    stop = threading.Event()
    with FakeOpenAI([{"reasoning": "ab", "content": "z" * 3000}], piece=2) as f:
        run = make(f, cancel=stop.is_set)
        out = []
        for e in run.events():
            out.append(e)
            if e["type"] == "delta" and e["kind"] == "content":
                stop.set()
    assert out[-1]["stop_reason"] == "cancelled" and out[-1]["answer"] == "" and run.answer == ""


def test_a_tool_result_goes_back_under_the_id_the_server_gave_the_call():
    turns = [{"tool_calls": [{"id": "call_xyz_789", "name": "calculator", "arguments": '{"expression": "1+1"}'}]}, {"content": "2"}]
    with FakeOpenAI(turns) as f:
        drain(make(f))
        second = f.requests[1]["messages"]
    assert second[2]["tool_calls"][0]["id"] == "call_xyz_789" and second[3]["tool_call_id"] == "call_xyz_789"


def test_active_time_before_a_resume_still_counts_toward_the_time_budget():
    now = [0.0]
    turns = [{"tool_calls": [call("calculator", expression="1+1")]}, {"tool_calls": [call("calculator", expression="2+2")]}, {"content": "x"}]
    with FakeOpenAI(turns) as f:
        run = make(f, budget=agent.Budget(max_steps=1, max_seconds=100), clock=lambda: now[0])
        out = []
        for e in run.events():
            out.append(e)
            if e["type"] == "tool":
                now[0] += 90
        assert out[-1]["stop_reason"] == "step_cap"
        out2 = []
        for e in run.resume(extra_steps=5):
            out2.append(e)
            if e["type"] == "tool":
                now[0] += 20
    assert out2[-1]["stop_reason"] == "time_cap"
