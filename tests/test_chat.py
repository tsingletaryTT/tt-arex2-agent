# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""The streaming chat client (arex2_agent/chat.py) against a scripted fake server."""
from __future__ import annotations

import socket
import threading
import time

import pytest

from arex2_agent import chat
from tests.fake_openai import FakeOpenAI

MSGS = [{"role": "user", "content": "hi"}]


def test_models_lists_the_served_model_names():
    with FakeOpenAI(model="served-name") as f:
        assert chat.ChatClient(f.url).models() == ["served-name"]


def test_a_reply_collects_reasoning_content_and_usage():
    turn = {"reasoning": "think about it carefully", "content": "The answer is 42.", "usage": {"prompt_tokens": 11, "completion_tokens": 29}}
    with FakeOpenAI([turn]) as f:
        r = chat.ChatClient(f.url, model="m").complete(MSGS)
    assert r.reasoning == "think about it carefully"
    assert r.content == "The answer is 42."
    assert r.finish_reason == "stop"
    assert (r.prompt_tokens, r.completion_tokens) == (11, 29)
    assert r.tool_calls == []
    assert r.seconds >= 0


def test_tool_call_arguments_split_over_many_chunks_are_joined():
    turn = {"tool_calls": [{"name": "calculator", "arguments": '{"expression": "(17 * 23) + 4"}'},
                           {"name": "wikipedia_search", "arguments": '{"query": "Ada Lovelace"}'}]}
    with FakeOpenAI([turn], piece=3) as f:
        r = chat.ChatClient(f.url, model="m").complete(MSGS)
    assert [(c.name, c.arguments) for c in r.tool_calls] == [("calculator", '{"expression": "(17 * 23) + 4"}'),
                                                              ("wikipedia_search", '{"query": "Ada Lovelace"}')]
    assert r.finish_reason == "tool_calls"
    assert all(c.id for c in r.tool_calls)


def test_the_delta_callback_sees_reasoning_before_content_in_order():
    seen = []
    with FakeOpenAI([{"reasoning": "abcdefghij", "content": "klmnopq"}], piece=4) as f:
        chat.ChatClient(f.url, model="m").complete(MSGS, on_delta=lambda kind, text: seen.append((kind, text)))
    assert "".join(t for k, t in seen if k == "reasoning") == "abcdefghij"
    assert "".join(t for k, t in seen if k == "content") == "klmnopq"
    kinds = [k for k, _ in seen]
    assert kinds.index("content") > max(i for i, k in enumerate(kinds) if k == "reasoning")


def test_the_request_asks_for_a_stream_with_usage_and_carries_the_settings():
    tools = [{"type": "function", "function": {"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
    with FakeOpenAI([{"content": "x"}]) as f:
        chat.ChatClient(f.url, model="m").complete(MSGS, tools=tools, max_tokens=777, temperature=0.3, top_p=0.8)
        body = f.requests[0]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert (body["model"], body["max_tokens"], body["temperature"], body["top_p"]) == ("m", 777, 0.3, 0.8)
    assert body["tools"] == tools and body["messages"] == MSGS


def test_no_tools_field_is_sent_when_there_are_no_tools():
    with FakeOpenAI([{"content": "x"}]) as f:
        chat.ChatClient(f.url, model="m").complete(MSGS)
        assert "tools" not in f.requests[0]


def test_an_api_key_is_sent_only_when_one_is_set():
    with FakeOpenAI([{"content": "a"}, {"content": "b"}]) as f:
        chat.ChatClient(f.url, model="m").complete(MSGS)
        chat.ChatClient(f.url, model="m", api_key="sekret").complete(MSGS)
        plain, keyed = f.headers_seen
    assert "Authorization" not in plain
    assert keyed["Authorization"] == "Bearer sekret"


def test_a_cut_off_reply_reports_finish_reason_length():
    with FakeOpenAI([{"reasoning": "long thought", "finish_reason": "length"}]) as f:
        r = chat.ChatClient(f.url, model="m").complete(MSGS)
    assert r.finish_reason == "length" and r.content == ""


def test_a_server_that_is_not_there_gives_a_plain_message_naming_the_address():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    with pytest.raises(chat.ServerDown) as e:
        chat.ChatClient(f"http://127.0.0.1:{port}/v1", model="m").complete(MSGS)
    assert f"127.0.0.1:{port}" in str(e.value)


def test_an_http_error_carries_the_status_and_the_servers_words():
    with FakeOpenAI([{"status": 503, "error": "engine is restarting"}]) as f:
        with pytest.raises(chat.ServerError) as e:
            chat.ChatClient(f.url, model="m").complete(MSGS)
    assert e.value.status == 503 and "engine is restarting" in str(e.value)


def test_a_server_that_stops_answering_times_out_with_a_plain_message():
    with FakeOpenAI([{"content": "late", "delay_s": 2.0}]) as f:
        with pytest.raises(chat.ServerDown) as e:
            chat.ChatClient(f.url, model="m", read_timeout=0.3).complete(MSGS)
    assert "no answer" in str(e.value)


def test_cancel_stops_the_read_and_reports_cancelled():
    long = {"reasoning": "x" * 4000, "content": "y" * 4000}
    count = []
    with FakeOpenAI([long], piece=2) as f:
        r = chat.ChatClient(f.url, model="m").complete(MSGS, on_delta=lambda k, t: count.append(t), cancel=lambda: len(count) >= 5)
    assert r.finish_reason == "cancelled"
    assert 5 <= len(count) < 200


def test_a_stream_that_ends_without_done_is_still_returned_with_what_arrived():
    class Cut(FakeOpenAI):
        def _stream(self, wfile, turn):
            self._send(wfile, {"reasoning_content": "partial thought"})
            wfile.flush()
    with Cut([{}]) as f:
        r = chat.ChatClient(f.url, model="m").complete(MSGS)
    assert r.reasoning == "partial thought" and r.finish_reason == "incomplete"


def test_model_name_defaults_to_the_first_served_model():
    with FakeOpenAI([{"content": "x"}], model="auto-picked") as f:
        c = chat.ChatClient(f.url)
        c.complete(MSGS)
        assert f.requests[0]["model"] == "auto-picked"


def test_a_models_request_that_fails_says_so_with_the_status():
    class Broken(FakeOpenAI):
        pass
    with Broken() as f:
        f.models_status = 502
        with pytest.raises(chat.ServerError) as e:
            chat.ChatClient(f.url).models()
    assert e.value.status == 502


@pytest.mark.parametrize("sep", ["\u2028", "\u2029", "\x85", "\x0b", "\x0c", "\x1c"])
def test_a_line_separator_character_inside_a_streamed_chunk_does_not_split_the_event(sep):
    text = f"before{sep}after"
    with FakeOpenAI([{"reasoning": text, "content": text}], piece=50, raw_unicode=True) as f:
        r = chat.ChatClient(f.url, model="m").complete(MSGS)
    assert r.content == text and r.reasoning == text and r.finish_reason == "stop"
