#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""A scripted stand-in for the model server, to try the page and take screenshots without a chip.

    python tools/demo_server.py --port 8000          # then: python demo/app.py   (choose "Offline corpus" in the page)

THIS IS NOT AREX-2. It replays a fixed script for the preset missions: the same tool calls, the same text, every time. Its model name is
"SCRIPTED-FAKE-MODEL (not AREX-2)" and the page shows that name in its badge. Anything made with it (the README screenshots) shows the page. It does not show the model.
The improvement-lab scores are real: the submitted code is run by the lab in its sandbox.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arex2_agent import lab, missions  # noqa: E402
from tests.fake_openai import FakeOpenAI  # noqa: E402

MODEL_NAME = "SCRIPTED-FAKE-MODEL (not AREX-2)"


def _call(name, **args):
    return {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}


def _urls():
    return {d["title"]: d["url"] for d in missions.load_corpus()}


def research_script():
    u = _urls()
    return [
        {"reasoning": "The mission has two parts: the architect's birth year and the opening year. I do not want to rely on memory for either date, so I will search first and read the pages.",
         "tool_calls": [_call("wikipedia_search", query="Sydney Opera House architect opened")],
         "usage": {"prompt_tokens": 410, "completion_tokens": 64}},
        {"reasoning": "The search names Jørn Utzon as the architect. I need his birth date and the formal opening date, so I will read both pages.",
         "tool_calls": [_call("wikipedia_page", title="Jørn Utzon"), _call("wikipedia_page", title="Sydney Opera House")],
         "usage": {"prompt_tokens": 620, "completion_tokens": 52}},
        {"reasoning": "Utzon was born on 9 April 1918. The building was formally opened on 20 October 1973. The difference in calendar years is 1973 - 1918. I will use the calculator and not do it in my head.",
         "tool_calls": [_call("calculator", expression="1973 - 1918")],
         "usage": {"prompt_tokens": 1900, "completion_tokens": 71}},
        {"content": ("Answer: 55 years.\n\nJørn Utzon, the architect of the Sydney Opera House, was born on 9 April 1918 in Copenhagen. "
                     "The Sydney Opera House was formally opened by Queen Elizabeth II on 20 October 1973. The difference between 1973 and 1918 is 55 years.\n\n"
                     f"Sources:\n- {u.get('Jørn Utzon', '')}\n- {u.get('Sydney Opera House', '')}"),
         "reasoning": "Both dates come from the pages I read, and the subtraction is from the calculator.", "usage": {"prompt_tokens": 2100, "completion_tokens": 160}},
    ]


PLAIN_RESEARCH = {"reasoning": "I remember Utzon was born around 1918 and the Opera House opened in 1973.",
                  "content": "From memory: Jørn Utzon was born in 1918 and the Sydney Opera House opened in 1973, so about 55 years. I did not check these dates against a source.",
                  "usage": {"prompt_tokens": 60, "completion_tokens": 85}}

COIN_ROUNDS = [
    ("Start with the obvious approach: take the biggest coin as many times as possible.",
     "def min_coins(coins, amount):\n    n = 0\n    for c in sorted(coins, reverse=True):\n        k, amount = divmod(amount, c)\n        n += k\n    return n if amount == 0 else -1\n"),
    ("The greedy version fails the trap cases: with coins 1, 3, 4 and amount 6 it answers 3 but 2 is possible. I need to consider every way to reach each amount. A memoised recursion is the quickest change.",
     "import functools\n\ndef min_coins(coins, amount):\n    @functools.lru_cache(None)\n    def best(a):\n        if a == 0:\n            return 0\n        r = min((best(a - c) + 1 for c in coins if c <= a), default=10**9)\n        return r\n    r = best(amount)\n    return r if r < 10**9 else -1\n"),
    ("The recursion is correct on the small cases but the large input hits the recursion limit. The same recurrence can run bottom-up with a list, which has no depth problem and is fast.",
     "def min_coins(coins, amount):\n    inf = amount + 1\n    dp = [0] + [inf] * amount\n    for a in range(1, amount + 1):\n        for c in coins:\n            if c <= a and dp[a - c] + 1 < dp[a]:\n                dp[a] = dp[a - c] + 1\n    return dp[amount] if dp[amount] <= amount else -1\n"),
]


def _assistant_turns(body) -> int:
    return sum(1 for m in body["messages"] if m["role"] == "assistant")


def script_fn(body):
    """Stateless: which turn to play depends only on how many assistant messages the conversation already has."""
    msgs = body["messages"]
    system = msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else ""
    prompt = next((m["content"] for m in msgs if m["role"] == "user"), "")
    n = _assistant_turns(body)
    if not body.get("tools"):                                         # the plain chat run
        if "Write `def" in prompt:
            return {"reasoning": "One shot, no way to test it.", "content": "```python\n" + COIN_ROUNDS[0][1] + "```", "usage": {"prompt_tokens": 300, "completion_tokens": 120}}
        return PLAIN_RESEARCH
    if system.startswith(lab.LAB_SYSTEM_PROMPT[:40]):
        if "min_coins" in prompt and n < len(COIN_ROUNDS):
            thought, code = COIN_ROUNDS[n]
            return {"reasoning": thought, "tool_calls": [_call("submit_solution", code=code)], "usage": {"prompt_tokens": 500 + 300 * n, "completion_tokens": 140}}
        if "min_coins" in prompt:
            return {"reasoning": "Full score on round 3.", "content": "Best code: the bottom-up version from round 3 (score 1.00).\n\nRound 1 was greedy and failed the trap cases. Round 2 was memoised "
                                                                      "recursion: correct on small inputs, but it hit the recursion limit on the large one. Round 3 runs the same recurrence bottom-up with a list.",
                    "usage": {"prompt_tokens": 1500, "completion_tokens": 160}}
        return {"content": "This scripted demo only knows the min_coins task in the improvement lab.", "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
    if "Sydney Opera House" in prompt and n < len(research_script()):
        return research_script()[n]
    return {"content": "This scripted demo only knows the Sydney Opera House mission. Run a real model server to try your own questions.", "usage": {"prompt_tokens": 100, "completion_tokens": 24}}


def make_server(delay_s: float = 0.004, piece: int = 6) -> FakeOpenAI:
    return FakeOpenAI(model=MODEL_NAME, piece=piece, chunk_delay_s=delay_s, script_fn=script_fn)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    srv = make_server()
    srv.server.server_close()
    from http.server import ThreadingHTTPServer
    srv.server = ThreadingHTTPServer(("127.0.0.1", a.port), srv.server.RequestHandlerClass)
    srv.port, srv.url = a.port, f"http://127.0.0.1:{a.port}/v1"
    print(f"scripted fake model (not AREX-2) at {srv.url}", flush=True)
    try:
        srv.server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
