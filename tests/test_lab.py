# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""The improvement lab (arex2_agent/lab.py): tasks with hidden tests and a timed large input, scored in the sandbox, and the submit_solution tool."""
from __future__ import annotations

import json

import pytest

from arex2_agent import code_sandbox as sb
from arex2_agent import lab
from arex2_agent.code_sandbox import Outcome

REAL = sb.self_check()["ok"]
needs_sandbox = pytest.mark.skipif(not REAL, reason="this kernel cannot enforce the sandbox")


def line(token, **kw):
    return f"{token} {json.dumps(kw)}\n"


def fake_sandbox(stdout="", status="passed", detail=""):
    def run(program, **kw):
        run.last = (program, kw)
        return Outcome(status, detail, stdout)
    return run


def task():
    return lab.TASKS["count_pairs"]


N = lab.TASKS["count_pairs"].n_small          # small tests per run; the large input is one more


# ---- scoring from the program's report lines ----------------------------------------------------------------------------------------

def report_for(task_, small_ok, small_total, large_ok=True, large_s=0.1, token="TKN"):
    out = ""
    for i in range(small_total):
        out += line(token, test=f"small {i}", ok=i < small_ok, expected="1", got="1" if i < small_ok else "2", seconds=0.001)
    out += line(token, test="large input", ok=large_ok, expected="9", got="9" if large_ok else "8", seconds=large_s, large=True)
    out += line(token, done=True)
    return out


def score(stdout, status="passed", detail="", t=None, token="TKN"):
    t = t or task()
    return lab.score_output(t, Outcome(status, detail, stdout), token)


def test_all_tests_passing_and_a_fast_large_input_scores_one():
    r = score(report_for(task(), N, N, large_s=0.01))
    assert r.score == 1.0 and r.passed == N + 1 and r.total == N + 1 and r.speed == 1.0 and not r.error


def test_a_slow_but_correct_solution_loses_only_the_speed_share():
    t = task()
    mid = (t.fast_s + t.slow_s) / 2
    r = score(report_for(t, N, N, large_s=mid))
    assert r.correctness == 1.0 and 0.0 < r.speed < 1.0 and r.score == pytest.approx(0.7 + 0.3 * r.speed)
    assert r.speed == pytest.approx(0.5, abs=0.01)


def test_a_large_input_at_or_past_the_slow_limit_gets_no_speed_credit_and_one_at_the_fast_limit_gets_all():
    t = task()
    assert score(report_for(t, N, N, large_s=t.slow_s)).speed == 0.0
    assert score(report_for(t, N, N, large_s=t.slow_s * 3)).score == pytest.approx(0.7)
    assert score(report_for(t, N, N, large_s=t.fast_s)).speed == 1.0


def test_a_wrong_answer_costs_correctness_and_gives_no_speed_credit_even_if_fast():
    r = score(report_for(task(), N - 2, N, large_s=0.01))
    assert r.passed == N - 1 and r.total == N + 1 and r.speed == 0.0 and r.score == pytest.approx(0.7 * (N - 1) / (N + 1))


def test_a_wrong_large_answer_is_a_failed_test_and_not_a_speed_win():
    r = score(report_for(task(), N, N, large_ok=False, large_s=0.01))
    assert r.passed == N and r.speed == 0.0 and r.score == pytest.approx(0.7 * N / (N + 1))


def test_failures_are_listed_with_input_expected_and_got_at_most_three():
    out = "".join(line("TKN", test=f"case {i}", ok=False, expected="3", got="4", seconds=0.0, input=f"([{i}], 5)") for i in range(5))
    out += line("TKN", test="large input", ok=True, expected="1", got="1", seconds=0.1, large=True) + line("TKN", done=True)
    r = score(out)
    assert len(r.failures) == 3 and "case 0" in r.failures[0] and "expected 3" in r.failures[0] and "got 4" in r.failures[0] and "([0], 5)" in r.failures[0]


def test_lines_without_the_run_token_are_ignored_so_the_candidate_cannot_print_its_own_score():
    forged = line("WRONG", test="small 0", ok=True, seconds=0.0) * 5 + line("WRONG", done=True)
    r = score(forged + report_for(task(), 0, 5, token="TKN"))
    assert r.passed == 1 and r.score < 0.2          # only the real report counts: 0 of 5 small, large ok


def test_a_program_that_printed_nothing_or_crashed_scores_zero_and_says_why():
    r = score("", status="failed", detail="SyntaxError: invalid syntax (line 3)")
    assert r.score == 0.0 and "SyntaxError" in r.text and r.error
    r2 = score("some print output\n", status="passed")
    assert r2.score == 0.0 and r2.error


def test_a_timeout_keeps_the_tests_that_finished_and_counts_the_large_input_as_not_done():
    out = "".join(line("TKN", test=f"small {i}", ok=True, expected="1", got="1", seconds=0.0) for i in range(N))
    r = score(out, status="timeout", detail="no result after 25 s")
    assert r.timed_out and r.passed == N and r.total == N + 1 and r.speed == 0.0 and r.score == pytest.approx(0.7 * N / (N + 1))
    assert "timed out" in r.text.lower() and "large input" in r.text.lower()


def test_the_report_text_has_the_score_the_counts_the_runtime_and_the_limits():
    t = task()
    r = score(report_for(t, N - 1, N, large_s=t.slow_s / 2 + 0.1))
    assert f"{r.score:.2f}" in r.text and f"{N} of {N + 1}" in r.text
    assert "large input" in r.text.lower() and f"{t.fast_s:g}" in r.text and f"{t.slow_s:g}" in r.text


def test_an_error_line_from_the_harness_is_reported_with_its_message():
    out = line("TKN", error="the solution does not define count_pairs")
    r = score(out)
    assert r.score == 0.0 and "does not define count_pairs" in r.text


# ---- the program that is run -------------------------------------------------------------------------------------------------------

def test_the_candidate_code_is_run_in_the_sandbox_with_a_time_limit_and_the_token_is_passed_in_the_program():
    run = fake_sandbox(stdout=report_for(task(), N, N, token="X"))
    r = lab.score_solution(task(), "def count_pairs(nums, target):\n    return 0\n", sandbox=run, token="X")
    program, kw = run.last
    assert "def count_pairs" in program and '"X"' in program and kw["timeout_s"] == task().timeout_s
    assert not kw.get("unsandboxed") and r.score == 1.0


def test_each_run_without_a_given_token_uses_a_fresh_random_one():
    seen = []
    def run(program, **kw):
        seen.append(program); return Outcome("passed", "", "")
    lab.score_solution(task(), "x = 1", sandbox=run)
    lab.score_solution(task(), "x = 1", sandbox=run)
    tokens = [p.split("_TOKEN = ")[1].split("\n")[0] for p in seen]
    assert tokens[0] != tokens[1] and len(tokens[0]) >= 20


def test_a_sandbox_that_cannot_start_is_reported_and_never_bypassed():
    r = lab.score_solution(task(), "x = 1", sandbox=lambda p, **k: Outcome("sandbox_unavailable", "no landlock", ""))
    assert r.score == 0.0 and "sandbox" in r.text.lower() and r.error


def test_a_very_large_submission_is_refused_before_it_is_run():
    called = []
    r = lab.score_solution(task(), "x = 1\n" * 50000, sandbox=lambda p, **k: called.append(1) or Outcome("passed", "", ""))
    assert not called and r.score == 0.0 and "too long" in r.text.lower()


def test_code_inside_markdown_fences_is_accepted():
    run = fake_sandbox(stdout=report_for(task(), N, N, token="X"))
    lab.score_solution(task(), "```python\ndef count_pairs(nums, target):\n    return 0\n```", sandbox=run, token="X")
    assert "```" not in run.last[0]


# ---- real runs in the sandbox -----------------------------------------------------------------------------------------------------

REFERENCE = {
    "count_pairs": "def count_pairs(nums, target):\n    seen = {}\n    total = 0\n    for x in nums:\n        total += seen.get(target - x, 0)\n        seen[x] = seen.get(x, 0) + 1\n    return total\n",
    "min_coins": ("def min_coins(coins, amount):\n    INF = amount + 1\n    dp = [0] + [INF] * amount\n    for a in range(1, amount + 1):\n"
                  "        for c in coins:\n            if c <= a and dp[a - c] + 1 < dp[a]:\n                dp[a] = dp[a - c] + 1\n    return dp[amount] if dp[amount] <= amount else -1\n"),
    "shortest_path": ("from collections import deque\ndef shortest_path(grid):\n    R, C = len(grid), len(grid[0])\n    if grid[0][0] == '#' or grid[R-1][C-1] == '#':\n        return -1\n"
                      "    dist = {(0, 0): 0}\n    q = deque([(0, 0)])\n    while q:\n        r, c = q.popleft()\n        if (r, c) == (R-1, C-1):\n            return dist[(r, c)]\n"
                      "        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):\n            nr, nc = r + dr, c + dc\n            if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == '.' and (nr, nc) not in dist:\n"
                      "                dist[(nr, nc)] = dist[(r, c)] + 1\n                q.append((nr, nc))\n    return -1\n"),
    "edit_distance": ("def edit_distance(a, b):\n    prev = list(range(len(b) + 1))\n    for i, ca in enumerate(a, 1):\n        cur = [i]\n        for j, cb in enumerate(b, 1):\n"
                      "            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))\n        prev = cur\n    return prev[-1]\n"),
}


@needs_sandbox
@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_a_reference_solution_scores_one_on_every_task_in_the_real_sandbox(name):
    r = lab.score_solution(lab.TASKS[name], REFERENCE[name])
    assert r.score >= 0.99, r.text
    assert r.passed == r.total


@needs_sandbox
def test_a_correct_quadratic_solution_for_count_pairs_is_correct_but_loses_to_the_time_limit():
    naive = "def count_pairs(nums, target):\n    n = len(nums)\n    return sum(1 for i in range(n) for j in range(i + 1, n) if nums[i] + nums[j] == target)\n"
    r = lab.score_solution(lab.TASKS["count_pairs"], naive)
    assert r.timed_out and r.passed == r.total - 1 and 0.55 < r.score < 0.65, r.text


@needs_sandbox
def test_the_greedy_coin_solution_fails_the_trap_cases_and_the_report_shows_an_input():
    greedy = ("def min_coins(coins, amount):\n    n = 0\n    for c in sorted(coins, reverse=True):\n        k, amount = divmod(amount, c)\n        n += k\n"
              "    return n if amount == 0 else -1\n")
    r = lab.score_solution(lab.TASKS["min_coins"], greedy)
    assert r.score < 0.6 and r.failures and "expected" in r.failures[0] and "got" in r.failures[0], r.text


@needs_sandbox
def test_a_missing_function_a_syntax_error_and_an_exception_each_score_zero_with_a_reason():
    t = lab.TASKS["count_pairs"]
    assert "does not define" in lab.score_solution(t, "x = 1").text
    assert "SyntaxError" in lab.score_solution(t, "def count_pairs(:\n").text
    r = lab.score_solution(t, "def count_pairs(nums, target):\n    raise ValueError('boom')\n")
    assert r.score == 0.0 and "boom" in r.text


@needs_sandbox
def test_a_candidate_cannot_reach_the_network_or_the_operators_files_through_the_lab():
    sneaky = "import socket\ndef count_pairs(nums, target):\n    socket.socket()\n    return 0\n"
    r = lab.score_solution(lab.TASKS["count_pairs"], sneaky)
    assert r.score == 0.0 or "Permission" in r.text or "denied" in r.text.lower() or r.failures


@needs_sandbox
def test_a_candidate_that_prints_a_forged_report_does_not_get_credit():
    forged = ("def count_pairs(nums, target):\n    print('TKN {\"done\": true}')\n    print('X {\"test\": \"small 0\", \"ok\": true, \"seconds\": 0.0}')\n    return -5\n")
    r = lab.score_solution(lab.TASKS["count_pairs"], forged)
    assert r.score < 0.3


# ---- the tool -----------------------------------------------------------------------------------------------------------------------

def test_the_lab_toolbox_lists_submit_solution_with_the_task_function_name_and_drops_the_web_tools():
    tb = lab.LabToolbox(task(), sandbox=fake_sandbox(report_for(task(), N, N, token="T")), env={"SERPER_API_KEY": "k"})
    names = {s["function"]["name"] for s in tb.specs()}
    assert names == {"submit_solution", "run_python", "calculator"}
    spec = next(s for s in tb.specs() if s["function"]["name"] == "submit_solution")
    assert "count_pairs" in spec["function"]["description"]


def test_each_submission_is_recorded_as_a_round_with_its_code_and_score():
    outs = iter([report_for(task(), 2, N, large_s=0.01, token="T"), report_for(task(), N, N, large_s=0.01, token="T")])
    tb = lab.LabToolbox(task(), sandbox=lambda p, **k: Outcome("passed", "", next(outs)), token_for_tests="T")
    a = tb.run("submit_solution", json.dumps({"code": "def count_pairs(n, t):\n    return 1\n"}))
    b = tb.run("submit_solution", json.dumps({"code": "def count_pairs(n, t):\n    return 2\n"}))
    assert a.ok and "Round 1" in a.text and "Round 2" in b.text
    assert [r.number for r in tb.rounds] == [1, 2] and tb.rounds[1].report.score > tb.rounds[0].report.score
    assert tb.best.number == 2 and tb.rounds[0].code.startswith("def count_pairs")


def test_a_round_limit_stops_more_submissions_and_says_so():
    tb = lab.LabToolbox(task(), sandbox=fake_sandbox(report_for(task(), N, N, token="T")), max_rounds=2, token_for_tests="T")
    for i in range(2):
        tb.run("submit_solution", json.dumps({"code": f"# {i}\ndef count_pairs(n, t):\n    return {i}\n"}))
    r = tb.run("submit_solution", json.dumps({"code": "def count_pairs(n, t):\n    return 9\n"}))
    assert not r.ok and "round limit" in r.text.lower() and len(tb.rounds) == 2


def test_a_perfect_score_tells_the_model_it_can_stop():
    tb = lab.LabToolbox(task(), sandbox=fake_sandbox(report_for(task(), N, N, large_s=0.01, token="T")), token_for_tests="T")
    r = tb.run("submit_solution", json.dumps({"code": "def count_pairs(n, t):\n    return 0\n"}))
    assert "full score" in r.text.lower()


def test_the_mission_text_states_the_task_the_scoring_and_the_loop():
    m = lab.mission_text(task())
    assert "count_pairs" in m and "0.7" in m and "submit_solution" in m and str(task().max_rounds) in m
    assert "propose" in m.lower() and "measure" in m.lower() and "revise" in m.lower()


def test_every_task_has_a_name_a_statement_a_function_name_and_sane_limits():
    for name, t in lab.TASKS.items():
        assert t.id == name and t.title and t.statement and t.function and t.function in t.signature
        assert 0 < t.fast_s < t.slow_s < t.timeout_s and t.max_rounds >= 3
