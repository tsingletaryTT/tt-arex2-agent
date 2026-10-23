# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""HTML fragments for the page, as pure functions of the run state.

Everything that came from the model, from a tool or from a web page goes through `esc` before it is placed in markup, and only http and https
addresses become links. The page runs on a LAN, so a page the agent reads must not be able to put script or an image into it. The fragments
state counts and measurements. They never say one answer is better or more correct than another.
"""
from __future__ import annotations

import html
import json
import re
from typing import Iterable, Optional

CSS = """
:root{--bg:#F1F8F8;--panel:#FFFFFF;--alt:#DCF0EE;--line:#C7D9D8;--ink:#092221;--soft:#3A5452;--accent:#1B8EB1;--teal:#74C5DF;--green:#2F7F6F;--green-bg:#E3F1EE;
--amber:#8A5A00;--amber-bg:#FDF1D6;--red:#B3361C;--red-bg:#FFE6E0;--blue-bg:#E1F1F7;--mono:ui-monospace,'Berkeley Mono',SFMono-Regular,Menlo,monospace}
.dark{--bg:#092221;--panel:#0F2F2E;--alt:#143B3A;--line:#24504F;--ink:#F1F8F8;--soft:#A9C4C2;--accent:#74C5DF;--teal:#74C5DF;--green:#6FABA0;--green-bg:#123B36;--amber:#F6BC42;--amber-bg:#3B3010;--red:#FF9E8A;--red-bg:#40201A;--blue-bg:#10343F}
.gradio-container,body{background:var(--bg)!important;color:var(--ink)}
.arex{font-family:'Neue Haas Unica Pro',system-ui,-apple-system,'Segoe UI',sans-serif;color:var(--ink);line-height:1.45}
.arex *{box-sizing:border-box}
.arex .head{border-left:6px solid var(--accent);padding:6px 14px;margin:4px 0 10px}
.arex .head h1{font-family:'Degular Display',system-ui,sans-serif;font-size:1.6rem;margin:0}
.arex .head p{margin:2px 0 0;color:var(--soft)}
.arex .badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:.78rem;background:var(--blue-bg);color:var(--ink);margin:2px 4px 2px 0}
.arex .badge.warn{background:var(--amber-bg);color:var(--amber)} .arex .badge.bad{background:var(--red-bg);color:var(--red)} .arex .badge.good{background:var(--green-bg);color:var(--green)}
.arex .step{border-left:4px solid var(--line);background:var(--panel);padding:8px 12px;margin:8px 0;border-radius:0 8px 8px 0}
.arex .step.live{border-left-color:var(--accent)}
.arex .step-head{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.arex .step-head .num{font-weight:700}.arex .meta{color:var(--soft);font-size:.82rem}
.arex .dots::after{content:'';display:inline-block;width:1.2em;animation:dots 1.2s steps(4,end) infinite;text-align:left}
@keyframes dots{0%{content:''}25%{content:'.'}50%{content:'..'}75%{content:'...'}}
@media (prefers-reduced-motion: reduce){.arex .dots::after{animation:none;content:'...'}}
.arex pre{white-space:pre-wrap;word-break:break-word;font-family:var(--mono);font-size:.82rem;background:var(--alt);padding:8px;border-radius:6px;max-height:320px;overflow:auto;margin:6px 0}
.arex details>summary{cursor:pointer;color:var(--soft);font-size:.86rem}
.arex .tool{border-left:3px solid var(--green);background:var(--green-bg);padding:6px 10px;margin:6px 0;border-radius:0 6px 6px 0}
.arex .tool.fail{border-left-color:var(--red);background:var(--red-bg)} .arex .tool.skipped{border-left-color:var(--amber);background:var(--amber-bg)}
.arex .tool code{font-family:var(--mono);font-size:.82rem;word-break:break-all}
.arex .meter{margin:6px 0}.arex .meter .bar{height:8px;background:var(--alt);border-radius:4px;overflow:hidden}
.arex .meter .bar i{display:block;height:100%;background:var(--accent)}.arex .meter.warn .bar i{background:#F6BC42}.arex .meter.over .bar i{background:#FA512E}
.arex .meter-label{display:flex;justify-content:space-between;font-size:.84rem;color:var(--soft)}
.arex .source{border-left:3px solid var(--accent);background:var(--panel);padding:6px 10px;margin:6px 0;border-radius:0 6px 6px 0}
.arex .kind{font-size:.74rem;padding:1px 7px;border-radius:999px;background:var(--blue-bg);margin-right:6px}.arex .kind.snippet{background:var(--amber-bg);color:var(--amber)}
.arex .quote{color:var(--soft);font-size:.86rem;margin:3px 0 0}
.arex .claim{padding:5px 10px;margin:5px 0;border-left:3px solid var(--line);background:var(--panel)}
.arex .claim.backed{border-left-color:var(--green)}.arex .claim.computed{border-left-color:var(--accent)}.arex .claim.unbacked{border-left-color:#F6BC42}
.arex .chip{font-size:.74rem;padding:1px 7px;border-radius:999px;margin-right:6px}
.arex .chip.backed{background:var(--green-bg);color:var(--green)}.arex .chip.computed{background:var(--blue-bg)}.arex .chip.unbacked{background:var(--amber-bg);color:var(--amber)}
.arex .note{color:var(--soft);font-size:.82rem;margin:6px 0}
.arex .notice{border-left:4px solid var(--accent);background:var(--blue-bg);padding:8px 12px;margin:8px 0}
.arex .notice.warn{border-left-color:#F6BC42;background:var(--amber-bg)}.arex .notice.error{border-left-color:#FA512E;background:var(--red-bg)}.arex .notice.good{border-left-color:var(--green);background:var(--green-bg)}
.arex .answer{border-left:5px solid var(--green);background:var(--panel);padding:10px 14px;margin:8px 0;font-size:1.02rem}
.arex table{border-collapse:collapse;width:100%;font-size:.9rem}.arex th,.arex td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top}
.arex svg text{fill:var(--soft);font-size:11px}.arex svg .line{stroke:var(--accent);fill:none;stroke-width:2}.arex svg .grid{stroke:var(--line)}
.arex svg circle{fill:var(--accent)}.arex svg circle.best{fill:#FA512E}
@media (max-width:700px){.arex .head h1{font-size:1.25rem}.arex table{font-size:.82rem}}
.inline-action{align-self:flex-end;margin-bottom:14px}
.tools-note{align-self:center}.tools-note p{margin:0;color:var(--soft)}
.settings-group{gap:4px}.settings-group p{margin:2px 0}
"""

STOP_TEXT = {
    "answered": "Finished with an answer.",
    "step_cap": "Stopped at the step limit. Continue to allow more steps.",
    "token_cap": "Stopped at the token budget. Continue to allow more tokens.",
    "time_cap": "Stopped at the time budget. Continue to allow more time.",
    "length": "The thinking budget ran out: a reply was cut off at the per-step token limit before it finished. Continue with a larger limit to give it more room.",
    "empty": "The model stopped inside its reasoning twice and gave neither an answer nor a tool call.",
    "repeated_call": "Stopped: the model asked for the same tool call three times.",
    "bad_calls": "Stopped: three tool calls in a row had arguments that were not valid JSON.",
    "server_down": "The model server could not be reached or stopped answering.",
    "cancelled": "Stopped with the stop button.",
}
KIND_TEXT = {"page": "read in full", "abstract": "abstract read", "snippet": "search snippet only"}


def esc(text) -> str:
    return html.escape("" if text is None else str(text), quote=True)


def safe_link(url: str, label: str) -> str:
    u = (url or "").strip()
    if re.match(r"^https?://", u, re.I) and not re.search(r"[\s<>\"']", u):
        return f'<a href="{esc(u)}" target="_blank" rel="noopener noreferrer">{esc(label)}</a>'
    return esc(label)


def _fmt_s(seconds) -> str:
    return f"{float(seconds or 0):,.0f} s"


def _tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return esc(text)
    return esc("... " + text[-limit:]) + f'<div class="note">showing the last {limit:,} of {len(text):,} characters</div>'


def _args(arguments: str) -> str:
    try:
        shown = json.dumps(json.loads(arguments), ensure_ascii=False)
    except (ValueError, TypeError):
        return esc((arguments or "")[:300]) + ' <span class="note">(not valid JSON)</span>'
    return esc(shown[:300] + ("..." if len(shown) > 300 else ""))


# ---- the timeline ---------------------------------------------------------------------------------------------------------------

def timeline_html(steps: Iterable[dict]) -> str:
    steps = list(steps)
    if not steps:
        return '<div class="arex"><div class="note">No steps yet. Pick a mission and press Run.</div></div>'
    out = ['<div class="arex">']
    for s in steps:
        live = bool(s.get("live"))
        meta = []
        if not live:
            meta.append(f"{float(s.get('seconds') or 0):.1f} s")
            if s.get("completion_tokens"):
                meta.append(f"{int(s['completion_tokens']):,} tokens")
        head = f'<span class="num">Step {int(s.get("n", 0))}</span>' + (f'<span class="meta dots">thinking</span>' if live else f'<span class="meta">{" · ".join(meta)}</span>')
        out.append(f'<div class="step{" live" if live else ""}"><div class="step-head">{head}</div>')
        reasoning = s.get("reasoning") or ""
        if reasoning:
            out.append(f'<details{" open" if live else ""}><summary>Reasoning ({len(reasoning):,} characters)</summary><pre>{_tail(reasoning, 1200 if live else 1500)}</pre></details>')
        if s.get("content") and s.get("tools"):
            out.append(f'<div class="note">Said: {esc(s["content"][:300])}</div>')
        for t in s.get("tools") or []:
            cls = "tool" if t.get("ok") else ("tool skipped" if t.get("executed") is False else "tool fail")
            state = "ok" if t.get("ok") else ("not run (repeated call)" if t.get("executed") is False else "failed")
            text = t.get("text") or ""
            out.append(f'<div class="{cls}"><div class="step-head"><b>{esc(t.get("name"))}</b> <code>{_args(t.get("arguments", ""))}</code>'
                       f'<span class="meta">{state} · {float(t.get("seconds") or 0):.2f} s</span></div>'
                       f'<details><summary>Result ({len(text)} characters)</summary><pre>{esc(text[:600])}</pre></details></div>')
        out.append("</div>")
    out.append("</div>")
    return "".join(out)


# ---- budgets ---------------------------------------------------------------------------------------------------------------------

def _meter(label: str, used: float, limit: float, fmt) -> str:
    pct = 0.0 if not limit else used / limit * 100
    cls = "over" if pct > 100 else "warn" if pct >= 75 else ""
    return (f'<div class="meter {cls}"><div class="meter-label"><span>{esc(label)}</span><span><b>{esc(fmt(used))} / {esc(fmt(limit))}</b> · {pct:.0f}%</span></div>'
            f'<div class="bar"><i style="width:{min(pct, 100):.0f}%"></i></div></div>')


def meters_html(budget: dict, steps: int, tokens: int, seconds: float) -> str:
    return ('<div class="arex">'
            + _meter("Steps", steps, budget["max_steps"], lambda v: f"{int(v):,}")
            + _meter("Tokens generated", tokens, budget["max_tokens"], lambda v: f"{int(v):,}")
            + _meter("Time", seconds, budget["max_seconds"], lambda v: f"{v:,.0f} s")
            + f'<div class="note">One reply may use up to {int(budget["step_tokens"]):,} tokens, thinking included.</div></div>')


# ---- the evidence board ----------------------------------------------------------------------------------------------------------

CHECK_NOTE = ("This check matches words and numbers against the text the agent retrieved. It does not judge whether a source is right, "
              "or whether a claim means what the source means.")


def evidence_html(sources: Iterable, claims: Iterable, cited: dict) -> str:
    sources, claims = list(sources), list(claims)
    out = ['<div class="arex">']
    if not sources:
        out.append('<div class="note">The agent has read nothing yet.</div>')
    for s in sources:
        quote = re.sub(r"\s+", " ", s.text or "")[:240]
        out.append(f'<div class="source"><span class="kind {esc(s.kind)}">{esc(KIND_TEXT.get(s.kind, s.kind))}</span>{safe_link(s.url, s.title or s.url)}'
                   f'<div class="quote">“{esc(quote)}”</div></div>')
    if claims:
        counts = {k: sum(c.status == k for c in claims) for k in ("backed", "computed", "unbacked")}
        out.append(f'<div class="note"><b>Claims in the answer:</b> {counts["backed"]} backed, {counts["computed"]} computed, {counts["unbacked"]} not backed (of {len(claims)}).</div>')
        for c in claims:
            names = esc(c.source_title or c.source_url) + "".join(f" and {esc(title or url)}" for url, title in getattr(c, "also_from", []))
            if c.status == "backed":
                chip = f'<span class="chip backed">backed by {names}</span>'
            elif c.status == "computed" and not (c.source_title or c.source_url):
                chip = '<span class="chip computed">computed: numbers from a calculation</span>'
            elif c.status == "computed":
                chip = f'<span class="chip computed">computed from {names} and a calculation</span>'
            else:
                miss = f" (no {esc(', '.join(c.missing_numbers))} in the retrieved text)" if c.missing_numbers else ""
                chip = f'<span class="chip unbacked">not backed by anything retrieved{miss}</span>'
            ev = f'<details><summary>matching text</summary><pre>{esc(c.evidence)}</pre></details>' if c.evidence else ""
            out.append(f'<div class="claim {esc(c.status)}">{chip}{esc(c.text)}{ev}</div>')
    if cited and cited.get("never_opened"):
        urls = "".join(f"<li><code>{esc(u)}</code></li>" for u in cited["never_opened"])
        out.append(f'<div class="notice warn">The answer cites addresses the agent never opened. A model can write a plausible link that does not exist:<ul>{urls}</ul></div>')
    out.append(f'<div class="note">{esc(CHECK_NOTE)}</div></div>')
    return "".join(out)


def answer_html(answer: str, claims: Iterable) -> str:
    if not (answer or "").strip():
        return ""
    return f'<div class="arex"><div class="answer">{esc(answer).replace(chr(10), "<br>")}</div></div>'


# ---- the comparison ---------------------------------------------------------------------------------------------------------------

def _found(answer: str, facts: Iterable[str]) -> int:
    flat = (answer or "").casefold().replace(",", "")
    return sum(1 for f in facts if str(f).casefold().replace(",", "") in flat)


def scorecard_html(agent: Optional[dict], plain: Optional[dict], expected_facts: Optional[list]) -> str:
    cols = [("Agent", agent)] + ([("Plain chat", plain)] if plain else [])
    rows = []

    def row(label, cells):
        rows.append(f"<tr><th>{esc(label)}</th>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    if expected_facts:
        n = len(expected_facts)
        row("Mentions the expected facts", [f"{_found(d.get('answer', ''), expected_facts)} of {n}" for _, d in cols])
    if any(d.get("lab_score") is not None for _, d in cols):
        row("Lab score", [(f"{d['lab_score']:.2f}" if d.get("lab_score") is not None else "no code to score") for _, d in cols])
    is_lab = any(d.get("lab_score") is not None for _, d in cols)
    if not is_lab:
        row("Sources read", [f"{d.get('sources', 0)}" if name == "Agent" else "no sources (no tools)" for name, d in cols])
        row("Claims backed by retrieved text", [
            (f"{d['claims']['backed'] + d['claims']['computed']} of {d['claims']['total']}" if d.get("claims") and d["claims"]["total"] else "none to check") if name == "Agent"
            else "no sources to check against" for name, d in cols])
    row("Steps", [f"{d.get('steps', 0)}" for _, d in cols])
    row("Tokens generated", [f"{int(d.get('completion_tokens') or 0):,}" for _, d in cols])
    row("Time", [_fmt_s(d.get("seconds")) for _, d in cols])
    row("How it ended", [esc(STOP_TEXT.get(d.get("stop_reason"), d.get("stop_reason", ""))) for _, d in cols])
    head = "".join(f"<th>{esc(name)}</th>" for name, _ in cols)
    sentence = ""
    if agent and agent.get("lab_score") is not None:
        n = int(agent.get("rounds") or 0)
        sentence = (f"The agent submitted {n} solution{'s' if n != 1 else ''} and measured each one"
                    + ("; plain chat wrote its code in one shot with no way to run it." if plain else "."))
    elif agent:
        sentence = (f"The agent opened {int(agent.get('sources', 0))} sources and made {int(agent.get('tool_calls', 0))} tool calls"
                    + ("; plain chat answered in one shot with no way to look anything up." if plain else "."))
    foot = ""
    if expected_facts:
        foot = ('<div class="note">“Mentions the expected facts” is a string match against facts checked on Wikipedia or arXiv when the mission was written. '
                "It is not a grade of the answer.</div>")
    return f'<div class="arex"><p>{esc(sentence)}</p><table><tr><th></th>{head}</tr>{"".join(rows)}</table>{foot}</div>'


# ---- the improvement lab ----------------------------------------------------------------------------------------------------------

def _large(report) -> str:
    if report.timed_out:
        return "timed out"
    if report.large_seconds is None:
        return "-"
    return f"{report.large_seconds:.2f} s" if report.large_seconds < 10 else f"{report.large_seconds:.1f} s"


def rounds_html(rounds: Iterable) -> str:
    rounds = list(rounds)
    if not rounds:
        return '<div class="arex"><div class="note">No submissions yet.</div></div>'
    best = max(rounds, key=lambda r: (r.report.score, -r.number))
    w, h, pad = 560, 190, 34
    xs = [pad + (w - 2 * pad) * (i / (len(rounds) - 1) if len(rounds) > 1 else 0.5) for i in range(len(rounds))]
    ys = [h - pad - (h - 2 * pad) * max(0.0, min(1.0, r.report.score)) for r in rounds]
    grid = "".join(f'<line class="grid" x1="{pad}" x2="{w - pad}" y1="{h - pad - (h - 2 * pad) * g:.0f}" y2="{h - pad - (h - 2 * pad) * g:.0f}"/>'
                   f'<text x="4" y="{h - pad - (h - 2 * pad) * g + 4:.0f}">{g:.1f}</text>' for g in (0.0, 0.5, 1.0))
    pts = " ".join(f"{x:.0f},{y:.0f}" for x, y in zip(xs, ys))
    dots = "".join(f'<circle class="{"best" if r is best else ""}" cx="{x:.0f}" cy="{y:.0f}" r="5"><title>Round {r.number}: {r.report.score:.2f}</title></circle>'
                   f'<text x="{x:.0f}" y="{h - 10}" text-anchor="middle">{r.number}</text>' for r, x, y in zip(rounds, xs, ys))
    svg = (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Score by round" width="100%">{grid}'
           f'<polyline class="line" points="{pts}"/>{dots}<text x="{w - pad}" y="14" text-anchor="end">best: round {best.number}, {best.report.score:.2f}</text></svg>')
    rows = "".join(f"<tr><td>Round {r.number}</td><td>{r.report.score:.2f}</td><td>{r.report.passed} of {r.report.total}</td>"
                   f"<td>{_large(r.report)}</td></tr>" for r in rounds)
    return (f'<div class="arex">{svg}<table><tr><th>Round</th><th>Score</th><th>Tests passed</th><th>Large input</th></tr>{rows}</table>'
            f'<details><summary>Code of the best round (round {best.number}, score {best.report.score:.2f})</summary><pre>{esc(best.code)}</pre></details>'
            '<div class="note">Score = 0.7 x correctness + 0.3 x speed on a large input. The lab defines this score. It measures this task only.</div></div>')


# ---- notices and badges -----------------------------------------------------------------------------------------------------------

def notice_html(kind: str, text: str) -> str:
    return f'<div class="arex"><div class="notice {esc(kind)}">{esc(text)}</div></div>'


def final_notice_html(final: dict) -> str:
    reason = final.get("stop_reason", "")
    base = STOP_TEXT.get(reason, reason)
    detail = final.get("detail") or ""
    text = base if (not detail or detail in base) else f"{base} {detail}"
    return notice_html("good" if reason == "answered" else "warn" if reason in ("step_cap", "token_cap", "time_cap", "length", "cancelled") else "error", text)


def server_badge_html(url: str, model: Optional[str], mode: str) -> str:
    if model is None:
        server = f'<span class="badge bad">Model server not reachable at {esc(url)}</span>'
    else:
        server = f'<span class="badge good">{esc(model)} · {esc(url)}</span>'
    tools = ('<span class="badge warn">offline corpus: tools read a small bundled set of pages, no network</span>' if mode == "offline"
             else '<span class="badge">live web tools: Wikipedia, arXiv, fetch_url</span>')
    return f'<div class="arex">{server}{tools}</div>'


# ---- the plain chat panel, the verified answer, Continue ---------------------------------------------------------------------------------

def plain_html(plain: Optional[dict], live: dict, running: bool) -> str:
    if plain is None and not running and not (live.get("reasoning") or live.get("content")):
        return ""
    head = '<div class="note"><b>Plain chat</b>: the same model, no tools, one shot.</div>'
    if plain is None:
        body = ""
        if live.get("reasoning"):
            body += f'<details open><summary>Reasoning so far ({len(live["reasoning"]):,} characters)</summary><pre>{_tail(live["reasoning"], 800)}</pre></details>'
        body += f'<div class="answer">{esc(live.get("content", "")).replace(chr(10), "<br>") or "answering"}<span class="dots"></span></div>'
        return f'<div class="arex">{head}{body}</div>'
    reasoning = (f'<details><summary>Reasoning ({len(plain.get("reasoning") or ""):,} characters)</summary><pre>{_tail(plain.get("reasoning") or "", 1200)}</pre></details>'
                 if plain.get("reasoning") else "")
    answer = (f'<div class="answer">{esc(plain["answer"]).replace(chr(10), "<br>")}</div>' if plain.get("answer") else "")
    ended = esc(STOP_TEXT.get(plain.get("stop_reason"), plain.get("stop_reason", "")))
    return (f'<div class="arex">{head}{reasoning}{answer}<div class="note">{ended} {int(plain.get("completion_tokens") or 0):,} tokens in '
            f'{float(plain.get("seconds") or 0):.0f} s.</div></div>')


def expected_html(mission: dict) -> str:
    v = (mission or {}).get("verified") or {}
    if not (mission or {}).get("expected") or not v.get("quotes"):
        return ""
    quotes = "".join(f'<li>{safe_link(q.get("url", ""), q.get("url", "") or "computed here")}: “{esc(q.get("quote", ""))}”</li>' for q in v["quotes"])
    note = f'<div class="note">{esc(v["note"])}</div>' if v.get("note") else ""
    return (f'<div class="arex"><details><summary>Show the verified answer</summary><p>{esc(mission["expected"])}</p>'
            f'<div class="note">Read from the sources below on {esc(v.get("read", ""))}.</div><ul>{quotes}</ul>{note}</details></div>')


MAX_STEP_TOKENS = 32768


def continue_kwargs(reason: str, settings) -> Optional[dict]:
    """What Continue should extend for a stop reason, as keyword arguments for Session.resume, or None when there is nothing to continue."""
    get = settings.get if isinstance(settings, dict) else (lambda k, d=None: getattr(settings, k, d))
    if reason == "length":
        cur = int(get("step_tokens", 4096))
        new = min(cur * 2, MAX_STEP_TOKENS)
        return {"step_tokens": new} if new > cur else None
    if reason == "step_cap":
        return {"extra_steps": 8}
    if reason == "token_cap":
        return {"extra_tokens": 30000}
    if reason == "time_cap":
        return {"extra_seconds": 900.0}
    return None


def continue_label(reason: str, settings) -> str:
    kw = continue_kwargs(reason, settings)
    if not kw:
        return ""
    if "step_tokens" in kw:
        return f"Continue with a larger limit ({kw['step_tokens']:,} tokens per reply)"
    if "extra_steps" in kw:
        return f"Continue with {kw['extra_steps']} more steps"
    if "extra_tokens" in kw:
        return f"Continue with {kw['extra_tokens']:,} more tokens"
    return f"Continue with {int(kw['extra_seconds'] // 60)} more minutes"
