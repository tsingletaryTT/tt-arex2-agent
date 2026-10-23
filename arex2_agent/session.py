# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""One run of a mission: the agent, an optional plain-chat run of the same question beside it, and the snapshots the page draws.

`Session.events()` is a generator of snapshot dicts. The agent and the plain chat each run on their own thread and put events on one queue;
this generator is the only code that changes the state, so a snapshot is always consistent. Snapshots are sent when something structural
happens (a step, a tool result, an answer) and, for streamed text, at most every 0.3 s. While the model is silent a heartbeat snapshot goes out
every second so the clock and the "thinking" marker keep moving. `cancel` is the stop button. `resume` continues a run that was cut off or hit a limit.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass, fields
from typing import Callable, Optional

from . import agent, chat, evidence, lab, missions, tools

THROTTLE_S = 0.3
HEARTBEAT_S = 1.0


@dataclass
class Settings:
    base_url: str = chat.DEFAULT_URL
    model: str = ""
    mode: str = "online"
    max_steps: int = 12
    max_tokens: int = 60000
    max_seconds: float = 1800.0
    step_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 0.95
    compare: bool = False

    def clean(self) -> "Settings":
        d = Settings()

        def num(value, default, lo, hi, cast):
            try:
                v = cast(value)
            except (TypeError, ValueError):
                return default
            if v != v:                                  # NaN
                return default
            return max(lo, min(hi, v))

        return Settings(base_url=(self.base_url or d.base_url).strip(), model=(self.model or "").strip(),
                        mode=self.mode if self.mode in ("online", "offline") else "online",
                        max_steps=num(self.max_steps, d.max_steps, 1, 40, int), max_tokens=num(self.max_tokens, d.max_tokens, 1000, 400000, int),
                        max_seconds=num(self.max_seconds, d.max_seconds, 30.0, 14400.0, float), step_tokens=num(self.step_tokens, d.step_tokens, 256, 32768, int),
                        temperature=num(self.temperature, d.temperature, 0.0, 2.0, float), top_p=num(self.top_p, d.top_p, 0.05, 1.0, float),
                        compare=bool(self.compare))


class Session:
    def __init__(self, mission: dict, settings: Settings, cancel: Optional[threading.Event] = None, toolbox=None,
                 client_factory: Callable = chat.ChatClient, clock: Callable = time.monotonic):
        self.mission, self.settings = mission, settings.clean()
        self.cancel = cancel or threading.Event()
        self._toolbox, self._factory, self._clock = toolbox, client_factory, clock
        self.run: Optional[agent.AgentRun] = None
        self.steps: list = []
        self.claims: list = []
        self.cited: dict = {"opened": 0, "never_opened": []}
        self.final: Optional[dict] = None
        self.agent_stats: Optional[dict] = None
        self.plain: Optional[dict] = None
        self.plain_live = {"reasoning": "", "content": ""}
        self.running = False
        self.tool_calls = 0
        self._computed: list = []
        self._used = 0.0
        self._seg_start: Optional[float] = None
        self._started_plain = False

    # -- setup ---------------------------------------------------------------------------------------------------------------
    @property
    def is_lab(self) -> bool:
        return self.mission.get("kind") == "lab"

    def _make_toolbox(self):
        if self._toolbox is not None:
            return self._toolbox
        if self.is_lab:
            return lab.LabToolbox(lab.TASKS[self.mission["task"]])
        if self.settings.mode == "offline":
            return missions.offline_toolbox()
        return tools.Toolbox(env=dict(os.environ))

    def _client(self):
        return self._factory(self.settings.base_url, model=self.settings.model or None)

    def _budget(self) -> agent.Budget:
        s = self.settings
        return agent.Budget(max_steps=s.max_steps, max_tokens=s.max_tokens, max_seconds=s.max_seconds, step_tokens=s.step_tokens)

    # -- the snapshot --------------------------------------------------------------------------------------------------------
    def _elapsed(self) -> float:
        return self._used + ((self._clock() - self._seg_start) if self._seg_start is not None and self.running else 0.0)

    def _sources(self) -> list:
        if self.run is None:
            return []
        for _ in range(5):
            try:
                return list(self.run.sources.values())
            except RuntimeError:                         # the agent thread added one while this list was built
                continue
        return []

    def snapshot(self) -> dict:
        run = self.run
        budget = self._budget().__dict__.copy()
        tb = self._toolbox if isinstance(self._toolbox, lab.LabToolbox) else getattr(self, "_lab_toolbox", None)
        return {"mission": self.mission, "settings": self.settings, "steps": [dict(s) for s in self.steps],
                "meters": {"budget": budget, "steps": run.step_count if run else 0, "tokens": run.completion_tokens if run else 0, "seconds": self._elapsed()},
                "sources": self._sources(), "claims": list(self.claims), "cited": dict(self.cited), "final": self.final, "answer": (self.final or {}).get("answer", ""),
                "agent": self.agent_stats, "plain": self.plain, "plain_live": dict(self.plain_live), "rounds": list(tb.rounds) if tb else [],
                "can_resume": bool(run and run.can_resume and self.final), "running": self.running}

    # -- running --------------------------------------------------------------------------------------------------------------
    def events(self):
        """Run the mission and yield snapshots until it ends."""
        self.running, self._used, self._seg_start = True, 0.0, self._clock()
        self.final, self.claims, self.agent_stats = None, [], None
        yield self.snapshot()
        try:
            client = self._client()
        except ValueError as e:
            self._finish_without_run(f"That is not a usable model server address: {e}")
            yield self.snapshot()
            return
        tb = self._make_toolbox()
        self._lab_toolbox = tb if isinstance(tb, lab.LabToolbox) else None
        system = lab.LAB_SYSTEM_PROMPT if self.is_lab else agent.SYSTEM_PROMPT
        s = self.settings
        self.run = agent.AgentRun(client, tb, self.mission["prompt"], self._budget(), system=system, clock=self._clock, cancel=self.cancel.is_set,
                                  temperature=s.temperature, top_p=s.top_p)
        yield from self._pump(self.run.events(), with_plain=s.compare)

    def resume(self, step_tokens: Optional[int] = None, extra_steps: int = 0, extra_tokens: int = 0, extra_seconds: float = 0.0):
        if self.run is None or not self.run.can_resume:
            yield self.snapshot()
            return
        self.running, self._seg_start, self.final = True, self._clock(), None
        self.cancel.clear()
        if step_tokens:
            self.settings.step_tokens = step_tokens
        self.settings.max_steps += extra_steps
        self.settings.max_tokens += extra_tokens
        self.settings.max_seconds += extra_seconds
        yield from self._pump(self.run.resume(step_tokens=step_tokens, extra_steps=extra_steps, extra_tokens=extra_tokens, extra_seconds=extra_seconds), with_plain=False)

    def _finish_without_run(self, detail: str):
        self.running = False
        self.final = {"type": "final", "stop_reason": "server_down", "detail": detail, "answer": "", "steps": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0, "sources": 0}
        self.agent_stats = self._stats_from_final()

    def _pump(self, agent_events, with_plain: bool):
        q: queue.Queue = queue.Queue()

        def agent_worker():
            try:
                for ev in agent_events:
                    q.put(("agent", ev))
            except Exception as e:                                       # nothing here may leave the page waiting
                q.put(("agent_error", e))
            finally:
                q.put(("agent_done", None))

        threading.Thread(target=agent_worker, daemon=True).start()
        plain_running = False
        if with_plain and not self._started_plain:
            self._started_plain, plain_running = True, True
            self.plain, self.plain_live = None, {"reasoning": "", "content": ""}
            threading.Thread(target=self._plain_worker, args=(q,), daemon=True).start()
        agent_done, last_yield = False, 0.0
        while not (agent_done and not plain_running):
            try:
                kind, payload = q.get(timeout=HEARTBEAT_S)
            except queue.Empty:
                yield self.snapshot()
                last_yield = self._clock()
                continue
            structural = True
            if kind == "agent":
                structural = self._apply(payload)
            elif kind == "agent_error":
                self._finalize({"type": "final", "stop_reason": "server_down", "detail": f"Unexpected error: {payload}", "answer": "", "steps": self.run.step_count,
                                "prompt_tokens": self.run.prompt_tokens, "completion_tokens": self.run.completion_tokens, "seconds": self._elapsed(), "sources": len(self.run.sources)})
            elif kind == "agent_done":
                agent_done = True
            elif kind == "plain_delta":
                self.plain_live[payload[0]] += payload[1]
                structural = False
            elif kind == "plain_done":
                self.plain, plain_running = payload, False
            now = self._clock()
            if structural or now - last_yield >= THROTTLE_S or (agent_done and not plain_running):
                if agent_done and not plain_running:
                    self.running = False
                    self._used = self._elapsed() if self.running else self._used
                yield self.snapshot()
                last_yield = now
        self.running = False

    def _plain_worker(self, q: queue.Queue):
        s = self.settings
        res = None
        try:
            res = agent.plain_chat(self._client(), self.mission["prompt"], step_tokens=s.step_tokens, temperature=s.temperature, top_p=s.top_p,
                                   on_delta=lambda kind, text: q.put(("plain_delta", (kind, text))), cancel=self.cancel.is_set)
            stats = {"answer": res.answer, "reasoning": res.reasoning, "completion_tokens": res.completion_tokens or (len(res.answer) + len(res.reasoning)) // 4,
                     "prompt_tokens": res.prompt_tokens, "seconds": res.seconds, "steps": 1, "sources": 0, "tool_calls": 0, "claims": None,
                     "stop_reason": res.stop_reason, "detail": res.detail, "lab_score": None}
            if self.is_lab and res.answer:
                stats["lab_score"] = lab.score_solution(lab.TASKS[self.mission["task"]], res.answer).score
        except Exception as e:
            stats = {"answer": "", "completion_tokens": 0, "seconds": 0.0, "steps": 1, "sources": 0, "tool_calls": 0, "claims": None,
                     "stop_reason": "server_down", "detail": f"Unexpected error: {e}", "lab_score": None}
        q.put(("plain_done", stats))

    # -- applying agent events -----------------------------------------------------------------------------------------------
    def _apply(self, ev: dict) -> bool:
        """Change the state for one agent event. Returns True when the snapshot should go out at once."""
        t = ev["type"]
        if t == "step":
            self.steps = [s for s in self.steps if s["n"] != ev["n"]]
            self.steps.append({"n": ev["n"], "reasoning": "", "content": "", "tools": [], "live": True, "seconds": 0.0, "completion_tokens": None})
            return True
        if t == "delta":
            self.steps[-1][ev["kind"]] += ev["text"]
            return False
        if t == "model":
            self.steps[-1].update(reasoning=ev["reasoning"], content=ev["content"], live=False, seconds=ev["seconds"],
                                  completion_tokens=ev["completion_tokens"], finish_reason=ev["finish_reason"])
            return True
        if t == "tool":
            self.steps[-1]["tools"].append(ev)
            if ev["executed"]:
                self.tool_calls += 1
                if ev["ok"] and ev["name"] in ("calculator", "run_python"):
                    self._computed.append(ev["text"])
            return True
        if t == "final":
            self._finalize(ev)
            return True
        return False

    def _finalize(self, ev: dict):
        for s in self.steps:
            s["live"] = False
        if ev["stop_reason"] == "length" and self.steps:
            self.steps[-1]["cut_off"] = True
        self.final = ev
        self.running = False
        self._used = ev.get("seconds", self._used)
        if ev.get("answer") and not self.is_lab:
            self.claims = evidence.check_claims(ev["answer"], self._sources(), self._computed)
            self.cited = evidence.cited_addresses(ev["answer"], self._sources())
        self.agent_stats = self._stats_from_final()

    def _stats_from_final(self) -> dict:
        f = self.final or {}
        tb = getattr(self, "_lab_toolbox", None)
        best = tb.best if tb else None
        return {"steps": f.get("steps", 0), "completion_tokens": f.get("completion_tokens", 0), "seconds": f.get("seconds", 0.0), "sources": f.get("sources", 0),
                "tool_calls": self.tool_calls, "answer": f.get("answer", ""), "claims": evidence.summarize(self.claims) if self.claims else None,
                "stop_reason": f.get("stop_reason", ""), "detail": f.get("detail", ""), "lab_score": best.report.score if best else None}
