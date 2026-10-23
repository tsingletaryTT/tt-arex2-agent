# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""The AREX-2 research page: a model trained to work in loops, run as a loop you can watch.

    python demo/app.py                        # talks to AREX2_URL, default http://127.0.0.1:8000/v1
    python serve/serve_arex2.py --ui          # boots the model on leased chips, then runs this page

Two kinds of mission. A research mission makes the agent look things up, read sources, calculate and answer with citations, and the page checks the
answer's claims against the text it actually retrieved. An improvement mission gives it a coding task with hidden tests: it submits code, reads a measured
score report (failed tests, runtime, errors) and revises, which is the propose, measure, reflect and revise loop AREX-2 was trained on. Beside each run the
page can ask the same model the same question once with no tools, so the difference is visible.

The page has no login. Bind it to 0.0.0.0 only on a network you trust: a visitor can use the model and its tools (web fetches go through a guard that
refuses private addresses, and code runs in a sandbox with no network).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr  # noqa: E402

from arex2_agent import chat, export, lab, missions, session, views  # noqa: E402

CUSTOM = "(write your own)"
MAX_PROMPT = 4000
EXPORT_DIR = Path(tempfile.gettempdir()) / "arex2-lab-exports"        # where "Download this run" files are written (the newest few are kept)
LAB_PREFIX = "Improvement lab: "
HEADER = ("<div class='arex'><div class='head'><h1>AREX-2 Research Lab</h1>"
          "<p>BAAI's 27B agent model, running on Tenstorrent Blackhole. It is trained to work in loops: look things up, run code, measure its own result, revise. "
          "This page runs that loop and shows every step. It is not a chat box.</p></div></div>")
ABOUT = """
### What you are looking at

**AREX-2** is an agent model: it learns to improve a solution over several rounds (propose, measure, reflect, revise) on tasks where the result can be
checked, and the same habit carries over to deep research. A chat box shows one reply. This page shows the loop.

**Research missions.** The agent searches Wikipedia and arXiv, reads pages, calculates and runs code, then answers with sources. Every source it opened is on
the *evidence board*, with the part of the page it read. Each claim in the answer is checked against that text: *backed*, *computed* (the numbers come from a
calculation) or *not backed*. This is a match of words and numbers. It does not tell you the source is right.

**Improvement lab.** The agent gets a coding task with hidden tests and a large input. It submits code, the lab runs it in a sandbox and returns a measured
report: which tests failed with what input, how long the large input took, any error. Then it revises. The chart shows the score per round. The score
(0.7 x correctness + 0.3 x speed) is defined by the lab and measures that task only.

**Agent against plain chat.** With the box ticked, the same question goes to the same model once, with no tools, at the same settings. The scorecard
compares sources, steps, tokens and time. For presets it also counts how many of the facts checked at build time each answer mentions (a string match,
not a grade).

### What it costs

On this hardware the model decodes at roughly 14 tokens a second per request when four requests run at once (measured with the drafter off and sampling on
the host). A thinking step can take minutes and a mission can take tens of minutes. The meters show steps, tokens and time against your budgets. A reply can
be cut off at the per-step limit: the page says the thinking budget ran out and offers to continue with a larger one.

### Offline corpus

*Offline corpus* serves the tools from a small bundled set of Wikipedia and arXiv pages (Wikipedia text under CC BY-SA 4.0), with no network. It answers the
preset missions. It is not a search engine.

### Limits

Tool results are untrusted text and are shown as plain text. The claim check is string matching. The lab demonstrates a loop. It is not an adversarial benchmark. The
page has no login.
"""


# ---- pure helpers (tested without a browser) --------------------------------------------------------------------------------------------

def preset_choices(kinds, strip_prefix: str = "") -> list:
    """(label, id) pairs for a mission picker. `strip_prefix` is cut from the labels only, so a tab that already says what its missions are does not repeat it."""
    def label(title: str) -> str:
        return title[len(strip_prefix):] if strip_prefix and title.startswith(strip_prefix) else title
    return [(label(m["title"]), m["id"]) for m in missions.all_missions() if m["kind"] in kinds] + [(CUSTOM, CUSTOM)]


def mission_from_inputs(preset_id: Optional[str], text: str, allowed_kinds=("research", "compute")) -> dict:
    """The mission to run. A preset whose text was edited counts as a custom mission, so the expected facts are not applied to a different question."""
    text = (text or "").strip()
    if not text:
        raise gr.Error("Write a mission, or pick a preset.")
    if len(text) > MAX_PROMPT:
        raise gr.Error(f"The mission is too long ({len(text)} characters; the limit is {MAX_PROMPT}).")
    if preset_id and preset_id != CUSTOM:
        try:
            m = missions.get(preset_id)
        except KeyError:
            m = None
        if m is not None and m["kind"] in allowed_kinds and m["prompt"].strip() == text:
            return m
    return {"id": "custom", "title": "Your mission", "kind": "research", "prompt": text, "expected": "", "facts": [], "derived": [], "needs": [], "verified": {}}


def settings_from_inputs(url, model, mode, max_steps, max_tokens, minutes, step_tokens, temperature, top_p, compare) -> session.Settings:
    return session.Settings(base_url=url or chat.DEFAULT_URL, model=model or "", mode="offline" if str(mode).startswith("Offline") else "online",
                            max_steps=max_steps, max_tokens=max_tokens, max_seconds=(minutes or 0) * 60, step_tokens=step_tokens, temperature=temperature,
                            top_p=top_p, compare=bool(compare)).clean()


def status_html(snap: dict) -> str:
    if snap["running"]:
        steps = snap["steps"]
        what = f"Step {steps[-1]['n']} is running." if steps else "Starting."
        return views.notice_html("", what + " Use Stop to end the run.")
    if snap["final"]:
        return views.final_notice_html(snap["final"])
    return ""


def render(snap: dict, lab_page: bool = False) -> dict:
    """Everything the page shows for one snapshot, as HTML strings."""
    meters = snap["meters"]
    m = snap["mission"]
    final = snap["final"]
    finished = bool(final) and not snap["running"]
    answer = views.answer_html(snap["answer"], snap["claims"]) if snap["answer"] else ""
    if finished and not lab_page:
        answer += views.expected_html(m)
    if lab_page:
        side = views.rounds_html(snap["rounds"])
    else:
        side = views.evidence_html(snap["sources"], snap["claims"], snap["cited"])
    score = ""
    if finished and snap["agent"]:
        score = views.scorecard_html(snap["agent"], snap["plain"], (m.get("facts") if m.get("kind") != "lab" else None) or None)
    plain = views.plain_html(snap["plain"], snap["plain_live"], running=snap["running"] and bool(snap["settings"].compare) and snap["plain"] is None)
    label = views.continue_label(final["stop_reason"], snap["settings"]) if (finished and snap["can_resume"]) else ""
    return {"notice": status_html(snap), "meters": views.meters_html(meters["budget"], meters["steps"], meters["tokens"], meters["seconds"]),
            "timeline": views.timeline_html(snap["steps"]), "answer": answer, "side": side, "score": score, "plain": plain, "continue_label": label}


def check_server(url: str, mode: str) -> str:
    url = (url or chat.DEFAULT_URL).strip()
    try:
        names = chat.ChatClient(url, connect_timeout=3, read_timeout=5).models()
        model = names[0] if names else None
    except (chat.ChatError, ValueError):
        model = None
    return views.server_badge_html(url, model, "offline" if str(mode).startswith("Offline") else "online")


def tools_label(mode) -> str:
    """The tools the next run will use, in words, for the line beside each Run button."""
    return "Tools: offline corpus (no network)" if str(mode).startswith("Offline") else "Tools: live web tools"


def lock_run():
    """While a run is going: Run is off, Stop is on, and the files of the last run are not offered (the run in progress will replace them)."""
    return gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False)


def unlock_run():
    return gr.update(interactive=True), gr.update(interactive=False)


def prepare_export(sess):
    """After a run ends: write its export and turn on Download and Copy. Nothing is offered for no session, a run still going, or a run with no end."""
    off = (gr.update(value=None, interactive=False), gr.update(interactive=False), "")
    if sess is None:
        return off
    snap = sess.snapshot()
    if snap.get("running") or not snap.get("final"):
        return off
    path = export.write_export(snap, EXPORT_DIR)
    if path is None:
        return off
    answer = snap.get("answer") or ""
    return gr.update(value=str(path), interactive=True), gr.update(interactive=bool(answer)), answer


COPY_JS = """(t) => { const text = t || '';
  if (navigator.clipboard && window.isSecureContext) { navigator.clipboard.writeText(text); }
  else { const a = document.createElement('textarea'); a.value = text; document.body.appendChild(a); a.select(); document.execCommand('copy'); a.remove(); }
  return [text]; }"""


def copied(text):
    gr.Info("Answer copied")


# ---- the page --------------------------------------------------------------------------------------------------------------------------

def build_app(default_url: Optional[str] = None) -> gr.Blocks:
    url0 = default_url or os.environ.get("AREX2_URL", chat.DEFAULT_URL)
    with gr.Blocks(title="AREX-2 Research Lab") as demo:
        gr.HTML(HEADER)
        badge = gr.HTML()
        # A control sits in the container whose scope it has. Everything here applies to every run on both tabs; what belongs to one tab (the mission,
        # the plain-chat box, Run and Stop, the tools line, the files of a finished run) is inside that tab.
        with gr.Accordion("Settings for every run", open=False):
            with gr.Column(elem_classes="settings-group"):
                gr.Markdown("**Connection**")
                with gr.Row():
                    url = gr.Textbox(label="Model server address", value=url0, scale=4)
                    check = gr.Button("Check the server", size="sm", scale=1, min_width=170, elem_classes="inline-action")
                with gr.Row():
                    model = gr.Textbox(label="Model name (blank: ask the server)", value="", scale=2)
                    mode = gr.Radio(["Live web tools", "Offline corpus (no network)"], value="Live web tools", label="Tools", scale=2)
            with gr.Column(elem_classes="settings-group"):
                gr.Markdown("**Limits**")
                with gr.Row():
                    max_steps = gr.Slider(1, 40, value=12, step=1, label="Steps")
                    max_tokens = gr.Slider(2000, 200000, value=60000, step=1000, label="Tokens generated in total")
                    minutes = gr.Slider(1, 240, value=30, step=1, label="Minutes")
            with gr.Column(elem_classes="settings-group"):
                gr.Markdown("**Sampling**")
                with gr.Row():
                    step_tokens = gr.Slider(512, 32768, value=4096, step=256, label="Tokens per reply, thinking included")
                    temperature = gr.Slider(0.0, 1.5, value=0.6, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
        shared = [url, model, mode, max_steps, max_tokens, minutes, step_tokens, temperature, top_p]
        demo.load(check_server, [url, mode], badge)
        check.click(check_server, [url, mode], badge, queue=False)
        mode.change(check_server, [url, mode], badge, queue=False)     # the badge says which tools the next run uses, so it follows the radio

        def make_tab(kinds, lab_page: bool, default_compare: bool):
            choices = preset_choices(kinds, LAB_PREFIX if lab_page else "")
            first = missions.get(choices[0][1]) if choices[0][1] != CUSTOM else None
            with gr.Row():
                preset = gr.Dropdown(choices, value=choices[0][1], label="Mission", scale=2)
                compare = gr.Checkbox(value=default_compare, label="Also ask plain chat (same model, no tools, one shot)", scale=2)
            prompt = gr.Textbox(value=first["prompt"] if first else "", lines=6 if lab_page else 4, label="Mission text",
                                interactive=not lab_page, info="Edit it freely; an edited preset runs as your own mission.")
            with gr.Row():
                run_btn = gr.Button("Run", variant="primary", scale=3)
                stop_btn = gr.Button("Stop", interactive=False, scale=1)               # on only while a run is going
                cont_btn = gr.Button("Continue", visible=False)
                tools_note = gr.Markdown(tools_label("Live web tools"), elem_classes="tools-note")      # what Run will use: the radio in the settings, said where the run starts
            with gr.Row(equal_height=True):
                notice = gr.HTML(scale=1)
                download = gr.DownloadButton("Download this run", interactive=False, size="sm", scale=0, min_width=160)
                copy_btn = gr.Button("Copy answer", interactive=False, size="sm", scale=0, min_width=130)
            meters = gr.HTML()
            answer_text = gr.Textbox(visible=False)                                      # the plain text of the answer, for Copy
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### The loop, step by step")
                    timeline = gr.HTML(views.timeline_html([]))
                with gr.Column(scale=1):
                    gr.Markdown("#### The result" if not lab_page else "#### Score by round")
                    if lab_page:
                        side = gr.HTML(views.rounds_html([]))
                        answer = gr.HTML()
                    else:
                        answer = gr.HTML()
                        gr.Markdown("#### Evidence board")
                        side = gr.HTML(views.evidence_html([], [], {"opened": 0, "never_opened": []}))
                    gr.Markdown("#### Agent against plain chat")
                    score, plain = gr.HTML(), gr.HTML()
            state = gr.State(None)
            outs = [state, notice, meters, timeline, answer, side, score, plain, cont_btn]

            def load_preset(pid):
                if pid == CUSTOM or not pid:
                    return gr.update(value="", interactive=True)
                return gr.update(value=missions.get(pid)["prompt"], interactive=not lab_page)
            preset.change(load_preset, preset, prompt, queue=False)

            def emit(sess, snap):
                r = render(snap, lab_page)
                cont = gr.update(visible=bool(r["continue_label"]), value=r["continue_label"] or "Continue")
                return sess, r["notice"], r["meters"], r["timeline"], r["answer"], r["side"], r["score"], r["plain"], cont

            def guarded(sess, snaps):
                try:
                    for snap in snaps:
                        yield emit(sess, snap)
                except Exception as e:                                          # the page shows a plain message and not a traceback
                    print(f"page error: {type(e).__name__}: {e}", file=sys.stderr)
                    yield (sess, views.notice_html("error", f"Something went wrong in the page: {type(e).__name__}: {e}"), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                           gr.skip(), gr.skip(), gr.update(visible=False))

            def run(pid, text, *rest):
                *vals, _old = rest
                mission = mission_from_inputs(pid, text, kinds)
                if lab_page and mission["kind"] != "lab":
                    raise gr.Error("Pick one of the improvement tasks.")
                settings = settings_from_inputs(*vals[:9], vals[9])
                sess = session.Session(mission, settings)
                yield from guarded(sess, sess.events())

            def resume(sess):
                if sess is None or not sess.final:
                    yield (sess, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.update(visible=False))
                    return
                kw = views.continue_kwargs(sess.final["stop_reason"], sess.settings)
                if not kw:
                    yield emit(sess, sess.snapshot())
                    return
                yield from guarded(sess, sess.resume(**kw))

            def stop(sess):
                if sess is not None:
                    sess.cancel.set()

            # Run and Continue: lock the buttons, run, unlock (also when the run was refused or failed), then offer the files of the finished run.
            def chain(button, fn, inputs, api_name):
                (button.click(lock_run, None, [run_btn, stop_btn, download, copy_btn], queue=False, api_name=False)
                 .then(fn, inputs, outs, api_name=api_name, concurrency_limit=3)
                 .then(unlock_run, None, [run_btn, stop_btn], queue=False, api_name=False)
                 .then(prepare_export, [state], [download, copy_btn, answer_text], queue=False, api_name=False))
            chain(run_btn, run, [preset, prompt, *shared, compare, state], "run_lab" if lab_page else "run_research")
            chain(cont_btn, resume, [state], "resume_lab" if lab_page else "resume_research")
            stop_btn.click(stop, [state], None, queue=False)
            copy_btn.click(copied, [answer_text], None, js=COPY_JS, queue=False, api_name=False)
            return preset, tools_note

        notes = []
        with gr.Tabs():
            with gr.Tab("Research missions"):
                notes.append(make_tab(("research", "compute"), False, True)[1])
            with gr.Tab("Improvement lab"):
                notes.append(make_tab(("lab",), True, True)[1])
        for note in notes:
            demo.load(tools_label, mode, note)
            mode.change(tools_label, mode, note, queue=False, api_name=False)
        with gr.Accordion("How to read this page", open=False):
            gr.Markdown(ABOUT)
    return demo


def launch(host: str = "127.0.0.1", port: int = 7872, url: Optional[str] = None, prevent_thread_lock: bool = False):
    demo = build_app(url).queue(max_size=16, default_concurrency_limit=3)
    return demo.launch(server_name=host, server_port=port, show_error=True, css=views.CSS, prevent_thread_lock=prevent_thread_lock)


if __name__ == "__main__":
    launch(os.environ.get("AREX2_UI_HOST", "127.0.0.1"), int(os.environ.get("AREX2_UI_PORT", "7872")))
