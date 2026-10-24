#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Take the README screenshots of the page running against the REAL AREX-2 (the `episod/arex-2-p300` package on 2 chips).

    gozer run --chips 2 --who "you:arex2" --reason "screenshots" -- .venv/bin/python serve/serve_arex2.py --ui
    python tools/screenshots_real.py http://127.0.0.1:7872/    # in any Python that has the `playwright` package; uses /usr/bin/google-chrome

It writes docs/settings-light.png, docs/lab-edit-distance-light.png, docs/research-done-light.png, docs/research-done-dark.png and
docs/research-live-light.png. Each is one run of the real model and not a benchmark, and the model's replies differ a little from run to run.

Before each screenshot the served model directory's path and the user's home directory are replaced in the page text, and the script stops
if a home path or a stand-in badge is still visible. Buttons are matched by their exact names: a match on "Run" alone also finds the
"Settings for every run" header and "Download this run".
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs"
CHROME = "/usr/bin/google-chrome"
# The served model directory is a path under the user's home; show `model-dir`. Any other home path becomes `~`.
REDACT = """() => {
  const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT); let n;
  while ((n = w.nextNode())) {
    n.nodeValue = n.nodeValue.replace(/\\/home\\/[^\\s·]*\\/model-dir/g, "model-dir").replace(/\\/home\\/[^\\s·]*/g, "~");
  }
  for (const e of document.querySelectorAll("input, textarea")) { e.value = e.value.replace(/\\/home\\/[^\\s·]*\\/model-dir/g, "model-dir"); }
}"""


def shot(page, name: str) -> None:
    page.evaluate(REDACT)
    text = page.inner_text("body")
    assert "/home/" not in text, f"a home path is still on the page for {name}"
    assert "SCRIPTED-FAKE-MODEL" not in text, f"the page points at the stand-in for {name}"
    page.screenshot(path=str(OUT / name), full_page=True)
    print("wrote", name, flush=True)


def fresh(browser, url: str, scheme: str):
    ctx = browser.new_context(viewport={"width": 1400, "height": 1000}, color_scheme=scheme)
    page = ctx.new_page()
    page.goto(url)
    page.wait_for_selector("text=Settings for every run", timeout=60000)
    return ctx, page


def wait_text(page, marker: str, ms: int) -> None:
    page.wait_for_function("(m) => document.body.innerText.includes(m)", arg=marker, timeout=ms)


def wait_plain_chat(page, ms: int) -> None:
    """The plain-chat panel finishes after the agent. Its last line ends 'N tokens in S s.' whether it answered or ran out of its token limit."""
    page.wait_for_function(
        "() => { const t = document.body.innerText; const i = t.indexOf('Plain chat:');"
        " return i >= 0 && /[0-9,]+ tokens in [0-9.]+ s/.test(t.slice(i)); }", timeout=ms)


def wait_settled(page, ms: int = 120000) -> None:
    """After a run the page unlocks Run, Download this run and Copy answer. A screenshot taken before that shows them greyed.
    Both tabs have these buttons, so the check uses the visible ones (a role query skips the hidden tab)."""
    expect(page.get_by_role("button", name="Download this run", exact=True)).to_be_enabled(timeout=ms)
    expect(page.get_by_role("button", name="Copy answer", exact=True)).to_be_enabled(timeout=ms)
    expect(page.get_by_role("button", name="Run", exact=True)).to_be_enabled(timeout=ms)
    # The page unlocks the buttons in steps as the plain-chat panel and the scorecard finish. Wait, then check again, so a re-lock is caught.
    page.wait_for_timeout(3000)
    expect(page.get_by_role("button", name="Download this run", exact=True)).to_be_enabled(timeout=ms)
    expect(page.get_by_role("button", name="Copy answer", exact=True)).to_be_enabled(timeout=ms)
    expect(page.get_by_role("button", name="Run", exact=True)).to_be_enabled(timeout=ms)


def pick_mission(page, option: str, nth: int) -> None:
    page.locator("input[role='listbox'], input[aria-label='Mission']").nth(nth).click()
    page.get_by_role("option", name=option, exact=True).click()


def run_button(page):
    return page.get_by_role("button", name="Run", exact=True)


def main(url: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME, headless=True, args=["--no-sandbox"])
        # 1. the settings, open, on the research tab
        ctx, page = fresh(browser, url, "light")
        page.get_by_text("Settings for every run").click()
        page.wait_for_selector("text=Connection")
        shot(page, "settings-light.png")
        ctx.close()
        # 2. the improvement lab: the model tests its own code and submits it to the sandbox (settings closed)
        ctx, page = fresh(browser, url, "light")
        page.get_by_role("tab", name="Improvement lab").click()
        pick_mission(page, "Edit distance between two strings", 1)
        t0 = time.time()
        run_button(page).last.click()
        wait_text(page, "Lab score", 2400000)
        wait_plain_chat(page, 900000)
        wait_settled(page)
        print("lab done in", round(time.time() - t0), "s", flush=True)
        shot(page, "lab-edit-distance-light.png")
        ctx.close()
        # 3. a research mission with the offline corpus, light and dark
        for scheme in ("light", "dark"):
            ctx, page = fresh(browser, url, scheme)
            page.get_by_text("Settings for every run").click()
            page.get_by_label("Offline corpus (no network)").check()
            page.get_by_text("Settings for every run").click()          # close it again
            t0 = time.time()
            run_button(page).first.click()
            wait_text(page, "Sources read", 1200000)
            wait_plain_chat(page, 900000)
            wait_settled(page)
            print(scheme, "research done in", round(time.time() - t0), "s", flush=True)
            shot(page, f"research-done-{scheme}.png")
            ctx.close()
        # 4. a research mission with live tools (the page's default)
        ctx, page = fresh(browser, url, "light")
        pick_mission(page, "Who wrote the Transformer paper", 0)
        t0 = time.time()
        run_button(page).first.click()
        wait_text(page, "Sources read", 1200000)
        wait_plain_chat(page, 900000)
        wait_settled(page)
        print("live research done in", round(time.time() - t0), "s", flush=True)
        shot(page, "research-live-light.png")
        ctx.close()
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7872/"))
