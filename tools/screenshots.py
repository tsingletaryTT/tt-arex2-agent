#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Take screenshots of the page running against tools/demo_stack.py, which serves a SCRIPTED stand-in model. It is not AREX-2.

The images are written to docs/stand-in-check/, which git ignores, so a stand-in screenshot cannot end up in the README by accident. The README's
screenshots come from the real model (see "What ran on the real model").

    python tools/demo_stack.py &                       # in the project's venv
    python tools/screenshots.py http://127.0.0.1:7899/  # in any Python that has the `playwright` package; uses /usr/bin/google-chrome
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs" / "stand-in-check"
CHROME = "/usr/bin/google-chrome"


def pick_offline(page):
    page.get_by_text("Model server and budgets").click()
    page.get_by_label("Offline corpus (no network)").check()


def finish(page, timeout_ms=180000, marker="Sources read"):
    """The scorecard appears only when the agent has finished (the plain chat panel finishes earlier and shares the 'Finished' wording)."""
    page.wait_for_function("(m) => document.body.innerText.includes(m)", arg=marker, timeout=timeout_ms)


def main(url: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME, headless=True, args=["--no-sandbox"])
        for scheme in ("light", "dark"):
            ctx = browser.new_context(viewport={"width": 1400, "height": 1000}, color_scheme=scheme)
            page = ctx.new_page()
            page.goto(url)
            page.wait_for_selector("text=SCRIPTED-FAKE-MODEL", timeout=30000)
            pick_offline(page)
            page.get_by_role("button", name="Run").first.click()
            page.wait_for_selector("text=Step 2", timeout=60000)
            if scheme == "light":
                page.screenshot(path=str(OUT / "research-running.png"), full_page=True)
            finish(page)
            page.screenshot(path=str(OUT / f"research-done-{scheme}.png"), full_page=True)
            if scheme == "light":
                page.get_by_role("tab", name="Improvement lab").click()
                page.locator("input[role='listbox'], input[aria-label='Mission']").nth(1).click()
                page.get_by_role("option", name="Improvement lab: Fewest coins for an amount").click()
                page.get_by_role("button", name="Run").last.click()
                page.wait_for_function("() => (document.body.innerText.match(/Round \\d/g) || []).length >= 3", timeout=240000)
                finish(page, 240000, marker="Lab score")
                page.screenshot(path=str(OUT / "lab-done-light.png"), full_page=True)
            ctx.close()
        browser.close()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7899/"))
