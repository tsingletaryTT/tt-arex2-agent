# tt-arex2-agent project log

## Original prompt (2026-10-22, from the tt-orchard session)
"build a gradio app for the model (compat with tt-discolike) that really shows off what it's capable of and disambiguates it from simple chat."
The model is BAAI/AREX-2, brought up and packaged by tt-orchard (`arex-2-p300`, 2 chips). Local repo only: nothing was pushed, uploaded or published.

## Key decisions
- AREX-2's card says it is trained to improve a solution over rounds (propose, measure, reflect, revise) on checkable tasks, plus deep research. So the page has two mission kinds:
  research missions (tools, sources, a claim check) and an improvement lab (code, a sandbox, a measured score per round). A chat box shows neither.
- Disambiguation from chat: every run can start a plain-chat run of the same question (same model, no tools, one shot, same settings) in parallel. The scorecard counts sources,
  claims backed, steps, tokens and time. For presets it also counts expected facts found in each answer, labelled as a string match and not a grade. No text says one answer is better.
- The evidence check matches words and numbers against retrieved text (a stretch of up to 3 sentences, 60% of content words, every number, a claim can use two sources, a sentence that
  states a calculation's result counts as computed). It says so on the page. It does not judge truth.
- Tool results are untrusted. `netguard` resolves names itself, refuses non-public addresses at every redirect, pins the connection to the checked address, limits ports, size and time.
  `run_python` and the lab use tt-orchard's Landlock and seccomp sandbox (copied, with stdout capture added). Everything shown on the page is escaped; only http and https become links.
- The launcher refuses to boot without a lease (`TT_VISIBLE_DEVICES`), reuses a server that already answers, and always stops what it started. tt-discolike's own parser reads `.disco/app.yaml`.
- Presets are built by `tools/build_missions.py` from live Wikipedia and arXiv reads with the quotes and the date kept; expected answers are computed from what was read.
- Screenshots use a scripted stand-in model whose name says `SCRIPTED-FAKE-MODEL (not AREX-2)`; the README says so.

## Log
- 2026-10-22: built test first, a mutation table per module, in this order: streaming chat client, network guard, sandbox copy, tools, agent loop, evidence check, lab, views, missions,
  session, launcher, page. About 550 mutations; each survivor was either given a test or judged equivalent (listed below). The suite has no test that needs a chip or the network.
- Found while testing, each fixed test first:
  - `resp.read(n)` blocks until n bytes arrive, so a body that trickles in never reached the fetch deadline; the guard now uses `read1`.
  - Gradio 6 takes `css` in `launch()`, not in `Blocks()`. Its dark mode is a `dark` class on the body; my media-query variables never applied, so dark mode showed light cards on a dark page. The palette is now under `.dark`.
  - A claim whose words are split over two sources never reached the coverage bar, and "The difference between 1973 and 1918 is 55 years" has no words a source holds. The first is now two-source coverage, the second a narrow rule: at least two numbers, all in one calculator or Python result.
  - Snapshots copied each step shallowly, so an old snapshot's tool list changed as the run went on. The session copies the lists.
  - Wikipedia answers 429 to quick bursts. `default_http` retries 429 and 503, honouring Retry-After up to 8 s; the preset builder caches pages on disk.
  - The chat client splits the event stream on newline bytes only, and a test sends U+2028, U+2029 and the other separators raw inside a chunk (tt-orchard's benchmark tooling broke on U+2028 in `str.splitlines()`).
- Mistakes of mine: a `ps | grep | xargs kill` line matched its own shell and ended it (use the port's pid); several tests asserted something the code could not fail (an `or` tail, a label that matched itself);
  my first screenshot wait matched the plain-chat panel's "Finished" text and photographed a run in progress.
- Equivalent mutants kept as they are: in `netguard._public` the explicit NAT64, 6to4, Teredo and IPv4-mapped branches (Python 3.12's `is_global` already refuses them; they guard other versions);
  `views` label wording changes; `evidence` second-source union check (rechecked one line later); `lab` slow edge (`>=` against `>` gives the same speed); `code_sandbox` read size (the file-size limit bounds it);
  the session's `running` flag in the final event (set again at the end of the pump) and `can_resume` without `final` (the loop clears it).

## 2026-10-23: first real run, three fixes, and the repo made ready to publish
Prompt (operator, through the parent session): publish a companion repository for the AREX-2 model card, with screenshots from the real model.
- Ran the page against the published package `episod/arex-2-p300` (2 chips, drafter on, sampling on the device) for the first time. The tool calls parsed through the `qwen3_coder` parser and
  `reasoning_content` came back. One research mission with the offline corpus (the Sydney Opera House architect; 4 steps; 14 to 29 s over three runs), one with live tools (the Transformer paper;
  3 steps, 15 s) and one lab task (edit distance; the model ran three tests of its own and submitted once, 1.00, 116 s). `min_coins` also ran once and scored 1.00 on its first submission.
  Plain chat on the edit-distance task ran out of the page's 4,096-token reply limit while thinking. The screenshots in `docs/` come from these runs; the stand-in screenshots are gone.
- Bug 1 (found in the first screenshot): the header badge said "live web tools" while the Offline corpus radio was selected, because only the page load and the Check button refreshed it. The radio's
  `change` now refreshes it. The test inspects the built app's events for a `change` on that radio that calls `check_server`. 3 mutations, all red. The two research screenshots predate the fix and
  still show the stale badge; the README says so.
- Bug 2 (found on the first boot, which failed): the launcher quoted the last 1,500 characters of the server log, which was vLLM's wrapper line "See root cause above". The cause was a kernel compile
  failure far earlier. `serve_arex2.first_error` finds the first line that states an error (an exception line, a build or compile failure, a TT_THROW, a full disk), skips the wrapper and indented
  traceback lines, and the launcher quotes it with its line number before the tail, in the early-exit and the deadline messages. Written from tt-orchard's `narrate.root_cause`, with its own tests built from the
  real lines. 15 mutations; 1 survived (an unreadable log), which now has a test and is red.
- The first real boot failed twice for a reason outside this repo: the shared disk was full (the base model's graft shard could not download), then a kernel compile failed. The retry passed.
- `tools/screenshots.py` now writes to `docs/stand-in-check/` (git-ignored) so a stand-in screenshot cannot reach the README; `generated/` (the chip runtime's output in the working directory) is ignored.
- Scrub: no home path, user name, token or host in any tracked file or commit. The author line of the commits is the operator's work address.
- Suite: 512 passed.
Not tested: the other four research missions and two of the four lab tasks; missions near the step or time limits; more than one request at a time; the claim check on more real answers (it marked the arXiv
answer "not backed" although the authors and date were in the retrieved text); a lab run where the score improves over rounds; starting the model from a package pulled from the Hub.

## 2026-10-23 (later): a layout pass from a design review of the page
Prompt (operator): critique the web UIs for buttons that sit in the wrong place, with a technical audience in mind. The test used: a control's scope should match its container (whole page, one tab, one run, one pane).
- **Settings for every run** replaces "Model server and budgets" and splits into three labelled groups: *Connection* (address with **Check the server** beside it, model name, tools), *Limits* (steps, tokens, minutes) and *Sampling* (tokens per reply, temperature, top-p). The check button used to be a full-width bar under the sliders, far from the address it tests.
- Each tab shows the tools mode in words beside Run ("Tools: live web tools" or "Tools: offline corpus (no network)"), driven by the radio. The radio stays in Connection because it applies to both tabs.
- **Download this run** and **Copy answer** sit beside the status line of each tab and turn on when a run ends (`arex2_agent/export.py`, `prepare_export`). The file holds the mission, settings, every step, the answer, the evidence, the claim check, the lab rounds and the scorecard. Every string passes through `redact` (bearer and authorization headers, `hf_`, `sk-`, `gh*_` tokens, key and password assignments, `/home` and `/Users` paths), the server address loses its login and path, and a model path is cut to its last part. The last 20 files are kept in `<tmp>/arex2-lab-exports`. Copy falls back to a selected textarea on a page served over plain http (checked in a browser with the clipboard interface removed).
- Run is off while a run is going and Stop is on only then; both come back when the run ends, also when it was refused or failed (chained with `.then`, which also runs after a failure; `.success` would skip it). The `Continue` button is locked the same way. Starting a run also turns off the files of the last run.
- "How to read this page" is a closed accordion at the bottom and no longer a tab. The lab tab's mission picker drops the repeated "Improvement lab: " prefix from its labels (ids and API names are unchanged).
- Found while checking it in a browser: a locator for the button named "Run" also matches the accordion header "Settings for every run" and "Download this run", because the match is a substring. Capture scripts must use an exact name. `tools/screenshots.py` does and now writes to `docs/stand-in-check/`, which git ignores. The screenshots the README shows (`docs/`) were taken before this layout and are not retaken here (no chips).
- Tests first: 8 in `tests/test_export.py` (10 after survivors) and 17 layout tests in `tests/test_app.py`, found through the app's own config (`get_config_file`: layout tree, event chains, component props). Mutations: 34 on `demo/app.py`, 28 on `export.py`, 5 on the CSS. The survivors each got a test: the lock reaching the Download and Copy buttons, a running snapshot that still holds an older final, Copy with no answer, the export folder, objects that are not plain data, NaN, and which files the prune removes. Equivalent: stripping the lab prefix from research titles (none starts with it).
- Not done: the same pass on the orchard UI was reviewed separately. The README screenshots show the old layout until they are retaken with the real model.

## Not done
- A first real run happened on 2026-10-23 (see above). Longer missions, the other tasks and concurrent requests have not run against the real model.
- The claim check thresholds (60%, 25%, window of 3) and the lab's speed limits are chosen values and need a look at real output.
- No `package/` notes and no tt-model package. The repository is ready to publish at https://github.com/tsingletaryTT/tt-arex2-agent; it had no remote when this entry was written.
- The page's own JavaScript is Gradio's; screenshots were taken with headless Chrome through playwright in a scratch venv (not a dependency of the repo).
