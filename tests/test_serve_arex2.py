# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""The launcher (serve/serve_arex2.py): boots the AREX-2 package's server on the chips this process holds, waits for it, and always stops it."""
from __future__ import annotations

import signal
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serve"))
import serve_arex2 as sv  # noqa: E402


class FakeProc:
    def __init__(self, exit_after_polls=None, returncode=3):
        self.pid, self.polls, self.exit_after, self.rc = 4242, 0, exit_after_polls, returncode
        self.waited = False

    def poll(self):
        self.polls += 1
        return self.rc if (self.exit_after is not None and self.polls > self.exit_after) else None

    def wait(self, timeout=None):
        self.waited = True
        return 0


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.now += s


def make(tmp_path, env=None, probe=None, proc=None, **kw):
    run_sh = tmp_path / "run.sh"
    run_sh.write_text("#!/bin/bash\n")
    calls = {"popen": [], "killpg": []}
    clock = Clock()

    def popen(argv, **k):
        calls["popen"].append((argv, k))
        return proc or FakeProc()

    def killpg(pgid, sig):
        calls["killpg"].append((pgid, sig))

    l = sv.Launcher(run_sh=run_sh, port=18000, env={"TT_VISIBLE_DEVICES": "0000:03:00.0,0000:04:00.0"} if env is None else env, log_dir=tmp_path / "runs",
                    popen=popen, probe=probe or (lambda url: None), sleep=clock.sleep, clock=clock, killpg=killpg, port_free=lambda port: True, **kw)
    l.calls, l.clock = calls, clock
    return l


# ---- refusing and reusing -------------------------------------------------------------------------------------------------------

def test_it_refuses_to_boot_without_a_chip_lease_so_it_never_opens_a_device_unleased(tmp_path):
    l = make(tmp_path, env={})
    with pytest.raises(sv.LauncherError) as e:
        l.boot()
    assert "lease" in str(e.value).lower() and "gozer" in str(e.value) and l.calls["popen"] == []


def test_a_server_that_already_answers_is_used_and_nothing_is_started_and_no_lease_is_needed(tmp_path):
    l = make(tmp_path, env={}, probe=lambda url: "already-running-model")
    assert l.boot() == "already-running-model"
    assert l.calls["popen"] == [] and not l.started
    l.stop()
    assert l.calls["killpg"] == []                                  # it did not start it, so it does not stop it


def test_a_missing_run_script_says_how_to_get_the_package(tmp_path):
    with pytest.raises(sv.LauncherError) as e:
        sv.find_run_sh(None, {}, home=tmp_path)
    assert "tt-model pull" in str(e.value) and "AREX2_RUN_SH" in str(e.value)


def test_the_run_script_comes_from_the_flag_then_the_environment_then_the_installed_package(tmp_path):
    a, b = tmp_path / "a.sh", tmp_path / "b.sh"
    a.write_text("x"); b.write_text("x")
    installed = tmp_path / ".cache" / "tt-model" / "models" / "episod" / "arex-2-p300"
    installed.mkdir(parents=True)
    (installed / "run.sh").write_text("x")
    assert sv.find_run_sh(str(a), {"AREX2_RUN_SH": str(b)}, home=tmp_path) == a
    assert sv.find_run_sh(None, {"AREX2_RUN_SH": str(b)}, home=tmp_path) == b
    assert sv.find_run_sh(None, {}, home=tmp_path) == installed / "run.sh"


def test_a_flag_that_names_a_missing_file_is_an_error_not_a_silent_fallback(tmp_path):
    with pytest.raises(sv.LauncherError) as e:
        sv.find_run_sh(str(tmp_path / "nope.sh"), {}, home=tmp_path)
    assert "nope.sh" in str(e.value)


def test_a_port_that_is_taken_by_something_else_is_refused(tmp_path):
    l = make(tmp_path)
    l.port_free = lambda port: False
    with pytest.raises(sv.LauncherError) as e:
        l.boot()
    assert "18000" in str(e.value) and "in use" in str(e.value).lower() and l.calls["popen"] == []


# ---- booting ----------------------------------------------------------------------------------------------------------------------

def test_it_runs_the_package_script_with_the_port_in_its_own_session_with_the_lease_in_the_environment(tmp_path):
    answers = iter([None, None, "AREX-2-served"])
    l = make(tmp_path, probe=lambda url: next(answers))
    assert l.boot() == "AREX-2-served"
    argv, kw = l.calls["popen"][0]
    assert argv == ["bash", str(tmp_path / "run.sh"), "--port", "18000"]
    assert kw["start_new_session"] is True and kw["env"]["TT_VISIBLE_DEVICES"] == "0000:03:00.0,0000:04:00.0"
    assert l.started and (tmp_path / "runs").is_dir()


def test_the_server_log_goes_to_a_file_and_its_path_is_known(tmp_path):
    l = make(tmp_path, probe=lambda url: "m" if l.calls["popen"] else None)
    l.boot()
    _, kw = l.calls["popen"][0]
    assert Path(l.log_path).parent == tmp_path / "runs" and kw["stdout"] is not None and kw["stderr"] is not None


def test_it_waits_with_pauses_between_checks_until_the_server_answers(tmp_path):
    n = []
    def probe(url):
        n.append(1)
        return "m" if len(n) >= 5 else None
    l = make(tmp_path, probe=probe)
    l.boot()
    assert len(n) >= 5 and l.clock.now > 0


def test_a_script_that_exits_early_fails_with_the_end_of_its_log(tmp_path):
    l = make(tmp_path, proc=FakeProc(exit_after_polls=2, returncode=7))
    def writing_popen(argv, **k):
        k["stdout"].write(b"loading...\nRuntimeError: SFPI mismatch\n"); k["stdout"].flush()
        return FakeProc(exit_after_polls=2, returncode=7)
    l._popen = writing_popen
    with pytest.raises(sv.LauncherError) as e:
        l.boot()
    assert "exited" in str(e.value) and "7" in str(e.value) and "SFPI mismatch" in str(e.value)


def test_a_server_that_never_comes_up_is_stopped_at_the_deadline_and_the_error_says_so(tmp_path):
    l = make(tmp_path, boot_deadline_s=100)
    with pytest.raises(sv.LauncherError) as e:
        l.boot()
    assert "100" in str(e.value) and "did not answer" in str(e.value)
    assert l.calls["killpg"] and l.calls["killpg"][0] == (4242, signal.SIGTERM)


# ---- stopping ---------------------------------------------------------------------------------------------------------------------

def test_stop_asks_the_whole_process_group_to_end_and_waits(tmp_path):
    l = make(tmp_path, probe=lambda url: "m" if l.calls["popen"] else None)
    l.boot()
    l.stop()
    assert l.calls["killpg"][0] == (4242, signal.SIGTERM)


def test_stop_kills_the_group_when_it_ignores_the_first_request(tmp_path):
    stubborn = FakeProc()
    l = make(tmp_path, proc=stubborn, probe=lambda url: "m" if l.calls["popen"] else None, grace_s=20)
    l.boot()
    l.stop()
    sigs = [s for _, s in l.calls["killpg"]]
    assert sigs == [signal.SIGTERM, signal.SIGKILL]


def test_stop_does_not_kill_when_the_process_ended_on_the_first_request(tmp_path):
    proc = FakeProc()
    l = make(tmp_path, proc=proc, probe=lambda url: "m" if l.calls["popen"] else None)
    l.boot()
    proc.exit_after, proc.polls = 0, 1                  # it ends as soon as it is asked
    l.stop()
    assert [s for _, s in l.calls["killpg"]] == [signal.SIGTERM]


def test_stop_twice_is_harmless(tmp_path):
    l = make(tmp_path, probe=lambda url: "m" if l.calls["popen"] else None)
    l.boot()
    l.stop()
    n = len(l.calls["killpg"])
    l.stop()
    assert len(l.calls["killpg"]) == n


def test_a_process_that_is_already_gone_is_not_an_error_to_stop(tmp_path):
    l = make(tmp_path, probe=lambda url: "m" if l.calls["popen"] else None)
    l.boot()
    def gone(pgid, sig):
        raise ProcessLookupError()
    l._killpg = gone
    l.stop()


# ---- the command line ---------------------------------------------------------------------------------------------------------------

def test_the_dry_run_prints_the_command_and_starts_nothing(tmp_path, capsys):
    run_sh = tmp_path / "run.sh"; run_sh.write_text("x")
    rc = sv.main(["--run-sh", str(run_sh), "--port", "18001", "--print"])
    out = capsys.readouterr().out
    assert rc == 0 and "bash" in out and "18001" in out and str(run_sh) in out


def test_the_defaults_match_the_tt_discolike_manifest(tmp_path):
    m = (Path(__file__).resolve().parent.parent / ".disco" / "app.yaml").read_text()
    assert "port: 7872" in m and "chips: 2" in m and "serve/serve_arex2.py" in m and "--ui" in m
    assert sv.DEFAULT_UI_PORT == 7872 and sv.DEFAULT_PORT == 8000


def test_a_server_that_ends_within_the_grace_period_is_not_killed():
    import tempfile
    tmp_path = Path(tempfile.mkdtemp())
    proc = FakeProc()
    l = make(tmp_path, proc=proc, probe=lambda url: "m" if l.calls["popen"] else None, grace_s=30)
    l.boot()
    proc.exit_after, proc.polls = proc.polls + 3, proc.polls          # it takes a few checks to end after the request
    l.stop()
    assert [s for _, s in l.calls["killpg"]] == [signal.SIGTERM]


def test_the_manifest_parses_with_tt_discolikes_own_parser_when_it_is_installed():
    import importlib.util
    root = Path.home() / "code" / "tt-discolike"
    if not (root / "discolike" / "manifest.py").exists():
        pytest.skip("tt-discolike is not checked out here")
    sys.path.insert(0, str(root))
    try:
        from discolike import manifest
    except ImportError as e:                       # its own dependencies (PyYAML) may be missing in this venv
        pytest.skip(f"tt-discolike cannot be imported here: {e}")
    m = manifest.parse_manifest(Path(__file__).resolve().parent.parent / ".disco" / "app.yaml")
    assert m.name == "arex2-research-lab" and m.port == 7872 and m.chips == 2 and "serve_arex2.py" in m.launch


# ---- the first real error in a server log -----------------------------------------------------------------------------------------
# Found on the first real-model boot: the launcher quoted only the end of the server's log, which was vLLM's wrapper line
# ("Engine core initialization failed. See root cause above."). The cause was a kernel compile failure 40 lines earlier.

REAL_FAILURE_LOG = "\n".join([
    "INFO 10-23 11:54:23 [api_utils.py:273] non-default args: {'port': 8103}",
    "2026-10-23 12:14:59.198 | info     |              Op | SDPA GQA K/V mcast: ENABLED (groups=4) (sdpa_program_factory.cpp:672)",
    "2026-10-23 12:16:03.815 | error    |    BuildKernels | brisc compile failure -- cmd: /opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++ -O2 -std=c++17",
    "2026-10-23 12:16:03.815 | critical |          Always | TT_THROW: brisc build failed. Log:  (assert.hpp:104)",
    "(EngineCore pid=2861523) ERROR 10-23 12:16:06 [core.py:1330] Traceback (most recent call last):",
    '(EngineCore pid=2861523) ERROR 10-23 12:16:06 [core.py:1330]   File "model.py", line 1183, in prefill',
    "(APIServer pid=2836737) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {'EngineCore': 1}",
])


def test_the_first_error_is_the_cause_and_not_vllms_wrapper_line_at_the_end():
    n, text = sv.first_error(REAL_FAILURE_LOG)
    assert n == 3 and "brisc compile failure" in text


def test_a_log_with_only_the_wrapper_line_has_no_first_error():
    assert sv.first_error("(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}") is None


def test_a_full_disk_is_found():
    log = "INFO loading\nINFO still loading\nOSError: [Errno 28] No space left on device\n"
    assert sv.first_error(log) == (3, "OSError: [Errno 28] No space left on device")
    assert sv.first_error("xet_get failed: IO Error: No space left on device (os error 28)\n")[0] == 1


def test_an_exception_line_behind_the_process_and_log_prefixes_is_found():
    log = "(EngineCore pid=7) INFO started\n(EngineCore pid=7) ERROR 10-23 12:16:06 [core.py:1330] RuntimeError: TT_THROW @ fabric_firmware_initializer.cpp:271\n"
    assert sv.first_error(log) == (2, "RuntimeError: TT_THROW @ fabric_firmware_initializer.cpp:271")


def test_an_indented_traceback_line_that_mentions_a_build_is_not_the_cause():
    log = "Traceback (most recent call last):\n  File \"x.py\", line 3\n    raise RuntimeError('build failed')\nValueError: bad value\n"
    assert sv.first_error(log) == (4, "ValueError: bad value")


def test_a_healthy_log_has_no_first_error():
    assert sv.first_error("INFO Application startup complete.\nINFO Uvicorn running on http://0.0.0.0:8103\n") is None


def test_a_long_line_is_cut():
    n, text = sv.first_error("RuntimeError: " + "x" * 2000)
    assert n == 1 and len(text) <= sv.FIRST_ERROR_CLIP


def test_the_launcher_names_the_first_error_and_its_line_before_the_tail_when_the_cause_is_far_from_the_end(tmp_path):
    l = make(tmp_path)
    filler = "\n".join(f"(EngineCore pid=1) INFO line {i} " + "y" * 60 for i in range(80))     # pushes the cause out of the 1,500-character tail
    log = (filler + "\n2026-10-23 12:16:03.815 | error    |    BuildKernels | brisc compile failure -- cmd: g++\n" + filler
           + "\n(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above.\n")
    def writing_popen(argv, **k):
        k["stdout"].write(log.encode()); k["stdout"].flush()
        return FakeProc(exit_after_polls=1, returncode=1)
    l._popen = writing_popen
    with pytest.raises(sv.LauncherError) as e:
        l.boot()
    msg = str(e.value)
    assert "First error (line 81): " in msg and "brisc compile failure" in msg
    assert msg.index("First error") < msg.index("End of its log")
    assert msg.count("brisc compile failure") == 1          # the tail is too short to repeat it, so it is quoted once


def test_the_deadline_message_names_the_first_error_too(tmp_path):
    l = make(tmp_path, boot_deadline_s=10)
    def writing_popen(argv, **k):
        k["stdout"].write(b"OSError: [Errno 28] No space left on device\n"); k["stdout"].flush()
        return FakeProc()
    l._popen = writing_popen
    with pytest.raises(sv.LauncherError) as e:
        l.boot()
    assert "First error (line 1): OSError: [Errno 28] No space left on device" in str(e.value)


def test_a_log_that_has_gone_still_gives_the_plain_error_with_no_first_error_line(tmp_path):
    l = make(tmp_path)
    def vanishing_popen(argv, **k):
        Path(k["stdout"].name).unlink()
        return FakeProc(exit_after_polls=1, returncode=2)
    l._popen = vanishing_popen
    with pytest.raises(sv.LauncherError) as e:
        l.boot()
    assert "exited with status 2" in str(e.value) and "First error" not in str(e.value) and "(no log)" in str(e.value)
