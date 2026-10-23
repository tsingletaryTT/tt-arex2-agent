# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""The sandbox the Python tool uses (arex2_agent/code_sandbox.py, copied from tt-orchard) and the output capture added here."""
from __future__ import annotations

import time

import pytest

from arex2_agent import code_sandbox as sb

pytestmark = pytest.mark.skipif(not sb.self_check()["ok"], reason="this kernel cannot enforce the sandbox (Landlock and seccomp)")


def test_the_self_check_passes_on_a_kernel_that_enforces_the_sandbox():
    assert sb.self_check()["ok"]


def test_what_a_program_prints_comes_back():
    out = sb.run_program("print(sum(range(10)))\nprint('done')")
    assert out.status == "passed" and out.stdout.split() == ["45", "done"]


def test_a_program_that_prints_a_lot_is_cut_to_the_keep_size():
    out = sb.run_program("print('x' * 500000)")
    assert len(out.stdout) <= sb.STDOUT_KEEP


def test_a_failing_program_reports_failed_with_the_error_text_and_its_earlier_output():
    out = sb.run_program("print('before')\nraise ValueError('boom')")
    assert out.status == "failed" and "boom" in out.detail and "before" in out.stdout


def test_a_program_cannot_open_a_network_socket():
    out = sb.run_program("import socket\ns = socket.socket()\n")
    assert out.status == "failed" and ("Permission" in out.detail or "denied" in out.detail.lower())


def test_a_program_cannot_read_the_operators_files():
    out = sb.run_program("import os\nprint(os.listdir('/home'))")
    assert out.status == "failed"
    out2 = sb.run_program("print(open('/etc/passwd').read()[:10])")
    assert out2.status == "failed"


def test_a_program_can_write_only_in_its_scratch_directory():
    ok = sb.run_program("open('note.txt', 'w').write('hi')\nprint(open('note.txt').read())")
    assert ok.status == "passed" and "hi" in ok.stdout
    bad = sb.run_program("open('/tmp/arex2_escape_probe', 'w').write('x')")
    assert bad.status == "failed"


def test_a_program_that_never_ends_is_killed_at_the_timeout():
    t0 = time.monotonic()
    out = sb.run_program("while True:\n    pass", timeout_s=1.0)
    assert out.status == "timeout" and time.monotonic() - t0 < 6


def test_the_programs_environment_holds_no_secrets():
    out = sb.run_program("import os\nprint(sorted(os.environ))")
    assert "HF_TOKEN" not in out.stdout and "SERPER_API_KEY" not in out.stdout and "SSH_AUTH_SOCK" not in out.stdout


def test_output_printed_before_a_timeout_is_kept():
    out = sb.run_program("print('progress', flush=True)\nwhile True:\n    pass", timeout_s=1.0)
    assert out.status == "timeout" and "progress" in out.stdout
