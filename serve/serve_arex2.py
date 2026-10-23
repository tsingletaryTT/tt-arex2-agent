#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Boot the AREX-2 server on the chips this process holds, wait for it, and (with --ui) run the research page.

    serve_arex2.py [--run-sh PATH] [--port 8000] [--endpoint URL] [--ui] [--ui-port 7872] [--print]

tt-discolike runs this under `gozer run --chips 2` (the manifest says chips: 2), so the chip lease already exists and TT_VISIBLE_DEVICES names the
leased chips. The script refuses to start the model without one: opening a device with no lease is how two agents end up fighting over a board.
If a server already answers at the port it is used as it is and is not stopped on exit.

The model is the `arex-2-p300` package's own run.sh (`tt-model pull episod/arex-2-p300`, or any run.sh named by --run-sh or AREX2_RUN_SH). That script
serves the vLLM OpenAI API with the tool-call and reasoning parsers the page relies on. With --ui the page runs in this process, so a SIGTERM stops
both the page and the server's whole process group (the server is started in its own session and is always stopped, first politely, then by force).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

DEFAULT_PORT = 8000
DEFAULT_UI_PORT = 7872
BOOT_DEADLINE_S = 45 * 60          # a cold first boot took about 26 minutes on the box this was written on
STOP_GRACE_S = 30
PACKAGE = "episod/arex-2-p300"
LOG_TAIL_CHARS = 1500


class LauncherError(Exception):
    pass


def find_run_sh(explicit: Optional[str], env: dict, home: Path = Path.home()) -> Path:
    """--run-sh, then AREX2_RUN_SH, then the installed tt-model package. A path that was named and does not exist is an error."""
    named = explicit or env.get("AREX2_RUN_SH")
    if named:
        p = Path(named).expanduser()
        if not p.is_file():
            raise LauncherError(f"the run script {p} does not exist")
        return p
    installed = Path(home) / ".cache" / "tt-model" / "models" / PACKAGE / "run.sh"
    if installed.is_file():
        return installed
    raise LauncherError(f"the AREX-2 package is not installed. Run `tt-model pull {PACKAGE}` (once it is published), or set AREX2_RUN_SH to the "
                        "run.sh of a built package.")


def server_model(url: str, timeout: float = 3.0) -> Optional[str]:
    """The served model name when a server answers /v1/models at `url`, else None."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=timeout) as resp:
            data = json.loads(resp.read(100000))
        return data["data"][0]["id"]
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None


def port_free(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


class Launcher:
    def __init__(self, run_sh: Path, port: int, env: dict, log_dir: Path, popen=subprocess.Popen, probe=None, sleep=time.sleep, clock=time.monotonic,
                 killpg=os.killpg, port_free=port_free, boot_deadline_s: float = BOOT_DEADLINE_S, grace_s: float = STOP_GRACE_S):
        self.run_sh, self.port, self.env, self.log_dir = Path(run_sh), port, dict(env), Path(log_dir)
        self._popen, self._sleep, self._clock, self._killpg, self.port_free = popen, sleep, clock, killpg, port_free
        self.probe = probe or (lambda url: server_model(url))
        self.boot_deadline_s, self.grace_s = boot_deadline_s, grace_s
        self.proc = None
        self.started = False
        self.log_path: Optional[Path] = None
        self.url = f"http://127.0.0.1:{port}/v1"

    def command(self) -> list:
        return ["bash", str(self.run_sh), "--port", str(self.port)]

    def boot(self) -> str:
        """Return the served model name, starting the server first unless one already answers."""
        existing = self.probe(self.url)
        if existing:
            return existing
        if not self.env.get("TT_VISIBLE_DEVICES"):
            raise LauncherError("no chip lease: TT_VISIBLE_DEVICES is empty. Start this under gozer (`gozer run --chips 2 -- python serve/serve_arex2.py --ui`) "
                                "or through tt-discolike, so the chips are yours before the model opens them.")
        if not self.port_free(self.port):
            raise LauncherError(f"port {self.port} is in use by something that is not a model server. Choose another with --port.")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"arex2-server-{int(time.time())}.log"
        log = open(self.log_path, "wb")
        self.proc = self._popen(self.command(), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=self.env, start_new_session=True)
        self.started = True
        start = self._clock()
        while True:
            code = self.proc.poll()
            if code is not None:
                log.flush()
                raise LauncherError(f"the server script exited with status {code} before the server answered. End of its log ({self.log_path}):\n{self._tail()}")
            model = self.probe(self.url)
            if model:
                return model
            if self._clock() - start > self.boot_deadline_s:
                self.stop()
                raise LauncherError(f"the server did not answer within {self.boot_deadline_s:g} s and was stopped. Its log is {self.log_path}:\n{self._tail()}")
            self._sleep(5)

    def _tail(self) -> str:
        try:
            return self.log_path.read_bytes()[-LOG_TAIL_CHARS:].decode("utf-8", "replace")
        except (OSError, AttributeError):
            return "(no log)"

    def stop(self) -> None:
        """Stop the server this launcher started: SIGTERM to its process group, then SIGKILL if it is still there after the grace period."""
        if not self.started or self.proc is None:
            return
        proc, self.proc = self.proc, None
        try:
            self._killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        waited = 0.0
        while proc.poll() is None and waited < self.grace_s:
            self._sleep(1)
            waited += 1
        if proc.poll() is None:
            try:
                self._killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-sh", default=None, help="the package's run.sh (default: AREX2_RUN_SH, then the installed tt-model package)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("AREX2_SERVER_PORT", DEFAULT_PORT)), help="the model server's port")
    ap.add_argument("--ui", action="store_true", help="run the Gradio page in this process after the server answers")
    ap.add_argument("--ui-port", type=int, default=int(os.environ.get("AREX2_UI_PORT", DEFAULT_UI_PORT)))
    ap.add_argument("--host", default=os.environ.get("AREX2_UI_HOST", "127.0.0.1"), help="the page's address (0.0.0.0 shares it on the LAN; the page has no login)")
    ap.add_argument("--print", dest="print_only", action="store_true", help="print the command and exit")
    a = ap.parse_args(argv)
    try:
        run_sh = find_run_sh(a.run_sh, os.environ)
        launcher = Launcher(run_sh, a.port, os.environ, Path(__file__).resolve().parent.parent / "runs")
        if a.print_only:
            print(" ".join(launcher.command()))
            return 0
        stop = {"flag": False}

        def on_signal(signum, frame):
            stop["flag"] = True
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
        try:
            model = launcher.boot()
            print(f"AREX-2 server ready at {launcher.url} (model {model})", flush=True)
            if a.ui:
                sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
                from demo import app as page
                page.build_app(default_url=launcher.url).queue().launch(server_name=a.host, server_port=a.ui_port, show_error=True)
            else:
                while launcher.proc is None or launcher.proc.poll() is None:
                    time.sleep(2)
        except KeyboardInterrupt:
            pass
        finally:
            launcher.stop()
        return 0
    except LauncherError as e:
        print(f"serve_arex2: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
