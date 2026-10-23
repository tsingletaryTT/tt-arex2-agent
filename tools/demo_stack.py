#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Run the page together with the scripted stand-in model (tools/demo_server.py), for trying the page with no chip and for screenshots.

    python tools/demo_stack.py [--ui-port 7899] [--model-port 18099]

The model is a script, not AREX-2 (the page's badge says so). Choose "Offline corpus" in the page to keep the tools off the network.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo import app  # noqa: E402
from tools import demo_server  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ui-port", type=int, default=7899)
    ap.add_argument("--model-port", type=int, default=18099)
    a = ap.parse_args(argv)
    from http.server import ThreadingHTTPServer
    srv = demo_server.make_server()
    srv.server.server_close()
    srv.server = ThreadingHTTPServer(("127.0.0.1", a.model_port), srv.server.RequestHandlerClass)
    srv.url = f"http://127.0.0.1:{a.model_port}/v1"
    import threading
    threading.Thread(target=srv.server.serve_forever, daemon=True).start()
    print(f"scripted model (not AREX-2): {srv.url}", flush=True)
    app.launch("127.0.0.1", a.ui_port, url=srv.url, prevent_thread_lock=True)
    print(f"page: http://127.0.0.1:{a.ui_port}/", flush=True)
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
