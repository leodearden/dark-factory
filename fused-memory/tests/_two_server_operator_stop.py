"""Child-process driver: one SIGTERM stops two embedded uvicorn servers served and torn down by main.py's own seams.

Run as ``python -I <this file>`` from a clean cwd; the stop's log lines are the
output, on stderr in main.py's LOG_FORMAT. Asserted by
``fused-memory/tests/test_operator_signal_handler.py::TestOneStopSignalStopsBothServersCleanly``.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

from starlette.applications import Starlette

import fused_memory.server.main as main

# -I keeps the script's own directory off sys.path; appended, it cannot shadow an installed module.
sys.path.append(str(Path(__file__).resolve().parent))
from _embedded_uvicorn_fixtures import ReadySignallingServer  # noqa: E402


def _server() -> ReadySignallingServer:
    return ReadySignallingServer(
        main._build_uvicorn_config(Starlette(), host='127.0.0.1', port=0, graceful_shutdown_timeout=5),
    )


async def _stop_with_one_sigterm() -> None:
    primary, recon = _server(), _server()
    main._install_operator_stop_handler(main._make_operator_stop_callback(primary, recon))
    signals_drained = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGUSR2, signals_drained.set)

    serving = asyncio.ensure_future(main._serve_primary_and_recon(primary, recon, primary_sockets=None))
    await primary.until_started()
    await recon.until_started()

    os.kill(os.getpid(), signal.SIGTERM)
    await serving

    # Signals reach the loop in arrival order through its wakeup fd, so any
    # SIGTERM a server re-raised as serve() returned is handled before this one.
    os.kill(os.getpid(), signal.SIGUSR2)
    await signals_drained.wait()


if __name__ == '__main__':
    asyncio.run(_stop_with_one_sigterm())
