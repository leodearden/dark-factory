"""uvicorn servers embedded in the fused-memory process, which owns their signals and loggers.

Every uvicorn server in the process shares its signal handlers and its
``uvicorn.*`` loggers, so no single server may own them. Their owner is
``fused_memory/server/main.py``: its operator-stop handler for signals, its
import-time ``logging.basicConfig`` for logs.

Contract: ``fused-memory/tests/server/test_embedded_uvicorn.py``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import uvicorn

_PROCESS_GLOBAL_LOGGING_SETTINGS = ('log_config', 'log_level', 'access_log')


class EmbeddedUvicornServer(uvicorn.Server):
    """A uvicorn server that leaves SIGTERM and SIGINT to the host process."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        """uvicorn's own capture_signals installs SIGTERM/SIGINT handlers for the life of serve() and re-raises each caught signal when serve() returns, so the host process must own those signals."""
        yield


def embedded_uvicorn_config(app: Any, **settings: Any) -> uvicorn.Config:
    """A ``uvicorn.Config`` for *app* that configures no logging; *settings* go to uvicorn unchanged."""
    for key in _PROCESS_GLOBAL_LOGGING_SETTINGS:
        if key in settings:
            raise ValueError(
                f'embedded_uvicorn_config refuses {key!r}: uvicorn applies it to the '
                f'process-global uvicorn.* loggers, which fused_memory/server/main.py owns'
            )
    return uvicorn.Config(app, log_config=None, **settings)
