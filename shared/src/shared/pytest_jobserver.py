"""Cooperative pytest jobserver client.

At session start, blocks on a FIFO until a token is available; at session
end (or SIGTERM), returns the token. If ``PYTEST_JOBSERVER_FIFO`` is unset or
the FIFO is missing, the plugin is a no-op — tests run unthrottled.

Modelled on GNU make's jobserver protocol; compatible in spirit with the
reify-jobserver.service FIFO (``/tmp/reify-jobserver``). Each pytest process
holds exactly one token for its lifetime — the seeded token count N is the
global ceiling on concurrent pytest invocations.

Enable per subproject by adding::

    pytest_plugins = ('shared.pytest_jobserver',)

to that subproject's top-level conftest.py.
"""
from __future__ import annotations

import atexit
import os
import signal

_fd: int | None = None
_tok: bytes | None = None


def _release() -> None:
    global _fd, _tok
    if _tok is not None and _fd is not None:
        try:
            os.write(_fd, _tok)
        except OSError:
            pass
        try:
            os.close(_fd)
        except OSError:
            pass
    _fd = None
    _tok = None


def pytest_configure(config) -> None:
    global _fd, _tok
    path = os.environ.get('PYTEST_JOBSERVER_FIFO', '/tmp/pytest-jobserver')
    if not path or not os.path.exists(path):
        return
    try:
        _fd = os.open(path, os.O_RDWR)
        _tok = os.read(_fd, 1)
    except OSError:
        _release()
        return
    atexit.register(_release)
    prev = signal.getsignal(signal.SIGTERM)

    def _handler(signum, frame):
        _release()
        if callable(prev) and prev not in (signal.SIG_DFL, signal.SIG_IGN):
            prev(signum, frame)
        else:
            os._exit(143)

    signal.signal(signal.SIGTERM, _handler)


def pytest_unconfigure(config) -> None:
    _release()
