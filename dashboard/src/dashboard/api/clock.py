"""The one place a dashboard route reads the present from.

A route takes ``app_clock(request.app)`` and calls it where it needs an
instant: once before its fan-out as ``render_at`` and, where it serves
``served_at``, once after. It passes those instants to every reader that takes
``now=``. ``app.py``'s lifespan installs :func:`live_clock`; a test installs
another with :func:`install_clock`.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from fastapi import FastAPI

from dashboard.data.utils import resolve_now

Clock = Callable[[], datetime]


def live_clock() -> datetime:
    return resolve_now(None)


def install_clock(app: FastAPI, clock: Clock) -> None:
    app.state.clock = clock


def app_clock(app: FastAPI) -> Clock:
    return app.state.clock
