"""The suite's clock helpers: fixed instants installed as a dashboard app's route clock."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

from fastapi import FastAPI

from dashboard.api.clock import Clock, app_clock, install_clock


def fixed_clock(at: datetime) -> Clock:
    """Return a clock that always answers *at*."""
    return lambda: at


@contextmanager
def installed_clock(app: FastAPI, clock: Clock) -> Iterator[None]:
    """Serve *app*'s routes from *clock* for the block, then restore the previous clock."""
    previous = app_clock(app)
    install_clock(app, clock)
    try:
        yield
    finally:
        install_clock(app, previous)
