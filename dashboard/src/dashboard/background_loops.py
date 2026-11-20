"""The background loops one dashboard lifespan starts, as one injectable value, with the production binding."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from fastapi import FastAPI

from dashboard.config import DashboardConfig
from dashboard.data.db import DbPool
from dashboard.http_pool import reaper_loop
from dashboard.loops import (
    _burndown_loop,
    _BurndownStore,
    _metrics_loop,
    _MetricsStore,
    _quality_loop,
    _QualityStore,
)

BurndownLoop = Callable[[_BurndownStore, DashboardConfig, httpx.AsyncClient], Coroutine[Any, Any, None]]
QualityLoop = Callable[[_QualityStore, DashboardConfig], Coroutine[Any, Any, None]]
ReaperLoop = Callable[[httpx.AsyncClient], Coroutine[Any, Any, None]]


class MetricsLoop(Protocol):
    def __call__(
        self,
        store: _MetricsStore,
        app: FastAPI,
        /,
        *,
        pool: DbPool,
        http_client: httpx.AsyncClient,
    ) -> Coroutine[Any, Any, None]: ...


@dataclass(frozen=True)
class BackgroundLoops:
    burndown: BurndownLoop
    metrics: MetricsLoop
    quality: QualityLoop
    reaper: ReaperLoop


PRODUCTION_LOOPS = BackgroundLoops(
    burndown=_burndown_loop,
    metrics=_metrics_loop,
    quality=_quality_loop,
    reaper=reaper_loop,
)
