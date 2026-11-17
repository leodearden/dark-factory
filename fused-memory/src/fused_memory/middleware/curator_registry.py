"""Lazy, off-loop, retrying load of one curator guard registry, and the health
it reports, so a registry that failed to load is distinguishable from one
that loaded empty.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Generic, TypeVar

from fused_memory.middleware.curator_types import exception_summary

logger = logging.getLogger(__name__)

E = TypeVar('E')


class RegistryState(StrEnum):
    UNCONFIGURED = 'unconfigured'
    NOT_LOADED = 'not_loaded'
    LOADED = 'loaded'
    FAILED = 'failed'


_STATE_RULE: dict[RegistryState, str] = {
    RegistryState.UNCONFIGURED: 'needs entry_count and error both None',
    RegistryState.NOT_LOADED: 'needs entry_count and error both None',
    RegistryState.LOADED: 'needs an entry_count and error None',
    RegistryState.FAILED: 'needs an error and failures >= 1',
}


@dataclass(frozen=True)
class RegistryHealth:
    """One guard registry's load state, as reported to operators."""

    label: str
    config_key: str
    state: RegistryState
    path: str | None = None
    entry_count: int | None = None
    error: str | None = None
    failures: int = 0

    def __post_init__(self) -> None:
        if not self._fields_fit_state():
            raise ValueError(
                f'RegistryHealth {self.label!r}: state={self.state.value} '
                f'{_STATE_RULE[self.state]}, got entry_count={self.entry_count!r}, '
                f'error={self.error!r}, failures={self.failures!r}'
            )

    def _fields_fit_state(self) -> bool:
        if self.state is RegistryState.LOADED:
            return self.entry_count is not None and self.error is None
        if self.state is RegistryState.FAILED:
            return self.error is not None and self.failures >= 1
        return self.entry_count is None and self.error is None

    def to_wire(self) -> dict[str, object]:
        return {
            'label': self.label,
            'config_key': self.config_key,
            'state': self.state.value,
            'path': self.path,
            'entry_count': self.entry_count,
            'error': self.error,
            'failures': self.failures,
        }


@dataclass(frozen=True)
class _Outcome(Generic[E]):
    """The settled result of the latest load attempt, replaced whole."""

    entries: list[E] | None = None
    path: Path | None = None
    error: str | None = None
    failures: int = 0
    failed_at: float | None = None

    def state(self) -> RegistryState:
        if self.entries is not None:
            return RegistryState.LOADED
        if self.error is not None:
            return RegistryState.FAILED
        return RegistryState.NOT_LOADED


class CuratorRegistry(Generic[E]):
    """Loads one guard registry off the event loop on first use. A failed load
    fails open and is retried after ``retry_seconds``; a good load is kept.
    """

    def __init__(
        self,
        *,
        label: str,
        config_key: str,
        loader: Callable[[Path], list[E]],
        configured_path: Callable[[], str | None],
        cwd: Path | None,
        retry_seconds: float,
        clock: Callable[[], float],
    ) -> None:
        self._label = label
        self._config_key = config_key
        self._loader = loader
        self._configured_path = configured_path
        self._cwd = cwd
        self._retry_seconds = retry_seconds
        self._clock = clock
        self._outcome: _Outcome[E] = _Outcome()
        self._lock = asyncio.Lock()

    async def entries(self) -> list[E] | None:
        """The loaded entries, or None when unconfigured or failing open."""
        cfg_path = self._configured_path()
        if cfg_path is None:
            return None
        if self._load_due():
            async with self._lock:
                if self._load_due():
                    await self._load(self._resolve(cfg_path))
        return self._outcome.entries

    def health(self) -> RegistryHealth:
        if self._configured_path() is None:
            return RegistryHealth(
                label=self._label, config_key=self._config_key,
                state=RegistryState.UNCONFIGURED,
            )
        outcome = self._outcome
        return RegistryHealth(
            label=self._label,
            config_key=self._config_key,
            state=outcome.state(),
            path=None if outcome.path is None else str(outcome.path),
            entry_count=None if outcome.entries is None else len(outcome.entries),
            error=outcome.error,
            failures=outcome.failures,
        )

    def _load_due(self) -> bool:
        outcome = self._outcome
        if outcome.state() is RegistryState.LOADED:
            return False
        if outcome.failed_at is None:
            return True
        return self._clock() - outcome.failed_at >= self._retry_seconds

    async def _load(self, path: Path) -> None:
        previous = self._outcome
        try:
            loaded = await asyncio.to_thread(self._loader, path)
        except Exception as exc:
            error = exception_summary(exc)
            self._outcome = _Outcome(
                path=path, error=error,
                failures=previous.failures + 1, failed_at=self._clock(),
            )
            logger.warning(
                'task_curator: %s registry load failed for %s, failing open '
                '(guard disabled) until a retry in %.0fs succeeds: %s',
                self._label, path, self._retry_seconds, error,
            )
            return
        # Never settle in a `finally`: CancelledError is a BaseException that
        # skips the except above, and must leave the outcome untouched (task 4201).
        self._outcome = _Outcome(entries=loaded, path=path, failures=previous.failures)

    def _resolve(self, cfg_path: str) -> Path:
        raw_path = Path(cfg_path)
        if raw_path.is_absolute():
            return raw_path
        if self._cwd is not None:
            return self._cwd / raw_path
        logger.warning(
            'task_curator: %s %r is relative but TaskCurator was constructed '
            'without cwd — resolving against process CWD which may be '
            'incorrect; use an absolute path in CuratorConfig',
            self._config_key, cfg_path,
        )
        return raw_path
