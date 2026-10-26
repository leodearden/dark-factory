"""await_best_effort returns the awaited value, and turns an Exception into a
logged None."""
from __future__ import annotations

import asyncio
import logging
import sqlite3

import pytest

from fused_memory.reconciliation.sqlite_failures import await_best_effort

LOGGER_NAME = 'test_sqlite_failures.await_best_effort'
LOGGER = logging.getLogger(LOGGER_NAME)


def _logged(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == LOGGER_NAME]


async def _returning(value: object) -> object:
    return value


async def _raising(exc: BaseException) -> object:
    raise exc


def _database_locked() -> sqlite3.OperationalError:
    err = sqlite3.OperationalError('database is locked')
    err.sqlite_errorname = 'SQLITE_BUSY'
    err.sqlite_errorcode = 5
    return err


@pytest.mark.asyncio
@pytest.mark.parametrize('value', [True, {'status': 'completed'}])
async def test_returns_the_awaited_value_and_logs_nothing(
    value: object, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    result = await await_best_effort(
        _returning(value), what='read', run_id='r1',
        logger=LOGGER, level=logging.ERROR,
    )

    assert result is value
    assert _logged(caplog) == []


@pytest.mark.asyncio
@pytest.mark.parametrize('level', [logging.WARNING, logging.ERROR])
async def test_an_exception_is_logged_once_at_the_given_level_and_returns_none(
    level: int, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    result = await await_best_effort(
        _raising(_database_locked()), what='is_run_completed', run_id='r1',
        logger=LOGGER, level=level,
    )

    assert result is None
    [record] = _logged(caplog)
    assert record.levelno == level
    message = record.getMessage()
    assert 'is_run_completed failed' in message
    assert 'run_id=r1' in message
    assert 'sqlite_errorname=SQLITE_BUSY' in message


@pytest.mark.asyncio
async def test_cancellation_propagates_and_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    with pytest.raises(asyncio.CancelledError):
        await await_best_effort(
            _raising(asyncio.CancelledError()), what='read', run_id='r1',
            logger=LOGGER, level=logging.ERROR,
        )

    assert _logged(caplog) == []
