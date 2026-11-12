"""A live OpenAI-SDK test skips only when its provider was never reached: the call
failed in the connect phase (ConnectError or ConnectTimeout) before any response.

`skip_if_provider_unreached` applies that rule to a live call the test awaits itself.
`FailOpenCauses` applies it to a fail-open that swallowed the live call's exception.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

import httpx
import openai
import pytest

_CONNECT_PHASE_CAUSES = (httpx.ConnectError, httpx.ConnectTimeout)


def unreached_provider_skip_reason(exc: BaseException) -> str | None:
    """The skip reason when *exc* shows the provider's connect phase failed
    (ConnectError or ConnectTimeout) before any response; None for every other failure.
    """
    cause = exc.__cause__
    if not (isinstance(exc, openai.APIConnectionError) and isinstance(cause, _CONNECT_PHASE_CAUSES)):
        return None
    return (
        f'{exc.request.url.host} was not reached: {type(cause).__name__} before any '
        f'response, so the live call observed nothing; a lasting outage is '
        f"scripts/lane_dependency_probe.py's to report"
    )


_T = TypeVar('_T')
_P = ParamSpec('_P')


async def skip_if_provider_unreached(call: Awaitable[_T]) -> _T:
    """Await *call*, skipping the test only when the provider was never reached."""
    try:
        return await call
    except openai.APIError as exc:
        if (reason := unreached_provider_skip_reason(exc)) is None:
            raise
        pytest.skip(reason)


def _described(exc: BaseException) -> str:
    cause = exc.__cause__
    caused_by = type(cause).__name__ if cause is not None else 'no __cause__'
    return f'{type(exc).__name__}({exc}) caused by {caused_by}'


class FailOpenCauses:
    """What a wrapped seam raised into its caller's fail-open, so a live test can
    tell an unreached provider from a real failure.
    """

    def __init__(self) -> None:
        self._raised: list[Exception] = []

    def recording(self, seam: Callable[_P, Awaitable[_T]]) -> Callable[_P, Awaitable[_T]]:
        """*seam*, recording each exception it raises before re-raising it unchanged."""

        async def _recorded(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            try:
                return await seam(*args, **kwargs)
            except Exception as exc:
                self._raised.append(exc)
                raise

        return _recorded

    def assert_no_fail_open_unless_unreached(self, fail_open_count: int) -> None:
        """Return when nothing failed open; skip when every counted fail-open was an
        unreached provider; fail otherwise.
        """
        if fail_open_count == 0:
            return
        reasons = [
            reason for exc in self._raised
            if (reason := unreached_provider_skip_reason(exc)) is not None
        ]
        if fail_open_count == len(self._raised) == len(reasons):
            pytest.skip(reasons[0])
        recorded = '; '.join(map(_described, self._raised)) or 'no recorded cause'
        pytest.fail(
            f'{fail_open_count} fail-open(s) counted, {len(self._raised)} raised through '
            f'the recorded seam: {recorded}. Only a connect-phase failure (ConnectError or '
            f'ConnectTimeout) before any response skips a live test; every other fail-open '
            f'fails it (plans/offline-lane-red-path-soak-2026-11-10.md "Rulings (iii)").'
        )
