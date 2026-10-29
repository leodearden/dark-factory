"""Tests for orchestrator.teardown — primitives that carry a teardown to a
confirmed end (task 6794)."""
from __future__ import annotations

import asyncio
import gc
import logging
import os
from pathlib import Path

import pytest

from orchestrator.teardown import complete_despite_cancellation, path_confirmed_absent


class TestPathConfirmedAbsent:
    def test_never_created_path_is_absent(self, tmp_path: Path) -> None:
        assert path_confirmed_absent(tmp_path / 'never') is True

    def test_existing_directory_is_not_absent(self, tmp_path: Path) -> None:
        d = tmp_path / 'd'
        d.mkdir()
        assert path_confirmed_absent(d) is False

    def test_existing_file_is_not_absent(self, tmp_path: Path) -> None:
        f = tmp_path / 'f'
        f.write_text('x')
        assert path_confirmed_absent(f) is False

    def test_dangling_symlink_is_not_absent(self, tmp_path: Path) -> None:
        link = tmp_path / 'link'
        link.symlink_to(tmp_path / 'gone')
        assert path_confirmed_absent(link) is False

    @pytest.mark.skipif(os.geteuid() == 0, reason='root bypasses permission bits')
    def test_unreadable_parent_is_unknown_not_absent(self, tmp_path: Path) -> None:
        locked = tmp_path / 'locked'
        locked.mkdir()
        inner = locked / 'inner'
        inner.mkdir()
        locked.chmod(0o000)
        try:
            with pytest.raises(PermissionError):
                os.lstat(inner)
            assert path_confirmed_absent(inner) is False
        finally:
            locked.chmod(0o700)


class TestCompleteDespiteCancellation:
    def test_uncancelled_returns_the_result(self) -> None:
        async def returns_42() -> int:
            return 42

        async def scenario() -> int:
            return await complete_despite_cancellation(returns_42())

        assert asyncio.run(scenario()) == 42

    def test_uncancelled_propagates_the_exception(self) -> None:
        async def boom() -> None:
            raise ValueError('boom')

        async def scenario() -> None:
            await complete_despite_cancellation(boom())

        with pytest.raises(ValueError, match='boom'):
            asyncio.run(scenario())

    def test_repeated_caller_cancellation_is_deferred_not_dropped(self) -> None:
        finished: list[bool] = []

        async def scenario() -> asyncio.Task[None]:
            gate = asyncio.Event()

            async def work() -> None:
                await gate.wait()
                finished.append(True)

            caller = asyncio.create_task(complete_despite_cancellation(work()))
            await asyncio.sleep(0)
            for _ in range(3):
                caller.cancel()
                await asyncio.sleep(0)
            assert not caller.done()
            assert finished == []
            gate.set()
            with pytest.raises(asyncio.CancelledError):
                await caller
            return caller

        caller = asyncio.run(scenario())
        assert finished == [True]
        assert caller.cancelled()

    def test_the_work_tasks_own_cancellation_propagates(self) -> None:
        async def work() -> None:
            current = asyncio.current_task()
            assert current is not None
            current.cancel()
            await asyncio.sleep(0)

        async def scenario() -> None:
            with pytest.raises(asyncio.CancelledError):
                await complete_despite_cancellation(work())

        asyncio.run(scenario())

    def test_caller_cancellation_wins_over_a_work_exception_and_chains_it(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.ERROR, logger='asyncio')
        raised: list[ValueError] = []

        async def scenario() -> BaseException | None:
            gate = asyncio.Event()

            async def work() -> None:
                await gate.wait()
                exc = ValueError('late')
                raised.append(exc)
                raise exc

            caller = asyncio.create_task(complete_despite_cancellation(work()))
            await asyncio.sleep(0)
            caller.cancel()
            await asyncio.sleep(0)
            gate.set()
            with pytest.raises(asyncio.CancelledError) as excinfo:
                await caller
            return excinfo.value.__cause__

        cause = asyncio.run(scenario())
        gc.collect()
        assert raised and cause is raised[0]
        assert not [
            r for r in caplog.records
            if r.name == 'asyncio' and 'never retrieved' in r.getMessage()
        ]
