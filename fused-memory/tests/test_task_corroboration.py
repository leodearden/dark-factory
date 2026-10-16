"""The task-corroboration primitives (``reconciliation/task_corroboration.py``).

``confirm_task_present`` classifies one ``get_task`` result, ``safe_get_task``
fetches one task with exceptions normalised, and ``corroborate_task_lookups``
is the one loop that corroborates many tasks, each in its own project.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import Any
from unittest.mock import AsyncMock

import pytest

from fused_memory.backends.task_backend_errors import TaskmasterError
from fused_memory.reconciliation.flag_dedup import confirm_task_absent
from fused_memory.reconciliation.task_corroboration import (
    LookupVerdict,
    TaskLookup,
    confirm_task_present,
    corroborate_task_lookups,
    safe_get_task,
)

# ---------------------------------------------------------------------------
# confirm_task_present (task 2525)
# ---------------------------------------------------------------------------


class TestConfirmTaskPresent:
    """``confirm_task_present(get_task_result)`` (task 2525).

    Returns True ONLY when the result positively confirms the task DOES exist
    (a plain task-record dict, no error keys, carrying at least one
    task-identity key). Any not-found, inconclusive, or non-dict result
    returns False (fail-safe: uncertain presence must never be treated as
    corroboration).
    """

    def test_true_for_valid_task_record(self):
        """Returns True for a valid task record dict carrying several identity keys."""

        task_record = {
            'id': '42',
            'title': 'Some real task',
            'status': 'in-progress',
            'dependencies': [],
        }
        assert confirm_task_present(task_record) is True, (
            'confirm_task_present must return True for a valid task record'
        )

    def test_true_for_dict_with_only_task_id_key(self):
        """Returns True when the only identity key present is 'task_id'."""

        minimal_record = {'task_id': '42'}
        assert confirm_task_present(minimal_record) is True, (
            "confirm_task_present must return True when 'task_id' is the only identity key"
        )

    def test_false_for_not_found_error_dict(self):
        """Returns False for the canonical not-found error dict (task is absent, not present)."""

        not_found = {
            'error': 'TASKMASTER_TOOL_ERROR: No tasks found for ID(s): 42',
            'error_type': 'TaskmasterError',
        }
        assert confirm_task_present(not_found) is False, (
            'confirm_task_present must return False for a not-found error dict'
        )

    def test_false_for_generic_inconclusive_error_dict(self):
        """Returns False for a generic/inconclusive error dict (e.g. timeout)."""

        timeout_error = {
            'error': 'Connection timeout reaching Taskmaster backend',
            'error_type': 'TimeoutError',
        }
        assert confirm_task_present(timeout_error) is False, (
            'confirm_task_present must return False for inconclusive errors (fail-safe)'
        )

    def test_false_for_none(self):
        """Returns False for None input (fail-safe)."""

        assert confirm_task_present(None) is False, (
            'confirm_task_present must return False for None (fail-safe)'
        )

    def test_false_for_empty_dict(self):
        """Returns False for an empty dict (no identity keys => not confirmed present)."""

        assert confirm_task_present({}) is False, (
            'confirm_task_present must return False for an empty dict (fail-safe)'
        )

    def test_false_for_non_dict_inputs(self):
        """Returns False for non-dict inputs (str, int, list, bool) — fail-safe."""

        for bad in ['a task exists', 42, [], True]:
            assert confirm_task_present(bad) is False, (
                f'confirm_task_present must return False for non-dict {bad!r} (fail-safe)'
            )


# ---------------------------------------------------------------------------
# safe_get_task (task 3051)
# ---------------------------------------------------------------------------


class TestSafeGetTask:
    """``safe_get_task(taskmaster, task_id, project_root)`` (task 3051).

    The single shared one-task fetch every cross-project corroboration path
    routes through: it
    returns the raw ``taskmaster.get_task`` result untouched on success and
    normalises ANY exception to ``{'error': str(exc), 'error_type':
    type(exc).__name__}``, so ``confirm_task_present`` / ``confirm_task_absent``
    classify the raised-exception path exactly as they classify the MCP
    wrapper's own error-dict path. That normalisation is also what lets every
    caller keep using a PLAIN ``asyncio.gather``.
    """

    @pytest.mark.asyncio
    async def test_returns_task_record_unchanged(self):
        """A successful task-record result passes through byte-identical."""

        record = {'id': '3045', 'title': 'A real task', 'status': 'pending'}
        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(return_value=record)

        result = await safe_get_task(taskmaster, '3045', '/repo/df')

        assert result is record
        assert confirm_task_present(result) is True

    @pytest.mark.asyncio
    async def test_returns_error_dict_unchanged(self):
        """A RETURNED error dict is passed through untouched, not re-wrapped."""

        error_dict = {
            'error': 'TASKMASTER_TOOL_ERROR: No tasks found for ID(s): 9999',
            'error_type': 'TaskmasterError',
        }
        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(return_value=error_dict)

        result = await safe_get_task(taskmaster, '9999', '/repo/df')

        assert result is error_dict
        assert confirm_task_absent(result) is True

    @pytest.mark.asyncio
    async def test_forwards_task_id_and_project_root_positionally(self):
        """task_id and project_root are forwarded POSITIONALLY, in that order."""

        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(return_value={'id': '7'})

        await safe_get_task(taskmaster, 7, '/repo/other')

        taskmaster.get_task.assert_awaited_once_with(7, '/repo/other')

    @pytest.mark.asyncio
    async def test_normalises_not_found_exception_to_absent_error_dict(self):
        """A RAISED TaskmasterError carrying the canonical not-found phrase
        normalises to a dict that ``confirm_task_absent`` classifies exactly as
        it classifies the returned error-dict path."""

        exc = TaskmasterError('TASKMASTER_TOOL_ERROR', 'No tasks found for ID(s): 9999')
        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(side_effect=exc)

        result = await safe_get_task(taskmaster, '9999', '/repo/df')

        assert result == {'error': str(exc), 'error_type': 'TaskmasterError'}
        assert confirm_task_absent(result) is True
        assert confirm_task_present(result) is False

    @pytest.mark.asyncio
    async def test_normalises_generic_exception_to_inconclusive_error_dict(self):
        """A generic exception normalises to the same shape and is INCONCLUSIVE:
        neither classifier fires, so it can neither corroborate presence nor
        confirm absence."""

        exc = RuntimeError('backend down')
        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(side_effect=exc)

        result = await safe_get_task(taskmaster, '3045', '/repo/df')

        assert result == {'error': 'backend down', 'error_type': 'RuntimeError'}
        assert confirm_task_present(result) is False
        assert confirm_task_absent(result) is False


# ---------------------------------------------------------------------------
# corroborate_task_lookups — the single corroboration loop (task 4873)
# ---------------------------------------------------------------------------


def _present_task(task_id: object, title: str = 'A real task') -> dict:
    return {'id': str(task_id), 'title': title, 'status': 'pending'}


def _not_found_task(task_id: object = '0') -> dict:
    return {
        'error': f'No tasks found for ID(s): {task_id}',
        'error_type': 'TaskmasterError',
    }


class _PerRootTaskmaster:
    """Taskmaster double whose ``get_task`` answers per ``(str(task_id), root)``.

    Replicated from ``tests/test_stage2_stats_audit.py::_StubTaskmaster``
    rather than cross-imported (module-private fixtures are not a shared
    contract). Unmapped pairs answer not-found; a mapped ``Exception`` is
    RAISED. ``max_concurrent`` is the peak in-flight lookup count, so one
    gathered batch (== N) is distinguishable from a sequential loop (== 1).
    """

    def __init__(self, results: dict[tuple[str, str], Any] | None = None) -> None:
        self.results: dict[tuple[str, str], Any] = dict(results or {})
        self.calls: list[tuple[Any, str]] = []
        self.max_concurrent = 0
        self._in_flight = 0

    async def get_task(self, task_id: Any, project_root: str) -> Any:
        self.calls.append((task_id, project_root))
        self._in_flight += 1
        self.max_concurrent = max(self.max_concurrent, self._in_flight)
        try:
            await asyncio.sleep(0)
            result = self.results.get((str(task_id), project_root), _not_found_task(task_id))
            if isinstance(result, Exception):
                raise result
            return result
        finally:
            self._in_flight -= 1


class _ExplodingKnownProjects(dict):
    """A known_projects mapping that raises while being resolved."""

    def get(self, key, default=None):  # type: ignore[override]
        raise ValueError(f'project resolution exploded on {key!r}')


class TestCorroborateTaskLookups:
    """``corroborate_task_lookups(taskmaster, known_projects, lookups)``.

    The ONE resolve-root -> single flat gather of ``safe_get_task`` -> classify
    loop that every cross-project corroboration in reconciliation shares. Each
    lookup carries its own ``accept`` predicate, so "the task exists",
    "the task exists with this title" and "the task's hints contain these
    entries" are call sites rather than copies of the loop.

    One verdict per lookup, in input order. Fail-CLOSED on confirmation (an
    uncertain answer is never CONFIRMED) and fail-SAFE for the caller (it never
    raises). UNCONFIRMED means "we asked and it did not confirm"; UNRESOLVABLE
    means "we could not ask".
    """

    _KNOWN = {'dark_factory': '/repo/df', 'reify': '/repo/reify'}

    @staticmethod
    def _lookup(project_id, task_id, accept=None):
        return TaskLookup(
            project_id=project_id,
            task_id=task_id,
            accept=accept if accept is not None else confirm_task_present,
        )

    @pytest.mark.asyncio
    async def test_present_task_with_confirm_task_present_is_confirmed(self):
        stub = _PerRootTaskmaster({('3045', '/repo/df'): _present_task('3045')})

        verdicts = await corroborate_task_lookups(
            stub, self._KNOWN, [self._lookup('dark_factory', '3045')],
        )

        assert verdicts == (LookupVerdict.CONFIRMED,)
        assert stub.calls == [('3045', '/repo/df')]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'get_task_result',
        [
            _not_found_task('3045'),
            {'error': 'Connection timeout', 'error_type': 'TimeoutError'},
            RuntimeError('backend down'),
            None,
            {},
            'a task exists',
        ],
        ids=['not-found', 'inconclusive-error', 'raised', 'none', 'empty-dict', 'str'],
    )
    async def test_non_present_results_are_unconfirmed_never_raised(self, get_task_result):
        stub = _PerRootTaskmaster({('3045', '/repo/df'): get_task_result})

        verdicts = await corroborate_task_lookups(
            stub, self._KNOWN, [self._lookup('dark_factory', '3045')],
        )

        assert verdicts == (LookupVerdict.UNCONFIRMED,)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('project_id', [None, '', '   ', 'unknown_project'])
    async def test_unresolvable_project_issues_no_lookup(self, project_id):
        stub = _PerRootTaskmaster()

        verdicts = await corroborate_task_lookups(
            stub,
            self._KNOWN,
            [self._lookup(project_id, '3045'), self._lookup('dark_factory', '1')],
        )

        assert verdicts[0] is LookupVerdict.UNRESOLVABLE
        assert verdicts[1] is LookupVerdict.UNCONFIRMED
        assert stub.calls == [('1', '/repo/df')]

    @pytest.mark.asyncio
    async def test_each_lookup_goes_to_its_own_project_root(self):
        """Same id under two projects is two lookups against two roots —
        Taskmaster ids are per-project."""
        stub = _PerRootTaskmaster({
            ('3045', '/repo/df'): _present_task('3045'),
            ('3045', '/repo/reify'): _not_found_task('3045'),
        })

        verdicts = await corroborate_task_lookups(
            stub,
            self._KNOWN,
            [self._lookup('dark_factory', '3045'), self._lookup('reify', '3045')],
        )

        assert verdicts == (
            LookupVerdict.CONFIRMED,
            LookupVerdict.UNCONFIRMED,
        )
        assert sorted(stub.calls) == [('3045', '/repo/df'), ('3045', '/repo/reify')]

    @pytest.mark.asyncio
    async def test_resolvable_lookups_go_out_as_one_concurrent_batch(self):
        stub = _PerRootTaskmaster()
        lookups = [self._lookup('dark_factory', str(i)) for i in range(5)]
        lookups.insert(2, self._lookup('nowhere', '99'))

        verdicts = await corroborate_task_lookups(stub, self._KNOWN, lookups)

        assert len(stub.calls) == 5
        assert stub.max_concurrent == 5
        assert verdicts[2] is LookupVerdict.UNRESOLVABLE

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'taskmaster_arg, known_projects_arg',
        [
            (None, {'dark_factory': '/repo/df'}),
            (False, {'dark_factory': '/repo/df'}),
            ('STUB', None),
            ('STUB', {}),
        ],
        ids=['taskmaster-none', 'taskmaster-false', 'known-projects-none', 'known-projects-empty'],
    )
    async def test_falsy_taskmaster_or_known_projects_is_all_unresolvable(
        self, taskmaster_arg, known_projects_arg,
    ):
        stub = _PerRootTaskmaster()
        taskmaster = stub if taskmaster_arg == 'STUB' else taskmaster_arg

        verdicts = await corroborate_task_lookups(
            taskmaster,
            known_projects_arg,
            [self._lookup('dark_factory', '3045'), self._lookup('reify', '9')],
        )

        assert verdicts == (LookupVerdict.UNRESOLVABLE,) * 2
        assert stub.calls == []

    @pytest.mark.asyncio
    async def test_raising_accept_predicate_is_unconfirmed_and_isolated(self):
        def exploding_accept(result: object) -> bool:
            raise ValueError('predicate exploded')

        stub = _PerRootTaskmaster({
            ('1', '/repo/df'): _present_task('1'),
            ('2', '/repo/df'): _present_task('2'),
        })

        verdicts = await corroborate_task_lookups(
            stub,
            self._KNOWN,
            [
                self._lookup('dark_factory', '1', accept=exploding_accept),
                self._lookup('dark_factory', '2'),
            ],
        )

        assert verdicts == (
            LookupVerdict.UNCONFIRMED,
            LookupVerdict.CONFIRMED,
        )

    @pytest.mark.asyncio
    async def test_accept_predicate_receives_the_raw_get_task_result(self):
        seen: list[object] = []
        record = _present_task('7', title='Exact title')

        def recording_accept(result: object) -> bool:
            seen.append(result)
            return False

        stub = _PerRootTaskmaster({('7', '/repo/df'): record})

        verdicts = await corroborate_task_lookups(
            stub, self._KNOWN, [self._lookup('dark_factory', '7', accept=recording_accept)],
        )

        assert seen == [record]
        assert verdicts == (LookupVerdict.UNCONFIRMED,)

    @pytest.mark.asyncio
    async def test_exploding_known_projects_degrades_to_all_unresolvable(self, caplog):
        caplog.set_level(logging.WARNING, logger='fused_memory.reconciliation.task_corroboration')

        verdicts = await corroborate_task_lookups(
            _PerRootTaskmaster(),
            _ExplodingKnownProjects(dark_factory='/repo/df'),
            [self._lookup('dark_factory', '3045'), self._lookup('reify', '9')],
        )

        assert verdicts == (LookupVerdict.UNRESOLVABLE,) * 2
        assert any(
            'reconciliation.corroborate_task_lookups' in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING
        )

    @pytest.mark.asyncio
    async def test_empty_lookups_is_an_empty_tuple_with_no_io(self):
        stub = _PerRootTaskmaster()

        verdicts = await corroborate_task_lookups(stub, self._KNOWN, [])

        assert verdicts == ()
        assert stub.calls == []

    def test_task_lookup_is_frozen(self):
        lookup = self._lookup('dark_factory', '1')

        for field in dataclasses.fields(lookup):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(lookup, field.name, None)
