"""Contract tests for the durable ``metadata.pending_since`` wait anchor.

Task 3816, PRD ``plans/scheduler-dispatch-scoring-and-lock-layer-prd.md`` §C1
(task alpha). The anchor is PRODUCED here (fused-memory status chokepoints)
and CONSUMED by the orchestrator scheduler's age term (task beta) and the
watchdog idle clock (task delta); this file owns the producer side.

Three layers, deliberately separated:

* :class:`TestStampPendingSinceTransitionTable` drives the pure helper
  ``stamp_pending_since`` with no database at all — the full C1 transition
  table, its invariants, and its fail-safe contract.
* :class:`TestPendingSinceThroughStatusWriters` drives the real backend and
  observes through ``get_task``, i.e. the user-observable boundary rows
  (PRD :496-498).
* :class:`TestPendingSinceBatchIdentity` drives the interceptor's CSV branch,
  the shape ``commit_planning`` uses.
"""

from __future__ import annotations

import json
import logging

import pytest
import pytest_asyncio
from shared.task_statuses import TaskStatus

from fused_memory.backends.sqlite_task_backend import (
    SqliteTaskBackend,
    stamp_pending_since,
)
from fused_memory.config.schema import TaskmasterConfig

_LOGGER_NAME = 'fused_memory.backends.sqlite_task_backend'

# Every origin status the store can hold, including the two the PRD's
# transition-table prose abbreviates away (`merge-deferred`, `infra-hold`).
# Imported from the closed vocabulary rather than hand-listed so a new member
# joins these sweeps without an edit here.
_ALL_ORIGINS = tuple(TaskStatus)
_NON_CANCELLED_ORIGINS = tuple(s for s in _ALL_ORIGINS if s != TaskStatus.CANCELLED)

_NOW = '2026-09-18T12:00:00.000Z'
_OLDER = '2026-08-06T10:00:00.000Z'


@pytest_asyncio.fixture
async def backend(tmp_path):
    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b = SqliteTaskBackend(cfg)
    await b.start()
    yield b
    await b.close()


class TestStampPendingSinceTransitionTable:
    """The pure C1 transition table, driven directly with no DB.

    | Transition                                  | Effect            |
    |---------------------------------------------|-------------------|
    | ``* -> pending``, key absent                | stamp ``now``     |
    | ``cancelled -> pending``                    | overwrite (D3)    |
    | other ``* -> pending``, key present         | unchanged         |
    | ``pending -> *`` (any exit)                 | unchanged, never cleared |
    """

    @pytest.mark.parametrize('old_status', _ALL_ORIGINS)
    @pytest.mark.parametrize(
        'new_status',
        tuple(s for s in _ALL_ORIGINS if s != TaskStatus.PENDING),
    )
    def test_non_pending_target_never_stamps_and_never_clears(
        self, old_status, new_status
    ):
        """Row 4 of the table: only a PENDING landing is a stamping event.

        Covers every exit from ``pending`` (notably ``pending -> done`` and
        ``pending -> cancelled``), and every non-pending-to-non-pending move.
        ``None`` means "no metadata column written", so an existing anchor is
        structurally incapable of being cleared on these paths.
        """
        blob = json.dumps({'pending_since': _OLDER, 'source': 'agent-followup'})
        assert (
            stamp_pending_since(
                blob, old_status=old_status, new_status=new_status, now=_NOW
            )
            is None
        )

    def test_cancelled_to_pending_overwrites_the_anchor(self):
        """D3's ONLY reset: un-cancelling restarts the wait from scratch."""
        blob = json.dumps({'pending_since': _OLDER})
        result = stamp_pending_since(
            blob,
            old_status=TaskStatus.CANCELLED,
            new_status=TaskStatus.PENDING,
            now=_NOW,
        )
        assert result is not None
        assert json.loads(result)['pending_since'] == _NOW

    @pytest.mark.parametrize(
        'old_status',
        tuple(s for s in _NON_CANCELLED_ORIGINS if s != TaskStatus.PENDING),
    )
    def test_non_cancelled_origin_with_anchor_present_is_unchanged(self, old_status):
        """A requeue/unblock/commit must not cost a task its accrued wait."""
        blob = json.dumps({'pending_since': _OLDER})
        assert (
            stamp_pending_since(
                blob,
                old_status=old_status,
                new_status=TaskStatus.PENDING,
                now=_NOW,
            )
            is None
        )

    @pytest.mark.parametrize('old_status', (*_ALL_ORIGINS, None))
    def test_any_origin_with_anchor_absent_stamps(self, old_status):
        """Row 1: an anchorless landing always gains one.

        ``old_status=None`` is the ``add_task`` insert case — there is no
        previous status at all.
        """
        result = stamp_pending_since(
            None if old_status is None else json.dumps({'source': 'x'}),
            old_status=old_status,
            new_status=TaskStatus.PENDING,
            now=_NOW,
        )
        assert result is not None
        assert json.loads(result)['pending_since'] == _NOW

    @pytest.mark.parametrize(
        'raw_blob',
        (
            '{}',
            json.dumps({'pending_since': ''}),
            json.dumps({'pending_since': '   '}),
            json.dumps({'pending_since': None}),
            json.dumps({'pending_since': 0}),
            json.dumps({'pending_since': 1723000000}),
            json.dumps({'pending_since': {'nested': 'object'}}),
            json.dumps({'pending_since': ['list']}),
        ),
    )
    def test_unusable_anchor_value_is_treated_as_absent(self, raw_blob):
        """A present-but-unusable value must not freeze the task anchorless.

        The reader contract parses the value; a blank, ``None`` or non-string
        one cannot be parsed, so leaving it in place would permanently pin the
        task at age 0. Treating it as absent repairs the row on the next
        landing.
        """
        result = stamp_pending_since(
            raw_blob,
            old_status=TaskStatus.IN_PROGRESS,
            new_status=TaskStatus.PENDING,
            now=_NOW,
        )
        assert result is not None
        assert json.loads(result)['pending_since'] == _NOW

    def test_sibling_metadata_keys_survive_the_stamp(self):
        """Sibling preservation is identical to ``_merge_metadata(mode='merge')``."""
        siblings = {
            'source': 'agent-followup',
            'modules': ['fused-memory'],
            'invariants': {'INV-5': 'one helper'},
            'spawned_from': '3816',
        }
        result = stamp_pending_since(
            json.dumps(siblings),
            old_status=TaskStatus.BLOCKED,
            new_status=TaskStatus.PENDING,
            now=_NOW,
        )
        assert result is not None
        merged = json.loads(result)
        assert merged['pending_since'] == _NOW
        for key, value in siblings.items():
            assert merged[key] == value, f'sibling {key} not preserved byte-for-value'

    def test_stamping_is_idempotent(self):
        """Feeding the output back returns ``None`` — no second write."""
        first = stamp_pending_since(
            json.dumps({'source': 'x'}),
            old_status=TaskStatus.IN_PROGRESS,
            new_status=TaskStatus.PENDING,
            now=_OLDER,
        )
        assert first is not None
        assert (
            stamp_pending_since(
                first,
                old_status=TaskStatus.IN_PROGRESS,
                new_status=TaskStatus.PENDING,
                now=_NOW,
            )
            is None
        )

    def test_anchor_is_monotone_non_decreasing_across_a_requeue_sequence(self):
        """The C1 invariant, driven as a sequence rather than a single cell.

        A task bounced through every non-``cancelled`` origin never has its
        anchor moved — not forwards (which would erase accrued wait) and not
        backwards (which would manufacture it).
        """
        blob = stamp_pending_since(
            None,
            old_status=None,
            new_status=TaskStatus.PENDING,
            now=_OLDER,
        )
        assert blob is not None
        for step, origin in enumerate(_NON_CANCELLED_ORIGINS):
            later = f'2026-09-18T12:00:{step:02d}.000Z'
            rewritten = stamp_pending_since(
                blob, old_status=origin, new_status=TaskStatus.PENDING, now=later
            )
            assert rewritten is None, f'origin {origin} moved the anchor'
            assert json.loads(blob)['pending_since'] == _OLDER

    def test_backfilled_marker_is_never_written_by_the_live_stamp(self):
        """``pending_since_backfilled`` belongs to the migration alone.

        If the live path wrote it too, the marker would stop identifying the
        back-filled population and D4's distortion would become uncountable.
        """
        result = stamp_pending_since(
            None, old_status=None, new_status=TaskStatus.PENDING, now=_NOW
        )
        assert result is not None
        assert 'pending_since_backfilled' not in json.loads(result)

    # --- fail-safe arms -------------------------------------------------
    #
    # Each case below passes a DISTINCT task_id: the warning routes through
    # `_warn_malformed_metadata_once`, a per-process dedup gate keyed on
    # (project_root, tag, task_id), so cases sharing an id would observe zero
    # records after the first.

    @pytest.mark.parametrize(
        ('raw_blob', 'task_id'),
        (
            ('{not json at all', 38160),
            ('{"unterminated": ', 38161),
            ('[1,2,3]', 38162),
            ('42', 38163),
            ('"str"', 38164),
            ('null', 38165),
            ('true', 38166),
        ),
    )
    def test_corrupt_or_non_dict_blob_fails_safe(self, raw_blob, task_id, caplog):
        """Never raises, never clobbers — a corrupt row stays movable.

        Today a corrupt blob does not block a status write at all, because
        ``set_task_status`` never touches the metadata column. A stamping
        scheme that raised would mean one corrupt row could no longer be moved
        out of ``pending`` — a new wedge. Returning ``None`` skips the stamp:
        the status write proceeds, the task reads as anchorless, and C1's
        reader contract maps that to age 0 (it loses age rather than jumping
        the queue).

        Clobbering is the other half, and it is why this helper does NOT
        delegate to ``_merge_metadata`` (design decision 3): that function
        falls back to ``incoming`` for valid-JSON-that-is-not-a-dict, i.e. it
        would REPLACE ``[1,2,3]`` with ``{"pending_since": ...}`` and destroy
        the bytes an operator needs to repair the row.
        """
        original = raw_blob
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = stamp_pending_since(
                raw_blob,
                old_status=TaskStatus.CANCELLED,
                new_status=TaskStatus.PENDING,
                now=_NOW,
                project_root='/tmp/proj',
                tag='master',
                task_id=task_id,
            )
        assert result is None, f'must not rewrite a malformed blob; got {result!r}'
        assert raw_blob == original, 'input bytes must be left exactly as found'
        malformed = [r for r in caplog.records if 'malformed metadata' in r.message]
        assert len(malformed) == 1, (
            f'expected exactly one deduped WARNING; got {[r.message for r in malformed]}'
        )

    def test_empty_metadata_column_stamps_rather_than_warning(self, caplog):
        """A NULL/empty metadata column is absence, not corruption."""
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = stamp_pending_since(
                '', old_status=None, new_status=TaskStatus.PENDING, now=_NOW
            )
        assert result is not None
        assert json.loads(result) == {'pending_since': _NOW}
        assert [r for r in caplog.records if 'malformed metadata' in r.message] == []

    def test_corrupt_blob_warns_without_the_optional_dedup_triple(self, caplog):
        """The triple is optional; omitting it takes the plain-logger path."""
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = stamp_pending_since(
                '{oops', old_status=None, new_status=TaskStatus.PENDING, now=_NOW
            )
        assert result is None
        assert [r for r in caplog.records if 'malformed metadata' in r.message] != []


class TestPendingSinceThroughStatusWriters:
    """The user-observable boundary rows (PRD :496-498), through the real backend.

    Driven against ``SqliteTaskBackend`` directly and observed through
    ``get_task`` — the product read path. The transition-LEGALITY gate lives
    in the interceptor, not the backend, so driving the backend is the right
    scope for the write rules themselves.
    """

    async def _anchor(self, backend, project_root, task_id) -> str | None:
        one = await backend.get_task(task_id, project_root=project_root)
        return (one['metadata'] or {}).get('pending_since')

    @pytest.mark.asyncio
    async def test_anchor_survives_a_requeue(self, backend, tmp_path):
        """Boundary row 1: dispatched then requeued keeps the accrued wait.

        The scored age must be measured from the ORIGINAL landing, not from
        the requeue — a task that has already waited must not lose that wait
        because the machine dropped it (D3 continuity).
        """
        project_root = str(tmp_path)
        dto = await backend.add_task(project_root=project_root, title='requeued')
        original = await self._anchor(backend, project_root, dto['id'])
        assert original is not None

        await backend.set_task_status(
            str(dto['id']), TaskStatus.IN_PROGRESS, project_root=project_root
        )
        await backend.set_task_status(
            str(dto['id']), TaskStatus.PENDING, project_root=project_root
        )
        assert await self._anchor(backend, project_root, dto['id']) == original

    @pytest.mark.asyncio
    async def test_only_un_cancelling_resets_the_anchor(self, backend, tmp_path):
        """Boundary row 2: the reset and the non-reset asserted side by side.

        Kept in ONE test so the CONTRAST is the assertion. Two separate tests
        could drift apart, leaving the ``cancelled`` exception asserted while
        the ``blocked`` rule silently acquired it too.
        """
        project_root = str(tmp_path)
        cancelled = await backend.add_task(project_root=project_root, title='cancel me')
        blocked = await backend.add_task(project_root=project_root, title='block me')
        before_cancel = await self._anchor(backend, project_root, cancelled['id'])
        before_block = await self._anchor(backend, project_root, blocked['id'])

        for task, away in (
            (cancelled, TaskStatus.CANCELLED),
            (blocked, TaskStatus.BLOCKED),
        ):
            await backend.set_task_status(
                str(task['id']), away, project_root=project_root
            )
            await backend.set_task_status(
                str(task['id']), TaskStatus.PENDING, project_root=project_root
            )

        after_cancel = await self._anchor(backend, project_root, cancelled['id'])
        after_block = await self._anchor(backend, project_root, blocked['id'])
        assert after_cancel != before_cancel, (
            'cancelled -> pending is the ONE reset (D3); the anchor must move'
        )
        assert after_cancel is not None and after_cancel > before_cancel
        assert after_block == before_block, (
            'blocked -> pending must keep the accrued wait, unlike un-cancelling'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('exit_status', (TaskStatus.DONE, TaskStatus.CANCELLED))
    async def test_exiting_pending_never_clears_the_anchor(
        self, backend, tmp_path, exit_status
    ):
        """Row 4: ``pending -> *`` leaves the key present AND unchanged."""
        project_root = str(tmp_path)
        dto = await backend.add_task(project_root=project_root, title='exiting')
        original = await self._anchor(backend, project_root, dto['id'])

        await backend.set_task_status(
            str(dto['id']), exit_status, project_root=project_root
        )
        assert await self._anchor(backend, project_root, dto['id']) == original

    @pytest.mark.asyncio
    async def test_anchorless_task_reaching_pending_gains_an_anchor(
        self, backend, tmp_path
    ):
        """A row that predates the anchor is repaired on its next landing."""
        project_root = str(tmp_path)
        dto = await backend.add_task(
            project_root=project_root, title='legacy', status=TaskStatus.DEFERRED,
        )
        assert await self._anchor(backend, project_root, dto['id']) is None

        await backend.set_task_status(
            str(dto['id']), TaskStatus.IN_PROGRESS, project_root=project_root
        )
        assert await self._anchor(backend, project_root, dto['id']) is None
        await backend.set_task_status(
            str(dto['id']), TaskStatus.PENDING, project_root=project_root
        )
        assert await self._anchor(backend, project_root, dto['id']) is not None

    @pytest.mark.asyncio
    async def test_corrupt_metadata_row_can_still_leave_pending(
        self, backend, tmp_path
    ):
        """The fail-safe, end to end: a corrupt blob must not wedge a row.

        Before this key existed a corrupt blob could not block a status write
        at all — ``set_task_status`` never touched the metadata column. A
        stamping scheme that raised would mean one corrupt row could no longer
        be moved out of ``pending``: a new wedge, and a loud regression of an
        unrelated invariant.
        """
        project_root = str(tmp_path)
        dto = await backend.add_task(project_root=project_root, title='corrupt')
        conn = await backend._get_connection(project_root)
        await conn.execute(
            "UPDATE tasks SET metadata = 'NOT_JSON_ANCHOR' WHERE id = ?",
            (int(dto['id']),),
        )
        await conn.commit()

        result = await backend.set_task_status(
            str(dto['id']), TaskStatus.BLOCKED, project_root=project_root
        )
        assert result['tasks'][0]['newStatus'] == TaskStatus.BLOCKED

        # And the re-entry: still no raise, still no anchor invented from a
        # blob that cannot be merged into.
        result = await backend.set_task_status(
            str(dto['id']), TaskStatus.PENDING, project_root=project_root
        )
        assert result['tasks'][0]['newStatus'] == TaskStatus.PENDING
        assert await self._anchor(backend, project_root, dto['id']) is None
