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

import copy
import json
import logging
from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from shared.task_statuses import TaskStatus

from fused_memory.backends.sqlite_task_backend import (
    _MACHINE_AUTHORED_METADATA_KEYS,
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

# The forged anchor a caller would supply to manufacture a queue jump: old
# enough that task beta's age term saturates, so it is unambiguous whether a
# stored value came from the machine clock or from the payload.
_ANCIENT = '2000-01-01T00:00:00.000Z'


def _assert_machine_clock(anchor: str | None, updated_at: str) -> None:
    """Assert ``anchor`` came from the machine's clock on THIS write.

    Bracketed against the row's own ``updatedAt`` rather than an imported
    ``_now()``: the two writers take their two ``_now()`` readings in
    OPPOSITE orders (``set_task_status`` binds ``updated_at`` first, the audit
    writer stamps the anchor first), so only a tolerance holds for both — and
    a tolerance measured against an observable field keeps this file off the
    module's internals.
    """
    assert anchor is not None, 'a pending landing must be anchored'
    assert anchor > _ANCIENT, f'the forged payload value was stored: {anchor!r}'
    skew = abs(datetime.fromisoformat(anchor) - datetime.fromisoformat(updated_at))
    assert skew < timedelta(seconds=5), (
        f'anchor {anchor!r} is not this write\'s clock (updatedAt={updated_at!r})'
    )


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

    @pytest.mark.parametrize(
        ('raw_blob', 'task_id'),
        (
            ({}, 38170),
            ({'pending_since': _ANCIENT}, 38171),
            ([1, 2, 3], 38172),
            (42, 38173),
        ),
    )
    def test_a_non_string_blob_fails_safe_rather_than_raising(
        self, raw_blob, task_id, caplog
    ):
        """The same fail-safe arm for a caller that bypassed ``str | None``.

        ``add_task`` feeds this helper the return of
        ``strip_machine_authored_metadata``, which deliberately ACCEPTS a dict
        and hands one straight back, so the two neighbouring lines have to
        agree on that posture — and they did not. ``json.loads`` raises
        ``TypeError`` (not ``ValueError``) for a dict, which escaped the parse
        guard into ``add_task``, where ``task_interceptor``'s legacy
        ``except TypeError`` two-step would swallow it, retry with ``metadata``
        dropped entirely and create the task with NO metadata and no error
        surfaced — the silent fail-soft the design invariants forbid. An empty
        dict was worse still: falsy, so it took the ABSENCE arm and was
        quietly answered with a ``str``.
        """
        original = copy.deepcopy(raw_blob)
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = stamp_pending_since(
                raw_blob,  # type: ignore[arg-type]
                old_status=None,
                new_status=TaskStatus.PENDING,
                now=_NOW,
                project_root='/tmp/proj',
                tag='master',
                task_id=task_id,
            )
        assert result is None, f'must not answer a non-string blob; got {result!r}'
        assert raw_blob == original, 'input must be left exactly as found'
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
        assert before_cancel is not None and before_block is not None, (
            'both rows land in `pending`, so the insert must have anchored both '
            f'before either leaves: {before_cancel!r}, {before_block!r}'
        )

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

    @pytest.mark.asyncio
    async def test_add_task_ignores_a_caller_supplied_anchor(
        self, backend, tmp_path, caplog
    ):
        """The anchor is MACHINE-authored: an insert cannot supply its own.

        Measured exploit (reviewer_comprehensive, robustness/authority-bypass):
        ``add_task(metadata='{"pending_since": "2000-01-01T00:00:00.000Z"}')``
        stored that value verbatim. Mechanism: on an INSERT the helper is
        called with ``old_status=None``, so its ``usable and old_status !=
        CANCELLED -> return None`` arm holds (``None != CANCELLED``) and the
        caller's value is PRESERVED rather than stamped.

        Consequence: task beta scores ``age(t) = AGE_BUDGET*a/(a+AGE_HALF_SECS)``
        with ``AGE_BUDGET=500`` inside ``TIER_WIDTH=1000``, so any caller of
        ``submit_task`` hands itself ~the full age bonus and jumps the
        intra-tier queue permanently — exactly the OVER-aging D4 excluded for
        the back-fill ("can under-age but never over-age, so it cannot
        manufacture a queue jump").

        Blessing ``pending_since`` (step 2) is what made the forgery SILENT:
        the unblessed sibling in the measured payload still minted a
        ``task_metadata.schema_warning code=unknown_key`` line and the forged
        anchor minted none. So the authority check has to be explicit — and
        the refusal must be on authority grounds, not by re-introducing the
        census line a legitimate machine stamp must never produce.
        """
        project_root = str(tmp_path)
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            dto = await backend.add_task(
                project_root=project_root, title='sneaky',
                status=TaskStatus.PENDING,
                metadata=json.dumps({'pending_since': _ANCIENT, 'files': ['keep.py']}),
            )

        one = await backend.get_task(dto['id'], project_root=project_root)
        stored = one['metadata'] or {}
        # Pinned to the identity a fresh pending insert already satisfies (D5:
        # one hoisted `now` binds both the anchor and `updated_at`), which is
        # sharper than merely asserting the forged value is gone.
        assert stored['pending_since'] == one['updatedAt'], (
            'a fresh pending insert must carry the INSERT CLOCK, not the '
            f'caller-supplied anchor: {stored.get("pending_since")!r}'
        )
        assert stored['files'] == ['keep.py'], (
            'stripping the forged anchor must not disturb the sibling keys'
        )
        census_msgs = [
            r.message for r in caplog.records
            if r.levelno >= logging.WARNING and 'task_metadata.schema_warning' in r.message
        ]
        assert census_msgs == [], (
            f'the forgery is refused on AUTHORITY grounds, not by a schema '
            f'census line; got: {census_msgs}'
        )


class TestPendingSinceThroughTheAuditWriter:
    """The audit-carrying writer stamps identically (task 3816, D1).

    ``_do_set_task_status_write`` routes to ``set_status_and_stamp_audit``
    whenever ``audit_fields`` is non-empty, and ``audit_fields`` is populated
    exactly by ``reopen_reason``/``reopen_from``/``reopen_at`` and
    ``done_provenance``. A reopen IS a ``blocked|cancelled -> pending`` move —
    the highest-value transition class for this contract, and the one carrying
    D3's only reset. Wiring only the PRD-named ``set_task_status`` would leave
    every reopened task silently anchorless.
    """

    _AUDIT = {
        'reopen_reason': 'requeued by the steward',
        'reopen_from': 'cancelled',
        'reopen_at': '2026-09-18T11:59:00.000000+00:00',
    }

    async def _metadata(self, backend, project_root, task_id) -> dict:
        one = await backend.get_task(task_id, project_root=project_root)
        return one['metadata'] or {}

    @pytest.mark.asyncio
    async def test_reopen_from_cancelled_resets_anchor_and_keeps_audit(
        self, backend, tmp_path
    ):
        """The D3 reset and the audit merge coexist in the one blob."""
        project_root = str(tmp_path)
        dto = await backend.add_task(project_root=project_root, title='reopened')
        original = (await self._metadata(backend, project_root, dto['id']))[
            'pending_since'
        ]
        await backend.set_task_status(
            str(dto['id']), TaskStatus.CANCELLED, project_root=project_root
        )

        await backend.set_status_and_stamp_audit(
            str(dto['id']), TaskStatus.PENDING, project_root,
            audit_fields=dict(self._AUDIT),
        )

        merged = await self._metadata(backend, project_root, dto['id'])
        assert merged['pending_since'] > original, 'un-cancelling must reset (D3)'
        for key, value in self._AUDIT.items():
            assert merged[key] == value, f'audit field {key} lost to the stamp'

    @pytest.mark.asyncio
    async def test_reopen_from_blocked_keeps_the_existing_anchor(
        self, backend, tmp_path
    ):
        """Not every reopen is a reset — only un-cancelling is."""
        project_root = str(tmp_path)
        dto = await backend.add_task(project_root=project_root, title='unblocked')
        original = (await self._metadata(backend, project_root, dto['id']))[
            'pending_since'
        ]
        await backend.set_task_status(
            str(dto['id']), TaskStatus.BLOCKED, project_root=project_root
        )

        await backend.set_status_and_stamp_audit(
            str(dto['id']), TaskStatus.PENDING, project_root,
            audit_fields={**self._AUDIT, 'reopen_from': 'blocked'},
        )

        merged = await self._metadata(backend, project_root, dto['id'])
        assert merged['pending_since'] == original
        assert merged['reopen_reason'] == self._AUDIT['reopen_reason']

    @pytest.mark.asyncio
    async def test_done_provenance_write_never_clears_the_anchor(
        self, backend, tmp_path
    ):
        """A ``* -> done`` audit write leaves the anchor exactly as found."""
        project_root = str(tmp_path)
        dto = await backend.add_task(project_root=project_root, title='finishing')
        original = (await self._metadata(backend, project_root, dto['id']))[
            'pending_since'
        ]

        await backend.set_status_and_stamp_audit(
            str(dto['id']), TaskStatus.DONE, project_root,
            audit_fields={'done_provenance': {'kind': 'merged', 'commit': 'a' * 40}},
        )

        merged = await self._metadata(backend, project_root, dto['id'])
        assert merged['pending_since'] == original
        assert merged['done_provenance']['kind'] == 'merged'

    @pytest.mark.asyncio
    async def test_a_reopen_routes_to_the_audit_writer(self, tmp_path):
        """Pins the test above to the REAL production path.

        Without this, the three tests above could be asserting the behaviour
        of a writer nothing actually calls for a reopen. Mirrors the routing
        idiom in test_task_interceptor.py.
        """
        from unittest.mock import AsyncMock

        from fused_memory.middleware.task_interceptor import TaskInterceptor
        from fused_memory.reconciliation.event_buffer import EventBuffer

        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(
            return_value={'id': '1', 'status': 'cancelled', 'title': 'T'}
        )
        taskmaster.get_tasks = AsyncMock(return_value={'tasks': []})
        taskmaster.set_task_status = AsyncMock(return_value={'success': True})
        taskmaster.set_status_and_stamp_audit = AsyncMock(return_value={'success': True})
        reconciler = AsyncMock()
        reconciler.reconcile_task = AsyncMock(return_value={'actions': []})
        buf = EventBuffer(db_path=tmp_path / 'eb.db', buffer_size_threshold=100)
        await buf.initialize()
        try:
            interceptor = TaskInterceptor(taskmaster, reconciler, buf)
            await interceptor.set_task_status(
                '1', TaskStatus.PENDING, str(tmp_path),
                reopen_reason='requeued by the steward',
            )
        finally:
            await buf.close()

        taskmaster.set_status_and_stamp_audit.assert_called_once()
        audit_fields = taskmaster.set_status_and_stamp_audit.call_args.kwargs[
            'audit_fields'
        ]
        assert audit_fields['reopen_from'] == 'cancelled'
        taskmaster.set_task_status.assert_not_called()


class TestPendingSinceBatchIdentity:
    """One identical anchor across a ``commit_planning`` batch (PRD rule 5).

    ``commit_planning`` (server/tools.py) does not write status itself: it
    hands a CSV to ``task_interceptor.set_task_status``, which splits it and
    loops ``_apply_status_transition`` per id. Each per-id backend call would
    otherwise compute its own ``_now()``, and since each id runs a full gated
    transaction the anchors drift by tens of milliseconds across a batch.

    That drift is NOT a tie. Under task beta's
    ``age(t) = AGE_BUDGET·a/(a+AGE_HALF_SECS)`` a 50 ms spread is a ~1e-4
    score delta — enough for the ``(-score, numeric_id, task_id)`` sort to
    order the batch by COMMIT SEQUENCE, when PRD rule 5 requires intra-batch
    order to fall through to CPM and then numeric id. So the assertion below
    is on set CARDINALITY, not on a tolerance window: a millisecond spread is
    exactly what a tolerance-based assertion would wave through.
    """

    async def _stack(self, tmp_path):
        from fused_memory.middleware.task_interceptor import TaskInterceptor
        from fused_memory.reconciliation.event_buffer import EventBuffer

        cfg = TaskmasterConfig(project_root=str(tmp_path))
        backend = SqliteTaskBackend(cfg)
        await backend.start()
        event_buffer = EventBuffer(
            db_path=tmp_path / 'batch_eb.db', buffer_size_threshold=100,
        )
        await event_buffer.initialize()
        interceptor = TaskInterceptor(backend, None, event_buffer)
        return interceptor, backend, event_buffer

    async def _anchors(self, backend, project_root, ids) -> list[str | None]:
        out = []
        for task_id in ids:
            one = await backend.get_task(task_id, project_root=project_root)
            out.append((one['metadata'] or {}).get('pending_since'))
        return out

    @pytest.mark.asyncio
    async def test_csv_flip_to_pending_stamps_one_identical_anchor(self, tmp_path):
        """The batch is one atomic flip, so it must read as one instant."""
        project_root = str(tmp_path)
        interceptor, backend, event_buffer = await self._stack(tmp_path)
        try:
            ids = []
            for n in range(4):
                dto = await backend.add_task(
                    project_root=project_root, title=f'parked {n}',
                    status=TaskStatus.DEFERRED,
                )
                ids.append(str(dto['id']))
            assert await self._anchors(backend, project_root, ids) == [None] * 4

            await interceptor.set_task_status(
                ','.join(ids), TaskStatus.PENDING, project_root,
            )

            anchors = await self._anchors(backend, project_root, ids)
            assert None not in anchors, f'every batch member must be anchored: {anchors}'
            # Re-bound as a narrowed list so the diagnostic below can sort it;
            # the assertion above is what makes dropping the None arm sound.
            stamped = [one for one in anchors if one is not None]
            assert len(set(stamped)) == 1, (
                'a commit_planning batch must stamp ONE identical anchor; got '
                f'{sorted(set(stamped))} — per-id _now() drift would order the '
                'batch by commit sequence instead of falling through to CPM'
            )
        finally:
            await backend.close()
            await event_buffer.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'target', (TaskStatus.DEFERRED, TaskStatus.CANCELLED)
    )
    async def test_csv_flip_to_a_non_pending_status_writes_no_anchor(
        self, tmp_path, target
    ):
        """The batch clock must not leak into non-landing transitions."""
        project_root = str(tmp_path)
        interceptor, backend, event_buffer = await self._stack(tmp_path)
        try:
            ids = []
            for n in range(3):
                dto = await backend.add_task(
                    project_root=project_root, title=f'held {n}',
                    status=TaskStatus.BLOCKED,
                )
                ids.append(str(dto['id']))

            await interceptor.set_task_status(
                ','.join(ids), target, project_root,
            )
            assert await self._anchors(backend, project_root, ids) == [None] * 3
        finally:
            await backend.close()
            await event_buffer.close()

    @pytest.mark.asyncio
    async def test_batch_member_with_an_existing_anchor_keeps_its_own(self, tmp_path):
        """The batch clock supplies ``now``; it does not override the table.

        A member that already waited must not be dragged forward to the batch
        stamp — that would erase its accrued wait, which is exactly what D3
        forbids.
        """
        project_root = str(tmp_path)
        interceptor, backend, event_buffer = await self._stack(tmp_path)
        try:
            veteran = await backend.add_task(project_root=project_root, title='veteran')
            older = (
                await backend.get_task(veteran['id'], project_root=project_root)
            )['metadata']['pending_since']
            await backend.set_task_status(
                str(veteran['id']), TaskStatus.BLOCKED, project_root=project_root
            )
            newcomer = await backend.add_task(
                project_root=project_root, title='newcomer',
                status=TaskStatus.DEFERRED,
            )

            await interceptor.set_task_status(
                f'{veteran["id"]},{newcomer["id"]}', TaskStatus.PENDING, project_root,
            )

            kept, stamped = await self._anchors(
                backend, project_root, [str(veteran['id']), str(newcomer['id'])]
            )
            assert kept == older, 'an existing anchor must survive the batch flip'
            assert stamped is not None and stamped > older
        finally:
            await backend.close()
            await event_buffer.close()


class TestPendingSinceIsMachineAuthored:
    """No CALLER may write the wait-anchor keys (task 3816 review remediation).

    A separate class from :class:`TestPendingSinceThroughStatusWriters`
    deliberately: the subject here is write AUTHORITY, not the C1 transition
    table. The ``add_task``-landing-in-``pending`` row stays with the boundary
    rows in that class (``test_add_task_ignores_a_caller_supplied_anchor``)
    because an insert into ``pending`` IS a status-writer boundary row; the
    four caller boundaries below are covered here.

    Why authority matters more than it looks: the anchor is the scheduler's
    INPUT, so a caller able to write it prices its own dispatch. Task beta
    scores ``age(t) = AGE_BUDGET*a/(a+AGE_HALF_SECS)`` with ``AGE_BUDGET=500``
    inside ``TIER_WIDTH=1000``, so a forged ancient value is ~the full age
    bonus and a permanent intra-tier queue jump — the OVER-aging PRD D4
    excluded even for the machine's own back-fill. Every one of the cases
    below was MEASURED storing the forged value before the sanitizer landed.
    """

    async def _metadata(self, backend, project_root, task_id) -> dict:
        one = await backend.get_task(task_id, project_root=project_root)
        return one['metadata'] or {}

    @pytest.mark.asyncio
    async def test_parked_insert_cannot_pre_age_its_later_commit(
        self, backend, tmp_path
    ):
        """Case A: ``deferred`` insert + ``commit_planning``'s landing flip.

        MEASURED to keep the forged value: the insert returned early on
        ``new_status != pending`` without stamping, so the forged key rode
        along and the later landing then hit the "key present -> unchanged"
        arm. This is the MORE exploitable path than the reviewed one, since
        ``planning_mode`` + ``commit_planning`` is how agents actually file
        tasks. Hence the strip at ``add_task`` is unconditional across every
        insert status, not scoped to ``pending``.
        """
        project_root = str(tmp_path)
        dto = await backend.add_task(
            project_root=project_root, title='parked', status=TaskStatus.DEFERRED,
            metadata=json.dumps({'pending_since': _ANCIENT, 'files': ['a.py']}),
        )
        parked = await self._metadata(backend, project_root, dto['id'])
        assert 'pending_since' not in parked, (
            'a parked row must accrue no wait until it is actually committed'
        )

        await backend.set_task_status(
            str(dto['id']), TaskStatus.PENDING, project_root=project_root
        )
        one = await backend.get_task(dto['id'], project_root=project_root)
        _assert_machine_clock(one['metadata']['pending_since'], one['updatedAt'])
        assert one['metadata']['files'] == ['a.py']

    @pytest.mark.asyncio
    async def test_update_task_cannot_move_the_anchor_backward(
        self, backend, tmp_path, caplog
    ):
        """Case B: the PUBLIC metadata writer, in its default merge mode.

        MEASURED to move a live anchor BACKWARD (``2026-09-19T06:29:11.384Z``
        -> ``2000-01-01T00:00:00.000Z``), breaking the monotone-non-decreasing
        invariant ``stamp_pending_since``'s own docstring asserts — and in the
        over-age direction, which is the exploitable one.

        The forged key is IGNORED, not rejected: ``update_task`` is reachable
        from blob round-trips, so raising would add a failure mode on data a
        caller merely echoed back (design decision 9). The call must still
        succeed and its other keys must still apply.

        The refusal must also NAME the row. This writer strips before tag
        normalization and ``_parse_task_id``, so it has no dedup triple — but
        an operator reading ``task_id=None`` on the one writer measured to
        move a live anchor backward cannot tell which task was targeted, so
        the identity travels on its own argument.
        """
        project_root = str(tmp_path)
        dto = await backend.add_task(project_root=project_root, title='public writer')
        original = (await self._metadata(backend, project_root, dto['id']))[
            'pending_since'
        ]

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = await backend.update_task(
                str(dto['id']), project_root=project_root,
                metadata=json.dumps({'pending_since': _ANCIENT, 'files': ['b.py']}),
            )
        assert result['updated'] is True, 'ignoring the forged key must not fail the call'

        stripped = [
            r.message for r in caplog.records
            if 'task_metadata.machine_authored_key_stripped' in r.message
        ]
        assert len(stripped) == 1, f'expected one strip WARNING; got {stripped}'
        assert f'task_id={dto["id"]} tag=' in stripped[0], (
            f'the warning must name the task it refused; got {stripped[0]!r}'
        )

        merged = await self._metadata(backend, project_root, dto['id'])
        assert merged['pending_since'] == original, (
            'the stored anchor must be untouched by a caller-supplied value'
        )
        assert merged['files'] == ['b.py'], (
            'the non-anchor keys of the same call must still be applied'
        )

    @pytest.mark.asyncio
    async def test_replace_mode_still_drops_the_anchor(self, backend, tmp_path):
        """Case B, second half: design decision 8 must hold UNCHANGED.

        ``metadata_mode='replace'`` deliberately wipes the blob, anchor
        included — fail-safe, since the row then reads as anchorless (age 0)
        rather than pre-aged. Stripping the INCOMING key must not accidentally
        start preserving the stored one through replace: that would be an
        unreviewed scope expansion, and it is asserted here so the sanitizer
        cannot silently acquire it later.
        """
        project_root = str(tmp_path)
        dto = await backend.add_task(project_root=project_root, title='replaced')
        assert 'pending_since' in await self._metadata(backend, project_root, dto['id'])

        await backend.update_task(
            str(dto['id']), project_root=project_root,
            metadata=json.dumps({'files': ['c.py']}), metadata_mode='replace',
        )
        replaced = await self._metadata(backend, project_root, dto['id'])
        assert 'pending_since' not in replaced, (
            "metadata_mode='replace' still wipes the anchor (D8) — fail-safe, "
            'the row reads as anchorless rather than pre-aged'
        )

    @pytest.mark.asyncio
    async def test_backfilled_marker_cannot_be_forged_by_a_caller(
        self, backend, tmp_path
    ):
        """Case C: the D4 census marker must identify exactly one population.

        MEASURED stored verbatim. ``pending_since_backfilled`` marks the rows
        the one-shot v4 -> v5 migration anchored from ``updated_at``, so the
        under-aging distortion D4 accepted stays countable. A caller able to
        set it makes that census meaningless.
        """
        project_root = str(tmp_path)
        dto = await backend.add_task(
            project_root=project_root, title='forged marker',
            metadata=json.dumps({'pending_since_backfilled': True, 'files': ['d.py']}),
        )
        stored = await self._metadata(backend, project_root, dto['id'])
        assert 'pending_since_backfilled' not in stored
        assert stored['files'] == ['d.py']
        assert 'pending_since' in stored, (
            "the machine's own anchor still lands on the same insert"
        )

    @pytest.mark.asyncio
    async def test_audit_fields_cannot_smuggle_an_anchor(self, backend, tmp_path):
        """Case D: falsifies the composition-order claim in the writer's comment.

        ``set_status_and_stamp_audit`` merges ``audit_fields`` FIRST and stamps
        SECOND, and the in-code comment claimed that order alone meant a caller
        passing ``pending_since`` inside ``audit_fields`` "cannot bypass the
        transition table". MEASURED FALSE on an ANCHORLESS row: the audit merge
        injects the key first, so the helper then sees it as already present
        and returns "unchanged" — persisting
        ``{"reopen_reason": "x", "pending_since": "2000-01-01T00:00:00.000Z"}``.
        Composition order was never sufficient; the audit fields have to be
        sanitized before the merge.
        """
        project_root = str(tmp_path)
        dto = await backend.add_task(
            project_root=project_root, title='reopened', status=TaskStatus.DEFERRED,
        )
        assert 'pending_since' not in await self._metadata(
            backend, project_root, dto['id']
        )

        await backend.set_status_and_stamp_audit(
            str(dto['id']), TaskStatus.PENDING, project_root,
            audit_fields={
                'reopen_reason': 'x',
                'reopen_from': 'deferred',
                'pending_since': _ANCIENT,
            },
        )

        one = await backend.get_task(dto['id'], project_root=project_root)
        _assert_machine_clock(one['metadata']['pending_since'], one['updatedAt'])
        assert one['metadata']['reopen_reason'] == 'x'
        assert one['metadata']['reopen_from'] == 'deferred', (
            'every legitimate audit field must still persist'
        )

    @pytest.mark.asyncio
    async def test_stamp_audit_metadata_cannot_smuggle_an_anchor(
        self, backend, tmp_path
    ):
        """Case E: the same read-modify-write merge shape, privileged.

        Reachable only from the interceptor, so the stakes are lower than the
        public writers — but asserted anyway so the invariant is UNIFORMLY
        enforced from one implementation rather than being a per-site
        judgement call (heuristic 10).
        """
        project_root = str(tmp_path)
        dto = await backend.add_task(project_root=project_root, title='audit stamp')
        original = (await self._metadata(backend, project_root, dto['id']))[
            'pending_since'
        ]

        await backend.stamp_audit_metadata(
            str(dto['id']), project_root,
            fields={'reopen_reason': 'y', 'pending_since': _ANCIENT},
        )

        merged = await self._metadata(backend, project_root, dto['id'])
        assert merged['pending_since'] == original
        assert merged['reopen_reason'] == 'y'

    @pytest.mark.asyncio
    async def test_a_stripped_forgery_is_countable(self, backend, tmp_path, caplog):
        """The refusal is observable, because the blessing made it silent.

        Before the Tier-A blessing (step 2) a forged ``pending_since`` minted a
        ``task_metadata.schema_warning code=unknown_key`` line; blessing it
        removed the only signal a forgery produced. The sanitizer restores
        one under its OWN token, deliberately distinct from both the schema
        census and the read-path malformed-blob census so the three never
        conflate.
        """
        project_root = str(tmp_path)
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            await backend.add_task(
                project_root=project_root, title='noisy forgery',
                metadata=json.dumps({'pending_since': _ANCIENT}),
            )

        stripped = [
            r.message for r in caplog.records
            if 'task_metadata.machine_authored_key_stripped' in r.message
        ]
        assert len(stripped) == 1, (
            f'expected exactly one countable strip WARNING; got: {stripped}'
        )
        assert 'pending_since' in stripped[0]
        conflated = [
            r.message for r in caplog.records
            if 'task_metadata.schema_warning' in r.message
            or 'malformed metadata' in r.message
        ]
        assert conflated == [], (
            f'the strip token must not conflate with the other two censuses; '
            f'got: {conflated}'
        )

    def test_machine_authored_keys_are_pinned_and_contained(self):
        """Drift guard on the two sets, pinning CONTAINMENT and not equality.

        The sets answer different questions — blessed is "the schema
        recognises this key on READ", machine-authored is "no caller may WRITE
        it" — and most blessed keys are legitimately caller-authored, so they
        are deliberately NOT in lockstep (design decision 10). What must hold
        is one direction: a key no caller may write still has to be a key the
        schema recognises, or every machine write would mint an
        ``unknown_key`` census line.
        """
        from shared.task_metadata import _BLESSED_METADATA_KEYS

        assert sorted(_MACHINE_AUTHORED_METADATA_KEYS) == [
            'pending_since', 'pending_since_backfilled',
        ]
        assert _MACHINE_AUTHORED_METADATA_KEYS <= _BLESSED_METADATA_KEYS, (
            'machine-authored keys must be blessed, or the status chokepoints '
            'mint an unknown_key census line on every write: '
            f'{sorted(_MACHINE_AUTHORED_METADATA_KEYS - _BLESSED_METADATA_KEYS)}'
        )
