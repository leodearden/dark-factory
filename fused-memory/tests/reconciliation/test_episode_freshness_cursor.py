"""Tests for the episode/Mem0 freshness cursor comparison (task 4574).

Regression coverage for the 894fbe90 incident: ``assemble_payload``'s "new
episodes since last reconciliation" filter compared timestamps as STRINGS —
``str(watermark.last_episode_timestamp)`` renders with a space separator,
while episode ``created_at`` values arrive from
``services/memory_service.py::_created_at_to_utc_iso`` as ISO-8601 with a
``T`` separator. Because ``'T'`` (0x54) sorts after ``' '`` (0x20), the
lexical ``>`` degenerated to date-granularity: any episode from the
watermark's own calendar day compared as "newer" regardless of its actual
time, so a same-day-earlier episode re-surfaced on every cycle forever.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import StageId, StageReport, Watermark
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.stages.base import BaseStage
from fused_memory.reconciliation.stages.memory_consolidator import (
    MemoryConsolidator,
    _is_newer_than_watermark,
)


def _scope(project_id: str, project_root: str) -> ProjectScope:
    """Build a ProjectScope from raw strings — DRYs the many test call sites."""
    return ProjectScope(ProjectId(project_id), ProjectRoot(project_root))


def _make_consolidator(project_root: str = '/tmp/test') -> MemoryConsolidator:
    """Build a MemoryConsolidator with mocked deps — mirrors test_assemble_payload_snapshot_filter.py."""
    config = ReconciliationConfig()
    memory_mock = AsyncMock()
    memory_mock.get_episodes = AsyncMock(return_value=[])
    memory_mock.mem0 = AsyncMock()
    memory_mock.mem0.get_all = AsyncMock(return_value={'results': []})
    memory_mock.get_status = AsyncMock(return_value={})

    stage = MemoryConsolidator(
        StageId.memory_consolidator,
        memory_mock,
        AsyncMock(),  # taskmaster
        AsyncMock(),  # journal
        config,
        scope=_scope('test_project', project_root),
    )
    stage.episode_limit = 5
    stage.memory_limit = 10
    return stage


INCIDENT_UUID = '894fbe90-2eee-4329-83b4-eddd9f81e48d'
# LATER the same calendar day as the incident episode's created_at below.
WATERMARK_TS = datetime(2026, 8, 20, 12, 0, 0, tzinfo=UTC)


class TestEpisodeFreshnessCursor:
    """``assemble_payload``'s episode filter must compare instants, not strings."""

    @pytest.mark.asyncio
    async def test_same_day_earlier_episode_excluded(self):
        """The 894fbe90 incident: an episode from earlier the SAME calendar
        day as the watermark must be excluded, not re-surfaced forever."""
        stage = _make_consolidator()
        stage.memory.get_episodes = AsyncMock(
            return_value=[
                {
                    'uuid': INCIDENT_UUID,
                    'created_at': '2026-08-20T01:52:27+00:00',
                    'content': 'incident episode content',
                }
            ]
        )
        watermark = Watermark(project_id='test_project', last_episode_timestamp=WATERMARK_TS)

        result = await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        assert '### New Episodes Since Last Reconciliation (0)' in result, (
            f'Expected header (0); got result:\n{result!r}'
        )
        assert INCIDENT_UUID not in result, (
            f'Incident episode must not re-surface; got result:\n{result!r}'
        )

    @pytest.mark.asyncio
    async def test_genuinely_newer_episode_included(self):
        """An episode from the NEXT calendar day must still be surfaced —
        guards against a fix that degenerates into "filter everything"."""
        stage = _make_consolidator()
        stage.memory.get_episodes = AsyncMock(
            return_value=[
                {
                    'uuid': 'newer-episode',
                    'created_at': '2026-08-21T03:00:00+00:00',
                    'content': 'genuinely newer content',
                }
            ]
        )
        watermark = Watermark(project_id='test_project', last_episode_timestamp=WATERMARK_TS)

        result = await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        assert '### New Episodes Since Last Reconciliation (1)' in result, (
            f'Expected header (1); got result:\n{result!r}'
        )
        assert 'newer-episode' in result

    @pytest.mark.asyncio
    async def test_older_episode_excluded(self):
        """An episode from the PREVIOUS calendar day must be excluded."""
        stage = _make_consolidator()
        stage.memory.get_episodes = AsyncMock(
            return_value=[
                {
                    'uuid': 'older-episode',
                    'created_at': '2026-08-19T23:59:59+00:00',
                    'content': 'older content',
                }
            ]
        )
        watermark = Watermark(project_id='test_project', last_episode_timestamp=WATERMARK_TS)

        result = await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        assert '### New Episodes Since Last Reconciliation (0)' in result, (
            f'Expected header (0); got result:\n{result!r}'
        )
        assert 'older-episode' not in result


class TestMem0FreshnessCursor:
    """The Mem0 ``new_memories`` filter carries the IDENTICAL string-compare
    defect as the episode filter, on a separate code path. Mem0 ``created_at``
    values are ISO-with-``T`` but — unlike episodes — are NOT normalized by
    ``_created_at_to_utc_iso``, so they can also carry non-UTC offsets."""

    @pytest.mark.asyncio
    async def test_same_day_earlier_memory_excluded(self):
        """A memory from earlier the SAME calendar day as the watermark must
        be excluded, not re-surfaced forever (mirrors the episode incident)."""
        stage = _make_consolidator()
        stage.memory.mem0.get_all = AsyncMock(
            return_value={
                'results': [
                    {
                        'id': 'same-day-earlier-mem',
                        'created_at': '2026-08-20T02:00:00+00:00',
                        'memory': 'same-day-earlier content',
                        'metadata': {'category': 'temporal_facts'},
                    }
                ]
            }
        )
        watermark = Watermark(project_id='test_project', last_memory_timestamp=WATERMARK_TS)

        result = await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        assert '### New Mem0 Memories Since Last Reconciliation (0)' in result, (
            f'Expected header (0); got result:\n{result!r}'
        )
        assert 'same-day-earlier-mem' not in result, (
            f'Same-day-earlier memory must not re-surface; got result:\n{result!r}'
        )

    @pytest.mark.asyncio
    async def test_genuinely_newer_memory_included(self):
        """A memory from the NEXT calendar day must still be surfaced."""
        stage = _make_consolidator()
        stage.memory.mem0.get_all = AsyncMock(
            return_value={
                'results': [
                    {
                        'id': 'newer-mem',
                        'created_at': '2026-08-21T03:00:00+00:00',
                        'memory': 'genuinely newer content',
                        'metadata': {'category': 'temporal_facts'},
                    }
                ]
            }
        )
        watermark = Watermark(project_id='test_project', last_memory_timestamp=WATERMARK_TS)

        result = await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        assert '### New Mem0 Memories Since Last Reconciliation (1)' in result, (
            f'Expected header (1); got result:\n{result!r}'
        )
        assert 'newer-mem' in result

    @pytest.mark.asyncio
    async def test_updated_at_fallback_newer_included(self):
        """No created_at, but a genuinely-newer updated_at: the existing
        created_at -> updated_at fallback must still surface the memory."""
        stage = _make_consolidator()
        stage.memory.mem0.get_all = AsyncMock(
            return_value={
                'results': [
                    {
                        'id': 'fallback-newer-mem',
                        'updated_at': '2026-08-21T03:00:00+00:00',
                        'memory': 'fallback newer content',
                        'metadata': {'category': 'temporal_facts'},
                    }
                ]
            }
        )
        watermark = Watermark(project_id='test_project', last_memory_timestamp=WATERMARK_TS)

        result = await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        assert '### New Mem0 Memories Since Last Reconciliation (1)' in result, (
            f'Expected header (1); got result:\n{result!r}'
        )
        assert 'fallback-newer-mem' in result

    @pytest.mark.asyncio
    async def test_updated_at_fallback_same_day_earlier_excluded(self):
        """No created_at, and a same-day-earlier updated_at: the fallback
        value must go through the same instant comparison and be excluded."""
        stage = _make_consolidator()
        stage.memory.mem0.get_all = AsyncMock(
            return_value={
                'results': [
                    {
                        'id': 'fallback-earlier-mem',
                        'updated_at': '2026-08-20T02:00:00+00:00',
                        'memory': 'fallback earlier content',
                        'metadata': {'category': 'temporal_facts'},
                    }
                ]
            }
        )
        watermark = Watermark(project_id='test_project', last_memory_timestamp=WATERMARK_TS)

        result = await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        assert '### New Mem0 Memories Since Last Reconciliation (0)' in result, (
            f'Expected header (0); got result:\n{result!r}'
        )
        assert 'fallback-earlier-mem' not in result, (
            f'Same-day-earlier fallback memory must not re-surface; got result:\n{result!r}'
        )


# LATER the same calendar day as the incident episode, matching WATERMARK_TS above.
_WM = datetime(2026, 8, 20, 12, 0, 0, tzinfo=UTC)


class TestIsNewerThanWatermarkTrue:
    """Pins the instant-comparison contract independently of payload assembly.

    Every case here is a genuinely-newer instant relative to _WM, expressed
    in a different textual shape than _WM itself — the whole point of
    comparing parsed instants rather than strings is that the shape must not
    matter.
    """

    @pytest.mark.parametrize(
        'raw_ts',
        [
            pytest.param('2026-08-20T13:00:00+00:00', id='plainly-newer'),
            pytest.param('2026-08-20T13:00:00Z', id='z-suffix'),
            pytest.param('2026-08-20T06:00:00-07:00', id='non-utc-offset-newer'),
            pytest.param('2026-08-20T13:00:00', id='naive-assumed-utc'),
            pytest.param('2026-08-20T12:00:00.000001+00:00', id='one-microsecond-after'),
        ],
    )
    def test_returns_true(self, raw_ts):
        assert _is_newer_than_watermark(raw_ts, _WM) is True, (
            f'Expected {raw_ts!r} to be newer than watermark {_WM.isoformat()!r}'
        )

    def test_non_utc_offset_guards_false_negative(self):
        """A non-UTC offset that IS genuinely newer must not be silently
        dropped by a naive string-style comparison.

        With watermark 2026-08-21T02:00:00+00:00, the string
        '2026-08-20T20:00:00-07:00' (== 2026-08-21T03:00:00Z, newer by one
        hour) sorts LESS than the watermark string lexically ('2026-08-20'
        < '2026-08-21'), so a str(watermark)-style comparison would wrongly
        exclude it. The instant comparison must include it.
        """
        watermark = datetime(2026, 8, 21, 2, 0, 0, tzinfo=UTC)
        raw_ts = '2026-08-20T20:00:00-07:00'  # == 2026-08-21T03:00:00+00:00
        assert _is_newer_than_watermark(raw_ts, watermark) is True, (
            f'Expected {raw_ts!r} (= 2026-08-21T03:00:00+00:00) to be newer than '
            f'watermark {watermark.isoformat()!r}'
        )


class TestIsNewerThanWatermarkFalse:
    """Cases that must NOT be classified as newer than _WM."""

    @pytest.mark.parametrize(
        'raw_ts',
        [
            pytest.param('2026-08-20T01:52:27+00:00', id='incident-same-day-earlier'),
            pytest.param('2026-08-20T12:00:00+00:00', id='exactly-equal'),
            pytest.param('2026-08-19T23:59:59+00:00', id='older'),
        ],
    )
    def test_returns_false(self, raw_ts):
        assert _is_newer_than_watermark(raw_ts, _WM) is False, (
            f'Expected {raw_ts!r} to NOT be newer than watermark {_WM.isoformat()!r}'
        )


class TestIsNewerThanWatermarkNaiveWatermarkSymmetry:
    """A naive watermark must behave identically to its UTC-aware twin."""

    @pytest.mark.parametrize(
        'raw_ts, expected',
        [
            pytest.param('2026-08-20T13:00:00+00:00', True, id='newer'),
            pytest.param('2026-08-20T01:52:27+00:00', False, id='same-day-earlier'),
            pytest.param('2026-08-20T12:00:00+00:00', False, id='exactly-equal'),
        ],
    )
    def test_naive_watermark_matches_aware_twin(self, raw_ts, expected):
        naive_wm = datetime(2026, 8, 20, 12, 0, 0)
        aware_wm = datetime(2026, 8, 20, 12, 0, 0, tzinfo=UTC)

        # Must not raise TypeError on naive-vs-aware datetime comparison.
        naive_result = _is_newer_than_watermark(raw_ts, naive_wm)

        assert naive_result is expected, (
            f'Expected _is_newer_than_watermark({raw_ts!r}, <naive watermark>) == {expected}, '
            f'got {naive_result}'
        )
        assert naive_result == _is_newer_than_watermark(raw_ts, aware_wm), (
            'Naive watermark must behave identically to its UTC-aware twin'
        )


class TestUndatableFreshnessRecords:
    """Undatable records (missing/empty/unparseable timestamp) are excluded
    from both 'new' lists AND counted/logged — never silently dropped, per
    design invariant INV-2 ``structured-facts-at-failure``. An undatable
    record must never become a permanent re-surfacer like 894fbe90."""

    @pytest.mark.asyncio
    async def test_undatable_episodes_and_memories_excluded_and_counted(self):
        stage = _make_consolidator()
        stage.memory.get_episodes = AsyncMock(
            return_value=[
                {'uuid': 'ep-none', 'created_at': None, 'content': 'x'},
                {'uuid': 'ep-empty', 'created_at': '', 'content': 'x'},
                {'uuid': 'ep-garbage', 'created_at': 'not-a-timestamp', 'content': 'x'},
            ]
        )
        stage.memory.mem0.get_all = AsyncMock(
            return_value={
                'results': [
                    {'id': 'mem-none', 'created_at': None, 'memory': 'x', 'metadata': {}},
                    {'id': 'mem-empty', 'created_at': '', 'memory': 'x', 'metadata': {}},
                    {'id': 'mem-garbage', 'created_at': 'not-a-timestamp', 'memory': 'x', 'metadata': {}},
                ]
            }
        )
        watermark = Watermark(
            project_id='test_project',
            last_episode_timestamp=WATERMARK_TS,
            last_memory_timestamp=WATERMARK_TS,
        )

        result = await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        assert '### New Episodes Since Last Reconciliation (0)' in result, (
            f'Expected header (0); got result:\n{result!r}'
        )
        assert '### New Mem0 Memories Since Last Reconciliation (0)' in result, (
            f'Expected header (0); got result:\n{result!r}'
        )
        for record_id in ('ep-none', 'ep-empty', 'ep-garbage', 'mem-none', 'mem-empty', 'mem-garbage'):
            assert record_id not in result, f'{record_id} must not surface; got result:\n{result!r}'

        assert stage._undatable_freshness_records == 6, (
            f'Expected 6 undatable records (3 episodes + 3 memories), '
            f'got {stage._undatable_freshness_records}'
        )

    @pytest.mark.asyncio
    async def test_run_copies_undatable_freshness_records_into_stats(self):
        """Mirrors TestRunSurfacesSnapshotStrippedStat in
        test_assemble_payload_snapshot_filter.py: run() must copy the
        pre-set instance attr into report.stats."""
        stage = _make_consolidator()
        stage.scope = _scope('test_project', stage.scope.project_root)
        stage._undatable_freshness_records = 4

        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
        )

        with patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id='run-undatable-stats',
            )

        assert report.stats.get('stage1_undatable_freshness_records') == 4, (
            f'Expected report.stats["stage1_undatable_freshness_records"]=4, '
            f'got stats={report.stats!r}'
        )

    @pytest.mark.asyncio
    async def test_stat_present_and_zero_on_clean_run(self):
        """The stat must be unconditionally present (no .get(..., 0) fallback
        needed downstream), matching the file's stage1_flag_markers_acknowledged
        / stage1_cycle_summary_ledger_written convention."""
        stage = _make_consolidator()
        stage.scope = _scope('test_project', stage.scope.project_root)

        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
        )

        with patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id='run-undatable-stats-clean',
            )

        assert 'stage1_undatable_freshness_records' in report.stats, (
            f'Expected key always present; got stats={report.stats!r}'
        )
        assert report.stats['stage1_undatable_freshness_records'] == 0

    @pytest.mark.asyncio
    async def test_undatable_count_resets_per_call(self):
        """Reusing one stage instance across two assemble_payload calls must
        NOT accumulate the counter — mirrors the _fetch_degraded_sources
        reset hazard already called out in assemble_payload."""
        stage = _make_consolidator()
        watermark = Watermark(project_id='test_project', last_episode_timestamp=WATERMARK_TS)

        stage.memory.get_episodes = AsyncMock(
            return_value=[{'uuid': 'ep-garbage', 'created_at': 'not-a-timestamp', 'content': 'x'}]
        )
        await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])
        assert stage._undatable_freshness_records == 1

        stage.memory.get_episodes = AsyncMock(
            return_value=[{'uuid': 'ep-newer', 'created_at': '2026-08-21T03:00:00+00:00', 'content': 'y'}]
        )
        await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])
        assert stage._undatable_freshness_records == 0, (
            f'Expected counter to reset per call, not accumulate; '
            f'got {stage._undatable_freshness_records}'
        )


class TestWatermarkCutoffDisclosure:
    """The '### Previous Reconciliation' section must disclose the episode
    and memory freshness cutoffs the filters above just compared against.

    Their invisibility is precisely why the 894fbe90 bug survived three
    full cycles of re-investigation with nobody able to see the cursor: the
    payload told the reader a filtered count ("New Episodes ... (1)") but
    never the instant it was filtered against, so there was nothing to
    cross-check the count with.
    """

    @pytest.mark.asyncio
    async def test_episode_and_memory_cutoffs_appear_in_isoformat_form(self):
        """Both cutoff instants must be disclosed, rendered via .isoformat()
        (a 'T' separator) — never a bare str(datetime) rendering, which
        would reintroduce the exact space-separator ambiguity task 4574
        removed from the comparison itself."""
        stage = _make_consolidator()
        episode_ts = datetime(2026, 8, 20, 12, 0, 0, tzinfo=UTC)
        memory_ts = datetime(2026, 8, 21, 6, 30, 0, tzinfo=UTC)
        watermark = Watermark(
            project_id='test_project',
            last_full_run_id='run-abc',
            last_full_run_completed=datetime(2026, 8, 22, 0, 0, 0, tzinfo=UTC),
            last_episode_timestamp=episode_ts,
            last_memory_timestamp=memory_ts,
        )

        result = await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        assert episode_ts.isoformat() in result, (
            f'Expected episode cutoff {episode_ts.isoformat()!r} disclosed in payload; '
            f'got:\n{result!r}'
        )
        assert memory_ts.isoformat() in result, (
            f'Expected memory cutoff {memory_ts.isoformat()!r} disclosed in payload; '
            f'got:\n{result!r}'
        )
        assert str(episode_ts) not in result, (
            f'Must never render the space-separated str(datetime) form '
            f'{str(episode_ts)!r} — that is the exact ambiguity task 4574 removes; '
            f'got:\n{result!r}'
        )
        assert str(memory_ts) not in result, (
            f'Must never render the space-separated str(datetime) form '
            f'{str(memory_ts)!r} — that is the exact ambiguity task 4574 removes; '
            f'got:\n{result!r}'
        )

    @pytest.mark.asyncio
    async def test_fresh_project_all_none_watermark_does_not_raise_or_emit_none(self):
        """A fresh project's all-None Watermark must degrade cleanly: no
        raise, and the literal 'None' must never appear as a stand-in for a
        missing cutoff. _format_watermark's existing last_full_run_completed
        is-None short-circuit ('First run — no previous reconciliation.')
        already covers this by construction; pinned here so a future edit
        cannot silently regress it while adding the episode/memory lines."""
        stage = _make_consolidator()
        watermark = Watermark(project_id='test_project')

        result = await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        section = result.split('### Previous Reconciliation', 1)[1].split('## Your Task', 1)[0]
        assert 'First run' in section, f'Expected first-run short-circuit; got:\n{section!r}'
        assert 'None' not in section, (
            f"Must never emit the literal 'None' in the Previous Reconciliation "
            f'section; got:\n{section!r}'
        )

    @pytest.mark.asyncio
    async def test_missing_episode_cutoff_omits_line_not_none_literal(self):
        """Mixed case: last_full_run_completed is set (so the first-run
        short-circuit does NOT fire) but last_episode_timestamp is None. The
        missing episode cutoff must be omitted entirely, never rendered as
        the literal 'None' — the memory cutoff (which IS set) must still
        appear."""
        stage = _make_consolidator()
        memory_ts = datetime(2026, 8, 21, 6, 30, 0, tzinfo=UTC)
        watermark = Watermark(
            project_id='test_project',
            last_full_run_id='run-abc',
            last_full_run_completed=datetime(2026, 8, 22, 0, 0, 0, tzinfo=UTC),
            last_episode_timestamp=None,
            last_memory_timestamp=memory_ts,
        )

        result = await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        section = result.split('### Previous Reconciliation', 1)[1].split('## Your Task', 1)[0]
        assert 'None' not in section, (
            f"Must never emit the literal 'None' for the missing episode cutoff; "
            f'got:\n{section!r}'
        )
        assert memory_ts.isoformat() in section, (
            f'Expected memory cutoff {memory_ts.isoformat()!r} still disclosed; '
            f'got:\n{section!r}'
        )
