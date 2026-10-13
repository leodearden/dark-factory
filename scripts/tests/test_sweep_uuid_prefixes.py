"""Tests for ``scripts/sweep_uuid_prefixes.py`` (task 5325, PRD C6).

Fixture ids are HEX-LETTERED (``bff81530``, ``cafe0123``) so a case or prefix
assertion cannot pass vacuously on a digits-only id.

No test touches a live store: escalation files are built under ``tmp_path``,
the tasks store is a seeded sqlite file, and the resolver, the mem0 corpus and
the MCP task client are injected fakes.
"""
import asyncio
import contextlib
import copy
import itertools
import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

# `import sweep_uuid_prefixes` resolves because scripts/tests/conftest.py puts
# scripts/ on sys.path; the script's own bootstrap then binds `shared` to this
# checkout.
import sweep_toolcall_markup as markup_sweep
import sweep_uuid_prefixes as sweep
from shared.project_layout import KNOWN_PROJECT_ROOTS_ENV
from shared.uuid_prefix import PrefixToken
from shared.uuid_prefix_guard import Candidate, Resolution, ResolverUnavailable

UNIQUE = 'bff81530'
FULL = 'bff81530-70e5-4ab6-939e-507299d1bcd2'
UNIQUE_CAFE = 'cafe0123'
FULL_CAFE = 'cafe0123-1111-4222-8333-444455556666'
FULL_OTHER = 'abcdef00-0000-4000-8000-00000000004a'
AMBIG = '208f1bdf'
AMBIG_A = '208f1bdf-0000-4000-8000-00000000000a'
AMBIG_B = '208f1bdf-0000-4000-8000-00000000000b'
NONE_TOKENS = ('deadbeef', '20260904')


def _unique(full_id: str) -> Resolution:
    return Resolution('unique', (Candidate('mem0', full_id, 'a preview'),))


RESOLUTIONS = {
    UNIQUE: _unique(FULL),
    UNIQUE_CAFE: _unique(FULL_CAFE),
    AMBIG: Resolution(
        'ambiguous',
        (
            Candidate('mem0', AMBIG_A, 'first'),
            Candidate('graphiti_node', AMBIG_B, 'second'),
        ),
    ),
    **{token: Resolution('none', ()) for token in NONE_TOKENS},
}

NO_UNWRITABLE: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# plan_record — the pure planner.
# ---------------------------------------------------------------------------


class TestPlanRecord:
    def test_every_unique_occurrence_is_expanded_at_its_span(self):
        detail = f'see {UNIQUE} and again {UNIQUE}; pair {UNIQUE}/{UNIQUE_CAFE} done'
        fields = {'detail': detail, 'other': 'untouched text'}
        snapshot = copy.deepcopy(fields)

        plan = sweep.plan_record(fields, RESOLUTIONS, NO_UNWRITABLE)

        assert plan.after == {
            'detail': detail.replace(UNIQUE, FULL).replace(UNIQUE_CAFE, FULL_CAFE),
            'other': 'untouched text',
        }
        assert [(token.token, full) for token, full in plan.expansions] == [
            (UNIQUE, FULL),
            (UNIQUE, FULL),
            (UNIQUE, FULL),
            (UNIQUE_CAFE, FULL_CAFE),
        ]
        assert fields == snapshot

    def test_an_ambiguous_token_is_reported_with_its_candidates_and_not_expanded(self):
        fields = {'detail': f'ambiguous {AMBIG} here'}

        plan = sweep.plan_record(fields, RESOLUTIONS, NO_UNWRITABLE)

        assert plan.expansions == ()
        assert plan.after is None
        assert plan.ambiguous == (
            sweep.AmbiguousSite(path=('detail',), token=AMBIG, candidates=(AMBIG_A, AMBIG_B)),
        )

    def test_none_tokens_are_only_counted(self):
        plan = sweep.plan_record(
            {'detail': f'{NONE_TOKENS[0]} on {NONE_TOKENS[1]}'}, RESOLUTIONS, NO_UNWRITABLE
        )

        assert plan.untouched == 2
        assert plan.expansions == ()
        assert plan.ambiguous == ()
        assert plan.after is None

    def test_glued_hex_runs_are_not_tokens_at_all(self):
        plan = sweep.plan_record(
            {'detail': 'recon-3f2a9c1e and episode_74b902f8'}, {}, NO_UNWRITABLE
        )

        assert plan == sweep.RecordPlan(
            expansions=(), ambiguous=(), unwritable=(), untouched=0, after=None
        )

    def test_a_nested_list_element_is_expanded(self):
        fields = {'metadata': {'cluster_memory_ids': [UNIQUE, FULL_OTHER], 'kind': 'x'}}

        plan = sweep.plan_record(fields, RESOLUTIONS, NO_UNWRITABLE)

        assert plan.after == {'metadata': {'cluster_memory_ids': [FULL, FULL_OTHER], 'kind': 'x'}}
        assert [token.path for token, _ in plan.expansions] == [
            ('metadata', 'cluster_memory_ids', 0)
        ]

    def test_a_token_in_an_unwritable_field_is_listed_and_left_alone(self):
        fields = {'test_strategy': f'check {UNIQUE}', 'details': f'{UNIQUE} cited'}

        plan = sweep.plan_record(fields, RESOLUTIONS, frozenset({'test_strategy'}))

        assert plan.unwritable == (
            PrefixToken(path=('test_strategy',), token=UNIQUE, start=6, end=14),
        )
        assert plan.after == {'test_strategy': f'check {UNIQUE}', 'details': f'{FULL} cited'}

    @pytest.mark.parametrize(
        ('fields', 'unwritable'),
        [
            ({'test_strategy': UNIQUE}, frozenset({'test_strategy'})),
            ({'detail': f'{AMBIG} {NONE_TOKENS[0]}'}, NO_UNWRITABLE),
            ({'detail': 'no hex at all'}, NO_UNWRITABLE),
        ],
        ids=['only-unwritable', 'only-ambiguous-and-none', 'no-tokens'],
    )
    def test_no_unique_writable_token_means_no_after_document(self, fields, unwritable):
        assert sweep.plan_record(fields, RESOLUTIONS, unwritable).after is None

    def test_the_after_document_is_a_fixed_point(self):
        fields = {
            'detail': f'{UNIQUE}/{UNIQUE_CAFE} and {AMBIG}',
            'metadata': {'ids': [UNIQUE]},
        }

        first = sweep.plan_record(fields, RESOLUTIONS, NO_UNWRITABLE)
        assert first.after is not None
        second = sweep.plan_record(first.after, RESOLUTIONS, NO_UNWRITABLE)

        assert second.expansions == ()
        assert second.after is None
        assert len(second.ambiguous) == len(first.ambiguous) == 1

    def test_counts_for_a_mixed_record(self):
        fields = {
            'title': f'{UNIQUE} x {AMBIG}',
            'details': f'{NONE_TOKENS[0]} {UNIQUE} {NONE_TOKENS[1]} {UNIQUE_CAFE}',
            'test_strategy': UNIQUE_CAFE,
            'metadata': {'ids': [AMBIG, NONE_TOKENS[0]]},
        }

        plan = sweep.plan_record(fields, RESOLUTIONS, frozenset({'test_strategy'}))

        assert len(plan.expansions) == 3
        assert len(plan.ambiguous) == 2
        assert plan.untouched == 3
        assert len(plan.unwritable) == 1

    def test_a_token_with_no_resolution_is_a_programming_error(self):
        with pytest.raises(KeyError):
            sweep.plan_record({'detail': UNIQUE}, {}, NO_UNWRITABLE)


# ---------------------------------------------------------------------------
# resolve_distinct — one resolution per distinct token, bounded concurrency.
# ---------------------------------------------------------------------------


class FakeResolver:
    """An async ``PrefixResolver`` that records how it was driven.

    *gate*, when given, holds every call until it is set, so a test can look at
    how many resolutions are in flight at once. A token in *unavailable* raises
    ``ResolverUnavailable`` for that store instead of answering.
    """

    def __init__(self, resolutions=None, *, gate=None, unavailable=None):
        self.resolutions = RESOLUTIONS if resolutions is None else resolutions
        self.gate = gate
        self.unavailable = unavailable or {}
        self.calls: list[tuple[str, str]] = []
        self.completed: list[str] = []
        self.cancelled: list[str] = []
        self.in_flight = 0
        self.peak = 0

    async def __call__(self, project_id: str, prefix: str) -> Resolution:
        self.calls.append((project_id, prefix))
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            if prefix in self.unavailable:
                raise ResolverUnavailable(self.unavailable[prefix])
            if self.gate is not None:
                await self.gate.wait()
            else:
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.cancelled.append(prefix)
            raise
        finally:
            self.in_flight -= 1
        self.completed.append(prefix)
        return self.resolutions.get(prefix, Resolution('none', ()))


async def _settle():
    for _ in range(50):
        await asyncio.sleep(0)


class TestResolveDistinct:
    def test_each_distinct_token_is_resolved_once_under_the_given_project(self):
        resolver = FakeResolver()
        tokens = [UNIQUE, AMBIG, UNIQUE, NONE_TOKENS[0], UNIQUE, AMBIG]

        result = asyncio.run(sweep.resolve_distinct('proj_a', tokens, resolver, {}))

        assert sorted(resolver.calls) == sorted(
            [('proj_a', UNIQUE), ('proj_a', AMBIG), ('proj_a', NONE_TOKENS[0])]
        )
        assert result == {
            UNIQUE: RESOLUTIONS[UNIQUE],
            AMBIG: RESOLUTIONS[AMBIG],
            NONE_TOKENS[0]: RESOLUTIONS[NONE_TOKENS[0]],
        }

    def test_memoised_tokens_are_carried_and_not_re_resolved(self):
        resolver = FakeResolver()
        memo = {UNIQUE: RESOLUTIONS[UNIQUE]}
        snapshot = dict(memo)

        result = asyncio.run(sweep.resolve_distinct('proj_a', [UNIQUE, UNIQUE_CAFE], resolver, memo))

        assert resolver.calls == [('proj_a', UNIQUE_CAFE)]
        assert result == {UNIQUE: RESOLUTIONS[UNIQUE], UNIQUE_CAFE: RESOLUTIONS[UNIQUE_CAFE]}
        assert memo == snapshot

    def test_in_flight_resolutions_never_exceed_the_bound(self):
        tokens = [f'{n:x}abcdef0' for n in range(10, 16)]

        async def scenario():
            gate = asyncio.Event()
            resolver = FakeResolver(gate=gate)
            pending = asyncio.create_task(
                sweep.resolve_distinct('proj_a', tokens, resolver, {}, concurrency=2)
            )
            await _settle()
            held = resolver.in_flight
            gate.set()
            return resolver, held, await pending

        resolver, held, result = asyncio.run(scenario())

        assert held == 2
        assert resolver.peak == 2
        assert sorted(result) == sorted(tokens)
        assert sorted(resolver.completed) == sorted(tokens)

    def test_an_unavailable_store_surfaces_as_itself_and_leaves_nothing_pending(self):
        tokens = [UNIQUE, AMBIG, UNIQUE_CAFE, NONE_TOKENS[0]]

        async def scenario():
            resolver = FakeResolver(gate=asyncio.Event(), unavailable={AMBIG: 'mem0'})
            with pytest.raises(ResolverUnavailable) as raised:
                await sweep.resolve_distinct('proj_a', tokens, resolver, {}, concurrency=4)
            return resolver, raised.value

        resolver, error = asyncio.run(scenario())

        assert error.store == 'mem0'
        started = {prefix for _, prefix in resolver.calls}
        assert started == set(resolver.completed) | set(resolver.cancelled) | {AMBIG}
        assert resolver.in_flight == 0

    def test_no_tokens_means_no_calls(self):
        resolver = FakeResolver()
        memo = {UNIQUE: RESOLUTIONS[UNIQUE]}

        result = asyncio.run(sweep.resolve_distinct('proj_a', [], resolver, memo))

        assert resolver.calls == []
        assert result == memo
        assert result is not memo


# ---------------------------------------------------------------------------
# Fixture builders shared by the lane and CLI tests.
# ---------------------------------------------------------------------------

PROJECT = 'proj_a'
ESC_DIR = ('data', 'escalations')
ESC_1_REL = 'data/escalations/archive/2026-09-01/esc-1-1.json'
ESC_1_KEY = f'esc-1-1 ({ESC_1_REL})'
ESC_2_KEY = 'esc-2-1 (data/escalations/esc-2-1.json)'


def escalation(esc_id: str, status: str, detail: str, **extra) -> dict:
    record = {
        'id': esc_id,
        'task_id': esc_id.split('-')[1],
        'agent_role': 'implementer',
        'severity': 'blocking',
        'category': 'infra_issue',
        'summary': 'a one-line summary',
        'detail': detail,
        'evidence': [],
        'status': status,
        'resolution': '',
        'created_at': '2026-08-08T00:00:00+00:00',
    }
    record.update(extra)
    return record


def write_escalation(path: Path, obj: dict, *, indent: int = 2) -> Path:
    """Write *obj* the way ``Escalation.to_json`` does: indent=2, NO trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent), encoding='utf-8')
    return path


def build_escalations(root: Path) -> dict[str, Path]:
    base = root.joinpath(*ESC_DIR)
    return {
        'esc-1-1': write_escalation(
            base / 'archive' / '2026-09-01' / 'esc-1-1.json',
            escalation(
                'esc-1-1',
                'resolved',
                f'the memory {UNIQUE} was cited\nand nothing else',
                evidence=[{'observation': f'recall returned {UNIQUE}', 'ref': 'run one'}],
            ),
        ),
        'esc-2-1': write_escalation(
            base / 'esc-2-1.json', escalation('esc-2-1', 'dismissed', f'which one is {AMBIG}')
        ),
        'esc-3-1': write_escalation(
            base / 'esc-3-1.json',
            escalation('esc-3-1', 'resolved', f'{NONE_TOKENS[0]} on {NONE_TOKENS[1]}'),
        ),
        'esc-4-1': write_escalation(
            base / 'esc-4-1.json', escalation('esc-4-1', 'pending', f'still open: {UNIQUE}')
        ),
        'esc-5-1': write_escalation(
            base / 'esc-5-1.json',
            escalation('esc-5-1', 'resolved', f'hand-edited {UNIQUE}'),
            indent=4,
        ),
    }


def bytes_under(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob('*'))
        if path.is_file()
    }


def run(root, *, apply, lanes=('escalations',), resolver=None, mem0=None, open_task_client=None):
    return asyncio.run(
        sweep.run_sweep(
            sweep.SweepScope(project_id=PROJECT, project_root=root),
            lanes=lanes,
            apply=apply,
            resolve=resolver if resolver is not None else FakeResolver(),
            mem0=mem0,
            open_task_client=open_task_client,
            rate=1000.0,
        )
    )


@pytest.fixture
def esc_root(tmp_path):
    root = tmp_path / PROJECT
    build_escalations(root)
    return root


# ---------------------------------------------------------------------------
# The run loop over the escalations lane.
# ---------------------------------------------------------------------------


class TestEscalationsLane:
    def test_a_dry_run_reports_and_diffs_but_writes_nothing(self, esc_root):
        before = bytes_under(esc_root)

        reports, diffs = run(esc_root, apply=False)

        assert bytes_under(esc_root) == before
        (report,) = reports
        assert report.lane == 'escalations'
        assert report.project_id == PROJECT
        assert (report.scanned, report.expandable, report.expanded, report.remaining) == (3, 2, 0, 2)
        assert report.untouched == 2
        assert [(site.record, site.token, site.candidates) for site in report.ambiguous] == [
            (ESC_2_KEY, AMBIG, (AMBIG_A, AMBIG_B))
        ]
        assert dict(report.refused) == {
            markup_sweep.REASON_NON_TERMINAL: 1,
            markup_sweep.REASON_FORMAT_NOT_REPRODUCIBLE: 1,
        }
        assert len(diffs) == 2
        assert all(diff.startswith(f'--- escalations {ESC_1_KEY} ') for diff in diffs)
        added = [
            line
            for diff in diffs
            for line in diff.splitlines()
            if line.startswith('+') and not line.startswith('+++')
        ]
        assert added == [f'+the memory {FULL} was cited', f'+recall returned {FULL}']

    def test_apply_expands_exactly_the_diffed_occurrences(self, esc_root):
        raw = (esc_root / ESC_1_REL).read_text(encoding='utf-8')
        before = bytes_under(esc_root)

        reports, diffs = run(esc_root, apply=True)

        after = bytes_under(esc_root)
        assert set(after) == set(before)
        assert {rel for rel in before if before[rel] != after[rel]} == {ESC_1_REL}
        new_text = (esc_root / ESC_1_REL).read_text(encoding='utf-8')
        assert new_text == raw.replace(UNIQUE, FULL)
        assert json.loads(new_text)['evidence'][0]['observation'] == f'recall returned {FULL}'
        (report,) = reports
        assert (report.expandable, report.expanded, report.remaining) == (2, 2, 0)
        assert report.failures == ()
        assert diffs == ()

    def test_a_second_apply_finds_nothing_and_exits_clean(self, esc_root):
        run(esc_root, apply=True)

        reports, _ = run(esc_root, apply=True)

        (report,) = reports
        assert (report.expandable, report.remaining) == (0, 0)
        assert len(report.ambiguous) == 1
        assert sweep.exit_code(reports) == markup_sweep.EXIT_CLEAN

    def test_an_interrupted_swap_leaves_the_target_whole_and_a_rerun_converges(
        self, esc_root, monkeypatch
    ):
        target = esc_root / ESC_1_REL
        raw = target.read_bytes()

        def interrupted(src, dst):
            raise KeyboardInterrupt

        with monkeypatch.context() as patch:
            patch.setattr(os, 'replace', interrupted)
            with pytest.raises(KeyboardInterrupt):
                run(esc_root, apply=True)

        assert target.read_bytes() == raw
        json.loads(raw)
        assert list(esc_root.rglob('.sweep.*.tmp')) == []

        (report,), _ = run(esc_root, apply=True)
        assert report.expanded == 2
        reports, _ = run(esc_root, apply=True)
        assert reports[0].remaining == 0
        assert sweep.exit_code(reports) == markup_sweep.EXIT_CLEAN

    def test_a_dry_run_with_work_left_exits_repairable_remains(self, esc_root):
        reports, _ = run(esc_root, apply=False)

        assert sweep.exit_code(reports) == markup_sweep.EXIT_REPAIRABLE_REMAINS

    def test_a_failed_swap_is_a_write_failure(self, esc_root, monkeypatch):
        def refused(src, dst):
            raise OSError(28, 'No space left on device')

        monkeypatch.setattr(os, 'replace', refused)

        reports, _ = run(esc_root, apply=True)

        (report,) = reports
        assert [(f.key, f.reason) for f in report.failures] == [
            (ESC_1_KEY, markup_sweep.REASON_WRITE_FAILED)
        ]
        assert 'No space left on device' in report.failures[0].error
        assert report.expanded == 0
        assert sweep.exit_code(reports) == markup_sweep.EXIT_WRITE_FAILED

    def test_an_unreachable_store_aborts_the_lane_before_any_write(self, esc_root):
        before = bytes_under(esc_root)

        reports, _ = run(
            esc_root, apply=True, resolver=FakeResolver(unavailable={UNIQUE: 'graphiti'})
        )

        assert bytes_under(esc_root) == before
        (report,) = reports
        assert report.aborted == 'graphiti'
        assert report.expanded == 0
        assert sweep.exit_code(reports) == sweep.EXIT_STORE_UNAVAILABLE


def lane_report(*, expandable=0, expanded=0, ambiguous=(), failures=(), aborted=None):
    return sweep.LaneReport(
        lane='escalations',
        project_id=PROJECT,
        scanned=1,
        expandable=expandable,
        expanded=expanded,
        untouched=0,
        ambiguous=ambiguous,
        unwritable=(),
        refused={},
        failures=failures,
        aborted=aborted,
    )


class TestExitCode:
    def test_precedence_is_aborted_then_failures_then_remaining(self):
        failure = sweep.WriteFailureRecord(key='k', reason='write-failed', error='e')
        remaining = lane_report(expandable=3, expanded=1)
        failed = lane_report(failures=(failure,))
        aborted = lane_report(aborted='mem0')

        assert sweep.exit_code((remaining, failed, aborted)) == sweep.EXIT_STORE_UNAVAILABLE
        assert sweep.exit_code((remaining, failed)) == markup_sweep.EXIT_WRITE_FAILED
        assert sweep.exit_code((remaining,)) == markup_sweep.EXIT_REPAIRABLE_REMAINS
        site = sweep.AmbiguousRecordSite(
            record='k', path=('detail',), token=AMBIG, candidates=(AMBIG_A, AMBIG_B)
        )
        assert sweep.exit_code((lane_report(ambiguous=(site,)),)) == markup_sweep.EXIT_CLEAN


# ---------------------------------------------------------------------------
# WritePacer.
# ---------------------------------------------------------------------------


class FakeTime:
    """A clock plus a sleep; *advance* makes a sleep move the clock."""

    def __init__(self, *, advance: bool):
        self.now = 100.0
        self.advance = advance
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        if self.advance:
            self.now += seconds


class TestWritePacer:
    def test_a_frozen_clock_sleeps_a_full_interval_between_writes(self):
        fake = FakeTime(advance=False)
        pacer = sweep.WritePacer(5.0, clock=fake.clock, sleep=fake.sleep)

        async def three():
            for _ in range(3):
                await pacer.wait()

        asyncio.run(three())

        assert fake.sleeps == [0.0, pytest.approx(0.2), pytest.approx(0.2)]

    def test_releases_are_spaced_by_at_least_the_interval(self):
        fake = FakeTime(advance=True)
        pacer = sweep.WritePacer(5.0, clock=fake.clock, sleep=fake.sleep)
        releases: list[float] = []

        async def paced():
            for gap in (0.0, 0.05, 0.5, 0.0):
                fake.now += gap
                await pacer.wait()
                releases.append(fake.now)

        asyncio.run(paced())

        spacings = [later - earlier for earlier, later in itertools.pairwise(releases)]
        assert all(spacing >= 0.2 - 1e-9 for spacing in spacings)
        assert fake.sleeps[2] == 0.0


# ---------------------------------------------------------------------------
# The tasks lane.
# ---------------------------------------------------------------------------

PENDING_SINCE = '2026-10-01T00:00:00Z'
REJECTION = {'success': False, 'error': 'done_provenance_via_update_task', 'hint': 'h'}


def build_tasks(make_tasks_db, root: Path) -> Path:
    directory = root / '.taskmaster' / 'tasks'
    directory.mkdir(parents=True, exist_ok=True)
    return make_tasks_db(
        [
            {
                'id': 1,
                'description': f'see memory {UNIQUE}',
                'metadata': {
                    'cluster_memory_ids': [UNIQUE, FULL_OTHER],
                    'pending_since': PENDING_SINCE,
                    'kind': 'x',
                },
            },
            {'id': 2, 'details': f'which one: {AMBIG}'},
            {'id': 3, 'test_strategy': f'recall {UNIQUE}'},
            {'id': 4, 'title': f'about {UNIQUE}'},
            {'id': 5, 'details': f'cites {UNIQUE}'},
        ],
        directory=directory,
    )


def stored_task(db: Path, task_id: int) -> dict:
    conn = sqlite3.connect(db)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute('SELECT * FROM tasks WHERE id = ?', (task_id,)).fetchone()
    finally:
        conn.close()
    task = dict(row)
    task['metadata'] = json.loads(task['metadata'] or '{}')
    return task


class FakeTaskServer:
    """The ``update_task`` half of fused-memory, applied to the seeded sqlite file.

    *replies* answers a task id with a canned reply and writes nothing.
    *errors* raises for a task id, the way the client does on a reply it
    cannot read. *side_effects* runs one extra statement inside a task's
    write, standing in for a concurrent writer.
    """

    def __init__(self, db: Path, *, replies=None, errors=None, side_effects=None):
        self.db = db
        self.replies = replies or {}
        self.errors = errors or {}
        self.side_effects = side_effects or {}
        self.calls: list[dict] = []
        self.opened_with: list[str | None] = []

    def opener(self, server_url):
        self.opened_with.append(server_url)
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def call_tool(self, name: str, arguments: dict) -> dict:
        assert name == 'update_task'
        self.calls.append(dict(arguments))
        task_id = arguments['id']
        if task_id in self.errors:
            raise self.errors[task_id]
        if task_id in self.replies:
            return self.replies[task_id]
        conn = sqlite3.connect(self.db)
        try:
            with conn:
                self._apply(conn, arguments)
                if task_id in self.side_effects:
                    conn.execute(self.side_effects[task_id])
        finally:
            conn.close()
        return {'id': task_id, 'updated': True}

    @staticmethod
    def _apply(conn: sqlite3.Connection, arguments: dict) -> None:
        key = (arguments['tag'], int(arguments['id']))
        for column in ('title', 'description', 'details'):
            if column in arguments:
                conn.execute(
                    f'UPDATE tasks SET {column} = ? WHERE tag = ? AND id = ?',
                    (arguments[column], *key),
                )
        if 'metadata' in arguments:
            assert arguments['metadata_mode'] == 'merge'
            (old,) = conn.execute(
                'SELECT metadata FROM tasks WHERE tag = ? AND id = ?', key
            ).fetchone()
            merged = {**json.loads(old or '{}'), **json.loads(arguments['metadata'])}
            conn.execute(
                'UPDATE tasks SET metadata = ? WHERE tag = ? AND id = ?',
                (json.dumps(merged), *key),
            )


@pytest.fixture
def task_root(tmp_path, make_tasks_db):
    root = tmp_path / PROJECT
    db = build_tasks(make_tasks_db, root)
    return root, db


def run_tasks(root, server, *, apply):
    return run(root, apply=apply, lanes=('tasks',), open_task_client=server.opener)


class TestTasksLane:
    def test_a_dry_run_never_dials_the_server(self, task_root):
        root, db = task_root
        server = FakeTaskServer(db)
        before = [stored_task(db, n) for n in range(1, 6)]

        reports, diffs = run_tasks(root, server, apply=False)

        assert server.opened_with == []
        assert [stored_task(db, n) for n in range(1, 6)] == before
        (report,) = reports
        assert (report.scanned, report.expandable, report.remaining) == (5, 4, 4)
        assert [site.record for site in report.ambiguous] == ['master#2']
        assert [(site.record, site.path) for site in report.unwritable] == [
            ('master#3', ('test_strategy',))
        ]
        labels = [diff.splitlines()[0] for diff in diffs]
        assert labels == [
            '--- tasks master#1 description',
            '--- tasks master#1 metadata.cluster_memory_ids[0]',
            '--- tasks master#4 title',
            '--- tasks master#5 details',
        ]

    def test_apply_sends_only_the_changed_column_and_metadata_key(self, task_root):
        root, db = task_root
        server = FakeTaskServer(db)

        reports, _ = run_tasks(root, server, apply=True)

        sent = {call['id']: call for call in server.calls}
        assert sorted(sent) == ['1', '4', '5']
        assert sent['1'] == {
            'id': '1',
            'project_root': str(root),
            'tag': 'master',
            'description': f'see memory {FULL}',
            'metadata': json.dumps({'cluster_memory_ids': [FULL, FULL_OTHER]}),
            'metadata_mode': 'merge',
        }
        assert stored_task(db, 1)['metadata'] == {
            'cluster_memory_ids': [FULL, FULL_OTHER],
            'pending_since': PENDING_SINCE,
            'kind': 'x',
        }
        assert stored_task(db, 3)['test_strategy'] == f'recall {UNIQUE}'
        assert server.opened_with == [None]
        assert (reports[0].expanded, reports[0].failures) == (4, ())

    def test_a_rejected_write_is_a_failure_and_the_lane_continues(self, task_root):
        root, db = task_root
        server = FakeTaskServer(db, replies={'4': REJECTION})

        reports, _ = run_tasks(root, server, apply=True)

        (report,) = reports
        assert [(f.key, f.reason) for f in report.failures] == [('master#4', 'write-rejected')]
        assert 'done_provenance_via_update_task' in report.failures[0].error
        assert stored_task(db, 4)['title'] == f'about {UNIQUE}'
        assert stored_task(db, 5)['details'] == f'cites {FULL}'
        assert report.expanded == 3
        assert sweep.exit_code(reports) == markup_sweep.EXIT_WRITE_FAILED

    def test_an_empty_reply_is_not_an_acceptance(self, task_root):
        root, db = task_root
        server = FakeTaskServer(db, replies={'4': {}})

        reports, _ = run_tasks(root, server, apply=True)

        assert [(f.key, f.reason) for f in reports[0].failures] == [('master#4', 'write-rejected')]

    def test_a_reply_that_is_not_json_is_a_failure_and_the_lane_continues(self, task_root):
        root, db = task_root
        html_page = json.JSONDecodeError('Expecting value', '<html>proxy error</html>', 0)
        server = FakeTaskServer(db, errors={'4': html_page})

        reports, _ = run_tasks(root, server, apply=True)

        (report,) = reports
        assert [(f.key, f.reason) for f in report.failures] == [
            ('master#4', markup_sweep.REASON_WRITE_FAILED)
        ]
        assert 'JSONDecodeError' in report.failures[0].error
        assert stored_task(db, 5)['details'] == f'cites {FULL}'
        assert report.expanded == 3
        assert sweep.exit_code(reports) == markup_sweep.EXIT_WRITE_FAILED

    def test_an_apply_with_no_task_client_is_a_programming_error(self, task_root):
        root, db = task_root

        with pytest.raises(ValueError, match='task client'):
            run(root, apply=True, lanes=('tasks',))

        assert stored_task(db, 1)['description'] == f'see memory {UNIQUE}'

    def test_a_row_changed_since_the_scan_is_not_sent(self, task_root):
        root, db = task_root
        server = FakeTaskServer(
            db, side_effects={'1': "UPDATE tasks SET details = 'rewritten elsewhere' WHERE id = 5"}
        )

        reports, _ = run_tasks(root, server, apply=True)

        assert '5' not in {call['id'] for call in server.calls}
        assert [(f.key, f.reason) for f in reports[0].failures] == [
            ('master#5', markup_sweep.REASON_CHANGED_UNDER_US)
        ]
        assert stored_task(db, 5)['details'] == 'rewritten elsewhere'

    def test_a_second_run_after_a_clean_apply_is_clean(self, task_root):
        root, db = task_root
        run_tasks(root, FakeTaskServer(db), apply=True)

        reports, _ = run_tasks(root, FakeTaskServer(db), apply=True)

        (report,) = reports
        assert (report.expandable, report.remaining) == (0, 0)
        assert [(site.record, site.path) for site in report.unwritable] == [
            ('master#3', ('test_strategy',))
        ]
        assert sweep.exit_code(reports) == markup_sweep.EXIT_CLEAN

    def test_an_unreadable_store_aborts_the_lane(self, tmp_path, project_root_with_tasks_db):
        root = tmp_path / PROJECT
        db = project_root_with_tasks_db(root)
        server = FakeTaskServer(db)

        reports, _ = run_tasks(root, server, apply=True)

        assert reports[0].aborted == 'tasks.db'
        assert server.calls == []
        assert sweep.exit_code(reports) == sweep.EXIT_STORE_UNAVAILABLE


class TestAssertWriteAccepted:
    @pytest.mark.parametrize(
        'reply', [{'success': False, 'error': 'e'}, {}, 'not a dict'], ids=['error', 'empty', 'text']
    )
    def test_a_reply_without_a_positive_signal_raises(self, reply):
        with pytest.raises(sweep.WriteRejectedError):
            sweep.assert_write_accepted(reply)

    def test_an_accepted_reply_returns_none(self):
        assert sweep.assert_write_accepted({'id': '1', 'updated': True}) is None


# ---------------------------------------------------------------------------
# The mem0 lane.
# ---------------------------------------------------------------------------

M1_BODY = f'cluster around {UNIQUE}; see also {UNIQUE}.'
MEMORY_ERROR = {'error_type': 'MemoryNotFound', 'error': 'gone'}


class FakeCorpus:
    """An in-memory ``Mem0Corpus``: *replies* answers an amend without applying it,
    and *drift* makes ``read_body`` return a body the scan never saw."""

    def __init__(self, bodies: dict[str, str], *, replies=None, drift=None):
        self.store = dict(bodies)
        self.replies = replies or {}
        self.drift = drift or {}
        self.amends: list[tuple[str, str, str]] = []

    async def bodies(self, project_id: str):
        for memory_id in sorted(self.store):
            yield memory_id, self.store[memory_id]

    async def read_body(self, project_id: str, memory_id: str) -> str | None:
        return self.drift.get(memory_id, self.store.get(memory_id))

    async def amend_body(self, project_id: str, memory_id: str, body: str) -> dict:
        self.amends.append((project_id, memory_id, body))
        if memory_id in self.replies:
            return self.replies[memory_id]
        self.store[memory_id] = body
        return {'status': 'updated', 'id': memory_id}


def mem0_corpus(**options) -> FakeCorpus:
    return FakeCorpus(
        {
            'm1': M1_BODY,
            'm2': f'which of {AMBIG}',
            'm3': f'{NONE_TOKENS[0]} on {NONE_TOKENS[1]}',
            'm4': f'lone {UNIQUE}',
        },
        **options,
    )


def run_mem0(corpus, *, apply):
    return run(None, apply=apply, lanes=('mem0',), mem0=corpus)


class TestMem0Lane:
    def test_a_dry_run_amends_nothing_and_needs_no_root(self):
        corpus = mem0_corpus()

        reports, diffs = run_mem0(corpus, apply=False)

        assert corpus.amends == []
        (report,) = reports
        assert (report.scanned, report.expandable, report.untouched) == (4, 3, 2)
        assert [site.record for site in report.ambiguous] == ['m2']
        assert [diff.splitlines()[0] for diff in diffs] == ['--- mem0 m1 data', '--- mem0 m4 data']

    def test_apply_amends_in_place_and_a_rejection_does_not_stop_the_lane(self):
        corpus = mem0_corpus(replies={'m4': MEMORY_ERROR})

        reports, _ = run_mem0(corpus, apply=True)

        assert corpus.amends == [
            (PROJECT, 'm1', M1_BODY.replace(UNIQUE, FULL)),
            (PROJECT, 'm4', f'lone {FULL}'),
        ]
        assert corpus.store['m1'] == M1_BODY.replace(UNIQUE, FULL)
        (report,) = reports
        assert [(f.key, f.reason) for f in report.failures] == [('m4', 'write-rejected')]
        assert 'MemoryNotFound' in report.failures[0].error
        assert report.expanded == 2
        assert sweep.exit_code(reports) == markup_sweep.EXIT_WRITE_FAILED

    def test_a_body_changed_since_the_scan_is_not_amended(self):
        corpus = mem0_corpus(drift={'m1': 'someone rewrote this'})

        reports, _ = run_mem0(corpus, apply=True)

        assert 'm1' not in {memory_id for _, memory_id, _ in corpus.amends}
        assert [(f.key, f.reason) for f in reports[0].failures] == [
            ('m1', markup_sweep.REASON_CHANGED_UNDER_US)
        ]

    def test_a_second_run_after_a_clean_apply_is_clean(self):
        corpus = mem0_corpus()
        run_mem0(corpus, apply=True)

        reports, _ = run_mem0(corpus, apply=True)

        assert reports[0].remaining == 0
        assert sweep.exit_code(reports) == markup_sweep.EXIT_CLEAN


class MissingCollection(Exception):
    """Stands in for Qdrant's 404 for a collection that was never created."""


class BackendDown(Exception):
    """Stands in for a Qdrant or embedding-client failure."""


#: The adapter's failure policy in these tests, shaped like the live one:
#: the missing-collection error is itself one of the backend's failures.
FAILURES = sweep.Mem0Failures(
    backend=(BackendDown, MissingCollection, OSError),
    is_missing_collection=lambda exc: isinstance(exc, MissingCollection),
)


def service_corpus(service):
    return sweep.ServiceMem0Corpus(service, FAILURES)


class FakeMemoryService:
    """The slice of ``MemoryService`` the production adapter touches."""

    def __init__(self, points, *, update_error=None, scroll_error=None, read_error=None):
        self.points = points
        self.update_error = update_error
        self.scroll_error = scroll_error
        self.read_error = read_error
        self.config = SimpleNamespace(mem0=SimpleNamespace(collection_prefix='fused'))
        self.mem0 = SimpleNamespace(scroll_collection_pages=self._scroll)
        self.scrolls: list[tuple[str, dict]] = []
        self.updates: list[dict] = []

    async def _scroll(self, collection, **options):
        self.scrolls.append((collection, options))
        if self.scroll_error is not None:
            raise self.scroll_error
        for point in self.points:
            yield point

    async def get_memory_by_id(self, *, project_id, memory_id):
        if self.read_error is not None:
            raise self.read_error
        for point in self.points:
            if str(point.id) == memory_id:
                return {'id': memory_id, 'content': 'c', 'metadata': point.payload}
        return None

    async def update_memory(self, **kwargs):
        self.updates.append(kwargs)
        if self.update_error is not None:
            raise self.update_error
        return {'status': 'updated', 'id': kwargs['memory_id']}


def service_points():
    return [
        SimpleNamespace(id=FULL_OTHER, payload={'data': f'cites {UNIQUE}', 'topic': 't'}),
        SimpleNamespace(id=AMBIG_A, payload={'topic': 'no body'}),
        SimpleNamespace(id=AMBIG_B, payload={'data': 7}),
    ]


class TestServiceMem0Corpus:
    def test_bodies_scroll_the_project_collection_and_skip_non_text(self):
        service = FakeMemoryService(service_points())
        corpus = service_corpus(service)

        async def collect():
            return [item async for item in corpus.bodies(PROJECT)]

        assert asyncio.run(collect()) == [(FULL_OTHER, f'cites {UNIQUE}')]
        ((collection, options),) = service.scrolls
        assert collection == 'fused_proj_a'
        assert options['with_payload'] is True

    def test_read_body_reads_the_payload_data_or_none(self):
        corpus = service_corpus(FakeMemoryService(service_points()))

        assert asyncio.run(corpus.read_body(PROJECT, FULL_OTHER)) == f'cites {UNIQUE}'
        assert asyncio.run(corpus.read_body(PROJECT, FULL)) is None

    def test_amend_body_is_a_content_only_same_id_amend(self):
        service = FakeMemoryService(service_points())

        reply = asyncio.run(service_corpus(service).amend_body(PROJECT, FULL_OTHER, f'cites {FULL}'))

        assert reply == {'status': 'updated', 'id': FULL_OTHER}
        assert service.updates == [
            {
                'memory_id': FULL_OTHER,
                'project_id': PROJECT,
                'content': f'cites {FULL}',
                'reason': sweep.WRITE_REASON,
                'agent_id': sweep.WRITE_SOURCE,
                '_source': sweep.WRITE_SOURCE,
            }
        ]

    @pytest.mark.parametrize(
        'failing',
        [
            {'update_error': ConnectionError('qdrant away')},
            {'read_error': BackendDown('qdrant away')},
        ],
        ids=['amend', 're-read'],
    )
    def test_a_backend_failure_on_write_is_a_lane_failure_not_a_crash(self, failing):
        service = FakeMemoryService(service_points(), **failing)

        reports, _ = run_mem0(service_corpus(service), apply=True)

        (report,) = reports
        assert [(f.key, f.reason) for f in report.failures] == [
            (FULL_OTHER, markup_sweep.REASON_WRITE_FAILED)
        ]
        (error,) = failing.values()
        assert type(error).__name__ in report.failures[0].error
        assert 'qdrant away' in report.failures[0].error

    @pytest.mark.parametrize(
        'failing',
        [
            {'update_error': TypeError("update_memory() got an unexpected keyword 'x'")},
            {'read_error': KeyError('metadata')},
        ],
        ids=['amend', 're-read'],
    )
    def test_a_bug_on_the_write_path_is_a_traceback_not_a_failure_row(self, failing):
        (error,) = failing.values()
        service = FakeMemoryService(service_points(), **failing)

        with pytest.raises(type(error)):
            run_mem0(service_corpus(service), apply=True)

    def test_a_scroll_that_cannot_reach_the_store_aborts_the_lane(self):
        service = FakeMemoryService(service_points(), scroll_error=ConnectionError('down'))

        reports, _ = run_mem0(service_corpus(service), apply=True)

        assert reports[0].aborted == 'mem0'
        assert service.updates == []
        assert sweep.exit_code(reports) == sweep.EXIT_STORE_UNAVAILABLE

    def test_a_missing_collection_is_an_empty_corpus_not_an_outage(self, caplog):
        service = FakeMemoryService(service_points(), scroll_error=MissingCollection('404'))

        with caplog.at_level('WARNING'):
            reports, diffs = run_mem0(service_corpus(service), apply=False)

        (report,) = reports
        assert (report.aborted, report.scanned, report.expandable) == (None, 0, 0)
        assert diffs == ()
        assert sweep.exit_code(reports) == markup_sweep.EXIT_CLEAN
        assert 'fused_proj_a' in caplog.text

    def test_a_scroll_failure_outside_the_backend_is_a_traceback_not_an_outage(self):
        budget = RuntimeError('scroll page budget exhausted')
        service = FakeMemoryService(service_points(), scroll_error=budget)

        with pytest.raises(RuntimeError, match='page budget'):
            run_mem0(service_corpus(service), apply=False)


class TestLiveMem0Failures:
    """The production policy, wherever the backend packages are installed."""

    def test_qdrants_missing_collection_is_recognised_and_bugs_are_not_backend_failures(self):
        mem0_client = pytest.importorskip('fused_memory.backends.mem0_client')
        from httpx import Headers
        from qdrant_client.http.exceptions import UnexpectedResponse

        policy = sweep.live_mem0_failures()
        missing = UnexpectedResponse(
            404, 'Not Found', b'{"status":{"error":"Not found: Collection `x` doesn\'t exist!"}}', Headers()
        )
        server_error = UnexpectedResponse(500, 'Internal Server Error', b'boom', Headers())

        assert isinstance(missing, policy.backend)
        assert policy.is_missing_collection(missing)
        assert isinstance(server_error, policy.backend)
        assert not policy.is_missing_collection(server_error)
        assert isinstance(TimeoutError(), policy.backend)
        assert not isinstance(mem0_client.ScrollPageBudgetExhausted('cap'), policy.backend)
        assert not isinstance(KeyError('metadata'), policy.backend)


# ---------------------------------------------------------------------------
# The CLI.
# ---------------------------------------------------------------------------


class FakeStores:
    """An injected ``open_stores``: yields the fakes and records how it was opened."""

    def __init__(self, *, resolver=None, corpus=None):
        self.resolver = resolver if resolver is not None else FakeResolver()
        self.corpus = corpus if corpus is not None else mem0_corpus()
        self.opened: list[tuple[str | None, bool]] = []

    @contextlib.asynccontextmanager
    async def open(self, config_path, *, mem0_apply):
        self.opened.append((config_path, mem0_apply))
        yield sweep.LiveStores(resolve=self.resolver, mem0=self.corpus)


def unexpected_task_client(server_url):
    raise AssertionError('the task client was opened')


@pytest.fixture
def cli_root(tmp_path, make_tasks_db):
    root = tmp_path / PROJECT
    build_escalations(root)
    build_tasks(make_tasks_db, root)
    return root


def cli(argv, stores, *, open_task_client=unexpected_task_client):
    return sweep.main(argv, open_stores=stores.open, open_task_client=open_task_client)


def scoped(root, *argv):
    return [*argv, '--project', PROJECT, '--project-root', str(root)]


class TestMain:
    def test_the_default_run_is_a_dry_run_over_every_lane(self, cli_root, capsys):
        stores = FakeStores()
        before = bytes_under(cli_root)

        code = cli(scoped(cli_root, '--lane', 'all'), stores)

        assert code == markup_sweep.EXIT_REPAIRABLE_REMAINS
        assert bytes_under(cli_root) == before
        assert stores.corpus.amends == []
        assert stores.opened == [(None, False)]
        out = capsys.readouterr().out
        for lane in ('mem0', 'tasks', 'escalations'):
            assert f'--- {lane} ' in out
        lines = out.splitlines()
        for lane, remaining in (('mem0', 3), ('tasks', 4), ('escalations', 2)):
            (summary,) = [line for line in lines if line.startswith(f'lane={lane} ')]
            for field in (f'project={PROJECT}', f'expandable={remaining}', 'ambiguous=1', 'untouched='):
                assert field in summary
            assert f'{lane}: {remaining} expandable remaining' in lines

    def test_apply_then_rerun_reaches_clean(self, cli_root, capsys):
        stores = FakeStores()

        assert cli(scoped(cli_root, '--apply', '--lane', 'escalations'), stores) == 0
        capsys.readouterr()
        code = cli(scoped(cli_root, '--lane', 'escalations'), stores)

        assert code == markup_sweep.EXIT_CLEAN
        assert 'escalations: 0 expandable remaining' in capsys.readouterr().out.splitlines()

    def test_only_a_mem0_apply_asks_for_mem0_writes(self, cli_root):
        stores = FakeStores()

        cli(scoped(cli_root, '--apply', '--lane', 'escalations'), stores)
        cli(scoped(cli_root, '--apply', '--lane', 'mem0'), stores)

        assert [mem0_apply for _, mem0_apply in stores.opened] == [False, True]

    def test_a_root_of_another_project_is_a_usage_error(self, cli_root, capsys):
        stores = FakeStores()

        with pytest.raises(SystemExit) as exited:
            cli(['--project', 'other', '--project-root', str(cli_root)], stores)

        assert exited.value.code == 2
        err = capsys.readouterr().err
        assert 'other' in err
        assert PROJECT in err
        assert stores.opened == []

    def test_a_root_needing_lane_without_a_root_is_a_usage_error(self, monkeypatch, capsys):
        monkeypatch.delenv(KNOWN_PROJECT_ROOTS_ENV, raising=False)
        stores = FakeStores()

        with pytest.raises(SystemExit) as exited:
            cli(['--lane', 'tasks', '--project', 'nowhere'], stores)

        assert exited.value.code == 2
        assert '--project-root' in capsys.readouterr().err
        assert stores.opened == []

    def test_the_mem0_lane_needs_no_root(self, monkeypatch):
        monkeypatch.delenv(KNOWN_PROJECT_ROOTS_ENV, raising=False)
        stores = FakeStores()

        cli(['--lane', 'mem0', '--project', 'nowhere'], stores)

        assert stores.opened == [(None, False)]

    @pytest.mark.parametrize('rate', ['0', '-1', 'fast'])
    def test_a_rate_that_is_not_positive_is_a_usage_error(self, cli_root, rate):
        stores = FakeStores()

        with pytest.raises(SystemExit) as exited:
            cli(scoped(cli_root, '--rate', rate), stores)

        assert exited.value.code == 2
        assert stores.opened == []

    def test_json_mode_prints_one_document_and_no_diffs(self, cli_root, capsys):
        code = cli(scoped(cli_root, '--lane', 'escalations', '--json'), FakeStores())

        document = json.loads(capsys.readouterr().out)
        assert document['exit_code'] == code == markup_sweep.EXIT_REPAIRABLE_REMAINS
        (lane,) = document['lanes']
        assert (lane['lane'], lane['expandable'], lane['remaining']) == ('escalations', 2, 2)
        assert lane['ambiguous'] == [
            {'record': ESC_2_KEY, 'path': ['detail'], 'token': AMBIG, 'candidates': [AMBIG_A, AMBIG_B]}
        ]
        assert lane['failures'] == []

    def test_an_unreachable_store_exits_store_unavailable_and_names_it(self, cli_root, capsys):
        stores = FakeStores(resolver=FakeResolver(unavailable={UNIQUE: 'mem0'}))

        code = cli(scoped(cli_root, '--lane', 'escalations'), stores)

        assert code == sweep.EXIT_STORE_UNAVAILABLE
        assert 'aborted: store=mem0' in capsys.readouterr().out.splitlines()
