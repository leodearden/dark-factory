"""The hold analysis: scripts/resource_report_holds.py.

PRD ``plans/per-consumer-resource-recording-prd.md`` leaf 2, contract C2 and
the §7 rows "Hold before go-live" and "Hold mapping, both shapes". Consumers
are built with ``sampler.consumers``, verify tags with
``shared.verify_scope_tag``, and the stores with resource_report_fixtures.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import _checkout as _checkout
import pytest
import resource_report_holds as holds
from resource_report_fixtures import memory_hold, project_root, samples_db, write_collection
from sampler.consumers import Consumer, Figure, Role, encode_consumer
from shared.project_layout import runs_db
from shared.verify_scope_tag import scope_tag_for

APP = '/user.slice/user-1000.slice/user@1000.service/app.slice'
DF_UNIT = f'{APP}/orchestrator-dark-factory.service'
T0 = datetime(2026, 11, 21, 18, 0, tzinfo=UTC)
KERNEL = Consumer(Role.KERNEL)
DF_SERVICE = Consumer(Role.SERVICE, 'orchestrator-dark-factory.service')
MiB = 2**20
GiB = 2**30


def at(seconds: int) -> datetime:
    return T0 + timedelta(seconds=seconds)


def ts(seconds: int) -> int:
    return int(at(seconds).timestamp())


def hold_block(text: str, moment: datetime) -> str:
    """The rendered block of the hold at *moment*: the paragraph its heading opens."""
    heading = f'hold {holds.format_utc(moment)}'
    return next(par for par in text.split('\n\n') if par.startswith(heading))


def hold_json(report: holds.HoldsReport, moment: datetime) -> Mapping[str, Any]:
    entries = json.loads(json.dumps(holds.report_json(report)))['holds']
    return next(entry for entry in entries if entry['at'] == holds.format_utc(moment))


def reading_at(report: holds.HoldsReport, moment: datetime) -> holds.HoldReading:
    return next(reading for reading in report.readings if reading.hold.at == moment)


class TestHoldMappingBothShapes:
    def test_the_unit_shape_maps_service_agents_and_verify_but_no_project(self, tmp_path: Path):
        root = tmp_path / 'solar-challenge'
        unit = 'orchestrator-my-solar-challenge.service'

        mapping = holds.hold_consumers(f'{APP}/{unit}', root)

        assert mapping.service == Consumer(Role.SERVICE, unit)
        assert mapping.agents == Consumer(Role.AGENTS, unit)
        assert mapping.project is None
        assert mapping.verify == Consumer(Role.VERIFY, scope_tag_for(root))
        assert mapping.unmapped_reason is None

    def test_the_unit_stem_and_the_verify_slug_are_not_assumed_equal(self, tmp_path: Path):
        root = tmp_path / 'solar-challenge'

        mapping = holds.hold_consumers(f'{APP}/orchestrator-my-solar-challenge.service', root)

        assert mapping.verify is not None
        assert mapping.verify.key is not None
        assert mapping.verify.key.startswith('solar-challenge-')
        assert not mapping.verify.key.startswith('my-solar-challenge')

    def test_the_slice_shape_keys_service_agents_and_project_by_the_slice(self, tmp_path: Path):
        root = tmp_path / 'dark-factory'
        project_slice = 'df-dark_factory.slice'

        mapping = holds.hold_consumers(f'{APP}/df.slice/{project_slice}', root)

        assert mapping.service == Consumer(Role.SERVICE, project_slice)
        assert mapping.agents == Consumer(Role.AGENTS, project_slice)
        assert mapping.project == Consumer(Role.PROJECT, project_slice)
        assert mapping.verify == Consumer(Role.VERIFY, scope_tag_for(root))
        assert mapping.verify.key is not None
        assert mapping.verify.key.startswith('dark-factory-')
        assert mapping.unmapped_reason is None

    def test_reify_s_unit_never_maps_its_legacy_slice(self, tmp_path: Path):
        mapping = holds.hold_consumers(f'{APP}/orchestrator-reify.service', tmp_path / 'reify')

        assert mapping.project is None
        assert Consumer(Role.PROJECT, 'reify.slice') not in {
            mapping.service, mapping.agents, mapping.project
        }

    def test_an_unresolved_own_cgroup_maps_only_verify(self, tmp_path: Path):
        root = tmp_path / 'dark-factory'

        mapping = holds.hold_consumers('', root)

        assert (mapping.service, mapping.agents, mapping.project) == (None, None, None)
        assert mapping.verify == Consumer(Role.VERIFY, scope_tag_for(root))
        assert mapping.unmapped_reason is not None
        assert 'unresolved' in mapping.unmapped_reason

    def test_a_hold_event_without_an_own_cgroup_maps_only_verify(self, tmp_path: Path):
        root = tmp_path / 'dark-factory'

        mapping = holds.hold_consumers(None, root)

        assert (mapping.service, mapping.agents, mapping.project) == (None, None, None)
        assert mapping.verify == Consumer(Role.VERIFY, scope_tag_for(root))
        assert mapping.unmapped_reason is not None
        assert 'no own cgroup' in mapping.unmapped_reason

    @pytest.mark.parametrize(
        ('own_cgroup', 'segment'),
        [
            (f'{APP}/orchestrator-watchdog.service', 'orchestrator-watchdog.service'),
            (f'{APP}/orchestrator-dark-factory.service/daemon', 'daemon'),
        ],
    )
    def test_a_last_segment_that_is_neither_shape_is_unmapped_by_name(
        self, tmp_path: Path, own_cgroup: str, segment: str
    ):
        root = tmp_path / 'dark-factory'

        mapping = holds.hold_consumers(own_cgroup, root)

        assert (mapping.service, mapping.agents, mapping.project) == (None, None, None)
        assert mapping.verify == Consumer(Role.VERIFY, scope_tag_for(root))
        assert mapping.unmapped_reason is not None
        assert segment in mapping.unmapped_reason


class TestHoldBeforeGoLive:
    """Collections every 30 s over [T0, T0+600]; nothing grows."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> Path:
        store = samples_db(tmp_path / 'load-samples.db')
        for seconds in range(0, 601, 30):
            write_collection(
                store, ts(seconds), {KERNEL: {Figure.ANON: 2**30}, DF_SERVICE: {Figure.ANON: 2**29}}
            )
        return store

    @pytest.fixture
    def root(self, tmp_path: Path, make_runs_db) -> Path:
        root = project_root(tmp_path, 'dark-factory')
        make_runs_db(
            [memory_hold(at(seconds), DF_UNIT) for seconds in (-60, 300, 60, 1000)],
            directory=runs_db(root).parent,
        )
        return root

    @pytest.fixture
    def report(self, root: Path, store: Path) -> holds.HoldsReport:
        return holds.read_holds_report(
            [root], store, since=T0 - timedelta(hours=1), window_seconds=120
        )

    def test_a_hold_before_the_first_collection_is_recorder_not_live(self, report):
        reading = reading_at(report, at(-60))

        assert reading.state is holds.HoldState.NOT_LIVE
        assert reading.window is None
        assert 'recorder not live at this hold' in hold_block(holds.render_text(report), at(-60))
        entry = hold_json(report, at(-60))
        assert entry['state'] == 'recorder_not_live'
        assert 'who_grew' not in entry
        assert 'who_paid' not in entry

    def test_a_hold_inside_the_collections_is_ranked_over_its_window(self, report):
        reading = reading_at(report, at(300))

        assert reading.state is holds.HoldState.RANKED
        assert reading.window is not None
        assert (reading.window.start, reading.window.end, reading.window.collections) == (
            at(180), at(300), 5
        )

    def test_an_empty_ranking_is_said_and_is_not_recorder_not_live(self, report):
        block = hold_block(holds.render_text(report), at(300))
        entry = hold_json(report, at(300))

        assert 'who grew: no consumer grew' in block
        assert 'who paid: no consumer paid' in block
        assert 'recorder not live' not in block
        assert (entry['state'], entry['who_grew'], entry['who_paid']) == ('ranked', [], [])

    def test_a_window_opening_before_go_live_starts_at_the_first_collection(self, report):
        reading = reading_at(report, at(60))

        assert reading.state is holds.HoldState.RANKED
        assert reading.window is not None
        assert (reading.window.start, reading.window.end, reading.window.collections) == (
            at(0), at(60), 3
        )

    def test_a_hold_past_the_last_collection_has_no_samples(self, report):
        reading = reading_at(report, at(1000))
        block = hold_block(holds.render_text(report), at(1000))

        assert reading.state is holds.HoldState.NO_SAMPLES
        assert reading.last_collection == at(600)
        assert holds.format_utc(at(600)) in block
        assert 'recorder not live' not in block
        assert hold_json(report, at(1000))['state'] == 'no_samples'

    def test_the_heading_names_the_root_the_task_and_the_arm_value(self, root, report):
        heading = hold_block(holds.render_text(report), at(300)).splitlines()[0]

        assert str(root) in heading
        assert 'task 5653' in heading
        assert 'mem_full_avg10 3.7' in heading

    def test_the_header_names_the_first_and_last_collection(self, report):
        samples_line = next(
            line for line in holds.render_text(report).splitlines() if line.startswith('samples db')
        )

        assert holds.format_utc(at(0)) in samples_line
        assert holds.format_utc(at(600)) in samples_line

    def test_the_readings_are_in_hold_order(self, report):
        assert [reading.hold.at for reading in report.readings] == [
            at(-60), at(60), at(300), at(1000)
        ]

    def test_with_no_collections_every_hold_is_recorder_not_live(self, tmp_path, root):
        empty = samples_db(tmp_path / 'empty' / 'load-samples.db')

        report = holds.read_holds_report(
            [root], empty, since=T0 - timedelta(hours=1), window_seconds=120
        )

        assert {reading.state for reading in report.readings} == {holds.HoldState.NOT_LIVE}
        assert 'no consumer collections' in holds.render_text(report)


def write_collections(
    store: Path,
    collections: Mapping[int, Mapping[Consumer, Mapping[Figure, float]]],
    *,
    unreadable: Mapping[int, frozenset[Consumer]] | None = None,
) -> None:
    """One collection per entry, keyed by its seconds after T0."""
    for seconds, consumers in collections.items():
        flagged = (unreadable or {}).get(seconds, frozenset())
        write_collection(store, ts(seconds), consumers, unreadable=flagged)


class TestRankings:
    """Collections at S = T0, T0+30 and E = T0+60; one hold at E+5, so the window is (S, E]."""

    T3CODE = Consumer(Role.INTERACTIVE, 't3code')
    SCCACHE = Consumer(Role.SHARED, 'sccache.service')
    REIFY_SLICE = Consumer(Role.PROJECT, 'reify.slice')
    UNATTRIBUTED = Consumer(Role.OTHER, 'app.slice')
    EXITED = Consumer(Role.EXITED, 'app.slice')
    UNREADABLE = Consumer(Role.OTHER, 'system.slice')

    @pytest.fixture
    def root(self, tmp_path: Path, make_runs_db) -> Path:
        root = project_root(tmp_path, 'dark-factory')
        make_runs_db([memory_hold(at(65), DF_UNIT)], directory=runs_db(root).parent)
        return root

    @pytest.fixture
    def verify(self, root: Path) -> Consumer:
        return Consumer(Role.VERIFY, scope_tag_for(root))

    @pytest.fixture
    def report(self, tmp_path: Path, root: Path, verify: Consumer) -> holds.HoldsReport:
        store = samples_db(tmp_path / 'load-samples.db')
        paid = {
            Figure.RECLAIM_KSWAPD_PAGES: 1000,
            Figure.RECLAIM_DIRECT_PAGES: 500,
            Figure.MEMORY_SOME_STALL_USEC: 2.0,
            Figure.MEMORY_FULL_STALL_USEC: 0.5,
        }
        before_the_window = {
            Figure.RECLAIM_KSWAPD_PAGES: 99_999,
            Figure.MEMORY_SOME_STALL_USEC: 100.0,
            Figure.MEMORY_FULL_STALL_USEC: 100.0,
        }

        def collection(step: int) -> dict[Consumer, dict[Figure, float]]:
            later = step > 0
            return {
                verify: {Figure.ANON: (1 + step / 2) * GiB, Figure.SWAP: 0},
                DF_SERVICE: {Figure.ANON: (600 - 100 * step) * MiB, Figure.SWAP: 100 * step * MiB},
                self.T3CODE: {Figure.ANON: 100 * MiB, **(paid if later else before_the_window)},
                self.SCCACHE: {Figure.ANON: 10 * MiB, Figure.RECLAIM_KSWAPD_PAGES: 50 if later else 0},
                self.REIFY_SLICE: {Figure.ANON: (50 + 5 * step) * MiB},
                self.UNATTRIBUTED: {
                    Figure.ANON: (100 + 150 * step) * MiB,
                    Figure.RECLAIM_DIRECT_PAGES: 100 if later else 0,
                },
                self.EXITED: {Figure.ANON: (25 * step) * MiB},
                KERNEL: {Figure.ANON: (200 + 10 * step) * MiB, Figure.RECLAIM_KSWAPD_PAGES: 25 if later else 0},
                self.UNREADABLE: {Figure.ANON: 5 * MiB},
            }

        write_collections(
            store,
            {0: collection(0), 30: collection(1), 60: collection(2)},
            unreadable={30: frozenset({self.UNREADABLE})},
        )
        return holds.read_holds_report([root], store, since=T0, window_seconds=120)

    @pytest.fixture
    def window(self, report: holds.HoldsReport) -> holds.Window:
        (reading,) = report.readings
        assert reading.state is holds.HoldState.RANKED
        assert reading.window is not None
        assert (reading.window.start, reading.window.end) == (at(0), at(60))
        return reading.window

    def test_who_grew_ranks_program_plus_swap_growth_without_the_aside_roles(self, window, verify):
        grew = holds.who_grew(window)

        assert [
            (row.consumer, row.grew, row.program_growth_bytes, row.swap_growth_bytes) for row in grew
        ] == [(verify, GiB, GiB, 0), (self.REIFY_SLICE, 10 * MiB, 10 * MiB, 0)]

    def test_a_swapped_out_service_has_not_grown(self, window):
        service = window.consumers[DF_SERVICE]

        assert (service.program_growth_bytes, service.swap_growth_bytes, service.grew) == (
            -200 * MiB, 200 * MiB, 0
        )
        assert DF_SERVICE not in {row.consumer for row in holds.who_grew(window)}

    def test_who_paid_ranks_reclaimed_pages_over_the_window_only(self, window):
        paid = holds.who_paid(window)

        assert [
            (
                row.consumer,
                row.reclaimed_pages,
                row.memory_some_stall_seconds,
                row.memory_full_stall_seconds,
            )
            for row in paid
        ] == [(self.T3CODE, 3000, 4.0, 1.0), (self.SCCACHE, 100, 0, 0)]

    def test_kernel_unattributed_and_exited_are_reported_beside_with_their_shares(self, window):
        total_growth = GiB + 10 * MiB + 300 * MiB + 50 * MiB + 20 * MiB
        total_reclaimed = 3000 + 100 + 200 + 50

        aside = {row.label: row for row in holds.aside(window)}

        assert {label: (row.grew, row.reclaimed_pages) for label, row in aside.items()} == {
            'kernel': (20 * MiB, 50),
            'unattributed': (300 * MiB, 200),
            'exited': (50 * MiB, 0),
        }
        assert aside['kernel'].growth_share == pytest.approx(20 * MiB / total_growth)
        assert aside['kernel'].reclaim_share == pytest.approx(50 / total_reclaimed)
        assert aside['unattributed'].growth_share == pytest.approx(300 * MiB / total_growth)
        assert aside['unattributed'].reclaim_share == pytest.approx(200 / total_reclaimed)
        assert aside['exited'].growth_share == pytest.approx(50 * MiB / total_growth)
        assert aside['exited'].reclaim_share == 0

    def test_a_consumer_unreadable_in_the_window_is_listed_incomplete(self, window):
        assert holds.incomplete(window) == (self.UNREADABLE,)

    def test_the_holding_project_block(self, report, verify):
        project = hold_json(report, at(65))['holding_project']

        assert project['service']['consumer'] == encode_consumer(DF_SERVICE)
        assert project['service']['status'] == 'present'
        assert project['service']['figures']['grew_bytes'] == 0
        assert project['service']['who_grew_rank'] is None
        assert project['verify']['consumer'] == encode_consumer(verify)
        assert project['verify']['who_grew_rank'] == 1
        assert project['agents']['consumer'] == encode_consumer(
            Consumer(Role.AGENTS, 'orchestrator-dark-factory.service')
        )
        assert project['agents']['status'] == 'absent'
        assert project['project']['status'] == 'not_mapped'
        assert project['project']['consumer'] is None
        assert project['project']['reason']

    def test_the_json_carries_the_full_rankings_aside_and_incomplete(self, report, verify):
        entry = hold_json(report, at(65))

        assert [row['consumer'] for row in entry['who_grew']] == [
            encode_consumer(verify), encode_consumer(self.REIFY_SLICE)
        ]
        assert entry['who_grew'][0]['grew_bytes'] == GiB
        assert [row['consumer'] for row in entry['who_paid']] == [
            encode_consumer(self.T3CODE), encode_consumer(self.SCCACHE)
        ]
        assert entry['who_paid'][0]['reclaimed_pages'] == 3000
        assert [row['label'] for row in entry['aside']] == ['kernel', 'unattributed', 'exited']
        assert entry['incomplete'] == [encode_consumer(self.UNREADABLE)]

    def test_the_text_marks_this_project_and_prints_the_aside_lines(self, report, verify):
        lines = hold_block(holds.render_text(report), at(65)).splitlines()

        assert any(
            encode_consumer(verify) in line and "[this project's verify]" in line for line in lines
        )
        labels = {line.strip().split(':')[0] for line in lines}
        assert {'kernel', 'unattributed', 'exited'} <= labels
        assert any(encode_consumer(self.UNREADABLE) in line for line in lines)

    def test_the_text_shows_bytes_in_binary_units(self, report, verify):
        block = hold_block(holds.render_text(report), at(65))

        assert '+1.00 GiB' in block
        assert '+10.0 MiB' in block
        assert '3000 pages' in block


class TestHoldMappingBothShapesEndToEnd:
    """The §7 row "Hold mapping, both shapes", from the runs.db events to the report's JSON."""

    SOLAR_SERVICE = Consumer(Role.SERVICE, 'orchestrator-my-solar-challenge.service')
    SLICE = 'df-dark_factory.slice'

    @pytest.fixture
    def roots(self, tmp_path: Path, make_runs_db) -> tuple[Path, Path]:
        solar = project_root(tmp_path, 'solar-challenge')
        make_runs_db(
            [memory_hold(at(65), f'{APP}/orchestrator-my-solar-challenge.service')],
            directory=runs_db(solar).parent,
        )
        factory = project_root(tmp_path, 'dark-factory')
        make_runs_db(
            [memory_hold(at(65), f'{APP}/df.slice/{self.SLICE}')], directory=runs_db(factory).parent
        )
        return solar, factory

    @pytest.fixture
    def report(self, tmp_path: Path, roots: tuple[Path, Path]) -> holds.HoldsReport:
        solar, _factory = roots
        store = samples_db(tmp_path / 'load-samples.db')
        solar_verify = Consumer(Role.VERIFY, scope_tag_for(solar))
        growth = {
            self.SOLAR_SERVICE: 50 * MiB,
            solar_verify: 200 * MiB,
            Consumer(Role.SERVICE, self.SLICE): 30 * MiB,
            Consumer(Role.AGENTS, self.SLICE): 70 * MiB,
            Consumer(Role.PROJECT, self.SLICE): 5 * MiB,
        }
        write_collections(
            store,
            {
                seconds: {
                    consumer: {Figure.ANON: 100 * MiB + grown * seconds / 60}
                    for consumer, grown in growth.items()
                }
                | {KERNEL: {Figure.ANON: GiB}}
                for seconds in (0, 30, 60)
            },
        )
        return holds.read_holds_report(list(roots), store, since=T0, window_seconds=120)

    def entry_for(self, report: holds.HoldsReport, root: Path) -> Mapping[str, Any]:
        entries = holds.report_json(report)['holds']
        return next(entry for entry in entries if entry['project_root'] == str(root))

    def test_the_unit_shape_resolves_service_and_verify(self, report, roots):
        solar, _factory = roots

        project = self.entry_for(report, solar)['holding_project']

        assert project['service']['consumer'] == encode_consumer(self.SOLAR_SERVICE)
        assert project['service']['figures']['grew_bytes'] == 50 * MiB
        assert project['verify']['consumer'] == encode_consumer(
            Consumer(Role.VERIFY, scope_tag_for(solar))
        )
        assert project['verify']['figures']['grew_bytes'] == 200 * MiB
        assert project['agents']['status'] == 'absent'
        assert project['project']['status'] == 'not_mapped'

    def test_the_slice_shape_resolves_service_agents_and_project_by_the_slice(self, report, roots):
        _solar, factory = roots

        project = self.entry_for(report, factory)['holding_project']

        assert {
            role: (project[role]['consumer'], project[role]['figures']['grew_bytes'])
            for role in ('service', 'agents', 'project')
        } == {
            'service': (encode_consumer(Consumer(Role.SERVICE, self.SLICE)), 30 * MiB),
            'agents': (encode_consumer(Consumer(Role.AGENTS, self.SLICE)), 70 * MiB),
            'project': (encode_consumer(Consumer(Role.PROJECT, self.SLICE)), 5 * MiB),
        }
        assert project['verify']['consumer'] == encode_consumer(
            Consumer(Role.VERIFY, scope_tag_for(factory))
        )
        assert project['verify']['status'] == 'absent'
