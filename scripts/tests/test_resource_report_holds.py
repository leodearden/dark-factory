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
from sampler.consumers import Consumer, Figure, Role
from shared.project_layout import runs_db
from shared.verify_scope_tag import scope_tag_for

APP = '/user.slice/user-1000.slice/user@1000.service/app.slice'
DF_UNIT = f'{APP}/orchestrator-dark-factory.service'
T0 = datetime(2026, 11, 21, 18, 0, tzinfo=UTC)
KERNEL = Consumer(Role.KERNEL)
DF_SERVICE = Consumer(Role.SERVICE, 'orchestrator-dark-factory.service')


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
