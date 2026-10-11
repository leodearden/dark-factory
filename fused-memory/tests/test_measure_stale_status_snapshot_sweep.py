"""Tests for scripts/measure_stale_status_snapshot_sweep.py (task 4851).

The script's pure core is driven with in-memory edges and a statuses dict;
no backend is involved.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from _fm_helpers import complete_paged_read, incomplete_paged_read, load_script_module
from reconciliation import plural_enum_shapes

from fused_memory.backends.graphiti_client import INCOMPLETE_SHORT_READ
from fused_memory.reconciliation.stale_status_snapshot_edge_sweep import (
    extract_snapshot_edge_task_ids,
)

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / 'scripts' / 'measure_stale_status_snapshot_sweep.py'
)
probe = load_script_module(SCRIPT_PATH, mod_name='measure_stale_status_snapshot_sweep')


def _edges(*facts: str) -> list[dict]:
    return [{'uuid': f'e{i}', 'fact': fact, 'name': ''} for i, fact in enumerate(facts, 1)]


E1 = 'Task 142 is an active pending task'
E2 = "Task 2401's follow-up is tracked as task 2405, which is in progress."
E3 = 'Task 7 is done'
E4 = 'Task 9 is pending'
E5 = 'Tasks 1020 and 1030 are pending.'
COVERAGE_STATUSES = {'142': 'done', '2405': 'done', '9': 'pending', '1020': 'done'}


class TestCoverage:
    def test_the_funnel_counts_and_the_unselected_sample(self):
        coverage = probe.measure_coverage(
            _edges(E1, E2, E3, E4, E5), COVERAGE_STATUSES, max_samples=10,
        )

        assert coverage.gate_passing == 4
        assert coverage.terminal_referencing == 3
        assert coverage.selected == 2
        assert coverage.unselected == 1
        assert coverage.unselected_samples == (E2,)

    def test_samples_are_capped(self):
        facts = [f'Task {n} is tracked; its follow-up, task {n}, which is in progress'
                 for n in range(100, 105)]
        statuses = {str(n): 'done' for n in range(100, 105)}

        coverage = probe.measure_coverage(_edges(*facts), statuses, max_samples=2)

        assert len(coverage.unselected_samples) == 2

    @pytest.mark.parametrize(
        'fact',
        [
            *plural_enum_shapes.PRECISION_GUARD_SHAPES,
            *plural_enum_shapes.GUARD_REJECTED_SUPPRESSION_SHAPES,
            *(fact for fact, _ in plural_enum_shapes.SUBJECT_POSITIVE_SHAPES),
            *plural_enum_shapes.ADVERBIAL_PREAMBLE_SHAPES,
            E1, E2, E5,
        ],
    )
    def test_the_lexical_probe_is_a_superset_of_extraction(self, fact):
        """Otherwise 'unselected' could hide a miss the probe never saw."""
        assert extract_snapshot_edge_task_ids(fact) <= (
            probe.lexically_referenced_task_ids(fact)
        )


GENERAL_RULE_FACTS = (
    'Task 5 is pending',
    'Task 6 is stalled',
    'Task 8 is an active task',
    'Task 11 is pending',
    'Task 12 is pending',
)
GENERAL_RULE_STATUSES = {
    '5': 'in-progress', '6': 'in-progress', '8': 'in-progress',
    '11': 'done', '12': 'pending',
}


class TestGeneralRule:
    def test_only_a_defined_contradiction_the_shipped_rules_miss_is_newly_retired(self):
        rule = probe.measure_general_rule(
            _edges(*GENERAL_RULE_FACTS), GENERAL_RULE_STATUSES, max_samples=10,
        )

        assert rule.newly_retired == 1
        assert rule.newly_retired_samples == ('Task 5 is pending',)
        assert rule.asserted[('stalled', 'in-progress')] == 1
        assert rule.asserted[('active', 'in-progress')] == 1
        assert rule.asserted[('pending', 'done')] == 1
        assert rule.asserted[('pending', 'pending')] == 1
        assert set(rule.undefined_contradiction) == {'stalled', 'active'}
        assert rule.verdict == probe.general_rule_verdict(1)

    def test_the_verdict_has_two_arms(self):
        assert probe.general_rule_verdict(0) == probe.DO_NOT_WIDEN_VERDICT
        assert probe.general_rule_verdict(3).startswith('PROVISIONAL')
        assert probe.general_rule_verdict(3) != probe.DO_NOT_WIDEN_VERDICT


class TestTokenInternalBreaks:
    def test_counts_only_what_the_shipped_predicate_calls_token_internal(self):
        breaks = probe.measure_token_internal_breaks(
            _edges(
                'See https://ci/build?ref=main for the log',
                'The a;b split',
                "The id 'review-&gt;') was dropped",
                'It is done. Next comes review',
            ),
            max_samples=5,
        )

        assert breaks.counts == {'.': 0, ';': 2, '!': 0, '?': 1}
        assert breaks.samples['?'] == ('See https://ci/build?ref=main for the log',)
        assert len(breaks.samples[';']) == 2
        assert breaks.samples['.'] == ()


def _edge_source(paged):
    async def source(project_id):
        return {'entity': _edges(E1, E4)}, paged

    return source


async def _status_source(project_root):
    return dict(COVERAGE_STATUSES)


async def _measure(paged):
    return await probe.measure_project(
        'dark_factory', '/repo',
        edge_source=_edge_source(paged), status_source=_status_source, max_samples=5,
    )


class TestFailClosed:
    @pytest.mark.asyncio
    async def test_an_incomplete_read_fails_the_run(self):
        project = await _measure(
            incomplete_paged_read(INCOMPLETE_SHORT_READ, rows_seen=2, expected_rows=9),
        )
        report = probe.build_report([project], measured_at='2026-10-10T00:00:00+00:00')

        assert project.complete is False
        assert project.incomplete_kind == INCOMPLETE_SHORT_READ
        assert probe.exit_code(report) == 1

    @pytest.mark.asyncio
    async def test_a_complete_read_passes(self):
        project = await _measure(complete_paged_read(rows_seen=2))
        report = probe.build_report([project], measured_at='2026-10-10T00:00:00+00:00')

        assert project.complete is True
        assert project.valid_edges == 2
        assert probe.exit_code(report) == 0

    def test_a_run_that_measured_nothing_fails(self):
        report = probe.build_report([], measured_at='2026-10-10T00:00:00+00:00')

        assert probe.exit_code(report) == 1


async def _report(paged=None, triage_note=None):
    project = await _measure(paged or complete_paged_read(rows_seen=2))
    return probe.build_report(
        [project], measured_at='2026-10-10T00:00:00+00:00', triage_note=triage_note,
    )


class TestRendering:
    @pytest.mark.asyncio
    async def test_markdown_carries_sections_headlines_and_verdict(self):
        report = await _report(triage_note='Misses are relative clauses.')

        markdown = probe.render_markdown(report)

        for header in ('## dark_factory', '### Coverage', '### General rule',
                       '### Token-internal clause breaks', '## Triage'):
            assert header in markdown
        coverage = report.projects[0].coverage
        assert f'| gate_passing | {coverage.gate_passing} |' in markdown
        assert f'| terminal_referencing | {coverage.terminal_referencing} |' in markdown
        assert report.projects[0].general_rule.verdict in markdown
        assert 'Misses are relative clauses.' in markdown

    @pytest.mark.asyncio
    async def test_json_round_trips_with_a_schema_version(self):
        report = await _report()

        payload = json.loads(probe.to_json(report))

        assert payload['schema_version'] == probe.SCHEMA_VERSION
        assert payload['complete'] is True
        assert payload['projects'][0]['coverage']['gate_passing'] == 2


class TestArtifactSafety:
    @pytest.mark.asyncio
    async def test_an_incomplete_report_writes_sidecars_only(self, tmp_path):
        json_out, md_out = tmp_path / 'r.json', tmp_path / 'r.md'
        json_out.write_text('ORIGINAL')
        md_out.write_text('ORIGINAL')
        report = await _report(
            incomplete_paged_read(INCOMPLETE_SHORT_READ, rows_seen=2, expected_rows=9),
        )

        written = probe.write_artifacts(report, json_out, md_out)

        assert json_out.read_text() == 'ORIGINAL'
        assert md_out.read_text() == 'ORIGINAL'
        assert written == (tmp_path / 'r.incomplete.json', tmp_path / 'r.incomplete.md')
        assert all(path.is_file() for path in written)

    @pytest.mark.asyncio
    async def test_a_complete_report_writes_the_canonical_paths(self, tmp_path):
        json_out, md_out = tmp_path / 'r.json', tmp_path / 'r.md'

        written = probe.write_artifacts(await _report(), json_out, md_out)

        assert written == (json_out, md_out)
        assert json.loads(json_out.read_text())['complete'] is True


class TestTaskStorePreflight:
    def test_a_root_without_a_populated_task_store_is_refused_before_any_read(
        self, tmp_path,
    ):
        assert probe.main(['--project', f'dark_factory={tmp_path}']) == 2
        assert not (tmp_path / '.taskmaster').exists()
