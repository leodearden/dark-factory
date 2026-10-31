import json

import pytest

from orchestrator.evals.elo import INDISTINGUISHABLE_THRESHOLD, JudgeState, TaskPool
from orchestrator.evals.elo_report import (
    build_report,
    compute_aggregate_ratings,
    compute_tiers,
    format_markdown,
    save_report,
)


def _two_task_state():
    # t1: A and B have played twice each; C has played no match at all.
    return JudgeState(per_task={
        't1': TaskPool(
            ratings={'A': 1100.0, 'B': 1000.0, 'C': 900.0},
            matches=[
                {'config_a': 'A', 'config_b': 'B'},
                {'config_a': 'B', 'config_b': 'A'},
            ],
        ),
        't2': TaskPool(ratings={'A': 1200.0, 'D': 1050.0}),
    })


def _sections(markdown):
    """Map each '## ' heading to the lines beneath it, up to the next one."""
    sections, current = {}, None
    for line in markdown.splitlines():
        if line.startswith('## '):
            current = line[3:]
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return sections


def _row_for(lines, config):
    return next(line for line in lines if line.split() and config in line.split())


class TestAggregateRatingsUnion:
    """The April all-tasks-INTERSECTION collapse is retired: the aggregate
    leaderboard spans the UNION of configs, each averaged over only the tasks in
    which it actually appears.
    """

    def _state(self):
        return JudgeState(per_task={
            't1': TaskPool(ratings={'A': 1100.0, 'B': 1000.0, 'C': 900.0}),
            't2': TaskPool(ratings={'A': 1200.0, 'B': 1050.0}),
        })

    def test_union_includes_config_present_in_only_one_task(self):
        agg = compute_aggregate_ratings(self._state())
        # C ran ONLY in t1 but MUST survive — no collapse-to-2-entries.
        assert set(agg.keys()) == {'A', 'B', 'C'}

    def test_aggregate_is_mean_over_tasks_where_config_appears(self):
        agg = compute_aggregate_ratings(self._state())
        assert agg['A'] == pytest.approx(1150.0)   # mean(1100, 1200)
        assert agg['B'] == pytest.approx(1025.0)   # mean(1000, 1050)
        assert agg['C'] == pytest.approx(900.0)    # only t1 → its t1 rating

    def test_empty_state_is_empty(self):
        assert compute_aggregate_ratings(JudgeState(per_task={})) == {}


class TestComputeTiers:
    """Configs within INDISTINGUISHABLE_THRESHOLD of their tier's top share a tier."""

    def test_no_ratings_is_no_tiers(self):
        assert compute_tiers({}) == []

    def test_tiers_are_measured_from_the_tier_top_in_descending_rating(self):
        step = 0.8 * INDISTINGUISHABLE_THRESHOLD
        # C is within the threshold of B but not of A, B's tier top.
        ratings = {'C': 1000.0 - 2 * step, 'A': 1000.0, 'B': 1000.0 - step}
        assert compute_tiers(ratings) == [['A', 'B'], ['C']]


class TestFormatMarkdown:
    """build_report then format_markdown renders the aggregate and per-task boards."""

    def test_title_opens_the_report(self):
        markdown = format_markdown(build_report(_two_task_state()))
        assert markdown.splitlines()[0] == '# Eval Judge Report'

    def test_aggregate_leaderboard_lists_the_union_of_configs(self):
        sections = _sections(format_markdown(build_report(_two_task_state())))
        aggregate = sections['Aggregate Leaderboard']
        for config in ('A', 'B', 'C', 'D'):
            assert _row_for(aggregate, config)

    def test_each_task_gets_its_own_section(self):
        sections = _sections(format_markdown(build_report(_two_task_state())))
        assert {'t1', 't2'} <= set(sections)

    def test_a_config_with_fewer_than_two_matches_is_marked_preliminary(self):
        t1 = _sections(format_markdown(build_report(_two_task_state())))['t1']
        assert _row_for(t1, 'C').split()[0] == '*'
        assert _row_for(t1, 'A').split()[0] != '*'


class TestSaveReport:
    def test_writes_round_trippable_json_ending_in_a_newline(self, tmp_path):
        report = build_report(_two_task_state())
        path = save_report(report, path=tmp_path / 'r.json')
        assert path == tmp_path / 'r.json'
        text = path.read_text()
        assert text.endswith('\n')
        assert json.loads(text) == report
