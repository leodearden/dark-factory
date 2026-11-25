"""verify_host_trial_report: the CLI end to end, over fixture project roots and a samples store."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from shared.project_layout import runs_db
from verify_host_trial_allowance import ALLOWANCE_RULE
from verify_host_trial_blocks import BLOCK_RULE
from verify_host_trial_decision import DECISION_RULE
from verify_host_trial_fixtures import (
    build_store,
    cap_hit,
    invocation,
    lifecycle,
    merge_finalized,
    merge_verify,
    project_root,
    samples_db,
    task_result,
)
from verify_host_trial_measures import FIGURE_RULES, FigureId
from verify_host_trial_report import main

SINCE = datetime(2026, 11, 11, tzinfo=UTC)
UNTIL = datetime(2026, 11, 23, tzinfo=UTC)
NOW = datetime(2026, 11, 25, tzinfo=UTC)
HOST = 'laptop'
COSTS = ['--host-cost-usd', '252', '--seat-bill-usd', '1375']


def day(n: float) -> datetime:
    return SINCE + timedelta(days=n)


@pytest.fixture
def store(tmp_path, make_runs_db):
    def build(name, events=(), **tables):
        return build_store(tmp_path, make_runs_db, name, events, **tables)

    return build


@pytest.fixture
def samples(tmp_path):
    return samples_db(tmp_path / 'load-samples.db')


def run(capsys, roots, samples, *extra: str) -> str:
    argv = ['--host', HOST, '--since', SINCE.date().isoformat(), '--until', UNTIL.date().isoformat(), *COSTS]
    argv += [arg for root in roots for arg in ('--project-root', str(root))]
    argv += ['--samples-db', str(samples), '--resamples', '500', *extra]
    assert main(argv, now=NOW) == 0
    return capsys.readouterr().out


def lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def decision_line(text: str) -> str:
    (line,) = [line for line in lines(text) if line.startswith('decision:')]
    return line


def landings(task_prefix: str, start: float, count: int) -> list[dict]:
    """*count* tasks landing one per hour from day *start*."""
    return [merge_finalized(day(start) + timedelta(hours=i), f'{task_prefix}{i}', 'done') for i in range(count)]


@pytest.fixture
def toggled(store):
    """Two stores whose lifecycle rows make on [0,3), off [3,6), on [6,9), off [9,12).

    The host verified in both on-blocks. Fleet landings are 6 per on-block and
    3 per off-block, so on is 2.00 a day, off 1.00, and every resample's effect
    is exactly +100%.
    """
    first = store('dark-factory', [
        merge_verify(day(-30), HOST, 600, 'old'),
        merge_verify(day(1), HOST, 600, 'a1'),
        merge_verify(day(7), HOST, 600, 'a2'),
        merge_verify(day(4), 'local', 600, 'a3'),
        lifecycle(day(3), 'verify_host_retired', HOST),
        lifecycle(day(6), 'verify_host_admitted', HOST),
        lifecycle(day(9), 'verify_host_retired', HOST),
        *landings('on1-', 0.5, 4),
        *landings('off1-', 3.5, 2),
        *landings('on2-', 6.5, 4),
        *landings('off2-', 9.5, 3),
    ], invocations=[invocation(day(1), 'max-a', 30.0)])
    second = store('reify', [
        merge_verify(day(2), HOST, 900, 'b1'),
        lifecycle(day(3) + timedelta(seconds=30), 'verify_host_retired', HOST),
        lifecycle(day(6) + timedelta(seconds=20), 'verify_host_admitted', HOST),
        *landings('on1-', 1, 2),
        *landings('off1-', 4, 1),
        *landings('on2-', 7, 2),
    ], task_results=[task_result('on2-0', 'done', day(7.5))])
    return first, second


class TestNoHostRows:
    @pytest.fixture
    def local_only(self, store):
        return store('dark-factory', [merge_verify(day(2), 'local', 600, '7'), *landings('t', 1, 5)])

    def test_no_verify_ran_on_this_host_and_no_comparison_is_printed(self, capsys, local_only, samples):
        text = run(capsys, [local_only], samples)

        assert 'no verify ran on this host' in text
        line = decision_line(text)
        assert 'no comparison' in line
        assert 'no verify ran on this host' in line
        assert '+0' not in line
        assert '0.0%' not in line

    def test_json_says_the_effect_is_null_with_its_reason(self, capsys, local_only, samples):
        report = json.loads(run(capsys, [local_only], samples, '--json'))

        assert report['decision']['effect'] is None
        assert report['decision']['reason']

    def test_lifecycle_rows_never_written_are_stated_as_such(self, capsys, local_only, samples):
        text = run(capsys, [local_only], samples)

        assert (
            'host lifecycle: no verify_host_admitted or verify_host_retired row ever written in any store read; '
            "block boundaries come from the host's first merge_verify only"
        ) in lines(text)

    def test_lifecycle_rows_that_exist_outside_the_window_are_a_different_state(self, capsys, store, samples):
        root = store('dark-factory', [lifecycle(day(-5), 'verify_host_admitted', HOST)])

        text = run(capsys, [root], samples)

        assert 'ever written in any store read' not in text
        assert 'host lifecycle: 1 verify_host_admitted or verify_host_retired row for laptop, 0 inside the window' in lines(text)


class TestToggledBlocks:
    def test_each_block_prints_its_state_bounds_days_and_opener(self, capsys, toggled, samples):
        text = lines(run(capsys, toggled, samples))

        assert (
            f'block 2: off [{day(3).isoformat()}, {day(6).isoformat()}) 3.00 days, '
            f'opened by retired at {day(3).isoformat()}'
        ) in text
        assert (
            f'block 1: on [{SINCE.isoformat()}, {day(3).isoformat()}) 3.00 days, '
            f'opened by first_verify at {(day(-30) - timedelta(seconds=600)).isoformat()}'
        ) in text
        assert sum(line.startswith('block ') for line in text) == 4

    def test_every_figure_opens_with_its_rule_line(self, capsys, toggled, samples):
        text = lines(run(capsys, toggled, samples))

        rules = {f'rule: {rule}' for rule in (BLOCK_RULE, ALLOWANCE_RULE, DECISION_RULE, *FIGURE_RULES.values())}
        assert {line for line in text if line.startswith('rule:')} == rules
        for figure, rule in FIGURE_RULES.items():
            at = [i for i, line in enumerate(text) if line == f'rule: {rule}']
            expected = 4 if figure is FigureId.CPU_PRESSURE else 4 * 2
            assert len(at) == expected, figure
            assert all(not text[i + 1].startswith('rule:') for i in at)

    def test_each_project_prints_its_figures_per_block(self, capsys, toggled, samples):
        text = run(capsys, toggled, samples)

        assert 'project dark-factory' in text
        assert 'project reify' in text

    def test_the_decision_line_compares_on_and_off_with_its_interval_and_the_cost_share(self, capsys, toggled, samples):
        line = decision_line(run(capsys, toggled, samples))

        assert 'on 2.00 landings/day' in line
        assert 'off 1.00 landings/day' in line
        assert '+100.0%' in line
        assert 'interval +100.0% to +100.0%' in line
        assert '18.3% of the seat bill' in line

    def test_json_rules_are_the_same_strings(self, capsys, toggled, samples):
        report = json.loads(run(capsys, toggled, samples, '--json'))

        assert report['rules'] == {
            'blocks': BLOCK_RULE,
            'decision': DECISION_RULE,
            'allowance': ALLOWANCE_RULE,
            'figures': {str(figure): rule for figure, rule in FIGURE_RULES.items()},
        }
        assert report['decision']['effect'] == 1.0
        assert [block['state'] for block in report['blocks']] == ['on', 'off', 'on', 'off']
        assert [block['landings'] for block in report['blocks']] == [6, 3, 6, 3]

    def test_every_store_is_byte_identical_after_a_run(self, capsys, toggled, samples):
        paths = [runs_db(root) for root in toggled] + [samples]
        before = [path.read_bytes() for path in paths]

        run(capsys, toggled, samples)
        run(capsys, toggled, samples, '--json')

        assert [path.read_bytes() for path in paths] == before


def test_a_root_without_a_runs_db_is_named_and_the_others_still_report(capsys, toggled, tmp_path, samples):
    absent = project_root(tmp_path, 'pump-web-ui')

    text = run(capsys, [*toggled, absent], samples)

    assert f'project pump-web-ui: missing ({runs_db(absent)})' in lines(text)
    assert 'project reify' in text
    assert '+100.0%' in decision_line(text)


def test_unparseable_rows_of_a_whole_store_source_are_labelled_whole_store(capsys, store, samples):
    plain_text_details = {**cap_hit(day(-180), 'max-a', None), 'details': 'Escalation watcher (auto)'}
    root = store(
        'dark-factory',
        account_events=[plain_text_details],
        invocations=[{**invocation(day(1), 'max-a', 1.0), 'cost_usd': 'n/a'}],
    )

    text = lines(run(capsys, [root], samples))
    (project,) = json.loads(run(capsys, [root], samples, '--json'))['projects']

    assert 'unparseable rows skipped: invocations 1, cap_hit 1 (whole store)' in text
    assert {source: project['sources'][source] for source in ('invocations', 'cap_hit')} == {
        'invocations': {'state': 'read', 'reason': None, 'skipped': 1, 'whole_store': False},
        'cap_hit': {'state': 'read', 'reason': None, 'skipped': 1, 'whole_store': True},
    }


def test_json_allowance_places_cycles_at_weekly_caps_only_and_counts_the_others(capsys, store, samples):
    weekly_reset = day(4)
    five_hour_reset = day(6) + timedelta(hours=5)
    five_hour = "You've hit your 5-hour limit · resets 7pm (Europe/London)"
    root = store(
        'dark-factory',
        account_events=[
            cap_hit(day(2), 'max-a', weekly_reset),
            cap_hit(day(6), 'max-a', five_hour_reset, reason=five_hour),
        ],
        invocations=[invocation(day(1), 'max-a', 30.0)],
    )

    allowance = json.loads(run(capsys, [root], samples, '--json'))['allowance']

    cycles = [cycle for cycle in allowance['cycles'] if cycle['account'] == 'max-a']
    assert [cycle['reset'] for cycle in cycles if cycle['capped']] == [weekly_reset.isoformat()]
    assert five_hour_reset.isoformat() not in {cycle['reset'] for cycle in cycles}
    assert allowance['other_caps'] == {'session': 1}


@pytest.mark.parametrize(
    'override',
    [
        ['--until', SINCE.date().isoformat()],
        ['--until', (SINCE - timedelta(days=1)).date().isoformat()],
        ['--host-cost-usd', '-1'],
        ['--seat-bill-usd', '0'],
        ['--seat-bill-usd', '-5'],
        ['--since', 'yesterday'],
    ],
    ids=['until-equals-since', 'until-before-since', 'negative-host-cost', 'zero-seat-bill', 'negative-seat-bill', 'bad-date'],
)
def test_usage_errors_exit_2(capsys, override, samples):
    argv = ['--host', HOST, '--since', SINCE.date().isoformat(), '--until', UNTIL.date().isoformat(), *COSTS]
    argv += ['--samples-db', str(samples), *override]

    with pytest.raises(SystemExit) as exit_:
        main(argv, now=NOW)

    assert exit_.value.code == 2


def test_the_roots_default_to_the_known_project_roots(capsys, monkeypatch, toggled, samples):
    monkeypatch.setenv('DASHBOARD_KNOWN_PROJECT_ROOTS', ','.join(str(root) for root in toggled))

    text = run(capsys, [], samples)

    assert 'project dark-factory: read' in text
    assert 'project reify: read' in text
