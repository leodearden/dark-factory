"""The MEMORY_EVALS metric row: one row per metric, one ``verdicts`` entry per fingerprint (task 10137).

A metric row carries its metric fields and trend ONCE, plus the judgement
fields the producer scopes per metric (``value``, ``limit_ref``,
``run_stamp``), plus ``verdicts``: one entry per fingerprint identity
``(eval_id, metric_id, item | window)``.  Every case goes through the public
builder over a tmp_path artifact tree, the way
``test_memory_evals_eval_build.py::_build`` does.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _dashboard_helpers import dump_artifact, write_escalation_record

from dashboard.config import DashboardConfig
from dashboard.data import memory_evals
from dashboard.data.escalation_corpus import measure_corpus, reconciliation_queue

_EVAL = 'eval-a'
_FINAL_RUN = datetime(2026, 7, 5, 3, 15, tzinfo=UTC)
_NOW = _FINAL_RUN + timedelta(hours=3)
_STAMP_FORMAT = '%Y%m%dT%H%M%SZ'
_FINAL_STAMP = _FINAL_RUN.strftime(_STAMP_FORMAT)
_GRANDFATHER_REF = 'grandfather_set_hash=' + 'f8c4' * 16
_ALPHA_REF = 'alpha=0.002777777777777778'

_TRIPWIRE = 'trip'
_WINDOW = 'prop'
_UNJUDGED = 'bare'

_METRIC_ROW_KEYS = frozenset({
    'metric_id', 'kind', 'rule_kind', 'current_value', 'n', 'denominator', 'direction', 'trend',
    'value', 'limit_ref', 'run_stamp', 'verdicts',
})
_QUIET_ITEM_KEYS = frozenset({'item', 'verdict', 'parity', 'escalation'})
_FULL_ENTRY_KEYS = _QUIET_ITEM_KEYS | {'verdict_detail', 'fingerprint'}


def _stamps(runs: int) -> list[str]:
    """*runs* daily stamps, oldest first, the newest always ``_FINAL_STAMP``."""
    return [(_FINAL_RUN - timedelta(days=age)).strftime(_STAMP_FORMAT) for age in reversed(range(runs))]


def _write_eval(root: Path, *, runs: int = 1, tripwire_value: float = 2.0) -> None:
    """``eval-a``: a tripwire, a proportion and an unjudged count, *runs* runs deep, with limits."""
    for stamp in _stamps(runs):
        dump_artifact(root / _EVAL / f'metrics-{stamp}.json', {
            'run_stamp': stamp,
            'metrics': [
                {'metric_id': _TRIPWIRE, 'kind': 'tripwire', 'value': tripwire_value, 'n': 8},
                {
                    'metric_id': _WINDOW, 'kind': 'proportion', 'value': 0.5, 'n': 15,
                    'denominator': 30, 'direction': 'lower_is_worse',
                },
                {'metric_id': _UNJUDGED, 'kind': 'count', 'value': 1.0, 'n': 3},
            ],
        })
    dump_artifact(root / _EVAL / 'limits-current.json', {
        'run_stamp': _FINAL_STAMP,
        'verdicts': [
            {'metric_id': _TRIPWIRE, 'rule_kind': 'tripwire'},
            {'metric_id': _WINDOW, 'rule_kind': 'proportion'},
            {'metric_id': _UNJUDGED, 'rule_kind': 'count'},
        ],
    })


def _item(item: str, verdict: Any, **overrides: Any) -> dict[str, Any]:
    """One tripwire item entry, shaped like ``shared/src/shared/memory_eval_verdicts.py::_tripwire_entries``."""
    return {
        'eval_id': _EVAL,
        'metric_id': _TRIPWIRE,
        'item': item,
        'verdict': verdict,
        'fingerprint': f'fp-{_TRIPWIRE}-{item}',
        'detail': f'Item {item!r} detail.',
        'value': 2.0,
        'limit_ref': _GRANDFATHER_REF,
        'run_stamp': _FINAL_STAMP,
        **overrides,
    }


def _window(verdict: str = 'no_alarm') -> dict[str, Any]:
    return {
        'eval_id': _EVAL,
        'metric_id': _WINDOW,
        'window': 'trailing-3',
        'verdict': verdict,
        'fingerprint': f'fp-{_WINDOW}',
        'detail': '12/30 against a pooled baseline of 72/90 over 3 run(s).',
        'value': 0.4,
        'limit_ref': _ALPHA_REF,
        'run_stamp': _FINAL_STAMP,
    }


def _write_verdicts(root: Path, entries: list[dict[str, Any]], *, storm: dict[str, Any] | None = None) -> None:
    body: dict[str, Any] = {'schema_version': 1, 'run_stamp': _FINAL_STAMP, 'entries': entries}
    if storm is not None:
        body['storm_escape'] = storm
    dump_artifact(root / 'verdicts-current.json', body)


def _build(tmp_path: Path) -> dict[str, Any]:
    queue = dataclasses.replace(
        reconciliation_queue(DashboardConfig(project_root=tmp_path, known_project_roots=[])),
        directory=_esc_dir(tmp_path),
    )
    return memory_evals.build_memory_evals(
        _root(tmp_path), measure_corpus((queue,), now=_NOW), queue.id, now=_NOW,
    )


def _root(tmp_path: Path) -> Path:
    return tmp_path / 'memory-evals'


def _esc_dir(tmp_path: Path) -> Path:
    esc_dir = tmp_path / 'escalations'
    esc_dir.mkdir(exist_ok=True)
    return esc_dir


def _metrics(payload: dict[str, Any]) -> list[dict[str, Any]]:
    (card,) = payload['evals']
    return card['metrics']


def _metric(payload: dict[str, Any], metric_id: str) -> dict[str, Any]:
    (row,) = [row for row in _metrics(payload) if row['metric_id'] == metric_id]
    return row


def _entry(payload: dict[str, Any], metric_id: str, item: str | None = None) -> dict[str, Any]:
    (entry,) = [entry for entry in _metric(payload, metric_id)['verdicts'] if entry['item'] == item]
    return entry


def _divergence_issues(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [issue for issue in payload['issues'] if issue['kind'] == 'divergent_verdict_field']


def _three_item_tree(tmp_path: Path) -> dict[str, Any]:
    root = _root(tmp_path)
    _write_eval(root)
    _write_verdicts(root, [
        _item('t-c', 'no_alarm'),
        _window(),
        _item('t-a', 'grandfathered'),
        _item('t-b', 'no_alarm'),
    ])
    return _build(tmp_path)


class TestOneRowPerMetric:
    def test_each_metric_is_one_row_holding_its_verdict_entries(self, tmp_path: Path) -> None:
        payload = _three_item_tree(tmp_path)

        assert [row['metric_id'] for row in _metrics(payload)] == [_UNJUDGED, _WINDOW, _TRIPWIRE]
        assert [entry['item'] for entry in _metric(payload, _TRIPWIRE)['verdicts']] == ['t-a', 't-b', 't-c']
        (window,) = _metric(payload, _WINDOW)['verdicts']
        assert (window['item'], window['verdict']) == (None, 'no_alarm')
        (unjudged,) = _metric(payload, _UNJUDGED)['verdicts']
        assert (unjudged['item'], unjudged['verdict'], unjudged['parity']) == (None, None, 'unjudged')

    def test_metric_rows_and_verdict_entries_split_the_fields(self, tmp_path: Path) -> None:
        payload = _three_item_tree(tmp_path)

        for row in _metrics(payload):
            assert set(row) == _METRIC_ROW_KEYS, row['metric_id']
            for entry in row['verdicts']:
                assert not set(entry) & (_METRIC_ROW_KEYS - {'verdicts'}), entry
        tripwire = _metric(payload, _TRIPWIRE)
        assert (tripwire['value'], tripwire['limit_ref'], tripwire['run_stamp']) == (
            2.0, _GRANDFATHER_REF, _FINAL_STAMP,
        )
        window = _metric(payload, _WINDOW)
        assert (window['value'], window['limit_ref'], window['run_stamp']) == (0.4, _ALPHA_REF, _FINAL_STAMP)
        unjudged = _metric(payload, _UNJUDGED)
        assert (unjudged['value'], unjudged['limit_ref'], unjudged['run_stamp']) == (None, None, None)

    def test_agreeing_entries_name_no_divergence(self, tmp_path: Path) -> None:
        payload = _three_item_tree(tmp_path)

        assert payload['issues'] == []


class TestQuietItemsAreTrimmed:
    @pytest.fixture
    def payload(self, tmp_path: Path) -> dict[str, Any]:
        root = _root(tmp_path)
        _write_eval(root)
        _write_verdicts(root, [
            _window(),
            _item('t-alarm-linked', 'alarm'),
            _item('t-alarm-unlinked', 'alarm'),
            _item('t-grandfathered', 'grandfathered'),
            _item('t-passes', 'no_alarm'),
            _item('t-recovered', 'no_alarm'),
            _item('t-unknown', 'improved'),
        ])
        esc_dir = _esc_dir(tmp_path)
        write_escalation_record(esc_dir, 'esc-alarm', dedupe_fingerprint=f'fp-{_TRIPWIRE}-t-alarm-linked')
        write_escalation_record(esc_dir, 'esc-recovered', dedupe_fingerprint=f'fp-{_TRIPWIRE}-t-recovered')
        return _build(tmp_path)

    @pytest.mark.parametrize('item', ['t-grandfathered', 't-passes'])
    def test_a_quiet_unlinked_item_ships_only_its_badge_fields(self, payload: dict[str, Any], item: str) -> None:
        entry = _entry(payload, _TRIPWIRE, item)

        assert set(entry) == _QUIET_ITEM_KEYS
        assert entry['escalation'] is None

    @pytest.mark.parametrize(('item', 'parity'), [
        ('t-alarm-unlinked', 'alarmed_unlinked'),
        ('t-alarm-linked', 'alarmed_open'),
        ('t-recovered', 'recovered_open'),
        ('t-unknown', 'unknown_verdict'),
    ])
    def test_an_item_worth_looking_up_keeps_its_detail_and_fingerprint(
        self, payload: dict[str, Any], item: str, parity: str,
    ) -> None:
        entry = _entry(payload, _TRIPWIRE, item)

        assert set(entry) == _FULL_ENTRY_KEYS
        assert entry['parity'] == parity
        assert entry['fingerprint'] == f'fp-{_TRIPWIRE}-{item}'
        assert entry['verdict_detail'] == f'Item {item!r} detail.'

    def test_a_window_entry_keeps_every_field_even_when_clear(self, payload: dict[str, Any]) -> None:
        entry = _entry(payload, _WINDOW)

        assert entry['parity'] == 'clear'
        assert set(entry) == _FULL_ENTRY_KEYS
        assert entry['verdict_detail'] == _window()['detail']


class TestTheJoinRunsPerEntry:
    def test_a_linked_item_carries_the_escalation_its_fingerprint_names(self, tmp_path: Path) -> None:
        root = _root(tmp_path)
        _write_eval(root)
        _write_verdicts(root, [_item('t-a', 'alarm'), _item('t-b', 'no_alarm')])
        write_escalation_record(_esc_dir(tmp_path), 'esc-a', dedupe_fingerprint=f'fp-{_TRIPWIRE}-t-a')

        entry = _entry(_build(tmp_path), _TRIPWIRE, 't-a')

        assert entry['parity'] == 'alarmed_open'
        assert entry['escalation']['id'] == 'esc-a'
        assert entry['escalation']['dedupe_fingerprint'] == entry['fingerprint']

    def test_a_triggered_storm_collapses_an_alarmed_item(self, tmp_path: Path) -> None:
        root = _root(tmp_path)
        _write_eval(root)
        _write_verdicts(
            root, [_item('t-a', 'alarm'), _item('t-b', 'no_alarm')],
            storm={'triggered': True, 'alarm_count': 1, 'aggregate_fingerprint': 'fp-aggregate'},
        )
        write_escalation_record(_esc_dir(tmp_path), 'esc-a', dedupe_fingerprint=f'fp-{_TRIPWIRE}-t-a')

        payload = _build(tmp_path)

        entry = _entry(payload, _TRIPWIRE, 't-a')
        assert (entry['parity'], entry['escalation']) == ('storm_collapsed', None)
        assert [(u['id'], u['reason']) for u in payload['unmatched_escalations']] == [
            ('esc-a', 'storm_suppressed'),
        ]


class TestDivergentMetricScopedFieldsAreNamed:
    @pytest.mark.parametrize(('field', 'first', 'other'), [
        ('limit_ref', _GRANDFATHER_REF, 'grandfather_set_hash=' + 'beef' * 16),
        ('value', 2.0, 3.0),
        ('run_stamp', _FINAL_STAMP, '20260704T031500Z'),
    ])
    def test_the_first_entry_wins_and_the_divergence_is_one_issue(
        self, tmp_path: Path, field: str, first: Any, other: Any,
    ) -> None:
        root = _root(tmp_path)
        _write_eval(root)
        _write_verdicts(root, [
            _item('t-c', 'no_alarm', **{field: other}),
            _item('t-a', 'no_alarm', **{field: first}),
            _item('t-b', 'no_alarm', **{field: other}),
        ])

        payload = _build(tmp_path)

        assert _metric(payload, _TRIPWIRE)[field] == first
        (issue,) = _divergence_issues(payload)
        assert (issue['eval_id'], issue['path']) == (_EVAL, str(root / 'verdicts-current.json'))
        assert repr(_TRIPWIRE) in issue['detail']
        assert repr(field) in issue['detail']
        assert '2 of 3' in issue['detail']
        assert issue['detail'].endswith('only the first is shown')

    def test_an_oversized_divergent_value_is_bounded_in_the_detail(self, tmp_path: Path) -> None:
        huge = 'h' * 5000
        root = _root(tmp_path)
        _write_eval(root)
        _write_verdicts(root, [_item('t-a', 'no_alarm', limit_ref=huge), _item('t-b', 'no_alarm')])

        (issue,) = _divergence_issues(_build(tmp_path))

        assert len(issue['detail']) < 1000
        assert huge not in issue['detail']


def _payload_bytes(payload: dict[str, Any]) -> int:
    """The size Starlette's JSONResponse serves."""
    return len(json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode())


class TestPayloadGrowsWithItemsNotItemsTimesRuns:
    def test_more_runs_cost_the_same_bytes_whatever_the_item_count(self, tmp_path: Path) -> None:
        def size(items: int, runs: int) -> int:
            base = tmp_path / f'{items}-items-{runs}-runs'
            _write_eval(_root(base), runs=runs, tripwire_value=0.0)
            _write_verdicts(_root(base), [
                _item(f't-{index:02d}', 'no_alarm', value=0.0) for index in range(items)
            ])
            return _payload_bytes(_build(base))

        assert size(40, 30) - size(40, 3) == size(1, 30) - size(1, 3)
