"""One eval's build inside the memory-evals payload: what its rows claim, and the order it names degradations (task 6629)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from _dashboard_helpers import dump_artifact, write_escalation_record

from dashboard.config import DashboardConfig
from dashboard.data import memory_evals
from dashboard.data.escalation_corpus import measure_corpus, reconciliation_queue

_RUN = '20260705T031500Z'
_NOW = datetime(2026, 7, 6, tzinfo=UTC)


def _metric(metric_id: str, kind: str = 'count') -> dict[str, Any]:
    return {'metric_id': metric_id, 'kind': kind, 'value': 1.0, 'n': 3}


def _write_eval(root: Path, eval_id: str, metrics: list[dict[str, Any]]) -> None:
    dump_artifact(root / eval_id / f'metrics-{_RUN}.json', {'run_stamp': _RUN, 'metrics': metrics})
    dump_artifact(root / eval_id / 'limits-current.json', {'run_stamp': _RUN, 'verdicts': []})


def _write_verdicts(root: Path, entries: list[dict[str, Any]]) -> None:
    dump_artifact(
        root / 'verdicts-current.json',
        {'schema_version': 1, 'run_stamp': _RUN, 'entries': entries},
    )


def _entry(eval_id: str, metric_id: str, verdict: str, fingerprint: str) -> dict[str, Any]:
    return {'eval_id': eval_id, 'metric_id': metric_id, 'verdict': verdict, 'fingerprint': fingerprint}


def _build(tmp_path: Path, root: Path, esc_dir: Path) -> dict[str, Any]:
    queue = dataclasses.replace(
        reconciliation_queue(DashboardConfig(project_root=tmp_path, known_project_roots=[])),
        directory=esc_dir,
    )
    return memory_evals.build_memory_evals(
        root, measure_corpus((queue,), now=_NOW), queue.id, now=_NOW,
    )


@pytest.fixture
def late_bug_in(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    """Arm a bug that fires after one eval's REAL build has run to completion."""

    def arm(eval_id: str) -> None:
        real = memory_evals._build_eval

        def _late(eval_dir: Path, *args: Any, **kwargs: Any) -> Any:
            built = real(eval_dir, *args, **kwargs)
            if eval_dir.name == eval_id:
                raise TypeError('late bug')
            return built

        monkeypatch.setattr(memory_evals, '_build_eval', _late)

    return arm


def _shared_fingerprint_tree(tmp_path: Path) -> tuple[Path, Path]:
    root, esc_dir = tmp_path / 'memory-evals', tmp_path / 'escalations'
    for eval_id in ('eval-a', 'eval-b'):
        _write_eval(root, eval_id, [_metric('m')])
    _write_verdicts(root, [
        _entry('eval-a', 'm', 'alarm', 'fp-shared'),
        _entry('eval-b', 'm', 'alarm', 'fp-shared'),
    ])
    write_escalation_record(esc_dir, 'esc-shared', dedupe_fingerprint='fp-shared')
    return root, esc_dir


def _single_alarm_tree(tmp_path: Path) -> tuple[Path, Path]:
    root, esc_dir = tmp_path / 'memory-evals', tmp_path / 'escalations'
    _write_eval(root, 'eval-a', [_metric('m')])
    _write_verdicts(root, [_entry('eval-a', 'm', 'alarm', 'fp-a')])
    write_escalation_record(esc_dir, 'esc-a', dedupe_fingerprint='fp-a')
    return root, esc_dir


def _only_row(payload: dict[str, Any], eval_id: str) -> dict[str, Any]:
    (row,) = next(ev for ev in payload['evals'] if ev['eval_id'] == eval_id)['metrics']
    return row


def _linked_id(row: dict[str, Any]) -> str | None:
    return row['escalation']['id'] if row['escalation'] else None


class TestRowClaimsShipWithTheirRow:
    def test_a_fingerprint_shared_across_evals_links_only_the_first_evals_row(
        self, tmp_path: Path,
    ) -> None:
        root, esc_dir = _shared_fingerprint_tree(tmp_path)

        payload = _build(tmp_path, root, esc_dir)

        first, second = _only_row(payload, 'eval-a'), _only_row(payload, 'eval-b')
        assert (first['parity'], _linked_id(first)) == ('alarmed_open', 'esc-shared')
        assert (second['parity'], _linked_id(second)) == ('alarmed_unlinked', None)
        assert payload['unmatched_escalations'] == []

    def test_an_eval_lost_to_a_bug_leaves_its_escalation_unmatched_not_vanished(
        self, tmp_path: Path, late_bug_in: Callable[[str], None],
    ) -> None:
        root, esc_dir = _single_alarm_tree(tmp_path)
        late_bug_in('eval-a')

        payload = _build(tmp_path, root, esc_dir)

        assert payload['evals'] == []
        assert [(u['id'], u['reason']) for u in payload['unmatched_escalations']] == [
            ('esc-a', 'no_matching_verdict'),
        ]

    def test_an_eval_lost_to_a_bug_cannot_take_a_link_from_a_shipped_row(
        self, tmp_path: Path, late_bug_in: Callable[[str], None],
    ) -> None:
        root, esc_dir = _shared_fingerprint_tree(tmp_path)
        late_bug_in('eval-a')

        payload = _build(tmp_path, root, esc_dir)

        assert [ev['eval_id'] for ev in payload['evals']] == ['eval-b']
        shipped = _only_row(payload, 'eval-b')
        assert (shipped['parity'], _linked_id(shipped)) == ('alarmed_open', 'esc-shared')
        assert payload['unmatched_escalations'] == []

    def test_an_eval_lost_to_a_bug_leaves_its_verdicts_orphaned(
        self, tmp_path: Path, late_bug_in: Callable[[str], None],
    ) -> None:
        root, esc_dir = _single_alarm_tree(tmp_path)
        late_bug_in('eval-a')

        payload = _build(tmp_path, root, esc_dir)

        assert [i['eval_id'] for i in payload['issues'] if i['kind'] == 'orphan_verdict'] == ['eval-a']

    def test_an_artifact_issue_named_before_the_bug_survives_it(
        self, tmp_path: Path, late_bug_in: Callable[[str], None],
    ) -> None:
        root, esc_dir = tmp_path / 'memory-evals', tmp_path / 'escalations'
        _write_eval(root, 'eval-a', [_metric('m', kind='histogram')])
        _write_verdicts(root, [])
        esc_dir.mkdir()
        late_bug_in('eval-a')

        payload = _build(tmp_path, root, esc_dir)

        assert [(i['kind'], i['eval_id']) for i in payload['issues']] == [
            ('unknown_kind', 'eval-a'),
            ('internal_error', 'eval-a'),
        ]


class TestIssueOrder:
    def test_one_evals_issues_are_named_in_build_order(self, tmp_path: Path) -> None:
        root, esc_dir = tmp_path / 'memory-evals', tmp_path / 'escalations'
        eval_dir = root / 'eval-a'
        eval_dir.mkdir(parents=True)
        (eval_dir / 'metrics-20260703T031500Z.json').write_text('{"metrics": [ not json')
        dump_artifact(eval_dir / 'metrics-20260704T031500Z.json', {
            'run_stamp': '20260704T031500Z',
            'metrics': [_metric('m-a'), _metric('m-a'), _metric('m-kind', kind='histogram')],
        })
        dump_artifact(eval_dir / 'metrics-20260705T031500Z.json', {
            'run_stamp': 'not-a-stamp',
            'metrics': [_metric('m-verdict'), _metric('m-z', kind='histogram2')],
        })
        _write_verdicts(root, [
            {'eval_id': 'eval-a', 'metric_id': 'm-verdict', 'verdict': 'bogus', 'fingerprint': 'fp-x'},
        ])
        esc_dir.mkdir()

        payload = _build(tmp_path, root, esc_dir)

        assert [i['kind'] for i in payload['issues']] == [
            'unreadable_metrics',
            'duplicate_metric_id',
            'missing_limits',
            'unparseable_run_stamp',
            'unknown_kind',
            'unknown_verdict',
            'unknown_kind',
        ]
