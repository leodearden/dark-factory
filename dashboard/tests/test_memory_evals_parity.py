"""Memory-eval parity gate: verdict rows and escalations agree, both ways (task 3217).

Every scenario is served through ``GET /api/v2/dashboard/memory-evals`` over a
tmp project root holding 3207's committed ``e1-retrieval-health`` metrics
series (copied, real bytes) plus a root verdicts artifact: either one of
3211's committed exemplars, copied verbatim, or a hand-seeded body naming the
same real metrics.  Escalations are written into the queue the route resolves
from config.

The join under test is whole-string equality between a verdict entry's
``fingerprint`` and an open ``eval_regression`` escalation's
``dedupe_fingerprint``.  The fingerprint identity is
``(eval_id, metric_id, item | window)``
(``shared/tests/fixtures/memory_eval_verdicts/README.md`` §"Entry granularity"),
so a tripwire contributes one row per item.

Artifact-only, like ``test_memory_evals_data.py``: exemplar fingerprints are
read with ``json.loads``, never through ``shared.memory_eval_*``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, TypeGuard

import pytest
from _dashboard_helpers import dump_artifact, write_escalation_record

from dashboard.config import DashboardConfig

_FIXTURES = Path(__file__).parents[2] / 'shared' / 'tests' / 'fixtures'
_METRICS_EVAL = _FIXTURES / 'memory_eval' / 'e1-retrieval-health'
_VERDICT_EXEMPLARS = _FIXTURES / 'memory_eval_verdicts'
_FILED = _VERDICT_EXEMPLARS / 'filed' / 'verdicts-current.json'
_STORM = _VERDICT_EXEMPLARS / 'storm' / 'verdicts-current.json'

_ROUTE_URL = '/api/v2/dashboard/memory-evals'
_EVAL_ID = _METRICS_EVAL.name
_RUN_STAMP = '20260704T031500Z'
_TRIPWIRE = 'topic-canonical-present'
_EARLIER_RUN_TIMESTAMP = '2026-07-03T03:15:00+00:00'


def _tree(tmp_path: Path, verdicts: Path | dict[str, Any]) -> DashboardConfig:
    """A project root holding the committed metrics series and *verdicts*."""
    config = DashboardConfig(project_root=tmp_path, known_project_roots=[])
    root = config.memory_evals_dir
    shutil.copytree(_METRICS_EVAL, root / _EVAL_ID)
    if isinstance(verdicts, Path):
        shutil.copy2(verdicts, root / 'verdicts-current.json')
    else:
        dump_artifact(root / 'verdicts-current.json', verdicts)
    esc_dir = config.reconciliation_escalations_dir
    assert esc_dir.is_relative_to(tmp_path), (
        f'escalation queue {esc_dir} resolved outside {tmp_path}; '
        'RECONCILIATION_DATA_DIR is set, so writing here would hit a live queue'
    )
    esc_dir.mkdir(parents=True, exist_ok=True)
    return config


def _serve(client, config: DashboardConfig) -> dict[str, Any]:
    client.app.state.config = config
    resp = client.get(_ROUTE_URL)
    assert resp.status_code == 200, resp.text
    return resp.json()['MEMORY_EVALS']


def _file(config: DashboardConfig, esc_id: str, fingerprint: str, **kw: Any) -> None:
    write_escalation_record(
        config.reconciliation_escalations_dir, esc_id, dedupe_fingerprint=fingerprint, **kw,
    )


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for card in payload['evals'] for row in card['metrics']]


def _row(payload: dict[str, Any], metric_id: str, item: str | None = None) -> dict[str, Any]:
    matches = [r for r in _rows(payload) if r['metric_id'] == metric_id and r['item'] == item]
    assert len(matches) == 1, (
        f'expected exactly one row for ({metric_id!r}, {item!r}), got {len(matches)}: '
        f'{[(r["metric_id"], r["item"]) for r in _rows(payload)]}'
    )
    return matches[0]


def _kinds(payload: dict[str, Any]) -> list[str]:
    return [issue['kind'] for issue in payload['issues']]


def _unmatched(payload: dict[str, Any]) -> list[tuple[str, str]]:
    return [(u['id'], u['reason']) for u in payload['unmatched_escalations']]


def _assert_both_directions(payload: dict[str, Any], filed: dict[str, str], *, storm: bool) -> None:
    """Parity in both directions over the pending escalations in *filed* (id -> fingerprint).

    FORWARD: an alarm row whose fingerprint equals a filed escalation's carries
    that escalation (or, under a storm, no link and ``storm_collapsed``); any
    row that does carry a link carries the one whose fingerprint equals its own.
    REVERSE: each filed id is reachable exactly once across the three places
    the payload can put it, and outside a storm that place is a metric row.
    """
    by_fingerprint = {fingerprint: esc_id for esc_id, fingerprint in filed.items()}
    rows = _rows(payload)
    for row in rows:
        if row['escalation'] is not None:
            assert row['escalation']['dedupe_fingerprint'] == row['fingerprint'], row
        if row['verdict'] != 'alarm' or row['fingerprint'] not in by_fingerprint:
            continue
        if storm:
            assert row['escalation'] is None, row
            assert row['parity'] == 'storm_collapsed', row
        else:
            assert row['escalation']['id'] == by_fingerprint[row['fingerprint']], row
            assert row['parity'] == 'alarmed_open', row

    homes: dict[str, list[str]] = {esc_id: [] for esc_id in filed}
    for row in rows:
        if row['escalation'] is not None:
            homes.setdefault(row['escalation']['id'], []).append('row')
    storm_link = (payload['storm_escape'] or {}).get('escalation')
    if storm_link is not None:
        homes.setdefault(storm_link['id'], []).append('storm_escape')
    for unmatched in payload['unmatched_escalations']:
        homes.setdefault(unmatched['id'], []).append('unmatched')
    assert {esc_id: len(places) for esc_id, places in homes.items()} == dict.fromkeys(filed, 1), homes
    if not storm:
        assert {esc_id: places[0] for esc_id, places in homes.items()} == dict.fromkeys(filed, 'row'), homes


def _exemplar_entries(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())['entries']


def _exemplar_fingerprint(entries: list[dict[str, Any]], metric_id: str, item: str | None = None) -> str:
    (fingerprint,) = [
        e['fingerprint'] for e in entries if e['metric_id'] == metric_id and e.get('item') == item
    ]
    return fingerprint


def _opaque(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _entry(metric_id: str, verdict: str, fingerprint: str, *, item: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        'eval_id': _EVAL_ID,
        'metric_id': metric_id,
        'verdict': verdict,
        'fingerprint': fingerprint,
        'value': 1.0,
        'limit_ref': 'hand-seeded',
        'run_stamp': _RUN_STAMP,
        'detail': f'{verdict} for {metric_id}',
    }
    if item is not None:
        entry['item'] = item
    return entry


def _verdicts(*entries: dict[str, Any], storm_escape: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        'schema_version': 1,
        'generator': 'shared.memory_eval_verdicts',
        'run_stamp': _RUN_STAMP,
        'entries': list(entries),
        'runner_failures': [],
        'storm_escape': storm_escape or {'triggered': False, 'alarm_count': 0, 'alarm_limit': 3},
    }


def _triggered_storm(alarm_count: int) -> dict[str, Any]:
    return {
        'triggered': True,
        'alarm_count': alarm_count,
        'alarm_limit': 1,
        'aggregate_fingerprint': _opaque('storm-aggregate'),
    }


class TestCommittedExemplarParity:
    """3211's committed exemplars, read through the route over 3207's committed series."""

    def test_filed_run_links_every_alarm_and_every_open_escalation(self, client, tmp_path: Path) -> None:
        config = _tree(tmp_path, _FILED)
        entries = _exemplar_entries(_FILED)
        filed = {
            'esc-canonical': _exemplar_fingerprint(entries, 'canonical-in-top-5'),
            'esc-dangling': _exemplar_fingerprint(entries, 'dangling-pointers'),
            'esc-worktree': _exemplar_fingerprint(entries, _TRIPWIRE, 't-worktree-lifecycle'),
            'esc-routing-recovered': _exemplar_fingerprint(entries, _TRIPWIRE, 't-routing-ladder'),
        }
        for esc_id, fingerprint in filed.items():
            _file(config, esc_id, fingerprint)

        payload = _serve(client, config)

        assert payload['issues'] == []
        assert sorted((r['metric_id'], r['item'] or '') for r in _rows(payload)) == sorted(
            (e['metric_id'], e.get('item') or '') for e in entries
        )
        for metric_id, item, esc_id in (
            ('canonical-in-top-5', None, 'esc-canonical'),
            ('dangling-pointers', None, 'esc-dangling'),
            (_TRIPWIRE, 't-worktree-lifecycle', 'esc-worktree'),
        ):
            row = _row(payload, metric_id, item)
            assert (row['verdict'], row['parity'], row['escalation']['id']) == ('alarm', 'alarmed_open', esc_id)
        recovered = _row(payload, _TRIPWIRE, 't-routing-ladder')
        assert (recovered['verdict'], recovered['parity']) == ('no_alarm', 'recovered_open')
        assert recovered['escalation']['id'] == 'esc-routing-recovered'
        grandfathered = _row(payload, _TRIPWIRE, 't-recon-watcher-triage')
        assert (grandfathered['parity'], grandfathered['escalation']) == ('grandfathered', None)
        assert [r['item'] for r in _rows(payload) if r['metric_id'] == _TRIPWIRE] == sorted(
            e['item'] for e in entries if e['metric_id'] == _TRIPWIRE
        )
        assert len([e for e in entries if e['metric_id'] == _TRIPWIRE]) == 8
        assert payload['unmatched_escalations'] == []
        _assert_both_directions(payload, filed, storm=False)

    def test_storm_run_links_only_the_aggregate(self, client, tmp_path: Path) -> None:
        config = _tree(tmp_path, _STORM)
        body = json.loads(_STORM.read_text())
        filed = {
            'esc-storm-aggregate': body['storm_escape']['aggregate_fingerprint'],
            'esc-earlier-dangling': _exemplar_fingerprint(body['entries'], 'dangling-pointers'),
        }
        _file(config, 'esc-storm-aggregate', filed['esc-storm-aggregate'])
        _file(config, 'esc-earlier-dangling', filed['esc-earlier-dangling'], timestamp=_EARLIER_RUN_TIMESTAMP)

        payload = _serve(client, config)

        assert payload['issues'] == []
        assert payload['storm_escape']['escalation']['id'] == 'esc-storm-aggregate'
        assert payload['storm_escape']['alarm_count'] == 3
        alarm_rows = [r for r in _rows(payload) if r['verdict'] == 'alarm']
        assert sorted((r['metric_id'], r['item'] or '') for r in alarm_rows) == [
            ('canonical-in-top-5', ''), ('dangling-pointers', ''), (_TRIPWIRE, 't-worktree-lifecycle'),
        ]
        assert {(r['parity'], r['escalation']) for r in alarm_rows} == {('storm_collapsed', None)}
        assert [r for r in _rows(payload) if r['escalation'] is not None] == []
        assert _unmatched(payload) == [('esc-earlier-dangling', 'storm_suppressed')]
        assert _row(payload, _TRIPWIRE, 't-recon-watcher-triage')['parity'] == 'grandfathered'
        _assert_both_directions(payload, filed, storm=True)


class TestHandSeededParityStates:
    """Each parity state, from a hand-written verdicts body over the committed series."""

    def test_alarmed_open_links_a_window_metric_and_each_tripwire_item(self, client, tmp_path: Path) -> None:
        filed = {
            'esc-window': _opaque('alarm window'),
            'esc-item-alpha': _opaque('alarm t-alpha'),
            'esc-item-beta': _opaque('alarm t-beta'),
        }
        config = _tree(tmp_path, _verdicts(
            _entry('canonical-in-top-5', 'alarm', filed['esc-window']),
            _entry(_TRIPWIRE, 'alarm', filed['esc-item-alpha'], item='t-alpha'),
            _entry(_TRIPWIRE, 'alarm', filed['esc-item-beta'], item='t-beta'),
        ))
        for esc_id, fingerprint in filed.items():
            _file(config, esc_id, fingerprint)

        payload = _serve(client, config)

        assert payload['issues'] == []
        for (metric_id, item), esc_id in (
            (('canonical-in-top-5', None), 'esc-window'),
            ((_TRIPWIRE, 't-alpha'), 'esc-item-alpha'),
            ((_TRIPWIRE, 't-beta'), 'esc-item-beta'),
        ):
            row = _row(payload, metric_id, item)
            assert (row['parity'], row['escalation']['id']) == ('alarmed_open', esc_id)
        _assert_both_directions(payload, filed, storm=False)

    def test_recovered_open_keeps_its_link_and_is_never_alarm_shaped(self, client, tmp_path: Path) -> None:
        filed = {
            'esc-window-recovered': _opaque('recovered window'),
            'esc-item-recovered': _opaque('recovered t-routing-ladder'),
        }
        config = _tree(tmp_path, _verdicts(
            _entry('dangling-pointers', 'no_alarm', filed['esc-window-recovered']),
            _entry(_TRIPWIRE, 'no_alarm', filed['esc-item-recovered'], item='t-routing-ladder'),
        ))
        for esc_id, fingerprint in filed.items():
            _file(config, esc_id, fingerprint)

        payload = _serve(client, config)

        assert payload['issues'] == []
        for (metric_id, item), esc_id in (
            (('dangling-pointers', None), 'esc-window-recovered'),
            ((_TRIPWIRE, 't-routing-ladder'), 'esc-item-recovered'),
        ):
            row = _row(payload, metric_id, item)
            assert (row['verdict'], row['parity'], row['escalation']['id']) == (
                'no_alarm', 'recovered_open', esc_id,
            )
        assert not [r for r in _rows(payload) if r['parity'].startswith('alarmed_')]
        _assert_both_directions(payload, filed, storm=False)

    def test_storm_collapses_per_metric_links_into_the_aggregate(self, client, tmp_path: Path) -> None:
        storm = _triggered_storm(alarm_count=2)
        filed = {'esc-aggregate': storm['aggregate_fingerprint']}
        config = _tree(tmp_path, _verdicts(
            _entry('canonical-in-top-5', 'alarm', _opaque('storm window')),
            _entry(_TRIPWIRE, 'alarm', _opaque('storm t-alpha'), item='t-alpha'),
            storm_escape=storm,
        ))
        _file(config, 'esc-aggregate', filed['esc-aggregate'])

        payload = _serve(client, config)

        assert payload['issues'] == []
        assert payload['storm_escape']['escalation']['id'] == 'esc-aggregate'
        for metric_id, item in (('canonical-in-top-5', None), (_TRIPWIRE, 't-alpha')):
            row = _row(payload, metric_id, item)
            assert (row['parity'], row['escalation']) == ('storm_collapsed', None)
        assert [r for r in _rows(payload) if r['escalation'] is not None] == []
        assert payload['unmatched_escalations'] == []
        _assert_both_directions(payload, filed, storm=True)

    @pytest.mark.parametrize('storm', [False, True], ids=['no-storm', 'storm'])
    def test_grandfathered_and_insufficient_data_never_render_alarm_shaped(
        self, client, tmp_path: Path, storm: bool,
    ) -> None:
        open_fingerprints = {
            'esc-grandfathered': _opaque('grandfathered t-known-bad'),
            'esc-insufficient': _opaque('insufficient superseded'),
        }
        storm_escape = _triggered_storm(alarm_count=1) if storm else None
        filed = dict(open_fingerprints)
        if storm_escape is not None:
            filed['esc-aggregate'] = storm_escape['aggregate_fingerprint']
        config = _tree(tmp_path, _verdicts(
            _entry(_TRIPWIRE, 'grandfathered', open_fingerprints['esc-grandfathered'], item='t-known-bad'),
            _entry(_TRIPWIRE, 'grandfathered', _opaque('grandfathered t-quiet'), item='t-quiet'),
            _entry('superseded-above-successor', 'insufficient_data', open_fingerprints['esc-insufficient']),
            _entry('search-latency-p50-ms', 'insufficient_data', _opaque('insufficient latency')),
            storm_escape=storm_escape,
        ))
        for esc_id, fingerprint in filed.items():
            _file(config, esc_id, fingerprint)

        payload = _serve(client, config)

        assert payload['issues'] == []
        parities = {
            'grandfathered-open': _row(payload, _TRIPWIRE, 't-known-bad')['parity'],
            'grandfathered': _row(payload, _TRIPWIRE, 't-quiet')['parity'],
            'insufficient-open': _row(payload, 'superseded-above-successor')['parity'],
            'insufficient': _row(payload, 'search-latency-p50-ms')['parity'],
        }
        if storm:
            assert parities == {
                'grandfathered-open': 'grandfathered', 'grandfathered': 'grandfathered',
                'insufficient-open': 'insufficient_data', 'insufficient': 'insufficient_data',
            }
            assert sorted(_unmatched(payload)) == [
                ('esc-grandfathered', 'storm_suppressed'), ('esc-insufficient', 'storm_suppressed'),
            ]
        else:
            assert parities == {
                'grandfathered-open': 'grandfathered_open', 'grandfathered': 'grandfathered',
                'insufficient-open': 'insufficient_data_open', 'insufficient': 'insufficient_data',
            }
        assert not set(parities.values()) & {'alarmed_open', 'alarmed_unlinked', 'storm_collapsed'}
        _assert_both_directions(payload, filed, storm=storm)


class TestTripwireItemsAreRows:
    """A tripwire's per-item entries are keyed ``(eval_id, metric_id, item)``."""

    def test_a_repeated_item_is_named_and_the_first_entry_wins(self, client, tmp_path: Path) -> None:
        first = _opaque('first t-dup')
        config = _tree(tmp_path, _verdicts(
            _entry(_TRIPWIRE, 'alarm', first, item='t-dup'),
            _entry(_TRIPWIRE, 'no_alarm', _opaque('second t-dup'), item='t-dup'),
        ))

        payload = _serve(client, config)

        (issue,) = payload['issues']
        assert issue['kind'] == 'duplicate_verdict_entry'
        assert repr(_TRIPWIRE) in issue['detail'] and repr('t-dup') in issue['detail']
        row = _row(payload, _TRIPWIRE, 't-dup')
        assert (row['verdict'], row['fingerprint']) == ('alarm', first)

    def test_distinct_items_are_distinct_rows_ordered_after_the_itemless_row(
        self, client, tmp_path: Path,
    ) -> None:
        config = _tree(tmp_path, _verdicts(
            _entry(_TRIPWIRE, 'no_alarm', _opaque('t-b'), item='t-b'),
            _entry(_TRIPWIRE, 'no_alarm', _opaque('whole tripwire')),
            _entry(_TRIPWIRE, 'alarm', _opaque('t-a'), item='t-a'),
        ))

        payload = _serve(client, config)

        assert payload['issues'] == []
        assert [r['item'] for r in _rows(payload) if r['metric_id'] == _TRIPWIRE] == [None, 't-a', 't-b']
        metric_ids = [r['metric_id'] for r in _rows(payload)]
        assert metric_ids == sorted(metric_ids)

    def test_an_unknown_verdict_on_each_item_names_its_own_item(self, client, tmp_path: Path) -> None:
        config = _tree(tmp_path, _verdicts(
            _entry(_TRIPWIRE, 'weird', _opaque('weird t-a'), item='t-a'),
            _entry(_TRIPWIRE, 'weird', _opaque('weird t-b'), item='t-b'),
        ))

        payload = _serve(client, config)

        details = [issue['detail'] for issue in payload['issues'] if issue['kind'] == 'unknown_verdict']
        assert len(details) == len(set(details)) == 2
        for detail, item in zip(sorted(details), ('t-a', 't-b'), strict=True):
            assert repr(_TRIPWIRE) in detail and repr(item) in detail

    def test_an_unkeyable_item_is_counted_and_its_siblings_still_render(self, client, tmp_path: Path) -> None:
        unkeyable = [_entry(_TRIPWIRE, 'alarm', _opaque(f'bad item {n}')) for n in range(3)]
        for entry, item in zip(unkeyable, ([], 5, ''), strict=True):
            entry['item'] = item
        config = _tree(tmp_path, _verdicts(
            *unkeyable,
            _entry(_TRIPWIRE, 'alarm', _opaque('t-ok'), item='t-ok'),
        ))

        payload = _serve(client, config)

        (issue,) = payload['issues']
        assert issue['kind'] == 'unidentified_verdicts'
        assert issue['detail'].startswith('3 ')
        assert '"item"' in issue['detail']
        assert [r['item'] for r in _rows(payload) if r['metric_id'] == _TRIPWIRE] == ['t-ok']

    def test_an_orphan_tripwire_is_one_issue_for_the_metric(self, client, tmp_path: Path) -> None:
        config = _tree(tmp_path, _verdicts(
            _entry('no-such-tripwire', 'alarm', _opaque('orphan t-a'), item='t-a'),
            _entry('no-such-tripwire', 'alarm', _opaque('orphan t-b'), item='t-b'),
        ))

        payload = _serve(client, config)

        (issue,) = payload['issues']
        assert issue['kind'] == 'orphan_verdict'
        assert repr('no-such-tripwire') in issue['detail']


_STRING_SURGERY_METHODS = frozenset({
    'split', 'rsplit', 'partition', 'rpartition', 'startswith', 'endswith',
    'removeprefix', 'removesuffix', 'find', 'rfind', 'index', 'rindex',
})


def _fingerprint_string_surgery(source: str) -> list[str]:
    """Code in *source* that takes a fingerprint apart as text, by string method, slice or ``re`` call.

    Name-keyed: an expression counts as fingerprint-bearing only when its text
    says ``fingerprint``.  ``TestFingerprintJoinIsStringEquality``'s near-miss
    cases are the behavioural gate; this scan only backs them up.
    """
    return [
        ast.unparse(node)
        for node in ast.walk(ast.parse(source))
        if any('fingerprint' in ast.unparse(operand) for operand in _text_operands(node))
    ]


def _text_operands(node: ast.AST) -> list[ast.AST]:
    """What *node* reads as text: a string method's receiver, a slice's target, or a ``re.*`` call's arguments."""
    if isinstance(node, ast.Attribute) and node.attr in _STRING_SURGERY_METHODS:
        return [node.value]
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
        return [node.value]
    if _calls_re(node):
        return [*node.args, *node.keywords]
    return []


def _calls_re(node: ast.AST) -> TypeGuard[ast.Call]:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == 're'
    )


class TestFingerprintJoinIsStringEquality:
    """The join is ``==`` on the whole fingerprint string; nothing reads inside it."""

    _BASE = _opaque('near-miss')

    @pytest.mark.parametrize(
        ('verdict_fingerprint', 'escalation_fingerprint'),
        [
            (_BASE, _BASE.upper()),
            (_BASE, _BASE + ' '),
            (_BASE, _BASE[:16]),
            (_BASE[:16], _BASE),
        ],
        ids=['case', 'trailing-whitespace', 'escalation-is-a-prefix', 'verdict-is-a-prefix'],
    )
    def test_a_near_miss_fingerprint_does_not_link(
        self, client, tmp_path: Path, verdict_fingerprint: str, escalation_fingerprint: str,
    ) -> None:
        config = _tree(tmp_path, _verdicts(_entry('dangling-pointers', 'alarm', verdict_fingerprint)))
        _file(config, 'esc-near-miss', escalation_fingerprint)

        payload = _serve(client, config)

        row = _row(payload, 'dangling-pointers')
        assert (row['parity'], row['escalation']) == ('alarmed_unlinked', None)
        assert _unmatched(payload) == [('esc-near-miss', 'no_matching_verdict')]

    def test_a_structured_looking_fingerprint_links_by_its_entry_not_its_text(
        self, client, tmp_path: Path,
    ) -> None:
        misleading = f'eval:{_EVAL_ID}|metric:dangling-pointers|item:x'
        config = _tree(tmp_path, _verdicts(
            _entry('canonical-in-top-5', 'alarm', misleading),
            _entry('dangling-pointers', 'alarm', _opaque('dangling')),
        ))
        _file(config, 'esc-misleading', misleading)

        payload = _serve(client, config)

        assert _row(payload, 'canonical-in-top-5')['escalation']['id'] == 'esc-misleading'
        assert _row(payload, 'dangling-pointers')['parity'] == 'alarmed_unlinked'
        assert payload['unmatched_escalations'] == []

    def test_the_reader_never_takes_a_fingerprint_apart(self) -> None:
        import dashboard.data.memory_evals as memory_evals_module

        source = Path(memory_evals_module.__file__).read_text()

        assert _fingerprint_string_surgery(source) == []

    @pytest.mark.parametrize(
        'injected',
        [
            "x = fingerprint.split('|')",
            'x = fingerprint[:8]',
            "x = re.split('[|]', fingerprint)",
            "x = re.fullmatch(pattern='[0-9a-f]+', string=entry['fingerprint'])",
        ],
    )
    def test_the_guard_flags_injected_string_surgery(self, injected: str) -> None:
        import dashboard.data.memory_evals as memory_evals_module

        source = Path(memory_evals_module.__file__).read_text()

        assert _fingerprint_string_surgery(f'{source}\n{injected}\n') != []

    @pytest.mark.parametrize(
        'injected',
        ['import re', "x = re.fullmatch('[0-9]{8}T[0-9]{6}Z', run_stamp)"],
    )
    def test_the_guard_ignores_a_regex_over_other_text(self, injected: str) -> None:
        assert _fingerprint_string_surgery(injected) == []

    def test_two_entries_sharing_a_fingerprint_are_named_and_link_one_row(
        self, client, tmp_path: Path,
    ) -> None:
        shared = _opaque('shared by t-a and t-b')
        config = _tree(tmp_path, _verdicts(
            _entry(_TRIPWIRE, 'alarm', shared, item='t-b'),
            _entry(_TRIPWIRE, 'alarm', shared, item='t-a'),
        ))
        _file(config, 'esc-shared', shared)

        payload = _serve(client, config)

        (issue,) = payload['issues']
        assert issue['kind'] == 'duplicate_verdict_fingerprint'
        assert repr('t-a') in issue['detail'] and repr('t-b') in issue['detail']
        linked = [(r['item'], r['escalation']['id']) for r in _rows(payload) if r['escalation'] is not None]
        assert linked == [('t-a', 'esc-shared')]
        assert _row(payload, _TRIPWIRE, 't-b')['parity'] == 'alarmed_unlinked'
        assert payload['unmatched_escalations'] == []
