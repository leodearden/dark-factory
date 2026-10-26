"""Tests for run_write_triage_config_matrix.py — μ, the C2'' configuration matrix.

μ scores every π arm with ι (``score_write_triage_pairs.py``) against λ's
verdict corpus, checks each arm against Γ3's bounds with Γ3's own predicate
(``scripts/check_write_triage_readiness_gate.py::check_require``), and names
the winner whose doc becomes ``write_triage_best_config.json``.

ι and the gate are the oracles here: every "copied verbatim" assertion is
checked against ι's own ``score_pairs`` output, and every bound assertion
against the gate's own ``check_require``. All three scripts are loaded by path
(``scripts/`` is not a package), lazily, exactly as
``test_score_write_triage_pairs.py`` does.
"""
from __future__ import annotations

import functools
import hashlib
import json
import types
from datetime import UTC, datetime, timedelta
from itertools import chain
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

PACKAGE = Path(__file__).resolve().parent.parent
SCRIPTS = PACKAGE / 'scripts'
CALIBRATION = PACKAGE / 'calibration'
SCRIPT_PATH = SCRIPTS / 'run_write_triage_config_matrix.py'
IOTA_PATH = SCRIPTS / 'score_write_triage_pairs.py'
GATE_PATH = Path(__file__).resolve().parents[2] / 'scripts' / 'check_write_triage_readiness_gate.py'

SNAPSHOT = 'snap'
SNAPSHOT_PATH = 'data/x/snapshot.json'
REFERENCE = 'gpt-4o-mini@5'
PRE_PSI = 'gpt-4o-mini@5+pre-psi'
SOL = 'gpt-6.1-sol:low@5'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(SCRIPT_PATH, mod_name='run_write_triage_config_matrix')


@functools.cache
def _mod_iota() -> types.ModuleType:
    return load_script_module(IOTA_PATH, mod_name='score_write_triage_pairs')


@functools.cache
def _mod_gate() -> types.ModuleType:
    return load_script_module(GATE_PATH, mod_name='check_write_triage_readiness_gate')


# --- synthetic fixtures ---------------------------------------------------------

def _row(
    arm: str,
    memory_id: str,
    outcome: str,
    judged: str | None = None,
    winner: str = 'w-parent',
    seconds: float | None = 1.0,
    model: str = 'gpt-4o-mini',
    prompt: int = 100,
    completion: int = 10,
    snapshot: str = SNAPSHOT,
    recon_marker: bool = False,
    declares_attach_keys: bool = False,
    slates: str = 'write-time',
) -> dict[str, Any]:
    """One ι-contract judge-band case row, as π2 writes it on write-time slates."""
    return {
        'arm': arm, 'memory_id': memory_id, 'band': 'judge', 'outcome': outcome,
        'judged_candidate_id': judged, 'band_winner_id': winner,
        'parse_failure': False, 'judge_seconds': seconds, 'judge_model': model,
        'usage': {'prompt_tokens': prompt, 'completion_tokens': completion},
        'snapshot_sha256': snapshot, 'recon_marker': recon_marker,
        'declares_attach_keys': declares_attach_keys, 'slates': slates,
    }


def _provenance(
    name: str,
    model: str,
    effort: str | None,
    width: int,
    wording: str,
    field_chars: int = 4000,
    provider: str = 'openai',
    cases_sha256: str = '0' * 64,
) -> dict[str, Any]:
    """One π provenance row, by the keys ``run_write_triage_population_arms.py::_arm_row`` writes."""
    return {
        'arm': name, 'model': model, 'provider': provider, 'reasoning_effort': effort,
        'width': width, 'wording': wording, 'field_chars': field_chars,
        'cases_path': f'data/x/arms/{name}.jsonl', 'cases_sha256': cases_sha256,
    }


def _population(
    arms: list[dict[str, Any]], n_judge_band: int, snapshot: str = SNAPSHOT,
) -> dict[str, Any]:
    """A doc shaped like π2's ``calibration/write_triage_population_write_time.json``."""
    return {
        'population': {
            'slates': 'write-time',
            'n_writes': n_judge_band + 3,
            'n_judge_band_write_time': n_judge_band,
            'projects': ['dark_factory', 'reify'],
            'frozen_at': '2026-10-05T10:01:34+00:00',
            'snapshot_sha256': snapshot,
            'snapshot_path': SNAPSHOT_PATH,
            't_high': 0.8868,
            't_low': 0.5230,
        },
        'arms': arms,
    }


def _c2_block(doc: dict[str, Any]) -> dict[str, Any]:
    """The C2 population block μ carries for a write-time population *doc*."""
    block = doc['population']
    return {
        'n_writes': block['n_writes'],
        'n_judge_band': block['n_judge_band_write_time'],
        'projects': block['projects'],
        'frozen_at': block['frozen_at'],
        'snapshot_sha256': block['snapshot_sha256'],
    }


def _vote(entry: str, target: str, verdict: str) -> dict[str, Any]:
    """A λ-style majority row."""
    return {
        'entry_id': entry, 'target_id': target, 'verdict': verdict,
        'rater': 'majority', 'batch': 'lambda-test',
    }


THREE_ARMS = [
    _provenance(REFERENCE, 'gpt-4o-mini', None, 5, 'shipped'),
    _provenance(PRE_PSI, 'gpt-4o-mini', None, 5, 'pre-psi'),
    _provenance(SOL, 'gpt-6.1-sol', 'low', 5, 'shipped'),
]

VERDICTS = [
    _vote('w1', 't1', 'SAME'),
    _vote('w2', 't2', 'CORRECTS'),
    _vote('w3', 't3', 'UNRELATED'),
    _vote('w3', 't3b', 'SAME'),
    _vote('w4', 't4', 'EXTENDS'),
]

T0 = datetime(2026, 10, 1, 12, tzinfo=UTC)
SLATE_OF = {'w1': ('t1',), 'w2': ('t2',), 'w3': ('t3', 't3b'), 'w4': ('t4',)}
"""Each write's write-time slate: every target it was rated against, all created before it."""
CREATED_AT: dict[str, datetime] = {
    **dict.fromkeys(SLATE_OF, T0),
    **{target: T0 - timedelta(days=1) for slate in SLATE_OF.values() for target in slate},
}


def _snapshot_doc() -> dict[str, Any]:
    """A frozen snapshot whose writes and slate candidates carry :data:`CREATED_AT`."""
    return {
        'writes': [
            {
                'memory_id': write, 'created_at': CREATED_AT[write].isoformat(),
                'candidates': [
                    {'memory_id': target, 'created_at': CREATED_AT[target].isoformat()}
                    for target in slate
                ],
            }
            for write, slate in SLATE_OF.items()
        ],
        'targets': {},
    }


def _write_snapshot(root: Path, doc: dict[str, Any] | None = None) -> str:
    """Write *doc* and its sidecar where the population's snapshot_path points; its digest."""
    path = root / SNAPSHOT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(_snapshot_doc() if doc is None else doc, sort_keys=True).encode()
    path.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    (path.parent / 'snapshot.sha256').write_text(digest + '\n')
    return digest


Answers = list[tuple[str, str, str | None]]

#: Misses the w2 contradiction, misfiles w3, contests the agreeing w4: 2 errors, 1 misfile.
MISSES: Answers = [
    ('w1', 'restated', 't1'), ('w2', 'amended', 't2'),
    ('w3', 'amended', 't3'), ('w4', 'contested', 't4'),
]
#: Contests w2 rightly but still misfiles w3 and contests w4: 1 error, 1 misfile.
PARTLY: Answers = [
    ('w1', 'restated', 't1'), ('w2', 'contested', 't2'),
    ('w3', 'amended', 't3'), ('w4', 'contested', 't4'),
]
#: Every write right: 0 errors, 0 misfiles.
RIGHT: Answers = [
    ('w1', 'restated', 't1'), ('w2', 'contested', 't2'),
    ('w3', 'restated', 't3b'), ('w4', 'amended', 't4'),
]
SOL_CALL = {'model': 'gpt-6.1-sol', 'seconds': 4.0, 'prompt': 2000, 'completion': 200}


def _arm_rows(arm: str, answers: Answers, **call: Any) -> list[dict[str, Any]]:
    """*arm*'s judge-band rows for writes w1..w4; w1, w2 share parent p1 and w3, w4 share p2."""
    return [
        _row(arm, write, outcome, judged, winner='p1' if write in ('w1', 'w2') else 'p2', **call)
        for write, outcome, judged in answers
    ]


def _three_arm_cases() -> dict[str, list[dict[str, Any]]]:
    """Three arms that differ in misfiles and contested-decision errors.

    The reference MISSES, pre-ψ is PARTLY right, and sol is RIGHT at a higher
    price and latency.
    """
    return {
        REFERENCE: _arm_rows(REFERENCE, MISSES),
        PRE_PSI: _arm_rows(PRE_PSI, PARTLY),
        SOL: _arm_rows(SOL, RIGHT, **SOL_CALL),
    }


def _shipped() -> Any:
    return _mod().ArmSettings('openai', 'gpt-4o-mini', None, 5, 'shipped', 4000)


def _bound(path: str, op: str, value: str) -> Any:
    return _mod().Bound(path, op, value)


def _build(
    arms: list[dict[str, Any]] | None = None,
    cases_by_arm: dict[str, list[dict[str, Any]]] | None = None,
    verdicts: list[dict[str, Any]] | None = None,
    *,
    n_judge_band: int = 4,
    bounds: list[Any] | None = None,
    shipped: Any = None,
) -> dict[str, Any]:
    population = _mod().Population.from_artifact(
        _population(THREE_ARMS if arms is None else arms, n_judge_band),
    )
    return _mod().build_matrix(
        population,
        _three_arm_cases() if cases_by_arm is None else cases_by_arm,
        VERDICTS if verdicts is None else verdicts,
        shipped=_shipped() if shipped is None else shipped,
        bounds=[_bound('quality.unrated_pairs', '<=', '0')] if bounds is None else bounds,
        created_at=CREATED_AT,
    )


def _oracle(cases: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """ι's own scores of *cases*, with the comparative block μ asks it for."""
    return _mod_iota().score_pairs(
        chain(*cases.values()), VERDICTS, reference_arm=REFERENCE, created_at=CREATED_AT,
    )


def _scored_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in matrix['arms'] if row['status'] == 'scored']


# --- step 1: every π arm scored by ι, verbatim -----------------------------------

def test_build_matrix_scores_every_pi_arm_with_iota() -> None:
    cases = _three_arm_cases()
    matrix = _build(cases_by_arm=cases)
    oracle = _oracle(cases)

    assert matrix['reference_arm'] == REFERENCE
    assert matrix['comparative_rules'] == oracle['comparative_rules']
    assert isinstance(matrix['comparative_rules'], dict)
    scored = _scored_rows(matrix)
    assert [row['arm'] for row in scored] == [REFERENCE, PRE_PSI, SOL]
    for row, provenance in zip(scored, THREE_ARMS, strict=True):
        iota = oracle['arms'][row['arm']]
        assert row['quality'] == iota['quality']
        assert row['runtime'] == iota['runtime']
        assert row['paired_vs_reference'] == iota['paired_vs_reference']
        assert row['comparative'] == iota['comparative']
        assert isinstance(row['comparative'], dict)
        assert row['selection'] == {
            'judge_provider': provenance['provider'],
            'judge_model': provenance['model'],
            'judge_reasoning_effort': provenance['reasoning_effort'],
            'judge_candidate_count': provenance['width'],
            'wording': provenance['wording'],
            'field_chars': provenance['field_chars'],
            'p95_judge_seconds': iota['runtime']['p95_judge_seconds'],
            'cost_per_write_usd': iota['runtime']['cost_per_write_usd'],
        }
    assert matrix['population'] == _c2_block(_population(THREE_ARMS, 4))
    assert matrix['verdict_corpus'] == oracle['verdict_corpus']
    assert matrix['list_prices'] == oracle['list_prices']


def test_shipped_reference_arm_must_be_unique() -> None:
    nowhere = _mod().ArmSettings('openai', 'gpt-4o-mini', None, 7, 'shipped', 4000)
    with pytest.raises(ValueError, match='matches no') as no_match:
        _build(shipped=nowhere)
    message = str(no_match.value)
    assert "candidate_count=7" in message
    assert all(name in message for name in (REFERENCE, PRE_PSI, SOL))

    twin = dict(THREE_ARMS[0], arm='gpt-4o-mini@5-again')
    cases = _three_arm_cases()
    cases['gpt-4o-mini@5-again'] = [
        dict(row, arm='gpt-4o-mini@5-again') for row in cases[REFERENCE]
    ]
    with pytest.raises(ValueError, match='gpt-4o-mini@5-again'):
        _build(arms=[*THREE_ARMS, twin], cases_by_arm=cases)


def test_shipped_settings_resolve_through_the_shipped_resolvers() -> None:
    service = types.SimpleNamespace(config=types.SimpleNamespace(
        llm=types.SimpleNamespace(provider='openai', model='gpt-4o-mini'),
        write_triage=types.SimpleNamespace(
            judge_provider=None, judge_model=None, judge_reasoning_effort='low',
            judge_candidate_count=20, judge_field_chars=4000,
        ),
    ))
    assert _mod().shipped_settings(service) == _mod().ArmSettings(
        'openai', 'gpt-4o-mini', 'low', 20, 'shipped', 4000,
    )


# --- step 3: Γ3's bounds and the winner rule --------------------------------------

#: Above gpt-4o-mini's $0.000021 per write and below sol's $0.006.
CHEAP = '0.001'


def _gate_checks(row: dict[str, Any], matrix: dict[str, Any], bounds: list[Any]) -> list[dict]:
    doc = {
        'quality': row['quality'], 'selection': row['selection'],
        'population': matrix['population'], 'comparative': row['comparative'],
    }
    return [
        _mod_gate().check_require({'best': doc}, 'best', b.path, b.op, b.value) for b in bounds
    ]


def test_every_arm_is_checked_against_the_bounds_the_way_gamma3_checks_them() -> None:
    bounds = [
        _bound('quality.misfile_rate_of_attaches', '<=', '0.3'),
        _bound('selection.cost_per_write_usd', '<=', CHEAP),
        _bound('population.n_judge_band', '>=', '4'),
        _bound('comparative.wrong_sibling_rate_placeable', '<=', '0.25'),
        _bound('comparative.harm_delta_vs_status_quo', '<=', '0'),
    ]
    matrix = _build(bounds=bounds)

    met = {}
    for row in _scored_rows(matrix):
        checks = _gate_checks(row, matrix, bounds)
        assert row['bounds'] == {'met': all(c['ok'] for c in checks), 'checks': checks}
        assert [c['note'] for c in checks] == [None] * len(checks)
        met[row['arm']] = row['bounds']['met']
    assert met == {REFERENCE: False, PRE_PSI: True, SOL: False}


def test_candidate_doc_is_what_gamma3_reads() -> None:
    [row] = [row for row in _scored_rows(_build()) if row['arm'] == SOL]
    population = _c2_block(_population(THREE_ARMS, 4))

    assert _mod().candidate_doc(row, population) == {
        'quality': row['quality'], 'selection': row['selection'],
        'population': population, 'comparative': row['comparative'],
    }


def test_a_metric_that_is_none_fails_its_bound() -> None:
    no_contradiction = [
        _vote(v['entry_id'], v['target_id'], 'SAME') if v['verdict'] == 'CORRECTS' else v
        for v in VERDICTS
    ]
    bounds = [_bound('quality.contradiction_recall', '>=', '0')]
    matrix = _build(verdicts=no_contradiction, bounds=bounds)

    for row in _scored_rows(matrix):
        assert row['quality']['contradiction_recall'] is None
        [check] = row['bounds']['checks']
        assert check['ok'] is False
        assert row['bounds']['met'] is False


def _harm_deltas(matrix: dict[str, Any]) -> dict[str, float]:
    return {
        row['arm']: row['comparative']['harm_delta_vs_status_quo'] for row in _scored_rows(matrix)
    }


def test_winner_is_the_qualifying_arm_with_the_lowest_harm_delta() -> None:
    bounds = [
        _bound('quality.misfile_rate_of_attaches', '<=', '0.3'),
        _bound('selection.cost_per_write_usd', '<=', CHEAP),
    ]
    matrix = _build(bounds=bounds)

    assert _harm_deltas(matrix) == {REFERENCE: 1.0, PRE_PSI: -1.0, SOL: -4.0}
    assert [row['arm'] for row in _scored_rows(matrix) if not row['bounds']['met']] == [SOL]
    assert matrix['winner'] == PRE_PSI
    rule = matrix['selection_rule']
    assert set(rule) == {'bounds', 'rule', 'fallback_applied'}
    assert rule['bounds'] == [
        {'path': b.path, 'op': b.op, 'value': b.value} for b in bounds
    ]
    assert isinstance(rule['rule'], str) and rule['rule']
    assert rule['fallback_applied'] is False


STORE = 'gpt-6-luna:low@5'
#: Stores every write flat: no contested-decision error at all, and no change from the status quo.
STORED: Answers = [(write, 'stored', None) for write in SLATE_OF]


def test_the_harm_delta_not_the_contested_decision_errors_picks_the_winner() -> None:
    arms = [*THREE_ARMS, _provenance(STORE, 'gpt-6-luna', 'low', 5, 'shipped')]
    cases = _three_arm_cases() | {STORE: _arm_rows(STORE, STORED, model='gpt-6-luna')}
    matrix = _build(arms=arms, cases_by_arm=cases,
                    bounds=[_bound('selection.cost_per_write_usd', '<=', CHEAP)])

    qualified = [row for row in _scored_rows(matrix) if row['bounds']['met']]
    assert [row['arm'] for row in qualified] == [REFERENCE, PRE_PSI, STORE]
    assert min(qualified, key=lambda row: row['quality']['contested_decision_errors'])['arm'] == (
        STORE
    )
    assert _harm_deltas(matrix)[STORE] == 0
    assert matrix['winner'] == PRE_PSI


def _ranked(arm: str, delta: float | None, cost: float | None, failed: int = 0) -> dict[str, Any]:
    """A scored row that fails *failed* bounds, shaped as ``_with_bounds`` writes it."""
    checks = [{'check': f'bound {index}', 'ok': False, 'note': None} for index in range(failed)]
    return {
        'arm': arm, 'status': 'scored', 'bounds': {'met': not failed, 'checks': checks},
        'comparative': {'harm_delta_vs_status_quo': delta},
        'selection': {'cost_per_write_usd': cost},
    }


def test_winner_ties_break_to_lower_cost_then_arm_name() -> None:
    select = _mod().select_winner
    skipped = {'arm': 'aaa', 'status': 'skipped', 'provider': 'anthropic', 'reason': 'no route'}

    assert select([_ranked('a', -3.0, 0.002), _ranked('b', -3.0, 0.001)]) == ('b', False)
    assert select([_ranked('a', -3.0, None), _ranked('b', -3.0, 0.5)]) == ('b', False)
    assert select([_ranked('z', -3.0, 0.001), _ranked('a', -3.0, 0.001)]) == ('a', False)
    assert select([skipped, _ranked('z', -3.0, 0.001)]) == ('z', False)
    with pytest.raises(ValueError):
        select([skipped])


def test_a_lower_harm_delta_wins_and_a_missing_one_ranks_last() -> None:
    select = _mod().select_winner

    assert select([_ranked('a', -1.0, 0.001), _ranked('b', -2.5, 0.009)]) == ('b', False)
    assert select([_ranked('a', None, 0.001), _ranked('b', 40.0, 0.5)]) == ('b', False)


def test_no_qualifying_arm_falls_back_to_the_lowest_harm_delta() -> None:
    matrix = _build(bounds=[_bound('population.n_judge_band', '>=', '5')])

    assert not any(row['bounds']['met'] for row in _scored_rows(matrix))
    assert matrix['winner'] == SOL
    assert matrix['selection_rule']['fallback_applied'] is True


def test_without_a_qualifying_arm_the_winner_fails_fewest_bounds_before_lowest_delta() -> None:
    select = _mod().select_winner

    assert select([
        _ranked('lowest-delta', -60.0, 0.017, failed=2), _ranked('one-miss', 10.0, 0.002, failed=1),
    ]) == ('one-miss', True)
    assert select([
        _ranked('a', 60.0, 0.001, failed=1), _ranked('b', 49.0, 0.001, failed=1),
    ]) == ('b', True)
    assert select([
        _ranked('close', -100.0, 0.001, failed=1), _ranked('qualified', 206.0, 0.001),
    ]) == ('qualified', False)

    matrix = _build(bounds=[
        _bound('population.n_judge_band', '>=', '5'),
        _bound('selection.cost_per_write_usd', '<=', CHEAP),
    ])
    failed = {
        row['arm']: sum(not check['ok'] for check in row['bounds']['checks'])
        for row in _scored_rows(matrix)
    }
    assert failed == {REFERENCE: 1, PRE_PSI: 1, SOL: 2}
    assert matrix['winner'] == PRE_PSI
    assert matrix['selection_rule']['fallback_applied'] is True


def test_empty_bounds_are_refused() -> None:
    with pytest.raises(ValueError, match='--require'):
        _build(bounds=[])


def test_skipped_arms_are_recorded_with_reasons() -> None:
    skipped_arms = _mod().SKIPPED_ARMS
    assert skipped_arms
    matrix = _build()

    assert matrix['arms'][len(THREE_ARMS):] == [
        {'arm': s.arm, 'status': 'skipped', 'provider': s.provider, 'reason': s.reason}
        for s in skipped_arms
    ]
    assert all(s.reason.strip() for s in skipped_arms)

    taken = skipped_arms[0].arm
    arms = [*THREE_ARMS[:2], dict(THREE_ARMS[2], arm=taken)]
    cases = _three_arm_cases()
    cases[taken] = [dict(row, arm=taken) for row in cases.pop(SOL)]
    with pytest.raises(ValueError, match='skipped'):
        _build(arms=arms, cases_by_arm=cases)


# --- step 5: best_config is the winner's candidate doc ------------------------------

QUALIFYING_BOUNDS = [
    ('quality.misfile_rate_of_attaches', '<=', '0.3'),
    ('selection.cost_per_write_usd', '<=', CHEAP),
]
UNMET_BOUNDS = [('population.n_judge_band', '>=', '5')]
UNEVENLY_UNMET_BOUNDS = [*UNMET_BOUNDS, ('selection.cost_per_write_usd', '<=', CHEAP)]
COMPARATIVE_BOUNDS = [
    ('comparative.wrong_sibling_rate_placeable', '<=', '0.11'),
    ('comparative.harm_delta_vs_status_quo', '<=', '0'),
]
UNMET_COMPARATIVE_BOUNDS = [('comparative.harm_delta_vs_status_quo', '<=', '-5')]


@pytest.mark.parametrize(
    ('spec', 'winner', 'gate_exit'),
    [
        (QUALIFYING_BOUNDS, PRE_PSI, 0),
        (UNMET_BOUNDS, SOL, 1),
        (UNEVENLY_UNMET_BOUNDS, PRE_PSI, 1),
        (COMPARATIVE_BOUNDS, SOL, 0),
        (UNMET_COMPARATIVE_BOUNDS, SOL, 1),
    ],
    ids=[
        'winner-qualified', 'fallback', 'fallback-fewest-failed-bounds',
        'comparative-qualified', 'comparative-fallback',
    ],
)
def test_best_config_copies_the_winner_verbatim(
    spec: list[tuple[str, str, str]], winner: str, gate_exit: int, tmp_path: Path,
) -> None:
    cases = _three_arm_cases()
    matrix = _build(cases_by_arm=cases, bounds=[_bound(*bound) for bound in spec])
    assert matrix['winner'] == winner
    [row] = [row for row in _scored_rows(matrix) if row['arm'] == winner]
    oracle = _oracle(cases)

    best = _mod().best_config(matrix)

    assert set(best) == {
        'quality', 'selection', 'population', 'comparative', 'comparative_rules', 'provenance',
    }
    assert best['quality'] == row['quality'] == oracle['arms'][winner]['quality']
    assert best['comparative'] == row['comparative'] == oracle['arms'][winner]['comparative']
    assert best['comparative_rules'] == matrix['comparative_rules'] == oracle['comparative_rules']
    assert best['selection'] == row['selection']
    assert set(best['selection']) == {
        'judge_provider', 'judge_model', 'judge_reasoning_effort', 'judge_candidate_count',
        'wording', 'field_chars', 'p95_judge_seconds', 'cost_per_write_usd',
    }
    assert best['population'] == matrix['population']
    assert set(best['population']) == {
        'n_writes', 'n_judge_band', 'projects', 'frozen_at', 'snapshot_sha256',
    }
    assert best['provenance'] == {
        'arm': winner,
        'reference_arm': matrix['reference_arm'],
        'meets_every_bound': row['bounds']['met'],
        'failed_bounds': [c['check'] for c in row['bounds']['checks'] if not c['ok']],
    }

    path = tmp_path / 'best_config.json'
    path.write_text(json.dumps(best, indent=2))
    requires = [arg for bound in spec for arg in ('--require', 'best', *bound)]
    assert _mod_gate().main(['--gate', 'G', '--report', 'best', str(path), *requires]) == gate_exit


# --- step 7: μ refuses inputs that drifted from what π published ----------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def _publish(
    root: Path, cases_by_arm: dict[str, list[dict[str, Any]]], snapshot: str = SNAPSHOT,
) -> dict[str, Any]:
    """Write each arm's rows under *root* and return a population doc that pins their digests."""
    arms = []
    for provenance in THREE_ARMS:
        path = root / provenance['cases_path']
        _write_rows(path, cases_by_arm[provenance['arm']])
        arms.append(dict(provenance, cases_sha256=_sha256(path)))
    return _population(arms, 4, snapshot=snapshot)


def _read(doc: dict[str, Any], root: Path) -> dict[str, list[dict[str, Any]]]:
    return _mod().read_arm_cases(_mod().Population.from_artifact(doc), root)


def test_read_arm_cases_returns_rows_keyed_by_arm(tmp_path: Path) -> None:
    cases = _three_arm_cases()
    doc = _publish(tmp_path, cases)

    assert _read(doc, tmp_path) == cases


def test_read_arm_cases_refuses_a_digest_mismatch(tmp_path: Path) -> None:
    doc = _publish(tmp_path, _three_arm_cases())
    path = tmp_path / THREE_ARMS[1]['cases_path']
    expected = _sha256(path)
    with path.open('a') as handle:
        handle.write('\n')

    with pytest.raises(_mod().MatrixInputError) as refusal:
        _read(doc, tmp_path)
    assert issubclass(_mod().MatrixInputError, ValueError)
    message = str(refusal.value)
    assert all(part in message for part in (PRE_PSI, str(path), expected, _sha256(path)))


def test_read_arm_cases_refuses_a_missing_file(tmp_path: Path) -> None:
    doc = _publish(tmp_path, _three_arm_cases())
    path = tmp_path / THREE_ARMS[2]['cases_path']
    path.unlink()

    with pytest.raises(_mod().MatrixInputError) as refusal:
        _read(doc, tmp_path)
    message = str(refusal.value)
    assert all(part in message for part in (SOL, str(path), '--data-root'))


@pytest.mark.parametrize(
    ('field', 'value'),
    [('arm', 'other'), ('snapshot_sha256', 'another-snapshot'), ('slates', 'frozen')],
)
def test_read_arm_cases_refuses_rows_of_another_arm_or_snapshot(
    field: str, value: str, tmp_path: Path,
) -> None:
    cases = _three_arm_cases()
    cases[REFERENCE][2] = dict(cases[REFERENCE][2], **{field: value})
    doc = _publish(tmp_path, cases)

    with pytest.raises(_mod().MatrixInputError) as refusal:
        _read(doc, tmp_path)
    message = str(refusal.value)
    assert REFERENCE in message
    assert value in message


def _without(row: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: value for name, value in row.items() if name != key}


def _with_arm(doc: dict[str, Any], index: int, arm: Any) -> dict[str, Any]:
    return dict(doc, arms=[arm if at == index else row for at, row in enumerate(doc['arms'])])


#: Each way of breaking a population doc, and the words its refusal must name.
MALFORMED_POPULATIONS: dict[str, tuple[Any, list[str]]] = {
    'arm-lacks-cases-path': (
        lambda doc: _with_arm(doc, 1, _without(doc['arms'][1], 'cases_path')),
        ['arms[1]', 'cases_path'],
    ),
    'arm-lacks-width': (
        lambda doc: _with_arm(doc, 0, _without(doc['arms'][0], 'width')), ['arms[0]', 'width'],
    ),
    'block-lacks-snapshot': (
        lambda doc: dict(doc, population=_without(doc['population'], 'snapshot_sha256')),
        ['snapshot_sha256'],
    ),
    'block-lacks-write-time-judge-band': (
        lambda doc: dict(doc, population=_without(doc['population'], 'n_judge_band_write_time')),
        ['n_judge_band_write_time'],
    ),
    'block-lacks-snapshot-path': (
        lambda doc: dict(doc, population=_without(doc['population'], 'snapshot_path')),
        ['snapshot_path'],
    ),
    'frozen-slates-artifact-without-slates': (
        lambda doc: dict(doc, population=_without(doc['population'], 'slates')),
        ['slates', 'None'],
    ),
    'frozen-slates': (
        lambda doc: dict(doc, population={**doc['population'], 'slates': 'frozen'}),
        ['slates', "'frozen'"],
    ),
    'no-arms': (lambda doc: _without(doc, 'arms'), ['arms']),
    'arm-not-an-object': (lambda doc: _with_arm(doc, 2, SOL), ['population artifact']),
    'not-an-object': (lambda doc: [doc], ['population artifact']),
}


def _malformed_population(case: str) -> tuple[Any, list[str]]:
    breaks, named = MALFORMED_POPULATIONS[case]
    return breaks(_population(THREE_ARMS, 4)), named


@pytest.mark.parametrize('case', MALFORMED_POPULATIONS)
def test_population_from_artifact_refuses_a_malformed_doc_naming_what_is_wrong(case: str) -> None:
    doc, named = _malformed_population(case)

    with pytest.raises(_mod().MatrixInputError) as refusal:
        _mod().Population.from_artifact(doc)
    message = str(refusal.value)
    assert all(word in message for word in named), message


def test_population_reads_the_write_time_artifact_into_the_c2_block() -> None:
    doc = _population(THREE_ARMS, 4)
    population = _mod().Population.from_artifact(doc)

    assert population.c2_block() == _c2_block(doc)
    assert population.c2_block()['n_judge_band'] == doc['population']['n_judge_band_write_time']
    assert _mod().POPULATION_KEYS == (
        'n_writes', 'n_judge_band', 'projects', 'frozen_at', 'snapshot_sha256',
    )


def test_read_record_created_at_reads_the_pinned_snapshot(tmp_path: Path) -> None:
    digest = _write_snapshot(tmp_path)
    population = _mod().Population.from_artifact(_population(THREE_ARMS, 4, snapshot=digest))

    assert _mod().read_record_created_at(population, tmp_path) == CREATED_AT


def test_read_record_created_at_refuses_a_missing_snapshot(tmp_path: Path) -> None:
    population = _mod().Population.from_artifact(_population(THREE_ARMS, 4))

    with pytest.raises(_mod().MatrixInputError) as refusal:
        _mod().read_record_created_at(population, tmp_path)
    message = str(refusal.value)
    assert str(tmp_path / SNAPSHOT_PATH) in message and '--data-root' in message, message


def test_read_record_created_at_refuses_a_snapshot_the_population_does_not_pin(
    tmp_path: Path,
) -> None:
    digest = _write_snapshot(tmp_path)
    pinned = 'f' * 64
    population = _mod().Population.from_artifact(_population(THREE_ARMS, 4, snapshot=pinned))

    with pytest.raises(_mod().MatrixInputError) as refusal:
        _mod().read_record_created_at(population, tmp_path)
    message = str(refusal.value)
    assert digest in message and pinned in message, message


def test_read_record_created_at_propagates_a_sidecar_mismatch(tmp_path: Path) -> None:
    digest = _write_snapshot(tmp_path)
    sidecar = 'e' * 64
    (tmp_path / SNAPSHOT_PATH).with_name('snapshot.sha256').write_text(sidecar + '\n')
    population = _mod().Population.from_artifact(_population(THREE_ARMS, 4, snapshot=digest))

    with pytest.raises(ValueError) as refusal:
        _mod().read_record_created_at(population, tmp_path)
    assert type(refusal.value) is ValueError
    assert digest in str(refusal.value) and sidecar in str(refusal.value)


def test_build_matrix_refuses_an_arm_whose_judge_band_disagrees_with_the_population() -> None:
    with pytest.raises(_mod().MatrixInputError) as refusal:
        _build(n_judge_band=5)
    message = str(refusal.value)
    assert REFERENCE in message
    assert '4' in message and '5' in message


def test_build_matrix_propagates_iota_refusal() -> None:
    verdicts = [v for v in VERDICTS if (v['entry_id'], v['target_id']) != ('w3', 't3')]

    with pytest.raises(_mod_iota().IncompleteCorpusError, match='w3 t3'):
        _build(verdicts=verdicts)


# --- step 9: wording attribution at matched width ------------------------------------

AT20 = 'gpt-4o-mini@20'
AT20_PRE_PSI = 'gpt-4o-mini@20+pre-psi'
LUNA_PRE_PSI = 'gpt-6-luna:low@5+pre-psi'


def test_wording_attribution_pairs_each_non_shipped_wording_arm_with_its_shipped_twin() -> None:
    arms = [
        *THREE_ARMS[:2],
        _provenance(AT20, 'gpt-4o-mini', None, 20, 'shipped'),
        _provenance(AT20_PRE_PSI, 'gpt-4o-mini', None, 20, 'pre-psi'),
        THREE_ARMS[2],
        _provenance(LUNA_PRE_PSI, 'gpt-6-luna', 'low', 5, 'pre-psi'),
    ]
    cases = _three_arm_cases() | {
        AT20: _arm_rows(AT20, RIGHT),
        AT20_PRE_PSI: _arm_rows(AT20_PRE_PSI, MISSES),
        LUNA_PRE_PSI: _arm_rows(LUNA_PRE_PSI, PARTLY, model='gpt-6-luna'),
    }
    matrix = _build(arms=arms, cases_by_arm=cases)
    attribution = matrix['wording_attribution']

    assert set(attribution) == {PRE_PSI, AT20_PRE_PSI, LUNA_PRE_PSI}
    for arm, twin in ((PRE_PSI, REFERENCE), (AT20_PRE_PSI, AT20)):
        oracle = _mod_iota().score_pairs(
            chain(cases[arm], cases[twin]), VERDICTS, reference_arm=twin,
        )
        assert attribution[arm] == {
            'shipped_twin': twin, 'paired': oracle['arms'][arm]['paired_vs_reference'],
        }
    [at20_pre_psi] = [row for row in _scored_rows(matrix) if row['arm'] == AT20_PRE_PSI]
    assert attribution[AT20_PRE_PSI]['paired'] != at20_pre_psi['paired_vs_reference']
    assert attribution[LUNA_PRE_PSI] == {'shipped_twin': None, 'paired': None}


# --- step 11: the CLI ---------------------------------------------------------------

MATRIX_NAME = 'write_triage_config_matrix.json'
BEST_NAME = 'write_triage_best_config.json'
CLI_BOUNDS = [('quality.unrated_pairs', '<=', '0'), ('population.n_judge_band', '>=', '4')]


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """A fake checkout holding π2's arm files and snapshot, the two committed inputs and an out dir.

    The fused-memory config it points CONFIG_PATH at ships gpt-4o-mini@5.
    """
    (tmp_path / '.git').mkdir()
    snapshot = _write_snapshot(tmp_path)
    cases = {
        arm: [dict(row, snapshot_sha256=snapshot) for row in rows]
        for arm, rows in _three_arm_cases().items()
    }
    doc = _publish(tmp_path, cases, snapshot=snapshot)
    population = tmp_path / 'population.json'
    population.write_text(json.dumps(doc, indent=2))
    verdicts = tmp_path / 'verdicts.jsonl'
    _write_rows(verdicts, VERDICTS)
    out = tmp_path / 'out'
    out.mkdir()
    config = tmp_path / 'config.yaml'
    config.write_text(
        'llm:\n  provider: openai\n  model: gpt-4o-mini\n'
        'write_triage:\n  judge_candidate_count: 5\n  judge_field_chars: 4000\n'
    )
    monkeypatch.setenv('CONFIG_PATH', str(config))
    return types.SimpleNamespace(
        root=tmp_path, cases=cases, doc=doc, population=population, verdicts=verdicts, out=out,
    )


def _main(cli: types.SimpleNamespace, bounds: list[tuple[str, str, str]] = CLI_BOUNDS) -> Any:
    argv = [
        '--population', str(cli.population), '--verdicts', str(cli.verdicts),
        '--data-root', str(cli.root), '--out-dir', str(cli.out),
        *[arg for bound in bounds for arg in ('--require', *bound)],
    ]
    try:
        return _mod().main(argv)
    except SystemExit as exit_:
        return exit_.code


def test_main_writes_both_artifacts(
    cli: types.SimpleNamespace, capsys: pytest.CaptureFixture[str],
) -> None:
    assert _main(cli) == 0

    expected = _mod().build_matrix(
        _mod().Population.from_artifact(cli.doc), cli.cases, VERDICTS,
        shipped=_shipped(), bounds=[_bound(*bound) for bound in CLI_BOUNDS],
        created_at=CREATED_AT,
    )
    matrix = json.loads((cli.out / MATRIX_NAME).read_text())
    assert matrix == expected | {'inputs': {
        'population_path': 'population.json',
        'population_sha256': _sha256(cli.population),
        'verdicts_path': 'verdicts.jsonl',
        'verdicts_sha256': _sha256(cli.verdicts),
    }}
    best = json.loads((cli.out / BEST_NAME).read_text())
    assert best == _mod().best_config(expected)
    assert all(isinstance(row['comparative'], dict) for row in _scored_rows(matrix))
    assert isinstance(best['comparative'], dict)

    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary['winner'] == expected['winner']
    assert summary['meets_every_bound'] == best['provenance']['meets_every_bound']
    assert {str(cli.out / MATRIX_NAME), str(cli.out / BEST_NAME)} <= set(summary.values())


def _written(cli: types.SimpleNamespace) -> list[str]:
    return sorted(path.name for path in cli.out.iterdir())


def test_main_refuses_an_incomplete_corpus_and_writes_nothing(
    cli: types.SimpleNamespace, capsys: pytest.CaptureFixture[str],
) -> None:
    _write_rows(cli.verdicts, [
        v for v in VERDICTS if (v['entry_id'], v['target_id']) != ('w3', 't3')
    ])
    stale = cli.out / BEST_NAME
    stale.write_bytes(b'{"stale": true}\n')

    assert _main(cli) == 1

    err = capsys.readouterr().err
    assert any('w3' in line and 't3' in line for line in err.splitlines())
    assert _written(cli) == [BEST_NAME]
    assert stale.read_bytes() == b'{"stale": true}\n'


def test_main_refuses_a_drifted_arm_file_and_writes_nothing(
    cli: types.SimpleNamespace, capsys: pytest.CaptureFixture[str],
) -> None:
    with (cli.root / THREE_ARMS[2]['cases_path']).open('a') as handle:
        handle.write('\n')

    assert _main(cli) == 1

    err = capsys.readouterr().err
    assert SOL in err and 'sha256' in err
    assert _written(cli) == []


def test_main_refuses_a_snapshot_the_population_does_not_pin_and_writes_nothing(
    cli: types.SimpleNamespace, capsys: pytest.CaptureFixture[str],
) -> None:
    pinned = cli.doc['population']['snapshot_sha256']
    other = _write_snapshot(cli.root, {**_snapshot_doc(), 'targets': {'t9': {'content': 'x'}}})

    assert _main(cli) == 1

    err = capsys.readouterr().err
    assert pinned in err and other in err, err
    assert _written(cli) == []


def test_the_default_population_is_the_write_time_artifact() -> None:
    args = _mod()._build_parser().parse_args(['--data-root', '.', '--require', 'a', '<=', '1'])
    assert args.population == CALIBRATION / 'write_triage_population_write_time.json'


def test_main_requires_at_least_one_bound(cli: types.SimpleNamespace) -> None:
    assert _main(cli, bounds=[]) not in (0, None)
    assert _written(cli) == []


def test_main_refuses_a_malformed_population_and_writes_nothing(
    cli: types.SimpleNamespace, capsys: pytest.CaptureFixture[str],
) -> None:
    doc, named = _malformed_population('arm-lacks-cases-path')
    cli.population.write_text(json.dumps(doc))

    assert _main(cli) == 1

    err = capsys.readouterr().err
    assert all(word in err for word in named)
    assert _written(cli) == []


def test_main_reports_an_out_dir_it_cannot_create(
    cli: types.SimpleNamespace, capsys: pytest.CaptureFixture[str],
) -> None:
    cli.out = cli.population / 'out'

    assert _main(cli) == 1

    err = capsys.readouterr().err
    assert str(cli.out) in err


def test_main_reports_a_failed_write_and_leaves_no_staged_file(
    cli: types.SimpleNamespace, capsys: pytest.CaptureFixture[str],
) -> None:
    (cli.out / MATRIX_NAME).mkdir()

    assert _main(cli) == 1

    err = capsys.readouterr().err
    assert str(cli.out) in err
    assert _written(cli) == [MATRIX_NAME]
    assert (cli.out / MATRIX_NAME).is_dir()


# --- step 13: the committed artifacts ---------------------------------------------------

def _committed(name: str) -> Any:
    return json.loads((CALIBRATION / name).read_text(encoding='utf-8'))


def _line_prefix_sha256s(path: Path) -> set[str]:
    """The digest of every line prefix: the verdict corpus is only appended to, so μ's pin is one of them."""
    digest = hashlib.sha256()
    prefixes = {digest.hexdigest()}
    for line in path.read_bytes().splitlines(keepends=True):
        digest.update(line)
        prefixes.add(digest.hexdigest())
    return prefixes


WRITE_TIME_POPULATION = 'write_triage_population_write_time.json'
GAMMA3_PATHS = (
    'comparative.wrong_sibling_rate_placeable',
    'comparative.harm_delta_vs_status_quo',
    'quality.contradiction_recall',
    'selection.p95_judge_seconds',
    'selection.cost_per_write_usd',
    'quality.unrated_pairs',
    'population.n_judge_band',
)
"""The key paths Γ3 reads (contract C2‴, PRD §12.3); their thresholds live in task 5808's args."""


def test_committed_best_config_is_the_committed_matrix_winner() -> None:
    mod = _mod()
    matrix = _committed(MATRIX_NAME)
    best = _committed(BEST_NAME)
    published = _committed(WRITE_TIME_POPULATION)
    scored = _scored_rows(matrix)

    assert [row['arm'] for row in scored] == [arm['arm'] for arm in published['arms']]
    assert matrix['arms'][len(scored):] == [skipped.row() for skipped in mod.SKIPPED_ARMS]
    assert all(row['reason'].strip() for row in matrix['arms'][len(scored):])
    for row, provenance in zip(scored, published['arms'], strict=True):
        settings = mod.ArmSettings.from_provenance(provenance).selection_keys()
        assert {key: row['selection'][key] for key in settings} == settings
    assert matrix['population'] == _c2_block(published)
    assert matrix['inputs']['population_path'] == f'fused-memory/calibration/{WRITE_TIME_POPULATION}'
    assert matrix['inputs']['population_sha256'] == _sha256(CALIBRATION / WRITE_TIME_POPULATION)
    assert matrix['inputs']['verdicts_sha256'] in _line_prefix_sha256s(
        CALIBRATION / 'write_triage_pair_verdicts.jsonl',
    )

    [reference] = [
        arm.name for arm in mod.Population.from_artifact(published).arms
        if arm.settings.selection_keys() == matrix['reference_settings']
    ]
    assert matrix['reference_arm'] == reference

    rule = matrix['selection_rule']
    bounds = [_bound(b['path'], b['op'], b['value']) for b in rule['bounds']]
    for row in scored:
        checks = _gate_checks(row, matrix, bounds)
        assert row['bounds'] == {'met': all(c['ok'] for c in checks), 'checks': checks}
        assert [c['note'] for c in checks] == [None] * len(checks)
    assert mod.select_winner(matrix['arms']) == (matrix['winner'], rule['fallback_applied'])

    assert best == mod.best_config(matrix)
    assert best['quality']['unrated_pairs'] == 0
    assert best['population']['n_judge_band'] == published['population']['n_judge_band_write_time']
    assert 'false_contested_rate' in best['quality']


def test_committed_best_config_answers_every_path_gamma3_reads() -> None:
    gate = _mod_gate()
    best = _committed(BEST_NAME)

    for path in GAMMA3_PATHS:
        actual = gate.resolve(best, path)
        assert gate.numeric_reason(actual) is None, (path, actual)


def test_committed_matrix_was_run_with_both_comparative_bounds() -> None:
    paths = {bound['path'] for bound in _committed(MATRIX_NAME)['selection_rule']['bounds']}
    assert {
        'comparative.wrong_sibling_rate_placeable', 'comparative.harm_delta_vs_status_quo',
    } <= paths


def test_committed_comparative_harm_follows_from_its_own_table_and_weights() -> None:
    for row in _scored_rows(_committed(MATRIX_NAME)):
        comparative = row['comparative']
        weights = comparative['harm_weights']
        totals = comparative['harm_total']
        for regime, table in comparative['outcomes'].items():
            expected = round(sum(weight * table[cell] for cell, weight in weights.items()), 4)
            assert totals[regime] == expected, (row['arm'], regime)
        assert comparative['harm_delta_vs_status_quo'] == pytest.approx(
            totals['arm'] - totals['status_quo'], abs=1e-4,
        ), row['arm']
