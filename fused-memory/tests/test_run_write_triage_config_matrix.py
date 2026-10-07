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
import types
from itertools import chain
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

SCRIPTS = Path(__file__).parent.parent / 'scripts'
SCRIPT_PATH = SCRIPTS / 'run_write_triage_config_matrix.py'
IOTA_PATH = SCRIPTS / 'score_write_triage_pairs.py'

SNAPSHOT = 'snap'
REFERENCE = 'gpt-4o-mini@5'
PRE_PSI = 'gpt-4o-mini@5+pre-psi'
SOL = 'gpt-6.1-sol:low@5'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(SCRIPT_PATH, mod_name='run_write_triage_config_matrix')


@functools.cache
def _mod_iota() -> types.ModuleType:
    return load_script_module(IOTA_PATH, mod_name='score_write_triage_pairs')


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
) -> dict[str, Any]:
    """One ι-contract judge-band case row, as π writes it."""
    return {
        'arm': arm, 'memory_id': memory_id, 'band': 'judge', 'outcome': outcome,
        'judged_candidate_id': judged, 'band_winner_id': winner,
        'parse_failure': False, 'judge_seconds': seconds, 'judge_model': model,
        'usage': {'prompt_tokens': prompt, 'completion_tokens': completion},
        'snapshot_sha256': snapshot,
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
    """A doc shaped like ``calibration/write_triage_population.json``."""
    return {
        'population': {
            'n_writes': n_judge_band + 3,
            'n_judge_band': n_judge_band,
            'n_judge_band_frozen': n_judge_band,
            'projects': ['dark_factory', 'reify'],
            'frozen_at': '2026-10-05T10:01:34+00:00',
            'snapshot_sha256': snapshot,
            'snapshot_path': 'data/x/snapshot.json',
        },
        'arms': arms,
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


def _three_arm_cases() -> dict[str, list[dict[str, Any]]]:
    """Four writes under three arms that differ in misfiles and contested-decision errors.

    The reference misses the w2 contradiction, misfiles w3 and contests the
    agreeing w4 (2 contested-decision errors, 1 misfile). pre-ψ gets the
    contest right but still misfiles w3 (0 errors, 1 misfile). sol gets every
    write right (0 errors, 0 misfiles) at a higher price and latency.
    """
    def parent(memory_id: str) -> str:
        return 'p1' if memory_id in ('w1', 'w2') else 'p2'

    def arm_rows(arm: str, answers: list[tuple[str, str, str | None]], **extra: Any):
        return [_row(arm, w, outcome, judged, winner=parent(w), **extra)
                for w, outcome, judged in answers]

    return {
        REFERENCE: arm_rows(REFERENCE, [
            ('w1', 'restated', 't1'), ('w2', 'amended', 't2'),
            ('w3', 'amended', 't3'), ('w4', 'contested', 't4'),
        ]),
        PRE_PSI: arm_rows(PRE_PSI, [
            ('w1', 'restated', 't1'), ('w2', 'contested', 't2'),
            ('w3', 'amended', 't3'), ('w4', 'amended', 't4'),
        ]),
        SOL: arm_rows(SOL, [
            ('w1', 'restated', 't1'), ('w2', 'contested', 't2'),
            ('w3', 'restated', 't3b'), ('w4', 'amended', 't4'),
        ], model='gpt-6.1-sol', seconds=4.0, prompt=2000, completion=200),
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
    )


def _scored_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in matrix['arms'] if row['status'] == 'scored']


# --- step 1: every π arm scored by ι, verbatim -----------------------------------

def test_build_matrix_scores_every_pi_arm_with_iota() -> None:
    cases = _three_arm_cases()
    matrix = _build(cases_by_arm=cases)
    oracle = _mod_iota().score_pairs(
        chain(*cases.values()), VERDICTS, reference_arm=REFERENCE,
    )

    assert matrix['reference_arm'] == REFERENCE
    scored = _scored_rows(matrix)
    assert [row['arm'] for row in scored] == [REFERENCE, PRE_PSI, SOL]
    for row, provenance in zip(scored, THREE_ARMS, strict=True):
        iota = oracle['arms'][row['arm']]
        assert row['quality'] == iota['quality']
        assert row['runtime'] == iota['runtime']
        assert row['paired_vs_reference'] == iota['paired_vs_reference']
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
    block = _population(THREE_ARMS, 4)['population']
    assert matrix['population'] == {
        key: block[key]
        for key in ('n_writes', 'n_judge_band', 'projects', 'frozen_at', 'snapshot_sha256')
    }
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
