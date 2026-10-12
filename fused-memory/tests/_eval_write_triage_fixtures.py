"""Builders shared by the eval_write_triage_* test modules. Not collected.

A builder lives here only when more than one of those modules uses it; one a
single module uses stays in that module. A builder that needs a script loads
it by path under the script's file stem, the key the scripts load each other
under, so it shares their module objects.
"""
from __future__ import annotations

import functools
import json
import types
from pathlib import Path

from _fm_helpers import load_script_module

from fused_memory.server.write_triage import OUTCOME_AMENDED, OUTCOME_JUDGE

SCRIPTS = Path(__file__).parent.parent / 'scripts'
FIXTURE_PATH = Path(__file__).parent / 'fixtures' / 'write_triage_calibration.jsonl'
ALIASES_PATH = (
    Path(__file__).parent / 'fixtures' / 'write_triage_calibration.canonical_aliases.json'
)


@functools.cache
def _scoring() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'eval_write_triage_scoring.py', 'eval_write_triage_scoring')


@functools.cache
def _cases_mod() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'eval_write_triage_cases.py', 'eval_write_triage_cases')


def committed_records() -> list[dict]:
    """The committed curator corpus, parsed with the stdlib.

    Parsed here rather than through either script's loader, so a loader bug
    cannot mask a data defect (and vice versa) — the discipline the sibling
    suite's own ``records`` fixture states.
    """
    assert FIXTURE_PATH.exists(), f'fixture missing: {FIXTURE_PATH}'
    return [
        json.loads(line)
        for line in FIXTURE_PATH.read_text().splitlines()
        if line.strip()
    ]


def rec(memory_id: str, cluster_id: str, label: str, *, category: str = 'procedural_knowledge') -> dict:
    """A minimal fixture record — only the keys the eval's pure core reads."""
    return {
        'memory_id': memory_id,
        'content': f'content of {memory_id}',
        'category': category,
        'cluster_id': cluster_id,
        'label': label,
    }


def corpus() -> list[dict]:
    """Three clusters, every label represented, deliberately unsorted on input.

    Unsorted so a determinism assertion cannot pass by accident on an input
    that was already in canonical order.

    ``cluster_id`` IS the canonical record's own ``memory_id`` — the fixture's
    referential-integrity invariant, pinned against the committed corpus by
    `tests/test_eval_write_triage_cases.py::TestBuildJudgeCases::test_the_synthetic_corpus_uses_the_fixtures_cluster_key`.
    A synthetic corpus that named clusters some other way would let the eval
    build slates out of ids that exist nowhere, and every test using this
    corpus would still pass.
    """
    return [
        rec('c2-dup-1', 'c2-canon', 'duplicate'),
        rec('c1-canon', 'c1-canon', 'canonical'),
        rec('c3-pseudo', 'c3-canon', 'pseudo_contradiction'),
        rec('c1-dup-2', 'c1-canon', 'duplicate'),
        rec('c2-canon', 'c2-canon', 'canonical'),
        rec('c1-dup-1', 'c1-canon', 'duplicate'),
        rec('c3-canon', 'c3-canon', 'canonical'),
        rec('c2-distinct', 'c2-canon', 'distinct'),
        rec('c3-dup-1', 'c3-canon', 'duplicate'),
    ]


PROVENANCE = {
    'fixture_path': 'tests/fixtures/write_triage_calibration.jsonl',
    'judge_provider': 'openai',
    'judge_model': 'gpt-4o-mini',
}


def slate(memory_id: str, *, candidates, band_winner_id, band,
          similarity: float | None = 0.7, canonical_present=True):
    """A `eval_write_triage_retrieval.Slate`-shaped stand-in.

    Duck-typed rather than imported: the retrieval module reaches the live
    server stack, and everything asserted here is the eval's own handling of
    what that module RETURNS.
    """
    return types.SimpleNamespace(
        memory_id=memory_id,
        candidates=tuple(candidates),
        attach_target_id=band_winner_id,
        band=band,
        similarity=similarity,
        canonical_present=canonical_present,
        retrieved_count=len(candidates),
        degraded=False,
        self_retrieved=False,
    )


def live(
    memory_id: str, *, canonical_id: str | None = None, category='procedural_knowledge',
    store_score: float = 0.7,
):
    """A retrieved store row in the eval's record shape, scored by ITS query."""
    return {
        'memory_id': memory_id,
        'content': f'content of {memory_id}',
        'category': category,
        'canonical_id': canonical_id or memory_id,
        'store_score': store_score,
        'metadata': {'store_score': store_score},
    }


ALIASES = {'old-canon': 'new-canon'}


def aliased_row(candidates, band_winner_id, answer=None) -> dict:
    """The dump row of a duplicate whose cluster canonical was rotated."""
    records = [
        rec('old-canon', 'old-canon', 'canonical'),
        rec('dup', 'old-canon', 'duplicate'),
    ]
    retrieved = slate('dup', candidates=candidates, band_winner_id=band_winner_id,
                      band=OUTCOME_JUDGE, canonical_present=False)
    plan = _cases_mod().plan_from_slates(records, [retrieved], provenance={}, aliases=ALIASES)
    return _scoring().case_row(
        0, plan.cases[0],
        answer or _scoring().JudgeAnswer(outcome=OUTCOME_AMENDED, verdict=OUTCOME_AMENDED),
        plan.candidate_records[0], plan.fixture_by_id,
    )
