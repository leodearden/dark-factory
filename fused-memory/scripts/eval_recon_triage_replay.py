#!/usr/bin/env python3
"""Replay recon's Stage-1 adjudication over rated write-triage attaches, read-only.

A seeded, stratified sample of the rubric v2.2 soak corpus is put to recon as
per-project worklists of attached children, through the real Stage-1 CLI
invocation with the read-only deny list. Recon answers one repair per child
from ``recon_triage_replay_scoring.py``'s vocabulary, and that module scores
the answers against the rated truth.

Subcommands, run from ``fused-memory/``:

- ``sample``: draw and write the frozen sample
  (``calibration/recon_triage_replay_sample.jsonl``, header line first).
- ``run --scratch DIR``: ask recon each batch's worklist, resumably, one JSON
  file per batch under ``DIR/batches/``.
- ``collect --scratch DIR``: write every parsed decision to
  ``calibration/recon_triage_replay_decisions.jsonl``.
- ``score``: score the committed sample and decisions into
  ``calibration/recon_triage_replay_scores.{json,md}``, from committed files alone.

Read-only rule: the soak bundle (``data/write-triage-soak-2026-10-19/``) and
the rating scratch (``data/write-triage-v21-rebase-2026-10-21/``) are read and
never written, and the replay's deny list keeps every store write out of
recon's reach.
"""
from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from fused_memory.server.write_triage import OUTCOME_AMENDED, OUTCOME_CONTESTED, OUTCOME_RESTATED
from fused_memory.utils.script_loader import load_script

_SCRIPTS = Path(__file__).resolve().parent


if TYPE_CHECKING:
    import recon_triage_replay_scoring as _scoring
    import write_triage_reader_rubric as _rubric
else:
    _scoring = load_script(_SCRIPTS / 'recon_triage_replay_scoring.py', 'recon_triage_replay_scoring')
    _rubric = load_script(_SCRIPTS / 'write_triage_reader_rubric.py', 'write_triage_reader_rubric')

SAMPLE_SEED = 7161
SAMPLE_PROJECTS = ('dark_factory', 'reify', 'solar_challenge', 'solar_challenge_platform')
_PER_OUTCOME = ((OUTCOME_CONTESTED, 20), (OUTCOME_AMENDED, 10), (OUTCOME_RESTATED, 10))

DEFAULT_ALLOCATION: Mapping[str, int] = MappingProxyType({
    _scoring.sample_stratum(outcome, project): count
    for project in SAMPLE_PROJECTS
    for outcome, count in _PER_OUTCOME
})
"""Headline children drawn per sample stratum: 160 over the four projects."""


def draw_sample(
    corpus_items: Iterable[Mapping[str, Any]],
    derivations: Mapping[str, _rubric.Derivation],
    *,
    allocation: Mapping[str, int],
    seed: int,
    touched: Mapping[str, str],
) -> list[dict[str, Any]]:
    """The headline draw of *allocation* from Decided items, then the touched children.

    Each stratum is drawn from its Decided items in entry-id order, the strata
    in name order from one ``random.Random(seed)``. A touched child (*touched*
    maps entry id to its live recon state) carries that state; one the draw
    missed is appended with ``headline`` false when it is in the corpus and
    Decided, so touching never moves the draw.
    """
    items = sorted(corpus_items, key=lambda item: item['entry_id'])
    decided = [
        item for item in items if isinstance(derivations.get(item['entry_id']), _rubric.Decided)
    ]
    draws = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for stratum, count in sorted(allocation.items()):
        pool = [
            item for item in decided
            if _scoring.sample_stratum(item['outcome'], item['project']) == stratum
        ]
        if count > len(pool):
            raise ValueError(
                f'stratum {stratum!r}: allocation asks {count} children, but only {len(pool)}'
                ' Decided items are available'
            )
        picked = sorted(draws.sample(pool, count), key=lambda item: item['entry_id'])
        rows += [_sample_row(item, headline=True, touched=touched) for item in picked]
    drawn = {row['entry_id'] for row in rows}
    rows += [
        _sample_row(item, headline=False, touched=touched)
        for item in decided if item['entry_id'] in touched and item['entry_id'] not in drawn
    ]
    return rows


def _sample_row(
    item: Mapping[str, Any], *, headline: bool, touched: Mapping[str, str],
) -> dict[str, Any]:
    return {
        'entry_id': item['entry_id'],
        'target_id': item['target_id'],
        'outcome': item['outcome'],
        'project': item['project'],
        'stratum': _scoring.sample_stratum(item['outcome'], item['project']),
        'headline': headline,
        'live_recon_state': touched.get(item['entry_id']),
    }


def touched_states(records: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Each touched child's live recon state, from the soak bundle's ``team/D/touched.json``."""
    return {record['id']: record['state'] for record in records}


def sample_header(
    *, seed: int, allocation: Mapping[str, int], inputs: Mapping[str, Path],
) -> dict[str, Any]:
    """The sample file's first line: the seed, the allocation and each input file's sha256."""
    return {
        'header': 'recon_triage_replay_sample',
        'seed': seed,
        'allocation': dict(sorted(allocation.items())),
        'inputs': {
            name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in inputs.items()
        },
    }
