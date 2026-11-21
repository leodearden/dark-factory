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
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from fused_memory.reconciliation.cli_stage_runner import STAGE3_DISALLOWED
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.stage1_payload_format import format_memories
from fused_memory.server.grouped_read import (
    AMENDMENT_KIND,
    CONTESTED_METADATA_KEY,
    PARENT_ID_KEY,
    SIGHTING_KIND,
)
from fused_memory.server.write_triage import OUTCOME_AMENDED, OUTCOME_CONTESTED, OUTCOME_RESTATED
from fused_memory.utils.script_loader import load_script

_SCRIPTS = Path(__file__).resolve().parent


if TYPE_CHECKING:
    import recon_triage_replay_scoring as _scoring
    import write_triage_rating_rounds as _rounds
    import write_triage_reader_rubric as _rubric
else:
    _scoring = load_script(_SCRIPTS / 'recon_triage_replay_scoring.py', 'recon_triage_replay_scoring')
    _rubric = load_script(_SCRIPTS / 'write_triage_reader_rubric.py', 'write_triage_reader_rubric')
    _rounds = load_script(_SCRIPTS / 'write_triage_rating_rounds.py', 'write_triage_rating_rounds')

Decision = _scoring.Decision

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


@dataclass(frozen=True)
class Batch:
    """One worklist put to recon in one Stage-1 run: children of a single project."""

    name: str
    project: str
    rows: tuple[Mapping[str, Any], ...]


def plan_batches(
    sample_rows: Iterable[Mapping[str, Any]], *, size: int, seed: int,
) -> list[Batch]:
    """Every sample row in exactly one single-project batch of at most *size* rows.

    Each project's rows are shuffled from entry-id order by one
    ``random.Random(seed)``, projects in name order, then split into the
    fewest batches of at most *size*, whose sizes differ by at most one.
    """
    if size < 1:
        raise ValueError(f'batch size {size} is not a positive integer')
    by_project: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sorted(sample_rows, key=lambda row: row['entry_id']):
        by_project[row['project']].append(row)
    draws = random.Random(seed)
    batches: list[Batch] = []
    for project, rows in sorted(by_project.items()):
        draws.shuffle(rows)
        count = math.ceil(len(rows) / size)
        batches += [
            Batch(
                name=f'{project}-{index:02d}', project=project,
                rows=tuple(rows[index * len(rows) // count:(index + 1) * len(rows) // count]),
            )
            for index in range(count)
        ]
    return batches


Pair = tuple[str, str]
"""A child and the parent it is attached under: (entry id, target id)."""


def load_texts(root: Path, projects: Mapping[str, str]) -> dict[Pair, _rounds.ShardEntry]:
    """The child and parent texts the raters saw, on the current basis, from a rating scratch.

    *projects* maps every entry id in the scratch's ``texts.json`` to its project.
    """
    return {
        (entry.entry_id, entry.target_id): entry
        for entry in _rounds.RatingScratch(root).entries(projects).values()
        if entry.basis == _rubric.Basis.CURRENT
    }


_STAGE1_HEADER = '## Reconciliation Run — Stage 1: Memory Consolidation'
SOAK_DATE = '2026-10-19'

_WORKLIST_CAVEAT = (
    f'Each child below is attached under the parent shown. Both texts are as of the {SOAK_DATE}'
    ' write-triage soak, not as the store holds them now; judge each pair as shown. Records'
    ' carry no category here, because the rated texts were captured without one.'
)

_WORKLIST_TASK = (
    'Decide one repair for every child in the worklist above, by the replay rules at the end'
    ' of your instructions. Return the decisions JSON as your final response and nothing else.'
)


def _child_kind(outcome: str) -> str:
    return SIGHTING_KIND if outcome == OUTCOME_RESTATED else AMENDMENT_KIND


def _child_block(position: int, total: int, row: Mapping[str, Any], texts: _rounds.ShardEntry) -> str:
    contested = row['outcome'] == OUTCOME_CONTESTED
    metadata: dict[str, Any] = {PARENT_ID_KEY: row['target_id'], 'kind': _child_kind(row['outcome'])}
    if contested:
        metadata[CONTESTED_METADATA_KEY] = True
    child, _ = format_memories([{'id': row['entry_id'], 'memory': texts.child_text, 'metadata': metadata}])
    parent, _ = format_memories([{'id': row['target_id'], 'memory': texts.parent_text, 'metadata': {}}])
    return '\n'.join([
        f'#### child {row["entry_id"]} ({position} of {total})',
        f'triage_outcome={row["outcome"]} {CONTESTED_METADATA_KEY}={"true" if contested else "false"}',
        'Child:',
        child,
        'Parent:',
        parent,
        '',
    ])


def render_worklist_payload(
    batch: Batch, texts: Mapping[Pair, _rounds.ShardEntry], *, project_id: str, project_root: str,
) -> str:
    """*batch* as a Stage-1 payload: the Stage-1 header, then one block per child and its parent."""
    total = len(batch.rows)
    return '\n'.join([
        _STAGE1_HEADER,
        f'## Project: {project_id}',
        '',
        f'### Replay worklist: attached children ({total}), batch {batch.name}',
        _WORKLIST_CAVEAT,
        '',
        *(
            _child_block(position, total, row, texts[row['entry_id'], row['target_id']])
            for position, row in enumerate(batch.rows, start=1)
        ),
        f'Use project_root="{project_root}" for tasks scoped to this project.',
        '',
        '## Your Task',
        _WORKLIST_TASK,
        '',
    ])


def _allowed_lines() -> str:
    return '\n'.join(
        f'- {outcome}: {", ".join(sorted(action.label for action in _scoring.allowed_actions(outcome)))}'
        for outcome in (OUTCOME_RESTATED, OUTCOME_AMENDED, OUTCOME_CONTESTED)
    )


_REPLAY_ADDENDUM = f"""

## Replay mode: read-only adjudication of attached children

This run replaces the consolidation task above. It is a read-only replay:
every store write is blocked, no recon-report server is attached, and nothing
you would normally record is kept. Your only output is the decisions JSON.

The worklist names children that write triage attached under a parent, each
with the outcome triage gave it: restated (a sighting, counted under the
parent but never shown), amended (shown to the parent's reader as an
amendment) or contested (shown as a dispute of the parent, {CONTESTED_METADATA_KEY}=true).
Judge each pair as shown, as of the soak. You may use your read tools for
context: the task store, memory metadata, other memories and the code.

Give exactly one decision per worklist child, chosen from these actions:

- `{Decision.KEEP}`: the outcome is right; a reader of the parent is served as filed.
- `{Decision.DETACH}`: the child's central claim is not its parent's claim, and it disputes
  nothing in it; it belongs standalone, not under this parent.
- `{Decision.CONTEST_UPHELD}`: the child shows something the parent states is false,
  outdated or replaced; the contest is right.
- `{Decision.NOT_A_CONTEST}`: a flagged contest disputes nothing in its parent; the child
  is about the parent's claim and should read as an amendment.
- `{Decision.RELABEL}` with relabel_to `{OUTCOME_CONTESTED}`: a sighting or amendment that does
  dispute something its parent states.
- `{Decision.RELABEL}` with relabel_to `{OUTCOME_AMENDED}`: a sighting carrying detail a reader
  acting on the parent needs, which a sighting hides.
- `{Decision.SUPERSEDED_PARENT}`: the child declares that it replaces its parent; it belongs
  standalone, with the parent marked superseded by it.

The actions each outcome allows:
{_allowed_lines()}
`{Decision.RELABEL}->X` is decision {Decision.RELABEL} with relabel_to X; relabel_to is
null for every other decision.

Answer with one JSON object: {{"decisions": [{{"child_id": ..., "decision": ...,
"relabel_to": ..., "reason": "<one line>", "evidence": ["<what you read>", ...]}}]}}.
"""



REPLAY_SYSTEM_PROMPT = STAGE1_SYSTEM_PROMPT + _REPLAY_ADDENDUM
"""Stage 1's system prompt verbatim, then the replay's vocabulary and output rules."""

REPLAY_DISALLOWED: tuple[str, ...] = tuple(STAGE3_DISALLOWED)
"""The read-only stage contract: Stage 1's deny list plus every memory, proposal and journal writer."""

REPLAY_OUTPUT_SCHEMA: Mapping[str, Any] = MappingProxyType({
    'type': 'object',
    'properties': {
        'decisions': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'child_id': {'type': 'string'},
                    'decision': {'type': 'string', 'enum': [decision.value for decision in Decision]},
                    'relabel_to': {'enum': [*sorted(_scoring.RELABEL_TARGETS), None]},
                    'reason': {'type': 'string'},
                    'evidence': {'type': 'array', 'items': {'type': 'string'}},
                },
                'required': ['child_id', 'decision', 'relabel_to', 'reason', 'evidence'],
            },
        },
    },
    'required': ['decisions'],
})
