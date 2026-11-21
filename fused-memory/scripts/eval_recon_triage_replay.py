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

import argparse
import asyncio
import hashlib
import json
import logging
import math
import random
import sys
from collections import defaultdict
from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from shared.config_dir import TaskConfigDir
from shared.safe_io import atomic_write_text
from shared.usage_gate import UsageGate

from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
from fused_memory.maintenance._utils import override_config_path
from fused_memory.models.scope import configured_known_projects_map
from fused_memory.reconciliation.cli_stage_runner import (
    STAGE3_DISALLOWED,
    StageResult,
    run_stage_via_cli,
)
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.stage1_payload_format import format_memories
from fused_memory.reconciliation.stages.base import recon_core_mcp_servers
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


SAMPLE_HEADER = 'recon_triage_replay_sample'
DECISIONS_HEADER = 'recon_triage_replay_decisions'


def sample_header(
    *, seed: int, allocation: Mapping[str, int], inputs: Mapping[str, Path],
) -> dict[str, Any]:
    """The sample file's first line: the seed, the allocation and each input file's sha256."""
    return {
        'header': SAMPLE_HEADER,
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


@dataclass(frozen=True)
class BatchDecisions:
    """A batch reply that parsed: one decision row per child, in the batch's order."""

    batch: str
    rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class BatchFailure:
    """A batch reply that did not parse; *reason* names the child and value at fault."""

    batch: str
    reason: str


class _Refused(ValueError):
    pass


def parse_batch_reply(batch: Batch, report: Mapping[str, Any]) -> BatchDecisions | BatchFailure:
    """*report*'s decisions for *batch*, or why the whole batch failed.

    Every child must be decided exactly once, with an action its triage
    outcome allows (``recon_triage_replay_scoring.action_for``); anything else
    fails the batch, so a partial reply never scores as a partial result.
    """
    try:
        return BatchDecisions(batch=batch.name, rows=_decision_rows(batch, report))
    except _Refused as refused:
        return BatchFailure(batch=batch.name, reason=str(refused))


def _decision_rows(batch: Batch, report: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    decisions = report.get('decisions')
    if not isinstance(decisions, list):
        raise _Refused(f'the reply has no decisions list (keys: {sorted(report)})')
    children = {row['entry_id']: row for row in batch.rows}
    decided: dict[str, dict[str, Any]] = {}
    for position, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            raise _Refused(f'decision {position} is not an object: {decision!r}')
        child = decision.get('child_id')
        if not isinstance(child, str) or child not in children:
            raise _Refused(f'decision {position} names child {child!r}, which is not in the batch')
        if child in decided:
            raise _Refused(f'child {child!r} is decided twice')
        decided[child] = _decision_row(children[child], decision)
    missing = [entry for entry in children if entry not in decided]
    if missing:
        raise _Refused(f'children {missing} have no decision')
    return tuple(decided[entry] for entry in children)


def _decision_row(row: Mapping[str, Any], decision: Mapping[str, Any]) -> dict[str, Any]:
    child = row['entry_id']
    reason, evidence = decision.get('reason'), decision.get('evidence')
    if not isinstance(reason, str):
        raise _Refused(f'child {child!r}: reason {reason!r} is not a string')
    if not (isinstance(evidence, list) and all(isinstance(item, str) for item in evidence)):
        raise _Refused(f'child {child!r}: evidence {evidence!r} is not a list of strings')
    word, relabel_to = decision.get('decision'), decision.get('relabel_to')
    try:
        action = _scoring.action_for(row['outcome'], str(word), relabel_to)
    except ValueError as exc:
        raise _Refused(f'child {child!r}: {exc}') from exc
    return {
        'entry_id': child,
        'target_id': row['target_id'],
        'decision': action.decision.value,
        'relabel_to': action.relabel_to,
        'reason': reason,
        'evidence': list(evidence),
    }


class ReplayRefused(Exception):
    """The replay refused its inputs, or a scratch that does not match them."""


@dataclass(frozen=True)
class Worklist:
    """One batch as recon is asked it: the batch and its Stage-1 payload."""

    batch: Batch
    payload: str


def worklists(
    batches: Iterable[Batch],
    texts: Mapping[Pair, _rounds.ShardEntry],
    *,
    project_roots: Mapping[str, str],
) -> list[Worklist]:
    """Each batch rendered for its project's root; refused for a project with no root or a child with no text."""
    rendered: list[Worklist] = []
    for batch in batches:
        root = project_roots.get(batch.project)
        if root is None:
            raise ReplayRefused(
                f'batch {batch.name}: project {batch.project!r} has no project root (known:'
                f' {sorted(project_roots)}); pass --project-root {batch.project}=PATH'
            )
        unrated = [
            row['entry_id'] for row in batch.rows if (row['entry_id'], row['target_id']) not in texts
        ]
        if unrated:
            raise ReplayRefused(f'batch {batch.name}: children {unrated} have no rated current-basis text')
        rendered.append(Worklist(
            batch=batch,
            payload=render_worklist_payload(batch, texts, project_id=batch.project, project_root=root),
        ))
    return rendered


@dataclass(frozen=True)
class AskReply:
    """What one ask of recon handed back: its report, what it cost, and any run error."""

    report: Mapping[str, Any]
    cost_usd: float
    error: str
    model: str


class Asker(Protocol):
    """An arm: asks one worklist and replies. *identity* is recorded in the scratch plan."""

    @property
    def identity(self) -> Mapping[str, str]: ...

    async def __call__(self, worklist: Worklist) -> AskReply: ...


RunStage = Callable[..., Awaitable[StageResult]]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class StageCliAsker:
    """Asks recon one worklist through the real Stage-1 CLI invocation, read-only.

    Each ask runs under a fresh session id and its own config dir under
    *config_base*, so every transcript is kept; *usage_gate* is the gate
    recon's harness runs its stages under.
    """

    def __init__(
        self,
        *,
        config: ReconciliationConfig,
        model: str,
        cwd: Path,
        config_base: Path,
        run_stage: RunStage = run_stage_via_cli,
        mcp_candidates: list[Path] | None = None,
        usage_gate: UsageGate | None = None,
    ) -> None:
        self._config = config
        self._model = model
        self._cwd = cwd
        self._config_base = config_base
        self._run_stage = run_stage
        self._servers = recon_core_mcp_servers(mcp_candidates)
        self._usage_gate = usage_gate

    @property
    def identity(self) -> Mapping[str, str]:
        return {
            'asker': 'stage', 'model': self._model,
            'system_prompt_sha256': _sha256_text(REPLAY_SYSTEM_PROMPT),
        }

    async def __call__(self, worklist: Worklist) -> AskReply:
        session_id = str(uuid4())
        result = await self._run_stage(
            system_prompt=REPLAY_SYSTEM_PROMPT,
            payload=worklist.payload,
            disallowed_tools=list(REPLAY_DISALLOWED),
            config=self._config,
            mcp_config={'mcpServers': dict(self._servers)},
            usage_gate=self._usage_gate,
            model=self._model,
            cwd=self._cwd,
            output_schema=dict(REPLAY_OUTPUT_SCHEMA),
            session_id=session_id,
            config_dir=TaskConfigDir(
                task_id=f'recon-replay-{worklist.batch.name}-{session_id}', base_dir=self._config_base,
            ),
        )
        return AskReply(
            report=result.report, cost_usd=result.cost_usd, error=result.error,
            model=result.model or self._model,
        )


FAKE_ASKER = 'fake'
STAGE_ASKER = 'stage'


class FakeAsker:
    """A deterministic arm, pure in the payload: each child's action is picked from
    those its outcome allows by the sha256 of the payload and the child's id."""

    @property
    def identity(self) -> Mapping[str, str]:
        return {'asker': FAKE_ASKER}

    async def __call__(self, worklist: Worklist) -> AskReply:
        return AskReply(
            report={'decisions': [_fake_decision(worklist.payload, row) for row in worklist.batch.rows]},
            cost_usd=0.0, error='', model=FAKE_ASKER,
        )


def _fake_decision(payload: str, row: Mapping[str, Any]) -> dict[str, Any]:
    choices = sorted(_scoring.allowed_actions(row['outcome']), key=lambda action: action.label)
    digest = hashlib.sha256(f'{payload}\0{row["entry_id"]}'.encode()).digest()
    action = choices[digest[0] % len(choices)]
    return {
        'child_id': row['entry_id'], 'decision': action.decision.value,
        'relabel_to': action.relabel_to, 'reason': 'fake arm: payload hash', 'evidence': [],
    }


class BatchStatus(StrEnum):
    """Whether a batch's recorded reply parsed."""

    OK = 'ok'
    FAILED = 'failed'


def _plan_path(scratch: Path) -> Path:
    return scratch / 'plan.json'


def _batch_path(scratch: Path, name: str) -> Path:
    return scratch / 'batches' / f'{name}.json'


def _write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + '\n', mkdir=True)


def _replay_plan(worklists: Sequence[Worklist], asker: Asker) -> dict[str, Any]:
    return {
        'asker': dict(asker.identity),
        'batches': {
            worklist.batch.name: {
                'project': worklist.batch.project,
                'children': [[row['entry_id'], row['target_id']] for row in worklist.batch.rows],
                'payload_sha256': _sha256_text(worklist.payload),
            }
            for worklist in worklists
        },
    }


def _claim_plan(scratch: Path, plan: Mapping[str, Any]) -> None:
    """Record *plan* in a fresh scratch, or refuse a scratch that records another."""
    path = _plan_path(scratch)
    if not path.exists():
        _write_json(path, plan)
        return
    recorded = json.loads(path.read_text(encoding='utf-8'))
    differing = sorted(key for key in {*plan, *recorded} if plan.get(key) != recorded.get(key))
    if differing:
        raise ReplayRefused(
            f'{path} records a different replay plan (differing: {differing}); its batches were'
            ' asked under other batches, payloads or arm. Use a fresh --scratch.'
        )


def _recorded(scratch: Path, name: str) -> dict[str, Any] | None:
    path = _batch_path(scratch, name)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


async def _asked(asker: Asker, worklist: Worklist) -> dict[str, Any]:
    """*worklist*'s batch record: the reply, parsed, or why it failed."""
    try:
        reply = await asker(worklist)
    except Exception as exc:  # recorded as a failed batch, which the next run re-asks
        reply = AskReply(report={}, cost_usd=0.0, error=f'{type(exc).__name__}: {exc}', model='')
    parsed = parse_batch_reply(worklist.batch, reply.report)
    decided = isinstance(parsed, BatchDecisions)
    return {
        'batch': worklist.batch.name,
        'project': worklist.batch.project,
        'status': (BatchStatus.OK if decided else BatchStatus.FAILED).value,
        'decisions': [dict(row) for row in parsed.rows] if decided else [],
        'reason': None if decided else parsed.reason,
        'model': reply.model,
        'cost_usd': reply.cost_usd,
        'error': reply.error,
        'payload_sha256': _sha256_text(worklist.payload),
        'report': dict(reply.report),
    }


async def run_replay(
    worklists: Sequence[Worklist],
    asker: Asker,
    scratch: Path,
    *,
    concurrency: int = 3,
    only: Collection[str] | None = None,
) -> dict[str, BatchStatus]:
    """Ask every worklist (or those *only* names) not yet parsed OK under *scratch*.

    The first run records the plan (batches, payload hashes, the arm's
    identity) in ``scratch/plan.json``; a later run must match it. Each
    batch's record is written as soon as its reply is in, so a rerun resumes.
    Returns each batch's status, in plan order.
    """
    if concurrency < 1:
        raise ReplayRefused(f'concurrency {concurrency} is not a positive integer')
    names = [worklist.batch.name for worklist in worklists]
    unknown = sorted(set(only or ()) - set(names))
    if unknown:
        raise ReplayRefused(f'--only names batches {unknown} that the plan does not hold; planned: {names}')
    _claim_plan(scratch, _replay_plan(worklists, asker))
    chosen = [worklist for worklist in worklists if only is None or worklist.batch.name in only]
    statuses: dict[str, BatchStatus] = {}
    gate = asyncio.Semaphore(concurrency)

    async def ask(worklist: Worklist) -> None:
        name = worklist.batch.name
        recorded = _recorded(scratch, name)
        if recorded is not None and recorded['status'] == BatchStatus.OK:
            statuses[name] = BatchStatus.OK
            return
        async with gate:
            record = await _asked(asker, worklist)
        _write_json(_batch_path(scratch, name), record)
        statuses[name] = BatchStatus(record['status'])

    await asyncio.gather(*(ask(worklist) for worklist in chosen))
    return {name: statuses[name] for name in names if name in statuses}


def _calibration_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    decisions = record['decisions']
    share = record['cost_usd'] / len(decisions) if decisions else 0.0
    return [
        {**decision, 'batch': record['batch'], 'model': record['model'], 'cost_usd': share}
        for decision in decisions
    ]


def _write_jsonl(path: Path, header: Mapping[str, Any], rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_write_text(path, ''.join(json.dumps(line, sort_keys=True) + '\n' for line in (header, *rows)))


def collect(
    scratch: Path,
    out: Path,
    *,
    allow_missing: bool = False,
    tool_health: Any = None,
) -> dict[str, Any]:
    """Write every parsed decision under *scratch* to *out*, a header line first; return the header.

    Refused while a planned batch is missing or failed, unless *allow_missing*:
    its children then have no row, and the scorer counts them as no decision.
    Each row's ``cost_usd`` is its batch's cost shared evenly over the batch.
    """
    plan_path = _plan_path(scratch)
    if not plan_path.exists():
        raise ReplayRefused(f'{plan_path} does not exist: run the replay into this scratch first')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    records = {name: _recorded(scratch, name) for name in sorted(plan['batches'])}
    missing = [name for name, record in records.items() if record is None]
    failed = [
        name for name, record in records.items()
        if record is not None and record['status'] != BatchStatus.OK
    ]
    if (missing or failed) and not allow_missing:
        raise ReplayRefused(
            f'batches without decisions: missing {missing}, failed {failed}; re-run them, or'
            ' collect with --allow-missing'
        )
    asked = [record for record in records.values() if record is not None]
    rows = sorted(
        (row for record in asked if record['status'] == BatchStatus.OK for row in _calibration_rows(record)),
        key=lambda row: row['entry_id'],
    )
    header = {
        'header': DECISIONS_HEADER,
        'asker': plan['asker'],
        'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        'batches': len(records),
        'decided': len(rows),
        'missing_batches': missing,
        'failed_batches': failed,
        'cost_usd': sum(record['cost_usd'] for record in asked),
        'tool_health': tool_health,
    }
    _write_jsonl(out, header, rows)
    return header


# ── the CLI ────────────────────────────────────────────────────────────────

EXIT_OK = 0
EXIT_INCOMPLETE = 1
EXIT_REFUSED = 2
REFUSALS = (ReplayRefused, OSError, ValueError)

_CALIBRATION = Path('calibration')
DEFAULT_ITEMS = _CALIBRATION / 'write_triage_v2_soak_items.jsonl'
DEFAULT_VERDICTS = _CALIBRATION / 'write_triage_v2_soak_verdicts_v2_2.jsonl'
DEFAULT_STRATA = _CALIBRATION / 'write_triage_v2_soak_strata.json'
DEFAULT_SAMPLE = _CALIBRATION / 'recon_triage_replay_sample.jsonl'
DEFAULT_DECISIONS = _CALIBRATION / 'recon_triage_replay_decisions.jsonl'
DEFAULT_SCORES_JSON = _CALIBRATION / 'recon_triage_replay_scores.json'
DEFAULT_SCORES_MD = _CALIBRATION / 'recon_triage_replay_scores.md'
_DATA = Path('/home/leo/src/dark-factory/data')
DEFAULT_TEXTS_ROOT = _DATA / 'write-triage-v21-rebase-2026-10-21'
DEFAULT_TOUCHED = _DATA / 'write-triage-soak-2026-10-19' / 'team' / 'D' / 'touched.json'
DEFAULT_RUBRIC = 'v2.2'


def load_config(config_path: str | None) -> FusedMemoryConfig:
    """The config as its file reads now; ``$CONFIG_PATH`` when *config_path* is None."""
    with override_config_path(config_path):
        return FusedMemoryConfig()


def registered_project_roots(config: FusedMemoryConfig) -> Mapping[str, str]:
    """Project id to root, from the registry recon itself routes by."""
    return configured_known_projects_map(config.taskmaster)


def recon_usage_gate(config: ReconciliationConfig) -> UsageGate | None:
    """The usage gate recon's harness runs its stages under, when the config enables one."""
    return UsageGate(config.usage_cap) if config.usage_cap.enabled else None


@dataclass(frozen=True)
class ReplayEnv:
    """What the replay reaches beyond its arguments. The defaults are production's."""

    config_loader: Callable[[str | None], FusedMemoryConfig] = load_config
    project_registry: Callable[[FusedMemoryConfig], Mapping[str, str]] = registered_project_roots
    run_stage: RunStage = run_stage_via_cli
    mcp_candidates: Sequence[Path] | None = None
    usage_gate_for: Callable[[ReconciliationConfig], UsageGate | None] = recon_usage_gate


def _json(path: Path, kind: type) -> Any:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ReplayRefused(f'{path}: not JSON ({exc})') from exc
    if not isinstance(value, kind):
        raise ReplayRefused(f'{path}: not a JSON {kind.__name__}')
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayRefused(f'{path} line {line_no}: not JSON ({exc})') from exc
        if not isinstance(row, dict):
            raise ReplayRefused(f'{path} line {line_no}: not a JSON object')
        rows.append(row)
    return rows


def _headed_jsonl(path: Path, header: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _jsonl(path)
    if not rows or rows[0].get('header') != header:
        raise ReplayRefused(f'{path}: the first line is not a {header!r} header')
    return rows[0], rows[1:]


def _corpus_texts(root: Path, items: Iterable[Mapping[str, Any]]) -> dict[Pair, _rounds.ShardEntry]:
    try:
        return load_texts(root, {item['entry_id']: item['project'] for item in items})
    except KeyError as exc:
        raise ReplayRefused(f'{root / "texts.json"}: {exc} is not keyed by the corpus items') from exc


def _sample_command(args: argparse.Namespace, _env: ReplayEnv) -> int:
    items = _jsonl(args.items)
    rows = draw_sample(
        items, _scoring.corpus_derivations(items, _jsonl(args.verdicts)),
        allocation=DEFAULT_ALLOCATION, seed=SAMPLE_SEED,
        touched=touched_states(_json(args.touched, list)),
    )
    texts = _corpus_texts(args.texts_root, items)
    unrated = [row['entry_id'] for row in rows if (row['entry_id'], row['target_id']) not in texts]
    if unrated:
        raise ReplayRefused(f'sampled children {unrated} have no rated current-basis text')
    header = sample_header(seed=SAMPLE_SEED, allocation=DEFAULT_ALLOCATION, inputs={
        'items': args.items, 'verdicts': args.verdicts, 'texts': args.texts_root / 'texts.json',
        'touched': args.touched,
    })
    _write_jsonl(args.out, header, rows)
    headline = sum(row['headline'] for row in rows)
    print(f'sample: {headline} headline and {len(rows) - headline} report-only children -> {args.out}')
    return EXIT_OK


def _asker(args: argparse.Namespace, env: ReplayEnv, config: FusedMemoryConfig) -> Asker:
    if args.asker == FAKE_ASKER:
        return FakeAsker()
    recon = config.reconciliation
    return StageCliAsker(
        config=recon,
        model=args.model or recon.agent_llm_model,
        cwd=args.cwd or Path(recon.explore_codebase_root),
        config_base=args.config_base or args.scratch / 'config',
        run_stage=env.run_stage,
        mcp_candidates=None if env.mcp_candidates is None else list(env.mcp_candidates),
        usage_gate=env.usage_gate_for(recon),
    )


def _run_command(args: argparse.Namespace, env: ReplayEnv) -> int:
    _, rows = _headed_jsonl(args.sample, SAMPLE_HEADER)
    texts = _corpus_texts(args.texts_root, _jsonl(args.items))
    config = env.config_loader(args.config)
    roots = {**env.project_registry(config), **dict(args.project_root or ())}
    planned = worklists(
        plan_batches(rows, size=args.batch_size, seed=SAMPLE_SEED), texts, project_roots=roots,
    )
    only = frozenset(args.only) if args.only else None

    async def replay() -> dict[str, BatchStatus]:
        return await run_replay(
            planned, _asker(args, env, config), args.scratch, concurrency=args.concurrency, only=only,
        )

    statuses = asyncio.run(replay())
    for name, status in statuses.items():
        print(f'{name}: {status.value}')
    return EXIT_OK if all(status is BatchStatus.OK for status in statuses.values()) else EXIT_INCOMPLETE


def _collect_command(args: argparse.Namespace, _env: ReplayEnv) -> int:
    tool_health = None if args.tool_health is None else _json(args.tool_health, dict)
    header = collect(args.scratch, args.out, allow_missing=args.allow_missing, tool_health=tool_health)
    print(
        f'collect: {header["decided"]} decisions from {header["batches"]} batches'
        f' (missing {header["missing_batches"]}, failed {header["failed_batches"]}) -> {args.out}'
    )
    return EXIT_OK


def _score_command(args: argparse.Namespace, _env: ReplayEnv) -> int:
    _, sample_rows = _headed_jsonl(args.sample, SAMPLE_HEADER)
    _, decisions = _headed_jsonl(args.decisions, DECISIONS_HEADER)
    report = _scoring.score_replay(
        sample_rows, _jsonl(args.items), _jsonl(args.verdicts), decisions,
        rubric=args.rubric, strata_populations=_json(args.strata, dict),
    )
    document = json.dumps(report, indent=2, sort_keys=True) + '\n'
    atomic_write_text(args.out_json, document)
    atomic_write_text(args.out_md, _scoring.render_markdown(json.loads(document)))
    print(f'score: {args.out_json} and {args.out_md}')
    return EXIT_OK


def _project_root(text: str) -> tuple[str, str]:
    project, separator, root = text.partition('=')
    if not (separator and project and root):
        raise argparse.ArgumentTypeError(f'{text!r} is not PROJECT=PATH')
    return project, root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='eval_recon_triage_replay.py',
        description='Replay recon Stage-1 adjudication of rated write-triage attaches, read-only.',
    )
    commands = parser.add_subparsers(dest='command', required=True)

    sample = commands.add_parser('sample', help='draw and write the frozen sample')
    sample.add_argument('--items', type=Path, default=DEFAULT_ITEMS)
    sample.add_argument('--verdicts', type=Path, default=DEFAULT_VERDICTS)
    sample.add_argument('--texts-root', type=Path, default=DEFAULT_TEXTS_ROOT)
    sample.add_argument('--touched', type=Path, default=DEFAULT_TOUCHED)
    sample.add_argument('--out', type=Path, default=DEFAULT_SAMPLE)

    run = commands.add_parser('run', help='ask recon each batch, resumably')
    run.add_argument('--sample', type=Path, default=DEFAULT_SAMPLE)
    run.add_argument('--items', type=Path, default=DEFAULT_ITEMS)
    run.add_argument('--texts-root', type=Path, default=DEFAULT_TEXTS_ROOT)
    run.add_argument('--scratch', type=Path, required=True)
    run.add_argument('--asker', choices=(STAGE_ASKER, FAKE_ASKER), default=STAGE_ASKER)
    run.add_argument('--batch-size', type=int, default=8)
    run.add_argument('--concurrency', type=int, default=3)
    run.add_argument('--only', action='append', metavar='BATCH', help='ask only this batch (repeatable)')
    run.add_argument('--config', help='fused-memory config YAML (default: $CONFIG_PATH)')
    run.add_argument('--model', help="default: the config's reconciliation.agent_llm_model")
    run.add_argument('--cwd', type=Path, help="default: the config's reconciliation.explore_codebase_root")
    run.add_argument('--project-root', type=_project_root, action='append', metavar='PROJECT=PATH',
                     help='a project root the registry lacks (repeatable)')
    run.add_argument('--config-base', type=Path, help='where per-ask config dirs go (default: SCRATCH/config)')

    gather = commands.add_parser('collect', help='write the parsed decisions')
    gather.add_argument('--scratch', type=Path, required=True)
    gather.add_argument('--out', type=Path, default=DEFAULT_DECISIONS)
    gather.add_argument('--allow-missing', action='store_true')
    gather.add_argument('--tool-health', type=Path, metavar='JSON',
                        help='a JSON object recording the tool-health check, kept in the header')

    score = commands.add_parser('score', help='score the committed sample and decisions')
    score.add_argument('--sample', type=Path, default=DEFAULT_SAMPLE)
    score.add_argument('--decisions', type=Path, default=DEFAULT_DECISIONS)
    score.add_argument('--items', type=Path, default=DEFAULT_ITEMS)
    score.add_argument('--verdicts', type=Path, default=DEFAULT_VERDICTS)
    score.add_argument('--strata', type=Path, default=DEFAULT_STRATA)
    score.add_argument('--rubric', default=DEFAULT_RUBRIC)
    score.add_argument('--out-json', type=Path, default=DEFAULT_SCORES_JSON)
    score.add_argument('--out-md', type=Path, default=DEFAULT_SCORES_MD)
    return parser


_COMMANDS: Mapping[str, Callable[[argparse.Namespace, ReplayEnv], int]] = MappingProxyType({
    'sample': _sample_command, 'run': _run_command, 'collect': _collect_command, 'score': _score_command,
})


def main(argv: Sequence[str] | None = None, *, env: ReplayEnv | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args, env or ReplayEnv())
    except REFUSALS as refusal:
        print(f'eval_recon_triage_replay refused: {refusal}', file=sys.stderr)
        return EXIT_REFUSED


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    sys.exit(main())
