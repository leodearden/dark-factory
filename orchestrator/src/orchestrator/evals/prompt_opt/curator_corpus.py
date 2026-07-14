"""Curator replay corpus builder (T5): tickets.db -> frontier-adjudicated +
human-spot-checked labeled CuratorReplayItems, split 2:1:7.

See plans/tier1-prompt-optimization-prd.md T5. Mirrors reviewer_trial's
corpus/mining/adjudication machinery (task 2495) but stays decoupled from
fused_memory: tickets.db rows are read as plain dicts by column name via
stdlib sqlite3 (read-only), and target_fingerprint/target_id use a local
minimal action representation -- no import of
``fused_memory.middleware.task_curator``. Every external effect (the
frontier label proposer) is dependency-injected so the builder is fully
hermetic in tests; see ``__main__.py`` for the operator-facing CLI that
wires the real tickets.db + a real frontier proposer.

Decisions != ground truth (PRD D-6): a ticket's RECORDED action/target
(persisted at task-creation time by the live curator) is retained only as
provenance -- gold labels always come from an injected frontier-adjudication
proposer, further checked by a deterministic, action-stratified human
spot-check subset (the Open-Q Sec9 tactical decision this task owns).
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from orchestrator.evals.reviewer_trial.adjudication import AdjudicationLog
from orchestrator.evals.reviewer_trial.mining import assign_split

logger = logging.getLogger(__name__)

__all__ = [
    'CuratorCorpusManifest',
    'CuratorReplayItem',
    'FrontierLabel',
    'FrontierProposerFn',
    'RecordedDecision',
    'build_curator_corpus',
    'propose_curator_label_frontier',
    'read_curator_decisions',
    'recover_recorded_action',
    'select_spot_check_subset',
]

_RECOVERABLE_ACTIONS = ('drop', 'combine', 'create')


@dataclass(frozen=True)
class RecordedDecision:
    """One recovered ticket decision -- provenance/weak signal, NOT a gold label.

    See PRD D-6: the recorded action is what the live curator historically
    decided, unverified -- ``build_curator_corpus`` always obtains the GOLD
    label from an injected frontier-adjudication proposer instead.
    """

    ticket_id: str
    candidate: dict
    action: str
    target_fingerprint: str | None
    target_id: str | None


def recover_recorded_action(
    status: str,
    result_json: str | None,
    task_id: str | None,
) -> tuple[str, str | None, str | None] | None:
    """Recover ``(action, target_fingerprint, target_id)`` from a ticket row.

    ``result_json['action']`` is authoritative when present and valid: drop
    AND combine BOTH persist ``status='combined'``, so status alone cannot
    disambiguate them -- only the embedded action can. Falls back to
    ``status == 'created' -> 'create'`` when *result_json* carries no
    recoverable action (the real add_task-result JSON has no ``'action'``
    key). ``target_id`` prefers ``result_json['id']`` (or the less common
    ``result_json['target_id']``), falling back to the ticket row's own
    *task_id* column when neither is present -- but only for drop/combine,
    since 'create' has no "target being combined into".

    Returns ``None`` (un-actionable, e.g. ``status='failed'``/``'pending'``,
    or a missing/unparseable *result_json* that isn't rescued by the
    'created' fallback) rather than raising -- callers skip these rows.
    """
    parsed: dict | None = None
    if result_json:
        try:
            candidate = json.loads(result_json)
        except (json.JSONDecodeError, TypeError):
            candidate = None
        if isinstance(candidate, dict):
            parsed = candidate

    action: str | None = None
    target_fingerprint: str | None = None
    target_id: str | None = None

    if parsed is not None:
        raw_action = parsed.get('action')
        if raw_action in _RECOVERABLE_ACTIONS:
            action = raw_action
            target_fingerprint = parsed.get('target_fingerprint')
            target_id = parsed.get('id') or parsed.get('target_id')

    if action is None and status == 'created':
        action = 'create'

    if action is None:
        return None

    if action in ('drop', 'combine') and target_id is None:
        target_id = task_id

    return (action, target_fingerprint, target_id)


def read_curator_decisions(db_path: Path) -> list[RecordedDecision]:
    """Read every recoverable curator decision from *db_path* (read-only).

    Opens the tickets.db directly via stdlib ``sqlite3`` (no writes, no
    ``fused_memory`` import) and skips rows :func:`recover_recorded_action`
    deems un-actionable. A row whose ``candidate_json`` fails to parse as a
    dict degrades to an empty candidate ``{}`` rather than dropping the
    row entirely -- the recovered action is still a meaningful signal even
    when the candidate payload itself is malformed.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            'SELECT ticket_id, candidate_json, status, task_id, result_json FROM tickets'
        ).fetchall()
    finally:
        conn.close()

    decisions: list[RecordedDecision] = []
    for row in rows:
        recovered = recover_recorded_action(row['status'], row['result_json'], row['task_id'])
        if recovered is None:
            continue
        action, target_fingerprint, target_id = recovered

        try:
            candidate = json.loads(row['candidate_json'])
        except (json.JSONDecodeError, TypeError):
            candidate = None
        if not isinstance(candidate, dict):
            logger.warning(
                'read_curator_decisions: unparseable candidate_json for ticket %s', row['ticket_id'],
            )
            candidate = {}

        decisions.append(RecordedDecision(
            ticket_id=row['ticket_id'],
            candidate=candidate,
            action=action,
            target_fingerprint=target_fingerprint,
            target_id=target_id,
        ))

    return decisions


# ---------------------------------------------------------------------------
# Human spot-check subset (Open-Q Sec9 -- this task's tactical decision)
# ---------------------------------------------------------------------------

_DEFAULT_SPOT_CHECK_FRACTION = 0.2
_DEFAULT_SPOT_CHECK_MINIMUM = 5
_DEFAULT_SPOT_CHECK_CAP = 200


def select_spot_check_subset(
    decisions: list[RecordedDecision],
    *,
    fraction: float = _DEFAULT_SPOT_CHECK_FRACTION,
    minimum: int = _DEFAULT_SPOT_CHECK_MINIMUM,
    cap: int = _DEFAULT_SPOT_CHECK_CAP,
    seed: int = 0,
    stratify_by_action: bool = True,
) -> list[str]:
    """Deterministic human spot-check subset of *decisions* (``ticket_id``s).

    The Open-Q Sec9 tactical decision this task owns: bound human review
    effort while keeping label confidence across the action distribution.
    By default (``stratify_by_action=True``) samples independently within
    each recorded-action stratum -- ``~fraction`` of the stratum, floored at
    *minimum* (so a stratum smaller than *minimum* is taken in full rather
    than padded past its own size) -- so every present action ends up
    represented in the human-reviewed subset rather than the sample being
    dominated by whichever action is most common. The combined subset is
    then trimmed to *cap* if it would otherwise exceed it, bounding total
    human effort regardless of corpus size.

    Same *decisions* + *seed* always yields the same subset (``random.Random``
    seeded per stratum, no wall-clock/real randomness) -- required for a
    reproducible spot-check protocol; a different *seed* yields a different
    sample.

    ``stratify_by_action=False`` samples flatly across all *decisions*
    instead (one "stratum" holding everything) -- provided for parity with
    a non-stratified sampling mode; :func:`build_curator_corpus` always uses
    the stratified default so drop/combine/create are each represented.
    """
    if not decisions:
        return []

    if stratify_by_action:
        groups: dict[str, list[str]] = {}
        for d in decisions:
            groups.setdefault(d.action, []).append(d.ticket_id)
    else:
        groups = {'_all': [d.ticket_id for d in decisions]}

    selected: list[str] = []
    for key in sorted(groups):
        ids = sorted(groups[key])
        rng = random.Random(f'{seed}:{key}')
        shuffled = ids[:]
        rng.shuffle(shuffled)
        k = min(len(shuffled), max(minimum, round(len(shuffled) * fraction)))
        selected.extend(shuffled[:k])

    if len(selected) > cap:
        # Reshuffle the combined selection (still seed-reproducible) rather
        # than truncating in per-stratum append order, which would silently
        # starve later-sorted strata whenever the combined set exceeds cap.
        rng = random.Random(f'{seed}:trim')
        rng.shuffle(selected)
        selected = selected[:cap]

    return sorted(set(selected))


# ---------------------------------------------------------------------------
# CuratorReplayItem + CuratorCorpusManifest (labeled corpus data model)
# ---------------------------------------------------------------------------

_ANNOTATIONS_DIRNAME = 'annotations'


@dataclass
class CuratorReplayItem:
    """One item in the curator replay corpus.

    ``recorded_action``/``recorded_target_fingerprint``/``recorded_target_id``
    are the ticket's historical (unverified) curator decision -- retained
    only as provenance/weak signal (PRD D-6: decisions != ground truth).
    ``gold_action``/``gold_target_fingerprint``/``gold_target_id`` are the
    frontier-adjudicated (+ possibly human-spot-checked) label
    :class:`~orchestrator.evals.prompt_opt.curator_scorer.CuratorActionScorer`
    actually grades against -- NEVER the recorded fields.
    """

    ticket_id: str
    candidate: dict
    recorded_action: str
    recorded_target_fingerprint: str | None
    recorded_target_id: str | None
    gold_action: str
    gold_target_fingerprint: str | None
    gold_target_id: str | None
    split: str | None = None
    provenance: dict = field(default_factory=dict)


@dataclass
class CuratorCorpusManifest:
    """Collection of :class:`CuratorReplayItem`\\ s with save/load.

    Mirrors ``reviewer_trial.corpus.CorpusManifest``'s on-disk shape: a
    top-level ``manifest.json`` listing ``(ticket_id, split)`` plus one
    ``annotations/<ticket_id>.json`` file per item holding the full item
    (candidate, recorded_*, gold_*, provenance).
    """

    items: list[CuratorReplayItem] = field(default_factory=list)
    version: str = '1.0'
    split_seed: int | None = None

    def get_item(self, ticket_id: str) -> CuratorReplayItem | None:
        for item in self.items:
            if item.ticket_id == ticket_id:
                return item
        return None

    def save(self, path: Path) -> None:
        """Save the manifest + one annotation file per item."""
        corpus_dir = path.parent
        ann_dir = corpus_dir / _ANNOTATIONS_DIRNAME
        ann_dir.mkdir(parents=True, exist_ok=True)

        manifest_data: dict[str, Any] = {'version': self.version, 'items': []}
        if self.split_seed is not None:
            manifest_data['split_seed'] = self.split_seed

        for item in self.items:
            ann_file = ann_dir / f'{item.ticket_id}.json'
            ann_file.write_text(json.dumps(asdict(item), indent=2))

            entry: dict[str, Any] = {'ticket_id': item.ticket_id}
            if item.split is not None:
                entry['split'] = item.split
            manifest_data['items'].append(entry)

        path.write_text(json.dumps(manifest_data, indent=2))

    @classmethod
    def load(cls, path: Path) -> CuratorCorpusManifest:
        """Load a manifest + its per-item annotation files."""
        corpus_dir = path.parent
        raw = json.loads(path.read_text())

        items: list[CuratorReplayItem] = []
        for entry in raw['items']:
            ticket_id = entry['ticket_id']
            ann_file = corpus_dir / _ANNOTATIONS_DIRNAME / f'{ticket_id}.json'
            data = json.loads(ann_file.read_text())
            items.append(CuratorReplayItem(**data))

        return cls(
            items=items,
            version=raw.get('version', '1.0'),
            split_seed=raw.get('split_seed'),
        )


# ---------------------------------------------------------------------------
# build_curator_corpus (frontier-adjudicated gold labels, PRD D-6)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrontierLabel:
    """A frontier-proposed GOLD label for one candidate: action + target.

    Never the recorded ticket decision (PRD D-6) -- always produced by the
    injected *frontier_proposer* seam in :func:`build_curator_corpus`.
    """

    action: str
    target_fingerprint: str | None = None
    target_id: str | None = None
    justification: str = ''


FrontierProposerFn = Callable[[dict], Awaitable[FrontierLabel]]


def _sample_stratified(decisions: list[RecordedDecision], n: int, seed: int) -> list[RecordedDecision]:
    """Deterministically sample down to at most *n* decisions, round-robin
    across recorded-action strata.

    Round-robin (rather than proportional allocation) keeps a bounded *n*
    representative of drop/combine/create even when one action dominates
    the raw ticket history, without needing precise proportional math.
    Returns *decisions* unchanged when it already has <= *n* items.
    """
    if n >= len(decisions):
        return list(decisions)

    by_action: dict[str, list[RecordedDecision]] = {}
    for d in decisions:
        by_action.setdefault(d.action, []).append(d)

    shuffled_by_action: dict[str, list[RecordedDecision]] = {}
    for action, items in by_action.items():
        ordered = sorted(items, key=lambda d: d.ticket_id)
        rng = random.Random(f'{seed}:sample:{action}')
        rng.shuffle(ordered)
        shuffled_by_action[action] = ordered

    sampled: list[RecordedDecision] = []
    cursors = dict.fromkeys(shuffled_by_action, 0)
    actions_cycle = sorted(shuffled_by_action)
    while len(sampled) < n:
        progressed = False
        for action in actions_cycle:
            if len(sampled) >= n:
                break
            cursor = cursors[action]
            pool = shuffled_by_action[action]
            if cursor < len(pool):
                sampled.append(pool[cursor])
                cursors[action] = cursor + 1
                progressed = True
        if not progressed:
            break  # every stratum exhausted before reaching n

    return sampled


async def build_curator_corpus(
    db_path: Path,
    *,
    n: int,
    seed: int,
    spot_check_size: int,
    frontier_proposer: FrontierProposerFn,
) -> tuple[CuratorCorpusManifest, AdjudicationLog]:
    """Build a labeled curator replay corpus from *db_path* (T5).

    Reads recorded decisions from tickets.db (:func:`read_curator_decisions`),
    optionally downsamples to *n* stratified by recorded action
    (:func:`_sample_stratified`), obtains a GOLD label for every sampled
    candidate from the injected *frontier_proposer* (NEVER the recorded
    decision -- PRD D-6: decisions != ground truth), flags a deterministic
    human spot-check subset capped at *spot_check_size*
    (:func:`select_spot_check_subset`), assigns a stable per-ticket 2:1:7
    split (reusing ``reviewer_trial.mining.assign_split``), and returns the
    assembled manifest alongside a companion :class:`AdjudicationLog`
    recording one frontier-proposal entry per item.

    Fully hermetic given a fake *frontier_proposer*: makes no real LLM call
    and only reads (never writes) *db_path*.
    """
    decisions = read_curator_decisions(db_path)
    sampled = _sample_stratified(decisions, n, seed)

    spot_check_ids = set(select_spot_check_subset(sampled, cap=spot_check_size, seed=seed))

    log = AdjudicationLog()
    items: list[CuratorReplayItem] = []
    for decision in sampled:
        label = await frontier_proposer(decision.candidate)
        item = CuratorReplayItem(
            ticket_id=decision.ticket_id,
            candidate=decision.candidate,
            recorded_action=decision.action,
            recorded_target_fingerprint=decision.target_fingerprint,
            recorded_target_id=decision.target_id,
            gold_action=label.action,
            gold_target_fingerprint=label.target_fingerprint,
            gold_target_id=label.target_id,
            provenance={'source_db': str(db_path)},
        )
        items.append(item)
        log.append(
            decision.ticket_id,
            frontier_proposal=[asdict(label)],
            in_spot_check_subset=decision.ticket_id in spot_check_ids,
            frontier_model='frontier_proposer',
            notes=label.justification,
        )

    assignment = assign_split([item.ticket_id for item in items], seed=str(seed))
    for item in items:
        item.split = assignment[item.ticket_id]

    manifest = CuratorCorpusManifest(items=items, split_seed=seed)
    return manifest, log


# JSON schema for the frontier curator-label proposal LLM output -- mirrors
# reviewer_trial.mining's _FRONTIER_LABEL_SCHEMA invoke_agent pattern but
# proposes a drop/combine/create action rather than a list of issues.
_FRONTIER_CURATOR_LABEL_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'action': {'type': 'string', 'enum': list(_RECOVERABLE_ACTIONS)},
        'target_fingerprint': {'type': ['string', 'null']},
        'target_id': {'type': ['string', 'null']},
        'justification': {'type': 'string'},
    },
    'required': ['action', 'justification'],
}


def _parse_frontier_label(data: Any) -> FrontierLabel | None:
    """Pure mapping: frontier structured-output dict -> :class:`FrontierLabel`.

    Returns ``None`` (never raises) when *data* isn't a dict, has no
    ``'action'`` key, or ``'action'`` isn't one of drop/combine/create.
    """
    if not isinstance(data, dict):
        return None
    action = data.get('action')
    if action not in _RECOVERABLE_ACTIONS:
        return None
    return FrontierLabel(
        action=action,
        target_fingerprint=data.get('target_fingerprint'),
        target_id=data.get('target_id'),
        justification=data.get('justification', ''),
    )


async def propose_curator_label_frontier(
    candidate: dict,
    model: str = 'opus',
    oauth_token: str | None = None,
    max_turns: int = 15,
) -> FrontierLabel:
    """Propose a GOLD curator label for *candidate* via a frontier model.

    Build-time only -- never called from a hermetic test (those inject a
    fake ``FrontierProposerFn``). Mirrors
    ``reviewer_trial.mining.propose_labels_frontier``'s ``invoke_agent``
    pattern: no tools, structured JSON output, cost always reported (see
    ``result.cost_usd``, surfaced to CLI callers via ``__main__.py``).

    Never raises: an unparseable/malformed response degrades to
    ``action='create'`` -- the live curator's own best-effort fallback
    semantics (a spurious "create" is far cheaper to have in a corpus than
    silently losing the candidate).
    """
    from orchestrator.agents.invoke import invoke_agent

    candidate_json = json.dumps(candidate, indent=2)
    prompt = f"""\
You are adjudicating a task-curation decision for an evaluation corpus.

## Candidate task

```json
{candidate_json}
```

## Instructions

Decide ONE action for this candidate: "drop" (already covered by existing
work), "combine" (should be folded into an existing pending task), or
"create" (genuinely new work). For "combine", also give target_fingerprint
(the verbatim title of the task it should combine into, if inferable from
the candidate's own context) and target_id (if inferable). Give a brief
justification. Output your decision as JSON.
"""
    system_prompt = (
        'You are a meticulous adjudicator proposing ground-truth curator-decision '
        'labels for an evaluation corpus. Be precise and conservative -- default to '
        '"create" when uncertain. Output ONLY valid JSON.'
    )

    result = await invoke_agent(
        prompt=prompt,
        system_prompt=system_prompt,
        cwd=Path('/home/leo/src/dark-factory'),
        model=model,
        max_turns=max_turns,
        max_budget_usd=1.0,
        output_schema=_FRONTIER_CURATOR_LABEL_SCHEMA,
        effort='high',
        allowed_tools=[],  # no tools needed -- all context is in the prompt
        oauth_token=oauth_token,
    )

    data = result.structured_output
    if not data:
        try:
            data = json.loads(result.output)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                'Frontier curator-label proposal produced unparseable output: %s',
                result.output[:200],
            )
            data = None

    label = _parse_frontier_label(data)
    if label is None:
        return FrontierLabel(action='create', justification='unparseable-frontier-output')
    return label
