"""LLM-judged task curator gate.

At task-creation time, the curator decides whether a candidate task should:

- **drop** — already covered by an existing non-cancelled task; return that task's id
  instead of creating a new one.
- **combine** — rewrite a pending task to subsume the candidate's work, producing a
  single coherent task instead of two fragments.
- **create** — genuinely new work; proceed with task creation.

The decision is made by an LLM against a layered corpus of existing tasks (review-chain
anchor + the parent's other follow-ups for an agent-followup candidate + module-lock
pool + embedding neighbors + dependency neighbors). See
docs/reify-task-fragmentation-report-2026-04-11.txt for the motivating analysis.

When the curator cannot judge a candidate it returns ``action="defer"``
(:meth:`CuratorDecision.deferred`, with the
:class:`~fused_memory.middleware.curation_deferral.DeferReason` that says why) rather
than creating anything un-deduped. The interceptor parks a deferred
submission and retries it until it gets a verdict, so task creation is never blocked
and never silently skips dedupe. A response that arrives but cannot be used still
degrades to ``action="create"``, marked ``degraded``.

A timed-out LLM call is classified from its own transcript when the curator is gated
(task 3995): a pre-turn stall (0 assistant turns) is the one outage that still creates,
marked ``degraded_by_zot``, and files a ZOT escalation that folds into one record per
project; a run killed with progress returns
a verdict the CLI accepted (its ``structured_output`` attachment) if the transcript
holds one, else escalates with its real turn count and tool sequence; any non-schema
tool use logs a loud WARNING. A schema-tool denial is always escalated, never salvaged.

Batch API (task 924)
--------------------
:meth:`TaskCurator.curate_prepared` is the one curation pipeline, for every batch
size: one ``invoke_with_cap_retry`` round-trip for up to ``CuratorConfig.batch_max``
candidates, and a single curation is a batch of one. Identical candidates in the same
batch are short-circuited via pre-batch payload-hash dedup; each unique candidate keeps
its own pool. Contract: plans/review-all-s11-curator-pipeline-prd.md §Contract
"Curation pipeline".
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid as uuid_mod
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Required, TypeAlias, TypedDict, TypeVar

from shared.cancel_dispositions import sibling_lookup_values
from shared.cli_invoke import (
    AgentResult,
    TranscriptEvidence,
    claude_binary_spec,
    invoke_with_cap_retry,
    is_zero_output_timeout,
    no_mcp_servers_config,
    resolve_claude_binary,
    transcript_evidence_for_session,
)
from shared.config_dir import (
    CONFIG_DIR_PREFIX,
    TaskConfigDir,
    sweep_stale_pid_dirs,
    sweep_stale_pid_dirs_once,
)
from shared.locking import files_to_modules
from shared.neutral_cwd import neutral_cli_cwd
from shared.prompt_artifact import PromptArtifactStore, PromptSpec, default_artifacts_root
from shared.streak import KeyedStreaks
from shared.task_fields import dependency_ids as _task_dependencies
from shared.task_metadata_wire import coerce_task_metadata
from shared.task_statuses import ACTIVE
from shared.transcript_archive import ArchiveBeforeDelete, archive_before_delete

from fused_memory.backends.sdk_client_lifecycle import close_wrapped_sdk_clients
from fused_memory.backends.task_backend_errors import TaskNotFoundError
from fused_memory.backends.task_backend_types import MetadataLookupRow
from fused_memory.mcp_tools.scheduler_state import effective_lock_depth
from fused_memory.middleware.cancelled_premise_blocklist import load_blocklist
from fused_memory.middleware.cancelled_premise_blocklist import (
    match_candidate as match_blocklisted_premise,
)
from fused_memory.middleware.candidate_key import compute_candidate_key
from fused_memory.middleware.curation_deferral import (
    POOL_UNAVAILABLE_ERRORS,
    DeferReason,
    pool_defer_reason,
)
from fused_memory.middleware.curator_types import (
    DEFAULT_PRIORITY,
    NON_CANCELLED_STATUSES,
    POOL_SOURCES,
    CandidateTask,
    CuratorDecision,
    CuratorFailureError,
    PoolEntry,
    PoolWithheld,
    PreparedCandidate,
    RewrittenTask,
    clip_for_prompt,
    declared_scope_of,
    embedding_text,
    empty_pool_sizes,
    exception_summary,
    is_combine_eligible,
    item_count_of,
    render_declared_scope,
)
from fused_memory.middleware.followup_hold import is_holder
from fused_memory.middleware.operational_ask_registry import load_operational_registry
from fused_memory.middleware.operational_ask_registry import (
    match_candidate as match_operational_ask,
)
from fused_memory.middleware.recon_claim_verification_guard import (
    AttributedClaim,
    extract_attributed_claims,
    make_source_and_history_probe,
    verify_attributed_claims,
)
from fused_memory.middleware.recon_code_fix_premise_guard import (
    load_premise_registry,
    verify_premise_refuted,
)
from fused_memory.middleware.recon_code_fix_premise_guard import (
    match_candidate as match_refuted_premise,
)
from fused_memory.middleware.zero_output_breaker import ZeroOutputBreaker
from fused_memory.reconciliation.cli_stage_runner import recon_archive_root
from fused_memory.reconciliation.context_assembler import estimate_tokens

if TYPE_CHECKING:
    from qdrant_client.models import ExtendedPointId
    from shared.cost_store import CostStore
    from shared.usage_gate import UsageGate

    from fused_memory.backends.task_backend_protocol import TaskBackendProtocol
    from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig
    from fused_memory.middleware.curator_escalator import CuratorEscalator

logger = logging.getLogger(__name__)


_STATUS_RANK = {
    'pending': 0,
    'in-progress': 1,
    'in_progress': 1,
    'blocked': 2,
    'deferred': 3,
    'done': 4,
}
_PRIORITY_RANK = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'polish': 4}
if DEFAULT_PRIORITY not in _PRIORITY_RANK:
    raise ValueError(f'{DEFAULT_PRIORITY!r} not in _PRIORITY_RANK')

# Expiry of this bound raises AllAccountsCappedException, which the curator turns
# into a defer decision for every LLM-bound candidate, so the ticket is parked and
# retried later. The shared default of 14 days would instead hold the ticket worker
# for up to 14 days, so a minutes-scale bound keeps the defer fast.
_CURATOR_CAP_WAIT_SANITY_SECS = 120.0

# Stem of a gated curator's per-process CLAUDE_CONFIG_DIR. The dead-PID sweep
# builds its prefix from the same constant, so the two cannot drift apart.
_CONFIG_DIR_TASK_PREFIX = 'fm-curator-'


def _archive_curator_config_dir(config_dir: Path, archive_root: Path) -> ArchiveBeforeDelete:
    """Archive a curator config dir's transcripts, then remove the dir. Never raises.

    The archive key is the dir's TaskConfigDir task_id, ``fm-curator-<pid>``.
    """
    outcome = archive_before_delete(
        config_dir, config_dir.name.removeprefix(CONFIG_DIR_PREFIX), archive_root=archive_root,
    )
    if outcome.archived or outcome.held:
        logger.info(
            'TaskCurator config dir %s archival: archived=%d already_current=%d held=%d '
            'removed=%s archive_root=%s',
            config_dir.name,
            outcome.archived,
            outcome.already_current,
            len(outcome.held),
            outcome.config_dir_removed,
            archive_root,
        )
    return outcome


def _reclaim_by_archiving(path: Path, *, archive_root: Path) -> None:
    """The dead-PID sweep's reclaim step, honouring ``shutil.rmtree``'s contract.

    A dir left holding unarchivable transcripts keeps its dead-PID name, so the
    next process's sweep retries it.
    """
    outcome = _archive_curator_config_dir(path, archive_root)
    if outcome.config_dir_removed:
        return
    if not path.exists():
        raise FileNotFoundError(path)
    if outcome.held:
        raise OSError(f'{len(outcome.held)} transcript(s) held, not archivable, in {path}')
    raise OSError(f'transcripts archived, but the dir could not be removed: {path}')


def _sweep_stale_curator_config_dirs_once(
    base_dir: Path | None, archive_root: Path | None,
) -> None:
    """Archive, then reclaim, curator config dirs whose owning process is dead. Never raises."""
    prefix = CONFIG_DIR_PREFIX + _CONFIG_DIR_TASK_PREFIX
    sweep_stale_pid_dirs_once(
        prefix,
        sweep=sweep_stale_pid_dirs,
        on_reclaimed=lambda reclaimed: logger.info(
            'TaskCurator: reclaimed %d stale config dir(s) under %s (dead-PID sweep)',
            reclaimed, prefix,
        ),
        on_failure=lambda _exc: logger.warning(
            'TaskCurator: dead-PID sweep of %s failed; continuing without it',
            prefix, exc_info=True,
        ),
        base_dir=base_dir,
        reclaim=None if archive_root is None else partial(
            _reclaim_by_archiving, archive_root=archive_root,
        ),
    )


@dataclass(frozen=True)
class _TranscriptScope:
    """Where one gated curator call's transcript lands, and its first-turn grace.

    config_dir and session_id together make ``transcript_turns`` stampable,
    which is what makes ``is_zero_output_timeout`` transcript-authoritative.
    The grace equals the call's own timeout because the curator's measured
    first-turn latency tail exceeds the 120s default (task 3995 plan).
    """

    config_dir: TaskConfigDir
    session_id: str
    startup_grace_secs: float

    def as_invoke_kwargs(self) -> dict[str, Any]:
        """The ``invoke_with_cap_retry`` kwargs that thread this scope."""
        return {
            'config_dir': self.config_dir,
            'session_id': self.session_id,
            'startup_grace_secs': self.startup_grace_secs,
        }


def _salvageable_verdict(
    agent_result: AgentResult, evidence: TranscriptEvidence | None,
) -> dict | None:
    """The verdict a failed call may be salvaged with, else None.

    Only a run that was KILLED (so its stdout never arrived) after the CLI
    accepted its verdict qualifies. Any other failure's stdout was already
    parsed by ``_parse_claude_output``, which owns stdout salvage, and a denied
    schema tool must reach its loud escalation.
    """
    if (
        not agent_result.timed_out
        or agent_result.schema_tool_denied
        or agent_result.structured_output is not None
        or evidence is None
    ):
        return None
    return evidence.accepted_schema_payload


# JSON schema for the curator's structured output — used by invoke_with_cap_retry
# to constrain the LLM's response.  See also CURATOR_BATCH_OUTPUT_SCHEMA below.
CURATOR_OUTPUT_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'required': ['action', 'justification'],
    'properties': {
        'action': {'type': 'string', 'enum': ['drop', 'combine', 'create']},
        'target_id': {'type': ['string', 'null']},
        # Verbatim echo of the target task's title (required for combine).
        # Verified against the live target before rewriting to prevent
        # silently clobbering an unrelated task when target_id is wrong.
        'target_fingerprint': {'type': ['string', 'null']},
        'justification': {'type': 'string'},
        'rewritten_task': {
            'type': ['object', 'null'],
            'properties': {
                'title': {'type': 'string'},
                'description': {'type': 'string'},
                'details': {'type': 'string'},
                'files_to_modify': {
                    'type': 'array',
                    'items': {'type': 'string'},
                },
                'priority': {'type': 'string'},
            },
        },
    },
}

# JSON schema for the batched curator output — wraps N per-item decisions.
# Each item mirrors the single-item shape plus candidate_index (for safe
# reordering) and batch_target_index (for within-batch duplicates).
_REWRITTEN_TASK_SCHEMA: dict[str, Any] = {
    'type': ['object', 'null'],
    'properties': {
        'title': {'type': 'string'},
        'description': {'type': 'string'},
        'details': {'type': 'string'},
        'files_to_modify': {'type': 'array', 'items': {'type': 'string'}},
        'priority': {'type': 'string'},
    },
}
CURATOR_BATCH_OUTPUT_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'required': ['decisions'],
    'properties': {
        'decisions': {
            'type': 'array',
            'items': {
                'type': 'object',
                'required': ['candidate_index', 'action', 'justification'],
                'properties': {
                    'candidate_index': {'type': 'integer'},
                    'action': {'type': 'string', 'enum': ['drop', 'combine', 'create']},
                    'target_id': {'type': ['string', 'null']},
                    'batch_target_index': {'type': ['integer', 'null']},
                    'target_fingerprint': {'type': ['string', 'null']},
                    'justification': {'type': 'string'},
                    'rewritten_task': _REWRITTEN_TASK_SCHEMA,
                },
            },
        },
    },
}


@dataclass
class BackfillResult:
    """Result of a backfill_corpus() call."""

    upserted: int = 0
    skipped: int = 0
    errors: int = 0


@dataclass
class PruneResult:
    """Result of a prune_orphans() call."""

    pruned: int = 0
    corpus_scanned: int = 0
    live: int = 0
    skipped: bool = False
    reason: str = ''


def _corpus_failed_deferral(
    candidate: CandidateTask, error: str, start: float,
) -> CuratorDecision:
    """Defer *candidate* because its corpus could not be assembled."""
    logger.warning(
        'task_curator: corpus assembly failed, deferring curation of %r: %s',
        candidate.title, error,
    )
    return CuratorDecision.deferred(
        DeferReason.CORPUS_FAILED, pool_sizes=empty_pool_sizes(), latency_ms=_elapsed_ms(start),
    )


# ----------------------------------------------------------------------
# Prompt CONTRACT / HEURISTICS split (task 2494, PRD tier1-prompt-optimization T3)
#
# _CURATOR_CONTRACT / _BATCH_CONTRACT are the FROZEN machine contract: the
# drop/combine/create action semantics, the output JSON-schema instructions,
# target_fingerprint semantics, and (batch-only) batch_target_index rules —
# everything the submit_task pipeline parses or verifies. It is un-editable
# by construction: a pinned artifact only ever supplies the heuristics
# argument to compose_prompt, never the contract (see PromptArtifactStore.resolve).
#
# _CURATOR_HEURISTICS is the EDITABLE combine guidance (the combine rules,
# the positive-signals bias-toward-combining list, and the rewrite-quality
# guidance) that a pinned artifact may override. Both specs share this same
# baseline block; only their contracts differ (batch adds the within-batch
# duplicate rules).
#
# Every section's PROSE below is copied VERBATIM from the pre-split prompts —
# no instruction is reworded or dropped. That said, the EMITTED prompt is NOT
# byte-identical to the pre-split _SYSTEM_PROMPT/_BATCH_SYSTEM_PROMPT text:
# compose_prompt() (shared/prompt_artifact.py) always renders CONTRACT, then
# the "\n\n---\n\n" separator, then HEURISTICS, which reorders sections
# relative to the original (intro -> combine rules -> positive signals ->
# rewriting -> output format -> combine safety) so that "## Output format"
# and "## Combine safety: target_fingerprint" now precede the combine
# guidance instead of following it. For the batch prompt, "## Within-batch
# duplicates" also moves from the end of the prompt to immediately before
# the heuristics block, since it lives in _BATCH_CONTRACT. Treat parity here
# as content-preservation (no instruction lost), not byte-identity — see
# TestCuratorPromptSplit's superset tests in test_task_curator.py.
# ----------------------------------------------------------------------

_CURATOR_CONTRACT = """\
You are the task curator for the dark-factory orchestrator. For each candidate \
task a reviewer or agent wants to create, you decide ONE of three actions:

- "drop": the candidate is already covered by an existing task in the pool. The \
  existing task may be pending, in-progress, blocked, or done — any of these mean \
  the work will happen or has happened, so creating a duplicate wastes a full \
  architect→implement→review cycle.

- "combine": the candidate should be folded into an existing PENDING task (marked \
  combine_eligible=true in the pool). You MUST rewrite the target task to coherently \
  subsume the candidate's work, producing a single unified task. Combining into a \
  non-pending task would silently drop the candidate's work because the workflow \
  has already moved past planning for that task.

- "create": genuinely new work that cannot be merged without loss. Default to \
  "create" when in doubt — a duplicated task is cheaper than a lost or garbled one.

## Output format

Return a single JSON object matching the provided schema. The `justification` field \
should briefly explain which rule triggered the decision. For "drop" and "combine", \
set `target_id` to the id of the matched pool task. For "combine", set \
`rewritten_task` to the full rewritten task fields. For "create", leave `target_id` \
and `rewritten_task` as null.

## Combine safety: target_fingerprint (required for combine)

When you choose "combine", you MUST also populate `target_fingerprint` with the \
VERBATIM title of the pool task you are targeting — copy the exact string \
immediately following `title:` from the pool entry whose id matches `target_id`. \
This fingerprint is verified against the live task before the rewrite is applied; \
if it does not match, the combine is refused and the candidate is created fresh \
instead of silently clobbering an unrelated task. Copy the title character-for-\
character — do not shorten, summarize, or reformat it. For "drop" and "create", \
leave `target_fingerprint` null.
"""

_CURATOR_HEURISTICS = """\
## Combine rules (ALL must hold)

1. Coherence. The rewritten task can be described in one sentence without the word \
   "and also". If you need "plus unrelated X", it's not one task.
2. Narrow lock. The union of files_to_modify must stay within the same 1–2 \
   module-lock keys the target already touches. If combining adds a new module \
   to the lock set, refuse and "create" instead.
3. Planning budget. The rewritten task can be planned by one architect in one \
   session: ≤ 15 files_to_modify, ≤ 5 distinct concerns at bullet level, combined \
   details under ~8000 chars. Each follow-up item a task carries (its \
   followup_items count, or its listed items) is one concern toward that ≤ 5.
4. No critical-path drag. If the target is high-priority and on a milestone critical \
   path, the candidate must itself be critical or tightly coupled to the target's \
   correctness. Do NOT weigh critical work down with adjacent nice-to-haves.

## Positive signals (bias toward combining when the above hold)

- Either task's declared-scope band is `narrow` (shown as `declared_scope: narrow` \
  or `band=narrow`): it declared small work when it was filed, so its orchestrator \
  overhead is disproportionate to the work. The band says what a task declared, not \
  what it will cost, and an `unstamped` band carries no information either way.
- The two tasks share a root cause or subsystem concept — fixing both together \
  means the implementer loads context once.
- They will serialize on the same module lock anyway (scheduler already serializes \
  tasks touching the same module key). Combining loses zero parallelism.
- They touch the same function body or adjacent lines — splitting invites a merge \
  conflict.

## Rewriting, not concatenating

When you combine, the rewritten task must be a single coherent rewrite:

- NEW title that names the unified concern (not "T and also C").
- NEW details structured as coherent bullets under that unified concern.
- PRESERVE every concrete code reference from both tasks VERBATIM: file paths, \
  line numbers, test names, function/class/symbol names. Do not paraphrase specifics \
  away — the implementer needs them.
- Union files_to_modify (verbatim, deduplicated).
- Carry the higher priority of the two.

If you cannot produce a coherent rewrite that satisfies the hard constraints above, \
the answer is "create", not a half-hearted combine.
"""

_BATCH_CONTRACT = _CURATOR_CONTRACT + """

## Within-batch duplicates

When you are given a BATCH of N candidates in a single call (sections labelled
`# Candidate batch_index=0`, `# Candidate batch_index=1`, …), use the following
additional rules for duplicates that exist WITHIN the batch:

1. **Drop to a sibling candidate**: if candidate A is a duplicate of another
   candidate B in this batch (neither has been materialised as a task yet), emit
   `action='drop'` for A and set `batch_target_index=<B's batch_index>`.  Leave
   `target_id` null in this case — there is no existing pool task to reference.

2. **Combine between two new batch items is NOT supported**: "combine" rewrites
   an *existing* task; two candidates that are both new cannot combine with each
   other.  Instead, choose one to "create" and emit `action='drop'` with
   `batch_target_index` pointing at it for the other.

3. **batch_target_index must be valid**: the value MUST reference a *different*
   candidate index that appears within the batch (0 ≤ batch_target_index < N,
   batch_target_index ≠ candidate_index).  Do not emit cycles (A→B and B→A).

4. For non-duplicate candidates, batch_target_index MUST be null.
"""

# Bumped whenever _CURATOR_CONTRACT/_BATCH_CONTRACT change materially, so a
# stale per-(prompt, model, harness) pinned artifact stops resolving and
# falls back to the (new) in-code baseline instead of pairing new heuristics
# with an outdated contract assumption.
_CURATOR_PROMPT_HARNESS_VERSION = 'curator-v1'

CURATOR_SINGLE_SPEC = PromptSpec(
    prompt_id='curator_single', contract=_CURATOR_CONTRACT, baseline_heuristics=_CURATOR_HEURISTICS,
)
CURATOR_BATCH_SPEC = PromptSpec(
    prompt_id='curator_batch', contract=_BATCH_CONTRACT, baseline_heuristics=_CURATOR_HEURISTICS,
)


def _scale_budget(base: float, per_entry: float, size: int, cap: float) -> float:
    """Compute a scaled LLM budget clamped at ``cap``.

    Both the single-call and batch budget-scaling formulas share this shape::

        budget = min(base + per_entry * size, cap)

    where ``size`` is ``len(pool)`` for single-call and ``(n - 1)`` for batch.
    Extracting the formula avoids drift between the two call sites when either
    constant is tuned.
    """
    return min(base + per_entry * size, cap)


@dataclass(frozen=True)
class _CuratorCall:
    """Everything one curator LLM call varies by its size, decided in one place."""

    spec: PromptSpec
    output_schema: dict[str, Any]
    timeout_seconds: float
    max_turns: int
    budget_usd: float
    label: str
    ledger_role: str
    project_id: str
    failure_context: str
    parse: Callable[[AgentResult, Sequence[PreparedCandidate], int], list[CuratorDecision]]

    @classmethod
    def single(cls, cfg: CuratorConfig, project_id: str, pool_size: int) -> _CuratorCall:
        """One candidate judged against a pool of *pool_size* entries.

        The budget scales by pool size, but its base and cap are both a flat
        $2.00 (task 1980 / esc-task-curator-194), so every pool size clamps to
        that.

        ``max_turns`` is never 1: see
        fused-memory/src/fused_memory/reconciliation/agent_loop.py::_AGENT_CLI_MAX_TURNS
        for the prose turn the model emits before calling ``StructuredOutput``.
        """
        return cls(
            spec=CURATOR_SINGLE_SPEC,
            output_schema=CURATOR_OUTPUT_SCHEMA,
            timeout_seconds=cfg.timeout_seconds,
            max_turns=cfg.max_turns,
            budget_usd=_scale_budget(
                cfg.max_budget_usd, cfg.per_pool_entry_budget_usd, pool_size,
                cfg.single_call_budget_cap_usd,
            ),
            label=f'task-curator[{project_id}]',
            ledger_role='task_curator',
            project_id=project_id,
            failure_context='curator LLM call failed:',
            parse=_parse_single_call,
        )

    @classmethod
    def batch(cls, cfg: CuratorConfig, project_id: str, n: int) -> _CuratorCall:
        """*n* candidates (n >= 2) judged in one call.

        Limits scale by ``n - 1``: the single-call limits already cover the
        first item, and each further item adds its slack, up to the batch caps.
        The budget's base and cap are both a flat $2.00 (task 1980), so its
        per-item term is inert.
        """
        return cls(
            spec=CURATOR_BATCH_SPEC,
            output_schema=CURATOR_BATCH_OUTPUT_SCHEMA,
            timeout_seconds=min(
                cfg.timeout_seconds + cfg.per_item_slack_seconds * (n - 1),
                cfg.batch_timeout_cap_seconds,
            ),
            max_turns=min(cfg.max_turns + cfg.per_item_turns * (n - 1), cfg.batch_turns_cap),
            budget_usd=_scale_budget(
                cfg.max_budget_usd, cfg.per_item_budget_usd, n - 1, cfg.batch_budget_cap_usd,
            ),
            label=f'task-curator-batch[{project_id}]',
            ledger_role='task_curator_batch',
            project_id=project_id,
            failure_context=f'curator batch LLM call failed: batch_size={n}',
            parse=_parse_batch_call,
        )


def _failure_error(
    agent_result: AgentResult, evidence: TranscriptEvidence | None, call: _CuratorCall,
) -> CuratorFailureError:
    """The :exc:`CuratorFailureError` for a failed *call*, with what the failure carried."""
    return CuratorFailureError(
        f'{call.failure_context} output={agent_result.output[:200]!r} '
        f'subtype={agent_result.subtype!r} turns={agent_result.turns} '
        f'transcript_turns={agent_result.transcript_turns} '
        f'timed_out={agent_result.timed_out} '
        f'duration_ms={agent_result.duration_ms} '
        f'schema_tool_denied={agent_result.schema_tool_denied} '
        f'configured_timeout_secs={call.timeout_seconds}',
        timed_out=agent_result.timed_out,
        duration_ms=agent_result.duration_ms,
        subtype=agent_result.subtype or None,
        schema_tool_denied=agent_result.schema_tool_denied,
        zero_output_timeout=is_zero_output_timeout(agent_result),
        proc_tree=agent_result.proc_tree,
        account_name=agent_result.account_name,
        cost_usd=agent_result.cost_usd,
        transcript_turns=agent_result.transcript_turns,
        tools_used=evidence.other_tool_uses if evidence else None,
    )


def _append_pool_truncation(
    lines: list[str], withheld: PoolWithheld | None,
) -> None:
    """Append the pool-truncation block to *lines*, if there is one.

    The one place the candidate-section renderer emits the census, so the fact
    and its guidance cannot end up worded differently on the single and batch
    paths. Appends nothing when the pool was complete.
    """
    rendered = withheld.render() if withheld is not None else None
    if rendered is not None:
        lines.append(rendered)
        lines.append('')


@dataclass(frozen=True)
class _SectionHeadings:
    """The two heading lines that tell one prompt's candidate section from another's."""

    candidate: str
    pool: str

    @classmethod
    def single(cls) -> _SectionHeadings:
        return cls(candidate='Candidate task (not yet created)', pool='Pool')

    @classmethod
    def batch(cls, index: int) -> _SectionHeadings:
        return cls(
            candidate=f'Candidate batch_index={index}', pool=f'Pool for batch_index={index}',
        )


def _render_candidate_section(
    candidate: CandidateTask,
    pool: list[PoolEntry],
    withheld: PoolWithheld | None,
    headings: _SectionHeadings,
    cfg: CuratorConfig,
) -> list[str]:
    """The lines of one candidate's prompt section: the candidate, its pool, the pool's census."""
    desc_cap = cfg.entry_description_chars
    details_cap = cfg.entry_details_chars

    lines = [
        f'# {headings.candidate}',
        f'  title: {candidate.title}',
        f'  priority: {candidate.priority}',
        f'  spawn_context: {candidate.spawn_context}',
    ]
    if candidate.spawned_from:
        lines.append(f'  spawned_from: {candidate.spawned_from}')
    if candidate.description:
        lines.append(f'  description: {clip_for_prompt(candidate.description, desc_cap)}')
    if candidate.details:
        lines.append(f'  details: {clip_for_prompt(candidate.details, details_cap)}')
    if candidate.files_to_modify:
        lines.append('  files_to_modify: ' + ', '.join(candidate.files_to_modify))
    lines.append(f'  declared_scope: {render_declared_scope(candidate.declared_scope)}')
    lines.append(f'  followup_items: {len(candidate.items)}')
    lines.append('')

    lines.append(f'# {headings.pool} ({len(pool)} tasks)')
    if not pool:
        lines.append('  (empty — no candidates; answer "create")')
    else:
        for entry in pool:
            lines.append(entry.render(desc_cap, details_cap))
            lines.append('')
    _append_pool_truncation(lines, withheld)
    return lines


def _render_single_prompt(
    candidate: CandidateTask,
    pool: list[PoolEntry],
    withheld: PoolWithheld | None,
    cfg: CuratorConfig,
) -> str:
    """The user prompt that asks for one candidate's decision."""
    return '\n'.join([
        *_render_candidate_section(candidate, pool, withheld, _SectionHeadings.single(), cfg),
        'Decide drop / combine / create per the system-prompt rules. '
        'Return only the JSON object required by the schema.',
    ])


def _render_batch_section(
    candidate: CandidateTask,
    pool: list[PoolEntry],
    index: int,
    withheld: PoolWithheld | None,
    cfg: CuratorConfig,
) -> str:
    """Candidate *index*'s section of a batch prompt, ending in a blank line.

    :meth:`TaskCurator.prepare_candidate` estimates a candidate's token cost
    from this same text, so the batch-sizing accumulator counts what is sent.
    """
    headings = _SectionHeadings.batch(index)
    return '\n'.join([*_render_candidate_section(candidate, pool, withheld, headings, cfg), ''])


def _render_batch_prompt(
    candidates: list[CandidateTask],
    pools: list[list[PoolEntry]],
    withheld_list: list[PoolWithheld | None] | None,
    cfg: CuratorConfig,
) -> str:
    """The user prompt that asks for one decision per candidate, addressed by batch_index.

    Pools stay per-candidate (never unioned), and so does each pool's census,
    which renders inside its own candidate's section so a truncated pool for
    one candidate is never read as a qualifier on another's. A short or absent
    *withheld_list* renders nothing for the missing entries, which is what an
    untruncated pool renders anyway.
    """
    censuses = withheld_list or []
    sections = [
        _render_batch_section(
            candidate, pool, i, censuses[i] if i < len(censuses) else None, cfg,
        )
        for i, (candidate, pool) in enumerate(zip(candidates, pools, strict=True))
    ]
    instruction = (
        'Return one decision per candidate as {"decisions": [...]} '
        'with a candidate_index field on each item, per the schema.'
    )
    return '\n'.join([*sections, instruction])


class _LazyRegistry:
    """One-shot, off-loop lazy load of a YAML registry file.

    Owns the ``(entries, attempted, lock)`` triple for exactly one registry
    so the double-checked-lock + ``asyncio.to_thread`` + fail-open idiom
    (see :meth:`entries`) exists once for all three curator guards
    (:meth:`TaskCurator._maybe_blocklist_drop`,
    :meth:`TaskCurator._maybe_route_deterministic`,
    :meth:`TaskCurator._maybe_premise_refuted_drop`) instead of being copied
    at each site. The idiom is subtle in exactly the way that punishes
    copies: the attempted flag must latch AFTER the load settles and never
    in a ``finally``, because ``asyncio.CancelledError`` is a
    ``BaseException`` (task 4201).
    """

    def __init__(self, config_key: str, label: str) -> None:
        # CuratorConfig field name for this registry's path — named in the
        # relative-path-without-cwd WARNING so an operator knows which
        # config field to make absolute.
        self._config_key = config_key
        # Short guard label ('blocklist' / 'operational-ask' / 'recon-premise')
        # — named in the load-error WARNING so an operator knows which guard
        # just went dark. Deliberately distinct from config_key: the two
        # warnings name genuinely different things, and 'recon-premise' does
        # not appear in 'recon_code_fix_premise_registry_path' (hyphen vs
        # underscore), which an existing test asserts on verbatim.
        self._label = label
        # _attempted (not _entries) is the not-yet-loaded-vs-settled signal
        # — see entries() below. _entries holds the loader's result once
        # settled, or None when the load raised and the guard failed open:
        # None therefore means "not yet attempted" OR "load raised", while
        # [] means only "loaded but empty". Every caller treats both falsy
        # cases identically (`if not entries: return None`). No hot-reload
        # — a server restart is required to pick up YAML changes.
        self._entries: list | None = None
        self._attempted = False
        # Guards the one-shot load. Needed because the load is offloaded via
        # asyncio.to_thread: the await point means a concurrent caller could
        # otherwise observe attempted=True while _entries is still None and
        # silently fail the guard open. One TaskCurator instance is shared
        # across projects while the interceptor's curator lock is
        # per-project, so concurrent entry is reachable. asyncio.Lock() is
        # loop-agnostic at construction on Python 3.13, so building it here
        # in __init__ is safe.
        self._lock = asyncio.Lock()

    def _resolve(self, cfg_path: str, cwd: Path | None) -> Path:
        """Resolve *cfg_path* against *cwd*, warning once when relative with no cwd."""
        raw_path = Path(cfg_path)
        if raw_path.is_absolute():
            return raw_path
        if cwd is not None:
            return cwd / raw_path
        logger.warning(
            'task_curator: %s %r is relative but TaskCurator was constructed '
            'without cwd — resolving against process CWD which may be '
            'incorrect; use an absolute path in CuratorConfig',
            self._config_key, cfg_path,
        )
        return raw_path

    async def entries(
        self, loader: Callable[[Path], list], cfg_path: str, cwd: Path | None,
    ) -> list | None:
        """Return this registry's parsed entries, loading at most once.

        Offloaded via ``asyncio.to_thread`` so the blocking read/YAML-parse
        work never stalls the event loop (INV-8
        loop-thread-occupancy-bounded, docs/legibility/design-invariants.md).

        Double-checked lock: the await point inside the offloaded load means
        a concurrent second caller could otherwise observe attempted=True
        while entries is still None and silently fail the guard open. The
        outer unlocked check keeps the steady-state path (every call after
        the first) lock-free; the inner re-check under the lock is what
        makes concurrent first calls correct. The flag is set only AFTER the
        assignment so nobody can observe the attempted-but-unassigned window.
        """
        if not self._attempted:
            async with self._lock:
                if not self._attempted:
                    path = self._resolve(cfg_path, cwd)
                    try:
                        self._entries = await asyncio.to_thread(loader, path)
                    except Exception as exc:
                        # The loaders never raise (their file-level contract is
                        # fused_memory/utils/safe_yaml.py::load_yaml_list_file),
                        # but asyncio.to_thread adds a raise path of its own
                        # (thread-pool failure or shutdown) that no loader-
                        # internal except can cover. Fail OPEN (guard
                        # disabled) rather than escaping into the curation
                        # pipeline, which calls the guard methods unguarded —
                        # an escape would defer every candidate in the batch.
                        logger.warning(
                            'task_curator: %s registry load errored for %s, '
                            'failing open (guard disabled): %s',
                            self._label, path, exc,
                        )
                        self._entries = None
                    # Settled outcome (loaded, or failed open) — latch the
                    # one-shot contract. Deliberately NOT a `finally`:
                    # asyncio.CancelledError is a BaseException, so it
                    # bypasses the except above, and latching in a `finally`
                    # would permanently disable the guard because an
                    # unrelated caller was cancelled mid-load. Leaving the
                    # flag clear on cancellation lets the next call retry.
                    self._attempted = True
        return self._entries


CuratorInvoke: TypeAlias = Callable[..., Awaitable[AgentResult]]


class TaskCurator:
    """LLM-judged drop/combine/create gate plus the Qdrant corpus backing it."""

    def __init__(
        self,
        config: FusedMemoryConfig,
        taskmaster: TaskBackendProtocol | None = None,
        usage_gate: UsageGate | None = None,
        cwd: Path | None = None,
        escalator: CuratorEscalator | None = None,
        prompt_store: PromptArtifactStore | None = None,
        config_dir_base: Path | None = None,
        cost_store: CostStore | None = None,
        invoke: CuratorInvoke = invoke_with_cap_retry,
    ) -> None:
        self._config = config
        self._invoke = invoke
        self._taskmaster = taskmaster
        self._usage_gate = usage_gate
        self._cost_store = cost_store
        self._cwd = cwd
        self._escalator = escalator
        self._prompt_store = prompt_store
        # Parent of the per-process config dir; None means the system tempdir.
        self._config_dir_base = config_dir_base
        self._config_dir: TaskConfigDir | None = None  # lazy, gated only
        self._qdrant_client = None  # AsyncQdrantClient, lazy
        self._embedder = None  # OpenAIEmbedder, lazy
        self._initialized_collections: set[str] = set()
        # Idempotency cache: payload_hash -> (decision, monotonic_time)
        self._decision_cache: dict[str, tuple[CuratorDecision, float]] = {}
        # Pre-LLM exact-match cache: project_id -> {normalized_hash: (task_id, ts)}
        # Catches the common "two triages race to create identical tasks"
        # case cheaply — skips embedding + LLM entirely on the second call.
        self._recent_creates: dict[str, dict[str, tuple[str, float]]] = {}
        # Cancelled-premise blocklist — lazy-loaded on first curation.
        self._blocklist = _LazyRegistry(
            config_key='cancelled_premise_blocklist_path', label='blocklist',
        )
        # Recon code-fix premise-verification registry — lazy-loaded on first
        # curation, same shape as _blocklist above. Unlike the blocklist,
        # a match here is re-verified against the live source tree on every
        # call (see recon_code_fix_premise_guard.verify_premise_refuted) —
        # only the YAML load itself is cached for the instance lifetime.
        self._premise_registry = _LazyRegistry(
            config_key='recon_code_fix_premise_registry_path', label='recon-premise',
        )
        # Operational-ask registry (filing-policy gate) — lazy-loaded on first
        # curation, same shape as _blocklist above. A match routes the
        # candidate straight to a deterministic PURE-GATE instead of the LLM;
        # unlike the blocklist, the resulting decision is deliberately NOT
        # written to the idempotency cache (see _maybe_route_deterministic).
        self._operational_registry = _LazyRegistry(
            config_key='operational_ask_registry_path', label='operational-ask',
        )
        # Consecutive-ZOT circuit breaker (task 1743): counts only ZOTs; a
        # non-ZOT failure neither increments nor resets it.
        self._zero_output_breaker = ZeroOutputBreaker(
            threshold=config.curator.zero_output_breaker_threshold,
            cooldown_seconds=config.curator.zero_output_breaker_cooldown_seconds,
        )
        # Degraded-curation streaks by project_id (task 4448). Distinct from
        # the ZOT breaker above: that one counts hung calls in order to stop
        # paying for them; these count degraded decisions of every cause, in
        # order to make a sustained outage visible. Keyed by project because
        # one curator serves every project, so a shared count would let one
        # project's healthy curations mask another's outage.
        self._degraded_streaks: KeyedStreaks[str] = KeyedStreaks(
            threshold=config.curator.degraded_streak_threshold,
        )

    # ------------------------------------------------------------------
    # Lazy init (mirrors task_dedup.py pattern)
    # ------------------------------------------------------------------

    async def _get_qdrant(self):
        if self._qdrant_client is None:
            from qdrant_client import AsyncQdrantClient

            self._qdrant_client = AsyncQdrantClient(
                url=self._config.mem0.qdrant_url,
                timeout=30,
            )
        return self._qdrant_client

    async def _get_embedder(self):
        if self._embedder is None:
            cfg = self._config.embedder
            if cfg.provider == 'openai' and cfg.providers.openai:
                api_key = cfg.providers.openai.api_key
                if api_key:
                    from graphiti_core.embedder import OpenAIEmbedder
                    from graphiti_core.embedder.openai import OpenAIEmbedderConfig

                    self._embedder = OpenAIEmbedder(
                        config=OpenAIEmbedderConfig(
                            api_key=api_key,
                            embedding_model=cfg.model,
                            base_url=cfg.providers.openai.api_url,
                            embedding_dim=cfg.dimensions,
                        ),
                    )
            if self._embedder is None:
                raise RuntimeError(
                    'TaskCurator requires an OpenAI embedder — check config.embedder',
                )
        return self._embedder

    def _resolve_curator_prompt(self, spec: PromptSpec) -> str:
        """Resolve *spec* to its composed prompt text via the artifact loader.

        Lazily builds ``self._prompt_store`` from :func:`default_artifacts_root`
        when no store was injected. :meth:`PromptArtifactStore.resolve` never
        raises — an absent or unverifiable pin falls back to
        ``spec.in_code_constant`` — so the curator's best-effort contract is
        preserved even when the artifacts root is missing. That fail-safe
        means a wrong or empty root (e.g. a worktree-local
        ``default_artifacts_root()`` that doesn't match the root an optimizer
        pinned into) never errors — it just silently never applies a pin. Set
        ``DARK_FACTORY_PROMPT_ARTIFACTS`` so curator and optimizer processes
        agree on one root; the one-time log line below records which root a
        lazily-constructed store resolved to, so an unexpectedly-empty root
        is diagnosable.
        """
        if self._prompt_store is None:
            root = default_artifacts_root()
            logger.info(
                'task_curator: no prompt_store injected; lazily resolved artifacts root to '
                '%s (set DARK_FACTORY_PROMPT_ARTIFACTS to override)',
                root,
            )
            self._prompt_store = PromptArtifactStore(root)
        return self._prompt_store.resolve(
            spec,
            executor_model=self._config.curator.model,
            harness_version=_CURATOR_PROMPT_HARNESS_VERSION,
        ).text

    def _collection_name(self, project_id: str) -> str:
        return f'task_curator_{project_id}'

    async def _ensure_collection(self, project_id: str) -> str:
        name = self._collection_name(project_id)
        if name in self._initialized_collections:
            return name
        client = await self._get_qdrant()
        if not await client.collection_exists(name):
            from qdrant_client.models import Distance, VectorParams

            await client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self._config.embedder.dimensions,
                    distance=Distance.COSINE,
                ),
            )
            logger.info('Created task curator collection: %s', name)
        self._initialized_collections.add(name)
        return name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Pre-LLM normalized-key cache (R3)
    # ------------------------------------------------------------------

    _RECENT_CREATES_TTL_SECS = 600.0  # 10 min — long enough to catch
    # concurrent triages but short enough that stale entries don't pin
    # memory after a task is removed.

    @staticmethod
    def _normalize_key(candidate: CandidateTask) -> str | None:
        """Stable 16-char hash over normalised (title, files_to_modify).

        Case + whitespace insensitive on the title, file-order insensitive.
        Designed to catch "two reviewers raced with the same suggestion"
        without depending on embeddings. See plan R3 — pre-LLM
        short-circuit.

        Delegates to :func:`fused_memory.middleware.candidate_key.compute_candidate_key`
        — the single owner of this key definition (fm-task-dedup W8 task A1) — rather
        than keeping its own hash. Returns ``None`` for a candidate whose title
        normalises to empty (see that function's docstring).
        """
        return compute_candidate_key(candidate.title, candidate.files_to_modify)

    def _evict_stale_recent_creates(self, project_id: str, now: float) -> None:
        bucket = self._recent_creates.get(project_id)
        if not bucket:
            return
        stale = [
            k for k, (_, ts) in bucket.items()
            if now - ts > self._RECENT_CREATES_TTL_SECS
        ]
        for k in stale:
            del bucket[k]

    async def _pre_llm_exact_match(
        self, candidate: CandidateTask, project_id: str, project_root: str,
    ) -> CuratorDecision | None:
        """Return a drop decision if a recently-created task matches
        verbatim on normalised title + file list. Otherwise return None.

        Verifies the cached task still exists (and isn't cancelled) so a
        stale cache entry doesn't redirect to a gone task.
        """
        bucket = self._recent_creates.get(project_id)
        if not bucket:
            return None
        now = time.monotonic()
        self._evict_stale_recent_creates(project_id, now)
        key = self._normalize_key(candidate)
        if key is None:
            # Title normalises to empty — no dedup signal to match on.
            return None
        entry = bucket.get(key)
        if entry is None:
            return None
        cached_id, _ts = entry
        if self._taskmaster is not None:
            try:
                task = await self._taskmaster.get_task(cached_id, project_root)
            except Exception:
                task = None
            status = (
                str(task.get('status', '')) if isinstance(task, dict) else ''
            )
            if not isinstance(task, dict) or status == 'cancelled':
                # Stale entry — drop it and fall through to full curate.
                bucket.pop(key, None)
                return None
        return CuratorDecision(
            action='drop',
            target_id=cached_id,
            justification='pre-llm-exact-match',
            pool_sizes=empty_pool_sizes(),
            latency_ms=0,
        )

    def note_created(
        self, project_id: str, candidate: CandidateTask, task_id: str,
    ) -> None:
        """Record a just-created task so identical concurrent candidates
        short-circuit on the pre-LLM exact-match cache.

        Call this *after* ``tm.add_task`` returns and within the
        project's add_task critical section so the next waiter sees it.

        No-ops when the candidate's normalised key is ``None`` (title
        normalises to empty) — there is no dedup signal worth caching.
        """
        key = self._normalize_key(candidate)
        if key is None:
            return
        bucket = self._recent_creates.setdefault(project_id, {})
        bucket[key] = (task_id, time.monotonic())

    async def _maybe_blocklist_drop(
        self, candidate: CandidateTask, payload_hash: str,
    ) -> CuratorDecision | None:
        """Return a REFUSAL decision if the candidate matches the cancelled-premise blocklist.

        The returned decision is ``action='refuse'`` — a deterministic verdict
        that creates NOTHING. It is deliberately not ``action='drop'``: a drop
        means "fold this into existing target task N" and carries a
        ``target_id``, so a targetless drop is inert at the dispatch chokepoint
        (``TaskInterceptor._dispatch_ticket_decision``) and fails open into
        creating the very candidate this guard exists to reject.

        Lazy-loads the blocklist from ``self._config.curator.cancelled_premise_blocklist_path``
        on the first call and caches it for the lifetime of this :class:`TaskCurator`
        instance (no hot-reload; a server restart is required to pick up YAML changes).
        The load runs off the event-loop thread via :class:`_LazyRegistry`
        (``asyncio.to_thread``), so it cannot stall the fused-memory event loop.

        Returns ``None`` when:
        - The blocklist path is not configured (``None``).
        - The blocklist file is missing, unreadable, or unparseable (one WARNING logged).
        - The blocklist is empty.
        - No entry matches the candidate.
        - The offloaded load raised (one WARNING logged; the guard then stays
          disabled for this TaskCurator instance rather than retrying per call).

        Never raises.
        """
        cfg_path = self._config.curator.cancelled_premise_blocklist_path
        if cfg_path is None:
            return None

        entries = await self._blocklist.entries(load_blocklist, cfg_path, self._cwd)
        if not entries:
            return None

        entry = match_blocklisted_premise(candidate, entries)
        if entry is None:
            return None

        decision = CuratorDecision(
            action='refuse',
            target_id=None,
            target_fingerprint=None,
            rewritten_task=None,
            justification=f'cancelled-premise-blocklist: {entry.name}: {entry.reason}',
            pool_sizes=empty_pool_sizes(),
            latency_ms=0,
        )
        self._store_cache(payload_hash, decision)
        logger.info(
            'task_curator: blocklist refusal (no task will be created) '
            'entry=%s candidate=%r',
            entry.name, candidate.title,
        )
        return decision

    async def _maybe_premise_refuted_drop(
        self, candidate: CandidateTask, payload_hash: str,
    ) -> CuratorDecision | None:
        """Return a REFUSAL decision if the candidate's premise is refuted by live source.

        The returned decision is ``action='refuse'`` — a deterministic verdict
        that creates NOTHING (see :meth:`_maybe_blocklist_drop` for why this is
        not ``action='drop'``).

        Lazy-loads the registry from
        ``self._config.curator.recon_code_fix_premise_registry_path`` on the first
        call and caches the parsed entries for the lifetime of this
        :class:`TaskCurator` instance (no hot-reload; a server restart is required
        to pick up YAML changes). The live source/test re-verification itself is
        NOT cached — it re-reads the cited files on every call, so the guard is
        self-correcting. Unlike ``_maybe_blocklist_drop``, the resulting REFUSAL
        DECISION is also deliberately NOT written to the idempotency cache
        (``self._decision_cache``): caching it would let a stale refusal resurface
        via ``_check_cache`` for up to ``idempotency_ttl_seconds`` after the
        cited source stops refuting the premise, silently suppressing a
        genuinely-fixed bug-fix task. ``payload_hash`` is accepted only for
        signature symmetry with ``_maybe_blocklist_drop`` and is unused here.

        Both blocking halves of this method run off the event-loop thread
        (task 4201, 2026-08-29): the one-shot registry load (via the shared
        :class:`_LazyRegistry`) and the per-match live-source re-verification
        are each dispatched via ``asyncio.to_thread``. The textual match
        (``match_refuted_premise``) is deliberately kept ON the loop — it is pure
        string work with no I/O, so the common case of a non-matching
        candidate pays no thread-pool dispatch. Measured on the shipped
        registry: registry load ~9.9ms (8.15ms of it ``yaml.safe_load``,
        only 21us the ``read_text``); worst per-match verification 1.2ms
        warm / 6.2ms cold on a 664 KB cited file; thread-dispatch overhead
        ~67us. This resolves an instance of INV-8
        ``loop-thread-occupancy-bounded`` (docs/legibility/design-invariants.md),
        mirroring the offload pattern already used by
        :meth:`_maybe_flag_unverified_claims` and
        :meth:`_flag_unverified_claims_of`.

        All three curator guards — this one, :meth:`_maybe_blocklist_drop`,
        and :meth:`_maybe_route_deterministic` — now share
        :class:`_LazyRegistry` for their one-shot lazy loads (task 5007), so
        the event-loop stall those loads can cause is bounded for the whole
        :meth:`curate_prepared` first-submission sequence, not just this guard.

        Returns ``None`` (fail-open) when:
        - The registry path is not configured (``None``).
        - The registry file is missing, unreadable, or unparseable, or the
          offloaded load raised (one WARNING logged; the guard then stays
          disabled for this TaskCurator instance rather than retrying per
          call). This is
          deliberate even for a cause that is transient in principle (a
          partially-written file observed mid-deploy, a momentary worker
          thread I/O error): :class:`_LazyRegistry.entries`'s
          ``except Exception`` does not distinguish exception type, so any
          raise latches the same as a permanently-malformed file, and a
          process restart is the recovery path. Only
          ``asyncio.CancelledError`` is excluded from the latch (see the
          comment on that except block) — every other exception is treated
          as permanent by design, not oversight.
        - The registry is empty.
        - No entry textually matches the candidate.
        - ``self._cwd`` is ``None`` — the source root cannot be resolved, so the
          premise cannot be verified.
        - The candidate matches an entry, but the live source no longer refutes
          its premise.
        - The offloaded live-source verification raised (one WARNING logged).

        Never raises.
        """
        cfg_path = self._config.curator.recon_code_fix_premise_registry_path
        if cfg_path is None:
            return None

        if self._cwd is None:
            logger.warning(
                'task_curator: recon_code_fix_premise_registry_path %r is configured but '
                'TaskCurator was constructed without cwd — cannot resolve a source root to '
                'verify premises against; guard disabled for this call',
                cfg_path,
            )
            return None

        entries = await self._premise_registry.entries(
            load_premise_registry, cfg_path, self._cwd,
        )
        if not entries:
            return None

        # Pure string ops over the registry, no I/O — deliberately kept ON the
        # loop so the overwhelmingly common non-matching candidate never pays a
        # thread-pool dispatch (measured 67us) nor extra curator-lock hold.
        # premise_refuted_entry composes exactly these two halves and already
        # short-circuits here; we split it so only the blocking half is offloaded.
        entry = match_refuted_premise(candidate, entries)
        if entry is None:
            return None

        # Off the event loop: verify_premise_refuted re-reads EVERY cited file
        # fresh on every call (that is what makes the guard self-correcting).
        # Measured on the shipped registry: the stage2_flag_query entry cites
        # tests/test_stages.py at 664 KB => 1.2 ms warm, 6.2 ms cold, 29 ms worst;
        # the 3-assertion entry is 170 us warm. curate_prepared() reaches this on
        # EVERY task submission while holding the per-project curator write lock
        # (task_interceptor.py _curator_lock, held around curate_prepared()).
        # Mirrors this file's claim-verification offloads.
        try:
            refuted = await asyncio.to_thread(verify_premise_refuted, entry, self._cwd)
        except Exception as exc:
            # Honour this method's documented "Never raises" contract. The guard
            # module fails open internally on OSError, but asyncio.to_thread adds
            # a raise path it cannot cover, and the curation pipeline calls
            # this unguarded — an escape would defer the whole batch.
            # Fail OPEN: let the candidate reach the architect rather than
            # silently refusing it on an unverified premise.
            logger.warning(
                'task_curator: recon-premise verification errored for entry=%s, '
                'failing open (candidate not dropped): %s',
                entry.name, exc,
            )
            return None
        if not refuted:
            return None

        decision = CuratorDecision(
            action='refuse',
            target_id=None,
            target_fingerprint=None,
            rewritten_task=None,
            justification=f'recon-premise-refuted: {entry.name}: {entry.reason}',
            pool_sizes=empty_pool_sizes(),
            latency_ms=0,
        )
        # Deliberately NOT self._store_cache(...)'d — see docstring. The guard
        # must self-correct the moment the cited source stops refuting the
        # premise, so every call re-verifies live source rather than trusting a
        # cached decision for the idempotency TTL.
        logger.info(
            'task_curator: recon-premise-refuted refusal (no task will be created) '
            'entry=%s candidate=%r',
            entry.name, candidate.title,
        )
        return decision

    @staticmethod
    def _claim_verification_text(candidate: CandidateTask) -> str:
        """Concatenated candidate text the claim-verification guard scans.

        Single source of truth for both :meth:`_maybe_flag_unverified_claims`
        (per-candidate extraction) and :meth:`_flag_unverified_claims_of`'s
        batch-level "does any candidate carry a claim" pre-check, so the two
        never drift.
        """
        return f'{candidate.title}\n{candidate.description}\n{candidate.details}'

    async def _maybe_flag_unverified_claims(
        self, candidate: CandidateTask, probe: Callable[[str], bool] | None = None,
    ) -> list[AttributedClaim]:
        """Return any of *candidate*'s attributed code-level claims that are unverified.

        Advisory backstop motivated by the task-2433 Stage 2 fabrication incident
        (see ``recon_claim_verification_guard``'s module docstring), but invoked
        from :meth:`curate_prepared` through :meth:`_flag_unverified_claims_of`
        — i.e. it applies to EVERY candidate the curator sees when the config
        flag is on, not only Stage-2-recon-filed ones (``CandidateTask`` carries no
        origin/source field that would let this narrow further). Extracts
        specific code-level tokens *candidate* attributes to a completed
        task/commit/ACTION (e.g. "task 2372 added
        metadata.done_provenance_invalidated=true") and verifies each against
        the live source tree + git history via *probe* (built from
        ``self._cwd`` via ``make_source_and_history_probe`` when not injected —
        tests inject a fake probe to stay git-free).

        Claim extraction runs FIRST and is pure/sync/cheap (a regex scan, no
        I/O): when *candidate* carries no attributed claim at all — the
        overwhelmingly common case, since a claim requires a code token AND a
        numbered task/ACTION/commit anchor within the co-occurrence window —
        this returns ``[]`` before a probe is ever built or a worker thread is
        ever spawned for it.

        Unlike :meth:`_maybe_premise_refuted_drop`, this hook NEVER drops the
        candidate or returns/mutates a :class:`CuratorDecision` — it only surfaces
        unverified claims via a grep-stable ``recon_claim_verification.unverified``
        WARNING (one per unverified claim, naming the token + attribution +
        candidate title) for human review, and returns the list of unverified
        claims so a caller may inspect them further.

        Returns ``[]`` (fail-open) when:
        - ``self._config.curator.recon_claim_verification_enabled`` is ``False``
          (the default — every existing caller is unaffected).
        - ``self._cwd`` is ``None`` — the source root cannot be resolved, so no
          claim can be verified (one WARNING logged).
        - *candidate*'s title/description/details carry no attributed claims at all.
        - Every attributed claim's token verifies present (self-correcting).
        - Probe construction or verification raises for any reason (one
          WARNING logged) — this is an advisory backstop and must never fail
          the task submission it rides along with.

        Never raises.
        """
        if not self._config.curator.recon_claim_verification_enabled:
            return []

        if self._cwd is None:
            logger.warning(
                'task_curator: recon_claim_verification_enabled is True but '
                'TaskCurator was constructed without cwd — cannot resolve a source '
                'root to verify attributed claims against; guard disabled for '
                'this call',
            )
            return []

        claims = extract_attributed_claims(self._claim_verification_text(candidate))
        if not claims:
            return []

        def _build_probe_and_verify(repo_root: Path) -> list[AttributedClaim]:
            local_probe = make_source_and_history_probe(repo_root)
            return verify_attributed_claims(claims, local_probe)

        try:
            if probe is None:
                # Off the event loop, in a SINGLE thread-pool hop: construction
                # resolves the git top level via _resolve_git_toplevel — a
                # blocking `git rev-parse --show-toplevel` subprocess bounded
                # only by a 10s timeout — and verification (git grep, and on a
                # miss — always true for a fabricated token, the exact case
                # this guard targets — a full-history `git log --all -S`
                # pickaxe too, up to ~10s each) then runs against the
                # freshly-built probe on that same worker thread. Only
                # reached when `claims` above is non-empty, so a candidate
                # with no attributed claim never pays for either. Mirrors
                # _flag_unverified_claims_of, which likewise only builds a
                # probe when at least one candidate in the batch carries a
                # claim.
                unverified = await asyncio.to_thread(_build_probe_and_verify, self._cwd)
            else:
                # probe was injected — _flag_unverified_claims_of builds one
                # probe per batch and shares it across candidates, or a test
                # supplies a fake. Still offload: a real probe's git grep,
                # and on a miss a full-history pickaxe, is blocking
                # regardless of who built it.
                unverified = await asyncio.to_thread(verify_attributed_claims, claims, probe)
        except Exception as exc:
            # Advisory backstop — must fail open rather than take the whole
            # task submission down with it (see "Never raises" above).
            logger.warning(
                'recon_claim_verification: guard errored, failing open: %s', exc,
            )
            return []

        for claim in unverified:
            logger.warning(
                'recon_claim_verification.unverified token=%s attribution=%s candidate=%r',
                claim.token, claim.attribution, candidate.title,
            )
        return unverified

    async def _maybe_route_deterministic(
        self, candidate: CandidateTask, payload_hash: str,
    ) -> CuratorDecision | None:
        """Return a route_deterministic decision if the candidate matches the
        operational-ask registry (the filing-policy gate).

        Lazy-loads the registry from
        ``self._config.curator.operational_ask_registry_path`` on the first
        call and caches the parsed entries for the lifetime of this
        :class:`TaskCurator` instance (no hot-reload; a server restart is
        required to pick up YAML changes). The load runs off the event-loop
        thread through the shared :class:`_LazyRegistry` (``asyncio.to_thread``),
        so it cannot stall the fused-memory event loop — mirrors
        ``_maybe_blocklist_drop``.

        Unlike ``_maybe_blocklist_drop``, the resulting decision is
        deliberately NOT written to the idempotency cache
        (``self._decision_cache``): once the first ask has materialised as a
        deterministic PURE-GATE task, a cached route decision must never
        resurface via ``_check_cache`` and spawn a SECOND gate for a re-filed
        duplicate. A re-filed duplicate is instead caught by the exact-match /
        idempotency-cache / LLM-combine checks that run before this one (see
        :meth:`_decide_before_llm`'s ordering) — mirrors the
        rationale of ``_maybe_premise_refuted_drop``. ``payload_hash`` is
        accepted only for signature symmetry with the other two guards and is
        unused here.

        KNOWN LIMITATION (residual duplicate-gate risk): the exact-match /
        idempotency-cache checks key off a payload fingerprint, so they only
        catch byte-identical (or near-identical) re-proposals. A PARAPHRASED
        re-proposal of the same operational ask — different wording, but
        still containing the registry's title/description substrings — will
        miss the fingerprint match, will not be cached, and will re-match
        this registry, filing a SECOND born-at-L2 pure-gate for work that is
        already gated. This is not auto-deduped against already-open gate
        tasks (no task-search lookup is performed here). Operators should
        expect this for paraphrased re-files and manually cancel/resolve the
        duplicate gate — see the "paraphrased re-proposals" guidance in
        config/operational_ask_registry.yaml's header.

        Returns ``None`` when:
        - The registry path is not configured (``None``).
        - The registry file is missing, unreadable, or unparseable (one
          WARNING logged by the loader).
        - The registry is empty.
        - No entry matches the candidate.
        - The offloaded load raised (one WARNING logged; the guard then stays
          disabled for this TaskCurator instance rather than retrying per call).

        Note (task δ demotion): a candidate whose ``metadata.execution_class``
        is ``"operational"`` or ``"decision"`` now returns ``None`` here too —
        not because it fell through to the last bullet, but because
        ``operational_ask_registry.match_candidate`` SKIPS tagged asks
        outright. Their routing is owned by the submit boundary
        ``operational_routing_guard.inject_operational_routing`` (task β),
        which coerces every tagged operational/decision ask to a deterministic
        pure-gate on the submit path; a second coercion in the curator would be
        lock-step duplication (INV-5). The curator retains ONLY the untagged
        title/description substring fallback (task 2687 introduced the axis;
        task δ narrowed it to a skip once the β boundary took ownership).

        Never raises.
        """
        cfg_path = self._config.curator.operational_ask_registry_path
        if cfg_path is None:
            return None

        entries = await self._operational_registry.entries(
            load_operational_registry, cfg_path, self._cwd,
        )
        if not entries:
            return None

        entry = match_operational_ask(candidate, entries)
        if entry is None:
            return None

        decision = CuratorDecision(
            action='route_deterministic',
            target_id=None,
            target_fingerprint=None,
            rewritten_task=None,
            justification=f'operational-ask-registry: {entry.name}: {entry.reason}',
            pool_sizes=empty_pool_sizes(),
            latency_ms=0,
        )
        # Deliberately NOT self._store_cache(...)'d — see docstring.
        logger.info(
            'task_curator: operational-ask-registry route entry=%s candidate=%r',
            entry.name, candidate.title,
        )
        return decision

    # ------------------------------------------------------------------
    # Backend-binary startup self-check (task 4448)
    # ------------------------------------------------------------------

    async def startup_self_check(self, project_id: str, project_root: str) -> bool:
        """Report whether the curator's backend CLI binary resolves. Never raises.

        An unresolvable binary degrades every curation; asking once at wiring
        time escalates it before the first candidate is filed. A verdict, not
        a precondition: a curator without its binary still fails open to
        ``action='create'``, whereas raising would reach the interceptor's
        ``except Exception -> return None`` and disable dedupe outright.
        """
        spec = claude_binary_spec()
        resolved = resolve_claude_binary()
        if resolved is not None:
            logger.info(
                'task_curator: backend binary %r resolves to %s', spec, resolved,
            )
            return True

        search_path = os.environ.get('PATH', '')
        logger.error(
            'task_curator: backend binary %r does not resolve (PATH=%s) — every '
            'curation for project %s will degrade to action=create WITHOUT '
            'dedupe until this is fixed. Set CLAUDE_BINARY to an absolute path, '
            'or pin Environment=PATH= in the fused-memory systemd unit.',
            spec, search_path, project_id,
        )
        if self._escalator is not None:
            await self._escalator.report_backend_binary_unresolvable(
                project_root=project_root,
                project_id=project_id,
                binary_spec=spec,
                search_path=search_path,
            )
        return False

    # ------------------------------------------------------------------
    # Class-agnostic degraded-streak alarm (task 4448)
    # ------------------------------------------------------------------

    async def _degraded_create(
        self,
        *,
        justification: str,
        pool_sizes: dict[str, int],
        start: float,
        candidate: CandidateTask,
        project_id: str,
        project_root: str,
        degraded_by_zot: bool = False,
        zot_escalation_id: str | None = None,
    ) -> CuratorDecision:
        """Build a degraded ``action='create'`` decision and count it.

        The only ``TaskCurator`` method that builds a create of its own, which
        is what lets the streak be counted without enumerating why a curation
        can degrade; fused-memory/tests/test_curator_arm_instrumentation_guard.py
        enforces that. Degrading is the designed fail-open, so this neither
        raises nor alters the decision; it only notices when degrading has
        stopped being occasional.
        """
        decision = CuratorDecision(
            action='create',
            justification=justification,
            pool_sizes=pool_sizes,
            latency_ms=_elapsed_ms(start),
            degraded=True,
            degraded_by_zot=degraded_by_zot,
            zot_escalation_id=zot_escalation_id,
        )
        await self._count_degraded(
            justification=justification,
            project_id=project_id,
            project_root=project_root,
            candidate_title=candidate.title,
        )
        return decision

    async def _count_degraded(
        self,
        *,
        justification: str,
        project_id: str,
        project_root: str,
        candidate_title: str,
    ) -> None:
        """Advance *project_id*'s streak by one degraded curation, alarming once.

        Shared by :meth:`_degraded_create` (single-candidate paths) and by
        :meth:`_decisions_from_result` (items a successful round-trip
        nonetheless failed to decide).  The alarm is class-agnostic by construction: it
        takes only the fact that a curation degraded, never the reason.
        """
        step = self._degraded_streaks.failure(project_id)
        if not step.fired:
            return

        threshold = self._config.curator.degraded_streak_threshold
        logger.error(
            'task_curator: %d consecutive degraded curations for project %s '
            '(threshold %d) — every candidate in that run was filed without '
            'dedupe. Last: %s',
            step.count, project_id, threshold, justification,
        )
        if self._escalator is not None:
            await self._escalator.report_consecutive_degraded(
                project_root=project_root,
                project_id=project_id,
                streak=step.count,
                threshold=threshold,
                last_justification=justification,
                candidate_title=candidate_title,
            )

    async def curate_prepared(
        self,
        prepared: Sequence[PreparedCandidate],
        project_id: str,
        project_root: str,
    ) -> list[CuratorDecision]:
        """Decide every prepared bundle: the one curation pipeline, at every batch size.

        Returns one decision per bundle, in input order. A candidate that
        cannot be judged gets :meth:`CuratorDecision.deferred` with the
        :class:`~fused_memory.middleware.curation_deferral.DeferReason` that
        says why. A zero-output hang is the one LLM outage that creates
        (``degraded_by_zot``), and an unusable response degrades to a
        ``create``.

        Raises only :exc:`asyncio.CancelledError` and a
        ``POOL_UNAVAILABLE_ERRORS`` error from a guard, which none of today's
        guards raises; the caller defers that under the pool's reason. Any
        other exception from a guard defers every bundle
        ``CURATOR_RAISED``, so a curator that raises instead of deciding parks
        its tickets rather than failing or creating them. A raise out of the
        LLM dispatch defers only the bundles it was given (see
        :meth:`_dispatch_deferring_raises`).

        The guards run in PRD decision-10 order (see :meth:`_run_pipeline` and
        :meth:`_decide_before_llm`); what none of them decides goes to
        :meth:`_dispatch_llm` in one call, over the bundles' own pools.
        """
        if not prepared:
            return []
        start = time.monotonic()
        try:
            return await self._run_pipeline(list(prepared), project_id, project_root, start)
        except POOL_UNAVAILABLE_ERRORS:
            raise
        except Exception:
            logger.exception(
                'task_curator: the curation pipeline raised for %d candidate(s) of '
                'project %s; deferring every one of them',
                len(prepared), project_id,
            )
            return _deferred_each(prepared, DeferReason.CURATOR_RAISED, start)

    async def _run_pipeline(
        self,
        bundles: list[PreparedCandidate],
        project_id: str,
        project_root: str,
        start: float,
    ) -> list[CuratorDecision]:
        """Guards 1-5 over the whole batch, then 6-10 per bundle, then one LLM dispatch."""
        candidates = [bundle.candidate for bundle in bundles]
        hashes = [candidate.payload_hash() for candidate in candidates]
        decided: dict[int, CuratorDecision] = {}

        # 1. A byte-identical later candidate drops onto its first occurrence;
        # the worker substitutes that sibling's resulting task at dispatch.
        first_of: dict[str, int] = {}
        for i, payload_hash in enumerate(hashes):
            if payload_hash in first_of:
                decided[i] = CuratorDecision(
                    action='drop',
                    batch_target_index=first_of[payload_hash],
                    justification='pre-batch-dedup: identical payload_hash',
                )
            else:
                first_of[payload_hash] = i

        # 2-3. Deterministic refusals, which create nothing.
        refusals: dict[int, CuratorDecision] = {}
        for i in first_of.values():
            refusal = await self._maybe_blocklist_drop(candidates[i], hashes[i])
            if refusal is None:
                refusal = await self._maybe_premise_refuted_drop(candidates[i], hashes[i])
            if refusal is not None:
                refusals[i] = refusal
        decided.update(refusals)

        # 4. A duplicate of a refused candidate inherits the refusal. Its
        # sibling drop would resolve against a task that was never created and
        # fail open into filing the very premise the guard refused. ``replace``
        # keeps it field-identical to the refusal it copies.
        for i, decision in list(decided.items()):
            sibling = decision.batch_target_index
            if sibling is not None and sibling in refusals:
                decided[i] = replace(refusals[sibling], batch_target_index=None)

        live = [i for i in first_of.values() if i not in refusals]

        # 5. Advisory only: flags unverified claims, decides nothing.
        await self._flag_unverified_claims_of([candidates[i] for i in live])

        # 6-10.
        llm_bound: list[int] = []
        for i in live:
            decision = await self._decide_before_llm(
                bundles[i], hashes[i], project_id, project_root, start,
            )
            if decision is None:
                llm_bound.append(i)
            else:
                decided[i] = decision

        if llm_bound:
            llm_decisions = await self._dispatch_deferring_raises(
                [bundles[i] for i in llm_bound], project_id, project_root, start,
            )
            for i, local in zip(llm_bound, llm_decisions, strict=True):
                decision = _to_input_space(local, llm_bound)
                # A sibling reference goes stale outside its batch, so only a
                # decision that stands alone may answer a later identical call.
                if decision.batch_target_index is None:
                    self._store_cache(hashes[i], decision)
                decided[i] = decision

        stamped = [_with_stream_census(decided[i], bundle) for i, bundle in enumerate(bundles)]
        for i in llm_bound:
            logger.info('task_curator: decision %s', stamped[i].to_log_fields())
        return stamped

    async def _dispatch_deferring_raises(
        self,
        bundles: list[PreparedCandidate],
        project_id: str,
        project_root: str,
        start: float,
    ) -> list[CuratorDecision]:
        """:meth:`_dispatch_llm`, with a raise deferring just *bundles*.

        A raise means the failure mapper itself failed, say an escalator that
        cannot write its state. Only the bundles sent to the LLM share that
        failure; a sibling a guard decided keeps its decision, exactly as it
        would alone (PRD B7). A pool error defers under the pool's reason,
        anything else ``CURATOR_RAISED``.
        """
        try:
            return await self._dispatch_llm(bundles, project_id, project_root, start)
        except Exception as exc:
            reason = (
                pool_defer_reason(exc) if isinstance(exc, POOL_UNAVAILABLE_ERRORS)
                else DeferReason.CURATOR_RAISED
            )
            logger.exception(
                'task_curator: the LLM dispatch raised for %d candidate(s) of project %s; '
                'deferring them %s',
                len(bundles), project_id, reason,
            )
            return _deferred_each(bundles, reason, start)

    async def _flag_unverified_claims_of(self, candidates: list[CandidateTask]) -> None:
        """Run the claim advisory (task 2438) over *candidates*, building the probe at most once.

        The probe is built off the event loop, since construction resolves the
        git top level with a blocking subprocess, and only when some candidate
        attributes a code-level claim to prior work; extraction is pure and
        cheap, so the common claimless batch never pays for git. The checks
        then fan out concurrently, because a fabricated token always runs the
        full git-grep-then-pickaxe path (up to ~10s each).
        """
        probe: Callable[[str], bool] | None = None
        enabled = self._config.curator.recon_claim_verification_enabled
        if enabled and self._cwd is not None and any(
            extract_attributed_claims(self._claim_verification_text(candidate))
            for candidate in candidates
        ):
            probe = await asyncio.to_thread(make_source_and_history_probe, self._cwd)
        await asyncio.gather(
            *(self._maybe_flag_unverified_claims(candidate, probe=probe) for candidate in candidates),
        )

    async def _decide_before_llm(
        self,
        bundle: PreparedCandidate,
        payload_hash: str,
        project_id: str,
        project_root: str,
        start: float,
    ) -> CuratorDecision | None:
        """Guards 6-10 for one bundle: a decision that needs no LLM call, else ``None``."""
        candidate = bundle.candidate

        # 6. Ahead of the cache: a cached 'create' is stale once its first
        # caller has landed the task and ``note_created`` it.
        exact = await self._pre_llm_exact_match(candidate, project_id, project_root)
        if exact is not None:
            self._store_cache(payload_hash, exact)
            return exact

        # 7.
        cached = self._check_cache(payload_hash)
        if cached is not None:
            return cached

        # 8. After the exact match and the cache, so a re-filed operational ask
        # folds into its existing gate rather than spawning a second one.
        route = await self._maybe_route_deterministic(candidate, payload_hash)
        if route is not None:
            return route

        # 9. A failed assembly's empty pool only LOOKS like a real one; the
        # prompt would tell the LLM to answer 'create' over it.
        if bundle.corpus_error is not None:
            return _corpus_failed_deferral(candidate, bundle.corpus_error, start)

        # 10. Only the LLM call is the cost the breaker exists to avoid.
        deferred = self._defer_while_breaker_open([bundle], start)
        return deferred[0] if deferred is not None else None

    def _defer_while_breaker_open(
        self, bundles: Sequence[PreparedCandidate], start: float,
    ) -> list[CuratorDecision] | None:
        """One breaker deferral per bundle while the zero-output breaker is open, else ``None``."""
        if not self._zero_output_breaker.is_open(time.monotonic()):
            return None
        logger.warning(
            'task_curator: zero-output-breaker open — deferring curation '
            'for candidates %r (consecutive ZOTs=%d)',
            [bundle.candidate.title for bundle in bundles],
            self._zero_output_breaker.consecutive_timeouts,
        )
        return [
            CuratorDecision.deferred(
                DeferReason.ZERO_OUTPUT_BREAKER_OPEN,
                pool_sizes=empty_pool_sizes(),
                latency_ms=_elapsed_ms(start),
            )
            for _ in bundles
        ]

    async def curate(
        self,
        candidate: CandidateTask,
        project_id: str,
        project_root: str,
    ) -> CuratorDecision:
        """Decide one candidate: :meth:`curate_prepared` over a freshly prepared batch of one.

        The corpus is assembled before any guard runs, even a guard that would
        have decided without it. A caller already holding a bundle calls
        :meth:`curate_prepared` instead.
        """
        bundle = await self.prepare_candidate(candidate, project_id, project_root)
        [decision] = await self.curate_prepared([bundle], project_id, project_root)
        return decision

    async def prepare_candidate(
        self,
        candidate: CandidateTask,
        project_id: str,
        project_root: str,
    ) -> PreparedCandidate:
        """Build the corpus + section-token estimate for one candidate.

        Separate from :meth:`curate_prepared` so the interceptor's curator
        worker can accumulate prepared candidates and decide, before fetching
        the next ticket, whether the in-flight batch has already exceeded the
        soft ``batch_token_threshold``. A corpus failure yields
        :meth:`PreparedCandidate.corpus_failed`, which the pipeline defers
        rather than curating over its empty pool.
        """
        try:
            pool, pool_sizes, withheld = await self._build_corpus(
                candidate, project_id, project_root,
            )
        except _CorpusUnavailableError as exc:
            logger.warning(
                'task_curator: no corpus stream answered for candidate %r, '
                'so its curation will defer: %s',
                candidate.title,
                exc,
            )
            return PreparedCandidate.corpus_failed(
                candidate, str(exc), unavailable_streams=exc.streams,
            )
        except Exception as exc:
            logger.warning(
                'task_curator: corpus assembly failed for candidate %r, '
                'so its curation will defer: %s',
                candidate.title,
                exc,
                exc_info=True,
            )
            return PreparedCandidate.corpus_failed(candidate, str(exc))
        # batch_index=0 is fine for the estimate — only a couple of digits of
        # rendered length difference at most across realistic batch sizes.
        # The census IS included: the estimate has to be taken over the section
        # actually emitted, or the accumulator under-counts every truncated
        # candidate.
        section = _render_batch_section(candidate, pool, 0, withheld, self._config.curator)
        return PreparedCandidate(
            candidate=candidate,
            pool=pool,
            pool_sizes=pool_sizes,
            prompt_tokens=estimate_tokens(section),
            withheld=withheld,
        )

    async def record_task(
        self,
        task_id: str,
        candidate: CandidateTask,
        project_id: str,
    ) -> None:
        """Upsert a task's embedding into the curator corpus.

        Called after a successful create. Fire-and-forget from the interceptor.
        """
        try:
            collection = await self._ensure_collection(project_id)
            embedder = await self._get_embedder()
            text = embedding_text(
                candidate.title, candidate.description, candidate.files_to_modify,
            )
            embedding = await embedder.create(text)

            from qdrant_client.models import PointStruct

            point_id = self._point_id(project_id, task_id)
            client = await self._get_qdrant()
            await client.upsert(
                collection_name=collection,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            'task_id': task_id,
                            'title': candidate.title,
                            'description': candidate.description[:1000],
                            'files_to_modify': candidate.files_to_modify or [],
                            'priority': candidate.priority or DEFAULT_PRIORITY,
                            'project_id': project_id,
                            'updated_at': datetime.now(UTC).isoformat(),
                        },
                    ),
                ],
            )
        except Exception:
            logger.warning(
                'task_curator: record_task failed for %s', task_id, exc_info=True,
            )

    async def reembed_task(
        self,
        task_id: str,
        candidate: CandidateTask,
        project_id: str,
    ) -> None:
        """Re-upsert a task's embedding after its description/details changed.

        Same deterministic point id as record_task, so this is idempotent.
        """
        await self.record_task(task_id, candidate, project_id)

    async def evict_task(self, project_id: str, task_id: str) -> None:
        """Delete a task's vector from the curator corpus.

        Called when a task is removed, so its embedding never lingers as a
        stale duplicate-detection neighbour. Reuses record_task's
        deterministic point-id scheme (_point_id) to target the exact point
        that record_task/backfill_corpus wrote. Best-effort — degrades to a
        logged warning on any error (see caller wrapping).
        """
        try:
            client = await self._get_qdrant()
            collection = self._collection_name(project_id)
            if not await client.collection_exists(collection):
                return

            from qdrant_client.models import PointIdsList

            point_id = self._point_id(project_id, task_id)
            await client.delete(
                collection_name=collection,
                points_selector=PointIdsList(points=[point_id]),
            )
        except Exception:
            logger.warning(
                'task_curator: evict_task failed for %s', task_id, exc_info=True,
            )

    @staticmethod
    def _point_id(project_id: str, task_id: str) -> str:
        """Deterministic UUID5 point ID for a task in a project.

        Shared by record_task() and backfill_corpus() to ensure idempotent overlap.
        """
        return str(uuid_mod.uuid5(uuid_mod.NAMESPACE_URL, f'{project_id}/{task_id}'))

    async def corpus_count(self, project_id: str) -> int:
        """Return the number of points in the curator corpus for a project.

        Returns 0 if the collection does not exist or an error occurs.
        This public method encapsulates collection-existence + count logic so
        callers (e.g. TaskInterceptor) don't need to access private methods.
        """
        try:
            client = await self._get_qdrant()
            collection_name = self._collection_name(project_id)
            if not await client.collection_exists(collection_name):
                return 0
            result = await client.count(collection_name=collection_name)
            return result.count
        except Exception:
            logger.warning(
                'task_curator: corpus_count failed for project %s', project_id, exc_info=True,
            )
            return 0

    async def search_corpus(
        self,
        query: str,
        project_id: str,
        *,
        limit: int = 10,
        score_threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Semantic nearest-neighbour search over the curator corpus.

        Read-only: embeds the raw ``query`` string and returns the most
        similar filed tasks for ``project_id`` ranked by cosine similarity
        (higher score = more similar). Never creates the collection — a
        missing collection (no tasks ever filed) yields ``[]``.

        Args:
            query: Free-text query, embedded as-is (NOT passed through
                ``embedding_text``, which composes stored-task fields).
            project_id: Project identifier selecting the collection.
            limit: Max number of hits to return.
            score_threshold: Drop hits below this cosine score server-side.

        Returns:
            List of dicts ``{task_id, title, description, files_to_modify,
            priority, updated_at, score}``. Returns ``[]`` on any error
            (Qdrant/embedder unavailable), mirroring corpus_count's
            best-effort degradation.
        """
        try:
            client = await self._get_qdrant()
            name = self._collection_name(project_id)
            if not await client.collection_exists(name):
                return []
            embedding = await (await self._get_embedder()).create(query)
            results = await client.query_points(
                collection_name=name,
                query=embedding,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )
            hits: list[dict[str, Any]] = []
            for point in results.points:
                payload = point.payload or {}
                hits.append(
                    {
                        'task_id': payload.get('task_id'),
                        'title': payload.get('title'),
                        'description': payload.get('description'),
                        'files_to_modify': payload.get('files_to_modify', []),
                        'priority': payload.get('priority'),
                        'updated_at': payload.get('updated_at'),
                        'score': point.score,
                    },
                )
            return hits
        except Exception:
            logger.warning(
                'task_curator: search_corpus failed for project %s', project_id, exc_info=True,
            )
            return []

    async def backfill_corpus(
        self,
        tasks: list[dict],
        project_id: str,
    ) -> BackfillResult:
        """Upsert embeddings for a flat list of existing task dicts.

        Idempotent: uses the same deterministic UUID5 point-ID scheme as
        record_task(). Re-running just re-upserts the same points.

        Args:
            tasks: Flat list of task dicts (as returned by flatten_task_tree).
            project_id: Project identifier used for the collection name and point IDs.

        Returns:
            BackfillResult with counts of upserted, skipped, and error tasks.
        """
        if not tasks:
            return BackfillResult()

        result = BackfillResult()
        collection = await self._ensure_collection(project_id)
        embedder = await self._get_embedder()
        client = await self._get_qdrant()

        from qdrant_client.models import PointStruct

        # Embed all tasks with bounded concurrency.
        sem = asyncio.Semaphore(10)
        points: list[PointStruct] = []

        async def _embed_one(task: dict) -> None:
            task_id = str(task.get('id', '') or '')
            title = str(task.get('title', '') or '').strip()
            if not title:
                result.skipped += 1
                return

            description = str(task.get('description', '') or '')
            files = _task_files(task)
            text = embedding_text(title, description, files)

            try:
                async with sem:
                    embedding = await embedder.create(text)
            except Exception:
                logger.warning(
                    'task_curator: backfill embed failed for task %s', task_id, exc_info=True,
                )
                result.errors += 1
                return

            point_id = self._point_id(project_id, task_id)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        'task_id': task_id,
                        'title': title,
                        'description': description[:1000],
                        'files_to_modify': files,
                        'priority': task.get('priority') or DEFAULT_PRIORITY,
                        'project_id': project_id,
                        'updated_at': datetime.now(UTC).isoformat(),
                    },
                )
            )

        await asyncio.gather(*[_embed_one(t) for t in tasks])

        if points:
            await client.upsert(collection_name=collection, points=points)
            result.upserted = len(points)

        logger.info(
            'task_curator: backfill_corpus complete project=%s upserted=%d skipped=%d errors=%d',
            project_id, result.upserted, result.skipped, result.errors,
        )
        return result

    async def prune_orphans(self, project_id: str, live_task_ids) -> PruneResult:
        """Health-gated reconciliation sweep: delete corpus points for tasks
        that no longer exist in a live snapshot.

        Orphans are computed by mapping ``live_task_ids`` through the same
        deterministic point-id scheme record_task/backfill_corpus use to
        write points (_point_id), then diffing against the actual corpus
        point ids — robust to payload drift, no reliance on payload task_id.

        Caller contract — tag scope: the corpus is keyed by ``project_id``
        alone (``_point_id``/``_collection_name`` take no ``tag``), and
        ``record_task``/``index_committed_tasks`` record a task's vector
        regardless of which tag it was filed under. ``live_task_ids`` MUST
        therefore already aggregate every tag the corpus could contain a
        vector for — if it only reflects a single tag's tasks, tasks filed
        under any other tag will look orphaned and be deleted here. As of
        task 2603 the sole caller (``BackfillManager.prune``) satisfies
        this: it enumerates every tag via ``list_tags`` and unions
        ``get_tasks(project_root, tag=t)`` across all of them before
        calling this method, failing safe (skipping the sweep entirely)
        if any tag's read fails — see the matching note on
        ``BackfillManager.prune``.
        """
        live = {str(t) for t in live_task_ids}
        if not live:
            # Hard gate: a total live-read failure surfaces as an empty
            # set. Without this, an empty live set would make every corpus
            # point look orphaned and prune_orphans would nuke the whole
            # corpus. Refuse before ever touching Qdrant.
            logger.warning(
                'task_curator: prune_orphans refused — empty live snapshot '
                'for %s (fail-safe, no prune)', project_id,
            )
            return PruneResult(skipped=True, reason='empty_live_snapshot')

        try:
            client = await self._get_qdrant()
            collection = self._collection_name(project_id)
            if not await client.collection_exists(collection):
                return PruneResult(live=len(live))

            corpus_ids: list[str] = []
            offset = None
            while True:
                records, offset = await client.scroll(
                    collection_name=collection,
                    limit=256,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                corpus_ids.extend(str(r.id) for r in records)
                if offset is None:
                    break

            live_point_ids = {self._point_id(project_id, tid) for tid in live}
            orphans: list[ExtendedPointId] = [
                pid for pid in corpus_ids if pid not in live_point_ids
            ]

            if orphans:
                from qdrant_client.models import PointIdsList

                await client.delete(
                    collection_name=collection,
                    points_selector=PointIdsList(points=orphans),
                )

            return PruneResult(
                pruned=len(orphans), corpus_scanned=len(corpus_ids), live=len(live),
            )
        except Exception:
            logger.warning(
                'task_curator: prune_orphans failed for %s', project_id, exc_info=True,
            )
            return PruneResult(skipped=True, reason='error', live=len(live))

    async def close(self) -> None:
        """Archive and remove this process's config dir, then release the clients.

        Archival goes first, so a hanging client close cannot starve it. The
        thread hop holds the delete as well as the archival, so a cancelled
        await cannot skip one and keep the other, which is the hazard
        ``archive_before_delete``'s no-``to_thread`` note guards against. A
        cancellation landing on that await still releases the clients.
        """
        config_dir, self._config_dir = self._config_dir, None
        archive_root = self._transcript_archive_root()
        try:
            if config_dir is not None and archive_root is not None:
                await asyncio.to_thread(
                    _archive_curator_config_dir, config_dir.path, archive_root,
                )
        finally:
            await self._close_clients()

    async def _close_clients(self) -> None:
        if self._qdrant_client is not None:
            import contextlib as _ctx

            with _ctx.suppress(Exception):
                await self._qdrant_client.close()
            self._qdrant_client = None
        embedder, self._embedder = self._embedder, None
        await close_wrapped_sdk_clients([embedder])

    # ------------------------------------------------------------------
    # Idempotency cache
    # ------------------------------------------------------------------

    def _check_cache(self, payload_hash: str) -> CuratorDecision | None:
        ttl = self._config.curator.idempotency_ttl_seconds
        now = time.monotonic()
        # Lazy eviction
        stale = [
            k for k, (_, ts) in self._decision_cache.items() if now - ts > ttl
        ]
        for k in stale:
            del self._decision_cache[k]
        entry = self._decision_cache.get(payload_hash)
        if entry is None:
            return None
        decision, ts = entry
        if now - ts > ttl:
            del self._decision_cache[payload_hash]
            return None
        return decision

    def _store_cache(self, payload_hash: str, decision: CuratorDecision) -> None:
        if decision.action == 'defer':
            return
        self._decision_cache[payload_hash] = (decision, time.monotonic())

    # ------------------------------------------------------------------
    # Corpus assembly
    # ------------------------------------------------------------------

    async def _build_corpus(
        self,
        candidate: CandidateTask,
        project_id: str,
        project_root: str,
    ) -> tuple[list[PoolEntry], dict[str, int], PoolWithheld]:
        """Assemble the candidate's pool for the LLM prompt, one stream per ``POOL_SOURCES`` entry.

        Returns ``(pool, pool_sizes, withheld)``. ``withheld`` is the
        :class:`PoolWithheld` census of everything the caps kept out, counted
        at EVERY cap rather than only at the final :func:`_trim_pool` — a
        stream cap discards entries the trim never sees, so the stream caps are
        where much of what the pool loses is lost (see the pool-cap block in
        ``config/schema.py::CuratorConfig``). A stream whose read fails is
        logged at WARNING and named in ``withheld.unavailable_streams``; when
        every stream it attempted failed, it raises ``_CorpusUnavailableError``
        instead, since an empty pool would then only look like no overlap.
        """
        # Resolve lock_depth PER PROJECT, not from a single global scalar:
        # fused-memory serves many projects and each orchestrator resolves its
        # own depth (3..12 across the fleet). The scheduler snapshot already
        # publishes the effective value, so we read it rather than
        # reimplementing orchestrator.config's layering here. lock_depth is
        # red-tier (changes only at orchestrator restart), so an hours-old
        # snapshot still carries the correct depth. Absent/unreadable snapshot
        # falls back COARSE (the config scalar) — see effective_lock_depth.
        #
        # Dispatched via to_thread because the underlying snapshot read is
        # blocking, matching recon_write_policy's handling of the same file.
        # Unlike that call site there is no existing thread hop to ride here
        # (_build_corpus is awaited straight off the event loop), so this adds
        # its own. It stays cheap because effective_lock_depth memoises per
        # project — this runs once per candidate but only reaches the disk
        # once per project per TTL, and the hop is noise next to the taskmaster
        # and embedding round-trips below.
        lock_depth = await asyncio.to_thread(
            effective_lock_depth, project_root, self._config.curator.lock_depth,
        )
        pool: list[PoolEntry] = []
        seen_ids: set[str] = set()
        reads = _StreamReads()
        has_store = self._taskmaster is not None

        def admit(entries: Iterable[PoolEntry]) -> None:
            for entry in entries:
                pool.append(entry)
                seen_ids.add(entry.task_id)

        # Stream 1: review-chain anchor. The store holds integer ids only, so
        # any other spawned_from names no task.
        anchor_row: dict | None = None
        if has_store and candidate.spawned_from and candidate.spawned_from.isdecimal():
            anchor_row = await reads.read(
                'anchor', self._anchor_row(candidate.spawned_from, project_root), None,
            )
        anchor_entry = _to_pool_entry(anchor_row, source='anchor', lock_depth=lock_depth)
        if anchor_entry is not None:
            admit([anchor_entry])

        # Stream 2: the parent's other follow-ups, for an agent-followup candidate
        sibling_overflow_ids: list[str] = []
        parent_id = _followup_parent_id(candidate)
        if has_store and parent_id is not None and self._config.curator.pool_sibling_cap >= 1:
            sibling_entries, sibling_overflow_ids = await reads.read(
                'sibling',
                self._sibling_stream(
                    parent_id, project_root, lock_depth=lock_depth, exclude=seen_ids,
                ),
                ([], []),
            )
            admit(sibling_entries)

        # Stream 3: module-lock pool
        candidate_modules = set(
            files_to_modules(candidate.files_to_modify, depth=lock_depth),
        )
        if not candidate_modules and anchor_entry is not None:
            candidate_modules = set(anchor_entry.module_keys)
        module_overflow_ids: list[str] = []
        if has_store and candidate_modules:
            module_entries, module_overflow_ids = await reads.read(
                'module',
                self._module_stream(
                    candidate_modules, project_root, lock_depth=lock_depth, exclude=seen_ids,
                ),
                ([], []),
            )
            admit(module_entries)

        # Stream 4: embedding neighbors
        embedding_entries, embedding_unvisited_ids, embedding_window_full = await reads.read(
            'embedding',
            self._embedding_stream(
                candidate, project_id, project_root, lock_depth=lock_depth, exclude=seen_ids,
            ),
            ([], [], False),
        )
        admit(embedding_entries)

        # Stream 5: dependency neighbors of the anchor
        dep_overflow_ids: list[str] = []
        if anchor_row is not None and anchor_entry is not None:
            dep_entries, dep_overflow_ids = await reads.read(
                'dependency',
                self._dependency_stream(
                    anchor_row, anchor_entry.task_id, project_root,
                    lock_depth=lock_depth, exclude=seen_ids,
                ),
                ([], []),
            )
            admit(dep_entries)

        if reads.every_attempt_failed():
            raise _CorpusUnavailableError(tuple(reads.unavailable))

        # Every stream has run, so seen_ids is now the complete admitted set
        # and a stream's overflow can be resolved into what the pool actually
        # lacks. Counting against seen_ids rather than the post-trim pool
        # keeps the stream counts and total_cap DISJOINT by construction: a
        # stream can only count ids that were never admitted, the trim only
        # ids that were, so `total` stays a count of distinct absent entries.
        sibling_withheld = _count_absent(sibling_overflow_ids, seen_ids)
        module_withheld = _count_absent(module_overflow_ids, seen_ids)
        embedding_withheld = _count_absent(embedding_unvisited_ids, seen_ids)
        dependency_withheld = _count_absent(dep_overflow_ids, seen_ids)

        # Final cap — trim weakest entries first (POOL_SOURCES reversed).
        pool, total_cap_dropped = _trim_pool(
            pool, self._config.curator.pool_total_cap,
        )

        pool_sizes = {
            source: sum(1 for e in pool if e.source == source) for source in POOL_SOURCES
        }
        withheld = PoolWithheld(
            by_source={
                'sibling': sibling_withheld,
                'module': module_withheld,
                'embedding': embedding_withheld,
                'dependency': dependency_withheld,
                'total_cap': total_cap_dropped,
            },
            caps={
                'sibling': self._config.curator.pool_sibling_cap,
                'module': self._config.curator.pool_module_cap,
                'embedding': self._config.curator.pool_embedding_cap,
                'dependency': self._config.curator.pool_dependency_cap,
                'total_cap': self._config.curator.pool_total_cap,
            },
            lower_bounds=('embedding',) if embedding_window_full else (),
            unavailable_streams=tuple(reads.unavailable),
        )
        return pool, pool_sizes, withheld

    async def _anchor_row(self, spawned_from: str, project_root: str) -> dict | None:
        """The review-chain anchor's task row, or ``None`` when no such task exists."""
        if self._taskmaster is None:
            return None
        try:
            return await self._taskmaster.get_task(spawned_from, project_root)
        except TaskNotFoundError:
            return None

    async def _module_stream(
        self,
        candidate_modules: set[str],
        project_root: str,
        *,
        lock_depth: int,
        exclude: set[str],
    ) -> tuple[list[PoolEntry], list[str]]:
        """The open-backlog tasks sharing a module with the candidate, ranked and capped.

        Returns ``(entries, overflow_ids)``: the admitted entries, and the
        ranked ids that ``pool_module_cap`` kept out.
        """
        if self._taskmaster is None:
            return [], []
        tasks_result = await self._taskmaster.get_tasks(
            project_root, statuses=list(_OPEN_BACKLOG_STATUSES),
        )
        matches: list[PoolEntry] = []
        for t in flatten_task_tree(tasks_result):
            tid = str(t.get('id', ''))
            if not tid or tid in exclude:
                continue
            # The read already asked for the open backlog; a backend that
            # ignored the filter must not widen the stream.
            if str(t.get('status', '')) not in _OPEN_BACKLOG_STATUSES:
                continue
            task_modules = set(files_to_modules(_task_files(t), depth=lock_depth))
            if not (task_modules & candidate_modules):
                continue
            entry = _to_pool_entry(t, source='module', lock_depth=lock_depth)
            if entry is not None:
                matches.append(entry)
        matches.sort(key=_module_sort_key)
        cap = self._config.curator.pool_module_cap
        # The sort key is status+priority, NOT relevance, so the entries this
        # cap discards are not the least-similar ones — a genuinely
        # overlapping task can sit just past it. Which of them the POOL
        # actually loses is not known yet: later streams can still admit one,
        # so the caller counts the absent ones once every stream has run.
        return matches[:cap], [e.task_id for e in matches[cap:]]

    async def _embedding_stream(
        self,
        candidate: CandidateTask,
        project_id: str,
        project_root: str,
        *,
        lock_depth: int,
        exclude: set[str],
    ) -> tuple[list[PoolEntry], list[str], bool]:
        """The candidate's nearest corpus neighbours by embedding, capped.

        Returns ``(entries, unvisited_ids, window_full)``: the admitted
        entries, the neighbour ids the cap left unvisited, and whether the
        retrieval window was full, which makes that count a floor.
        """
        collection = await self._ensure_collection(project_id)
        embedder = await self._get_embedder()
        text = embedding_text(candidate.title, candidate.description, candidate.files_to_modify)
        embedding = await embedder.create(text)
        client = await self._get_qdrant()
        cap = self._config.curator.pool_embedding_cap
        # Over-fetch so filtering out stream 1-3 overlaps still leaves K.
        overfetch = cap + 20
        results = await client.query_points(
            collection_name=collection,
            query=embedding,
            limit=overfetch,
            with_payload=True,
        )
        points = list(results.points)
        entries: list[PoolEntry] = []
        for i, point in enumerate(points):
            payload = point.payload or {}
            tid = str(payload.get('task_id', ''))
            if not tid or tid in exclude:
                continue
            # We need full task data — try to fetch it, fall back to the
            # qdrant payload if taskmaster is unavailable.
            entry = await self._fetch_entry_for_neighbor(
                tid, payload, source='embedding',
                lock_depth=lock_depth, project_root=project_root,
            )
            if entry is None:
                continue
            entries.append(entry)
            if len(entries) >= cap:
                # The neighbours the cap left unvisited, by id — the caller
                # counts them against what the pool finally holds, so a
                # neighbour another stream already pooled is not reported as
                # lost. It stays an upper bound in one narrower respect: an
                # unvisited point might have been unresolvable via
                # _fetch_entry_for_neighbor and so never eligible. These are
                # ordered by embedding distance, so the closest of them are
                # the most likely duplicates in the corpus.
                unvisited_ids = [str((p.payload or {}).get('task_id', '')) for p in points[i + 1:]]
                # And a hard ceiling in the other direction, unlike the
                # module/dependency streams, which are exhaustive over the
                # open backlog: this tail can hold at most `overfetch - cap`
                # ids because that is all qdrant was asked for. A full window
                # means neighbours past it were never fetched, so the count is
                # a FLOOR and is reported as one rather than as a count of
                # everything excluded.
                return entries, unvisited_ids, len(points) >= overfetch
        return entries, [], False

    async def _dependency_stream(
        self,
        anchor_row: dict,
        anchor_id: str,
        project_root: str,
        *,
        lock_depth: int,
        exclude: set[str],
    ) -> tuple[list[PoolEntry], list[str]]:
        """The anchor's open-backlog dependencies and dependents, in id order and capped.

        Only the admitted ids are read in full, so the stream costs at most
        ``pool_dependency_cap`` row reads. Returns ``(entries, overflow_ids)``.
        """
        taskmaster = self._taskmaster
        if taskmaster is None:
            return [], []
        edges = await taskmaster.get_dependency_edges(project_root)
        neighbours = (
            set(_task_dependencies(anchor_row)) | _dependents_of(anchor_id, edges)
        ) - exclude - {anchor_id}
        if not neighbours:
            return [], []
        statuses = await taskmaster.get_statuses_raw(project_root, ids=sorted(neighbours))
        open_ids = sorted(
            (tid for tid, status in statuses.items()
             if tid in neighbours and status in _OPEN_BACKLOG_STATUSES),
            key=int,
        )
        cap = self._config.curator.pool_dependency_cap
        entries: list[PoolEntry] = []
        for tid in open_ids[:cap]:
            try:
                task = await taskmaster.get_task(tid, project_root)
            except TaskNotFoundError:
                continue
            entry = _to_pool_entry(task, source='dependency', lock_depth=lock_depth)
            if entry is not None:
                entries.append(entry)
        return entries, open_ids[cap:]

    async def _sibling_stream(
        self,
        parent_id: str,
        project_root: str,
        *,
        lock_depth: int,
        exclude: set[str],
    ) -> tuple[list[PoolEntry], list[str]]:
        """The non-cancelled tasks sharing *parent_id* as ``spawned_from``, ranked and capped.

        It runs ahead of the module, embedding and dependency streams, so a
        sibling is never left to rank in by its files or its embedding.
        Returns ``(entries, overflow_ids)``: the admitted entries, and the
        ranked ids that ``pool_sibling_cap`` kept out.
        """
        taskmaster = self._taskmaster
        cap = self._config.curator.pool_sibling_cap
        if taskmaster is None:
            return [], []
        rows = await self._sibling_lookup_rows(taskmaster, parent_id, project_root)
        ranked = sorted(
            (row for row in rows if str(row['task_id']) not in exclude),
            key=_sibling_rank,
        )
        entries: list[PoolEntry] = []
        overflow_ids: list[str] = []
        for row in ranked:
            tid = str(row['task_id'])
            if len(entries) >= cap:
                overflow_ids.append(tid)
                continue
            entry = await self._fetch_entry_for_neighbor(
                tid, dict(row), source='sibling',
                lock_depth=lock_depth, project_root=project_root,
            )
            if entry is not None:
                entries.append(entry)
        return entries, overflow_ids

    @staticmethod
    async def _sibling_lookup_rows(
        taskmaster: TaskBackendProtocol, parent_id: str, project_root: str,
    ) -> list[MetadataLookupRow]:
        """The tasks naming *parent_id* in any stored form; a form whose lookup fails is skipped.

        Each form is looked up on its own, so one failure keeps the rows the
        other forms returned. A decimal id past SQLite's INTEGER range is one
        such failure: its int form is refused before the store is read. When
        EVERY form fails, the last failure is raised: nothing the store
        vouched for came back.
        """
        rows: dict[int, MetadataLookupRow] = {}
        failure: Exception | None = None
        answered = False
        for value in sibling_lookup_values(parent_id):
            try:
                page = await taskmaster.find_tasks_by_metadata(
                    project_root, 'spawned_from', value,
                    statuses=list(NON_CANCELLED_STATUSES), limit=_SIBLING_LOOKUP_PAGE,
                )
            except Exception as exc:
                logger.warning(
                    'task_curator: sibling lookup failed for parent %s as %r: %s',
                    parent_id, value, exc,
                )
                failure = exc
                continue
            answered = True
            rows.update((row['task_id'], row) for row in page['results'])
        if not answered and failure is not None:
            raise failure
        return list(rows.values())

    async def _fetch_entry_for_neighbor(
        self,
        tid: str,
        payload: dict,
        *,
        source: str,
        lock_depth: int,
        project_root: str,
    ) -> PoolEntry | None:
        """Fetch a full task for an embedding neighbour or a sibling, falling back to *payload*.

        The fallback is for an inconclusive read only: a row that was read and
        that ``_to_pool_entry`` rejects (a follow-up holder) is excluded.

        ``get_task`` is called without a ``tag`` — it resolves to
        ``DEFAULT_TAG``. That does NOT match the Qdrant corpus's own scope:
        the corpus is keyed by ``project_id`` ALONE (see the tag-scope
        caller-contract note on ``prune_orphans`` above), and
        ``record_task``/``index_committed_tasks`` write a task's vector
        regardless of which tag it was filed under. So a
        ``TaskNotFoundError`` here does NOT prove the task is gone — it
        also fires for a live task filed under a non-default tag, which is
        then excluded from the pool exactly like a genuine deletion (task
        2521 RC2). This is an accepted tradeoff, not a correctness proof:
        worst case a near-identical re-file of a non-default-tag task is
        filed as a duplicate instead of combined — the same "duplicate is
        cheap" bias this method already applies to inconclusive errors
        below. As of task 2603 the orphan-prune sweep
        (``BackfillManager.prune``/``prune_orphans``) is tag-complete;
        this pool-building lookup's own ``DEFAULT_TAG``-only ``get_task``
        call remains a separate, accepted limitation — not fixed here.
        """
        if self._taskmaster is not None:
            try:
                task = await self._taskmaster.get_task(tid, project_root)
                return _to_pool_entry(task, source=source, lock_depth=lock_depth)
            except TaskNotFoundError:
                # Zero-row absence under DEFAULT_TAG: usually the task was
                # removed and this Qdrant point is an orphan, but it also
                # fires for a live non-default-tag task (see the docstring
                # tag-scope note above) — exclude either way rather than
                # fabricating a live-looking entry from the stale payload.
                # A duplicate is a cheap failure; silently losing a real
                # candidate to a fabricated "existing" match is not.
                return None
            except Exception:
                # Inconclusive (e.g. TASKMASTER_UNAVAILABLE, a raw driver
                # error, or _to_pool_entry raising on a malformed task): bias
                # to the cheap failure and keep the conservative thin entry
                # built below, rather than risk dropping a live candidate.
                pass
        # Fallback: build a thin entry from the qdrant payload or lookup row
        return PoolEntry(
            task_id=tid,
            title=str(payload.get('title', '')),
            description=str(payload.get('description', '')),
            details='',
            files_to_modify=list(payload.get('files_to_modify', []) or []),
            module_keys=files_to_modules(
                payload.get('files_to_modify', []) or [], depth=lock_depth,
            ),
            status='unknown',
            priority=str(payload.get('priority', DEFAULT_PRIORITY)),
            source=source,
            combine_eligible=False,  # create-safe: neither combine- nor drop-eligible
        )

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _transcript_config_dir(self) -> TaskConfigDir | None:
        """This process's curator config dir, or None when there is no UsageGate.

        Gate-less, no per-call OAuth token is passed, so the CLI needs the login
        in its default config dir, which an isolated one would hide. Concurrent
        gated calls share this dir safely: each authenticates from its own
        ``CLAUDE_CODE_OAUTH_TOKEN``, never from the ``.credentials.json`` they
        all rewrite (``shared/tests/test_config_dir_credentials_live.py``).

        The dead-PID sweep runs on a worker thread and is not awaited: it
        reclaims other processes' leftovers, so no call depends on it, and a
        readdir of a large system tempdir must stall neither the server's event
        loop nor this call (task 7335). It archives each dead dir's transcripts
        before removing the dir.
        """
        if self._usage_gate is None:
            return None
        if self._config_dir is None:
            asyncio.get_running_loop().run_in_executor(
                None,
                _sweep_stale_curator_config_dirs_once,
                self._config_dir_base,
                self._transcript_archive_root(),
            )
            self._config_dir = TaskConfigDir(
                f'{_CONFIG_DIR_TASK_PREFIX}{os.getpid()}', base_dir=self._config_dir_base,
            )
        return self._config_dir

    def _transcript_archive_root(self) -> Path | None:
        """Where curator transcripts are archived: recon's root, under recon's kill switch."""
        return recon_archive_root(self._config.reconciliation)

    def _ledger_kwargs(self, role: str) -> dict[str, Any]:
        """The ``invoke_with_cap_retry`` kwargs that land a call in the invocations ledger.

        ``run_id`` is read from the gate, so curator invocations and the gate's
        own account events share one run key.
        """
        run_id = (self._usage_gate.run_id or '') if self._usage_gate is not None else ''
        return {'cost_store': self._cost_store, 'role': role, 'run_id': run_id}

    def _transcript_scope(self, timeout_seconds: float) -> _TranscriptScope | None:
        """A fresh transcript scope for one call, or None when there is no UsageGate."""
        config_dir = self._transcript_config_dir()
        if config_dir is None:
            return None
        return _TranscriptScope(config_dir, str(uuid_mod.uuid4()), timeout_seconds)

    async def _read_failure_evidence(
        self, transcript_scope: _TranscriptScope,
    ) -> TranscriptEvidence | None:
        """What a failed call's own transcript recorded, or None when unknown.

        Reads once per failed call, off the event loop (a bisected batch fails
        as several concurrent calls), and never raises: a transcript fault must
        not replace the LLM failure on the synchronous add_task path.
        """
        session_id = transcript_scope.session_id
        try:
            evidence = await asyncio.to_thread(
                transcript_evidence_for_session, transcript_scope.config_dir.path, session_id,
            )
        except Exception:
            logger.warning(
                'TaskCurator: could not read the transcript of failed session %s; '
                'reporting the failure without it', session_id, exc_info=True,
            )
            return None
        if evidence is not None and evidence.other_tool_uses:
            logger.warning(
                'TaskCurator: pure-classifier contract violated: failed session %s '
                'used tools %s over %d assistant turn(s), but its registry should '
                "hold only StructuredOutput. Check the schema + '*' -> --tools '' "
                'substitution in shared.cli_invoke.build_claude_argv.',
                session_id, list(evidence.other_tool_uses), evidence.assistant_turns,
            )
        return evidence

    async def _resolve_failed_result(
        self, agent_result: AgentResult, transcript_scope: _TranscriptScope | None,
    ) -> tuple[AgentResult, TranscriptEvidence | None]:
        """Read a failed call's transcript evidence, and salvage its verdict if allowed.

        Returns ``(result, evidence)``. A successful or gate-less result comes
        back unchanged and unread. A failed gated one gets ONE evidence read,
        for reporting.
        Salvage is only for a run KILLED after the CLI accepted its verdict (the
        transcript's ``structured_output`` attachment). The result then becomes
        a success carrying that verdict, which meets exactly the parsing and
        validation a returned one does. A denied or non-kill failure is never
        salvaged.

        Known limitation: when ``invoke_with_cap_retry`` re-mints the session id
        for a fresh retry (after a cap hit or a pre-turn rejection,
        ``_reset_for_fresh_retry``), the curator's session id names an EARLIER
        attempt. Evidence is then absent or stale. That can only cause a missed
        salvage, never a false one, since a capped or rejected attempt has no
        acceptance record.
        """
        if agent_result.success or transcript_scope is None:
            return agent_result, None
        evidence = await self._read_failure_evidence(transcript_scope)
        verdict = _salvageable_verdict(agent_result, evidence)
        if verdict is None:
            return agent_result, evidence
        logger.warning(
            'TaskCurator: salvaged an accepted verdict from the transcript of killed '
            'session %s (subtype=%s, transcript_turns=%s); stdout never delivered it: %.300s',
            transcript_scope.session_id, agent_result.subtype,
            agent_result.transcript_turns, json.dumps(verdict, default=str),
        )
        salvaged = replace(
            agent_result, success=True, structured_output=verdict, schema_salvaged=True,
        )
        return salvaged, evidence

    async def _invoke_curator(self, prompt: str, call: _CuratorCall) -> AgentResult:
        """One LLM round-trip for *call*, returning a successful result.

        A failed call raises :exc:`CuratorFailureError` (see
        :func:`_failure_error`), unless its transcript holds a verdict the CLI
        accepted, which :meth:`_resolve_failed_result` salvages into a success.
        """
        system_prompt = self._resolve_curator_prompt(call.spec)
        transcript_scope = self._transcript_scope(call.timeout_seconds)

        agent_result: AgentResult = await self._invoke(
            usage_gate=self._usage_gate,
            label=call.label,
            project_id=call.project_id,
            prompt=prompt,
            system_prompt=system_prompt,
            # Task 1989: neutral CLI cwd decouples per-call cost from the filing
            # project's CLAUDE.md/MEMORY.md; self._cwd stays project-root for
            # Python-side premise/blocklist resolution.
            cwd=neutral_cli_cwd(),
            model=self._config.curator.model,
            max_turns=call.max_turns,
            max_budget_usd=call.budget_usd,
            # Pure classifier: build_claude_argv turns '*' into --tools '', which
            # removes built-in and deferred tools; the two MCP kwargs remove MCP.
            disallowed_tools=['*'],
            mcp_config=no_mcp_servers_config(),
            strict_mcp_config=True,
            output_schema=call.output_schema,
            permission_mode='bypassPermissions',
            timeout_seconds=call.timeout_seconds,
            cap_wait_sanity_secs=_CURATOR_CAP_WAIT_SANITY_SECS,
            park_on_frozen_pool=False,
            **self._ledger_kwargs(call.ledger_role),
            **(transcript_scope.as_invoke_kwargs() if transcript_scope else {}),
        )

        agent_result, evidence = await self._resolve_failed_result(agent_result, transcript_scope)
        if not agent_result.success:
            raise _failure_error(agent_result, evidence, call)
        return agent_result

    async def _dispatch_llm(
        self,
        bundles: list[PreparedCandidate],
        project_id: str,
        project_root: str,
        start: float,
    ) -> list[CuratorDecision]:
        """Judge *bundles* in one LLM call: the single spec for one, the batch spec for more.

        The only caller of :meth:`_invoke_curator`. An open zero-output
        breaker defers the call's bundles before any invocation, at every
        level: the top-level call, each bisected half and each leaf. A failed
        call over two or more bundles bisects, each half dispatched again over
        its own prepared pools and never back through the guard chain, so the
        bottom of a bisection is a single-spec invocation. A pool error is not
        bisected: a smaller batch cannot cure an account pool that cannot
        serve, and each half would repeat the cap-retry wait. Every failure
        that is not bisected is decided by :meth:`_failure_decision`.

        Decisions come back in *bundles*-local index space.
        """
        deferred = self._defer_while_breaker_open(bundles, start)
        if deferred is not None:
            return deferred
        cfg = self._config.curator
        if len(bundles) == 1:
            [only] = bundles
            prompt = _render_single_prompt(only.candidate, only.pool, only.withheld, cfg)
            call = _CuratorCall.single(cfg, project_id, len(only.pool))
        else:
            prompt = _render_batch_prompt(
                [bundle.candidate for bundle in bundles],
                [bundle.pool for bundle in bundles],
                [bundle.withheld for bundle in bundles],
                cfg,
            )
            call = _CuratorCall.batch(cfg, project_id, len(bundles))
        try:
            agent_result = await self._invoke_curator(prompt, call)
        except Exception as exc:
            if len(bundles) >= 2 and not isinstance(exc, POOL_UNAVAILABLE_ERRORS):
                return await self._bisect(bundles, exc, project_id, project_root, start)
            return [
                await self._failure_decision(
                    exc, bundle, project_id=project_id, project_root=project_root, start=start,
                )
                for bundle in bundles
            ]
        return await self._decisions_from_result(
            agent_result, call, bundles, project_id, project_root, start,
        )

    async def _bisect(
        self,
        bundles: list[PreparedCandidate],
        exc: Exception,
        project_id: str,
        project_root: str,
        start: float,
    ) -> list[CuratorDecision]:
        """Dispatch each half of a failed batch, concurrently, rebasing the right half's indices.

        A half that raises cancels its sibling before the raise propagates, so
        a batch the pipeline has given up on leaves no LLM call running.
        """
        mid = len(bundles) // 2
        detail = (
            f'subtype={exc.subtype} timed_out={exc.timed_out}'
            if isinstance(exc, CuratorFailureError)
            else exception_summary(exc)
        )
        logger.warning(
            'task_curator: bisecting a size-%d batch on %s (%s) → halves of %d and %d',
            len(bundles), type(exc).__name__, detail, mid, len(bundles) - mid,
        )
        async with asyncio.TaskGroup() as halves:
            left = halves.create_task(
                self._dispatch_llm(bundles[:mid], project_id, project_root, start),
            )
            right = halves.create_task(
                self._dispatch_llm(bundles[mid:], project_id, project_root, start),
            )
        return left.result() + [
            _shift_batch_target_index(decision, mid) for decision in right.result()
        ]

    async def _decisions_from_result(
        self,
        agent_result: AgentResult,
        call: _CuratorCall,
        bundles: list[PreparedCandidate],
        project_id: str,
        project_root: str,
        start: float,
    ) -> list[CuratorDecision]:
        """Parse a successful *call* into one decision per bundle, and account for it."""
        # A completed round-trip disproves a wedged backend, which is all the
        # ZOT breaker asserts, so it closes however unusable the payload turns
        # out to be (task 4143). That includes a breaker a concurrent bisect
        # half has just opened: closing that cooldown early is deliberate
        # (pinned by TestZeroOutputBreakerBatchReset).
        self._zero_output_breaker.close()

        decisions = call.parse(agent_result, bundles, _elapsed_ms(start))

        # The streak, unlike the breaker, needs a usable DECISION: the parsers
        # fail items open to a degraded 'create' without raising. One usable
        # decision suffices; the other items' degradation is then the designed
        # per-item fail-open.
        if any(not decision.degraded for decision in decisions):
            self._degraded_streaks.success(project_id)
        else:
            for bundle, decision in zip(bundles, decisions, strict=True):
                await self._count_degraded(
                    justification=decision.justification,
                    project_id=project_id,
                    project_root=project_root,
                    candidate_title=bundle.candidate.title,
                )
        return decisions

    async def _failure_decision(
        self,
        exc: Exception,
        bundle: PreparedCandidate,
        *,
        project_id: str,
        project_root: str,
        start: float,
    ) -> CuratorDecision:
        """The one mapping from a failed LLM call to *bundle*'s decision.

        A pool that cannot serve defers under its pool reason. Any other
        failure is reported through :meth:`_escalate_failure` and then defers
        ``LLM_FAILED``, except a zero-output hang, which counts on the breaker
        and creates, marked ``degraded_by_zot``.
        """
        candidate = bundle.candidate
        if isinstance(exc, POOL_UNAVAILABLE_ERRORS):
            logger.warning(
                'task_curator: account pool unavailable for %r — deferring: %s',
                candidate.title, exc,
            )
            return CuratorDecision.deferred(
                pool_defer_reason(exc), pool_sizes=bundle.pool_sizes, latency_ms=_elapsed_ms(start),
            )

        if not isinstance(exc, CuratorFailureError):
            logger.warning(
                'task_curator: LLM call failed with unexpected %s', type(exc).__name__,
                exc_info=exc,
            )
        escalation = await self._escalate_failure(
            exc, bundle, project_id=project_id, project_root=project_root,
        )

        if not (isinstance(exc, CuratorFailureError) and exc.zero_output_timeout):
            logger.warning(
                'task_curator: LLM call failed, deferring curation of %r', candidate.title,
            )
            return CuratorDecision.deferred(
                DeferReason.LLM_FAILED, pool_sizes=bundle.pool_sizes, latency_ms=_elapsed_ms(start),
            )

        # A fresh timestamp, so the cooldown runs from when the hang was
        # observed rather than from before the (up to 180s) call started.
        if self._zero_output_breaker.record_timeout(time.monotonic()):
            logger.warning(
                'task_curator: zero-output-breaker OPEN after %d consecutive ZOTs; '
                'short-circuiting curator LLM calls for %.0fs',
                self._zero_output_breaker.consecutive_timeouts,
                self._zero_output_breaker.cooldown_seconds,
            )
        return await self._degraded_create(
            justification=escalation.zot_justification(exc),
            pool_sizes=bundle.pool_sizes,
            start=start,
            candidate=candidate,
            project_id=project_id,
            project_root=project_root,
            degraded_by_zot=True,
            zot_escalation_id=escalation.escalation_id,
        )

    async def _escalate_failure(
        self,
        exc: Exception,
        bundle: PreparedCandidate,
        *,
        project_id: str,
        project_root: str,
    ) -> _Escalation:
        """Report a failed LLM call to the escalator, and say what came of it.

        With no orchestrator running, the escalator raises
        :exc:`CuratorFailureError` instead of filing. That raise is logged and
        returned as the outcome's ``refusal``, so the failure is still decided
        exactly as on the orchestrator path.
        """
        if self._escalator is None:
            logger.warning('task_curator: LLM failure with no escalator wired: %s', exc)
            return _Escalation(escalated=False)
        try:
            escalation_id = await self._escalator.report_failure(
                project_root=project_root,
                project_id=project_id,
                candidate_title=bundle.candidate.title,
                pool_sizes=bundle.pool_sizes,
                **_failure_report(exc),
            )
        except CuratorFailureError as refusal:
            logger.error(
                'task_curator: no orchestrator to escalate the curator failure for %r '
                'to (%s); deciding it here',
                bundle.candidate.title, exception_summary(refusal),
            )
            return _Escalation(escalated=False, refusal=refusal)
        return _Escalation(escalated=True, escalation_id=escalation_id)


@dataclass(frozen=True)
class _Escalation:
    """What reporting one curator failure to the escalator came to."""

    escalated: bool
    escalation_id: str | None = None
    refusal: CuratorFailureError | None = None

    def zot_justification(self, exc: Exception) -> str:
        """The justification of the create a zero-output hang *exc* degrades to.

        Only a hang an escalator accepted is ``llm-error-escalated``. One it
        refused cites the refusal; with no escalator wired, the hang itself.
        """
        if self.escalated:
            return 'llm-error-escalated'
        return f'curator-failed: {exception_summary(self.refusal or exc)}'


class _FailureReport(TypedDict, total=False):
    """The ``report_failure`` keyword arguments that describe a failed call itself."""

    justification: Required[str]
    timed_out: bool | None
    duration_ms: int | None
    schema_tool_denied: bool
    zero_output_timeout: bool
    account_name: str | None
    proc_tree: str | None
    subtype: str | None
    cost_usd: float | None
    transcript_turns: int | None
    tools_used: tuple[str, ...] | None


# ----------------------------------------------------------------------
# Pure helpers (module-level — easier to unit-test)
# ----------------------------------------------------------------------


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _deferred_each(
    bundles: Sequence[PreparedCandidate], reason: DeferReason, start: float,
) -> list[CuratorDecision]:
    latency_ms = _elapsed_ms(start)
    return [
        CuratorDecision.deferred(reason, pool_sizes=bundle.pool_sizes, latency_ms=latency_ms)
        for bundle in bundles
    ]


def _failure_report(exc: Exception) -> _FailureReport:
    """The ``report_failure`` fields that describe *exc* itself."""
    if isinstance(exc, CuratorFailureError):
        return {
            'justification': str(exc),
            'timed_out': exc.timed_out,
            'duration_ms': exc.duration_ms,
            'schema_tool_denied': exc.schema_tool_denied,
            'zero_output_timeout': exc.zero_output_timeout,
            'account_name': exc.account_name or None,
            'proc_tree': exc.proc_tree or None,
            'subtype': exc.subtype,
            'cost_usd': exc.cost_usd,
            'transcript_turns': exc.transcript_turns,
            'tools_used': exc.tools_used,
        }
    name = type(exc).__name__
    return {
        'justification': f'unexpected-exception: {name}: {exc}',
        'subtype': f'unexpected-exception:{name}',
    }


def _to_input_space(decision: CuratorDecision, llm_bound: list[int]) -> CuratorDecision:
    """Rebase *decision*'s sibling reference from dispatch-local to input positions."""
    local = decision.batch_target_index
    if local is None:
        return decision
    if 0 <= local < len(llm_bound):
        return replace(decision, batch_target_index=llm_bound[local])
    # _parse_batch_decisions already degrades an out-of-range reference.
    logger.warning(
        'task_curator: LLM batch_target_index=%d out of dispatch range [0, %d); clearing it',
        local, len(llm_bound),
    )
    return replace(decision, batch_target_index=None)


_T = TypeVar('_T')


@dataclass
class _StreamReads:
    """The corpus streams one ``_build_corpus`` call attempted, and those whose read failed."""

    attempted: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)

    async def read(self, stream: str, read: Awaitable[_T], fallback: _T) -> _T:
        """Await *stream*'s *read*; a failure is logged, recorded, and answered with *fallback*."""
        self.attempted.append(stream)
        try:
            return await read
        except Exception as exc:
            logger.warning(
                'task_curator: corpus stream %s unavailable: %s', stream, exception_summary(exc),
            )
            self.unavailable.append(stream)
            return fallback

    def every_attempt_failed(self) -> bool:
        """Whether some stream was attempted and every attempted stream failed."""
        return bool(self.attempted) and len(self.unavailable) == len(self.attempted)


class _CorpusUnavailableError(RuntimeError):
    """Every corpus stream ``_build_corpus`` attempted failed, so no pool can be trusted."""

    def __init__(self, streams: tuple[str, ...]) -> None:
        super().__init__(f'every corpus stream failed: {", ".join(streams)}')
        self.streams = streams


def _with_stream_census(decision: CuratorDecision, bundle: PreparedCandidate) -> CuratorDecision:
    """*decision*, naming the corpus streams that failed for its *bundle*."""
    if bundle.withheld is None or not bundle.withheld.unavailable_streams:
        return decision
    return replace(decision, unavailable_streams=bundle.withheld.unavailable_streams)


def _count_absent(task_ids: Iterable[str], admitted: set[str]) -> int:
    """How many DISTINCT ids in *task_ids* the pool never admitted.

    The census counts what the prompt is MISSING, so an id a later stream
    picked up is not withheld however early a cap skipped it. Counting has to
    run against the finished admitted set for that reason; counting at the
    moment each cap fires reports entries that end up present.
    """
    return len({tid for tid in task_ids if tid and tid not in admitted})


def _task_files(task: dict) -> list[str]:
    """Extract files_to_modify from a raw taskmaster task dict.

    Priority: top-level ``files_to_modify`` (curator wire shape) →
    ``metadata.files`` (canonical orchestrator field) →
    ``metadata.files_to_modify`` (legacy curator-internal name).
    """
    files = task.get('files_to_modify')
    if files is None:
        meta = task.get('metadata') or {}
        if isinstance(meta, dict):
            files = meta.get('files') or meta.get('files_to_modify')
    if not files:
        return []
    if isinstance(files, str):
        return [files]
    return [str(f) for f in files if f]


# The module and dependency streams pool the open backlog only: a done task
# reaches the pool through the similarity-ordered embedding stream.
_OPEN_BACKLOG_STATUSES: tuple[str, ...] = tuple(sorted(status.value for status in ACTIVE))


def _dependents_of(anchor_id: str, edges: Mapping[int, Sequence[int]]) -> set[str]:
    """The ids of the tasks whose dependency list in *edges* names *anchor_id*."""
    if not anchor_id.isdecimal():
        return set()
    anchor = int(anchor_id)
    return {str(tid) for tid, deps in edges.items() if anchor in deps}


_SIBLING_STREAM_SPAWN_CONTEXT = 'agent_followup'
# One lookup page, far above the measured maximum of 11 siblings per parent;
# rows past it are neither ranked nor counted.
_SIBLING_LOOKUP_PAGE = 100


def _followup_parent_id(candidate: CandidateTask) -> str | None:
    """The parent whose follow-ups form *candidate*'s sibling stream, if it has one."""
    if candidate.spawn_context != _SIBLING_STREAM_SPAWN_CONTEXT:
        return None
    return candidate.spawned_from


def _sibling_rank(row: MetadataLookupRow) -> tuple[bool, int, int]:
    """Active before done, then by priority, then newest first."""
    return (
        row['status'] not in ACTIVE,
        _PRIORITY_RANK.get(row['priority'], 99),
        -row['task_id'],
    )


def _to_pool_entry(
    task: dict | None, *, source: str, lock_depth: int,
) -> PoolEntry | None:
    if not task or not isinstance(task, dict) or is_holder(task):
        return None
    tid = str(task.get('id', ''))
    if not tid:
        return None
    status = str(task.get('status', 'unknown'))
    files = _task_files(task)
    metadata = coerce_task_metadata(task.get('metadata')) or {}
    return PoolEntry(
        task_id=tid,
        title=str(task.get('title', '')),
        description=str(task.get('description', '') or ''),
        details=str(task.get('details', '') or ''),
        files_to_modify=files,
        module_keys=files_to_modules(files, depth=lock_depth),
        status=status,
        priority=str(task.get('priority', DEFAULT_PRIORITY)),
        source=source,
        combine_eligible=is_combine_eligible(task),
        item_count=item_count_of(metadata),
        declared_scope=declared_scope_of(metadata),
    )


def flatten_task_tree(tasks_result: Mapping[str, Any]) -> list[dict]:
    """Walk a get_tasks DTO and return a flat list of task dicts."""
    out: list[dict] = []

    def _walk(items: Any) -> None:
        if not isinstance(items, list):
            return
        for t in items:
            if isinstance(t, dict):
                out.append(t)
                _walk(t.get('subtasks') or [])

    _walk(tasks_result.get('tasks'))
    return out


def _module_sort_key(entry: PoolEntry) -> tuple[int, int, str]:
    status_rank = _STATUS_RANK.get(entry.status, 99)
    priority_rank = _PRIORITY_RANK.get(entry.priority, 99)
    return (status_rank, priority_rank, entry.task_id)


def _trim_pool(
    pool: list[PoolEntry], total_cap: int,
) -> tuple[list[PoolEntry], int]:
    """Trim a pool that exceeds the total cap, dropping the weakest first.

    Weak = the reverse of ``POOL_SOURCES``, preserving anchor always.

    Returns ``(kept, dropped_n)``. ``dropped_n`` is how many entries the cap
    removed, and it feeds the :class:`PoolWithheld` census so the prompt can
    say that the pool the LLM is reading is incomplete. Reporting the
    shortfall alongside the result, rather than logging it, is INV-11
    ``no-silent-fail-soft``: the consumer that makes the combine-vs-create
    call is the LLM, and the prompt is the only surface it reads.
    """
    if len(pool) <= total_cap:
        return pool, 0
    weakest_order = list(reversed(POOL_SOURCES))
    result = list(pool)
    for source in weakest_order:
        if len(result) <= total_cap:
            break
        # drop tail-most entries of this source class first
        indices = [i for i, e in enumerate(result) if e.source == source]
        # keep at least one anchor
        if source == 'anchor':
            indices = indices[1:]
        # drop from the end so earlier (higher-priority) keeps survive
        for i in reversed(indices):
            if len(result) <= total_cap:
                break
            result.pop(i)
    kept = result[:total_cap]
    return kept, len(pool) - len(kept)


def _parse_decision_dict(
    raw: dict,
    *,
    pool: list[PoolEntry],
    pool_sizes: dict[str, int],
    latency_ms: int,
    cost_usd: float,
) -> CuratorDecision:
    """Validate and parse a single raw decision dict into a ``CuratorDecision``.

    Called by both :func:`_parse_decision` (single-item) and
    :func:`_parse_batch_decisions` (per-item inside a batch).  The function
    **never raises**: every malformed-input path fails open to
    ``CuratorDecision(action='create', ...)`` instead of propagating an
    exception.  Per-item isolation in the batch path is provided by the
    ``try/except`` wrapper in :func:`_parse_batch_decisions`, not by
    ``ValueError`` from this function.
    """
    action = str(raw.get('action', '')).lower()
    if action not in ('drop', 'combine', 'create'):
        return CuratorDecision(
            action='create',
            justification=f'invalid-action: {action!r}',
            pool_sizes=pool_sizes,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            degraded=True,
        )

    justification = str(raw.get('justification', ''))
    target_id = raw.get('target_id')
    if target_id is not None:
        target_id = str(target_id)
    target_fingerprint = raw.get('target_fingerprint')
    if target_fingerprint is not None:
        target_fingerprint = str(target_fingerprint)

    # batch_target_index: valid only for 'drop'; carries the sibling index when
    # the candidate is a duplicate of another batch item (not yet materialised).
    # Wrap int() coercion: if the LLM emits a non-numeric string or other
    # schema-drifted value, treat it as absent (degrade gracefully like every
    # other invalid field in this parser) rather than raising ValueError out
    # of the single-item path where no try/except wraps us.
    batch_target_index_raw = raw.get('batch_target_index')
    batch_target_index: int | None
    if batch_target_index_raw is None:
        batch_target_index = None
    else:
        try:
            batch_target_index = int(batch_target_index_raw)
        except (TypeError, ValueError):
            batch_target_index = None

    # Safety: drop/combine must reference a pool task id that actually exists,
    # UNLESS this is a within-batch drop (batch_target_index set, target_id None).
    valid_ids = {e.task_id for e in pool}
    if action in ('drop', 'combine'):
        # Guard: LLM emitted BOTH a pool target_id and a within-batch
        # batch_target_index.  We cannot resolve the intent safely, so degrade.
        if action == 'drop' and batch_target_index is not None and target_id is not None:
            return CuratorDecision(
                action='create',
                justification=(
                    f'ambiguous-drop: both target_id={target_id!r} and '
                    f'batch_target_index={batch_target_index} set; '
                    'cannot determine intent'
                ),
                pool_sizes=pool_sizes,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                degraded=True,
            )
        is_within_batch_drop = (
            action == 'drop'
            and batch_target_index is not None
            and target_id is None
        )
        if not is_within_batch_drop:
            # Two causes, two branches: they fall on opposite sides of the
            # degraded line (see CuratorDecision.degraded).
            if not target_id:
                return CuratorDecision(
                    action='create',
                    justification=(
                        f'missing-target: action={action} has no target_id'
                    ),
                    pool_sizes=pool_sizes,
                    latency_ms=latency_ms,
                    cost_usd=cost_usd,
                    degraded=True,
                )
            if target_id not in valid_ids:
                return CuratorDecision(
                    action='create',
                    justification=(
                        f'invalid-target: action={action} target_id={target_id!r}; '
                        f'not in pool'
                    ),
                    pool_sizes=pool_sizes,
                    latency_ms=latency_ms,
                    cost_usd=cost_usd,
                    degraded=False,
                )
        # RC3 create-safe guard: never drop/combine against a pool entry whose
        # status is unconfirmable ('unknown' — e.g. a thin fallback entry built
        # after a TRANSIENT get_task failure, see _fetch_entry_for_neighbor).
        # Placed before the combine_eligible check below so it is authoritative
        # even if an unknown entry were ever wrongly marked combine_eligible=True.
        # Resolved once here (guaranteed present after the valid_ids check above)
        # and reused by the combine_eligible check that follows.
        target_entry = next((e for e in pool if e.task_id == target_id), None) if target_id else None
        if target_entry is not None and target_entry.status == 'unknown':
            return CuratorDecision(
                action='create',
                justification=(
                    f'unknown-status-target: {target_id} status=unknown, '
                    f'create-safe (RC3)'
                ),
                pool_sizes=pool_sizes,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                degraded=False,
            )
        # combine-only tasks must also be combine_eligible (pending status).
        if action == 'combine' and target_entry is not None and not target_entry.combine_eligible:
            return CuratorDecision(
                action='create',
                justification=(
                    f'invalid-combine-target: {target_id} has status '
                    f'{target_entry.status}, not pending'
                ),
                pool_sizes=pool_sizes,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                degraded=False,
            )

    rewritten: RewrittenTask | None = None
    if action == 'combine':
        rt = raw.get('rewritten_task')
        if not isinstance(rt, dict):
            return CuratorDecision(
                action='create',
                justification='combine-missing-rewrite',
                pool_sizes=pool_sizes,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                degraded=True,
            )
        try:
            rewritten = RewrittenTask(
                title=str(rt.get('title', '')),
                description=str(rt.get('description', '')),
                details=str(rt.get('details', '')),
                files_to_modify=[str(f) for f in (rt.get('files_to_modify') or [])],
                priority=str(rt.get('priority', DEFAULT_PRIORITY)),
            )
        except Exception as exc:
            return CuratorDecision(
                action='create',
                justification=f'rewrite-parse-failed: {exc}',
                pool_sizes=pool_sizes,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                degraded=True,
            )
        if not rewritten.title or not rewritten.details:
            return CuratorDecision(
                action='create',
                justification='rewrite-empty-title-or-details',
                pool_sizes=pool_sizes,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                degraded=True,
            )

    return CuratorDecision(
        action=action,  # type: ignore[arg-type]
        target_id=target_id,
        target_fingerprint=target_fingerprint,
        rewritten_task=rewritten,
        justification=justification,
        pool_sizes=pool_sizes,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        batch_target_index=batch_target_index,
    )


def _parse_decision(
    agent_result: AgentResult,
    *,
    pool_sizes: dict[str, int],
    latency_ms: int,
    pool: list[PoolEntry],
) -> CuratorDecision:
    """Parse a single-item :class:`AgentResult` into a :class:`CuratorDecision`.

    Extracts the raw dict from ``structured_output`` (or falls back to
    ``json.loads(output)``) then delegates to :func:`_parse_decision_dict`.
    Any parse failure returns ``action='create'`` — never raises.
    """
    raw = agent_result.structured_output
    if raw is None:
        # Try to parse from raw output
        try:
            raw = json.loads(agent_result.output)
        except Exception as exc:
            return CuratorDecision(
                action='create',
                justification=f'parse-failed: {exc}; output={agent_result.output[:200]}',
                pool_sizes=pool_sizes,
                latency_ms=latency_ms,
                cost_usd=agent_result.cost_usd,
                degraded=True,
            )

    if not isinstance(raw, dict):
        return CuratorDecision(
            action='create',
            justification=f'parse-failed: not-a-dict ({type(raw).__name__})',
            pool_sizes=pool_sizes,
            latency_ms=latency_ms,
            cost_usd=agent_result.cost_usd,
            degraded=True,
        )

    return _parse_decision_dict(
        raw,
        pool=pool,
        pool_sizes=pool_sizes,
        latency_ms=latency_ms,
        cost_usd=agent_result.cost_usd,
    )


def _shift_batch_target_index(
    decision: CuratorDecision, offset: int,
) -> CuratorDecision:
    """Add *offset* to ``decision.batch_target_index`` (no-op if None).

    Used by :meth:`TaskCurator._bisect` to rebase the
    half-local indices returned by a bisected sub-call into the input-local
    space of the helper invocation that issued the bisect.
    """
    bti = decision.batch_target_index
    if bti is None:
        return decision
    return replace(decision, batch_target_index=bti + offset)


def _parse_batch_decisions(
    agent_result: AgentResult,
    *,
    pools: list[list[PoolEntry]],
    pool_sizes_list: list[dict[str, int]],
    latency_ms: int,
) -> list[CuratorDecision]:
    """Parse a batched LLM response into one :class:`CuratorDecision` per candidate.

    The model may return decisions out of order; we re-index by ``candidate_index``.
    Per-item parse failures degrade only that item to ``action='create'``; other
    items are unaffected.  Whole-batch structural failures degrade ALL items.
    """
    n = len(pools)
    cost_per_item = (agent_result.cost_usd or 0.0) / max(n, 1)

    # Try to extract the decisions list from structured_output or raw JSON.
    raw_decisions: list[Any] | None = None
    try:
        so = agent_result.structured_output
        if isinstance(so, dict) and isinstance(so.get('decisions'), list):
            raw_decisions = so['decisions']
        else:
            parsed = json.loads(agent_result.output or '{}')
            if isinstance(parsed, dict) and isinstance(parsed.get('decisions'), list):
                raw_decisions = parsed['decisions']
    except Exception:
        logger.warning(
            'failed to extract decisions list from agent_result; '
            'degrading entire batch to action=create',
            exc_info=True,
        )

    # Build a lookup from candidate_index → raw dict.
    index_to_raw: dict[int, dict] = {}
    if raw_decisions is not None:
        for item in raw_decisions:
            if isinstance(item, dict):
                idx = item.get('candidate_index')
                if isinstance(idx, int) and 0 <= idx < n:
                    index_to_raw[idx] = item

    results: list[CuratorDecision] = []
    for i in range(n):
        if i not in index_to_raw:
            results.append(CuratorDecision(
                action='create',
                justification='batch-item-missing',
                pool_sizes=pool_sizes_list[i] if i < len(pool_sizes_list) else {},
                latency_ms=latency_ms,
                cost_usd=cost_per_item,
                degraded=True,
            ))
            continue
        try:
            decision = _parse_decision_dict(
                index_to_raw[i],
                pool=pools[i],
                pool_sizes=pool_sizes_list[i] if i < len(pool_sizes_list) else {},
                latency_ms=latency_ms,
                cost_usd=cost_per_item,
            )
        except Exception as exc:  # noqa: BLE001
            decision = CuratorDecision(
                action='create',
                justification=f'batch-item-parse-failed: {exc}',
                pool_sizes=pool_sizes_list[i] if i < len(pool_sizes_list) else {},
                latency_ms=latency_ms,
                cost_usd=cost_per_item,
                degraded=True,
            )
        results.append(decision)

    # --- Post-parse: validate and sanitize batch_target_index references ---
    # 1. Out-of-range or self-referential batch_target_index → degrade to create.
    # 2. Cycle detection: walk the graph of i → batch_target_index[i]; any node
    #    that is part of a cycle is degraded to create with 'batch-cycle'.
    for i, d in enumerate(results):
        if d.action == 'drop' and d.batch_target_index is not None:
            bti = d.batch_target_index
            if bti == i or not (0 <= bti < n):
                results[i] = CuratorDecision(
                    action='create',
                    justification=(
                        f'batch-target-out-of-range: batch_target_index={bti!r} '
                        f'for item {i} (n={n})'
                    ),
                    pool_sizes=d.pool_sizes,
                    latency_ms=d.latency_ms,
                    cost_usd=d.cost_usd,
                    degraded=True,
                )

    # Build the directed graph for cycle detection (only batch-drop edges).
    def _find_cycle_members() -> set[int]:
        """Return the set of indices that participate in any cycle.

        Uses recursive DFS.  Recursion depth is bounded by the batch size
        (``n``), which is capped at ``CuratorConfig.batch_max`` (default 5).
        If ``batch_max`` is ever raised to hundreds, convert to an iterative
        DFS with an explicit stack to avoid Python's 1000-frame limit.
        """
        # Simple DFS with colour marking: 0=unvisited, 1=in-stack, 2=done.
        colour = [0] * n
        cycle_nodes: set[int] = set()

        def dfs(node: int, path: list[int]) -> None:
            if colour[node] == 2:
                return
            if colour[node] == 1:
                # Found cycle — mark all nodes currently in the path back to
                # where the cycle starts.
                cycle_start = path.index(node)
                cycle_nodes.update(path[cycle_start:])
                return
            colour[node] = 1
            path.append(node)
            d = results[node]
            if d.action == 'drop' and d.batch_target_index is not None:
                target = d.batch_target_index
                if 0 <= target < n and target != node:
                    dfs(target, path)
            path.pop()
            colour[node] = 2

        for start in range(n):
            if colour[start] == 0:
                dfs(start, [])
        return cycle_nodes

    cycle_members = _find_cycle_members()
    for i in cycle_members:
        d = results[i]
        results[i] = CuratorDecision(
            action='create',
            justification=f'batch-cycle: item {i} is part of a batch_target_index cycle',
            pool_sizes=d.pool_sizes,
            latency_ms=d.latency_ms,
            cost_usd=d.cost_usd,
            degraded=True,
        )

    return results


def _parse_single_call(
    agent_result: AgentResult, bundles: Sequence[PreparedCandidate], latency_ms: int,
) -> list[CuratorDecision]:
    """:attr:`_CuratorCall.parse` for the single spec: the one bundle's decision."""
    [only] = bundles
    return [_parse_decision(
        agent_result, pool_sizes=only.pool_sizes, latency_ms=latency_ms, pool=only.pool,
    )]


def _parse_batch_call(
    agent_result: AgentResult, bundles: Sequence[PreparedCandidate], latency_ms: int,
) -> list[CuratorDecision]:
    """:attr:`_CuratorCall.parse` for the batch spec: one decision per bundle."""
    return _parse_batch_decisions(
        agent_result,
        pools=[bundle.pool for bundle in bundles],
        pool_sizes_list=[bundle.pool_sizes for bundle in bundles],
        latency_ms=latency_ms,
    )
