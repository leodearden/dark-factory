"""The curator's data vocabulary and the pure helpers over it, doing no I/O.

Contract: plans/review-all-s11-curator-pipeline-prd.md §Contract.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from shared.task_statuses import TaskStatus

from fused_memory.middleware.curation_deferral import DeferReason, deferral_justification
from fused_memory.utils.clip_disclosure import elide_text


class CuratorFailureError(RuntimeError):
    """Raised when the curator's LLM call fails and cannot be salvaged.

    The curator no longer degrades silently to ``action='create'`` on LLM
    errors; instead the interceptor translates this into an L1 escalation or
    a hard failure at the MCP boundary so operators notice breakage.

    Attaches ``timed_out``, ``duration_ms``, ``subtype``, ``schema_tool_denied``,
    and ``cost_usd`` from the underlying ``AgentResult`` so
    :class:`CuratorEscalator` can surface them in the L1 escalation detail and
    so the bisecting fallback in
    :meth:`TaskCurator._call_llm_batch_with_fallback` can branch on the
    failure subtype without leaking the underlying ``AgentResult`` outside
    :meth:`TaskCurator._invoke_curator`, whose failures
    ``task_curator.py::_failure_error`` maps to this one error.
    Defaults make the attributes safe to read even when the failure
    originates outside that call.

    ``schema_tool_denied`` is the CLI-2.1.168 regression signal: the synthetic
    ``StructuredOutput`` schema tool was permission-denied (a systemic deny-list
    break, not a flaky candidate).  When set, :class:`CuratorEscalator` raises a
    distinct, un-suppressed escalation so the deny-list gets fixed promptly.

    On a killed run ``turns`` and ``cost_usd`` are empty-stdout defaults, so
    ``transcript_turns`` is the observation, and ``tools_used`` lists the
    non-schema tools the call's transcript recorded (None when it was not read).
    """

    def __init__(
        self,
        message: str,
        *,
        timed_out: bool | None = None,
        duration_ms: int | None = None,
        subtype: str | None = None,
        schema_tool_denied: bool = False,
        zero_output_timeout: bool = False,
        proc_tree: str = '',
        account_name: str = '',
        cost_usd: float | None = None,
        transcript_turns: int | None = None,
        tools_used: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(message)
        self.timed_out = timed_out
        self.duration_ms = duration_ms
        self.subtype = subtype
        self.schema_tool_denied = schema_tool_denied
        self.zero_output_timeout = zero_output_timeout
        self.proc_tree = proc_tree
        self.account_name = account_name
        self.cost_usd = cost_usd
        self.transcript_turns = transcript_turns
        self.tools_used = tools_used


_EXCEPTION_SUMMARY_MAX_CHARS = 120


def exception_summary(exc: BaseException) -> str:
    """How an exception is written into a decision justification or a ticket reason.

    ``<ExcType>: <first line of the message>``, the line cut to
    ``_EXCEPTION_SUMMARY_MAX_CHARS``: diagnostic enough to name a missing
    binary, yet bounded, so identical failures group under one reason.
    """
    lines = str(exc).strip().splitlines()
    first_line = lines[0][:_EXCEPTION_SUMMARY_MAX_CHARS] if lines else ''
    name = type(exc).__name__
    return f'{name}: {first_line}' if first_line else name


# 'drop', 'combine' and 'create' are the only actions the LLM may request — see
# task_curator.py::CURATOR_OUTPUT_SCHEMA and ::CURATOR_BATCH_OUTPUT_SCHEMA, whose
# enums stay exactly those three. 'route_deterministic' and 'refuse' are DETERMINISTIC-ONLY
# verdicts, reachable solely from a `_maybe_*` guard backed by an
# operator-curated YAML registry; the model must never be able to emit either.
# 'refuse' in particular means "this candidate's premise is dead — create
# NOTHING", which is deliberately distinct from 'drop' ("fold into existing
# target task N", which requires a target). 'defer' — "the curator could not
# judge this, judge it again later" — is built only by CuratorDecision.deferred.
Action = Literal['drop', 'combine', 'create', 'route_deterministic', 'refuse', 'defer']


DEFAULT_PRIORITY = 'medium'  # canonical fallback used by both record_task and backfill_corpus


@dataclass
class CandidateTask:
    """A task about to be created, fed to the curator for a drop/combine/create decision."""

    title: str
    description: str = ''
    details: str = ''
    files_to_modify: list[str] = field(default_factory=list)
    priority: str = DEFAULT_PRIORITY
    spawned_from: str | None = None  # task id of the review-chain anchor
    spawn_context: str = 'manual'  # review | steward-triage | planning | manual
    # metadata.execution_class ('code_tdd' | 'operational' | 'decision' | None) —
    # consulted by operational_ask_registry.match_candidate. Its routing axis
    # (task 2687) is DEMOTED (task δ): a tagged operational/decision candidate is
    # now SKIPPED (match_candidate returns None) because the submit boundary
    # inject_operational_routing (task β) already owns tagged-ask routing;
    # untagged asks still use the substring fallback. Also part of payload_hash
    # (see below).
    execution_class: str | None = None

    def payload_hash(self) -> str:
        """Stable hash over fields the curator actually reads.

        Two candidates with the same hash get the same cached decision — this
        prevents combine-on-combine drift when a creation is retried.
        """
        h = hashlib.sha256()
        h.update(self.title.encode())
        h.update(b'\x00')
        h.update(self.description.encode())
        h.update(b'\x00')
        h.update(self.details.encode())
        h.update(b'\x00')
        h.update('\n'.join(sorted(self.files_to_modify)).encode())
        h.update(b'\x00')
        h.update((self.spawned_from or '').encode())
        h.update(b'\x00')
        # execution_class is hashed too (task 2687; rationale corrected task δ).
        # The match_candidate routing axis is now DEMOTED — a tagged
        # operational/decision candidate returns None from the route guard
        # rather than routing (task δ) — but two candidates differing ONLY in
        # execution_class can STILL resolve to different curator decisions: a
        # tagged operational candidate returns None from _maybe_route_deterministic
        # and proceeds to the LLM create/drop/combine path (which IS cached),
        # while an otherwise-identical untagged candidate may substring-route or
        # take a different LLM decision. Dropping execution_class from the hash
        # would let one's cached decision be wrongly returned for the other, so
        # it must remain part of the key.
        h.update((self.execution_class or '').encode())
        # 16-char sha256-hex shape — canonical owner: orchestrator.agents.triage.sha256_16.
        # Any change to length or algorithm must be mirrored there and at the
        # other mirror site, candidate_key.py::compute_candidate_key.
        return h.hexdigest()[:16]

    @classmethod
    def from_rewritten_task(cls, rt: RewrittenTask) -> CandidateTask:
        """Construct a CandidateTask from a RewrittenTask produced by the combine action.

        Centralises the field mapping so callers don't hand-enumerate fields
        and silently drop any that ``RewrittenTask`` gains in the future.
        ``spawned_from`` and ``spawn_context`` are left at their defaults
        because a rewritten task has no direct spawning relationship.
        """
        return cls(
            title=rt.title,
            description=rt.description,
            details=rt.details,
            files_to_modify=rt.files_to_modify,
            priority=rt.priority,
        )


@dataclass
class RewrittenTask:
    """A task description produced by the combine action."""

    title: str
    description: str
    details: str
    files_to_modify: list[str]
    priority: str


@dataclass
class CuratorDecision:
    """Result of a curator call. Always returned — never raised."""

    action: Action
    target_id: str | None = None
    target_fingerprint: str | None = None
    rewritten_task: RewrittenTask | None = None
    justification: str = ''
    pool_sizes: dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0
    cost_usd: float = 0.0
    # Set only by curate_batch when a candidate is a duplicate of another
    # candidate in the same batch (neither yet materialised as a task).
    # The worker substitutes the sibling's resulting task_id at dispatch time.
    batch_target_index: int | None = None
    # True for a fail-open create that no dedupe judgement stands behind: the
    # response itself was unusable, or the curator never got one. False for a
    # usable decision that POOL STATE vetoed. Every fail-open create states it
    # explicitly, as enforced (with the classification rule spelled out) by
    # fused-memory/tests/test_curator_arm_instrumentation_guard.py.
    degraded: bool = False
    # Structured signal that dedupe was silently skipped for this create by a
    # zero-output hang; never derived from ``justification``, which is shared.
    degraded_by_zot: bool = False
    zot_escalation_id: str | None = None
    defer_reason: DeferReason | None = None

    def __post_init__(self) -> None:
        if (self.action == 'defer') != (self.defer_reason is not None):
            raise ValueError(
                "CuratorDecision.defer_reason must be set iff action == 'defer', got "
                f'action={self.action!r} defer_reason={self.defer_reason!r}'
            )

    @classmethod
    def deferred(
        cls,
        reason: DeferReason,
        *,
        pool_sizes: dict[str, int] | None = None,
        latency_ms: int = 0,
    ) -> CuratorDecision:
        return cls(
            action='defer',
            defer_reason=reason,
            justification=deferral_justification(reason),
            pool_sizes=dict(pool_sizes or {}),
            latency_ms=latency_ms,
        )

    def to_log_fields(self) -> dict[str, Any]:
        return {
            'action': self.action,
            'target_id': self.target_id,
            'target_fingerprint': self.target_fingerprint,
            'justification': self.justification[:200],
            'pool_sizes': self.pool_sizes,
            'latency_ms': self.latency_ms,
            'cost_usd': self.cost_usd,
            'defer_reason': self.defer_reason,
        }


@dataclass
class PreparedCandidate:
    """Per-candidate batch input prepared by :meth:`TaskCurator.prepare_candidate`.

    Carries the candidate, its assembled corpus pool, pool sizes, the
    :class:`PoolWithheld` census of what the caps kept out of that pool, and
    the estimated user-prompt token count for this candidate's section of the
    batched prompt.  The token estimate is what the worker uses to decide
    whether adding this candidate to the in-flight batch would exceed the
    soft ``batch_token_threshold``.

    ``corpus_error`` is set when assembly FAILED and the pool above is the
    empty degradation rather than a real (possibly empty) corpus.  The two
    are indistinguishable from the fields alone, and they call for opposite
    handling: a real corpus is curated over, a failed one defers the
    candidate as ``corpus-failed``.
    """

    candidate: CandidateTask
    pool: list[PoolEntry]
    pool_sizes: dict[str, int]
    prompt_tokens: int
    withheld: PoolWithheld | None = None
    corpus_error: str | None = None


@dataclass(frozen=True)
class PoolWithheld:
    """Per-cap census of the pool entries a cap kept out of the prompt.

    The consumer is the CURATOR LLM, and the prompt is the only surface it
    reads — so the pool's incompleteness travels in :meth:`render`, not in a
    log line nobody at the decision point sees (INV-11
    ``no-silent-fail-soft``). Without it the LLM cannot distinguish "no
    duplicate exists" from "the duplicate was cut at a cap", and answers
    ``create`` or ``combine`` as if the pool were complete.

    ``by_source`` counts the eligible entries each cap excluded, keyed by the
    stream that lost them (``module`` / ``embedding`` / ``dependency``) plus
    ``total_cap`` for the final ``task_curator.py::_trim_pool`` pass. ``caps``
    carries the configured limits that produced those counts, so a reader can
    tell a pool one entry over its cap from one that lost forty.

    The ``module`` and ``dependency`` counts are corpus-exhaustive — both
    streams scan every task. The ``embedding`` count is NOT: it is taken over
    a fixed qdrant retrieval window (``pool_embedding_cap + 20`` points), so
    at most 20 neighbours can ever be reported however many near-duplicates
    the corpus holds. A stream whose count is ceilinged that way is named in
    ``lower_bounds`` and :meth:`render` says so, because the alternative is
    the LLM reading a ceiling as an exact count on the one stream ordered by
    actual similarity — the stream whose withheld entries are the likeliest
    duplicates of all.

    Every count is deliberately per-STREAM and not only at ``total_cap``:
    under stock config the stream caps are the binding constraints and
    ``total_cap`` never fires at all (see the pool-cap block in
    ``config/schema.py::CuratorConfig``), so a census keyed on the final trim
    alone would be permanently silent.

    An empty census renders nothing: a pool that fit says nothing about
    fitting, exactly as a complete result surface carries no delta.
    """

    by_source: Mapping[str, int] = field(default_factory=dict)
    caps: Mapping[str, int] = field(default_factory=dict)
    lower_bounds: tuple[str, ...] = ()

    @property
    def total(self) -> int:
        """How many eligible entries the caps kept out of the pool, in total."""
        return sum(self.by_source.values())

    def render(self) -> str | None:
        """Render the prompt block, or ``None`` when nothing was withheld."""
        if self.total == 0:
            return None
        payload: dict[str, object] = {
            'withheld': dict(self.by_source), 'caps': dict(self.caps),
        }
        if self.lower_bounds:
            payload['lower_bounds'] = list(self.lower_bounds)
        block = (
            f'pool_truncated: {json.dumps(payload, sort_keys=True)}\n'
            '  The pool above is INCOMPLETE — the counts above are tasks that '
            'matched this candidate but did not fit. Absence of a duplicate in '
            'a truncated pool is not proof that no duplicate exists, so prefer '
            '"create" over a speculative "combine" or "drop" when the pool '
            'offers no clearly-overlapping task.'
        )
        if self.lower_bounds:
            block += (
                '\n  The streams named in "lower_bounds" withheld AT LEAST '
                'that many: the retrieval window they were counted over was '
                'itself full, so the true number is that count or higher.'
            )
        return block


POOL_SOURCES: Final = ('anchor', 'module', 'embedding', 'dependency')


def empty_pool_sizes() -> dict[str, int]:
    """Per-source pool sizes of a decision reached without a corpus."""
    return dict.fromkeys(POOL_SOURCES, 0)


@dataclass
class PoolEntry:
    """One task as seen by the curator — extracted from raw taskmaster data."""

    task_id: str
    title: str
    description: str
    details: str
    files_to_modify: list[str]
    module_keys: list[str]
    status: str
    priority: str
    source: str  # one of POOL_SOURCES
    combine_eligible: bool

    def render(self, desc_cap: int, details_cap: int) -> str:
        """Render this entry as a human-readable block for the LLM prompt.

        Truncation goes through :func:`clip_for_prompt`, the same helper the
        CANDIDATE block uses, so the two sides of the prompt cannot drift.
        """
        desc = clip_for_prompt(self.description, desc_cap)
        det = clip_for_prompt(self.details, details_cap)
        files = ', '.join(self.files_to_modify) if self.files_to_modify else '(none)'
        return (
            f'[Task {self.task_id}] status={self.status} priority={self.priority} '
            f'source={self.source} combine_eligible={self.combine_eligible}\n'
            f'  title: {self.title}\n'
            f'  description: {desc}\n'
            f'  details: {det}\n'
            f'  files_to_modify: {files}'
        )


def embedding_text(title: str, description: str, files_to_modify: list[str]) -> str:
    """Compose the text every curator corpus embedding is computed over.

    Single owner: a caller comparing cosine scores against stored tasks must
    compose its query with this, or the scores are not comparable.
    """
    parts = [title]
    if description:
        parts.append(description)
    if files_to_modify:
        parts.append('\n'.join(files_to_modify))
    return '\n\n'.join(parts)


def clip_for_prompt(text: str, cap: int) -> str:
    """Clip *text* to *cap* characters for the curator prompt, marking how much was elided.

    The counted marker is owned by ``utils/clip_disclosure.py::elide_text``;
    this is the curator prompt's single route to it. Both sides of that prompt
    route through here — the POOL side (:meth:`PoolEntry.render`) and the
    CANDIDATE side (``task_curator.py::_render_candidate_section``, which both
    the single and batch prompts use) — so the two cannot drift apart (INV-5
    ``no-lockstep-duplication``). They previously did: the pool side appended a
    bare ``…`` while the candidate side clipped silently, so the reading LLM
    could not tell a short description from a truncated one. A clip at a
    non-positive *cap* raises ``ValueError`` rather than rendering a marker.

    The elided COUNT is part of the marker because the size of the loss is
    what decides whether it mattered: "2 characters trimmed" and "1827
    characters of concrete file references gone" are not the same event, and
    1827 was the measured mean elision at the pre-2026-09-18 cap.
    """
    return elide_text(text, cap, field='curator_prompt').text


def is_combine_eligible_status(status: str) -> bool:
    """Single source of truth for combine STATUS eligibility (task 4035).

    Called from BOTH the curator's selection snapshot
    (``_to_pool_entry.combine_eligible``) and the interceptor's
    execution-time combine guard (``_execute_combine``). Keeping one
    definition is the point: the two sites previously hand-copied
    ``status == 'pending'`` and silently diverged, letting 20.2% of combines
    land on non-pending targets mid-planning.

    Status ONLY. Liveness (``claimant_run_id``) is execution-side and
    deliberately NOT part of this predicate — the selection snapshot cannot
    observe it (D11).

    Fails closed: anything outside the canonical vocabulary (an unknown
    status, a blank, a case variant) is not eligible.
    """
    return status == TaskStatus.PENDING.value
