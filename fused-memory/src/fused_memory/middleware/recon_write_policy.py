"""Server-side gate rejecting recon-stage writes to tasks they must not touch.

W5-ζ (task 2224). Consulted from
:class:`fused_memory.middleware.task_interceptor.TaskInterceptor` ONLY when
the caller's ``agent_id`` starts with ``'recon-stage-'`` — enforcing this
gate server-side means the post-hoc reconciliation guards
(``reconciliation/stages/task_knowledge_sync.py``'s
``_check_stall_guard_freshness`` and related) become redundant; deleting
them is downstream consumer W5-μ's job, NOT this module's.

Three independent early-return gates in :func:`check`:

1. ``op == 'update_task'`` AND ``live_status`` is terminal (done/cancelled)
   AND NOT ``is_annotation_clear`` -> ``ReconTerminalWriteRejected``. The
   ``is_annotation_clear`` bypass exempts a non-load-bearing, metadata-only,
   merge-mode write touching only :data:`CLEARABLE_ANNOTATION_KEYS` (e.g.
   ``possible_scope_mismatch`` — task 2684; see
   :func:`is_terminal_annotation_clear`) OR touching only
   :data:`X_ANNOTATION_PREFIX`-prefixed forward-compat annotation keys
   (task 2695; see :func:`is_terminal_annotation_add`). A mandatory
   :data:`_CAUSATION_TRACING_KEYS` co-key (currently ``_causation_id``,
   which every recon-stage write embeds per the run's Reconciliation
   Context block) riding alongside either kind of content is transparent
   to both checks (task 2697; see :func:`_annotation_content_keys`) — but a
   payload of tracing co-keys alone is still not exempt. Callers should pass
   :func:`is_terminal_annotation_exempt` — the single-parse combined form of
   the two predicates. It never loosens Gates 2/3.
2. ``op == 'set_task_status'`` AND a live workflow is detected for the task
   -> ``ReconLiveWorkflowWriteRejected``. The caller's ``live_status`` and
   (optionally) the task's ``task_metadata`` are forwarded to the detector
   as ``status``/``task_kind``/``pure_gate``, so the project-wide
   orchestrator-lock signal is dropped for tasks it is not evidence for
   (task 3751 — see :func:`check`'s Gate 2 paragraph).
3. ``snapshot_token is not None`` AND it disagrees with ``live_status``
   (op-agnostic) -> ``ReconStaleSnapshotRejected``.

Design note: this module is a SEPARATE gate from
``shared.task_transitions.is_legal_transition`` (the (from, to, actor)
transition-legality table) — never a third transition table. That table
answers "is old->new legal for this actor class"; this module answers "may
this recon-stage caller touch this task at all". The two compose
independently; this module never imports/extends ``shared.task_transitions``.

The recon-stage scoping (``agent_id.startswith('recon-stage-')``) lives at
the call site in ``task_interceptor.py``, not here — :func:`check` is a pure
function that always runs its gates when called.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from shared.task_statuses import TERMINAL as TERMINAL_STATUSES

from fused_memory.services.live_workflow_detector import (
    is_pure_gate_metadata,
    is_workflow_live_for_task,
)

# Metadata keys carrying the snapshot status a recon-stage caller observed
# before writing. Promoted (Open Q3) from
# reconciliation/stages/task_knowledge_sync.py's _STAGE2_STALL_SNAPSHOT_KEYS
# so the server enforces the same keys the post-hoc guard checked.
SNAPSHOT_TOKEN_KEYS: tuple[str, ...] = ('snapshot_status', 'observed_status')

# Stable, machine-readable Verdict.corrective_path token for the Gate 1
# (terminal-write) rejection. Names the sanctioned same-status
# done_provenance repair seam
# (``set_task_status(task_id, 'done', done_provenance={...})``, which routes
# an already-done task to
# ``task_interceptor._repair_done_provenance_same_status`` — task 2401)
# rather than the destructive reopen path. Never used by the Gate 2
# (live-workflow) or Gate 3 (stale-snapshot) rejections — see :class:`Verdict`.
TERMINAL_CORRECTIVE_PATH = 'set_task_status_done_provenance_repair'

# Frozen-by-default allowlist (task 2684) of metadata keys a terminal-task
# ``update_task`` write may clear via :func:`is_terminal_annotation_clear`.
# Mirrors the ``_BLESSED_METADATA_KEYS`` convention in
# ``shared.task_metadata``: a newly introduced metadata key stays BLOCKED on
# terminal tasks until deliberately added here — the guard can never be
# silently loosened by an unrelated new key. Seeded with the one known
# non-load-bearing annotation marker
# (``TaskInterceptor._attach_possible_scope_mismatch`` — an advisory
# ``{matched_paths, suggested_project, source}`` blob with no bearing on
# lifecycle/scheduling/locks/provenance). ``status`` and ``done_provenance``
# are intentionally never eligible here — the ``update_task`` write-authority
# floor already unconditionally rejects both for every caller, so they can
# never reach this predicate.
CLEARABLE_ANNOTATION_KEYS: frozenset[str] = frozenset({'possible_scope_mismatch'})

# The sanctioned forward-compat annotation-key namespace (task 2695), used
# by :func:`is_terminal_annotation_add`. Mirrors the bare ``'x_'`` literal in
# ``shared/src/shared/task_metadata.py:863`` (``parse_metadata`` admits any
# key where ``key.startswith('x_')`` silently — no schema warning, no
# allowlist entry required). Defined as a local constant rather than an
# import because ``task_metadata.py`` exposes no shared constant of its own
# to import (it uses the bare literal directly) — this is an intentional
# literal-mirror; keep the two in sync by hand if either changes.
X_ANNOTATION_PREFIX = 'x_'

# Allowlist (robustness amendment, task 2684) of ``update_task`` kwargs that
# NEVER disqualify the terminal-annotation-clear exemption. Deliberately an
# allowlist of permitted kwargs, NOT a denylist of known content fields: a
# denylist (title/description/details/prompt/priority/dependencies) has the
# OPPOSITE default-safety posture from :data:`CLEARABLE_ANNOTATION_KEYS`'s
# frozen-by-default allowlist — a *new* content-mutating ``update_task``
# parameter added later (e.g. a hypothetical ``owner`` or ``subtasks`` field)
# would silently pass through a denylist until someone remembered to add it
# there, permitting a content mutation on a terminal task alongside a
# sanctioned annotation clear. An allowlist fails CLOSED instead: an
# unrecognized kwarg disqualifies the exemption until deliberately added
# here, matching :data:`CLEARABLE_ANNOTATION_KEYS`'s posture so both checks
# in :func:`is_terminal_annotation_clear` agree.
#
# This is a *kwarg-name* allowlist (is this top-level ``update_task``
# parameter one the exemption may ignore?) — a different axis from
# :data:`CLEARABLE_ANNOTATION_KEYS`, which is a *metadata-key* allowlist
# (is this key inside the ``metadata`` dict one the exemption may clear?).
# The two compose: both must hold for the exemption to apply.
#
# ``metadata``/``metadata_mode``/``append`` are specially inspected below
# (not merely ignored). ``tag`` selects the tag-scoped row
# (``WHERE tag = ? AND id = ?`` in ``sqlite_task_backend.update_task``) and
# is never itself written — addressing, not content. ``status`` is included
# for the same reason :data:`CLEARABLE_ANNOTATION_KEYS` never eligibility-
# checks it: the write-authority floor unconditionally rejects a non-None
# ``status`` for every caller, before any DB write is attempted, regardless
# of this predicate's verdict — carrying forward the original design
# decision rather than silently reversing it. ``agent_id`` never actually
# reaches ``update_kwargs`` in practice (the ``task_interceptor.update_task``
# call site captures it as a separate keyword-only parameter, out of
# ``**kwargs``) — listed here only so a direct caller of this predicate is
# not surprised.
_ANNOTATION_CLEAR_ALLOWED_KWARGS: frozenset[str] = frozenset({
    'metadata', 'metadata_mode', 'append', 'tag', 'status', 'agent_id',
})

# Frozen allowlist (task 2697) of non-load-bearing recon-stage
# causation-tracing co-keys that are transparent to (excluded from) the
# per-key annotation-eligibility checks in :func:`is_terminal_annotation_clear`,
# :func:`is_terminal_annotation_add`, and :func:`is_terminal_annotation_exempt`.
# The run's Reconciliation Context block
# (``reconciliation/stages/base.py``) mandates ``_causation_id`` on EVERY
# recon-stage write, so a compliant recon-stage caller's metadata payload
# always carries it ALONGSIDE any real annotation content (a clear or an
# x_ add) — without this allowlist, that mandatory co-key made both
# ALL-clearable and ALL-x_ checks fail unconditionally, leaving the task
# 2684/2695 exemptions practically unreachable. Mirrors ``_causation_id``'s
# Tier-A blessed-key status in ``shared.task_metadata._BLESSED_METADATA_KEYS``
# (see :class:`TestCausationTracingKeysConsistencyWithBlessedMetadata`).
# Fail-CLOSED like its sibling allowlists: only these EXACT keys are
# transparent to the eligibility check — every other key, including any
# other blessed-but-load-bearing key (``branch_base_sha``, ``escalation_id``,
# ``before_done_*`` stamps, …), still disqualifies. See
# :func:`_annotation_content_keys`.
_CAUSATION_TRACING_KEYS: frozenset[str] = frozenset({'_causation_id'})

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """Outcome of a :func:`check` call.

    Mirrors :class:`fused_memory.middleware.path_scope_guard.PathGuardVerdict`'s
    shape (``is_rejection`` property; ``to_error_dict()`` -> ``{}`` on ok /
    structured dict on rejection) so callers branch on ``error_type``
    uniformly across guards.

    Fields:
        outcome: ``'ok'`` or ``'rejection'``.
        op: The operation checked (``'update_task'`` or ``'set_task_status'``).
        task_id: The task identifier checked.
        agent_id: The caller's agent_id.
        error_type: Stable error-type string for callers to branch on
            (``'ReconTerminalWriteRejected'`` / ``'ReconLiveWorkflowWriteRejected'``
            / ``'ReconStaleSnapshotRejected'``). Empty on ok.
        reason: Human-readable rejection message. Empty on ok.
        hint: Actionable follow-up guidance. Empty on ok.
        live_status: The task's live status as observed by the caller.
        target_status: The status the caller was attempting to set
            (``set_task_status`` only; ``None`` for ``update_task``).
        snapshot_token: The snapshot status token the caller's write was
            based on, when supplied.
        corrective_path: Stable, machine-readable token naming the
            sanctioned seam a caller should use instead, when one exists.
            Empty on ok and on rejections with no sanctioned redirect
            (Gate 2 live-workflow, Gate 3 stale-snapshot). Populated only
            on the Gate 1 terminal-write rejection, where it names the
            same-status ``done_provenance`` repair seam
            (``set_task_status(..., 'done', done_provenance={...})``) —
            see :data:`TERMINAL_CORRECTIVE_PATH`.
    """

    outcome: Literal['ok', 'rejection']
    op: str = ''
    task_id: str = ''
    agent_id: str = ''
    error_type: str = ''
    reason: str = ''
    hint: str = ''
    live_status: str = ''
    target_status: str | None = None
    snapshot_token: str | None = None
    corrective_path: str = ''

    @property
    def is_rejection(self) -> bool:
        return self.outcome == 'rejection'

    def to_error_dict(self) -> dict:
        """Return a structured MCP error dict, or ``{}`` on ok."""
        if not self.is_rejection:
            return {}
        return {
            'error': self.reason,
            'error_type': self.error_type,
            'agent_id': self.agent_id,
            'task_id': self.task_id,
            'op': self.op,
            'live_status': self.live_status,
            'target_status': self.target_status,
            'snapshot_token': self.snapshot_token,
            'hint': self.hint,
            'corrective_path': self.corrective_path,
        }


def _reject(
    *,
    op: str,
    task_id: str,
    agent_id: str,
    error_type: str,
    reason: str,
    hint: str,
    live_status: str,
    target_status: str | None = None,
    snapshot_token: str | None = None,
    corrective_path: str = '',
) -> Verdict:
    """Small internal factory for a rejection :class:`Verdict`."""
    return Verdict(
        outcome='rejection',
        op=op,
        task_id=task_id,
        agent_id=agent_id,
        error_type=error_type,
        reason=reason,
        hint=hint,
        live_status=live_status,
        target_status=target_status,
        snapshot_token=snapshot_token,
        corrective_path=corrective_path,
    )


# ---------------------------------------------------------------------------
# check()
# ---------------------------------------------------------------------------


def check(
    op: str,
    *,
    task_id: str,
    project_root: str,
    agent_id: str,
    target_status: str | None,
    live_status: str,
    snapshot_token: str | None,
    is_annotation_clear: bool = False,
    task_metadata: object = None,
) -> Verdict:
    """Decide whether a recon-stage caller may perform *op* on *task_id*.

    Pure function — always runs its gates; the recon-stage scoping
    (``agent_id.startswith('recon-stage-')``) lives at the call site in
    ``task_interceptor.py``, not here.

    Gate 1 (terminal): ``op == 'update_task'`` AND ``live_status`` is
    terminal (done/cancelled) AND NOT ``is_annotation_clear`` ->
    ``ReconTerminalWriteRejected``. ``is_annotation_clear`` (task 2684,
    default ``False`` for full backward compatibility) is a non-load-bearing
    exemption: the caller should pass
    ``is_terminal_annotation_exempt(update_kwargs)`` (equivalently,
    ``is_terminal_annotation_clear(update_kwargs) or
    is_terminal_annotation_add(update_kwargs)``, but parsing the metadata
    payload only once — task 2695 efficiency amendment) so a metadata-only,
    merge-mode write touching only :data:`CLEARABLE_ANNOTATION_KEYS` (a
    clear) OR only :data:`X_ANNOTATION_PREFIX`-prefixed keys (an add, task
    2695) bypasses this gate. It bypasses Gate 1 ONLY — Gates 2/3 still
    compose (e.g. a stale ``snapshot_token`` still fails Gate 3).

    Gate 2 (live workflow): ``op == 'set_task_status'`` AND a live workflow
    is detected for ``task_id`` -> ``ReconLiveWorkflowWriteRejected``. The
    caller's ``live_status`` is forwarded as ``is_workflow_live_for_task``'s
    ``status`` kwarg so the project-wide orchestrator-lock signal is
    suppressed for tasks in a status the orchestrator never actively
    dispatches (done/cancelled/deferred) — see
    ``live_workflow_detector.ORCH_LIVE_INELIGIBLE_STATUSES``.

    Gate 3 (stale snapshot, op-agnostic, checked last): ``snapshot_token is
    not None`` AND it disagrees with ``live_status`` ->
    ``ReconStaleSnapshotRejected``. Checked last so a terminal/live-workflow
    rejection takes precedence over a stale-snapshot one.
    """
    if op == 'update_task' and live_status in TERMINAL_STATUSES and not is_annotation_clear:
        return _reject(
            op=op,
            task_id=task_id,
            agent_id=agent_id,
            error_type='ReconTerminalWriteRejected',
            reason=(
                f'recon-stage write blocked: task {task_id} is {live_status!r} '
                '(terminal); recon-stage agents may not write to terminal tasks.'
            ),
            hint=(
                'Terminal tasks (done, cancelled) are frozen against '
                'recon-stage update_task writes. The ONLY recon-stage '
                'correction on a terminal task is to done_provenance on a '
                'done task, via set_task_status(task_id, \'done\', '
                'done_provenance={\'kind\': \'merged\'|\'found_on_main\', '
                '\'commit\': <ancestor-checked sha>}) — a done->done repair '
                'that does not reopen the task and writes done_provenance '
                'ONLY. Other terminal-task string content fields (e.g. '
                'details, description, title) have NO recon-stage corrective '
                'path: file a human-gated workaround task to correct them. '
                'Note: update_task can never write done_provenance — it is '
                'rejected for every caller by the write-authority floor.'
            ),
            live_status=live_status,
            target_status=target_status,
            snapshot_token=snapshot_token,
            corrective_path=TERMINAL_CORRECTIVE_PATH,
        )

    if op == 'set_task_status' and is_workflow_live_for_task(
        task_id, project_root, status=live_status,
    ):
        return _reject(
            op=op,
            task_id=task_id,
            agent_id=agent_id,
            error_type='ReconLiveWorkflowWriteRejected',
            reason=(
                f'recon-stage write blocked: task {task_id} has a live '
                'workflow (worktree/branch or orchestrator lock active); '
                'recon-stage agents may not write status while a workflow '
                'is live.'
            ),
            hint=(
                'A live orchestrator workflow (worktree, recent commit, or '
                'project-level lock) is active for this task. Recon-stage '
                'status writes would race the pipeline and are rejected; '
                'wait for the workflow to complete.'
            ),
            live_status=live_status,
            target_status=target_status,
            snapshot_token=snapshot_token,
        )

    if snapshot_token is not None and snapshot_token != live_status:
        return _reject(
            op=op,
            task_id=task_id,
            agent_id=agent_id,
            error_type='ReconStaleSnapshotRejected',
            reason=(
                f'recon-stage write blocked: task {task_id} snapshot_token '
                f'{snapshot_token!r} does not match live_status '
                f'{live_status!r}; the snapshot this write was based on is '
                'stale.'
            ),
            hint=(
                'The task status changed after this recon-stage write was '
                'computed. Re-observe the live status and retry, or drop '
                'the snapshot_status/observed_status metadata key if '
                'freshness is not required.'
            ),
            live_status=live_status,
            target_status=target_status,
            snapshot_token=snapshot_token,
        )

    return Verdict(
        outcome='ok',
        op=op,
        task_id=task_id,
        agent_id=agent_id,
        live_status=live_status,
        target_status=target_status,
        snapshot_token=snapshot_token,
    )


# ---------------------------------------------------------------------------
# extract_snapshot_token()
# ---------------------------------------------------------------------------


def _coerce_metadata_dict(metadata: object) -> dict | None:
    """Coerce an ``update_task``-style ``metadata`` payload to a ``dict``.

    Shared coercion idiom (factored out of :func:`extract_snapshot_token`,
    task 2684) mirroring ``_reject_done_provenance_in_update_metadata``'s
    inline coercion in ``task_interceptor.py``. A ``dict`` passes through
    unchanged; a ``str`` is parsed via ``json.loads`` and kept only if the
    result is itself a ``dict``. Anything else — including a JSON string
    that parses to a non-dict (e.g. a list), invalid JSON, or a non-str/dict
    value such as ``None`` or ``42`` — returns ``None``.
    """
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            loaded = json.loads(metadata)
        except (ValueError, TypeError):
            return None
        return loaded if isinstance(loaded, dict) else None
    return None


def extract_snapshot_token(metadata: object) -> str | None:
    """Extract the snapshot status token from ``update_task``'s ``metadata`` kwarg.

    Coerces *metadata* via :func:`_coerce_metadata_dict`. Returns the value
    of the first :data:`SNAPSHOT_TOKEN_KEYS` entry present, when that value
    is itself a non-empty ``str``. Returns ``None`` when *metadata* is not a
    dict / JSON-object string, contains neither key, or the first present
    key's value is not a non-empty string (e.g. ``None``, a number, or a
    bool) — such values are treated as an absent token rather than coerced
    via ``str()``, so they can never spuriously disagree with a live status.
    """
    parsed = _coerce_metadata_dict(metadata)
    if parsed is None:
        return None
    for key in SNAPSHOT_TOKEN_KEYS:
        if key in parsed:
            val = parsed[key]
            return val if isinstance(val, str) and val else None
    return None


def _pure_terminal_annotation_merge(update_kwargs: dict) -> dict | None:
    """Return the parsed ``metadata`` dict iff *update_kwargs* is a pure,
    allowlisted-kwargs, merge-mode, non-empty metadata write — else ``None``.

    Shared eligibility core (task 2695) for both
    :func:`is_terminal_annotation_clear` and :func:`is_terminal_annotation_add`:
    both predicates layer their own per-key check (every key in
    :data:`CLEARABLE_ANNOTATION_KEYS`, or every key starting with
    :data:`X_ANNOTATION_PREFIX`, respectively) on top of the dict this
    function returns.

    Returns non-``None`` iff ALL of the following hold:

    (a) Every key in *update_kwargs* with a non-``None`` value is in
        :data:`_ANNOTATION_CLEAR_ALLOWED_KWARGS` — an unrecognized kwarg
        (any ``update_task`` parameter not on that allowlist, including a
        content-mutating field added after this predicate was written)
        disqualifies unconditionally: fail CLOSED, not open.
    (b) The effective metadata merge mode is ``'merge'`` — classified inline
        (``metadata_mode == 'merge'``, or both ``metadata_mode`` and ``append``
        omitted → the merge default) WITHOUT importing the backend (keeps this
        module dependency-light). Anything else — an explicit non-merge
        ``metadata_mode``, or ``append`` in any non-``None`` state (incl.
        ``append=False``) — is treated as non-merge and disqualifies here. NB:
        ``sqlite_task_backend._resolve_metadata_mode`` now REJECTS a bare
        ``append=False`` metadata write outright (the task-2180 metadata-wipe
        guard) rather than mapping it to 'replace'; that does not affect this
        predicate, whose own conservative classification already treats
        ``append=False`` as non-merge → disqualify. ``replace`` would clobber
        load-bearing keys; ``additive`` cannot overwrite/clear the target key
        (scalar-collision old-wins) — only ``merge`` both preserves other keys
        and overwrites the target.
    (c) ``update_kwargs['metadata']`` coerces (via :func:`_coerce_metadata_dict`)
        to a non-empty ``dict`` — the dict returned on success.
    """
    if any(
        key not in _ANNOTATION_CLEAR_ALLOWED_KWARGS and value is not None
        for key, value in update_kwargs.items()
    ):
        return None

    metadata_mode = update_kwargs.get('metadata_mode')
    append = update_kwargs.get('append')
    is_merge_mode = metadata_mode == 'merge' or (metadata_mode is None and append is None)
    if not is_merge_mode:
        return None

    parsed = _coerce_metadata_dict(update_kwargs.get('metadata'))
    if not parsed:
        return None

    return parsed


def _annotation_content_keys(parsed: dict) -> list[str]:
    """Return *parsed*'s keys with :data:`_CAUSATION_TRACING_KEYS` stripped.

    The real-annotation-content keys a per-key eligibility check (ALL
    clearable / ALL x_-prefixed) should evaluate — the mandatory
    recon-stage ``_causation_id`` tracing co-key (task 2697) is transparent
    to that check and never itself counts as (or disqualifies) content.
    """
    return [key for key in parsed if key not in _CAUSATION_TRACING_KEYS]


def is_terminal_annotation_clear(update_kwargs: dict) -> bool:
    """Is *update_kwargs* a pure, allowlisted, terminal-task annotation clear?

    True iff :func:`_pure_terminal_annotation_merge` succeeds AND, once
    :data:`_CAUSATION_TRACING_KEYS` co-keys (e.g. the mandatory recon-stage
    ``_causation_id`` tracing tag, task 2697) are stripped via
    :func:`_annotation_content_keys`, there is at least one remaining
    content key AND every one of them is in :data:`CLEARABLE_ANNOTATION_KEYS`
    — used to compute Gate 1's ``is_annotation_clear`` bypass (task 2684)
    for a non-load-bearing annotation-only ``update_task`` write against a
    terminal task. See :func:`_pure_terminal_annotation_merge` for the
    shared (allowlisted-kwargs, merge-mode, non-empty-metadata) eligibility
    checks. A payload consisting only of tracing co-keys (e.g. a bare
    ``{'_causation_id': ...}``) has no real annotation content and is NOT
    exempt.

    A "clear" is a merge-overwrite (including to ``None``); this predicate
    has no notion of key deletion (metadata has no such primitive).
    """
    parsed = _pure_terminal_annotation_merge(update_kwargs)
    if parsed is None:
        return False
    content = _annotation_content_keys(parsed)
    return bool(content) and all(key in CLEARABLE_ANNOTATION_KEYS for key in content)


def is_terminal_annotation_add(update_kwargs: dict) -> bool:
    """Is *update_kwargs* a pure, allowlisted, terminal-task x_-annotation add?

    True iff :func:`_pure_terminal_annotation_merge` succeeds AND, once
    :data:`_CAUSATION_TRACING_KEYS` co-keys (e.g. the mandatory recon-stage
    ``_causation_id`` tracing tag, task 2697) are stripped via
    :func:`_annotation_content_keys`, there is at least one remaining
    content key AND every one of them starts with :data:`X_ANNOTATION_PREFIX`
    (``'x_'``) — the sanctioned forward-compat annotation namespace
    (``shared.task_metadata.parse_metadata`` silently admits any
    ``x_``-prefixed key; see :data:`X_ANNOTATION_PREFIX`). See
    :func:`_pure_terminal_annotation_merge` for the shared (allowlisted-kwargs,
    merge-mode, non-empty-metadata) eligibility checks. A payload consisting
    only of tracing co-keys has no real annotation content and is NOT exempt.

    Widens task 2684's terminal-write exemption
    (:func:`is_terminal_annotation_clear`) to also cover ADDING (or
    overwriting) x_-prefixed forward-compat annotation keys on a terminal
    task, not just clearing :data:`CLEARABLE_ANNOTATION_KEYS`.

    x_-only limitation: a metadata payload mixing a
    :data:`CLEARABLE_ANNOTATION_KEYS` key with an ``x_``-prefixed key is NOT
    exempted by either predicate and remains rejected on a terminal task —
    real Stage-2 remediations are either a clear or an x_ add, never both in
    the same ``update_task`` call, so this is treated as a known, immaterial
    limitation rather than composing the two allowlists. This limitation
    survives stripping tracing co-keys: a mixed clearable+x_ residual still
    satisfies neither ALL-branch.
    """
    parsed = _pure_terminal_annotation_merge(update_kwargs)
    if parsed is None:
        return False
    content = _annotation_content_keys(parsed)
    return bool(content) and all(key.startswith(X_ANNOTATION_PREFIX) for key in content)


def is_terminal_annotation_exempt(update_kwargs: dict) -> bool:
    """Is *update_kwargs* exempt from Gate 1's terminal-write rejection?

    Equivalent to ``is_terminal_annotation_clear(update_kwargs) or
    is_terminal_annotation_add(update_kwargs)`` — this is the combined form
    :func:`check`'s ``is_annotation_clear`` argument wants (efficiency
    amendment, task 2695). Calling the two predicates separately each runs
    :func:`_pure_terminal_annotation_merge` (and, for JSON-string metadata,
    a second ``json.loads``) on the SAME payload; this function parses once
    and derives both the clear and add key-checks from that single result.

    Like both predicates above, evaluates its ALL-clearable / ALL-x_
    conditions against :func:`_annotation_content_keys` (parsed) — the
    mandatory recon-stage ``_causation_id`` tracing co-key (task 2697) is
    transparent to the check — and requires at least one remaining content
    key, so a tracing-only payload is NOT exempt.

    Behavior-identical to the two-call OR form, including the x_-only
    limitation: a metadata payload mixing a :data:`CLEARABLE_ANNOTATION_KEYS`
    key with an ``x_``-prefixed key satisfies neither ALL-clearable nor
    ALL-x_-prefixed, so it is still NOT exempt (see
    :func:`is_terminal_annotation_add`'s docstring) — even after tracing
    co-keys are stripped, since only ``_causation_id`` (not the clearable or
    x_ keys) is ever removed.
    """
    parsed = _pure_terminal_annotation_merge(update_kwargs)
    if parsed is None:
        return False
    content = _annotation_content_keys(parsed)
    if not content:
        return False
    return (
        all(key in CLEARABLE_ANNOTATION_KEYS for key in content)
        or all(key.startswith(X_ANNOTATION_PREFIX) for key in content)
    )
