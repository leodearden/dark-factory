"""Judges whether a done write's ``done_provenance`` is admissible.

A done write meets four checks, in order: the caller bar on the kind it
claims, the parse through ``shared/src/shared/task_metadata.py::DoneProvenance``,
the git evidence (a cited commit resolves and is on main), and freshness
against the task's last reopen. The per-kind contract lives on
``DoneProvenance``; the author-facing table is in ``docs/task-authoring.md``.
"""

import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any, Literal, NamedTuple, Protocol, get_args

from pydantic import ValidationError
from shared.git_async import GitResult
from shared.task_metadata import DoneProvenance
from shared.task_metadata_wire import coerce_task_metadata

from fused_memory.middleware.done_provenance_authz import (
    DETERMINISTIC_CALLER_ERROR_TYPE,
    DETERMINISTIC_PROVENANCE_KINDS,
    resolve_deterministic_provenance_authorization,
)

logger = logging.getLogger(__name__)


def _render_done_provenance_kinds() -> str:
    """Every ``DoneProvenance.kind``, quoted, for a caller-facing remedy (I2).

    Derived from the shared Literal so a new kind reaches every refusal that
    names the vocabulary without a second edit.
    """
    kinds = get_args(DoneProvenance.model_fields['kind'].annotation)
    return ', '.join(f'"{k}"' for k in kinds[:-1]) + f', or "{kinds[-1]}"'


_DONE_PROVENANCE_KINDS_TEXT = _render_done_provenance_kinds()
_CALLER_SELECTABLE_KINDS_GUIDANCE = (
    '{"kind": "merged", "commit": "<merge-sha>"} after the merge_request '
    'landed the commit on main; '
    '{"kind": "found_on_main", "commit": "<sha>", "note": "<text>"} when the '
    'implementation is already on main from a sibling task (the commit is '
    'ancestor-checked); or '
    '{"kind": "operational-verified", "escalation_id": "<esc-id>", '
    '"note": "<text>"} for a no-code operational ask closed via a resolved '
    'escalation. The deterministic-* kinds are stamped by DeterministicRunner '
    'and never supplied by hand. The per-kind table in docs/task-authoring.md '
    'is canonical.'
)
_KIND_REQUIRED_REASON = (
    f'done_provenance.kind is required (must be {_DONE_PROVENANCE_KINDS_TEXT}). '
    f'Pass done_provenance= one of: {_CALLER_SELECTABLE_KINDS_GUIDANCE}'
)


# ── Git-probe failure carriers (task 3455) ──────────────────────────────
#
# Both git probes used by validate_done_provenance (_resolve_commit_sha,
# _verify_commit_on_main) can fail two ways that need OPPOSITE remedies:
# git ran and affirmatively rejected the input, or the probe never
# completed (timeout / missing binary / unexpected error). Each returns a
# NamedTuple whose leading field names that distinction, so the semantics
# travel with the value instead of living in a call-site variable name.
# The authoritative branch-by-branch contract is each probe's docstring.
# The never-completed outcome is itself a carrier, _ProbeUnavailable,
# produced in one place (_git_probe); each probe maps it onto its own
# honest not-yet-confirmed arm.


_PROBE_TIMEOUT_SECONDS = 5.0


class GitProbeRunner(Protocol):
    """Runs one git argv for the done-provenance probes; production passes ``shared.git_async.run_git``."""

    async def __call__(self, cmd: Sequence[str], *, timeout: float | None = None) -> GitResult: ...


class _ProbeUnavailable(NamedTuple):
    """A git probe never rendered a verdict. See _git_probe."""

    detail: str


class _CommitResolutionFailure(NamedTuple):
    """`git rev-parse --verify` did not yield a SHA. See _resolve_commit_sha."""

    confirmed_absent: bool
    reason: str


class _AncestorCheckFailure(NamedTuple):
    """`git merge-base --is-ancestor` did not accept. See _verify_commit_on_main."""

    confirmed_off_main: bool
    detail: str


async def _git_probe(
    git_runner: GitProbeRunner, project_root: str, *args: str,
) -> GitResult | _ProbeUnavailable:
    """Run ``git -C <project_root> <args>`` within the probe timeout.

    Returns git's answer, whatever its exit code, or a
    :class:`_ProbeUnavailable` when git never answered: a missing binary, a
    timeout, or any other exception from the runner. The catch is broad on
    purpose: ``args`` can carry caller-supplied text, and an embedded NUL
    makes the spawn raise ``ValueError``.
    """
    try:
        result = await git_runner(
            ['git', '-C', project_root, *args], timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return _ProbeUnavailable('git binary not found')
    except Exception as exc:
        logger.warning(
            'git %s probe in %s raised; reporting the check as unavailable',
            args[0], project_root, exc_info=True,
        )
        return _ProbeUnavailable(f'{type(exc).__name__}: {exc}')
    if result.timed_out:
        return _ProbeUnavailable(f'git {args[0]} timed out')
    return result


def provenance_kind(raw: object) -> str | None:
    """The ``kind`` a raw ``done_provenance`` payload CLAIMS.

    Pure and total: a non-dict payload, a missing key, a non-``str`` value or
    a blank one all read as ``None``. It reports the CLAIM, not a validated
    kind — a refused write's row must still say what the caller asserted, and
    an unreadable claim must not be able to skip the audit row by raising.

    Shared by every journal site and by the caller bar
    (:func:`_provenance_caller_refusal`), so an accepted and a refused row
    report the kind identically and the bar judges the same claim the
    journal records (SPOT).
    """
    if not isinstance(raw, dict):
        return None
    kind = raw.get('kind')
    if not isinstance(kind, str):
        return None
    return kind.strip() or None


def provenance_stamp_now() -> str:
    """The production clock for ``done_provenance.stamped_at`` (task 3576).

    ``middleware/task_interceptor.py::TaskInterceptor`` stamps with it unless
    constructed with another ``provenance_stamp_clock``. Same convention as
    the interceptor's other timestamp sites (``reopen_at``, ``combined_at``):
    ``datetime.now(UTC).isoformat()``.
    """
    return datetime.now(UTC).isoformat()


def _provenance_caller_refusal(
    task_id: str,
    claimed_kind: str | None,
    *,
    is_recon_stage: bool,
    agent_id: str | None,
    config: Any,
) -> dict | None:
    """The done_provenance caller bar, judged on the kind a payload CLAIMS.

    Returns the refusal payload, or ``None`` when this caller may record that
    kind. Runs before the payload's shape is parsed, so an unauthorized caller
    is refused on identity rather than handed a shape hint first.
    """
    if claimed_kind in DETERMINISTIC_PROVENANCE_KINDS:
        # Factor 1 (PRD D11): config-free, and FIRST. A recon stage may not
        # self-authorize a runner-stamped close, and an operator who adds
        # 'recon-stage-' to the allowlist must not thereby reopen incident
        # 5156.
        if is_recon_stage:
            return _done_provenance_error(
                task_id,
                f'done_provenance with kind={claimed_kind!r} cannot be recorded by a '
                'recon-stage caller — a recon stage may not self-authorize a '
                'close that asserts a DeterministicRunner observation. This '
                'refusal is config-free: granting the caller prefix does not '
                'lift it.',
                error_type=DETERMINISTIC_CALLER_ERROR_TYPE,
            )
        # Factor 2: the configured caller allowlist, read live so the leaf
        # stays green-tier hot-reloadable.
        decision = resolve_deterministic_provenance_authorization(config, agent_id=agent_id)
        if not decision.allowed:
            return _done_provenance_error(
                task_id,
                decision.error or '',
                error_type=decision.error_type,
            )
    if claimed_kind == 'operational-verified' and is_recon_stage:
        return _done_provenance_error(
            task_id,
            'done_provenance with kind="operational-verified" cannot be '
            'recorded by a recon-stage caller — a recon stage may not '
            'self-authorize an operational close.',
        )
    return None


def _done_provenance_parse_reason(exc: ValidationError) -> str:
    """The caller-facing ``reason`` for a payload ``DoneProvenance`` refused.

    Routes on the first error's structured fields only. A missing or unknown
    kind gets the self-correcting remedy naming every kind; a per-kind
    requirement (``loc == ()``) is the model's own message, verbatim.
    """
    err = exc.errors()[0]
    loc = err['loc']
    if loc == ('kind',):
        if err['type'] == 'missing' or err['input'] is None:
            return _KIND_REQUIRED_REASON
        return f'done_provenance.kind must be {_DONE_PROVENANCE_KINDS_TEXT} (got {err["input"]!r})'
    if loc == ():
        return err['msg']
    field = '.'.join(str(part) for part in loc)
    return f'done_provenance.{field}: {err["msg"]} (got {err["input"]!r})'


async def validate_done_provenance(
    task_id: str,
    raw: object,
    project_root: str,
    *,
    require: bool,
    is_recon_stage: bool = False,
    agent_id: str | None = None,
    config: Any = None,
    git_runner: GitProbeRunner,
    stamp_clock: Callable[[], str],
) -> tuple[dict | None, dict | None]:
    """Authorize, parse and resolve done_provenance for set_task_status(done).

    The per-kind contract — which fields each kind requires and which it
    records — lives on ``shared/src/shared/task_metadata.py::DoneProvenance``;
    the author-facing table is in ``docs/task-authoring.md``. This function
    owns only what that model cannot know, in this order:

    1. THE CALLER BAR (:func:`_provenance_caller_refusal`), on the CLAIMED
       kind. The ``deterministic-*`` family (PRD C5 / D11, task 5241) asserts
       a MACHINE observation whose only live producer is DeterministicRunner:
       a ``recon-stage-`` caller is refused unconditionally (config-free,
       incident 5156), and any other caller must match a prefix in
       ``reconciliation.deterministic_provenance_allowed_agent_prefixes``
       (deny-on-missing; see ``middleware/done_provenance_authz.py``). Both
       refusals carry ``error_type='DeterministicProvenanceCallerNotPermitted'``.
       A recon stage is also refused ``operational-verified``, with no
       ``error_type``: it may not self-authorize an operational close.

       The identity is SELF-REPORTED. Under stateless HTTP a tool call
       carries no clientInfo for ``server/tools.py::_resolve_identity`` to
       fall back to, so a caller that sends no ``agent_id`` arrives as
       ``None`` and is refused; the orchestrator therefore sends
       ``agent_id='orchestrator'`` explicitly
       (``orchestrator/src/orchestrator/scheduler.py::Scheduler.set_task_status``).
       The bar DETERS a cooperating caller; it is not a security boundary.
       The residual is made visible rather than prevented — every done
       write, accepted or refused, leaves a write-journal row naming the
       resolved caller and the provenance kind.

    2. PARSE AT THE BOUNDARY through ``DoneProvenance.parse_for_write``, which
       also enforces the write-time-only requirements. A refusal's reason is
       built by :func:`_done_provenance_parse_reason`; the record keeps
       ``recorded_evidence()``.

    3. GIT: a supplied ``commit`` must resolve via ``git rev-parse`` and be an
       ancestor of main (``git merge-base --is-ancestor <sha> main``), so a
       steward cannot record a SHA that is only on a feature branch. The full
       SHA is recorded, plus the original ``commit_input`` when a short hash
       or ref was resolved.

    4. ``stamped_at`` for ``found_on_main`` is written here, server-side
       (task 3576); a caller-supplied value is discarded, never parsed.

    Returns ``(error_payload, resolved_provenance)``. When the error payload
    is non-None the transition must be aborted. When ``require`` is False,
    missing/empty provenance logs a warning and returns ``(None, None)`` so
    the transition proceeds; malformed provenance always errors regardless of
    ``require`` — we never want to record corrupt provenance on the task.

    ``is_recon_stage`` mirrors
    ``middleware/task_interceptor.py::TaskInterceptor._apply_status_transition``'s
    ``is_recon_stage_write`` classification (``agent_id.startswith(
    'recon-stage-')``). It is passed SEPARATELY from ``agent_id`` rather than
    re-derived here so this function and the transition gate can never
    disagree about who is a recon stage. ``agent_id`` and ``config`` default
    to the denying value, so a caller that forgets to thread them refuses
    rather than silently opening the bar.
    """
    if raw is None or raw == {}:
        if require:
            return _done_provenance_missing_error(task_id), None
        logger.warning(
            'set_task_status(%s, done) called without done_provenance; '
            'Stage-2 reconciliation will treat this task as provenance-unknown. '
            'To record verified evidence, pass done_provenance= one of: %s',
            task_id,
            _CALLER_SELECTABLE_KINDS_GUIDANCE,
        )
        return None, None

    if not isinstance(raw, dict):
        return (
            _done_provenance_error(
                task_id,
                'done_provenance must be an object with keys "kind", "commit", and/or "note"',
            ),
            None,
        )

    refusal = _provenance_caller_refusal(
        task_id,
        provenance_kind(raw),
        is_recon_stage=is_recon_stage,
        agent_id=agent_id,
        config=config,
    )
    if refusal is not None:
        return refusal, None

    try:
        provenance = DoneProvenance.parse_for_write(
            {key: value for key, value in raw.items() if key != 'stamped_at'}
        )
    except ValidationError as exc:
        return _done_provenance_error(task_id, _done_provenance_parse_reason(exc)), None

    resolved = provenance.recorded_evidence()
    commit_input = provenance.commit
    if commit_input is not None:
        sha_or_err = await _resolve_commit_sha(project_root, commit_input, git_runner=git_runner)
        if isinstance(sha_or_err, _CommitResolutionFailure):
            # Only a git-issued rejection is a "not found" verdict; see
            # _resolve_commit_sha for the branch-by-branch contract.
            verb = 'not found in' if sha_or_err.confirmed_absent else 'could not be resolved in'
            return _done_provenance_error(
                task_id,
                f'commit {commit_input!r} {verb} {project_root}: {sha_or_err.reason}',
            ), None
        resolved['commit'] = sha_or_err
        if sha_or_err != commit_input:
            resolved['commit_input'] = commit_input
        # The ancestor backstop applies to every kind that cites a commit,
        # not only the kinds that require one.
        ancestor_result = await _verify_commit_on_main(
            project_root, sha_or_err, git_runner=git_runner,
        )
        if ancestor_result is not None:
            # rc=1 is the only confirmed off-main verdict; every other
            # failure is NOT-YET-CONFIRMED and carries the opposite remedy.
            # See _verify_commit_on_main for the full contract.
            prefix = (
                'is not on main'
                if ancestor_result.confirmed_off_main
                else 'could not be confirmed on main'
            )
            return _done_provenance_error(
                task_id,
                f'kind={provenance.kind!r} but commit {sha_or_err} {prefix}: '
                f'{ancestor_result.detail}.',
            ), None

    if provenance.kind == 'found_on_main':
        # THE SINGLE WRITE SITE for done_provenance.stamped_at (task 3576).
        #
        # Every found_on_main producer funnels through this function: the
        # fresh-done path and the same-status repair seam in
        # middleware/task_interceptor.py (TaskInterceptor._apply_status_transition,
        # _repair_done_provenance_same_status), agent-authored
        # set_task_status calls, and the orchestrator's
        # Scheduler.mark_done(kind='found_on_main', ...) which reaches here
        # via set_task_status. So no producer-side change is needed, and no
        # future producer can forget to stamp. That matters: this defect
        # class's history (1087 -> 1091 -> 1180 -> 2372 -> 2500 -> 2648 ->
        # 2787 -> 2794 -> 3057) is a chain of point fixes each reopened by an
        # unguarded seam the previous fix missed.
        #
        # Scoped to found_on_main deliberately — it is the
        # attribution-by-inference kind the soak gate watches. `merged`
        # already carries independent landing evidence (merge-queue journal +
        # is_ancestor + commit citation), and stamping it would perturb the
        # exact-equality merged-kind assertions across the existing suites
        # for no benefit.
        #
        # A caller-supplied value is DISCARDED with a warning rather than
        # rejected. Rejection would break the same-status repair seam:
        # _repair_done_provenance_same_status re-submits a previously-stored
        # blob, which post-3576 carries its own stamped_at, so a hard reject
        # would fail every repair of an already-stamped blob. Refreshing on
        # repair is also the correct semantics — a repair is a fresh
        # assertion of the attribution, and a repair that leaves the task
        # still misattributed SHOULD re-enter the gate window. The warning
        # keeps the override visible rather than silent.
        supplied_stamp = raw.get('stamped_at')
        if supplied_stamp is not None:
            logger.warning(
                'done_provenance.stamped_at is server-generated; discarding '
                'caller-supplied value %r for task %s',
                supplied_stamp,
                task_id,
            )
        resolved['stamped_at'] = stamp_clock()

    return None, resolved


async def _verify_commit_on_main(
    project_root: str, sha: str, *, git_runner: GitProbeRunner,
) -> _AncestorCheckFailure | None:
    """Verify ``sha`` is reachable from ``main`` via ``merge-base --is-ancestor``.

    Returns None on success (commit is on main). On failure, returns an
    :class:`_AncestorCheckFailure`:

    - ``confirmed_off_main=True`` only for rc=1 — git affirmatively
      determined ``sha`` is NOT reachable from ``main``. This is the only
      case that actually means "not on main", and the only one whose
      remedy is to re-derive the landing SHA.
    - ``confirmed_off_main=False`` for every other failure (a non-1 git
      error such as an unresolvable ``main`` ref, a timeout, a missing git
      binary, or any unexpected exception) — the check never completed, so
      the caller must treat it as NOT-YET-CONFIRMED rather than a
      not-on-main verdict. The remedy there is the opposite one: fetch /
      fix the checkout / retry. Task 3455: conflating the two under a
      single "is not on main" wording is the same defect class as task
      3119's premature outcome assertions. Any failure branch added later
      inherits ``False``, i.e. the weaker (honest) claim, by default.

    Used as a backstop for kind="merged" provenance so that a SHA from a
    feature branch cannot pass as a merge commit.
    """
    result = await _git_probe(git_runner, project_root, 'merge-base', '--is-ancestor', sha, 'main')
    if isinstance(result, _ProbeUnavailable):
        return _AncestorCheckFailure(False, result.detail)

    # Exit codes from `git merge-base --is-ancestor`:
    #   0 — sha is reachable from main
    #   1 — sha is NOT reachable from main
    #   other — git error (e.g. unknown ref `main`)
    if result.returncode == 0:
        return None
    if result.returncode == 1:
        return _AncestorCheckFailure(True, 'commit is not an ancestor of main')
    return _AncestorCheckFailure(False, result.stderr or 'merge-base failed')


async def _resolve_commit_sha(
    project_root: str, commit: str, *, git_runner: GitProbeRunner,
) -> str | _CommitResolutionFailure:
    """Resolve a tree-ish to a 40-char SHA via ``git rev-parse --verify --quiet``.

    Returns the SHA on success, or a :class:`_CommitResolutionFailure` whose
    ``confirmed_absent`` flag says whether git actually rendered a verdict
    (task 3455, mirroring :func:`_verify_commit_on_main`'s rc=1 policy):

    - ``confirmed_absent=True`` ONLY for rc=1 under ``--verify --quiet`` —
      git ran inside a readable repository, evaluated the ref, and rejected
      it (missing, unknown name, not peelable to a commit, ambiguous
      prefix). Only then may the caller say the commit was "not found in
      <project_root>".
    - ``confirmed_absent=False`` for every other outcome: rc=128, which git
      returns for a project_root that is not a readable git repository or
      does not exist (task 5274); a timeout; a missing git binary; an
      unexpected exception; or empty output on rc=0.
      rev-parse never rendered a verdict on the ref, so the honest report
      is "could not be resolved", with the same fetch / fix-the-checkout /
      retry remedy as an unconfirmed ancestor check. Any failure branch
      added later inherits ``False``, i.e. the weaker claim, by default.

    Uses a commit-peeling suffix (``^{commit}``) so a tag that points at a
    commit resolves to the commit SHA directly.
    """
    result = await _git_probe(
        git_runner, project_root, 'rev-parse', '--verify', '--quiet', f'{commit}^{{commit}}',
    )
    if isinstance(result, _ProbeUnavailable):
        return _CommitResolutionFailure(False, result.detail)
    if result.returncode == 1:
        return _CommitResolutionFailure(
            True,
            result.stderr or 'does not name exactly one commit (an unknown name, or an '
            'ambiguous short prefix: pass the full 40-char SHA)',
        )
    if result.returncode != 0:
        return _CommitResolutionFailure(False, result.stderr or 'rev-parse failed')
    if len(result.stdout) < 7:
        return _CommitResolutionFailure(False, 'empty rev-parse output')
    return result.stdout


# ── Reopen-freshness gate (task 2674, PRD task alpha) ───────────────────
#
# Closes Face B (re-derivation clobber, task-1175 shape): a legitimate
# done->pending reopen re-marked done within one dispatch tick citing a
# PRE-reopen commit. Fires only for commit-bearing done_provenance kinds
# ('merged'/'found_on_main') on a task whose metadata.reopen_at is set
# (task 2649 stamps this atomically on every terminal-exit reopen).


async def _resolve_commit_committed_at(
    project_root: str, sha: str, *, git_runner: GitProbeRunner,
) -> str | None:
    """Resolve a commit's committer date via ``git show -s --format=%cI``.

    Returns the stripped ``%cI`` (strict ISO-8601, offset-aware) string on
    success, or None on any failure (git binary missing, non-zero exit,
    timeout, empty output). Runs through :func:`_git_probe`, like the other
    done-provenance probes, so it shares their timeout and error handling.
    """
    result = await _git_probe(git_runner, project_root, 'show', '-s', '--format=%cI', sha)
    if isinstance(result, _ProbeUnavailable) or not result.ok:
        return None
    return result.stdout or None


def _extract_reopen_at(before: Any) -> str | None:
    """Return ``metadata.reopen_at`` from a Taskmaster ``before`` snapshot.

    Uses ``shared.task_metadata_wire.coerce_task_metadata`` so a JSON-string
    metadata blob is handled identically to an already-parsed dict. Returns
    None when absent, empty, or unresolvable — callers treat that as "never
    reopened" (the reopen-freshness gate stays inert).
    """
    raw = before.get('metadata') if isinstance(before, dict) else None
    parsed = coerce_task_metadata(raw)
    if not isinstance(parsed, dict):
        return None
    reopen_at = parsed.get('reopen_at')
    return reopen_at if isinstance(reopen_at, str) and reopen_at else None


def _parse_aware_utc(value: str) -> datetime | None:
    """Parse an ISO-8601 datetime string to an aware UTC :class:`datetime`.

    A naive (no ``tzinfo``) input is assumed to already be UTC; an aware
    input is converted via ``astimezone(UTC)``. Returns None on any parse
    failure — the reopen-freshness gate's caller treats an unparseable
    date as fail-closed (stale), never as a silent pass.
    """
    try:
        dt = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _done_evidence_stale_error(
    task_id: str,
    evidence_commit: str | None,
    evidence_committed_at: str | None,
    reopen_at: str | None,
    agent_id: str | None,
) -> dict:
    """Structured error returned when the reopen-freshness gate rejects (enforce-mode).

    Structured fields only, no prose — MCP callers (scheduler, steward)
    branch on ``error`` uniformly, matching the flat refusal shape of the
    interceptor's other done-write gates
    (``middleware/task_interceptor.py::_terminal_exit_error``,
    ``middleware/task_interceptor.py::_done_gate_error``).
    """
    return {
        'success': False,
        'error': 'done_evidence_stale',
        'task_id': task_id,
        'evidence_commit': evidence_commit,
        'evidence_committed_at': evidence_committed_at,
        'reopen_at': reopen_at,
        'agent_id': agent_id,
    }


def _done_evidence_stale_override_invalid_error(task_id: str, agent_id: str | None) -> dict:
    """Structured error for a present-but-invalid ``stale_evidence_override``.

    Mode-independent (returned in both enforce and warn) — an explicit
    malformed override, or one supplied by a recon-stage caller, is always
    a loud caller error, never a silent fall-through to the plain
    :func:`_done_evidence_stale_error` rejection.
    """
    return {
        'success': False,
        'error': 'done_evidence_stale_override_invalid',
        'task_id': task_id,
        'agent_id': agent_id,
        'hint': (
            'stale_evidence_override requires a non-recon-stage caller and '
            'non-empty escalation_id + reason fields'
        ),
    }


def _valid_stale_evidence_override(override: object, is_recon_stage: bool) -> bool:
    """True iff ``override`` is a well-formed, non-recon-stage bypass assertion.

    A recon-stage caller may never self-authorize a stale-evidence bypass —
    the override is a human/orchestrator-sanctioned exception, not a claim
    a reconciliation stage can make about its own write. Otherwise valid
    iff ``override`` is a dict carrying non-empty string ``escalation_id``
    and ``reason`` fields (both are recorded verbatim as Stage-2 audit
    substrate — see :func:`check_reopen_freshness`).
    """
    if is_recon_stage:
        return False
    if not isinstance(override, dict):
        return False
    escalation_id = override.get('escalation_id')
    reason = override.get('reason')
    return (
        isinstance(escalation_id, str)
        and bool(escalation_id)
        and isinstance(reason, str)
        and bool(reason)
    )


StaleEvidenceMode = Literal['warn', 'enforce']


async def check_reopen_freshness(
    task_id: str,
    *,
    resolved_provenance: dict | None,
    raw_done_provenance: object,
    before: Any,
    project_root: str,
    agent_id: str | None,
    is_recon_stage: bool,
    mode: StaleEvidenceMode,
    git_runner: GitProbeRunner,
) -> dict | None:
    """Reject (enforce) or no-op (warn) a done-write citing PRE-reopen evidence.

    Inert (returns None immediately) unless ALL of:
      - ``resolved_provenance`` is a dict with ``kind`` in
        ``DoneProvenance.COMMIT_REQUIRED_KINDS`` (commitless kinds —
        deterministic-* — self-evidence and are exempt);
      - it carries a truthy ``commit``;
      - ``before`` (the pre-write task snapshot) has a non-empty
        ``metadata.reopen_at`` (a task that was never reopened has nothing
        to be stale against).

    When the gate fires, the evidence commit's committer date and
    ``reopen_at`` are both normalised to aware UTC and compared. Evidence
    dated before the reopen is STALE. Fail-CLOSED (task 2674 S10): an
    unparseable date on either side (git failure / parse failure) is ALSO
    treated as stale — a gate that cannot prove freshness must not certify
    it, so the comparison never silently passes on indeterminate input.
    This deliberately does not distinguish a transient git failure (binary
    hiccup, timeout) from a genuinely pre-reopen commit — both fail closed
    identically; see ``task_status.done_evidence_stale_unresolved_commit``
    below for the observability seam that at least makes the two cases
    distinguishable in logs ahead of the task-gamma enforce flip.

    Whenever the evidence commit's committer date cannot be resolved at
    all (``_resolve_commit_committed_at`` returns None), a
    ``task_status.done_evidence_stale_unresolved_commit`` WARNING (stable
    grep anchor) is logged unconditionally — in both modes, and whether or
    not an override subsequently bypasses the gate — purely as a
    diagnostic aid; it does not itself change the accept/reject outcome.

    On stale, BEFORE the enforce/warn mode dispatch: a well-formed,
    non-recon-stage ``raw_done_provenance['stale_evidence_override']``
    (validated by :func:`_valid_stale_evidence_override`) bypasses the gate
    unconditionally (both modes). The override is mutated onto
    ``resolved_provenance`` VERBATIM — the caller persists
    ``resolved_provenance`` into ``audit_fields['done_provenance']``, so
    this is Stage-2 audit substrate, not a live cross-service escalation
    lookup (fused-memory has no orchestrator-escalation client).

    Absent any override: ``mode == 'enforce'`` returns the typed
    :func:`_done_evidence_stale_error` rejection (the caller must abort the
    write); otherwise (warn mode) emits one ``task_status.done_evidence_stale_warn``
    census WARNING (stable grep anchor) and returns None (the write
    proceeds).

    A present-but-invalid override (malformed shape, or a recon-stage
    caller) is distinguished from "no override": it returns the typed
    :func:`_done_evidence_stale_override_invalid_error` rejection,
    mode-independently (enforce and warn alike) — an explicit malformed
    override is always a loud caller error, never a silent fall-through to
    the plain ``done_evidence_stale`` rejection or (worse) an accept. Only
    the no-override stale path proceeds to the enforce/warn mode dispatch.

    Rollout-safety note: this makes the override-invalid rejection the
    ONE case where 'warn' mode is not purely observational — every other
    branch of 'warn' mode (no override, or a *valid* override) never
    blocks a write. This is intentional (a malformed override is a caller
    bug, not a staleness verdict the rollout mode should suppress — see
    task 2674 design decisions), not an oversight; callers relying on
    warn-mode being unconditionally non-blocking during the alpha rollout
    must still handle ``done_evidence_stale_override_invalid``.
    """
    if not isinstance(resolved_provenance, dict):
        return None
    if resolved_provenance.get('kind') not in DoneProvenance.COMMIT_REQUIRED_KINDS:
        return None
    evidence_commit = resolved_provenance.get('commit')
    if not evidence_commit:
        return None
    reopen_at = _extract_reopen_at(before)
    if not reopen_at:
        return None

    evidence_committed_at = await _resolve_commit_committed_at(
        project_root, evidence_commit, git_runner=git_runner,
    )
    if evidence_committed_at is None:
        # Distinguishable from a genuine date-based staleness verdict: this
        # fires whenever `git show` itself failed (binary missing, timeout,
        # non-zero exit) rather than because the comparison found the
        # evidence dated before reopen_at. The disposition is unchanged
        # (still fail-closed — see this function's docstring), but this gives
        # operators a grep-stable way to tell "couldn't prove freshness
        # because git errored" apart from "proved stale" ahead of the
        # task-gamma enforce flip. Logged unconditionally (both modes),
        # before the override/mode dispatch below.
        logger.warning(
            'task_status.done_evidence_stale_unresolved_commit task_id=%s '
            'evidence_commit=%s reopen_at=%s agent_id=%s',
            task_id, evidence_commit, reopen_at, agent_id,
        )
    evidence_dt = _parse_aware_utc(evidence_committed_at) if evidence_committed_at else None
    reopen_dt = _parse_aware_utc(reopen_at)

    stale = evidence_dt is None or reopen_dt is None or evidence_dt < reopen_dt
    if not stale:
        return None

    override = (
        raw_done_provenance.get('stale_evidence_override')
        if isinstance(raw_done_provenance, dict)
        else None
    )
    if override is not None:
        if _valid_stale_evidence_override(override, is_recon_stage):
            resolved_provenance['stale_evidence_override'] = {
                'escalation_id': override['escalation_id'],
                'reason': override['reason'],
            }
            logger.info(
                'task_status.done_evidence_stale_override_accepted task_id=%s escalation_id=%s',
                task_id, override['escalation_id'],
            )
            return None
        return _done_evidence_stale_override_invalid_error(task_id, agent_id)

    if mode == 'enforce':
        return _done_evidence_stale_error(
            task_id, evidence_commit, evidence_committed_at, reopen_at, agent_id,
        )
    logger.warning(
        'task_status.done_evidence_stale_warn task_id=%s evidence_commit=%s '
        'evidence_committed_at=%s reopen_at=%s agent_id=%s',
        task_id, evidence_commit, evidence_committed_at, reopen_at, agent_id,
    )
    return None


def _done_provenance_missing_error(task_id: str) -> dict:
    return {
        'success': False,
        'error': 'done_provenance_required',
        'task_id': task_id,
        'hint': (
            'Cannot mark task done: done_provenance is required when '
            'reconciliation.require_done_provenance is enabled. Pass '
            f'done_provenance= one of: {_CALLER_SELECTABLE_KINDS_GUIDANCE}'
        ),
    }


def _done_provenance_error(
    task_id: str,
    reason: str,
    *,
    error_type: str | None = None,
) -> dict:
    """The single builder for every ``done_provenance_invalid`` refusal.

    ``error_type`` names a machine-readable refusal CLASS for the caller-bar
    rejections (task 5241). The key is included only when supplied, so every
    pre-existing refusal payload stays byte-identical — a consumer keying on
    its absence is unaffected, and one that keys on its presence learns
    something no ``reason`` substring match could tell it reliably.
    """
    payload = {
        'success': False,
        'error': 'done_provenance_invalid',
        'task_id': task_id,
        'reason': reason,
    }
    if error_type is not None:
        payload['error_type'] = error_type
    return payload
