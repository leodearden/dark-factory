#!/usr/bin/env python3
"""scripts/legibility/nightly.py — nightly trickle pipeline assembly (PRD task ε).

Assembles the per-project nightly trickle: inventory+sample (β) -> digest
(α) -> code (δ) -> merge (γ) -> docs-only commit -> census trigger (ζ). See
plans/confusion-reduction-prd.md §5.5 (pipeline), §7.4 (per-project config),
decisions 7/8 (fail-loud contract, liveness probes git history never),
boundary test §8.8.

Every stage is reached behind a dependency-injection seam (``invoke`` for
the LLM, ``status_fetcher`` for the census, ``poster`` for escalation,
``committer`` for git) plus module-level functions a caller can monkeypatch
-- mirrors the established seam convention (coder.py's ``invoke`` override,
census_trigger.py's injected ``status_fetcher``). This is what the
systemd ``legibility-trickle@.service`` template runs nightly, and what
``install-trickle-timer.sh``/``check_trickle_liveness.sh`` install and probe.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Self-bootstrap for standalone `python scripts/legibility/nightly.py` runs
# (and the systemd ExecStart, which invokes this file directly) -- must run
# BEFORE the `legibility.*` imports below, since a direct script invocation
# puts only scripts/legibility/ (not scripts/) on sys.path. Skipped under
# pytest/normal package import: __name__ is 'legibility.nightly'. Mirrors
# sampling.py/census_trigger.py's identical guard.
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legibility import census_trigger, codebook, coder, digest, inventory, sampling  # noqa: E402
from legibility.config import LegibilityConfig, load_config  # noqa: E402

logger = logging.getLogger('legibility.nightly')


# ---------------------------------------------------------------------------
# resolve_config_path — map a bare project_id to its legibility.yaml
# ---------------------------------------------------------------------------

def _default_search_roots() -> list[Path]:
    """Default search roots for :func:`resolve_config_path`.

    Env ``LEGIBILITY_SEARCH_ROOTS`` (``os.pathsep``-split) if set, else the
    dark-factory repo's own parent directory -- mirrors
    ``skills/factory-init/scripts/find_escalation_port.known_project_roots``'s
    sibling-repos-under-``/home/leo/src`` convention. Each returned path is a
    PARENT directory: :func:`resolve_config_path` globs one level down for
    candidate project roots, it is not itself a project root.
    """
    env_value = os.environ.get('LEGIBILITY_SEARCH_ROOTS')
    if env_value:
        return [Path(p) for p in env_value.split(os.pathsep) if p]
    repo_root = Path(__file__).resolve().parents[2]
    return [repo_root.parent]


def resolve_config_path(
    project_id: str, search_roots: Sequence[str | Path] | None = None,
) -> Path:
    """Resolve *project_id* to its ``docs/legibility/legibility.yaml`` path.

    *search_roots* (default: :func:`_default_search_roots`) is a list of
    PARENT directories; each is globbed one level down for candidate
    project-root directories, and each candidate's
    ``docs/legibility/legibility.yaml`` (if present and loadable) is matched
    against its own authoritative ``project_id`` field via
    :func:`legibility.config.load_config` -- never a directory-name guess.
    A candidate whose config fails to load (malformed YAML, schema error) is
    skipped rather than aborting the whole search. Raises
    ``FileNotFoundError`` if no candidate matches.
    """
    roots = (
        [Path(r) for r in search_roots] if search_roots is not None
        else _default_search_roots()
    )
    for root in roots:
        if not root.is_dir():
            continue
        for candidate_root in sorted(root.iterdir()):
            if not candidate_root.is_dir():
                continue
            config_path = candidate_root / 'docs' / 'legibility' / 'legibility.yaml'
            if not config_path.is_file():
                continue
            try:
                cfg = load_config(config_path)
            except Exception:
                continue
            if cfg.project_id == project_id:
                return config_path
    raise FileNotFoundError(
        f'no legibility.yaml found for project_id={project_id!r} under '
        f'search roots {[str(r) for r in roots]!r}'
    )


# ---------------------------------------------------------------------------
# select_scored_records / select_digest_sessions — inventory -> score ->
# classify -> stratified sample
# ---------------------------------------------------------------------------

def select_scored_records(
    cfg: LegibilityConfig, projects_root: Path | str, target_date: date,
) -> list[sampling.ScoredRecord]:
    """Enumerate *target_date*'s sessions for *cfg* and assemble a
    :class:`~legibility.sampling.ScoredRecord` per session.

    Reuses ``inventory.enumerate_sessions`` plus sampling's own private
    one-pass helpers (``_score_and_find_first_turn`` /
    ``_first_user_turn_text``) -- the EXACT loop ``sampling.main`` uses --
    rather than duplicating the score+first-turn pass or adding a new public
    function to the already-landed β module.
    """
    sessions = inventory.enumerate_sessions(projects_root, cfg.cwd_prefixes, target_date)

    scored: list[sampling.ScoredRecord] = []
    for session in sessions:
        counts, first_turn = sampling._score_and_find_first_turn(session.path)
        stratum = sampling.classify_agent_class(first_turn, session.path)
        scored.append(
            sampling.ScoredRecord(
                session=session,
                stratum=stratum,
                counts=counts,
                first_turn_text=sampling._first_user_turn_text(first_turn),
            )
        )
    return scored


def select_digest_sessions(
    cfg: LegibilityConfig, projects_root: Path | str, target_date: date,
) -> list[sampling.ScoredRecord]:
    """The budget-bounded, stratified subset of *target_date*'s sessions to
    digest -- :func:`select_scored_records` narrowed by
    ``sampling.stratified_sample``."""
    scored = select_scored_records(cfg, projects_root, target_date)
    return sampling.stratified_sample(scored, cfg).selected


# ---------------------------------------------------------------------------
# build_digests — render one digest per selected session, isolating crashes
# ---------------------------------------------------------------------------

DEFAULT_MAX_DIGEST_BYTES = 15360


def build_digests(
    selected: Sequence[sampling.ScoredRecord],
    *,
    max_bytes: int = DEFAULT_MAX_DIGEST_BYTES,
    build=digest.build_digest,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Render one confusion digest per *selected* record via *build*
    (default :func:`legibility.digest.build_digest`), passing beta's already
    -authoritative ``rec.stratum`` as ``agent_class_override`` -- alpha never
    re-guesses when the caller already knows.

    Any exception raised by *build* for a given record is isolated: it is
    captured as ``(session_basename, reason)`` in the returned
    ``extractor_failures`` list rather than propagated or fabricated into a
    placeholder digest, so a driving caller (:func:`run_nightly`) can treat
    a non-empty ``extractor_failures`` as the extractor-crash fail-loud
    trigger (PRD decision 8).
    """
    digests: list[str] = []
    extractor_failures: list[tuple[str, str]] = []

    for record in selected:
        try:
            rendered = build(
                record.path, agent_class_override=record.stratum, max_bytes=max_bytes,
            )
        except Exception as exc:  # noqa: BLE001 - isolate, never propagate/fabricate
            extractor_failures.append((record.path.name, str(exc)))
            continue
        digests.append(rendered)

    return digests, extractor_failures


# ---------------------------------------------------------------------------
# git helpers — docs-only commit of the codebook, ref-lock retry, never stash
# ---------------------------------------------------------------------------

_LOCK_ERROR_PATTERNS: tuple[str, ...] = (
    'cannot lock ref',
    'index.lock',
    'another git process',
    'unable to create',
)
"""Case-insensitive stderr substrings identifying a transient git ref/index
lock contention (the merge worker/hooks racing for the lock in the shared
main checkout) -- worth retrying. Any other failure (e.g. "nothing to
commit") is NOT retried."""

_DEFAULT_COMMIT_RETRIES = 5
_DEFAULT_COMMIT_BACKOFF_SECS = 0.2


def _git_status_changed(repo: Path | str, relpath: Path | str, *, runner=None) -> bool:
    """True iff ``git status --porcelain -- relpath`` reports a change.

    *runner* defaults to ``subprocess.run`` looked up at CALL time (not
    bound as a function-default), so a test can monkeypatch
    ``nightly.subprocess.run`` and have it take effect.
    """
    run = runner if runner is not None else subprocess.run
    result = run(
        ['git', '-C', str(repo), 'status', '--porcelain', '--', str(relpath)],
        capture_output=True, text=True,
    )
    return bool((result.stdout or '').strip())


@dataclass
class GitCommitResult:
    """Outcome of :func:`_git_commit_docs_only`.

    ``ok=False`` (never a raised exception) on any failure -- including
    exhausted ref-lock retries or a genuine no-op (nothing to commit) --
    so the caller (:func:`run_nightly`) can escalate rather than crash.
    """

    ok: bool
    sha: str | None = None
    stderr: str = ''
    attempts: int = 0


def _git_commit_docs_only(
    repo: Path | str,
    paths: Sequence[Path | str],
    message: str,
    *,
    runner=None,
    retries: int = _DEFAULT_COMMIT_RETRIES,
    backoff: float = _DEFAULT_COMMIT_BACKOFF_SECS,
) -> GitCommitResult:
    """Commit ONLY *paths* (``git commit --only <paths> -m message``) in
    *repo* -- never any other dirty/staged path, and NEVER ``git stash``
    (CLAUDE.md's machine-operated-main-checkout rules).

    Retries up to *retries* attempts when the commit's stderr matches a
    known transient ref/index-lock pattern (:data:`_LOCK_ERROR_PATTERNS`);
    any other failure (e.g. a genuine no-op, "nothing to commit") fails
    immediately without retrying. On success, resolves the new commit sha
    via a follow-up ``git rev-parse HEAD`` (through the same *runner*).
    Returns ``ok=False`` (never raises) after the final failed attempt.
    """
    run = runner if runner is not None else subprocess.run
    str_paths = [str(p) for p in paths]

    stderr = ''
    for attempt in range(1, retries + 1):
        result = run(
            ['git', '-C', str(repo), 'commit', '--only', *str_paths, '-m', message],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            sha_result = run(
                ['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                capture_output=True, text=True,
            )
            sha = (sha_result.stdout or '').strip() or None
            return GitCommitResult(ok=True, sha=sha, stderr='', attempts=attempt)

        stderr = result.stderr or ''
        is_lock_error = any(
            pattern in stderr.lower() for pattern in _LOCK_ERROR_PATTERNS
        )
        if is_lock_error and attempt < retries:
            if backoff:
                time.sleep(backoff)
            continue
        return GitCommitResult(ok=False, sha=None, stderr=stderr, attempts=attempt)

    return GitCommitResult(ok=False, sha=None, stderr=stderr, attempts=retries)


# ---------------------------------------------------------------------------
# escalation — fail-loud contract (PRD decision 8): best-effort escalate_info
# ---------------------------------------------------------------------------

_ESCALATION_TOOL_NAME = 'escalate_info'
_ESCALATION_AGENT_ROLE = 'legibility-trickle'
_ESCALATION_CATEGORY = 'infra_issue'
_ESCALATION_SEVERITY = 'info'


def _build_escalation_arguments(cfg: LegibilityConfig, summary: str, detail: str) -> dict:
    """Build the ``escalate_info`` MCP tool arguments (PRD decision 8): a
    synthetic ``task_id`` labels the trickle as the source since it is a
    timer-driven agent, not a Taskmaster task."""
    return {
        'task_id': f'legibility-trickle-{cfg.project_id}',
        'agent_role': _ESCALATION_AGENT_ROLE,
        'category': _ESCALATION_CATEGORY,
        'severity': _ESCALATION_SEVERITY,
        'summary': summary,
        'detail': detail,
    }


def _default_poster(url: str, envelope: dict) -> None:
    """Post *envelope* to *url* via a real (lazily-imported) httpx POST.

    ``httpx`` is imported lazily since it is not a ``scripts/`` dependency
    -- mirrors ``census_trigger.default_status_fetcher``. Raises on any
    network/HTTP failure; :func:`post_escalation` wraps this best-effort.
    """
    import httpx

    response = httpx.post(url, json=envelope, timeout=10.0)
    response.raise_for_status()


def post_escalation(
    cfg: LegibilityConfig, summary: str, detail: str, *, poster=None,
) -> bool:
    """Best-effort escalate_info POST for a fail-loud trigger (PRD decision
    8): extractor crash, coder storm, or commit failure.

    Posts an MCP ``tools/call`` JSON-RPC envelope for tool
    ``escalate_info`` to ``http://localhost:<cfg.escalation_port>/mcp``
    (via *poster*, default :func:`_default_poster`). NEVER raises: any
    failure (poster exception, network error) is logged as a warning and
    swallowed, returning False -- a down escalation server must not mask
    the underlying failure, since the caller's non-zero exit code is the
    authoritative loud signal regardless of whether this POST succeeded.
    Returns True on a successful post.
    """
    poster_fn = poster if poster is not None else _default_poster
    url = f'http://localhost:{cfg.escalation_port}/mcp'
    arguments = _build_escalation_arguments(cfg, summary, detail)
    envelope = {
        'jsonrpc': '2.0',
        'id': 1,
        'method': 'tools/call',
        'params': {'name': _ESCALATION_TOOL_NAME, 'arguments': arguments},
    }
    try:
        poster_fn(url, envelope)
        return True
    except Exception as exc:  # noqa: BLE001 - best-effort, never propagate
        logger.warning(
            'legibility trickle: escalation post failed (best-effort, run '
            'still exits non-zero): %s', exc,
        )
        return False


# ---------------------------------------------------------------------------
# evaluate_census_step — census trigger (ζ) evaluation + best-effort launch
# ---------------------------------------------------------------------------

_CENSUS_ENTRYPOINT_NAME = 'census.py'
"""scripts/legibility/census.py -- task η. NOT a dependency of ε: this
module must keep working whether or not η has landed on main yet."""


def _default_entrypoint_exists() -> bool:
    return (Path(__file__).resolve().parent / _CENSUS_ENTRYPOINT_NAME).exists()


def _default_census_launcher() -> None:
    """Best-effort subprocess launch of the census entrypoint (task η)."""
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / _CENSUS_ENTRYPOINT_NAME)],
        check=False,
    )


def evaluate_census_step(
    cfg: LegibilityConfig,
    *,
    now=None,
    status_fetcher=None,
    decide=census_trigger.decide_for_project,
    entrypoint_exists=None,
    launcher=None,
) -> tuple[str, bool]:
    """Evaluate the periodic-census trigger (ζ) at the end of a nightly
    run, returning ``(one_line_decision, fire)``.

    *decide* (default ``census_trigger.decide_for_project``, never raises
    -- fail-safe) makes the FIRE/NO-FIRE call. On NO-FIRE, *launcher* is
    never called. On FIRE: if *entrypoint_exists* (default: does
    ``scripts/legibility/census.py`` -- task η -- exist) is False, this
    logs a LOUD "FIRE-WITHOUT-LAUNCH" warning and returns without calling
    *launcher* -- η is NOT a dependency of ε, so a fired trigger before η
    lands must never crash or fail the nightly run. If the entrypoint is
    present, *launcher* (default: best-effort subprocess launch) is called
    once; any launcher failure is caught and logged, never propagated --
    this function never raises and never fails the run.
    """
    if entrypoint_exists is None:
        entrypoint_exists = _default_entrypoint_exists
    if launcher is None:
        launcher = _default_census_launcher

    decision = decide(cfg.project_root, now=now, status_fetcher=status_fetcher)
    line = 'census trigger: {} -- {}'.format(
        'FIRE' if decision.fire else 'NO-FIRE', '; '.join(decision.reasons),
    )

    if not decision.fire:
        return line, False

    if not entrypoint_exists():
        logger.warning(
            'census trigger FIRED but the census entrypoint '
            '(scripts/legibility/census.py, task η) is not on main -- '
            'FIRE-WITHOUT-LAUNCH; no census started'
        )
        return line, True

    try:
        launcher()
    except Exception as exc:  # noqa: BLE001 - best-effort, never fail the run
        logger.warning('legibility trickle: census launcher failed (best-effort): %s', exc)

    return line, True


# ---------------------------------------------------------------------------
# run_nightly — full pipeline assembly: inventory -> sample -> digest ->
# code -> merge -> docs-only commit -> census (PRD §5.5). This step wires
# the happy path only; the three decision-8 fail-loud triggers (extractor
# crash, coder storm, commit failure) are layered on by later steps.
# ---------------------------------------------------------------------------

_CODEBOOK_RELPATH = Path('docs') / 'legibility' / 'confusion-codebook.yaml'

DEFAULT_PROJECTS_ROOT = Path.home() / '.claude' / 'projects'


@dataclass
class NightlyResult:
    """Outcome of one :func:`run_nightly` run.

    ``exit_code`` is the authoritative pass/fail signal for the systemd
    ExecStart / CLI caller: 0 on success (including a genuine no-change
    night), non-zero on a fail-loud trigger (PRD decision 8, layered on by
    later steps). ``applied`` is the total number of codebook mutations
    (matched sightings + applied candidates) across every coding record
    this run merged -- 0 on a dedup-only or empty-coding night, which is
    exactly what gates the dump/commit below, giving a re-run its
    idempotency for free (PRD §6.7/§8.8).
    """

    exit_code: int = 0
    commit_made: bool = False
    applied: int = 0
    coder_status: str | None = None
    census_line: str | None = None
    census_fire: bool = False
    escalated: bool = False
    reason: str | None = None


def run_nightly(
    *,
    config_path: Path | str | None = None,
    project_id: str | None = None,
    projects_root: Path | str | None = None,
    target_date: date | None = None,
    now: datetime | None = None,
    invoke=None,
    status_fetcher=None,
    poster=None,
    committer=None,
) -> NightlyResult:
    """Run one full nightly trickle pass for a single project (PRD §5.5).

    *config_path* (preferred) or *project_id* (resolved via
    :func:`resolve_config_path`) selects the project -- exactly one of the
    two is expected to identify it, mirroring the ``nightly.py run`` CLI's
    ``--config``/``--project-id`` pair. *target_date* defaults to
    yesterday UTC; *projects_root* defaults to ``~/.claude/projects``.
    *now* threads through to :func:`evaluate_census_step`'s clock seam.

    Happy-path wiring only in this step: inventory+sample -> digest ->
    code -> merge -> docs-only commit (only when the merge actually
    applied something AND the dump changed the working tree -- a
    dedup-only or empty-coding night therefore commits nothing) -> census
    trigger evaluation. This step's happy path always returns
    ``exit_code=0``; the decision-8 fail-loud triggers are added by later
    steps.
    """
    if config_path is not None:
        resolved_config_path = Path(config_path)
    else:
        if project_id is None:
            raise ValueError('run_nightly requires either config_path or project_id')
        resolved_config_path = resolve_config_path(project_id)
    cfg = load_config(resolved_config_path)

    if target_date is None:
        target_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    if projects_root is None:
        projects_root = DEFAULT_PROJECTS_ROOT
    commit_fn = committer if committer is not None else _git_commit_docs_only

    selected = select_digest_sessions(cfg, projects_root, target_date)
    digests, extractor_failures = build_digests(selected)

    if extractor_failures:
        # A digest builder raise is exceptional (build_digests already
        # isolates per-record failures, so a non-empty extractor_failures
        # means something crashed outright): fail loud (decision 8) and
        # never proceed to coder/merge/commit.
        summary = (
            f'legibility trickle extractor crashed on {len(extractor_failures)} session(s)'
        )
        detail = '; '.join(f'{session}: {reason}' for session, reason in extractor_failures)
        escalated = post_escalation(cfg, summary, detail, poster=poster)
        return NightlyResult(
            exit_code=1,
            escalated=escalated,
            reason=summary,
        )

    codebook_path = Path(cfg.project_root) / _CODEBOOK_RELPATH
    cb = codebook.load(codebook_path)

    run = coder.code_digests(
        digests, cb, project=cfg.project_id, model=cfg.models.trickle, invoke=invoke,
    )

    if run.status == 'failure':
        # >50% of digests failed to code (PRD §5.3/§6.8 storm threshold):
        # fail loud (decision 8) and skip merge/dump/commit entirely -- the
        # codebook is left untouched rather than partially applied.
        summary = f'legibility trickle coder storm: {run.failed}/{run.total} digests failed'
        detail = '; '.join(f'{session}: {reason}' for session, reason in run.failures)
        escalated = post_escalation(cfg, summary, detail, poster=poster)
        return NightlyResult(
            exit_code=1,
            coder_status=run.status,
            escalated=escalated,
            reason=summary,
        )

    applied = 0
    for record in run.records:
        cb, stats = codebook.apply_coding_record(cb, record)
        applied += stats['matched'] + stats['candidates_applied']

    validation_errors = codebook.validate(cb)
    if validation_errors:
        logger.warning(
            'legibility trickle: merged codebook failed validation, not dumped: %s',
            '; '.join(validation_errors),
        )

    commit_made = False
    if applied > 0 and not validation_errors:
        codebook.dump(cb, codebook_path)
        if _git_status_changed(cfg.project_root, _CODEBOOK_RELPATH):
            message = f'legibility: nightly trickle sightings for {target_date.isoformat()}'
            commit_result = commit_fn(cfg.project_root, [_CODEBOOK_RELPATH], message)
            commit_made = bool(commit_result.ok)

    census_line, census_fire = evaluate_census_step(cfg, now=now, status_fetcher=status_fetcher)

    return NightlyResult(
        exit_code=0,
        commit_made=commit_made,
        applied=applied,
        coder_status=run.status,
        census_line=census_line,
        census_fire=census_fire,
    )


if __name__ == '__main__':
    raise SystemExit(0)
