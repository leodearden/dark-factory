#!/usr/bin/env python3
"""scripts/legibility/nightly.py — nightly trickle pipeline assembly (PRD task ε).

Assembles the per-project nightly trickle: inventory+sample (β) -> digest
(α) -> code (δ) -> merge (γ) -> docs-only commit -> census trigger (ζ). See
plans/confusion-reduction-prd.md §5.5 (pipeline), §7.4 (per-project config),
decisions 7/8 (fail-loud contract, liveness probes git history never),
boundary test §8.8.

Every stage is reached behind a dependency-injection seam (``invoke`` for
the LLM, ``status_fetcher`` for the census -- defaulting to the real
MCP-backed ``get_statuses`` fetcher for ``cfg.project_root``, ``poster``
for escalation, ``committer`` for git) plus module-level functions a caller
can monkeypatch -- mirrors the established seam convention (coder.py's
``invoke`` override, census_trigger.py's injected ``status_fetcher``).
Each seam resolves ``None`` to its real implementation, so a production
entrypoint that injects nothing still gets the real thing rather than
nothing at all (task 4148). This is what the
systemd ``legibility-trickle@.service`` template runs nightly, and what
``install-trickle-timer.sh``/``check_trickle_liveness.sh`` install and probe.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import NamedTuple

# Self-bootstrap for standalone `python scripts/legibility/nightly.py` runs
# (and the systemd ExecStart, which invokes this file directly) -- must run
# BEFORE the `legibility.*` imports below, since a direct script invocation
# puts only scripts/legibility/ (not scripts/) on sys.path. Skipped under
# pytest/normal package import: __name__ is 'legibility.nightly'. Mirrors
# sampling.py/census_trigger.py's identical guard.
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legibility import (  # noqa: E402
    census_trigger,
    codebook,
    coder,
    digest,
    invariants,
    inventory,
    sampling,
    session_ledger,
    session_runner,
    trickle_state,
    unlanded,
)
from legibility.config import (  # noqa: E402
    LegibilityConfig,
    TrickleCensusCaps,
    _require_absolute,
    configure_logging,
    load_config,
)

logger = logging.getLogger('legibility.nightly')


# ---------------------------------------------------------------------------
# resolve_config / resolve_config_path — map a bare project_id to its legibility.yaml
# ---------------------------------------------------------------------------

def _default_search_roots() -> list[Path]:
    """Default search roots for :func:`resolve_config`.

    Env ``LEGIBILITY_SEARCH_ROOTS`` (``os.pathsep``-split) if set, else the
    dark-factory repo's own parent directory -- mirrors
    ``skills/factory-init/scripts/find_escalation_port.known_project_roots``'s
    sibling-repos-under-``/home/leo/src`` convention. Each returned path is a
    PARENT directory: :func:`resolve_config` globs one level down for
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
    """Resolve *project_id* to its ``docs/legibility/legibility.yaml`` path
    (:func:`resolve_config`, without the config it loaded on the way)."""
    path, _cfg = resolve_config(project_id, search_roots)
    return path


def resolve_config(
    project_id: str, search_roots: Sequence[str | Path] | None = None,
) -> tuple[Path, LegibilityConfig]:
    """Resolve *project_id* to its ``docs/legibility/legibility.yaml`` path
    and the config loaded from it.

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
                return config_path, cfg
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
    *, skip_sessions: frozenset[str] = frozenset(),
) -> list[sampling.ScoredRecord]:
    """Enumerate *target_date*'s sessions for *cfg* and score each one with
    :func:`legibility.sampling.score_session`, the loop ``sampling.main``
    uses too. A session whose first ``sessionId`` is in *skip_sessions* is
    dropped before scoring, as ``census_window.SessionBatchSource._select``
    drops a ledgered one."""
    sessions = inventory.enumerate_sessions(
        projects_root, cfg.cwd_prefixes, target_date,
        agent_transcript_roots=inventory.resolve_agent_transcript_roots(
            cfg.project_root, cfg.agent_transcript_roots
        ),
    )
    if skip_sessions:
        uncoded = [
            session for session in sessions
            if sampling.peek_session_id(session.path) not in skip_sessions
        ]
        if len(uncoded) < len(sessions):
            logger.info(
                'legibility trickle: project=%s date=%s skipped %d already-coded session(s)',
                cfg.project_id, target_date.isoformat(), len(sessions) - len(uncoded),
            )
        sessions = uncoded
    return [sampling.score_session(session) for session in sessions]


DEFAULT_MAX_DIGEST_BYTES = sampling.DEFAULT_DIGEST_MAX_BYTES
"""Per-digest soft byte cap (PRD §7.2) — an ALIAS, not a second definition.

Both the COST basis (:func:`select_digest_sessions`) and the RENDER basis
(:func:`build_digests`) read this one name, so what the sampler charged is
provably the digest that later gets produced. It binds to
:data:`sampling.DEFAULT_DIGEST_MAX_BYTES` rather than re-declaring ``15360``
because ``DEFAULT_MAX_DIGEST_BYTES`` and ``DEFAULT_DIGEST_MAX_BYTES`` are one
word-order swap apart — trivially confusable at a call site, and if they were
independent literals a drift between them would be invisible to every test
(reviewer_comprehensive, task 3268 amendment pass). The alias is kept because
this is the established public name for the nightly cap.

The remaining independent literal is :func:`legibility.digest.build_digest`'s
own ``max_bytes=15360`` default. ``digest.py`` is outside task 3268's locked
scope, so collapsing all three into a single ``digest.DEFAULT_MAX_BYTES``
(the reviewer's preferred shape, and where the §7.2 cap naturally belongs) is
left to a follow-up; ``TestDigestByteCostFn.test_module_constant_matches_the_
renderers_own_default`` pins the two against each other in the meantime so
drift fails a test instead of silently mis-charging the budget."""


def select_digest_sessions(
    cfg: LegibilityConfig, projects_root: Path | str, target_date: date,
    *, rendered: dict[tuple, str] | None = None,
    skip_sessions: frozenset[str] = frozenset(),
) -> sampling.SampleResult:
    """The budget-bounded, stratified sample of *target_date*'s sessions to
    digest -- :func:`select_scored_records` narrowed by
    ``sampling.stratified_sample``.

    Returns the WHOLE :class:`~legibility.sampling.SampleResult` (the
    selection is ``.selected``), not just the selection. That is deliberate
    and load-bearing: this function used to end in ``.selected`` and throw
    the accounting away, so a night that enumerated real signal and then
    discarded ALL of it on the byte budget (``selected == []`` with
    ``budget_skipped > 0``) was indistinguishable from a night with nothing
    to sample -- which is exactly what happened, unobserved, every night
    from 2026-07-16 to 2026-07-29 (task 2573 computed ``budget_skipped``;
    task 2581's consumer here dropped it; task 3270 carries it through).
    There is deliberately NO ``.selected``-only sibling: leaving the lossy
    accessor available is the mechanism that produced the incident.

    The byte budget (``cfg.budgets.max_daily_digest_bytes``) is charged
    against each candidate's ACTUAL rendered digest size, at exactly the
    :data:`DEFAULT_MAX_DIGEST_BYTES` cap :func:`build_digests` will render
    with -- the same constant on both sides, so ``bytes_used`` describes the
    digests this night actually produces. Charging the RAW transcript size
    instead over-charged the budget by 20x-500x (a 0.5-6.5MB session renders
    to a ~15KB digest), which made a single real session unaffordable
    against the entire nightly cap and selected NOTHING at all, every night,
    from 2026-07-16 to 2026-07-29.

    Costing renders transcripts, but only POST-dedupe, POST-``top_fraction``
    candidates -- a small number of extra renders, not one per enumerated
    session -- and each at most once (``stratified_sample`` memoizes
    ``cost_fn`` per session path). Pass *rendered* -- a caller-owned dict --
    to also carry those renders FORWARD to :func:`build_digests`, which then
    emits the already-rendered digest instead of building it a second time;
    :func:`run_nightly` does exactly that. Without it each selected session is
    rendered twice per night, and rendering is super-linear in signal-item
    count (:func:`legibility.digest._truncate_sections` re-renders the whole
    digest per trimmed item), so the sharing is worth roughly a 2x on the
    dominant cost of the job. What sharing does NOT recover: a candidate the
    budget ends up SKIPPING was still rendered once to find out what it would
    have cost. That is the irreducible price of a real byte budget rather than
    an estimate -- bounded by the candidate count, and the alternative (a
    count cap) is what PRD §5.2 point 2 rules out. A costing render
    that FAILS is charged ZERO rather than propagating — the accurate charge
    (no digest was produced) and the one that keeps the record cheap enough
    to reach :func:`build_digests`, where the same failure is reported
    through the structured ``extractor_failures`` channel (PRD decision 8)
    instead of aborting the night from the sampler. Cheap is not free of the
    budget entirely: a zero-cost record can still fall inside a reserve
    group that is unaffordable as a whole, or sit behind the greedy fill's
    strict halt.

    *skip_sessions* is forwarded to :func:`select_scored_records`, so a
    skipped session never enters the returned accounting.
    """
    scored = select_scored_records(
        cfg, projects_root, target_date, skip_sessions=skip_sessions,
    )
    return sampling.stratified_sample(
        scored, cfg,
        cost_fn=sampling.digest_byte_cost_fn(
            max_bytes=DEFAULT_MAX_DIGEST_BYTES, rendered=rendered,
        ),
    )


# ---------------------------------------------------------------------------
# build_digests — render one digest per selected session, isolating crashes
# ---------------------------------------------------------------------------


def build_digests(
    selected: Sequence[sampling.ScoredRecord],
    *,
    max_bytes: int = DEFAULT_MAX_DIGEST_BYTES,
    build=digest.build_digest,
    rendered: dict[tuple, str] | None = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Render one confusion digest per *selected* record via *build*
    (default :func:`legibility.digest.build_digest`), passing beta's already
    -authoritative ``rec.stratum`` as ``agent_class_override`` -- alpha never
    re-guesses when the caller already knows.

    *rendered* is the render cache :func:`select_digest_sessions` populated
    while COSTING these same records against the byte budget. A record whose
    digest is already in it is emitted from the cache rather than rendered a
    second time -- the cost basis and the render basis are the same constant
    by construction (:data:`DEFAULT_MAX_DIGEST_BYTES`), so the cached text is
    byte-identical to what a re-render would produce, and
    :func:`sampling.render_cache_key` makes a mismatched cap or builder a
    structural MISS rather than a wrong digest. Skipping the re-render matters
    because :func:`legibility.digest._truncate_sections` is super-linear in
    signal-item count (one item popped and the whole digest re-rendered per
    iteration): seconds to minutes per multi-MB session.

    A costing render that FAILED is deliberately absent from the cache, so it
    is re-attempted here and raises again -- which is how it reaches the
    structured report below.

    Any exception raised by *build* for a given record is isolated: it is
    captured as ``(session_basename, reason)`` in the returned
    ``extractor_failures`` list rather than propagated or fabricated into a
    placeholder digest, so a driving caller (:func:`run_nightly`) can treat
    a non-empty ``extractor_failures`` as the extractor-crash fail-loud
    trigger (PRD decision 8).
    """
    digests: list[str] = []
    extractor_failures: list[tuple[str, str]] = []
    cache: dict[tuple, str] = {} if rendered is None else rendered

    for record in selected:
        key = sampling.render_cache_key(record.path, max_bytes, build)
        cached = cache.get(key)
        if cached is not None:
            digests.append(cached)
            continue

        try:
            text = build(
                record.path, agent_class_override=record.stratum, max_bytes=max_bytes,
            )
        except Exception as exc:  # noqa: BLE001 - isolate, never propagate/fabricate
            extractor_failures.append((record.path.name, str(exc)))
            continue
        cache[key] = text
        digests.append(text)

    return digests, extractor_failures


# ---------------------------------------------------------------------------
# git status — did tonight's dump change the codebook? (The commit itself is
# unlanded.commit_only.)
# ---------------------------------------------------------------------------

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
    """Post *envelope* to *url* over the MCP streamable-HTTP transport.

    Delegates to ``census_trigger.post_mcp_envelope``, which single-sources
    the whole transport contract for this subsystem: the required
    Accept/Content-Type pair, the session handshake, and response-body
    decoding. Raises on any network/HTTP failure; :func:`post_escalation`
    wraps this best-effort.

    Why this is not a bare POST any more (task 3644): the escalation server
    is a STATEFUL streamable-HTTP server. A session-less ``tools/call`` is
    rejected at the transport layer, before the tool ever runs, with
    ``400 Bad Request`` / ``"Missing session ID"`` -- and with
    :func:`post_escalation` swallowing that best-effort, EVERY trickle
    fail-loud escalation (extractor crash, coder storm, commit failure) was
    silently dropped. ``post_mcp_envelope`` performs the
    ``initialize`` -> ``notifications/initialized`` handshake on that 400 and
    retries once, and decodes the SSE-framed reply that server then sends.

    Keeps returning None and discarding the decoded body: the caller only
    needs "did it land", and preserving the ``(url, envelope) -> None`` seam
    signature keeps every injected ``poster=`` in the test suite working.
    """
    census_trigger.post_mcp_envelope(url, envelope, timeout=10.0)


def post_escalation(
    cfg: LegibilityConfig, summary: str, detail: str, *, poster=None,
    level: int = logging.ERROR,
) -> bool:
    """Best-effort escalate_info POST for a fail-loud trigger (PRD decision
    8): extractor crash, coder storm, or commit failure.

    Also used by the exit-0 notices that are NOT fail-loud triggers, which
    pass ``level=logging.WARNING``: budget suppression, the barren streak,
    the deletion directive, and (task 4736) the CAPPED-night deferral. The
    level rule is exactly that — ERROR iff this escalation is itself the
    fail-loud trigger returning ``exit_code=1``, WARNING iff it deliberately
    leaves the exit code untouched.

    JOURNALS THE PAIR FIRST, AT *level*, BEFORE ANYTHING ELSE HAPPENS.
    That single line is the contract this function now carries: EVERY
    escalation this module posts is written to the journal here, at the
    call site's severity, before the envelope is built and before the POST
    is attempted. Two properties fall out of putting it here rather than
    at each of the six call sites, and neither survives a per-branch log
    call:

    * A future fail-loud branch CANNOT FORGET IT. The log sits on the only
      path to the POST, so a new branch inherits journal visibility by
      construction rather than by whoever adds it remembering.
    * A DOWN ESCALATION SERVER NO LONGER COSTS THE DIAGNOSIS. On
      2026-08-18 (esc-legibility-trickle-reify-3) the only copy of
      "coder storm: 6/6 digests failed ... [Errno 2] No such file or
      directory: claude" lived inside the archived escalation JSON, while
      ``journalctl --user -u legibility-trickle@reify.service`` showed
      nothing but an unexplained benign 400 and ``status=1/FAILURE``.
      Ordering the write BEFORE the POST is what makes the reason durable
      independent of whether the POST lands.

    *level* IS THE RULE, AND IT IS ABOUT THIS ESCALATION, NOT THE NIGHT:
    ERROR iff this escalation is ITSELF the fail-loud trigger returning
    ``exit_code=1``, WARNING iff it deliberately leaves the exit code
    untouched (the run may still fail LATER, for an unrelated reason). The
    four decision-8 fail-loud branches take the ERROR default and pass
    nothing, so the cannot-forget property above is untouched. The three
    exit-0 sites -- the budget-suppression door, the barren streak and the
    deletion-directive aggregate -- pass ``logging.WARNING``, because each
    deliberately leaves ``exit_code=0`` (a non-zero exit would make
    ``check_trickle_liveness.sh`` fail every night for a timer that is
    running perfectly, and an ERROR journal line is a weaker version of
    that same false alarm). Tying the level to what THIS escalation does
    to the exit code is what keeps ``journalctl -p err`` a list of ACTUAL
    trickle failures rather than a mixed feed a healthy-but-barren timer
    also appears in.

    The rule is DELIBERATELY NOT stated as "matches the unit's own systemd
    ``Result``", which reads tighter but is false in two directions -- so
    a future maintainer must not "fix" a WARNING site to match an observed
    ``Result=failed``. :func:`_escalate_barren_streak` is reached through
    :func:`_record_trickle_progress` from ``run_nightly``'s ``finally``
    block, so a night that crashes, storms or fails its commit AND has hit
    the streak posts a WARNING escalation on a unit that reports failure.
    The deletion-directive aggregate posts at WARNING before execution can
    still fall into the validation-failure or commit-failure branch below
    it and exit 1. Both WARNINGs are correct: neither escalation is why
    the night failed, and the branch that IS why logs its own ERROR here.

    *detail* is logged IN FULL -- never truncated, never ``repr()``-ed.
    ``census_trigger._bounded_repr`` is this package's precedent for
    bounding text before a log sink and is deliberately not reused:
    clipping the aggregate reason is precisely the defect this closes.
    The ``'%s -- %s'`` grammar is carried verbatim from the two call sites
    that already logged it, so the journal line shape (and any operator
    grep keyed on it) is unchanged; no prefix is added because every
    summary already begins with "legibility trickle".

    Posts an MCP ``tools/call`` JSON-RPC envelope for tool
    ``escalate_info`` to ``http://localhost:<cfg.escalation_port>/mcp``
    (via *poster*, default :func:`_default_poster`). NEVER raises: any
    failure (poster exception, network error) is logged as a warning and
    swallowed, returning False -- a down escalation server must not mask
    the underlying failure, since the caller's non-zero exit code is the
    authoritative loud signal regardless of whether this POST succeeded.
    Returns True on a successful post.
    """
    logger.log(level, '%s -- %s', summary, detail)
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


_RELATIVE_CENSUS_ARG_DETAIL = (
    "A relative one would resolve against the trickle's cwd "
    "(legibility-trickle@.service's WorkingDirectory) -- task 3269's defect."
)

_SCHEMA_DEFAULT_TRICKLE_CAPS = TrickleCensusCaps()


def _census_cap_args(caps: TrickleCensusCaps) -> list[str]:
    """census.py's cost-control flags for *caps*; a ``None`` cap omits its flag."""
    args: list[str] = []
    if caps.max_batches is not None:
        args += ['--max-batches', str(caps.max_batches)]
    if caps.max_verify_clusters is not None:
        args += ['--max-verify-clusters', str(caps.max_verify_clusters)]
    return args


def _default_census_launcher(
    project_root: str | Path,
    *,
    config_path: str | Path | None = None,
    caps: TrickleCensusCaps = _SCHEMA_DEFAULT_TRICKLE_CAPS,
) -> None:
    """Best-effort subprocess launch of the census entrypoint (task η)
    against the project named by *project_root*.

    The target is an ARGUMENT, never inherited from the process cwd (task
    3269): ``legibility-trickle@.service`` pins one ``WorkingDirectory`` for
    every ``%i`` instance, so an argv-less launch censused dark_factory for
    every project. Fixing that in the unit file was rejected -- it would move
    the invariant somewhere no test reaches. *config_path*, when given, also
    pins the EXACT ``legibility.yaml`` the trickle loaded. A relative value
    for either would resolve against that same cwd, so it is refused with
    ``ValueError`` before anything is launched.

    Captures the census exit code and, on a NON-ZERO exit, emits ONE loud
    warning (PRD decision 8: degradation never silent -- the silent-census
    incident, task 2952). census.py's main() escalates every failure once its
    config has loaded; a failure before that (a bad ``--project-root``, an
    unreadable config, an import-time crash, a signal kill) leaves only its
    stderr and this warning. Keeps ``check=False``:
    census runs AFTER the trickle's own commit work, so a census failure must
    never crash or fail the nightly run.

    The census inherits this process's environment unchanged: census.py
    opens its own pooled session runner, so each of its stages rotates over
    the whole account pool per invocation (task 6042).

    *caps* is the trickle's census bound (task 5900, task 5782's precondition
    (b)), emitted as census.py's own ``--max-batches`` /
    ``--max-verify-clusters`` flags, which remain the single cap mechanism.
    Omitted, it is the bounded ``TrickleCensusCaps`` schema default, never
    uncapped. ``None`` has one meaning here, on a caps FIELD only: the
    explicit uncapped opt-out for that one flag.
    """
    argv = [
        sys.executable,
        str(Path(__file__).resolve().parent / _CENSUS_ENTRYPOINT_NAME),
        '--project-root', _require_absolute(
            str(project_root), field_name='census project_root',
            detail=_RELATIVE_CENSUS_ARG_DETAIL,
        ),
    ]
    if config_path is not None:
        argv += ['--config', _require_absolute(
            str(config_path), field_name='census config_path',
            detail=_RELATIVE_CENSUS_ARG_DETAIL,
        )]
    argv += _census_cap_args(caps)
    result = subprocess.run(argv, check=False)
    if result.returncode != 0:
        logger.warning(
            "legibility trickle: census subprocess exited non-zero (returncode=%s); "
            "its own stderr above carries the cause -- the nightly trickle's own "
            "commit work is unaffected",
            result.returncode,
        )


def evaluate_census_step(
    cfg: LegibilityConfig,
    *,
    now=None,
    status_fetcher=None,
    decide=census_trigger.decide_for_project,
    entrypoint_exists=None,
    launcher=None,
    config_path: Path | str | None = None,
) -> tuple[str, bool]:
    """Evaluate the periodic-census trigger (ζ) at the end of a nightly
    run, returning ``(one_line_decision, fire)``.

    *decide* (default ``census_trigger.decide_for_project``) makes the
    FIRE/NO-FIRE call. On NO-FIRE, *launcher* is never called. On FIRE: if
    *entrypoint_exists* (default: does ``scripts/legibility/census.py`` --
    task η -- exist) is False, this logs a LOUD "FIRE-WITHOUT-LAUNCH"
    warning and returns without calling *launcher* -- η is NOT a dependency
    of ε, so a fired trigger before η lands must never crash or fail the
    nightly run. If the entrypoint is present, *launcher* (default:
    best-effort subprocess launch) is called once.

    The *launcher* seam's contract is ``launcher(project_root, *,
    config_path=None, caps)``, always called with the CONFIG's
    ``project_root`` AND the config's ``census.trickle_caps`` -- never
    argv-less, so the census target is never left to the process cwd (task
    3269), and the census bound is decided by the project's legibility.yaml
    here, where config maps to launch, never by the launcher's own fallback
    (task 5900). *config_path* is forwarded unchanged.

    This function never raises and never fails the run, and that guarantee is
    its OWN: both the *decide* call and the *launcher* call are guarded here,
    each degrading to one WARNING. It is deliberately not delegated to
    ``decide_for_project``'s own never-raises contract -- *decide* is an
    injected seam any caller can replace, so a promise about the default
    callee could not hold for an arbitrary one (task 4085). A failed
    evaluation returns a synthetic line keeping the
    ``census trigger: NO-FIRE -- ...`` grammar every consumer of
    ``NightlyResult.census_line`` reads, and returns before the
    entrypoint/launcher block so it can never start a census.

    The decision line is also emitted to the journal at INFO from here --
    exactly once per run, on every path, and BEFORE any launch (task 4148).
    """
    if entrypoint_exists is None:
        entrypoint_exists = _default_entrypoint_exists
    if launcher is None:
        launcher = _default_census_launcher

    try:
        decision = decide(cfg.project_root, now=now, status_fetcher=status_fetcher)
    except Exception as exc:  # noqa: BLE001 - the census trigger must never fail the run
        # The exception text is BOUNDED before it reaches either sink, and both
        # sinks are single-line: the WARNING becomes one nightly journal line,
        # and `census_line` is a one-line field on `NightlyResult`. An escaping
        # exception's message is arbitrary -- a `StatusFetchUnavailable`
        # chained from a large get_statuses payload, or a multi-line YAML /
        # pydantic error -- so it is truncated (and its newlines escaped) by
        # census_trigger's own log-hygiene helper rather than a second copy of
        # that logic here. Formatted once, so the two sinks cannot drift.
        detail = f'{type(exc).__name__}: {census_trigger._bounded_repr(str(exc))}'
        logger.warning(
            'census trigger evaluation failed (%s) -- NO-FIRE; the nightly '
            'run is unaffected', detail,
        )
        decision = None
        line = f'census trigger: NO-FIRE -- trigger evaluation failed ({detail})'
    else:
        line = 'census trigger: {} -- {}'.format(
            'FIRE' if decision.fire else 'NO-FIRE', '; '.join(decision.reasons),
        )

    # Task 4148 (amendment). Emitted HERE -- inside this function, before the
    # entrypoint/launcher block -- rather than at run_nightly's call site,
    # because on the FIRE path `launcher()` runs the census SYNCHRONOUSLY
    # (_default_census_launcher -> subprocess.run(census.py), whose stages
    # carry the 120/900/1800s timeouts in config.py's Timeouts). Logging after
    # this function returned would withhold the REASON for a running census
    # until that subprocess finished -- tens of minutes later -- and lose it
    # entirely if the unit is killed, times out, or the box reboots
    # mid-census: precisely the FIRE case this line exists to make visible.
    # An operator tailing `journalctl --user -u legibility-trickle@<project>`
    # now sees a census start together with the reason it started.
    #
    # ONE log site covering both the decided and the failed-evaluation path,
    # so the journal line and the `census_line` returned on NightlyResult can
    # never drift. Always-on INFO in the module's established style (cf. the
    # sampler summary and no-change-night lines in run_nightly); main()'s
    # config.configure_logging() is what makes INFO visible in the journal at
    # all (gap (b) of the trickle-legibility incident).
    logger.info('legibility trickle: %s', line)

    # A failed evaluation has established NOTHING, so it must never be able to
    # start a census -- hence `decision is None` returns here, BEFORE
    # `entrypoint_exists` is even consulted. That ordering is a safety
    # property rather than a style choice.
    if decision is None or not decision.fire:
        return line, False

    if not entrypoint_exists():
        logger.warning(
            'census trigger FIRED but the census entrypoint '
            '(scripts/legibility/census.py, task η) is not on main -- '
            'FIRE-WITHOUT-LAUNCH; no census started'
        )
        return line, True

    try:
        launcher(cfg.project_root, config_path=config_path, caps=cfg.census.trickle_caps)
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
    later steps). ``applied`` is the night's codebook mutation count,
    ``codebook.MergeCounts.mutations`` -- 0 on a dedup-only or empty-coding
    night, which is exactly what gates the dump/commit, giving a re-run its
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
    """Why this run went the way it did, when there is a why.

    Carries the escalation summary of whichever fail-loud trigger ended the
    run (extractor crash, coder storm, validation failure, commit failure),
    AND — on an otherwise-successful night — the fact that records were
    dropped: a deletion-directive skip keeps ``exit_code`` 0 by design, but
    is never an unremarkable night. ``None`` only when nothing notable
    happened."""

    budget_suppressed: bool = False
    """True when this night found real signal and then discarded ALL of it on
    the digest-byte budget (``not sample.selected and
    sample.budget_skipped > 0``) -- NOT a no-change night.

    A structured fact rather than a log line to scrape: the two states are
    otherwise identical in every observable a caller has (``applied == 0``,
    ``commit_made is False``, ``exit_code == 0``), which is precisely why the
    live trickle's 14 fully-suppressed nights (2026-07-16..29) went
    unnoticed. ``exit_code`` deliberately stays 0 -- see
    :func:`_report_sample_outcome`."""

    capped: bool = False
    """True when this night's coding was DEFERRED by a usage/auth cap rather
    than failing (``coder.is_cap_deferral(run)``) — task 4736.

    A structured fact rather than a log line to scrape, in the shape of
    ``budget_suppressed`` above and for the same reason: without it a
    deferred night and a genuine coder storm are indistinguishable in every
    observable a caller has beyond the exit code.

    ``exit_code`` is deliberately 0. An all-accounts-capped night is a NORMAL
    operating condition, not an incident (Leo's directive; sibling task
    4503): the systemd unit must not fail, no L0/L2 promotion should fire,
    and nothing is fabricated to fill the gap. On 2026-08-24 the absence of
    this branch turned expected weather into ``exit_code=1`` plus an
    ERROR-level escalation.

    VOCABULARY RESERVATION for task 4514. This task lands the token ``capped``
    for a run whose coding was deferred by a usage cap. 4514's fourth
    ``classify_run`` outcome must ACCOUNT for it rather than collapsing
    ``capped`` into ``failed``: the two warrant different operator responses —
    a capped night needs no action and self-clears at the reset, a failed
    night does not. ``trickle_state.py`` and ``check_trickle_progress.py`` are
    deliberately NOT touched here so 4514 owns those files uncontended (which
    is also why this night still recorded ``outcome=productive``:
    ``classify_run`` keys only on ``selected_count > 0``). What this task does
    change is the input 4514 will read — the exit code recorded into
    trickle-state.json for a capped night is now an honest 0 rather than a
    misleading 1."""

    barren_escalated: bool = False
    """True when THIS run crossed the barren-streak threshold and posted the
    streak escalation (see :func:`_record_trickle_progress`).

    Distinct from both of its neighbours, and deliberately not folded into
    either. ``escalated`` covers the decision-8 triggers (extractor crash,
    coder storm, validation failure, commit failure) plus
    :func:`_report_sample_outcome`'s per-night suppression notice.
    ``budget_suppressed`` is a PER-NIGHT fact about one run's sampler
    accounting. This one is about a WINDOW: N consecutive runs that
    digested nothing despite real signal arriving. Different predicate,
    different window, different remedy prompt."""

    rollback: unlanded.Rollback | None = None
    """What this run did with a codebook dump whose commit did not land: where
    the refused dump was quarantined and whether the checkout is back at HEAD.
    Set only on the commit-failure branch; a structured fact rather than the
    escalation prose that also carries it."""

    ledger_rows_written: int = 0
    """Coded sessions this run added to the session ledger."""

    ledger_rows_failed: int = 0
    """Coded sessions this run failed to ledger — counted, so a ledger fault
    is a structured fact and not only a journal line (the next census
    re-mines them)."""

    retry_due: tuple[date, ...] = ()
    """The backlog dates this invocation should retry next, oldest first.
    Set only by a scheduled pass that finished and whose state was
    recorded; otherwise empty."""


class _LedgerWrite(NamedTuple):
    written: int
    failed: int
    error: str | None


def _ledger_fault(
    cfg: LegibilityConfig, target_date: date, *, failed: int, error: Exception,
) -> _LedgerWrite:
    outcome = _LedgerWrite(written=0, failed=failed, error=str(error))
    logger.warning(
        'legibility trickle ledger: project=%s date=%s rows_written=%d '
        'rows_failed=%d error=%s',
        cfg.project_id, target_date.isoformat(), outcome.written,
        outcome.failed, outcome.error,
    )
    return outcome


def _trickle_run_ref(cfg: LegibilityConfig, target_date: date) -> str:
    return f'trickle-{cfg.project_id}-{target_date:%Y%m%d}'


def _ledger_coded_sessions(
    cfg: LegibilityConfig, records, target_date: date, now: datetime | None,
) -> _LedgerWrite:
    """Record every merged coding in the session ledger so the census skips
    those sessions. A ledger fault -- rows that will not build (e.g. a naive
    *now*) or a write that fails -- is logged and counted, never raised."""
    records = list(records)
    try:
        rows = session_ledger.rows_for(
            records,
            coded_by=session_ledger.CodedBy.TRICKLE,
            run_ref=_trickle_run_ref(cfg, target_date),
            instrument_version=digest.DIGEST_INSTRUMENT_VERSION,
            coded_at=now if now is not None else datetime.now(UTC),
        )
    except ValueError as exc:
        return _ledger_fault(cfg, target_date, failed=len(records), error=exc)
    try:
        written = session_ledger.record_codings(
            session_ledger.ledger_path(cfg.project_id), rows,
        )
    except session_ledger.LedgerError as exc:
        return _ledger_fault(cfg, target_date, failed=len(rows), error=exc)
    logger.info(
        'legibility trickle ledger: project=%s date=%s rows_written=%d rows_failed=%d',
        cfg.project_id, target_date.isoformat(), written, 0,
    )
    return _LedgerWrite(written=written, failed=0, error=None)


def _report_merge(
    cfg: LegibilityConfig,
    target_date: date,
    merge: codebook.MergeResult,
    invariant_slugs: Sequence[str],
) -> None:
    """Journal the night's one merge line, and warn once naming the
    ``invariant_violated`` values the merger dropped."""
    logger.info(
        'legibility trickle: project=%s date=%s merge: %s',
        cfg.project_id, target_date.isoformat(), merge.summary(),
    )
    codebook.warn_unknown_invariant_slugs(
        merge.invariant_slugs, invariant_slugs, run=_trickle_run_ref(cfg, target_date),
    )


def _report_sample_outcome(
    cfg: LegibilityConfig,
    sample: sampling.SampleResult,
    target_date: date,
    *,
    poster=None,
) -> bool:
    """Report *sample*'s accounting, loudly if the night was totally
    suppressed. Returns whether an escalation was posted.

    Always emits the operator's grep anchor at INFO -- what was enumerated,
    what was dropped and why, what the budget bought -- prefixed with the
    project id and target date, since one journal interleaves every project's
    timer. Its visibility depends on ``config.configure_logging()`` being
    called from ``main()``; the two halves are one contract.

    On ``not sample.selected and sample.budget_skipped > 0`` -- which the
    :class:`~legibility.sampling.SampleResult` conservation invariant makes
    exactly "candidates existed and were ALL discarded on the byte budget" --
    it additionally posts ONE best-effort :func:`post_escalation`,
    reusing the decision-8 ``escalate_info``/``infra_issue``/severity=info
    envelope unchanged. THAT CALL IS ALSO WHAT JOURNALS THE PAIR (task
    4511): ``post_escalation`` logs ``summary -- detail`` itself, here at
    ``logging.WARNING``, so this function no longer logs it separately and
    there is exactly one such line per night, not two.

    IT NOW REPORTS BOTH BARREN DOORS AT WARNING, BUT ESCALATES ONLY THE
    BUDGET ONE PER-NIGHT (task 3340). The sibling door -- ``not
    sample.selected and sample.budget_skipped <= 0 and
    sample.below_sampling_cut > 0``, i.e. real signal held back by the
    SAMPLING cut, which never even reached the budget phase -- was
    entirely unobserved before and now gets a WARNING naming its OWN
    remedy (``sampling.top_fraction`` / ``sampling.per_stratum_min``,
    never ``budgets.max_daily_digest_bytes``; ``SampleResult``'s docstring
    is explicit that conflating the two is wrong). It posts no per-night
    escalation: escalation for BOTH doors is owned by the edge-triggered
    barren-STREAK signal in :func:`_record_trickle_progress`, so per-night
    alarm volume does not rise.

    Two deliberate non-behaviors. It does NOT touch the exit code: a non-zero
    exit flips ``legibility-trickle@<project>.service`` to ``Result=failed``,
    so ``check_trickle_liveness.sh`` would fail every night for a timer that
    is running perfectly -- trading a silent-failure mode for a permanent
    false alarm, and burning the one signal that currently works. And it does
    not stop the run: a suppressed night has no reason to skip the census
    trigger. Partial truncation (``budget_skipped > 0`` with a non-empty
    selection) stays at INFO too -- that is the budget working as designed,
    and it is fully visible in the always-on line.

    IT REPEATS EVERY NIGHT, BY DESIGN -- know this before you tune it.
    Unlike the other decision-8 triggers (extractor crash, coder storm,
    commit failure), which fire on exceptional events, total suppression is
    caused by PERSISTENT CONFIG STATE: ``budgets.max_daily_digest_bytes``
    too small for this project's session sizes. An undersized budget nobody
    fixes therefore posts one info escalation per project per night, for as
    long as it stays unfixed, under the same synthetic
    ``task_id=legibility-trickle-<project_id>``
    (:func:`_build_escalation_arguments`).

    Repetition was chosen over de-duplicating repeats behind a state file
    for two reasons. It is self-limiting by construction -- the escalation
    stops the very first night the budget is raised, and the operator action
    that stops it is exactly the action that fixes the underlying data loss,
    so a nightly reminder is proportionate to a project silently digesting
    NOTHING. And a "only escalate when the previous run was not suppressed"
    latch has the worse failure mode for this specific incident: whoever
    misses the single first-night escalation gets silence thereafter, which
    is the precise pathology (14 indistinguishable nights, 2026-07-16..29)
    this whole task exists to end. The recurring-alarm-fatigue risk is real
    and accepted with eyes open; the WARNING line -- now emitted by
    :func:`post_escalation` on this function's behalf, at the
    ``logging.WARNING`` passed below -- is the per-night signal either way,
    and the severity stays ``info`` so this never gates anything. If the
    repeats do become noise before the budget is fixed, latch them here
    rather than downgrading the WARNING.
    """
    summary_line = sampling.format_sample_summary_line(
        sample, max_bytes=cfg.budgets.max_daily_digest_bytes,
    )
    logger.info(
        'legibility trickle sampler: project=%s date=%s %s',
        cfg.project_id, target_date.isoformat(), summary_line,
    )

    if not sample.selected and sample.budget_skipped <= 0 and sample.below_sampling_cut > 0:
        # The SIBLING barren door, unobserved before task 3340: real,
        # distinct, non-duplicate signal ranked below its stratum's cut and
        # never reached the budget phase at all. WARNING only -- no
        # per-night escalation, because the barren STREAK owns escalation
        # for this mode (see _record_trickle_progress), so per-night alarm
        # volume does not rise for a condition that, like suppression,
        # repeats every night until someone changes config.
        logger.warning(
            'legibility trickle night produced nothing: project=%s date=%s %s. '
            'Real confusion signal WAS found and then held back by the '
            'SAMPLING CUT (below_sampling_cut=%d, budget_skipped=0) -- this '
            'is NOT a no-change night, and raising '
            'budgets.max_daily_digest_bytes will not help: these records '
            'never competed for budget. Raise sampling.top_fraction or '
            'sampling.per_stratum_min instead.',
            cfg.project_id, target_date.isoformat(), summary_line,
            sample.below_sampling_cut,
        )
        return False

    if sample.selected or sample.budget_skipped <= 0:
        return False

    summary = (
        f'legibility trickle night totally suppressed by the digest byte budget: '
        f'{sample.budget_skipped} candidate(s) discarded, 0 selected'
    )
    detail = (
        f'project={cfg.project_id} date={target_date.isoformat()} {summary_line}. '
        f'Real confusion signal WAS found and then entirely discarded on '
        f'budgets.max_daily_digest_bytes='
        f'{cfg.budgets.max_daily_digest_bytes} -- this is NOT a no-change '
        f'night. Either the budget is too small for this project\'s session '
        f'sizes or the cost basis is over-charging; raise the budget or '
        f'investigate the sampler cost basis.'
    )
    return post_escalation(
        cfg, summary, detail, poster=poster, level=logging.WARNING,
    )


def _record_trickle_progress(
    cfg: LegibilityConfig,
    sample: sampling.SampleResult,
    target_date: date,
    result: NightlyResult,
    *,
    recorder=None,
    retry_recorder=None,
    poster=None,
    now: datetime | None = None,
    retry: bool,
) -> None:
    """Record WHY this night went the way it did, best-effort.

    Maps *sample*'s six conservation counters plus *result* onto
    :func:`trickle_state.record_run`, which classifies the run
    failed/productive/quiet/barren and folds it into the running barren
    and failed streaks that :mod:`check_trickle_progress` reads. This is
    the write half of the pair; ``check_trickle_liveness.sh`` still
    answers the separate "did the unit run" question and is untouched.

    CLASSIFICATION READS ``exit_code`` FIRST (task 4514). A non-zero exit
    classifies ``failed`` whatever the sampler counters say, because the
    counters answer "did signal flow IN" and cannot answer "did the
    pipeline FINISH". The counters are still recorded in full, so the
    first question stays answerable.

    THE OLD CONTRACT WAS WRONG, AND NOT AS A MISLABEL. It read: "a run
    that crashed AFTER selecting digests still classifies ``productive``".
    On 2026-08-18 that was applied to a reify run exactly as prescribed —
    six digests selected, exit 1, ``applied=0``, ``commit_made=false`` —
    and recorded ``productive``. It was a VOCABULARY HOLE: a permanently
    broken coder would sample > 0, storm, exit 1, and record
    ``productive`` / streak 0 / a FRESH ``last_productive_at`` every night
    forever, while ``check_trickle_progress.py`` printed "OK: last run was
    productive 0h ago" indefinitely. That is the 2026-07-16..29
    silent-degradation shape (task 3270) entering through a different
    door.

    THE MITIGATION IT LEANED ON WAS NOT OPERATIONALLY REAL. The old text
    claimed crashes "are already caught loudly by
    ``check_trickle_liveness.sh`` (``Result=failed``)". True as a
    statement about that script's LOGIC; false as a claim about the
    world, because NOTHING INVOKED IT — see OPERATIONS.md §"Legibility
    trickle health probe (04:30)" for the full account. What runs the
    probes now is ``legibility-trickle-health@<project>.timer`` ->
    ``scripts/legibility/check_trickle_health.py``, and the old claim is
    true again only while that timer is installed FOR THIS PROJECT. The
    one partial mitigation that predates it is
    :func:`_escalate_barren_streak`'s detail string ending "Probe on
    demand with: check_trickle_progress.py ...", which is
    DISCOVERABILITY, not level-triggering: it makes the probe
    hand-runnable, not bound.

    An unfinished night's date enters trickle_state's retry backlog
    (``scripts/legibility/trickle_state.py::fold_retry_backlog``) and is
    retried by :func:`run_scheduled_night`; operator view in OPERATIONS.md
    §"Legibility trickle health probe (04:30)". A *retry* pass records only
    its backlog fold: no streak escalation, no ``retry_due``.
    """
    recorded_at = now if now is not None else datetime.now(UTC)
    doc = _record_run_state(
        cfg.project_id, sample, target_date, result,
        recorder=recorder, retry_recorder=retry_recorder, now=recorded_at, retry=retry,
    )
    if retry:
        return
    _escalate_barren_streak(cfg, doc, target_date, result, poster=poster)
    if isinstance(doc, dict) and trickle_state.finished(
        exit_code=result.exit_code, capped=result.capped,
    ):
        result.retry_due = trickle_state.due_retries(doc, today=recorded_at.date())


def _record_run_state(
    project_id: str,
    sample: sampling.SampleResult,
    target_date: date,
    result: NightlyResult,
    *,
    recorder,
    retry_recorder,
    now: datetime | None,
    retry: bool,
) -> dict | None:
    """Record one night through *recorder* (default
    :func:`trickle_state.record_run`) and return the document it wrote, or
    ``None`` when the write failed. A *retry* pass is recorded through
    *retry_recorder* (default :func:`trickle_state.record_retry`) instead.

    BEST-EFFORT, ALWAYS: the whole call is wrapped in ``try/except`` ->
    one WARNING, swallowed — mirroring :func:`post_escalation`'s
    established contract. A state-write failure must never mask or replace
    the run's own outcome; observability must not become a new failure
    mode.
    """
    recorded_at = now if now is not None else datetime.now(UTC)
    try:
        if retry:
            record_retry = (
                retry_recorder if retry_recorder is not None
                else trickle_state.record_retry
            )
            return record_retry(
                project_id,
                target_date=target_date,
                recorded_at=recorded_at,
                exit_code=result.exit_code,
                capped=result.capped,
                commit_made=result.commit_made,
            )
        record = recorder if recorder is not None else trickle_state.record_run
        return record(
            project_id,
            target_date=target_date,
            recorded_at=recorded_at,
            exit_code=result.exit_code,
            total_records=sample.total_records,
            zero_signal_dropped=sample.zero_signal_dropped,
            dedupe_collapsed=sample.dedupe_collapsed,
            below_sampling_cut=sample.below_sampling_cut,
            budget_skipped=sample.budget_skipped,
            selected_count=len(sample.selected),
            applied=result.applied,
            commit_made=result.commit_made,
            budget_suppressed=result.budget_suppressed,
            capped=result.capped,
        )
    except Exception as exc:  # noqa: BLE001 — deliberately total
        logger.warning(
            'legibility trickle: could not record run state for project=%s '
            'date=%s (%s: %s); the run itself is unaffected, but the '
            'progress probe will read a stale streak',
            project_id, target_date.isoformat(), type(exc).__name__, exc,
        )
        return None


def _escalate_barren_streak(
    cfg: LegibilityConfig,
    doc,
    target_date: date,
    result: NightlyResult,
    *,
    poster=None,
) -> None:
    """Post ONE escalation on the run where the barren streak first REACHES
    :data:`trickle_state.DEFAULT_MAX_BARREN_RUNS`. Best-effort; mutates
    *result* in place.

    EDGE-TRIGGERED BY EXACT EQUALITY, not ``>=``. That is what makes it
    fire once per streak and re-arm only after a productive or quiet run
    resets the counter to 0. Level-triggering would re-fire every night
    from the threshold on, re-opening task 3270's alarm-fatigue debate;
    a one-shot latch would instead reproduce the worse failure mode 3270
    explicitly rejected -- whoever misses the single first escalation gets
    silence thereafter, which is the precise pathology (14
    indistinguishable nights) this whole task exists to end. Re-arming
    after a reset is the middle path, and it is exactly what
    ``_report_sample_outcome``'s own docstring invited ("If the repeats do
    become noise before the budget is fixed, latch them here").

    This is a GENUINELY DIFFERENT signal from that per-night notice, not a
    duplicate of it: different predicate (a window, not one run),
    different door coverage (BOTH barren doors, not just the budget one),
    and a different remedy prompt.

    ``result.exit_code`` IS NOT TOUCHED. A non-zero exit flips
    ``legibility-trickle@<project>.service`` to ``Result=failed``, so
    ``check_trickle_liveness.sh`` would fail every night for a timer that
    is running perfectly -- trading a silent-failure mode for a permanent
    false alarm and burning the one signal that currently works. This is
    the identical refusal :func:`_report_sample_outcome` already records,
    for the identical reason -- and it is why the :func:`post_escalation`
    call below passes ``logging.WARNING`` rather than taking the ERROR
    default: an escalation that does not itself flip the exit code must
    not show up in ``journalctl -p err``, even on a night that later fails
    for an unrelated reason (this one is posted from ``run_nightly``'s
    ``finally``, so it can accompany one). That call is ALSO the single
    site that journals
    ``summary -- detail`` for this escalation (task 4511); this function
    does not log the pair itself, so a streak night produces exactly one
    such line.

    The health probe's own non-zero exit is safe for exactly this reason,
    and only this reason: it runs in a SEPARATE unit
    (``legibility-trickle-health@<project>.service``), so it can never
    flip ``legibility-trickle@<project>.service`` to ``Result=failed`` and
    invert ``check_trickle_liveness.sh`` into the permanent false alarm
    this paragraph refuses.
    """
    if not isinstance(doc, dict):
        return
    if doc.get('outcome') != trickle_state.OUTCOME_BARREN:
        return
    if doc.get('consecutive_barren_runs') != trickle_state.DEFAULT_MAX_BARREN_RUNS:
        return

    counters = doc.get('counters') or {}
    budget_skipped = counters.get('budget_skipped') or 0
    below_sampling_cut = counters.get('below_sampling_cut') or 0

    # DOOR-SPECIFIC remedy. SampleResult's docstring (sampling.py) is
    # explicit that the two doors have different fixes: raising the byte
    # budget recovers a budget_skipped record and does nothing whatever for
    # a below_sampling_cut one. Never conflate them.
    remedies = []
    if budget_skipped > 0:
        remedies.append(
            'raise budgets.max_daily_digest_bytes (candidates competed for '
            'budget and were ALL discarded on it)'
        )
    if below_sampling_cut > 0:
        remedies.append(
            'raise sampling.top_fraction or sampling.per_stratum_min (these '
            'records ranked below their stratum cut and never competed for '
            'budget, so a bigger byte budget will not recover them)'
        )
    remedy = '; '.join(remedies) or (
        'inspect the recorded counters: no door counter is set, which should '
        'be impossible for a barren run'
    )

    summary = (
        f'legibility trickle has produced nothing for '
        f'{trickle_state.DEFAULT_MAX_BARREN_RUNS} consecutive runs'
    )
    detail = (
        f'project={cfg.project_id} date={target_date.isoformat()} '
        f'outcome=barren consecutive_barren_runs='
        f'{doc.get("consecutive_barren_runs")} '
        f'below_sampling_cut={below_sampling_cut} '
        f'budget_skipped={budget_skipped} '
        f'last_productive_at={doc.get("last_productive_at")}. '
        f'Real confusion signal reached the sampling/budget stage on each of '
        f'those runs and NOTHING was digested -- three consecutive means '
        f'persistent config state, not a run of bad luck. Remedy: {remedy}. '
        f'Probe on demand with: check_trickle_progress.py '
        f'{cfg.project_id} {trickle_state.DEFAULT_MAX_BARREN_RUNS}'
    )
    if post_escalation(
        cfg, summary, detail, poster=poster, level=logging.WARNING,
    ):
        result.barren_escalated = True


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
    recorder=None,
    retry_recorder=None,
    retry: bool = False,
) -> NightlyResult:
    """Run one full nightly trickle pass for a single project (PRD §5.5).

    *config_path* (preferred) or *project_id* (resolved via
    :func:`resolve_config`) selects the project -- exactly one of the
    two is expected to identify it, mirroring the ``nightly.py run`` CLI's
    ``--config``/``--project-id`` pair. *target_date* defaults to
    yesterday UTC; *projects_root* defaults to ``~/.claude/projects``.
    *now* threads through to :func:`evaluate_census_step`'s clock seam and
    to the run-state recorder's ``recorded_at``.

    After the config loads, every failure is recorded as a failed night,
    returned as ``exit_code=1``, and escalated once (:func:`_crashed_night`).
    A config that does not load raises. It is first recorded as a failed
    night when *project_id* selected the project (the systemd unit's
    ``--project-id %i``), and left unrecorded under *config_path*, which
    names no project to record against.

    *status_fetcher* defaults to the real MCP-backed ``get_statuses``
    fetcher for ``cfg.project_root``
    (:func:`census_trigger.default_status_fetcher`) -- inject to override.
    ``None`` means "build the real one", NOT "no fetcher": that asymmetry
    with every other seam here is what kept the census's tasks-landed
    condition dead on the production path (task 4148). A test wanting the
    fail-safe path injects a raising/empty fake instead.

    *invoke* reads the same way (task 5488): ``None`` means "open ONE
    pooled session runner for the night and code every digest through it"
    (:func:`session_runner.open_pooled_runner`, the orchestrator's own runner
    and account pool, task 6042), not "no invoker". The runner is opened
    inside the recorded ``try`` and closed on every exit path.

    *recorder* (default :func:`trickle_state.record_run`) is the run-state
    seam, alongside the existing ``invoke``/``status_fetcher``/``poster``/
    ``committer`` ones. Called exactly once per run from a ``finally``
    spanning every exit path, so the progress probe
    (``check_trickle_progress.py``) can tell a barren night from a quiet
    one — the distinction ``check_trickle_liveness.sh`` structurally
    cannot make, and the reason 2026-07-16..29 went unnoticed for 14
    nights — and, since task 4514, a FAILED night from a productive one,
    which is the distinction that let a crashed run read as productive
    indefinitely. The seam's contract is all four outcomes. It records the
    scheduled pass only; *retry_recorder* (default
    :func:`trickle_state.record_retry`, which rewrites only the backlog) is
    the same seam for a retry pass.

    *retry* marks a pass over a date the retry backlog owes, run by
    :func:`run_scheduled_night` after the scheduled pass. It skips sessions
    the session ledger already holds, never evaluates the census trigger,
    and records through *retry_recorder*.

    Happy-path wiring only in this step: inventory+sample -> digest ->
    code -> merge -> docs-only commit (only when the merge actually
    applied something AND the dump changed the working tree -- a
    dedup-only or empty-coding night therefore commits nothing) -> census
    trigger evaluation. This step's happy path always returns
    ``exit_code=0``; the decision-8 fail-loud triggers are added by later
    steps.
    """
    if target_date is None:
        target_date = (datetime.now(UTC) - timedelta(days=1)).date()
    if config_path is not None:
        resolved_config_path = Path(config_path).resolve()
        cfg = load_config(resolved_config_path)
    elif project_id is not None:
        resolved_config_path, cfg = _load_config_for_project(
            project_id, target_date,
            recorder=recorder, retry_recorder=retry_recorder, now=now, retry=retry,
        )
    else:
        raise ValueError('run_nightly requires either config_path or project_id')
    try:
        return _run_configured_nightly(
            cfg, resolved_config_path,
            projects_root=projects_root, target_date=target_date, now=now,
            invoke=invoke, status_fetcher=status_fetcher, poster=poster,
            committer=committer, recorder=recorder, retry_recorder=retry_recorder,
            retry=retry,
        )
    except Exception as exc:
        return _crashed_night(cfg, exc, poster=poster)


def _load_config_for_project(
    project_id: str,
    target_date: date,
    *,
    recorder,
    retry_recorder,
    now: datetime | None,
    retry: bool,
) -> tuple[Path, LegibilityConfig]:
    """Resolve *project_id*'s config (:func:`resolve_config`), recording a
    failed night for *project_id* before re-raising when none resolves."""
    try:
        return resolve_config(project_id)
    except Exception:
        _record_run_state(
            project_id, _nothing_sampled(), target_date, NightlyResult(exit_code=1),
            recorder=recorder, retry_recorder=retry_recorder, now=now, retry=retry,
        )
        raise


def _ledgered_sessions(cfg: LegibilityConfig) -> frozenset[str]:
    """The sessions *cfg*'s session ledger already holds; empty when it is
    absent or unreadable."""
    snapshot = session_ledger.read_ledger(session_ledger.ledger_path(cfg.project_id))
    if snapshot.state is session_ledger.LedgerState.UNREADABLE:
        logger.warning(
            'legibility trickle: session ledger unreadable, so no already-coded '
            'session is skipped this retry: %s',
            snapshot.error,
        )
    return snapshot.sessions


def _nothing_sampled() -> sampling.SampleResult:
    """The counters of a night whose sampling never returned."""
    return sampling.SampleResult(
        selected=[], per_stratum_counts={}, zero_signal_dropped=0, bytes_used=0,
    )


def _run_configured_nightly(
    cfg: LegibilityConfig,
    resolved_config_path: Path,
    *,
    projects_root: Path | str | None,
    target_date: date,
    now: datetime | None,
    invoke,
    status_fetcher,
    poster,
    committer,
    recorder,
    retry_recorder,
    retry: bool,
) -> NightlyResult:
    """:func:`run_nightly`'s pass once *cfg* has loaded."""
    # ONE recording point for all five return sites AND an unexpected
    # raise: a single try/finally cannot forget a branch, the way five
    # separate record calls could. Two load-bearing, non-obvious
    # subtleties, both deliberate:
    #
    # (1) `return result` binds the OBJECT REFERENCE before `finally`
    #     runs, so _record_trickle_progress mutating `result` in place
    #     is visible to the caller — that is how the barren-streak
    #     escalation flag reaches the returned NightlyResult.
    # (2) The pre-seeded `result` and `sample` below are what the night
    #     knows when a stage raises. The `except` arm builds the crash
    #     result ON `result` (_crashed_night), so the suppression facts and
    #     any escalation already posted survive, and that is what gets
    #     RECORDED: exit_code=1, with zero counters if sampling never
    #     returned and the real ones otherwise. Recording only on the happy
    #     path would let the progress probe read a stale streak as healthy
    #     after repeated crashes.
    sample = _nothing_sampled()
    result = NightlyResult(
        exit_code=1,
        reason='legibility trickle run crashed before completing',
    )
    runner = None
    try:
        if projects_root is None:
            projects_root = DEFAULT_PROJECTS_ROOT
        commit_fn = committer if committer is not None else unlanded.commit_only
        # Task 4148. The systemd path is `nightly.py run --project-id %i` ->
        # main() -> run_nightly, and main() never built a fetcher -- so
        # `status_fetcher=None` reached `compute_tasks_landed` on every real run
        # and hit its `status_fetcher is None` arm (census_trigger.py:735-739;
        # journal 2026-08-10/11/12), leaving condition (b) tasks-landed unable to
        # fire at all. Every OTHER seam here already resolves None to its real
        # implementation (committer -> unlanded.commit_only above, recorder ->
        # trickle_state.record_run, poster -> _default_poster); status_fetcher
        # alone resolved to nothing, and that asymmetry is how three prior repairs
        # of this same code (2953's 406 fix, 3291's baseline fix, 4085's fail-safe
        # arms) each left the wiring loop open.
        #
        # Defaulted HERE rather than in main() because `cfg.project_root` is the
        # first point in the chain where the ABSOLUTE root is known (config.py's
        # validator guarantees absolute -- exactly what the MCP wire requires
        # post-3291) while main() holds only --config/--project-id; and it is the
        # identical value `evaluate_census_step` hands to `decide`, so the fetcher
        # and the decision can never disagree about which project they are for.
        # Deliberately NOT wrapped in try/except: the construction does no I/O
        # (a resolve, an environ read, a returned closure), and every call-time
        # failure is already fail-safe inside compute_tasks_landed.
        #
        # HAZARD, for whoever writes the next run_nightly test (amendment): this
        # default is LIVE, so a test that seeds census-state.json with a valid
        # non-negative `last_census_done_count` and injects no `status_fetcher`
        # issues a REAL POST to $FUSED_MEMORY_MCP_URL (default localhost:8002) --
        # and on a dev box where fused-memory is actually up, a stale fixture
        # baseline against the real done-count can produce a genuine FIRE that
        # subprocess-launches scripts/legibility/census.py (real LLM spend, real
        # git writes). Such a test MUST fake httpx (conftest's install_fake_httpx)
        # AND stub `_default_census_launcher`; see the task-4148 block in
        # scripts/tests/test_legibility_nightly.py for the worked pattern.
        status_fetcher = (
            status_fetcher if status_fetcher is not None
            else census_trigger.default_status_fetcher(cfg.project_root)
        )

        # One render cache for the whole run: select_digest_sessions renders each
        # candidate to CHARGE it against the byte budget, and build_digests reuses
        # that render instead of paying for it twice. Rendering is super-linear in
        # signal-item count, so this is roughly a 2x on the job's dominant cost.
        rendered: dict[tuple, str] = {}
        skip_sessions = _ledgered_sessions(cfg) if retry else frozenset()
        sample = select_digest_sessions(
            cfg, projects_root, target_date, rendered=rendered, skip_sessions=skip_sessions,
        )

        # The operator's grep anchor plus the total-suppression signal, emitted on
        # EVERY run whatever the outcome. Placed BEFORE the digest stage so the
        # numbers are reported even when a later stage fails loud and returns
        # early. Its VISIBILITY depends on main()'s config.configure_logging()
        # call -- the journal's root logger defaults to WARNING, so an INFO line
        # nobody configured for is an INFO line nobody sees; the two halves are
        # one contract.
        budget_suppressed = not sample.selected and sample.budget_skipped > 0
        suppression_escalated = _report_sample_outcome(
            cfg, sample, target_date, poster=poster,
        )
        result = replace(
            result, budget_suppressed=budget_suppressed, escalated=suppression_escalated,
        )

        # Task 5488's lesson: `invoke=None` -- what main() passes -- must
        # resolve to the real pooled invoker, never to nothing. ONE runner for
        # the whole night, because cap state lives in its gate and only there:
        # a per-digest pool would re-try every account it had just capped.
        # Opened inside this `try` so even a failure to open it is recorded
        # (task 5635).
        if invoke is None:
            runner = session_runner.open_pooled_runner(
                label=f'legibility-trickle[{cfg.project_id}]',
            )
            invoke = runner.invoker(coder.TRICKLE_CODER_STAGE)

        digests, extractor_failures = build_digests(sample.selected, rendered=rendered)

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
            result = NightlyResult(
                exit_code=1,
                escalated=escalated or suppression_escalated,
                budget_suppressed=budget_suppressed,
                reason=summary,
            )
            return result

        codebook_path = Path(cfg.project_root) / _CODEBOOK_RELPATH
        try:
            cb = codebook.load(codebook_path)
        except FileNotFoundError:
            # First-ever nightly run for a project onboarded without a
            # pre-seeded codebook: start from an empty v2 document rather than
            # letting a bare FileNotFoundError escape -- this module's
            # contract is a clean fail-loud escalation or a graceful default,
            # never an uncaught traceback. codebook.dump() below (and
            # unlanded.commit_only's own never-tracked-path handling) takes
            # it from there once/if this run's coding actually applies
            # something.
            logger.info(
                'legibility trickle: no codebook at %s yet, starting from an empty v2 document',
                codebook_path,
            )
            cb = {'version': 2, 'entries': [], 'candidates': []}
        invariant_slugs = invariants.read_slugs(cfg.project_root)

        run = coder.code_digests(
            digests, cb, project=cfg.project_id, model=cfg.models.trickle,
            invariant_slugs=invariant_slugs, invoke=invoke,
        )

        if coder.is_cap_deferral(run):
            # A capped night is a DEFERRAL, not a failure (task 4736).
            # Placed immediately above the storm branch because it is a
            # refinement of it: the batch really did exceed the storm
            # threshold, but its failures are majority CAPS -- digests the
            # CLI never actually looked at. Charging those to the coder made
            # 2026-08-24 read as an infra incident for a condition ruled
            # normal (Leo's directive; sibling task 4503).
            #
            # Same skip-everything discipline as the storm branch below --
            # no merge, no dump, no commit, nothing fabricated -- but
            # exit_code 0, and the escalation journalled at WARNING rather
            # than ERROR, per post_escalation's level rule: ERROR iff the
            # escalation is itself the fail-loud trigger returning
            # exit_code=1.
            summary = (
                f'legibility trickle coder DEFERRED: {run.capped}/{run.total} '
                f'digests found no pool account with headroom'
            )
            # Joined exactly as the storm branch does, so the marker text
            # reaches BOTH the journal and the escalation detail. Without it
            # the operator reads a deferral with no stated cause.
            detail = '; '.join(f'{session}: {reason}' for session, reason in run.failures)
            escalated = post_escalation(
                cfg, summary, detail, poster=poster, level=logging.WARNING,
            )
            result = NightlyResult(
                exit_code=0,
                coder_status=run.status,
                capped=True,
                escalated=escalated or suppression_escalated,
                budget_suppressed=budget_suppressed,
                reason=summary,
            )
            return result

        if run.status == 'failure':
            # >50% of digests failed to code (PRD §5.3/§6.8 storm threshold):
            # fail loud (decision 8) and skip merge/dump/commit entirely -- the
            # codebook is left untouched rather than partially applied.
            summary = f'legibility trickle coder storm: {run.failed}/{run.total} digests failed'
            detail = '; '.join(f'{session}: {reason}' for session, reason in run.failures)
            escalated = post_escalation(cfg, summary, detail, poster=poster)
            result = NightlyResult(
                exit_code=1,
                coder_status=run.status,
                escalated=escalated or suppression_escalated,
                budget_suppressed=budget_suppressed,
                reason=summary,
            )
            return result

        merge = codebook.apply_coding_records(cb, run.records, invariant_slugs=invariant_slugs)
        cb = merge.codebook
        applied = merge.counts.mutations
        _report_merge(cfg, target_date, merge, invariant_slugs)

        # ONE escalation for the whole night, not one per record -- mirroring
        # the extractor-crash and coder-storm triggers above, which both
        # aggregate every failure into a single summary+detail. A SYSTEMIC
        # cause (a coder prompt regression, a model that starts emitting
        # `action: delete` for every digest) makes every record in the batch
        # deletion-shaped, and a per-record POST would turn that into an
        # escalation storm -- exactly the shape a human is least able to read.
        # Posted BEFORE the validation/commit fail-loud returns below so it
        # cannot be lost to an early return.
        deletion_escalated = False
        deletion_reason: str | None = None
        if merge.skipped_deletions:
            deletion_reason = (
                f'legibility trickle: {len(merge.skipped_deletions)} coding record(s) '
                f'carried a deletion directive; skipped'
            )
            # WARNING, not the ERROR default: this branch leaves
            # exit_code=0 on purpose (loud but non-fatal, the same shape
            # census.py uses for its mass-rejection signal), so it must not
            # land in `journalctl -p err`.
            deletion_escalated = post_escalation(
                cfg, deletion_reason,
                '; '.join(skipped.describe() for skipped in merge.skipped_deletions),
                poster=poster, level=logging.WARNING,
            )

        validation_errors = codebook.validate(cb)
        if validation_errors:
            # A merge that leaves the codebook failing its own validator is
            # exceptional on par with the other decision-8 triggers (extractor
            # crash / coder storm / commit failure): silently discarding real
            # coding work behind a green exit_code=0 would mask every night's
            # sightings from here on. Fail loud instead -- codebook_path is
            # left untouched (dump/commit are both skipped).
            summary = (
                f'legibility trickle merged codebook failed validation '
                f'({len(validation_errors)} error(s))'
            )
            detail = '; '.join(validation_errors)
            escalated = post_escalation(cfg, summary, detail, poster=poster)
            result = NightlyResult(
                exit_code=1,
                applied=applied,
                coder_status=run.status,
                escalated=escalated or suppression_escalated or deletion_escalated,
                budget_suppressed=budget_suppressed,
                reason=summary,
            )
            return result

        commit_made = False
        if merge.mutated:
            codebook.dump(cb, codebook_path)
            if _git_status_changed(cfg.project_root, _CODEBOOK_RELPATH):
                message = f'legibility: nightly trickle sightings for {target_date.isoformat()}'
                commit_result = commit_fn(cfg.project_root, [_CODEBOOK_RELPATH], message)
                if not commit_result.ok:
                    # Fail loud (decision 8), and never leave the refused dump
                    # in the checkout -- why, and the remedy applied, are in
                    # scripts/legibility/unlanded.py's module docstring.
                    rollback = unlanded.roll_back(
                        cfg.project_root, [_CODEBOOK_RELPATH],
                        project_id=cfg.project_id,
                        label=f'trickle-{target_date.isoformat()}',
                    )
                    summary = 'legibility trickle codebook commit failed'
                    detail = f'{commit_result.stderr}\n\n{rollback.describe()}'
                    escalated = post_escalation(cfg, summary, detail, poster=poster)
                    result = NightlyResult(
                        exit_code=1,
                        applied=applied,
                        coder_status=run.status,
                        escalated=escalated or suppression_escalated or deletion_escalated,
                        budget_suppressed=budget_suppressed,
                        reason=summary,
                        rollback=rollback,
                    )
                    return result
                commit_made = True

        if not commit_made:
            # Covers every case that reaches here without committing: an
            # empty/dedup-only coding run (applied == 0), or a merge that
            # applied but produced a byte-identical dump (git status
            # unchanged) -- both genuine no-change nights, never conflated
            # with a coding failure (PRD §6.7). A merged-but-invalid codebook
            # is NOT a no-change night -- it fails loud above instead, before
            # reaching here.
            logger.info(
                'legibility trickle: no-change night: nothing committed (applied=%d)', applied,
            )

        # Before the census step: it may launch the census synchronously, and
        # that census must skip what this night coded.
        ledger = _ledger_coded_sessions(cfg, merge.merged_records, target_date, now)
        ledger_reason = (
            f'legibility trickle: {ledger.failed} coded session(s) not ledgered, '
            f'the next census re-mines them ({ledger.error})'
            if ledger.error is not None else None
        )

        # Task 4148. The decision was computed and returned on NightlyResult
        # below, but never LOGGED -- so only the census's failure modes
        # (census_trigger's own WARNINGs, FIRE-WITHOUT-LAUNCH, 4085's
        # evaluation-failed line) ever reached the journal. With condition (b)
        # now wired, a healthy evaluation would otherwise be indistinguishable
        # from the still-dead one in
        # `journalctl --user -u legibility-trickle@dark_factory` -- the same
        # channel that diagnosed this task. Deliberately NOT logged here at
        # the call site (amendment): `evaluate_census_step` LAUNCHES the
        # census synchronously before returning, so a log line here would only
        # reach the journal after the whole census subprocess finished -- see
        # the log site inside that function for the full reasoning. Do not
        # re-add one here; that would double the line, not advance it.
        # Task 3269: the census is pinned to the config THIS run loaded -- never
        # a fresh resolution, never the process cwd. A retry pass skips it: the
        # scheduled pass already evaluated it this invocation, and a census that
        # failed did not advance census-state, so a second evaluation could
        # launch a second paid census.
        if retry:
            logger.info(
                'legibility trickle: census trigger not evaluated on a retry pass '
                '(project=%s date=%s)',
                cfg.project_id, target_date.isoformat(),
            )
            census_line, census_fire = None, False
        else:
            census_line, census_fire = evaluate_census_step(
                cfg, now=now, status_fetcher=status_fetcher,
                config_path=resolved_config_path,
            )

        result = NightlyResult(
            exit_code=0,
            commit_made=commit_made,
            applied=applied,
            coder_status=run.status,
            census_line=census_line,
            census_fire=census_fire,
            escalated=suppression_escalated or deletion_escalated,
            budget_suppressed=budget_suppressed,
            # A dropped record is never an unremarkable night, even though it
            # is deliberately not exit-worthy: when the escalation server is
            # down `post_escalation` returns False, and without this the whole
            # never-delete contract violation would survive only as a WARNING
            # line in the journal -- `escalated` False, `reason` empty, every
            # other field identical to a clean run.
            reason='; '.join(filter(None, (deletion_reason, ledger_reason))) or None,
            ledger_rows_written=ledger.written,
            ledger_rows_failed=ledger.failed,
        )
        return result
    except Exception as exc:
        result = _crashed_night(cfg, exc, poster=poster, known=result)
        return result
    finally:
        try:
            if runner is not None:
                runner.close()
        finally:
            _record_trickle_progress(
                cfg, sample, target_date, result,
                recorder=recorder, retry_recorder=retry_recorder,
                poster=poster, now=now, retry=retry,
            )


def _crashed_night(
    cfg: LegibilityConfig, exc: Exception, *, poster, known: NightlyResult | None = None,
) -> NightlyResult:
    """The fail-loud result of a pass that raised: one ERROR escalation
    carrying the full traceback, on top of *known* -- what the night had
    already established, such as a budget suppression and the escalation
    posted for it."""
    summary = f'legibility trickle run crashed: {type(exc).__name__}: {exc}'
    detail = ''.join(traceback.format_exception(exc))
    escalated = post_escalation(cfg, summary, detail, poster=poster)
    known = known if known is not None else NightlyResult()
    return replace(
        known, exit_code=1, escalated=known.escalated or escalated, reason=summary,
    )


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def run_scheduled_night(
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
    recorder=None,
    retry_recorder=None,
) -> tuple[NightlyResult, ...]:
    """The scheduled :func:`run_nightly` pass over *target_date*, then one
    ``retry=True`` pass per date its :attr:`NightlyResult.retry_due` names,
    in order. Every other argument reaches every pass unchanged. Each pass
    records itself; a raise from any pass propagates."""

    def run_pass(pass_date: date | None, *, retry: bool) -> NightlyResult:
        return run_nightly(
            config_path=config_path, project_id=project_id, projects_root=projects_root,
            target_date=pass_date, now=now, invoke=invoke, status_fetcher=status_fetcher,
            poster=poster, committer=committer, recorder=recorder,
            retry_recorder=retry_recorder, retry=retry,
        )

    scheduled = run_pass(target_date, retry=False)
    results = [scheduled]
    for retry_date in scheduled.retry_due:
        logger.info(
            'legibility trickle: retrying %s from the retry backlog', retry_date.isoformat(),
        )
        results.append(run_pass(retry_date, retry=True))
    return tuple(results)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint -- what the systemd ``legibility-trickle@.service``
    ``ExecStart`` and an operator both invoke.

    ``run`` -- run one nightly trickle pass. Exactly one of ``--config``
    (a direct path to a project's ``legibility.yaml``) or ``--project-id``
    (resolved via :func:`resolve_config_path`) is expected to identify the
    project; both are simply forwarded to :func:`run_nightly`, which
    already prefers ``config_path`` over ``project_id`` when both are
    given -- so ``--config`` takes precedence for free, no extra logic
    here. ``--projects-root`` defaults to ``None``, letting
    :func:`run_nightly`'s own default (``~/.claude/projects``) apply. An
    undated run is the scheduled night (yesterday UTC) plus any retry the
    backlog owes (:func:`run_scheduled_night`); a ``--date`` run is a single
    pass. ``--dry-run`` is accepted for forward operator visibility but not
    yet wired to any behavior change. Returns the first non-zero
    :attr:`NightlyResult.exit_code` among the passes, else 0.

    ``resolve-config <project_id>`` -- print the resolved
    ``legibility.yaml`` path (via :func:`resolve_config_path`) and return
    0, or print an error to stderr and return 1 if no project matches --
    this is what ``install-trickle-timer.sh`` shells out to.

    Configures logging FIRST, before arg parsing, so the summary line and
    every other INFO line this module emits is actually visible in the
    journal (see :func:`legibility.config.configure_logging` — without it
    root sits at WARNING under the systemd ExecStart and they are all
    dropped). Doing it here rather than at import keeps pytest/caplog and
    any library importer untouched.
    """
    configure_logging()

    parser = argparse.ArgumentParser(
        prog='nightly.py', description='Legibility nightly trickle pipeline (PRD task ε).',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    run_parser = subparsers.add_parser('run', help='Run one nightly trickle pass.')
    run_parser.add_argument('--config', default=None, help="Path to the project's legibility.yaml.")
    run_parser.add_argument(
        '--project-id', default=None, help='Project id, resolved via resolve-config.',
    )
    run_parser.add_argument(
        '--projects-root', default=None,
        help='Root dir of encoded session transcripts (default: ~/.claude/projects).',
    )
    run_parser.add_argument(
        '--date', default=None, type=_parse_date,
        help=(
            'Target date YYYY-MM-DD, run as a single pass (default: yesterday '
            'UTC, then any date the retry backlog owes).'
        ),
    )
    run_parser.add_argument(
        '--dry-run', action='store_true',
        help='Accepted for forward compatibility; not yet wired to any behavior change.',
    )

    resolve_parser = subparsers.add_parser(
        'resolve-config', help="Print a project's resolved legibility.yaml path.",
    )
    resolve_parser.add_argument('project_id')

    args = parser.parse_args(argv)

    if args.command == 'resolve-config':
        try:
            path = resolve_config_path(args.project_id)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(str(path))
        return 0

    # args.command == 'run'
    if not args.config and not args.project_id:
        parser.error('run requires --config or --project-id')

    if args.date is None:
        results = run_scheduled_night(
            config_path=args.config,
            project_id=args.project_id,
            projects_root=args.projects_root,
        )
    else:
        results = (run_nightly(
            config_path=args.config,
            project_id=args.project_id,
            projects_root=args.projects_root,
            target_date=args.date,
        ),)
    return next((result.exit_code for result in results if result.exit_code != 0), 0)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
