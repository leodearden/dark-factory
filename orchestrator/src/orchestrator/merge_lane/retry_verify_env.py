"""The failed-only merge-verify retry producer: the narrowed env a reify retry runs under.

The post-merge verify calls :func:`build_attempt0_payload` on an infra-transient
red and hands the payload to :func:`assemble_retry_verify_env`, whose result it
merges into the retry's ``MergeVerifySpec.verify_env``.
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from shared import proc_group

from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.shadow import (
    build_fail_fast_map,
    did_not_pass_subset,
    nextest_filter_ids,
    parse_nextest_list_planned,
    parse_per_test_results,
    select_failed_run_all_members,
)
from orchestrator.merge_lane.types import MergeRequest
from orchestrator.verify import VerifyResult

logger = logging.getLogger('orchestrator.merge_lane')


# ---------------------------------------------------------------------------
# Failed-only merge-verify retry PRODUCER (PRD verify-retry-failed-only D2).
#
# When ``MergeRequest.retry_failed_only`` is set (D1, task 2833), this
# orchestrator produces the retry contract that reify's verify.sh (α/β/γ)
# consumes.  These REIFY_VERIFY_RETRY_* / REIFY_RUN_ALL_MEMBER_SUBSET /
# REIFY_GUI_RETRY_SPECS env keys are BRAND NEW — this producer defines them:
#
# OWNERSHIP (task 3059).  DF owns the subset CONTENT end to end: it is derived
# from attempt-0's OWN VerifyResult (per-test verdicts + failed run_all members)
# unioned with a DF-run ``cargo nextest list`` planned probe — see
# :func:`build_attempt0_payload`.  The ONLY thing sourced from reify is the
# attempt-0 pin: ``tree_oid`` and ``profiles``, read from the sidecar reify
# actually writes at ``target/reify-verify-attempt.json`` (reify tasks
# 5287/5548 — see :data:`_REIFY_ATTEMPT_SIDECAR_RELPATH`).  There is no
# cross-repo obligation for the subset itself; a missing or stale sidecar
# simply routes the retry to a FULL verify.
#
#   REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_DEBUG    absolute path to a newline
#   REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_RELEASE  file of EXACT {did-not-pass}
#                                                   nextest ids for that cargo
#                                                   profile (one id per line).
#                                                   Files (not env values) because
#                                                   a subset can be thousands of
#                                                   ids — too large for an env.
#   REIFY_RUN_ALL_MEMBER_SUBSET   SPACE-delimited {failed run_all members}.
#   REIFY_GUI_RETRY_SPECS         SPACE-delimited {failed gui spec files}.
#
#     SPACE, not comma — reify WORD-SPLITS both values:
#       verify.sh:2579  _mk_ra_toks=(${REIFY_RUN_ALL_MEMBER_SUBSET})
#       verify.sh:2141  for _gui_retry_tok in $_gui_retry_specs
#     and the gui shell-safety allowlist is [A-Za-z0-9._/ -], which EXCLUDES
#     ','.  A comma-joined gui value is therefore rejected wholesale (loud full
#     fallback, :2156) and a comma-joined run_all value collapses into one
#     unmatchable token.
#
#     An EMPTY value is the deliberate SAFE fallback meaning "run this suite in
#     FULL", never "run nothing": verify.sh:2545 and :2127 both gate the subset
#     on the value being non-empty.
#   REIFY_VERIFY_RETRY_TREE_OID   the attempt-0-pinned content-tree OID, taken
#                                 from reify's own ``git rev-parse HEAD:`` stamp
#                                 in target/reify-verify-attempt.json and
#                                 corroborated against DF's independent
#                                 get_head_tree_hash read (INV-3); reify then
#                                 re-checks it a third time from its side.
#   REIFY_VERIFY_RETRY_SCOPE      always 'failed_only' — the retry-scope marker.
#
# The dict is merged into the dispatched MergeVerifySpec.verify_env, and the
# filter files are written into THIS host's merge_wt, the shared persistent
# merge-verify lane reify's verify.sh reads: LOCAL dispatch only (see
# verify_runner.py::MergeVerifySpec).
# ---------------------------------------------------------------------------

# Subdir under merge_wt holding the per-profile nextest filter files.
_RETRY_FILTER_SUBDIR = '.reify-verify-retry'

# DF-side MIRROR of reify's REIFY_VERIFY_RETRY_MAX_SUBSET storm-escape ceiling
# (verify.sh:1703, default 5000).  reify refuses to narrow a profile whose
# subset reaches it and runs that profile in FULL — so a subset at or above the
# ceiling is not a narrowed retry, and DF must not charge it to the larger
# narrowed budget.  Mirrored (not read from reify) because this is a DF-side
# COST decision; a drift in reify's default only makes DF conservative.
_REIFY_VERIFY_RETRY_MAX_SUBSET = 5000


def _materially_narrows(planned: Sequence[str], subset: Sequence[str]) -> bool:
    """True when ``subset`` is a REAL, reify-acceptable reduction of ``planned``.

    Three ways a produced subset fails to narrow anything, all of which make
    reify run the profile in FULL:

    * **empty** — reify's per-profile "retry refused: no subset" fallback fires
      (``verify.sh:1698``);
    * **the whole plan** — reachable whenever attempt-0 reds before any test
      line is emitted (compile-gate red, ``env_transient``,
      ``semaphore_timeout``): ``parse_per_test_results`` returns ``{}``, so
      every planned test is ``'not-started'``;
    * **at/over the ceiling** — reify rejects it wholesale
      (:data:`_REIFY_VERIFY_RETRY_MAX_SUBSET`).

    Each is SAFE (the retry runs everything), but calling any of them
    "narrowed" is a lie with a price: the caller would switch to the larger
    ``MAX_POST_MERGE_VERIFY_NARROWED_RETRIES`` budget on the premise that a
    narrowed retry is cheap, so the merge lane would pay TWO full re-verifies
    where the legacy ``max_enospc`` budget paid one.
    """
    return (
        bool(subset)
        and len(subset) < len(planned)
        and len(subset) <= _REIFY_VERIFY_RETRY_MAX_SUBSET
    )


def _build_retry_verify_env(
    *,
    nextest_subset_debug: list[str],
    nextest_subset_release: list[str],
    run_all_members: list[str],
    gui_specs: list[str],
    tree_oid: str,
    filter_dir: Path,
) -> dict[str, str]:
    """Write the per-profile nextest filter files and build the REIFY_* env dict.

    Writes ``nextest-retry-debug.filter`` / ``nextest-retry-release.filter`` (one
    exact test id per line, deterministic order preserved from the passed lists)
    into ``filter_dir/.reify-verify-retry`` and returns the env dict documented
    in the module comment above.  ``run_all_members`` and ``gui_specs`` ship as
    SPACE-delimited env values (reify word-splits both, and the gui allowlist
    excludes ','); the nextest subsets ship as file paths.

    Callers must pass nextest ids already mapped through
    :func:`nextest_filter_ids` — this function writes the lines verbatim.

    Args:
        nextest_subset_debug: {did-not-pass} nextest ids for the debug profile.
        nextest_subset_release: {did-not-pass} nextest ids for the release profile.
        run_all_members: {failed} run_all member ids (pass-through, not transformed).
        gui_specs: {failed} gui spec files (pass-through, not transformed).
        tree_oid: attempt-0-pinned content-tree OID (REIFY_VERIFY_RETRY_TREE_OID).
        filter_dir: directory the filter files are written under (the
            dispatching host's merge_wt; produced for a local dispatch only).

    Returns:
        The REIFY_VERIFY_RETRY_* env dict to merge into MergeVerifySpec.verify_env.
    """
    retry_dir = Path(filter_dir) / _RETRY_FILTER_SUBDIR
    retry_dir.mkdir(parents=True, exist_ok=True)

    debug_file = retry_dir / 'nextest-retry-debug.filter'
    release_file = retry_dir / 'nextest-retry-release.filter'
    debug_file.write_text('\n'.join(nextest_subset_debug))
    release_file.write_text('\n'.join(nextest_subset_release))

    return {
        'REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_DEBUG': str(debug_file),
        'REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_RELEASE': str(release_file),
        # SPACE-delimited: reify word-splits both, and the gui allowlist
        # excludes ',' outright.  See the module comment above.
        'REIFY_RUN_ALL_MEMBER_SUBSET': ' '.join(run_all_members),
        'REIFY_GUI_RETRY_SPECS': ' '.join(gui_specs),
        'REIFY_VERIFY_RETRY_TREE_OID': tree_oid,
        'REIFY_VERIFY_RETRY_SCOPE': 'failed_only',
    }


@dataclasses.dataclass(frozen=True)
class _Attempt0Payload:
    """The attempt-0 result the failed-only retry is constructed from.

    **DF CONSTRUCTS this** (see :func:`build_attempt0_payload`) from attempt-0's
    own ``VerifyResult.test_output`` plus a ``cargo nextest list`` planned probe.
    Only the tree-OID pin and the profile list come from reify's sidecar
    (:data:`_REIFY_ATTEMPT_SIDECAR_RELPATH`); the subset CONTENT is DF-owned end
    to end.  Passing it as an explicit parameter keeps the subset/env/gate logic
    pure and fully unit-testable.

    Fields:
        tree_oid: attempt-0-pinned content-tree OID the retry is corroborated
            against (INV-3) — reify's ``git rev-parse HEAD:`` stamp.
        profiles: the cargo profiles attempt-0 PLANNED to run, in reify's order.
        debug_planned / debug_verdicts: the debug-profile nextest plan
            (authoritative full set) and the parsed attempt-0 verdicts (RUN tests
            only) — combined via :func:`build_fail_fast_map`.
        release_planned / release_verdicts: same, for the release profile.
        run_all_members: {failed} run_all member ids (pass-through).
        gui_specs: {failed} gui spec files (pass-through).
    """

    tree_oid: str
    profiles: tuple[str, ...]
    debug_planned: list[str]
    debug_verdicts: dict[str, str]
    release_planned: list[str]
    release_verdicts: dict[str, str]
    run_all_members: list[str]
    gui_specs: list[str]


async def assemble_retry_verify_env(
    git_ops: GitOps,
    req: MergeRequest,
    merge_wt: Path,
    attempt0: _Attempt0Payload,
) -> dict[str, str] | None:
    """INV-3 tree-OID corroboration gate → the retry env, or None for full verify.

    **Two INDEPENDENT reads of the same fact** (belt-and-suspenders):

    1. DF's own ground truth — ``git_ops.get_head_tree_hash(merge_wt)``;
    2. reify's own ``git rev-parse HEAD:`` stamp, carried in
       ``attempt0.tree_oid`` from its ``target/reify-verify-attempt.json``
       sidecar (:func:`_load_reify_attempt_sidecar`).

    A disagreement (rebased tree) or an unreadable OID (None — an ABSENCE of
    corroboration, not a match) fails safe: log a WARNING and return None so the
    caller leaves the spec untouched and the existing ``_reverify_rebased_tree``
    (M4) route runs a FULL re-verify.  reify then independently re-checks the
    same pin from its own side (verify.sh's ``_RETRY_SUBSET_ELIGIBLE`` guard), so
    the tree is corroborated on both sides of the seam.

    On a match, builds the per-profile {did-not-pass} nextest subsets via
    :func:`build_fail_fast_map` → :func:`did_not_pass_subset`, then maps them
    through :func:`nextest_filter_ids` at the write boundary so the filter files
    carry nextest's ``test(=...)`` domain while the internal plan/verdict space
    stays uniform.  Failed run_all members / gui specs pass through unchanged.

    **Third gate: the subset must actually narrow something.** A subset that is
    empty, equal to the whole plan, or over reify's storm-escape ceiling makes
    reify run the profile in FULL — so returning an env dict for it would let
    the caller charge a full re-verify to the larger narrowed-retry budget.  See
    :func:`_materially_narrows`.

    Args:
        git_ops: GitOps for DF's own tree-OID read.
        req: the merge request (for ``task_id`` in log lines).
        merge_wt: the merge worktree — both the tree-OID probe target and the
            filter-file directory.
        attempt0: the DF-constructed attempt-0 payload.

    Returns:
        The REIFY_VERIFY_RETRY_* env dict on a corroborated tree whose subset
        materially narrows, else None.
    """
    current = await git_ops.get_head_tree_hash(merge_wt)
    if current is None or current != attempt0.tree_oid:
        logger.warning(
            'Task %s: retry tree OID %r does not match attempt-0 %r '
            '(rebased or unknown tree) — falling back to full verify '
            '(M4 _reverify_rebased_tree route)',
            req.task_id, current, attempt0.tree_oid,
        )
        return None

    debug_subset = did_not_pass_subset(
        build_fail_fast_map(attempt0.debug_planned, attempt0.debug_verdicts)
    )
    release_subset = did_not_pass_subset(
        build_fail_fast_map(attempt0.release_planned, attempt0.release_verdicts)
    )

    # MATERIAL-NARROWING gate: a retry that narrows NOTHING must not be called
    # narrowed.  `narrowed` is what unlocks the larger MAX_POST_MERGE_VERIFY_
    # NARROWED_RETRIES budget, on the premise that a narrowed retry re-runs only
    # the {did-not-pass} subset and is therefore cheap.  When that premise is
    # false (see :func:`_materially_narrows`) the lane would pay TWO FULL
    # re-verifies where the legacy max_enospc budget paid one — a real
    # worst-case regression for exactly the infra-transient class this branch
    # handles.  Returning None here routes the caller to the legacy budget AND
    # leaves `spec` untouched, so nothing about the retry changes except its
    # accounting.  Both profiles are checked because only the FIRST one is ever
    # populated and the payload does not name which that was.
    if not (
        _materially_narrows(attempt0.debug_planned, debug_subset)
        or _materially_narrows(attempt0.release_planned, release_subset)
    ):
        logger.warning(
            'Task %s: failed-only retry would narrow NOTHING '
            '(nextest debug %d/%d, release %d/%d; ceiling %d) — reify would run '
            'these profiles in full anyway, so this is charged to the legacy '
            'budget as a FULL retry, not the narrowed one',
            req.task_id,
            len(debug_subset), len(attempt0.debug_planned),
            len(release_subset), len(attempt0.release_planned),
            _REIFY_VERIFY_RETRY_MAX_SUBSET,
        )
        return None

    # Per-suite sizes at the narrowing site, so the cost premise behind the
    # larger narrowed budget is auditable from the log rather than inferred.
    logger.info(
        'Task %s: narrowed failed-only retry — nextest debug %d/%d, '
        'release %d/%d, run_all %d, gui %d',
        req.task_id,
        len(debug_subset), len(attempt0.debug_planned),
        len(release_subset), len(attempt0.release_planned),
        len(attempt0.run_all_members), len(attempt0.gui_specs),
    )

    # Map into nextest's matcher domain ONLY here, at the single write boundary:
    # a filter file of full "<binary-id> <test-name>" parse keys is non-empty
    # (so reify's "retry refused: no subset" fallback never fires) yet matches
    # ZERO tests — a FALSE-GREEN retry.  See :func:`nextest_filter_ids`.
    return _build_retry_verify_env(
        nextest_subset_debug=nextest_filter_ids(debug_subset),
        nextest_subset_release=nextest_filter_ids(release_subset),
        run_all_members=list(attempt0.run_all_members),
        gui_specs=list(attempt0.gui_specs),
        tree_oid=current,
        filter_dir=merge_wt,
    )


# reify's attempt-0 sidecar — the REAL one, relative to the worktree root.
#
# Producer: reify ``scripts/verify.sh`` ``add_test_passes()``, which stamps it as
# its FIRST plan line so the pin survives a RED psi-gate / compile-gate / nextest
# pole (reify task 5548).  ``verify.sh:738`` defines the path as
# ``_ATTEMPT_SIDECAR_PATH="${REIFY_VERIFY_ATTEMPT_SIDECAR:-target/reify-verify-attempt.json}"``.
#
# Schema (exactly three keys; see tests/fixtures/reify_verify_retry/ for the
# verbatim captured bytes and PROVENANCE.md)::
#
#     {"tree_oid": "<git rev-parse HEAD:>",
#      "profiles": "debug release",          # SPACE-DELIMITED STRING, not a list
#      "timestamp": "2026-07-30T04:56:41Z"}
#
# ``profiles`` records what the attempt PLANNED to run — because the stamp
# precedes the test poles, it is NEVER proof that a profile actually executed.
# That is precisely why only the FIRST profile may be narrowed (see
# :func:`build_attempt0_payload`).
#
# ``tree_oid`` is reify's own ``git rev-parse HEAD:`` — the same value DF's
# ``git_ops.get_head_tree_hash`` produces, which is what makes the INV-3
# corroboration two genuinely INDEPENDENT reads rather than one read twice.
_REIFY_ATTEMPT_SIDECAR_RELPATH = 'target/reify-verify-attempt.json'

# The cargo profiles DF can emit a per-profile nextest filter-file env key for.
# A `profiles` value naming anything outside this set aborts to a full verify:
# there is no REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_<X> key for a third
# profile, so DF could not satisfy reify's "a filter file for EVERY profile
# named in `profiles`, or fall back to a full verify" obligation
# (verify.sh:219-230), and silently ignoring it would narrow a profile whose
# attempt-0 pass never ran.
_KNOWN_REIFY_PROFILES = frozenset({'debug', 'release'})


@dataclasses.dataclass(frozen=True)
class _ReifyAttemptSidecar:
    """The parsed contents of reify's ``target/reify-verify-attempt.json``.

    Fields:
        tree_oid: reify's ``git rev-parse HEAD:`` stamp for the attempt-0 tree.
        profiles: the cargo profiles attempt-0 PLANNED to run, in the order
            reify named them (the sidecar's space-delimited ``profiles``
            string, split).  Never proof a profile executed.
    """

    tree_oid: str
    profiles: tuple[str, ...]


def _load_reify_attempt_sidecar(merge_wt: Path) -> _ReifyAttemptSidecar | None:
    """Tolerantly load reify's attempt-0 sidecar from ``merge_wt``.

    Never raises.  Returns None — routing the caller to a FULL verify — when the
    sidecar is absent, unreadable, not JSON, not an object, missing ``tree_oid``,
    carries an empty/whitespace-only ``profiles``, or names a profile outside
    :data:`_KNOWN_REIFY_PROFILES`.

    Args:
        merge_wt: the merge worktree root the sidecar path is relative to.

    Returns:
        The parsed :class:`_ReifyAttemptSidecar`, or None.
    """
    path = Path(merge_wt) / _REIFY_ATTEMPT_SIDECAR_RELPATH
    try:
        raw = path.read_text()
    except OSError:
        # An absent sidecar is an ordinary outcome (non-reify project, or a
        # verify that died before add_test_passes) — no warning, just full.
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as exc:
        logger.warning(
            'reify attempt-0 sidecar at %s is not valid JSON (%s) — full verify',
            path, exc,
        )
        return None
    if not isinstance(data, dict) or 'tree_oid' not in data:
        logger.warning(
            'reify attempt-0 sidecar at %s missing tree_oid / not an object '
            '— full verify', path,
        )
        return None

    tree_oid = str(data['tree_oid'])
    # `profiles` is a SPACE-DELIMITED STRING in reify's real bytes, not a list.
    raw_profiles = data.get('profiles')
    profiles = tuple(raw_profiles.split()) if isinstance(raw_profiles, str) else ()
    if not profiles:
        logger.warning(
            'reify attempt-0 sidecar at %s has empty/unparseable profiles %r '
            '— full verify', path, raw_profiles,
        )
        return None
    unknown = [p for p in profiles if p not in _KNOWN_REIFY_PROFILES]
    if unknown:
        logger.warning(
            'reify attempt-0 sidecar at %s names unsupported profile(s) %r '
            '(DF has no per-profile filter-file env key for them, so it cannot '
            'satisfy reify\'s every-profile obligation) — full verify',
            path, unknown,
        )
        return None
    return _ReifyAttemptSidecar(tree_oid=tree_oid, profiles=profiles)


# Bounds the planned-set probe so it can NEVER wedge the merge lane.  In a warm
# merge lane the binaries are already built, so `nextest list` links and lists in
# seconds; this ceiling only matters on a cold lane where the probe would have to
# compile — and there, timing out into a full verify is the correct outcome.
#
# That "warm ⇒ seconds" premise is only true if the probe runs under the SAME
# build environment as the verify (the shared sccache backend et al) — which is
# why `verify_env` is threaded all the way down to the spawn.  Without it the
# probe compiles cold on exactly the lanes this capability targets.
_NEXTEST_LIST_PROBE_TIMEOUT_SECS = 300

# Per-signal grace for the timeout path's group kill (SIGTERM, then SIGKILL).
# Kept short: the probe is a `nextest list`, so there is nothing to flush.
_PROBE_KILL_GRACE_SECS = 2.0

# How much of a failed probe's stderr is carried into the WARNING.  Enough to
# tell 'cargo-nextest not installed' from 'unknown flag' from 'workspace failed
# to compile' from 'lockfile out of date' (structured-facts-at-failure, see
# docs/legibility/design-invariants.md); short enough that a runaway build log
# cannot flood the orchestrator log.
_PROBE_STDERR_TAIL_CHARS = 500


def _stderr_tail(stderr: str) -> str:
    """Last :data:`_PROBE_STDERR_TAIL_CHARS` of ``stderr``, or an explicit marker."""
    tail = stderr.strip()[-_PROBE_STDERR_TAIL_CHARS:]
    return tail if tail else '<empty>'


async def _kill_probe_process_tree(
    proc: asyncio.subprocess.Process, pgid: int
) -> None:
    """Terminate the probe's whole process GROUP via the shared helper.

    ``cargo nextest list`` spawns ``rustc`` / build-script children that a bare
    ``proc.kill()`` leaves running.  Since the probe exists precisely to be
    bounded (:data:`_NEXTEST_LIST_PROBE_TIMEOUT_SECS`), leaking compiles into
    the merge lane on the timeout path would defeat the ceiling.

    Delegates to :func:`shared.proc_group.terminate_process_group` — the
    repo's single blessed SIGTERM→SIGKILL group-kill (landed on main; see also
    ``verify.py``) — rather than re-implementing it.  ``pgid`` MUST be the value
    captured immediately after the ``start_new_session=True`` spawn (where
    ``pgid == proc.pid`` by POSIX guarantee): calling ``os.getpgid(proc.pid)``
    here instead would be unsafe, because by this point the process may already
    have been reaped and its pid recycled onto an unrelated group.
    """
    await proc_group.terminate_process_group(
        proc, pgid, grace_secs=_PROBE_KILL_GRACE_SECS
    )


async def _run_probe_cmd(
    argv: list[str],
    *,
    cwd: Path,
    timeout_secs: float,
    env: Mapping[str, str] | None = None,
) -> tuple[int, str, str]:
    """Run ``argv`` in ``cwd``, returning ``(returncode, stdout, stderr)``.

    A thin, separately-patchable subprocess seam: it exists so
    :func:`_probe_nextest_planned` can be unit-tested hermetically without
    shelling out.  Raises :class:`TimeoutError` on timeout and :class:`OSError`
    (incl. :class:`FileNotFoundError`) on spawn failure — the caller converts
    both into a fail-safe ``None``.

    ``env`` carries the verify's OWN environment overrides
    (``MergeVerifySpec.verify_env`` — the shared sccache backend:
    ``RUSTC_WRAPPER``, ``SCCACHE_*``, ``CARGO_INCREMENTAL``, …) layered OVER
    ``os.environ``.  Threading it is load-bearing, not cosmetic: on a lane where
    the sccache wrapper is *why* the merge worktree is warm, a probe run without
    it compiles from scratch — burning minutes of lane CPU/disk and usually
    timing out, i.e. silently degrading every narrowed retry to a full verify
    while still paying the probe cost.  ``None`` inherits the orchestrator
    environment unchanged (the legacy behaviour, kept for direct/test callers).

    stderr is CAPTURED rather than discarded so the caller's failure WARNING can
    name the actual reason instead of a bare exit code.
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=({**os.environ, **env} if env else None),
        # Own session/process group so the timeout path can kill the whole build
        # tree — see :func:`_kill_probe_process_tree`.
        start_new_session=True,
    )
    # Capture the pgid NOW, while proc is guaranteed alive and unreaped: with
    # start_new_session=True, pgid == proc.pid by POSIX guarantee.  Reading it
    # later (os.getpgid) could hit a recycled pid and signal a stranger.
    pgid = proc.pid
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_secs
        )
    except (TimeoutError, asyncio.CancelledError):
        # Hard-bounded: a cancelled communicate() can leave proc.wait() pending
        # on still-open pipe transports, so the cleanup itself gets a ceiling.
        # Never trade a probe timeout for an unbounded await in the merge lane.
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                _kill_probe_process_tree(proc, pgid),
                timeout=_PROBE_KILL_GRACE_SECS * 3,
            )
        raise
    return (
        proc.returncode or 0,
        stdout.decode('utf-8', errors='replace'),
        stderr.decode('utf-8', errors='replace'),
    )


# Planned-set source: DECISION (task 3059).
#
# Candidate (a) — THIS: a `cargo nextest list` probe run in the warm merge lane.
# Chosen because the binaries are already built there, so it links and lists
# WITHOUT compiling; it is DF-owned (no cross-repo obligation — which is the
# whole point of this leaf); and it emits ids in exactly
# `parse_per_test_results`' key space, so `build_fail_fast_map` composes with no
# translation.
#
# Candidate (b) — the warm shadow baseline's prior full per-test map — is NOT
# VIABLE: `data/orchestrator/warm_verify_shadow.json` persists only
# `ShadowCompareState` cadence fields (merges_since_last_shadow,
# last_shadow_run_at).  The full per-test map lives in-process only
# (`_warm_capture` → `build_warm_shadow_results`) and is gone by the next merge.
#
# Candidate (c) — a reify-emitted planned list — is the phantom cross-repo
# dependency this leaf exists to DELETE.  Re-introducing it would recreate the
# exact "DONE producer ≠ wired at the seam" failure.
#
# KNOWN IMPRECISION, and why it is safe: DF cannot reproduce reify's per-pass
# selector (`--workspace` vs `$AFFECTED_ALL_FLAGS` vs `$_RELEASE_ALL_FLAGS`) or
# its heavy-exclude filterset, so this probe may return a SUPERSET of what
# attempt-0 actually planned.  A superset only adds extra 'not-started' ids to
# the retry subset — it runs MORE tests, never FEWER — and reify's
# `REIFY_VERIFY_RETRY_MAX_SUBSET` ceiling (default 5000) is the storm escape if
# it ever approaches the whole suite.
#
# THE INVARIANT: ``None`` is NEVER treated as an empty plan.  It routes to a
# FULL verify.  Narrowing on a partial plan would silently skip every test the
# probe failed to see.
async def _probe_nextest_planned(
    merge_wt: Path,
    profile: str,
    *,
    timeout_secs: float,
    verify_env: Mapping[str, str] | None = None,
) -> list[str] | None:
    """Probe the merge lane for the ``profile``'s planned nextest set.

    Totally fail-safe: any non-zero exit, empty/unparseable stdout, timeout, or
    spawn failure returns ``None`` (→ full verify), logging a WARNING that names
    the profile and the reason (including a bounded stderr tail on a non-zero
    exit, so 'cargo-nextest absent' is distinguishable from 'workspace failed to
    compile').

    Args:
        merge_wt: the warm merge worktree to run the probe in.
        profile: ``'debug'`` or ``'release'`` — selects the ``--release`` flag.
        timeout_secs: hard ceiling on the probe.
        verify_env: the verify's own env overrides (``MergeVerifySpec.verify_env``)
            so the probe models the SAME build the verify performs — notably the
            shared sccache backend that is why a warm lane is warm.  See
            :func:`_run_probe_cmd`.

    Returns:
        Planned test ids in :func:`parse_per_test_results`' key space, or None.
    """
    argv = ['cargo', 'nextest', 'list', '--workspace', '--message-format', 'json']
    if profile == 'release':
        argv.append('--release')

    try:
        rc, stdout, stderr = await _run_probe_cmd(
            argv, cwd=Path(merge_wt), timeout_secs=timeout_secs, env=verify_env
        )
    except TimeoutError:
        logger.warning(
            'nextest list probe for profile %r timed out after %ss — full verify',
            profile, timeout_secs,
        )
        return None
    except OSError as exc:
        # FileNotFoundError (no cargo on PATH) is the common shape here.
        logger.warning(
            'nextest list probe for profile %r could not run (%s) — full verify',
            profile, exc,
        )
        return None

    if rc != 0:
        logger.warning(
            'nextest list probe for profile %r exited %s — full verify '
            '(stderr tail: %s)',
            profile, rc, _stderr_tail(stderr),
        )
        return None
    if not stdout.strip():
        logger.warning(
            'nextest list probe for profile %r produced empty stdout — full verify',
            profile,
        )
        return None

    planned = parse_nextest_list_planned(stdout)
    if planned is None:
        logger.warning(
            'nextest list probe for profile %r produced unparseable stdout '
            '— full verify', profile,
        )
    return planned


async def build_attempt0_payload(
    merge_wt: Path,
    verify: VerifyResult,
    *,
    verify_env: Mapping[str, str] | None = None,
) -> _Attempt0Payload | None:
    """Build the retry payload from attempt-0's OWN result — no cross-repo handoff.

    Sources, all DF-owned except the two pins:

    * ``verify.test_output`` → per-test verdicts (:func:`parse_per_test_results`);
      the {failed} run_all members → :func:`select_failed_run_all_members`;
    * a ``cargo nextest list`` probe → the authoritative planned set;
    * reify's sidecar → ONLY ``tree_oid`` (the INV-3 pin) and ``profiles``.

    **Profile attribution — the load-bearing rule.** ``verify.test_output`` is a
    single blended blob across BOTH nextest passes, and the verdict key
    (``"<binary-id> <test-name>"``) does not carry the profile.  So a ``pass``
    observed in profile 1 is INDISTINGUISHABLE from a test that never ran in
    profile 2.  ``verify.sh:219-230`` makes "never narrow a profile that never
    ran" DF's seam obligation.  Therefore **only the FIRST profile named in the
    sidecar is narrowed**; every later profile gets an empty subset, which reify
    turns into its loud per-profile "retry refused: no subset" FULL fallback.

    Under fail-fast the red lands in the first (debug) pass, which is where the
    savings are, so the capability still fires meaningfully.

    Fail-safe throughout: a None from the sidecar loader or from the first
    profile's probe returns None, routing the caller to a FULL verify.

    Args:
        merge_wt: the merge worktree (sidecar location and probe cwd).
        verify: attempt-0's own failing :class:`VerifyResult`.
        verify_env: attempt-0's own ``MergeVerifySpec.verify_env``, threaded to
            the probe so it models the SAME build (sccache backend et al).

    Returns:
        The constructed :class:`_Attempt0Payload`, or None for a full verify.
    """
    sidecar = _load_reify_attempt_sidecar(merge_wt)
    if sidecar is None:
        return None

    # Only the FIRST profile can be narrowed — see the rule above.
    first = sidecar.profiles[0]
    planned = await _probe_nextest_planned(
        merge_wt,
        first,
        timeout_secs=_NEXTEST_LIST_PROBE_TIMEOUT_SECS,
        verify_env=verify_env,
    )
    if planned is None:
        # Never narrow on a partial plan: an unknown plan cannot distinguish
        # "this test passed" from "this test was never listed".
        return None

    verdicts = parse_per_test_results(verify.test_output)

    # PLAN-COVERAGE gate: the probe must SEE everything attempt-0 saw fail.
    #
    # `build_fail_fast_map` iterates `planned` only and documents "keys present
    # in verdicts but not in planned are ignored".  So a test attempt-0 reported
    # FAIL/TIMEOUT that the probe did not list is silently DROPPED from the
    # retry subset — the retry then never re-runs the actual failure and reports
    # PASS.  A FALSE GREEN, the worst outcome available here.
    #
    # The comment above claims the probe returns a SUPERSET of attempt-0's plan
    # (`--workspace` ⊇ reify's `-p` selectors).  That holds TODAY by coincidence
    # of reify's flags; a future `--all-features` / `--all-targets` /
    # `--cargo-profile` change on reify's side would break it with no signal.
    # This is the one form of plan-incompleteness DF can detect IN-PROCESS, so
    # the module's standing rule — never narrow on an incomplete plan — is
    # enforced here rather than assumed.
    #
    # Only nextest-shaped keys ("<binary-id> <test-name>", hence the space) are
    # comparable: `parse_per_test_results` also emits bare libtest paths, which
    # `cargo nextest list` never produces and which are not evidence of drift.
    planned_ids = set(planned)
    uncovered = sorted(
        key
        for key, verdict in verdicts.items()
        if verdict != 'pass' and ' ' in key and key not in planned_ids
    )
    if uncovered:
        logger.warning(
            'nextest list probe for profile %r did not cover %d did-not-pass '
            'id(s) attempt-0 observed (%s%s) — the probed plan is INCOMPLETE, '
            'so narrowing would silently drop the real failure; full verify',
            first, len(uncovered), ', '.join(uncovered[:5]),
            ', …' if len(uncovered) > 5 else '',
        )
        return None
    # A later profile's planned set is never even requested — it could not be
    # used, and probing it would cost a `nextest list` for nothing.
    per_profile: dict[str, tuple[list[str], dict[str, str]]] = {
        p: ([], {}) for p in _KNOWN_REIFY_PROFILES
    }
    per_profile[first] = (planned, verdicts)

    return _Attempt0Payload(
        tree_oid=sidecar.tree_oid,
        profiles=sidecar.profiles,
        debug_planned=per_profile['debug'][0],
        debug_verdicts=per_profile['debug'][1],
        release_planned=per_profile['release'][0],
        release_verdicts=per_profile['release'][1],
        run_all_members=select_failed_run_all_members(verify.test_output, verify.run_all_failed_members),
        # Deliberately empty (task 3059): no real reify gui failure log was
        # available to pin a fixture to, and authoring one from prose is the
        # drift class this leaf corrects.  verify.sh:2127-2158 treats an empty
        # value as "run the FULL gui suite" — unambiguously safe.
        gui_specs=[],
    )
