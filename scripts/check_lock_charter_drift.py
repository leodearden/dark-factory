#!/usr/bin/env python3
"""Cross-repo lock-charter reconciliation — EXPLICITLY INVOKED, never collected by pytest.

WHY THIS IS A SCRIPT AND NOT A TEST.  The lock-charter predicate is mirrored
across two repositories: reify's ``scripts/lock-charter-guard.sh`` (the α copy,
bash) and dark-factory's ``shared.locking`` / ``fused_memory.middleware.
lock_charter_guard`` (the γ copies, Python).  Keeping them in agreement is real
and necessary work.  Doing it from INSIDE dark-factory's pytest suite is what
caused the 2026-07-29 incident:

  * reify widened its allowlist 36 -> 58 extensions in a single commit;
  * dark-factory's in-suite Tier-2 drift guard shells out to that script LIVE,
    so every dark-factory worktree's full suite went red simultaneously;
  * task 3118's ``post_merge_verify`` then failed on a SEVEN-FILE,
    orchestrator-only diff that could not possibly have caused it.

The rule that came out of it, and the rule this file implements: **a sibling
repo's mutable working state must not sit on this repo's merge-verify critical
path.**  A reify commit is reify's news.  It should surface here as a drift
REPORT from a run somebody asked for — an operator, a scheduled job — not as a
red gate on an unrelated dark-factory branch that has no way to fix it.

WHY THERE IS NO SKIP STATE.  The in-suite guards degrade to a pytest SKIP when
no reify checkout is discoverable, and a skip is indistinguishable from a pass
to anything reading an exit code.  That conflation is the second half of this
task's charter, and it would be self-defeating to rebuild it here.  So this
script has exactly three outcomes and no fourth:

    EXIT_OK = 0           the mirrors agree
    EXIT_DRIFT = 1        they disagree, and the report says how
    EXIT_UNAVAILABLE = 2  the question could not be answered at all

"I could not look" is reported exactly as loudly as "I looked and found drift",
because a scheduled reconciliation job whose sibling checkout has silently gone
missing is the case where a green is most dangerous: it reads as "the mirrors
agree" forever, while nothing is being compared.

WHERE THE γ VECTOR COMES FROM, and why it is not the obvious import.  The
vectors are read from ``shared.locking``, NOT from
``fused_memory.middleware.lock_charter_guard``.  Both carry the same canonical
allowlist and ``shared/tests/test_locking.py::TestFileExtensionsDriftGuard``
pins that copy independently, so the two cannot disagree about what
dark-factory declares — but only ``shared`` is importable from this script's
gate lane (``uv run --project shared``; ``shared/pyproject.toml`` declares no
fused-memory dependency).  Importing the fused-memory copy would make this file
uncollectable in the lane that lints and type-checks it.

NOTHING AT MODULE SCOPE RESOLVES A REAL CHECKOUT.  Reify discovery happens
inside `resolve`, per call.  Module-level resolution is what makes the in-suite
guards' constants evaluate at IMPORT time and therefore un-steerable by a test's
``monkeypatch``; here it would additionally mean a bare ``--help`` depended on
the ambient ``REIFY_ROOT``.

Contract pinned by ``tests/scripts/test_check_lock_charter_drift.py``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import shared.reify_checkout as reify_checkout
from shared.locking import EXTENSIONLESS_FILENAMES, FILE_EXTENSIONS, is_file_path

# --------------------------------------------------------------------------
# The three-outcome contract.  These numbers are API: a cron entry or CI lane
# wiring this in will branch on them, and `rc != 0` / `rc == 1` must never
# collide.  Do NOT add a fourth — see "WHY THERE IS NO SKIP STATE" above.
# --------------------------------------------------------------------------
EXIT_OK = 0
EXIT_DRIFT = 1
EXIT_UNAVAILABLE = 2

_STATUS_BY_EXIT = {
    EXIT_OK: 'ok',
    EXIT_DRIFT: 'drift',
    EXIT_UNAVAILABLE: 'unavailable',
}

#: reify-relative path of the emitter this script compares against.  Also the
#: MARKER that identifies a reify checkout usable by this caller — the same
#: marker fused-memory's in-suite guard binds, so one ``export REIFY_ROOT=...``
#: steers both to the same root.
GUARD_RELPATH = Path('scripts') / 'lock-charter-guard.sh'

#: Bound on every subprocess this script runs — the guard script and ``git``.
#: Generous (the real guard is a pure `printf`) but finite: an unbounded call
#: would let a hung sibling script, or a ``git`` stuck on a lock or a network
#: filesystem, hang a scheduled job indefinitely with no output to diagnose.
SUBPROCESS_TIMEOUT_SECS = 30.0

Report = dict[str, Any]


class GuardUnavailable(Exception):
    """A subprocess asking reify a question — its guard script or ``git`` — could not run, or failed.

    Deliberately a narrow, module-local exception rather than a reused builtin:
    callers catch it to convert "could not look" into `EXIT_UNAVAILABLE`, and a
    broader catch would swallow genuine bugs in this file into a confident-
    looking unavailable report.

    Raised — never returned as an empty vector — because an empty vector is a
    real, meaningful answer ("reify declares nothing"), and collapsing a broken
    emitter into it would manufacture a full-width DRIFT report accusing reify
    of dropping its entire allowlist.
    """


def _run_or_unavailable(argv: list[str], *, what: str, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run *argv* to completion and return its result, or raise `GuardUnavailable`.

    Owns only "did the process answer at all"; each caller judges its rc.  Every
    could-not-answer escape — a timeout, an ``OSError`` (no binary, permission
    bits, ENOENT), output that is not UTF-8 — becomes `GuardUnavailable`,
    because an escaping exception exits 1, which a caller reads as DRIFT: a
    false accusation against reify.  The codec is pinned rather than taken from
    the locale, so whether the same bytes are readable never depends on who ran
    this.
    """
    try:
        return subprocess.run(argv, capture_output=True, encoding='utf-8', timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise GuardUnavailable(
            f'{what} did not finish within {timeout}s — treated as unavailable '
            f'rather than waited on indefinitely'
        ) from exc
    except OSError as exc:
        raise GuardUnavailable(f'could not run {what}: {exc}') from exc
    except UnicodeDecodeError as exc:
        raise GuardUnavailable(
            f'{what} produced output that is not valid UTF-8 ({exc}) — its '
            f'answer could not be read'
        ) from exc


def _guard_vector(script: Path, flag: str, *, timeout: float = SUBPROCESS_TIMEOUT_SECS) -> list[str]:
    """Return the sorted vector emitted by reify's guard *script* for *flag*.

    LOUD on failure, by design, and the distinction matters.  The in-suite
    copies of this call (``_reify_guard_vector`` in
    ``fused-memory/tests/test_lock_charter_guard.py`` and
    ``shared/tests/test_locking.py``) hard-assert ``rc == 0`` for the same
    reason recorded there: measured 2026-08-09 on the live script, an
    unrecognised flag prints usage to stderr and exits 2, so a
    skip-on-any-non-zero would silently retire the comparison the moment reify
    renamed or dropped an emitter, leaving it green forever while enforcing
    nothing.

    Every could-not-answer path funnels into `GuardUnavailable`: a missing file,
    an ``OSError`` (no bash, permission bits, ENOENT), a non-zero rc, and a
    timeout.  Letting an ``OSError`` escape instead would exit with Python's own
    code 1 — which a caller reads as DRIFT, a false accusation against reify.
    """
    if not script.is_file():
        raise GuardUnavailable(
            f'reify guard script {script} is not a file — no cross-repo '
            f'comparison was performed'
        )
    try:
        result = subprocess.run(
            ['bash', str(script), flag],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise GuardUnavailable(
            f'reify guard script {script} {flag} did not finish within '
            f'{timeout}s — treated as unavailable rather than waited on '
            f'indefinitely'
        ) from exc
    except OSError as exc:
        raise GuardUnavailable(f'could not run {script} {flag}: {exc}') from exc

    if result.returncode != 0:
        raise GuardUnavailable(
            f'reify guard script {script} {flag} exited {result.returncode}, '
            f'expected 0 — the emitter this comparison depends on is gone, '
            f'renamed, or broken.  Check `bash {script} --help`.\n'
            f'  stdout: {result.stdout[:400]!r}\n'
            f'  stderr: {result.stderr[:400]!r}'
        )
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def _compare(reify: list[str], dark_factory: list[str]) -> dict[str, list[str]]:
    """Symmetric difference of one vector, reported in BOTH directions.

    Both halves are load-bearing.  A checker that computed only
    ``reify - dark_factory`` would look correct on a WIDENING (the 2026-07-29
    incident's direction) while being blind to a NARROWING — reify dropping an
    extension dark-factory still declares — which is the direction that makes
    dark-factory's declarations unenforceable on the α side.
    """
    left, right = set(reify), set(dark_factory)
    return {
        'only_in_reify': sorted(left - right),
        'only_in_dark_factory': sorted(right - left),
    }


#: The two vectors compared, each as (report key, guard flag, γ-side value).
#: They are STRUCTURALLY different inputs to ``is_file_path`` —
#: EXTENSIONLESS_FILENAMES exists precisely because those names would be
#: rejected by the extension test — so both are swept and neither may mask the
#: other.  Comparing only the first would report OK while `Dockerfile` silently
#: became undeclarable on one side.
_VECTORS = (
    ('extensions', '--list-extensions', FILE_EXTENSIONS),
    ('extensionless', '--list-extensionless', EXTENSIONLESS_FILENAMES),
)


def check_vectors(reify_repo: Path, *, timeout: float = SUBPROCESS_TIMEOUT_SECS) -> Report:
    """Compare reify's declared allowlist vectors against dark-factory's.

    Returns a report dict carrying ``status``, ``exit_code``, a per-vector
    ``vectors`` mapping of ``only_in_reify`` / ``only_in_dark_factory``, and a
    human-readable ``detail``.  Never raises for a reify-side problem: an
    unrunnable guard becomes an `EXIT_UNAVAILABLE` report, because a traceback
    out of a scheduled job is an outcome nobody can branch on.
    """
    script = Path(reify_repo) / GUARD_RELPATH
    vectors: dict[str, dict[str, list[str]]] = {}
    try:
        for key, flag, declared in _VECTORS:
            vectors[key] = _compare(_guard_vector(script, flag, timeout=timeout), sorted(declared))
    except GuardUnavailable as exc:
        return {
            'status': _STATUS_BY_EXIT[EXIT_UNAVAILABLE],
            'exit_code': EXIT_UNAVAILABLE,
            'vectors': {},
            'detail': (
                f'cross-repo vector comparison NOT PERFORMED: {exc}  '
                f'This is not a pass — nothing was compared.'
            ),
        }

    drifted = {k: v for k, v in vectors.items() if v['only_in_reify'] or v['only_in_dark_factory']}
    if not drifted:
        return {
            'status': _STATUS_BY_EXIT[EXIT_OK],
            'exit_code': EXIT_OK,
            'vectors': vectors,
            'detail': (
                f'α/γ lock-charter vectors agree ({len(FILE_EXTENSIONS)} extensions, '
                f'{len(EXTENSIONLESS_FILENAMES)} extensionless names) against {script}'
            ),
        }

    lines = [
        'α/γ lock-charter DRIFT detected against '
        f'{script}:',
    ]
    for key, diff in drifted.items():
        lines.append(
            f'  {key}: only in reify {diff["only_in_reify"]!r}; '
            f'only in dark-factory {diff["only_in_dark_factory"]!r}'
        )
    # WHOSE FAULT, in terms.  The 2026-07-29 incident's real cost was
    # misattribution — an orchestrator-only diff blamed for a sibling repo's
    # commit — so a reader of a scheduled run must not have to infer this.
    lines.append(
        'This is a CROSS-REPO MIRROR gap owned by a mirror task: reify and '
        'dark-factory declare different lock-charter vectors.  It is NOT a '
        'fault in any dark-factory branch under test, and no dark-factory '
        'branch can fix it by changing its own diff.'
    )
    return {
        'status': _STATUS_BY_EXIT[EXIT_DRIFT],
        'exit_code': EXIT_DRIFT,
        'vectors': vectors,
        'detail': '\n'.join(lines),
    }


def tracked_paths(
    repo_root: Path, *, timeout: float = SUBPROCESS_TIMEOUT_SECS,
) -> tuple[list[str], list[str]]:
    """``(non_gitlink_paths, gitlink_paths)`` for *repo_root*'s tracked corpus.

    ONE parse of the corpus, split by git's OWN mode column, so nothing
    downstream can disagree about what "the corpus" means.  ``git ls-files -s``
    (not plain ``ls-files``) is what makes that column available; mode
    ``160000`` is a submodule gitlink.

    Gitlinks are returned SEPARATELY rather than discarded, matching the
    rationale already recorded on ``_tracked_entries`` in the in-suite copies:
    deriving the exclusion from the mode column rather than from hardcoded
    submodule names means it keeps working the day reify renames or adds one.

    Raises `GuardUnavailable` whenever ``git`` did not answer: a non-zero rc
    (not a checkout) here, and every other could-not-answer escape — including
    a tracked path whose name is not UTF-8 — via `_run_or_unavailable`.
    Returning an empty corpus would report a vacuous, confident OK.
    """
    result = _run_or_unavailable(
        ['git', '-C', str(repo_root), 'ls-files', '-s', '-z'],
        what=f'`git ls-files` in {repo_root}',
        timeout=timeout,
    )
    if result.returncode != 0:
        raise GuardUnavailable(
            f'`git ls-files` in {repo_root} exited {result.returncode} — not a '
            f'git checkout, so the tracked corpus was never swept.\n'
            f'  stderr: {result.stderr[:400]!r}'
        )
    paths: list[str] = []
    gitlinks: list[str] = []
    for entry in result.stdout.split('\0'):
        if not entry:
            continue
        # Format: "<mode> <sha> <stage>\t<path>"
        meta, _, path = entry.partition('\t')
        if not path:
            continue
        if meta.split(' ', 1)[0] == '160000':
            gitlinks.append(path)
        else:
            paths.append(path)
    return paths, gitlinks


def check_corpus(reify_repo: Path, *, timeout: float = SUBPROCESS_TIMEOUT_SECS) -> Report:
    """Sweep reify's tracked corpus and report paths the predicate REJECTS.

    Direction: **corpus -> allowlist**.  This is the failure mode that produced
    the originating 2026-07-28 incident and the one the vector comparison
    structurally cannot catch — both repos can agree perfectly on a vector that
    is missing the same extension.  ``tests/infra/run-all-classification.
    manifest`` was a genuinely tracked file whose extension was simply absent
    from the allowlist, so `is_file_path` called it a directory and
    ``update_task`` rejected the declaration with a LockCharterViolation.

    Findings are keyed by EXTENSION, valued by the FIRST offending tracked path.
    The actionable unit is the extension — that is what gets allowlisted — and
    one witness per extension is enough to prove it is real; a reify-scale
    corpus can carry hundreds of files behind a single missing entry.

    Only DOTTED final segments can produce a finding.  A no-dot segment
    (``README``) is also rejected by the predicate, but there is no token to
    allowlist: reporting it under a key of ``''`` would tell the reader to add
    an empty extension.  Widening ``EXTENSIONLESS_FILENAMES`` is a separate,
    name-by-name judgement and stays out of this sweep.
    """
    try:
        paths, gitlinks = tracked_paths(Path(reify_repo), timeout=timeout)
    except GuardUnavailable as exc:
        return {
            'status': _STATUS_BY_EXIT[EXIT_UNAVAILABLE],
            'exit_code': EXIT_UNAVAILABLE,
            'findings': {},
            'detail': (
                f'tracked-corpus sweep NOT PERFORMED: {exc}  '
                f'This is not a pass — nothing was swept.'
            ),
        }

    findings: dict[str, str] = {}
    for path in paths:
        segment = path.rsplit('/', 1)[-1]
        if '.' not in segment or is_file_path(path):
            continue
        findings.setdefault(segment.rsplit('.', 1)[1], path)

    if not findings:
        return {
            'status': _STATUS_BY_EXIT[EXIT_OK],
            'exit_code': EXIT_OK,
            'findings': {},
            'detail': (
                f'every one of {len(paths)} tracked non-gitlink paths in '
                f'{reify_repo} classifies as a file ({len(gitlinks)} gitlink(s) '
                f'excluded by mode 160000)'
            ),
        }

    lines = [
        f'{len(findings)} tracked extension(s) in {reify_repo} are REJECTED by '
        f'is_file_path, so a declaration naming such a path is refused as a '
        f'directory:'
    ]
    lines += [f'  .{ext} — e.g. {path}' for ext, path in sorted(findings.items())]
    lines.append(
        'This is a CROSS-REPO MIRROR gap owned by a mirror task: the shared '
        'allowlist does not cover reify\'s tracked corpus.  It is NOT a fault '
        'in any dark-factory branch under test.'
    )
    return {
        'status': _STATUS_BY_EXIT[EXIT_DRIFT],
        'exit_code': EXIT_DRIFT,
        'findings': findings,
        'detail': '\n'.join(lines),
    }


#: Severity order for folding part exit codes: an UNAVAILABLE half can never be
#: masked by an OK (or even a DRIFT) half.  `max` over this ranking rather than
#: over the raw codes, so the ordering stays explicit and survives any future
#: renumbering of the constants themselves.
_SEVERITY = {EXIT_OK: 0, EXIT_DRIFT: 1, EXIT_UNAVAILABLE: 2}


def check(reify_repo: Path, *, timeout: float = SUBPROCESS_TIMEOUT_SECS) -> Report:
    """Run both halves and fold them into one report, worst outcome wins.

    Ordering is UNAVAILABLE > DRIFT > OK.  This is the same "absence is never a
    pass" invariant as the individual halves, applied to composition: a reify
    checkout can easily have a git index and no guard script, and reporting OK
    because the half that COULD run agreed would be the silent green of a pytest
    skip, one level up.  Likewise a drifting half plus an unavailable half is
    not a complete drift report — the unexamined half may hold worse.

    Both halves ALWAYS run; neither short-circuits the other.  A single
    invocation should tell the operator everything that is knowable right now,
    and a report that stopped at the first problem would need re-running to
    find the second.

    The per-part sub-reports are kept under ``parts`` so a ``--json`` consumer
    can see exactly WHICH half was unavailable rather than inferring it.
    """
    parts = {
        'vectors': check_vectors(Path(reify_repo), timeout=timeout),
        'corpus': check_corpus(Path(reify_repo), timeout=timeout),
    }
    exit_code = max((p['exit_code'] for p in parts.values()), key=lambda c: _SEVERITY[c])
    return {
        'status': _STATUS_BY_EXIT[exit_code],
        'exit_code': exit_code,
        'parts': parts,
        'detail': '\n'.join(f'[{name}] {p["detail"]}' for name, p in parts.items()),
    }


def resolve(explicit: str | None = None) -> reify_checkout.ReifyCheckout:
    """Locate the reify checkout to compare against, with its PROVENANCE.

    Precedence: an explicit ``--reify-repo`` out-ranks everything (the operator
    saying "this one, now"), and everything else is delegated VERBATIM to
    `shared.reify_checkout.resolve_reify_checkout`, which already owns the
    ``$REIFY_ROOT``-then-nearest-ancestor order.

    THREE THINGS ARE DELIBERATELY NOT DONE HERE, each of which has already been
    a bug in this lineage:

      * No local copy of the ancestor walk.  `shared.reify_checkout` exists to
        be the one implementation; the stale WIP this task discarded carried a
        private ``resolve_reify_repo``, which is how two call sites drift.
      * No re-read of ``os.environ['REIFY_ROOT']``.  The env arm belongs to the
        resolver, so the answer and its provenance are produced together — an
        ambient re-read is what lets a message say "(named by REIFY_ROOT)" about
        a root that was actually DISCOVERED, blaming the operator for a path
        they never typed (see `reify_skip_reason`'s docstring).
      * No ``Path(__file__).parents[N]`` hop.  Measured in the resolver's
        docstring: ``.worktrees/<id>`` contributes exactly two segments, so
        parents[5] is right in a worktree and parents[3] is right in a bare
        checkout — no fixed index is right in both, and this script is run from
        both.

    *start* is this module's own file, and that is what makes the answer
    independent of the CALLER's cwd: an operator in a worktree and a cron job in
    the main checkout resolve the same sibling.

    The explicit arm reports ``named_by_env=False`` because a flag-named root
    did not come from the env var — claiming otherwise would misattribute the
    choice in any downstream message.
    """
    if explicit is not None:
        return reify_checkout.ReifyCheckout(Path(explicit).expanduser().resolve(), False)
    return reify_checkout.resolve_reify_checkout(GUARD_RELPATH, start=Path(__file__))


def main(argv: list[str] | None = None) -> int:
    """Run the reconciliation and return its exit code.

    STREAM ROUTING IS PART OF THE CONTRACT.  An OK line goes to stdout; DRIFT
    and UNAVAILABLE go to stderr.  The wrapper this is meant to live in — a cron
    entry, a scheduled lane — discards stdout and surfaces stderr, so a drift
    report written to stdout would be silently thrown away by exactly the
    caller that most needs it.

    ``--json`` is the MACHINE channel and goes to stdout for EVERY status,
    including drift and unavailable: a consumer that piped ``--json`` and got
    nothing on the run it most needs to parse would have to fall back to
    scraping the human text.

    A resolution problem short-circuits before any comparison, and its message
    comes from `shared.reify_checkout.reify_skip_reason` rather than a local
    string.  That single-sourcing is what keeps the two arms distinct — the
    benign standalone-checkout MISS ("nobody has reify here") versus an
    operator's ``REIFY_ROOT`` naming a path that is not there — which two
    hand-rolled strings in two call sites reliably conflate.
    """
    parser = argparse.ArgumentParser(
        prog='check_lock_charter_drift.py',
        description=(
            'Reconcile reify and dark-factory lock-charter declarations. '
            'Exits 0 (agree), 1 (drift), or 2 (could not check). There is no '
            'skip: 2 is a LOUD "not checked", never a pass.'
        ),
    )
    parser.add_argument(
        '--reify-repo',
        default=None,
        help='reify checkout to compare against (out-ranks $REIFY_ROOT and discovery)',
    )
    parser.add_argument(
        '--json', action='store_true', help='emit the full report as JSON on stdout'
    )
    args = parser.parse_args(argv)

    checkout = resolve(args.reify_repo)
    reason = reify_checkout.reify_skip_reason(
        GUARD_RELPATH, checkout.root, named_by_env=checkout.named_by_env
    )
    if reason is not None:
        report: Report = {
            'status': _STATUS_BY_EXIT[EXIT_UNAVAILABLE],
            'exit_code': EXIT_UNAVAILABLE,
            'parts': {},
            'detail': (
                f'cross-repo lock-charter reconciliation NOT PERFORMED: {reason}.  '
                f'This is not a pass — nothing was compared.'
            ),
        }
    else:
        assert checkout.root is not None  # reify_skip_reason returned None
        report = check(checkout.root)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report['exit_code'] == EXIT_OK:
        print(report['detail'])
    else:
        print(report['detail'], file=sys.stderr)
    return report['exit_code']


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
