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

import subprocess
from pathlib import Path
from typing import Any

from shared.locking import EXTENSIONLESS_FILENAMES, FILE_EXTENSIONS

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

#: Bound on a single guard-script invocation.  Generous — the real script is a
#: pure `printf` — but finite: an unbounded call would let a hung sibling script
#: hang a scheduled job indefinitely, which is the never-answers failure of a
#: missing script with no output at all to diagnose it from.
GUARD_TIMEOUT_SECS = 30.0

Report = dict[str, Any]


class GuardUnavailable(Exception):
    """The reify guard script could not be run, or ran and failed.

    Deliberately a narrow, module-local exception rather than a reused builtin:
    callers catch it to convert "could not look" into `EXIT_UNAVAILABLE`, and a
    broader catch would swallow genuine bugs in this file into a confident-
    looking unavailable report.

    Raised — never returned as an empty vector — because an empty vector is a
    real, meaningful answer ("reify declares nothing"), and collapsing a broken
    emitter into it would manufacture a full-width DRIFT report accusing reify
    of dropping its entire allowlist.
    """


def _guard_vector(script: Path, flag: str, *, timeout: float = GUARD_TIMEOUT_SECS) -> list[str]:
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


def check_vectors(reify_repo: Path, *, timeout: float = GUARD_TIMEOUT_SECS) -> Report:
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
