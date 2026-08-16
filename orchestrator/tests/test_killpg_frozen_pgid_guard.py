"""Guard test: never re-derive a pgid inside a ``killpg`` dispatch (task 845).

The invariant: a pgid handed to ``os.killpg`` MUST be a value frozen at a
``start_new_session=True`` spawn (where ``pgid == proc.pid`` by POSIX
guarantee), never re-derived at kill time via ``os.getpgid(proc.pid)``.

Why: once a process has been reaped, the kernel is free to recycle its pid.
``os.getpgid`` on a recycled pid returns the *new* owner's group — which, in
the incidents that produced task 845, resolved to the user ``systemd --user``
manager's group, so the ``killpg`` killed the operator's entire login session.
``shared/src/shared/proc_group.py``'s module docstring ("Why the caller
captures pgid") is the canonical statement of the contract; task 3884 was the
leaf that closed the last two ``killpg(os.getpgid(...))`` sites in the repo
(``deterministic_runner._terminate_process_tree`` and this package's
``test_warm_lane_bash_suite._run_bash_suite``).

Allowlist-free, following ``test_raw_semaphore_access_guard.py``: the
requirement is ZERO sites, not a frozen residual, so any hit at all is a
violation.

SCOPE -- what "in the repo" means here, stated precisely because this guard is
unusual among its siblings in claiming a REPO-WIDE rather than package-local
invariant. The sweep covers every git-tracked ``.py`` file, unioned with every
``.py`` under the package/script roots (so untracked work-in-progress inside a
known package is covered too), and ``test_sweep_covers_every_tracked_python_file``
holds that claim honest. Deliberately OUT of scope: sibling checkout trees under
``.eval-worktrees/``, ``.claude/worktrees/`` and ``.worktrees/``. Those hold
*other revisions of this same repo*, many predating the fix -- ~30 files under
them still carry the literal unsafe idiom -- so sweeping them would make this
guard permanently red in the main checkout over code that is not this tree's.
Both file sources exclude them structurally rather than by allowlist.

The detector matches the COMPOSITION — a ``getpgid`` call nested inside a
``killpg`` call's arguments — rather than any ``os.getpgid`` call. That is
deliberate: ``git_ops.py``'s ``pgid = os.getpgid(pid)`` renders a "kernel FLOCK
holders: ..." diagnostics string and never signals anything, so it is correct
as written and is excluded *structurally*, leaving no allowlist entry to rot.

AST-based (not text/regex) so docstring and comment mentions of the unsafe
idiom are never mistaken for a real call site. This polarity is load-bearing
and has live in-repo witnesses: both ``shared/src/shared/proc_group.py``'s
module docstring and ``df_pytest_isolation.py``'s ``_run_isolated`` docstring
discuss ``killpg(os.getpgid(...))`` in prose today, and
``deterministic_runner.py``'s own docstring carried the literal string until
task 3884 removed it. A regex guard would fail on all three.
"""
from __future__ import annotations

import ast
import functools
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_SKIP_PARTS = {'.worktrees', '.venv', 'node_modules', '__pycache__'}

# Minimum number of .py files the sweep must actually scan. Measured at 1593
# unique files, so the floor sits ~8x below reality: low enough not to become
# brittle as packages move, high enough that a filter regression trips it.
#
# This is the crude COLLAPSE detector only -- a raw-count floor cannot see a
# partial hole (1516 of 1593 files once sailed past it with an entire tree
# unguarded). `test_sweep_covers_every_tracked_python_file` owns real
# completeness; this stays because it needs no git and still trips on a total
# collapse, e.g. the absolute-vs-relative `.worktrees` filter bug.
_MIN_SCANNED_FILES = 200


def _called_name(node: ast.Call) -> str | None:
    """Return the bare function name a :class:`ast.Call` resolves to.

    Handles both the attribute spelling (``os.killpg(...)`` -> ``'killpg'``)
    and the bare-name spelling produced by ``from os import killpg``
    (``killpg(...)`` -> ``'killpg'``).
    """
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _killpg_over_getpgid(source: str) -> list[tuple[int, str]]:
    """Return ``(lineno, description)`` for each ``killpg(... getpgid(...) ...)``
    composition in *source*.

    A hit is an :class:`ast.Call` resolving to the name ``killpg`` that has,
    anywhere within its own ``args``/``keywords``, a nested :class:`ast.Call`
    resolving to the name ``getpgid``. Only the composition is flagged — a bare
    ``pgid = os.getpgid(pid)`` (diagnostics, never signalled) is not a hit.

    AST-based, returning ``[]`` on a :class:`SyntaxError` so a stray
    non-Python-3 file in the tree cannot break the guard.
    """
    # Token prefilter before the (expensive) parse.  This is EXACTLY as strong
    # as the detector, not an approximation: every hit below is an ast.Call
    # whose func is an ast.Attribute with ``.attr == 'killpg'`` or an ast.Name
    # with ``.id == 'killpg'``, and both spellings require the literal token
    # `killpg` to appear in the source text.  A dynamic spelling that avoids
    # the token (``getattr(os, 'kill' + 'pg')``) is invisible to the AST walk
    # too, so nothing detectable is lost.  Worth it: ~97% of the repo's 1593
    # swept files never mention killpg, and skipping ast.parse on them takes
    # the ratchet from ~13.5s to ~0.5s.
    if 'killpg' not in source:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _called_name(node) != 'killpg':
            continue
        # Walk only this call's ARGUMENTS, not the whole subtree from the
        # enclosing statement, so an unrelated sibling getpgid cannot pin a
        # false positive on the killpg.
        for arg in [*node.args, *(kw.value for kw in node.keywords)]:
            if any(
                isinstance(inner, ast.Call) and _called_name(inner) == 'getpgid'
                for inner in ast.walk(arg)
            ):
                hits.append((node.lineno, 'killpg(<...>getpgid(...))'))
                break
    return hits


# ---------------------------------------------------------------------------
# _killpg_over_getpgid(source) -- inline-fixture unit tests, both polarities so
# the guard cannot silently stop detecting.
# ---------------------------------------------------------------------------


def test_flags_attribute_spelling() -> None:
    """The exact idiom task 845 was caused by."""
    source = (
        "def cleanup(proc):\n"
        "    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)\n"
    )
    assert _killpg_over_getpgid(source) == [(2, 'killpg(<...>getpgid(...))')]


def test_flags_bare_name_spelling() -> None:
    """``from os import killpg, getpgid`` must not launder the violation."""
    source = (
        "def cleanup(pid, sig):\n"
        "    killpg(getpgid(pid), sig)\n"
    )
    assert _killpg_over_getpgid(source) == [(2, 'killpg(<...>getpgid(...))')]


def test_does_not_flag_frozen_pgid() -> None:
    """The blessed form: a pgid captured at spawn, passed straight through."""
    source = (
        "def cleanup(pgid):\n"
        "    os.killpg(pgid, signal.SIGKILL)\n"
    )
    assert _killpg_over_getpgid(source) == []


def test_does_not_flag_bare_getpgid_read() -> None:
    """git_ops.py's diagnostics read: a getpgid that never reaches a killpg is
    correct as written and must be excluded structurally, not by allowlist."""
    source = (
        "def describe(pid):\n"
        "    pgid = os.getpgid(pid)\n"
        "    return f'kernel FLOCK holders: {pgid}'\n"
    )
    assert _killpg_over_getpgid(source) == []


def test_does_not_flag_docstring_or_comment_mention() -> None:
    """Prose discussing the unsafe idiom is NOT a call site -- proves the
    detector is AST-based, not text/regex. Three real in-repo files (this one,
    shared/proc_group.py, df_pytest_isolation.py) depend on this polarity."""
    source = (
        '"""Never write os.killpg(os.getpgid(proc.pid), signal.SIGKILL)."""\n'
        "# os.killpg(os.getpgid(pid), sig)  <- the task-845 footgun, do not copy\n"
        "SAFE_FORM = 'os.killpg(pgid, signal.SIGKILL)'\n"
    )
    assert _killpg_over_getpgid(source) == []


# ---------------------------------------------------------------------------
# Tree-scan: the actual ratchet.
# ---------------------------------------------------------------------------


@functools.cache
def _git_tracked_python() -> tuple[Path, ...]:
    """Every .py file git tracks in this checkout, or ``()`` if git is unavailable.

    ``git ls-files`` is the authoritative answer to "what is actually in this
    repo": it is complete by construction (no hand-maintained root list to grow
    holes) and it excludes every generated, ephemeral or sibling-checkout tree
    without an allowlist, because none of them are tracked.

    NUL-delimited (``-z``) rather than newline-delimited on purpose: a path
    containing a space or a newline would silently truncate the tracked set,
    and a truncated ground truth makes the completeness test below *vacuously*
    pass -- the exact failure mode it exists to prevent.

    Cached (and returning an immutable tuple, so a caller cannot corrupt the
    shared value) because four tests in this module consume the sweep and the
    subprocess would otherwise be re-run once per test.
    """
    try:
        out = subprocess.run(
            ['git', '-C', str(REPO_ROOT), 'ls-files', '-z', '--', '*.py'],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ()
    return tuple(REPO_ROOT / p for p in out.split('\0') if p)


def _root_glob_python() -> list[Path]:
    """Every .py file under the repo's package/script roots.

    The second source of the sweep, and the one that covers a brand-new .py an
    implementer has written but not yet ``git add``ed -- work git cannot see
    yet, but that lives inside a known package root.

    The skip decision is made on the path RELATIVE to REPO_ROOT, never on the
    absolute path. Every orchestrator-dispatched task runs in a worktree rooted
    at ``<repo>/.worktrees/<id>/``, so REPO_ROOT's own absolute path contains
    the substring ``.worktrees`` and an absolute-path filter would match EVERY
    file -- scanning nothing while still reporting green.
    """
    roots = [
        *sorted(REPO_ROOT.glob('*/src/*')),
        *sorted(REPO_ROOT.glob('*/tests')),
        REPO_ROOT / 'scripts',
        REPO_ROOT / 'hooks',
    ]
    files: list[Path] = [
        # Repo-root modules (conftest.py, df_pytest_isolation.py) are swept
        # non-recursively: df_pytest_isolation.py is the one root module that
        # actually does process-group killing, so it belongs in scope.
        p for p in sorted(REPO_ROOT.glob('*.py'))
    ]
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob('*.py')):
            if _SKIP_PARTS & set(path.relative_to(REPO_ROOT).parts):
                continue
            files.append(path)
    return files


@functools.cache
def _python_files() -> tuple[Path, ...]:
    """The sweep's file set: git-tracked UNION package roots, deduped.

    Both sources are load-bearing, and neither alone is sufficient:

    * git-tracked alone would miss a brand-new .py written but not yet
      ``git add``ed. The roots keep covering in-progress work.
    * the roots alone are what produced this guard's coverage hole, and they
      are unmaintainable: the obvious widening (``REPO_ROOT/'tests'`` plus a
      ``*/scripts`` glob) STILL misses
      ``skills/factory-init/scripts/find_escalation_port.py``, because
      ``*/scripts`` matches at depth 2 and that path is depth 3. A
      hand-maintained root list keeps growing new holes; ``git ls-files`` is
      complete by construction.

    ``is_file()`` is required, not defensive: ``git ls-files`` lists INDEX
    entries, so a tracked-but-deleted file would raise FileNotFoundError in the
    ratchet's ``read_text`` and turn a coverage win into a crash. The ``set``
    also fixes the roots' mutual overlap (``*/tests`` matches ``scripts/tests``,
    which ``REPO_ROOT/'scripts'`` rglobs into as well).

    *** Do NOT "simplify" this into a whole-repo rglob from REPO_ROOT. *** The
    main checkout carries ``.eval-worktrees/`` and ``.claude/worktrees/`` --
    full copies of THIS repo at older, pre-fix revisions -- and ~30 files under
    them still contain the literal ``os.killpg(os.getpgid(proc.pid),
    signal.SIGKILL)``. A whole-repo sweep would go permanently red in the main
    checkout on code belonging to other revisions. Both sources here are
    structurally immune: git never lists those trees (untracked/ignored) and
    the root globs never reach them. Re-verify that immunity before adding any
    third source.

    Cached, returning an immutable tuple: four tests in this module consume the
    sweep, and without this the ``git ls-files`` subprocess plus the full rglob
    (~0.25s) would run once per test. The cache is session-scoped, which is
    correct here -- the tree does not change mid-run.
    """
    candidates = [*_git_tracked_python(), *_root_glob_python()]
    return tuple(sorted({p for p in candidates if p.is_file()}))


def test_sweep_is_not_vacuous() -> None:
    """The sweep must actually scan the tree.

    Load-bearing, not stylistic: a zero-file sweep passes the ratchet below
    green, so without this the guard could look healthy while checking nothing
    -- and it would look healthy specifically inside task worktrees, i.e.
    everywhere branches are verified.
    """
    scanned = _python_files()
    relative = {str(p.relative_to(REPO_ROOT)) for p in scanned}

    # Witnesses reachable from `_root_glob_python()` alone, so these hold
    # unconditionally -- including on a machine with no git.
    for required in (
        'orchestrator/src/orchestrator/deterministic_runner.py',
        'orchestrator/tests/test_killpg_frozen_pgid_guard.py',
        'shared/src/shared/proc_group.py',
    ):
        assert required in relative, (
            f'{required} missing from the sweep -- the root globs or the skip '
            f'filter regressed (skip must be applied to the RELATIVE path)'
        )

    # One witness per tree the original hand-maintained root globs missed
    # entirely, pinning that specific hole shut. The top-level tests/scripts
    # tree in particular holds the repo's highest density of live
    # process-group killing outside the two sites task 3884 closed.
    #
    # These are asserted only when the git source is LIVE, because none of the
    # three is reachable from `_root_glob_python()` (whose roots are `*/src/*`,
    # `*/tests`, `scripts`, `hooks` and root-level `*.py` -- top-level `tests/`,
    # `fused-memory/scripts/` and `skills/**` match none of them); they enter
    # the sweep solely via `git ls-files`. Asserting them unconditionally would
    # mean that on a git-less machine this test hard-fails pointing at "the
    # root globs or the skip filter", sending a debugger at the wrong
    # subsystem, while the test that actually owns completeness
    # (`test_sweep_covers_every_tracked_python_file`) skips cleanly.
    if _git_tracked_python():
        for required in (
            'tests/scripts/test_drain_process_leak_isolation.py',
            'fused-memory/scripts/audit_duplicate_memories.py',
            'skills/factory-init/scripts/find_escalation_port.py',
        ):
            assert required in relative, (
                f'{required} is git-tracked but missing from the sweep -- the '
                f'git ls-files source regressed or its output is being '
                f'filtered. This is one of the trees the original root globs '
                f'missed entirely; do not "fix" this by deleting the witness.'
            )

    assert len(scanned) >= _MIN_SCANNED_FILES, (
        f'sweep scanned only {len(scanned)} files, expected >= '
        f'{_MIN_SCANNED_FILES} -- the guard is vacuous. Most likely the skip '
        f'filter is being applied to the absolute path, which contains '
        f'".worktrees" for every task worktree.'
    )


def test_sweep_covers_every_tracked_python_file() -> None:
    """Every .py file git tracks must be inside the sweep.

    This is the *completeness* assertion, and it is a different instrument from
    ``test_sweep_is_not_vacuous`` above. A raw-count floor is structurally
    incapable of detecting a PARTIAL hole: a sweep can miss an entire tree and
    still sail past the floor by three orders of magnitude. Only a set
    difference against an independently-derived ground truth catches that, and
    ``git ls-files`` is the right ground truth -- it is exactly "the content
    that can land on main".
    """
    tracked = {p for p in _git_tracked_python() if p.is_file()}
    if not tracked:
        pytest.skip('git unavailable -- completeness cannot be established')

    missing = sorted(
        str(p.relative_to(REPO_ROOT)) for p in tracked - set(_python_files())
    )
    if missing:
        shown = '\n  '.join(missing[:20])
        more = f'\n  ... and {len(missing) - 20} more' if len(missing) > 20 else ''
        raise AssertionError(
            f'{len(missing)} git-tracked .py file(s) are OUTSIDE this guard\'s '
            f'sweep, i.e. UNGUARDED: someone can re-introduce '
            f'`os.killpg(os.getpgid(p.pid), SIGKILL)` in any of them and ship '
            f'green, which defeats the ratchet entirely.\n\n'
            f'Fix by WIDENING `_python_files()` so it covers them -- do NOT '
            f'trim this assertion or add an exclusion. The sweep is meant to '
            f'be complete by construction; a hand-maintained root list is the '
            f'mechanism that produced this hole in the first place.\n\n'
            f'Unguarded files:\n  {shown}{more}'
        )


def test_sweep_yields_no_duplicates() -> None:
    """The sweep must visit each file exactly once.

    The roots OVERLAP (``*/tests`` matches ``scripts/tests``, which
    ``REPO_ROOT/'scripts'`` also rglobs into; symmetrically ``*/scripts`` would
    match ``tests/scripts`` under a ``REPO_ROOT/'tests'`` root), so an
    append-without-dedupe loop double-counts whole trees. That inflates the
    number the vacuity floor is compared against, re-reads and re-parses every
    affected file, and reports any real offender twice in the failure message.
    """
    scanned = _python_files()
    assert len(scanned) == len(set(scanned)), (
        f'sweep visited {len(scanned)} paths but only '
        f'{len(set(scanned))} are distinct -- the file sources overlap and are '
        f'not deduped. Collect them into a set before returning.'
    )


def test_no_killpg_over_getpgid_in_repo() -> None:
    """No file in the repo may re-derive a pgid inside a killpg dispatch."""
    offenders: list[str] = []

    for py_file in _python_files():
        try:
            source = py_file.read_text(encoding='utf-8')
        except (UnicodeDecodeError, OSError):
            # The sweep now reaches script trees never parsed before. One
            # undecodable or unreadable file must not take the whole ratchet
            # down -- same posture as the detector's SyntaxError -> [].
            continue
        for lineno, description in _killpg_over_getpgid(source):
            offenders.append(
                f'{py_file.relative_to(REPO_ROOT)}:{lineno}: {description}'
            )

    if offenders:
        offender_list = '\n  '.join(offenders)
        raise AssertionError(
            'killpg() dispatched against a pgid re-derived by os.getpgid() at '
            'kill time. Once the target has been reaped its pid may be '
            'recycled, so getpgid can return an unrelated group -- in task '
            "845's incidents, the user `systemd --user` group, killing the "
            'whole login session.\n\n'
            'Fix: capture the pgid immediately after the '
            '`start_new_session=True` spawn (`pgid = proc.pid`, which POSIX '
            'guarantees at that moment) and pass that frozen int to killpg. '
            'Also short-circuit on `proc.returncode is not None` -- the frozen '
            'number itself goes stale once the leader is reaped. See '
            "shared/src/shared/proc_group.py's module docstring for the "
            'canonical contract.'
            f'\n\nOffending sites:\n  {offender_list}'
        )
