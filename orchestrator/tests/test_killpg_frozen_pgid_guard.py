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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_SKIP_PARTS = {'.worktrees', '.venv', 'node_modules', '__pycache__'}

# Minimum number of .py files the sweep must actually scan. Measured at 1567 in
# this repo, so the floor sits ~7.8x below reality: low enough not to become
# brittle as packages move, high enough that a filter regression trips it.
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


def _python_files() -> list[Path]:
    """Every .py file under the repo's package/script roots.

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


def test_sweep_is_not_vacuous() -> None:
    """The sweep must actually scan the tree.

    Load-bearing, not stylistic: a zero-file sweep passes the ratchet below
    green, so without this the guard could look healthy while checking nothing
    -- and it would look healthy specifically inside task worktrees, i.e.
    everywhere branches are verified.
    """
    scanned = _python_files()
    relative = {str(p.relative_to(REPO_ROOT)) for p in scanned}

    for required in (
        'orchestrator/src/orchestrator/deterministic_runner.py',
        'orchestrator/tests/test_killpg_frozen_pgid_guard.py',
        'shared/src/shared/proc_group.py',
    ):
        assert required in relative, (
            f'{required} missing from the sweep -- the root globs or the skip '
            f'filter regressed (skip must be applied to the RELATIVE path)'
        )

    assert len(scanned) >= _MIN_SCANNED_FILES, (
        f'sweep scanned only {len(scanned)} files, expected >= '
        f'{_MIN_SCANNED_FILES} -- the guard is vacuous. Most likely the skip '
        f'filter is being applied to the absolute path, which contains '
        f'".worktrees" for every task worktree.'
    )


def test_no_killpg_over_getpgid_in_repo() -> None:
    """No file in the repo may re-derive a pgid inside a killpg dispatch."""
    offenders: list[str] = []

    for py_file in _python_files():
        source = py_file.read_text(encoding='utf-8')
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
