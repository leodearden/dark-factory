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

WHAT COUNTS AS A SITE. The detector flags a ``getpgid`` value REACHING a
``killpg``, in either of its two spellings: the single-expression composition
``killpg(... getpgid(...) ...)``, and the hoisted two-statement form
``pgid = os.getpgid(p.pid)`` followed by ``os.killpg(pgid, ...)`` in the same
function scope. Both are the identical defect — a pgid re-derived at kill time
from a possibly-recycled pid — and the hoisted one is the likelier product of a
refactor, since the failure message below says to "pass that frozen int to
killpg". Taint is scope-local, and the residual gap that leaves is stated
precisely in ``_killpg_over_getpgid``'s docstring rather than papered over.

What is deliberately NOT flagged is any ``os.getpgid`` call that never reaches
a ``killpg``: ``git_ops.py``'s ``pgid = os.getpgid(pid)`` renders a "kernel
FLOCK holders: ..." diagnostics string and never signals anything, so it is
correct as written and is excluded *structurally*, leaving no allowlist entry
to rot. Keeping that exclusion structural under the widened detector is what
the scope partitioning buys: that same file separately does ``os.killpg(pgid,
0)`` as a liveness probe on a pgid read from a lock file, ~1000 lines away in a
different method, and a module-wide taint walk would pair the two into a false
violation.

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
from pathlib import Path

import pytest
from _orch_helpers import WHOLE_TREE_SCAN_TEST_TIMEOUT
from _tree_scan import (
    REPO_ROOT,
    TREE_PASS_XDIST_GROUP,
    ParsedModule,
    TreeAnalysis,
    TreePass,
    checkout_python_files,
    git_tracked_python_files,
)

# The family mark and shared group: _orch_helpers.py::WHOLE_TREE_SCAN_TEST_TIMEOUT, _tree_scan.py.
pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.xdist_group(TREE_PASS_XDIST_GROUP),
]

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


# A ``killpg`` call node belongs to exactly one of these scopes, so taint is
# never leaked between sibling functions. Lambdas and comprehensions are
# deliberately NOT scope boundaries here: they close over the enclosing
# function's locals, so a `lambda: os.killpg(pgid, 9)` must still see the
# `pgid = os.getpgid(...)` bound outside it.
_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)

_DIRECT = 'killpg(<...>getpgid(...))'
_INDIRECT = 'killpg(<name bound to getpgid(...) in the same function>)'


def _contains_call_to(node: ast.AST, func_name: str) -> bool:
    """True if *node*'s subtree contains a call resolving to *func_name*."""
    return any(
        isinstance(inner, ast.Call) and _called_name(inner) == func_name
        for inner in ast.walk(node)
    )


def _own_nodes(scope: ast.AST) -> list[ast.AST]:
    """Every node lexically inside *scope* but NOT inside a nested function.

    Partitioning matters: a plain ``ast.walk`` from the module would pull every
    function's body into one namespace, so ``git_ops.py``'s diagnostics-only
    ``pgid = os.getpgid(pid)`` (line ~1865) would taint the unrelated
    ``os.killpg(pgid, 0)`` liveness probe ~1000 lines away in a different
    method, whose ``pgid`` comes from ``read_lock_holder_pgid``. Measured: that
    exact false positive appears without this partitioning and vanishes with it.
    """
    collected: list[ast.AST] = []

    def descend(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _SCOPE_NODES):
                continue  # visited separately, as its own scope
            collected.append(child)
            descend(child)

    descend(scope)
    return collected


def _pgid_tainted_names(nodes: list[ast.AST]) -> set[str]:
    """Local names bound, in this scope, to an expression containing ``getpgid``.

    Covers ``pgid = os.getpgid(pid)``, the annotated ``pgid: int = ...``, the
    walrus ``(pgid := os.getpgid(pid))`` and tuple-unpacking spellings.
    """
    tainted: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        # AnnAssign.value is optional (`pgid: int` binds nothing);
        # NamedExpr.value is always present.
        elif isinstance(node, ast.AnnAssign | ast.NamedExpr) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not _contains_call_to(value, 'getpgid'):
            continue
        for target in targets:
            for name in ast.walk(target):
                if isinstance(name, ast.Name):
                    tainted.add(name.id)
    return tainted


def _killpg_over_getpgid(source: str) -> list[tuple[int, str]]:
    """Return ``(lineno, description)`` for each unsafe ``killpg`` dispatch.

    Two forms are flagged, because they are the same defect:

    1. the DIRECT composition ``killpg(... getpgid(...) ...)`` — an
       :class:`ast.Call` resolving to ``killpg`` with a nested :class:`ast.Call`
       resolving to ``getpgid`` anywhere in its own ``args``/``keywords``;
    2. the INDIRECT, two-statement spelling, where a local is bound from a
       ``getpgid`` call and that local is then passed to ``killpg`` **within the
       same function scope**::

           pgid = os.getpgid(proc.pid)   # re-derived at kill time
           os.killpg(pgid, signal.SIGKILL)

    Form 2 matters more than it looks: it is the shape a refactor naturally
    produces, precisely because this module's own failure message says to "pass
    that frozen int to killpg", which reads as an instruction to hoist the
    ``getpgid`` into a local. Flagging only form 1 would have let the exact
    task-845 footgun ship green in a new spelling.

    What is still NOT flagged, stated so the ratchet is not over-trusted: taint
    is scope-local, so a pgid derived by ``getpgid`` in one function and passed
    as an argument to another function that calls ``killpg`` is invisible here.
    Tracking that needs cross-procedural analysis, which is well past what an
    AST lint should attempt. A bare ``pgid = os.getpgid(pid)`` that never
    reaches a ``killpg`` in its scope stays correctly unflagged — that is
    ``git_ops.py``'s diagnostics read, excluded structurally, not by allowlist.

    AST-based, returning ``[]`` on a :class:`SyntaxError` so a stray
    non-Python-3 file in the tree cannot break the guard.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return _killpg_over_getpgid_in(tree)


def _killpg_over_getpgid_in(tree: ast.Module) -> list[tuple[int, str]]:
    """:func:`_killpg_over_getpgid` over an already-parsed module."""
    # Keyed by lineno so the direct form always wins over the indirect one when
    # a single call qualifies as both, and so the result is deterministic.
    hits: dict[int, str] = {}
    scopes: list[ast.AST] = [
        tree, *(n for n in ast.walk(tree) if isinstance(n, _SCOPE_NODES))
    ]
    for scope in scopes:
        nodes = _own_nodes(scope)
        tainted = _pgid_tainted_names(nodes)
        for node in nodes:
            if not isinstance(node, ast.Call) or _called_name(node) != 'killpg':
                continue
            # Only this call's ARGUMENTS are inspected, not the whole subtree of
            # the enclosing statement, so an unrelated sibling getpgid cannot
            # pin a false positive on the killpg.
            args = [*node.args, *(kw.value for kw in node.keywords)]
            if any(_contains_call_to(arg, 'getpgid') for arg in args):
                hits[node.lineno] = _DIRECT
            elif tainted and any(
                isinstance(inner, ast.Name) and inner.id in tainted
                for arg in args
                for inner in ast.walk(arg)
            ):
                hits.setdefault(node.lineno, _INDIRECT)
    return sorted(hits.items())


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


def test_flags_two_statement_spelling() -> None:
    """The hoisted form is the same defect and must not ship green.

    This is the spelling a refactor most naturally produces, because the
    ratchet's own failure message says to "pass that frozen int to killpg" --
    which reads as an instruction to hoist the getpgid into a local. The pgid
    is still re-derived at kill time from a possibly-recycled pid, so the
    task-845 hazard is identical to the single-expression form.
    """
    source = (
        "def cleanup(proc):\n"
        "    pgid = os.getpgid(proc.pid)\n"
        "    os.killpg(pgid, signal.SIGKILL)\n"
    )
    assert _killpg_over_getpgid(source) == [(3, _INDIRECT)]


def test_flags_hoisted_binding_in_its_other_spellings() -> None:
    """An annotated binding and a closure over it are the same violation.

    Lambdas and comprehensions are deliberately not scope boundaries in the
    detector: they close over the enclosing function's locals, so a killpg
    inside one still sees the tainted name bound outside it.
    """
    annotated = (
        "def cleanup(proc):\n"
        "    pgid: int = os.getpgid(proc.pid)\n"
        "    os.killpg(pgid, signal.SIGKILL)\n"
    )
    assert _killpg_over_getpgid(annotated) == [(3, _INDIRECT)]

    closure = (
        "def cleanup(proc):\n"
        "    pgid = os.getpgid(proc.pid)\n"
        "    atexit.register(lambda: os.killpg(pgid, signal.SIGKILL))\n"
    )
    assert _killpg_over_getpgid(closure) == [(3, _INDIRECT)]


def test_does_not_flag_getpgid_from_a_different_function() -> None:
    """Taint is scope-local -- the precision half of the widened detector.

    This is not a hypothetical: ``git_ops.py`` binds ``pgid = os.getpgid(pid)``
    for a diagnostics string in one function and, ~1000 lines away in a
    different method, does ``os.killpg(pgid, 0)`` as a liveness probe on a pgid
    read from a lock file. Both are correct. A module-wide taint walk flags that
    pair as a violation; scope partitioning is what keeps this ratchet
    allowlist-free rather than accumulating exemptions for correct code.
    """
    source = (
        "def describe(pid):\n"
        "    pgid = os.getpgid(pid)\n"
        "    return f'kernel FLOCK holders: {pgid}'\n"
        "\n"
        "def is_held(self):\n"
        "    pgid = read_lock_holder_pgid(self.base)\n"
        "    os.killpg(pgid, 0)\n"
    )
    assert _killpg_over_getpgid(source) == []


def test_known_gap_taint_does_not_reach_a_nested_def() -> None:
    """Documented boundary: a nested ``def`` does not inherit the outer taint.

    Pinned as a test rather than left implicit so the ratchet's limits are
    legible and nobody over-trusts it. Inheriting taint across a nested ``def``
    would need shadowing analysis (an inner parameter named ``pgid`` is a
    genuinely different value), and this spelling has zero incidence in the
    repo today -- whereas the lambda/comprehension closure above, which IS
    covered, is the form that actually occurs.

    The broader residual, stated in ``_killpg_over_getpgid``'s docstring: a
    pgid derived in one function and passed as an ARGUMENT to another that
    calls killpg is likewise invisible. Catching that needs cross-procedural
    analysis, well past what an AST lint should attempt.
    """
    source = (
        "def outer(proc):\n"
        "    pgid = os.getpgid(proc.pid)\n"
        "    def inner():\n"
        "        os.killpg(pgid, signal.SIGKILL)\n"
        "    return inner\n"
    )
    assert _killpg_over_getpgid(source) == []


# ---------------------------------------------------------------------------
# Tree-scan: the actual ratchet.
# ---------------------------------------------------------------------------


def _killpg_sites(module: ParsedModule) -> tuple[tuple[int, str], ...]:
    """Unsafe killpg dispatches in *module*; none if it cannot be read or parsed.

    Every finding is a call named ``killpg``, so a module that cannot name it is
    never parsed: ~97% of the swept files.  A dynamic spelling that avoids the
    name (``getattr(os, 'kill' + 'pg')``) is invisible to the walk as well.
    """
    try:
        if not module.may_name('killpg'):
            return ()
        tree = module.tree
    except (UnicodeDecodeError, OSError, SyntaxError):
        # The sweep reaches script trees parsed nowhere else. One undecodable,
        # unreadable or unparseable file must not take the whole ratchet down.
        return ()
    return tuple(_killpg_over_getpgid_in(tree))


ANALYSIS = TreeAnalysis(
    name=Path(__file__).stem,
    files=functools.partial(checkout_python_files, REPO_ROOT),
    extract=_killpg_sites,
)
TREE_ANALYSES = (ANALYSIS,)


def test_sweep_flags_a_compatibility_spelling_the_parser_normalizes(tmp_path: Path) -> None:
    """``ｋｉｌｌｐｇ`` parses as ``killpg``, so the sweep must not skip the file unparsed."""
    path = tmp_path / 'cleanup.py'
    path.write_text('os.ｋｉｌｌｐｇ(os.getpgid(proc.pid), signal.SIGKILL)\n', encoding='utf-8')

    assert _killpg_sites(ParsedModule(path)) == ((1, _DIRECT),)


def test_sweep_is_not_vacuous(tree_pass: TreePass) -> None:
    """The sweep must actually scan the tree.

    Load-bearing, not stylistic: a zero-file sweep passes the ratchet below
    green, so without this the guard could look healthy while checking nothing
    -- and it would look healthy specifically inside task worktrees, i.e.
    everywhere branches are verified.
    """
    scanned = tree_pass.facts(ANALYSIS).keys()
    relative = {str(p.relative_to(REPO_ROOT)) for p in scanned}

    # Witnesses reachable from the on-disk package roots alone, so these hold
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
    # three is reachable from the on-disk package roots (`*/src/*`,
    # `*/tests`, `scripts`, `hooks` and root-level `*.py` -- top-level `tests/`,
    # `fused-memory/scripts/` and `skills/**` match none of them); they enter
    # the sweep solely via `git ls-files`. Asserting them unconditionally would
    # mean that on a git-less machine this test hard-fails pointing at "the
    # root globs or the skip filter", sending a debugger at the wrong
    # subsystem, while the test that actually owns completeness
    # (`test_sweep_covers_every_tracked_python_file`) skips cleanly.
    if git_tracked_python_files(REPO_ROOT):
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


def test_sweep_covers_every_tracked_python_file(tree_pass: TreePass) -> None:
    """Every .py file git tracks must be inside the sweep.

    This is the *completeness* assertion, and it is a different instrument from
    ``test_sweep_is_not_vacuous`` above. A raw-count floor is structurally
    incapable of detecting a PARTIAL hole: a sweep can miss an entire tree and
    still sail past the floor by three orders of magnitude. Only a set
    difference against an independently-derived ground truth catches that, and
    ``git ls-files`` is the right ground truth -- it is exactly "the content
    that can land on main".
    """
    tracked = {p for p in git_tracked_python_files(REPO_ROOT) if p.is_file()}
    if not tracked:
        pytest.skip('git unavailable -- completeness cannot be established')

    missing = sorted(
        str(p.relative_to(REPO_ROOT)) for p in tracked - tree_pass.facts(ANALYSIS).keys()
    )
    if missing:
        shown = '\n  '.join(missing[:20])
        more = f'\n  ... and {len(missing) - 20} more' if len(missing) > 20 else ''
        raise AssertionError(
            f'{len(missing)} git-tracked .py file(s) are OUTSIDE this guard\'s '
            f'sweep, i.e. UNGUARDED: someone can re-introduce '
            f'`os.killpg(os.getpgid(p.pid), SIGKILL)` in any of them and ship '
            f'green, which defeats the ratchet entirely.\n\n'
            f'Fix by WIDENING `_tree_scan.py::checkout_python_files` so it covers them -- do NOT '
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
    scanned = checkout_python_files(REPO_ROOT)
    assert len(scanned) == len(set(scanned)), (
        f'sweep visited {len(scanned)} paths but only '
        f'{len(set(scanned))} are distinct -- the file sources overlap and are '
        f'not deduped. Collect them into a set before returning.'
    )


def test_no_killpg_over_getpgid_in_repo(tree_pass: TreePass) -> None:
    """No file in the repo may re-derive a pgid inside a killpg dispatch."""
    offenders = [
        f'{py_file.relative_to(REPO_ROOT)}:{lineno}: {description}'
        for py_file, sites in tree_pass.facts(ANALYSIS).items()
        for lineno, description in sites
    ]

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
            'NOTE "frozen" means captured AT SPAWN: hoisting the same '
            '`os.getpgid(proc.pid)` call into a local one line above the '
            'killpg is NOT a fix -- it is the same re-derivation, and it is '
            'flagged here too. '
            'Also short-circuit on `proc.returncode is not None` -- the frozen '
            'number itself goes stale once the leader is reaped. See '
            "shared/src/shared/proc_group.py's module docstring for the "
            'canonical contract.'
            f'\n\nOffending sites:\n  {offender_list}'
        )
