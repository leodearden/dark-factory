"""Repo-level structural fence around the anti-regrowth atomic-write guard.

Step 3 of task 3388 creates this module with the two BOUNDARY tests only; the
guard itself (``_SRC_TREES``, ``_ALLOWED_RENAMERS``, ``_find_renamers``,
``_iter_source_files``, ``TestNoRegrownAtomicWriters``) is relocated here from
``shared/tests/test_safe_io.py`` in step 4, which also writes this docstring's
placement rationale and its cross-references to the sibling repo-level sweeps
already living in this directory.
"""
import ast
from pathlib import Path

# tests/scripts/<file>.py and shared/tests/<file>.py are both exactly two
# levels below the repo root, so this constant carries over verbatim from the
# guard's old home and is shared by the boundary tests below and by the
# relocated guard (task 3388 step 4).
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The ONE file this module's boundary test pins.  Deliberately a single file
#: rather than a walk of ``shared/tests/**/*.py`` — see the scope-limit block
#: in test_atomic_write_guard_does_not_scan_sibling_package_trees.
_GUARDED_FILE = 'shared/tests/test_safe_io.py'

#: Scan roots that name a package ``shared`` does not own.  Matched as exact
#: STRING LITERALS via ast, never as substrings of prose: a comment or
#: docstring in the guarded file may legitimately discuss another package (the
#: relocated guard's own scope block does), and only a scan root — a literal
#: fed to a directory walk — can make shared's suite depend on that package.
_SIBLING_SCAN_ROOTS = (
    'orchestrator/src',
    'escalation/src',
    'fused-memory/src',
    'fused-memory/scripts',
    'scripts',
)

#: One real, long-lived module per declared tree.  ``_SRC_TREES`` yielding
#: *something* is not enough — a tree that rglobs only a stray ``__init__.py``
#: would satisfy a bare non-empty check while scanning nothing that matters.
_CONTROL_MODULES = {
    'shared/src': 'shared/src/shared/safe_io.py',
    'orchestrator/src': 'orchestrator/src/orchestrator/digest.py',
    'escalation/src': 'escalation/src/escalation/queue.py',
    'fused-memory/src': 'fused-memory/src/fused_memory/reconciliation/event_queue.py',
    'fused-memory/scripts': 'fused-memory/scripts/bake_off_storage_shape.py',
    'scripts': 'scripts/legibility/codebook.py',
}


def test_atomic_write_guard_does_not_scan_sibling_package_trees():
    """``shared/tests/test_safe_io.py`` declares no sibling package as a scan root.

    THE INVARIANT THIS DELIVERS, stated at its true width.  The atomic-write
    anti-regrowth guard must not be turnable red by a refactor in a package
    ``shared`` does not own.  Task 3388's worked example: the guard allowlists
    ``orchestrator/src/orchestrator/digest.py::write_digest_entry`` and flags it
    there as a prime migration candidate — so while the guard lived in
    ``shared/tests``, migrating or renaming that orchestrator function turned
    the SHARED suite red, at a site no orchestrator author would think to look.
    Relocating the guard to this repo-level directory (step 4) is what removes
    that coupling, and this test is what keeps it removed.

    SCOPE LIMIT — READ THIS BEFORE TRUSTING A GREEN RUN.  This test pins ONE
    file, not ``shared/tests`` as a whole.  It does NOT establish that shared's
    suite is standalone-runnable against a lone ``dark-factory-shared``
    checkout, and after task 3388 that suite is **not** standalone-runnable.
    The five OTHER cross-tree gates below live in ``shared/tests`` and are
    deliberately NOT covered here; each carries its own comment arguing for its
    cross-tree reach, so narrowing them is a design question this task did not
    settle rather than an oversight it missed:

      * ``silent_fallthrough_scan.py`` — ``_SCOPE_ROOTS`` (7 roots:
        orchestrator/src, fused-memory/src, dashboard/src, escalation/src,
        shared/src, sampler/src, scripts) and a hard ``RuntimeError`` when the
        ``shared/src``/``orchestrator/src`` sentinels are absent, paired with
        ``silent_fallthrough_allowlist.py`` (13 sibling-path literals, as
        ``(path, qualname, hash, reason)`` tuples).
      * ``config_dir_archival_allowlist.py`` — 10 sibling-path literals, as
        ``{'path': ..., 'qualname': ...}`` dicts.
      * ``test_auth_failed.py`` — ``_PRODUCTION_SRC_ROOTS = ('shared/src',
        'orchestrator/src')`` with a hard ``assert root.is_dir()``.
      * ``test_silent_fallthrough_gate.py`` — 3 sibling-path literals,
        hard-asserted (``assert candidate in files``).
      * ``test_capability_manifest.py`` — 21 sibling-path literals; most are
        synthetic fixtures, but it hard-asserts on REAL files in ``scripts/``
        (``committed_file_mode('scripts/check_method_param_wiring.py') ==
        '100755'``, and a superset assertion naming
        ``scripts/gc_agent_transcripts.py``).

    Counts measured first-hand at commit 6b68a87fd6 by an ast sweep of
    ``shared/tests/**/*.py`` for non-docstring string constants ending in
    ``.py`` and beginning with a sibling package directory — stated as a method
    plus a number so the claim stays falsifiable rather than becoming the kind
    of stale prose task 3388 exists to eliminate.  Re-run the sweep; do not
    re-trust this sentence.

    WHY THE NARROW FORM.  The plan specified a walk of ``shared/tests/**/*.py``
    asserting the whole directory names no sibling tree.  Measured, that test
    is red forever: the gates above are the falsifying evidence, they are
    outside this task's file scope, and each is load-bearing where it sits.
    Ticket ``tkt_0RT7TDAAH2TS88BR88TZ1E3QMP`` tracks the real question — where
    this family of repo-wide gates should live, and what it should do when a
    tree it names is absent.  Until that lands, this test is a fence around one
    gate and must not be read as evidence about the others.
    """
    target = _REPO_ROOT / _GUARDED_FILE
    assert target.is_file(), (
        f'{_GUARDED_FILE} not found under {_REPO_ROOT}. This test pins one named '
        f'file, so a move or rename must update _GUARDED_FILE rather than let the '
        f'check silently pass over a file that is no longer there.'
    )

    offenders = [
        f'{_GUARDED_FILE}:{node.lineno}: {node.value!r}'
        for node in ast.walk(ast.parse(target.read_text(encoding='utf-8')))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value in _SIBLING_SCAN_ROOTS
    ]

    assert not offenders, (
        f'{_GUARDED_FILE} declares a scan root in a package shared does not own:\n  '
        + '\n  '.join(offenders)
        + '\nA guard that walks sibling package trees cannot live inside one '
        'package\'s suite: a refactor in that sibling (migrating the allowlisted '
        'orchestrator digest.write_digest_entry, say) turns SHARED red, at a site '
        'no orchestrator author would think to look. Move the cross-tree guard to '
        'a repo-level suite — tests/scripts/test_atomic_write_regrowth.py is where '
        'the atomic-write one lives — rather than widening this exception.'
    )


def test_every_declared_tree_contributes_scanned_files():
    """Every tree in ``_SRC_TREES`` actually contributes files to the scan.

    THE ANTI-VACUITY FENCE, and the load-bearing one of the two.  The guard's
    own ``assert root.is_dir()`` catches a DELETED tree.  It does not catch a
    tree that is present and rglobs nothing useful — which is exactly how a
    fence that was just WIDENED reports green while silently scanning less than
    it did before, and is the failure mode task 3388 exists to close.  So this
    asserts two things per tree: that it yields at least one scanned file, and
    that one named, long-lived control module from it is in the scanned set.
    """
    scanned = {relpath for relpath, _ in _iter_source_files()}

    assert set(_CONTROL_MODULES) == set(_SRC_TREES), (
        'Every declared tree needs a named control module, or the anti-vacuity '
        'check degrades to "the tree yielded something" for the unpaired one.\n'
        f'  declared but uncontrolled: {sorted(set(_SRC_TREES) - set(_CONTROL_MODULES))}\n'
        f'  controlled but undeclared: {sorted(set(_CONTROL_MODULES) - set(_SRC_TREES))}'
    )

    empty = [tree for tree in _SRC_TREES
             if not any(p.startswith(f'{tree}/') for p in scanned)]
    assert not empty, (
        f'Declared scan trees contributed NO files: {empty}. The tree exists (the '
        'is_dir assertion passed) but rglobbed nothing, so the guard is scanning '
        'less than its _SRC_TREES claims while still reporting green.'
    )

    missing = sorted(
        control for control in _CONTROL_MODULES.values() if control not in scanned
    )
    assert not missing, (
        f'Control modules absent from the scanned set: {missing}. Either the guard '
        'stopped reaching into that tree, or the module genuinely moved — in which '
        'case repoint _CONTROL_MODULES at another long-lived module in the same '
        'tree rather than deleting the entry, which would restore the vacuity this '
        'test exists to prevent.'
    )
