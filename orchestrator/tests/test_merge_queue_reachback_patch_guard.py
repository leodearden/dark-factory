"""Guard test: no NEW orchestrator.merge_queue.<private> reach-back string-path
monkeypatches in orchestrator/tests/*.py (merge-queue-reliability PRD, scope epsilon).

merge_queue.py re-exports private (`_`-prefixed) helpers from four satellite
modules -- merge_gates, merge_drift, merge_shadow, merge_liveness -- via
`# noqa: F401 re-export shim` import blocks near the top of the file.  Several
of those satellites (and workflow.py) reach back through
``orchestrator.merge_queue.<name>`` at call time via a function-local deferred
import specifically so the existing test suite's string-path monkeypatches on
that historical path keep resolving post-extraction.  Because monkeypatch
resolves at the LOOKUP site rather than the definition site, patching the
satellite's own copy does not intercept a reach-back consumer -- only the
merge_queue-side patch does.

This guard does not force those sites to migrate (most of them cannot, until
the back-imports they depend on are deleted in a later PRD scope).  It only
freezes the CURRENT reach-back patch surface: the ALLOWLIST below is the full
residual, and any NEW `orchestrator.merge_queue.<private>` patch -- a new name
anywhere, or an existing forbidden name spreading into a new test file --
fails this test.  As later scopes delete back-imports and shrink the shim,
``_forbidden_reachback_names()`` narrows automatically and the allowlist can
shrink to match.

This test uses AST parsing (not regex) so comments and docstrings that merely
mention the string -- including the reach-back notes in the satellite module
docstrings, and this file's own ALLOWLIST literal -- are never mistaken for a
real patch call site.
"""
from __future__ import annotations

import ast
from pathlib import Path

_THIS_FILE = Path(__file__).name
_TESTS_DIR = Path(__file__).parent
_SRC_DIR = Path(__file__).parent.parent / 'src' / 'orchestrator'
_MERGE_QUEUE_PATH = _SRC_DIR / 'merge_queue.py'
_MERGE_QUEUE_PATCH_PREFIX = 'orchestrator.merge_queue.'

_SATELLITE_MODULES = {
    'orchestrator.merge_gates',
    'orchestrator.merge_drift',
    'orchestrator.merge_shadow',
    'orchestrator.merge_liveness',
}


def _forbidden_reachback_names() -> set[str]:
    """The merge_queue-private reach-back patch surface: `_`-prefixed names
    merge_queue.py re-exports from the four satellite modules via its shim
    import blocks.

    Derived by AST-parsing merge_queue.py itself rather than a hardcoded
    list, so the forbidden set self-narrows as later PRD scopes delete names
    from the shim.
    """
    source = _MERGE_QUEUE_PATH.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(_MERGE_QUEUE_PATH))
    forbidden: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _SATELLITE_MODULES:
            forbidden.update(alias.name for alias in node.names if alias.name.startswith('_'))
    return forbidden


def _find_merge_queue_private_patches(source: str, forbidden: set[str]) -> list[tuple[int, str]]:
    """Return ``(lineno, leaf)`` for each ``patch(...)`` / ``<attr>.patch(...)`` /
    ``<attr>.setattr(...)`` call in *source* whose first positional argument is
    a string constant ``orchestrator.merge_queue.<leaf>`` with *leaf* in
    *forbidden*.

    AST-based (returns ``[]`` on a ``SyntaxError``) so comments, docstrings,
    and the ``ALLOWLIST`` literal below are never mistaken for a real patch
    call site.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        is_patch_or_setattr = (
            (isinstance(func, ast.Attribute) and func.attr in ('patch', 'setattr'))
            or (isinstance(func, ast.Name) and func.id == 'patch')
        )
        if not is_patch_or_setattr:
            continue
        first_arg = node.args[0]
        if not (isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str)):
            continue
        if not first_arg.value.startswith(_MERGE_QUEUE_PATCH_PREFIX):
            continue
        leaf = first_arg.value[len(_MERGE_QUEUE_PATCH_PREFIX):]
        if leaf in forbidden:
            hits.append((node.lineno, leaf))
    return hits


# ALLOWLIST -- the current reach-back patch residual: (relative_test_path,
# leaf_name) pairs.  This IS the scope-zeta..lambda worklist -- each pair
# clears only once its consumer's back-import is deleted so the site can
# repoint to the defining satellite module.
#
# Pre-1 / steps 3-4 (task 2157) classified all 16 forbidden names actually
# patched via orchestrator.merge_queue in the suite (spanning the 36 pairs
# below) as reach-back-locked: for each, some exercised consumer resolves the
# name via a function-local `from orchestrator.merge_queue import X`
# elsewhere in src/ (a satellite module, or workflow.py), or via a bare
# reference in merge_queue.py's own body -- so patching the satellite copy
# directly would not intercept the call and would turn the test RED. Zero
# pairs were safe to repoint opportunistically. Lock site per name:
#
#   _acquire_warm_verify_worktree        merge_queue.py:8761  (_run_inflight_verify, own body)
#   _check_plan_files_touched_in_branch  workflow.py:5466     (_submit_to_merge_queue)
#   _check_plan_targets_in_tree          merge_queue.py:2644  (classify_and_merge, own body)
#   _check_post_merge_equivalence        merge_gates.py:366   (_run_equivalence_gate)
#   _check_post_merge_pyright            merge_gates.py:444   (_run_pyright_gate)
#   _commit_is_linear                    merge_gates.py:538   (_finalize_advanced_merge)
#   _finalize_advanced_merge             merge_queue.py:3313  (_do_train_merge, own body)
#   _map_advance_failure                 merge_queue.py:3286  (_do_train_merge, own body)
#   _maybe_run_drift_check               merge_queue.py:9156  (_finalize_inflight, own body)
#   _maybe_schedule_shadow_compare        merge_queue.py:9150  (_finalize_inflight, own body)
#   _rebase_delta_touched_overlap         merge_gates.py:1358  (_reverify_rebased_tree)
#   _resolve_second_parent                merge_gates.py:538   (_finalize_advanced_merge)
#   _reverify_rebased_tree                merge_queue.py:9195  (_finalize_inflight, own body)
#   _run_cold_shadow_verify               merge_shadow.py:852  (_run_shadow_compare)
#   _run_drift_check                      merge_drift.py:254   (_maybe_run_drift_check)
#   _run_shadow_compare                   merge_shadow.py:1012 (_maybe_schedule_shadow_compare)
#
# A name drops off this table -- and its ALLOWLIST pairs become genuine
# repoint candidates -- once a later scope deletes the back-import that
# locks it.
ALLOWLIST: frozenset[tuple[str, str]] = frozenset({
    ('test_atomic_train_merge.py', '_check_post_merge_equivalence'),
    ('test_merge_drift.py', '_run_drift_check'),
    ('test_merge_gates.py', '_check_post_merge_equivalence'),
    ('test_merge_gates.py', '_check_post_merge_pyright'),
    ('test_merge_gates.py', '_rebase_delta_touched_overlap'),
    ('test_merge_guard_pipeline.py', '_check_plan_targets_in_tree'),
    ('test_merge_queue.py', '_check_plan_targets_in_tree'),
    ('test_merge_queue.py', '_check_post_merge_equivalence'),
    ('test_merge_queue.py', '_check_post_merge_pyright'),
    ('test_merge_queue.py', '_commit_is_linear'),
    ('test_merge_queue.py', '_finalize_advanced_merge'),
    ('test_merge_queue.py', '_rebase_delta_touched_overlap'),
    ('test_merge_queue.py', '_resolve_second_parent'),
    ('test_merge_queue.py', '_reverify_rebased_tree'),
    ('test_merge_queue_concurrent_verify.py', '_maybe_schedule_shadow_compare'),
    ('test_merge_queue_equivalence.py', '_check_post_merge_pyright'),
    ('test_merge_queue_invariant_integration_gate.py', '_check_post_merge_equivalence'),
    ('test_merge_queue_invariant_integration_gate.py', '_check_post_merge_pyright'),
    ('test_merge_queue_invariant_integration_gate.py', '_reverify_rebased_tree'),
    ('test_merge_queue_multihost_wiring.py', '_maybe_run_drift_check'),
    ('test_merge_queue_multihost_wiring.py', '_maybe_schedule_shadow_compare'),
    ('test_merge_queue_multihost_wiring.py', '_run_drift_check'),
    ('test_merge_queue_train_attribution.py', '_finalize_advanced_merge'),
    ('test_merge_queue_warm_cold_shadow.py', '_maybe_schedule_shadow_compare'),
    ('test_merge_queue_warm_cold_shadow.py', '_run_cold_shadow_verify'),
    ('test_merge_queue_warm_cold_shadow.py', '_run_shadow_compare'),
    ('test_merge_shadow.py', '_run_cold_shadow_verify'),
    ('test_merge_shadow.py', '_run_shadow_compare'),
    ('test_merge_speculation.py', '_acquire_warm_verify_worktree'),
    ('test_merge_speculation.py', '_finalize_advanced_merge'),
    ('test_merge_speculation.py', '_map_advance_failure'),
    ('test_merge_speculation.py', '_maybe_run_drift_check'),
    ('test_merge_speculation.py', '_maybe_schedule_shadow_compare'),
    ('test_merge_speculation.py', '_reverify_rebased_tree'),
    ('test_merge_speculation.py', '_run_cold_shadow_verify'),
    ('test_workflow.py', '_check_plan_files_touched_in_branch'),
})


# ---------------------------------------------------------------------------
# _find_merge_queue_private_patches(source, forbidden) -- inline-fixture unit
# tests.
# ---------------------------------------------------------------------------


def test_find_merge_queue_private_patches_flags_reachback_patch_call() -> None:
    """A `patch('orchestrator.merge_queue.<forbidden>', ...)` string-path is flagged."""
    source = (
        "from unittest.mock import patch\n"
        "\n"
        "def test_something():\n"
        "    with patch('orchestrator.merge_queue._check_post_merge_equivalence', object()):\n"
        "        pass\n"
    )
    forbidden = {'_check_post_merge_equivalence'}
    hits = _find_merge_queue_private_patches(source, forbidden)
    assert [leaf for _lineno, leaf in hits] == ['_check_post_merge_equivalence']


def test_find_merge_queue_private_patches_ignores_repointed_satellite_patch() -> None:
    """A patch already repointed to the defining satellite module is NOT flagged."""
    source = (
        "from unittest.mock import patch\n"
        "\n"
        "def test_something():\n"
        "    with patch('orchestrator.merge_gates._check_post_merge_equivalence', object()):\n"
        "        pass\n"
    )
    forbidden = {'_check_post_merge_equivalence'}
    assert _find_merge_queue_private_patches(source, forbidden) == []


def test_find_merge_queue_private_patches_ignores_public_name() -> None:
    """A public/non-satellite orchestrator.merge_queue.<name> patch is NOT flagged
    (it is simply absent from the forbidden set)."""
    source = (
        "from unittest.mock import patch\n"
        "\n"
        "def test_something():\n"
        "    with patch('orchestrator.merge_queue.run_scoped_verification', object()):\n"
        "        pass\n"
    )
    forbidden = {'_check_post_merge_equivalence'}  # run_scoped_verification is not in it
    assert _find_merge_queue_private_patches(source, forbidden) == []


def test_find_merge_queue_private_patches_ignores_docstring_mention() -> None:
    """A comment/docstring merely mentioning the string is NOT flagged -- proves
    the detector is AST-based, not text/regex."""
    source = (
        '"""Reach-back note: production code patches '
        'orchestrator.merge_queue._check_post_merge_equivalence."""\n'
        "# patch('orchestrator.merge_queue._check_post_merge_equivalence', m)  example only\n"
    )
    forbidden = {'_check_post_merge_equivalence'}
    assert _find_merge_queue_private_patches(source, forbidden) == []


def test_find_merge_queue_private_patches_flags_monkeypatch_setattr() -> None:
    """A `monkeypatch.setattr('orchestrator.merge_queue.<forbidden>', ...)` string-path
    is flagged too, not just `patch(...)`."""
    source = (
        "def test_something(monkeypatch):\n"
        "    monkeypatch.setattr('orchestrator.merge_queue._run_drift_check', object())\n"
    )
    forbidden = {'_run_drift_check'}
    hits = _find_merge_queue_private_patches(source, forbidden)
    assert [leaf for _lineno, leaf in hits] == ['_run_drift_check']


# ---------------------------------------------------------------------------
# _forbidden_reachback_names() -- derives the forbidden set from merge_queue.py
# itself (the four satellite shim blocks), not a hardcoded list.
# ---------------------------------------------------------------------------


def test_forbidden_reachback_names_derives_from_shim_blocks() -> None:
    forbidden = _forbidden_reachback_names()

    # Private names re-exported from the four satellite shim blocks.
    assert '_check_post_merge_equivalence' in forbidden
    assert '_run_cold_shadow_verify' in forbidden
    assert '_run_drift_check' in forbidden

    # Public re-exports are never forbidden, even when re-exported from one of
    # the four satellite blocks.
    assert 'check_merge_liveness_margin' not in forbidden
    assert 'SpeculativeMergeWorker' not in forbidden
    assert 'enqueue_merge_request' not in forbidden

    # Reach-back names owned by non-satellite modules (git_ops.py, verify.py)
    # are never forbidden -- only the four satellite blocks are in scope.
    assert 'run_scoped_verification' not in forbidden
    assert '_run' not in forbidden


# ---------------------------------------------------------------------------
# Required fixture: demonstrate fail-on-new.
# ---------------------------------------------------------------------------


def test_guard_flags_synthetic_new_patch() -> None:
    """A brand-new orchestrator.merge_queue.<forbidden> patch, under a synthetic
    filename not in ALLOWLIST, must be reported as a violation."""
    forbidden = _forbidden_reachback_names()
    leaf = sorted(forbidden)[0]
    synthetic_source = (
        "from unittest.mock import patch\n"
        f"patch('orchestrator.merge_queue.{leaf}', object())\n"
    )
    synthetic_file = '__synthetic_not_in_allowlist__.py'

    hits = _find_merge_queue_private_patches(synthetic_source, forbidden)
    violations = {(synthetic_file, hit_leaf) for _lineno, hit_leaf in hits} - ALLOWLIST

    assert violations == {(synthetic_file, leaf)}, (
        f'expected the synthetic new patch on {leaf!r} to be flagged as a '
        f'violation, got {violations!r}'
    )


# ---------------------------------------------------------------------------
# Tree-scan: the actual ratchet.
# ---------------------------------------------------------------------------


def test_no_new_merge_queue_private_reachback_patches() -> None:
    """No orchestrator test file may introduce a NEW orchestrator.merge_queue.<private>
    reach-back string-path monkeypatch beyond the frozen ALLOWLIST residual."""
    forbidden = _forbidden_reachback_names()
    violations: set[tuple[str, str]] = set()
    offenders: list[str] = []

    for py_file in sorted(_TESTS_DIR.rglob('*.py')):
        if py_file.name == _THIS_FILE:
            continue  # skip the guard itself (its docstring/fixtures/ALLOWLIST mention the pattern)
        source = py_file.read_text(encoding='utf-8')
        rel = str(py_file.relative_to(_TESTS_DIR))
        for lineno, leaf in _find_merge_queue_private_patches(source, forbidden):
            pair = (rel, leaf)
            if pair not in ALLOWLIST:
                violations.add(pair)
                offenders.append(f'{rel}:{lineno}: orchestrator.merge_queue.{leaf}')

    if violations:
        offender_list = '\n  '.join(sorted(offenders))
        raise AssertionError(
            'New orchestrator.merge_queue.<private> reach-back string-path '
            'monkeypatch(es) found that are not in ALLOWLIST.\n'
            'Patch the defining satellite module directly instead '
            '(orchestrator.merge_gates / merge_drift / merge_shadow / '
            'merge_liveness) if the exercised consumer resolves the name '
            'outside the merge_queue reach-back path. If a merge_queue patch '
            'is genuinely unavoidable (the consumer reaches back through '
            'merge_queue), add the (file, name) pair to ALLOWLIST with '
            'justification.\n'
            f'\nOffending sites:\n  {offender_list}'
        )
