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


# ---------------------------------------------------------------------------
# _find_merge_queue_private_patches(source, forbidden) -- inline-fixture unit
# tests.  The helper and ALLOWLIST do not exist yet (added in step-2), so
# every test below fails with NameError until then.
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
