"""Guard test: no NEW orchestrator.merge_queue.<private> reach-back
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

Both idioms observed in this suite are detected: the string-path form
(``patch('orchestrator.merge_queue.<leaf>', ...)`` /
``monkeypatch.setattr('orchestrator.merge_queue.<leaf>', ...)``) and the
object-path form (``monkeypatch.setattr(merge_queue, '<leaf>', ...)`` /
``patch.object(merge_queue, '<leaf>', ...)``, where ``merge_queue`` is bound
via ``import orchestrator.merge_queue as merge_queue`` or
``from orchestrator import merge_queue``, or referenced via the bare
``orchestrator.merge_queue`` attribute chain). Both forms resolve to the
identical reach-back lookup site at runtime, so both must freeze the same
surface -- see `_merge_queue_module_aliases()`.

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
# Last path segment of each satellite module (e.g. 'merge_gates'). A relative
# shim import (`from .merge_gates import ...`) carries no package prefix --
# `ast.ImportFrom.module` is just the leaf name and `.level > 0` -- so it
# cannot be matched against the fully-qualified `_SATELLITE_MODULES` strings.
# See `_forbidden_reachback_names_from_source()`.
_SATELLITE_LEAF_NAMES = {name.rsplit('.', 1)[-1] for name in _SATELLITE_MODULES}


def _forbidden_reachback_names_from_source(source: str) -> set[str]:
    """The merge_queue-private reach-back patch surface: `_`-prefixed names
    re-exported from the four satellite modules via shim import blocks in
    *source*.

    Recognizes both the current absolute-import shim form
    (``from orchestrator.merge_gates import ...``) and a relative-import
    form (``from .merge_gates import ...``), matching on the last path
    segment when ``node.level > 0`` since a relative import has no package
    prefix to compare against ``_SATELLITE_MODULES``. Without this, a future
    switch to relative imports would silently collapse the forbidden set to
    empty and this ratchet would pass vacuously instead of loudly failing
    (see the non-empty assertion in
    `test_no_new_merge_queue_private_reachback_patches`).

    Split out from `_forbidden_reachback_names()` so the relative-import
    branch is unit-testable against a synthetic snippet without touching
    merge_queue.py.
    """
    tree = ast.parse(source)
    forbidden: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        expected_modules = _SATELLITE_LEAF_NAMES if node.level else _SATELLITE_MODULES
        if node.module in expected_modules:
            forbidden.update(alias.name for alias in node.names if alias.name.startswith('_'))
    return forbidden


def _forbidden_reachback_names() -> set[str]:
    """The merge_queue-private reach-back patch surface, derived by
    AST-parsing merge_queue.py itself rather than a hardcoded list, so the
    forbidden set self-narrows as later PRD scopes delete names from the
    shim.
    """
    source = _MERGE_QUEUE_PATH.read_text(encoding='utf-8')
    return _forbidden_reachback_names_from_source(source)


def _merge_queue_module_aliases(tree: ast.AST) -> set[str]:
    """Names bound directly to the `orchestrator.merge_queue` module object by
    an import statement anywhere in *tree* (module-level or function-local),
    e.g. ``import orchestrator.merge_queue as merge_queue`` or
    ``from orchestrator import merge_queue``.

    Used by `_find_merge_queue_private_patches()` to recognize the
    object-path reach-back idiom (``setattr(merge_queue, '<leaf>', ...)`` /
    ``patch.object(merge_queue, '<leaf>', ...)``), which targets the exact
    same reach-back lookup site as the string-path form without embedding
    the module path as a string constant.
    """
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'orchestrator.merge_queue' and alias.asname:
                    aliases.add(alias.asname)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module == 'orchestrator'
            and not node.level
        ):
            for alias in node.names:
                if alias.name == 'merge_queue':
                    aliases.add(alias.asname or 'merge_queue')
    return aliases


def _find_merge_queue_private_patches(source: str, forbidden: set[str]) -> list[tuple[int, str]]:
    """Return ``(lineno, leaf)`` for each reach-back patch call in *source*
    targeting a merge_queue-private *leaf* in *forbidden*, in either idiom:

    * string-path -- ``patch(...)`` / ``<attr>.patch(...)`` /
      ``<attr>.setattr(...)`` whose first positional argument is a string
      constant ``orchestrator.merge_queue.<leaf>``; or
    * object-path -- ``<attr>.setattr(<ref>, '<leaf>', ...)`` /
      ``patch.object(<ref>, '<leaf>', ...)`` where ``<ref>`` is a name bound
      to the `orchestrator.merge_queue` module (see
      `_merge_queue_module_aliases()`) or the bare ``orchestrator.merge_queue``
      attribute chain, and the second positional argument is the string
      constant ``'<leaf>'``.

    Both idioms resolve to the identical reach-back lookup site at runtime,
    so both must be frozen by the same ratchet.

    AST-based (returns ``[]`` on a ``SyntaxError``) so comments, docstrings,
    and the ``ALLOWLIST`` literal below are never mistaken for a real patch
    call site.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    merge_queue_aliases = _merge_queue_module_aliases(tree)

    def _is_merge_queue_ref(expr: ast.expr) -> bool:
        if isinstance(expr, ast.Name):
            return expr.id in merge_queue_aliases
        return (
            isinstance(expr, ast.Attribute)
            and expr.attr == 'merge_queue'
            and isinstance(expr.value, ast.Name)
            and expr.value.id == 'orchestrator'
        )

    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        is_setattr = isinstance(func, ast.Attribute) and func.attr == 'setattr'
        is_dotted_patch = isinstance(func, ast.Attribute) and func.attr == 'patch'
        is_bare_patch = isinstance(func, ast.Name) and func.id == 'patch'
        is_patch_object = (
            isinstance(func, ast.Attribute)
            and func.attr == 'object'
            and (
                (isinstance(func.value, ast.Name) and func.value.id == 'patch')
                or (isinstance(func.value, ast.Attribute) and func.value.attr == 'patch')
            )
        )

        leaf: str | None = None

        # String-path form: the dotted path IS the first positional arg.
        if is_setattr or is_dotted_patch or is_bare_patch:
            first_arg = node.args[0]
            if (
                isinstance(first_arg, ast.Constant)
                and isinstance(first_arg.value, str)
                and first_arg.value.startswith(_MERGE_QUEUE_PATCH_PREFIX)
            ):
                leaf = first_arg.value[len(_MERGE_QUEUE_PATCH_PREFIX):]

        # Object-path form: first positional arg is a merge_queue module
        # reference, second positional arg is the leaf name string.
        if leaf is None and (is_setattr or is_patch_object) and len(node.args) >= 2:
            target, name_arg = node.args[0], node.args[1]
            if (
                _is_merge_queue_ref(target)
                and isinstance(name_arg, ast.Constant)
                and isinstance(name_arg.value, str)
            ):
                leaf = name_arg.value

        if leaf is not None and leaf in forbidden:
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
    ('test_merge_queue_lifecycle_registry.py', '_finalize_advanced_merge'),
    ('test_merge_queue_lifecycle_registry.py', '_reverify_rebased_tree'),
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
# _find_merge_queue_private_patches(source, forbidden) -- object-path form:
# setattr(merge_queue, '<leaf>', ...) / patch.object(merge_queue, '<leaf>', ...).
# This is the same reach-back lookup site as the string-path form above, just
# without embedding the module path as a string constant.
# ---------------------------------------------------------------------------


def test_find_merge_queue_private_patches_flags_object_form_setattr_via_from_import() -> None:
    """`monkeypatch.setattr(merge_queue, '<forbidden>', ...)` -- object-path form
    with `merge_queue` bound via `from orchestrator import merge_queue` (the
    idiom already used for `merge_queue._DEBUG_ASSERTS` elsewhere in this
    suite) -- is flagged just like the string-path form."""
    source = (
        "from orchestrator import merge_queue\n"
        "\n"
        "def test_something(monkeypatch):\n"
        "    monkeypatch.setattr(merge_queue, '_check_post_merge_equivalence', object())\n"
    )
    forbidden = {'_check_post_merge_equivalence'}
    hits = _find_merge_queue_private_patches(source, forbidden)
    assert [leaf for _lineno, leaf in hits] == ['_check_post_merge_equivalence']


def test_find_merge_queue_private_patches_flags_patch_object_form_via_import_as() -> None:
    """`patch.object(merge_queue, '<forbidden>', ...)` -- object-path form with
    `merge_queue` bound via `import orchestrator.merge_queue as merge_queue`
    -- is flagged."""
    source = (
        "import orchestrator.merge_queue as merge_queue\n"
        "from unittest.mock import patch\n"
        "\n"
        "def test_something():\n"
        "    with patch.object(merge_queue, '_run_drift_check', object()):\n"
        "        pass\n"
    )
    forbidden = {'_run_drift_check'}
    hits = _find_merge_queue_private_patches(source, forbidden)
    assert [leaf for _lineno, leaf in hits] == ['_run_drift_check']


def test_find_merge_queue_private_patches_flags_object_form_dotted_attribute_chain() -> None:
    """`patch.object(orchestrator.merge_queue, '<forbidden>', ...)` -- the bare
    dotted attribute chain with no `as` alias -- is also recognized as a
    merge_queue module reference."""
    source = (
        "import orchestrator.merge_queue\n"
        "from unittest.mock import patch\n"
        "\n"
        "def test_something():\n"
        "    with patch.object(orchestrator.merge_queue, '_run_drift_check', object()):\n"
        "        pass\n"
    )
    forbidden = {'_run_drift_check'}
    hits = _find_merge_queue_private_patches(source, forbidden)
    assert [leaf for _lineno, leaf in hits] == ['_run_drift_check']


def test_find_merge_queue_private_patches_ignores_object_form_on_other_module() -> None:
    """`patch.object(merge_gates, '<forbidden>', ...)` targets the defining
    satellite module directly (already repointed) -- object-path form must
    NOT be flagged just because it is a `.object(...)` call; the target must
    actually resolve to the merge_queue module."""
    source = (
        "import orchestrator.merge_gates as merge_gates\n"
        "from unittest.mock import patch\n"
        "\n"
        "def test_something():\n"
        "    with patch.object(merge_gates, '_check_post_merge_equivalence', object()):\n"
        "        pass\n"
    )
    forbidden = {'_check_post_merge_equivalence'}
    assert _find_merge_queue_private_patches(source, forbidden) == []


def test_find_merge_queue_private_patches_ignores_unrelated_merge_queue_attribute() -> None:
    """A `workflow.merge_queue` attribute (e.g. an `asyncio.Queue` instance
    stored on an unrelated object, an idiom already used elsewhere in this
    suite) is NOT a reference to the `orchestrator.merge_queue` module and
    must not be flagged just because its attribute name happens to be
    `merge_queue`."""
    source = (
        "from unittest.mock import patch\n"
        "\n"
        "def test_something(workflow):\n"
        "    with patch.object(workflow.merge_queue, '_run_drift_check', object()):\n"
        "        pass\n"
    )
    forbidden = {'_run_drift_check'}
    assert _find_merge_queue_private_patches(source, forbidden) == []


def test_merge_queue_module_aliases_recognizes_import_forms() -> None:
    """`_merge_queue_module_aliases()` collects names bound to the
    `orchestrator.merge_queue` module via either import form, and ignores
    unrelated satellite imports."""
    source = (
        "import orchestrator.merge_queue as merge_queue\n"
        "from orchestrator import merge_queue as mq2\n"
        "from orchestrator import merge_queue\n"
        "import orchestrator.merge_gates as merge_gates\n"
    )
    tree = ast.parse(source)
    assert _merge_queue_module_aliases(tree) == {'merge_queue', 'mq2'}


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


def test_forbidden_reachback_names_from_source_handles_relative_imports() -> None:
    """If merge_queue.py's shim blocks ever switch to relative-import form
    (`from .merge_gates import ...`), the forbidden set must still be
    derived correctly instead of silently collapsing to empty (which would
    make the tree-scan ratchet pass vacuously rather than loudly fail --
    see the non-empty assertion in
    `test_no_new_merge_queue_private_reachback_patches`)."""
    source = (
        "from .merge_gates import (\n"
        "    _check_post_merge_equivalence,  # noqa: F401 re-export shim\n"
        "    PostMergePyrightResult,  # noqa: F401 re-export shim\n"
        ")\n"
        "from .merge_drift import _run_drift_check  # noqa: F401 re-export shim\n"
    )
    forbidden = _forbidden_reachback_names_from_source(source)
    assert forbidden == {'_check_post_merge_equivalence', '_run_drift_check'}


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
    assert forbidden, (
        'expected a non-empty forbidden reach-back name set derived from '
        "merge_queue.py's satellite shim blocks -- an empty set would mean "
        'this ratchet is passing vacuously (e.g. merge_queue.py stopped '
        'using an import form _forbidden_reachback_names_from_source() '
        'recognizes). See test_forbidden_reachback_names_derives_from_shim_blocks '
        'and test_forbidden_reachback_names_from_source_handles_relative_imports '
        'for the contract this depends on.'
    )
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
