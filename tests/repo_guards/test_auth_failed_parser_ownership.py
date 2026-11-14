"""`shared.invocation_outcome` single-sources the resets parser across every production tree.

The UsageGate auth_failed lifecycle these guards protect is tested in
shared/tests/test_auth_failed.py. These guards walk the session's
whole-first-party tree, so they live here rather than in shared's suite.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from silent_fallthrough_scan import ParsedFile

from shared import invocation_outcome

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

#: tests/repo_guards/test_auth_failed_parser_ownership.py -> parents[0]=tests/repo_guards,
#: parents[1]=tests, parents[2]=repo root. Mirrors capability_manifest_corpus.REPO_ROOT;
#: correct inside a `.worktrees/<id>` checkout too.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The ONE module allowed to define — and so to call unqualified — the bare
#: `_parse_resets_at` / `_extract_cap_message` names. Everywhere else the
#: convention (already followed by shared/src/shared/usage_gate.py's
#: `import ... as _parse_resets_at_strict`) is to import under the explicit
#: alias, so a reader can tell at the call site WHICH copy is running.
_STRICT_PARSE_OWNER = 'shared/src/shared/invocation_outcome.py'

#: The names `_STRICT_PARSE_OWNER` single-sources.
_OWNED_NAMES = ('_parse_resets_at', '_extract_cap_message')

#: Roots whose ABSENCE means the walk below is broken rather than the tree
#: merely reorganised: these two held the retired fork and its re-export.
_REQUIRED_SRC_ROOTS = ('shared/src', 'orchestrator/src')


def _production_src_roots() -> list[str]:
    """Every workspace package's `<pkg>/src` tree, as a posix repo-relative path.

    DISCOVERED rather than hardcoded: a fixed list covers only the trees
    someone remembered to name, so a re-fork in a package added later — or in
    one simply overlooked, `dashboard/src` being the pointed example, since
    the dashboard is the consumer a fabricated reset time would mislead —
    would sail past a guard still reporting green. The ownership scan walks
    the session's shared first-party tree, whose scope is a fixed list and
    which also applies `iter_first_party_files`' test-code exclusions, so the
    discovered contract is this discovery PLUS
    `test_shared_tree_spans_every_production_src_root` and
    `test_shared_tree_drops_no_production_module`. Scoped to `<pkg>/src`
    on purpose: `.worktrees/` sits at the REPO root, so unlike a root-level
    rglob (the trap capability_manifest_corpus.py documents) this glob cannot
    wander into a sibling task's checkout.
    """
    found = sorted(
        p.relative_to(_REPO_ROOT).as_posix() for p in _REPO_ROOT.glob('*/src') if p.is_dir()
    )
    # Loud, not silently narrowed: a guard that quietly stops scanning a tree
    # it can no longer find is worse than no guard.
    missing = [rel for rel in _REQUIRED_SRC_ROOTS if rel not in found]
    assert not missing, f'production source root(s) missing under {_REPO_ROOT}: {missing}'
    return found


class _Site(NamedTuple):
    """One place the scan found an owned name. `detail` is the defined name
    for a definition, the source line for a call."""

    path: str
    lineno: int
    detail: str

    def __str__(self) -> str:
        return f'{self.path}:{self.lineno}: {self.detail}'


class _OwnedNameSites(NamedTuple):
    calls: tuple[_Site, ...]
    definitions: tuple[_Site, ...]


def _owned_name_sites(records: Sequence[ParsedFile]) -> _OwnedNameSites:
    """Module-level DEFINITIONS and bare-name CALLS of the owned names in *records*.

    Walks each record's already-parsed tree READ-ONLY (the trees are the
    session's shared ones) and does no I/O. A record that failed to parse is
    skipped here; the `owned_name_sites` fixture refuses one up front.

    AST-based, not grep-based: the retired fork and its history are discussed
    at length in comments and docstrings across `usage_gate.py` and
    `invocation_outcome.py`, and a textual scan would count that prose as
    callers. Only `ast.Call` nodes and MODULE-LEVEL `ast.FunctionDef` nodes
    count, so `_parse_resets_at_strict(...)` (a different name) and every
    mention in prose are excluded by construction.

    The substring pre-filter is an optimisation, never a narrowing: every node
    reported here carries an owned name as a literal identifier, so a file
    whose text contains neither name cannot hold one, and its tree is never
    walked.
    """
    calls: list[_Site] = []
    definitions: list[_Site] = []
    for record in records:
        tree = record.tree
        if tree is None or not any(name in record.source for name in _OWNED_NAMES):
            continue
        lines = record.source.splitlines()
        definitions.extend(
            _Site(record.relpath, node.lineno, node.name)
            for node in tree.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name in _OWNED_NAMES
        )
        # ast.walk is breadth-first, so a single file's hits are not
        # source-ordered; both lists are sorted before being returned.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name):
                called = func.id
            elif isinstance(func, ast.Attribute):
                called = func.attr
            else:
                continue
            if called in _OWNED_NAMES:
                calls.append(_Site(record.relpath, node.lineno, lines[node.lineno - 1].strip()))
    return _OwnedNameSites(tuple(sorted(calls)), tuple(sorted(definitions)))


@pytest.fixture(scope='module')
def owned_name_sites(first_party_tree: tuple[ParsedFile, ...]) -> _OwnedNameSites:
    """The ownership scan over the shared tree's production `<pkg>/src` records."""
    prefixes = tuple(f'{root}/' for root in _production_src_roots())
    production = [record for record in first_party_tree if record.relpath.startswith(prefixes)]
    unparseable = [record.relpath for record in production if record.syntax_error is not None]
    assert not unparseable, f'production module(s) that do not parse: {unparseable}'
    return _owned_name_sites(production)


class TestSingleResetsParserOwnership:
    """`shared.invocation_outcome` is the SINGLE owner of `_parse_resets_at`
    and `_extract_cap_message`.

    WHAT IS MECHANICALLY ENFORCED, stated exactly so the claim and the scan
    cannot drift apart: across EVERY workspace member's `<pkg>/src` tree, no
    module but the owner may define either name at module level or call either
    bare name; and `orchestrator/src/orchestrator/usage_gate.py` — the shim
    that used to re-export both — may not bind either name. Test trees,
    `scripts/` and prose anywhere are deliberately NOT scanned: a copy in one
    of those cannot reach production.

    Task 4042 moved the fabricating `usage_gate._parse_resets_at` fork's ONE
    live call site — `_handle_auth_failure` — onto the non-fabricating
    `shared.invocation_outcome` copy, because restoring the 401/403 body
    snippet re-armed the `'resets' in reason.lower()` branch: a body merely
    CONTAINING the substring "resets" without a parseable phrase would
    otherwise persist an invented `now + 1h` recovery time onto a revoked
    token (PRD 7.1.a — see
    `shared/tests/test_auth_failed.py::TestAuthFailedPersistsResetsAt`).

    Task 4357 then retired the fork itself, along with its caller-free
    `_extract_cap_message` twin and the orchestrator re-export that kept both
    importable. This class is what keeps the tree converged: with the fork
    gone, the failure mode that would restore the fabrication regression is
    RE-forking — someone re-adding a module-local parser, or re-exporting one
    — and nothing else in the tree catches that.

    Do not relax any assertion here. If a second copy is genuinely wanted,
    that is a design change: argue it, don't let a guard erode.
    """

    def test_scan_finds_the_owner_defining_and_calling_both_names(self, owned_name_sites):
        # Anti-vacuity: proves the AST walk resolves BOTH node kinds it
        # reports, so a future rename cannot turn the guards below into no-ops
        # that pass because they found nothing at all.
        sites = owned_name_sites
        assert sites.calls, 'AST scan found no call of either owned name — guard is vacuous'
        owner_defs = {s.detail for s in sites.definitions if s.path == _STRICT_PARSE_OWNER}
        assert owner_defs == set(_OWNED_NAMES), (
            f'AST scan did not find both owned names defined in {_STRICT_PARSE_OWNER} '
            f'(found {sorted(owner_defs)}) — the definition guard is vacuous.'
        )

    def test_only_the_owner_calls_the_bare_names(self, owned_name_sites):
        # Exact match on the structured `path`, not a prefix match on a
        # rendered string — and NO assertion on the call COUNT: a legitimate
        # new call inside the owner's own module must not fail this guard.
        offenders = [
            str(site) for site in owned_name_sites.calls
            if site.path != _STRICT_PARSE_OWNER
        ]
        assert offenders == [], (
            'new production caller(s) of a bare owned name: '
            f'{offenders}. Outside {_STRICT_PARSE_OWNER} the bare names resolve '
            'to nothing at all (NameError) — unless a module-local fork has '
            'been re-introduced, which is the regression this guards: the '
            'retired fork invented `now + 1h` on parse failure. Import the '
            'strict copy as `_parse_resets_at_strict` instead — see '
            'shared/tests/test_auth_failed.py::TestAuthFailedPersistsResetsAt for why.'
        )

    def test_handle_auth_failure_uses_the_single_parser(self):
        # Belt-and-braces on the specific regression: the module that owns
        # _handle_auth_failure must reach the strict copy under its alias.
        import shared.usage_gate as usage_gate_module

        assert usage_gate_module._parse_resets_at_strict is invocation_outcome._parse_resets_at

    def test_no_production_module_redefines_the_names(self, owned_name_sites):
        """No module but the owner may DEFINE either function.

        Scans every production tree rather than only `usage_gate.py` where
        the fork used to live: with the fork gone, a re-fork is as likely to
        appear in whichever module next wants to parse a reset string —
        `dashboard/src/dashboard/data/costs.py`, the consumer a fabricated
        value would mislead, being the pointed candidate.

        An AST `FunctionDef` scan rather than `hasattr`/`vars()`: a module
        legitimately imports the strict parser under an alias, and a name
        bound by `import ... as _parse_resets_at_strict` must not be mistaken
        for a local definition. Scanning defs also catches a re-fork that is
        never imported anywhere — i.e. before it has a caller for
        `test_only_the_owner_calls_the_bare_names` to find.
        """
        refork = [
            str(site) for site in owned_name_sites.definitions
            if site.path != _STRICT_PARSE_OWNER
        ]
        assert refork == [], (
            f'{refork} re-define(s) a name owned by {_STRICT_PARSE_OWNER}. Both '
            'live ONLY there; a second copy is exactly the un-guarded drift '
            'surface task 4357 removed, and the copy that used to live in '
            'shared/src/shared/usage_gate.py fabricated `now + 1h` on parse '
            'failure.'
        )

    def test_orchestrator_shim_does_not_reexport_the_retired_names(self, first_party_tree):
        """The orchestrator shim is public-surface-only.

        `shared.usage_gate.__all__` lists no underscore names, so the star
        import cannot reintroduce these — only an explicit re-export tuple
        could, which is how they used to survive there.

        An AST scan of the FILE rather than `import orchestrator.usage_gate`
        + `hasattr`, for two independent reasons. (1) Layering: `shared/`
        must not import `orchestrator/` (program decision #4 — the same note
        appears in `test_server_error.py` and
        `test_invocation_outcome_boundary.py`). (2) That import is not even
        available here: verify runs `cd shared && uv run pytest` FIRST in its
        chain, which syncs the workspace venv down to `shared`'s own
        dependencies, so `orchestrator` is not installed and the import
        raises `ModuleNotFoundError` — the guard would go red for a reason
        with nothing to do with what it guards, and go green again only if
        some earlier command happened to leave an `--all-packages` venv
        behind. Reading the source keeps the assertion true of the file,
        which is what "does not re-export" actually means. The AST walked is
        the session's shared one (`first_party_tree`).
        """
        shim_rel = 'orchestrator/src/orchestrator/usage_gate.py'
        shim = next((r for r in first_party_tree if r.relpath == shim_rel), None)
        assert shim is not None, (
            f'orchestrator usage_gate shim missing from the shared tree: {shim_rel}'
        )
        assert shim.tree is not None, f'{shim_rel} does not parse: {shim.syntax_error}'
        bound: set[str] = set()
        for node in shim.tree.body:
            if isinstance(node, ast.Import | ast.ImportFrom):
                # A `from ... import *` alias is literally named '*' and binds
                # nothing statically — it cannot reintroduce these two, since
                # shared.usage_gate.__all__ lists no underscore names.
                bound.update(alias.asname or alias.name for alias in node.names)
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                bound.add(node.name)
            elif isinstance(node, ast.Assign):
                bound.update(t.id for t in node.targets if isinstance(t, ast.Name))
        leaked = sorted(bound & set(_OWNED_NAMES))
        assert leaked == [], (
            f'orchestrator/src/orchestrator/usage_gate.py re-exports {leaked}. '
            f'Both live ONLY in {_STRICT_PARSE_OWNER}; re-exporting them from a '
            'second module is what let the retired fabricating fork stay '
            'importable.'
        )

    def test_shared_tree_spans_every_production_src_root(self, first_party_tree):
        """The shared tree must cover every DISCOVERED `<pkg>/src` root.

        The ownership scan walks the session's shared first-party tree, whose
        scope is a fixed list. This keeps the discovered contract: a package
        added later is caught here rather than silently left unscanned. Its
        complement, ``test_silent_fallthrough_gate.py::TestGateSelfIntegrity::test_every_scope_root_contributes_a_scanned_module``,
        checks the DECLARED roots.
        """
        relpaths = [record.relpath for record in first_party_tree]
        missing = [
            root_rel
            for root_rel in _production_src_roots()
            if not any(rel.startswith(f'{root_rel}/') for rel in relpaths)
        ]
        assert missing == [], (
            f'discovered production source root(s) absent from the shared '
            f'first-party tree: {missing}. Add them to '
            f'tests/repo_guards/silent_fallthrough_scan.py::SCOPE_ROOTS; never narrow '
            f'this discovery. (A `<pkg>/src` holding no Python at all also trips '
            f'this check.)'
        )

    def test_shared_tree_drops_no_production_module(self, first_party_tree):
        """Every `*.py` under a discovered `<pkg>/src` root is a shared-tree record.

        The shared tree applies `iter_first_party_files`' test-code exclusions:
        path parts named `tests`, `mem0` or `graphiti`, and files named
        `test_*.py` or `conftest.py`. No production module matches them today.
        One that ever did would be silently skipped by the ownership scan, so a
        re-fork hidden there would pass a guard that claims every `<pkg>/src`.
        """
        in_tree = {record.relpath for record in first_party_tree}
        dropped = sorted(
            relpath
            for root_rel in _production_src_roots()
            for path in (_REPO_ROOT / root_rel).rglob('*.py')
            if (relpath := path.relative_to(_REPO_ROOT).as_posix()) not in in_tree
        )
        assert dropped == [], (
            f'production module(s) under a discovered `<pkg>/src` root that the '
            f'shared first-party tree excludes: {dropped}. The ownership scan '
            f'cannot see them. Rename them out of the exclusions in '
            f'tests/repo_guards/silent_fallthrough_scan.py::iter_first_party_files, or '
            f'narrow those exclusions; never let this scan narrow silently.'
        )


class TestOwnershipScanUsesTheSharedTree:
    """The ownership scan walks the session's shared ASTs and does no I/O of its own."""

    def test_the_ownership_scan_reads_nothing_and_parses_nothing(
        self, first_party_tree, monkeypatch
    ):
        """Scoped by ``monkeypatch.context()``: pytest's own failure report calls
        ``ast.parse``, so the patch must be gone before a failure is rendered."""

        def no_parse(*_args, **_kwargs):
            raise AssertionError('ast.parse called: the ownership scan re-parsed a file')

        def no_read(*_args, **_kwargs):
            raise AssertionError('pathlib.Path.read_text called: the ownership scan re-read a file')

        with monkeypatch.context() as patched:
            patched.setattr(ast, 'parse', no_parse)
            patched.setattr(Path, 'read_text', no_read)
            sites = _owned_name_sites(first_party_tree)
        owner_defs = {s.detail for s in sites.definitions if s.path == _STRICT_PARSE_OWNER}
        assert owner_defs == set(_OWNED_NAMES)
