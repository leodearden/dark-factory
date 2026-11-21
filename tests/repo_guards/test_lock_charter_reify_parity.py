"""The lock-charter predicate's cross-source tier: shared.locking against reify and the tracked indexes.

Split out of ``shared/tests/test_locking.py`` (task 5654), which keeps the
pure ``shared.locking`` unit tests, the Tier-1 canonical-vector pins among them.
The drift guards and sweeps here read outside shared: reify's
``scripts/lock-charter-guard.sh`` emitters, and ``git ls-files`` over this
checkout and the reify sibling. shared's verify cache key covers neither, so
they live in this never-cached module. The other classes test the
reify-resolution support those guards run on, which lives here with them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import shared.reify_checkout as reify_checkout
import shared.testing_reify_layout as reify_layout
from shared.locking import EXTENSIONLESS_FILENAMES, FILE_EXTENSIONS, is_file_path
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

# ---------------------------------------------------------------------------
# _resolve_reify_checkout — marker-bound adapter over `shared.reify_checkout`,
# the single source of layout-independent reify-checkout discovery AND of the
# skip wording that goes with it (established by task 3843 in the γ suite,
# fused-memory/tests/test_lock_charter_guard.py, and promoted to shared/ by
# task 3978 so orchestrator's scripts/verify.sh gate resolves through the same
# walk).  Feeds the module-level constants below once they are rewired to it.
# ---------------------------------------------------------------------------

_REIFY_GUARD_RELPATH = Path('scripts') / 'lock-charter-guard.sh'


def _resolve_reify_checkout(start: Path | None = None) -> reify_checkout.ReifyCheckout:
    """Locate the reify checkout carrying THIS module's guard script.

    Marker-bound adapter over the shared single source.  Every semantic —
    discovery order, REIFY_ROOT precedence, the measured worktree-vs-bare-
    checkout evidence table (and why "change parents[5] to parents[3]" is a
    REGRESSION rather than a fix), and why an override absent on disk is
    honored VERBATIM — lives in `shared.reify_checkout.resolve_reify_checkout`.
    Do not restate or re-derive them here, and do not reintroduce a local copy
    of the ancestor walk.

    Returns the shared `ReifyCheckout` — root AND provenance — rather than a
    bare path, so the skip reason is formatted from the very resolution that
    produced the root, instead of re-deriving provenance separately and
    risking the two disagreeing.

    *start* defaults to THIS file, and that default belongs here rather than
    in the shared helper: the walk has to begin at the CALL SITE, and shared's
    own ``__file__`` would start it in ``shared/src/shared/`` instead of here.
    """
    # Called through the module attribute on purpose: the delegation pin in
    # TestReifyCheckoutAdapter patches `reify_checkout.resolve_reify_checkout`,
    # which a `from ... import resolve_reify_checkout` binding would put out of
    # its reach.
    return reify_checkout.resolve_reify_checkout(
        _REIFY_GUARD_RELPATH, start=start or Path(__file__)
    )


# Repo roots for the tracked-corpus self-audit sweep.  dark-factory is
# resolved from this file (works in a worktree too: <root>/tests/repo_guards/ ->
# parents[2] — deliberately UNCHANGED here, since parents[2] is correct in
# BOTH a bare checkout and a worktree: it names the checkout THIS test is
# running inside, main root or worktree root, either of which is a valid
# `git ls-files` target).  reify is a sibling checkout, resolved
# layout-independently via `shared.reify_checkout` (through the
# `_resolve_reify_checkout` adapter above) rather than a fixed parents[N]
# index — see that module's docstring for the measured worktree-vs-bare
# evidence table.  These constants are evaluated at IMPORT time, so
# REIFY_ROOT must be exported BEFORE pytest starts to steer them; a
# mid-session `monkeypatch.setenv` only affects direct
# `_resolve_reify_checkout()` calls (exactly what TestReifyCheckoutAdapter
# does above; it never touches these constants).
_DF_REPO_ROOT = Path(__file__).parents[2]
_REIFY_CHECKOUT = _resolve_reify_checkout()
_REIFY_REPO_ROOT: Path | None = _REIFY_CHECKOUT.root
_REIFY_GUARD_SCRIPT: Path | None = (
    _REIFY_REPO_ROOT / _REIFY_GUARD_RELPATH if _REIFY_REPO_ROOT is not None else None
)
# The skip wording comes from the shared builder too, not a hand-rolled
# string: it separates a discovery MISS (nobody has reify checked out here —
# benign) from a REIFY_ROOT naming a path that is not there (an operator
# typo, which must NAME the bad path).  Matches the wording the γ suite and
# orchestrator's verify gate emit for the same condition.
_REIFY_SKIP_REASON: str | None = reify_checkout.reify_skip_reason(
    _REIFY_GUARD_RELPATH, _REIFY_REPO_ROOT, named_by_env=_REIFY_CHECKOUT.named_by_env
)


def _reify_guard_vector(flag: str) -> list[str]:
    """Return the sorted vector emitted by the reify guard script's *flag*.

    Verbatim counterpart of the γ suite's helper of the same name, duplicated
    for the same reason the corpora are: the two suites are not linked by an
    import, and each must be able to pin its own copy of the predicate against
    bash on its own.

    LOUD on failure.  The callers skipif on the script's ABSENCE — a standalone
    checkout legitimately has no reify sibling — but a script that EXISTS and
    then fails is real signal and is raised, never skipped.  Measured
    2026-08-09: an unrecognised flag prints usage and exits 2, so a
    skip-on-any-non-zero would retire the guard the moment reify renamed an
    emitter, leaving it green forever while enforcing nothing.
    """
    assert _REIFY_GUARD_SCRIPT is not None, (
        'every caller is skipif-guarded on _REIFY_GUARD_SCRIPT being a real '
        'file, so reaching here with None means a caller lost its guard — fail '
        'loudly rather than stringify None into a bogus `bash None` invocation'
    )
    result = subprocess.run(
        ['bash', str(_REIFY_GUARD_SCRIPT), flag],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f'reify {_REIFY_GUARD_SCRIPT.name} {flag} exited {result.returncode}, '
        f'expected 0 — the emitter this cross-source drift guard depends on is '
        f'gone, renamed, or broken.  Check `bash {_REIFY_GUARD_SCRIPT} --help`.  '
        f'Hard failure, not a skip: silently retiring a cross-source guard would '
        f'let the three copies of the predicate diverge with every test green.\n'
        f'  stdout: {result.stdout[:400]!r}\n'
        f'  stderr: {result.stderr[:400]!r}'
    )
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


# ---------------------------------------------------------------------------
# GITLINK EXCLUSION (load-bearing, not a measurement artefact): entries with
# mode 160000 are submodule mount points — `graphiti` and `mem0` in
# dark-factory — and are deliberately NOT admitted.  They are extension-less
# and would otherwise qualify, but admitting them would let a task declare an
# entire vendored submodule as its lock charter, which is strictly worse than
# the bug being fixed.  They stay DIRECTORIES by design.  The sweep below
# asserts this filter is doing real work rather than passing vacuously.
# ---------------------------------------------------------------------------

def _tracked_entries(repo_root: Path) -> tuple[list[str], list[str]] | None:
    """``(non_gitlink_paths, gitlink_paths)`` for *repo_root*, or None if not a checkout.

    ``git ls-files -s`` (not plain ``ls-files``) is what makes the mode column
    available.  Mode ``160000`` means a submodule gitlink; those entries are
    returned SEPARATELY rather than discarded, because the sweeps need them to
    prove the exclusion is doing real work.  See the GITLINK EXCLUSION note
    above for why the filter is load-bearing rather than cosmetic.
    """
    result = subprocess.run(
        ['git', '-C', str(repo_root), 'ls-files', '-s', '-z'],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
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


def _tracked_file_paths(repo_root: Path) -> list[str] | None:
    """Tracked NON-GITLINK paths under *repo_root*, or None if not a checkout."""
    entries = _tracked_entries(repo_root)
    return None if entries is None else entries[0]


def _skip_unless_checkout(repo: str, repo_root: Path | None) -> Path:
    """Skip unless *repo_root* names a real checkout directory; else return it.

    Marker-bound adapter over `shared.reify_checkout.checkout_skip_reason`,
    which owns BOTH arms' wording — why a discovery MISS is not an operator's
    REIFY_ROOT typo, why the set-but-absent arm names the path verbatim, and
    why it does not borrow the stronger marker-based reason.  Do not restate or
    re-derive them here, and do not reintroduce a local copy of either arm; the
    same helper in fused-memory/tests/test_lock_charter_guard.py delegates to
    the identical builder, which is the point (task 4259).

    What is LOCAL is only the binding: this call site's guard-script marker,
    and turning the reason into a ``pytest.skip`` — the shared module is
    pytest-free.

    Called through the module attribute on purpose, for the same reason
    `_resolve_reify_checkout` notes: the delegation pin in
    TestSkipUnlessCheckout patches `reify_checkout.checkout_skip_reason`, which
    a ``from ... import`` binding would put out of its reach.
    """
    reason = reify_checkout.checkout_skip_reason(
        repo, repo_root, marker=_REIFY_GUARD_RELPATH
    )
    if reason is not None:
        pytest.skip(reason)
    assert repo_root is not None  # narrows for the return
    return repo_root


class TestReifyCheckoutAdapter:
    """`_resolve_reify_checkout` — this module's binding to the shared resolver."""

    def test_resolver_delegates_to_the_shared_single_source(self, tmp_path, monkeypatch):
        """The ancestor walk must live in ONE place — shared.reify_checkout.

        Deliberately white-box, and deliberately the only genuinely-RED
        assertion available for what is otherwise a behaviour-preserving
        refactor: in the ``.worktrees/<id>`` layout this suite normally runs
        in, ``parents[5]`` already resolves correctly, so every BLACK-BOX
        assertion in this file stays green both before and after this module
        grows its own copy of the ancestor walk — a faithful local duplicate
        would satisfy them all while defeating the single-sourcing this task
        exists to establish (the task's "do not each grow a private copy of
        the resolver" directive). What this pin buys is that a SECOND copy of
        the walk cannot quietly reappear in this file.
        """
        sentinel = tmp_path / 'sentinel-reify'
        start = tmp_path / 'dark-factory' / 'shared' / 'tests' / 'test_x.py'
        seen = {}

        def _stub(marker, start=None):
            seen['marker'] = Path(marker)
            seen['start'] = start
            return reify_checkout.ReifyCheckout(sentinel, False)

        monkeypatch.setattr(reify_checkout, 'resolve_reify_checkout', _stub)

        assert _resolve_reify_checkout(start).root == sentinel, (
            'this module must DELEGATE to '
            'shared.reify_checkout.resolve_reify_checkout rather than keep a '
            'private copy of the ancestor walk'
        )
        assert seen['marker'] == _REIFY_GUARD_RELPATH, (
            'the shared resolver must be bound to THIS call site marker '
            f'(expected {_REIFY_GUARD_RELPATH}, got {seen.get("marker")})'
        )
        assert seen['start'] == start

    def test_module_constants_track_the_resolver(self):
        """Single-source-of-truth pin against the call sites re-diverging.

        Covers the skip reason too: it must be the SHARED builder's output
        for this module's own resolution, not a hand-rolled string that can
        drift from the wording the γ suite or orchestrator's verify gate use
        for the same condition.  Checked two ways: an INDEPENDENT invariant
        first — read straight off the filesystem, mirroring
        ``reify_skip_reason``'s own documented contract — and only then the
        recomputed-call equality.  The recomputation alone would catch the
        constants re-diverging from a fresh call but would credit a
        hand-rolled string that happened to match it; the independent half
        below cannot be fooled that way.

        RED today (before the constants are rewired): ``_REIFY_SKIP_REASON``
        does not exist yet in this module, so referencing it below raises
        ``NameError`` regardless of whether the first assertion happens to
        agree by coincidence in this developer's worktree.
        """
        checkout = _resolve_reify_checkout()

        assert checkout.root == _REIFY_REPO_ROOT
        if _REIFY_REPO_ROOT is not None:
            assert _REIFY_GUARD_SCRIPT == _REIFY_REPO_ROOT / _REIFY_GUARD_RELPATH
            assert _REIFY_GUARD_SCRIPT is not None  # narrows for .is_file() below
            if _REIFY_GUARD_SCRIPT.is_file():
                assert _REIFY_SKIP_REASON is None, (
                    f'the guard script is present at {_REIFY_GUARD_SCRIPT}, so '
                    f'the gate must ADMIT the run: {_REIFY_SKIP_REASON!r}'
                )
            else:
                assert _REIFY_SKIP_REASON is not None
                assert str(_REIFY_REPO_ROOT) in _REIFY_SKIP_REASON, (
                    f'the reason must name the resolved root: '
                    f'{_REIFY_SKIP_REASON!r}'
                )
        else:
            assert _REIFY_SKIP_REASON is not None
            assert 'REIFY_ROOT' in _REIFY_SKIP_REASON, (
                f'the reason must name the override: {_REIFY_SKIP_REASON!r}'
            )
            assert 'scripts/lock-charter-guard.sh' in _REIFY_SKIP_REASON, (
                f'the reason must name the marker: {_REIFY_SKIP_REASON!r}'
            )

        shared_reason = reify_checkout.reify_skip_reason(
            _REIFY_GUARD_RELPATH, checkout.root, named_by_env=checkout.named_by_env
        )
        assert shared_reason == _REIFY_SKIP_REASON


# ---------------------------------------------------------------------------
# Planted-synthetic-layout harness — `shared.testing_reify_layout` (task 4259,
# consolidating this copy with orchestrator's).  Why a real reify checkout
# cannot express the defect, why the RED is host-independent in both
# directions, and how a copy's imports resolve from the parent process while
# its ``__file__`` is the planted path: all in that module's docstring.  Only
# the local binding lives here.
# ---------------------------------------------------------------------------


@pytest.fixture
def planted_env(monkeypatch):
    """Neutralize the ambient env that would otherwise steer a module copy.

    REIFY_ROOT would override the discovery the planted-layout cases exist to
    exercise.  Returns monkeypatch so a case can set REIFY_ROOT back to a
    value it chose itself.
    """
    monkeypatch.delenv('REIFY_ROOT', raising=False)
    return monkeypatch


def _load_module_copy(tmp_path: Path, tests_relpath: str, *, plant_guard: bool = True):
    """Import a COPY of THIS module from a synthetic checkout layout.

    Marker-bound adapter over `shared.testing_reify_layout.plant_reify_layout`,
    which owns every semantic — what gets planted, where the copy is written,
    why its ``__file__`` must be the planted path, the ``sys.modules`` pop, and
    the ambient-contamination precondition on the ``plant_guard=False`` arm.
    Do not restate or re-derive them here, and do not reintroduce a local copy
    of the planting or the ``spec_from_file_location`` dance.

    What is LOCAL is only the binding: THIS module's ``__file__`` (the walk has
    to start at the CALL SITE) and its guard-script marker.  ``plant_guard=``
    keeps its local spelling for the one call site that passes it.

    Called through the module attribute on purpose: the delegation pin in
    TestPlantedLayoutHarnessAdapter patches
    `reify_layout.plant_reify_layout`, which a ``from ... import`` binding
    would put out of its reach.
    """
    return reify_layout.plant_reify_layout(
        __file__,
        tmp_path,
        tests_relpath,
        marker=_REIFY_GUARD_RELPATH,
        plant_marker=plant_guard,
    )


# Literals, not the `shared.testing_reify_layout` builders: those render
# ``<package>/tests``, which cannot spell this module's ``tests/repo_guards``.
_BARE_LAYOUT = 'dark-factory/tests/repo_guards'
_WORKTREE_LAYOUT = 'dark-factory/.worktrees/4080/tests/repo_guards'


def _with_worktree_segments(layout: str, task_id: str) -> str:
    """*layout* with ``.worktrees/<task_id>`` inserted after its checkout segment."""
    checkout, _, tests_dir = layout.partition('/')
    return f'{checkout}/.worktrees/{task_id}/{tests_dir}'


class TestPlantedLayoutHarnessAdapter:
    """`_load_module_copy` — this module's binding to the shared harness.

    The same shape, one layer up, as `TestReifyCheckoutAdapter` below the
    resolver: deliberately white-box, and deliberately the only genuinely-RED
    assertion available for what is otherwise a behaviour-preserving refactor.
    Every BLACK-BOX assertion in `TestReifyCheckoutPlantedLayout` stays green
    both before and after this module grows a private copy of the planting +
    ``spec_from_file_location`` dance, because a faithful duplicate behaves
    identically — which is exactly how the two copies task 4259 consolidates
    came to diverge unnoticed in the first place.  What these pins buy is that
    a SECOND copy of the harness cannot quietly reappear here.
    """

    @pytest.mark.parametrize('layout', [_BARE_LAYOUT, _WORKTREE_LAYOUT])
    def test_load_module_copy_delegates_to_the_shared_single_source(
        self, tmp_path, monkeypatch, layout
    ):
        """The planting + loading must live in ONE place.

        RED before step-6: the local helper runs its own planting and
        ``spec_from_file_location`` dance and returns a really-loaded module,
        so the sentinel never comes back.
        """
        sentinel = object()
        seen = {}

        def _stub(source, tmp, tests_relpath, *, marker, plant_marker=True):
            seen.update(
                source=source,
                tmp_path=tmp,
                tests_relpath=tests_relpath,
                marker=marker,
                plant_marker=plant_marker,
            )
            return sentinel

        monkeypatch.setattr(reify_layout, 'plant_reify_layout', _stub)

        assert _load_module_copy(tmp_path, layout) is sentinel, (
            'this module must DELEGATE to '
            'shared.testing_reify_layout.plant_reify_layout rather than keep a '
            'private copy of the planted-layout harness'
        )
        assert Path(seen['source']) == Path(__file__), (
            'the harness must be bound to THIS module\'s file — the walk has to '
            f'start at the CALL SITE (got {seen.get("source")!r})'
        )
        assert seen['tmp_path'] == tmp_path
        assert seen['tests_relpath'] == layout
        assert seen['marker'] == _REIFY_GUARD_RELPATH, (
            'the shared harness must be bound to THIS call site\'s marker '
            f'(expected {_REIFY_GUARD_RELPATH}, got {seen.get("marker")!r})'
        )
        assert seen['plant_marker'] is True

    def test_plant_guard_false_reaches_the_shared_harness(self, tmp_path, monkeypatch):
        """The local ``plant_guard=`` kwarg carries through as ``plant_marker=``.

        The local spelling is preserved for the existing call site; the shared
        one is where the meaning lives.  RED before step-6: the local helper
        never calls the shared harness at all.
        """
        seen = {}

        def _stub(source, tmp, tests_relpath, *, marker, plant_marker=True):
            seen['plant_marker'] = plant_marker
            return object()

        monkeypatch.setattr(reify_layout, 'plant_reify_layout', _stub)

        _load_module_copy(tmp_path, _BARE_LAYOUT, plant_guard=False)

        assert seen['plant_marker'] is False

    def test_layout_constants_differ_only_by_the_worktree_segments(self):
        """The worktree layout is the bare one plus ``.worktrees/<id>``, as the builders shape it.

        The constants are literals (see their comment), so they cannot be
        compared with the builders' output. Instead the builders' own insertion
        is checked for a package they can spell, and the literals must follow
        it: a builder change that moves the two segments fails here rather than
        leaving these literals silently describing a different layout.
        """
        assert reify_layout.worktree_layout('shared', '4080') == _with_worktree_segments(
            reify_layout.bare_layout('shared'), '4080'
        )
        assert _with_worktree_segments(_BARE_LAYOUT, '4080') == _WORKTREE_LAYOUT


class TestReifyCheckoutPlantedLayout:
    """The four cross-repo guards, proven ARMED from a bare-checkout layout.

    Every case here loads a COPY of this very module from a synthetic
    ancestry — see the harness comment above for why a real checkout cannot
    express the defect.
    """

    def test_gate_resolves_against_the_planted_checkout_not_this_machine(
        self, tmp_path, planted_env
    ):
        """The guard must find the reify sibling of the tree it is RUNNING in.

        Not of this developer's machine.  RED today: the bare-layout copy's
        constants are still ``parents[5]``-derived, which lands off-tree for
        this layout (measured: the ancestor five levels up from a file at
        ``<tmp>/src/dark-factory/shared/tests/`` is outside ``<tmp>`` itself).
        """
        mod = _load_module_copy(tmp_path, _BARE_LAYOUT)

        expected = tmp_path / 'src' / 'reify'
        assert expected == mod._REIFY_REPO_ROOT, (
            f'resolved {mod._REIFY_REPO_ROOT!r} instead of the reify checkout '
            f'planted beside it at {expected!r} — a fixed parents[N] index '
            f"answers for this developer's machine rather than for the tree "
            f'under test'
        )
        assert expected / 'scripts' / 'lock-charter-guard.sh' == mod._REIFY_GUARD_SCRIPT

    def test_worktree_and_bare_layouts_resolve_to_the_same_planted_checkout(
        self, tmp_path, planted_env
    ):
        """No fixed parents[N] index can satisfy both layouts.

        RED today: the worktree-layout copy's ``parents[5]`` happens to be
        correct here (mirroring why the real worktree suite sees 0 skips),
        but the bare-layout copy's does not, so the two disagree.
        """
        bare = _load_module_copy(tmp_path, _BARE_LAYOUT)
        worktree = _load_module_copy(tmp_path, _WORKTREE_LAYOUT)

        expected = tmp_path / 'src' / 'reify'
        assert expected == worktree._REIFY_REPO_ROOT
        assert worktree._REIFY_REPO_ROOT == bare._REIFY_REPO_ROOT, (
            f"'.worktrees/<id>' adds exactly two path segments, so a fixed "
            f'parents[N] cannot be correct in both layouts (got '
            f'worktree={worktree._REIFY_REPO_ROOT!r} vs bare={bare._REIFY_REPO_ROOT!r})'
        )

    def test_gate_admits_the_run_from_a_planted_checkout(self, tmp_path, planted_env):
        """Resolution is not enough — the skip decision must ADMIT the run.

        RED today: ``_REIFY_SKIP_REASON`` does not exist in this module yet.
        """
        mod = _load_module_copy(tmp_path, _BARE_LAYOUT)

        assert mod._REIFY_SKIP_REASON is None, (
            f'the gate would SKIP against a planted checkout that really '
            f'carries scripts/lock-charter-guard.sh: {mod._REIFY_SKIP_REASON!r}'
        )

    def test_four_cross_repo_guards_are_armed_in_a_bare_checkout(self, tmp_path, planted_env):
        """The direct expression of "silently skipping 4 cross-repo lock guards".

        Reads the DECORATED functions' evaluated ``pytestmark`` on the
        bare-layout module copy: decorators evaluate their arguments at
        IMPORT time, which is exactly the moment this bug silently flips a
        guard off. The two Tier-2 ``skipif`` conditions must evaluate False
        (armed, not silently retired), and the two parametrized sweeps'
        ``reify`` argvalue must carry the checkout planted BESIDE the bare
        copy, not an off-tree answer.

        RED today: the bare copy's ``_REIFY_GUARD_SCRIPT`` is an off-tree,
        non-existent path, so both skipif conditions evaluate True (SKIPPED),
        and both sweeps' ``reify`` argvalue is that same off-tree
        ``_REIFY_REPO_ROOT`` rather than the planted checkout.
        """
        mod = _load_module_copy(tmp_path, _BARE_LAYOUT)
        expected_root = tmp_path / 'src' / 'reify'

        def _skipif_condition(func):
            marks = [m for m in func.pytestmark if m.name == 'skipif']
            assert marks, f'{func.__name__} must carry a skipif mark'
            return marks[0].args[0]

        def _parametrize_reify_argvalue(func):
            # Select by argnames rather than blindly taking marks[0]: a
            # stacked second parametrize (e.g. a flag axis) would otherwise
            # let this silently pick an arbitrary mark.
            marks = [
                m
                for m in func.pytestmark
                if m.name == 'parametrize' and 'repo_root' in m.args[0]
            ]
            assert len(marks) == 1, (
                f"{func.__name__} must carry exactly one parametrize mark "
                f"whose argnames include 'repo_root' (found {len(marks)}) — "
                f'a restructure of this sweep must fail here with a message '
                f'naming what changed, not with an opaque error from '
                f'indexing marks[0]'
            )
            # Tolerate rows wrapped in pytest.param(...): dict(argvalues)
            # would raise ValueError the moment any row carries marks/id
            # instead of being a bare 2-tuple.
            rows = [getattr(argvalue, 'values', argvalue) for argvalue in marks[0].args[1]]
            reify_rows = [row for row in rows if row[0] == 'reify']
            assert len(reify_rows) == 1, (
                f"{func.__name__} parametrize must carry exactly one row "
                f"whose first element is 'reify' (found {len(reify_rows)} "
                f'in {rows!r})'
            )
            return reify_rows[0][1]

        assert (
            _skipif_condition(
                mod.TestExtensionlessFilenamesDriftGuard.test_extensionless_drift_guard_vs_reify_script
            )
            is False
        ), 'guard 1/4: extensionless drift Tier-2 skipif must be ARMED (condition False) in a bare checkout'
        assert (
            _skipif_condition(
                mod.TestFileExtensionsDriftGuard.test_extension_drift_guard_vs_reify_script
            )
            is False
        ), 'guard 2/4: extension drift Tier-2 skipif must be ARMED (condition False) in a bare checkout'
        assert (
            _parametrize_reify_argvalue(
                mod.TestExtensionlessFilenames.test_no_canonical_name_is_also_a_directory
            )
            == expected_root
        ), 'guard 3/4: reify parametrization must carry the planted checkout, not an off-tree answer'
        assert (
            _parametrize_reify_argvalue(
                mod.TestExtensionlessFilenames.test_every_tracked_extensionless_file_is_allowlisted
            )
            == expected_root
        ), 'guard 4/4: reify parametrization must carry the planted checkout, not an off-tree answer'

    def test_discovery_miss_skips_instead_of_answering_from_this_machine(
        self, tmp_path, planted_env
    ):
        """THE structural pin against a hardcoded/relative default.

        Nothing named reify exists anywhere in the planted ancestry, so the
        honest answer is None + a skip.  A fixed parents[N] index instead
        always produces SOME path — possibly one that exists on this
        developer's machine — so the guard would silently run (or silently
        retire) against a checkout unrelated to the tree under test.

        RED today, structurally: the current ``_REIFY_REPO_ROOT`` is a raw
        ``parents[N] / 'reify'`` expression that is never None regardless of
        whether anything exists on disk at that path, and
        ``_REIFY_SKIP_REASON`` does not exist yet.

        This case's RED depends on an environmental precondition: no ancestor
        of the copied module may carry ``reify/scripts/lock-charter-guard.sh``.
        That precondition is enforced by the harness itself as of task 4259 —
        `plant_reify_layout` raises `AmbientReifyCheckoutError` naming the
        contaminating ancestors on the ``plant_guard=False`` arm — so an
        unusual ``--basetemp`` or an ambient ``/tmp/reify`` checkout fails with
        a message blaming the environment rather than the resolver, and it does
        so for orchestrator's copy of this case too.
        """
        mod = _load_module_copy(tmp_path, _BARE_LAYOUT, plant_guard=False)

        assert mod._REIFY_REPO_ROOT is None, (
            f'no reify checkout exists in the planted ancestry, but the '
            f'module resolved {mod._REIFY_REPO_ROOT!r} — an off-tree answer'
        )
        reason = mod._REIFY_SKIP_REASON
        assert isinstance(reason, str) and reason
        assert 'REIFY_ROOT' in reason, f'the skip must name the override: {reason!r}'
        assert 'scripts/lock-charter-guard.sh' in reason, (
            f'the skip must name the marker: {reason!r}'
        )

    def test_env_override_wins_over_the_planted_checkout_and_names_the_bad_path(
        self, tmp_path, planted_env
    ):
        """REIFY_ROOT must WIN over a discoverable planted sibling, verbatim.

        Ported from orchestrator/tests/test_verify_role_integration.py's
        ``test_env_override_is_honored_verbatim_and_names_the_bad_path`` (task
        3978).  Closes the one arm the ``planted_env`` fixture's docstring has
        always promised ("a case can set REIFY_ROOT back to a value it chose
        itself") but that no case here previously exercised: every other case
        in this class only relies on the fixture's ``delenv``.  Without this,
        ``_REIFY_REPO_ROOT`` / ``_REIFY_GUARD_SCRIPT`` / ``_REIFY_SKIP_REASON``
        had no end-to-end coverage of the override arm at all — only
        transitively, via the delegation pin in TestReifyCheckoutAdapter and
        shared/tests/test_reify_checkout.py.  A future edit that pre-resolved
        the root by discovery and only consulted REIFY_ROOT as a fallback
        would leave every other test in this file green.

        A typo'd REIFY_ROOT that silently fell back to a discoverable sibling
        would answer for a DIFFERENT repo than the operator named; honoring it
        verbatim instead makes the typo self-evident in ``pytest -rs`` output.
        """
        bad_root = tmp_path / 'does-not-exist' / 'reify-typo'
        planted_env.setenv('REIFY_ROOT', str(bad_root))

        mod = _load_module_copy(tmp_path, _BARE_LAYOUT)

        assert bad_root.resolve() == mod._REIFY_REPO_ROOT, (
            f'REIFY_ROOT must win over the discoverable planted sibling, not '
            f'be shadowed by it (got {mod._REIFY_REPO_ROOT!r})'
        )
        assert (tmp_path / 'src' / 'reify') != mod._REIFY_REPO_ROOT, (
            "a typo'd REIFY_ROOT silently falling back to a discovered "
            'checkout would answer for a DIFFERENT repo than the operator named'
        )
        reason = mod._REIFY_SKIP_REASON
        assert isinstance(reason, str) and reason
        assert str(bad_root) in reason, (
            f'the skip reason must name the bad path verbatim so a REIFY_ROOT '
            f'typo is self-evident in `pytest -rs` output (got {reason!r})'
        )


class TestSkipUnlessCheckout:
    """`_skip_unless_checkout` — the reachability precondition step-6 wires
    both reify-parametrized sweeps through, so a None or set-but-absent reify
    root SKIPS honestly instead of raising ``AttributeError`` at the sweeps'
    inline ``repo_root.is_dir()``.

    RED today: `_skip_unless_checkout` does not exist in this module, so
    every case below fails with ``NameError``.  Ported in shape from
    fused-memory/tests/test_lock_charter_guard.py:721-762's helper of the
    same name, but pinned MORE thoroughly than that suite pins its own copy:
    the γ suite only exercises the helper indirectly through its own
    parametrized sweeps, which never observe the ``None`` arm on a host where
    reify IS discoverable.  These cases call the helper directly, so the
    ``None`` arm is pinned host-independently regardless of what any given
    machine has checked out.
    """

    def test_discovery_miss_skips_with_the_shared_wording(self):
        """A None root is the discovery-miss arm — nobody has a reify sibling.

        The wording must come from the shared builder, not a local string, so
        it cannot drift from what the Tier-2 skipifs above and orchestrator's
        verify gate say about the same condition.  ``pytest.skip.Exception``
        is pytest's own public handle for the outcome ``pytest.skip`` raises
        (documented on ``pytest.skip`` itself), so capturing it here needs no
        dependency on the private ``_pytest.outcomes`` import path.

        Checked two ways: an INDEPENDENT invariant first — the reason must
        name the override var and the marker, the same fact
        TestReifyCheckoutPlantedLayout's
        ``test_discovery_miss_skips_instead_of_answering_from_this_machine``
        pins on a planted copy — and only then the recomputed-call equality,
        which is what actually proves the wording is DELEGATED rather than
        merely similar-looking.
        """
        with pytest.raises(pytest.skip.Exception) as excinfo:
            _skip_unless_checkout('reify', None)

        reason = str(excinfo.value)
        assert isinstance(reason, str) and reason, (
            f'a None root is the discovery-miss arm, which always yields a '
            f'reason from the shared builder — a falsy one here would turn '
            f'this skip into a phantom pass, got {reason!r}'
        )
        assert 'REIFY_ROOT' in reason, f'the skip must name the override: {reason!r}'
        assert 'scripts/lock-charter-guard.sh' in reason, (
            f'the skip must name the marker: {reason!r}'
        )
        assert reason == reify_checkout.reify_skip_reason(
            _REIFY_GUARD_RELPATH, None, named_by_env=False
        ), 'the discovery-miss wording must come from the shared builder, not a local string'
        assert reason == reify_checkout.checkout_skip_reason(
            'reify', None, marker=_REIFY_GUARD_RELPATH
        ), (
            'and it must route through `checkout_skip_reason`, so BOTH arms of '
            'one skip decision keep a single home'
        )

    def test_set_but_absent_root_skips_naming_the_path(self, tmp_path):
        """A REIFY_ROOT-shaped path that is not on disk must be NAMED in the
        skip reason, so an operator's typo is self-evident in `pytest -rs`
        output instead of silently falling back to discovery.

        Same two-way shape as the case above, and as
        `test_module_constants_track_the_resolver` documents: the INDEPENDENT
        invariant first (the reason names the bad path), then the
        recomputed-call equality against the shared builder.
        """
        missing = tmp_path / 'no-such-reify-checkout'

        with pytest.raises(pytest.skip.Exception) as excinfo:
            _skip_unless_checkout('reify', missing)

        reason = str(excinfo.value)
        assert str(missing) in reason, (
            f'the skip reason must name the bad path so a REIFY_ROOT typo is '
            f'self-evident: {reason!r}'
        )
        assert reason == reify_checkout.checkout_skip_reason(
            'reify', missing, marker=_REIFY_GUARD_RELPATH
        ), (
            'the set-but-absent wording must come from the shared builder, not '
            'a local f-string'
        )

    def test_present_directory_is_returned_unchanged(self, tmp_path):
        """A real directory must be RETURNED, not skipped.

        Otherwise the helper could skip everything vacuously and the sweeps
        it guards would never run against a real checkout.
        """
        assert _skip_unless_checkout('reify', tmp_path) == tmp_path

    def test_skip_unless_checkout_delegates_to_the_shared_builder(
        self, tmp_path, monkeypatch
    ):
        """The skip DECISION must come from `checkout_skip_reason`, not a local copy.

        The genuinely-RED pin for this helper, in the shape
        `TestReifyCheckoutAdapter::test_resolver_delegates_to_the_shared_single_source`
        established: the recomputed-equality assertions in the two cases above
        cannot tell delegation from a hand-rolled string that happens to agree,
        so they stay green whether or not the call is made — which is why they
        ride along with those cases' independent invariants rather than
        standing as cases of their own.  This one CAN tell: it patches the
        shared builder to answer for a root that is a perfectly good directory,
        which a local implementation would ADMIT.

        RED before step-6: the local helper decides both arms itself, so a real
        directory is returned unchanged and no skip is raised.
        """
        seen = {}

        def _stub(repo, root, *, marker):
            seen.update(repo=repo, root=root, marker=marker)
            return 'stub reason from the shared builder'

        monkeypatch.setattr(reify_checkout, 'checkout_skip_reason', _stub)

        with pytest.raises(pytest.skip.Exception) as excinfo:
            _skip_unless_checkout('reify', tmp_path)

        assert str(excinfo.value) == 'stub reason from the shared builder', (
            'this module must DELEGATE the skip decision to '
            'shared.reify_checkout.checkout_skip_reason rather than keep a '
            'private copy of the two arms'
        )
        assert seen['repo'] == 'reify'
        assert seen['root'] == tmp_path
        assert seen['marker'] == _REIFY_GUARD_RELPATH, (
            'the shared builder must be bound to THIS call site\'s marker '
            f'(expected {_REIFY_GUARD_RELPATH}, got {seen.get("marker")!r})'
        )


class TestReifyGuardVectorNoneGuard:
    """`_reify_guard_vector` must fail LOUDLY on a None guard script.

    Its docstring already commits the helper to LOUD-on-failure for a script
    that exists and then fails; this pins the same commitment for the
    ``None`` arm the step-4 Optional widening opened up (project
    no-silent-fail-soft invariant), so a caller that lost its skipif guard
    fails with a clear AssertionError instead of shelling out to `bash None`.
    """

    def test_fails_loudly_when_guard_script_is_none(self, monkeypatch):
        """A None `_REIFY_GUARD_SCRIPT` must raise an AssertionError naming
        the constant, never shell out to `bash None` and let bash's resulting
        non-zero exit read as a real cross-source drift failure.

        RED today: calling this with `_REIFY_GUARD_SCRIPT` patched to None
        instead raises `AttributeError` (`None.name` inside the message that
        formats the non-zero-exit assertion) — confusing rather than loud,
        and not the AssertionError this case pins.
        """
        monkeypatch.setattr(sys.modules[__name__], '_REIFY_GUARD_SCRIPT', None)

        with pytest.raises(AssertionError, match='_REIFY_GUARD_SCRIPT'):
            _reify_guard_vector('--list-extensions')


class TestExtensionlessFilenamesDriftGuard:
    """Tier-2: shared.locking.EXTENSIONLESS_FILENAMES against reify's emitter.

    Tier-1 is
    ``shared/tests/test_locking.py::TestExtensionlessFilenamesDriftGuard``.
    """

    @pytest.mark.skipif(_REIFY_SKIP_REASON is not None, reason=_REIFY_SKIP_REASON or '')
    def test_extensionless_drift_guard_vs_reify_script(self):
        """Tier-2 (cross-source): this α copy must match reify --list-extensionless.

        WHY THIS EXISTS ON THE α SIDE TOO, rather than being left to the γ
        suite's identical test.  Tier-1 compares this frozenset against
        ``_CANONICAL_EXTENSIONLESS``, a literal in shared/tests/test_locking.py.
        An edit that changes both together — the natural way to "add a name" —
        therefore passes every guard in the shared tree while silently
        diverging from the γ copy and from bash.  The whole hole was that
        nothing in this package could observe the other two copies.

        With both suites pinned against the same bash emitter the α↔γ agreement
        follows transitively: α == bash here and γ == bash there implies α == γ,
        which is what lets the two Python copies stay verbatim duplicates
        without a re-export linking them.  Neither Tier-2 alone gives that; both
        together do, and that is the precise sense in which three-way agreement
        is ENFORCED.

        A present-but-failing script is a hard failure, not a skip — see
        _reify_guard_vector.
        """
        script_names = _reify_guard_vector('--list-extensionless')
        assert script_names == sorted(EXTENSIONLESS_FILENAMES), (
            f'α/γ/bash drift detected on the extension-less vector!\n'
            f'  reify --list-extensionless        : {script_names!r}\n'
            f'  α EXTENSIONLESS_FILENAMES         : {sorted(EXTENSIONLESS_FILENAMES)!r}\n'
            f'Update EXTENSIONLESS_FILENAMES here, in '
            f'fused-memory/src/fused_memory/middleware/lock_charter_guard.py, '
            f"and reify's own scripts/lock-charter-guard.sh _EXTLESS, plus "
            f'_CANONICAL_EXTENSIONLESS in both test copies — all five places.'
        )


class TestExtensionlessFilenames:
    """Extension-less tracked FILES classify as files (dark_factory #3248).

    The hazard these pin, stated concretely: two tasks editing
    ``hooks/project-checks`` concurrently could both hold NO lock on it, because
    a charter that strips to empty degrades to a per-task ``task-<id>`` synthetic
    lock that conflicts with nothing.
    """


    @pytest.mark.parametrize(
        ('repo', 'repo_root'),
        [('dark-factory', _DF_REPO_ROOT), ('reify', _REIFY_REPO_ROOT)],
    )
    def test_no_canonical_name_is_also_a_directory(
        self, repo: str, repo_root: Path | None
    ):
        """No member may appear as a NON-FINAL component of any tracked path.

        This is what bounds the risk of matching on a bare basename.  A member
        that were ALSO a directory name somewhere in the tree would make that
        directory declarable as a lock charter — the exact over-wide-charter
        failure the guard exists to prevent.  Measured zero collisions across
        both repos on 2026-08-09; this pins that property permanently rather
        than leaving it as a one-time observation.

        SCOPE, stated plainly so the guard is not over-read: it bounds the risk
        for these TWO checkouts only, while the predicate governs lock charters
        for every project the orchestrator targets.  Several members are generic
        basenames (`cargo`, `Dockerfile`, `LICENSE`, `pre-commit`), so a third
        project with a real `tools/cargo/` DIRECTORY would have it classified as
        a file and retained as a subtree-wide prefix lock, unswept by this test.
        That is why admitting a generic basename is a cross-project commitment —
        see the CROSS-PROJECT SCOPE note on the EXTENSIONLESS_FILENAMES vector.
        """
        repo_root = _skip_unless_checkout(repo, repo_root)
        paths = _tracked_file_paths(repo_root)
        if paths is None:
            pytest.skip(f'{repo_root} is not a git checkout (git ls-files failed)')
        assert paths, f'git ls-files returned no tracked paths for {repo_root}'

        collisions: dict[str, str] = {}
        for path in paths:
            for component in path.split('/')[:-1]:
                if component in EXTENSIONLESS_FILENAMES:
                    collisions.setdefault(component, path)

        assert not collisions, (
            f'{len(collisions)} EXTENSIONLESS_FILENAMES member(s) are also '
            f'DIRECTORY names in {repo}, so declaring that directory would now '
            f'be accepted as a file-level lock charter:\n'
            + '\n'.join(f'  {name} — e.g. {path}' for name, path in sorted(collisions.items()))
            + '\nRemove the name from the allowlist, or rename the directory.'
        )

    @pytest.mark.parametrize(
        ('repo', 'repo_root'),
        [('dark-factory', _DF_REPO_ROOT), ('reify', _REIFY_REPO_ROOT)],
    )
    def test_every_tracked_extensionless_file_is_allowlisted(
        self, repo: str, repo_root: Path | None
    ):
        """Corpus->allowlist self-audit: the 9th extensionless file must go RED.

        Turns "add 8 names" into a self-auditing invariant.  When someone adds a
        new extension-less tracked file to either repo, this test names it here
        rather than letting it surface later as a LockCharterViolation at
        submit time — or, worse, as a silent under-lock.

        This sweep IS duplicated in the γ suite, unlike the extension-corpus
        sweep which is deliberately α-free (see
        test_accept_corpus_covers_every_canonical_extension for that rationale).
        The difference is the failure mode: the extension vector is fully pinned
        by the two drift guards, whereas this one gates a REPO-level property —
        a newly-added tracked file changes the correct answer without any test
        vector changing, so whichever suite runs must be able to see it.

        Gitlink mount points are excluded by _tracked_file_paths and must NOT be
        admitted; that exclusion is asserted below.  The gitlink set is DERIVED
        from git's own mode column rather than hardcoded, which matters in both
        directions: hardcoding ``('graphiti', 'mem0')`` made the assertion
        vacuously true for the reify parametrization (reify vendors no
        submodules), and would have gone red for no real defect if dark-factory
        ever de-vendored them.  A repo with no submodules now skips the
        sub-assertion explicitly instead of passing it silently — so read this
        guard as non-vacuous exactly where gitlinks actually exist, which today
        is the dark-factory parametrization.
        """
        repo_root = _skip_unless_checkout(repo, repo_root)
        entries = _tracked_entries(repo_root)
        if entries is None:
            pytest.skip(f'{repo_root} is not a git checkout (git ls-files failed)')
        paths, gitlinks = entries
        assert paths, f'git ls-files returned no tracked paths for {repo_root}'

        # The gitlink filter must be doing real work, not passing vacuously.
        if gitlinks:
            overlap = sorted(set(gitlinks) & set(paths))
            assert not overlap, (
                f'{overlap!r} appear as BOTH mode-160000 gitlinks and ordinary '
                f'tracked paths in {repo}; _tracked_entries must partition the '
                f'corpus, and admitting a submodule mount point would let a task '
                f'declare an entire vendored submodule as its lock charter'
            )
            for gitlink in gitlinks:
                assert not is_file_path(gitlink), (
                    f'{gitlink!r} is a submodule mount point (mode 160000) and '
                    f'must stay a DIRECTORY — see the GITLINK EXCLUSION note on '
                    f'the corpus.  Admitting it would make an entire vendored '
                    f'submodule declarable as a lock charter, strictly worse '
                    f'than the bug #3248 fixed.'
                )

        unknown: dict[str, str] = {}
        for path in paths:
            seg = path.rsplit('/', 1)[-1]
            if '.' in seg:
                continue
            if not is_file_path(path):
                unknown.setdefault(seg, path)

        assert not unknown, (
            f'{len(unknown)} extension-less tracked {repo} file(s) classify as '
            f'DIRECTORIES, so any task declaring one in metadata.files is either '
            f'rejected with a LockCharterViolation or — silently — stripped to an '
            f'empty charter and given a task-<id> synthetic lock that conflicts '
            f'with nothing:\n'
            + '\n'.join(f'  {name} — e.g. {path}' for name, path in sorted(unknown.items()))
            + '\nAdd each name to EXTENSIONLESS_FILENAMES in '
            'shared/src/shared/locking.py AND '
            'fused-memory/src/fused_memory/middleware/lock_charter_guard.py, '
            'and to reify scripts/lock-charter-guard.sh _EXTLESS (all three must '
            'agree — omitting reify reds '
            'test_extensionless_drift_guard_vs_reify_script), '
            'plus _CANONICAL_EXTENSIONLESS and _EXTENSIONLESS_ACCEPT_PATHS in '
            'both test copies.'
        )


class TestFileExtensionsDriftGuard:
    """Tier-2: shared.locking.FILE_EXTENSIONS against reify's emitter.

    Tier-1 is ``shared/tests/test_locking.py::TestFileExtensionsDriftGuard``.
    """

    @pytest.mark.skipif(_REIFY_SKIP_REASON is not None, reason=_REIFY_SKIP_REASON or '')
    def test_extension_drift_guard_vs_reify_script(self):
        """Tier-2 (cross-source): this α copy must match reify --list-extensions.

        The extension-half counterpart of
        TestExtensionlessFilenamesDriftGuard::test_extensionless_drift_guard_vs_reify_script,
        added for the same reason and closing the same structural hole: before
        this, ONLY the γ suite invoked the bash emitter, so an α-only edit that
        moved FILE_EXTENSIONS and the local _CANONICAL_EXTENSIONS together
        passed everything in this package while diverging from the other two
        copies.  That hole predates #3248 — the extension vector has been
        α-unpinned against bash since the two Python copies were split — and is
        fixed here rather than left as the one asymmetry between the suites.
        """
        script_exts = _reify_guard_vector('--list-extensions')
        assert script_exts == sorted(FILE_EXTENSIONS), (
            f'α/γ/bash drift detected on the extension vector!\n'
            f'  reify --list-extensions : {script_exts!r}\n'
            f'  α FILE_EXTENSIONS       : {sorted(FILE_EXTENSIONS)!r}\n'
            f'Update FILE_EXTENSIONS here, in lock_charter_guard.py, and in '
            f"reify's scripts/lock-charter-guard.sh _EXTS, plus "
            f'_CANONICAL_EXTENSIONS in both test copies.'
        )
