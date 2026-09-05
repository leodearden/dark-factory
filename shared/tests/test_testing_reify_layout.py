"""Unit tests for shared.testing_reify_layout — the planted-layout test harness.

The single home of the "load a COPY of a test module out of a SYNTHETIC
checkout tree" technique that every reify-dependent suite needs (task 4259,
consolidating the two copies that had grown in shared/tests/test_locking.py and
orchestrator/tests/test_verify_role_integration.py).  These cases OWN the
harness's semantics; the two consumers keep thin marker-bound adapters over it
and pin only their own wiring, exactly as shared/tests/test_reify_checkout.py
owns the resolver's semantics one layer down.

Every case here builds its own SYNTHETIC source module in ``tmp_path`` rather
than planting a real consumer suite, so the harness is pinned host-
independently and a change to either consumer cannot silently rewrite what this
file measures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from shared.testing_reify_layout import (
    AmbientReifyCheckoutError,
    ambient_reify_roots,
    bare_layout,
    plant_reify_layout,
    planted_reify_root,
    worktree_layout,
)

# The two real markers in play, one per consumer call site — the same pair
# shared/tests/test_reify_checkout.py parametrizes the resolver over.  Using
# both here is what proves the marker really is a parameter of the harness and
# not a default hardcoded to whichever suite was ported first.
_VERIFY_MARKER = Path('scripts') / 'verify.sh'
_GUARD_MARKER = Path('scripts') / 'lock-charter-guard.sh'

_BOTH_MARKERS = pytest.mark.parametrize('marker', [_VERIFY_MARKER, _GUARD_MARKER])

#: A probe module that records, at IMPORT time, where it was loaded from and
#: what reify checkout it resolves from THAT location.  Import-time is the whole
#: point: it is the moment a real suite's module-level constants are computed.
_PROBE_TEMPLATE = """
from pathlib import Path

import shared.reify_checkout as reify_checkout

PROBE_ID = {probe_id!r}
PROBE_FILE = Path(__file__).resolve()
MARKER = Path({marker!s})
CHECKOUT = reify_checkout.resolve_reify_checkout(MARKER, start=Path(__file__))
REIFY_ROOT = CHECKOUT.root
"""


@pytest.fixture(autouse=True)
def _no_ambient_reify_root(monkeypatch):
    """Neutralize REIFY_ROOT for every case in this file.

    The override arm short-circuits `resolve_reify_checkout` before the
    ancestor walk, so an operator's ambient ``export REIFY_ROOT=...`` would
    make every planted-layout assertion below answer for that path instead of
    for the tree under test.  The harness itself is env-agnostic; this fixture
    only removes the ambient steering, it does not stand in for anything the
    harness does.
    """
    monkeypatch.delenv('REIFY_ROOT', raising=False)


def _write_probe(tmp_path: Path, name: str, marker: str | Path = _VERIFY_MARKER) -> Path:
    """Write a synthetic probe module at ``tmp_path/<name>.py`` and return it.

    Deliberately written OUTSIDE ``tmp_path/'src'`` so it is not itself inside
    the planted tree: `plant_reify_layout` must COPY it in, and a harness that
    forgot to parameterize the source would copy something else entirely.
    """
    source = tmp_path / f'{name}.py'
    source.write_text(
        _PROBE_TEMPLATE.format(probe_id=name, marker=repr(str(marker)))
    )
    return source


def _tests_dir(tmp_path: Path, tests_relpath: str) -> Path:
    """Where `plant_reify_layout` is contracted to put the copy."""
    return tmp_path / 'src' / tests_relpath


class TestPlantReifyLayout:
    """`plant_reify_layout` — copy a source module into a synthetic ancestry."""

    def test_copies_the_source_module_not_the_harness(self, tmp_path):
        """THE pin: the copy is of *source*, planted at *tests_relpath*.

        A harness that forgot to parameterize the source would copy its own
        ``__file__`` — and, in the ``.worktrees/<id>`` layout every consumer
        suite normally runs in, most black-box assertions would still pass.
        Two facts are checked: the returned module's ``__file__`` is the
        PLANTED path (not the source's own), and its contents came from the
        source (``PROBE_ID`` exists only in the probe).
        """
        source = _write_probe(tmp_path, 'alpha_probe')

        mod = plant_reify_layout(
            source, tmp_path, bare_layout('shared'), marker=_VERIFY_MARKER
        )

        expected = _tests_dir(tmp_path, bare_layout('shared')) / 'test_copy_probe.py'
        assert mod.__file__ is not None, f'{mod!r} was loaded from a file and must have one'
        assert Path(mod.__file__) == expected, (
            f'the copy must be loaded from the PLANTED path {expected!r}, not '
            f'{mod.__file__!r} — a copy that keeps the source\'s own __file__ '
            f'walks the real ancestry and measures this host, not the planted tree'
        )
        assert mod.PROBE_ID == 'alpha_probe', (
            'the planted module must be a copy of *source*, not of the harness'
        )
        assert expected.resolve() == mod.PROBE_FILE

    @_BOTH_MARKERS
    def test_planted_marker_is_what_the_copy_resolves(self, tmp_path, marker):
        """A planted ``reify/<marker>`` is the answer the copy resolves to.

        Parametrized over BOTH real markers so the marker is proven to be a
        parameter rather than a default baked in from whichever consumer was
        ported first.
        """
        source = _write_probe(tmp_path, 'beta_probe', marker=marker)

        mod = plant_reify_layout(source, tmp_path, bare_layout('shared'), marker=marker)

        assert (planted_reify_root(tmp_path) / marker).is_file(), (
            'plant_marker=True must write a real file at '
            f'{planted_reify_root(tmp_path) / marker!r}'
        )
        assert planted_reify_root(tmp_path).resolve() == mod.REIFY_ROOT, (
            f'the copy must resolve the checkout planted beside it '
            f'({planted_reify_root(tmp_path)!r}), not an off-tree answer '
            f'({mod.REIFY_ROOT!r})'
        )

    def test_plant_marker_false_plants_nothing(self, tmp_path):
        """``plant_marker=False`` leaves the planted reify root absent.

        This is the arm that expresses a genuine discovery MISS, so anything
        appearing under `planted_reify_root` would silently convert the miss
        into a hit and retire the case that depends on it.
        """
        source = _write_probe(tmp_path, 'gamma_probe')

        mod = plant_reify_layout(
            source, tmp_path, bare_layout('shared'), marker=_VERIFY_MARKER, plant_marker=False
        )

        assert not planted_reify_root(tmp_path).exists(), (
            f'plant_marker=False must plant nothing under '
            f'{planted_reify_root(tmp_path)!r}'
        )
        assert mod.REIFY_ROOT is None, (
            f'nothing named reify exists in the planted ancestry, but the copy '
            f'resolved {mod.REIFY_ROOT!r} — an off-tree answer'
        )

    def test_no_sys_modules_entry_survives_the_call(self, tmp_path):
        """The temporary ``sys.modules`` entry must not outlive the call.

        A copy left behind would be importable by name from an unrelated test
        and would keep the whole planted tree alive in the interpreter.
        """
        source = _write_probe(tmp_path, 'delta_probe')
        before = set(sys.modules)

        mod = plant_reify_layout(
            source, tmp_path, bare_layout('shared'), marker=_VERIFY_MARKER
        )

        assert mod.__name__ not in sys.modules, (
            f'{mod.__name__!r} outlived the call — the harness must pop its '
            f'temporary sys.modules entry'
        )
        leaked = [name for name in set(sys.modules) - before if sys.modules[name] is mod]
        assert not leaked, f'the loaded copy is still registered under {leaked!r}'

    def test_sys_modules_is_clean_even_when_the_copy_raises(self, tmp_path):
        """The pop must be in a ``finally`` — pins it against a bare sequence.

        A copy whose import-time constant resolution blows up is exactly the
        case a consumer suite is investigating when it reaches for this
        harness, so that is precisely when a leaked entry would be least
        noticed.  The exception must propagate unchanged as well: swallowing it
        would turn an import-time failure into a silently empty module.
        """
        source = tmp_path / 'boom_probe.py'
        source.write_text("raise RuntimeError('probe boom')\n")
        before = set(sys.modules)

        with pytest.raises(RuntimeError, match='probe boom'):
            plant_reify_layout(
                source, tmp_path, bare_layout('shared'), marker=_VERIFY_MARKER
            )

        assert set(sys.modules) - before == set(), (
            'a copy that raised during exec must still be popped from '
            'sys.modules — the pop belongs in a finally'
        )

    def test_two_sources_at_one_layout_get_distinct_module_names(self, tmp_path):
        """Two suites planting the same layout must not clobber each other.

        The internal module name is DERIVED from the source and the layout
        rather than supplied by each caller, so uniqueness is a property of the
        harness instead of every caller remembering to pick a distinct prefix.
        """
        first_source = _write_probe(tmp_path, 'epsilon_probe')
        second_source = _write_probe(tmp_path, 'zeta_probe')
        layout = bare_layout('shared')

        first = plant_reify_layout(first_source, tmp_path, layout, marker=_VERIFY_MARKER)
        second = plant_reify_layout(second_source, tmp_path, layout, marker=_VERIFY_MARKER)

        assert first is not second
        assert first.__name__ != second.__name__, (
            f'both copies were loaded under {first.__name__!r}, so one suite\'s '
            f'copy can clobber another\'s in sys.modules'
        )
        assert (first.PROBE_ID, second.PROBE_ID) == ('epsilon_probe', 'zeta_probe')


class TestLayoutBuilders:
    """`bare_layout` / `worktree_layout` — the two checkout shapes, stated once."""

    def test_bare_layout_renders_a_plain_checkout(self):
        assert bare_layout('shared') == 'dark-factory/shared/tests'

    def test_worktree_layout_renders_a_task_worktree(self):
        assert worktree_layout('shared', '4080') == 'dark-factory/.worktrees/4080/shared/tests'

    def test_both_layouts_resolve_to_the_same_planted_checkout(self, tmp_path):
        """The load-bearing fact, now stated in ONE place.

        ``.worktrees/<id>`` contributes exactly two path segments, so no fixed
        ``parents[N]`` index can be correct in both layouts.  Two copies planted
        under the same tmp_path must nonetheless agree on the checkout beside
        them.
        """
        source = _write_probe(tmp_path, 'eta_probe')

        bare = plant_reify_layout(
            source, tmp_path, bare_layout('shared'), marker=_VERIFY_MARKER
        )
        worktree = plant_reify_layout(
            source, tmp_path, worktree_layout('shared', '4080'), marker=_VERIFY_MARKER
        )

        expected = planted_reify_root(tmp_path).resolve()
        assert expected == bare.REIFY_ROOT
        assert worktree.REIFY_ROOT == bare.REIFY_ROOT, (
            f"'.worktrees/<id>' adds exactly two path segments, so a fixed "
            f'parents[N] cannot be correct in both layouts (got '
            f'worktree={worktree.REIFY_ROOT!r} vs bare={bare.REIFY_ROOT!r})'
        )


class TestPlantedReifyRoot:
    """`planted_reify_root` — the one place that knows the synthetic layout."""

    def test_is_the_reify_sibling_of_the_planted_src_tree(self, tmp_path):
        assert planted_reify_root(tmp_path) == tmp_path / 'src' / 'reify'


class TestAmbientReifyRoots:
    """`ambient_reify_roots` — naming an environment problem as one."""

    def test_clean_synthetic_tree_has_no_ambient_roots(self, tmp_path):
        start = _tests_dir(tmp_path, bare_layout('shared')) / 'test_x.py'
        assert ambient_reify_roots(start, _VERIFY_MARKER) == []

    def test_reports_a_real_checkout_planted_above_the_tests_dir(self, tmp_path):
        """A real ``reify/<marker>`` above the planted tree is contamination.

        It shadows nothing when a marker is planted (nearest-first resolution
        wins), but it makes a deliberate discovery MISS unexpressible.
        """
        marker_file = tmp_path / 'reify' / _VERIFY_MARKER
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        marker_file.write_text('#!/bin/sh\necho stub\n')
        start = _tests_dir(tmp_path, bare_layout('shared')) / 'test_x.py'

        roots = ambient_reify_roots(start, _VERIFY_MARKER)

        assert tmp_path.resolve() in roots, (
            f'the contaminating ancestor {tmp_path!r} must be reported, got {roots!r}'
        )
        assert roots[0] == tmp_path.resolve(), (
            f'ancestors must be reported nearest-first, matching the order '
            f'resolve_reify_checkout itself walks them (got {roots!r})'
        )

    def test_a_marker_below_the_start_is_not_an_ancestor(self, tmp_path):
        """Only ANCESTORS count — a sibling/descendant checkout steers nothing."""
        marker_file = tmp_path / 'src' / 'elsewhere' / 'reify' / _VERIFY_MARKER
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        marker_file.write_text('#!/bin/sh\necho stub\n')
        start = _tests_dir(tmp_path, bare_layout('shared')) / 'test_x.py'

        assert ambient_reify_roots(start, _VERIFY_MARKER) == []


class TestAmbientContaminationGuard:
    """The precondition `plant_reify_layout` enforces on the discovery-MISS arm.

    Promoted into the harness from shared/tests/test_locking.py, where it was
    an inline block in ONE case; the copy in
    orchestrator/tests/test_verify_role_integration.py it was ported from never
    grew it.  That divergence is the reason this module exists, so the check
    belongs to the single source rather than to whichever caller remembered it.
    """

    def test_plant_marker_false_raises_when_a_real_checkout_is_above(self, tmp_path):
        """An unusual ``--basetemp`` or an ambient ``/tmp/reify`` must be NAMED.

        Without this the behavioural assertion downstream fails with a message
        that blames the resolver instead of naming the real cause.
        """
        marker_file = tmp_path / 'reify' / _VERIFY_MARKER
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        marker_file.write_text('#!/bin/sh\necho stub\n')
        source = _write_probe(tmp_path, 'theta_probe')

        with pytest.raises(AmbientReifyCheckoutError) as excinfo:
            plant_reify_layout(
                source,
                tmp_path,
                bare_layout('shared'),
                marker=_VERIFY_MARKER,
                plant_marker=False,
            )

        message = str(excinfo.value)
        assert str(tmp_path) in message, (
            f'the contaminating ancestor must be named so the failure blames '
            f'the environment, not the resolver: {message!r}'
        )
        assert str(_VERIFY_MARKER) in message, (
            f'the marker must be named — contamination is marker-specific: {message!r}'
        )

    def test_clean_tree_does_not_raise(self, tmp_path):
        source = _write_probe(tmp_path, 'iota_probe')

        mod = plant_reify_layout(
            source, tmp_path, bare_layout('shared'), marker=_VERIFY_MARKER, plant_marker=False
        )

        assert mod.REIFY_ROOT is None

    def test_planted_marker_arm_is_not_guarded(self, tmp_path):
        """With a marker planted, an ancestor checkout is harmless.

        The planted marker is the NEAREST ancestor hit, so nearest-first
        resolution already shadows anything above it.  Raising here would
        reject a perfectly expressible case.
        """
        marker_file = tmp_path / 'reify' / _VERIFY_MARKER
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        marker_file.write_text('#!/bin/sh\necho stub\n')
        source = _write_probe(tmp_path, 'kappa_probe')

        mod = plant_reify_layout(
            source, tmp_path, bare_layout('shared'), marker=_VERIFY_MARKER
        )

        assert planted_reify_root(tmp_path).resolve() == mod.REIFY_ROOT, (
            'the nearer planted checkout must win over the one above the tree'
        )
