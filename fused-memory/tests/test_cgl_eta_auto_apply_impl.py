"""Tests for scripts/cgl_eta_auto_apply_impl.py's INHERITED store-mutation guard.

Loaded via importlib so the script (``scripts/`` is not a package and is not on
PYTHONPATH) can be tested without sys.path pollution — the same idiom as
``test_migrate_cross_graph_leak.py``.

Scope note: this file covers exactly one property (task 4293) — that this
script's sole mutation path is guarded, even though the script contains no
probe of its own and a ``grep -rln assert_store_mutation_allowed
fused-memory/scripts/`` will never name it.

Why that needs a test at all. This is the one script in the tree that applies
with NO flag: it has no argparse, and ``main`` builds
``types.SimpleNamespace(apply=True, ...)`` unconditionally and hands it to
``migrate_cross_graph_leak.run``. Guarding it a second time in its own file
would double-probe every run for no gain, so the deliberate choice is to
inherit migrate's guard. That inheritance is real but FRAGILE, and it depends
on two facts a future refactor could silently break:

  * the guard must live in migrate's ``run()``. This script never touches
    migrate's CLI, so a guard in ``main()``/``build_arg_parser()`` would be
    bypassed entirely;
  * ``_load_migrate()`` EXECUTES the real migrate file via
    ``importlib.util.spec_from_file_location`` and never registers it in
    ``sys.modules``, so a ``sys.modules``-based interception would not reach
    it either.

Both are asserted below — behaviourally for the first, structurally for the
second — so the day either stops being true, this file goes red rather than
this script quietly becoming the only unguarded bulk-apply in the repo.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.maintenance.cross_graph_move import (
    CreateResult,
    SubgraphEdgeResult,
)

SCRIPT_PATH = (
    Path(__file__).parent.parent / 'scripts' / 'cgl_eta_auto_apply_impl.py'
)
MIGRATE_PATH = (
    Path(__file__).parent.parent / 'scripts' / 'migrate_cross_graph_leak.py'
)


def _load_module() -> types.ModuleType:
    """Load cgl_eta_auto_apply_impl.py from its file path.

    The module is registered in sys.modules under its name so that
    reflection-based decorators work correctly.
    """
    mod_name = 'cgl_eta_auto_apply_impl'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()


def _load_migrate_from_this_tree(monkeypatch) -> types.ModuleType:
    """Drive the script's OWN ``_load_migrate()``, pointed at THIS checkout.

    The script hard-codes ``SCRIPT = /home/leo/src/dark-factory/fused-memory/
    scripts/migrate_cross_graph_leak.py`` — an absolute path into the MAIN
    checkout. Left alone, this suite would assert against whatever migrate
    happens to be on main rather than the one sitting beside it in this
    worktree, which is exactly backwards for a test whose job is to catch a
    change to that file. Repointing ``SCRIPT`` is the smallest edit that makes
    the test track its own tree; ``test_the_script_loads_migrate_by_file_path``
    below pins the unpatched value so the redirect cannot paper over a change
    to what production actually loads.

    Everything else is the production path verbatim: the script's own
    ``_load_migrate`` executing the real file.
    """
    monkeypatch.setattr(_mod, 'SCRIPT', MIGRATE_PATH)
    return _mod._load_migrate()


def _manifest_path(migrate, tmp_path) -> Path:
    """A one-MOVE-node reviewed manifest — enough that a run reaching the apply
    loop WOULD issue a Phase-A create, a Phase-B recreate and a Phase-C delete.
    Without a mutating node the zero-mutation assertions would be vacuous."""
    node = {
        'uuid': 'u-move',
        'name': 'N',
        'source_graph': 'reify',
        'target_graph': 'dark_factory',
        'disposition': migrate.MOVE,
        'edge_count': 0,
        'episode_count': 0,
    }
    manifest = migrate.build_manifest([node], {'reify': 1}, dry_run=True)
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return path


def _shim() -> tuple[types.SimpleNamespace, MagicMock]:
    """Production's backend shim shape: ``SimpleNamespace(graphiti=backend)``.

    Returns the shim and the single graph mock every ``_graph_for`` lookup
    resolves to, so a test can assert no write ever reached a graph.
    """
    empty = MagicMock()
    empty.result_set = []
    graph = MagicMock()
    graph.query = AsyncMock(return_value=empty)
    graph.ro_query = AsyncMock(side_effect=lambda *_a, **_kw: empty)
    graphiti = MagicMock()
    graphiti._graph_for = MagicMock(return_value=graph)
    graphiti.list_graphs = AsyncMock(return_value=['reify'])
    graphiti._require_falkor_client = MagicMock(return_value=MagicMock())
    return types.SimpleNamespace(graphiti=graphiti), graph


# ===========================================================================
# Tests: the inherited --apply store-mutation preflight (task 4293)
# ===========================================================================

class TestInheritedStoreMutationPreflight:
    """This script mutates only through ``migrate_cross_graph_leak.run()``,
    so migrate's guard is this script's guard."""

    def test_the_loaded_migrate_module_exposes_the_guard_at_module_scope(
        self, monkeypatch,
    ):
        """The structural precondition the inheritance rests on.

        A guard reachable only from migrate's ``main()``/``build_arg_parser()``
        would leave this script — which never touches migrate's CLI — applying
        unguarded, and would fail here. Module scope is also what lets the
        behavioural test below rig the probe at all.
        """
        migrate = _load_migrate_from_this_tree(monkeypatch)

        assert hasattr(migrate, 'assert_store_mutation_allowed'), (
            'migrate must import the preflight at MODULE scope — this script '
            'inherits its guard and has none of its own'
        )
        assert hasattr(migrate, 'StoreMutationUnavailable')

    def test_the_script_loads_migrate_by_file_path_not_via_sys_modules(
        self, monkeypatch,
    ):
        """``_load_migrate`` executes the real file and never consults (or
        populates) ``sys.modules``.

        Pinned because it is the other half of the inheritance claim: the guard
        is inherited from the FILE, so intercepting or substituting migrate
        through the module registry neither helps nor is required. The
        unpatched ``SCRIPT`` is asserted here too, so the repoint in
        ``_load_migrate_from_this_tree`` cannot hide a change to what
        production actually executes.
        """
        assert _mod.SCRIPT == _mod.FM / 'scripts' / 'migrate_cross_graph_leak.py'
        assert _mod.SCRIPT.name == MIGRATE_PATH.name

        migrate = _load_migrate_from_this_tree(monkeypatch)

        assert migrate is not sys.modules.get('migrate_cross_graph_leak'), (
            '_load_migrate must execute the file itself, so a sys.modules '
            'entry is neither consulted nor sufficient'
        )

    @pytest.mark.asyncio
    async def test_the_unconditional_apply_refuses_when_the_store_is_unwritable(
        self, tmp_path, monkeypatch,
    ):
        """The behavioural claim, driven through this script's own loader.

        ``main`` builds ``SimpleNamespace(apply=True, manifest=..., ...)``
        unconditionally (there is no flag to withhold) and awaits
        ``migrate.run`` — so this call IS the script's entire mutation surface.
        Denying the probe on the module ``_load_migrate()`` returns must abort
        it before a single graph write, with the refusal PROPAGATING: swallowed,
        it would become blocked rows and an exit code the deterministic task's
        predicate contract reads as a clean apply.
        """
        migrate = _load_migrate_from_this_tree(monkeypatch)
        manifest = _manifest_path(migrate, tmp_path)
        shim, graph = _shim()

        create = AsyncMock()
        recreate = AsyncMock()
        delete = AsyncMock()
        monkeypatch.setattr(migrate, 'create_moved_node', create)
        monkeypatch.setattr(migrate, 'recreate_subgraph_relationships', recreate)
        monkeypatch.setattr(migrate, 'delete_source_node', delete)

        def _raise(*_args, **_kwargs):
            raise migrate.StoreMutationUnavailable('SENTINEL-store-unwritable')

        monkeypatch.setattr(migrate, 'assert_store_mutation_allowed', _raise)

        apply_args = types.SimpleNamespace(
            apply=True, manifest=str(manifest), page_size=1000, config=None,
        )

        with pytest.raises(
            migrate.StoreMutationUnavailable, match='SENTINEL-store-unwritable'
        ):
            await migrate.run(apply_args, shim)

        create.assert_not_awaited()
        recreate.assert_not_awaited()
        delete.assert_not_awaited()
        graph.query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_apply_is_unchanged_when_the_preflight_passes(
        self, tmp_path, monkeypatch,
    ):
        """Non-vacuity for the test above: with the probe passing, the SAME
        call migrates exactly as before. Without this, a manifest that silently
        stopped producing dispatches would make the zero-mutation assertions
        pass for the wrong reason."""
        migrate = _load_migrate_from_this_tree(monkeypatch)
        manifest = _manifest_path(migrate, tmp_path)
        shim, _ = _shim()

        create = AsyncMock(return_value=CreateResult(
            uuid='u-move', source_graph='reify', target_graph='dark_factory',
        ))
        recreate = AsyncMock(return_value=SubgraphEdgeResult())
        delete = AsyncMock(return_value=None)
        monkeypatch.setattr(migrate, 'create_moved_node', create)
        monkeypatch.setattr(migrate, 'recreate_subgraph_relationships', recreate)
        monkeypatch.setattr(migrate, 'delete_source_node', delete)
        monkeypatch.setattr(
            migrate, 'assert_store_mutation_allowed', lambda **_kw: None,
        )

        apply_args = types.SimpleNamespace(
            apply=True, manifest=str(manifest), page_size=1000, config=None,
        )

        report = await migrate.run(apply_args, shim)

        assert report['dry_run'] is False
        create.assert_awaited_once()
        delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_the_probe_names_the_operation_and_runs_once(
        self, tmp_path, monkeypatch,
    ):
        """One probe for the whole bulk apply, attributable in a log.

        This script's run is the widest one migrate has — the WHOLE real census
        — so a per-node probe would be both the wrong contract and thousands of
        filesystem round-trips.
        """
        migrate = _load_migrate_from_this_tree(monkeypatch)
        manifest = _manifest_path(migrate, tmp_path)
        shim, _ = _shim()

        monkeypatch.setattr(migrate, 'create_moved_node', AsyncMock(return_value=CreateResult(
            uuid='u-move', source_graph='reify', target_graph='dark_factory',
        )))
        monkeypatch.setattr(
            migrate, 'recreate_subgraph_relationships',
            AsyncMock(return_value=SubgraphEdgeResult()),
        )
        monkeypatch.setattr(migrate, 'delete_source_node', AsyncMock(return_value=None))

        calls: list[dict] = []
        monkeypatch.setattr(
            migrate, 'assert_store_mutation_allowed', lambda **kw: calls.append(kw),
        )

        apply_args = types.SimpleNamespace(
            apply=True, manifest=str(manifest), page_size=1000, config=None,
        )

        await migrate.run(apply_args, shim)

        assert len(calls) == 1, 'probed ONCE per run, not once per manifest node'
        assert 'migrate_cross_graph_leak' in calls[0]['operation']
        assert '--apply' in calls[0]['operation']

    @pytest.mark.asyncio
    async def test_the_scripts_own_dry_run_census_is_never_gated(
        self, monkeypatch,
    ):
        """Step 2 of this script's sequence is a fresh dry-run census
        (``SimpleNamespace(apply=False, ...)`` at the ``migrate.run`` above the
        apply one). It mutates nothing, so it must not be gated on write
        capability: the census has standalone diagnostic value and must stay
        obtainable from a read-only environment, exactly as ``--apply``-less
        runs do everywhere else in this contract.

        The trade this placement accepts, stated plainly: because the ONLY
        probe in this script's sequence is the inherited one inside
        ``migrate.run(apply=True)`` at step 5, a write-denied environment pays
        for this census, the cross-target recovery dump and the full FalkorDB
        backup before the refusal arrives. An early probe in ``main`` would
        buy that back; see the comment at the step-5 call site for why a
        single probe site was preferred anyway."""
        migrate = _load_migrate_from_this_tree(monkeypatch)
        shim, _ = _shim()

        calls: list[dict] = []
        monkeypatch.setattr(
            migrate, 'assert_store_mutation_allowed', lambda **kw: calls.append(kw),
        )

        dry_args = types.SimpleNamespace(
            apply=False, manifest=None, page_size=1000, config=None,
        )

        manifest = await migrate.run(dry_args, shim)

        assert manifest['dry_run'] is True
        assert calls == [], 'a census mutates nothing, so it must not probe'
