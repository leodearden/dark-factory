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

import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import load_script_module
from _graph_access_fake import FakeGraphAccess
from _store_mutation_preflight_contract import SENTINEL, deny

from fused_memory.maintenance.cross_graph_move import (
    CreateResult,
    SubgraphEdgeResult,
)
from fused_memory.utils.store_mutation_preflight import (
    StoreMutationUnavailable,
    assert_store_mutation_allowed,
)

SCRIPT_PATH = (
    Path(__file__).parent.parent / 'scripts' / 'cgl_eta_auto_apply_impl.py'
)
MIGRATE_PATH = (
    Path(__file__).parent.parent / 'scripts' / 'migrate_cross_graph_leak.py'
)


_mod = load_script_module(SCRIPT_PATH, mod_name='cgl_eta_auto_apply_impl')


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


@pytest.fixture
def make_shim(make_backend, mock_config):
    """Factory fixture: () -> production's backend shim shape, ``SimpleNamespace(graphiti=backend)``.

    The backend is real, over one raw graph wired under every graph key the
    manifest names. The factory returns the shim and that raw graph, so a
    test can assert no write ever reached a graph.
    """
    def _factory() -> tuple[types.SimpleNamespace, MagicMock]:
        empty = MagicMock()
        empty.result_set = []
        graph = MagicMock()
        graph.query = AsyncMock(return_value=empty)
        graph.ro_query = AsyncMock(side_effect=lambda *_a, **_kw: empty)
        backend = make_backend(
            mock_config, graph_access=FakeGraphAccess({'reify': graph, 'dark_factory': graph}),
        )
        return types.SimpleNamespace(graphiti=backend), graph

    return _factory


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

        assert (
            getattr(migrate, 'assert_store_mutation_allowed', None)
            is assert_store_mutation_allowed
        ), (
            'migrate must import the REAL preflight at MODULE scope — this '
            'script inherits its guard and has none of its own'
        )

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
        self, make_shim, tmp_path, monkeypatch,
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
        shim, graph = make_shim()

        create = AsyncMock()
        recreate = AsyncMock()
        delete = AsyncMock()
        monkeypatch.setattr(migrate, 'create_moved_node', create)
        monkeypatch.setattr(migrate, 'recreate_subgraph_relationships', recreate)
        monkeypatch.setattr(migrate, 'delete_source_node', delete)

        deny(migrate, monkeypatch)

        apply_args = types.SimpleNamespace(
            apply=True, manifest=str(manifest), page_size=1000, config=None,
        )

        with pytest.raises(
            StoreMutationUnavailable, match=SENTINEL
        ):
            await migrate.run(apply_args, shim)

        create.assert_not_awaited()
        recreate.assert_not_awaited()
        delete.assert_not_awaited()
        graph.query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_apply_is_unchanged_when_the_preflight_passes(
        self, make_shim, tmp_path, monkeypatch,
    ):
        """Non-vacuity for the test above: with the probe passing, the SAME
        call migrates exactly as before. Without this, a manifest that silently
        stopped producing dispatches would make the zero-mutation assertions
        pass for the wrong reason."""
        migrate = _load_migrate_from_this_tree(monkeypatch)
        manifest = _manifest_path(migrate, tmp_path)
        shim, _ = make_shim()

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
        self, make_shim, tmp_path, monkeypatch,
    ):
        """One probe for the whole bulk apply, attributable in a log.

        This script's run is the widest one migrate has — the WHOLE real census
        — so a per-node probe would be both the wrong contract and thousands of
        filesystem round-trips.
        """
        migrate = _load_migrate_from_this_tree(monkeypatch)
        manifest = _manifest_path(migrate, tmp_path)
        shim, _ = make_shim()

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
        self, make_shim, monkeypatch,
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
        shim, _ = make_shim()

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


class TestDumpCrossTargetRecoveryDump:
    """dump_cross_target(backend, migrate, manifest) over a real backend."""

    @pytest.mark.asyncio
    async def test_a_dropped_edge_is_dumped_with_its_compact_embedding(
        self, tmp_path, monkeypatch, make_backend, mock_config,
    ):
        monkeypatch.setattr(_mod, 'OUTDIR', tmp_path)
        migrate = _load_migrate_from_this_tree(monkeypatch)
        manifest = {'nodes': [{
            'uuid': 'u1', 'disposition': 'MOVE',
            'source_graph': 'reify', 'target_graph': 'dark_factory',
        }]}
        reify = MagicMock()
        reify.ro_query = AsyncMock(return_value=MagicMock(
            result_set=[['e1', 'REL', 'fact', 'u1', 'A', 'x9', 'B']],
        ))
        reify.execute_command = AsyncMock(return_value=['h', [[[12, [b'0.5', b'1.25']]]], 'stats'])
        dark_factory = MagicMock()
        dark_factory.ro_query = AsyncMock(return_value=MagicMock(result_set=[]))
        backend = make_backend(
            mock_config, graph_access=FakeGraphAccess({'reify': reify, 'dark_factory': dark_factory}),
        )

        dest = await _mod.dump_cross_target(backend, migrate, manifest)

        dump = json.loads(dest.read_text())
        assert dump['count'] == 1
        assert dump['edges'][0]['fact_embedding_compact'] == '[0.5, 1.25]'
        reify.execute_command.assert_awaited_once_with(
            'GRAPH.RO_QUERY', 'reify',
            "MATCH ()-[e:RELATES_TO {uuid: 'e1'}]-() RETURN e.fact_embedding",
            '--compact',
        )
