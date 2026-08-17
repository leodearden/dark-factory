"""Tests for scripts/invalidate_fabricated_shipping_edges.py.

Loaded via importlib so the script (``scripts/`` is not a package and is not
on PYTHONPATH) can be tested without sys.path pollution — the same idiom as
``test_cleanup_count_snapshots.py`` / ``test_tag_cgl_eta_rehome_scope.py``.

Scope note: this file covers the ``--apply`` store-mutation preflight only
(task 4293). The script has no injectable ``run(args, memory=...)`` seam —
``_run`` builds its own ``FusedMemoryConfig``/``MemoryService`` — but that
costs nothing here, because the guard is placed ABOVE those lazy imports: a
denied probe must abort while never importing or constructing a backend at
all, which is precisely what makes these tests hermetic.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).parent.parent
    / 'scripts'
    / 'invalidate_fabricated_shipping_edges.py'
)


def _load_module() -> types.ModuleType:
    """Load invalidate_fabricated_shipping_edges.py from its file path.

    The module is registered in sys.modules under its name so that
    @dataclass and other reflection-based decorators work correctly
    (they call sys.modules.get(cls.__module__)).
    """
    mod_name = 'invalidate_fabricated_shipping_edges'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # required for @dataclass __module__ lookup
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()


@pytest.fixture(autouse=True)
def _neutralise_store_mutation_preflight(monkeypatch):
    """Keep this MOCK-unit suite independent of the REAL ``~/.mem0``.

    ``_run(...)`` with ``--apply`` runs a fail-closed capability preflight
    before it builds a backend (task 4293). That probe touches the real
    filesystem, so without this fixture every test here would pass or fail
    according to whether the machine running pytest happens to be able to
    write mem0's history directory -- and it genuinely cannot inside an agent
    sandbox, which is the whole reason the guard exists. This suite is
    deliberately MOCK-unit (no live store at all), so the environment must not
    be an input to it.

    ``TestRunApplyStoreMutationPreflight`` re-rigs this per test -- to refuse,
    to record, or to pass -- so the guard's own behaviour is still pinned
    explicitly rather than assumed away.

    Deliberately NOT ``raising=False``: if the guard is ever removed from the
    script this fixture must break loudly rather than silently no-op.
    """
    monkeypatch.setattr(_mod, 'assert_store_mutation_allowed', lambda **_kw: None)


class _BackendWasConstructed(RuntimeError):
    """Raised by the ``MemoryService`` sentinel below.

    Deliberately distinct from ``StoreMutationUnavailable`` so a test can tell
    "the guard refused before the backend stack was reached" from "execution
    ran on into the normal path" -- the two halves this file has to
    distinguish, since it has no injectable service seam.
    """


def _install_backend_sentinel(monkeypatch):
    """Make constructing a ``MemoryService`` an observable, loud event.

    ``_run`` imports ``MemoryService`` lazily from
    ``fused_memory.services.memory_service``, so patching the attribute on
    that module is what the lazy import resolves.
    """
    import fused_memory.services.memory_service as mem_svc

    def _explode(*_args, **_kwargs):
        raise _BackendWasConstructed('MemoryService was constructed')

    monkeypatch.setattr(mem_svc, 'MemoryService', _explode)


# ===========================================================================
# Tests: --apply store-mutation preflight
# ===========================================================================

class TestRunApplyStoreMutationPreflight:
    """``--apply`` refuses to START when this process cannot write mem0's store.

    Ported from ``test_sweep_toolcall_xml_leak.TestRunApplyStoreMutationPreflight``
    (task 3686), which is the in-repo precedent for this contract.

    ``_run`` invalidates its edges in a sequential loop whose body is wrapped
    in ``except Exception`` -> ``logger.error`` -> continue, and then returns
    **0** regardless. ``StoreMutationUnavailable`` subclasses ``RuntimeError``,
    so a probe inside that loop would be swallowed into N error lines and a
    ZERO exit code -- a silent success reported over a half-applied
    invalidation. Only a run-wide probe ahead of the backend stack bounds that.
    """

    def _args(self, apply: bool = True, keep_unverified: bool = False):
        return argparse.Namespace(
            project='dark_factory',
            project_root='/some/path',
            config=None,
            apply=apply,
            keep_unverified=keep_unverified,
        )

    @staticmethod
    def _deny(monkeypatch):
        """Rig the preflight to refuse, as it would inside an agent sandbox."""
        def _raise(*_args, **_kwargs):
            raise _mod.StoreMutationUnavailable('SENTINEL-store-unwritable')

        monkeypatch.setattr(_mod, 'assert_store_mutation_allowed', _raise)

    @staticmethod
    def _fail_closed_records(caplog) -> list:
        """The guard site's OWN diagnosis.

        ``main`` has no handler at all here -- it hands ``_run`` straight to
        ``asyncio.run`` -- so the refusal exits as an uncaught traceback and
        this ERROR record is the ONLY place the operator is told what was
        refused and what to do instead. Pinned on the fail-closed marker and
        the remedy noun ONLY, so every other word stays free to reword.

        Asserting on message CONTENT is deliberate, and is the narrow exception
        to the repo's don't-pin-guard-message-prose norm (task 3799): the record
        this test is about is defined BY its content -- mere record-existence
        would still pass if the whole diagnosis were replaced by "boom",
        precisely the regression this exists to catch. Verified non-vacuous:
        mutating the marker in the script turns this assertion red (task 4127
        amendment).

        NOTE the logger name is ``invalidate_shipping_edges``, which is NOT the
        module name -- filtering on the module name would silently match
        nothing and make every assertion below vacuous.
        """
        return [
            rec for rec in caplog.records
            if rec.name == 'invalidate_shipping_edges'
            and rec.levelname == 'ERROR'
            and 'NOT started (fail-closed)' in rec.getMessage()
            and 'MCP server' in rec.getMessage()
        ]

    @pytest.mark.asyncio
    async def test_apply_refuses_to_start_when_the_store_is_unwritable(
        self, monkeypatch
    ):
        """The whole point: refuse to start rather than half-complete.

        The refusal must PROPAGATE. Swallowed at the invalidation loop it
        would become N ``logger.error`` lines and a return of 0 -- an exit code
        an operator and any CI caller would read as a clean run.
        """
        self._deny(monkeypatch)

        with pytest.raises(
            _mod.StoreMutationUnavailable, match='SENTINEL-store-unwritable'
        ):
            await _mod._run(self._args(apply=True))

    @pytest.mark.asyncio
    async def test_it_aborts_before_constructing_any_backend(self, monkeypatch):
        """Nothing is imported, connected to, or scanned by a run that was
        never going to be allowed to mutate.

        The sentinel makes construction loud, so the assertion is behavioural:
        seeing ``StoreMutationUnavailable`` rather than ``_BackendWasConstructed``
        proves the guard sits above the lazy backend imports -- and therefore
        above the ``ro_query`` candidate scan and the ``update_edge`` calls
        that follow it.
        """
        self._deny(monkeypatch)
        _install_backend_sentinel(monkeypatch)

        with pytest.raises(_mod.StoreMutationUnavailable):
            await _mod._run(self._args(apply=True))

    @pytest.mark.asyncio
    async def test_a_dry_run_is_never_gated_on_write_capability(self, monkeypatch):
        """A read-only run mutates nothing, so it must not require the ability
        to mutate -- the candidate report stays obtainable from anywhere.

        Reaching the backend sentinel is the proof that the dry run ran ON past
        the guard site rather than being turned back at it, and the empty
        recorder is the proof it was never probed at all.
        """
        calls: list[dict] = []
        monkeypatch.setattr(
            _mod, 'assert_store_mutation_allowed', lambda **kw: calls.append(kw)
        )
        _install_backend_sentinel(monkeypatch)

        with pytest.raises(_BackendWasConstructed):
            await _mod._run(self._args(apply=False))

        assert calls == [], 'a dry run mutates nothing, so it must not probe'

    @pytest.mark.asyncio
    async def test_the_probe_names_the_operation_and_apply_runs_on(
        self, monkeypatch
    ):
        """A passing preflight leaves ``--apply`` exactly as it was, and the
        refusal stays attributable in a log.

        Reaching the backend sentinel proves the guard is a gate and not a
        detour: with the probe passing, execution continues into the normal
        path. The operation string identifies this script and its mutating
        mode, and is probed once for the RUN, not once per candidate edge.
        """
        calls: list[dict] = []
        monkeypatch.setattr(
            _mod, 'assert_store_mutation_allowed', lambda **kw: calls.append(kw)
        )
        _install_backend_sentinel(monkeypatch)

        with pytest.raises(_BackendWasConstructed):
            await _mod._run(self._args(apply=True))

        assert len(calls) == 1, 'probed ONCE per run, not once per edge'
        assert 'invalidate_fabricated_shipping_edges' in calls[0]['operation']
        assert '--apply' in calls[0]['operation']

    @pytest.mark.asyncio
    async def test_the_refusal_is_loud(self, monkeypatch, caplog):
        """The guard site logs its own fail-closed diagnosis before raising.

        Nothing downstream will: this script's ``main`` has no blanket handler,
        so without this record the operator sees a bare traceback naming an
        exception class and no remedy.
        """
        self._deny(monkeypatch)

        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(_mod.StoreMutationUnavailable),
        ):
            await _mod._run(self._args(apply=True))

        assert self._fail_closed_records(caplog), (
            'nothing else explains this traceback -- the guard site must log '
            'the fail-closed diagnosis before raising; got: '
            f'{[rec.getMessage() for rec in caplog.records]}'
        )
