"""Tests for scripts/repair_recon_citation.py (task 3065).

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution — mirrors the pattern in test_prune_recon_cycle_summaries.py.

The script owns NO copy of the repair logic: it parses flags, opens the journal
and the memory service, and delegates to
``reconciliation.citation_repair.repair_memory_citation`` with ``apply`` taken
straight from ``--apply``. So these tests assert the DELEGATION (one call, ids
forwarded verbatim, apply threaded) and the exit-code mapping — the repair
semantics themselves are covered in tests/reconciliation/test_citation_repair.py.
A script-side dry-run that re-derived "what would change" would be a second site
that must agree byte-for-byte with the real one and cannot be kept in agreement.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'repair_recon_citation.py'

TARGET_RUN = '06a4466d-cdc0-49ac-8e99-e6723be39392'
FINDING = '5e85117e-51fc-4a7f-8ca7-e26078dbd3f2'
DANGLING = 'beacf7fc-b76a-4c0b-876d-f4cf6d906d42'
SUCCESSOR = '746b4ab9-ca3c-418b-982a-32b85bfcf94b'


def _load_module() -> types.ModuleType:
    mod_name = 'repair_recon_citation'
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


def _parse(*argv: str):
    return _mod.build_parser().parse_args(list(argv))


_REQUIRED = (
    '--target-run-id', TARGET_RUN,
    '--finding-id', FINDING,
    '--memory-id', DANGLING,
)


# ===========================================================================
# build_parser
# ===========================================================================


class TestBuildParser:
    """Flag surface and defaults, asserted without invoking main()."""

    def test_parses_the_repair_identifiers(self):
        args = _parse(*_REQUIRED, '--replacement-memory-id', SUCCESSOR)
        assert args.target_run_id == TARGET_RUN
        assert args.finding_id == FINDING
        assert args.memory_id == DANGLING
        assert args.replacement_memory_id == SUCCESSOR

    def test_dry_run_is_the_default(self):
        """The write must be opted into, per the one-shot-script convention.

        This script rewrites a historical audit record; a default that wrote
        would make a typo'd id destructive on first invocation.
        """
        assert _parse(*_REQUIRED).apply is False
        assert _parse(*_REQUIRED, '--apply').apply is True

    def test_store_defaults_to_mem0(self):
        assert _parse(*_REQUIRED).store == 'mem0'
        assert _parse(*_REQUIRED, '--store', 'graphiti').store == 'graphiti'

    def test_replacement_defaults_to_none_for_drop_only_mode(self):
        assert _parse(*_REQUIRED).replacement_memory_id is None

    def test_data_dir_flag_exists(self):
        args = _parse(*_REQUIRED, '--data-dir', '/tmp/recon')
        assert Path(args.data_dir) == Path('/tmp/recon')


# ===========================================================================
# run() — delegation
# ===========================================================================


class TestRunDelegates:
    """``run`` forwards to the shared function and returns its outcome verbatim."""

    @pytest.fixture
    def spy(self, monkeypatch):
        called = AsyncMock(return_value={'status': 'dry_run', 'stage': 's'})
        monkeypatch.setattr(_mod.citation_repair, 'repair_memory_citation', called)
        return called

    @pytest.mark.asyncio
    async def test_forwards_ids_and_apply_verbatim(self, spy):
        journal, memory = object(), object()
        args = _parse(
            *_REQUIRED, '--replacement-memory-id', SUCCESSOR, '--apply'
        )

        outcome = await _mod.run(args, journal=journal, memory=memory)

        assert spy.await_count == 1
        call = spy.await_args
        assert call.args[0] is journal
        assert call.args[1] is memory
        assert call.kwargs['target_run_id'] == TARGET_RUN
        assert call.kwargs['finding_id'] == FINDING
        assert call.kwargs['memory_id'] == DANGLING
        assert call.kwargs['replacement_memory_id'] == SUCCESSOR
        assert call.kwargs['store'] == 'mem0'
        assert call.kwargs['apply'] is True
        assert outcome == {'status': 'dry_run', 'stage': 's'}

    @pytest.mark.asyncio
    async def test_dry_run_threads_apply_false(self, spy):
        await _mod.run(args := _parse(*_REQUIRED), journal=None, memory=None)
        assert args.apply is False
        assert spy.await_args.kwargs['apply'] is False

    @pytest.mark.asyncio
    async def test_omitted_replacement_forwards_none(self, spy):
        """Drop-only mode: no replacement is not an empty string."""
        await _mod.run(_parse(*_REQUIRED), journal=None, memory=None)
        assert spy.await_args.kwargs['replacement_memory_id'] is None

    @pytest.mark.asyncio
    async def test_stamps_a_repaired_by_naming_this_script(self, spy):
        """Provenance must say a human ran the operator script, not a run id."""
        await _mod.run(_parse(*_REQUIRED), journal=None, memory=None)
        assert 'repair_recon_citation' in spy.await_args.kwargs['repaired_by']


# ===========================================================================
# Exit-code mapping
# ===========================================================================


class TestExitCode:
    """Structured facts on stdout; 0 only when the repair actually resolved."""

    def test_success_statuses_exit_zero(self):
        assert _mod.exit_code_for({'status': 'repaired'}) == 0
        assert _mod.exit_code_for({'status': 'dry_run'}) == 0

    def test_any_error_exits_one(self):
        assert _mod.exit_code_for({'error': 'citation_not_dangling'}) == 1
        assert _mod.exit_code_for({'error': 'run_still_live'}) == 1

    def test_unrecognised_outcome_exits_one(self):
        """An outcome with neither key is a contract break, not a success."""
        assert _mod.exit_code_for({}) == 1

    def test_outcome_is_printed_as_json(self, capsys):
        """JSON, not prose — the operator gets the structured facts (INV-2)."""
        import json

        outcome = {'error': 'replacement_not_found', 'error_type': 'X'}
        code = _mod.report(outcome)

        assert code == 1
        assert json.loads(capsys.readouterr().out) == outcome
