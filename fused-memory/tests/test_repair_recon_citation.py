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

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _fm_helpers import load_script_module

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'repair_recon_citation.py'

TARGET_RUN = '06a4466d-cdc0-49ac-8e99-e6723be39392'
FINDING = '5e85117e-51fc-4a7f-8ca7-e26078dbd3f2'
DANGLING = 'beacf7fc-b76a-4c0b-876d-f4cf6d906d42'
SUCCESSOR = '746b4ab9-ca3c-418b-982a-32b85bfcf94b'


_mod = load_script_module(SCRIPT_PATH, mod_name='repair_recon_citation')


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

    def test_replacement_defaults_to_none_for_drop_only_mode(self):
        assert _parse(*_REQUIRED).replacement_memory_id is None

    def test_reason_defaults_to_the_dangling_class(self):
        """The default must stay the class this script shipped with (task 5552).

        ``memory_not_found`` is the corroboration an operator gets without
        saying anything; ``wrong_memory`` removes a citation that still
        resolves, so it has to be asked for explicitly.
        """
        assert _parse(*_REQUIRED).reason == 'memory_not_found'

    def test_reason_accepts_the_wrong_memory_class(self):
        assert _parse(*_REQUIRED, '--reason', 'wrong_memory').reason == 'wrong_memory'

    def test_reason_outside_the_enum_is_rejected_by_argparse(self):
        """Closed at the CLI boundary, mirroring ``--store``'s ``choices``.

        The repair also refuses an unknown reason with ``invalid_reason``, but
        an operator typo should not need a journal open to be told.
        """
        with pytest.raises(SystemExit):
            _parse(*_REQUIRED, '--reason', 'detach')

    def test_justification_defaults_to_none(self):
        assert _parse(*_REQUIRED).justification is None

    def test_justification_round_trips_the_supplied_prose(self):
        prose = 'mis-cites task 168 rolling summary; claim confirmed via get_task(182)'
        assert _parse(*_REQUIRED, '--justification', prose).justification == prose


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
        await _mod.run(_parse(*_REQUIRED), journal=None, memory=None)
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

    @pytest.mark.asyncio
    async def test_forwards_the_defect_class_and_its_justification(self, spy):
        """Both reach the repair as kwargs; the script gates neither (task 5552)."""
        prose = 'cites the pre-consolidation summary, not the decision it claims'
        args = _parse(*_REQUIRED, '--reason', 'wrong_memory', '--justification', prose)

        await _mod.run(args, journal=None, memory=None)

        assert spy.await_args.kwargs['reason'] == 'wrong_memory'
        assert spy.await_args.kwargs['justification'] == prose

    @pytest.mark.asyncio
    async def test_omitted_reason_and_justification_forward_their_defaults(self, spy):
        """Forwarded EXPLICITLY, not left to the function's own defaults.

        A script that omitted the kwargs would still repair today, and would
        silently stop tracking the enum the day either default changed.
        """
        await _mod.run(_parse(*_REQUIRED), journal=None, memory=None)

        assert spy.await_args.kwargs['reason'] == 'memory_not_found'
        assert spy.await_args.kwargs['justification'] is None


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

    def test_the_reason_scoped_refusals_exit_one(self):
        """The task-5552 refusals need no mapping entry — but they are pinned.

        ``exit_code_for`` allowlists the two SUCCESS statuses rather than
        enumerating errors, so a new refusal exits 1 for free. That is the
        property worth pinning: it is what keeps the script from having to be
        edited every time the repair grows a gate.
        """
        assert _mod.exit_code_for({'error': 'citation_not_resolving'}) == 1
        assert _mod.exit_code_for({'error': 'justification_required'}) == 1
        assert _mod.exit_code_for({'error': 'invalid_reason'}) == 1
        assert _mod.exit_code_for({'error': 'replacement_is_victim'}) == 1

    def test_a_run_status_is_never_read_as_an_outcome_status(self):
        """The real ``run_still_live`` shape, with its own status key present.

        ``run_still_live`` is the one refusal that also reports the TARGET RUN's
        status. It reports it under ``run_status`` precisely so this function —
        which discriminates on ``status`` — cannot read a run status as an
        outcome status. Fed the dual-key shape, the exit code must still be 1;
        that a RunStatus value never literally equals 'repaired' or 'dry_run' is
        luck, and this pins the boundary rather than relying on it.
        """
        assert _mod.exit_code_for(
            {
                'error': 'run_still_live',
                'error_type': 'ReconCitationRunStillLive',
                'run_status': 'interrupted',
            }
        ) == 1
        # The overloaded shape this rename retired. It still exits 1 — but only
        # because no RunStatus value happens to spell 'repaired' or 'dry_run'.
        # Pinned in both directions so a regression that reintroduces the
        # overload is caught here rather than by a future enum member.
        assert _mod.exit_code_for({'error': 'run_still_live', 'status': 'running'}) == 1

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


# ===========================================================================
# main() — live wiring
# ===========================================================================


class TestMain:
    """What ``main()`` actually constructs, and what it closes.

    The ``--data-dir`` fallback, the journal/memory construction and the
    ``finally`` teardown live ONLY here — asserting that argparse stores the
    string it was handed exercises no project code at all.
    """

    @pytest.fixture
    def wiring(self, monkeypatch):
        """Stand in for the three live constructions ``main()`` makes.

        ``_run_live`` imports each one INSIDE the function, so patching the
        defining module is what the call actually resolves through.
        """
        seen: dict[str, Any] = {'closed': []}

        class FakeJournal:
            def __init__(self, data_dir):
                seen['data_dir'] = data_dir
                seen['journal'] = self

            async def initialize(self):
                seen['journal_initialized'] = True

            async def close(self):
                seen['closed'].append('journal')

        class FakeMemory:
            def __init__(self, config):
                seen['memory_config'] = config

            async def initialize(self):
                seen['memory_initialized'] = True

            async def close(self):
                seen['closed'].append('memory')

        class FakeConfig:
            reconciliation = SimpleNamespace(data_dir='/configured/recon')

        monkeypatch.setattr(
            'fused_memory.config.schema.FusedMemoryConfig', FakeConfig
        )
        monkeypatch.setattr(
            'fused_memory.reconciliation.journal.ReconciliationJournal', FakeJournal
        )
        monkeypatch.setattr(
            'fused_memory.services.memory_service.MemoryService', FakeMemory
        )
        return seen

    @staticmethod
    def _patch_repair(monkeypatch, **kwargs) -> AsyncMock:
        spy = AsyncMock(**kwargs)
        monkeypatch.setattr(_mod.citation_repair, 'repair_memory_citation', spy)
        return spy

    def test_data_dir_flag_selects_the_journal_location(
        self, monkeypatch, wiring, capsys
    ):
        self._patch_repair(monkeypatch, return_value={'status': 'dry_run'})
        monkeypatch.setattr(
            sys, 'argv', ['repair_recon_citation', *_REQUIRED, '--data-dir', '/tmp/recon']
        )

        assert _mod.main() == 0

        assert wiring['data_dir'] == Path('/tmp/recon')
        assert wiring['journal_initialized'] is True
        assert wiring['memory_initialized'] is True
        capsys.readouterr()

    def test_omitted_data_dir_falls_back_to_the_configured_one(
        self, monkeypatch, wiring, capsys
    ):
        """The default is the CONFIGURED reconciliation dir, not a literal.

        An operator who omits the flag must hit the same journal the running
        server owns; a wrong default would silently repair a different DB.
        """
        self._patch_repair(monkeypatch, return_value={'status': 'dry_run'})
        monkeypatch.setattr(sys, 'argv', ['repair_recon_citation', *_REQUIRED])

        assert _mod.main() == 0

        assert wiring['data_dir'] == Path('/configured/recon')
        capsys.readouterr()

    def test_teardown_closes_memory_then_journal(self, monkeypatch, wiring, capsys):
        self._patch_repair(monkeypatch, return_value={'status': 'repaired'})
        monkeypatch.setattr(sys, 'argv', ['repair_recon_citation', *_REQUIRED, '--apply'])

        assert _mod.main() == 0

        assert wiring['closed'] == ['memory', 'journal']
        capsys.readouterr()

    def test_journal_is_closed_even_when_the_repair_raises(
        self, monkeypatch, wiring
    ):
        """A leaked SQLite handle would outlive a crash and hold the WAL open."""
        self._patch_repair(monkeypatch, side_effect=RuntimeError('boom'))
        monkeypatch.setattr(sys, 'argv', ['repair_recon_citation', *_REQUIRED])

        with pytest.raises(RuntimeError, match='boom'):
            _mod.main()

        assert wiring['closed'] == ['memory', 'journal']
