"""Tests for audit_unverified_completion_claims.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution — mirrors the pattern in test_audit_found_on_main_provenance.py
/ test_audit_duplicate_memories.py.
"""
from __future__ import annotations

import importlib.util
import types
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).parent.parent / 'scripts' / 'audit_unverified_completion_claims.py'
)


def _load_module() -> types.ModuleType:
    """Load audit_unverified_completion_claims.py from its file path.

    The module is registered in sys.modules under its name so that
    @dataclass and other reflection-based decorators work correctly
    (they call sys.modules.get(cls.__module__)).
    """
    import sys  # noqa: PLC0415

    mod_name = 'audit_unverified_completion_claims'
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
parse_category = _mod.parse_category
IN_SCOPE_CATEGORIES = _mod.IN_SCOPE_CATEGORIES


class TestParseCategory:
    """The category label survives ONLY as the Episodic source_description.

    Graphiti persists no ``category`` property anywhere — not on Entity nodes,
    not on RELATES_TO edges. ``MemoryService`` stamps ``add_memory:<category>``
    (memory_service.py:2804) and ``replay_from_mem0:<category>`` (:3302), and
    ``graphiti_client`` may prepend ``[temporal:<ctx>] `` (:700) and — new from
    task 3142 — ``[unverified_claim] `` (:702). Recovering the label from that
    string is the only way to honour this task's category scope.
    """

    def test_add_memory_temporal_facts(self) -> None:
        assert parse_category('add_memory:temporal_facts') == 'temporal_facts'

    def test_add_memory_decisions_and_rationale(self) -> None:
        assert (
            parse_category('add_memory:decisions_and_rationale')
            == 'decisions_and_rationale'
        )

    def test_replay_from_mem0_prefix(self) -> None:
        assert parse_category('replay_from_mem0:temporal_facts') == 'temporal_facts'

    def test_temporal_context_prefix_is_stripped(self) -> None:
        assert (
            parse_category('[temporal:sprint-4] add_memory:temporal_facts')
            == 'temporal_facts'
        )

    def test_unverified_claim_prefix_is_stripped(self) -> None:
        """THE case that matters most.

        Task 3142 stamps ``[unverified_claim] `` at graphiti_client.py:702.
        Without stripping it, every episode the new write-time gate ALREADY
        flagged falls out of this sweep's population — precisely the records
        most worth reading.
        """
        assert (
            parse_category('[unverified_claim] add_memory:temporal_facts')
            == 'temporal_facts'
        )

    def test_both_prefixes_any_order(self) -> None:
        assert (
            parse_category(
                '[unverified_claim] [temporal:x] add_memory:decisions_and_rationale'
            )
            == 'decisions_and_rationale'
        )
        assert (
            parse_category(
                '[temporal:x] [unverified_claim] add_memory:decisions_and_rationale'
            )
            == 'decisions_and_rationale'
        )

    def test_caller_supplied_add_episode_description_is_uncategorized(self) -> None:
        assert parse_category('reconciliation stage 2 cycle summary') is None

    def test_unknown_category_does_not_mint_a_phantom_bucket(self) -> None:
        """Validated against models.enums.MemoryCategory, so a typo yields None."""
        assert parse_category('add_memory:not_a_real_category') is None

    def test_empty_and_none(self) -> None:
        assert parse_category('') is None
        assert parse_category(None) is None
        assert parse_category('   ') is None

    def test_in_scope_categories(self) -> None:
        assert IN_SCOPE_CATEGORIES == frozenset(
            {'temporal_facts', 'decisions_and_rationale'}
        )
