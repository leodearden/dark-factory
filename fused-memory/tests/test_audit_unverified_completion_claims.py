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
CorpusRecord = _mod.CorpusRecord
ScannedRecord = _mod.ScannedRecord
scan_records = _mod.scan_records

KNOWN_PROJECTS = frozenset({'dark_factory', 'reify'})

#: esc-3085-1 instance (2), verbatim: a filing/dispatch claim naming a ticket
#: id that does not exist in the registry.
ESC_3085_1_INSTANCE_2 = (
    'reify task 5638 was reported unactionable and re-filed into '
    "dark_factory's task tree as ticket tkt_0RRRC5AASJ9Z630VP4PCN9H376"
)

#: esc-3085-1 instance (1): an applied-work claim about a still-open task.
ESC_3085_1_INSTANCE_1 = "task 5422's de-flake fix has been applied"


def _record(
    uuid: str = 'ep-1',
    text: str = '',
    category: str | None = 'temporal_facts',
    project_id: str = 'reify',
    created_at: str = '2026-07-26T00:00:00Z',
    source_description: str = 'add_memory:temporal_facts',
    name: str = 'episode',
) -> object:
    """Build a CorpusRecord without touching a store."""
    return CorpusRecord(
        uuid=uuid,
        kind='episode',
        text=text,
        source_description=source_description,
        category=category,
        project_id=project_id,
        created_at=created_at,
        name=name,
    )


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


class TestScanRecords:
    """The imported extractor, run over the corpus projection.

    The script contributes NO claim-detection regex of its own — the whole
    point of reusing completion_claim_gate is that a second, drifting copy of
    the negation/aspirational strippers is what stops "has not yet landed"
    from being read as a completion (the gate's own docstring, :20-26).
    """

    def _scan(self, records: list[object]) -> list[object]:
        return scan_records(
            records,
            default_project_id='reify',
            known_project_ids=KNOWN_PROJECTS,
            categories=IN_SCOPE_CATEGORIES,
        )

    def test_esc_3085_1_instance_2_yields_the_ticket_claim(self) -> None:
        """Ticket > commit > task precedence (completion_claim_gate.py:405-411).

        The clause names BOTH task 5638 and the ticket; it is ONE claim about
        the ticket — the most specific ref is what the writer asserted into
        existence.
        """
        scanned = self._scan([_record(text=ESC_3085_1_INSTANCE_2)])
        assert len(scanned) == 1
        claims = scanned[0].claims
        assert len(claims) == 1
        assert claims[0].kind == 'filing_dispatch'
        assert claims[0].subject == 'ticket'
        assert claims[0].ref == 'tkt_0RRRC5AASJ9Z630VP4PCN9H376'

    def test_esc_3085_1_instance_1_yields_the_task_claim(self) -> None:
        scanned = self._scan([_record(text=ESC_3085_1_INSTANCE_1)])
        assert len(scanned) == 1
        claims = scanned[0].claims
        assert len(claims) == 1
        assert claims[0].kind == 'applied_work'
        assert claims[0].subject == 'task'
        assert claims[0].ref == '5422'
        assert claims[0].project_id == 'reify'

    def test_negated_claim_yields_nothing(self) -> None:
        """Proves the imported strippers are LIVE, not re-derived."""
        scanned = self._scan(
            [_record(text='the fix has not been applied for task 5422')]
        )
        assert scanned == []

    def test_out_of_scope_category_is_excluded(self) -> None:
        scanned = self._scan([
            _record(uuid='ep-a', text=ESC_3085_1_INSTANCE_1,
                    category='entities_and_relations'),
            _record(uuid='ep-b', text=ESC_3085_1_INSTANCE_1, category=None),
        ])
        assert scanned == []

    def test_records_without_claims_are_dropped(self) -> None:
        scanned = self._scan([
            _record(uuid='ep-a', text='an ordinary observation about the tree'),
            _record(uuid='ep-b', text=''),
            _record(uuid='ep-c', text=ESC_3085_1_INSTANCE_1),
        ])
        assert [s.record.uuid for s in scanned] == ['ep-c']

    def test_output_order_is_deterministic(self) -> None:
        """Sorted by (category, created_at, uuid) — never dict iteration order."""
        records = [
            _record(uuid='ep-z', text=ESC_3085_1_INSTANCE_1,
                    category='temporal_facts', created_at='2026-07-27T00:00:00Z'),
            _record(uuid='ep-a', text=ESC_3085_1_INSTANCE_1,
                    category='temporal_facts', created_at='2026-07-26T00:00:00Z'),
            _record(uuid='ep-m', text=ESC_3085_1_INSTANCE_1,
                    category='decisions_and_rationale',
                    created_at='2026-07-28T00:00:00Z'),
            _record(uuid='ep-b', text=ESC_3085_1_INSTANCE_1,
                    category='temporal_facts', created_at='2026-07-26T00:00:00Z'),
        ]
        expected = ['ep-m', 'ep-a', 'ep-b', 'ep-z']
        assert [s.record.uuid for s in self._scan(records)] == expected
        assert [s.record.uuid for s in self._scan(list(reversed(records)))] == expected

    def test_record_project_id_overrides_the_default(self) -> None:
        scanned = self._scan(
            [_record(text=ESC_3085_1_INSTANCE_1, project_id='dark_factory')]
        )
        assert scanned[0].claims[0].project_id == 'dark_factory'
