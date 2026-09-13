"""Tests for scripts/backfill_entity_standing_decision.py (task 2900 η).

The one-shot migration that turns the reify mem0 record ``b0057f3d`` — the
sole live ``recurring_flag_standing_decision`` — into an
``entity_standing_decision`` ledger row (PRD
``plans/stage1-entity-standing-decision-prd.md`` decision 5), and stamps the
ad-hoc mem0 originals evidence-only (decision 6 / Open Question 6).

The script lives under ``scripts/``, which is not a package and not on
PYTHONPATH, so it is loaded through the SHARED ``load_script_module`` helper
rather than a local ``spec_from_file_location`` copy — see the mandate in
``tests/conftest.py`` (tasks 3738 / 3895).

The fixture corpus below is not invented: it mirrors the reify records
MEASURED for entity ``f02a32ea`` on 2026-09-13. That matters most for the two
``stage1_flag_suppression`` records, which share the entity uuid but are a
RATIFIED machine-read kind (``flag_dedup.filter_suppressed`` consumes them) —
they are what makes the kind half of the selection predicate load-bearing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _fm_helpers import load_script_module

from fused_memory.utils.validation import is_full_uuid

SCRIPT_PATH = (
    Path(__file__).parent.parent / 'scripts' / 'backfill_entity_standing_decision.py'
)

_mod = load_script_module(SCRIPT_PATH, mod_name='backfill_entity_standing_decision')


# ---------------------------------------------------------------------------
# The measured live corpus (reify, 2026-09-13)
# ---------------------------------------------------------------------------

#: The entity every migrated/stamped record concerns — reify's 'orchestrator'
#: node. Spelled HERE, in the test's fixture data, and deliberately NOT in the
#: script: the script DERIVES it from the fetched source record (SPOT, plan
#: design decision 3), so a copy pinned in the script is exactly the drift this
#: suite exists to forbid.
ENTITY_UUID = 'f02a32ea-0efd-4865-94b4-97a412d8ffda'

SOURCE_ID = 'b0057f3d-dc53-4cf8-9d1f-9959bd0897bd'
CORRECTION_ID = 'baf8ca57-9f36-431b-a9b9-17c82fadd22d'
OFF_ENTITY_CORRECTION_ID = '12c3a5ce-1a0e-4c6b-9d3c-6b5a2e0f7a41'
OFF_ENTITY_UUID = '6643a9a7-5e58-4e1e-8e4f-3f1c2d9b7a10'


def _record(memory_id: str, kind: str | None, entity_uuid: str | None) -> dict:
    """One mem0 record envelope as ``get_memories_by_metadata`` returns it."""
    metadata: dict[str, object] = {}
    if kind is not None:
        metadata['kind'] = kind
    if entity_uuid is not None:
        metadata['entity_uuid'] = entity_uuid
    return {'id': memory_id, 'content': f'record {memory_id}', 'metadata': metadata}


SOURCE_RECORD = _record(SOURCE_ID, 'recurring_flag_standing_decision', ENTITY_UUID)


# ---------------------------------------------------------------------------
# resolve_source_entity_uuid — the SPOT derivation, loudly validated
# ---------------------------------------------------------------------------


class TestResolveSourceEntityUuid:
    """The entity uuid comes from the fetched record, or the run stops."""

    def test_derives_the_uuid_from_the_live_source_payload(self) -> None:
        assert _mod.resolve_source_entity_uuid(SOURCE_RECORD) == ENTITY_UUID

    @pytest.mark.parametrize(
        'record',
        [
            pytest.param(None, id='not_found_none'),
            pytest.param({'found': False}, id='not_found_envelope'),
        ],
    )
    def test_missing_source_record_fails_loudly(self, record: object) -> None:
        with pytest.raises(_mod.BackfillSourceInvalid) as excinfo:
            _mod.resolve_source_entity_uuid(record)
        message = str(excinfo.value)
        assert SOURCE_ID in message
        assert 'hint' in message.lower()

    def test_wrong_kind_fails_loudly_naming_both_kinds(self) -> None:
        drifted = _record(SOURCE_ID, 'investigation_outcome', ENTITY_UUID)
        with pytest.raises(_mod.BackfillSourceInvalid) as excinfo:
            _mod.resolve_source_entity_uuid(drifted)
        message = str(excinfo.value)
        assert _mod.SOURCE_KIND in message
        assert 'investigation_outcome' in message

    @pytest.mark.parametrize(
        ('entity_uuid', 'label'),
        [
            pytest.param(None, 'missing', id='missing'),
            pytest.param('', 'empty', id='empty'),
            pytest.param('f02a32ea', 'short_hex', id='truncated'),
            pytest.param(
                'f02a32ea0efd486594b497a412d8ffda', 'undashed', id='undashed'
            ),
        ],
    )
    def test_non_canonical_entity_uuid_fails_loudly(
        self, entity_uuid: str | None, label: str
    ) -> None:
        """Canonicality is judged by the shared ``is_full_uuid`` (INV-5)."""
        assert not is_full_uuid(entity_uuid)
        broken = _record(SOURCE_ID, _mod.SOURCE_KIND, entity_uuid)
        with pytest.raises(_mod.BackfillSourceInvalid) as excinfo:
            _mod.resolve_source_entity_uuid(broken)
        assert 'entity_uuid' in str(excinfo.value)

    def test_the_error_is_module_typed_not_a_bare_valueerror(self) -> None:
        """A caller can catch this migration's defect without swallowing others."""
        assert issubclass(_mod.BackfillSourceInvalid, ValueError)
        assert _mod.BackfillSourceInvalid is not ValueError


class TestPinnedConstants:
    """Exactly ONE memory id is pinned for the source, and the entity uuid is
    pinned NOWHERE (plan design decision 3 — SPOT).

    Asserted against the module's live globals rather than by eyeballing the
    source, so a second copy added later fails this suite instead of drifting
    quietly out of agreement with the record it claims to describe.
    """

    @staticmethod
    def _pinned_uuids() -> list[tuple[str, str]]:
        """Every canonical-UUID string reachable from a module-level global."""
        found: list[tuple[str, str]] = []
        for name, value in vars(_mod).items():
            if name.startswith('__'):
                continue
            candidates = (
                [value]
                if isinstance(value, str)
                else list(value)
                if isinstance(value, (tuple, list, frozenset, set))
                else []
            )
            found.extend(
                (name, item) for item in candidates if is_full_uuid(item)
            )
        return found

    def test_source_memory_id_is_the_live_record(self) -> None:
        assert _mod.SOURCE_MEMORY_ID == SOURCE_ID

    def test_source_id_is_pinned_exactly_once(self) -> None:
        pins = [name for name, value in self._pinned_uuids() if value == SOURCE_ID]
        assert pins == ['SOURCE_MEMORY_ID']

    def test_entity_uuid_is_never_pinned(self) -> None:
        """It is DERIVED from the fetched record, so no global may hold it."""
        assert [
            name for name, value in self._pinned_uuids() if value == ENTITY_UUID
        ] == []
