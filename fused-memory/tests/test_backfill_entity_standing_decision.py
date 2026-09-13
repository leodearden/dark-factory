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

from fused_memory.memory_metadata import (
    EXPERIMENTAL_KEY_PREFIX,
    classify_unknown_keys,
)
from fused_memory.reconciliation.standing_decision_constants import (
    GROUNDS_STRUCTURAL_SIZE_CONFLATION,
)
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

#: The two records PRD decision 5 demotes, in the order the selector must
#: return them. Spelled as ``sorted(...)`` rather than hand-transcribed: the
#: contract is "deterministic, independent of scroll order", and a literal pair
#: would assert a particular lexicographic accident instead.
EXPECTED_STAMP_TARGETS = sorted([SOURCE_ID, CORRECTION_ID])


def _record(memory_id: str, kind: str | None, entity_uuid: str | None) -> dict:
    """One mem0 record envelope as ``get_memories_by_metadata`` returns it."""
    metadata: dict[str, object] = {}
    if kind is not None:
        metadata['kind'] = kind
    if entity_uuid is not None:
        metadata['entity_uuid'] = entity_uuid
    return {'id': memory_id, 'content': f'record {memory_id}', 'metadata': metadata}


SOURCE_RECORD = _record(SOURCE_ID, 'recurring_flag_standing_decision', ENTITY_UUID)

#: The six records the entity-scoped scroll returned for ``f02a32ea`` on
#: 2026-09-13, plus one ``stage1_finding_correction`` belonging to a DIFFERENT
#: entity — the record the research doc's pinned id list would have stamped.
LIVE_ENTITY_SCROLL = [
    SOURCE_RECORD,
    _record(CORRECTION_ID, 'stage1_finding_correction', ENTITY_UUID),
    _record('39528550-6d0c-4a5f-8a55-9b2c1e7d4f03', 'flag_correction', ENTITY_UUID),
    _record('a04ae6e8-3b71-4c2a-9f18-5d6e8c0a1b29', 'flag_correction', ENTITY_UUID),
    _record('aa46fbad-0c2e-4e7b-8a19-2f7d5b3c6e84', 'stage1_flag_suppression', ENTITY_UUID),
    _record('d79f6b28-4a13-45c9-b6e2-8c0f1a9d7e35', 'stage1_flag_suppression', ENTITY_UUID),
    _record(OFF_ENTITY_CORRECTION_ID, 'stage1_finding_correction', OFF_ENTITY_UUID),
]


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


# ---------------------------------------------------------------------------
# select_evidence_only_targets — both halves of the predicate are load-bearing
# ---------------------------------------------------------------------------


class TestSelectEvidenceOnlyTargets:
    """Selection is (kind ∈ allowlist) AND (entity_uuid == the source's)."""

    def test_selects_exactly_the_two_unratified_ad_hoc_records(self) -> None:
        assert _mod.select_evidence_only_targets(
            LIVE_ENTITY_SCROLL, ENTITY_UUID
        ) == EXPECTED_STAMP_TARGETS

    def test_order_is_deterministic_across_input_orderings(self) -> None:
        """Two runs over the same corpus must plan the same stamps in the same
        order, whatever order the scroll happened to return."""
        shuffled = list(reversed(LIVE_ENTITY_SCROLL))
        assert _mod.select_evidence_only_targets(
            shuffled, ENTITY_UUID
        ) == _mod.select_evidence_only_targets(LIVE_ENTITY_SCROLL, ENTITY_UUID)

    def test_ratified_machine_read_suppressions_are_excluded(self) -> None:
        """``stage1_flag_suppression`` shares the entity uuid but is CONSUMED by
        ``flag_dedup.filter_suppressed``.

        Stamping those evidence-only would be a live behaviour change, not a
        bookkeeping annotation — which is why the entity-scoped scroll alone is
        not a safe stamp gate.
        """
        selected = _mod.select_evidence_only_targets(LIVE_ENTITY_SCROLL, ENTITY_UUID)
        suppressions = [
            record['id']
            for record in LIVE_ENTITY_SCROLL
            if record['metadata'].get('kind') == 'stage1_flag_suppression'
        ]
        assert len(suppressions) == 2
        assert not set(suppressions) & set(selected)
        assert 'stage1_flag_suppression' not in _mod.DEMOTED_AD_HOC_KINDS

    def test_flag_corrections_are_excluded(self) -> None:
        """``flag_correction`` is not one of the two kinds the PRD demotes."""
        selected = _mod.select_evidence_only_targets(LIVE_ENTITY_SCROLL, ENTITY_UUID)
        assert '39528550-6d0c-4a5f-8a55-9b2c1e7d4f03' not in selected
        assert 'a04ae6e8-3b71-4c2a-9f18-5d6e8c0a1b29' not in selected

    def test_off_entity_correction_is_excluded(self) -> None:
        """A right-kind record about a DIFFERENT entity.

        Measured 2026-09-13: reify now holds 5 ``stage1_finding_correction``
        records (the research doc measured 2) and only ``baf8ca57`` concerns
        this entity, so a pinned id list would stamp an unrelated record.
        """
        assert OFF_ENTITY_CORRECTION_ID not in _mod.select_evidence_only_targets(
            LIVE_ENTITY_SCROLL, ENTITY_UUID
        )

    @pytest.mark.parametrize(
        'malformed',
        [
            pytest.param({'id': 'no-metadata-at-all'}, id='no_metadata'),
            pytest.param({'id': 'null-metadata', 'metadata': None}, id='null_metadata'),
            pytest.param(_record('no-kind', None, ENTITY_UUID), id='no_kind'),
            pytest.param(_record('no-entity', 'stage1_finding_correction', None), id='no_entity'),
            pytest.param({'metadata': {'kind': 'stage1_finding_correction',
                                       'entity_uuid': ENTITY_UUID}}, id='no_id'),
        ],
    )
    def test_malformed_records_are_excluded_not_crashed_on(self, malformed: dict) -> None:
        selected = _mod.select_evidence_only_targets(
            [*LIVE_ENTITY_SCROLL, malformed], ENTITY_UUID
        )
        assert selected == EXPECTED_STAMP_TARGETS

    def test_empty_corpus_selects_nothing(self) -> None:
        assert _mod.select_evidence_only_targets([], ENTITY_UUID) == []


# ---------------------------------------------------------------------------
# The two pure builders — evidence refs, and the Open-Question-6 stamp
# ---------------------------------------------------------------------------


class TestBuildEvidenceRefs:
    """Every cited id reaches β verbatim, exactly once, source first."""

    @staticmethod
    def _refs() -> list[dict]:
        return _mod.build_evidence_refs(EXPECTED_STAMP_TARGETS, ENTITY_UUID)

    def test_cites_every_stamp_target_and_every_pinned_human_record(self) -> None:
        mem0_ids = [
            ref['id'] for ref in self._refs() if ref['type'] == 'mem0'
        ]
        assert set(mem0_ids) == {
            *EXPECTED_STAMP_TARGETS,
            *_mod.HUMAN_EVIDENCE_MEMORY_IDS,
        }

    def test_cites_the_opening_escalation_as_a_foreign_ref(self) -> None:
        escalation_refs = [
            ref for ref in self._refs() if ref['type'] == 'escalation'
        ]
        assert escalation_refs == [
            {'type': 'escalation', 'id': _mod.ESCALATION_EVIDENCE_ID}
        ]

    def test_the_source_record_is_cited_first(self) -> None:
        """The row's provenance reads as "this record, plus its corroboration"."""
        assert self._refs()[0] == {'type': 'mem0', 'id': _mod.SOURCE_MEMORY_ID}

    def test_no_id_is_cited_twice(self) -> None:
        ids = [ref['id'] for ref in self._refs()]
        assert len(ids) == len(set(ids))

    def test_refs_are_bare_and_unresolved(self) -> None:
        """β's ``resolve_evidence_refs`` owns ``locally_resolved`` (INV-5).

        A builder that pre-stamped resolution would be a second, un-run opinion
        about whether an id exists.
        """
        assert all(set(ref) == {'type', 'id'} for ref in self._refs())

    def test_an_empty_stamp_plan_still_cites_the_pinned_evidence(self) -> None:
        """An idempotent re-run stamps nothing, but the row's provenance must not
        shrink to depend on what a previous run happened to leave unstamped."""
        refs = _mod.build_evidence_refs([], ENTITY_UUID)
        ids = {ref['id'] for ref in refs}
        assert _mod.SOURCE_MEMORY_ID in ids
        assert set(_mod.HUMAN_EVIDENCE_MEMORY_IDS) <= ids
        assert _mod.ESCALATION_EVIDENCE_ID in ids


class TestBuildEvidenceOnlyPatch:
    """PRD Open Question 6: four flat Tier-C scalar keys."""

    MIGRATED_AT = '2026-09-13T12:00:00+00:00'

    @classmethod
    def _patch(cls) -> dict:
        return _mod.build_evidence_only_patch(
            entity_uuid=ENTITY_UUID,
            grounds=_mod.GROUNDS,
            migrated_at=cls.MIGRATED_AT,
        )

    def test_carries_exactly_the_four_decided_keys(self) -> None:
        assert set(self._patch()) == {
            'x_standing_decision_status',
            'x_standing_decision_entity_uuid',
            'x_standing_decision_grounds',
            'x_standing_decision_migrated_at',
        }

    def test_status_marks_the_record_evidence_only(self) -> None:
        assert self._patch()['x_standing_decision_status'] == 'evidence_only'

    def test_grounds_and_entity_are_carried_as_separate_fields(self) -> None:
        """NOT δ's joined ``f'{uuid}:{grounds}'`` id — structured data, not a
        meaningful string, and no second site re-deriving that join."""
        patch = self._patch()
        assert patch['x_standing_decision_entity_uuid'] == ENTITY_UUID
        assert patch['x_standing_decision_grounds'] == GROUNDS_STRUCTURAL_SIZE_CONFLATION

    def test_migrated_at_is_echoed_verbatim(self) -> None:
        assert self._patch()['x_standing_decision_migrated_at'] == self.MIGRATED_AT

    def test_every_key_carries_the_experimental_prefix(self) -> None:
        assert all(
            key.startswith(EXPERIMENTAL_KEY_PREFIX) for key in self._patch()
        )

    def test_every_value_is_a_queryable_scalar_string(self) -> None:
        """Flat scalars, never one nested dict: Qdrant payload filters do
        exact-match on scalars, so an operator can enumerate every demoted
        record with ``get_memories_by_metadata(reify, {...: 'evidence_only'})``.
        """
        assert all(isinstance(value, str) for value in self._patch().values())

    def test_the_stamp_adds_no_unknown_key_census_line(self) -> None:
        """Checked against the REAL validator, not by eyeballing the prefixes."""
        assert classify_unknown_keys(self._patch()) == []
