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

import asyncio
import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from _fm_helpers import load_script_module

from fused_memory.memory_metadata import (
    EXPERIMENTAL_KEY_PREFIX,
    classify_unknown_keys,
)
from fused_memory.reconciliation.recon_ledger import ReconLedgerStore
from fused_memory.reconciliation.standing_decision_constants import (
    GROUNDS_STRUCTURAL_SIZE_CONFLATION,
    STANDING_DECISION_TTL_DAYS,
    STATE_ACTIVE,
    STATE_EXPIRED,
    STATE_REVOKED,
)
from fused_memory.reconciliation.standing_decision_writer import (
    EVIDENCE_TYPE_OPERATOR_AUTHORIZATION,
    LedgerUnavailable,
)
from fused_memory.utils.store_mutation_preflight import StoreMutationUnavailable
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
        return _mod.build_evidence_refs(EXPECTED_STAMP_TARGETS)

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
        refs = _mod.build_evidence_refs([])
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


# ---------------------------------------------------------------------------
# plan_backfill — the whole decision surface, unit-testable without a store
# ---------------------------------------------------------------------------


def _stamped(record: dict) -> dict:
    """The same record after a previous run's evidence-only stamp."""
    patched = {**record, 'metadata': {**record['metadata']}}
    patched['metadata'][_mod.EVIDENCE_ONLY_STATUS_KEY] = _mod.EVIDENCE_ONLY_STATUS
    return patched


def _scroll_with(*stamped_ids: str) -> list[dict]:
    """The live scroll, with *stamped_ids* already carrying the stamp."""
    return [
        _stamped(record) if record.get('id') in stamped_ids else record
        for record in LIVE_ENTITY_SCROLL
    ]


class _Row:
    """The fields ``plan_backfill`` reads off an existing ledger row."""

    def __init__(self, state: str) -> None:
        self.state = state
        self.entity_uuid = ENTITY_UUID
        self.flag_type = GROUNDS_STRUCTURAL_SIZE_CONFLATION


class TestIsAlreadyStamped:
    def test_keys_on_the_status_field(self) -> None:
        assert not _mod.is_already_stamped(SOURCE_RECORD)
        assert _mod.is_already_stamped(_stamped(SOURCE_RECORD))

    def test_a_malformed_record_is_not_stamped(self) -> None:
        assert not _mod.is_already_stamped({'id': 'x'})
        assert not _mod.is_already_stamped({'id': 'x', 'metadata': None})


class TestPlanBackfill:
    """Four corpus/ledger states, one frozen plan each."""

    def test_fresh_corpus_plans_the_row_and_both_stamps(self) -> None:
        plan = _mod.plan_backfill(SOURCE_RECORD, LIVE_ENTITY_SCROLL, None)
        assert plan.entity_uuid == ENTITY_UUID
        assert plan.grounds == GROUNDS_STRUCTURAL_SIZE_CONFLATION
        assert plan.needs_ledger_write is True
        assert plan.stamp_targets == tuple(EXPECTED_STAMP_TARGETS)
        assert plan.evidence_refs == tuple(
            _mod.build_evidence_refs(EXPECTED_STAMP_TARGETS)
        )

    def test_a_fully_migrated_corpus_plans_nothing(self) -> None:
        """Idempotent re-run: an ACTIVE row plus both originals stamped."""
        plan = _mod.plan_backfill(
            SOURCE_RECORD,
            _scroll_with(*EXPECTED_STAMP_TARGETS),
            _Row(STATE_ACTIVE),
        )
        assert plan.needs_ledger_write is False
        assert plan.stamp_targets == ()

    def test_a_partially_stamped_corpus_completes_only_what_is_missing(self) -> None:
        plan = _mod.plan_backfill(
            SOURCE_RECORD, _scroll_with(SOURCE_ID), _Row(STATE_ACTIVE)
        )
        assert plan.needs_ledger_write is False
        assert plan.stamp_targets == (CORRECTION_ID,)

    @pytest.mark.parametrize('state', [STATE_EXPIRED, STATE_REVOKED])
    def test_a_lapsed_row_is_re_established_not_silently_skipped(
        self, state: str
    ) -> None:
        """``get_active_entity_standing_decision`` gates on ``state='active'``,
        so a TTL-expired or revoked row leaves γ/δ blind — the migration must
        write again rather than read "a row exists" as "the work is done"."""
        plan = _mod.plan_backfill(
            SOURCE_RECORD, _scroll_with(*EXPECTED_STAMP_TARGETS), _Row(state)
        )
        assert plan.needs_ledger_write is True

    def test_the_plan_is_frozen(self) -> None:
        """Nothing between planning and applying may edit the decision."""
        plan = _mod.plan_backfill(SOURCE_RECORD, LIVE_ENTITY_SCROLL, None)
        with pytest.raises(FrozenInstanceError):
            plan.needs_ledger_write = False

    def test_an_invalid_source_stops_the_plan(self) -> None:
        with pytest.raises(_mod.BackfillSourceInvalid):
            _mod.plan_backfill(None, LIVE_ENTITY_SCROLL, None)


# ---------------------------------------------------------------------------
# run_backfill — the live legs, against a REAL ledger and a faked mem0
# ---------------------------------------------------------------------------

#: The edge count the faked graphiti reports. β samples it at decision time and
#: ζ's growth sweep later compares against it, so the row must carry exactly
#: what was sampled — not a default and not a recount.
FAKE_EDGE_COUNT = 7


@pytest_asyncio.fixture
async def ledger(tmp_path):
    """A REAL, function-scoped ``ReconLedgerStore`` on a fresh tmp db."""
    store = ReconLedgerStore(tmp_path / 'reconciliation.db')
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()


def _memory_service(ledger_obj, *, scroll: list[dict] | None = None):
    """An AsyncMock mem0/graphiti façade with a REAL ledger mounted.

    Serves the four reads/writes ``run_backfill`` and β make: the source
    fetch, the entity scroll, the stamp write, and β's edge sample plus its
    best-effort mem0 mirror.
    """
    service = AsyncMock()
    service.recon_ledger = ledger_obj
    service.get_memory_by_id = AsyncMock(return_value=SOURCE_RECORD)
    service.get_memories_by_metadata = AsyncMock(
        return_value=LIVE_ENTITY_SCROLL if scroll is None else scroll
    )
    service.update_memory = AsyncMock(
        return_value={'status': 'updated', 'store': 'mem0'}
    )
    service.graphiti.get_valid_edges_for_node = AsyncMock(
        return_value=[{'uuid': f'edge-{n}'} for n in range(FAKE_EDGE_COUNT)]
    )
    return service


async def _rows(ledger_obj) -> list:
    return await ledger_obj.list_entity_standing_decisions(_mod.PROJECT_ID)


class TestRunBackfillDryRun:
    """A dry run reads and decides everything, and writes nothing."""

    @pytest.mark.asyncio
    async def test_writes_no_ledger_row_and_stamps_nothing(self, ledger) -> None:
        service = _memory_service(ledger)
        await _mod.run_backfill(service, apply=False)
        assert await _rows(ledger) == []
        service.update_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_report_still_names_the_work_it_would_do(self, ledger) -> None:
        service = _memory_service(ledger)
        report = _mod.run_backfill(service, apply=False)
        report = await report
        assert report['apply'] is False
        assert report['entity_uuid'] == ENTITY_UUID
        assert report['ledger'] == 'would_write'
        assert [row['memory_id'] for row in report['records']] == EXPECTED_STAMP_TARGETS
        assert {row['outcome'] for row in report['records']} == {'would_stamp'}

    @pytest.mark.asyncio
    async def test_the_report_is_json_serializable(self, ledger) -> None:
        report = await _mod.run_backfill(_memory_service(ledger), apply=False)
        assert json.loads(json.dumps(report)) == report


class TestRunBackfillApply:
    """``--apply`` writes exactly one row, then stamps exactly two records."""

    @pytest.mark.asyncio
    async def test_writes_one_active_row_for_the_decided_entity(self, ledger) -> None:
        await _mod.run_backfill(_memory_service(ledger), apply=True)
        rows = await _rows(ledger)
        assert len(rows) == 1
        assert rows[0].state == STATE_ACTIVE
        assert rows[0].entity_uuid == ENTITY_UUID
        assert rows[0].flag_type == GROUNDS_STRUCTURAL_SIZE_CONFLATION

    @pytest.mark.asyncio
    async def test_the_row_carries_the_freshly_sampled_edge_count(self, ledger) -> None:
        await _mod.run_backfill(_memory_service(ledger), apply=True)
        payload = json.loads((await _rows(ledger))[0].payload_json)
        assert payload['edge_count_at_decision'] == FAKE_EDGE_COUNT

    @pytest.mark.asyncio
    async def test_the_row_expires_after_the_shared_ttl(self, ledger) -> None:
        await _mod.run_backfill(_memory_service(ledger), apply=True)
        row = (await _rows(ledger))[0]
        span = datetime.fromisoformat(row.expires_at) - datetime.fromisoformat(
            row.created_at
        )
        assert span == timedelta(days=STANDING_DECISION_TTL_DAYS)

    @pytest.mark.asyncio
    async def test_the_row_cites_every_evidence_ref_plus_the_operator_bypass(
        self, ledger
    ) -> None:
        await _mod.run_backfill(_memory_service(ledger), apply=True)
        evidence = json.loads((await _rows(ledger))[0].payload_json)['evidence']
        mem0_ids = {
            ref['id'] for ref in evidence if ref['type'] == _mod.EVIDENCE_TYPE_MEM0
        }
        assert mem0_ids == {
            SOURCE_ID,
            CORRECTION_ID,
            *_mod.HUMAN_EVIDENCE_MEMORY_IDS,
        }
        assert {
            ref['type'] for ref in evidence
        } >= {EVIDENCE_TYPE_OPERATOR_AUTHORIZATION}

    @pytest.mark.asyncio
    async def test_the_foreign_escalation_ref_is_not_locally_resolved(
        self, ledger
    ) -> None:
        """β marks a non-mem0 ref unresolved WITHOUT a lookup — the escalation
        queue is not this project's mem0 corpus."""
        await _mod.run_backfill(_memory_service(ledger), apply=True)
        evidence = json.loads((await _rows(ledger))[0].payload_json)['evidence']
        escalation = [
            ref for ref in evidence if ref['type'] == _mod.EVIDENCE_TYPE_ESCALATION
        ]
        assert escalation == [
            {
                'type': _mod.EVIDENCE_TYPE_ESCALATION,
                'id': _mod.ESCALATION_EVIDENCE_ID,
                'locally_resolved': False,
            }
        ]

    @pytest.mark.asyncio
    async def test_stamps_exactly_the_two_demoted_originals(self, ledger) -> None:
        service = _memory_service(ledger)
        await _mod.run_backfill(service, apply=True)
        stamped = [
            call.kwargs['memory_id'] for call in service.update_memory.await_args_list
        ]
        assert sorted(stamped) == EXPECTED_STAMP_TARGETS

    @pytest.mark.asyncio
    async def test_never_stamps_a_machine_read_or_off_entity_record(
        self, ledger
    ) -> None:
        service = _memory_service(ledger)
        await _mod.run_backfill(service, apply=True)
        stamped = {
            call.kwargs['memory_id'] for call in service.update_memory.await_args_list
        }
        assert not stamped & {
            'aa46fbad-0c2e-4e7b-8a19-2f7d5b3c6e84',
            'd79f6b28-4a13-45c9-b6e2-8c0f1a9d7e35',
            OFF_ENTITY_CORRECTION_ID,
        }

    @pytest.mark.asyncio
    async def test_each_stamp_is_a_metadata_only_merge_of_the_four_keys(
        self, ledger
    ) -> None:
        service = _memory_service(ledger)
        await _mod.run_backfill(service, apply=True)
        for call in service.update_memory.await_args_list:
            assert call.kwargs['metadata_mode'] == 'merge'
            assert call.kwargs['project_id'] == _mod.PROJECT_ID
            assert call.kwargs.get('content') is None
            patch = call.kwargs['metadata_patch']
            assert patch == _mod.build_evidence_only_patch(
                entity_uuid=ENTITY_UUID,
                grounds=_mod.GROUNDS,
                migrated_at=patch[_mod.EVIDENCE_ONLY_MIGRATED_AT_KEY],
            )

    @pytest.mark.asyncio
    async def test_a_second_apply_run_is_a_no_op(self, ledger) -> None:
        """Idempotence end-to-end: the second run sees its own row and its own
        stamps, and does nothing."""
        first = _memory_service(ledger)
        await _mod.run_backfill(first, apply=True)

        already_stamped = _scroll_with(*EXPECTED_STAMP_TARGETS)
        second = _memory_service(ledger, scroll=already_stamped)
        report = await _mod.run_backfill(second, apply=True)

        assert len(await _rows(ledger)) == 1
        second.update_memory.assert_not_awaited()
        assert report['ledger'] == 'already_migrated'
        assert report['records'] == []


# ---------------------------------------------------------------------------
# Failure ordering — no original is ever marked superseded by a missing row
# ---------------------------------------------------------------------------


class TestLedgerWriteFailsBeforeAnyStamp:
    """β raises rather than returning a status, and the raise must land BEFORE
    the stamp loop begins."""

    @pytest.mark.asyncio
    async def test_an_unwired_ledger_aborts_with_no_stamp(self) -> None:
        service = _memory_service(None)
        with pytest.raises(LedgerUnavailable):
            await _mod.run_backfill(service, apply=True)
        service.update_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_failed_edge_sample_aborts_with_no_stamp(self, ledger) -> None:
        """β lets a sampling failure propagate rather than persist a bogus count
        that would corrupt ζ's growth sweep — so no row exists to be cited."""
        service = _memory_service(ledger)
        service.graphiti.get_valid_edges_for_node = AsyncMock(
            side_effect=RuntimeError('graphiti unreachable')
        )
        with pytest.raises(RuntimeError, match='graphiti unreachable'):
            await _mod.run_backfill(service, apply=True)
        assert await _rows(ledger) == []
        service.update_memory.assert_not_awaited()


class TestStampFailureNeverRollsBackTheRow:
    """The row is authoritative; the stamps are advisory. A stamp that does not
    land is reported, not repaired by destroying the row."""

    @staticmethod
    def _refusing_service(ledger_obj, failing_id: str):
        """``update_memory`` REFUSES *failing_id* by RETURNING an error envelope
        — which is how it reports a not-found, rather than raising."""
        service = _memory_service(ledger_obj)

        async def _update(**kwargs):
            if kwargs['memory_id'] == failing_id:
                return {
                    'error': 'memory not found',
                    'error_type': 'MemoryNotFound',
                }
            return {'status': 'updated', 'store': 'mem0'}

        service.update_memory = AsyncMock(side_effect=_update)
        return service

    @pytest.mark.asyncio
    async def test_the_row_survives_a_refused_stamp(self, ledger) -> None:
        service = self._refusing_service(ledger, CORRECTION_ID)
        report = await _mod.run_backfill(service, apply=True)
        assert report['ledger'] == 'written'
        assert len(await _rows(ledger)) == 1

    @pytest.mark.asyncio
    async def test_the_refusal_is_reported_with_its_error_type(self, ledger) -> None:
        report = await _mod.run_backfill(
            self._refusing_service(ledger, CORRECTION_ID), apply=True
        )
        outcomes = {row['memory_id']: row for row in report['records']}
        assert outcomes[CORRECTION_ID]['outcome'] == 'stamp_error'
        assert outcomes[CORRECTION_ID]['error_type'] == 'MemoryNotFound'

    @pytest.mark.asyncio
    async def test_the_other_stamp_still_lands(self, ledger) -> None:
        """Per-record isolation: one refusal must not abandon the rest."""
        report = await _mod.run_backfill(
            self._refusing_service(ledger, CORRECTION_ID), apply=True
        )
        outcomes = {row['memory_id']: row['outcome'] for row in report['records']}
        assert outcomes[SOURCE_ID] == 'stamped'

    @pytest.mark.asyncio
    async def test_a_raising_update_is_also_captured_per_record(self, ledger) -> None:
        """``update_memory`` has TWO failure shapes; a vocabulary rejection
        RAISES where a not-found returns an envelope."""
        service = _memory_service(ledger)
        service.update_memory = AsyncMock(side_effect=ValueError('nope'))
        report = await _mod.run_backfill(service, apply=True)
        assert len(await _rows(ledger)) == 1
        assert {row['outcome'] for row in report['records']} == {'stamp_error'}
        assert {row['error_type'] for row in report['records']} == {'ValueError'}

    @pytest.mark.asyncio
    async def test_a_partial_outcome_exits_non_zero(self, ledger) -> None:
        """An operator must not read a half-stamped corpus as a clean run."""
        report = await _mod.run_backfill(
            self._refusing_service(ledger, CORRECTION_ID), apply=True
        )
        assert _mod.resolve_exit_code(report) != 0

    @pytest.mark.asyncio
    async def test_a_clean_run_exits_zero(self, ledger) -> None:
        report = await _mod.run_backfill(_memory_service(ledger), apply=True)
        assert _mod.resolve_exit_code(report) == 0

    @pytest.mark.asyncio
    async def test_a_re_run_completes_the_stamp_that_failed(self, ledger) -> None:
        """Both legs are independently idempotent, so the repair is a re-run —
        which is the whole reason the row is not rolled back."""
        await _mod.run_backfill(
            self._refusing_service(ledger, CORRECTION_ID), apply=True
        )
        retry = _memory_service(ledger, scroll=_scroll_with(SOURCE_ID))
        report = await _mod.run_backfill(retry, apply=True)
        assert report['ledger'] == 'already_migrated'
        assert report['records'] == [{'memory_id': CORRECTION_ID, 'outcome': 'stamped'}]
        assert len(await _rows(ledger)) == 1


# ---------------------------------------------------------------------------
# The live target — WHICH ledger an --apply would write, and whether it is read
# ---------------------------------------------------------------------------


def _config(data_dir: Path, *, enabled: bool = True) -> SimpleNamespace:
    """The two ``config.reconciliation`` leaves the target resolution reads."""
    return SimpleNamespace(
        reconciliation=SimpleNamespace(
            data_dir=str(data_dir), recon_ledger_enabled=enabled
        )
    )


class TestResolveLedgerDbPath:
    """The target is absolutized once, from the configured data_dir."""

    def test_names_the_ledger_file_inside_the_configured_data_dir(
        self, tmp_path: Path
    ) -> None:
        assert _mod.resolve_ledger_db_path(_config(tmp_path)) == (
            tmp_path / _mod.LEDGER_DB_FILENAME
        )

    def test_a_relative_data_dir_is_resolved_against_the_cwd(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The default ``./data/reconciliation`` names a DIFFERENT file from
        every directory — which is the whole hazard, so the resolution must be
        absolute and the report must be able to quote it."""
        monkeypatch.chdir(tmp_path)
        resolved = _mod.resolve_ledger_db_path(_config(Path('./data/reconciliation')))
        assert resolved.is_absolute()
        assert resolved == tmp_path / 'data' / 'reconciliation' / (
            _mod.LEDGER_DB_FILENAME
        )


def _initialized_ledger_db(db_path: Path) -> Path:
    """Build a REAL ledger at *db_path* the way a deployment's is — α's own DDL.

    A ``touch()`` will not stand in for this, and that is precisely what the
    verdict below exists to say: an empty file IS a valid empty SQLite
    database, so only the SCHEMA distinguishes a deployment's ledger from a
    stray file sitting at the resolved path.
    """
    async def _build() -> None:
        store = ReconLedgerStore(db_path)
        await store.initialize()
        await store.close()

    asyncio.run(_build())
    return db_path


class TestClassifyLedgerTarget:
    """One read-only probe of the target file, four named verdicts.

    Every leg below was MEASURED against sqlite 3.50.4 on 2026-09-13 before it
    was asserted. The two that look like implementation detail are the ones
    that decide the classifier's shape:

    * a MISSING path and a DIRECTORY both raise ``sqlite3.OperationalError``
      from the read-only open (``'unable to open database file'`` and
      ``'disk I/O error'`` respectively on this build). The exception TYPE
      cannot tell them apart and the message text is a build detail, so the
      classifier consults ``db_path.exists()`` FIRST and only then interprets
      an open failure;
    * an empty file opens CLEANLY and reports an empty ``sqlite_master``. So
      ``exists()`` alone — the shape the gate had before this pass — cannot
      tell a stray file from a real ledger.
    """

    def test_a_real_initialized_ledger_is_live(self, tmp_path: Path) -> None:
        db_path = _initialized_ledger_db(tmp_path / _mod.LEDGER_DB_FILENAME)
        assert _mod.classify_ledger_target(db_path) is _mod.LedgerTargetState.LIVE

    def test_an_absent_path_is_missing(self, tmp_path: Path) -> None:
        assert _mod.classify_ledger_target(
            tmp_path / 'nowhere' / _mod.LEDGER_DB_FILENAME
        ) is _mod.LedgerTargetState.MISSING

    def test_classifying_an_absent_path_creates_nothing(self, tmp_path: Path) -> None:
        """The probe opens ``mode=ro``, which cannot create the file. This is
        what lets the verdict be taken BEFORE the dry run decides whether to
        attach a real ledger at all — a classifier that created what it
        classified would arm the very gate it reports on."""
        missing = tmp_path / 'nowhere' / _mod.LEDGER_DB_FILENAME
        _mod.classify_ledger_target(missing)
        assert not missing.exists()
        assert not missing.parent.exists()

    def test_a_bare_touched_file_has_no_schema(self, tmp_path: Path) -> None:
        """The stray-file shape: something is there, but α's table is not, so
        no running server has ever opened it as its ledger."""
        empty = tmp_path / _mod.LEDGER_DB_FILENAME
        empty.touch()
        assert _mod.classify_ledger_target(empty) is _mod.LedgerTargetState.NO_SCHEMA

    def test_a_directory_at_the_target_is_undetermined(self, tmp_path: Path) -> None:
        """It EXISTS, so it is not MISSING; it cannot be read, so the probe
        cannot prove the target unread either."""
        a_dir = tmp_path / _mod.LEDGER_DB_FILENAME
        a_dir.mkdir()
        assert _mod.classify_ledger_target(a_dir) is (
            _mod.LedgerTargetState.UNDETERMINED
        )

    def test_a_non_sqlite_file_is_undetermined(self, tmp_path: Path) -> None:
        garbage = tmp_path / _mod.LEDGER_DB_FILENAME
        garbage.write_bytes(b'not a database, just bytes' * 16)
        assert _mod.classify_ledger_target(garbage) is (
            _mod.LedgerTargetState.UNDETERMINED
        )

    @pytest.mark.asyncio
    async def test_a_ledger_a_running_server_holds_open_is_still_live(
        self, tmp_path: Path
    ) -> None:
        """The case the real migration actually runs against, and the reason the
        probe is read-only rather than an open-for-write reachability test: the
        server has this db OPEN in WAL mode throughout. A probe that could not
        read a busy ledger would invent a brand-new way to block the one-shot
        run it exists to protect."""
        db_path = tmp_path / _mod.LEDGER_DB_FILENAME
        store = ReconLedgerStore(db_path)
        await store.initialize()
        try:
            assert _mod.classify_ledger_target(db_path) is (
                _mod.LedgerTargetState.LIVE
            )
        finally:
            await store.close()

    def test_the_verdicts_are_a_closed_named_vocabulary(self) -> None:
        """A verdict is carried into the JSON report, so it is a ``StrEnum``
        (the house shape, cf. ``models/reconciliation.py``) rather than a bare
        string: closed at the type level, and still JSON-serializable."""
        assert issubclass(_mod.LedgerTargetState, str)
        assert {member.value for member in _mod.LedgerTargetState} == {
            'live', 'missing', 'no_schema', 'undetermined',
        }
        assert json.dumps(_mod.LedgerTargetState.LIVE) == '"live"'


class TestAssertLedgerTargetLive:
    """Every way to write a valid row nothing reads is refused, loudly.

    The gate CONSUMES a verdict rather than taking one, so each leg below
    classifies the target shape it describes and hands the answer over — the
    same one probe, same two consumers, that ``main`` performs.
    """

    @staticmethod
    def _existing_db(tmp_path: Path) -> Path:
        """A REAL ledger, not a ``touch()``ed file: the accept path now turns
        on α's schema being there, which is the whole point of the verdict."""
        return _initialized_ledger_db(tmp_path / _mod.LEDGER_DB_FILENAME)

    @staticmethod
    def _assert_target(db_path: Path, *, enabled: bool = True) -> None:
        _mod.assert_ledger_target_live(
            state=_mod.classify_ledger_target(db_path),
            db_path=db_path,
            recon_ledger_enabled=enabled,
        )

    def test_a_seeded_ledger_is_accepted(self, tmp_path: Path) -> None:
        self._assert_target(self._existing_db(tmp_path))

    def test_an_unreadable_target_is_accepted_rather_than_refused(
        self, tmp_path: Path
    ) -> None:
        """Fail-OPEN on UNDETERMINED. The gate must refuse whenever it can
        PROVE the row would be unread; refusing on a probe that established
        nothing would invent a new way to block the legitimate one-shot run,
        with no remedy available inside the script."""
        garbage = tmp_path / _mod.LEDGER_DB_FILENAME
        garbage.write_bytes(b'not a database' * 16)
        self._assert_target(garbage)

    def test_a_disabled_ledger_is_refused_even_though_the_file_is_there(
        self, tmp_path: Path
    ) -> None:
        """``server/main.py`` wires ``set_recon_ledger`` only under the flag, so
        the row's only consumer is not attached at all."""
        with pytest.raises(_mod.LedgerTargetUnusable) as excinfo:
            self._assert_target(self._existing_db(tmp_path), enabled=False)
        assert 'recon_ledger_enabled' in str(excinfo.value)

    def test_a_missing_ledger_is_refused_and_the_message_names_the_path(
        self, tmp_path: Path
    ) -> None:
        """``initialize()`` would otherwise CREATE it — a silent no-effect
        success for a migration that is unlikely to be re-run."""
        missing = tmp_path / 'nowhere' / _mod.LEDGER_DB_FILENAME
        with pytest.raises(_mod.LedgerTargetUnusable) as excinfo:
            self._assert_target(missing)
        assert str(missing) in str(excinfo.value)
        assert not missing.exists()

    def test_a_stray_unseeded_file_is_refused_separately_from_a_missing_one(
        self, tmp_path: Path
    ) -> None:
        """Same outcome — ``initialize()`` seeds the schema and the row lands
        unread — but a DIFFERENT remedy: the path is occupied rather than
        wrong, so the two refusals must not read alike. An operator told to
        "re-run from the right directory" when they already are has nowhere to
        go; the file needs deleting."""
        stray = tmp_path / _mod.LEDGER_DB_FILENAME
        stray.touch()
        with pytest.raises(_mod.LedgerTargetUnusable) as excinfo:
            self._assert_target(stray)
        message = str(excinfo.value)
        assert str(stray) in message
        assert _mod.LEDGER_TABLE_NAME in message

        missing = tmp_path / 'nowhere' / _mod.LEDGER_DB_FILENAME
        with pytest.raises(_mod.LedgerTargetUnusable) as other:
            self._assert_target(missing)
        assert message != str(other.value)

    def test_a_disabled_ledger_is_reported_ahead_of_a_missing_file(
        self, tmp_path: Path
    ) -> None:
        """When both are wrong the flag is the actionable fact: the row would be
        unreadable wherever it landed, so the path is not the interesting one."""
        with pytest.raises(_mod.LedgerTargetUnusable) as excinfo:
            self._assert_target(
                tmp_path / _mod.LEDGER_DB_FILENAME, enabled=False
            )
        assert 'recon_ledger_enabled' in str(excinfo.value)

    def test_the_refusal_is_module_typed_and_fail_closed(self) -> None:
        """A ``RuntimeError`` like ``StoreMutationUnavailable``, and for the same
        reason — but its own type, so a caller can tell the two refusals apart."""
        assert issubclass(_mod.LedgerTargetUnusable, RuntimeError)
        assert not issubclass(_mod.LedgerTargetUnusable, StoreMutationUnavailable)


# ---------------------------------------------------------------------------
# main() — the two run-wide refusals, the artifact, and the graded exit
# ---------------------------------------------------------------------------


class _LiveHarness:
    """Everything ``main()``'s live leg constructs, faked, with a call TRACE.

    The trace is the point. Both run-wide refusals are claims about ORDER —
    they must fire before any store is touched — and order is exactly what a
    return-value assertion cannot see. ``StoreMutationUnavailable`` and
    :class:`LedgerTargetUnusable` are both ``RuntimeError`` subclasses, so
    either probe moved down into ``run_backfill`` would be swallowed by
    ``_stamp_one``'s per-record ``except Exception`` and re-reported as N
    ``stamp_error`` rows — with every other test in this module still green.

    The recon ledger is the REAL ``ReconLedgerStore`` on a tmp db, so
    :meth:`rows` reads back what the run actually persisted rather than what
    it reported.
    """

    def __init__(
        self,
        tmp_path: Path,
        *,
        ledger_enabled: bool = True,
        create_db: bool = True,
        update_response: dict | None = None,
    ) -> None:
        self.trace: list[str] = []
        self.operations: list[str] = []
        self.services: list[AsyncMock] = []
        self.stamps: list[dict] = []
        self.ledger_enabled = ledger_enabled
        self.update_response = update_response or {'status': 'updated'}
        self.data_dir = tmp_path / 'data' / 'reconciliation'
        self.data_dir.mkdir(parents=True)
        self.db_path = self.data_dir / _mod.LEDGER_DB_FILENAME
        if create_db:
            # A REAL ledger, carrying α's schema — the shape an operator aiming
            # at a running deployment actually has. A bare ``touch()`` is the
            # OTHER shape (a stray file), and the gate now tells them apart.
            _initialized_ledger_db(self.db_path)

    def _make_config(self) -> SimpleNamespace:
        self.trace.append('config')
        return _config(self.data_dir, enabled=self.ledger_enabled)

    def _reader(self, value, label: str):
        async def _read(*_args, **_kwargs):
            self.trace.append(label)
            return value
        return _read

    async def _stamp(self, **kwargs) -> dict:
        self.trace.append('update_memory')
        self.stamps.append(kwargs)
        return dict(self.update_response)

    def _make_service(self, _config_obj) -> AsyncMock:
        self.trace.append('service_constructed')
        service = _memory_service(None)
        service.set_recon_ledger = lambda ledger: setattr(
            service, 'recon_ledger', ledger
        )
        service.get_memory_by_id = AsyncMock(
            side_effect=self._reader(SOURCE_RECORD, 'source_fetch')
        )
        service.get_memories_by_metadata = AsyncMock(
            side_effect=self._reader(list(LIVE_ENTITY_SCROLL), 'scroll')
        )
        service.update_memory = AsyncMock(side_effect=self._stamp)
        self.services.append(service)
        return service

    def run(self, monkeypatch, argv: list[str], *, probe_error=None) -> int:
        """Install the fakes and drive the real ``main(argv)``."""
        def _probe(*, operation: str) -> None:
            self.trace.append('probe')
            self.operations.append(operation)
            if probe_error is not None:
                raise probe_error

        monkeypatch.setattr(_mod, 'assert_store_mutation_allowed', _probe)
        monkeypatch.setattr(
            'fused_memory.config.schema.FusedMemoryConfig', self._make_config
        )
        monkeypatch.setattr(
            'fused_memory.services.memory_service.MemoryService', self._make_service
        )
        return _mod.main(argv)

    def report(self, capsys) -> dict:
        """The JSON artifact ``main`` printed."""
        return json.loads(capsys.readouterr().out.split('\nDRY RUN')[0])

    def rows(self) -> list:
        """What the run actually left in the ledger, read back independently."""
        async def _read():
            store = ReconLedgerStore(self.db_path)
            await store.initialize()
            try:
                return await store.list_entity_standing_decisions(_mod.PROJECT_ID)
            finally:
                await store.close()
        return asyncio.run(_read())


class TestMainStoreMutationPreflight:
    """One probe per run, before the scan and before the first mutation."""

    def test_apply_probes_exactly_once(self, tmp_path, monkeypatch, capsys) -> None:
        harness = _LiveHarness(tmp_path)
        assert harness.run(monkeypatch, ['--apply']) == 0
        capsys.readouterr()
        assert harness.trace.count('probe') == 1

    def test_nothing_is_constructed_or_read_before_the_probe(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        harness = _LiveHarness(tmp_path)
        harness.run(monkeypatch, ['--apply'])
        capsys.readouterr()
        assert harness.trace[0] == 'probe'
        # ...and the run really did go on to touch the store, so the leg above
        # is not passing on a trivially short trace.
        assert {'service_constructed', 'source_fetch', 'scroll'} <= set(harness.trace)

    def test_a_refused_probe_propagates_having_touched_nothing(
        self, tmp_path, monkeypatch
    ) -> None:
        harness = _LiveHarness(tmp_path)
        with pytest.raises(StoreMutationUnavailable):
            harness.run(
                monkeypatch,
                ['--apply'],
                probe_error=StoreMutationUnavailable('denied'),
            )
        assert harness.trace == ['probe']
        assert harness.services == []
        assert harness.stamps == []
        assert harness.rows() == []

    def test_the_probed_operation_names_this_script_and_the_flag(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """An operator reading the refusal must be able to tell WHICH operation
        was refused — so the string is read from the script, not invented."""
        harness = _LiveHarness(tmp_path)
        harness.run(monkeypatch, ['--apply'])
        capsys.readouterr()
        (operation,) = harness.operations
        assert SCRIPT_PATH.stem in operation
        assert '--apply' in operation

    def test_a_dry_run_never_probes(self, tmp_path, monkeypatch, capsys) -> None:
        """The probe gates MUTATION. A rehearsal writes nothing, so gating it
        would only deny an operator the report they need to plan the real run."""
        harness = _LiveHarness(tmp_path)
        assert harness.run(monkeypatch, []) == 0
        capsys.readouterr()
        assert 'probe' not in harness.trace
        assert harness.operations == []
        assert harness.stamps == []
        assert harness.rows() == []


class TestMainRefusesADeadLedgerTarget:
    """``--apply`` stops before the store when the row would land unread."""

    def test_a_missing_ledger_file_refuses_instead_of_creating_one(
        self, tmp_path, monkeypatch
    ) -> None:
        harness = _LiveHarness(tmp_path, create_db=False)
        with pytest.raises(_mod.LedgerTargetUnusable) as excinfo:
            harness.run(monkeypatch, ['--apply'])
        assert str(harness.db_path) in str(excinfo.value)
        assert not harness.db_path.exists()
        assert harness.trace == ['probe', 'config']
        assert harness.services == []

    def test_a_switched_off_ledger_refuses(self, tmp_path, monkeypatch) -> None:
        harness = _LiveHarness(tmp_path, ledger_enabled=False)
        with pytest.raises(_mod.LedgerTargetUnusable):
            harness.run(monkeypatch, ['--apply'])
        assert harness.trace == ['probe', 'config']
        assert harness.rows() == []

    def test_the_refusal_follows_the_mutation_probe(
        self, tmp_path, monkeypatch
    ) -> None:
        """Capability first, then destination: a process that may not mutate at
        all should hear that, not a complaint about which file it aimed at."""
        harness = _LiveHarness(tmp_path, create_db=False)
        with pytest.raises(StoreMutationUnavailable):
            harness.run(
                monkeypatch,
                ['--apply'],
                probe_error=StoreMutationUnavailable('denied'),
            )
        assert harness.trace == ['probe']

    def test_a_dry_run_is_never_gated_on_the_target(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """A rehearsal against a dead target is exactly how an operator DISCOVERS
        it, so the report must still be produced."""
        harness = _LiveHarness(tmp_path, ledger_enabled=False, create_db=False)
        assert harness.run(monkeypatch, []) == 0
        assert harness.report(capsys)['ledger'] == 'would_write'


class TestMainReportNamesItsTarget:
    """The artifact says which ledger it wrote and whether anything reads it."""

    def test_the_reported_path_is_absolute_and_is_the_file_that_was_written(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        harness = _LiveHarness(tmp_path)
        harness.run(monkeypatch, ['--apply'])
        reported = Path(harness.report(capsys)['ledger_db_path'])
        assert reported.is_absolute()
        assert reported == harness.db_path.resolve()
        assert len(harness.rows()) == 1

    def test_a_dry_run_names_the_target_too(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        harness = _LiveHarness(tmp_path)
        harness.run(monkeypatch, [])
        assert harness.report(capsys)['ledger_db_path'] == str(
            harness.db_path.resolve()
        )

    def test_the_consumer_flag_is_echoed(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """So a report from a ``recon_ledger_enabled=False`` host cannot be read
        as a landed migration."""
        harness = _LiveHarness(tmp_path, ledger_enabled=False)
        harness.run(monkeypatch, [])
        assert harness.report(capsys)['recon_ledger_enabled'] is False


class TestMainCliContract:
    """``--json-out``, and the exit code an automated caller reads."""

    def test_json_out_is_written_in_addition_to_stdout(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        harness = _LiveHarness(tmp_path)
        out = tmp_path / 'report.json'
        assert harness.run(monkeypatch, ['--json-out', str(out)]) == 0
        written = out.read_text()
        assert capsys.readouterr().out.startswith(written)
        assert json.loads(written)['apply'] is False

    def test_a_clean_apply_exits_zero(self, tmp_path, monkeypatch, capsys) -> None:
        assert _LiveHarness(tmp_path).run(monkeypatch, ['--apply']) == 0
        capsys.readouterr()

    def test_a_refused_stamp_exits_non_zero(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """``update_memory`` REPORTS a refusal rather than raising it, and a
        half-stamped corpus must not read as a completed migration."""
        harness = _LiveHarness(
            tmp_path, update_response={'error_type': 'MemoryNotFound', 'error': 'gone'}
        )
        assert harness.run(monkeypatch, ['--apply']) == 1
        assert {row['outcome'] for row in harness.report(capsys)['records']} == {
            'stamp_error'
        }
