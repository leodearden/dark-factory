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

import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta
from pathlib import Path
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
            _mod.build_evidence_refs(EXPECTED_STAMP_TARGETS, ENTITY_UUID)
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
