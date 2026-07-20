"""Tests for the ``metadata.merge_retry_pending`` registered sub-model (task 2795).

``merge_retry_pending`` is the durable record of a merge-phase resume
obligation: when a merge-phase escalation is resolved via ``resume`` the
orchestrator stamps ``{branch_head, base_sha, resolved_at}`` here so the
in-place merge retry survives an orchestrator restart (Reify 5166). It is
modelled as a *registered* metadata sub-model
(``register_metadata_submodel('merge_retry_pending', MergeRetryPending)``)
mirroring the ``milestone`` / ``routing`` precedent — so it lands in
``parse_metadata``'s ``known_fields`` (no ``unknown_key`` census warning),
is typed/validated at the fused-memory write boundary, and — unlike an
optional typed field — is absent from ``model_dump()`` when unset (no
``None``-noise on every task).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from shared.task_metadata import (
    MergeRetryPending,
    TaskMetadata,
    parse_metadata,
)

_WELL_FORMED_SLICE = {
    'branch_head': 'a' * 40,
    'base_sha': 'b' * 40,
    'resolved_at': '2026-07-20T05:00:00+00:00',
}


class TestMergeRetryPendingModel:
    """The MergeRetryPending sub-model in isolation."""

    def test_constructs_with_required_fields(self):
        m = MergeRetryPending(**_WELL_FORMED_SLICE)
        assert m.branch_head == 'a' * 40
        assert m.base_sha == 'b' * 40
        assert m.resolved_at == '2026-07-20T05:00:00+00:00'

    def test_missing_branch_head_raises(self):
        with pytest.raises(ValidationError):
            MergeRetryPending(base_sha='b' * 40, resolved_at='2026-07-20T05:00:00+00:00')

    def test_extra_fields_allowed_and_round_trip(self):
        # extra='allow' so a newer writer's field survives round-trip.
        m = MergeRetryPending(x_note='keep', **_WELL_FORMED_SLICE)
        assert m.model_dump()['x_note'] == 'keep'


class TestMergeRetryPendingParse:
    """parse_metadata behaviour for the registered merge_retry_pending slice."""

    def test_read_well_formed_slice_typed_and_no_unknown_key_warning(self):
        blob = {'merge_retry_pending': dict(_WELL_FORMED_SLICE)}
        model, warnings = parse_metadata(blob, direction='read')
        # (a) typed slice attached (registered sub-model → __pydantic_extra__).
        assert isinstance(model.merge_retry_pending, MergeRetryPending)
        assert model.merge_retry_pending.branch_head == 'a' * 40
        # (a) NO unknown_key warning for this key (it is in known_fields).
        assert not any(
            w.code == 'unknown_key' and w.field == 'merge_retry_pending' for w in warnings
        )
        # A well-formed slice warns not at all.
        assert warnings == []

    def test_model_dump_round_trips_the_three_fields_as_plain_dict(self):
        # (b) I1 round-trip: model_dump re-emits the slice as a plain dict
        # carrying all three fields, not a lingering BaseModel instance.
        blob = {'merge_retry_pending': dict(_WELL_FORMED_SLICE)}
        model, warnings = parse_metadata(blob, direction='read')
        assert warnings == []
        dumped = model.model_dump()['merge_retry_pending']
        assert dumped == _WELL_FORMED_SLICE
        assert not isinstance(dumped, BaseModel)

    def test_absent_slice_not_emitted_in_model_dump(self):
        # (c) no None-noise: an unset registered sub-model is simply absent
        # from model_dump(), unlike a `field | None = None` typed field.
        assert 'merge_retry_pending' not in TaskMetadata().model_dump()
        model, warnings = parse_metadata({'task_kind': 'normal'}, direction='read')
        assert warnings == []
        assert 'merge_retry_pending' not in model.model_dump()

    # (d) A malformed value — a bare string (not a mapping) or a dict missing
    # the required branch_head — warns invalid_submodel under read and raises
    # under write+enforce.
    _MALFORMED_CASES = [
        pytest.param('not-a-dict', id='bare_string'),
        pytest.param(
            {'base_sha': 'b' * 40, 'resolved_at': '2026-07-20T05:00:00+00:00'},
            id='dict_missing_branch_head',
        ),
    ]

    @pytest.mark.parametrize('bad_value', _MALFORMED_CASES)
    def test_malformed_slice_read_warns_invalid_submodel(self, bad_value):
        model, warnings = parse_metadata(
            {'merge_retry_pending': bad_value}, direction='read'
        )
        assert len(warnings) == 1
        assert warnings[0].code == 'invalid_submodel'
        assert warnings[0].field == 'merge_retry_pending'
        # I1: the raw offending value is retained verbatim.
        assert model.model_dump()['merge_retry_pending'] == bad_value

    @pytest.mark.parametrize('bad_value', _MALFORMED_CASES)
    def test_malformed_slice_write_enforce_raises(self, bad_value):
        with pytest.raises((ValidationError, TypeError)):
            parse_metadata(
                {'merge_retry_pending': bad_value}, direction='write', enforce=True
            )
