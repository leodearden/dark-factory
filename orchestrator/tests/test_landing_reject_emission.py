"""Tests for orchestrator.landing_reject_emission (task 4496).

The two ``validate_landing_evidence`` call sites that file no escalation — the
already-landed gate's ancestry arm and the reconcile sweep's MARK_DONE arm —
record each reject as a ``landing_evidence_rejected`` runs.db row, gated on a
new-or-changed signature per (site, task_id).  Exercised through the public
surface only, against a real EventStore so the payload's JSON-serialisability
is proven end to end.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from _landing_evidence_helpers import landing_reject_rows

from orchestrator.event_store import EventStore
from orchestrator.landing_reject_emission import (
    LandingRejectRecorder,
    LandingRejectSite,
    landing_reject_payload,
)
from orchestrator.merge_lane.landing_evidence import LandingMethod, LandingReason, LandingVerdict

_CITATION = 'a' * 40
_TIP = 'b' * 40


def _probe(**overrides: Any) -> dict[str, Any]:
    probe: dict[str, Any] = {
        'task_id': '42',
        'branch': 'task/42',
        'branch_tip_sha': _TIP,
        'citation': _CITATION,
        'method': LandingMethod.merge_marker,
        'effect_check_sha': _CITATION,
        'delivered_checks_state': 'none_declared',
        'reason': LandingReason.effect_absent,
        'diverged_paths': ['src/x.py'],
        'aggregate_survival': 0.4,
        'added_lines_total': 50,
    }
    probe.update(overrides)
    return probe


def _reject(reason: LandingReason = LandingReason.effect_absent, **overrides: Any) -> LandingVerdict:
    return LandingVerdict(
        accepted=False, evidence_sha=None, reason=reason,
        probe=_probe(reason=reason, **overrides),
    )


def _accept() -> LandingVerdict:
    return LandingVerdict(
        accepted=True, evidence_sha=_CITATION, reason=LandingReason.ok,
        probe=_probe(reason=LandingReason.ok),
    )


@pytest.fixture
def store(tmp_path: Path) -> EventStore:
    return EventStore(tmp_path / 'runs.db', 'run-test')


def _observe(
    recorder: LandingRejectRecorder,
    verdict: LandingVerdict,
    store: EventStore | None,
    *,
    site: LandingRejectSite = LandingRejectSite.reconcile_sweep,
    task_id: str = '42',
    task_status: str = 'in-progress',
) -> bool:
    return recorder.observe(
        verdict, event_store=store, site=site, task_id=task_id, task_status=task_status,
    )


class TestLandingRejectPayload:
    def test_payload_is_the_probe_plus_the_site_keys(self) -> None:
        verdict = _reject()

        payload = landing_reject_payload(
            verdict, site=LandingRejectSite.reconcile_sweep, task_status='in-progress',
        )

        for key, value in verdict.probe.items():
            assert payload[key] == value, key
        assert payload['site'] == 'reconcile_sweep'
        assert payload['task_status'] == 'in-progress'
        json.dumps(payload)

    def test_site_keys_win_over_a_colliding_probe_key(self) -> None:
        verdict = _reject(site='from-probe', task_status='from-probe')

        payload = landing_reject_payload(
            verdict, site=LandingRejectSite.already_landed_gate_ancestry,
            task_status='pending',
        )

        assert payload['site'] == 'already_landed_gate_ancestry'
        assert payload['task_status'] == 'pending'


class TestLandingRejectRecorder:
    def test_an_unserialisable_probe_value_still_yields_a_row(
        self, store: EventStore,
    ) -> None:
        assert _observe(LandingRejectRecorder(), _reject(opaque=object()), store) is True

        [row] = landing_reject_rows(store)
        assert isinstance(row['data']['opaque'], str)

    def test_a_first_reject_writes_one_row_and_an_identical_repeat_none(
        self, store: EventStore,
    ) -> None:
        recorder = LandingRejectRecorder()

        assert _observe(recorder, _reject(), store) is True
        assert _observe(recorder, _reject(), store) is False

        [row] = landing_reject_rows(store)
        assert row['task_id'] == '42'
        assert row['data']['site'] == 'reconcile_sweep'
        assert row['data']['reason'] == 'effect_absent'

    @pytest.mark.parametrize('changed', [
        {'citation': 'c' * 40},
        {'effect_check_sha': 'd' * 40},
        {'reason': LandingReason.no_citation},
    ])
    def test_a_changed_signature_writes_a_second_row(
        self, store: EventStore, changed: dict[str, Any],
    ) -> None:
        recorder = LandingRejectRecorder()
        _observe(recorder, _reject(), store)

        assert _observe(recorder, _reject(**changed), store) is True

        assert len(landing_reject_rows(store)) == 2

    def test_another_site_or_task_is_an_independent_key(
        self, store: EventStore,
    ) -> None:
        recorder = LandingRejectRecorder()
        _observe(recorder, _reject(), store)

        assert _observe(
            recorder, _reject(), store, site=LandingRejectSite.already_landed_gate_ancestry,
        ) is True
        assert _observe(recorder, _reject(), store, task_id='43') is True

        assert [(r['task_id'], r['data']['site']) for r in landing_reject_rows(store)] == [
            ('42', 'reconcile_sweep'),
            ('42', 'already_landed_gate_ancestry'),
            ('43', 'reconcile_sweep'),
        ]

    def test_an_accept_writes_nothing_and_rearms_the_next_reject(
        self, store: EventStore,
    ) -> None:
        recorder = LandingRejectRecorder()
        _observe(recorder, _reject(), store)

        assert _observe(recorder, _accept(), store) is False
        assert _observe(recorder, _reject(), store) is True

        assert len(landing_reject_rows(store)) == 2

    def test_release_rearms_every_site_for_that_task_only(
        self, store: EventStore,
    ) -> None:
        recorder = LandingRejectRecorder()
        for site in LandingRejectSite:
            _observe(recorder, _reject(), store, site=site)
        _observe(recorder, _reject(), store, task_id='43')

        recorder.release('42')

        assert [_observe(recorder, _reject(), store, site=site) for site in LandingRejectSite] == [
            True, True,
        ]
        assert _observe(recorder, _reject(), store, task_id='43') is False

    def test_a_reject_seen_with_no_event_store_is_announced_once_one_is_bound(
        self, store: EventStore,
    ) -> None:
        recorder = LandingRejectRecorder()

        assert _observe(recorder, _reject(), None) is False
        assert _observe(recorder, _reject(), store) is True

    def test_a_raising_event_store_leaves_the_reject_unannounced(
        self, store: EventStore,
    ) -> None:
        recorder = LandingRejectRecorder()
        raising = MagicMock()
        raising.emit.side_effect = RuntimeError('store exploded')

        assert _observe(recorder, _reject(), raising) is False
        assert _observe(recorder, _reject(), store) is True

        assert len(landing_reject_rows(store)) == 1

    def test_a_payload_that_cannot_be_built_leaves_the_reject_unannounced(
        self, store: EventStore,
    ) -> None:
        recorder = LandingRejectRecorder()

        assert _observe(recorder, _reject(opaque={('non', 'str'): 'key'}), store) is False
        assert _observe(recorder, _reject(), store) is True

        assert len(landing_reject_rows(store)) == 1
