"""The INV-4 mint-storm alarm (task 4932): `emit_entity_mint_storm_escalation`.

The FIRE half of the storm escape whose counter half is
`MemoryService._entity_mint_storm_counters`, a per-`agent_id`
`shared.storm_counter.StormCounter` recording one event per MINT.

WHY THIS FILE EXISTS AT ALL. The service-side suite
(`tests/test_entity_mint_service.py`) monkeypatches this symbol out entirely, on
purpose — the `escalation` package is a DEFENSIVE OPTIONAL import, so a service
test that filed for real would be environment-coupled. That leaves the module's
own body untested, and every one of its load-bearing claims is a documented
design decision with no other pin: the `HAS_ESCALATION=False` degradation, the
queue-construction failure, the fingerprint over `(category, finding_category,
agent:agent_id)` ONLY, the UNBOUNDED dedupe fold, `level=1`, and the
submit-failure path. The module swallows every exception and returns None, so a
regression in any of them loses the alarm SILENTLY — which is the exact failure
shape the module docstring argues against.

Modelled on `tests/middleware/test_mem0_update_storm_escalator.py` (the sibling
alarm) for the queue-reading helper and the fold/attribution legs, and on
`tests/test_referent_repair_storm_escalator.py` for the module-FUNCTION shape:
`project_root` arrives as an explicit argument, so there is no queue cache to
own and no `set_known_projects` lifecycle to assert.
"""

from __future__ import annotations

import json
import logging

import pytest

from fused_memory.middleware import entity_mint_storm_escalator as emse_mod
from fused_memory.middleware.entity_mint_storm_escalator import (
    emit_entity_mint_storm_escalation,
)

pytestmark = pytest.mark.skipif(
    not emse_mod.HAS_ESCALATION,
    reason='escalation package unavailable (minimal env); the HAS_ESCALATION '
           'no-op arm is covered separately below',
)

_PROJECT = 'dark_factory'


def _filed(root) -> list[dict]:
    """The parsed escalation payloads under ``{root}/data/escalations``."""
    queue_dir = root / 'data' / 'escalations'
    if not queue_dir.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(queue_dir.glob('esc-*.json'))]


def _emit(root, *, agent_id='curator-repair', count=10, threshold=10,
          window_seconds=3600.0, project_id=_PROJECT):
    return emit_entity_mint_storm_escalation(
        str(root),
        project_id=project_id,
        agent_id=agent_id,
        count=count,
        threshold=threshold,
        window_seconds=window_seconds,
    )


class TestTheFiledEscalation:
    def test_files_exactly_one_escalation_and_returns_its_id(self, tmp_path):
        esc_id = _emit(tmp_path)

        assert esc_id is not None
        payloads = _filed(tmp_path)
        assert len(payloads) == 1, f'expected one escalation file, got {payloads}'
        assert payloads[0]['id'] == esc_id

    def test_lands_in_the_affected_projects_own_queue(self, tmp_path):
        """`project_root` is an explicit argument for exactly this reason.

        The caller (`MemoryService._record_entity_mint`) resolves it from
        `_known_projects` and NEVER defaults it to the server cwd, where no
        operator watches; this module simply honours whatever root it is given.
        """
        root_a = tmp_path / 'a'
        root_b = tmp_path / 'b'

        _emit(root_a)
        _emit(root_b, agent_id='curator-other')

        assert len(_filed(root_a)) == 1
        assert len(_filed(root_b)) == 1

    def test_carries_the_category_severity_role_and_anchor(self, tmp_path):
        _emit(tmp_path)
        payload = _filed(tmp_path)[0]

        assert payload['category'] == 'entity_mint_storm', (
            'its OWN category, not the mem0_update one — an operator paged for '
            'a mint storm must not be sent to read update_memory journal rows'
        )
        assert payload['severity'] == 'blocking'
        assert payload['agent_role'] == 'fused-memory/entity-mint-guard'
        assert payload['task_id'] == 'entity-mint-storm'

    def test_is_born_at_l1_like_every_sibling_fused_memory_escalator(self, tmp_path):
        """L0 routes to a task's steward; this anchor is never dispatched.

        Filed by a background server process under a synthetic anchor task id,
        so an L0 entry would have no consumer at all and would merely wait out
        `orphan_l0_timeout_secs` before being promoted to exactly where L1 puts
        it immediately.
        """
        _emit(tmp_path)
        assert _filed(tmp_path)[0]['level'] == 1

    def test_the_summary_names_the_agent_the_count_and_the_project(self, tmp_path):
        _emit(tmp_path, agent_id='recon-stage-1', count=37, threshold=10)
        summary = _filed(tmp_path)[0]['summary']

        assert 'recon-stage-1' in summary, summary
        assert '37' in summary, summary
        assert _PROJECT in summary, summary

    def test_the_detail_carries_every_operator_input(self, tmp_path):
        _emit(tmp_path, agent_id='recon-stage-1', count=37, threshold=10,
              window_seconds=1800.0)
        detail = _filed(tmp_path)[0]['detail']

        for expected in ('recon-stage-1', '37', '10', '1800', _PROJECT):
            assert expected in detail, (
                f'the operator must be able to judge the incident without '
                f'reading the server log; {expected!r} missing from {detail!r}'
            )

    def test_the_detail_names_the_knobs_that_actually_stop_minting(
        self, tmp_path,
    ):
        """An operator reading the alarm has to be told which knob stops the
        burst; naming both leaves is the executable half of that. The prose
        framing around them is deliberately left unpinned."""
        _emit(tmp_path)
        payload = _filed(tmp_path)[0]
        detail = payload['detail']

        assert 'entity_mint.enabled' in detail, detail
        assert 'entity_mint.storm_threshold' in detail, detail

    def test_the_detail_names_the_write_journal_as_the_evidence_trail(self, tmp_path):
        """The alarm's whole value is pointing somewhere; nothing else records
        which nodes a mint loop created."""
        _emit(tmp_path)
        detail = _filed(tmp_path)[0]['detail']

        assert 'ensure_entity_node' in detail, detail
        assert 'delete_entity' in detail, (
            'the reversal path is only clean while a minted node is edgeless, '
            f'so the alarm has to say so: {detail!r}'
        )


class TestDedupeFold:
    """A sustained storm pages ONCE; two storming agents page twice."""

    def test_the_fingerprint_is_over_category_finding_and_agent_only(self, tmp_path):
        """Deliberately NOT over count or window: both change on every breach,
        so including either would mint a fresh escalation per recount and
        defeat the very folding the fingerprint exists to provide."""
        from escalation.dedupe import compute_content_fingerprint

        _emit(tmp_path, count=10, threshold=10, window_seconds=3600.0)

        expected = compute_content_fingerprint(
            emse_mod._CATEGORY,
            emse_mod._FINDING_CATEGORY,
            affected_ids=['agent:curator-repair'],
        )
        assert _filed(tmp_path)[0]['dedupe_fingerprint'] == expected

        # And it is genuinely count/threshold/window independent.
        _emit(tmp_path, count=999, threshold=5, window_seconds=60.0)
        assert {p['dedupe_fingerprint'] for p in _filed(tmp_path)} == {expected}

    def test_a_sustained_storm_folds_into_one_pending_parent(self, tmp_path):
        first = _emit(tmp_path, count=10)
        second = _emit(tmp_path, count=41)
        third = _emit(tmp_path, count=63)

        assert first is not None
        assert second == first, 'a repeat breach must fold into the pending parent'
        assert third == first

        payloads = _filed(tmp_path)
        assert len(payloads) == 1, f'expected one surviving record, got {payloads}'
        assert payloads[0]['id'] == first
        assert payloads[0]['dedupe_count'] == 2
        assert len(payloads[0]['dedupe_children']) == 2

    def test_the_fold_window_is_unbounded(self, tmp_path):
        """`infra_dedupe_window_secs=float('inf')`, so age never un-folds it.

        Backdates the pending parent on disk by a decade — under any finite
        window `find_dedupe_parent`'s age filter would drop it and the second
        breach would page again. A mint storm can smoulder for days; each
        recount must still land on the same record.
        """
        first = _emit(tmp_path)
        queue_dir = tmp_path / 'data' / 'escalations'
        parent_path = next(iter(queue_dir.glob('esc-*.json')))
        payload = json.loads(parent_path.read_text())
        payload['timestamp'] = '2016-01-01T00:00:00+00:00'
        parent_path.write_text(json.dumps(payload))

        second = _emit(tmp_path, count=999)

        assert second == first, (
            'a decade-old pending parent must still absorb the recount'
        )
        assert len(_filed(tmp_path)) == 1

    def test_two_agents_produce_two_escalations(self, tmp_path):
        """Attribution is the point: the operator's first question is WHICH
        caller is looping, and one entry naming neither cannot answer it."""
        first = _emit(tmp_path, agent_id='recon-stage-1')
        second = _emit(tmp_path, agent_id='curator-repair')

        assert first is not None
        assert second is not None
        assert second != first
        assert len(_filed(tmp_path)) == 2

    def test_an_unattributed_burst_still_files(self, tmp_path):
        """`'<unattributed>'` is what the caller passes for a missing agent_id;
        it is a real fingerprint value, not a reason to drop the alarm."""
        esc_id = _emit(tmp_path, agent_id='<unattributed>')

        assert esc_id is not None
        assert '<unattributed>' in _filed(tmp_path)[0]['summary']


class TestNeverRaises:
    """Dispatched through `asyncio.to_thread` off the live MCP request path.

    The mints being complained about have ALREADY committed by the time this
    runs, so a raise here would turn a completed, landed mint into a tool
    exception because the COMPLAINT about it failed.
    """

    def test_a_queue_construction_failure_returns_none(self, tmp_path, monkeypatch):
        def _explode(*_a, **_kw):
            raise OSError('cannot create queue dir')

        monkeypatch.setattr(emse_mod, 'EscalationQueue', _explode)

        assert _emit(tmp_path) is None

    def test_an_unwritable_project_root_returns_none_and_logs(
        self, tmp_path, caplog,
    ):
        """The real-world shape of the branch above: a `project_root` that
        cannot hold a queue directory.

        A plain FILE where `data/` must be — constructing the queue mkdirs
        through it and raises `NotADirectoryError`. Deterministic without root,
        unlike a permissions trick.
        """
        blocked = tmp_path / 'blocked'
        blocked.mkdir()
        (blocked / 'data').write_text('not a directory')

        with caplog.at_level(logging.ERROR, logger=emse_mod.__name__):
            result = _emit(blocked)

        assert result is None
        assert caplog.records, 'a swallowed failure must still be visible'
        assert 'curator-repair' in caplog.text

    def test_a_submit_failure_returns_none_and_logs(self, tmp_path, monkeypatch, caplog):
        class _BrokenQueue:
            def __init__(self, *_a, **_kw):
                pass

            def get_pending(self):
                return []

            def make_id(self, task_id):
                return f'esc-{task_id}-1'

            def submit(self, _esc):
                raise OSError('read-only filesystem')

        monkeypatch.setattr(emse_mod, 'EscalationQueue', _BrokenQueue)

        with caplog.at_level(logging.ERROR, logger=emse_mod.__name__):
            result = _emit(tmp_path)

        assert result is None
        assert caplog.records, 'a swallowed failure must still be visible'
        assert not _filed(tmp_path)


def test_without_the_escalation_package_it_no_ops(tmp_path, monkeypatch, caplog):
    """The minimal-env path: WARNED, nothing filed, `None` returned.

    The optional `escalation` workspace package stays optional — minting must
    behave identically with or without it — but the degradation is LOUD, because
    a burst that goes unescalated is precisely the thing an operator needs told.
    """
    monkeypatch.setattr(emse_mod, 'HAS_ESCALATION', False)

    with caplog.at_level(logging.WARNING, logger=emse_mod.__name__):
        result = emit_entity_mint_storm_escalation(
            str(tmp_path),
            project_id=_PROJECT,
            agent_id='curator-repair',
            count=10,
            threshold=10,
            window_seconds=3600.0,
        )

    assert result is None
    assert not (tmp_path / 'data' / 'escalations').exists(), (
        'a no-op must not leave a queue directory behind'
    )
    text = caplog.text
    assert 'curator-repair' in text, text
    assert '10' in text, 'the WARN must carry the burst count'
