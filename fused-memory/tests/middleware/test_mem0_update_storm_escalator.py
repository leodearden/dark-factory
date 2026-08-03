"""Tests for Mem0UpdateStormEscalator — the INV-4 escape hatch for update_memory.

Task 3088. In-place content amendment is a silent-rewrite primitive: an agent
stuck in a rewrite loop leaves no new records behind, so nothing downstream can
tell it is happening. The storm counter (``server/storm_counter.StormCounter``)
detects the burst; this escalator is the channel that makes it reach an operator.

Modelled on ``tests/test_scope_violation_escalator.py`` — same defensive-import
skip marker, same per-``project_root`` queue-cache assertion, same
"queue I/O failure never propagates" contract.
"""

from __future__ import annotations

import json
import logging

import pytest

from fused_memory.middleware import mem0_update_storm_escalator as storm_mod
from fused_memory.middleware.mem0_update_storm_escalator import Mem0UpdateStormEscalator


def _read_escalations(tmp_path):
    """Return the parsed escalation payloads under ``{tmp_path}/data/escalations``."""
    queue_dir = tmp_path / 'data' / 'escalations'
    if not queue_dir.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(queue_dir.glob('esc-*.json'))]


@pytest.mark.skipif(
    not storm_mod.HAS_ESCALATION,
    reason='escalation package not installed in this environment',
)
class TestStormEscalationEnabled:
    def test_constructs_zero_arg(self):
        """Constructed unconditionally inside MemoryService.__init__ — no deps to hand it.

        The map is delivered later by a setter, exactly because MemoryService is
        built before ``build_known_projects_map`` runs at server startup.
        """
        esc = Mem0UpdateStormEscalator()
        assert esc._queues == {}

    def test_writes_blocking_escalation_under_project_root(self, tmp_path):
        esc = Mem0UpdateStormEscalator()
        esc.set_known_projects({'dark_factory': str(tmp_path)})

        esc_id = esc.report_storm(
            project_id='dark_factory',
            agent_id='recon-stage-1',
            count=20,
            threshold=20,
            window_seconds=3600.0,
        )

        assert esc_id is not None
        payloads = _read_escalations(tmp_path)
        assert len(payloads) == 1, f'expected one escalation file, found: {payloads}'
        payload = payloads[0]
        assert payload['id'] == esc_id
        assert payload['category'] == 'mem0_in_place_update_storm'
        assert payload['severity'] == 'blocking'
        # The operator needs the offending agent, the burst size and the window
        # to judge the incident without reading the server log.
        assert 'recon-stage-1' in payload['detail']
        assert '20' in payload['detail']
        assert '3600' in payload['detail']
        assert 'recon-stage-1' in payload['summary']

    def test_dedupe_fingerprint_is_over_category_and_agent_only(self, tmp_path):
        """The fingerprint must not vary with count/window — else folding is defeated.

        A fingerprint that folded in the burst size would mint a fresh
        escalation on every recount, which is precisely the paging flood
        ``submit_or_dedupe`` exists to prevent.
        """
        from escalation.dedupe import compute_content_fingerprint

        esc = Mem0UpdateStormEscalator()
        esc.set_known_projects({'dark_factory': str(tmp_path)})
        esc.report_storm(
            project_id='dark_factory',
            agent_id='recon-stage-1',
            count=20,
            threshold=20,
            window_seconds=3600.0,
        )

        payload = _read_escalations(tmp_path)[0]
        expected = compute_content_fingerprint(
            'mem0_in_place_update_storm',
            storm_mod._FINDING_CATEGORY,
            affected_ids=['agent:recon-stage-1'],
        )
        assert payload['dedupe_fingerprint'] == expected

        # And it is genuinely count/window independent.
        other = Mem0UpdateStormEscalator()
        other.set_known_projects({'dark_factory': str(tmp_path)})
        other.report_storm(
            project_id='dark_factory',
            agent_id='recon-stage-1',
            count=999,
            threshold=5,
            window_seconds=60.0,
        )
        payloads = _read_escalations(tmp_path)
        fingerprints = {p['dedupe_fingerprint'] for p in payloads}
        assert fingerprints == {expected}, (
            'variable burst detail must not change the fingerprint'
        )

    def test_sustained_storm_from_one_agent_folds_into_one_parent(self, tmp_path):
        """A sustained storm pages ONCE, not once per breach."""
        esc = Mem0UpdateStormEscalator()
        esc.set_known_projects({'dark_factory': str(tmp_path)})

        first = esc.report_storm(
            project_id='dark_factory',
            agent_id='recon-stage-1',
            count=20,
            threshold=20,
            window_seconds=3600.0,
        )
        second = esc.report_storm(
            project_id='dark_factory',
            agent_id='recon-stage-1',
            count=41,
            threshold=20,
            window_seconds=3600.0,
        )
        third = esc.report_storm(
            project_id='dark_factory',
            agent_id='recon-stage-1',
            count=63,
            threshold=20,
            window_seconds=3600.0,
        )

        assert first is not None
        assert second == first, 'a repeat storm must fold into the pending parent'
        assert third == first

        payloads = _read_escalations(tmp_path)
        assert len(payloads) == 1, f'expected one surviving escalation, found: {payloads}'
        assert payloads[0]['id'] == first
        assert payloads[0]['dedupe_count'] == 2
        assert len(payloads[0]['dedupe_children']) == 2

    def test_distinct_agents_do_not_fold_together(self, tmp_path):
        """Attribution is the point: two runaway agents are two incidents."""
        esc = Mem0UpdateStormEscalator()
        esc.set_known_projects({'dark_factory': str(tmp_path)})

        first = esc.report_storm(
            project_id='dark_factory',
            agent_id='recon-stage-1',
            count=20,
            threshold=20,
            window_seconds=3600.0,
        )
        second = esc.report_storm(
            project_id='dark_factory',
            agent_id='recon-stage-3',
            count=20,
            threshold=20,
            window_seconds=3600.0,
        )

        assert first is not None
        assert second is not None
        assert second != first
        assert len(_read_escalations(tmp_path)) == 2

    def test_caches_one_queue_per_project_root(self, tmp_path):
        esc = Mem0UpdateStormEscalator()
        root_a = tmp_path / 'a'
        root_b = tmp_path / 'b'
        esc.set_known_projects({'proj_a': str(root_a), 'proj_b': str(root_b)})

        esc.report_storm(
            project_id='proj_a', agent_id='recon-stage-1',
            count=20, threshold=20, window_seconds=3600.0,
        )
        first_queue = esc._queues[str(root_a)]
        esc.report_storm(
            project_id='proj_a', agent_id='recon-stage-3',
            count=20, threshold=20, window_seconds=3600.0,
        )
        esc.report_storm(
            project_id='proj_b', agent_id='recon-stage-1',
            count=20, threshold=20, window_seconds=3600.0,
        )

        assert sorted(esc._queues) == sorted([str(root_a), str(root_b)])
        assert esc._queues[str(root_a)] is first_queue, 'queue must be cached, not rebuilt'
        # Each project's escalations land in its OWN queue directory.
        assert len(_read_escalations(root_a)) == 2
        assert len(_read_escalations(root_b)) == 1

    def test_unset_map_warns_and_submits_nothing(self, tmp_path, caplog):
        """Setter never called → structured WARN, no submission, no cwd fallback."""
        esc = Mem0UpdateStormEscalator()
        with caplog.at_level(logging.WARNING, logger=storm_mod.__name__):
            result = esc.report_storm(
                project_id='dark_factory',
                agent_id='recon-stage-1',
                count=20,
                threshold=20,
                window_seconds=3600.0,
            )

        assert result is None
        assert esc._queues == {}, 'no queue may be built for an unresolvable project'
        text = caplog.text
        assert 'dark_factory' in text
        assert '20' in text, 'the WARN must carry the storm count'
        assert '3600' in text, 'the WARN must carry the window'
        assert 'recon-stage-1' in text
        assert not _read_escalations(tmp_path)

    def test_unknown_project_id_warns_and_submits_nothing(self, tmp_path, caplog):
        """project_id absent from the injected map → same no-op posture.

        Explicitly NOT a fall back to ``config.taskmaster.project_root``, which
        defaults to ``'.'`` and would file into the server's cwd where no
        operator is watching.
        """
        esc = Mem0UpdateStormEscalator()
        esc.set_known_projects({'dark_factory': str(tmp_path)})
        with caplog.at_level(logging.WARNING, logger=storm_mod.__name__):
            result = esc.report_storm(
                project_id='some_other_project',
                agent_id='recon-stage-1',
                count=25,
                threshold=20,
                window_seconds=3600.0,
            )

        assert result is None
        assert esc._queues == {}
        assert 'some_other_project' in caplog.text
        assert not _read_escalations(tmp_path)

    def test_queue_failure_returns_none_does_not_raise(self, tmp_path, monkeypatch):
        """Queue I/O failure is swallowed — the alarm is additive to the write."""
        from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

        def boom(self, _esc):
            raise RuntimeError('disk full')

        monkeypatch.setattr(EscalationQueue, 'submit', boom)
        esc = Mem0UpdateStormEscalator()
        esc.set_known_projects({'dark_factory': str(tmp_path)})

        result = esc.report_storm(
            project_id='dark_factory',
            agent_id='recon-stage-1',
            count=20,
            threshold=20,
            window_seconds=3600.0,
        )
        assert result is None  # no exception propagates


class TestStormEscalationDisabled:
    def test_no_op_when_escalation_pkg_unavailable(self, tmp_path, monkeypatch):
        """The optional ``escalation`` package stays optional.

        Same no-op posture as an unresolvable project_root, so the alarm has ONE
        degraded behaviour, not two.
        """
        monkeypatch.setattr(storm_mod, 'HAS_ESCALATION', False)
        esc = Mem0UpdateStormEscalator()
        esc.set_known_projects({'dark_factory': str(tmp_path)})

        result = esc.report_storm(
            project_id='dark_factory',
            agent_id='recon-stage-1',
            count=20,
            threshold=20,
            window_seconds=3600.0,
        )
        assert result is None
        assert not _read_escalations(tmp_path)
