"""The INV-4 repair-storm alarm (task 3672, PRD leaf eta).

The FIRE half of the storm escape whose counter half is
`MemoryService._referent_repair_streaks`.  When a project's repair pass has
moved at least one edge endpoint in ten CONSECUTIVE episodes, the scanner or
the resolver has regressed — the measured base rate is ~0.22% of live
task-mentioning edges, so a sustained streak is anomalous by construction —
and that must not be absorbed silently.

THE ESCALATION IS THE ALARM, NOT A HALT.  Repairs continue while it fires;
this module is never a rate limiter.  It therefore inherits the never-raise
contract every fused-memory escalator has, for the same reason: it is called
from the live write path, and a raise here would fail a write because the
COMPLAINT ABOUT the write failed.

Modelled on `middleware/candidate_key_escalation` (the module-function shape)
rather than `mem0_update_storm_escalator` (the class shape): attribution here
is per-PROJECT, not per-agent.
"""

from __future__ import annotations

import json

import pytest

from fused_memory.middleware import referent_repair_storm_escalator as rrse_mod
from fused_memory.middleware.referent_repair_storm_escalator import (
    emit_referent_repair_storm_escalation,
)

pytestmark = pytest.mark.skipif(
    not rrse_mod.HAS_ESCALATION,
    reason='escalation package unavailable (minimal env); the HAS_ESCALATION '
           'no-op arm is covered separately below',
)


def _records() -> list[dict]:
    """The INV-2 structured evidence, as `ReferentRepair.to_dict()` renders it."""
    return [
        {
            'edge_uuid': 'e1',
            'which_end': 'source',
            'outcome': 'repaired',
            'old_endpoint_uuid': 'n-3129',
            'new_endpoint_uuid': 'n-3127',
            'intended_referent': 'Task 3127',
            'check': 'set-membership',
            'minted': False,
            'moved': True,
            'summaries_refreshed': ['n-3129', 'n-3127'],
            'deleted_emptied_node': '',
            'reason': '',
        },
    ]


def _emit(tmp_path, **overrides) -> str | None:
    kwargs = {
        'project_id': 'dark_factory',
        'streak': 10,
        'threshold': 10,
        'repairs': 1,
        'records': _records(),
    }
    kwargs.update(overrides)
    return emit_referent_repair_storm_escalation(str(tmp_path), **kwargs)


def _filed(tmp_path) -> list[dict]:
    queue_dir = tmp_path / 'data' / 'escalations'
    return [json.loads(p.read_text()) for p in sorted(queue_dir.glob('esc-*.json'))]


class TestTheFiledEscalation:
    """One escalation, into the affected project's OWN queue."""

    def test_files_exactly_one_escalation_and_returns_its_id(self, tmp_path):
        esc_id = _emit(tmp_path)

        assert isinstance(esc_id, str)
        filed = _filed(tmp_path)
        assert len(filed) == 1
        assert filed[0]['id'] == esc_id

    def test_lands_in_the_affected_projects_own_queue(self, tmp_path):
        """`{project_root}/data/escalations`, resolved from the caller's
        project_root — never the server cwd, where no operator watches."""
        _emit(tmp_path)
        assert (tmp_path / 'data' / 'escalations').is_dir()

    def test_carries_the_category_severity_role_and_anchor(self, tmp_path):
        esc_id = _emit(tmp_path)
        assert esc_id is not None
        record = _filed(tmp_path)[0]

        assert record['category'] == 'referent_repair_storm'
        assert record['severity'] == 'blocking'
        assert record['agent_role'] == 'fused-memory/referent-repair-guard'
        assert record['task_id'] == rrse_mod._ANCHOR_TASK_ID
        assert esc_id.startswith(f'esc-{rrse_mod._ANCHOR_TASK_ID}-'), (
            'the id must be minted by queue.make_id off the stable anchor, so '
            'the series is greppable and dedupe can find it'
        )

    def test_is_born_at_l0_so_it_routes_to_the_steward(self, tmp_path):
        """The correct ladder entry for an agent-filed alarm: L0 -> steward,
        who re-escalates if it needs the auto-watcher. Self-assigning a higher
        level buys no faster attention, only noise."""
        _emit(tmp_path)
        assert _filed(tmp_path)[0]['level'] == 0

    def test_the_summary_names_the_project_and_the_streak(self, tmp_path):
        _emit(tmp_path, project_id='reify', streak=12, threshold=10)
        summary = _filed(tmp_path)[0]['summary']

        assert 'reify' in summary
        assert '12' in summary

    def test_the_detail_carries_every_operator_input(self, tmp_path):
        _emit(tmp_path, project_id='dark_factory', streak=11, threshold=10, repairs=4)
        detail = _filed(tmp_path)[0]['detail']

        assert 'dark_factory' in detail
        assert str(tmp_path) in detail
        assert '11' in detail
        assert '10' in detail
        assert '4' in detail

    def test_the_detail_ships_the_structured_records_not_just_a_count(
        self, tmp_path,
    ):
        """INV-2: the alarm ships the EVIDENCE. A count alone cannot tell an
        operator whether the resolver is mis-targeting one node or the scanner
        is firing on everything."""
        _emit(tmp_path)
        detail = _filed(tmp_path)[0]['detail']

        for token in ('e1', 'n-3129', 'n-3127', 'Task 3127', 'set-membership'):
            assert token in detail, f'{token!r} missing from the escalation detail'

    def test_the_suggested_action_names_where_to_look(self, tmp_path):
        _emit(tmp_path)
        action = _filed(tmp_path)[0]['suggested_action']

        assert 'canonical_labels' in action, 'the scanner'
        assert 'referent_resolution' in action, 'the resolver'
        assert 'referent_repair_counts' in action, 'the read side'

    def test_the_suggested_action_states_repairs_were_not_halted(self, tmp_path):
        """The escalation is the ALARM, not a halt — an operator who assumes
        writes are parked will mis-triage the urgency."""
        _emit(tmp_path)
        action = _filed(tmp_path)[0]['suggested_action'].lower()

        assert 'not halted' in action or 'were not halted' in action


class TestDedupeFold:
    """A SUSTAINED storm folds into one pending parent, not one page per breach."""

    def test_a_second_call_while_the_first_is_pending_reuses_its_id(self, tmp_path):
        first = _emit(tmp_path)
        second = _emit(tmp_path, streak=11)

        assert first is not None
        assert second == first
        assert len(_filed(tmp_path)) == 1, (
            'a storm breaches the threshold on EVERY subsequent episode; '
            'filing one escalation per breach would bury the queue'
        )

    def test_dedupe_is_keyed_on_the_pending_anchor(self, tmp_path):
        """`queue.get_by_task(anchor, status='pending')` — per-PROJECT, which
        is the whole reason for the module-function shape."""
        first = _emit(tmp_path)
        assert first is not None
        from escalation.queue import EscalationQueue

        EscalationQueue(tmp_path / 'data' / 'escalations').resolve(
            first, 'scanner regression fixed',
        )

        third = _emit(tmp_path)
        assert third is not None
        assert third != first, (
            'once the prior alarm is resolved a fresh storm must be able to '
            'file again, or the alarm is one-shot for the process lifetime'
        )
        # `resolve` archives the first out of the queue root, so the root now
        # holds exactly the fresh alarm — the dedupe lookup is scoped to
        # PENDING, not to "an escalation was ever filed".
        live = _filed(tmp_path)
        assert [record['id'] for record in live] == [third]

    def test_two_projects_do_not_dedupe_against_each_other(self, tmp_path):
        """Attribution is per-project: `reify` storming must not be silenced
        by an open `dark_factory` alarm."""
        root_a = tmp_path / 'a'
        root_b = tmp_path / 'b'
        root_a.mkdir()
        root_b.mkdir()

        id_a = _emit(root_a, project_id='dark_factory')
        id_b = _emit(root_b, project_id='reify')

        assert id_a is not None
        assert id_b is not None
        assert len(_filed(root_a)) == 1
        assert len(_filed(root_b)) == 1


class TestNeverRaises:
    """Called from the live write path: a raise here fails a write because the
    COMPLAINT about the write failed."""

    def test_a_submit_failure_returns_none_and_logs(self, tmp_path, monkeypatch, caplog):
        class _BrokenQueue:
            def __init__(self, *_a, **_kw):
                pass

            def get_by_task(self, *_a, **_kw):
                return []

            def make_id(self, task_id):
                return f'esc-{task_id}-1'

            def submit(self, _esc):
                raise OSError('read-only filesystem')

        monkeypatch.setattr(rrse_mod, 'EscalationQueue', _BrokenQueue)

        with caplog.at_level('ERROR'):
            assert _emit(tmp_path) is None
        assert caplog.records, 'a swallowed failure must still be visible'

    def test_a_get_by_task_failure_falls_through_to_filing(self, tmp_path, monkeypatch):
        """A read failure must not BLOCK the alarm — better a possible
        duplicate than a silenced storm."""
        real_queue = rrse_mod.EscalationQueue

        class _UnreadableQueue(real_queue):  # type: ignore[misc,valid-type]
            def get_by_task(self, *_a, **_kw):
                raise OSError('queue scan failed')

        monkeypatch.setattr(rrse_mod, 'EscalationQueue', _UnreadableQueue)

        esc_id = _emit(tmp_path)
        assert isinstance(esc_id, str)
        assert len(_filed(tmp_path)) == 1

    def test_a_queue_construction_failure_returns_none(self, tmp_path, monkeypatch):
        def _explode(*_a, **_kw):
            raise OSError('cannot create queue dir')

        monkeypatch.setattr(rrse_mod, 'EscalationQueue', _explode)

        assert _emit(tmp_path) is None


def test_without_the_escalation_package_it_no_ops(tmp_path, monkeypatch, caplog):
    """The minimal-env path: logged, nothing filed, `None` returned. The
    repair pass must behave identically whether or not the optional
    `escalation` workspace package is installed."""
    monkeypatch.setattr(rrse_mod, 'HAS_ESCALATION', False)

    with caplog.at_level('DEBUG'):
        result = emit_referent_repair_storm_escalation(
            str(tmp_path),
            project_id='dark_factory',
            streak=10,
            threshold=10,
            repairs=1,
            records=_records(),
        )

    assert result is None
    assert not (tmp_path / 'data' / 'escalations').exists()
    assert caplog.records, 'a no-op alarm must still say so'
