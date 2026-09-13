"""The durable-write dead-letter alarm (task 3583): `emit_dead_letter_escalation`.

The PUSH half of the signal whose seam is
`DurableWriteQueue._notify_dead_letter` and whose caller is
`MemoryService._report_queue_dead_letter`.

WHY THIS FILE EXISTS AT ALL, and why its assertions are worth their weight. The
module swallows every exception and returns None, so a regression in any of its
load-bearing claims loses the alarm SILENTLY — which is the precise failure
shape the alarm exists to remove. Worse, this escalation is the ONLY evidence
that survives routine `delete_dead_letters` cleanup: 26 of the 28 dead rows in
the esc-3561-3 investigation had already been swept by the time anyone looked,
and every pull-side counter had gone back to zero with them.

Modelled on `tests/middleware/test_entity_mint_storm_escalator.py` (the newest
sibling) for the `_filed` helper, the module-level skipif, and the
never-raise checklist including the queue-CONSTRUCTION failure the
fresh-per-call shape makes reachable.
"""

from __future__ import annotations

import json
import logging

import pytest

from fused_memory.middleware import dead_letter_escalator as dle_mod
from fused_memory.middleware.dead_letter_escalator import emit_dead_letter_escalation

pytestmark = pytest.mark.skipif(
    not dle_mod.HAS_ESCALATION,
    reason='escalation package unavailable (minimal env); the HAS_ESCALATION '
           'no-op arm is covered separately below',
)

_PROJECT = 'proj1'


def _filed(root):
    """The parsed escalation payloads under ``{root}/data/escalations``."""
    queue_dir = root / 'data' / 'escalations'
    if not queue_dir.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(queue_dir.glob('esc-*.json'))]


def _emit(root, *, project_id=_PROJECT, operation='add_episode',
          group_id=_PROJECT, item_id=7, attempts=5,
          error='NodeNotFoundError: node abc not found', post_execute=False,
          content_preview='an episode that never landed', write_op_id='W1'):
    return emit_dead_letter_escalation(
        str(root),
        project_id=project_id,
        operation=operation,
        group_id=group_id,
        item_id=item_id,
        attempts=attempts,
        error=error,
        post_execute=post_execute,
        content_preview=content_preview,
        write_op_id=write_op_id,
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

        The caller (`MemoryService._report_queue_dead_letter`) resolves it from
        `_known_projects` and NEVER defaults it to the server cwd, where no
        operator watches; this module honours whatever root it is given.
        """
        root_a = tmp_path / 'a'
        root_b = tmp_path / 'b'

        _emit(root_a, project_id='proj_a', group_id='proj_a')
        _emit(root_b, project_id='proj_b', group_id='proj_b')

        assert len(_filed(root_a)) == 1
        assert len(_filed(root_b)) == 1

    def test_carries_the_category_severity_role_and_anchor(self, tmp_path):
        _emit(tmp_path)
        payload = _filed(tmp_path)[0]

        assert payload['category'] == 'durable_write_dead_letter'
        assert payload['severity'] == 'blocking'
        assert payload['agent_role'] == 'fused-memory/dead-letter-guard'
        assert payload['task_id'] == 'durable-write-dead-letter', (
            'a synthetic anchor, so the ids form a greppable '
            'esc-durable-write-dead-letter-N series'
        )
        assert payload['id'].startswith('esc-durable-write-dead-letter-')

    def test_is_born_at_l1_like_every_sibling_fused_memory_escalator(self, tmp_path):
        """L0 routes to a task's steward; this anchor is never dispatched.

        Filed by a background server process under a synthetic anchor task id,
        so an L0 entry would have no consumer at all and would merely wait out
        `orphan_l0_timeout_secs` before being promoted to exactly where L1 puts
        it immediately.
        """
        _emit(tmp_path)
        assert _filed(tmp_path)[0]['level'] == 1


class TestNeverRaises:
    """Dispatched through `asyncio.to_thread` from inside the queue worker.

    The item this complains about is ALREADY COMMITTED dead by the time this
    runs, so a raise here buys nothing and costs the worker: it would propagate
    back through `_notify_dead_letter` into `_process_item`. Every failure mode
    degrades to a log line plus `None`.
    """

    def test_a_queue_construction_failure_returns_none(self, tmp_path):
        """The failure mem0's cached-queue shape HIDES.

        Constructing an `EscalationQueue` creates its directory, so a read-only
        root raises at construction — which a per-call cache would defer to an
        arbitrary later call. Building fresh inside its own try/except contains
        it here.
        """
        blocked = tmp_path / 'blocked'
        blocked.mkdir()
        (blocked / 'data').write_text('not a directory')

        assert _emit(blocked) is None

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

        monkeypatch.setattr(dle_mod, 'EscalationQueue', _BrokenQueue)

        with caplog.at_level(logging.ERROR, logger=dle_mod.__name__):
            result = _emit(tmp_path)

        assert result is None
        assert caplog.records, 'a swallowed failure must still be visible'
        assert 'add_episode' in caplog.text, caplog.text
        assert not _filed(tmp_path)

    def test_without_the_escalation_package_it_no_ops(self, tmp_path, monkeypatch, caplog):
        """The minimal-env path: WARNED, nothing filed, `None` returned.

        The degradation is LOUD because a permanently-lost durable write that
        goes unescalated is precisely the thing an operator needs told.
        """
        monkeypatch.setattr(dle_mod, 'HAS_ESCALATION', False)

        with caplog.at_level(logging.WARNING, logger=dle_mod.__name__):
            result = _emit(tmp_path)

        assert result is None
        assert not (tmp_path / 'data' / 'escalations').exists(), (
            'a no-op must not leave a queue directory behind'
        )
        text = caplog.text
        assert 'add_episode' in text, text
        assert _PROJECT in text, text
