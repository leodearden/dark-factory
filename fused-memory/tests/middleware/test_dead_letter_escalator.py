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


class TestDedupeFold:
    """Folds on `(project, operation, error class)` — the triple an operator
    needs to tell "the same failure again" from "a new failure mode".

    The fingerprint route, not the per-project open-escalation anchor scan
    `referent_repair_storm_escalator` uses: that shape folds every event in a
    project into ONE entry and so cannot attribute a burst. Here attribution is
    the entire point — an alarm that collapsed a NodeNotFoundError storm on
    add_episode together with an unrelated TimeoutError on add_memory_graphiti
    would hide the second behind the first.
    """

    def test_the_fingerprint_is_over_project_operation_and_error_class_only(
        self, tmp_path,
    ):
        """Deliberately NOT over item_id, attempts or content_preview: all
        three change on every death, so including any would mint a fresh
        escalation per death and defeat the folding this exists to provide."""
        from escalation.dedupe import compute_content_fingerprint

        _emit(tmp_path)

        expected = compute_content_fingerprint(
            dle_mod._CATEGORY,
            dle_mod._FINDING_CATEGORY,
            affected_ids=[
                f'project:{_PROJECT}',
                'operation:add_episode',
                'error:NodeNotFoundError',
            ],
        )
        assert _filed(tmp_path)[0]['dedupe_fingerprint'] == expected

        # And it is genuinely item/attempt/content independent.
        _emit(tmp_path, item_id=999, attempts=55, content_preview='something else')
        assert {p['dedupe_fingerprint'] for p in _filed(tmp_path)} == {expected}

    def test_a_repeat_death_folds_into_the_pending_parent(self, tmp_path):
        """The 28-dead-writes case pages ONCE, not 28 times."""
        first = _emit(tmp_path)
        second = _emit(tmp_path, item_id=8)
        third = _emit(tmp_path, item_id=9)

        assert first is not None
        assert second == first
        assert third == first

        payloads = _filed(tmp_path)
        assert len(payloads) == 1, f'expected one surviving record, got {payloads}'
        assert payloads[0]['id'] == first
        assert payloads[0]['dedupe_count'] == 2
        assert len(payloads[0]['dedupe_children']) == 2

    def test_the_same_error_class_with_a_different_message_still_folds(
        self, tmp_path,
    ):
        """The reason the fingerprint keys on the CLASS, not the message.

        Every error in the esc-3561-3 corpus was `node <uuid> not found` with a
        DIFFERENT uuid per write. A message-keyed fingerprint would have minted
        28 separate escalations and reproduced the paging storm the fold exists
        to prevent.
        """
        first = _emit(tmp_path, error='NodeNotFoundError: node abc not found')
        second = _emit(tmp_path, error='NodeNotFoundError: node def not found')

        assert second == first
        assert len(_filed(tmp_path)) == 1

    def test_the_post_execute_prefix_does_not_defeat_the_fold(self, tmp_path):
        """`POST_EXECUTE_DEAD_PREFIX` is prepended to the error text, so the
        class parse has to strip it or a post-execute death would never fold
        with anything."""
        from fused_memory.services.durable_queue import POST_EXECUTE_DEAD_PREFIX

        plain = _emit(tmp_path, error='RuntimeError: callback keeps failing')
        prefixed = _emit(
            tmp_path,
            error=f'{POST_EXECUTE_DEAD_PREFIX}RuntimeError: callback keeps failing',
            post_execute=True,
        )

        assert prefixed == plain
        assert len(_filed(tmp_path)) == 1
        assert dle_mod._error_class(
            f'{POST_EXECUTE_DEAD_PREFIX}TimeoutError: too slow'
        ) == 'TimeoutError'

    def test_a_different_operation_mints_a_new_escalation(self, tmp_path):
        """add_memory's Graphiti leg dying is not the same news as add_episode
        dying, and one entry naming neither cannot be triaged."""
        first = _emit(tmp_path, operation='add_episode')
        second = _emit(tmp_path, operation='add_memory_graphiti')

        assert first is not None and second is not None
        assert second != first
        assert len(_filed(tmp_path)) == 2

    def test_a_different_error_class_mints_a_new_escalation(self, tmp_path):
        first = _emit(tmp_path, error='NodeNotFoundError: node abc not found')
        second = _emit(tmp_path, error='TimeoutError: backend did not respond')

        assert first is not None and second is not None
        assert second != first
        assert len(_filed(tmp_path)) == 2

    def test_a_different_project_mints_a_new_escalation(self, tmp_path):
        """Asserted in ONE root, so it is the fingerprint being tested rather
        than two queues that could not have collided anyway."""
        first = _emit(tmp_path, project_id='proj1', group_id='proj1')
        second = _emit(tmp_path, project_id='proj2', group_id='proj2')

        assert first is not None and second is not None
        assert second != first
        assert len(_filed(tmp_path)) == 2

    def test_the_fold_window_is_unbounded(self, tmp_path):
        """A durable write can die once every few days for months — which is
        exactly what esc-3561-3 was. Under any finite window the next death
        would page again."""
        first = _emit(tmp_path)
        parent_path = next(iter((tmp_path / 'data' / 'escalations').glob('esc-*.json')))
        payload = json.loads(parent_path.read_text())
        payload['timestamp'] = '2016-01-01T00:00:00+00:00'
        parent_path.write_text(json.dumps(payload))

        second = _emit(tmp_path, item_id=8)

        assert second == first
        assert len(_filed(tmp_path)) == 1

    def test_an_unparseable_error_still_files(self, tmp_path):
        """`None` or a message with no class prefix is a reason to fold under
        `'unknown'`, never a reason to drop the alarm."""
        from escalation.dedupe import compute_content_fingerprint

        assert _emit(tmp_path, error=None) is not None
        assert _filed(tmp_path)[0]['dedupe_fingerprint'] == compute_content_fingerprint(
            dle_mod._CATEGORY,
            dle_mod._FINDING_CATEGORY,
            affected_ids=[
                f'project:{_PROJECT}', 'operation:add_episode', 'error:unknown',
            ],
        )
