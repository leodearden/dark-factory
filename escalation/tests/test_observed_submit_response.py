"""Tests for `escalation.queue.observed_submit_response` — the ladder's front door.

Task 3236 made this function report OBSERVED state rather than write intent, so
a filing swallowed by a concurrent resolver is no longer reported as 'queued'.
Task 5368 closes the hole that fix left open: when the post-write re-read itself
fails (returns nothing, or raises), the response was byte-identical to a
CONFIRMED persist — so a filer whose escalation never landed was told it had.

The contract these tests freeze: an UNCONFIRMED persist is never shaped like a
confirmed one.  `'queued'` means, and only means, "the re-read found the record
pending on disk"; `'accepted_unpersisted'` means durability is unconfirmed and
the caller must keep driving its blocked task.

`observed_submit_response` had no direct test before this file — it was
exercised only incidentally through the server — so both the function-level
contract (below) and the server-boundary envelope it feeds live here together.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from _filing_tools import call_blocker as _blocker
from _filing_tools import call_info as _info

from escalation.models import (
    ACTION_KEEP_DRIVING,
    ACTION_TERMINATE_CLEANLY,
    FILER_ACTIONS,
    Escalation,
)
from escalation.queue import EscalationQueue, observed_submit_response
from escalation.server import create_server

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_QUEUE_LOGGER = 'escalation.queue'

_COMMON_KWARGS: dict[str, Any] = {
    'task_id': 'task-5368',
    'agent_role': 'implementer',
    'category': 'infra_issue',
    'summary': 'post-write re-read could not confirm persistence',
}


def _submit_pending(queue: EscalationQueue, *, level: int = 1) -> str:
    """Write one genuinely pending escalation and return its id."""
    esc = Escalation(
        id=queue.make_id('task-5368'),
        task_id='task-5368',
        agent_role='implementer',
        severity='blocking',
        category='infra_issue',
        summary='a filing whose persistence we then interrogate',
        level=level,
    )
    return queue.submit(esc)


def _unlink_record(queue: EscalationQueue, esc_id: str) -> Path:
    """Delete the record's file, so a later `get` misses it for the RIGHT reason."""
    path = queue.queue_dir / f'{esc_id}.json'
    path.unlink()
    return path


def _tear_record(queue: EscalationQueue, esc_id: str) -> Path:
    """Leave a torn write behind: a real file that parses into no record."""
    path = queue.queue_dir / f'{esc_id}.json'
    text = path.read_text()
    path.write_text(text[: len(text) // 2])
    return path


def _error_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


# ---------------------------------------------------------------------------
# Function-level contract
# ---------------------------------------------------------------------------


class TestObservedSubmitResponseShapes:
    """The three response shapes must stay mutually distinguishable."""

    def test_confirmed_pending_is_queued_with_no_persist_check(self, tmp_path: Path):
        """(a) A re-read that FINDS the record pending is the one durable shape.

        `queued` carries exactly id/status/level and no `persist_check` — the
        absence of that key is itself the signal that nothing was degraded.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        esc_id = _submit_pending(queue, level=1)

        result = observed_submit_response(queue, esc_id, fallback_level=1)

        assert result == {'id': esc_id, 'status': 'queued', 'level': 1}, (
            f'Confirmed-persist shape drifted: {result}'
        )
        assert 'persist_check' not in result, (
            f'A confirmed persist must not carry a persist_check verdict: {result}'
        )

    def test_absent_record_reports_accepted_unpersisted(self, tmp_path: Path):
        """(b) Nothing on disk → unpersisted, discriminated as 'absent'.

        `EscalationQueue.get` searches the queue root, the archive, a targeted
        archive re-probe and a TOCTOU re-locate retry, and the None it returns
        is re-probed for a surviving path before the verdict is written — so
        'absent' is asserted here against a real empty filesystem rather than a
        stub, because a stub cannot distinguish the two causes it now decides.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        esc_id = _submit_pending(queue, level=1)
        removed = _unlink_record(queue, esc_id)

        result = observed_submit_response(queue, esc_id, fallback_level=1)

        assert not removed.exists(), 'Precondition: the record must really be gone'

        assert result['status'] == 'accepted_unpersisted', (
            f"Expected 'accepted_unpersisted' for an absent record, got: {result}"
        )
        assert result['persist_check'] == 'absent', (
            f"Expected persist_check='absent', got: {result}"
        )
        assert result['level'] == 1, f'Level echo lost on the absent branch: {result}'
        assert result['id'] == esc_id, f'Escalation id lost: {result}'

    def test_torn_write_reports_unreadable_not_absent(self, tmp_path: Path):
        """(b') A file that IS on disk but parses into nothing is not 'absent'.

        `get` answers None for a record it cannot parse exactly as it does for
        one that is nowhere.  A torn or half-written file is the very "accepted
        but not durable" failure this response exists to name, so reporting it
        as a missing file sends the operator hunting for something that is
        sitting right there — the one case where the two verdicts diverge in
        practice is the one that must not lie.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        esc_id = _submit_pending(queue, level=1)
        torn = _tear_record(queue, esc_id)

        result = observed_submit_response(queue, esc_id, fallback_level=1)

        assert torn.exists(), 'Precondition: the torn file must still be on disk'
        assert result['status'] == 'accepted_unpersisted', (
            f'A torn write is still an unconfirmed persist: {result}'
        )
        assert result['persist_check'] == 'unreadable', (
            f"A torn write is unreadable, not absent: {result}"
        )
        assert result['level'] == 1, f'Level echo lost on the torn-write branch: {result}'

    def test_unreadable_record_reports_accepted_unpersisted(self, tmp_path: Path):
        """(c) A re-read that RAISES → unpersisted, discriminated as 'unreadable'.

        The read failed, so the record's state is simply unknown — materially
        different from 'absent' for an operator debugging the outage, and the
        exception must not propagate to the filer.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        esc_id = _submit_pending(queue, level=1)

        def _boom(escalation_id: str):
            raise OSError('simulated unreadable escalation file')

        queue.get = _boom  # type: ignore[method-assign]

        result = observed_submit_response(queue, esc_id, fallback_level=1)

        assert result['status'] == 'accepted_unpersisted', (
            f"Expected 'accepted_unpersisted' for an unreadable record, got: {result}"
        )
        assert result['persist_check'] == 'unreadable', (
            f"Expected persist_check='unreadable', got: {result}"
        )
        assert result['level'] == 1, f'Level echo lost on the unreadable branch: {result}'
        assert result['id'] == esc_id, f'Escalation id lost: {result}'

    def test_unpersisted_response_differs_from_confirmed(self, tmp_path: Path):
        """(d) The defect asserted directly: the two shapes must not re-converge.

        Same id, same level — if a future edit makes an unconfirmed persist
        indistinguishable from a confirmed one again, this fails.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        esc_id = _submit_pending(queue, level=1)

        confirmed = observed_submit_response(queue, esc_id, fallback_level=1)
        queue.get = lambda escalation_id: None  # type: ignore[method-assign]
        unpersisted = observed_submit_response(queue, esc_id, fallback_level=1)

        assert unpersisted != confirmed, (
            'An UNCONFIRMED persist is shaped identically to a confirmed one — '
            f'the S8-13 defect: {unpersisted!r}'
        )

    @pytest.mark.parametrize('break_get', ['absent', 'unreadable'])
    def test_no_fallback_level_fabricates_no_level_key(self, tmp_path: Path, break_get: str):
        """(e) A legacy caller passing no fallback_level gets no fabricated level."""
        queue = EscalationQueue(tmp_path / 'esc')
        esc_id = _submit_pending(queue, level=1)

        if break_get == 'absent':
            _unlink_record(queue, esc_id)
        else:
            def _boom(escalation_id: str):
                raise OSError('simulated unreadable escalation file')

            queue.get = _boom  # type: ignore[method-assign]

        result = observed_submit_response(queue, esc_id, fallback_level=None)

        assert result['status'] == 'accepted_unpersisted', f'Unexpected status: {result}'
        assert 'level' not in result, (
            f'level must never be fabricated when the caller supplied none: {result}'
        )


class TestUnpersistedLogsAtError:
    """(f) A filing whose durability is unconfirmed is not a routine degrade."""

    def test_absent_branch_logs_error_naming_the_id(self, tmp_path: Path, caplog):
        queue = EscalationQueue(tmp_path / 'esc')
        esc_id = _submit_pending(queue, level=1)
        _unlink_record(queue, esc_id)

        with caplog.at_level(logging.ERROR, logger=_QUEUE_LOGGER):
            observed_submit_response(queue, esc_id, fallback_level=1)

        matching = [r for r in _error_records(caplog) if esc_id in r.getMessage()]
        assert matching, (
            f'Expected an ERROR naming {esc_id}; got: '
            f'{[(r.levelname, r.getMessage()) for r in caplog.records]}'
        )

    def test_unreadable_branch_logs_error_naming_the_id(self, tmp_path: Path, caplog):
        queue = EscalationQueue(tmp_path / 'esc')
        esc_id = _submit_pending(queue, level=1)

        def _boom(escalation_id: str):
            raise OSError('simulated unreadable escalation file')

        queue.get = _boom  # type: ignore[method-assign]

        with caplog.at_level(logging.ERROR, logger=_QUEUE_LOGGER):
            observed_submit_response(queue, esc_id, fallback_level=1)

        matching = [r for r in _error_records(caplog) if esc_id in r.getMessage()]
        assert matching, (
            f'Expected an ERROR naming {esc_id}; got: '
            f'{[(r.levelname, r.getMessage()) for r in caplog.records]}'
        )

    def test_the_two_causes_do_not_share_one_sentence(self, tmp_path: Path, caplog):
        """(g) A log that cannot tell the two apart necessarily lies about one.

        Deliberately wording-agnostic: what is pinned is that a file left on
        disk and a file that is gone produce DIFFERENT operator-facing
        sentences, never a particular phrasing.  The `persist_check` field is
        the machine-readable half of the same distinction.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        gone_id = _submit_pending(queue, level=1)
        _unlink_record(queue, gone_id)
        torn_id = _submit_pending(queue, level=1)
        _tear_record(queue, torn_id)

        with caplog.at_level(logging.ERROR, logger=_QUEUE_LOGGER):
            observed_submit_response(queue, gone_id, fallback_level=1)
            observed_submit_response(queue, torn_id, fallback_level=1)

        said = {
            esc_id: [r.getMessage() for r in _error_records(caplog) if esc_id in r.getMessage()]
            for esc_id in (gone_id, torn_id)
        }
        assert all(said.values()), f'Both causes must log an ERROR: {said}'
        gone_sentence = said[gone_id][0].replace(gone_id, '<id>')
        torn_sentence = said[torn_id][0].replace(torn_id, '<id>')
        assert gone_sentence != torn_sentence, (
            f'One sentence for both causes cannot be true of both: {gone_sentence!r}'
        )


# ---------------------------------------------------------------------------
# Server boundary: the envelope the agent actually reads
# ---------------------------------------------------------------------------


class TestUnpersistedReachesTheAgentFacingEnvelope:
    """The status must change the INSTRUCTION, not just the label beside it.

    `escalate_blocker` appended `action='terminate_cleanly'` to every response
    unconditionally.  That key, not `status`, is what the filer acts on — an
    agent told to terminate cleanly removes its task from every recovery path,
    believing a human will see the escalation.  Changing only `status` would
    leave the envelope self-contradictory: a filing marked unpersisted while
    still being told to stand down.
    """

    @pytest.mark.asyncio
    async def test_healthy_queue_still_terminates_cleanly(self, tmp_path: Path):
        """(a) The undegraded path is unchanged — queued, terminate_cleanly."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _blocker(server, level=1, **_COMMON_KWARGS)

        assert result['status'] == 'queued', f'Unexpected status: {result}'
        assert result['action'] == ACTION_TERMINATE_CLEANLY, (
            f'The healthy blocker path must still stand the agent down: {result}'
        )
        assert result['level'] == 1, f'Level echo missing: {result}'

    @pytest.mark.asyncio
    async def test_absent_record_tells_the_blocker_to_keep_driving(self, tmp_path: Path):
        """(b) Nothing persisted → the agent must NOT stand down.

        This is the assertion that closes S8-13's wrong decision: with no
        record on disk, no L1 or L2 drain will ever see this filing, so
        terminating cleanly strands the task in silence.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        queue.get = lambda escalation_id: None  # type: ignore[method-assign]

        result = await _blocker(server, level=1, **_COMMON_KWARGS)

        assert result['status'] == 'accepted_unpersisted', f'Unexpected status: {result}'
        assert result['action'] == ACTION_KEEP_DRIVING, (
            f'Expected {ACTION_KEEP_DRIVING!r} on an unpersisted filing, got: {result}'
        )
        assert result['action'] != ACTION_TERMINATE_CLEANLY
        assert result['level'] == 1, f'Level echo missing: {result}'

    @pytest.mark.asyncio
    async def test_unreadable_record_tells_the_blocker_to_keep_driving(self, tmp_path: Path):
        """(c) An unreadable re-read carries the same instruction as an absent one."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        def _boom(escalation_id: str):
            raise OSError('simulated unreadable escalation file')

        queue.get = _boom  # type: ignore[method-assign]

        result = await _blocker(server, level=1, **_COMMON_KWARGS)

        assert result['status'] == 'accepted_unpersisted', f'Unexpected status: {result}'
        assert result['action'] == ACTION_KEEP_DRIVING, (
            f'Expected {ACTION_KEEP_DRIVING!r} on an unpersisted filing, got: {result}'
        )
        assert result['level'] == 1, f'Level echo missing: {result}'

    @pytest.mark.asyncio
    async def test_info_path_surfaces_status_and_grows_no_action_key(self, tmp_path: Path):
        """(d) escalate_info never had an `action`, and must not acquire one."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        queue.get = lambda escalation_id: None  # type: ignore[method-assign]

        result = await _info(server, **_COMMON_KWARGS)

        assert result['status'] == 'accepted_unpersisted', f'Unexpected status: {result}'
        assert 'action' not in result, (
            f'The info path has no instruction to give and must stay that way: {result}'
        )
        assert 'level' in result, f'Level echo missing: {result}'

    @pytest.mark.asyncio
    async def test_born_at_l2_bypass_also_keeps_driving(self, tmp_path: Path):
        """(e) The L2 front door is covered too — it does NOT route through dedupe.

        A born-at-L2 filing reaches `observed_submit_response` by its own path
        in `_submit_or_dedupe`, so it needs its own assertion; and it is the
        filing whose loss costs the most, being the one addressed to a human.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        queue.get = lambda escalation_id: None  # type: ignore[method-assign]

        result = await _blocker(
            server, severity='critical',
            **{**_COMMON_KWARGS, 'agent_role': 'orchestrator-watcher-supervisor'},
        )

        assert result['status'] == 'accepted_unpersisted', f'Unexpected status: {result}'
        assert result['action'] == ACTION_KEEP_DRIVING, (
            f'Expected {ACTION_KEEP_DRIVING!r} on an unpersisted L2 filing, got: {result}'
        )
        assert 'level' in result, f'Level echo missing: {result}'


# ---------------------------------------------------------------------------
# The filer-facing action vocabulary
# ---------------------------------------------------------------------------


class TestFilerActionVocabulary:
    """`action` is an instruction NO CODE READS — its only consumer is the
    agent reading the tool result.  That makes the wire values load-bearing
    prose: `orchestrator.agents.roles.ESCALATION_LADDER_CORE` quotes them
    verbatim to tell an agent which response means "stop" and which means
    "keep driving", and a role prompt cannot be an f-string.  A rename here
    would therefore decouple the instruction from the response it describes
    while raising no import error anywhere — so the values are frozen by name
    and the vocabulary is closed, letting the prompt's copy be pinned against
    the emission site from across the package boundary.

    This is NOT `escalation.server.RESOLVE_ACTIONS`, the handler-side
    `resolve_issue` disposition; the two vocabularies merely share a key name.

    Closure is pinned where it has content — `action in FILER_ACTIONS` on each
    emitting branch below, which a future branch inventing an undeclared value
    would fail.  Asserting the tuple's own contents back at it would restate
    one line of models.py and could only fail when someone edited that line.
    """

    def test_wire_values_are_frozen(self):
        """(a) The strings are the contract, not merely the names bound to them."""
        assert ACTION_TERMINATE_CLEANLY == 'terminate_cleanly'
        assert ACTION_KEEP_DRIVING == 'keep_driving'

    @pytest.mark.asyncio
    async def test_healthy_branch_emits_the_named_terminate_action(self, tmp_path: Path):
        """(c)+(d) The observed-persist branch, asserted through the names.

        Asserting through the constant is what makes the emission site and the
        vocabulary provably the same object rather than two literals that
        happen to match today.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        result = await _blocker(server, level=1, **_COMMON_KWARGS)

        assert result['action'] == ACTION_TERMINATE_CLEANLY, f'Unexpected action: {result}'
        assert result['action'] in FILER_ACTIONS, (
            f'An action no agent has been told how to read: {result}'
        )

    @pytest.mark.asyncio
    async def test_unpersisted_branch_emits_the_named_keep_driving_action(
        self, tmp_path: Path,
    ):
        """(c)+(d) The unconfirmed-persist branch, asserted through the names."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        queue.get = lambda escalation_id: None  # type: ignore[method-assign]

        result = await _blocker(server, level=1, **_COMMON_KWARGS)

        assert result['action'] == ACTION_KEEP_DRIVING, f'Unexpected action: {result}'
        assert result['action'] in FILER_ACTIONS, (
            f'An action no agent has been told how to read: {result}'
        )


# ---------------------------------------------------------------------------
# What the `keep_driving` retry actually costs
# ---------------------------------------------------------------------------


class TestRepeatFilingIsNotGenerallyFolded:
    """Why `keep_driving` bounds the retry at ONE re-file.

    `keep_driving` tells a filer to keep working and re-file.  An earlier
    revision justified that as unconditionally safe — "the dedupe gate
    collapses a repeat filing into one record" — and left the loop unbounded.
    These tests measure the gate instead of asserting the claim, and the
    measurement is why the instruction now says ONCE:

    - the stock server config folds ONLY `category='infra_issue'`
      (`DedupeConfig()`), so five of the six documented blocker categories
      never fold at all; and
    - a born-at-L2 severity bypasses `_submit_or_dedupe`'s gate entirely — the
      highest-cost route, where each repeat is another record addressed to a
      human.

    Note the inversion relative to what a "bounded recovery" test would look
    like under the other fix: the bound lives in the prompt and the tool
    docstring, not in the gate, so what is assertable HERE is the cost the
    bound exists to cap — N re-files really do mint N records.  A future change
    that made folding universal would fail these tests, which is the intended
    signal: the prose would then be wrong in the opposite direction.
    """

    @staticmethod
    def _pending_ids(queue: EscalationQueue) -> set[str]:
        return {esc.id for esc in queue.get_pending()}

    @pytest.mark.asyncio
    async def test_non_infra_repeat_mints_a_second_record(self, tmp_path: Path):
        """(a) A repeat `scope_violation` filing does NOT fold."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        kwargs = {**_COMMON_KWARGS, 'category': 'scope_violation'}

        first = await _blocker(server, **kwargs)
        second = await _blocker(server, **kwargs)

        assert second['status'] != 'dedup_skipped', (
            f'scope_violation is outside the stock DedupeConfig categories, so a '
            f'repeat cannot fold: {second}'
        )
        assert first['id'] != second['id'], (
            f'Two filings, one id — the gate folded a category it does not cover: '
            f'{first} / {second}'
        )
        assert len(self._pending_ids(queue)) == 2, (
            f'Expected two distinct pending records: {queue.get_pending()}'
        )

    @pytest.mark.asyncio
    async def test_born_at_l2_repeat_mints_a_second_record(self, tmp_path: Path):
        """(b) The human-paging route is the one where a repeat costs the most.

        `_submit_or_dedupe` returns before reaching the gate for
        `severity in BORN_AT_L2_SEVERITIES`, so N re-files are N records — and
        N pages.  This is exactly the path
        `test_born_at_l2_bypass_also_keeps_driving` proves returns
        `keep_driving`.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)
        kwargs = {
            **_COMMON_KWARGS,
            'agent_role': 'orchestrator-watcher-supervisor',
            'severity': 'critical',
        }

        first = await _blocker(server, **kwargs)
        second = await _blocker(server, **kwargs)

        assert first['status'] == 'queued' and second['status'] == 'queued', (
            f'Born-at-L2 filings never dedupe: {first} / {second}'
        )
        assert first['id'] != second['id'], (
            f'A born-at-L2 repeat must mint its own record: {first} / {second}'
        )
        assert len(self._pending_ids(queue)) == 2, (
            f'Expected two distinct pending L2 records: {queue.get_pending()}'
        )

    @pytest.mark.asyncio
    async def test_infra_issue_repeat_is_the_one_path_that_folds(self, tmp_path: Path):
        """(c) The narrow truth the old claim over-generalised from.

        Pins the positive case too, so "only infra_issue folds" is asserted
        from both sides rather than inferred from two negatives.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue)

        first = await _blocker(server, **_COMMON_KWARGS)
        second = await _blocker(server, **_COMMON_KWARGS)

        assert second['status'] == 'dedup_skipped', (
            f'infra_issue inside the window is the one category the stock '
            f'DedupeConfig folds: {second}'
        )
        assert second['parent_id'] == first['id'], f'Folded into a stranger: {second}'
        assert len(self._pending_ids(queue)) == 1, (
            f'A folded repeat must not leave a second pending record: '
            f'{queue.get_pending()}'
        )
