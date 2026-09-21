"""The WRITE paths' queue scans run OFF the event loop (task 5648).

WHY they must — this server has no process of its own and shares the
ORCHESTRATOR's loop — and, just as importantly, WHY the flock'd writes beside
them deliberately do NOT hop, is stated ONCE beside the production code it
justifies: the rationale block above
``escalation/src/escalation/dedupe.py::submit_or_dedupe_off_loop``.  Task
4391's read-path sibling ``test_pending_scan_off_loop.py`` points at its own
block the same way; restating either here would give the argument a second
home in a file nobody edits alongside it.

The fact that IS local to this file: only the READ halves hop.  Every probe
below therefore asserts it FIRED before asserting which thread it ran on — a
scan that never happened is a stubbed-out scan, not a passing hop, and the two
are indistinguishable from the thread set alone.

Neither pattern here is a TIMING claim, for the reason 4391 gives: racing a
heartbeat against a scan makes the verdict depend on wall-clock scheduling on
a loaded host.  ``asyncio.to_thread`` always runs its callable on an executor
worker, so a recorded thread id either equals the loop thread's or it does
not; and a happens-before ORDER is a fact rather than a duration.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest
from _filing_tools import call_blocker, call_info
from _pending_tool_fixtures import _file

from escalation.models import Escalation
from escalation.queue import EscalationQueue
from escalation.server import create_server

# The one category the stock ``DedupeConfig`` folds, and therefore the only
# one whose filings reach ``find_dedupe_parent`` at all.  Filing anything else
# would short-circuit in pure memory and the scan under test would never run.
DEDUPED_CATEGORY = 'infra_issue'


class _ThreadProbeQueue(EscalationQueue):
    """A REAL queue that records which thread each probed call ran on.

    ONE probe keyed by method name rather than a subclass per test: the suites
    below differ in WHICH read they care about, not in how a thread id is
    recorded, so a class apiece would be three copies of one mechanism.

    Delegates through ``super()`` and is deliberately NOT a mock — the shape
    ``test_pins_recovery_annotation.py::_CountingQueue`` established and
    ``test_pending_scan_off_loop.py`` inherited.  Every probed call still does
    the real glob and JSON parse, so no test here can pass because the scan it
    was measuring had been stubbed out from under it.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.threads: dict[str, list[int]] = {}

    def _record(self, method: str) -> None:
        self.threads.setdefault(method, []).append(threading.get_ident())

    def get_pending(self):
        self._record('get_pending')
        return super().get_pending()

    def _recover_seq_from_disk(self, key):
        # The method that actually SCANS, rather than ``make_id`` around it:
        # ``make_id`` also fires on the counter-present path, where there is no
        # scan to hop, so probing it would let this test pass for the wrong
        # reason.  Reaching a private name is the price of naming the branch
        # precisely, and ``make_id``'s own docstring is what declares it.
        self._record('_recover_seq_from_disk')
        return super()._recover_seq_from_disk(key)


def _assert_off_loop(queue: _ThreadProbeQueue, method: str) -> None:
    """Assert *method* ran, and never on the thread calling this.

    The caller is a test body, which runs ON the event loop, so its own id is
    exactly the one the scan must not have used.  Firing is checked first
    because an absent probe and a hopped probe produce the same empty thread
    set, and only one of them is the thing under test.
    """
    observed = queue.threads.get(method, [])
    assert observed, f'the {method} probe recorded no call at all'
    loop_thread = threading.get_ident()
    assert loop_thread not in observed, (
        f'{method} ran ON the event-loop thread {loop_thread}; '
        f'observed threads: {observed}'
    )


async def _await_event(event: threading.Event) -> None:
    """Wait for a worker thread's ``threading.Event`` without blocking the loop."""
    while not event.is_set():
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
class TestDedupeScanThread:
    """The dedupe parent scan on the two agent filing tools.

    Parametrized over both tools rather than split: they differ only in which
    wrapper reaches the same ``_submit_or_dedupe``, and pinning them together
    is what stops a fix landing on one and reading as if the two filing paths
    had different concurrency semantics.
    """

    @pytest.mark.parametrize('call_tool', [call_blocker, call_info], ids=['blocker', 'info'])
    async def test_dedupe_scan_does_not_run_on_the_loop_thread(self, tmp_path, call_tool):
        queue = _ThreadProbeQueue(tmp_path / 'esc')
        server = create_server(queue, startup_sweep=False)
        kwargs = {
            'task_id': '5648',
            'agent_role': 'implementer',
            'category': DEDUPED_CATEGORY,
            'summary': 'falkordb refused the connection on port 6379',
        }

        first = await call_tool(server, **kwargs)
        second = await call_tool(server, **kwargs)

        _assert_off_loop(queue, 'get_pending')
        # And the answer is still right: the second filing folded into the
        # first, which is what proves the hopped scan really looked.
        assert first['status'] == 'queued'
        assert second['status'] == 'dedup_skipped', second
        assert second['parent_id'] == first['id']


class _RendezvousQueue(EscalationQueue):
    """A REAL queue that parks on ENTRY to ``get_pending``, so the loop can act.

    The park BRACKETS the real scan rather than interposing inside it — record
    ``'scan-start'``, wait, record ``'scan-end'``, and only then delegate to
    ``super()``.  So what the loop does while parked happens-before the glob,
    not during it.  That is exactly enough to pin the ORDER this test exists
    for, and deliberately NOT a claim about interleaving WITHIN the directory
    read (``test_queue.py::TestScanSurvivesRecordArchivedMidScan`` owns that).

    The release wait is bounded at 2.0 s, following the same reasoning as
    ``test_pending_scan_off_loop.py::_RendezvousQueue``: under an INLINE
    implementation nothing on the loop can ever set the release event, so the
    wait must time out rather than hang — which turns the inline case into a
    clean order-assertion failure in ~2 s instead of a test that sits until
    this suite's per-test cap (see escalation/pyproject.toml).
    """

    RELEASE_TIMEOUT = 2.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.events: list[str] = []
        self.scan_entered = threading.Event()
        self.release = threading.Event()

    def get_pending(self):
        self.events.append('scan-start')
        self.scan_entered.set()
        self.release.wait(timeout=self.RELEASE_TIMEOUT)
        self.events.append('scan-end')
        return super().get_pending()


@pytest.mark.asyncio
class TestLoopStaysLiveDuringDedupeScan:
    async def test_loop_runs_and_can_file_while_the_dedupe_scan_is_in_flight(self, tmp_path):
        """The same property stated behaviourally, and what a submit gets to do.

        Asserting an ORDER makes this a happens-before fact rather than a
        duration, so it does not depend on how loaded the host is.  The record
        filed from the loop thread lands between the tool ENTERING the dedupe
        scan and the queue read beginning — a submit concurrent with a filing,
        which is the interleaving the hop introduces and which the TOCTOU
        guard in ``attach_or_submit`` already absorbs.  The check that it was
        harmless is that both records are whole on disk afterwards.
        """
        queue = _RendezvousQueue(tmp_path / 'esc')
        server = create_server(queue, startup_sweep=False)

        filing = asyncio.create_task(call_blocker(
            server,
            task_id='5648',
            agent_role='implementer',
            category=DEDUPED_CATEGORY,
            summary='falkordb refused the connection on port 6379',
        ))
        try:
            # Polled rather than awaited directly — a threading.Event is not
            # awaitable — and bounded, so a scan that never enters fails here
            # instead of spinning to the per-test cap.  Under an INLINE
            # implementation this line is not reached until the scan has
            # already finished, which is what the order assertion below
            # catches.
            await asyncio.wait_for(_await_event(queue.scan_entered), 3.0)
            queue.events.append('loop-alive')
            # The concurrent mutation, from the loop thread, while it parks.
            # Its default category is outside the dedupe gate, so it cannot
            # become a fold target for the filing still in flight and the two
            # records stay independent.
            concurrent = _file(queue, '5649', level=0)
        finally:
            queue.release.set()
        filed = await filing

        assert queue.events == ['scan-start', 'loop-alive', 'scan-end'], (
            f'the loop did not run while the dedupe scan was in flight: {queue.events}'
        )
        # Both records are well-formed: the submit that landed while the tool
        # was parked produced no partial or malformed record, and the filing it
        # raced was neither dropped nor folded into it.
        assert filed['status'] == 'queued', filed
        on_disk = {
            path.stem: Escalation.from_json(path.read_text())
            for path in queue.queue_dir.glob('esc-*.json')
        }
        assert set(on_disk) == {filed['id'], concurrent.id}, sorted(on_disk)
        assert on_disk[filed['id']].category == DEDUPED_CATEGORY
        assert on_disk[concurrent.id].summary == concurrent.summary


# The id-NAMESPACE key every mint test below shares.  ``make_id`` names the
# counter ``esc-{key}.seq`` and the id stem ``esc-{key}-{n}`` from it, and all
# three tools derive it from their ``task_id`` argument.
MINT_KEY = '5648'


async def _promote(server, **kwargs):
    """Invoke ``promote_to_l2`` on *server* and return its response dict.

    Local to this module ON PURPOSE.  ``_filing_tools`` shares the call SHAPE
    of the two async FILING tools and says explicitly that the KWARGS are not
    shared because each suite's carry local meaning — which is true here twice
    over: the mint suite needs a ``task_id`` that is a fresh id-namespace key,
    and the promote-scan suite needs ``severity`` OMITTED so the member
    derivation it is measuring actually runs.  Only one other module owns such
    a helper (``test_server.py::_promote_to_l2``), so this is the second, which
    is below the third-copy threshold that moved the filing dance out.
    """
    tool = await server.get_tool('promote_to_l2')
    return await tool.fn(**kwargs)


# The three minting tools under one signature, so the parametrized test below
# dispatches over them without a per-tool branch.  Each carries its OWN kwargs
# — they are not a shared constant, because only `promote_to_l2` has members to
# name and only the filing tools have a category.  The two that ignore
# *member_id* still take it: a uniform slot is cheaper to read than a table of
# mismatched arities.
async def _mint_via_blocker(server, member_id: str):
    return await call_blocker(
        server,
        task_id=MINT_KEY,
        agent_role='implementer',
        category=DEDUPED_CATEGORY,
        summary='falkordb refused the connection on port 6379',
    )


async def _mint_via_info(server, member_id: str):
    return await call_info(
        server,
        task_id=MINT_KEY,
        agent_role='implementer',
        category=DEDUPED_CATEGORY,
        summary='falkordb refused the connection on port 6379',
    )


async def _mint_via_promote(server, member_id: str):
    return await _promote(
        server,
        task_id=MINT_KEY,
        agent_role='escalation-watcher-auto',
        member_ids=[member_id],
        root_cause='falkordb unreachable',
        evidence='the member L1 names a refused connection',
        options=['A: restart falkordb', 'B: fail the lane'],
        summary='one cluster, one cause',
    )


def _seed_for_recovery_branch(queue: EscalationQueue) -> str:
    """Leave *queue* in the state whose next mint must scan, and return a member id.

    Two properties, both load-bearing:

    - An ARCHIVED record under ``MINT_KEY``, so the archive half of
      ``_recover_seq_from_disk``'s scan has real work rather than rglob'ing an
      empty subtree.  Produced by a real submit+resolve, not a hand-placed
      file, so the archive layout is the queue's own.
    - NO ``esc-{MINT_KEY}.seq`` counter, which is what sends the next mint down
      the absent-counter branch.  Every id here is explicit precisely so no
      counter is ever written; ``make_id``'s docstring records that this branch
      is not only a repair path — every brand-new key's FIRST mint takes it.

    The member L1 is filed under a DIFFERENT key, so it is invisible to the
    scoped ``esc-{MINT_KEY}-*`` glob and cannot perturb the recovered sequence.
    """
    def _esc(esc_id: str, task_id: str, summary: str) -> Escalation:
        return Escalation(
            id=esc_id, task_id=task_id, agent_role='implementer',
            severity='info', category=DEDUPED_CATEGORY, summary=summary,
        )

    archived = _esc(f'esc-{MINT_KEY}-1', MINT_KEY, 'an older filing under this key')
    queue.submit(archived)
    queue.resolve(archived.id, resolution='archived so the recovery scan has real work')
    assert not (queue.queue_dir / f'esc-{MINT_KEY}.seq').exists()
    assert (queue.queue_dir / 'archive').exists(), 'the seed did not reach the archive'

    member = _esc('esc-member-1', 'member-task', 'a member L1 for the promote arm')
    queue.submit(member)
    return member.id


@pytest.mark.asyncio
class TestMintScanThread:
    """``make_id``'s recovery scan, on all three tools that mint on a write path.

    The site the task description characterised as an flock plus an atomic
    write + fsync also GLOBS the queue root and RGLOBS the entire archive on
    the absent-counter branch — the branch ``make_id``'s own docstring says
    every brand-new key's first mint takes.  Unlike the dedupe scan, which the
    stock ``DedupeConfig`` restricts to one category, this one fires for every
    filing.

    Parametrized over all three tools because a hop applied to one and not the
    others would read as if they had different concurrency semantics.
    """

    @pytest.mark.parametrize(
        'mint',
        [_mint_via_blocker, _mint_via_info, _mint_via_promote],
        ids=['blocker', 'info', 'promote'],
    )
    async def test_mint_recovery_scan_does_not_run_on_the_loop_thread(self, tmp_path, mint):
        queue = _ThreadProbeQueue(tmp_path / 'esc')
        member_id = _seed_for_recovery_branch(queue)
        server = create_server(queue, startup_sweep=False)
        # The seeding above ran on this thread and is not the subject; only
        # what the TOOL does is measured.
        queue.threads.clear()

        result = await mint(server, member_id)

        _assert_off_loop(queue, '_recover_seq_from_disk')
        # The recovered sequence is strictly greater than the archived record's,
        # which is what proves the scan really looked rather than being skipped.
        assert result['id'] == f'esc-{MINT_KEY}-2', result


@pytest.mark.asyncio
class TestConcurrentMintsUnderOneKey:
    """The invariant the hop must not break, written before the hop lands.

    ``escalation_id_lock`` takes ``fcntl.flock``, which is per-open-file-
    description, so two WORKER THREADS opening the counter sidecar separately
    serialise on it exactly as two OS processes do — the contract ``make_id``'s
    docstring already states for concurrent minters.  Hopping the mint adds
    in-process threads to a set that lock was already built to cover; this
    test is what says so out loud.
    """

    async def test_two_concurrent_filings_under_one_key_get_distinct_ids(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue, startup_sweep=False)

        def _file_one(summary):
            # category OUTSIDE the dedupe gate: the mint is the subject here,
            # so the two filings are kept independent rather than letting one
            # fold into the other and leave a single record to inspect.
            return call_blocker(
                server,
                task_id=MINT_KEY,
                agent_role='implementer',
                category='scope_violation',
                summary=summary,
            )

        first, second = await asyncio.gather(
            _file_one('first concurrent filing'),
            _file_one('second concurrent filing'),
        )

        assert first['id'] != second['id'], (first, second)
        on_disk = {
            path.stem: Escalation.from_json(path.read_text())
            for path in queue.queue_dir.glob('esc-*.json')
        }
        assert set(on_disk) == {first['id'], second['id']}, sorted(on_disk)
        assert {rec.summary for rec in on_disk.values()} == {
            'first concurrent filing', 'second concurrent filing',
        }
