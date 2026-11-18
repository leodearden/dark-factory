"""Self-tests for the verify-merge boundary harness in _verify_boundary_harness."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from _verify_boundary_ceilings import ROW5_HOLDER_TEARDOWN_CEILING_SECS
from _verify_boundary_harness import (
    TaggedPopen,
    assert_pre_kill_tree_exited,
    kill_holder_tree,
    read_direct_children,
    read_proc_state,
    sleeper_spec,
    snapshot_pre_kill_tree,
    sweep_tagged,
    tagged_pids,
    wait_for_marker_stable,
    wait_for_pgid_file,
    wait_pids_exited,
    wait_subtree_live,
)
from shared.proc_group import StatFields

from orchestrator.verify_cancel import collect_descendants, read_ppid_map

# ---------------------------------------------------------------------------
# Task 2819 -- deterministic unit coverage for wait_for_marker_stable, the
# create+utimensat settle helper that de-flakes the Row 5 marker-retention
# assertion in test_laptop_warm_verify_boundary.py.  Both tests inject a
# private mtime-reader seam
# (_read_mtime_ns) so there is ZERO real timing -- a real-timing settle test
# would itself be load-sensitive (fixing a flake with a flake).
# ---------------------------------------------------------------------------


def test_wait_for_marker_stable_returns_settled_not_intermediate(tmp_path):
    """wait_for_marker_stable returns the SETTLED mtime, not the intermediate one.

    Models GNU ``touch``'s two-syscall create->utimensat drift (task 2819):
    the injected reader yields the intermediate create-time mtime a few
    times, then the final settled mtime for the remainder of the scripted
    sequence.  Uses a real tmp_path marker so the existence gate
    (wait_for_marker) passes instantly, and interval=0 so the settle loop
    itself burns no real wall-clock time.
    """
    marker = tmp_path / 'warm.marker'
    marker.touch()

    sequence = [100, 100, 100, 150, 150, 150, 150, 150, 150, 150, 150, 150]
    calls = iter(sequence)

    def fake_read_mtime_ns(_path):
        return next(calls)

    result = wait_for_marker_stable(
        marker, interval=0, stable_reads=6, _read_mtime_ns=fake_read_mtime_ns,
    )

    assert result == 150, (
        f'expected the settled mtime (150), not the create-time intermediate '
        f'(100); got {result}'
    )


def test_wait_for_marker_stable_raises_when_never_settles(tmp_path):
    """wait_for_marker_stable raises AssertionError if the mtime never quiets down.

    An always-changing reader can never satisfy the consecutive-unchanged
    window, so with a tiny timeout this must raise promptly rather than
    hang or silently return an unsettled value.
    """
    marker = tmp_path / 'warm.marker'
    marker.touch()

    call_count = [0]

    def fake_read_mtime_ns(_path):
        call_count[0] += 1
        return call_count[0]

    with pytest.raises(AssertionError):
        wait_for_marker_stable(
            marker, timeout=0.05, interval=0, _read_mtime_ns=fake_read_mtime_ns,
        )


# ---------------------------------------------------------------------------
# Task 4014 -- deterministic unit coverage for wait_subtree_live's descendant
# discovery poll (the "no descendant appeared within 20.0s" flake).  Every
# test here injects the two private seams (_probe_children / _ppid_map) and
# runs with interval=0, so the POLL LOOP itself has ZERO real timing -- the
# same task-2819 precedent as the wait_for_marker_stable tests above.
# Fixing a flake with a flake is the failure mode to avoid, and the 3689
# steward failed to reproduce this one in 19 of 19 attempts (including
# synthetic pressure at load 100 / ~8000 processes), so a load-based test
# would be non-deterministic in BOTH directions.  The efficiency property
# that actually prevents sampling-rate collapse is therefore pinned
# STRUCTURALLY, by call count, which is exact at any load.
#
# Deliberate, narrow exceptions to "zero real subprocess" (task 4312,
# joining test_read_direct_children_sees_a_real_fork_including_off_main_thread
# and task 5945's
# test_read_direct_children_skips_a_sibling_thread_that_exits_mid_probe
# below): the two timeout-diagnostic tests spawn a real `proc` because the
# diagnostic message's CONTENT -- the leader's actual returncode and, where
# applicable, its stderr tail -- is exactly what is under test.  Each test
# covers BOTH discovery waiters (wait_subtree_live and wait_for_pgid_file),
# since they share one exit diagnostic.  The poll loop each one drives still
# runs zero ticks (timeout=0), so timing stays zero even there.
# ---------------------------------------------------------------------------


def test_wait_subtree_live_gates_the_proc_walk_on_a_cheap_probe():
    """The expensive full /proc rescan runs only once the cheap probe goes positive.

    This is the load-bearing efficiency property of the discovery poll.
    Measured on this box: ``read_ppid_map()`` costs 49.67 ms at 917 live
    processes, while one ``/proc/<pid>/task/<tid>/children`` read costs
    0.0183 ms -- ~2700x.  Against the 50 ms poll interval the full rescan
    already DOUBLES the effective sampling period at idle, and the poller
    burns ~917 file reads per tick competing for CPU with the very leader it
    is waiting to see fork; under a full-suite storm (~8000 procs) the scan
    is ~400ms+ and the sampling period collapses by ~10x.  So the cost is
    worst exactly when the test needs the samples most.

    The call-count assertion below pins "the sampling rate cannot collapse
    under a large process table" as a STRUCTURAL property rather than a
    timing hope: 26 ticks may cost at most ONE expensive walk.
    """
    probe_calls = [0]
    ppid_map_calls = [0]

    def fake_probe(_pid):
        # 25 cheap negatives (leader live, hasn't forked yet), then a child.
        probe_calls[0] += 1
        return {5678} if probe_calls[0] > 25 else set()

    def fake_ppid_map():
        ppid_map_calls[0] += 1
        return {5678: 1234}

    result = wait_subtree_live(
        1234, interval=0, _probe_children=fake_probe, _ppid_map=fake_ppid_map,
    )

    assert result == {5678}, (
        f'expected the production walker\'s descendant set {{5678}}; got {result}'
    )
    assert probe_calls[0] == 26, (
        f'expected the cheap probe on every one of the 26 ticks; '
        f'got {probe_calls[0]} calls'
    )
    assert ppid_map_calls[0] <= 1, (
        f'the expensive read_ppid_map() walk ran {ppid_map_calls[0]} times across '
        f'26 ticks -- it must run ONLY on the tick the cheap probe goes positive, '
        f'or the sampling rate collapses exactly when the process table is largest'
    )


def test_wait_subtree_live_skips_a_vanished_transient_and_returns_the_durable_set():
    """A transient child that exits between probe and walk is skipped, not returned.

    The first child the CLI forks for a not-yet-materialised worktree is a
    SHORT-LIVED ``git`` subprocess (``git worktree add`` / ``git reset`` /
    ``git clean`` inside ``acquire_host_verify_worktree``), not the eventual
    build shell.  Scripted here deterministically: the probe sees pid 111 on
    tick 1, nothing on ticks 2-3, then the durable 300s sleeper (222) from
    tick 4; the paired walk comes back EMPTY on its first call because 111
    had already exited by the time the rescan ran.

    The helper must keep polling rather than hand back that empty set.  Two
    reasons: it makes the return contract "a NON-EMPTY descendant set"
    total, and it biases what a caller observes toward the durable sleeper --
    a caller would otherwise mistake a transient ``git`` child for the
    durable build.
    """
    probe_calls = [0]
    ppid_map_calls = [0]

    def fake_probe(_pid):
        probe_calls[0] += 1
        if probe_calls[0] == 1:
            return {111}  # transient `git worktree add`
        if probe_calls[0] <= 3:
            return set()  # transient has exited; sleeper not forked yet
        return {222}  # the durable 300s sleeper

    def fake_ppid_map():
        ppid_map_calls[0] += 1
        # First walk races the transient's exit and sees nothing.
        return {} if ppid_map_calls[0] == 1 else {222: 1234}

    result = wait_subtree_live(
        1234, interval=0, _probe_children=fake_probe, _ppid_map=fake_ppid_map,
    )

    assert result == {222}, (
        f'expected the durable sleeper {{222}} -- never the empty set a walk '
        f'racing the transient git child returns; got {result}'
    )


def test_wait_subtree_live_falls_back_to_the_full_walk_when_children_unreadable():
    """A probe that returns None degrades to today's exact per-tick full walk.

    ``/proc/<pid>/task/<tid>/children`` requires CONFIG_PROC_CHILDREN, so the
    probe is TRI-STATE and its third state is load-bearing:

    * ``set()``  -- leader live, no children yet (cheap negative: skip the walk)
    * ``{...}``  -- worth confirming with the production walker
    * ``None``   -- CANNOT probe (no ``children`` file, or the ``/proc/<pid>``
      entry is gone)

    ``None`` must fall through to the full ``collect_descendants(pgid,
    read_ppid_map())`` walk for that tick.  Conflating it with the cheap
    negative would make the helper spin until timeout on a kernel where the
    probe can never go positive -- trading a flake for a hard, universal
    failure on that platform.

    The second half pins the vanishing-pid case that makes ``None``
    reachable at all on a normal kernel: the leader can exit mid-poll, and
    that must yield ``None``, never turn this helper's timeout diagnostic
    into an unhandled OSError.  A pid above ``pid_max`` cannot exist, so it
    stands in deterministically with no process to spawn or reap.
    """
    ppid_map_calls = [0]

    def fake_probe(_pid):
        return None  # kernel without CONFIG_PROC_CHILDREN

    def fake_ppid_map():
        ppid_map_calls[0] += 1
        return {5678: 1234} if ppid_map_calls[0] > 3 else {}

    result = wait_subtree_live(
        1234, interval=0, _probe_children=fake_probe, _ppid_map=fake_ppid_map,
    )

    assert result == {5678}, (
        f'a None (unprobeable) tick must fall back to the full walk, exactly as '
        f'this helper behaved before task 4014; got {result}'
    )

    pid_max = int(Path('/proc/sys/kernel/pid_max').read_text().strip())
    assert read_direct_children(pid_max + 1) is None, (
        'a pid that does not exist must probe as None (cannot probe), not raise -- '
        'a leader exiting mid-poll must not turn the timeout path into an OSError'
    )


def _time_out_subtree_live(proc: subprocess.Popen, tmp_path: Path) -> None:
    """Drive :func:`wait_subtree_live` straight into its timeout branch for *proc*."""
    wait_subtree_live(
        1234,
        proc=proc,
        proc_label='leader',
        timeout=0,
        interval=0,
        _probe_children=lambda _pid: set(),
        _ppid_map=lambda: {},
    )


def _time_out_build_pgid_file(proc: subprocess.Popen, tmp_path: Path) -> None:
    """Drive :func:`wait_for_pgid_file` straight into its timeout branch for *proc*."""
    wait_for_pgid_file(
        tmp_path / 'never-written.pgid', proc=proc, proc_label='leader', timeout=0,
    )


_DISCOVERY_WAIT_TIMEOUTS = pytest.mark.parametrize(
    'time_out',
    [_time_out_subtree_live, _time_out_build_pgid_file],
    ids=['wait_subtree_live', 'wait_for_pgid_file'],
)


@_DISCOVERY_WAIT_TIMEOUTS
def test_discovery_wait_timeout_reports_leader_returncode_and_stderr_tail(time_out, tmp_path):
    """The timeout diagnostic names the LEADER's real rc and stderr tail (291a75a919).

    291a75a919 ("De-flake alpha: make wait_subtree_live's timeout path
    self-diagnosing") added the ``proc``/``proc_label`` reporting now owned
    by :func:`describe_proc_exit`, which both discovery waiters append on
    timeout, so a watchdog self-kill (rc == 1, no ``Error:`` line), an
    exception exit (rc == 1 WITH an ``Error:`` line), and a merely slow
    leader (rc is None) are distinguishable from the failure message alone
    -- exactly the signal an incident needs. It shipped with no test (task
    4312).

    Deterministic per the task-4014 precedent above: each waiter runs with
    ``timeout=0`` (and :func:`wait_subtree_live` with seams that report no
    children), so the poll body never executes a single tick -- the timeout
    branch fires immediately, with zero real timing.  What IS real is *proc*: an actually-spawned,
    already-``wait()``-ed subprocess with a KNOWN returncode (3) and a
    KNOWN stderr payload (``SENTINEL-XYZ``), so this pins the diagnostic's
    CONTENT rather than merely the presence of the words "rc"/"stderr" --
    the vacuous-guard trap this repo has hit before: a message reporting
    nothing (e.g. ``rc=None; stderr tail:\n``) would also contain those
    labels.
    """
    proc = subprocess.Popen(
        [sys.executable, '-c', 'import sys; sys.stderr.write("SENTINEL-XYZ\\n"); sys.exit(3)'],
        stderr=subprocess.PIPE,
    )
    try:
        proc.wait()
        assert proc.returncode == 3  # sanity: the known rc actually landed

        with pytest.raises(AssertionError) as exc_info:
            time_out(proc, tmp_path)

        message = str(exc_info.value)
        assert 'leader rc=3' in message, (
            f'expected the leader\'s actual returncode (3), reported under its '
            f'proc_label, in the timeout message; got: {message!r}'
        )
        assert 'SENTINEL-XYZ' in message, (
            f'expected the leader\'s actual stderr tail content in the timeout '
            f'message; got: {message!r}'
        )
    finally:
        if proc.stderr is not None:
            with contextlib.suppress(OSError):
                proc.stderr.close()


@_DISCOVERY_WAIT_TIMEOUTS
def test_discovery_wait_timeout_reports_rc_none_and_omits_stderr_for_a_live_leader(
    time_out, tmp_path,
):
    """The rc=None ("merely slow leader") taxonomy branch and its anti-hang guard (291a75a919).

    Sibling of
    :func:`test_discovery_wait_timeout_reports_leader_returncode_and_stderr_tail`
    above, covering the two properties that rc=3 case cannot reach (task 4312
    review):

    * the ``rc is None`` taxonomy branch itself -- a leader that is merely
      slow (still running when the poll gives up) is the case an incident is
      most likely to actually hit, and it was entirely unpinned.
    * the load-bearing guard :func:`describe_proc_exit`'s docstring calls out:
      stderr is read ONLY when ``proc.poll()`` is not ``None``, precisely
      because a LIVE leader's pipe write end can still be held open by an
      inherited-fd helper it spawned, and a buffered read on that pipe would
      block until EOF -- hanging this helper, and the rest of the suite,
      rather than raising.  A regression that swapped the guarded
      ``select()``-bounded raw read for an unconditional ``proc.stderr.read()``
      would still pass every other test in this module while reintroducing
      exactly that hang.

    *proc* here is a real, still-running subprocess (``time.sleep(300)``, well
    outside this test's own runtime) so ``proc.poll()`` is genuinely ``None``
    -- checking that costs one non-blocking ``waitpid(WNOHANG)`` syscall, so
    this stays zero real TIMING even though *proc* itself is real, same as
    its rc=3 sibling.  ``stderr=PIPE`` is set (mirroring real call sites) so a
    guard regression would hang on the open write end rather than raise
    ``AttributeError`` on a ``None`` pipe, which would make this test pass for
    the wrong reason.
    """
    proc = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(300)'],
        stderr=subprocess.PIPE,
    )
    try:
        with pytest.raises(AssertionError) as exc_info:
            time_out(proc, tmp_path)

        message = str(exc_info.value)
        assert 'leader rc=None' in message, (
            f'expected the merely-slow-leader taxonomy branch (rc=None, '
            f'reported under its proc_label) in the timeout message; got: '
            f'{message!r}'
        )
        assert 'stderr tail' not in message, (
            f'stderr must be read ONLY when proc.poll() is not None -- seeing '
            f'"stderr tail" here means the anti-hang poll()-gate regressed to '
            f'an unconditional read, which would hang on a live leader whose '
            f'pipe write end is still open; got: {message!r}'
        )
    finally:
        proc.kill()
        proc.wait()
        if proc.stderr is not None:
            with contextlib.suppress(OSError):
                proc.stderr.close()


#: A child that stays alive but is never waited on -- killed by the test that
#: spawns it.  ``sys.executable`` rather than ``sleep`` so nothing depends on
#: PATH; the probe only needs the fork, not a finished exec.
_DURABLE_CHILD_ARGV = [sys.executable, '-c', 'import time; time.sleep(30)']


def test_read_direct_children_sees_a_real_fork_including_off_main_thread():
    """The real probe's POSITIVE path, against the real kernel interface.

    The three tests above all inject ``_probe_children``, so they pin
    :func:`wait_subtree_live`'s gate LOGIC and never execute
    :func:`read_direct_children` itself; the only unmocked assertion on it is
    the negative (``None``) case immediately above.  A regression in the
    positive path -- a wrong ``/proc`` path segment, iterating only the main
    tid, a broken ``int(token)`` parse -- would return ``set()`` for a leader
    that HAS forked.  Every deterministic test in this module would stay green
    while all five real rows failed with "no descendant appeared within Ns":
    the exact flake task 4014 exists to remove, made permanent.

    Two halves, both with ZERO real timing -- no sleeps, no polling.
    ``Popen`` has already forked when it returns and the kernel lists the child
    from that moment, so there is nothing to wait for (the ``Event`` waits in
    (b) are handoff barriers, not timing assertions).

    (a) A child forked from THIS thread must be listed.  Pins path
        construction and token parsing.

    (b) A child forked from a SECONDARY thread, probed while that thread is
        still alive, must also be listed.  This is the half that pins the
        per-thread ``task/*/`` iteration the helper's docstring justifies:
        ``children`` is a PER-THREAD file, so a bare
        ``/proc/<pid>/task/<pid>/children`` read would MISS this child.  The
        forking thread is held alive until after the probe precisely because a
        thread that EXITS has its children re-parented onto the group leader,
        which would let a main-tid-only implementation pass and make this half
        vacuous.
    """
    child = subprocess.Popen(_DURABLE_CHILD_ARGV)
    try:
        probed = read_direct_children(os.getpid())
        assert probed is not None, (
            'probing this live test process must not report "cannot probe" -- '
            'either the /proc path is built wrong or this kernel lacks '
            'CONFIG_PROC_CHILDREN (in which case every row silently runs on the '
            'full-walk fallback and task 4014 buys nothing)'
        )
        assert child.pid in probed, (
            f'just-forked direct child {child.pid} is missing from the probe '
            f'{sorted(probed)} -- a probe that cannot see a fork returns the cheap '
            f'NEGATIVE forever, so wait_subtree_live never spends the confirming '
            f'walk and every real row fails with "no descendant appeared"'
        )
    finally:
        child.kill()
        child.wait()

    forked: dict[str, subprocess.Popen] = {}
    spawned = threading.Event()
    release = threading.Event()

    def fork_off_main_thread() -> None:
        forked['proc'] = subprocess.Popen(_DURABLE_CHILD_ARGV)
        spawned.set()
        release.wait(timeout=30.0)  # stay alive so the child stays MINE

    thread = threading.Thread(target=fork_off_main_thread, daemon=True)
    thread.start()
    try:
        assert spawned.wait(timeout=30.0), 'the helper thread never forked its child'
        off_main = forked['proc']
        probed = read_direct_children(os.getpid())
        assert probed is not None and off_main.pid in probed, (
            f'child {off_main.pid}, forked from a still-live secondary thread, is '
            f'missing from the probe {probed if probed is None else sorted(probed)} '
            f'-- children is a PER-THREAD file, so this is what a regression to a '
            f'single /proc/<pid>/task/<pid>/children read looks like'
        )
    finally:
        release.set()
        thread.join(timeout=30.0)
        off_main = forked.get('proc')
        if off_main is not None:
            off_main.kill()
            off_main.wait()


def test_read_direct_children_skips_a_sibling_thread_that_exits_mid_probe():
    """A sibling thread exiting mid-probe must not make a LIVE process read as "cannot probe".

    The probe lists ``/proc/<pid>/task`` and then reads each tid's
    ``children``; a thread that exits in between has no ``children`` left to
    read.  That is routine in any threaded process -- pytest itself brackets
    every test with faulthandler / pytest-timeout watchdog threads -- and it
    says nothing about the threads still alive.  Task 5945: the recorded
    flake of the test above was exactly ``assert None is not None`` on a live
    xdist worker.

    Two churner threads keep starting and joining no-op threads while this
    thread probes.  The probe loop is bounded by a CONDITION -- at least
    ``required_exits`` sibling-thread exits observed while probing -- not by
    a probe count or a clock, so its power does not depend on scheduler
    speed.  The ``Event`` waits are handoff barriers, not timing assertions.

    No single exit is guaranteed to land between the listing and a
    ``children`` read, so this catches the pre-fix collapse only
    probabilistically; the deterministic regression guard is
    :func:`test_read_direct_children_tri_state_over_a_fake_task_listing`.  This
    test confirms, on the real kernel, the behaviour that fake listing models:
    an exited thread's tid directory is gone, so its failed read is skipped.
    """
    required_exits = 200
    stop = threading.Event()
    exit_counts = [0, 0]
    ready = [threading.Event() for _ in exit_counts]

    def churn(slot: int) -> None:
        while not stop.is_set():
            sibling = threading.Thread(target=lambda: None)
            try:
                sibling.start()
            except RuntimeError:
                return  # "can't start new thread" -- fails the exit-count assert
            sibling.join()
            exit_counts[slot] += 1
            ready[slot].set()

    churners = [threading.Thread(target=churn, args=(slot,)) for slot in range(len(exit_counts))]
    child = subprocess.Popen(_DURABLE_CHILD_ARGV)
    probes = none_count = missing_count = 0
    try:
        for churner in churners:
            churner.start()
        for event in ready:
            assert event.wait(timeout=30.0), 'churner never completed a thread lifecycle'
        exits_before = sum(exit_counts)
        while sum(exit_counts) - exits_before < required_exits and any(
            churner.is_alive() for churner in churners
        ):
            probed = read_direct_children(os.getpid())
            probes += 1
            if probed is None:
                none_count += 1
            elif child.pid not in probed:
                missing_count += 1
        exits_during = sum(exit_counts) - exits_before
    finally:
        stop.set()
        for churner in churners:
            churner.join(timeout=30.0)
        child.kill()
        child.wait()

    assert exits_during >= required_exits, (
        f'only {exits_during} sibling-thread exits (needed {required_exits}) '
        f'overlapped {probes} probes -- the churners stopped early, so the churn '
        f'never overlapped the probes and this test proved nothing'
    )
    assert none_count == 0, (
        f'{none_count} of {probes} probes of this LIVE process reported "cannot '
        f'probe" across {exits_during} sibling-thread exits -- a thread that exits '
        f'between the task/ listing and its children read must be skipped, not '
        f'collapse the whole probe to None'
    )
    assert missing_count == 0, (
        f'{missing_count} of {probes} answered probes missed direct child '
        f'{child.pid}, forked from this thread -- it must appear in every one'
    )


def _build_fake_task_listing(
    proc_root: Path,
    pid: int,
    *,
    live: dict[int, list[int]],
    without_children_file: tuple[int, ...] = (),
    exited: tuple[int, ...] = (),
) -> None:
    task_dir = proc_root / str(pid) / 'task'
    task_dir.mkdir(parents=True)
    for tid, kids in live.items():
        (task_dir / str(tid)).mkdir()
        (task_dir / str(tid) / 'children').write_text(''.join(f'{kid} ' for kid in kids))
    for tid in without_children_file:
        (task_dir / str(tid)).mkdir()
    for tid in exited:
        # Listed, yet `children` raises FileNotFoundError and exists() is False: a thread released after the listing.
        (task_dir / str(tid)).symlink_to(proc_root / 'gone' / str(tid))


@pytest.mark.parametrize(
    ('live', 'without_children_file', 'exited', 'expected', 'consequence'),
    [
        pytest.param(
            {1234: [5678, 9012], 1240: []}, (), (1241,), {5678, 9012},
            'None would drop wait_subtree_live to the full read_ppid_map walk on '
            'every tick a sibling thread exits -- the cost task 4014 removed',
            id='exited-thread-beside-a-live-one-is-skipped',
        ),
        pytest.param(
            {1234: [5678]}, (1238,), (), None,
            'a live thread whose children cannot be read hides its children, so '
            'any answer here can be a false cheap negative that makes '
            'wait_subtree_live skip its walk every tick and spin to its timeout '
            'on a kernel without CONFIG_PROC_CHILDREN',
            id='live-thread-without-children-file-cannot-probe',
        ),
        pytest.param(
            {}, (), (1234,), None,
            'set() here is a false cheap negative for a process that exited '
            'mid-probe, which makes wait_subtree_live skip its walk instead of '
            'falling back to it',
            id='every-listed-thread-exited-cannot-probe',
        ),
    ],
)
def test_read_direct_children_tri_state_over_a_fake_task_listing(
    tmp_path, live, without_children_file, exited, expected, consequence,
):
    """The probe's tri-state on the branches this kernel cannot produce on demand."""
    pid = 4321
    _build_fake_task_listing(
        tmp_path, pid, live=live, without_children_file=without_children_file, exited=exited,
    )

    result = read_direct_children(pid, _proc_root=tmp_path)

    assert result == expected, f'expected {expected!r}, got {result!r} -- {consequence}'


def test_wait_pids_exited_counts_an_unreaped_zombie_as_exited():
    """A terminated-but-unreaped child is not a survivor, although signal 0 still answers.

    ``os.waitid(..., WNOWAIT)`` blocks until the child has exited without
    reaping it, so the zombie is produced by a condition wait, never a sleep.
    """
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])
    try:
        child.kill()
        os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
        try:
            os.kill(child.pid, 0)
        except ProcessLookupError:
            pytest.fail(
                'harness bug: an unreaped zombie must still answer signal 0, or '
                'this test is not reproducing the state a signal-0 probe counts '
                'as alive'
            )

        assert wait_pids_exited({child.pid}, timeout=0) == {}

        child.wait()
        assert wait_pids_exited({child.pid}, timeout=0) == {}
    finally:
        if child.returncode is None:
            child.kill()
            child.wait()


def test_wait_pids_exited_reports_a_live_child_as_a_survivor():
    """A live child survives the wait, parented to this process and not terminated."""
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])
    try:
        survivors = wait_pids_exited({child.pid}, timeout=0)
        assert set(survivors) == {child.pid}
        assert survivors[child.pid].ppid == os.getpid()
        assert survivors[child.pid].terminated is False
    finally:
        child.kill()
        child.wait()


def test_wait_pids_exited_decides_on_a_probe_taken_after_its_last_sleep():
    """A deadline crossed DURING the sleep still gets one fresh probe before the verdict.

    The scripted clock jumps past the deadline while the loop sleeps.  A loop
    that re-checks the deadline before re-probing returns the stale ``R``
    observation; the verdict must come from the second probe, which sees ``Z``.
    """
    reads: list[int] = []

    def read_state(pid: int) -> StatFields:
        reads.append(pid)
        state = 'R' if len(reads) == 1 else 'Z'
        return StatFields(comm='sleep', state=state, ppid=1, pgrp=pid)

    ticks = iter([0.0, 0.5])

    survivors = wait_pids_exited(
        {7}, timeout=1.0, interval=0,
        _read_state=read_state, _clock=lambda: next(ticks, 99.0),
    )

    assert survivors == {}
    assert reads == [7, 7]


def test_wait_pids_exited_reports_each_survivor_with_its_last_observed_state():
    """timeout=0 is exactly one probe; a pid with no /proc entry is not a survivor."""
    reads: list[int] = []
    asleep = StatFields(comm='sleep', state='S', ppid=4242, pgrp=4242)

    def sleeping(pid: int) -> StatFields:
        reads.append(pid)
        return asleep

    assert wait_pids_exited({7}, timeout=0, _read_state=sleeping) == {7: asleep}
    assert reads == [7]

    states: dict[int, StatFields | None] = {7: asleep, 8: None}
    assert wait_pids_exited({7, 8}, timeout=0, _read_state=states.__getitem__) == {7: asleep}


def test_sleeper_spec_build_pgid_file_names_the_blocking_session_leader(tmp_path):
    """The build pgid file names the build shell's own session, and that pid BECOMES the sleeper.

    Runs the command exactly as verify.py::_run_cmd does on its default
    (non-cgroup) path: ``/bin/bash -c`` in a new session.  Because the reported
    pid itself execs into the blocker, nothing forks after the handle is
    published, so a snapshot taken once the file exists cannot miss the sleeper.
    """
    build_pgf = tmp_path / 'build.pgid'
    cmd = sleeper_spec(30.0, build_pgid_file=build_pgf).verify_commands[0].test_command
    assert cmd is not None
    cwd = tmp_path / 'build'
    cwd.mkdir()
    proc = subprocess.Popen(['/bin/bash', '-c', cmd], cwd=cwd, start_new_session=True)
    try:
        build_pgid = wait_for_pgid_file(build_pgf, timeout=10.0)
        assert build_pgid == proc.pid, (
            f'the file must hold the build shell pid {proc.pid}; got {build_pgid}'
        )
        assert os.getpgid(proc.pid) == proc.pid, 'the build shell must lead its own pgid'

        comm_path = Path(f'/proc/{build_pgid}/comm')
        deadline = time.monotonic() + 10.0
        comm = comm_path.read_text().strip()
        while comm != 'sleep' and time.monotonic() < deadline:
            time.sleep(0.02)
            comm = comm_path.read_text().strip()
        assert comm == 'sleep', (
            f'pid {build_pgid} must exec into the blocking sleeper; comm is {comm!r}'
        )
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_pre_kill_tree_verdict_sees_a_session_escaped_build_after_its_leader_is_reaped(tmp_path):
    """A build that outlives its reaped leader fails the verdict; once killed, it passes.

    The stand-in leader leads its own session, as
    ``orchestrator/src/orchestrator/verify_cancel.py::start_own_process_group``
    makes the real one, and reproduces verify.py::_run_cmd's escape: it starts
    the :func:`sleeper_spec` build in another new session, then is SIGKILLed
    and reaped while the build runs on.  That severs the ppid chain and empties
    the leader's group, so a walk rooted at the leader and a signal-0 probe of
    its group both read the tree as gone.  The verdict judges the pids
    snapshotted while the leader was alive, so it must still name the
    surviving build.
    """
    build_pgf = tmp_path / 'build.pgid'
    sleep_secs = float(f'274.{os.getpid() % 1000:03d}')
    cmd = sleeper_spec(sleep_secs, build_pgid_file=build_pgf).verify_commands[0].test_command
    assert cmd is not None
    cwd = tmp_path / 'build'
    cwd.mkdir()
    leader = subprocess.Popen(
        [
            sys.executable, '-c',
            'import subprocess, sys, time\n'
            'subprocess.Popen(["/bin/bash", "-c", sys.argv[1]], start_new_session=True, '
            'stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n'
            'time.sleep(300)\n',
            cmd,
        ],
        cwd=cwd,
        start_new_session=True,
    )
    build = None
    try:
        build = wait_for_pgid_file(build_pgf, timeout=10.0)
        tree = snapshot_pre_kill_tree(leader.pid, build)
        assert {leader.pid, build} <= tree.pids
        assert os.getpgid(leader.pid) == leader.pid, (
            f'harness bug: leader pid={leader.pid} must lead its own group, or the '
            f'signal-0 group probe below proves nothing'
        )

        leader.kill()
        leader.wait(timeout=5)

        build_state = read_proc_state(build)
        assert build_state is not None and not build_state.terminated, (
            f'harness bug: build pid={build} is not running ({build_state}) after '
            f'its leader was reaped, so the verdict below would prove nothing'
        )
        assert build_state.ppid != leader.pid, (
            f'harness bug: build pid={build} still names the reaped leader '
            f'pid={leader.pid} as its parent -- it was never reparented'
        )
        assert build not in collect_descendants(leader.pid, read_ppid_map()), (
            f'harness bug: the ppid walk from pid={leader.pid} still reaches build '
            f'pid={build}, so this test does not isolate the severed-chain case'
        )
        with pytest.raises(ProcessLookupError):
            os.killpg(leader.pid, 0)

        with pytest.raises(AssertionError) as exc_info:
            assert_pre_kill_tree_exited(tree, timeout=0, after='its leader was reaped')
        message = str(exc_info.value)
        assert f'build={build}' in message, message
        assert f'{build}: (' in message, (
            f'the verdict must render the surviving build pid={build}; got {message!r}'
        )
        assert 'its leader was reaped' in message, message

        os.kill(build, signal.SIGKILL)
        assert_pre_kill_tree_exited(tree, timeout=5.0, after='the build was SIGKILLed')
    finally:
        if leader.returncode is None:
            leader.kill()
            leader.wait(timeout=5)
        if build is not None:
            for pid in wait_pids_exited({build}, timeout=0):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signal.SIGKILL)


# ---------------------------------------------------------------------------
# Task 4092 -- deterministic-ish unit coverage for kill_holder_tree, the
# shared teardown helper that reaps a spawn_verify_merge holder AND every
# descendant it forked (including start_new_session escapes -- verify.py's
# `_run_cmd` runs every build/test command via
# ``create_subprocess_shell(..., start_new_session=True)``, so a killed
# leader alone leaves its build orphaned, reparented to init, for up to its
# full sleep duration).
#
# Every test here spawns a lightweight Popen stand-in (never the real CLI,
# never a git repo) -- the property under test is purely "does a
# session-escaped descendant get reaped" / "is the caller's own process
# group ever signalled", both of which this reproduces exactly.  See the
# plan's design_decisions for why a real verify-merge holder is not used
# here: it would drag in _setup_verify_repo's git work, config YAML, the
# ~9s CLI import and the load-scaled discovery waits, fixing a flake with a
# flake (this module's own task-2819 banner above warns against exactly
# this).  The two real-CLI call sites remain covered end-to-end by the Row
# 5 / lane-lock tests in test_laptop_warm_verify_boundary.py plus their
# post-run pgrep check.
#
# Task 5301: the helper now routes through
# orchestrator/src/orchestrator/verify_cancel.py::kill_process_tree while the
# leader is pinned, plus a per-holder tag sweep that also reaches the orphans
# of an already-reaped leader.
# ---------------------------------------------------------------------------


def test_kill_holder_tree_reaps_a_session_escaped_grandchild():
    """kill_holder_tree reaps BOTH the leader and a start_new_session grandchild.

    Reproduces the exact escape verify.py's ``_run_cmd`` produces
    (``create_subprocess_shell(cmd, start_new_session=True)``) with a
    lightweight stand-in leader: ``subprocess.Popen`` running a `-c` snippet
    that forks a ``sleep`` grandchild via ``start_new_session=True`` and
    then blocks itself for a long time, mirroring ``sleeper_spec``'s shape
    without needing the real CLI or a throwaway git repo.

    The sleep duration is derived from this test process's own pid so it
    cannot collide with an unrelated ``sleep`` on a shared dev box -- in
    particular with this very module's own ``sleeper_spec`` 300s sleeper.

    Asserts BOTH the leader is reaped and every captured grandchild has
    actually EXITED, not merely been signalled.  A zombie awaiting its
    subreaper's reap counts as exited -- see :func:`wait_pids_exited`.  It is
    self-cleaning (a finally that SIGKILLs any surviving captured pid) so a
    failing/RED run of this test never itself leaks the orphan it exists to
    pin.
    """
    sleep_secs = f'271.{os.getpid() % 1000:03d}'
    leader = TaggedPopen([
        sys.executable, '-c',
        f'import subprocess, time\n'
        f'subprocess.Popen(["sleep", "{sleep_secs}"], start_new_session=True)\n'
        f'time.sleep(300)\n',
    ])
    grandchildren: set[int] = set()
    try:
        discovery_deadline = time.monotonic() + 10.0
        while time.monotonic() < discovery_deadline:
            grandchildren = collect_descendants(leader.pid, read_ppid_map())
            if grandchildren:
                break
            time.sleep(0.05)
        assert grandchildren, (
            f'no descendant of leader pid={leader.pid} appeared within 10s -- '
            f'harness bug (the stand-in leader never forked its sleep '
            f'grandchild), not a seam defect under test'
        )

        kill_holder_tree(leader, timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS)

        assert leader.poll() is not None, (
            'kill_holder_tree must reap the leader -- poll() is still None '
            'after the call returned'
        )

        survivors = wait_pids_exited(grandchildren, timeout=5.0)
        state_by_pid = {pid: (st.state, st.ppid) for pid, st in sorted(survivors.items())}
        assert not survivors, (
            f'kill_holder_tree left session-escaped descendant(s) of leader '
            f'pid={leader.pid} running, as {{pid: (state, ppid)}}: {state_by_pid} '
            f'-- the exact orphan task 4092 exists to fix.  State S/T means '
            f'SIGKILL never reached it; R/D means it was signalled but has '
            f'not yet run to exit'
        )
    finally:
        if leader.poll() is None:
            leader.kill()
            leader.wait(timeout=5)
        for pid in grandchildren:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)


def test_kill_holder_tree_never_signals_the_callers_own_process_group():
    """kill_holder_tree must NEVER killpg the CALLER's own process group.

    This is the guard that makes this task's literally-prescribed
    ``os.killpg(os.getpgid(holder.pid), SIGKILL)`` one-liner unwritable.
    The lane-lock holder
    (``test_live_verify_merge_holds_lane_lock_real_subprocess``) is spawned
    WITHOUT ``--request-id``, so ``cli.py`` never calls ``os.setsid`` on it
    (:func:`~orchestrator.verify_cancel.start_own_process_group` is gated on
    ``if request_id is not None:``) and the holder stays in THIS process's
    (pytest's) own group -- ``os.getpgid(holder.pid) == os.getpgid(0)``. A
    naive killpg backstop there would SIGKILL the pytest worker running
    this very test.

    Reproduces that exact shape: a leader spawned via plain
    ``subprocess.Popen`` with no ``start_new_session``.  The precondition
    assertion proves this test really exercises the shared-group case
    rather than silently testing nothing.

    kill_holder_tree's guarded killpg backstop (see its own docstring)
    fires only when the holder is provably its own process-group leader
    (``pgid == proc.pid``) AND that group is provably not the caller's
    own (``pgid != os.getpgid(0)``). A holder that never setsid'd -- like
    this one -- fails the first condition, so the backstop must stay
    silent. This test pins that guard directly against the SHIPPED
    (killpg-aware) helper: it fails the moment a future edit drops either
    condition, or otherwise makes the backstop unconditional.
    """
    leader = TaggedPopen([sys.executable, '-c', 'import time; time.sleep(300)'])
    try:
        assert os.getpgid(leader.pid) == os.getpgid(0), (
            'harness bug: a plain subprocess.Popen with no start_new_session '
            'must inherit the caller\'s own process group, or this test is '
            'not exercising the shared-group case it exists to guard'
        )

        killpg_calls: list[tuple[int, int]] = []

        def spy(pgid, sig):
            killpg_calls.append((pgid, sig))

        kill_holder_tree(
            leader, timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS, _killpg=spy,
        )

        assert leader.poll() is not None, (
            'kill_holder_tree must still reap the leader even with the '
            'killpg backstop suppressed -- a guard satisfied by a helper '
            'that does nothing at all would be worthless'
        )

        own_pgid = os.getpgid(0)
        assert all(pgid != own_pgid for pgid, _sig in killpg_calls), (
            f"kill_holder_tree called killpg with the CALLER'S OWN process "
            f'group {own_pgid}: {killpg_calls} -- this would SIGKILL the '
            f'pytest worker itself'
        )
        assert killpg_calls == [], (
            f'kill_holder_tree must not call killpg at all when the holder '
            f"shares the caller's process group (never setsid'd): "
            f'{killpg_calls}'
        )
    finally:
        if leader.poll() is None:
            leader.kill()
            leader.wait(timeout=5)


def test_kill_holder_tree_killpg_backstop_fires_for_a_setsid_leader():
    """The guarded killpg backstop's POSITIVE path: it must actually fire.

    test_kill_holder_tree_never_signals_the_callers_own_process_group above
    only pins the NEGATIVE path -- killpg must stay silent when the holder
    never setsid'd. Nothing asserts that the backstop actually fires when
    the holder DID setsid, which is exactly the shape spawn_verify_merge
    produces when the CLI is launched with ``--request-id`` (cli.py:578-581
    gates ``start_own_process_group()`` on ``if request_id is not None:``).
    A regression that made the guard's admission condition unconditionally
    false -- e.g. inverting the ``pgid == proc.pid`` comparison, or
    dropping the branch entirely -- would leave both existing
    kill_holder_tree killpg unit tests green: exactly the "guard reporting
    PASS while guarding nothing" failure mode the boundary suite's
    anti-vacuity discipline exists to prevent (see the banner of
    test_laptop_warm_verify_boundary.py::
    test_every_scaled_discovery_wait_is_covered_by_its_test_timeout_mark).

    Reproduces the setsid'd shape directly with ``start_new_session=True``,
    which is precisely what makes a Popen leader its own process-group
    leader (``os.getpgid(leader.pid) == leader.pid``). The two precondition
    assertions below prove BOTH halves of the backstop's admission
    condition actually hold before kill_holder_tree runs, so a pass here
    cannot be a vacuous accident of the harness's own process group.
    """
    leader = TaggedPopen(
        [sys.executable, '-c', 'import time; time.sleep(300)'],
        start_new_session=True,
    )
    try:
        leader_pgid = os.getpgid(leader.pid)
        assert leader_pgid == leader.pid, (
            'harness bug: start_new_session=True must make the leader its '
            'own process-group leader, or this test is not exercising the '
            "killpg backstop's admission condition it exists to guard"
        )
        assert leader_pgid != os.getpgid(0), (
            "harness bug: the setsid'd leader's group must differ from "
            "the caller's own, or this test is not exercising the killpg "
            "backstop's admission condition it exists to guard"
        )

        killpg_calls: list[tuple[int, int]] = []

        def spy(pgid, sig):
            killpg_calls.append((pgid, sig))

        kill_holder_tree(
            leader, timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS, _killpg=spy,
        )

        assert leader.poll() is not None, (
            "kill_holder_tree must reap a setsid'd leader the same as any "
            'other'
        )
        assert killpg_calls == [(leader_pgid, signal.SIGKILL)], (
            f"kill_holder_tree must killpg a leader that provably setsid'd "
            f'into its own, non-caller process group -- got {killpg_calls}, '
            f'expected exactly one call with pgid={leader_pgid}'
        )
    finally:
        if leader.poll() is None:
            leader.kill()
            leader.wait(timeout=5)


def _wait_until_zombie(pid: int, *, timeout: float = 5.0) -> bool:
    """Poll ``/proc/<pid>/stat`` until *pid* is a zombie (state ``Z``).

    Deliberately does NOT use ``Popen.wait()``/``poll()``: those reap the
    process, which is precisely the state transition the caller here needs
    to NOT happen.  Reads the state via :func:`read_proc_state`.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = read_proc_state(pid)
        if state is None:
            return False  # already reaped by someone else, or gone
        if state.state == 'Z':
            return True
        time.sleep(0.02)
    return False


def test_kill_holder_tree_does_not_reap_a_leader_that_exits_mid_teardown():
    """A leader that exits DURING teardown must not be reaped before the backstop.

    Closes the window between the entry-time ``proc.returncode is not
    None`` short circuit and the killpg backstop.  The two existing killpg
    unit tests only cover the endpoints -- "already reaped at entry"
    (``test_kill_holder_tree_is_safe_when_the_leader_already_exited``) and
    "alive throughout"
    (``test_kill_holder_tree_killpg_backstop_fires_for_a_setsid_leader``)
    -- and both stay GREEN with a ``poll()`` reintroduced at the leader
    kill, so neither can catch this.

    THE HAZARD.  ``Popen.poll()`` is not a passive read: it calls
    ``os.waitpid(pid, WNOHANG)``.  A ``poll()`` at the leader-kill step
    would therefore REAP a leader that exited since entry and FREE
    ``proc.pid`` -- after which the backstop's ``pgid == proc.pid`` test
    compares a pre-reap pgid against a freed pid, and fires
    ``killpg(pgid, SIGKILL)`` at a group the kernel is free to have
    reassigned to a stranger.  On a shared dev box or CI host that is an
    unrelated user's entire process group.

    THIS IS THE ROW 5 HOLDER'S DESIGNED BEHAVIOUR, not an exotic race: its
    ``finally`` calls ``heartbeat_holder.stop_heartbeats()`` immediately
    before ``kill_holder_tree``, and stopping heartbeats ARMS the CLI's
    stdin-watchdog self-kill, so the holder exiting on its own inside the
    teardown window is exactly what the surrounding code sets up.

    THE ASSERTION.  The ``_ppid_map_provider`` seam is used to land the
    leader's exit strictly inside the window: the provider SIGKILLs the
    leader and waits (via ``/proc``, never ``wait()``) until it is a
    ZOMBIE, so the leader is provably exited-but-unreaped by the time the
    helper reaches the leader kill.  The killpg spy then records
    ``proc.returncode`` AT CALL TIME.  Under the fixed helper that is
    ``None`` -- nothing in the helper reaped it, so the pid is still
    PINNED and ``pgid`` provably still denotes the holder's own group.
    Reintroduce the ``poll()`` guard and the spy sees a non-``None``
    returncode: the pid was freed before the backstop fired, and the test
    goes RED.
    """
    leader = TaggedPopen(
        [sys.executable, '-c', 'import time; time.sleep(300)'],
        start_new_session=True,
    )
    try:
        leader_pgid = os.getpgid(leader.pid)
        assert leader_pgid == leader.pid, (
            'harness bug: start_new_session=True must make the leader its '
            'own process-group leader, or this test is not exercising the '
            "killpg backstop's admission condition it exists to guard"
        )
        assert leader_pgid != os.getpgid(0), (
            "harness bug: the setsid'd leader's group must differ from the "
            "caller's own, or this test is not exercising the killpg "
            "backstop's admission condition it exists to guard"
        )
        assert leader.returncode is None, (
            'harness bug: the leader must be un-reaped at entry, or '
            "kill_holder_tree's already-reaped short circuit fires and this "
            'test exercises nothing'
        )

        became_zombie: list[bool] = []

        def provider_that_kills_the_leader():
            """Land the leader's exit strictly inside the teardown window."""
            os.kill(leader.pid, signal.SIGKILL)
            became_zombie.append(_wait_until_zombie(leader.pid))
            return {}  # no descendants; the pid-pinning property is under test

        killpg_calls: list[tuple[int, int, int | None]] = []

        def killpg_spy(pgid, sig):
            # Record the leader's REAPED-ness as the backstop sees it.
            killpg_calls.append((pgid, sig, leader.returncode))

        kill_holder_tree(
            leader,
            timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS,
            _ppid_map_provider=provider_that_kills_the_leader,
            _killpg=killpg_spy,
        )

        assert became_zombie == [True], (
            f'harness bug: the leader never became an unreaped zombie inside '
            f'the teardown window ({became_zombie}) -- this test is not '
            f'exercising the exit-mid-teardown race it exists to pin'
        )
        assert killpg_calls, (
            'kill_holder_tree must still fire its killpg backstop for a '
            "leader that setsid'd and then exited mid-teardown -- same-group "
            'stragglers still need reaping'
        )
        assert [(pgid, sig) for pgid, sig, _rc in killpg_calls] == [
            (leader_pgid, signal.SIGKILL)
        ], (
            f'kill_holder_tree must killpg exactly the holder\'s own group -- '
            f'got {killpg_calls}, expected pgid={leader_pgid}'
        )
        assert all(rc is None for _pgid, _sig, rc in killpg_calls), (
            f'kill_holder_tree REAPED the leader before firing its killpg '
            f'backstop (returncode at killpg time: '
            f'{[rc for _p, _s, rc in killpg_calls]}) -- proc.pid was FREED, '
            f'so pgid={leader_pgid} may already belong to a stranger. This is '
            f'the pid-recycling hazard the backstop is documented to avoid; '
            f'a poll()/wait() must not run before the backstop.'
        )
        assert leader.poll() is not None, (
            'kill_holder_tree must still reap the leader by the time it '
            'returns -- pinning the pid across the backstop is not a licence '
            'to leak a zombie'
        )
    finally:
        if leader.poll() is None:
            leader.kill()
            leader.wait(timeout=5)


def test_kill_holder_tree_is_safe_when_the_leader_already_exited():
    """An already-reaped leader means proc.pid is a FREE pid -- derive NOTHING from it.

    Four converted call sites reap the leader with ``wait()`` inside their
    ``try`` block BEFORE the ``finally`` runs ``kill_holder_tree``
    (``test_watchdog_timeout_env_override_fires_fast_without_heartbeat``,
    ``test_cancel_verify_tree_kills_under_live_watchdog``,
    ``test_ssh_dropped_mid_build_tree_killed_via_eof_dispatcher_alive``,
    ``test_heartbeat_starved_hard_partition_tree_killed_via_timeout``) --
    so on their GREEN path ``proc.pid`` no longer refers to the holder at
    all by the time this helper runs.
    ``os.getpgid(proc.pid)`` and ``collect_descendants(proc.pid, ...)``
    would then describe whatever process happens to OWN that pid now, and
    every descendant of that stranger would be SIGKILLed (and killpg'd by
    the backstop, if it happened to be its own group leader).  This is not
    theoretical: pid recycling is observed on this fleet
    (verify_cancel.py:313-315 -- pid_max=4194304, and the laptop's own pid
    counter demonstrably wrapped on 2026-08-11).

    Three cases; (a) and (b) share ONE already-reaped leader:

    (a) REAL-PATH no-signal contract: every seam is spied rather than
        faked, and NONE may fire -- in particular the ppid-map provider
        must never be CALLED AT ALL, which is what proves the short
        circuit lands before the snapshot phase rather than merely seeing
        a map with no descendants in it.  This leader spawned nothing, so
        the tag sweep finds nothing to signal either (the sweep itself is
        pinned by ``test_kill_holder_tree_sweeps_the_session_escaped_orphan_of_an_already_reaped_leader``).
        Also pins that the already-reaped path still closes stdin, since
        the lane-lock site
        (``test_live_verify_merge_holds_lane_lock_real_subprocess``)
        delegates its stdin cleanup to this helper.

    (b) PID-REUSE MODEL -- the case that makes this RED non-vacuous.
        Against a reaped leader the REAL ``/proc`` map almost always has
        no descendants, so asserting against it would pass by luck even
        without the short circuit.  A SYNTHETIC ppid map gives the reaped
        (free) pid fabricated children, modeling a stranger process
        reusing that pid -- ``collect_descendants`` accepts any dict and
        is cycle-safe (verify_cancel.py:197), and every pid is fake with
        ``_kill`` spied, so this can never signal a real process even if
        the guard under test is missing.

    (c) the OSError-degradation coverage (a)/(b) can no longer reach once
        the short circuit exists, since it never gets far enough to call
        the ppid-map provider -- moved onto a LIVE leader so the
        ``except OSError: ppid_map = {}`` branch stays under test.
    """
    # (a) + (b): one already-reaped leader, shared.
    leader = TaggedPopen([sys.executable, '-c', 'pass'], stdin=subprocess.PIPE)
    leader.wait(timeout=10)

    # (a) REAL-PATH: every seam spied; none may fire for an already-reaped
    # leader, and the ppid-map provider must not even be CALLED.
    kill_calls_a: list[tuple[int, int]] = []
    killpg_calls_a: list[tuple[int, int]] = []
    ppid_map_call_count = [0]

    def kill_spy_a(pid, sig):
        kill_calls_a.append((pid, sig))

    def killpg_spy_a(pgid, sig):
        killpg_calls_a.append((pgid, sig))

    def ppid_map_spy_a():
        ppid_map_call_count[0] += 1
        return {}

    kill_holder_tree(
        leader,
        timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS,
        _ppid_map_provider=ppid_map_spy_a,
        _kill=kill_spy_a,
        _killpg=killpg_spy_a,
    )

    assert kill_calls_a == [], (
        f'kill_holder_tree signalled pid(s) {kill_calls_a} for an '
        f'already-reaped leader -- proc.pid is FREE, so this can only '
        f'reach an unrelated stranger process'
    )
    assert killpg_calls_a == [], (
        f'kill_holder_tree called killpg{killpg_calls_a} for an '
        f'already-reaped leader'
    )
    assert ppid_map_call_count[0] == 0, (
        'kill_holder_tree consulted the ppid-map provider for an '
        'already-reaped leader -- the short circuit must land BEFORE the '
        'snapshot phase, not merely happen to see a map with no '
        'descendants in it'
    )
    assert leader.stdin is not None and leader.stdin.closed, (
        'kill_holder_tree must still close stdin on the already-reaped '
        'early-return path -- the lane-lock site '
        '(test_live_verify_merge_holds_lane_lock_real_subprocess) '
        'delegates its stdin cleanup to this helper'
    )

    # (b) PID-REUSE MODEL: a synthetic ppid map gives the reaped (free)
    # leader pid fabricated children, modeling a stranger process reusing
    # that pid.  Every pid below is fake and _kill/_killpg stay spied, so
    # this can never signal a real process.
    synthetic_ppid_map = {
        leader.pid: 1,
        99990001: leader.pid,
        99990002: 99990001,
    }
    kill_calls_b: list[tuple[int, int]] = []
    killpg_calls_b: list[tuple[int, int]] = []

    def kill_spy_b(pid, sig):
        kill_calls_b.append((pid, sig))

    def killpg_spy_b(pgid, sig):
        killpg_calls_b.append((pgid, sig))

    kill_holder_tree(
        leader,
        timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS,
        _ppid_map_provider=lambda: synthetic_ppid_map,
        _kill=kill_spy_b,
        _killpg=killpg_spy_b,
    )

    assert kill_calls_b == [], (
        f'kill_holder_tree signalled fabricated pid(s) {kill_calls_b}, '
        f"reachable only through a stranger's reuse of the reaped "
        f"leader's pid {leader.pid} -- an already-reaped leader must "
        f'signal NOTHING'
    )
    assert killpg_calls_b == [], (
        f'kill_holder_tree called killpg{killpg_calls_b} against a '
        f"stranger's fabricated process group for an already-reaped "
        f'leader'
    )

    # (c) OSError-degradation coverage, moved onto a LIVE leader: once (a)
    # short-circuits before ever calling the ppid-map provider, a reaped
    # leader can no longer exercise the `except OSError: ppid_map = {}`
    # guard.  _kill is deliberately left as the REAL os.kill here so the
    # leader is genuinely SIGKILLed and reaped fast, well inside the 5.0s
    # ceiling -- spying it would leave the process alive for the whole
    # timeout and emit the "did not exit" warning instead.
    def raising_ppid_map_provider():
        raise OSError('simulated /proc read racing process exit')

    live_leader = TaggedPopen(
        [sys.executable, '-c', 'import time; time.sleep(300)'],
    )
    try:
        kill_holder_tree(
            live_leader,
            timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS,
            _ppid_map_provider=raising_ppid_map_provider,
        )
        assert live_leader.poll() is not None, (
            'kill_holder_tree must still reap a LIVE leader when the '
            'ppid-map provider raises OSError (a /proc read racing '
            'process exit)'
        )
    finally:
        if live_leader.poll() is None:
            live_leader.kill()
            live_leader.wait(timeout=5)


def test_kill_holder_tree_sweeps_the_session_escaped_orphan_of_an_already_reaped_leader():
    """An already-reaped leader's session-escaped orphan is still killed; other holders are not.

    Models the four rows that ``wait()`` the leader inside their ``try``
    before the ``finally`` tears down: by then the build is reparented away,
    leads its own session, and the ppid walk provably cannot reach it.  Only
    the holder's inherited tag still names it.  The bystander is a different
    holder's tree, which Row 5's waiter teardown must never reach.
    """
    sleep_secs = f'272.{os.getpid() % 1000:03d}'
    leader = TaggedPopen(
        [
            sys.executable, '-c',
            'import subprocess\n'
            f'child = subprocess.Popen(["sleep", "{sleep_secs}"], start_new_session=True, '
            'stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n'
            'print(child.pid, flush=True)\n',
        ],
        stdout=subprocess.PIPE,
    )
    bystander = TaggedPopen([sys.executable, '-c', 'import time; time.sleep(300)'])
    try:
        assert leader.stdout is not None
        grandchild = int(leader.stdout.readline())
        leader.wait(timeout=10)

        assert leader.returncode is not None, (
            "harness bug: the leader must already be reaped, or kill_holder_tree's "
            'already-reaped short circuit never fires and this test exercises nothing'
        )
        orphan_state = read_proc_state(grandchild)
        assert orphan_state is not None and not orphan_state.terminated, (
            f'harness bug: orphan pid={grandchild} is not running ({orphan_state}) '
            f'before teardown, so its exit afterwards would prove nothing'
        )
        assert orphan_state.ppid != leader.pid, (
            f'harness bug: orphan pid={grandchild} still names the reaped leader '
            f'pid={leader.pid} as its parent -- it was never reparented'
        )
        assert grandchild not in collect_descendants(leader.pid, read_ppid_map()), (
            f'harness bug: the ppid walk from pid={leader.pid} still reaches orphan '
            f'pid={grandchild}, so this test does not isolate the severed-chain case'
        )

        kill_holder_tree(leader, timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS)

        survivors = wait_pids_exited({grandchild}, timeout=5.0)
        state_by_pid = {pid: (st.state, st.ppid) for pid, st in sorted(survivors.items())}
        assert not survivors, (
            f'kill_holder_tree left the session-escaped orphan of already-reaped '
            f'leader pid={leader.pid} running, as {{pid: (state, ppid)}}: '
            f'{state_by_pid} -- the #4962 leak'
        )
        assert bystander.poll() is None, (
            "kill_holder_tree reached a different holder's tree -- the sweep "
            'must be scoped to this holder alone'
        )
    finally:
        sweep_tagged(leader.tag)
        bystander.kill()
        bystander.wait(timeout=5)
        if leader.stdout is not None:
            leader.stdout.close()


# ---------------------------------------------------------------------------
# Task 5301 -- per-holder env tag.  A TaggedPopen stamps a uuid into its
# child's environment; every descendant inherits it across fork, setsid and
# exec, so tagged_pids() names the holder's whole tree even after the /proc
# ppid chain to a session-escaped build is severed.
# test_laptop_warm_verify_boundary.py::
# test_spawned_verify_merge_build_carries_the_holder_tag pins the premise end
# to end: verify.py::_target_subprocess_env must pass the tag through to the
# build it spawns.
# ---------------------------------------------------------------------------


def test_tagged_pids_finds_a_session_escaped_grandchild_and_nothing_else():
    """tagged_pids(tag) names exactly one holder's tree, session escapes included."""
    sleep_secs = f'273.{os.getpid() % 1000:03d}'
    leader = TaggedPopen(
        [
            sys.executable, '-c',
            'import subprocess, time\n'
            f'child = subprocess.Popen(["sleep", "{sleep_secs}"], start_new_session=True, '
            'stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n'
            'print(child.pid, flush=True)\n'
            'time.sleep(300)\n',
        ],
        stdout=subprocess.PIPE,
    )
    bystander = TaggedPopen([sys.executable, '-c', 'import time; time.sleep(300)'])
    try:
        assert leader.stdout is not None
        grandchild = int(leader.stdout.readline())
        assert os.getsid(grandchild) == grandchild, (
            f'harness bug: grandchild pid={grandchild} must lead its own '
            f'session, or this test is not exercising the escape the sweep '
            f'exists for'
        )

        assert isinstance(leader.tag, str) and leader.tag
        assert leader.tag != bystander.tag, 'tags are per holder, never shared'

        found = tagged_pids(leader.tag)
        assert {leader.pid, grandchild} <= found, (
            f'tagged_pids missed part of the leader pid={leader.pid} tree '
            f'(grandchild pid={grandchild}): found {sorted(found)}'
        )
        assert bystander.pid not in found, (
            "another holder's tree must never match this holder's tag"
        )
        assert os.getpid() not in found
    finally:
        sweep_tagged(leader.tag)
        sweep_tagged(bystander.tag)
        leader.wait(timeout=5)
        bystander.wait(timeout=5)
        if leader.stdout is not None:
            leader.stdout.close()
