"""The repo's single async subprocess-spawn primitive for git (and friends).

WHY THIS EXISTS. Two independent call sites needed the same carefully-built
child-process primitive, and only one of them had it:

  - orchestrator ``git_ops._run`` grew it over several incidents (the
    ``LC_ALL=C`` locale pin, optional stdin feeding, the task-2608
    cancellation kill+reap). It is now a thin adapter over this module,
    keeping only its own ``WorktreeMissing`` taxonomy on top.
  - the fused-memory live-workflow probes were still calling BLOCKING
    ``subprocess.run`` on the event-loop thread (task 3778), fanned out per
    task across a reconciliation sweep — measured at a 15-43 s event-loop
    stall.

Rather than clone the primitive into fused-memory, it lives here once
(INV-5, no lockstep duplication) with the two things the fused-memory side
additionally needs and the orchestrator side never grew: a per-call
``timeout`` and a concurrency bound.

WHAT ASYNC DOES AND DOES NOT BUY YOU (task 3778). Be precise about this,
because it is the trap this module exists to bound:
``asyncio.create_subprocess_exec`` still performs the fork/exec INLINE on
the event-loop thread — the ``kernel_clone`` wchan samples in 3778's
root-cause report ARE that fork. Making a probe "async" therefore does NOT
zero its loop occupancy; it only stops the loop being blocked for the
child's whole RUNTIME. What bounds the residual fork cost is
:data:`MAX_CONCURRENT_SPAWNS`: at most that many spawns are ever in flight
on one loop, so a caller that fans out over hundreds of tasks cannot
convert its fan-out directly into loop-thread fork storms.

LOCALE IS LOAD-BEARING. ``LC_ALL=C``/``LANG=C`` are forced in the child so
git always emits English-locale diagnostics. ``git_ops._git_clean_failure_
is_benign`` substring-matches that English warning text; under a translated
locale the matcher silently stops recognising it, defeating the R3
ENOENT-tolerance fix for the 4892-class warm-lane FAULT. Do not "simplify"
this away.

CANCELLATION SAFETY (task 2608). If the ``communicate()`` await is
interrupted — a caller-side ``asyncio.wait_for``, an outer cancellation, or
this module's own ``timeout`` — the spawned child would otherwise keep
running as an orphan holding its stdout/stderr pipes open. For a
persistently-hung script that recurred every scheduler sweep, leaking a
process and file descriptors each time. The child is best-effort killed and
reaped on every interrupted path.

FAIL-OPEN VS RAISE. A non-zero returncode and a timeout are RETURNED as
data (``GitResult.ok`` is False), never raised: every fused-memory probe
reads "git said no" as *signal absent* and must not have an exception
thrown through it. ``FileNotFoundError`` from the spawn itself is the one
error that PROPAGATES — ``git_ops`` re-classifies it into ``WorktreeMissing``
to tell a vanished worktree (a recoverable race) apart from a missing
binary on ``PATH``, and swallowing it here would destroy that distinction.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

logger = logging.getLogger(__name__)

#: Maximum simultaneous child spawns per event loop. The fork happens on the
#: loop thread (see the module docstring), so this is a bound on loop
#: OCCUPANCY, not merely on process count. Sized to keep a recon-sweep
#: fan-out useful (calls still overlap) while a burst of hundreds of probes
#: cannot translate into hundreds of inline forks.
MAX_CONCURRENT_SPAWNS = 8

#: Returncode reported for a timed-out call whose reaped child left no signal
#: of its own. 124 is what ``timeout(1)`` uses, so it reads correctly in logs.
TIMEOUT_RETURNCODE = 124

#: Per-running-loop semaphores. Deliberately NOT a module-level singleton:
#: since 3.10 asyncio primitives bind lazily to the loop of first use and
#: thereafter raise ``RuntimeError: ... is bound to a different event loop``.
#: pytest builds a fresh loop per async test and the fused-memory server has
#: been restarted in-process, so a singleton would work exactly once and then
#: fail with an error that looks nothing like a concurrency bug. Weak keys
#: also let a dead loop's semaphore (and its waiters) be collected.
_LOOP_SEMAPHORES: MutableMapping[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
    WeakKeyDictionary()
)


@dataclass(frozen=True)
class GitResult:
    """The outcome of one child process run.

    ``timed_out`` is carried separately from ``returncode`` because a caller
    that wants to distinguish "git answered, negatively" from "git never
    answered" cannot do so from an exit code alone.
    """

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        """True iff the child exited 0 within its timeout."""
        return self.returncode == 0 and not self.timed_out


def _semaphore_for_running_loop() -> asyncio.Semaphore:
    """Return this loop's spawn semaphore, creating it on first use."""
    loop = asyncio.get_running_loop()
    sem = _LOOP_SEMAPHORES.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(MAX_CONCURRENT_SPAWNS)
        _LOOP_SEMAPHORES[loop] = sem
    return sem


async def _kill_and_reap(proc: asyncio.subprocess.Process) -> None:
    """Best-effort kill + reap of an interrupted child. Never raises."""
    with contextlib.suppress(ProcessLookupError):
        proc.kill()  # may already have exited
    with contextlib.suppress(BaseException):
        await proc.wait()  # reap is best-effort; never let it mask the original error


async def run_git(
    cmd: Sequence[str],
    cwd: Path | str | None = None,
    *,
    input_text: str | None = None,
    timeout: float | None = None,
) -> GitResult:
    """Spawn *cmd* asynchronously and return its :class:`GitResult`.

    :param cmd: argv, e.g. ``['git', 'worktree', 'list', '--porcelain']``.
    :param cwd: working directory for the child. Unlike ``git_ops._run``
        this function does NOT pre-flight it — a missing directory surfaces
        as ``FileNotFoundError`` and the caller decides what that means.
    :param input_text: when given, the child is spawned with ``stdin=PIPE``
        and this text is written to it (the ``git patch-id`` path). When
        ``None`` stdin is not piped and the child inherits the parent's, so
        the capability is inert unless used.
    :param timeout: seconds to wait for the child. On expiry the child is
        killed and reaped and a ``timed_out=True`` result is RETURNED (with
        a non-zero returncode) rather than raised — see "fail-open vs raise"
        in the module docstring. ``None`` waits indefinitely.

    :raises FileNotFoundError: the binary (or *cwd*) does not exist.

    Concurrency: at most :data:`MAX_CONCURRENT_SPAWNS` calls are in flight
    per event loop; excess callers wait on the semaphore.
    """
    async with _semaphore_for_running_loop():
        return await _spawn_and_communicate(cmd, cwd, input_text, timeout)


async def _spawn_and_communicate(
    cmd: Sequence[str],
    cwd: Path | str | None,
    input_text: str | None,
    timeout: float | None,
) -> GitResult:
    """The primitive itself, run under the caller's semaphore slot."""
    # Force a stable C locale so child output is always English and amenable
    # to substring matching (see the module docstring — this is load-bearing).
    env = {**os.environ, 'LC_ALL': 'C', 'LANG': 'C'}
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd is not None else None,
        stdin=asyncio.subprocess.PIPE if input_text is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    communicate = proc.communicate(
        input=input_text.encode() if input_text is not None else None,
    )
    try:
        if timeout is None:
            stdout, stderr = await communicate
        else:
            stdout, stderr = await asyncio.wait_for(communicate, timeout=timeout)
    except TimeoutError:
        # Our own timeout fired. wait_for already cancelled communicate(), so
        # the child is still alive: kill + reap it before degrading. This
        # branch DEGRADES (returns a result), so it logs loudly — a silent
        # timeout is indistinguishable from a clean negative answer.
        await _kill_and_reap(proc)
        logger.warning(
            'git_async.timeout cmd=%s cwd=%s timeout=%.3fs rc=%s '
            '(child killed and reaped; returning fail-open timed-out result)',
            ' '.join(cmd), cwd, timeout, proc.returncode,
        )
        return GitResult(
            returncode=proc.returncode or TIMEOUT_RETURNCODE,
            stdout='',
            stderr=f'timed out after {timeout}s',
            timed_out=True,
        )
    except BaseException:
        # The await was interrupted by something OTHER than our own timeout —
        # most commonly asyncio.CancelledError from a caller-side wait_for.
        # Kill + reap so the child does not leak as an orphan (task 2608),
        # then propagate the original exception unchanged. No log here: this
        # path re-raises, so it is not a silent fallthrough.
        await _kill_and_reap(proc)
        raise
    return GitResult(
        returncode=proc.returncode if proc.returncode is not None else 1,
        stdout=stdout.decode().strip(),
        stderr=stderr.decode().strip(),
    )
