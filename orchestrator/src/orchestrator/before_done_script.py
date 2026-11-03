"""Run a before_done script subprocess to completion under its own timeout,
reporting an overrun as the measured ``ScriptTimeout``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import assert_never

from shared.proc_group import SigkillDispatch, sigkill_process_group

logger = logging.getLogger(__name__)

# Task 2090: bound the reap after a whole-process-group SIGKILL on a
# before_done subprocess timeout.  A process stuck in an uninterruptible
# state (or one that otherwise ignores SIGKILL) must never be allowed to hang
# `terminate_process_tree` — and therefore `run_script` — forever.
# 5s comfortably covers a normal kernel-mediated reap while staying bounded.
REAP_GRACE_SECS: float = 5.0


class ProcessTeardown(Enum):
    """What ``terminate_process_tree`` actually DISPATCHED — never what died.

    That distinction is the whole point of this type (task 4252 reviewer
    finding): the timeout branch used to describe its teardown as a
    whole-group SIGKILL unconditionally, while the branch it takes MOST often
    dispatches no signal at all, so the escalation told an operator a
    surviving process was gone when it was still running.  A caller can only
    state what was dispatched if the teardown RETURNS it.

    Each member's value IS the one clause every operator-facing site renders
    (via ``clause`` below), so the enum and the prose cannot drift apart:
    there is deliberately no second table keyed by member.

    Nothing serialises these — this is an in-process return value, hence a
    plain ``Enum`` rather than a ``StrEnum``.
    """

    #: ``os.killpg`` reached the whole group, so processes the script spawned
    #: died with it.
    GROUP_KILLED = 'its whole process group was SIGKILLed'
    #: The whole-group signal was refused as unsafe
    #: (``shared.proc_group.unsafe_pgid_reason``) or the ``killpg`` itself
    #: raised, so ONLY the direct child was signalled — anything it spawned
    #: SURVIVES (the refusal branch's own log line already says as much).
    DIRECT_KILLED = (
        'only its direct child process was signalled — the whole-group signal '
        'was refused or failed, so nothing the script spawned was killed'
    )
    #: ``proc`` was already reaped, so the pid-recycling guard dispatched
    #: NOTHING and the whole tree below it SURVIVES.  See
    #: ``shared/src/shared/proc_group.py::sigkill_process_group`` for why
    #: refusing to signal beats reaping an orphan.
    NOT_SIGNALLED = (
        'nothing was signalled at all — the script had already been reaped, so '
        'signalling its possibly-recycled pid was refused, and nothing the '
        'script spawned was killed'
    )

    @property
    def clause(self) -> str:
        """The canonical clause describing this dispatch, for operator text.

        Reads as a continuation of "… exceeded its own per-script timeout
        (Ns) *and* <clause>", which is how both the ``ScriptTimeout`` message
        and ``ScriptTimeout.budget_line`` render it — one definition, so
        the exception's own message and every escalation detail state the
        same fact.
        """
        return self.value

    @classmethod
    def from_dispatch(cls, dispatch: SigkillDispatch) -> ProcessTeardown:
        """The disposition for what ``sigkill_process_group`` reports dispatching."""
        match dispatch:
            case SigkillDispatch.GROUP_KILLED:
                return cls.GROUP_KILLED
            case SigkillDispatch.DIRECT_KILLED:
                return cls.DIRECT_KILLED
            case SigkillDispatch.NOT_SIGNALLED:
                return cls.NOT_SIGNALLED
            case _:
                assert_never(dispatch)


class ScriptTimeout(Exception):
    """Raised by ``run_script`` when its INNER per-subprocess
    ``asyncio.wait_for`` fires — i.e. the script itself overran
    ``before_done['timeout_secs']`` and the Layer-A teardown ran (task 4065).

    THIS DOCSTRING IS THE CANONICAL EXPLANATION of the classification.  The
    other 4065 sites (docs/deterministic-runner.md, and in
    ``orchestrator/src/orchestrator/deterministic_runner.py``:
    ``DeterministicRunner._default_run_script``,
    ``DeterministicRunner._run_predicate``, both seam wrappers, the
    ``script_runner`` ctor arg) deliberately carry one-line pointers here rather than restating it, so
    the next change to this rule has ONE place to update, not ten that can
    independently rot.

    Why an exception rather than a ``(rc, tail)`` return: the previous
    ``return 1, '<script timed out after Ns>'`` was indistinguishable, at the
    ``run_fn`` seam, from a predicate check script genuinely EXITING non-zero
    — which on the γ-predicate path is a milestone VERDICT ("the invariant
    does not hold"), complete with a ``gate_escalated_at`` stamp latching the
    task into the resolve-to-done path.  A timed-out script produced no exit
    code and therefore no verdict; it is an INFRA fault.  A sentinel rc (e.g.
    GNU ``timeout``'s 124) would merely move the collision — a script that
    genuinely exits with the sentinel would be misclassified the other way —
    whereas an exception type cannot collide with any exit code at all.

    Deliberately NOT a ``TimeoutError`` subclass: both ``run_fn`` seam
    wrappers (``_invoke_run_fn_translating_timeout`` and ``_run_predicate``'s
    local ``_invoke_run_fn``) translate a seam-internal ``TimeoutError`` into
    ``RuntimeError``, precisely so that an ``except TimeoutError`` around the
    OUTER ``asyncio.wait_for`` can only ever mean "the outer wall-clock guard
    itself fired".  A ``TimeoutError`` subclass would be swallowed by that
    translation and re-reported as a generic "unexpected error" — trading one
    misattribution for another, one layer down.

    ONLY the default runner raises this.  An injected ``script_runner`` keeps
    the plain ``(rc, tail)`` contract, so a custom runner's non-zero rc is
    still an honest verdict on the predicate path.  Classification keys on
    this exception TYPE — never on substring-matching the tail, which would
    misclassify any check script that merely PRINTS "timed out".

    Carries exactly three things, ALL MEASURED (task 4252 + reviewer
    amendments) — nothing here is fabricated, and no ``(rc, tail)`` pair is
    ferried for a caller to restore:

    * ``timeout_secs`` — the budget the script overran.
    * ``exit_code`` — the script's OWN exit code if it had already exited,
      else ``None``.  Non-``None`` in exactly the task-2090 shape:
      ``communicate()`` waits on the merged stdout/stderr pipe, so a
      grandchild holding its write end can time the read out AFTER the script
      itself exited, and ``Process.returncode`` is populated by the child
      watcher independently of ``communicate()``.  Claiming "no exit code"
      there would be an affirmative falsehood about a script that in fact ran
      to completion.  A signal death (negative ``returncode``) and an
      incomplete reap both normalize to ``None``: neither produced an exit
      code.
    * ``teardown`` — WHICH signal the Layer-A teardown dispatched
      (``ProcessTeardown``), observed and returned by
      ``terminate_process_tree`` rather than re-derived here.

    ``exit_code`` is NOT a sound proxy for whether a kill was dispatched, in
    EITHER direction.  The usual exited-script case above short-circuits the
    ``killpg`` entirely (the pid-recycling guard refuses to signal a reaped
    leader), so an exit code comes with NO signal at all; conversely a script
    that exits in the narrow window between the timeout firing and the
    ``killpg`` gets both an exit code AND a real whole-group kill.  That is
    precisely why the disposition is measured and carried instead of inferred
    by a caller.

    Captured output is never carried in any case — that read was still in
    flight when the timeout fired.  Every caller — both deploy paths and the
    predicate path — owns a dedicated ``except ScriptTimeout`` arm, each
    placed BEFORE its ``except Exception`` catch-all, and reports the timeout
    as what it was.
    """

    def __init__(
        self,
        timeout_secs: float,
        *,
        exit_code: int | None = None,
        teardown: ProcessTeardown,
    ) -> None:
        self.timeout_secs = timeout_secs
        self.exit_code = exit_code
        self.teardown = teardown
        # `teardown` is REQUIRED and keyword-only so no construction site can
        # silently fall back to assuming a kill — the defect this carries the
        # measurement to fix.
        if exit_code is None:
            super().__init__(
                f'script timed out after {timeout_secs}s and {teardown.clause} '
                f'— no exit code was produced'
            )
        else:
            super().__init__(
                f'script timed out after {timeout_secs}s: the script itself '
                f'exited with code {exit_code}, but a process it spawned '
                f'outlived it holding the output pipe open, and '
                f'{teardown.clause}'
            )

    def budget_line(self, subject: str) -> str:
        """The sentence EVERY ``ScriptTimeout`` arm opens with: which budget was
        overrun, and WHICH SIGNAL the teardown then dispatched.

        Shared by the deploy arms (via ``deploy_fact_lines``) and by
        ``DeterministicRunner._run_predicate``'s arm, which differ only in
        *subject* — so the formatted budget expression has ONE definition to
        change rather than two copies to keep in step (reviewer amendment).

        The trailing clause is ``self.teardown.clause`` (task 4252), so all
        three arms state what was actually signalled.  Every one of them
        previously asserted a whole-group SIGKILL unconditionally, including
        on the branch that dispatches nothing at all — centralising this
        sentence is precisely what let one edit correct all three.
        """
        return (
            f'{subject} exceeded its own per-script timeout '
            f"({self.timeout_secs}s = before_done['timeout_secs']) and "
            f'{self.teardown.clause}.'
        )

    def deploy_fact_lines(self, *, script: Path) -> list[str]:
        """The FACT sentences both DEPLOY arms print for this timeout.

        Shared so the two deploy branches cannot come to say different things
        about the same event — the anti-drift property
        ``DeterministicRunner._invoke_run_fn_translating_timeout`` supplied for
        this case until it stopped catching the exception (task 4252).

        The class docstring says WHY a timed-out script is an infra fault
        rather than a ``(rc, tail)`` return; it stays the single canonical
        explanation and this method does not restate it.

        ``DeterministicRunner._run_predicate``'s own ``ScriptTimeout`` arm
        shares only ``budget_line``: the CONSEQUENCE differs (a predicate
        timeout produces no milestone VERDICT, and that path's summary and
        category are separately pinned), and all three predicate infra arms
        file the same category, so their remaining wording is the only thing
        telling a human which guard fired.

        The sentences are built from the three measured facts this exception
        carries — budget, the script's own exit code if it had one, and which
        signal the teardown dispatched.

        Args:
            script: the absolute path the runner executed
                (``DeterministicRunner._before_done_paths``), the search key the
                survivor sentence hands the operator when that teardown left the
                script's children running.  Not on the exception: it is the
                deploy's own configuration, not something the timeout measured.
        """
        group_killed = self.teardown is ProcessTeardown.GROUP_KILLED
        lines = [
            self.budget_line('Deploy script'),
        ]
        if self.exit_code is None:
            if self.teardown is ProcessTeardown.NOT_SIGNALLED:
                # This cell is reachable only as a signal death, and the plain
                # "had not exited" spelling below would be false in it:
                # NOT_SIGNALLED means the script was ALREADY reaped when the
                # teardown ran, and a reaped script with no exit code is one that
                # died of a signal (a negative returncode is not an exit code).
                why_no_code = (
                    'the script had already died of a SIGNAL rather than exiting, and '
                    'a signal death produces no exit code'
                )
            else:
                why_no_code = 'the script had not exited when the timeout fired'
            lines.append(
                f'No exit code was produced — {why_no_code}. No output was captured '
                f'either: the merged stdout/stderr read was still in flight. The '
                f'script DID run, so it may have applied PART of its effect: inspect '
                f'out-of-band before resolving.'
            )
        elif group_killed:
            lines.append(
                f'The script ITSELF had already exited with code {self.exit_code}: '
                f'the timeout fired because a process it spawned outlived it, '
                f'still holding the merged stdout/stderr pipe open, and that '
                f'survivor was killed with the group. So the script ran to '
                f'completion — but its exit code never reached the deploy '
                f'classifier, so this deploy is NOT recorded as successful even '
                f'when that code is 0, and no output was captured because the '
                f'read was still in flight. Check out-of-band what the surviving '
                f'child was and whether killing it left the effect half-applied.'
            )
        else:
            lines.append(
                f'The script ITSELF had already exited with code {self.exit_code}: '
                f'the timeout fired because a process it spawned outlived it, '
                f'still holding the merged stdout/stderr pipe open. So the script '
                f'ran to completion — but its exit code never reached the deploy '
                f'classifier, so this deploy is NOT recorded as successful even '
                f'when that code is 0, and no output was captured because the read '
                f'was still in flight.'
            )
        if not group_killed:
            lines.append(_survivor_processes_line(script))
        return lines

    @property
    def summary_phrase(self) -> str:
        """The parenthetical both deploy SUMMARIES carry for this timeout.

        The summary is the headline a human reads first, so it must not assert
        "no exit code" for the case where the script did produce one (see the
        class docstring).  Defined once so the two branch-distinct summaries
        cannot come to characterise the same event differently.
        """
        if self.exit_code is None:
            return 'no exit code'
        return f'script exited rc={self.exit_code}, output pipe held open'


def _survivor_processes_line(script: Path) -> str:
    """The actionable sentence for a teardown that left the script's children
    alive — printed by BOTH deploy arms whenever the whole-group kill was not
    the signal dispatched (task 4252 reviewer finding).

    Deliberately does NOT offer the pid or pgid as something to kill: the
    runner declined to signal that id precisely because it may by now belong
    to an unrelated group, so handing it to a human would be the task-845
    footgun the guard exists to prevent.  The script path is the safe search
    key, and is the one thing the operator needs that this text can supply.
    """
    return (
        f'Whatever the script spawned was therefore NOT killed and may still be '
        f'running — including anything still holding the output pipe open. It '
        f'must be located and killed out-of-band before resolving: search by the '
        f'script path ({script}), NOT by pid, since the pid the runner refused '
        f'to signal may by now belong to an unrelated process group (task 845).'
    )


async def run_script(
    script: Path,
    args: Sequence[str],
    *,
    cwd: Path,
    env_overrides: Mapping[str, str] | None,
    timeout_secs: float,
    reap_grace_secs: float,
) -> tuple[int, str]:
    """Run *script* to completion under *timeout_secs*.

    Adapts ``service_restart.py``'s spawn pattern but AWAITS completion
    (blocking cross-unit deploy — no self-kill risk on this path).  The child
    sees ``os.environ`` overlaid with *env_overrides*, so it keeps PATH, HOME
    and XDG_RUNTIME_DIR; an empty or absent override means full inherit.

    Returns:
        (rc, output_tail) — rc is the process return code; output_tail is
        the last 2000 chars of combined stdout/stderr.  Returned ONLY when
        the script ran to completion; a timeout raises instead (below).

    Raises:
        ScriptTimeout — the script overran *timeout_secs*.  Task 2090's
            Layer-A teardown runs FIRST, before the raise, and the raise
            carries WHICH signal that teardown dispatched — for an
            already-reaped child, none at all.  An infra fault, not a
            verdict; see the ``ScriptTimeout`` docstring for why it is
            deliberately not a ``(1, tail)`` return.
    """
    env = {**os.environ, **env_overrides} if env_overrides else None
    proc = await asyncio.create_subprocess_exec(
        str(script), *args,
        env=env,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    # Capture the pgid NOW, while proc is guaranteed alive and unreaped: with
    # start_new_session=True, pgid == proc.pid by POSIX guarantee.  Reading it
    # later via os.getpgid could hit a recycled pid and signal a stranger's
    # group (task 845) — see shared/src/shared/proc_group.py's module
    # docstring, "Why the caller captures pgid".
    pgid = proc.pid
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_secs)
        tail = (stdout or b'').decode(errors='replace')[-2000:]
        return proc.returncode or 0, tail
    except TimeoutError:
        # Task 2090: the deploy script may spawn grandchildren (e.g.
        # restart-fused-memory.sh --drain forking systemctl/curl/journalctl/
        # sleep/the restarted daemon) that inherit the write end of the
        # merged stdout pipe above.  Killing only this direct child leaves
        # them alive as orphans — still holding that pipe open and still
        # running indefinitely.  start_new_session=True (above) makes this
        # process its own session/group leader so the WHOLE tree can be
        # torn down together instead of leaking orphaned processes.
        #
        # The teardown stays FIRST: task 4065 only changed how the timeout
        # is REPORTED, not the Layer-A guarantee that the process group is
        # dead before this frame unwinds.  `terminate_process_tree` never
        # raises (see its docstring), so the raise below is always reached.
        teardown = await terminate_process_tree(proc, pgid, reap_grace_secs=reap_grace_secs)
        # Both facts below are MEASURED here, not inferred: the script may
        # already have exited while a grandchild held the pipe open, and that
        # same case makes the teardown skip the killpg.  See ScriptTimeout's
        # ``exit_code`` and ``teardown`` fields.  A negative returncode is a
        # signal death and None an incomplete reap: neither is an exit code.
        rc = proc.returncode
        raise ScriptTimeout(
            timeout_secs,
            exit_code=rc if rc is not None and rc >= 0 else None,
            teardown=teardown,
        ) from None


async def terminate_process_tree(
    proc: asyncio.subprocess.Process, pgid: int, *, reap_grace_secs: float,
) -> ProcessTeardown:
    """SIGKILL *proc*'s whole process group, then bound the reap (task 2090).

    The signal choice is
    ``shared/src/shared/proc_group.py::sigkill_process_group``'s: the whole
    group, a direct ``proc.kill()`` when the group kill is refused or
    fails, or nothing once the leader is reaped.  This helper renders that
    choice as a ``ProcessTeardown`` and bounds the reap by
    *reap_grace_secs*, so a process stuck in an uninterruptible
    state cannot hang it (and therefore ``run_script``) forever.
    It never raises, so a timeout branch can always return cleanly.

    Returns:
        Which signal was DISPATCHED (``ProcessTeardown``) — the caller
        builds operator-facing text about the teardown and must state
        what actually happened rather than assume the whole-group kill
        (task 4252).  A caller may NOT conclude from this that anything
        actually DIED: the reap below is bounded and can be abandoned, and
        ``NOT_SIGNALLED``/``DIRECT_KILLED`` mean processes the script
        spawned were deliberately left alone.

    Args:
        proc: the timed-out child, spawned with ``start_new_session=True``
            so its process group is its own, not the orchestrator's.
        pgid: the process group id, which MUST be the value captured
            immediately after that spawn (where ``pgid == proc.pid`` by
            POSIX guarantee), never refreshed via ``os.getpgid``.
            ``shared/src/shared/proc_group.py``'s module docstring ("Why
            the caller captures pgid") is the canonical statement of this
            contract.
        reap_grace_secs: the bound on reaping *proc* after the signal.
    """
    teardown = ProcessTeardown.from_dispatch(sigkill_process_group(proc, pgid))

    try:
        await asyncio.wait_for(proc.wait(), timeout=reap_grace_secs)
    except TimeoutError:
        logger.warning(
            'process %s did not exit within '
            'reap_grace_secs=%s after termination signal — abandoning reap '
            '(process may be unkillable)',
            proc.pid, reap_grace_secs,
        )
    # An abandoned reap deliberately does NOT downgrade the disposition: a
    # signal WAS dispatched, and what the reap would have proved is the
    # separate question the Returns: block tells the caller not to ask of
    # this value.
    return teardown
