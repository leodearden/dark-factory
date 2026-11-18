"""orchestrator.session_registry — durable session-registry substrate (Attention Rail T3).

PRD: plans/session-attention-rail-prd.md T3 (spine root; §4.1-4.2, §4.8, §6 G5).

This module is the fleet registry CLI over the global, cross-project session
registry at ``~/.claude/fleet/sessions/<slug>/record.json``: the launching,
exit, refresh, set-display and reap verbs, the lease verbs, and the decision
verbs through ``decision_cli``. The record schema and its store live in
``session_records``, the leases in ``session_leases``, the sweeps and reaps in
``session_reaper``, and THIS session's identity in ``session_identity``.

It imports only the stdlib, this checkout's stdlib-only ``shared`` leaves and
its stdlib-only siblings, layered so each imports only downward:
``fleet_files``, ``session_identity``, ``session_windows`` <-
``session_records`` / ``session_leases`` / ``session_reaper`` <-
``session_registry``, and ``decision_registry`` <- ``decision_cli`` <-
``session_registry``. Nothing else from the orchestrator, so it can be:

- invoked directly by ``skills/spawn/spawn-claude.sh`` and the SKILL.md verb
  lines via an absolute path (no venv/PYTHONPATH/install required: the
  bootstrap below reaches ``shared/src`` from ``__file__``), and
- imported as ``orchestrator.session_registry`` by downstream Python
  consumers.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

# NOTE: this module is executed by absolute path from
# skills/spawn/spawn-claude.sh and the SKILL.md verb lines with no venv,
# install or PYTHONPATH, so the only non-stdlib imports allowed at module
# scope are this checkout's stdlib-only `shared` leaves, reached through the
# bootstrap below. It APPENDS, so an installed or PYTHONPATH `shared` wins.
# No third-party import, and no unguarded `from orchestrator import <sibling>`:
# running a script by path puts only the script's OWN directory on sys.path,
# so intra-orchestrator imports fail there even though they work for
# consumers that import this module with PYTHONPATH set. Stdlib-only
# siblings are imported through the load-mode block below the shared imports.
# Pinned by the tier-1 (bare shell, NO PYTHONPATH) row of
# _BARE_SHELL_ENTRYPOINTS in
# test_session_registry.py::TestStdlibOnlySelfContainment, and mutation-tested
# there via _MUST_BE_REJECTED.
if (
    _CHECKOUT_SHARED_SRC := str(Path(__file__).resolve().parents[3] / 'shared' / 'src')
) not in sys.path and Path(_CHECKOUT_SHARED_SRC).is_dir():
    sys.path.append(_CHECKOUT_SHARED_SRC)

from shared.session_liveness import capture_for_record

# Siblings are imported relative to orchestrator.session_registry, or else from this
# file's own directory: loaded by path, as tier-1 __main__ or under a foreign dotless
# name, that directory is on sys.path only if Python put it there (not under -P, -I or
# PYTHONSAFEPATH, nor for a foreign-name load), so it is borrowed for these imports
# alone. Each mode is pinned in test_session_registry.py::TestStdlibOnlySelfContainment.
# TYPE_CHECKING points the type checker at the package arm, the only one it can resolve.
if TYPE_CHECKING or __package__:
    from . import decision_cli, session_identity, session_leases, session_reaper, session_records
    from .fleet_files import fleet_root
else:
    _OWN_DIR = str(Path(__file__).resolve().parent)
    _OWN_DIR_BORROWED = _OWN_DIR not in sys.path
    if _OWN_DIR_BORROWED:
        sys.path.append(_OWN_DIR)
    try:
        import decision_cli
        import session_identity
        import session_leases
        import session_reaper
        import session_records
        from fleet_files import fleet_root
    finally:
        if _OWN_DIR_BORROWED:
            sys.path.remove(_OWN_DIR)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project token from cwd: the enclosing git checkout's main working tree
# (reads git's on-disk layout, gitrepository-layout(5); never raises)
# ---------------------------------------------------------------------------


def _enclosing_checkout(path: Path) -> Path | None:
    """The nearest of *path* and its ancestors holding a ``.git`` entry."""
    for candidate in (path, *path.parents):
        if os.path.exists(candidate / '.git'):
            return candidate
    return None


def _read_git_path(pointer_file: Path, prefix: str = '') -> Path | None:
    """The path a git pointer file (gitfile, ``commondir``) holds on its first line.

    The line must start with *prefix*; a relative path is resolved against
    the pointer file's directory and lexically normalised. Unreadable,
    unprefixed or empty -> None.
    """
    try:
        lines = pointer_file.read_text(encoding='utf-8').splitlines()
    except (OSError, ValueError):
        return None
    if not lines or not lines[0].startswith(prefix):
        return None
    raw = lines[0][len(prefix):].strip()
    if not raw:
        return None
    return Path(os.path.normpath(pointer_file.parent / raw))


def _main_working_tree(checkout: Path) -> Path:
    """The main working tree *checkout* belongs to.

    *checkout* itself when its ``.git`` is a directory, or when its gitfile
    does not lead to a common dir named ``.git`` (submodule, bare-backed
    worktree, broken gitfile).
    """
    dotgit = checkout / '.git'
    if os.path.isdir(dotgit):
        return checkout
    gitdir = _read_git_path(dotgit, prefix='gitdir:')
    common_dir = _read_git_path(gitdir / 'commondir') if gitdir is not None else None
    if common_dir is not None and common_dir.name == '.git':
        return common_dir.parent
    return checkout


def _cwd_project_token(cwd: str) -> str:
    """``parse_spawn_identity``'s default project -- see its step 3."""
    checkout = _enclosing_checkout(Path(cwd)) if os.path.isabs(cwd) else None
    if checkout is not None:
        name = _main_working_tree(checkout).name
        if name:
            return name
    return os.path.basename(cwd.rstrip('/')) or 'unknown'


# ---------------------------------------------------------------------------
# CLI + fail-soft (PRD: a registry fault must never change the spawn's exit code)
# ---------------------------------------------------------------------------

_TITLE_RE = re.compile(r'^(?P<role>[^:]+):(?P<project>[^#\s]+)(?:#(?P<task_id>\S+))?')
"""Matches the documented spawn terminal-title convention (skills/spawn/SKILL.md):
``'<role>:<project>#<task-id> <short-slug>'``, task-id and short-slug optional."""


@dataclass(frozen=True)
class SpawnIdentity:
    """Resolved role/project/task_id/escalation_id for a `launching` write."""

    role: str
    project: str
    task_id: str | None
    escalation_id: str | None


def parse_spawn_identity(
    env: Mapping[str, str],
    title: str,
    prompt: str,
    cwd: str,
) -> SpawnIdentity:
    """Resolve spawn identity: env > title-parse > defaults.

    1. ``CLAUDE_SPAWN_ROLE``/``PROJECT``/``TASK_ID``/``ESCALATION_ID`` in *env*,
       when set to a non-empty value.
    2. For whichever of role/project/task_id env did not supply, a best-effort
       parse of the documented terminal-title convention
       ``'<role>:<project>#<task-id> <short-slug>'`` (skills/spawn/SKILL.md).
       The trailing short-slug is always discarded; a project-level title
       (no ``#``) yields ``task_id=None``.
    3. Defaults: ``role='session'``; ``project`` is the basename of the
       enclosing git checkout's MAIN working tree (a linked worktree resolves
       through its gitfile's ``gitdir`` and ``commondir`` to the checkout it
       belongs to), else ``basename(cwd.rstrip('/'))`` or ``'unknown'``. A
       relative *cwd* is not resolved against the process cwd. The token is
       written raw (unfolded); the cockpit folds at its read boundary.

    *prompt* takes no part in this resolution; it is accepted so callers have
    one function for the whole spawn-identity cascade.
    """
    role = env.get('CLAUDE_SPAWN_ROLE') or None
    project = env.get('CLAUDE_SPAWN_PROJECT') or None
    task_id = env.get('CLAUDE_SPAWN_TASK_ID') or None
    escalation_id = env.get('CLAUDE_SPAWN_ESCALATION_ID') or None

    if (role is None or project is None) and title:
        m = _TITLE_RE.match(title.strip())
        if m:
            role = role or m.group('role')
            project = project or m.group('project')
            if task_id is None:
                task_id = m.group('task_id')

    if role is None:
        role = 'session'
    if project is None:
        project = _cwd_project_token(cwd)

    return SpawnIdentity(role=role, project=project, task_id=task_id, escalation_id=escalation_id)


def start_detached_reap(root: Path | str | None = None) -> bool:
    """Start ``session_reaper.run_reap`` on *root*'s fleet in a detached ``reap`` child and return at once.

    The child gets its own session, no inherited fds, DEVNULL stdio and ``/``
    as its cwd, so it can never hold a caller's pipe open nor keep the
    caller's worktree in use while it runs or queues. It inherits
    ``os.environ`` (so ``DISPLAY`` carries over) with ``CLAUDE_FLEET_ROOT``
    pinned to the absolute fleet root. A start failure is logged to stderr
    only and returns False.
    """
    try:
        env = {**os.environ, 'CLAUDE_FLEET_ROOT': str(fleet_root(root).absolute())}
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), 'reap'],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd='/',
            env=env,
            start_new_session=True,
            close_fds=True,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        logger.warning('start_detached_reap: could not start the reap child', exc_info=True)
        return False
    return True


def _parse_end_of_work_inputs(env: Mapping[str, str]) -> tuple[bool, session_records.T3End | None]:
    """Parse (self_exit, t3_end) strictly; an act is set only on the T3 lane, where self-exit means exactly archive."""
    self_exit_raw = env.get('CLAUDE_SPAWN_SELF_EXIT', '') or ''
    t3_end_raw = env.get('CLAUDE_SPAWN_T3_END', '') or ''
    self_exit = self_exit_raw == '1'
    t3_end = session_records.parse_t3_end_input(t3_end_raw)
    if t3_end is not None and self_exit != (t3_end is session_records.T3End.ARCHIVE):
        raise ValueError(
            f'CLAUDE_SPAWN_SELF_EXIT={self_exit_raw!r} contradicts CLAUDE_SPAWN_T3_END={t3_end_raw!r}: '
            'a T3 spawn self-exits exactly when its act is archive'
        )
    return self_exit, t3_end


def _run_launching(env: Mapping[str, str]) -> str:
    """Build + write a LAUNCHING record from CLAUDE_SPAWN_* env; return its dir.

    Every spawn still drives the full mark-then-delete drain
    (``session_reaper.run_reap``),
    now in a detached ``reap`` child (``start_detached_reap``), so the
    launcher never waits on an O(N) registry pass. That child's stdio is
    DEVNULL and a start failure only logs to stderr, so the printed record dir
    stays the only stdout -- spawn-claude.sh captures it into
    ``SESSION_RECORD_DIR``. Concurrent spawns' reaps serialise on
    ``session_reaper.reap_lock``.
    """
    title = env.get('CLAUDE_SPAWN_TITLE', '') or ''
    prompt = env.get('CLAUDE_SPAWN_PROMPT', '') or ''
    cwd = env.get('CLAUDE_SPAWN_CWD') or os.getcwd()
    launcher_pid_raw = env.get('CLAUDE_SPAWN_LAUNCHER_PID')
    launcher_pid = int(launcher_pid_raw) if launcher_pid_raw else os.getpid()
    self_exit, t3_end = _parse_end_of_work_inputs(env)

    identity = parse_spawn_identity(env, title, prompt, cwd)
    slug = session_records.build_session_slug(identity.role, identity.project, identity.task_id, launcher_pid)
    record_dir = session_records.record_path_for_slug(slug).parent

    record = session_records.SessionRecord(
        session_slug=slug,
        status=session_records.Status.LAUNCHING,
        title=title,
        role=identity.role,
        project=identity.project,
        task_id=identity.task_id,
        escalation_id=identity.escalation_id,
        prompt=prompt,
        cwd=cwd,
        launcher_pid=launcher_pid,
        launcher_identity=capture_for_record(launcher_pid),
        start_ts=datetime.now(UTC).isoformat(),
        result_file=str(record_dir / session_records.RESULT_FILENAME),
        transcript_path=session_records.transcript_path_for_cwd(cwd),
        self_exit=self_exit,
        t3_end=t3_end,
    )
    session_records.write_record(record)
    start_detached_reap()
    return str(record_dir)


def slug_root_from_record_dir(record_dir: str) -> tuple[str, Path]:
    """Derive (slug, root) from a record dir printed by `launching`.

    The dir is ``<root>/sessions/<slug>``, so its name is the slug and its
    grandparent is the root -- this is how the `exit`/`refresh` CLI
    subcommands (which only ever receive this printed dir, not a root) find
    their way back to the same session_records.write_record/session_records.read_record key.
    """
    path = Path(record_dir)
    return path.name, path.parent.parent


def _run_exit(record_dir: str, code: int) -> None:
    slug, root = slug_root_from_record_dir(record_dir)
    record = session_records.update_status(slug, root=root, status=session_records.Status.EXITED, exit_code=code)
    session_reaper.drop_t3_thread_pointer_if_stale(record, root)


def _run_refresh(record_dir: str, status_value: str) -> None:
    slug, root = slug_root_from_record_dir(record_dir)
    record = session_records.refresh_record(slug, root=root, status=session_records.Status(status_value))
    if record.status in session_records.TERMINAL_STATUSES:
        session_reaper.drop_t3_thread_pointer_if_stale(record, root)


def _run_set_display(
    record_dir: str,
    kind: str,
    wm_title: str,
    wm_window_id: str | None,
    tmux_target: str | None,
    t3_thread_id: str | None = None,
) -> None:
    slug, root = slug_root_from_record_dir(record_dir)
    session_records.update_display(
        slug,
        root=root,
        kind=kind,
        wm_title=wm_title,
        wm_window_id=wm_window_id,
        tmux_target=tmux_target,
        t3_thread_id=t3_thread_id,
    )


def _run_lease_claim(
    name: str, slug: str, pid: int | None, policy_value: str, *, record_slug: str = ''
) -> None:
    """Run the ``lease-claim`` verb: ALWAYS prints a ``decision=<value>`` line + message.

    *pid* is the claimant's long-lived session pid. Resolving it in code
    rather than in the SKILL.md makes the correct pid STRUCTURAL: the
    ``--pid $$`` defect (task 3994) existed because it depended on every skill
    doc getting one shell token right.

    ON THE CLI PATH IT ARRIVES ALREADY RESOLVED (task 4248): ``main()`` resolves
    the session_identity.SessionIdentity ONCE and derives both the slug and this pid from it, so
    the identity token and the pid written into the lease body cannot
    desynchronise. The ``pid is None`` fallback below is
    therefore a defensive path for a DIRECT caller (a test, a future in-process
    caller), not the CLI's route -- and it stays, so this function remains
    correct standalone rather than depending on a caller it cannot see.
    *record_slug* likewise arrives resolved from ``main()``, from that same
    pid; it is never resolved here, so the fallback path records ''.

    This carries its OWN fail-open guard, independent of main()'s outer
    try/except: a fault raised by claim_lease itself (a corrupt lease body,
    an unwritable leases_dir, ...) must never surface as a silent failure or
    -- worse -- a false stand-down. On any exception here, the caller is
    still told to PROCEED (fail-open), exactly as if the lease were free,
    because a lease-substrate fault must never block a watcher cycle or a
    /unblock session.
    """
    try:
        if pid is None:
            pid = session_identity.resolve_session_pid()
        holder = session_leases.LeaseHolder(
            session_slug=slug,
            pid=pid,
            start_ts=datetime.now(UTC).isoformat(),
            record_slug=record_slug,
        )
        claim = session_leases.claim_lease(name, holder=holder, policy=session_leases.LeasePolicy(policy_value))
    except Exception:
        logger.error('lease-claim %s failed', name, exc_info=True)
        print(f'decision={session_leases.LeaseDecision.PROCEED.value}')
        print(f'lease-claim {name} faulted; proceeding (fail-open)')
        # `slug=` is emitted here too, while `holder_liveness=` deliberately is
        # not: the fault made the HOLDER unknown, not US. This is also where
        # the line is most useful -- the claim's outcome is least certain.
        print(f'slug={slug}')
        return
    print(f'decision={claim.decision.value}')
    print(claim.message)
    # ADDITIVE and LAST (task 3994 defect 4; extended by task 4248): `decision=`
    # stays line 1 and the message line 2, so a skill parser that reads only the
    # first two lines is unaffected -- which is why these are appended lines
    # rather than a fourth LeaseDecision token a not-yet-reloaded skill would
    # face as an unrecognised value. Lease SEMANTICS are unchanged: we never
    # auto-steal. This only lets a stand-down name a machine-readable owner, so
    # a watcher can file a DecisionRecord for a provably-gone holder instead of
    # exiting silently (INV-6/INV-7).
    #
    # TWO appended lines now: `holder_liveness=` (about the CONTENDING HOLDER)
    # and, last, `slug=` (about THIS CALLER). The second exists because the
    # skills no longer pass `--slug` at all: it makes the CLI-derived identity
    # LEGIBLE, so an agent whose later heartbeat returns `result=refused` can
    # cross-check what it claimed as against `lease-show`'s holder_slug. It is a
    # DIAGNOSTIC, explicitly NOT a value for the skills to thread into the next
    # verb -- a shell variable cannot survive to the next Claude Code Bash tool
    # call (each is a fresh `/bin/bash -c`), which is precisely why the
    # derivation had to move into the CLI rather than be printed for quoting.
    #
    # The MUTATING verbs deliberately gain no `slug=` line: `_run_lease_mutation`
    # already names the presented slug alongside the real holder in its refusal
    # message, so a second line would restate what the caller was just told.
    #
    # THREE values, not two: an ACQUIRED claim has no contending holder at all
    # -- the holder is this caller -- so it prints `none`. This first shipped
    # emitting `held` on a successful claim, which reads as "someone else has
    # it": an ambiguous machine-readable token in a change whose entire purpose
    # is removing ambiguous lease signals. The skills only branch on this line
    # under `decision=stand-down`, so `none` narrows the vocabulary rather than
    # breaking a reader. The line is absent only on the fail-open path above,
    # where a substrate fault means we genuinely know nothing about a holder
    # and must not assert one either way.
    #
    # `holder_record=` (task 4237), last of all, is a SECOND, INDEPENDENT orphan
    # axis: the holder's registry record, per HolderRecordState -- the
    # claimant's own on an acquired claim, the existing holder's on a contended
    # one. It deliberately does not change the `orphaned` predicate below: every
    # lease claimed before LeaseHolder.record_slug existed reads `unlinked`, so
    # requiring corroboration would silence today's orphan signal for that whole
    # population. Reading the two lines together is the skills' job; collapsing
    # them into one verdict is what task 3994 defect 3 forbade.
    if claim.acquired:
        print('holder_liveness=none')
        print(f'slug={slug}')
        print(f'holder_record={claim.holder_record_state.value}')
        return
    # A SINGLE signal, deliberately: `orphaned` means exactly "the pid
    # recorded in the lease body is not running". This predicate used to
    # AND in `claim.holder_session_state in ('absent', 'exited')`, which
    # WHENEVER THERE WAS A HOLDER measured as a structural constant 'absent'
    # -- a lease slug is not a session-registry record key (see
    # LeaseHolder.session_slug) -- so dropping it is an honesty fix, not a
    # behaviour change. The one case where that conjunct was NOT constant is
    # an unreadable body, whose holder is None and which therefore resolved
    # 'unknown' -> held; `claim.holder is not None` reproduces that
    # explicitly, on its real reason (no holder means no pid to probe, so
    # there is nothing to call orphaned) rather than as a side effect of a
    # withdrawn lookup. The emitted values are byte-identical before and
    # after, in every branch that HAS a contending holder.
    #
    # It is sound only because session_identity.resolve_session_pid now records the
    # long-lived `claude` pid; the old `--pid $$` was the transient Bash-tool
    # wrapper, always dead, which would have made this a tautology. Both
    # error directions are safe: pid REUSE yields a false `held` (silence,
    # the status quo, never a steal), and a false `orphaned` still triggers
    # only a NON-destructive response (file a DecisionRecord, then exit --
    # never auto-steal, which is the duplicate-spawn incident path). A holder
    # claimed on session_identity.resolve_session_pid's DEGRADED path (pid 0: an unresolvable
    # CLAUDE_PID, or a shared `claude agents` host) therefore reads `orphaned`
    # while possibly alive -- the
    # deliberate, bounded cost of keeping such a lease reapable at all.
    orphaned = claim.holder is not None and not claim.holder_alive
    print(f'holder_liveness={"orphaned" if orphaned else "held"}')
    # THIS CALLER's slug, never the holder's -- see the block comment above.
    print(f'slug={slug}')
    print(f'holder_record={claim.holder_record_state.value}')


def _run_lease_mutation(
    verb: str,
    name: str,
    slug: str,
    mutate: Callable[[], session_leases.LeaseMutation],
) -> None:
    """Drive a mutating lease verb and print ``result=<value>`` + a human line.

    The machine-readable token is ALWAYS the first line, mirroring
    ``lease-claim``'s ``decision=`` so an agent parses all four lease verbs
    with one rule. On REFUSED/FORCED a second line names the lease's actual
    holder, so a caller told "no" also learns WHO owns it and can act
    (inspect with ``lease-show``) rather than guess.

    The holder is read BEFORE *mutate* runs: a FORCED ``lease-release``
    deletes the very file the holder's identity lives in, so reading after
    the fact would report ``<unreadable lease body>`` for exactly the case
    where naming the displaced holder matters most.

    ``--force`` reaches the mutation through *mutate*'s closure and nowhere
    else -- this helper takes no ``force`` parameter, deliberately. It had one
    briefly, unread by any line of the body, which invites the next reader to
    assume the wording below branches on it; the RESULT (FORCED vs REFUSED) is
    the only thing that decides what is printed.
    """
    holder = session_leases.lease_status(name)
    result = mutate()
    print(f'result={result.value}')
    if result not in (session_leases.LeaseMutation.REFUSED, session_leases.LeaseMutation.FORCED):
        return
    holder_slug = holder.holder_slug or '<unreadable lease body>'
    holder_pid = holder.holder_pid if holder.holder_pid is not None else -1
    alive_word = 'alive' if holder.holder_pid_alive else 'not running'
    holder_facts = (
        f'{holder_slug} (pid {holder_pid} {alive_word}, heartbeat {int(holder.heartbeat_age_secs)}s ago)'
    )
    if result is session_leases.LeaseMutation.REFUSED:
        print(
            f'{verb} {name} refused: held by {holder_facts}, not by {slug} '
            f'— inspect with `lease-show --name {name}`'
        )
    else:
        print(f'{verb} {name} FORCED by {slug} over holder {holder_facts}')


def _run_lease_heartbeat(name: str, slug: str, force: bool = False) -> None:
    _run_lease_mutation(
        'lease-heartbeat',
        name,
        slug,
        lambda: session_leases.heartbeat_lease(name, slug=slug, force=force),
    )


def _run_lease_release(name: str, slug: str, force: bool = False) -> None:
    _run_lease_mutation(
        'lease-release',
        name,
        slug,
        lambda: session_leases.release_lease(name, slug=slug, force=force),
    )


def _run_lease_show(name: str) -> None:
    """Run the ``lease-show`` verb: print one lease's full state as key=value lines.

    The replacement for ``cat <lease>`` + ``stat -c %y`` archaeology (task
    3994 defect 3), in the same machine-readable ``<key>=<value>`` form as
    ``decision=``/``result=`` so an agent parses every lease verb with one
    rule. ``state=`` is the discriminator: the holder_* identity lines are
    printed only when the body actually parsed, so an unreadable body reads
    as unreadable rather than as a holder named ``<unknown>``. Booleans are
    rendered lowercase (``true``/``false``) so a shell test is trivial.
    ``holder_record=`` and ``holder_record_slug=`` are ADDITIVE and LAST,
    the convention ``_run_lease_claim`` states.

    Read-only and fail-soft, like lease_status itself: it never mutates the
    lease and never raises.
    """
    status = session_leases.lease_status(name)
    print(f'name={status.name}')
    print(f'state={status.state}')
    if status.state == 'absent':
        return
    if status.holder_slug is not None:
        print(f'holder_slug={status.holder_slug}')
    if status.holder_pid is not None:
        print(f'holder_pid={status.holder_pid}')
    print(f'holder_pid_alive={str(status.holder_pid_alive).lower()}')
    print(f'heartbeat_ts={status.heartbeat_ts}')
    print(f'heartbeat_age_secs={int(status.heartbeat_age_secs)}')
    print(f'reclaimable={str(status.reclaimable).lower()}')
    print(f'holder_record={status.holder_record_state.value}')
    if status.holder_record_slug is not None:
        print(f'holder_record_slug={status.holder_record_slug}')


def _run_lease_reap() -> list[session_leases.ReapedLease]:
    return session_leases.reap_stale_leases()


def _resolve_filing_record_slug(local_id: str, project: str) -> str:
    """Return the record_slug a write-decision filing of *local_id* stamps.

    That is resolve_own_record_slug's slug, else '' after a WARNING, because a
    blank one leaves that cockpit row unable to focus its filer's session for
    the record's life (task 7527).
    """
    slug = session_leases.resolve_own_record_slug()
    if slug is not None:
        return slug
    logger.warning(
        'write-decision --id %s --project %s: $%s=%r resolves to no vouched session record '
        '(unset, unusable or a shared `claude agents` host pid; or no sessions-by-pid/<pid> '
        'pointer whose record still names that pid as its owner), so this decision gets a '
        'BLANK record_slug and its cockpit row cannot focus the filing session (Enter-to-focus) '
        'for the life of the record. This alone does not refuse the filing; an id on stdout is '
        'what says it landed.',
        local_id,
        project,
        session_identity.SESSION_PID_ENV,
        os.environ.get(session_identity.SESSION_PID_ENV),
    )
    return ''


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='session_registry')
    sub = parser.add_subparsers(dest='verb', required=True)

    sub.add_parser('launching', help='write a LAUNCHING record from CLAUDE_SPAWN_* env')

    exit_p = sub.add_parser('exit', help='set status=exited + exit_code on a record')
    exit_p.add_argument('--record', required=True, help='record dir, as printed by `launching`')
    exit_p.add_argument('--code', required=True, type=int)

    refresh_p = sub.add_parser('refresh', help='upsert a record with a new status (hook seam)')
    refresh_p.add_argument('--record', required=True, help='record dir')
    refresh_p.add_argument('--status', required=True, choices=[s.value for s in session_records.Status])

    set_display_p = sub.add_parser(
        'set-display', help='stamp display (kind/title/target) on a record'
    )
    set_display_p.add_argument('--record', required=True, help='record dir, as printed by `launching`')
    set_display_p.add_argument('--kind', required=True, choices=[k.value for k in session_records.DisplayKind])
    set_display_p.add_argument('--tmux-target', default=None)
    set_display_p.add_argument('--wm-title', default='')
    set_display_p.add_argument('--wm-window-id', default=None)
    set_display_p.add_argument('--t3-thread-id', default=None)

    sub.add_parser('reap', help='sweep and remove stale session records')

    lease_claim_p = sub.add_parser('lease-claim', help='claim a single-owner-per-role lease (T7)')
    lease_claim_p.add_argument('--name', required=True, help='lease name, see build_lease_name')
    lease_claim_p.add_argument(
        '--slug',
        default=None,
        help="this claimant's own session_slug; defaults to "
        'default_lease_slug(--name) (`<name>-$CLAUDE_PID`, or '
        '`<name>-$CLAUDE_PID-$CLAUDE_CODE_SESSION_ID` under a shared `claude agents` '
        'host). Only pass this as '
        'an explicit operator override -- the derived value is what every '
        'later lease-heartbeat/lease-release re-derives and matches against',
    )
    lease_claim_p.add_argument(
        '--pid',
        type=int,
        default=None,
        help="this claimant's own long-lived session pid; defaults to "
        'resolve_session_pid() ($CLAUDE_PID, or 0 under a shared `claude agents` host, '
        'whose pid outlives this session). Only pass this as an explicit '
        'operator override -- never `$$`, which is the transient Bash-tool shell. '
        'This is the lease body\'s LIVENESS pid only: it does not supply the '
        "slug's identity (which always derives from $CLAUDE_PID), so --pid alone "
        'does not satisfy an underivable --slug -- lease-heartbeat/lease-release '
        'have no --pid, and a slug only one verb can derive is not an identity',
    )
    lease_claim_p.add_argument(
        '--policy',
        choices=[p.value for p in session_leases.LeasePolicy],
        default=session_leases.LeasePolicy.STAND_DOWN.value,
        help='how to respond when the lease is already held live',
    )

    lease_heartbeat_p = sub.add_parser('lease-heartbeat', help='touch a held lease (heartbeat)')
    lease_heartbeat_p.add_argument('--name', required=True)

    lease_release_p = sub.add_parser('lease-release', help='release a held lease')
    lease_release_p.add_argument('--name', required=True)

    # Ownership (task 3994): both MUTATING verbs act only for the lease's
    # actual holder, identified by slug. AMENDED BY TASK 4248 -- the slug is
    # now DERIVED (default_lease_slug, in main()) rather than required from the
    # caller. That does NOT retract 3994's guard. The defect 3994 fixed was that
    # both verbs mutated UNCONDITIONALLY, with no ownership check at all; the
    # check still runs, and it now compares against a slug derived from THIS
    # caller's own $CLAUDE_PID, so a stranger session derives a different slug
    # and is still `result=refused` with the lease body untouched. What is
    # retired is only the requirement that the token be SPELLED IN SHELL, which
    # is exactly what let it drift: 3994 left three SKILL.md files assembling
    # `<role>-<project>-${CLAUDE_PID:-$PPID}` by hand, and $PPID is measurably
    # not stable across Claude Code tool calls. `required=True` was a forcing
    # function to make the call sites pass a slug at all; the call sites have
    # now been updated, and the derivation is the stronger guarantee -- a caller
    # can no longer MIS-spell a token it never spells. An unresolvable
    # $CLAUDE_PID still exits 2 (see main()); it never falls back silently.
    for _lease_mutation_p in (lease_heartbeat_p, lease_release_p):
        _lease_mutation_p.add_argument(
            '--slug',
            default=None,
            help='the slug you claimed this lease with; a mismatch is refused. '
            'Defaults to default_lease_slug(--name) (`<name>-$CLAUDE_PID`, or '
            '`<name>-$CLAUDE_PID-$CLAUDE_CODE_SESSION_ID` under a shared `claude agents` '
            'host), the same derivation lease-claim uses -- pass it only as an operator '
            'override',
        )
        _lease_mutation_p.add_argument(
            '--force',
            action='store_true',
            help='operator recovery: act even when you are not the holder; logged loudly',
        )

    lease_show_p = sub.add_parser(
        'lease-show',
        help="inspect a lease (holder, pid liveness, heartbeat freshness); "
        'read-only. Use this instead of `cat`, which cannot show freshness',
    )
    lease_show_p.add_argument('--name', required=True, help='lease name, see build_lease_name')

    sub.add_parser('lease-reap', help='sweep and remove stale leases')

    decision_cli.add_verb_parsers(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    ALWAYS returns 0 for a session or lease verb: a fault in its core logic
    is logged loudly (stderr, via the standard logging machinery) and
    swallowed here rather than raised, so a registry fault can never change
    the exit code of the bash caller (spawn-claude.sh) that invokes this
    script. That covers runtime FAULTS only: a malformed invocation exits 2
    from argparse's parse_args() below, before the try/except.

    A decision verb goes to decision_cli.run_verb, which owns the decision
    half of this policy, including why close-decision and reopen-decision
    exit non-zero on a refusal. The filer's record slug it needs is session
    identity, so this module hands it _resolve_filing_record_slug.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ONE derivation point for the lease slug, shared by all three lease verbs
    # (task 4248). It lives HERE, in main(), and not inside _run_lease_claim /
    # _run_lease_mutation, because all three must derive a BYTE-IDENTICAL slug
    # or the whole scheme fails -- a claim the holder's own later heartbeat
    # cannot match is precisely the drift this change removes. One shared point
    # makes that structural instead of a three-way convention nobody enforces.
    # An explicit --slug always wins: it stays the operator's override.
    #
    # ONE IDENTITY RESOLUTION TOO. `lease-claim` both WRITES a liveness pid
    # (into the lease body) and EMBEDS an identity (the derived slug), so both
    # derive from the one session_identity.SessionIdentity resolved here: to default_lease_slug
    # via identity=, and to _run_lease_claim via args.pid
    # (session_identity.lease_liveness_pid). Pinned by
    # test_main_lease_claim_derives_the_body_pid_and_the_slug_pid_from_one_resolution.
    # The two legitimately differ under a shared `claude agents` host (body pid
    # 0, slug carrying the host pid and session id) and under an explicit --pid,
    # which overrides the BODY's liveness pid only, never the slug -- the
    # identity has to stay derivable by the mutating verbs, which have no --pid.
    # The body pid also keys the holder's `record_slug` lookup below.
    #
    # Resolution is DEMAND-DRIVEN, and only lease-claim's liveness pid can log:
    # a degradation WARNING is useful only when it is TRUE of the work being
    # done, and a heartbeat writes no pid.
    if args.verb in ('lease-claim', 'lease-heartbeat', 'lease-release'):
        derives_body_pid = args.verb == 'lease-claim' and args.pid is None
        identity = None
        if args.slug is None or derives_body_pid:
            identity = session_identity.resolve_session_identity()
            if derives_body_pid:
                args.pid = session_identity.lease_liveness_pid(identity)
            if args.slug is None:
                args.slug = session_leases.default_lease_slug(args.name, identity=identity)
        # FAIL LOUDLY, never silently. default_lease_slug returns None only
        # when this session's identity is underivable, and it refuses to
        # synthesize `<name>-0` (or `<name>-<shared host pid>`) because that
        # token is IDENTICAL for two such sessions -- each would then pass the
        # other's _is_lease_owner check, reopening 3994 defect 1. So we stop here.
        #
        # This is NOT a breach of _run_lease_claim's fail-open contract.
        # Fail-open covers lease-SUBSTRATE faults (a corrupt lease body, an
        # unwritable leases_dir) -- conditions the caller cannot fix and must
        # not be blocked by. An underivable slug is a CALLER/ENVIRONMENT INPUT
        # error, exactly the class argparse already exits 2 for, and it is
        # fixable at the call site (`--slug`). Note also that pre-4248 a
        # slug-less lease verb exited 2 as well: the undeterminable case KEEPS
        # its current exit code, and only the derivable case gained a default.
        #
        # parser.error raises SystemExit -- a BaseException -- so it
        # deliberately escapes main()'s `except Exception`, which swallows
        # faults and returns 0. That escape is the point: this must reach the
        # shell as a nonzero exit, not as a silently-successful no-op.
        if not args.slug:
            cause = (
                f'${session_identity.SESSION_PID_ENV}={identity.claude_pid} names a shared '
                f'`claude agents` host, so this session\'s identity needs '
                f'${session_identity.SESSION_ID_ENV}, which is unset '
                'or unusable'
                if identity is not None and identity.shared_host
                else f'${session_identity.SESSION_PID_ENV} is unset or unusable'
            )
            parser.error(
                f'cannot derive the lease slug for --name {args.name}: {cause}, '
                'so this session has no stable identity to claim, '
                'heartbeat or release a lease with. Refusing to invent one -- a synthesized '
                'token would collide with every other degraded session and let each act on '
                "the other's lease. Fix the environment, or pass an explicit "
                '--slug <stable-token> that this session will re-use on every later lease verb. '
                'Note --pid does NOT substitute for --slug: it sets the lease body\'s liveness '
                'pid, not this session\'s identity, and lease-heartbeat/lease-release have no '
                '--pid at all -- only --slug is honoured by all three verbs.'
            )
        if args.verb == 'lease-claim':
            args.record_slug = ''
            if args.pid is not None:
                args.record_slug = session_records.resolve_session_slug_for_pid(args.pid) or ''

    if args.verb in decision_cli.DECISION_VERBS:
        return decision_cli.run_verb(args, filing_record_slug=_resolve_filing_record_slug)

    try:
        if args.verb == 'launching':
            print(_run_launching(os.environ))
        elif args.verb == 'exit':
            _run_exit(args.record, args.code)
        elif args.verb == 'refresh':
            _run_refresh(args.record, args.status)
        elif args.verb == 'set-display':
            _run_set_display(
                args.record,
                args.kind,
                args.wm_title,
                args.wm_window_id,
                args.tmux_target,
                args.t3_thread_id,
            )
        elif args.verb == 'reap':
            session_reaper.run_reap()
        elif args.verb == 'lease-claim':
            _run_lease_claim(
                args.name, args.slug, args.pid, args.policy, record_slug=args.record_slug
            )
        elif args.verb == 'lease-heartbeat':
            _run_lease_heartbeat(args.name, args.slug, args.force)
        elif args.verb == 'lease-release':
            _run_lease_release(args.name, args.slug, args.force)
        elif args.verb == 'lease-show':
            _run_lease_show(args.name)
        elif args.verb == 'lease-reap':
            _run_lease_reap()
        else:
            logger.error('session_registry %s: the parser accepts this verb but no runner handles it', args.verb)
    except Exception:
        logger.error('session_registry %s failed', args.verb, exc_info=True)
        return 0

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
