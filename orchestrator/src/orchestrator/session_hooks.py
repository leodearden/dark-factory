"""orchestrator.session_hooks — Claude Code hooks trio (Attention Rail T6).

PRD: plans/session-attention-rail-prd.md T6 (§4.6, §6 G5, §7 T6).

Backs the SessionStart/Notification/Stop hooks wired into
``~/.claude/settings.json`` (via ``skills/spawn/hooks/install-hooks.sh``).
These three hooks fire on *every* Claude Code session, including Leo's
hand-launched ones — SessionStart is how a hand-launched session first gets
captured into the session registry.

This module imports ``orchestrator.session_registry`` read-only (PRD §6 G5:
consumers import the shared record contract, they never re-derive it).

A spawned session's hooks key its record on ``CLAUDE_SPAWN_SESSION_ID``, the
slug of the record spawn-claude.sh created; a session without one (T3-hosted
or hand-launched) keys on a registry pointer, else on its stdin
``session_id``. See ``docs/session-hook-ownership.md`` §Record keying.

IMPORT CONTRACT. ``skills/spawn/hooks/*.sh`` run this file by absolute path
from a plain shell with only ``PYTHONPATH=<repo>/orchestrator/src``: no venv,
no install. ``from orchestrator import ...`` works; a module-scope ``shared``
or third-party import makes every hook die quietly. Pinned and mutation-tested
by the ``session_hooks.py`` row of
``orchestrator/tests/test_session_registry.py::TestStdlibOnlySelfContainment``.
``session_ownership_probes`` carries the same contract under its own row.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from orchestrator import session_registry
from orchestrator.session_ownership_probes import EventProbes, ProcessTree
from orchestrator.session_self_exit import SelfExit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hook stdin JSON parsing (tolerant -- every field optional)
# ---------------------------------------------------------------------------


def parse_hook_input(raw: str) -> dict[str, Any]:
    """Tolerantly parse the Claude Code hook stdin JSON.

    Recognized keys: ``session_id``, ``cwd``, ``hook_event_name``,
    ``message``, ``transcript_path``, ``background_tasks``,
    ``session_crons`` — every one of them optional. Empty,
    malformed, or non-object input all yield ``{}`` rather than raising, so a
    hook fault here can never block a session.
    """
    if not raw or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Identity + slug resolution
# ---------------------------------------------------------------------------


def resolve_hook_identity(
    hook_input: Mapping[str, Any],
    env: Mapping[str, str],
) -> session_registry.SpawnIdentity:
    """Resolve role/project/task_id/escalation_id for one hook event.

    Delegates to ``session_registry.py::parse_spawn_identity``: ``CLAUDE_SPAWN_*``
    env wins when present, else its defaults (``role='session'``, ``project``
    from the checkout enclosing *cwd*). *cwd* is the stdin ``cwd``, else
    ``os.getcwd()``. Callers describing the event's own session (record body,
    OSC retitle) pass the env from ``_own_identity_env``; only the slug
    derivation in ``_resolve_hook_slug`` passes the raw env. See
    ``docs/session-hook-ownership.md`` §Forked-inheritor identity.
    """
    cwd = str(hook_input.get('cwd') or os.getcwd())
    title = env.get('CLAUDE_SPAWN_TITLE', '') or ''
    prompt = env.get('CLAUDE_SPAWN_PROMPT', '') or ''
    return session_registry.parse_spawn_identity(env, title, prompt, cwd)


_SPAWNER_IDENTITY_ENV_KEYS = frozenset({
    'CLAUDE_SPAWN_ROLE',
    'CLAUDE_SPAWN_PROJECT',
    'CLAUDE_SPAWN_TASK_ID',
    'CLAUDE_SPAWN_ESCALATION_ID',
    'CLAUDE_SPAWN_TITLE',
    'CLAUDE_SPAWN_PROMPT',
})
"""``CLAUDE_SPAWN_*`` keys ``parse_spawn_identity``/``hook_display_title`` read.

``_own_identity_env`` strips them on the fork path, where they describe the
spawner. The other spawner-describing keys (``CLAUDE_SPAWN_PARENT_ID``,
``CLAUDE_SPAWN_LAUNCHER_PID``, ``CLAUDE_SPAWN_WM_TITLE``) are read by name at
their own call sites in ``run_session_start``. ``CLAUDE_SPAWN_PROMPT`` takes
no part in identity resolution today and is listed so the set stays complete.
See ``docs/session-hook-ownership.md`` §Forked-inheritor identity.
"""


def _hook_session_id(hook_input: Mapping[str, Any]) -> str:
    """The hook's stdin ``session_id``, stripped -- ``''`` when there is none.

    ONE definition of "a usable Claude Code session_id", shared by the
    ownership probe, the binding stamp and the hand-launched slug builder so
    the three can never disagree about what counts as a discriminator.
    Missing, null, empty and whitespace-only all collapse to ``''``: a blank
    token must fall back to ``'unknown'`` when building a slug, exactly as a
    wholly absent one does, and must never be bound as an owner.
    """
    return str(hook_input.get('session_id') or '').strip()


_OWNER_ONLY_SESSION_START_SOURCES = frozenset({'clear', 'compact'})
"""SessionStart ``source`` values only a process already holding a session can emit.

``/clear`` and compaction re-mint ``session_id`` in place. Membership is
necessary but not sufficient to forgive a re-mint: ``_env_slug_ownership``
also requires the event's owning claude pid to equal the record's
``claude_owner_pid``. ``'resume'`` is excluded because a new nested process
reports it too. See ``docs/session-hook-ownership.md`` §Session-id re-mints.
"""


def _is_owner_only_session_start(hook_input: Mapping[str, Any]) -> bool:
    """Did this event come from a source only the owning process can produce?

    ``source`` is a SessionStart-only stdin field, so this is only ever
    consulted from the SessionStart handler's own call path -- the
    ``allow_remint`` seam on ``_env_slug_ownership``/``_resolve_hook_slug``.
    Notification and Stop stay under the strict bind-once rule rather than
    trusting a field their event shape has no business carrying.
    """
    return (
        str(hook_input.get('source') or '').strip().lower()
        in _OWNER_ONLY_SESSION_START_SOURCES
    )


@dataclass(frozen=True)
class _RecordSnapshot:
    """One tri-state read of ``<root>/sessions/<slug>/record.json``.

    * PRESENT -- ``record`` is the body, ``unreadable`` False.
    * ABSENT -- ``record`` None, ``unreadable`` False. No record exists yet;
      a refresh may create one.
    * UNREADABLE -- ``record`` None, ``unreadable`` True. A record exists but
      could not be read; a refresh must not overwrite it.

    Frozen: one observation, so every decision derived from it saw the same
    bytes. ``slug`` travels with the body so a snapshot is never applied to a
    record it was not read from. See ``docs/session-hook-ownership.md``
    §One read, one decision, one write.
    """

    slug: str
    record: session_registry.SessionRecord | None
    unreadable: bool = False


def _read_record_snapshot(slug: str, root: Path | str | None) -> _RecordSnapshot:
    """Read *slug*'s record once, as a tri-state ``_RecordSnapshot``.

    NEVER raises. ``FileNotFoundError`` maps to ABSENT silently -- an
    ordinary fresh spawn must log nothing -- and every other fault maps to
    UNREADABLE with a WARNING, since a degradation that leaves no trace
    violates the repo's no-silent-fail-soft norm.

    This is the module's single reading seam. Both consumers -- the
    ownership probe and the status refresh -- go through it, and a hook
    event that reads the slug it writes reads it exactly ONCE, the snapshot
    travelling forward on ``_HookSlugResolution``.

    The refresh path needs both the prior status AND the prior binding to
    recognise the launch window (see ``_in_unbound_launch_window``), which
    is why the whole body travels rather than a single field.
    """
    try:
        return _RecordSnapshot(slug, session_registry.read_record(slug, root=root))
    except FileNotFoundError:
        return _RecordSnapshot(slug, None)
    except Exception:
        # A corrupt body, an unreadable fleet root, anything. Callers choose
        # what to do with it (the probe adopts, the refresh declines to
        # synthesize over it) -- but nobody gets to be quiet about it.
        logger.warning(
            'session record read failed for slug %s; treating it as unreadable',
            slug,
            exc_info=True,
        )
        return _RecordSnapshot(slug, None, unreadable=True)


def _env_slug_ownership(
    slug: str,
    hook_input: Mapping[str, Any],
    root: Path | str | None,
    *,
    allow_remint: bool = False,
    probes: EventProbes,
) -> tuple[bool, bool, _RecordSnapshot | None]:
    """Is candidate *slug*, an inherited ``CLAUDE_SPAWN_SESSION_ID`` or a pointer's, this session's own?

    Returns ``(owned, may_bind, snapshot)``:

    * ``owned`` -- adopt the candidate slug (True) or fall through to the
      hand-launched keying (False). False only when ownership is positively
      disproved: a bound ``claude_session_id`` that mismatches, or, on an
      unbound record, an ``owner_ppid_verdict`` of False.
    * ``may_bind`` -- whether this event may stamp its ``session_id`` onto the
      record. False when ownership was adopted but not proven, which leaves
      the record open for its real owner.
    * ``snapshot`` -- the record read made to answer, so an adopting caller
      need not read again. None only when nothing was read.

    Fail-soft: every unresolvable probe adopts, never forks. *allow_remint*
    (SessionStart only) forgives a mismatch from an owner-only ``source`` when
    the owning claude pid matches the record's ``claude_owner_pid``. *probes*
    is this event's ``EventProbes``. See ``docs/session-hook-ownership.md``
    §Owner or inheritor and §Fail-soft direction.
    """
    hook_session_id = _hook_session_id(hook_input)
    if not hook_session_id:
        # No discriminator, and the 'unknown' slug fallback is not a valid binding.
        return True, True, None
    snapshot = _read_record_snapshot(slug, root)
    if snapshot.unreadable:
        # Cannot answer: adopt without binding (logged in _read_record_snapshot).
        return True, False, snapshot
    record = snapshot.record
    if record is None:
        # First sight of this slug: adopt, and bind only on positive proof.
        return True, probes.owner_ppid_verdict() is True, snapshot
    bound = (record.claude_session_id or '').strip()
    if not bound:
        # Unbound: the OWNER_PPID verdict decides. An unproven event (None)
        # adopts but does not bind.
        verdict = probes.owner_ppid_verdict()
        return verdict is not False, verdict is True, snapshot
    if bound == hook_session_id:
        return True, True, snapshot
    # A re-mint is forgiven only for an owner-only SessionStart source
    # presented by the owning process, i.e. the same claude pid.
    if not (allow_remint and _is_owner_only_session_start(hook_input)):
        # Forked onto the hand-launched keying, whose slug embeds this very
        # session_id: binding there is tautological, never a capture.
        return False, True, snapshot
    owner_pid = record.claude_owner_pid
    current_pid = probes.owning_claude_pid()
    if owner_pid is None or current_pid is None:
        # Unprovable, not disproved: adopt.
        return True, True, snapshot
    return owner_pid == current_pid, True, snapshot


def _env_slug_is_owned(
    slug: str,
    hook_input: Mapping[str, Any],
    root: Path | str | None,
    *,
    allow_remint: bool = False,
    probes: EventProbes,
) -> bool:
    """Adopt-or-fork half of ``_env_slug_ownership`` -- see its docstring.

    Kept as the narrow, boolean question for callers that only need the
    identity decision; the binding decision travels separately, through
    ``_resolve_hook_slug``.
    """
    return _env_slug_ownership(slug, hook_input, root, allow_remint=allow_remint, probes=probes)[0]


@dataclass(frozen=True)
class _HookSlugResolution:
    """One hook event's resolved registry identity -- see ``_resolve_hook_slug``.

    Named fields rather than a tuple, so a mis-ordered unpack is a type error
    rather than a silent ownership inversion.

    INVARIANT: ``snapshot is None or snapshot.slug == slug``.

    ``session_id_pointed`` -- adopted through its session-id pointer, so sid
    equality proved ownership and a dead ``claude_owner_pid`` may be reclaimed.
    """

    slug: str
    rejected_env_slug: str | None
    may_bind: bool
    snapshot: _RecordSnapshot | None
    session_id_pointed: bool = False


def _adopt_pointer_slug(
    slug: str | None,
    hook_input: Mapping[str, Any],
    root: Path | str | None,
    *,
    allow_remint: bool,
    probes: EventProbes,
    session_id_pointed: bool,
) -> _HookSlugResolution | None:
    """Adopt pointer-resolved *slug* iff ``_env_slug_ownership`` owns it and its record reads."""
    if slug is None:
        return None
    owned, may_bind, snapshot = _env_slug_ownership(
        slug, hook_input, root, allow_remint=allow_remint, probes=probes
    )
    if not owned or snapshot is None or snapshot.record is None:
        return None
    return _HookSlugResolution(slug, None, may_bind, snapshot, session_id_pointed)


def _pointer_resolution(
    hook_input: Mapping[str, Any],
    root: Path | str | None,
    *,
    allow_remint: bool,
    probes: EventProbes,
) -> _HookSlugResolution | None:
    """The record a registry pointer names for this event, adopted; None falls through.

    The stdin ``session_id``'s pointer first, then the owning claude's pid
    pointer, which survives a re-mint. A pid-pointed record bound to another
    session id is adopted only as a re-mint (``_env_slug_ownership``). See
    ``docs/session-hook-ownership.md`` §Record keying.
    """
    session_id = _hook_session_id(hook_input)
    if not session_id:
        return None
    by_session_id = _adopt_pointer_slug(
        session_registry.resolve_session_slug_for_claude_session_id(session_id, root=root),
        hook_input,
        root,
        allow_remint=allow_remint,
        probes=probes,
        session_id_pointed=True,
    )
    if by_session_id is not None:
        return by_session_id
    pid = probes.owning_claude_pid()
    if pid is None:
        return None
    return _adopt_pointer_slug(
        session_registry.resolve_session_slug_for_pid(pid, root=root),
        hook_input,
        root,
        allow_remint=allow_remint,
        probes=probes,
        session_id_pointed=False,
    )


def _resolve_hook_slug(
    hook_input: Mapping[str, Any],
    env: Mapping[str, str],
    root: Path | str | None,
    *,
    allow_remint: bool = False,
    probes: EventProbes,
) -> _HookSlugResolution:
    """Resolve this hook event's registry identity.

    * ``slug`` -- exactly what ``hook_session_slug`` returns.
    * ``rejected_env_slug`` -- the sanitized ``CLAUDE_SPAWN_SESSION_ID`` when
      one was present but disowned: this event belongs to a nested ``claude``
      being forked onto its own record, and the value names its spawner.
      None on every other path.
    * ``may_bind`` -- ``_env_slug_ownership``'s second bit. Callers that bind
      must honour it.
    * ``snapshot`` -- the ownership probe's own record read, attached ONLY on
      the adopt branch, where the slug read is the slug returned. On the fork
      branch it would describe the spawner's record, not the one returned.

    With no ``CLAUDE_SPAWN_SESSION_ID``, ``_pointer_resolution`` is tried
    before the hand-launched keying.

    *allow_remint* (SessionStart only) and *probes* are forwarded to
    ``_env_slug_ownership``. See ``docs/session-hook-ownership.md``
    §One read, one decision, one write.
    """
    spawn_session_id = (env.get('CLAUDE_SPAWN_SESSION_ID') or '').strip() or None
    rejected: str | None = None
    if spawn_session_id is not None:
        # Sanitize BEFORE probing, so an adversarial token can no more escape
        # sessions_dir when read than when written.
        candidate = session_registry.sanitize_slug(spawn_session_id)
        owned, may_bind, snapshot = _env_slug_ownership(
            candidate,
            hook_input,
            root,
            allow_remint=allow_remint,
            probes=probes,
        )
        if owned:
            # The slug READ is the slug RETURNED, so the snapshot describes
            # exactly the record the caller is about to refresh.
            return _HookSlugResolution(candidate, None, may_bind, snapshot)
        rejected = candidate
    else:
        pointed = _pointer_resolution(hook_input, root, allow_remint=allow_remint, probes=probes)
        if pointed is not None:
            return pointed

    # Raw env on the fork branch too: the slug is an opaque key that all three
    # hook events must derive identically. See docs/session-hook-ownership.md
    # §Forked-inheritor identity.
    identity = resolve_hook_identity(hook_input, env)
    session_id = _hook_session_id(hook_input) or 'unknown'
    # session_id fills the launcher_pid slot as the uniqueness token; the int
    # annotation does not bind, since build_session_slug only str()s it.
    slug = session_registry.build_session_slug(
        identity.role,
        identity.project,
        identity.task_id,
        session_id,  # type: ignore[arg-type]
    )
    # This slug embeds session_id itself, so binding it is tautological.
    # No snapshot: any read above was of a DIFFERENT record than this slug.
    return _HookSlugResolution(slug, rejected, True, None)


def hook_session_slug(
    hook_input: Mapping[str, Any],
    env: Mapping[str, str],
    *,
    root: Path | str | None = None,
    allow_remint: bool = False,
    process_tree: ProcessTree | None = None,
) -> str:
    """Build the record-identity slug for one hook event.

    Returns the sanitized ``CLAUDE_SPAWN_SESSION_ID`` as-is when it is present
    (blank counts as absent) and owned by this session: it is already
    spawn-claude.sh's record slug, and feeding it back through
    ``build_session_slug`` would prefix it twice. Otherwise, including on a
    proven mismatch, returns ``session_registry.build_session_slug`` keyed on
    the stdin ``session_id`` (``'unknown'`` when blank), which is the same for
    SessionStart, Notification and Stop. See ``docs/session-hook-ownership.md``
    §Record keying and §Owner or inheritor.

    Pass *root* so the ownership probe reads the registry the caller writes.
    *allow_remint* is SessionStart-only (see ``_env_slug_ownership``).
    *process_tree* is the tree the probes read; None reads the live one.

    A thin wrapper over ``_resolve_hook_slug`` that discards its rejection,
    binding and snapshot bits, so the handlers call that directly instead.
    """
    probes = EventProbes(env, process_tree)
    return _resolve_hook_slug(hook_input, env, root, allow_remint=allow_remint, probes=probes).slug


def _own_identity_env(
    env: Mapping[str, str],
    resolution: _HookSlugResolution,
) -> Mapping[str, str]:
    """The env this event's own identity (record body, OSC retitle) resolves from.

    The raw hook env, unless ``_resolve_hook_slug`` rejected an inherited
    ``CLAUDE_SPAWN_SESSION_ID``. Then the ``_SPAWNER_IDENTITY_ENV_KEYS`` values
    describe the spawner and are stripped, so identity falls through to this
    session's own defaults. The slug is not derived from this env. See
    ``docs/session-hook-ownership.md`` §Forked-inheritor identity.
    """
    if resolution.rejected_env_slug is None:
        return env
    return {k: v for k, v in env.items() if k not in _SPAWNER_IDENTITY_ENV_KEYS}


def _bind_claude_session_id(
    record: session_registry.SessionRecord,
    hook_input: Mapping[str, Any],
    *,
    allow_rebind: bool = False,
    reclaim_dead_owner: bool = False,
    probes: EventProbes,
) -> bool:
    """Stamp this hook's ``session_id`` and owning claude pid onto an unbound *record*.

    Bind-once: an already-bound record is left alone, and a blank stdin
    ``session_id`` binds nothing (``'unknown'`` is never a valid binding).
    *allow_rebind* lets SessionStart re-bind when ``_is_owner_only_session_start``
    holds, because ``/clear`` re-mints the id inside the owning process; never on
    Notification/Stop. ``claude_owner_pid`` comes from *probes* and is None when
    unresolvable. The first bind also stamps ``claude_launch_session_id``,
    which no rebind touches. *reclaim_dead_owner* lets an already-bound record
    take this event's claude pid via ``_reclaim_dead_owner_pid``. Returns
    whether *record* was mutated. See ``docs/session-hook-ownership.md``
    §Owner or inheritor and §Session-id re-mints.
    """
    hook_session_id = _hook_session_id(hook_input)
    if not hook_session_id:
        return False
    bound = (record.claude_session_id or '').strip()
    if bound == hook_session_id:
        return reclaim_dead_owner and _reclaim_dead_owner_pid(record, probes)
    if bound and not allow_rebind:
        return False
    if not bound and record.claude_launch_session_id is None:
        record.claude_launch_session_id = hook_session_id
    record.claude_session_id = hook_session_id
    # The pid later tells a re-mint inside the owner from a nested claude;
    # None, when unresolvable, reads back as "cannot prove ownership".
    record.claude_owner_pid = probes.owning_claude_pid()
    return True


def _reclaim_dead_owner_pid(record: session_registry.SessionRecord, probes: EventProbes) -> bool:
    """Re-stamp *record*'s ``claude_owner_pid`` with this event's claude pid iff no live process holds it.

    A live recorded owner keeps the record: that is a nested ``claude
    --resume`` of the same session id. Returns whether *record* was mutated.
    """
    pid = probes.owning_claude_pid()
    owner = record.claude_owner_pid
    if pid is None or owner == pid or (owner is not None and session_registry.pid_alive(owner)):
        return False
    record.claude_owner_pid = pid
    return True


def _stamp_session_pointers(
    record: session_registry.SessionRecord,
    probes: EventProbes,
    root: Path | str | None,
    *,
    newly_bound: bool,
) -> None:
    """Point the session id this event just bound, and its claude pid, at the written *record*.

    Each pointer asserts exactly what the written record already proves: the
    session-id pointer only when this event changed the binding, the
    pid pointer whenever ``claude_owner_pid`` is this event's owning claude
    pid. The pid rule covers every lane (adopt, fork, a withheld launch
    window, an unproven adopt, a blank session_id) without branching on them.
    Fail-soft: both registry writers never raise.
    """
    if newly_bound and record.claude_session_id:
        session_registry.write_claude_session_pointer(
            record.claude_session_id, record.session_slug, root=root
        )
    pid = probes.owning_claude_pid()
    if pid is None or record.claude_owner_pid != pid:
        return
    session_registry.write_session_pointer(pid, record.session_slug, root=root)


# ---------------------------------------------------------------------------
# Pure OSC-retitle + display-title helpers (PRD §4.6)
# ---------------------------------------------------------------------------

_GLYPH_BY_STATUS: dict[session_registry.Status, str] = {
    session_registry.Status.RUNNING: '⚙',
    session_registry.Status.AWAITING_INPUT: '⏸ AWAITING',
    session_registry.Status.IDLE: '✅',
}
"""PRD §4.6 default glyph prefixes: running=gear, awaiting-input=pause+label,
idle=check. Statuses this hook trio never emits (LAUNCHING/EXITED/
FAILED_TO_START) have no entry and fall back to no glyph prefix."""


def osc_retitle_sequence(status: session_registry.Status, title: str) -> str:
    """Build the emulator-agnostic terminal-retitle OSC escape sequence.

    Pure string builder -- PRD §4.6: ``\\033]0;<title>\\007``, no konsole
    DBus. *status* selects the glyph prefix (see ``_GLYPH_BY_STATUS``); the
    result is ``'<glyph-prefix> <title>'`` spliced into that OSC template.
    """
    prefix = _GLYPH_BY_STATUS.get(session_registry.Status(status), '')
    label = f'{prefix} {title}' if prefix else title
    return f'\033]0;{label}\007'


def hook_display_title(
    identity: session_registry.SpawnIdentity,
    env: Mapping[str, str],
    record: session_registry.SessionRecord | None = None,
) -> str:
    """Resolve the display title to retitle a hook's terminal tab with.

    Prefers an explicit title -- ``CLAUDE_SPAWN_TITLE`` env, else a non-empty
    ``record.title`` (persisted by an earlier hook in this session, e.g.
    SessionStart) -- and only falls back to a project-derived title
    (``'<role>:<project>#<task_id>'``, or ``'<role>:<project>'`` when there is
    no task_id) matching the documented spawn terminal-title convention
    (skills/spawn/SKILL.md) when neither explicit source is present.
    """
    explicit = env.get('CLAUDE_SPAWN_TITLE') or (record.title if record else '') or ''
    if explicit:
        return explicit
    if identity.task_id:
        return f'{identity.role}:{identity.project}#{identity.task_id}'
    return f'{identity.role}:{identity.project}'


# ---------------------------------------------------------------------------
# SessionStart handler: hand-launched-capture + refresh
# ---------------------------------------------------------------------------


def _resolve_parent_session_id(env: Mapping[str, str]) -> str | None:
    """Resolve the spawning session's slug from ``CLAUDE_SPAWN_PARENT_ID``.

    A hand-launched root has no such env var (or it is blank) and stays
    None -- ``parent_session_id`` encodes ONLY genuine spawn parentage
    (Fleet Cockpit C1, PRD §6.1).
    """
    return env.get('CLAUDE_SPAWN_PARENT_ID') or None


_WM_WINDOW_ID_ATTEMPTS = 5
_WM_WINDOW_ID_RETRY_SLEEP_SECS = 0.2
_WMCTRL_TIMEOUT_SECS = 2


def _wmctrl_list(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Fail-soft default runner for ``_resolve_wm_window_id``.

    Wraps ``subprocess.run(['wmctrl', '-l'], ...)`` and never raises, but
    distinguishes *why* it failed so the caller can react differently:

    - A genuinely-missing ``wmctrl`` binary (``FileNotFoundError``) is a
      permanent failure -- yields the rc=127 sentinel
      ``_resolve_wm_window_id`` short-circuits on (no point retrying a
      binary that will never appear).
    - Any other ``OSError``/``SubprocessError`` -- notably
      ``subprocess.TimeoutExpired`` from a momentarily-hung ``wmctrl`` -- is
      transient, not permanent, so it yields a distinct rc=124 sentinel that
      the retry loop keeps riding out exactly like a nonzero/empty probe.
      Conflating the two would abort all remaining attempts on a single
      slow probe instead of retrying it.
    """
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=_WMCTRL_TIMEOUT_SECS)
    except FileNotFoundError:
        return subprocess.CompletedProcess(argv, returncode=127, stdout='')
    except (OSError, subprocess.SubprocessError):
        return subprocess.CompletedProcess(argv, returncode=124, stdout='')


def _resolve_wm_window_id(
    title: str,
    *,
    run: Callable[[list[str]], subprocess.CompletedProcess[str]] = _wmctrl_list,
    attempts: int = _WM_WINDOW_ID_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> str | None:
    """Best-effort resolve *title*'s live X11 window id via ``wmctrl -l``.

    Terminals map their X11 window asynchronously, so the window may not yet
    appear in ``wmctrl -l`` on the first probe right after spawn -- this
    retries up to *attempts* times, sleeping ``_WM_WINDOW_ID_RETRY_SLEEP_SECS``
    between attempts (never after the last one). Each ``wmctrl -l`` line is
    parsed via ``split(None, 3)`` into ``(window_id, desktop, host, title)`` --
    identical to ``WmBackend.is_alive``'s proven parse
    (cockpit/src/cockpit/backends/wm.py) -- and this returns the window id of
    the line whose title field matches *title* EXACTLY, never as a substring,
    so a short marker can't false-positive against an unrelated longer window
    title. NOTE: this parse is intentionally duplicated (not shared/imported)
    across the orchestrator/cockpit package boundary -- keep the two in sync
    if either changes (see WmBackend.is_alive's matching note).

    Fully fail-soft: a missing ``wmctrl``, a nonzero return code on every
    attempt, no matching line, or any unexpected exception (from *run*,
    parsing, or *sleep*) all return ``None`` rather than raising -- a
    resolution miss must degrade cleanly to ``display=None``, never worse.

    ``_wmctrl_list``'s two failure sentinels are handled distinctly:
    a returncode of 127 (genuinely-missing ``wmctrl`` binary) is a permanent
    failure, so it short-circuits on the first probe instead of paying the
    full *attempts* x sleep cost; a returncode of 124 (transient failure,
    e.g. a ``wmctrl`` invocation that timed out) is treated exactly like any
    other non-matching probe and retried.

    Worst-case added SessionStart latency (this runs synchronously inside
    the hook): the common "window not mapped yet" miss costs
    ``(attempts - 1) * _WM_WINDOW_ID_RETRY_SLEEP_SECS`` of sleeping plus
    *attempts* fast ``wmctrl -l`` calls (~0.8s at the current defaults); the
    pathological case of ``wmctrl`` itself repeatedly timing out costs up to
    ``attempts * _WMCTRL_TIMEOUT_SECS`` on top of that (~10.8s at the current
    defaults) before giving up and leaving ``display=None``.
    """
    try:
        for attempt in range(attempts):
            result = run(['wmctrl', '-l'])
            if result.returncode == 127:
                return None
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    columns = line.split(None, 3)
                    if len(columns) < 4:
                        continue
                    window_id, _desktop, _host, line_title = columns
                    if line_title == title:
                        return window_id
            if attempt < attempts - 1:
                sleep(_WM_WINDOW_ID_RETRY_SLEEP_SECS)
    except Exception:
        logger.warning("wmctrl window-id resolution failed; degrading to display=None", exc_info=True)
        return None
    return None


def _canonical_window_id(raw: str | None) -> str | None:
    """Canonicalize a window id to wmctrl -l's zero-padded ``0x%08x`` form.

    ``int(raw, 0)`` base-autodetects, so a bare DECIMAL ``$WINDOWID`` (what a
    real terminal emulator exports, e.g. ``'52428807'``) and an already-hex
    id (e.g. ``'0x3200007'``) both parse to the same integer, which is then
    reformatted as ``0x%08x`` -- byte-identical to wmctrl -l's own
    ``0x%08lx`` first column (e.g. ``'0x03200007'``). This makes a captured
    id self-consistent with the marker path (``_resolve_wm_window_id``, which
    already returns wmctrl's canonical column) and with every string-consumer
    (the window-gone reaper, dashboards). Fail-soft: a missing (``None``) or
    unparseable id returns ``None`` so the caller can fall back to the raw
    value rather than crashing the hook.

    NOTE: this parse/canonicalize logic is mirrored (duplicated, not
    shared/imported, since cockpit must never import orchestrator -- see
    ``cockpit/src/cockpit/backends/base.py``'s docstring) in
    ``cockpit/src/cockpit/backends/wm.py::_window_id_int``, which keys on the
    identical ``int(s, 0)`` contract (there returning the int; here the
    canonical string). Keep the two in sync.
    """
    try:
        return f'0x{int(raw, 0):08x}'  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None


def _resolve_display(
    env: Mapping[str, str],
    title: str,
    *,
    allow_spawn_marker: bool = True,
) -> session_registry.Display | None:
    """Resolve this session's best-effort focus target from env (Fleet Cockpit C2).

    TMUX takes precedence over WINDOWID (hybrid window model, PRD §3 fork 1):
    a session running inside tmux is best focused via the C4 tmux backend
    even when a WINDOWID is also present (e.g. a terminal emulator hosting
    the tmux client), so ``TMUX`` is checked first. ``tmux_target`` is the
    pane spec from ``TMUX_PANE`` when present, else ``None`` (best-effort,
    never guessed). The 'wm' case carries *title* as ``wm_title`` -- the
    primary focus key (PRD §6.1: wmctrl -a by title) -- alongside the
    best-effort ``WINDOWID`` optimization.

    Third and last, when neither TMUX nor WINDOWID is present (the common
    case for konsole/gnome-terminal spawns, which do not export WINDOWID --
    task 2510 / Fleet Cockpit C10 fix): if ``CLAUDE_SPAWN_WM_TITLE`` is set,
    resolve its live X11 window id via ``_resolve_wm_window_id`` and, on a
    hit, stamp a 'wm' Display carrying that EXACT marker as ``wm_title``
    (never the possibly-since-churned *title* param). Gated strictly on this
    explicit env var -- no fallback to *title* itself -- so a hand-launched
    session (no CLAUDE_SPAWN_WM_TITLE) never invokes wmctrl here, and a
    resolver miss (or no marker at all) returns None exactly like today.

    *allow_spawn_marker* is False on the forked-inheritor path (task 4193
    review): ``CLAUDE_SPAWN_WM_TITLE`` names the SPAWNING session's window,
    so resolving it for a nested ``claude`` would hand that session's row
    the PARENT's window id -- cockpit focus on the nested row would raise
    the parent's terminal, and ``mark_windowless_wm_sessions_exited`` would
    treat the two rows as sharing one window. TMUX/WINDOWID are deliberately
    left alone even there: those are set by the terminal the nested process
    genuinely runs in, not by the spawn namespace.
    """
    if env.get('TMUX'):
        return session_registry.Display(
            kind=session_registry.DisplayKind.TMUX.value,
            tmux_target=env.get('TMUX_PANE') or None,
        )
    if env.get('WINDOWID'):
        # Canonicalize to wmctrl's 0x%08x column form (a real emulator exports
        # $WINDOWID DECIMAL); fall back to the raw value if it won't parse.
        return session_registry.Display(
            kind=session_registry.DisplayKind.WM.value,
            wm_title=title,
            wm_window_id=_canonical_window_id(env.get('WINDOWID')) or env.get('WINDOWID'),
        )
    marker = env.get('CLAUDE_SPAWN_WM_TITLE') if allow_spawn_marker else None
    if marker:
        window_id = _resolve_wm_window_id(marker)
        if window_id is not None:
            return session_registry.Display(
                kind=session_registry.DisplayKind.WM.value,
                wm_title=marker,
                wm_window_id=window_id,
            )
    return None


def run_session_start(
    hook_input: Mapping[str, Any],
    env: Mapping[str, str],
    root: Path | str | None = None,
    *,
    process_tree: ProcessTree | None = None,
) -> session_registry.SessionRecord:
    """SessionStart hook handler: capture or refresh this session's record.

    With no record at the slug yet, this is the session's first sight: a rich
    record is built from the resolved identity with status RUNNING. An
    existing record is refreshed to RUNNING in place, its other fields kept.
    Both then get best-effort enrichment (``parent_session_id`` from
    ``CLAUDE_SPAWN_PARENT_ID``, display from ``_resolve_display`` unless the
    record's is T3's), a binding when ``may_bind`` allows
    (``_bind_claude_session_id``, re-binding on an owner-only ``source``), one
    ``write_record`` and ``_stamp_session_pointers``.

    The record is read once per event: on the adopt path the ownership
    probe's snapshot is reused. An UNREADABLE body is re-read so the fault
    propagates instead of being overwritten (see ``_RecordSnapshot``).

    On the fork path, where an inherited env slug was rejected:

    * ``parent_session_id`` is the rejected slug, i.e. the spawner;
    * ``CLAUDE_SPAWN_WM_TITLE`` is not resolved for display;
    * ``launcher_pid`` is ``EventProbes.nested_claude_liveness_pid``, never an
      inherited ``CLAUDE_SPAWN_LAUNCHER_PID``;
    * identity and title resolve from ``_own_identity_env``.

    *process_tree* is the tree the probes read; None reads the live one. See
    ``docs/session-hook-ownership.md`` §Forked-inheritor identity and
    §One read, one decision, one write.
    """
    probes = EventProbes(env, process_tree)
    resolution = _resolve_hook_slug(
        hook_input, env, root, allow_remint=True, probes=probes
    )
    slug, forked_from, may_bind = (
        resolution.slug,
        resolution.rejected_env_slug,
        resolution.may_bind,
    )
    identity_env = _own_identity_env(env, resolution)
    identity = resolve_hook_identity(hook_input, identity_env)
    # The event's only read of its slug: the probe's snapshot on the adopt path.
    snapshot = resolution.snapshot or _read_record_snapshot(slug, root)
    record = snapshot.record
    if record is None and snapshot.unreadable:
        # Unreadable, not absent: re-read so a corrupt body propagates to
        # main() instead of being overwritten. FileNotFoundError means absent.
        with contextlib.suppress(FileNotFoundError):
            record = session_registry.read_record(slug, root=root)
    if record is None:
        cwd = str(hook_input.get('cwd') or os.getcwd())
        launcher_pid_raw = env.get('CLAUDE_SPAWN_LAUNCHER_PID')
        if forked_from is not None:
            # An inherited CLAUDE_SPAWN_LAUNCHER_PID would be the SPAWNER's,
            # and the terminal's session leader outlives this nested claude.
            launcher_pid = probes.nested_claude_liveness_pid()
        elif launcher_pid_raw:
            launcher_pid = int(launcher_pid_raw)
        else:
            launcher_pid = probes.hand_launched_liveness_pid()
        record = session_registry.SessionRecord(
            session_slug=slug,
            status=session_registry.Status.RUNNING,
            title=hook_display_title(identity, identity_env, record=None),
            role=identity.role,
            project=identity.project,
            task_id=identity.task_id,
            escalation_id=identity.escalation_id,
            cwd=cwd,
            launcher_pid=launcher_pid,
            start_ts=datetime.now(UTC).isoformat(),
            transcript_path=session_registry.transcript_path_for_cwd(cwd),
        )

    record.status = session_registry.Status.RUNNING
    if forked_from is not None:
        # The rejected env slug IS this nested claude's spawner.
        record.parent_session_id = forked_from
    else:
        parent_session_id = _resolve_parent_session_id(env)
        if parent_session_id is not None:
            record.parent_session_id = parent_session_id
    title = record.title or hook_display_title(identity, identity_env, record)
    display = _resolve_display(env, title, allow_spawn_marker=forked_from is None)
    if display is not None and not session_registry.is_t3_record(record):
        record.display = display
    # may_bind, not forked_from, guards the bind: an adopted-but-unproven
    # event refreshes the record without claiming it.
    newly_bound = may_bind and _bind_claude_session_id(
        record,
        hook_input,
        allow_rebind=_is_owner_only_session_start(hook_input),
        reclaim_dead_owner=resolution.session_id_pointed,
        probes=probes,
    )
    session_registry.write_record(record, root=root)
    _stamp_session_pointers(record, probes, root, newly_bound=newly_bound)
    return record


# ---------------------------------------------------------------------------
# Notification / Stop handlers: refresh status, return the OSC retitle
# ---------------------------------------------------------------------------


def _extract_question(hook_input: Mapping[str, Any]) -> session_registry.Question | None:
    """Extract a pending Question from a Notification hook's ``message``.

    Returns None -- leaving any prior ``record.question`` untouched by the
    caller -- when ``message`` is absent or blank/whitespace-only. Latest-
    question semantics: an empty Notification must never clobber a real
    question stamped by an earlier one.
    """
    message = hook_input.get('message')
    if not message or not str(message).strip():
        return None
    return session_registry.Question(text=str(message), asked_at=datetime.now(UTC).isoformat())


_LAUNCH_WINDOW_WITHHOLD_MAX_SECS = 600.0
"""How long an event of unknown provenance is withheld from a LAUNCHING, unbound record.

Nothing else bounds that window. 600s is well past the observed launch-window
p90 and maximum, so only a stuck spawn falls through. See
``docs/session-hook-ownership.md`` §The launch window."""


def _launch_window_withholding_expired(
    prior: session_registry.SessionRecord,
) -> bool:
    """Has *prior* sat in the unbound launch window past the withhold bound?

    A missing or unparseable ``start_ts`` counts as expired: a record with no
    clock cannot be shown to be inside the window, and treating it as inside
    would withhold forever.
    """
    raw = (prior.start_ts or '').strip()
    if not raw:
        return True
    try:
        started = datetime.fromisoformat(raw)
    except ValueError:
        return True
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    return (datetime.now(UTC) - started).total_seconds() > _LAUNCH_WINDOW_WITHHOLD_MAX_SECS


def _in_unbound_launch_window(
    prior: session_registry.SessionRecord | None,
) -> bool:
    """Is *prior* a spawn-created record still LAUNCHING and unbound?

    In that window the record has no ``claude_session_id`` discriminator, so an
    event's provenance is unknown. A bound LAUNCHING record is not in the
    window. This answers only the record-shape question; whether to withhold
    is ``_withhold_from_launching``'s decision.
    """
    if prior is None or prior.status is not session_registry.Status.LAUNCHING:
        return False
    return not (prior.claude_session_id or '').strip()


def _withhold_from_launching(
    prior: session_registry.SessionRecord,
    *,
    probes: EventProbes,
) -> bool:
    """Given a record in the unbound launch window, withhold this event?

    No when ``owner_ppid_verdict`` positively identifies the owner, or when the
    window has expired (``_launch_window_withholding_expired``); otherwise the
    event's provenance is unknowable, so yes. *probes* is the same observation
    the adopt/fork decision used. See ``docs/session-hook-ownership.md``
    §The launch window.
    """
    if probes.owner_ppid_verdict() is True:
        return False
    return not _launch_window_withholding_expired(prior)


def _run_status_refresh_and_retitle(
    hook_input: Mapping[str, Any],
    env: Mapping[str, str],
    root: Path | str | None,
    status: session_registry.Status,
    *,
    question: session_registry.Question | None = None,
    process_tree: ProcessTree | None,
    self_exit: SelfExit | None = None,
) -> str:
    """Shared refresh-then-retitle body for ``run_notification``/``run_stop``.

    Resolves the record through the ownership check, so a nested inheritor
    refreshes its own record, not its spawner's. One read, one decision, one
    write: the event reads its slug once (the probe's snapshot on the adopt
    path), decides the launch-window withhold from that same snapshot, and
    issues one ``write_record`` carrying status, question and binding.

    An event that ``_withhold_from_launching`` holds back from a record in the
    unbound launch window writes only the mtime heartbeat: no status, question
    or binding. Every lane, the UNREADABLE fault included, goes through the one
    shared withhold/refresh/bind/question/write block; do not give a lane its
    own copy. ``_stamp_session_pointers`` follows every write. Returns the OSC
    retitle for *status*.

    *self_exit* (Stop only) is evaluated on an owner event, the event that may
    bind; its stamp rides the one write and its action follows the pointer.

    See ``docs/session-hook-ownership.md`` §One read, one decision, one write
    and §The launch window.
    """
    probes = EventProbes(env, process_tree)
    resolution = _resolve_hook_slug(hook_input, env, root, probes=probes)
    slug, may_bind = resolution.slug, resolution.may_bind
    # The event's only read of its slug: the probe's snapshot on the adopt
    # path, a first read on every other.
    snapshot = resolution.snapshot or _read_record_snapshot(slug, root)

    prior = snapshot.record
    if prior is None and snapshot.unreadable:
        # Unreadable, not absent: re-read, then fall through to the shared
        # block so the withhold is still decided. A corrupt body propagates to
        # main(); FileNotFoundError means absent.
        with contextlib.suppress(FileNotFoundError):
            prior = session_registry.read_record(slug, root=root)
    # Decided from the same snapshot as the body written; status=None makes
    # the refresh a heartbeat-only bump. `prior is not None` narrows the type.
    in_launch_window = (
        prior is not None
        and _in_unbound_launch_window(prior)
        and _withhold_from_launching(prior, probes=probes)
    )
    record = session_registry.apply_refresh(
        slug, prior, status=None if in_launch_window else status
    )
    # Status, question, binding and the self-exit stamp land in one body. A
    # withheld event never acts as the owner, nor does an adopted-but-unproven
    # one (may_bind False).
    owner_event = not in_launch_window and may_bind
    newly_bound = owner_event and _bind_claude_session_id(
        record, hook_input, reclaim_dead_owner=resolution.session_id_pointed, probes=probes
    )
    if question is not None and not in_launch_window:
        record.question = question
    evaluation = (
        self_exit.evaluate(record, hook_input, owner_pid=probes.owning_claude_pid(), root=root)
        if self_exit is not None and owner_event
        else None
    )
    if evaluation is not None:
        evaluation.stamp(record)
    # Unconditional: the heartbeat is owed on every event, a withheld one too.
    session_registry.write_record(record, root=root)
    _stamp_session_pointers(record, probes, root, newly_bound=newly_bound)
    if self_exit is not None and evaluation is not None:
        self_exit.enact(evaluation, record, root)
    identity_env = _own_identity_env(env, resolution)
    identity = resolve_hook_identity(hook_input, identity_env)
    title = hook_display_title(identity, identity_env, record)
    return osc_retitle_sequence(status, title)


def run_notification(
    hook_input: Mapping[str, Any],
    env: Mapping[str, str],
    root: Path | str | None = None,
    *,
    process_tree: ProcessTree | None = None,
) -> str:
    """Notification hook handler: status -> AWAITING_INPUT, stamp any question, return its OSC retitle.

    Resolves the record through the ownership check (task 4193): a nested
    inheritor's question lands on its own record, never on the spawning
    session's. *process_tree* is the tree this event's ownership probes
    read; None reads the live one (see ``ProcessTree``).
    """
    return _run_status_refresh_and_retitle(
        hook_input,
        env,
        root,
        session_registry.Status.AWAITING_INPUT,
        question=_extract_question(hook_input),
        process_tree=process_tree,
    )


def run_stop(
    hook_input: Mapping[str, Any],
    env: Mapping[str, str],
    root: Path | str | None = None,
    *,
    process_tree: ProcessTree | None = None,
    self_exit: SelfExit | None = None,
) -> str:
    """Stop hook handler: status -> IDLE, return its OSC retitle.

    Resolves the record through the ownership check (task 4193): a nested
    inheritor finishing its turn can no longer idle the spawning session
    mid-turn. *process_tree* is the tree this event's ownership probes
    read; None reads the live one (see ``ProcessTree``). *self_exit* is the
    opt-in self-exit decision; None uses ``SelfExit()``, which signals with
    the live ``os.kill``.
    """
    return _run_status_refresh_and_retitle(
        hook_input,
        env,
        root,
        session_registry.Status.IDLE,
        process_tree=process_tree,
        self_exit=SelfExit() if self_exit is None else self_exit,
    )


# ---------------------------------------------------------------------------
# Settings merge (MERGE-not-clobber, PRD §4 decision 5)
# ---------------------------------------------------------------------------

_HOOK_SCRIPTS: dict[str, str] = {
    'SessionStart': 'session-start.sh',
    'Notification': 'notification.sh',
    'Stop': 'stop.sh',
}
_HOOK_TIMEOUT_SECS = 10
"""Mirrors ~/.claude/hooks/worktree-hookspath-{capture,restore}.sh's own timeout."""


def merge_hook_settings(settings: Mapping[str, Any], script_dir: str | Path) -> dict[str, Any]:
    """Idempotently merge the SessionStart/Notification/Stop trio into *settings*.

    Deep-copies *settings* and adds ONLY the three event keys under
    ``hooks`` that are not already present, each pointing (by absolute path
    into *script_dir*) at its matching entrypoint script. Every existing
    event key (``PreToolUse``/``PostToolUse``/...) and every other top-level
    key (``env``/``permissions``/``statusLine``/...) is left byte-identical
    -- *settings* itself is never mutated. Safe to call twice: an event key
    already present (from a prior merge) is left untouched, not duplicated.
    """
    merged = copy.deepcopy(dict(settings))
    hooks = merged.setdefault('hooks', {})
    script_dir_path = Path(script_dir)
    for event, script_name in _HOOK_SCRIPTS.items():
        if event in hooks:
            continue
        hooks[event] = [
            {
                'matcher': '*',
                'hooks': [
                    {
                        'type': 'command',
                        'command': str(script_dir_path / script_name),
                        'timeout': _HOOK_TIMEOUT_SECS,
                    }
                ],
            }
        ]
    return merged


def _hooks_script_dir() -> Path:
    """Absolute path to the in-repo skills/spawn/hooks/ dir.

    Resolves to the PRIMARY checkout's ``skills/spawn/hooks/``, even when
    this file is itself executing from inside a linked git worktree (e.g.
    ``.worktrees/<id>/``). Otherwise the ephemeral worktree's path gets
    baked into the absolute hook commands written into
    ``~/.claude/settings.json``, and every hook breaks the moment that
    worktree is reaped.

    Runs ``git -C <this file's dir> rev-parse --path-format=absolute
    --git-common-dir``: the *common* git dir always points at the primary
    repo's ``.git``, even from a linked worktree, so its parent is the
    primary checkout root. Falls back to the previous
    ``Path(__file__).resolve().parents[3]`` behaviour whenever git is
    absent, this isn't a git checkout, the command errors, or the resolved
    directory doesn't actually contain the three hook scripts -- install
    must stay robust and never raise.
    """
    fallback = Path(__file__).resolve().parents[3] / 'skills' / 'spawn' / 'hooks'
    file_dir = Path(__file__).resolve().parent
    try:
        result = subprocess.run(
            ['git', '-C', str(file_dir), 'rev-parse', '--path-format=absolute', '--git-common-dir'],
            capture_output=True,
            text=True,
            timeout=_HOOK_TIMEOUT_SECS,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return fallback

    git_common_dir = result.stdout.strip()
    if not git_common_dir:
        return fallback

    primary_root = Path(git_common_dir).resolve().parent
    candidate = primary_root / 'skills' / 'spawn' / 'hooks'
    if all((candidate / script_name).is_file() for script_name in _HOOK_SCRIPTS.values()):
        return candidate
    return fallback


_DEFAULT_SETTINGS_PATH = Path.home() / '.claude' / 'settings.json'


def _run_install(settings_path: str | Path | None = None) -> None:
    """Merge the hooks trio into the real ``~/.claude/settings.json`` (or *settings_path*).

    Reads the current settings (a missing file reads as ``{}``), merges in
    the trio via ``merge_hook_settings``, writes a timestamped ``.bak`` of
    the pre-merge bytes when the file already existed, then atomically
    replaces the settings file (tmp file in the same dir + ``os.replace``,
    mirroring ``session_registry.write_record``) so a crash mid-write can
    never leave it truncated or corrupt.
    """
    path = Path(settings_path) if settings_path is not None else _DEFAULT_SETTINGS_PATH
    try:
        raw = path.read_text()
    except FileNotFoundError:
        raw = ''
    settings = json.loads(raw) if raw.strip() else {}

    merged = merge_hook_settings(settings, _hooks_script_dir())

    if raw:
        stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
        path.with_name(f'{path.name}.{stamp}.bak').write_text(raw)

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(suffix='.tmp', prefix=path.stem, dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(merged, f, indent=2)
            f.write('\n')
        os.replace(tmp_path_str, str(path))
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path_str)
        raise


# ---------------------------------------------------------------------------
# CLI + fail-soft (PRD: a hook fault must never block a session or turn)
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='session_hooks')
    sub = parser.add_subparsers(dest='verb', required=True)
    sub.add_parser('session-start', help='SessionStart hook: capture/refresh the record (status=running)')
    sub.add_parser('notification', help='Notification hook: status=awaiting-input, print the OSC retitle')
    sub.add_parser('stop', help='Stop hook: status=idle, print the OSC retitle')
    install_p = sub.add_parser('install', help='merge the hooks trio into ~/.claude/settings.json')
    install_p.add_argument(
        '--settings-path',
        default=None,
        help='override the settings.json path (default: ~/.claude/settings.json)',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    The three hook verbs (``session-start``/``notification``/``stop``)
    ALWAYS return 0: a fault in their core logic is logged loudly (stderr,
    via the standard logging machinery) and swallowed here rather than
    raised, so a hook fault can never block a Claude Code session or turn
    (mirrors session_registry.main's fail-soft contract).

    ``install`` is different: it is a human-invoked one-shot command, not a
    per-event hook, so a fault is logged AND propagated as exit code 1 --
    otherwise a failed install (e.g. a permission error writing
    settings.json) would report success via exit 0 while having written
    nothing, and a non-interactive caller (install-hooks.sh) could never
    detect it.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.verb in ('session-start', 'notification', 'stop'):
            hook_input = parse_hook_input(sys.stdin.read())
            env = os.environ
            if args.verb == 'session-start':
                run_session_start(hook_input, env)
            elif args.verb == 'notification':
                print(run_notification(hook_input, env))
            else:
                print(run_stop(hook_input, env))
        elif args.verb == 'install':
            _run_install(args.settings_path)
    except Exception:
        logger.error('session_hooks %s failed', args.verb, exc_info=True)
        return 1 if args.verb == 'install' else 0

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
