"""orchestrator.session_hooks — Claude Code hooks trio (Attention Rail T6).

PRD: plans/session-attention-rail-prd.md T6 (§4.6, §6 G5, §7 T6).

Backs the SessionStart/Notification/Stop hooks wired into
``~/.claude/settings.json`` (via ``skills/spawn/hooks/install-hooks.sh``).
These three hooks fire on *every* Claude Code session, including Leo's
hand-launched ones — SessionStart is how a hand-launched session first gets
captured into the session registry.

This module imports ``orchestrator.session_registry`` read-only (PRD §6 G5:
consumers import the shared record contract, they never re-derive it). All
three hook handlers key their registry record on ``CLAUDE_SPAWN_SESSION_ID``
(env) when present — spawn-claude.sh's own ``launching``-record slug for a
spawned session — falling back to the Claude Code ``session_id`` delivered
on each hook's stdin JSON for a hand-launched session (see
``hook_session_slug``).

RESOLVED (task 2511): spawned sessions used to get TWO registry records —
this trio's session_id-keyed one and spawn-claude.sh's pid-keyed
``launching``/``exited`` one — because ``session_id`` and ``launcher_pid``
are different uniqueness tokens that can never converge (see
``run_session_start``'s docstring). Now that spawn-claude.sh exports
``CLAUDE_SPAWN_SESSION_ID`` (its own launching record's slug) into the
spawned session's environment, ``hook_session_slug`` adopts it directly, so
all three hooks — plus spawn-claude.sh's own ``launching``/``exit`` writes —
target the SAME record: one record advances
launching -> running -> idle/awaiting-input -> exited (exited is still
written by spawn-claude.sh's ``finish()``). Hand-launched sessions (no
CLAUDE_SPAWN_SESSION_ID) still key on session_id, exactly as before.
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from orchestrator import session_registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hook stdin JSON parsing (tolerant -- every field optional)
# ---------------------------------------------------------------------------


def parse_hook_input(raw: str) -> dict[str, Any]:
    """Tolerantly parse the Claude Code hook stdin JSON.

    Recognized keys: ``session_id``, ``cwd``, ``hook_event_name``,
    ``message``, ``transcript_path`` — every one of them optional. Empty,
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

    Delegates to ``session_registry.parse_spawn_identity``: ``CLAUDE_SPAWN_*``
    env wins when present; otherwise falls back to its documented defaults
    (``role='session'``, ``project=basename(cwd)``) since hook stdin carries
    no title to parse. *cwd* comes from the hook's stdin JSON, falling back
    to ``os.getcwd()`` when absent.
    """
    cwd = str(hook_input.get('cwd') or os.getcwd())
    title = env.get('CLAUDE_SPAWN_TITLE', '') or ''
    prompt = env.get('CLAUDE_SPAWN_PROMPT', '') or ''
    return session_registry.parse_spawn_identity(env, title, prompt, cwd)


def hook_session_slug(
    hook_input: Mapping[str, Any],
    env: Mapping[str, str],
) -> str:
    """Build the record-identity slug for one hook event.

    Prefers ``CLAUDE_SPAWN_SESSION_ID`` (env) when present: that value is
    already spawn-claude.sh's own ``launching``-record slug, so it is
    returned DIRECTLY (sanitized via ``session_registry.sanitize_slug``),
    NOT fed back through ``build_session_slug`` -- doing so would
    double-prefix it (e.g. ``'session-cockpit-session-cockpit-3215033'``)
    and miss the pre-existing launching record. This is what lets all three
    hook events converge on the SAME pid-keyed record spawn-claude.sh
    created (module docstring). A missing, empty, or whitespace-only value
    is treated as absent (mirrors ``_resolve_parent_session_id``'s
    ``env.get(...) or None`` idiom, extended with a ``.strip()`` so a
    blank-but-non-empty token also falls through).

    Otherwise reuses ``session_registry.build_session_slug`` with the
    hook's ``session_id`` as the uniqueness token in place of
    ``launcher_pid`` -- ``session_id`` is stable across the
    SessionStart/Notification/Stop events of one Claude Code session, so
    all three deterministically resolve to the same ``record.json`` (the
    hand-launched fallback).
    """
    spawn_session_id = (env.get('CLAUDE_SPAWN_SESSION_ID') or '').strip() or None
    if spawn_session_id is not None:
        return session_registry.sanitize_slug(spawn_session_id)

    identity = resolve_hook_identity(hook_input, env)
    session_id = str(hook_input.get('session_id') or 'unknown')
    # session_id (str) deliberately fills the launcher_pid slot as the
    # uniqueness token (see module docstring); build_session_slug only ever
    # str()s this argument, so the int annotation is not a real runtime
    # constraint here.
    return session_registry.build_session_slug(
        identity.role,
        identity.project,
        identity.task_id,
        session_id,  # type: ignore[arg-type]
    )


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


def _hand_launched_liveness_pid() -> int:
    """A durable pid-liveness proxy for a hand-launched session's new record.

    ``os.getppid()`` would resolve to THIS hook invocation's own bash
    entrypoint (session-start.sh), which exits within seconds of this call
    returning. ``reap_stale_records``'s ``stale_pid`` rule reclaims a
    non-terminal record once its ``launcher_pid`` is dead AND
    ``NON_TERMINAL_HEARTBEAT_TTL`` has elapsed since its last write -- so a
    hand-launched session that goes idle for over an hour with no further
    hook write would be reaped even though its terminal is still open,
    defeating T6's registry-visibility goal.

    ``os.getsid(0)`` instead resolves to THIS process's POSIX *session
    leader* -- ordinarily the interactive shell (or ``claude`` itself) that
    owns the controlling terminal. Claude Code does not ``setsid`` the hook
    subprocesses it spawns, so they inherit that same session id; the leader
    lives for the terminal's whole lifetime, which is the liveness signal T6
    actually wants. Falls back to ``os.getppid()`` on the rare
    platform/sandbox where ``getsid`` raises.
    """
    try:
        return os.getsid(0)
    except OSError:
        return os.getppid()


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
        return None
    return None


def _resolve_display(env: Mapping[str, str], title: str) -> session_registry.Display | None:
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
    """
    if env.get('TMUX'):
        return session_registry.Display(
            kind=session_registry.DisplayKind.TMUX.value,
            tmux_target=env.get('TMUX_PANE') or None,
        )
    if env.get('WINDOWID'):
        return session_registry.Display(
            kind=session_registry.DisplayKind.WM.value,
            wm_title=title,
            wm_window_id=env.get('WINDOWID'),
        )
    marker = env.get('CLAUDE_SPAWN_WM_TITLE')
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
) -> session_registry.SessionRecord:
    """SessionStart hook handler: capture or refresh this session's record.

    When no record exists yet at this session's slug, this IS the session's
    first sight (PRD hand-launched-capture signal -- true for every hand-
    launched session), so a RICH record is built:
    role/project/task_id/escalation_id/title/cwd/transcript_path all
    populated from the resolved identity, status=RUNNING. When a record
    already exists under this slug, it is refreshed to RUNNING in place --
    every other already-populated field survives untouched. Both paths then
    receive the same best-effort enrichment (parent_session_id from
    ``CLAUDE_SPAWN_PARENT_ID``, see ``_resolve_parent_session_id``; display
    from ``TMUX``/``WINDOWID``, see ``_resolve_display``) before a single
    ``write_record`` call, which bumps the record's mtime heartbeat.

    RESOLVED dual-record split for SPAWNED sessions (task 2511): this slug
    used to be keyed on the Claude Code ``session_id`` (module docstring),
    while spawn-claude.sh's own ``launching`` write keys ITS record on the
    integer ``launcher_pid`` -- a different uniqueness token that
    ``session_id`` could never converge with, since spawn-claude.sh cannot
    know the session_id Claude Code will assign at launch time. A spawned
    session used to get TWO registry records: spawn-claude.sh's pid-keyed
    LAUNCHING/RUNNING/EXITED record, and this hook trio's session_id-keyed
    RUNNING/AWAITING_INPUT/IDLE record -- they did not merge into one,
    regardless of hook-vs-spawn-claude.sh timing. Per the PRD addendum, this
    is now fixed: spawn-claude.sh exports ``CLAUDE_SPAWN_SESSION_ID`` (its
    own launching record's slug) into the spawned session's own
    environment, and ``hook_session_slug`` adopts that token directly (see
    its docstring) -- so this handler's read/refresh/write all target
    spawn-claude.sh's pre-existing record, converging launching -> running
    -> idle/awaiting-input -> exited onto ONE record. Hand-launched sessions
    (no CLAUDE_SPAWN_SESSION_ID) are unaffected -- they still key on
    session_id exactly as before.
    """
    identity = resolve_hook_identity(hook_input, env)
    slug = hook_session_slug(hook_input, env)
    try:
        record = session_registry.read_record(slug, root=root)
    except FileNotFoundError:
        cwd = str(hook_input.get('cwd') or os.getcwd())
        launcher_pid_raw = env.get('CLAUDE_SPAWN_LAUNCHER_PID')
        launcher_pid = int(launcher_pid_raw) if launcher_pid_raw else _hand_launched_liveness_pid()
        record = session_registry.SessionRecord(
            session_slug=slug,
            status=session_registry.Status.RUNNING,
            title=hook_display_title(identity, env, record=None),
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
    parent_session_id = _resolve_parent_session_id(env)
    if parent_session_id is not None:
        record.parent_session_id = parent_session_id
    title = record.title or hook_display_title(identity, env, record)
    display = _resolve_display(env, title)
    if display is not None:
        record.display = display
    session_registry.write_record(record, root=root)
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


def _run_status_refresh_and_retitle(
    hook_input: Mapping[str, Any],
    env: Mapping[str, str],
    root: Path | str | None,
    status: session_registry.Status,
    *,
    question: session_registry.Question | None = None,
) -> str:
    """Shared refresh-then-retitle body for run_notification/run_stop."""
    identity = resolve_hook_identity(hook_input, env)
    slug = hook_session_slug(hook_input, env)
    record = session_registry.refresh_record(slug, root=root, status=status)
    if question is not None:
        record.question = question
        session_registry.write_record(record, root=root)
    title = hook_display_title(identity, env, record)
    return osc_retitle_sequence(status, title)


def run_notification(
    hook_input: Mapping[str, Any],
    env: Mapping[str, str],
    root: Path | str | None = None,
) -> str:
    """Notification hook handler: status -> AWAITING_INPUT, stamp any question, return its OSC retitle."""
    return _run_status_refresh_and_retitle(
        hook_input,
        env,
        root,
        session_registry.Status.AWAITING_INPUT,
        question=_extract_question(hook_input),
    )


def run_stop(
    hook_input: Mapping[str, Any],
    env: Mapping[str, str],
    root: Path | str | None = None,
) -> str:
    """Stop hook handler: status -> IDLE, return its OSC retitle."""
    return _run_status_refresh_and_retitle(hook_input, env, root, session_registry.Status.IDLE)


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
    worktree is reaped (observed live 2026-07-08: worktree 2288 was reaped
    after authoring this trio, orphaning all three hook commands).

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
