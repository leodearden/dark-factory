"""orchestrator.session_hooks — Claude Code hooks trio (Attention Rail T6).

PRD: plans/session-attention-rail-prd.md T6 (§4.6, §6 G5, §7 T6).

Backs the SessionStart/Notification/Stop hooks wired into
``~/.claude/settings.json`` (via ``skills/spawn/hooks/install-hooks.sh``).
These three hooks fire on *every* Claude Code session, including Leo's
hand-launched ones — SessionStart is how a hand-launched session first gets
captured into the session registry.

This module imports ``orchestrator.session_registry`` read-only (PRD §6 G5:
consumers import the shared record contract, they never re-derive it). All
three hook handlers key their registry record on the Claude Code
``session_id`` delivered on each hook's stdin JSON — the only identity that
is present in, and stable across, all three events for one session.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import os
import sys
import tempfile
from collections.abc import Mapping
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

    Reuses ``session_registry.build_session_slug`` with the hook's
    ``session_id`` as the uniqueness token in place of ``launcher_pid`` --
    ``session_id`` is stable across the SessionStart/Notification/Stop
    events of one Claude Code session, so all three deterministically
    resolve to the same ``record.json``.
    """
    identity = resolve_hook_identity(hook_input, env)
    session_id = str(hook_input.get('session_id') or 'unknown')
    return session_registry.build_session_slug(
        identity.role, identity.project, identity.task_id, session_id
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


def run_session_start(
    hook_input: Mapping[str, Any],
    env: Mapping[str, str],
    root: Path | str | None = None,
) -> session_registry.SessionRecord:
    """SessionStart hook handler: capture or refresh this session's record.

    When no record exists yet at this session's slug, this IS the session's
    first sight (PRD hand-launched-capture signal -- true for every hand-
    launched session, and also for a spawned one if this hook somehow runs
    before a future spawn-claude.sh write), so a RICH record is written:
    role/project/task_id/escalation_id/title/cwd/transcript_path all
    populated from the resolved identity, status=RUNNING. When a record
    already exists under this slug, it is refreshed to RUNNING in place via
    ``refresh_record`` -- every other already-populated field survives
    untouched, and the write bumps the record's mtime heartbeat.
    """
    identity = resolve_hook_identity(hook_input, env)
    slug = hook_session_slug(hook_input, env)
    try:
        session_registry.read_record(slug, root=root)
    except FileNotFoundError:
        cwd = str(hook_input.get('cwd') or os.getcwd())
        launcher_pid_raw = env.get('CLAUDE_SPAWN_LAUNCHER_PID')
        launcher_pid = int(launcher_pid_raw) if launcher_pid_raw else os.getppid()
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
        session_registry.write_record(record, root=root)
        return record
    return session_registry.refresh_record(slug, root=root, status=session_registry.Status.RUNNING)


# ---------------------------------------------------------------------------
# Notification / Stop handlers: refresh status, return the OSC retitle
# ---------------------------------------------------------------------------


def _run_status_refresh_and_retitle(
    hook_input: Mapping[str, Any],
    env: Mapping[str, str],
    root: Path | str | None,
    status: session_registry.Status,
) -> str:
    """Shared refresh-then-retitle body for run_notification/run_stop."""
    identity = resolve_hook_identity(hook_input, env)
    slug = hook_session_slug(hook_input, env)
    record = session_registry.refresh_record(slug, root=root, status=status)
    title = hook_display_title(identity, env, record)
    return osc_retitle_sequence(status, title)


def run_notification(
    hook_input: Mapping[str, Any],
    env: Mapping[str, str],
    root: Path | str | None = None,
) -> str:
    """Notification hook handler: status -> AWAITING_INPUT, return its OSC retitle."""
    return _run_status_refresh_and_retitle(
        hook_input, env, root, session_registry.Status.AWAITING_INPUT
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
    """Absolute path to the in-repo skills/spawn/hooks/ dir, derived from this file's own path."""
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / 'skills' / 'spawn' / 'hooks'


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

    ALWAYS returns 0: a fault in any subcommand's core logic is logged
    loudly (stderr, via the standard logging machinery) and swallowed here
    rather than raised, so a hook fault can never block a Claude Code
    session or turn (mirrors session_registry.main's fail-soft contract).
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
        return 0

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
