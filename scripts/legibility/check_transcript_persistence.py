#!/usr/bin/env python3
"""scripts/legibility/check_transcript_persistence.py — registry↔transcript
reconciliation DETECTOR (task 2893).

Cross-checks fleet session-registry records
(``~/.claude/fleet/sessions/<slug>/record.json``) against
``~/.claude/projects`` transcripts and alarms loudly when a COMPLETED
spawn-launched interactive session — a record WITH a substantive prompt, a
terminal status, and a ``start_ts`` inside a lookback window — produced no
plausibly-matching transcript. This is the detector for the transcript-loss
regression (a suppressed session-persistence path), NOT the preventer: the
``CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1`` fix lands separately. Shipping
the detector means any future silent transcript-loss regression is caught in
~a day of the nightly probe.

Natural home: ``scripts/legibility/`` alongside ``check_trickle_liveness.sh``
/ ``inventory.py`` / ``nightly.py``, whose conventions this module mirrors —
a ``__main__`` sys.path self-bootstrap, dependency-injection seams, and a
``main() -> exit-code`` CLI a systemd/watcher probe can surface.

Registry enumeration composes existing public ``session_registry`` helpers
(``sessions_dir`` + ``read_record``, skipping missing/corrupt like the
reapers) rather than adding an iterator to that heavily-tested orchestrator
contract file — see the plan's design decisions.
"""
from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Self-bootstrap for standalone `python scripts/legibility/check_transcript_persistence.py`
# runs (and any systemd ExecStart invoking this file directly) -- must run
# BEFORE the orchestrator/legibility imports below, since a direct script
# invocation puts only scripts/legibility/ (not scripts/ or orchestrator/src)
# on sys.path. Skipped under pytest/normal package import: __name__ is
# 'legibility.check_transcript_persistence', and the repo-root conftest
# already provides orchestrator/src. Mirrors nightly.py's identical guard,
# extended to also add orchestrator/src for `from orchestrator import ...`.
if __name__ == '__main__':
    _HERE = Path(__file__).resolve()
    sys.path.insert(0, str(_HERE.parent.parent))  # scripts/
    sys.path.insert(0, str(_HERE.parents[2] / 'orchestrator' / 'src'))  # <repo>/orchestrator/src

from legibility import inventory, sampling  # noqa: E402
from orchestrator import session_registry  # noqa: E402

DEFAULT_LOOKBACK = timedelta(hours=48)
"""Default registry lookback window. Terminal records are reaped after the
24h TERMINAL_TTL, so a 48h window stays populated while tolerating probe lag."""


# ---------------------------------------------------------------------------
# Registry enumeration + completed-spawn filtering
# ---------------------------------------------------------------------------

def iter_registry_records(
    fleet_root: Path | str | None = None,
) -> Iterator[session_registry.SessionRecord]:
    """Yield every readable ``SessionRecord`` under the fleet registry.

    Composes ``session_registry.sessions_dir(fleet_root).iterdir()`` +
    ``read_record`` in sorted (directory-name) order, skipping a slug dir
    whose ``record.json`` is missing (``FileNotFoundError``) or unparseable
    (``CorruptSessionRecord``) — the same tolerate-missing/corrupt idiom
    ``reap_stale_records`` uses. *fleet_root* is the root-injection param
    (default ``None`` -> real ``~/.claude/fleet``), so tests never touch the
    real tree. Yields nothing when the sessions dir does not exist.
    """
    base = session_registry.sessions_dir(fleet_root)
    if not base.is_dir():
        return
    for slug_dir in sorted(base.iterdir()):
        if not slug_dir.is_dir():
            continue
        try:
            record = session_registry.read_record(slug_dir.name, root=fleet_root)
        except (FileNotFoundError, session_registry.CorruptSessionRecord):
            continue
        yield record


def _parse_start_ts(value: str | None) -> datetime | None:
    """Parse a record's ISO-8601 ``start_ts`` to a UTC-aware datetime, or None.

    A naive timestamp is assumed UTC. Returns ``None`` for an empty or
    unparseable value — the caller then skips the record (it cannot bound
    the lookback window), tolerating lag rather than alarming.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def find_completed_spawn_records(
    records: Iterable[session_registry.SessionRecord],
    *,
    now: datetime,
    lookback: timedelta,
) -> list[session_registry.SessionRecord]:
    """Filter *records* to COMPLETED spawn sessions worth checking.

    Keeps a record iff ALL hold:
      (a) it has a non-empty (after ``strip``) prompt — spawn records always
          carry a substantive prompt; a promptless record cannot be matched
          per-session and is skipped;
      (b) its status is terminal (``EXITED``/``FAILED_TO_START``) — an
          in-flight session's transcript may still be flushing, so alarming
          on it would false-positive;
      (c) its ``start_ts`` parses and lies within ``[now - lookback, now]``.

    Records failing any condition (in-flight, promptless, out-of-window,
    unparseable ``start_ts``) are skipped defensively.
    """
    window_start = now - lookback
    kept: list[session_registry.SessionRecord] = []
    for record in records:
        if not record.prompt or not record.prompt.strip():
            continue
        if record.status not in session_registry.TERMINAL_STATUSES:
            continue
        start = _parse_start_ts(record.start_ts)
        if start is None:
            continue
        if not (window_start <= start <= now):
            continue
        kept.append(record)
    return kept
