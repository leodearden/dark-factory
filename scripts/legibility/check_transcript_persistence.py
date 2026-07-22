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

import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
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

DEFAULT_SKEW = timedelta(hours=6)
"""Default clock-skew tolerance for the WEAK mtime fallback window."""

PREFIX_LEN = 200
"""Number of leading prompt characters used as the strong-match needle. A
prefix (not the whole prompt) tolerates a transcript that wraps/suffixes the
prompt while staying specific enough to defeat the same-cwd sibling confound."""

MIN_MATCH_LEN = 32
"""Minimum usable prompt-prefix length. A prompt shorter than this is deemed
too short to match reliably (a short common phrase risks false positives), so
matching falls back to the WEAK file-mtime time window instead."""


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


# ---------------------------------------------------------------------------
# Per-session transcript matching (STRONG prompt-prefix; WEAK mtime fallback)
# ---------------------------------------------------------------------------

def _usable_prompt_prefix(prompt: str | None) -> str | None:
    """Return the strong-match prompt prefix, or None when the prompt is not usable.

    The prefix is ``prompt.strip()[:PREFIX_LEN]``; it is usable only when at
    least ``MIN_MATCH_LEN`` characters long. A None/empty/too-short prompt
    returns ``None`` — the caller then uses the WEAK mtime fallback.
    """
    if not prompt:
        return None
    prefix = prompt.strip()[:PREFIX_LEN]
    if len(prefix) < MIN_MATCH_LEN:
        return None
    return prefix


def find_matching_transcript(
    record: session_registry.SessionRecord,
    projects_root: Path | str,
    *,
    now: datetime,
    skew: timedelta,
) -> Path | None:
    """Find the transcript file under *projects_root* that belongs to *record*, or None.

    The expected dir is ``projects_root/<encoded-cwd>`` (via
    ``inventory.encode_cwd`` — the same lossy ``/``/``.`` -> ``-`` encoding
    Claude Code itself uses). Missing dir -> ``None``.

    Matching is PER-SESSION to defeat the same-cwd confound (sibling
    headless-agent transcripts share the encoded-cwd dir):
      - STRONG (usable prompt): the record's prompt prefix is contained in a
        candidate transcript's first-user-turn text (via the ``sampling``
        helpers). This is the dominant path — spawn records carry a
        substantive prompt — so a usable-prompt session with only sibling
        transcripts is correctly flagged MISSING (returns ``None``).
      - WEAK (prompt too short/empty to match reliably): the first candidate
        whose file mtime lies within ``[start_ts - skew, now + skew]``. Only
        reached when the prompt is NOT usable, so a usable-prompt session
        never falls through to this confound-tolerant path. An unparseable
        ``start_ts`` cannot bound the window, so the fallback yields ``None``.

    Candidates are scanned in sorted order; the first match's ``Path`` is
    returned.
    """
    session_dir = Path(projects_root) / inventory.encode_cwd(record.cwd)
    if not session_dir.is_dir():
        return None
    candidates = sorted(session_dir.glob('*.jsonl'))

    prefix = _usable_prompt_prefix(record.prompt)
    if prefix is not None:
        for path in candidates:
            _counts, first_turn = sampling._score_and_find_first_turn(path)
            text = sampling._first_user_turn_text(first_turn)
            if prefix in text:
                return path
        return None

    # WEAK fallback — prompt too short/empty for a reliable content match.
    start = _parse_start_ts(record.start_ts)
    if start is None:
        return None
    window_start = start - skew
    window_end = now + skew
    for path in candidates:
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        except OSError:
            continue
        if window_start <= mtime <= window_end:
            return path
    return None


# ---------------------------------------------------------------------------
# find_missing_transcripts — the finding list
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MissingTranscript:
    """One "session ran, no transcript" finding.

    Carries the identifying facts the escalation detail names so a human (or
    a follow-up probe) can locate the lost session: its registry slug, real
    ``cwd``, prompt prefix, ``start_ts``, exit code, and the encoded
    ``~/.claude/projects/<enc>`` dir the transcript should have lived in.
    """

    session_slug: str
    cwd: str
    prompt_prefix: str
    start_ts: str
    exit_code: int | None
    expected_dir: Path


def find_missing_transcripts(
    records: Iterable[session_registry.SessionRecord],
    projects_root: Path | str,
    cwd_prefixes: Sequence[str],
    *,
    now: datetime,
    lookback: timedelta,
    skew: timedelta = DEFAULT_SKEW,
) -> list[MissingTranscript]:
    """Return a :class:`MissingTranscript` for every lost session in *records*.

    Scopes the fleet-wide registry to THIS project — a record is checked only
    when it is a completed in-window spawn
    (:func:`find_completed_spawn_records`) AND its real ``cwd`` is a project
    member (``inventory.is_member`` against *cwd_prefixes*, path-component
    semantics that exclude a ``-cockpit`` sibling). For each such record with
    no matching transcript (:func:`find_matching_transcript` returns
    ``None``), one finding is emitted. In-flight, promptless, out-of-window,
    and foreign-project records never produce a finding.
    """
    projects_root = Path(projects_root)
    completed = find_completed_spawn_records(records, now=now, lookback=lookback)
    findings: list[MissingTranscript] = []
    for record in completed:
        if not inventory.is_member(record.cwd, cwd_prefixes):
            continue
        if find_matching_transcript(record, projects_root, now=now, skew=skew) is not None:
            continue
        findings.append(
            MissingTranscript(
                session_slug=record.session_slug,
                cwd=record.cwd,
                prompt_prefix=record.prompt.strip()[:PREFIX_LEN],
                start_ts=record.start_ts,
                exit_code=record.exit_code,
                expected_dir=projects_root / inventory.encode_cwd(record.cwd),
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Bonus preventer guard (pure; opt-in via --check-preventer)
# ---------------------------------------------------------------------------

_FORCE_PERSISTENCE_RE = re.compile(
    r'CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=\s*(["\']?)1\1(?![0-9])'
)
"""Matches ``CLAUDE_CODE_FORCE_SESSION_PERSISTENCE`` set to ``1`` in
``export VAR=1`` / plain ``VAR=1`` / quoted ``VAR="1"`` forms, while rejecting
``=0`` and any longer number (``=10``) via the closing backreference +
no-trailing-digit lookahead."""


def payload_exports_force_persistence(script_text: str) -> bool:
    """Return True iff *script_text* sets ``CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1``.

    A PURE function over the script text — the detector's opt-in
    (``--check-preventer``) guard that the separately-landing preventer is in
    place. False when the token is absent or explicitly set to ``0``. Never
    reads the real committed ``spawn-claude.sh``; callers pass its text in.
    """
    return bool(_FORCE_PERSISTENCE_RE.search(script_text))
