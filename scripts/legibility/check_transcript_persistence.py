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

import argparse
import logging
import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

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

from legibility import census_trigger, config, inventory, sampling  # noqa: E402

from orchestrator import session_registry  # noqa: E402

logger = logging.getLogger('legibility.check_transcript_persistence')

DEFAULT_LOOKBACK = timedelta(hours=48)
"""Default registry lookback window. Terminal records are reaped after the
24h TERMINAL_TTL, so a 48h window stays populated while tolerating probe lag."""

DEFAULT_SKEW = timedelta(hours=6)
"""Default clock-skew tolerance for the WEAK mtime fallback window."""

DEFAULT_PROJECTS_ROOT = Path.home() / '.claude' / 'projects'
"""Default transcript archive root — the encoded-cwd dirs Claude Code writes.
Injected in tests so the real ``~/.claude`` tree is never touched."""

DEFAULT_SPAWN_SCRIPT = Path(__file__).resolve().parents[2] / 'skills' / 'spawn' / 'spawn-claude.sh'
"""Default spawn script the ``--check-preventer`` guard reads. Tests always
pass an explicit ``spawn_script_path`` fixture — never the real committed file
(which does not carry the ``CLAUDE_CODE_FORCE_SESSION_PERSISTENCE`` token on
this branch; the preventer lands separately)."""

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


def _first_user_turn(path: Path) -> dict[str, Any] | None:
    """Return *path*'s first non-sidechain/non-meta user-turn record, or None.

    A LIGHT alternative to ``sampling._score_and_find_first_turn`` for the
    STRONG match: that helper must read the WHOLE file to also compute the
    5-class confusion score (tool_error/not_found/self_correct/df_guard/
    interrupt) — all of which the strong match discards. This loop instead
    early-returns on the first matching user record, so matching a record
    against a busy encoded-cwd dir full of large sibling transcripts costs
    O(bytes-until-first-user-turn) per candidate rather than O(filesize)
    (reviewer_comprehensive/efficiency, task 2893 amendment). The
    first-turn selection predicate is identical to the scorer's, so
    ``sampling._first_user_turn_text`` flattens the result the same way.
    Degrades to ``None`` on an unreadable file, mirroring the scorer's
    ``except OSError`` contract.
    """
    try:
        for record in inventory.iter_json_lines(path):
            if (
                record.get('type') == 'user'
                and not record.get('isSidechain')
                and not record.get('isMeta')
            ):
                return record
    except OSError:
        return None
    return None


def _normalize_ws(text: str) -> str:
    """Collapse each run of whitespace to a single space (ends stripped).

    The STRONG match is prompt-prefix containment in a transcript's first
    user turn. Claude Code may reflow a stored prompt's whitespace (wrap long
    lines, normalize newlines) WITHOUT altering its words, which would break a
    raw substring check and FALSE-POSITIVE a present transcript as MISSING.
    Normalizing both the needle and the candidate text means whitespace-only
    differences never mask a real match. This does NOT normalize a prompt that
    was semantically expanded/rewritten before storage (e.g. a slash-command
    template) — that residual false-positive class is named in the escalation
    detail so a human reading the alarm can rule it out.
    """
    return ' '.join(text.split())


def find_matching_transcript(
    record: session_registry.SessionRecord,
    projects_root: Path | str,
    *,
    now: datetime,
    skew: timedelta,
) -> Path | None:
    """Find the transcript file under *projects_root* that belongs to *record*, or None.

    The expected dir is ``projects_root/<encoded-cwd>`` (via
    ``inventory.encode_cwd`` — the same lossy ``/``/``.``/``_`` -> ``-``
    encoding Claude Code itself uses). Missing dir -> ``None``.

    THIS IS THE ONE CONSUMER THAT USES ``encode_cwd`` AS A DIRECT LOOKUP KEY.
    Everywhere else in ``inventory.py`` the encoding is only a cheap superset
    pre-filter over a directory listing, re-checked against the session's real
    ``cwd`` by ``inventory.is_member``; here there is no such backstop. An
    encoder that renders even one character differently from Claude Code
    resolves to a directory that does not exist, this function returns
    ``None``, and :func:`find_missing_transcripts` emits a FALSE-POSITIVE
    "session ran, no transcript" finding — which escalates — naming an
    ``expected_dir`` that was never going to exist. That is why the encoder
    must be exact, and why it is pinned to real on-disk dir names by
    ``test_legibility_inventory.py``'s ``TestEncoderLockstep`` (task 3272,
    which found the ``_`` rule missing and this detector alarming on
    dark-factory's own ``.eval-worktrees/df_task_<N>/`` sessions).

    Matching is PER-SESSION to defeat the same-cwd confound (sibling
    headless-agent transcripts share the encoded-cwd dir):
      - STRONG (usable prompt): the record's prompt prefix is contained
        (whitespace-normalized) in a candidate transcript's first-user-turn
        text — read via the light :func:`_first_user_turn` helper, not the
        full confusion-scorer. This is the dominant path — spawn records carry
        a substantive prompt — so a usable-prompt session with only sibling
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
        # spawn-claude.sh persists record.prompt as the caller's ORIGINAL
        # prompt, BEFORE appending its result-handback trailer, so the prompt
        # prefix is genuinely a PREFIX of what claude received (the trailer is
        # a suffix) — containment, not equality, is the right test. Both sides
        # are whitespace-normalized so a reflowed transcript still matches.
        needle = _normalize_ws(prefix)
        for path in candidates:
            text = _normalize_ws(sampling._first_user_turn_text(_first_user_turn(path)))
            if needle in text:
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
    r'(?:^[ \t]*(?:export[ \t]+)?'
    r'|(?<=[;{}])[ \t]*export[ \t]+)'
    r'CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=(["\']?)1\1(?![0-9])',
    re.MULTILINE,
)
"""Matches ``CLAUDE_CODE_FORCE_SESSION_PERSISTENCE`` set to ``1`` in
``export VAR=1`` / plain ``VAR=1`` / quoted ``VAR="1"`` forms, while rejecting
``=0`` and any longer number (``=10``) via the closing backreference +
no-trailing-digit lookahead.

Two ASSIGNMENT-context alternatives, so a commented-out or documentary
reference (``# do NOT set CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1``) or the
token embedded in a larger identifier (``FOO_CLAUDE_..._PERSISTENCE=1``) does
NOT falsely report the preventer as present:

1. Line start (optional indentation, optional ``export`` prefix) — the
   original anchored form, for a script that sets the var on its own line.
2. Mid-line, immediately after a ``;``, ``{``, or ``}`` boundary (optional
   indentation), with a MANDATORY ``export`` keyword — covers a payload
   built by string concatenation, e.g. ``skills/spawn/spawn-claude.sh``'s
   ``${wm_title_export}export CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1; cd ...``
   (task 2923), where the real export is neither at line start nor preceded
   by whitespace. Requiring ``export`` (not optional, unlike alternative 1)
   on this branch keeps a bare mid-line ``VAR=1`` embedded in prose from
   matching, and the explicit boundary-char lookbehind keeps a ``# ...``
   comment continuation from matching (a space is not a boundary char).
   ``"``/``'`` are deliberately EXCLUDED from the boundary set — including
   them would match a merely-quoted/echoed token, e.g.
   ``echo "export CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1"``, as if it were
   a real export (reviewer_comprehensive/robustness, task 2923 amendment);
   the shipped ``spawn-claude.sh`` payload only ever needs ``;``/``{``/``}``.

No ``\\s*`` after ``=`` — a space there (``VAR= 1``) is not a valid shell
assignment."""


def payload_exports_force_persistence(script_text: str) -> bool:
    """Return True iff *script_text* sets ``CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1``.

    A PURE function over the script text — the detector's opt-in
    (``--check-preventer``) guard that the separately-landing preventer is in
    place. False when the token is absent or explicitly set to ``0``. Never
    reads the real committed ``spawn-claude.sh``; callers pass its text in.
    """
    return bool(_FORCE_PERSISTENCE_RE.search(script_text))


# ---------------------------------------------------------------------------
# escalation — self-contained best-effort escalate_info mirror
# ---------------------------------------------------------------------------
# Mirrors nightly.py's escalate_info shape by CONVENTION, not import: importing
# nightly pulls the whole trickle pipeline at module load. Same JSON-RPC
# tools/call envelope, escalate_info tool, infra_issue/info classification,
# http://localhost:<port>/mcp URL, and best-effort never-raises contract.

_ESCALATION_TOOL_NAME = 'escalate_info'
_ESCALATION_AGENT_ROLE = 'legibility-transcript-check'
_ESCALATION_CATEGORY = 'infra_issue'
_ESCALATION_SEVERITY = 'info'


def _build_escalation_arguments(
    findings: Sequence[MissingTranscript],
    cfg: config.LegibilityConfig,
    *,
    force_persistence_ok: bool | None = None,
) -> dict:
    """Build the ``escalate_info`` MCP tool arguments for a transcript-loss alarm.

    A synthetic ``task_id`` (``legibility-transcript-check-<project_id>``)
    labels the detector as the source since it is a timer-driven probe, not a
    Taskmaster task. The ``summary`` names the count of lost sessions; the
    ``detail`` names each finding's slug and real ``cwd`` (plus expected dir,
    ``start_ts``, exit code, and prompt prefix) so a human can locate every
    lost session. When *force_persistence_ok* is not ``None`` (the preventer
    guard was evaluated), its verdict is appended — a MISSING preventer
    strengthens the diagnosis (the known regression cause is present).
    """
    count = len(findings)
    summary = (
        f'legibility transcript-loss: {count} completed spawn session(s) ran '
        f'with NO matching transcript (project {cfg.project_id})'
    )
    detail_lines = [
        f'{count} completed spawn-launched interactive session(s) in project '
        f'{cfg.project_id!r} produced no plausibly-matching transcript under '
        f'~/.claude/projects — the "session ran, no transcript" regression. '
        f'Per-session (slug / cwd / start_ts / exit_code / expected_dir / prompt):',
    ]
    for finding in findings:
        detail_lines.append(
            f'  - {finding.session_slug}  cwd={finding.cwd}  '
            f'start_ts={finding.start_ts}  exit_code={finding.exit_code}  '
            f'expected_dir={finding.expected_dir}  prompt={finding.prompt_prefix!r}'
        )
    detail_lines.append(
        'NOTE: matching is prompt-prefix containment (whitespace-normalized) in '
        'a transcript first user turn under expected_dir. A session whose prompt '
        'was expanded/rewritten before storage (e.g. a slash-command template, '
        'where record.prompt holds the literal argv rather than the stored text) '
        'is a known false-positive class — inspect expected_dir before acting.'
    )
    if force_persistence_ok is not None:
        verdict = 'present' if force_persistence_ok else 'MISSING'
        detail_lines.append(
            f'preventer guard (CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1 in '
            f'spawn-claude.sh): {verdict}'
        )
    return {
        'task_id': f'legibility-transcript-check-{cfg.project_id}',
        'agent_role': _ESCALATION_AGENT_ROLE,
        'category': _ESCALATION_CATEGORY,
        'severity': _ESCALATION_SEVERITY,
        'summary': summary,
        'detail': '\n'.join(detail_lines),
    }


def _default_poster(url: str, envelope: dict) -> None:
    """Post *envelope* to *url* via a real (lazily-imported) httpx POST.

    ``httpx`` is imported lazily so that importing this module -- for its
    unit-tested pure core -- never needs it, and so the tests can substitute
    a stub for the real POST. It is NOT lazy for availability: httpx is a
    direct dependency of ``shared`` (``shared/pyproject.toml``,
    ``httpx>=0.27``, task 2965), and the systemd unit runs this script under
    ``uv run --frozen --project shared``, so it is always importable in
    production. Mirrors ``nightly._default_poster``. Raises on any
    network/HTTP failure; :func:`post_findings` wraps this best-effort.
    """
    import httpx

    response = httpx.post(
        url,
        json=envelope,
        # Required by the streamable-HTTP MCP transport -- single-sourced
        # in census_trigger (already imported here) so a transport change is
        # a one-line edit, not four lockstep edits with a silent-406 risk.
        headers=census_trigger.MCP_STREAMABLE_HTTP_HEADERS,
        timeout=10.0,
    )
    response.raise_for_status()


def post_findings(
    cfg: config.LegibilityConfig,
    findings: Sequence[MissingTranscript],
    *,
    force_persistence_ok: bool | None = None,
    poster=None,
) -> bool:
    """Best-effort ``escalate_info`` POST for the transcript-loss alarm.

    Posts an MCP ``tools/call`` JSON-RPC envelope for tool ``escalate_info``
    to ``http://localhost:<cfg.escalation_port>/mcp`` (via *poster*, default
    :func:`_default_poster`). NEVER raises: any failure (poster exception,
    network error) is logged as a warning and swallowed, returning ``False``
    — a down escalation server must not mask the finding, since
    :func:`run_check`'s non-zero exit code is the authoritative loud signal
    regardless of whether this POST succeeded. Returns ``True`` on a
    successful post.
    """
    poster_fn = poster if poster is not None else _default_poster
    url = f'http://localhost:{cfg.escalation_port}/mcp'
    arguments = _build_escalation_arguments(
        findings, cfg, force_persistence_ok=force_persistence_ok,
    )
    envelope = {
        'jsonrpc': '2.0',
        'id': 1,
        'method': 'tools/call',
        'params': {'name': _ESCALATION_TOOL_NAME, 'arguments': arguments},
    }
    try:
        poster_fn(url, envelope)
        return True
    except Exception as exc:  # noqa: BLE001 - best-effort, never propagate
        logger.warning(
            'legibility transcript-check: escalation post failed '
            '(best-effort, run still exits non-zero): %s', exc,
        )
        return False


# ---------------------------------------------------------------------------
# run_check — the timer-invocable check + its CheckResult + main() CLI
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckResult:
    """The outcome of one :func:`run_check` pass.

    ``exit_code`` is the authoritative loud signal (0 healthy, 1 = at least
    one lost session OR — under ``--check-preventer`` — a failed preventer
    guard) a systemd/watcher probe surfaces. ``missing`` is the finding list,
    ``escalated`` whether the best-effort POST succeeded,
    ``force_persistence_ok`` the preventer verdict (``None`` when not checked),
    and ``reason`` a human-readable one-line summary.
    """

    exit_code: int
    missing: list[MissingTranscript]
    escalated: bool
    force_persistence_ok: bool | None
    reason: str


def _load_config(
    *, config_path: Path | str | None, project_id: str | None,
) -> config.LegibilityConfig:
    """Load the project's :class:`LegibilityConfig` from *config_path* or *project_id*.

    ``--config`` takes precedence. Resolving a bare *project_id* lazily
    imports ``nightly.resolve_config_path`` — only when needed — so the common
    ``--config`` path never pulls the trickle pipeline (census/codebook/coder/
    digest) at module load. Raises ``ValueError`` if neither is given.
    """
    if config_path is not None:
        return config.load_config(config_path)
    if project_id is not None:
        from legibility import nightly  # lazy: avoid pulling the trickle pipeline
        return config.load_config(nightly.resolve_config_path(project_id))
    raise ValueError('run_check requires config_path or project_id')


def _read_spawn_script(spawn_script_path: Path | str | None) -> str:
    """Read the spawn script text best-effort (empty string on any read error).

    An unreadable script yields ``''`` — for which
    :func:`payload_exports_force_persistence` is ``False``, i.e. the preventer
    guard fails safe (a missing/unreadable spawn script is not evidence the
    preventer is in place)."""
    path = Path(spawn_script_path) if spawn_script_path is not None else DEFAULT_SPAWN_SCRIPT
    try:
        return path.read_text(encoding='utf-8')
    except OSError:
        return ''


def _build_reason(
    missing: Sequence[MissingTranscript],
    force_persistence_ok: bool | None,
    check_preventer: bool,
) -> str:
    """Compose the one-line human-readable :attr:`CheckResult.reason`."""
    parts: list[str] = []
    if missing:
        parts.append(
            f'{len(missing)} completed spawn session(s) ran with NO transcript'
        )
    else:
        parts.append('no lost transcripts detected')
    if check_preventer:
        if force_persistence_ok:
            parts.append('preventer guard present')
        else:
            parts.append(
                'preventer guard FAILED '
                '(CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1 not exported)'
            )
    return '; '.join(parts)


def run_check(
    *,
    config_path: Path | str | None = None,
    project_id: str | None = None,
    fleet_root: Path | str | None = None,
    projects_root: Path | str | None = None,
    now: datetime | None = None,
    lookback: timedelta = DEFAULT_LOOKBACK,
    poster=None,
    check_preventer: bool = False,
    spawn_script_path: Path | str | None = None,
) -> CheckResult:
    """Run one registry↔transcript reconciliation pass and return a :class:`CheckResult`.

    Loads the project config (``config_path`` preferred; else *project_id*
    resolved lazily), enumerates the fleet registry under *fleet_root*
    (default real ``~/.claude/fleet``), and computes the lost-session findings
    scoped to ``cfg.cwd_prefixes`` against *projects_root* (default
    ``~/.claude/projects``). When there are findings, best-effort escalates via
    :func:`post_findings`. When *check_preventer*, additionally evaluates the
    bonus guard on the spawn script (``spawn_script_path`` or the default);
    a failed guard is a loud (non-zero exit) signal even with no lost sessions,
    and — when there are also lost sessions — enriches the escalation detail.

    ``exit_code`` is 1 iff there is at least one lost session OR (under
    *check_preventer*) the guard fails; else 0. Escalation failure never
    changes the exit code (the non-zero code is the authoritative signal).
    """
    if now is None:
        now = datetime.now(UTC)
    cfg = _load_config(config_path=config_path, project_id=project_id)
    root = Path(projects_root) if projects_root is not None else DEFAULT_PROJECTS_ROOT

    records = iter_registry_records(fleet_root=fleet_root)
    missing = find_missing_transcripts(
        records, root, cfg.cwd_prefixes, now=now, lookback=lookback,
    )

    force_persistence_ok: bool | None = None
    if check_preventer:
        force_persistence_ok = payload_exports_force_persistence(
            _read_spawn_script(spawn_script_path)
        )
    preventer_failed = check_preventer and force_persistence_ok is False

    escalated = False
    if missing:
        escalated = post_findings(
            cfg, missing, force_persistence_ok=force_persistence_ok, poster=poster,
        )

    exit_code = 1 if (missing or preventer_failed) else 0
    reason = _build_reason(missing, force_persistence_ok, check_preventer)
    return CheckResult(
        exit_code=exit_code,
        missing=missing,
        escalated=escalated,
        force_persistence_ok=force_persistence_ok,
        reason=reason,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint — what a systemd ``ExecStart`` / operator / watcher runs.

    Mirrors ``nightly.py``'s ``run`` subcommand flags. Exactly one of
    ``--config`` or ``--project-id`` identifies the project. Prints the
    one-line reason (to stderr on a non-zero exit, stdout otherwise) and
    returns :attr:`CheckResult.exit_code`. Uses the default (real httpx)
    poster — a down escalation server is swallowed best-effort, the non-zero
    exit remains the authoritative loud signal.
    """
    parser = argparse.ArgumentParser(
        prog='check_transcript_persistence.py',
        description='Legibility registry<->transcript reconciliation detector (task 2893).',
    )
    parser.add_argument('--config', default=None, help="Path to the project's legibility.yaml.")
    parser.add_argument(
        '--project-id', default=None, help='Project id, resolved via resolve_config_path.',
    )
    parser.add_argument(
        '--projects-root', default=None,
        help='Root dir of encoded session transcripts (default: ~/.claude/projects).',
    )
    parser.add_argument(
        '--fleet-root', default=None,
        help='Fleet session-registry root (default: ~/.claude/fleet).',
    )
    parser.add_argument(
        '--lookback-hours', default=48.0, type=float,
        help='Registry lookback window in hours (default: 48).',
    )
    parser.add_argument(
        '--check-preventer', action='store_true',
        help='Also assert spawn-claude.sh exports CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1.',
    )
    args = parser.parse_args(argv)

    if not args.config and not args.project_id:
        parser.error('requires --config or --project-id')

    result = run_check(
        config_path=args.config,
        project_id=args.project_id,
        projects_root=args.projects_root,
        fleet_root=args.fleet_root,
        lookback=timedelta(hours=args.lookback_hours),
        check_preventer=args.check_preventer,
    )
    print(result.reason, file=sys.stderr if result.exit_code else sys.stdout)
    return result.exit_code


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
