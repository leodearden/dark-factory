"""Chronic pool-infra flake auto-file detector (task 2358).

Policy (Leo, 2026-07-08, verify-flakiness survey follow-up): after a verify
completes, detect a chronic pool-infra flake and auto-file a medium-priority
De-flake fix task into the project's task tree — non-blocking (the gate
stays green; retry-once already absorbed the failure) — so the flake debt
becomes visible, owned work instead of a warning nobody reads.

Substrate (reify task 5142, lands separately, cross-project runtime — NOT a
DF-code dependency): reify's ``tests/infra/run_all.sh`` persists every
serial-retry pass to ``<project_root>/data/verify-logs/flaky-ledger.jsonl``
(``{ts, test, role, flaky_count_window}`` per line) and emits a
line-anchored ``=== CHRONIC-FLAKY test=<name> count=<n> window=<m> ===``
marker when a test is flaky ``>= threshold`` times in the last ``window``
ledger-recorded runs.

Detection strategy (hedges reify's windowing ambiguity): the MARKER (parsed
from verify output) is reify's own authoritative "this is chronic" trigger;
the LEDGER read (from the stable project-root path) supplies the rich
evidence (dates/counts/roles) for the filed task's description AND an
independent/fallback trigger via this module's own configured
threshold/window. :func:`maybe_file_chronic_flake_tasks` unions both.

This module never talks to the fused-memory MCP directly (mirrors
``offline_lane.OfflineLaneTaskClient``'s cross-project scope boundary) —
:class:`ChronicFlakeTaskClient` is the pluggable seam, and
:class:`SchedulerChronicFlakeTaskClient` is the concrete adapter over a
duck-typed scheduler exposing ``dispatch_tool`` (self-contained here to
avoid a workflow<->harness import cycle).

Non-blocking guarantee: :func:`maybe_file_chronic_flake_tasks` is internally
catch-all-defensive (mirrors ``compute_failing_test_set_fingerprint``'s
fail-safe philosophy) so a ledger/MCP/search error can never fail the
verify/merge path; callers (``TaskWorkflow._maybe_file_chronic_flakes``)
wrap it in a second try/except as belt-and-suspenders.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from shared.safe_io import load_json_or_warn
from shared.task_statuses import TERMINAL

if TYPE_CHECKING:
    from orchestrator.config import OrchestratorConfig

logger = logging.getLogger(__name__)

# Default relpath (under project_root) for the persistent per-test filing
# rate-limit ledger — sibling to the scheduler's other small persisted-state
# JSON files under data/orchestrator/.
_DEFAULT_FILINGS_RELPATH = Path('data') / 'orchestrator' / 'chronic_flake_filings.json'


# ---------------------------------------------------------------------------
# CHRONIC-FLAKY marker parsing (step-3/step-4)
# ---------------------------------------------------------------------------


@dataclass
class ChronicFlakeMarker:
    """A single parsed ``=== CHRONIC-FLAKY test=<name> count=<n> window=<m> ===``
    marker line emitted by reify's ``tests/infra/run_all.sh`` (reify task 5142)."""

    test: str
    count: int
    window: int


# Line-anchored (lstrip + fullmatch semantics): ``^\s*`` tolerates leading
# whitespace only (never an arbitrary log/harness prefix), and ``\s*$``
# tolerates trailing whitespace only — so a marker embedded MID-LINE
# (assertion prose quoting the token) or prefixed by a harness/log tag
# cannot match. Mirrors the anchored discipline of
# ``verify._match_clock_marker`` (reify esc-4791-52 / task 4998 regression:
# a substring match let infra tests' own assertion prose masquerade as a
# real marker).
_CHRONIC_FLAKY_MARKER_RE = re.compile(
    r'^\s*=== CHRONIC-FLAKY test=(?P<test>\S+) count=(?P<count>\d+) window=(?P<window>\d+) ===\s*$'
)


def _match_chronic_flaky_marker(line: str) -> ChronicFlakeMarker | None:
    """Return a :class:`ChronicFlakeMarker` if *line* is a genuine, anchored
    CHRONIC-FLAKY marker; else ``None``.

    See module-level ``_CHRONIC_FLAKY_MARKER_RE`` docstring comment for the
    anchoring discipline. Malformed count/window (non-digit) fails the
    regex itself; the ``int()`` calls are defense-in-depth and should never
    raise given the ``\\d+`` capture groups.
    """
    m = _CHRONIC_FLAKY_MARKER_RE.match(line)
    if m is None:
        return None
    try:
        count = int(m.group('count'))
        window = int(m.group('window'))
    except (TypeError, ValueError):
        return None
    return ChronicFlakeMarker(test=m.group('test'), count=count, window=window)


def parse_chronic_flaky_markers(output: str) -> list[ChronicFlakeMarker]:
    """Scan *output* line-by-line and return every valid CHRONIC-FLAKY marker,
    in order. Non-matching lines (plain output, embedded/prefixed
    occurrences) are silently skipped. Returns ``[]`` for falsy *output*."""
    if not output:
        return []
    markers = []
    for line in output.splitlines():
        marker = _match_chronic_flaky_marker(line)
        if marker is not None:
            markers.append(marker)
    return markers


# ---------------------------------------------------------------------------
# Flaky ledger read + chronic-test computation (step-5/step-6)
# ---------------------------------------------------------------------------


def read_flaky_ledger(path: str | Path) -> list[dict]:
    """Best-effort read of reify's ``flaky-ledger.jsonl`` (task 5142):
    one ``{ts, test, role, flaky_count_window}`` JSON object per line.

    Tolerant by design — this is evidence-gathering for a non-blocking
    filing decision, never a correctness-critical parse: blank lines,
    malformed JSON, and well-formed-but-non-dict rows are logged and
    skipped rather than raising. A missing ledger file (reify:5142 not yet
    landed, or no flakes recorded yet) returns ``[]``.
    """
    ledger_path = Path(path)
    if not ledger_path.exists():
        return []
    entries: list[dict] = []
    for line in ledger_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            logger.warning('Skipping malformed flaky-ledger line: %s', line[:120])
            continue
        if not isinstance(row, dict):
            logger.warning('Skipping non-dict flaky-ledger line: %s', line[:120])
            continue
        entries.append(row)
    return entries


@dataclass
class ChronicFlakeEvidence:
    """Evidence bundle for a test computed as chronic from the last
    ``window`` flaky-ledger entries — feeds
    :func:`build_chronic_flake_fix_task_arguments`'s description."""

    test: str
    count: int
    window: int
    dates: list[str]
    roles: list[str]
    entries: list[dict]


def compute_chronic_flakes(
    entries: list[dict], threshold: int, window: int
) -> list[ChronicFlakeEvidence]:
    """Group the last *window* ledger *entries* by ``test`` and flag every
    test occurring ``>= threshold`` times as chronic.

    Entries are assumed chronologically ordered (oldest first, as appended
    by reify's ``run_all.sh``), so "the last `window` entries" is a plain
    tail slice. Sub-threshold tests are excluded. Returns ``[]`` for empty
    *entries* or when nothing meets *threshold*.
    """
    if not entries:
        return []
    considered = entries[-window:] if window > 0 else []
    by_test: dict[str, list[dict]] = {}
    for entry in considered:
        if not isinstance(entry, dict):
            continue
        test = entry.get('test')
        if not test:
            continue
        by_test.setdefault(test, []).append(entry)
    evidence = []
    for test, matches in by_test.items():
        count = len(matches)
        if count < threshold:
            continue
        dates = [str(match.get('ts', '')) for match in matches]
        roles = sorted({str(match['role']) for match in matches if match.get('role')})
        evidence.append(
            ChronicFlakeEvidence(
                test=test, count=count, window=window, dates=dates, roles=roles, entries=matches
            )
        )
    return evidence


# ---------------------------------------------------------------------------
# De-flake fix-task argument builder (step-7/step-8)
# ---------------------------------------------------------------------------

# Fixed root-cause guidance, always appended to the filed task's
# description. Deliberately steers AWAY from the laziest "fix": widening a
# sleep/timeout, which just makes the flake rarer and harder to reproduce
# later. Condition-polling (wait for the actual state the test needs) and
# structural asserts (assert on the state itself, not on prose/log output)
# are the durable fix per the infra-test-wallclock-deflake toolkit.
_ROOT_CAUSE_INSTRUCTION = (
    'ROOT-CAUSE this, do not paper over it: replace wall-clock sleeps with '
    'condition-polling (wait for the actual state the test depends on) and '
    'prefer structural asserts (assert on state, not on log/prose output) '
    'per the infra-test-wallclock-deflake toolkit. NEVER "fix" this with a '
    'blind timeout bump — that only makes the flake rarer and harder to '
    'reproduce, it does not remove it.'
)


def build_chronic_flake_fix_task_arguments(
    evidence: ChronicFlakeEvidence, project_root: str | Path
) -> dict:
    """Build the ``submit_task`` argument block for an auto-filed De-flake
    fix task (task 2358).

    Modeled on :func:`~orchestrator.workflow.build_offline_lane_fix_task_arguments`
    (title/description/priority/project_root/metadata shape). Unlike the
    offline-lane red-fix task, this fires on a test that is CURRENTLY
    GREEN overall (retry-once absorbed the flake) — so it is filed at
    ``priority='medium'``, non-blocking, purely to make the flake debt
    visible and owned.
    """
    dates_str = ', '.join(evidence.dates) if evidence.dates else '(none recorded)'
    roles_str = ', '.join(evidence.roles) if evidence.roles else '(none recorded)'
    title = f'De-flake {evidence.test}: chronic pool flake (auto-filed)'
    description = (
        f'The chronic-flake auto-file detector observed {evidence.test} flaky '
        f'{evidence.count} time(s) within the last {evidence.window} '
        f'flaky-ledger-recorded run(s) — at or above the configured chronic '
        f'threshold.\n\n'
        f'Evidence (from data/verify-logs/flaky-ledger.jsonl):\n'
        f'  test:    {evidence.test}\n'
        f'  count:   {evidence.count}\n'
        f'  window:  {evidence.window}\n'
        f'  dates:   {dates_str}\n'
        f'  role(s): {roles_str}\n\n'
        f'{_ROOT_CAUSE_INSTRUCTION}\n\n'
        f'This task was auto-filed by the chronic-flake detector (task 2358) '
        f'after a verify completed. The gate stayed green — retry-once '
        f'already absorbed the failure — so this is non-blocking, visible '
        f'flake debt, not an incident.'
    )
    return {
        'title': title,
        'description': description,
        'priority': 'medium',
        'project_root': str(project_root),
        'metadata': {
            'spawn_context': 'chronic_flake_auto_file',
            'chronic_flake_test': evidence.test,
            'chronic_flake_count': evidence.count,
            'chronic_flake_window': evidence.window,
            'roles': evidence.roles,
        },
    }


# ---------------------------------------------------------------------------
# FilingLedger: persistent per-test rate-limit store (step-9/step-10)
# ---------------------------------------------------------------------------


class FilingLedger:
    """Persistent JSON store of the last auto-filed timestamp per test
    (``{test: last_filed_iso}``), backing the per-test rate limit that keeps
    :func:`maybe_file_chronic_flake_tasks` from re-filing the same chronic
    flake on every verify run.

    Loaded eagerly at construction time (mirrors
    :class:`~orchestrator.landed_outbox.LandedOutbox`'s warm-at-construction
    cache) via ``shared.safe_io.load_json_or_warn`` — fail-open: a missing
    file (first run) or a corrupt file (disk/process failure) both yield an
    empty ledger rather than raising, since losing rate-limit history only
    risks filing a little sooner than ideal, never a correctness bug.

    ``now`` is always caller-injected (never ``datetime.now()`` internally)
    so callers/tests can drive rate-limit-elapsed scenarios deterministically.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._data: dict[str, str] = self.load()

    def load(self) -> dict[str, str]:
        """(Re)load the on-disk ledger; best-effort — a missing or corrupt
        file yields ``{}`` rather than raising."""
        data, ok = load_json_or_warn(self._path, default={}, on_corrupt='warn')
        if not ok or not isinstance(data, dict):
            return {}
        return {
            test: last_filed_iso
            for test, last_filed_iso in data.items()
            if isinstance(test, str) and isinstance(last_filed_iso, str)
        }

    def should_file(self, test: str, now: datetime, days: int) -> bool:
        """True if *test* has never been filed, or its last filing was at
        least *days* days before *now*."""
        last_filed_iso = self._data.get(test)
        if last_filed_iso is None:
            return True
        try:
            last_filed = datetime.fromisoformat(last_filed_iso)
        except ValueError:
            # Hand-edited/schema-drifted entry — fail-open (treat as unfiled).
            return True
        return now - last_filed >= timedelta(days=days)

    def record(self, test: str, now: datetime) -> None:
        """Record *test* as filed at *now* (in-memory only — call
        :meth:`save` to persist)."""
        self._data[test] = now.isoformat()

    def save(self) -> None:
        """Best-effort persist the in-memory ledger to disk (mkdir parents)."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data))
        except OSError as exc:
            logger.warning(
                'chronic_flake: failed to save filing ledger at %s: %s', self._path, exc,
            )


# ---------------------------------------------------------------------------
# maybe_file_chronic_flake_tasks: detect + dedup + file (step-11/step-12)
# ---------------------------------------------------------------------------


class ChronicFlakeTaskClient(Protocol):
    """Pluggable task-filing seam so this module never talks to the
    fused-memory MCP directly — mirrors
    ``offline_lane.OfflineLaneTaskClient``'s cross-project scope boundary.
    :class:`SchedulerChronicFlakeTaskClient` (step-15/step-16) is the
    concrete adapter over a duck-typed scheduler."""

    async def submit_task(self, arguments: dict) -> str:
        """Submit a new task from a ``submit_task``-shaped argument block
        (see :func:`build_chronic_flake_fix_task_arguments`) and return a
        best-effort id (submit is two-phase server-side, so the id is
        log-only — never relied on for dedup)."""
        ...

    async def search_tasks(self, query: str) -> list[dict]:
        """Semantic search over already-filed tasks; each result carries at
        least ``title`` and ``status`` (mirrors the fused-memory
        ``search_tasks`` MCP tool's result shape)."""
        ...


def _merge_evidence_by_test(
    verify_output: str, ledger_entries: list[dict], threshold: int, window: int,
) -> dict[str, ChronicFlakeEvidence]:
    """Union the ledger-computed chronic tests with the CHRONIC-FLAKY
    markers parsed from *verify_output*, keyed by test name.

    The ledger's own evidence (dates/roles/entries) is preferred when a test
    is chronic by BOTH signals; a marker-only test (chronic per reify's own
    determination but not — or not yet — at DF's configured threshold in the
    ledger) falls back to a thin evidence record built from the marker's own
    fields. Dict insertion order (ledger-chronic tests first, in
    ``compute_chronic_flakes``' order, then marker-only tests in the order
    they appear in *verify_output*) is what :func:`maybe_file_chronic_flake_tasks`
    treats as "deterministic order".
    """
    evidence_by_test: dict[str, ChronicFlakeEvidence] = {
        evidence.test: evidence
        for evidence in compute_chronic_flakes(ledger_entries, threshold, window)
    }
    for marker in parse_chronic_flaky_markers(verify_output):
        if marker.test in evidence_by_test:
            continue
        evidence_by_test[marker.test] = ChronicFlakeEvidence(
            test=marker.test, count=marker.count, window=marker.window,
            dates=[], roles=[], entries=[],
        )
    return evidence_by_test


def _is_open_status(status: str | None) -> bool:
    """True if *status* denotes a live, still-open task.

    Mirrors the ``status in TERMINAL_STATUSES or status == 'deferred'``
    idiom used elsewhere (e.g. ``TaskWorkflow._resolve_and_resubmit``) to
    mean "closed" — this is the negation. ``None``/``''`` (task no longer
    exists, or an unrecognised search-result shape) is treated as NOT open,
    fail-safe towards filing rather than silently dedup-suppressing forever.
    """
    if not status:
        return False
    return status not in TERMINAL and status != 'deferred'


async def _has_open_dedup_match(task_client: ChronicFlakeTaskClient, test: str) -> bool:
    """True if a ``search_tasks`` result's title mentions *test* and that
    task is still OPEN (the local dedup guard, layer (b) — see module
    docstring).

    A ``search_tasks`` failure degrades to "no match" (fail-open towards
    filing, never towards blocking) rather than propagating — part of the
    non-blocking guarantee (step-13/step-14): a flaky search backend must
    never prevent a legitimate filing.
    """
    try:
        results = await task_client.search_tasks(test)
    except Exception:
        logger.warning(
            'chronic_flake: search_tasks failed for %s; treating as no match', test, exc_info=True,
        )
        return False
    for result in results or []:
        if not isinstance(result, dict):
            continue
        if test in (result.get('title') or '') and _is_open_status(result.get('status')):
            return True
    return False


async def maybe_file_chronic_flake_tasks(
    verify_output: str,
    config: OrchestratorConfig,
    task_client: ChronicFlakeTaskClient | None,
    *,
    now: datetime | None = None,
    ledger_path: str | Path | None = None,
    filings_path: str | Path | None = None,
) -> list[str]:
    """After a verify completes, detect chronic pool-infra flakes and
    auto-file a medium-priority De-flake fix task per test — see the module
    docstring for the full policy. Returns the list of test names actually
    filed this call (``[]`` when disabled, unwired, or nothing chronic/dedup
    -eligible was found).

    *ledger_path*/*filings_path* default to ``<project_root>/<ledger_relpath>``
    and ``<project_root>/data/orchestrator/chronic_flake_filings.json``
    respectively when not given (real callers); tests pass tmp fixtures
    explicitly. *now* defaults to the current UTC time.

    Non-blocking guarantee (step-13/step-14): this function never raises.
    The whole body below the enabled/task_client gates runs under a
    catch-all so a ledger/config surprise degrades to ``[]`` instead of
    propagating, and each per-test filing attempt is independently
    try/excepted so one bad test (a raising ``submit_task``) cannot stop
    the rest from being considered — mirrors
    ``compute_failing_test_set_fingerprint``'s fail-safe philosophy and
    ``offline_lane``'s fail-open call-site pattern.
    """
    cfg = config.chronic_flake
    if not cfg.enabled:
        return []
    if task_client is None:
        logger.info('chronic_flake: no task_client wired; skipping (log-only no-op)')
        return []

    try:
        project_root = Path(config.project_root)
        if ledger_path is None:
            ledger_path = project_root / cfg.ledger_relpath
        if filings_path is None:
            filings_path = project_root / _DEFAULT_FILINGS_RELPATH
        if now is None:
            now = datetime.now(UTC)

        try:
            ledger_entries = read_flaky_ledger(ledger_path)
        except OSError:
            logger.warning(
                'chronic_flake: failed to read flaky ledger at %s', ledger_path, exc_info=True,
            )
            ledger_entries = []

        evidence_by_test = _merge_evidence_by_test(
            verify_output, ledger_entries, cfg.threshold, cfg.window
        )
        if not evidence_by_test:
            return []

        filings = FilingLedger(filings_path)
        filed: list[str] = []
        for test, evidence in evidence_by_test.items():
            try:
                if not filings.should_file(test, now, cfg.rate_limit_days):
                    continue
                if await _has_open_dedup_match(task_client, test):
                    continue
                arguments = build_chronic_flake_fix_task_arguments(evidence, project_root)
                await task_client.submit_task(arguments)
                filings.record(test, now)
                filed.append(test)
            except Exception:
                logger.warning(
                    'chronic_flake: failed to file De-flake task for %s', test, exc_info=True,
                )
                continue

        filings.save()
        return filed
    except Exception:
        logger.warning(
            'chronic_flake: maybe_file_chronic_flake_tasks failed unexpectedly', exc_info=True,
        )
        return []


# ---------------------------------------------------------------------------
# SchedulerChronicFlakeTaskClient: concrete adapter over dispatch_tool
# (step-15/step-16)
# ---------------------------------------------------------------------------


def _unwrap_dispatch_envelope(result: object) -> dict:
    """Best-effort unwrap of a ``dispatch_tool`` response down to the
    underlying tool's own returned dict, tolerating whichever transport
    shape it happens to hand back.

    Generalises ``Harness._extract_task_id``'s envelope normalisation
    (``{'task_id': ...}`` direct, ``{'structuredContent': {...}}`` /
    ``{'result': {...}}`` nested, or a ``content`` list of text blocks
    carrying JSON) so both id-extraction (:func:`extract_task_id`) and
    results-list extraction (search_tasks) share one seam. Returns ``{}``
    for a non-dict *result* or when no known shape unwraps.
    """
    if not isinstance(result, dict):
        return {}
    for key in ('structuredContent', 'result'):
        inner = result.get(key)
        if isinstance(inner, dict):
            return inner
    content = result.get('content')
    if isinstance(content, list):
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get('type') == 'text':
                try:
                    parsed = json.loads(str(chunk.get('text') or ''))
                except (TypeError, ValueError):
                    continue
                if isinstance(parsed, dict):
                    return parsed
    return result


def extract_task_id(result: object) -> str:
    """Best-effort extraction of a task/ticket id out of a
    ``dispatch_tool('submit_task', ...)`` response envelope.

    Self-contained (no ``harness``/``scheduler`` import — avoids a
    workflow<->harness import cycle) so both
    :class:`SchedulerChronicFlakeTaskClient` and any future adapter can
    share it. Handles the direct ``{'task_id': ...}`` shape (e.g.
    planning-mode submissions) AND ``{'ticket': ...}`` — ``submit_task``'s
    actual two-phase response, ``{"ticket": "tkt_<id>"}`` — plus the
    ``structuredContent``/``content``-text-block nested shapes via
    :func:`_unwrap_dispatch_envelope`. Never raises; an
    unrecognised/unparseable envelope returns ``''``. This id is
    **log-only** (see the module docstring's dedup-layer note) — never
    relied on for correctness.
    """
    envelope = _unwrap_dispatch_envelope(result)
    for key in ('task_id', 'ticket'):
        if envelope.get(key):
            return str(envelope[key])
    return ''


def _extract_results_list(result: object) -> list[dict]:
    """Best-effort extraction of a ``search_tasks`` response's ``results``
    list from a ``dispatch_tool`` envelope (see
    :func:`_unwrap_dispatch_envelope`). Non-dict results within the list,
    or the absence of a ``results`` list altogether, degrade to ``[]``
    rather than raising."""
    envelope = _unwrap_dispatch_envelope(result)
    results = envelope.get('results')
    if not isinstance(results, list):
        return []
    return [entry for entry in results if isinstance(entry, dict)]


# Per-call dispatch_tool timeout for submit_task — matches
# harness._OfflineLaneTaskClient.submit_fix_task's precedent.
_SUBMIT_TASK_TIMEOUT_SECS = 30


class SchedulerChronicFlakeTaskClient:
    """Concrete :class:`ChronicFlakeTaskClient` adapter over a duck-typed
    scheduler exposing ``dispatch_tool`` (the ``SchedulerFacade`` surface).

    Self-contained here (not in ``harness.py``) so this module never talks
    to the fused-memory MCP directly and avoids a workflow<->harness import
    cycle — mirrors ``harness._OfflineLaneTaskClient``'s scope boundary,
    just re-derived locally rather than imported.
    """

    def __init__(self, scheduler: Any, project_root: str | Path) -> None:
        self._scheduler = scheduler
        self._project_root = str(project_root)

    async def submit_task(self, arguments: dict) -> str:
        """Submit *arguments* (see
        :func:`build_chronic_flake_fix_task_arguments`) via
        ``dispatch_tool('submit_task', ...)`` and return a best-effort id
        (log-only — submit is two-phase server-side)."""
        result = await self._scheduler.dispatch_tool(
            'submit_task', arguments, timeout=_SUBMIT_TASK_TIMEOUT_SECS,
        )
        return extract_task_id(result)

    async def search_tasks(self, query: str) -> list[dict]:
        """Semantic search over already-filed tasks via
        ``dispatch_tool('search_tasks', ...)``; returns the parsed
        ``results`` list (each entry carrying at least ``title``/``status``),
        or ``[]`` for a malformed/unrecognised response envelope."""
        result = await self._scheduler.dispatch_tool(
            'search_tasks', {'project_root': self._project_root, 'query': query},
        )
        return _extract_results_list(result)
