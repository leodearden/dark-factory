"""FN-label mining machinery for the reviewer_trial corpus (task 2495 / PRD D-6).

Mines "false negative" (FN) candidates from the offline, gitignored
orchestrator run history — ``data/orchestrator/runs.db`` and
``data/escalations/*.json`` in the main repo checkout — tasks whose review
phase PASSed but where a bug surfaced downstream. These sources are
build-time-only inputs consumed by the ``mine`` CLI subcommand
(``__main__.py``) when generating the committed corpus; they are never
runtime dependencies. Unit tests exercise this module exclusively against
synthetic fixtures (``tests/_reviewer_trial_mining_fixtures.py``) — never
the real gitignored data.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MIN_VERIFY_ATTEMPTS = 2

# Outcomes that themselves indicate a downstream bug signal after review passed.
_DOWNSTREAM_OUTCOMES = {'requeued', 'blocked'}


@dataclass
class FnCandidate:
    """A task flagged as a false-negative (FN) candidate by ``mine_fn_candidates``.

    "Review PASSed" is operationalized as ``review_cycles >= 1`` — the
    review loop ran and the task moved past the review phase to verify/
    merge/requeue/etc. ``task_results`` has no explicit review-verdict
    column, so this is the closest available proxy (refined against the
    real runs.db during corpus generation — see PRD D-6 / Open-Q §9).

    ``signal_reason`` names every downstream bug signal that fired, for
    provenance/audit (persisted into the mined diff's annotation
    ``provenance`` — see ``corpus.CorpusDiff.provenance``).
    """

    task_id: str
    project_id: str
    outcome: str
    review_cycles: int
    verify_attempts: int
    merge_sha: str | None
    signal_reason: list[str] = field(default_factory=list)


def mine_fn_candidates(
    db_path: Path,
    min_verify_attempts: int = DEFAULT_MIN_VERIFY_ATTEMPTS,
) -> list[FnCandidate]:
    """Mine FN candidates from a ``runs.db`` (``task_results`` + ``events``).

    A task is a candidate when ``review_cycles >= 1`` (review phase ran)
    AND at least one downstream bug signal is present:

    - ``outcome`` in ``{'requeued', 'blocked'}``
    - ``verify_attempts >= min_verify_attempts``
    - a ``escalation_created`` event exists for the task_id in ``events``

    Read-only: opens the database file directly via stdlib ``sqlite3``, no
    writes. Returns ``[]`` when nothing matches.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT task_id, project_id, outcome, review_cycles, verify_attempts
            FROM task_results
            WHERE review_cycles >= 1
            """
        ).fetchall()

        escalated_task_ids = {
            r['task_id'] for r in conn.execute(
                "SELECT DISTINCT task_id FROM events WHERE event_type = 'escalation_created'"
            ).fetchall()
        }
        merge_sha_by_task: dict[str, str | None] = {}
        for r in conn.execute(
            "SELECT task_id, data FROM events WHERE event_type = 'merge_finalized'"
        ).fetchall():
            try:
                data = json.loads(r['data'])
            except (json.JSONDecodeError, TypeError):
                continue
            merge_sha_by_task[r['task_id']] = data.get('merge_sha')
    finally:
        conn.close()

    candidates: list[FnCandidate] = []
    for row in rows:
        task_id = row['task_id']
        outcome = row['outcome']
        verify_attempts = row['verify_attempts']

        signal_reason: list[str] = []
        if outcome in _DOWNSTREAM_OUTCOMES:
            signal_reason.append(f'outcome:{outcome}')
        if verify_attempts >= min_verify_attempts:
            signal_reason.append(
                f'verify_attempts>={min_verify_attempts} (actual={verify_attempts})'
            )
        if task_id in escalated_task_ids:
            signal_reason.append('escalation_created event')

        if not signal_reason:
            continue

        candidates.append(FnCandidate(
            task_id=task_id,
            project_id=row['project_id'],
            outcome=outcome,
            review_cycles=row['review_cycles'],
            verify_attempts=verify_attempts,
            merge_sha=merge_sha_by_task.get(task_id),
            signal_reason=signal_reason,
        ))

    return candidates


@dataclass
class EscalationRef:
    """A reference to an escalation record (``data/escalations/esc-<id>.json``)."""

    task_id: str
    category: str
    severity: str
    summary: str
    level: int
    path: Path


def mine_escalation_refs(esc_dir: Path) -> dict[str, list[EscalationRef]]:
    """Parse ``esc-<id>.json`` escalation records, keyed by task_id.

    Only files matching ``esc-*.json`` are considered, which naturally
    excludes ``*.json.lock`` sidecars (wrong suffix) and non-escalation
    state files like ``b3-state.json`` (wrong prefix) — both of which live
    alongside the real escalations in ``data/escalations/``.

    Tolerates malformed JSON in an otherwise-matching file by skipping it
    rather than raising — offline mining over ~2,400 files must not abort
    on one bad record.
    """
    refs: dict[str, list[EscalationRef]] = {}
    for path in sorted(esc_dir.glob('esc-*.json')):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        task_id = data.get('task_id')
        if not task_id:
            continue

        ref = EscalationRef(
            task_id=task_id,
            category=data.get('category', ''),
            severity=data.get('severity', ''),
            summary=data.get('summary', ''),
            level=data.get('level', 0),
            path=path,
        )
        refs.setdefault(task_id, []).append(ref)

    return refs


def recover_diff(merge_sha: str, repo_path: Path) -> str | None:
    """Recover the unified diff text for a commit via ``git show``.

    Tries a plain ``git show --format= <sha>`` first (works for ordinary
    single-parent commits). A genuine merge commit typically produces no
    output from that alone (no direct per-parent diff is shown by default),
    so this falls back to the first-parent diff (``git diff <sha>^1 <sha>``)
    — i.e. what the merge changed relative to the branch it landed on.

    Returns ``None`` (never raises) when the sha is unknown/invalid or both
    attempts yield no diff.
    """
    result = subprocess.run(
        ['git', 'show', '--format=', merge_sha],
        cwd=repo_path, capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    if result.stdout.strip():
        return result.stdout

    fallback = subprocess.run(
        ['git', 'diff', f'{merge_sha}^1', merge_sha],
        cwd=repo_path, capture_output=True, text=True,
    )
    if fallback.returncode != 0 or not fallback.stdout.strip():
        return None
    return fallback.stdout


_SPLIT_NAMES = ('train', 'selection', 'test')


def _stable_hash(seed: str, diff_id: str) -> str:
    """Stable (deterministic across processes/runs) hash of seed+diff_id."""
    return hashlib.sha256(f'{seed}:{diff_id}'.encode()).hexdigest()


def assign_split(
    diff_ids: list[str],
    seed: str,
    ratios: tuple[int, int, int] = (2, 1, 7),
) -> dict[str, str]:
    """Deterministically assign each diff_id to train/selection/test.

    Orders ``diff_ids`` by a stable hash of ``f"{seed}:{diff_id}"`` — a
    reproducible pseudo-random permutation independent of input order and
    of any other seed — then slices the ordering into cumulative buckets
    sized by *ratios* (train:selection:test, default 2:1:7). Slicing a
    sorted permutation (rather than bucketing each id independently by
    hash-modulo) gives near-exact proportions for any N rather than a
    statistical approximation that only converges at large N.

    Same *diff_ids* + same *seed* always yields the same assignment
    (determinism required for a reproducible TEST split — see PRD §2).
    """
    total = sum(ratios)
    ordered = sorted(diff_ids, key=lambda d: _stable_hash(seed, d))
    n = len(ordered)

    boundaries = []
    running = 0
    for r in ratios:
        running += r
        boundaries.append(round(n * running / total))

    assignment: dict[str, str] = {}
    start = 0
    for name, end in zip(_SPLIT_NAMES, boundaries):
        for diff_id in ordered[start:end]:
            assignment[diff_id] = name
        start = end

    return assignment
