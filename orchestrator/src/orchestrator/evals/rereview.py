"""Re-review pre-cutoff eval results with the new 1× opus reviewer.

Replays the REVIEWER_COMPREHENSIVE reviewer (VARIANT_A, 1× opus, $5,
effort=high) against the existing eval worktree diffs so results graded by
the old 5× sonnet panel become directly comparable to post-cutoff results.

Only top-level ``metrics`` values are patched in place; original sonnet
values are preserved as ``legacy_sonnet_*`` fields. Original reviewer
artifacts in ``.eval-worktrees/<task>/run-<id>/.task/reviews/`` are NOT
touched.

Usage::

    python -m orchestrator.evals.rereview --enumerate
    python -m orchestrator.evals.rereview --run --scope FILE [--force]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from orchestrator.evals.metrics import EvalMetrics, compute_composite
from orchestrator.evals.reviewer_trial.corpus import CorpusDiff
from orchestrator.evals.reviewer_trial.runner import _run_single_reviewer
from orchestrator.evals.reviewer_trial.variants import VARIANT_A
from orchestrator.evals.snapshots import get_diff

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Commit 594658fbe3 replaced the 5× sonnet panel with the 1× opus reviewer.
# Override via ``--cutoff`` if this is off.
CUTOFF = datetime(2026, 4, 8, 10, 32, 0, tzinfo=timezone.utc)

RESULTS_DIR = Path(__file__).parent / 'results'
RE_REVIEWS_DIR = RESULTS_DIR / 're_reviews'
PROGRESS_FILE = RE_REVIEWS_DIR / '_progress.jsonl'

MAX_CONCURRENCY = 24
STAGGER_SECS = 5.0

# Median per-review cost observed in reviewer_trial data.
_EST_COST_PER_REVIEW = 3.0

_HEX = set('0123456789abcdef')


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    result_path: Path
    task_id: str
    config_name: str
    run_id: str
    worktree_path: Path
    base_commit: str
    current_blocking: int
    current_suggestions: int
    mtime: datetime


@dataclass
class RereviewOutcome:
    status: str   # 'done' | 'skip' | 'error'
    run_id: str = ''
    cost: float = 0.0
    blocking: int = 0
    suggestions: int = 0
    message: str = ''

    @classmethod
    def ok(
        cls, run_id: str, cost: float, blocking: int, suggestions: int,
    ) -> 'RereviewOutcome':
        return cls(
            status='done', run_id=run_id, cost=cost,
            blocking=blocking, suggestions=suggestions,
        )

    @classmethod
    def skip(cls, run_id: str, message: str) -> 'RereviewOutcome':
        return cls(status='skip', run_id=run_id, message=message)

    @classmethod
    def error(cls, run_id: str, message: str) -> 'RereviewOutcome':
        return cls(status='error', run_id=run_id, message=message)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def atomic_write_json(path: Path, data: dict) -> None:
    """Write ``data`` to ``path`` via a temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def append_jsonl(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(entry) + '\n')


def _extract_run_id(result_path: Path, data: dict) -> str:
    """Pull run_id from the result data, filename suffix, or worktree path."""
    rid = data.get('run_id') or ''
    if rid:
        return rid
    # Filename convention: task_id__config_name__<runid>.json
    parts = result_path.stem.split('__')
    tail = parts[-1] if len(parts) >= 3 else ''
    if len(tail) == 8 and all(c in _HEX for c in tail):
        return tail
    wt = data.get('worktree_path', '')
    if wt:
        name = Path(wt).name
        if name.startswith('run-'):
            return name[len('run-'):]
    return ''


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

def enumerate_candidates(cutoff: datetime, force: bool) -> list[Candidate]:
    """Return done-outcome pre-cutoff results whose worktree is intact."""
    cands: list[Candidate] = []
    for path in sorted(RESULTS_DIR.glob('*.json')):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning('Cannot read %s: %s', path.name, e)
            continue

        if data.get('outcome') != 'done':
            continue

        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if mtime >= cutoff:
            continue

        wt_raw = data.get('worktree_path')
        if not wt_raw:
            continue
        worktree = Path(wt_raw)
        if not worktree.is_dir():
            continue

        meta_file = worktree / '.task' / 'metadata.json'
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        base_commit = meta.get('base_commit')
        if not base_commit:
            continue

        metrics = data.get('metrics') or {}
        if not force and 'legacy_sonnet_review_blocking_issues' in metrics:
            continue

        run_id = _extract_run_id(path, data)
        if not run_id:
            logger.warning('No run_id for %s; skipping', path.name)
            continue

        cands.append(Candidate(
            result_path=path,
            task_id=data.get('task_id', ''),
            config_name=data.get('config_name', ''),
            run_id=run_id,
            worktree_path=worktree,
            base_commit=base_commit,
            current_blocking=int(metrics.get('review_blocking_issues', 0)),
            current_suggestions=int(metrics.get('review_suggestions', 0)),
            mtime=mtime,
        ))

    cands.sort(key=lambda c: (c.task_id, c.config_name, c.run_id))
    return cands


def print_enumeration(cands: list[Candidate]) -> None:
    if not cands:
        print('No candidates.')
        return
    tw = max(len('task_id'), max(len(c.task_id) for c in cands))
    cw = max(len('config_name'), max(len(c.config_name) for c in cands))
    header = (
        f'{"task_id":<{tw}}  {"config_name":<{cw}}  '
        f'{"run_id":<8}  {"mtime":<16}  {"blocking":>8}  {"suggestions":>11}'
    )
    print(header)
    print('-' * len(header))
    for c in cands:
        mt = c.mtime.strftime('%Y-%m-%dT%H:%M')
        print(
            f'{c.task_id:<{tw}}  {c.config_name:<{cw}}  '
            f'{c.run_id:<8}  {mt:<16}  '
            f'{c.current_blocking:>8}  {c.current_suggestions:>11}'
        )
    print()
    print(f'Total: {len(cands)} candidates')
    est = len(cands) * _EST_COST_PER_REVIEW
    print(f'Estimated cost: ${est:.2f}  (median ${_EST_COST_PER_REVIEW:.0f}/review)')


# ---------------------------------------------------------------------------
# Scope loading
# ---------------------------------------------------------------------------

def load_scope(path: Path) -> set[str]:
    ids: set[str] = set()
    for raw_line in path.read_text().splitlines():
        line = raw_line.split('#', 1)[0].strip()
        if line:
            ids.add(line)
    return ids


# ---------------------------------------------------------------------------
# Re-review pipeline
# ---------------------------------------------------------------------------

def recompute_composite_from_metrics(metrics_dict: dict) -> float:
    """Rebuild an EvalMetrics from the patched dict and call compute_composite."""
    known = set(EvalMetrics.__dataclass_fields__)
    filtered = {k: v for k, v in metrics_dict.items() if k in known}
    m = EvalMetrics(**filtered)
    return compute_composite(m)


async def rereview_one(cand: Candidate) -> RereviewOutcome:
    # 1. Compute the diff (pinned to metadata base_commit via get_diff).
    try:
        diff_text = await get_diff(cand.worktree_path)
    except Exception as e:
        return RereviewOutcome.error(cand.run_id, f'get_diff failed: {e}')
    if not diff_text.strip():
        return RereviewOutcome.skip(cand.run_id, 'empty diff')

    # 2. Invoke the production reviewer via reviewer_trial's proven path.
    corpus_diff = CorpusDiff(
        diff_id=cand.run_id,
        language='python',
        source='real_world',
        diff_text=diff_text,
        description='',
        ground_truth=[],
        cwd=cand.worktree_path,
    )
    spec = VARIANT_A.reviewers[0]

    try:
        name, review, cost = await _run_single_reviewer(spec, corpus_diff, max_retries=2)
    except Exception as e:
        return RereviewOutcome.error(cand.run_id, f'reviewer raised: {e}')

    if review is None or review.get('verdict') == 'ERROR':
        return RereviewOutcome.error(
            cand.run_id, 'reviewer returned no parseable result',
        )

    issues = review.get('issues') or []
    blocking = sum(1 for i in issues if i.get('severity') == 'blocking')
    suggestions = sum(1 for i in issues if i.get('severity') == 'suggestion')

    # 3. Dump the raw review JSON (atomic).
    raw_path = (
        RE_REVIEWS_DIR
        / f'{cand.task_id}__{cand.config_name}__{cand.run_id}.json'
    )
    atomic_write_json(raw_path, {
        'task_id': cand.task_id,
        'config_name': cand.config_name,
        'run_id': cand.run_id,
        'reviewer': name,
        'review': review,
        'cost_usd': cost,
        'timestamp': datetime.now(tz=timezone.utc).isoformat(),
        'base_commit': cand.base_commit,
        'diff_len': len(diff_text),
    })

    # 4. Patch the result file (atomic).
    try:
        result = json.loads(cand.result_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return RereviewOutcome.error(cand.run_id, f'reread of result failed: {e}')

    m = result.setdefault('metrics', {})
    # Preserve the ORIGINAL sonnet values only on first re-review. Under
    # --force reruns, the legacy fields must still point at the original
    # sonnet values, not the previous opus result.
    if 'legacy_sonnet_review_blocking_issues' not in m:
        m['legacy_sonnet_review_blocking_issues'] = m.get('review_blocking_issues', 0)
        m['legacy_sonnet_review_suggestions'] = m.get('review_suggestions', 0)
        m['legacy_sonnet_composite_score'] = m.get('composite_score', 0.0)

    m['review_blocking_issues'] = blocking
    m['review_suggestions'] = suggestions
    m['composite_score'] = recompute_composite_from_metrics(m)
    atomic_write_json(cand.result_path, result)

    # 5. Append progress.
    append_jsonl(PROGRESS_FILE, {
        'run_id': cand.run_id,
        'task_id': cand.task_id,
        'config_name': cand.config_name,
        'status': 'done',
        'cost_usd': cost,
        'blocking': blocking,
        'suggestions': suggestions,
        'ts': datetime.now(tz=timezone.utc).isoformat(),
    })
    return RereviewOutcome.ok(cand.run_id, cost, blocking, suggestions)


async def run_rereviews(cands: list[Candidate]) -> list[RereviewOutcome]:
    if not cands:
        return []
    RE_REVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _gated(i: int, cand: Candidate) -> RereviewOutcome:
        # Fixed launch-time stagger: first MAX_CONCURRENCY tasks ramp at
        # STAGGER_SECS intervals, later tasks queue on the sem (no extra
        # stagger once the pool is full).
        await asyncio.sleep(i * STAGGER_SECS)
        async with sem:
            try:
                return await rereview_one(cand)
            except Exception as e:  # pragma: no cover - defensive
                logger.exception('rereview_one crashed for %s', cand.run_id)
                return RereviewOutcome.error(cand.run_id, f'unhandled: {e}')

    tasks = [asyncio.create_task(_gated(i, c)) for i, c in enumerate(cands)]

    outcomes: list[RereviewOutcome] = []
    done = 0
    spent = 0.0
    errors = 0
    skipped = 0
    for coro in asyncio.as_completed(tasks):
        r = await coro
        outcomes.append(r)
        done += 1
        if r.status == 'done':
            spent += r.cost
        elif r.status == 'error':
            errors += 1
            append_jsonl(PROGRESS_FILE, {
                'run_id': r.run_id,
                'status': 'error',
                'message': r.message,
                'ts': datetime.now(tz=timezone.utc).isoformat(),
            })
        elif r.status == 'skip':
            skipped += 1
        print(
            f'[{done}/{len(cands)}] ${spent:.2f}  '
            f'errors={errors}  skipped={skipped}  last={r.run_id}:{r.status}',
            flush=True,
        )
    return outcomes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            'Re-review pre-cutoff eval results with the 1x opus reviewer. '
            'Run with --enumerate to preview, then --run --scope FILE.'
        ),
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        '--enumerate', action='store_true',
        help='Print candidate table and cost estimate (no API calls)',
    )
    mode.add_argument(
        '--run', action='store_true',
        help='Run re-reviews for candidates listed in --scope',
    )
    p.add_argument(
        '--scope', type=Path,
        help='File of newline-delimited run_ids (required with --run; '
             'blanks and # comments allowed)',
    )
    p.add_argument(
        '--force', action='store_true',
        help='Re-review even if legacy_sonnet_* already set',
    )
    p.add_argument(
        '--cutoff', type=str, default=None,
        help='Override cutoff as ISO8601 (default: hardcoded constant)',
    )
    return p.parse_args()


async def _amain() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    cutoff = CUTOFF
    if args.cutoff:
        try:
            parsed = datetime.fromisoformat(args.cutoff)
        except ValueError:
            print(f'Invalid --cutoff: {args.cutoff}', file=sys.stderr)
            return 2
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        cutoff = parsed

    cands = enumerate_candidates(cutoff=cutoff, force=args.force)

    if args.enumerate:
        print_enumeration(cands)
        return 0

    # --run mode
    if not args.scope:
        print('--run requires --scope FILE', file=sys.stderr)
        return 2
    if not args.scope.exists():
        print(f'Scope file not found: {args.scope}', file=sys.stderr)
        return 2

    scope = load_scope(args.scope)
    if not scope:
        print(f'Scope file {args.scope} has no run_ids', file=sys.stderr)
        return 2

    enum_ids = {c.run_id for c in cands}
    missing = scope - enum_ids
    if missing:
        print(
            f'Scope contains run_ids not in enumeration: '
            f'{sorted(missing)}\n'
            f'(filtered out because post-cutoff, outcome != done, missing '
            f'worktree/metadata, or already re-reviewed)\n'
            f'Use --force to re-review already-re-reviewed entries, or '
            f'--cutoff to widen the time window.',
            file=sys.stderr,
        )
        return 2

    selected = [c for c in cands if c.run_id in scope]
    print(f'Running re-review on {len(selected)} candidate(s)...')
    outcomes = await run_rereviews(selected)

    n_done = sum(1 for o in outcomes if o.status == 'done')
    n_err = sum(1 for o in outcomes if o.status == 'error')
    n_skip = sum(1 for o in outcomes if o.status == 'skip')
    total_cost = sum(o.cost for o in outcomes if o.status == 'done')
    print()
    print(
        f'Summary: done={n_done}  errors={n_err}  skipped={n_skip}  '
        f'cost=${total_cost:.2f}',
    )
    if n_err:
        print('Error details:')
        for o in outcomes:
            if o.status == 'error':
                print(f'  {o.run_id}: {o.message}')
        return 1
    return 0


def main() -> None:
    sys.exit(asyncio.run(_amain()))


if __name__ == '__main__':
    main()
