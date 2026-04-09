#!/usr/bin/env python3
"""Phase-1 diagnostic for the reviewer panel restructuring trial.

Walks every `.task/reviews/reviewer_*.json` under .worktrees and
.eval-worktrees, computes per-reviewer activity, redundancy/uniqueness,
and pairwise overlap stats.

Two notions of "same issue":
1. coarse  — same file path (stripping :line)
2. medium  — same file path AND same category string

We do NOT do semantic dedup; the LLM-assisted matcher is Phase 6 work.
The point of Phase 1 is to learn whether the data even motivates the
panel changes before we build the rest of the trial.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

REPO = Path('/home/leo/src/dark-factory')
SEARCH_ROOTS = [REPO / '.worktrees', REPO / '.eval-worktrees']

REVIEWER_NAMES = [
    'test_analyst',
    'reuse_auditor',
    'architect_reviewer',
    'performance',
    'robustness',
]


def find_review_dirs() -> list[Path]:
    dirs: list[Path] = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob('.task/reviews'):
            if any(p.glob('reviewer_*.json')):
                dirs.append(p)
    return sorted(set(dirs))


def task_label(reviews_dir: Path) -> str:
    """Friendly label like '.worktrees/516' or 'eval/df_task_12/run-x'."""
    parts = reviews_dir.relative_to(REPO).parts
    # parts ends with .task/reviews — strip those
    return '/'.join(parts[:-2])


def load_review(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def normalize_location(loc: str) -> str:
    """Strip line numbers and whitespace from a location string."""
    if not loc:
        return ''
    # Take part before first colon (the file path)
    head = loc.split(':', 1)[0].strip()
    return head


def main() -> int:
    review_dirs = find_review_dirs()
    print(f'Found {len(review_dirs)} task review directories\n')

    # ── per-reviewer aggregates ────────────────────────────────────────
    per_rev_total_issues: Counter = Counter()
    per_rev_blocking: Counter = Counter()
    per_rev_suggestions: Counter = Counter()
    per_rev_present: Counter = Counter()       # tasks where reviewer ran
    per_rev_pass_verdict: Counter = Counter()  # tasks where verdict=PASS
    per_rev_error: Counter = Counter()         # tasks where verdict=ERROR
    per_rev_categories: dict[str, Counter] = defaultdict(Counter)

    # ── per-task aggregates (for redundancy analysis) ──────────────────
    # Each task: {reviewer_name: [(location_normalized, category, severity), ...]}
    per_task: dict[str, dict[str, list[tuple[str, str, str]]]] = {}
    blocked_tasks: list[str] = []  # tasks where any reviewer flagged blocking

    for d in review_dirs:
        label = task_label(d)
        per_task[label] = {}

        for name in REVIEWER_NAMES:
            f = d / f'reviewer_{name}.json'
            if not f.exists():
                continue
            data = load_review(f)
            if data is None:
                continue

            per_rev_present[name] += 1
            verdict = data.get('verdict', '')
            if verdict == 'PASS':
                per_rev_pass_verdict[name] += 1
            if verdict == 'ERROR':
                per_rev_error[name] += 1

            issues = data.get('issues', []) or []
            findings: list[tuple[str, str, str]] = []
            for issue in issues:
                sev = issue.get('severity', 'unknown')
                loc = normalize_location(issue.get('location', ''))
                cat = (issue.get('category') or '').strip()
                per_rev_total_issues[name] += 1
                if sev == 'blocking':
                    per_rev_blocking[name] += 1
                else:
                    per_rev_suggestions[name] += 1
                per_rev_categories[name][cat] += 1
                findings.append((loc, cat, sev))
            per_task[label][name] = findings

        # Did any reviewer mark this task as blocked?
        if any(
            (loc and sev == 'blocking')
            for findings in per_task[label].values()
            for (loc, _cat, sev) in findings
        ):
            blocked_tasks.append(label)

    n_tasks = len(per_task)

    # ── per-reviewer summary table ─────────────────────────────────────
    print('=' * 78)
    print('PER-REVIEWER SUMMARY')
    print('=' * 78)
    print(f'{"reviewer":<22} {"runs":>5} {"PASS":>5} {"ERR":>4} '
          f'{"issues":>7} {"block":>6} {"sugg":>6} {"i/run":>7}')
    print('-' * 78)
    for name in REVIEWER_NAMES:
        runs = per_rev_present[name]
        if runs == 0:
            continue
        i_per_run = per_rev_total_issues[name] / runs
        print(
            f'{name:<22} {runs:>5} {per_rev_pass_verdict[name]:>5} '
            f'{per_rev_error[name]:>4} {per_rev_total_issues[name]:>7} '
            f'{per_rev_blocking[name]:>6} {per_rev_suggestions[name]:>6} '
            f'{i_per_run:>7.1f}'
        )
    print()

    # ── redundancy: per task, group findings by location ──────────────
    # For each task, count how many reviewers hit each location.
    # Then bucket: 1 reviewer = unique, 2 = pair, 3+ = pile-on.
    unique_finds_per_rev: Counter = Counter()
    unique_blocking_per_rev: Counter = Counter()
    pile_on_count: Counter = Counter()  # how many reviewers per location
    pile_on_blocking: Counter = Counter()  # same but for blocking only

    for label, by_rev in per_task.items():
        # location → set of reviewers that mentioned it
        # location → list of (reviewer, severity)
        loc_to_reviewers: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for rev, findings in by_rev.items():
            seen_locs_for_this_rev: set[str] = set()
            for loc, _cat, sev in findings:
                if not loc:
                    continue
                # dedup within a reviewer (same loc flagged twice = once)
                if loc in seen_locs_for_this_rev:
                    continue
                seen_locs_for_this_rev.add(loc)
                loc_to_reviewers[loc].append((rev, sev))

        for loc, hits in loc_to_reviewers.items():
            n_hits = len(hits)
            pile_on_count[n_hits] += 1
            any_blocking = any(s == 'blocking' for _, s in hits)
            if any_blocking:
                pile_on_blocking[n_hits] += 1
            if n_hits == 1:
                rev = hits[0][0]
                unique_finds_per_rev[rev] += 1
                if hits[0][1] == 'blocking':
                    unique_blocking_per_rev[rev] += 1

    print('=' * 78)
    print('REDUNDANCY: how many reviewers hit each location')
    print('=' * 78)
    print(f'{"reviewers_at_loc":<20} {"locations":>11} {"of which blocking":>20}')
    print('-' * 78)
    total_locations = sum(pile_on_count.values())
    for n in sorted(pile_on_count.keys()):
        pct = 100 * pile_on_count[n] / total_locations if total_locations else 0
        b = pile_on_blocking[n]
        print(f'{n:<20} {pile_on_count[n]:>11} ({pct:>4.1f}%) {b:>15}')
    print(f'{"TOTAL":<20} {total_locations:>11}')
    print()

    # ── unique-find rate per reviewer ─────────────────────────────────
    print('=' * 78)
    print('UNIQUE FINDS PER REVIEWER (location flagged by only this reviewer)')
    print('=' * 78)
    print(f'{"reviewer":<22} {"unique":>7} {"of_total":>10} {"unique_block":>14}')
    print('-' * 78)
    for name in REVIEWER_NAMES:
        if per_rev_present[name] == 0:
            continue
        u = unique_finds_per_rev[name]
        total = per_rev_total_issues[name]
        ub = unique_blocking_per_rev[name]
        pct = 100 * u / total if total else 0
        print(f'{name:<22} {u:>7} {pct:>9.1f}% {ub:>14}')
    print()

    # ── pairwise overlap ──────────────────────────────────────────────
    # For each pair (A, B): of all locations either flagged in any task,
    # how many were flagged by both?
    pair_both: Counter = Counter()
    pair_either: Counter = Counter()
    pair_blocking_both: Counter = Counter()
    for label, by_rev in per_task.items():
        rev_locs: dict[str, set[str]] = {}
        rev_blocking_locs: dict[str, set[str]] = {}
        for rev, findings in by_rev.items():
            rev_locs[rev] = {loc for loc, _, _ in findings if loc}
            rev_blocking_locs[rev] = {
                loc for loc, _, sev in findings if loc and sev == 'blocking'
            }
        for a, b in combinations(REVIEWER_NAMES, 2):
            if a not in rev_locs or b not in rev_locs:
                continue
            both = rev_locs[a] & rev_locs[b]
            either = rev_locs[a] | rev_locs[b]
            pair_both[(a, b)] += len(both)
            pair_either[(a, b)] += len(either)
            pair_blocking_both[(a, b)] += len(
                rev_blocking_locs[a] & rev_blocking_locs[b]
            )

    print('=' * 78)
    print('PAIRWISE OVERLAP (Jaccard similarity of flagged locations)')
    print('=' * 78)
    print(f'{"pair":<48} {"both":>6} {"either":>7} {"jacc":>6}')
    print('-' * 78)
    pairs_sorted = sorted(
        combinations(REVIEWER_NAMES, 2),
        key=lambda p: -(pair_both[p] / pair_either[p] if pair_either[p] else 0),
    )
    for a, b in pairs_sorted:
        if pair_either[(a, b)] == 0:
            continue
        jacc = pair_both[(a, b)] / pair_either[(a, b)]
        print(
            f'{a + " × " + b:<48} {pair_both[(a, b)]:>6} '
            f'{pair_either[(a, b)]:>7} {jacc:>6.2f}'
        )
    print()

    # ── what categories does each reviewer focus on? ───────────────────
    print('=' * 78)
    print('TOP CATEGORIES PER REVIEWER (top 5)')
    print('=' * 78)
    for name in REVIEWER_NAMES:
        if not per_rev_categories[name]:
            continue
        top = per_rev_categories[name].most_common(5)
        cats = ', '.join(f'{c}({n})' for c, n in top)
        print(f'  {name:<22} {cats}')
    print()

    # ── tasks that ended in blocking issues ────────────────────────────
    print('=' * 78)
    print(f'TASKS WITH BLOCKING ISSUES IN LAST REVIEW: {len(blocked_tasks)}/{n_tasks}')
    print('=' * 78)
    for label in sorted(blocked_tasks):
        print(f'  {label}')
        # Show which reviewers flagged blocking and what
        for rev, findings in per_task[label].items():
            blocks = [(loc, cat) for loc, cat, sev in findings if sev == 'blocking']
            if blocks:
                for loc, cat in blocks:
                    print(f'    [{rev}] {cat} @ {loc}')
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
