"""Elo rating system for pairwise eval judging.

DEMOTED (task 2477 λ / decision 10): the primary eval leaderboard is now
:func:`orchestrator.evals.report.build_composite_report` — every run yields a
dense composite score, whereas pairwise judge data is sparse and this Elo path
was ~32% self-admitted noise. The pairwise machinery below is RETAINED, not
deleted (``judge.run_elo_tournament`` and the CLI ``--judge`` path still use
it), but is meaningful only when the pairwise data is dense — gated by
:func:`is_pairwise_data_dense`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STATE_FILE = Path(__file__).parent / 'judge_state.json'

DEFAULT_RATING = 1500.0
K_FACTOR = 32.0
MAX_PER_PAIR = 3
INDISTINGUISHABLE_THRESHOLD = 50.0


@dataclass
class TaskPool:
    """Elo pool for one task."""

    ratings: dict[str, float] = field(default_factory=dict)
    matches: list[dict[str, Any]] = field(default_factory=list)
    pair_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'ratings': self.ratings,
            'matches': self.matches,
            'pair_counts': self.pair_counts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskPool:
        return cls(
            ratings=data.get('ratings', {}),
            matches=data.get('matches', []),
            pair_counts=data.get('pair_counts', {}),
        )


@dataclass
class JudgeState:
    """Full persisted state across all tasks."""

    per_task: dict[str, TaskPool] = field(default_factory=dict)
    updated_at: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'per_task': {k: v.to_dict() for k, v in self.per_task.items()},
            'updated_at': self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JudgeState:
        per_task = {
            k: TaskPool.from_dict(v)
            for k, v in data.get('per_task', {}).items()
        }
        return cls(per_task=per_task, updated_at=data.get('updated_at', ''))


def load_state(path: Path = STATE_FILE) -> JudgeState:
    """Load persisted judge state, or return empty state."""
    if path.exists():
        with open(path) as f:
            return JudgeState.from_dict(json.load(f))
    return JudgeState()


def save_state(state: JudgeState, path: Path = STATE_FILE) -> None:
    """Persist judge state to disk."""
    state.updated_at = datetime.now(UTC).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(state.to_dict(), f, indent=2)
        f.write('\n')


def _pair_key(a: str, b: str) -> str:
    """Canonical key for an unordered pair."""
    return f'{min(a, b)}|{max(a, b)}'


def is_pairwise_data_dense(
    pool: TaskPool, *, min_matches_per_pair: int = 3,
) -> bool:
    """Whether *pool* has DENSE pairwise judge data (task 2477 λ / decision 10).

    The Open-Q3 gate — "drop Elo unless pairwise judge data is dense" — that
    decides whether this task's demoted Elo path is meaningful. Returns:

    - ``False`` when the pool has fewer than 2 configs (no pair is possible).
    - otherwise ``True`` iff EVERY unordered config pair (keyed via
      :func:`_pair_key` over the sorted ``pool.ratings`` keys) has been judged at
      least *min_matches_per_pair* times per ``pool.pair_counts``.

    A single under-filled (or unjudged → 0) pair makes the pool non-dense, so a
    caller can fall back to :func:`report.build_composite_report` (the dense
    composite-score leaderboard) rather than trusting a sparse Elo ranking. Pure
    query over ``pool.ratings`` / ``pool.pair_counts``.
    """
    configs = sorted(pool.ratings.keys())
    if len(configs) < 2:
        return False
    return all(
        pool.pair_counts.get(_pair_key(a, b), 0) >= min_matches_per_pair
        for a, b in combinations(configs, 2)
    )


def expected_score(rating_a: float, rating_b: float) -> float:
    """Standard Elo expected score for player A."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(
    rating_a: float,
    rating_b: float,
    result: float,
    k: float = K_FACTOR,
) -> tuple[float, float]:
    """Update Elo ratings.  *result*: 1.0=A wins, 0.0=B wins, 0.5=tie."""
    ea = expected_score(rating_a, rating_b)
    new_a = rating_a + k * (result - ea)
    new_b = rating_b + k * ((1 - result) - (1 - ea))
    return new_a, new_b


def ensure_config_in_pool(pool: TaskPool, config_name: str) -> None:
    """Seed a config at the default rating if not already present."""
    if config_name not in pool.ratings:
        pool.ratings[config_name] = DEFAULT_RATING


def next_matchup(
    pool: TaskPool, max_per_pair: int = MAX_PER_PAIR,
) -> tuple[str, str] | None:
    """Pick the pair with the smallest rating gap that hasn't hit match cap.

    Returns ``None`` when all pairs have been judged *max_per_pair* times.
    """
    candidates: list[tuple[float, str, str]] = []
    for a, b in combinations(sorted(pool.ratings.keys()), 2):
        key = _pair_key(a, b)
        if pool.pair_counts.get(key, 0) >= max_per_pair:
            continue
        gap = abs(pool.ratings[a] - pool.ratings[b])
        candidates.append((gap, a, b))

    if not candidates:
        return None
    candidates.sort()  # smallest gap first → most informative match
    return candidates[0][1], candidates[0][2]


def record_match(
    pool: TaskPool,
    config_a: str,
    config_b: str,
    winner: str,
    confidence: float,
    reasoning: str,
) -> None:
    """Record a match result: update Elo ratings and bookkeeping."""
    result_val = {'A': 1.0, 'B': 0.0, 'tie': 0.5}.get(winner, 0.5)
    ra, rb = pool.ratings[config_a], pool.ratings[config_b]
    new_a, new_b = update_elo(ra, rb, result_val)
    pool.ratings[config_a] = round(new_a, 1)
    pool.ratings[config_b] = round(new_b, 1)

    key = _pair_key(config_a, config_b)
    pool.pair_counts[key] = pool.pair_counts.get(key, 0) + 1

    pool.matches.append({
        'config_a': config_a,
        'config_b': config_b,
        'winner': winner,
        'confidence': confidence,
        'reasoning': reasoning,
        'elo_before': {'a': ra, 'b': rb},
        'elo_after': {
            'a': pool.ratings[config_a],
            'b': pool.ratings[config_b],
        },
    })
