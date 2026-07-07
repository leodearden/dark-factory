"""Pure priority-scoring library for the Fleet Cockpit (PRD section 6.3 / C3).

score() is a pure function: every input (including `now`) is injected, so
identical inputs always yield an identical score. Weights come from a
Priorities config (see priorities.default.yaml / load_priorities), never
from a global or a clock read.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ScoringItem:
    """A scoreable queue item.

    Any object exposing these six attributes works — score() reads
    attributes, not isinstance — so a real rail C1 DecisionRecord/
    SessionRecord satisfies this structurally once a caller derives
    severity/category from the linked escalation or decision kind.
    """

    severity: str
    category: str
    project: str
    filed_at: datetime
    manual_boost: int
    state: str


@dataclass(frozen=True)
class Defaults:
    """Fallback weights used when a severity/category/project key is unmapped."""

    severity: float
    category: float
    project: float


@dataclass(frozen=True)
class AgeCurve:
    """Bounded-linear age bonus: max_bonus * min(1, age_seconds / saturation_seconds)."""

    max_bonus: float
    saturation_seconds: float


@dataclass(frozen=True)
class ManualBoostConfig:
    """Weight applied to item.manual_boost, clamped to [min, max] first."""

    weight: float
    min: int
    max: int


@dataclass(frozen=True)
class Priorities:
    """Weight configuration for score(). See priorities.default.yaml for the bundled default."""

    severity_weights: dict[str, float]
    category_weights: dict[str, float]
    project_weights: dict[str, float]
    defaults: Defaults
    age_curve: AgeCurve
    manual_boost: ManualBoostConfig

    @classmethod
    def default(cls) -> Priorities:
        """Sane, non-negative hardcoded defaults (mirrored in priorities.default.yaml)."""
        return cls(
            severity_weights={'critical': 5.0, 'high': 3.0, 'medium': 1.5, 'low': 0.5},
            category_weights={'security': 2.0, 'bug': 1.0, 'feature': 0.5, 'chore': 0.2},
            project_weights={},
            defaults=Defaults(severity=1.0, category=0.5, project=0.0),
            age_curve=AgeCurve(max_bonus=2.0, saturation_seconds=float(7 * 24 * 3600)),
            manual_boost=ManualBoostConfig(weight=1.0, min=-5, max=5),
        )


# Integer-separated state tiers so the open band [2, 3) sits strictly above the
# answered band [1, 2) and the dropped/unknown band [0, 1) for ANY urgency in
# the half-open interval [0, 1) — independent of weights, age, or boost.
STATE_TIER: dict[str, float] = {
    'open': 2.0,
    'answered': 1.0,
    'dropped': 0.0,
}


def score(item: ScoringItem, weights: Priorities, now: datetime) -> float:
    """Score `item` for queue ordering. Pure: `now` is injected, no RNG."""
    clamped_boost = max(weights.manual_boost.min, min(item.manual_boost, weights.manual_boost.max))
    # max(0.0, ...) tolerates clock skew (now earlier than filed_at) by treating it as age 0.
    age_seconds = max(0.0, (now - item.filed_at).total_seconds())
    age_term = weights.age_curve.max_bonus * min(
        1.0, age_seconds / weights.age_curve.saturation_seconds
    )
    raw = (
        weights.severity_weights.get(item.severity, weights.defaults.severity)
        + weights.category_weights.get(item.category, weights.defaults.category)
        + weights.project_weights.get(item.project, weights.defaults.project)
        + weights.manual_boost.weight * clamped_boost
        + age_term
    )
    raw = max(0.0, raw)
    urgency = raw / (1.0 + raw)
    return STATE_TIER.get(item.state, 0.0) + urgency
