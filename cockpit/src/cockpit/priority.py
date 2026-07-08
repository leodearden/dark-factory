"""Pure priority-scoring library for the Fleet Cockpit (PRD section 6.3 / C3).

score() is a pure function: every input (including `now`) is injected, so
identical inputs always yield an identical score. Weights come from a
Priorities config (see priorities.default.yaml / load_priorities), never
from a global or a clock read.
"""

from __future__ import annotations

import importlib.resources
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


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
    """Score `item` for queue ordering. Pure: `now` is injected, no RNG.

    `now` and `item.filed_at` may independently be naive or tz-aware; a naive
    datetime is assumed to already be UTC so a caller mixing the two shapes
    never raises TypeError — a view must never crash on a timestamp shape it
    didn't control.
    """
    clamped_boost = max(weights.manual_boost.min, min(item.manual_boost, weights.manual_boost.max))
    filed_at = (
        item.filed_at if item.filed_at.tzinfo is not None else item.filed_at.replace(tzinfo=UTC)
    )
    now_aware = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    # max(0.0, ...) tolerates clock skew (now earlier than filed_at) by treating it as age 0.
    age_seconds = max(0.0, (now_aware - filed_at).total_seconds())
    saturation_seconds = weights.age_curve.saturation_seconds
    if saturation_seconds > 0:
        age_term = weights.age_curve.max_bonus * min(1.0, age_seconds / saturation_seconds)
    else:
        # A non-positive saturation_seconds is a misconfiguration (e.g. a
        # hand-edited priorities.yaml with age_curve.saturation_seconds: 0).
        # Treat it as instant saturation rather than dividing by zero/negative
        # — this branch is what keeps score() a total function over any
        # operator-supplied config.
        age_term = weights.age_curve.max_bonus if age_seconds > 0 else 0.0
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


def _default_priorities_path() -> Path:
    """Where an operator's editable priorities.yaml lives (outside the repo)."""
    return Path.home() / '.claude' / 'fleet' / 'priorities.yaml'


def _read_bundled_defaults_text() -> str:
    """Read the package-bundled priorities.default.yaml as raw text.

    Shared by _load_bundled_defaults (which parses it) and
    ensure_priorities_file (which writes it out verbatim) so the
    bundled-resource lookup lives in exactly one place.
    """
    defaults_path = importlib.resources.files('cockpit').joinpath('priorities.default.yaml')
    with importlib.resources.as_file(defaults_path) as p:
        return p.read_text()


def _load_bundled_defaults() -> dict[str, Any]:
    """Load the package-bundled priorities.default.yaml."""
    return yaml.safe_load(_read_bundled_defaults_text()) or {}


def _priorities_from_dict(data: dict[str, Any]) -> Priorities:
    """Build a Priorities from a parsed YAML dict, defaulting missing sections/fields."""
    fallback = Priorities.default()
    defaults_section = data.get('defaults') or {}
    age_curve_section = data.get('age_curve') or {}
    manual_boost_section = data.get('manual_boost') or {}
    return Priorities(
        severity_weights=data.get('severity_weights') or fallback.severity_weights,
        category_weights=data.get('category_weights') or fallback.category_weights,
        project_weights=data.get('project_weights') or fallback.project_weights,
        defaults=Defaults(
            severity=defaults_section.get('severity', fallback.defaults.severity),
            category=defaults_section.get('category', fallback.defaults.category),
            project=defaults_section.get('project', fallback.defaults.project),
        ),
        age_curve=AgeCurve(
            max_bonus=age_curve_section.get('max_bonus', fallback.age_curve.max_bonus),
            saturation_seconds=age_curve_section.get(
                'saturation_seconds', fallback.age_curve.saturation_seconds
            ),
        ),
        manual_boost=ManualBoostConfig(
            weight=manual_boost_section.get('weight', fallback.manual_boost.weight),
            min=manual_boost_section.get('min', fallback.manual_boost.min),
            max=manual_boost_section.get('max', fallback.manual_boost.max),
        ),
    )


def load_priorities(path: Path | None = None) -> Priorities:
    """Load Priorities from *path* (default ``~/.claude/fleet/priorities.yaml``).

    Fail-soft, per the cockpit's hard constraint (a view must never be a
    dependency): a missing user file silently falls back to the
    package-bundled defaults; a malformed user file falls back to the same
    defaults with a logged WARNING. This function never raises.
    """
    target = path if path is not None else _default_priorities_path()

    if target.exists():
        try:
            with open(target) as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as exc:
            logger.warning('load_priorities: failed to parse %s: %s — using defaults', target, exc)
            return Priorities.default()

        if not isinstance(data, dict):
            logger.warning(
                'load_priorities: expected a YAML mapping in %s, got %s — using defaults',
                target,
                type(data).__name__,
            )
            return Priorities.default()

        return _priorities_from_dict(data)

    return _priorities_from_dict(_load_bundled_defaults())


def ensure_priorities_file(path: Path | None = None) -> None:
    """Materialize an editable priorities.yaml at *path* if it doesn't exist yet.

    Default target is ``~/.claude/fleet/priorities.yaml``. If the target
    already exists, this is a no-op — never clobbers operator edits (e.g.
    from the C9b weight editor). Otherwise the parent dirs are created and
    the bundled ``priorities.default.yaml`` contents are copied in. Fail-soft
    on write error (logged, never raised) per the cockpit's hard constraint
    that a view must never be a dependency.
    """
    target = path if path is not None else _default_priorities_path()

    if target.exists():
        return

    try:
        contents = _read_bundled_defaults_text()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)
    except OSError as exc:
        logger.warning('ensure_priorities_file: failed to write %s: %s', target, exc)
