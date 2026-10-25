"""scripts/legibility/plan_deviation_census.py — the nightly plan-deviation
census: counts recorded plan deviations and plan-deviation notes across tasks
and escalates a rate rise or a dominant pattern.

Design and rationale: plans/plan-deviation-recording-prd.md D8-D11.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping

from legibility import state_docs
from legibility.config import PlanDeviationCensus

logger = logging.getLogger('legibility.plan_deviation_census')

RECENT_WINDOW_DAYS = 7

_MAX_REPR_CHARS = 80


def _bounded(text: str, limit: int = _MAX_REPR_CHARS) -> str:
    """*text* on one line, cut to *limit* characters with an elision marker."""
    flat = text.replace('\n', '\\n')
    return flat if len(flat) <= limit else f'{flat[:limit]}...'


# ---------------------------------------------------------------------------
# settings — the raw config block, validated per field, never coerced
# ---------------------------------------------------------------------------

def _number(name: str, raw: object) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ValueError(f'{name}: expected a number, got {raw!r}')
    if not math.isfinite(raw):
        raise ValueError(f'{name}: expected a finite number, got {raw!r}')
    return raw


def _rate_rise_factor(name: str, raw: object) -> float:
    value = _number(name, raw)
    if value < 1.0:
        raise ValueError(f'{name}: expected a number >= 1.0, got {raw!r}')
    return value


def _share(name: str, raw: object) -> float:
    value = _number(name, raw)
    if not 0 < value <= 1:
        raise ValueError(f'{name}: expected a number in (0, 1], got {raw!r}')
    return value


def _positive_int(name: str, raw: object) -> int:
    value = state_docs.as_int(name, raw)
    if value < 1:
        raise ValueError(f'{name}: expected an integer >= 1, got {raw!r}')
    return value


def _horizon_days(name: str, raw: object) -> int:
    value = state_docs.as_int(name, raw)
    if value <= RECENT_WINDOW_DAYS:
        raise ValueError(f'{name}: expected an integer > {RECENT_WINDOW_DAYS}, got {raw!r}')
    return value


def _meta_roots(name: str, raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list | tuple) or not raw:
        raise ValueError(f'{name}: expected a non-empty list of paths, got {raw!r}')
    roots = tuple(state_docs.as_str(name, entry) for entry in raw)
    if not all(roots):
        raise ValueError(f'{name}: expected non-empty paths, got {raw!r}')
    return roots


_FIELD_PARSERS: Mapping[str, Callable[[str, object], object]] = {
    'enabled': state_docs.as_bool,
    'rate_rise_factor': _rate_rise_factor,
    'min_records': _positive_int,
    'pattern_share': _share,
    'prose_only_min': _positive_int,
    'cooldown_days': _positive_int,
    'horizon_days': _horizon_days,
    'meta_roots': _meta_roots,
}


def settings_from_mapping(raw: object) -> PlanDeviationCensus:
    """The census settings from the raw ``plan_deviation_census`` block.

    ``None`` means every default. Each known key is validated on its own; a
    rejected value keeps that field's default, so one typo cannot disarm the
    other triggers. Every rejection and unknown key is reported in exactly
    one WARNING for the block. Never raises."""
    if raw is None:
        return PlanDeviationCensus()
    if not isinstance(raw, Mapping):
        logger.warning(
            'plan_deviation_census block is a %s, not a mapping; using every default',
            type(raw).__name__,
        )
        return PlanDeviationCensus()
    accepted: dict[str, object] = {}
    rejected: list[str] = []
    unknown: list[str] = []
    for key, value in raw.items():
        parse = _FIELD_PARSERS.get(key) if isinstance(key, str) else None
        if parse is None:
            unknown.append(_bounded(repr(key)))
            continue
        try:
            accepted[key] = parse(key, value)
        except ValueError as exc:
            rejected.append(
                f'{key}={_bounded(repr(value))} ({type(value).__name__}): {exc}'
            )
    if rejected or unknown:
        logger.warning(
            'plan_deviation_census block: defaults kept for rejected value(s) [%s]; '
            'unknown key(s) ignored [%s]',
            '; '.join(rejected), ', '.join(unknown),
        )
    return PlanDeviationCensus.model_validate(accepted)
