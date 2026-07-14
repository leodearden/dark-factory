"""scripts/legibility/census_trigger.py — periodic legibility census trigger
evaluator + census-state reader.

See plans/confusion-reduction-prd.md §6 (task ζ: fire logic + hard floor),
§7.4 (per-project census config block), §7.5 (census state contract),
§8.5 (boundary-test matrix — day-9-no-spike/day-7+130-landed/
day-6+4-candidates-in-72h/day-4+spike -> no-fire/fire/fire/no-fire(floor)).

Evaluated at the end of each nightly trickle run (wired by PRD task ε) and
via the standalone `evaluate` CLI subcommand below.

Extended census-state READ contract (for task η, which WRITES/advances
docs/legibility/census-state.json): in addition to the §7.5 minimal shape
`{last_census_at, last_census_report}`, this module reads an OPTIONAL
`last_census_done_count` integer baseline — the fused-memory get_statuses()
done-task count as of the last census, used to compute the "tasks landed
since last census" delta for condition (b). fused-memory's get_statuses
returns only a `{id: status}` status snapshot with no timestamps, so that
delta is uncomputable without a persisted baseline. When
`last_census_done_count` is absent (never censused, or η not yet writing
it), condition (b) fails SAFE — it never fires — rather than guessing.

This module deliberately does NOT import task β's `legibility.yaml` config
loader (β has not landed; this task's only dependency is γ / codebook.py):
`load_census_config` reads the `census:` block directly with a light
pyyaml read, falling back to the §7.4 defaults hardcoded in `CensusConfig`.

The get_statuses done-count fetch is injected (`status_fetcher`), not a
hardcoded MCP/HTTP client: the scripts/ test env (`uv run --project
shared`) has no httpx and no MCP client available (see
shared/pyproject.toml), so the pure decision core stays fully unit-testable
and "a failing get_statuses fails SAFE" is testable with a raising fake.
`default_status_fetcher` provides a best-effort glue implementation for the
standalone CLI; task ε injects the real MCP-backed fetcher.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("legibility.census_trigger")


def _as_utc(value: datetime | None) -> datetime | None:
    """Normalize a datetime to timezone-aware UTC. A naive datetime (e.g.
    parsed from a bare `YYYY-MM-DD` codebook date) is assumed to already be
    UTC. `None` passes through unchanged."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# CensusConfig — §7.4 census: block, with hardcoded defaults
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CensusConfig:
    """The six §7.4 census-trigger thresholds, with their documented
    defaults. `from_mapping` merges a partial override mapping (e.g. the
    `census:` sub-dict of a project's legibility.yaml) over these
    defaults."""

    max_interval_days: int = 10
    tasks_landed_threshold: int = 120
    tasks_landed_min_days: int = 7
    novelty_spike_count: int = 4
    novelty_spike_window_hours: int = 72
    floor_days: int = 5

    @classmethod
    def from_mapping(cls, mapping: dict | None) -> "CensusConfig":
        """Build a CensusConfig by merging `mapping` (shaped like §7.4's
        `census:` block, e.g. `{"max_interval_days": 3, "novelty_spike":
        {"count": 9}}`) over the defaults. Keys absent from `mapping`
        (including either nested `novelty_spike` key) keep their default
        value. `mapping=None` (or `{}`) returns plain defaults."""
        defaults = cls()
        mapping = mapping or {}
        novelty_spike = mapping.get("novelty_spike") or {}
        return cls(
            max_interval_days=mapping.get("max_interval_days", defaults.max_interval_days),
            tasks_landed_threshold=mapping.get(
                "tasks_landed_threshold", defaults.tasks_landed_threshold
            ),
            tasks_landed_min_days=mapping.get(
                "tasks_landed_min_days", defaults.tasks_landed_min_days
            ),
            novelty_spike_count=novelty_spike.get("count", defaults.novelty_spike_count),
            novelty_spike_window_hours=novelty_spike.get(
                "window_hours", defaults.novelty_spike_window_hours
            ),
            floor_days=mapping.get("floor_days", defaults.floor_days),
        )


# ---------------------------------------------------------------------------
# Decision + evaluate() — pure §8.5 decision core (no I/O)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    """The census-trigger verdict: whether to fire, plus one human-readable
    reason line per evaluated condition (and the floor), for logging /
    CLI display."""

    fire: bool
    reasons: list[str]


def evaluate(
    *,
    now: datetime,
    last_census_at: datetime | None,
    never_censused: bool,
    tasks_landed: int | None,
    candidate_first_seens: list[datetime],
    config: CensusConfig,
) -> Decision:
    """Pure decision core for the §6/§8.5 fire logic. Fires at the earliest
    of condition (a) max_interval_days, (b) tasks_landed_min_days +
    tasks_landed_threshold, (c) novelty_spike — currently only (a) is
    implemented; (b)/(c)/the hard floor are added by later steps. No I/O:
    all inputs are plain values so the full §8.5 matrix is testable without
    a filesystem or a live get_statuses call.
    """
    now_utc = _as_utc(now)
    last_utc = _as_utc(last_census_at)
    days_since = (
        (now_utc - last_utc).total_seconds() / 86400.0 if last_utc is not None else None
    )

    reasons: list[str] = []

    cond_a = days_since is not None and days_since >= config.max_interval_days
    if days_since is not None:
        reasons.append(
            "max-interval: {:.1f}d since last census (threshold {}d){}".format(
                days_since, config.max_interval_days, " -> FIRE" if cond_a else ""
            )
        )
    else:
        reasons.append("max-interval: no last-census anchor available -> N/A")

    cond_b = (
        days_since is not None
        and days_since >= config.tasks_landed_min_days
        and tasks_landed is not None
        and tasks_landed >= config.tasks_landed_threshold
    )
    if tasks_landed is None:
        reasons.append("tasks-landed: delta unavailable (no baseline/fetcher) -> N/A")
    elif days_since is None or days_since < config.tasks_landed_min_days:
        reasons.append(
            "tasks-landed: {} landed but only {:.1f}d elapsed (min {}d) -> N/A".format(
                tasks_landed, days_since if days_since is not None else 0.0,
                config.tasks_landed_min_days,
            )
        )
    else:
        reasons.append(
            "tasks-landed: {} landed since last census (threshold {}){}".format(
                tasks_landed, config.tasks_landed_threshold, " -> FIRE" if cond_b else ""
            )
        )

    fire = cond_a or cond_b

    return Decision(fire=fire, reasons=reasons)
