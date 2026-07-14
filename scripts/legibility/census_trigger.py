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

This module does NOT import task β's `legibility.yaml` config *loader*
(`legibility.config.load_config`): that loader requires four mandatory
top-level fields and raises `pydantic.ValidationError` on malformed input,
which is the wrong contract for this module's fail-safe/silent-defaults
`census:` block reader -- `load_census_config` instead reads the `census:`
block directly with a light pyyaml read, falling back to defaults on a
missing/malformed file. Those defaults ARE sourced from β's
`legibility.config.Census` pydantic model (scripts/legibility/config.py),
though -- now that β has landed, it is the single source of truth for the
six §7.4 threshold values, so `CensusConfig`'s fields read their defaults
from it rather than re-hardcoding them (see the `CensusConfig` docstring).

The get_statuses done-count fetch is injected (`status_fetcher`), not a
hardcoded MCP/HTTP client: the scripts/ test env (`uv run --project
shared`) has no httpx and no MCP client available (see
shared/pyproject.toml), so the pure decision core stays fully unit-testable
and "a failing get_statuses fails SAFE" is testable with a raising fake.
`default_status_fetcher` provides a best-effort glue implementation for the
standalone CLI; task ε injects the real MCP-backed fetcher.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from legibility import codebook
from legibility.config import Census as _LegibilityCensus

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

# Single source of truth for the six §7.4 threshold *values*: task β's
# `Census` pydantic model (scripts/legibility/config.py), instantiated once
# at import time. `CensusConfig` below stays its own flat dataclass (not
# `Census` itself) because its nested `novelty_spike.count`/`window_hours`
# shape doesn't match the flat attributes `evaluate()` and `from_mapping`
# read/merge -- but its field *defaults* are pulled from here so the two
# schemas cannot silently drift apart (review finding, task 2579 amendment
# pass: config.py did not exist yet when this module was first written).
_CENSUS_DEFAULTS = _LegibilityCensus()


@dataclass(frozen=True)
class CensusConfig:
    """The six §7.4 census-trigger thresholds. Field defaults are sourced
    from `legibility.config.Census` (see `_CENSUS_DEFAULTS` above), not
    re-hardcoded here. `from_mapping` merges a partial override mapping
    (e.g. the `census:` sub-dict of a project's legibility.yaml) over these
    defaults."""

    max_interval_days: int = _CENSUS_DEFAULTS.max_interval_days
    tasks_landed_threshold: int = _CENSUS_DEFAULTS.tasks_landed_threshold
    tasks_landed_min_days: int = _CENSUS_DEFAULTS.tasks_landed_min_days
    novelty_spike_count: int = _CENSUS_DEFAULTS.novelty_spike.count
    novelty_spike_window_hours: int = _CENSUS_DEFAULTS.novelty_spike.window_hours
    floor_days: int = _CENSUS_DEFAULTS.floor_days

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

    window_start = now_utc - timedelta(hours=config.novelty_spike_window_hours)
    in_window = [
        fs for fs in candidate_first_seens if window_start <= _as_utc(fs) <= now_utc
    ]
    cond_c = len(in_window) >= config.novelty_spike_count
    reasons.append(
        "novelty-spike: {} candidate(s) within {}h (threshold {}){}".format(
            len(in_window),
            config.novelty_spike_window_hours,
            config.novelty_spike_count,
            " -> FIRE" if cond_c else "",
        )
    )

    triggered = cond_a or cond_b or cond_c

    floor_blocks = (
        not never_censused and days_since is not None and days_since < config.floor_days
    )
    if floor_blocks:
        reasons.append(
            "floor: only {:.1f}d since last census (floor {}d) -> BLOCKS all conditions".format(
                days_since, config.floor_days
            )
        )
    elif never_censused:
        reasons.append("floor: never censused -> exempt")

    fire = triggered and not floor_blocks

    return Decision(fire=fire, reasons=reasons)


# ---------------------------------------------------------------------------
# load_census_state — §7.5 census-state.json reader (three-valued)
# ---------------------------------------------------------------------------

def load_census_state(path: str | Path) -> tuple[str, dict | None]:
    """Read `docs/legibility/census-state.json` (§7.5, extended with the
    optional `last_census_done_count` baseline documented in this module's
    docstring). Three-valued result distinguishing "never censused" from
    "fail safe":

    - path does not exist -> `("missing", None)`, no warning logged. A
      project that has never run a census is a normal, expected state, not
      a degradation.
    - unreadable / invalid JSON / non-dict top level / unparseable
      `last_census_at` -> `("malformed", None)` + exactly one WARNING.
      Callers must fail SAFE (never fire) rather than guess a timestamp.
    - otherwise -> `("ok", data)`.
    """
    path = Path(path)
    if not path.exists():
        return "missing", None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("census state at %s is malformed: %s", path, exc)
        return "malformed", None

    if not isinstance(data, dict):
        logger.warning(
            "census state at %s is malformed: expected a JSON object, got %s",
            path,
            type(data).__name__,
        )
        return "malformed", None

    last_census_at = data.get("last_census_at")
    if last_census_at is not None:
        try:
            datetime.fromisoformat(last_census_at)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "census state at %s is malformed: unparseable last_census_at %r: %s",
                path,
                last_census_at,
                exc,
            )
            return "malformed", None

    return "ok", data


# ---------------------------------------------------------------------------
# codebook_signal — extract the never-censused anchor + novelty-spike dates
# from a task γ / codebook.load() dict (§7.3 candidates schema)
# ---------------------------------------------------------------------------

def _parse_date(value: object) -> datetime | None:
    """Parse an ISO-ish date/datetime string; return None (never raise) for
    anything unparseable or non-string, so one bad codebook record can't
    take down the whole evaluation."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def codebook_signal(codebook: dict) -> tuple[datetime | None, list[datetime]]:
    """Extract the census-trigger signal from a `codebook.load()`-shaped
    dict (task γ / 2575's schema): the earliest structured sighting or
    candidate `first_seen`/sighting date across the WHOLE codebook (used to
    anchor condition (a) when `never_censused`), and every candidate's
    `first_seen` date on its own (used for condition (c)'s novelty-spike
    window count). Unparseable dates are skipped rather than raising.

    Returns `(earliest_or_None, sorted_candidate_first_seens)`. An empty
    codebook (`entries: []`, `candidates: []`) returns `(None, [])`.
    """
    all_dates: list[datetime] = []
    candidate_first_seens: list[datetime] = []

    for entry in codebook.get("entries") or []:
        for sighting in entry.get("sightings") or []:
            parsed = _parse_date(sighting.get("date"))
            if parsed is not None:
                all_dates.append(parsed)

    for candidate in codebook.get("candidates") or []:
        first_seen = _parse_date(candidate.get("first_seen"))
        if first_seen is not None:
            candidate_first_seens.append(first_seen)
            all_dates.append(first_seen)
        for sighting in candidate.get("sightings") or []:
            parsed = _parse_date(sighting.get("date"))
            if parsed is not None:
                all_dates.append(parsed)

    earliest = min(all_dates) if all_dates else None
    return earliest, sorted(candidate_first_seens)


# ---------------------------------------------------------------------------
# load_census_config — §7.4 census: block reader (independent of task β)
# ---------------------------------------------------------------------------

def load_census_config(project_root: str | Path) -> CensusConfig:
    """Read `<project_root>/docs/legibility/legibility.yaml`'s `census:`
    sub-dict (§7.4) directly via a light pyyaml read -- deliberately NOT
    task β's `legibility.config.load_config` loader, whose four required
    top-level fields and raise-on-malformed-input contract are the wrong
    shape for this fail-safe reader (see module docstring). An absent file
    returns defaults silently (no legibility.yaml is a normal pre-adoption
    state). A malformed file (YAML parse error, non-dict top level, or a
    non-dict `census:` block) returns defaults plus exactly one WARNING --
    never raises. Those defaults are `Census`'s, not independently
    hardcoded -- see `CensusConfig`.
    """
    path = Path(project_root) / "docs" / "legibility" / "legibility.yaml"
    if not path.exists():
        return CensusConfig()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("legibility config at %s is malformed: %s", path, exc)
        return CensusConfig()

    if not isinstance(data, dict):
        logger.warning(
            "legibility config at %s is malformed: expected a YAML mapping, got %s",
            path,
            type(data).__name__,
        )
        return CensusConfig()

    census_block = data.get("census")
    if census_block is not None and not isinstance(census_block, dict):
        logger.warning(
            "legibility config at %s has a malformed census: block (expected a mapping, got %s)",
            path,
            type(census_block).__name__,
        )
        return CensusConfig()

    return CensusConfig.from_mapping(census_block)


# ---------------------------------------------------------------------------
# compute_tasks_landed + default_status_fetcher — fail-safe done-count delta
# ---------------------------------------------------------------------------

class StatusFetchUnavailable(Exception):
    """Raised by a status_fetcher (in particular `default_status_fetcher`'s
    returned callable) when the get_statuses done-count cannot be obtained
    for any reason -- missing dependency, no network, a non-2xx response, a
    malformed payload. A dedicated exception (rather than letting an
    arbitrary one propagate) lets `compute_tasks_landed` -- and any other
    caller -- catch fetch failures deterministically."""


def compute_tasks_landed(*, state: dict | None, status_fetcher) -> int | None:
    """Fail-SAFE "tasks done since last census" delta for condition (b).

    Returns `None` (never fires condition (b)) plus exactly one WARNING
    when: `status_fetcher` is `None`; `state` has no `last_census_done_count`
    baseline (§7.5 extended read contract, see module docstring); or calling
    `status_fetcher` raises for any reason. Otherwise counts `"done"` values
    in the fetcher's wrapped `{"statuses": {id: status}}` envelope (matching
    get_statuses' real shape -- fused-memory/src/fused_memory/server/tools.py:2665)
    and returns `current_done - baseline`.
    """
    baseline = (state or {}).get("last_census_done_count")
    if baseline is None:
        logger.warning(
            "tasks-landed: no last_census_done_count baseline in census state "
            "-- condition (b) fails safe (no fire)"
        )
        return None

    if status_fetcher is None:
        logger.warning(
            "tasks-landed: no status_fetcher configured -- condition (b) fails safe (no fire)"
        )
        return None

    try:
        payload = status_fetcher()
    except Exception as exc:  # noqa: BLE001 - any fetch failure must fail safe
        logger.warning("tasks-landed: status_fetcher failed: %s", exc)
        return None

    statuses = payload.get("statuses") or {} if isinstance(payload, dict) else {}
    current_done = sum(1 for status in statuses.values() if status == "done")
    return current_done - baseline


_FUSED_MEMORY_URL_ENV_VAR = "FUSED_MEMORY_MCP_URL"
_DEFAULT_FUSED_MEMORY_URL = "http://localhost:8002"  # dashboard.config.DEFAULT_FUSED_MEMORY_URLS[0]


def default_status_fetcher(project_root: str | Path):
    """Return a zero-arg best-effort get_statuses caller for the standalone
    `evaluate` CLI (task ε injects the real MCP-backed fetcher for the
    nightly trickle instead -- see module docstring). Reads the fused-memory
    MCP endpoint from the `FUSED_MEMORY_MCP_URL` env var, defaulting to
    `http://localhost:8002`. `httpx` is imported lazily since it is not a
    scripts/ dependency (see module docstring); that ImportError, along with
    any network/HTTP/parse failure, is wrapped as `StatusFetchUnavailable`
    so callers can catch fetch failures deterministically rather than a bare
    Exception.
    """
    project_root_str = str(project_root)
    url = os.environ.get(_FUSED_MEMORY_URL_ENV_VAR, _DEFAULT_FUSED_MEMORY_URL)

    def _fetch() -> dict:
        try:
            import httpx
        except ImportError as exc:
            raise StatusFetchUnavailable("httpx is not installed") from exc

        try:
            response = httpx.post(
                f"{url}/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "get_statuses",
                        "arguments": {"project_root": project_root_str},
                    },
                },
                timeout=10.0,
            )
            response.raise_for_status()
            return response.json()
        except StatusFetchUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - any failure must fail safe
            raise StatusFetchUnavailable(
                f"get_statuses unreachable at {url}: {exc}"
            ) from exc

    return _fetch


# ---------------------------------------------------------------------------
# decide_for_project — high-level assembly (config + state + codebook signal)
# ---------------------------------------------------------------------------

def decide_for_project(
    project_root: str | Path,
    *,
    now: datetime | None = None,
    status_fetcher=None,
) -> Decision:
    """Assemble the full §6/§8.5 census-trigger Decision for `project_root`:
    loads the `census:` config (§7.4), the census state (§7.5, extended),
    and the codebook novelty signal (task γ), computes the tasks-landed
    delta via `status_fetcher`, and calls the pure `evaluate()` core. This
    is what task ε's nightly trickle imports and what the `evaluate` CLI
    subcommand below calls.

    `now` defaults to the current UTC time. A malformed census state
    short-circuits to a fail-safe `Decision(fire=False, ...)` immediately
    (before touching the codebook or `status_fetcher`) -- `load_census_state`
    has already logged its one WARNING, so nothing else needs to.
    """
    project_root = Path(project_root)
    now = now if now is not None else datetime.now(timezone.utc)

    config = load_census_config(project_root)

    state_status, state = load_census_state(
        project_root / "docs" / "legibility" / "census-state.json"
    )
    if state_status == "malformed":
        return Decision(
            fire=False,
            reasons=["census state is malformed -- failing safe (no fire)"],
        )

    never_censused = state_status == "missing"

    codebook_path = project_root / "docs" / "legibility" / "confusion-codebook.yaml"
    try:
        codebook_data = codebook.load(codebook_path)
        if not isinstance(codebook_data, dict):
            raise ValueError(f"expected a YAML mapping, got {type(codebook_data).__name__}")
        earliest_sighting, candidate_first_seens = codebook_signal(codebook_data)
    except Exception as exc:  # noqa: BLE001 - a bad codebook must fail safe, not crash
        logger.warning("codebook at %s is unreadable: %s", codebook_path, exc)
        earliest_sighting, candidate_first_seens = None, []

    if never_censused:
        last_census_at = earliest_sighting
    else:
        raw_last_census_at = (state or {}).get("last_census_at")
        last_census_at = (
            datetime.fromisoformat(raw_last_census_at) if raw_last_census_at else None
        )

    tasks_landed = compute_tasks_landed(state=state, status_fetcher=status_fetcher)

    return evaluate(
        now=now,
        last_census_at=last_census_at,
        never_censused=never_censused,
        tasks_landed=tasks_landed,
        candidate_first_seens=candidate_first_seens,
        config=config,
    )


# ---------------------------------------------------------------------------
# CLI — `evaluate --project-root <path>` (always exits 0; fail-safe, never
# crashes the nightly trickle that calls this at the end of a run -- PRD
# task ε)
# ---------------------------------------------------------------------------

def _cmd_evaluate(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root)
    status_fetcher = default_status_fetcher(project_root)
    decision = decide_for_project(project_root, status_fetcher=status_fetcher)

    print(f"DECISION: {'FIRE' if decision.fire else 'NO-FIRE'}")
    for reason in decision.reasons:
        print(reason)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="census_trigger",
        description="Legibility periodic-census trigger evaluator "
        "(plans/confusion-reduction-prd.md §6/§8.5). Always exits 0 -- "
        "fail-safe, never crashes the nightly trickle.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate whether the periodic census should fire"
    )
    evaluate_parser.add_argument("--project-root", default=".")
    evaluate_parser.set_defaults(func=_cmd_evaluate)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001 - never crash the nightly trickle
        logger.warning("census_trigger evaluate failed unexpectedly: %s", exc)
        print(f"DECISION: NO-FIRE (evaluator error: {exc})")
        return 0


if __name__ == "__main__":
    sys.exit(main())
