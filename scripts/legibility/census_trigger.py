"""scripts/legibility/census_trigger.py — periodic legibility census trigger
evaluator + census-state reader.

See plans/confusion-reduction-prd.md §6 (task ζ: fire logic), §7.4
(per-project census config block), §7.5 (census state contract) and §8.5
(boundary-test matrix). The novelty-spike and floor contract lives in
plans/census-incremental-prd.md §4.7, superseding §8.5's
day-6+4-candidates-in-72h row: the spike is relative to a trailing baseline,
and the floor is anchored on the session watermark.

Evaluated at the end of each nightly trickle run (wired by PRD task ε) and
via the standalone `evaluate` CLI subcommand below.

Extended census-state READ contract (for task η, which WRITES/advances
docs/legibility/census-state.json): in addition to the §7.5 minimal shape
`{last_census_at, last_census_report}` and the optional run identity and
`session_watermark` of plans/census-incremental-prd.md §4.2, this module
reads an OPTIONAL `last_census_done_count` integer baseline — the fused-memory get_statuses()
done-task count as of the last census, used to compute the "tasks landed
since last census" delta for condition (b). fused-memory's get_statuses
returns only a `{id: status}` status snapshot with no timestamps, so that
delta is uncomputable without a persisted baseline. That baseline is
THREE-VALUED (task 3291):

  * an int — a real observed done-count as of the last census. A real `0`
    (a project with no done tasks) is included and is NOT treated as
    missing;
  * `null` — the count could not be OBSERVED at census time. Condition (b)
    fails SAFE, exactly as for an absent key, until the next successful
    census; condition (a) (`max_interval_days`) remains the unconditional
    backstop;
  * absent — never censused, or η not yet writing it. Condition (b) fails
    SAFE.

ANY other on-disk value is a fourth, MALFORMED case (task 4085): the state
file is hand-seedable (skills/census/SKILL.md), so a quoted `"2872"`, a float,
a JSON `true` or a negative count are all expected operator-authored inputs.
Those also fail SAFE — one WARNING naming the value and its type, condition
(b) inert, conditions (a)/(c) unaffected — rather than crashing (`"2872"`
raised `TypeError` out of the subtraction) or silently computing a nonsense
delta (`true` is an `int` subclass, so it meant a baseline of 1). The value is
VALIDATED, never coerced: see `compute_tasks_landed`.

Only a real int ever arms condition (b). A FABRICATED `0` used to be the
fourth case, and it was the dangerous one: `census.py` wrote 0 whenever the
get_statuses call failed, and that 0 was persisted as a real baseline on
2026-07-24 and on 2026-07-31. It is UNSOUND rather than immediately
visible, and the distinction matters for anyone debugging over-firing —
these are the measured facts (2026-07-31 decision replayed against the
pre-task code at merge-base `2b0afe47d9`, with the historical state and
codebook from `3e462e1b56^`):

  * while the fetch was ALSO broken the fabricated 0 was self-cancelling.
    The same relative-project_root defect zeroed `current_done` too, so the
    delta was `0 - 0 = 0` — far under the 120 threshold. Replay of the real
    pre-fix census.py gate reports `tasks-landed: 0 landed since last
    census (threshold 120)`: condition (b) did NOT fire. In the nightly
    trickle of the day it could not fire at all, since `run_nightly`
    defaulted `status_fetcher` to nothing and (b) was structurally N/A
    there. That hole is CLOSED: task 4148 made the seam default to
    `default_status_fetcher(cfg.project_root)`, so (b) is live on the
    nightly path and the sentence above is history, not current behaviour;
  * but it ARMS (b) the moment the fetch is repaired: with a working
    fetcher the same replay reports `tasks-landed: 2872 landed since last
    census (threshold 120) -> FIRE`, ~24x over. So the root-cause fix below
    would have DETONATED the poisoned baseline had the two not landed
    together with the on-disk data repair.

What actually fired the 2026-07-31 census was condition (c): `novelty-spike:
25 candidate(s) within 72h (threshold 4)`. Condition (a) did not fire either
(7.4d elapsed, 10d threshold). Recalibrating the novelty-spike thresholds is
NOT part of task 3291 — see the follow-up filed against
docs/legibility/legibility.yaml.

Task 3291 removed the fabricated-0 path; `census.advance_census_state` now
persists `null` for an unobservable count, and `extract_done_count` below is
the single chokepoint that refuses to turn an unusable payload into a
number.

This module does NOT import task β's `legibility.yaml` config *loader*
(`legibility.config.load_config`): that loader requires four mandatory
top-level fields and raises `pydantic.ValidationError` on malformed input,
which is the wrong contract for this module's fail-safe/silent-defaults
`census:` block reader -- `load_census_config` instead reads the `census:`
block directly with a light pyyaml read, falling back to defaults on a
missing/malformed file. Those defaults ARE sourced from β's
`legibility.config.Census` pydantic model (scripts/legibility/config.py),
though -- now that β has landed, it is the single source of truth for the
census threshold values, so `CensusConfig`'s fields read their defaults
from it rather than re-hardcoding them (see the `CensusConfig` docstring).

The get_statuses done-count fetch is injected (`status_fetcher`) so the
decision core stays pure and "a failing get_statuses fails SAFE" is testable
with a raising fake, whatever is listening on the host.
`default_status_fetcher` is the shared production fetcher for the `evaluate`
CLI, `census.py`'s `main()` and the nightly trickle's `run_nightly`. It calls
get_statuses through `shared.mcp_session.SyncMcpSession`, which hands back
the tool's unwrapped reply, and it resolves `project_root` to an ABSOLUTE
path first because fused-memory's `_normalize_project_root` rejects a
relative one. Whatever the fetcher returns is validated by
`extract_statuses` (and counted by `extract_done_count`), never read by an
inline `.get("statuses")` -- see those functions for why.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, overload

import yaml
from legibility import codebook
from legibility.config import Census as _LegibilityCensus

if (_SHARED_SRC := str(Path(__file__).resolve().parents[2] / "shared" / "src")) not in sys.path:
    sys.path.insert(0, _SHARED_SRC)

from shared.mcp_session import McpCallError, SyncMcpSession

logger = logging.getLogger("legibility.census_trigger")


@overload
def _as_utc(value: datetime) -> datetime: ...


@overload
def _as_utc(value: None) -> None: ...


def _as_utc(value: datetime | None) -> datetime | None:
    """Normalize a datetime to timezone-aware UTC. A naive datetime (e.g.
    parsed from a bare `YYYY-MM-DD` codebook date) is assumed to already be
    UTC. `None` passes through unchanged.

    Overloaded because the pass-through is exactly correlated with the input:
    the plain `datetime | None -> datetime | None` signature discarded that,
    so `evaluate`'s `now` (declared non-Optional) came back Optional and made
    every subsequent arithmetic on it a type error. The overloads state what
    the body already does; the runtime implementation is unchanged."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


# ---------------------------------------------------------------------------
# CensusConfig — §7.4 census: block, with hardcoded defaults
# ---------------------------------------------------------------------------

# Single source of truth for the census threshold *values*: task β's
# `Census` pydantic model (scripts/legibility/config.py), instantiated once
# at import time. `CensusConfig` below stays its own flat dataclass (not
# `Census` itself) because its nested `novelty_spike` block's shape doesn't
# match the flat attributes `evaluate()` and `from_mapping`
# read/merge -- but its field *defaults* are pulled from here so the two
# schemas cannot silently drift apart (review finding, task 2579 amendment
# pass: config.py did not exist yet when this module was first written).
_CENSUS_DEFAULTS = _LegibilityCensus()


@dataclass(frozen=True)
class CensusConfig:
    """The census-trigger thresholds. Field defaults are sourced
    from `legibility.config.Census` (see `_CENSUS_DEFAULTS` above), not
    re-hardcoded here. `from_mapping` merges a partial override mapping
    (e.g. the `census:` sub-dict of a project's legibility.yaml) over these
    defaults, validating each override so an unusable hand-edited value
    falls back to the default here rather than reaching `evaluate()`.

    Every field is therefore a non-negative `int` by construction, which is
    what lets `evaluate()` compare and arithmetic on them unguarded."""

    max_interval_days: int = _CENSUS_DEFAULTS.max_interval_days
    tasks_landed_threshold: int = _CENSUS_DEFAULTS.tasks_landed_threshold
    tasks_landed_min_days: int = _CENSUS_DEFAULTS.tasks_landed_min_days
    novelty_spike_count: int = _CENSUS_DEFAULTS.novelty_spike.count
    novelty_spike_window_hours: int = _CENSUS_DEFAULTS.novelty_spike.window_hours
    novelty_spike_multiple: int = _CENSUS_DEFAULTS.novelty_spike.multiple
    novelty_spike_baseline_days: int = _CENSUS_DEFAULTS.novelty_spike.baseline_days
    floor_days: int = _CENSUS_DEFAULTS.floor_days

    @classmethod
    def from_mapping(cls, mapping: dict | None) -> CensusConfig:
        """Build a CensusConfig by merging `mapping` (shaped like §7.4's
        `census:` block, e.g. `{"max_interval_days": 3, "novelty_spike":
        {"count": 9}}`) over the defaults. Keys absent from `mapping`
        (including either nested `novelty_spike` key) keep their default
        value. `mapping=None` (or `{}`) returns plain defaults.

        Every override is VALIDATED, never coerced, exactly as
        `compute_tasks_landed` validates the hand-seeded
        `last_census_done_count` baseline (task 4085) -- legibility.yaml is
        the same operator-authored-input class, so a value that is not a
        non-negative `int` is an expected input, not a hypothetical. A bad
        value is REJECTED: that one field falls back to its default and the
        whole batch is reported in exactly one WARNING. This method NEVER
        raises.

        Per-field fallback, not a whole-block reject, because the
        thresholds are independent: one typo'd `novelty_spike.count` must not
        also disarm the (a) max-interval backstop.

        The stakes are the never-raises contract of everything above this.
        Unvalidated, a quoted `max_interval_days: '10'` reaches `evaluate()`'s
        `days_since >= config.max_interval_days` and raises `TypeError` out of
        `load_census_config` (documented "never raises") and out of
        `decide_for_project` (same) into `census.main`, which calls it
        unguarded. A `novelty_spike.window_hours: null` reaches
        `timedelta(hours=None)`; a non-mapping `novelty_spike:` used to raise
        `AttributeError` from the `.get` below. `bool` is rejected explicitly
        because it is an `int` subclass, so `max_interval_days: true` would
        otherwise silently mean 1 -- a daily ~$100 census.
        """
        defaults = cls()
        mapping = mapping or {}
        rejected: list[str] = []

        novelty_spike = mapping.get("novelty_spike")
        if novelty_spike is None:
            novelty_spike = {}
        elif not isinstance(novelty_spike, dict):
            rejected.append(
                f"novelty_spike={_bounded_repr(novelty_spike)} "
                f"({type(novelty_spike).__name__}, expected a mapping)"
            )
            novelty_spike = {}

        def _threshold(label: str, source: dict, key: str, default: int) -> int:
            if key not in source:
                return default
            value = source[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                rejected.append(
                    f"{label}={_bounded_repr(value)} ({type(value).__name__})"
                )
                return default
            return value

        config = cls(
            max_interval_days=_threshold(
                "max_interval_days", mapping, "max_interval_days",
                defaults.max_interval_days,
            ),
            tasks_landed_threshold=_threshold(
                "tasks_landed_threshold", mapping, "tasks_landed_threshold",
                defaults.tasks_landed_threshold,
            ),
            tasks_landed_min_days=_threshold(
                "tasks_landed_min_days", mapping, "tasks_landed_min_days",
                defaults.tasks_landed_min_days,
            ),
            novelty_spike_count=_threshold(
                "novelty_spike.count", novelty_spike, "count",
                defaults.novelty_spike_count,
            ),
            novelty_spike_window_hours=_threshold(
                "novelty_spike.window_hours", novelty_spike, "window_hours",
                defaults.novelty_spike_window_hours,
            ),
            novelty_spike_multiple=_threshold(
                "novelty_spike.multiple", novelty_spike, "multiple",
                defaults.novelty_spike_multiple,
            ),
            novelty_spike_baseline_days=_threshold(
                "novelty_spike.baseline_days", novelty_spike, "baseline_days",
                defaults.novelty_spike_baseline_days,
            ),
            floor_days=_threshold(
                "floor_days", mapping, "floor_days", defaults.floor_days,
            ),
        )

        # ONE warning for the whole block, not one per field: this is a single
        # operator fault (a hand-edited census: block) and the message is one
        # nightly journal line. Every rejected value is named, with its type,
        # so the fix is obvious without opening the file.
        if rejected:
            logger.warning(
                "legibility config census: block has %d unusable value(s), each "
                "falling back to its default -- every §7.4 threshold must be a "
                "non-negative int (unquoted, not a float/bool): %s",
                len(rejected),
                "; ".join(rejected),
            )

        return config


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


class FloorAnchorKind(Enum):
    SESSION_WATERMARK = "session watermark"
    LAST_CENSUS_AT = "last_census_at (no watermark yet)"
    EARLIEST_CODEBOOK_DATE = "earliest codebook date (never censused)"


@dataclass(frozen=True)
class FloorAnchor:
    """The instant the census floor is measured from, and where it came from."""

    kind: FloorAnchorKind
    at: datetime


def _floor_condition(
    now_utc: datetime, anchor: FloorAnchor | None, config: CensusConfig
) -> tuple[bool, str]:
    if anchor is None:
        return True, (
            "floor: no anchor (no session watermark, last_census_at or codebook date)"
            " -> BLOCKS all conditions"
        )
    days = (now_utc - _as_utc(anchor.at)).total_seconds() / 86400.0
    blocks = days < config.floor_days
    line = f"floor: {days:.1f}d since {anchor.kind.value} (floor {config.floor_days}d)"
    return blocks, line + (" -> BLOCKS all conditions" if blocks else "")


def _count_within(first_seens_utc: list[datetime], *, end: datetime, hours: int) -> int:
    start = end - timedelta(hours=hours)
    return sum(1 for fs in first_seens_utc if start <= fs <= end)


def _novelty_condition(
    now_utc: datetime, candidate_first_seens: list[datetime], config: CensusConfig
) -> tuple[bool, str]:
    seen = [_as_utc(fs) for fs in candidate_first_seens]
    hours = config.novelty_spike_window_hours
    baseline_days = config.novelty_spike_baseline_days
    count = _count_within(seen, end=now_utc, hours=hours)
    head = f"novelty-spike: {count} within {hours}h"

    history_days = (now_utc - min(seen)).total_seconds() / 86400.0 if seen else 0.0
    if history_days < baseline_days:
        return False, (
            f"{head} (baseline needs {baseline_days}d of candidate history, "
            f"have {history_days:.1f}d) -> N/A"
        )

    baseline = (
        statistics.median(
            _count_within(seen, end=now_utc - timedelta(days=k), hours=hours)
            for k in range(1, baseline_days + 1)
        )
        if baseline_days
        else 0.0
    )
    spike = count >= config.novelty_spike_count and count >= config.novelty_spike_multiple * baseline
    line = f"{head} (baseline median {baseline:g}, x{config.novelty_spike_multiple})"
    return spike, line + (" -> FIRE" if spike else "")


def evaluate(
    *,
    now: datetime,
    last_census_at: datetime | None,
    floor_anchor: FloorAnchor | None,
    tasks_landed: int | None,
    candidate_first_seens: list[datetime],
    config: CensusConfig,
) -> Decision:
    """Pure decision core for the census fire logic. Fires at the earliest
    of condition (a) max_interval_days, (b) tasks_landed_min_days +
    tasks_landed_threshold, (c) novelty_spike, subject to a floor measured
    from `floor_anchor` (plans/census-incremental-prd.md §4.7 Floor): inside
    floor_days of the anchor nothing fires. The floor never fires by itself,
    and its reason line always names its anchor. The novelty condition is
    relative to the median of the daily window counts over the trailing
    baseline_days, and N/A until that much candidate history exists (PRD
    §4.7 (b)). No I/O: all inputs are plain values so the full matrix is
    testable without a filesystem or a live get_statuses call.
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
            f"tasks-landed: {tasks_landed} landed but only {days_since if days_since is not None else 0.0:.1f}d elapsed (min {config.tasks_landed_min_days}d) -> N/A"
        )
    else:
        reasons.append(
            "tasks-landed: {} landed since last census (threshold {}){}".format(
                tasks_landed, config.tasks_landed_threshold, " -> FIRE" if cond_b else ""
            )
        )

    cond_c, novelty_line = _novelty_condition(now_utc, candidate_first_seens, config)
    reasons.append(novelty_line)

    floor_blocks, floor_line = _floor_condition(now_utc, floor_anchor, config)
    reasons.append(floor_line)

    fire = (cond_a or cond_b or cond_c) and not floor_blocks

    return Decision(fire=fire, reasons=reasons)


# ---------------------------------------------------------------------------
# load_census_state — §7.5 census-state.json reader (three-valued)
# ---------------------------------------------------------------------------

CENSUS_STATE_RELPATH = Path("docs") / "legibility" / "census-state.json"


def census_state_path(project_root: str | Path) -> Path:
    """The ONE spelling of a project's census-state.json path: the census
    writes it, the trigger reads it, and the nightly records it for
    ``check_census_progress.py`` to re-read."""
    return Path(project_root) / CENSUS_STATE_RELPATH


_STATE_TIMESTAMP_KEYS = ("last_census_at", "session_watermark")


def _unparseable_timestamp(value: object) -> Exception | None:
    """Why *value* is not a usable state timestamp, or None when it is
    (absent and null are usable: the key is optional)."""
    if value is None:
        return None
    if not isinstance(value, str):
        return TypeError(f"expected an ISO-8601 string, got {type(value).__name__}")
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        return exc
    return None


def load_census_state(path: str | Path) -> tuple[str, dict | None]:
    """Read `docs/legibility/census-state.json` (§7.5, extended with the
    optional `last_census_done_count` baseline documented in this module's
    docstring, and with plans/census-incremental-prd.md §4.2's optional
    `last_census_run_id`, `last_census_as_of_sha` and `session_watermark`).
    Three-valued result distinguishing "never censused" from "fail safe":

    - path does not exist -> `("missing", None)`, no warning logged. A
      project that has never run a census is a normal, expected state, not
      a degradation.
    - unreadable / invalid JSON / non-dict top level / an unparseable
      `last_census_at` or `session_watermark` -> `("malformed", None)` +
      exactly one WARNING. Callers must fail SAFE (never fire) rather than
      guess a timestamp.
    - otherwise -> `("ok", data)`.
    """
    path = Path(path)
    if not path.exists():
        return "missing", None

    try:
        with open(path, encoding="utf-8") as f:
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

    for key in _STATE_TIMESTAMP_KEYS:
        error = _unparseable_timestamp(data.get(key))
        if error is not None:
            logger.warning(
                "census state at %s is malformed: unparseable %s %r: %s",
                path,
                key,
                data.get(key),
                error,
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
    anchor condition (a) and the floor when never censused), and every candidate's
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

    A file that parses but holds an unusable threshold VALUE (a quoted
    `'10'`, a float, a JSON `true`, a negative) is handled one level down by
    `CensusConfig.from_mapping`, which rejects that field alone -- default
    plus one WARNING for the batch -- so the other thresholds stay live.
    Between the two, this function returns a fully-typed `CensusConfig` for
    any input whatsoever, which is what makes its never-raises contract, and
    `decide_for_project`'s, actually hold.
    """
    path = Path(project_root) / "docs" / "legibility" / "legibility.yaml"
    if not path.exists():
        return CensusConfig()

    try:
        with open(path, encoding="utf-8") as f:
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


# Every `StatusFetchUnavailable` message below is re-emitted verbatim by
# `compute_tasks_landed`'s WARNING and by `census.run_census`'s "done-count
# unobservable (%s)" WARNING, i.e. into the nightly systemd journal. An
# unbounded `{payload!r}` there would dump a whole malformed-but-large body
# (get_statuses over a big project is thousands of entries) into a single log
# line -- the opposite of the legible failure this module is for. Diagnosis
# only ever needs the shape, so the reprs are truncated and the type
# name/key list, which is what actually identifies the fault, is always
# reported outside the truncated part.
_MAX_REPR_CHARS = 200
_MAX_REPORTED_KEYS = 20


def _bounded_repr(value, limit: int = _MAX_REPR_CHARS) -> str:
    """`repr(value)` truncated to *limit* chars with an explicit elision
    marker, so a log line stays one readable line regardless of payload size."""
    text = repr(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... (repr truncated, {len(text)} chars total)"


def _bounded_keys(mapping) -> str:
    """A bounded, order-stable summary of a mapping's keys. Sorts by `str` so
    a payload with mixed-type keys cannot raise `TypeError` from inside the
    error path itself -- a failure while reporting a failure is the worst
    possible place to be clever."""
    keys = sorted(mapping, key=str)
    shown = keys[:_MAX_REPORTED_KEYS]
    suffix = f", ... ({len(keys)} total)" if len(keys) > _MAX_REPORTED_KEYS else ""
    return f"{shown}{suffix}"


def extract_done_count(payload) -> int:
    """Count `"done"` values in a get_statuses payload, or raise
    `StatusFetchUnavailable` if *payload* is not a usable get_statuses
    envelope.

    The count is taken over `extract_statuses(payload)`, the SOLE place a
    get_statuses payload is validated, and that validation exists because
    the idiom it replaces silently fabricated baselines. Both `compute_tasks_landed` below and `census.run_census`
    used to inline `sum(1 for v in (payload.get("statuses") or {}).values()
    if v == "done")`. That `or {}` swallowed *every* unusable payload into a
    done-count of 0 -- no warning, no exception, no way to tell a real
    empty project from a failed call. fused-memory's `@mcp_tool_errors`
    returns its `{"error", "error_type"}` dict on an `isError: false`
    JSON-RPC response, so the client unwraps it as though it were a
    genuine result and the old idiom read it as 0. That fabricated 0
    was persisted as a real census baseline on 2026-07-24 and on 2026-07-31
    (task 3291) — an unsound baseline that stayed latent only for as long as
    the fetch itself was broken (the same defect zeroed `current_done`, so
    the delta was `0 - 0`), and that arms condition (b) with a delta of
    ~2872 — every done task ever, ~24x its 120 threshold — as soon as the
    fetch is repaired. See the module docstring for the replayed
    measurements, including what actually fired the 2026-07-31 census.

    `{"statuses": {}}` is deliberately a VALID done-count of 0: a project
    with no done tasks is a real, expected state, and
    `census.advance_census_state`'s contract explicitly guarantees a real 0
    is never dropped as falsy. The distinction that matters is
    presence-of-shape, not emptiness-of-content -- only a missing or
    non-mapping `statuses` key, a non-dict payload, or the tool-error
    envelope is treated as unavailable.

    Note this strictness deliberately lives HERE and not in the MCP client:
    `SyncMcpSession.call_tool` hands every caller the tool's reply as-is,
    and `submit_task`'s callers legitimately receive and inspect its
    `{"error": ...}` rejection dict. Raising at the get_statuses-specific
    boundary closes the hole for both done-count consumers without touching
    unrelated paths.

    Raises:
        StatusFetchUnavailable: with a message naming the offending shape --
            or quoting the server's own `error` text -- so the operator sees
            the real cause rather than a generic complaint. The identifying
            facts (type name, key list, container length) are always stated
            in full; any embedded value repr is bounded, because this message
            is re-emitted into the nightly journal as a single log line (see
            `_bounded_repr`).
    """
    return count_done(extract_statuses(payload))


def count_done(statuses: Mapping[str, Any]) -> int:
    """How many tasks of an `extract_statuses` mapping are `"done"`."""
    return sum(1 for status in statuses.values() if status == "done")


def extract_statuses(payload) -> Mapping[str, Any]:
    """The `{task id string: status}` mapping of a get_statuses payload, or
    raise `StatusFetchUnavailable` if *payload* is not a usable get_statuses
    envelope (see `extract_done_count` for why each shape is rejected).

    `{"statuses": {}}` is a valid, empty answer. A census run reads one
    snapshot through here and derives both its outcome feed and its
    done-count from it, so the two cannot disagree.
    """
    if not isinstance(payload, dict):
        raise StatusFetchUnavailable(
            f"get_statuses payload is not a dict (got {type(payload).__name__}): "
            f"{_bounded_repr(payload)}"
        )

    # The fused-memory tool-error envelope, checked before the shape
    # complaint so the operator sees the actual server-side reason (e.g.
    # "project_root must be a non-empty absolute path") instead of merely
    # being told a `statuses` key was missing.
    if "statuses" not in payload and "error" in payload:
        raise StatusFetchUnavailable(
            f"get_statuses returned a tool-error envelope "
            f"({payload.get('error_type') or 'error'}): "
            f"{_bounded_repr(payload.get('error'))}"
        )

    if "statuses" not in payload:
        raise StatusFetchUnavailable(
            f"get_statuses payload has no 'statuses' key (keys: {_bounded_keys(payload)})"
        )

    statuses = payload["statuses"]
    if not isinstance(statuses, Mapping):
        sized = f", len={len(statuses)}" if hasattr(statuses, "__len__") else ""
        raise StatusFetchUnavailable(
            f"get_statuses 'statuses' is not a mapping "
            f"(got {type(statuses).__name__}{sized}): {_bounded_repr(statuses)}"
        )

    return statuses


def compute_tasks_landed(*, state: dict | None, status_fetcher) -> int | None:
    """Fail-SAFE "tasks done since last census" delta for condition (b).

    Returns `None` (never fires condition (b)) plus exactly one WARNING
    when: `status_fetcher` is `None`; `state` has no `last_census_done_count`
    baseline (§7.5 extended read contract, see module docstring);
    `last_census_done_count` is PRESENT but is not a non-negative int -- a
    malformed hand-seeded baseline (task 4085; see below); calling
    `status_fetcher` raises for any reason; or the fetched payload is not a
    usable `{"statuses": {id: status}}` envelope (matching get_statuses' real
    shape -- fused-memory/src/fused_memory/server/tools.py:2665). In
    particular fused-memory's `{"error", "error_type"}` tool-error dict --
    which rides an `isError: false` JSON-RPC response and so reaches the
    fetcher as the tool's reply -- fails SAFE here rather than counting
    zero; see `extract_done_count` for why counting zero was the defect
    (task 3291). Otherwise returns `current_done - baseline`.

    A bad payload and an unreachable server deliberately share ONE code path
    and ONE warning, so there is no separate failure mode to reason about --
    only the warning TEXT distinguishes them in the journal.

    The baseline is VALIDATED, never coerced. `int("2872")` would silently
    bless a malformed state file and arm an expensive census run on an
    unaudited number -- exactly the defect class task 3291 closed on the fetch
    side (see `extract_done_count` on the fabricated-0 baseline). A value that
    is not what the documented contract says is REPORTED, not repaired behind
    the operator's back.
    """
    baseline = (state or {}).get("last_census_done_count")
    if baseline is None:
        logger.warning(
            "tasks-landed: no last_census_done_count baseline in census state "
            "-- condition (b) fails safe (no fire)"
        )
        return None

    # Checked BEFORE the status_fetcher guard on purpose: this is a state-side
    # fault with a different owner (the operator's state file), and reporting
    # the MORE SPECIFIC fault first is what keeps a mis-seeded baseline
    # visible instead of being masked by the generic "no status_fetcher
    # configured" line whenever a caller reaches here without one. Both public
    # entrypoints (`decide_for_project`, this function) still default
    # `status_fetcher=None`, so that arm stays live for any caller that
    # supplies nothing -- it is just no longer the NIGHTLY TRICKLE, whose
    # `run_nightly` has built `default_status_fetcher(cfg.project_root)` since
    # task 4148 (before that it passed None on every real run, which is what
    # made this ordering load-bearing in production rather than merely in
    # principle).
    # `bool` is rejected explicitly because it is an `int` subclass, so a
    # hand-seeded JSON `true` would otherwise mean `baseline == 1` and arm
    # condition (b) with every done task ever, silently. A negative count is
    # definitionally impossible and inflates the delta in the fire-ward
    # direction. `StatusFetchUnavailable` is deliberately NOT raised here --
    # that exception is a FETCH-side signal that `census.run_census` reads as
    # "the done-count could not be observed"; this is a different fault with a
    # different owner (the operator's state file).
    if not isinstance(baseline, int) or isinstance(baseline, bool) or baseline < 0:
        logger.warning(
            "tasks-landed: last_census_done_count is not a non-negative int "
            "(got %s: %s) -- condition (b) fails safe (no fire); the census-state "
            "baseline is hand-seedable (skills/census/SKILL.md) and this one is "
            "unusable",
            type(baseline).__name__,
            _bounded_repr(baseline),
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

    # Extraction is guarded separately from the fetch so the warning TEXT can
    # distinguish "server unreachable" from "server answered with something
    # unusable" -- but both land on the same return-None fail-safe branch.
    try:
        current_done = extract_done_count(payload)
    except StatusFetchUnavailable as exc:
        logger.warning("tasks-landed: get_statuses returned an unusable payload: %s", exc)
        return None

    return current_done - baseline


_FUSED_MEMORY_URL_ENV_VAR = "FUSED_MEMORY_MCP_URL"
_DEFAULT_FUSED_MEMORY_URL = "http://localhost:8002"  # dashboard.config.DEFAULT_FUSED_MEMORY_URLS[0]


def default_status_fetcher(project_root: str | Path):
    """Return a zero-arg get_statuses caller for *project_root*.

    The shared production factory for the `evaluate` CLI, `census.py`'s
    `main()` and the nightly trickle. It calls fused-memory at
    `$FUSED_MEMORY_MCP_URL` (default `http://localhost:8002`) and returns
    the tool's reply, the `{"statuses": {...}}` mapping
    `compute_tasks_landed` reads. Any failure raises `StatusFetchUnavailable`
    naming the URL.

    *project_root* is resolved to an ABSOLUTE path here, at the wire, for
    every consumer at once: the server resolves the argument against its own
    cwd, and fused-memory's `_normalize_project_root` rejects a relative
    path. An absolute root is sent unchanged.
    """
    project_root_str = str(Path(project_root).resolve())
    base_url = os.environ.get(_FUSED_MEMORY_URL_ENV_VAR, _DEFAULT_FUSED_MEMORY_URL)

    def _fetch() -> dict:
        try:
            with SyncMcpSession(
                base_url, client_name="legibility-census-trigger", timeout=10.0,
            ) as session:
                return session.call_tool("get_statuses", {"project_root": project_root_str})
        except McpCallError as exc:
            raise StatusFetchUnavailable(
                f"get_statuses unreachable at {base_url}: {exc}"
            ) from exc

    return _fetch


# ---------------------------------------------------------------------------
# decide_for_project — high-level assembly (config + state + codebook signal)
# ---------------------------------------------------------------------------

def _trigger_anchors(
    state: dict | None, *, never_censused: bool, earliest_codebook_date: datetime | None
) -> tuple[datetime | None, FloorAnchor | None]:
    """The instant max-interval and tasks-landed measure from, and the floor's anchor."""
    if never_censused:
        if earliest_codebook_date is None:
            return None, None
        return earliest_codebook_date, FloorAnchor(
            FloorAnchorKind.EARLIEST_CODEBOOK_DATE, earliest_codebook_date
        )
    state = state or {}
    last_census_at = _parse_date(state.get("last_census_at"))
    watermark = _parse_date(state.get("session_watermark"))
    if watermark is not None:
        return last_census_at, FloorAnchor(FloorAnchorKind.SESSION_WATERMARK, watermark)
    if last_census_at is None:
        return None, None
    return last_census_at, FloorAnchor(FloorAnchorKind.LAST_CENSUS_AT, last_census_at)


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

    This function NEVER raises: it always returns a `Decision`. Every risky
    input is guarded -- the config load and its individual threshold VALUES
    (`load_census_config` / `CensusConfig.from_mapping`), the census state
    (`load_census_state`), the codebook load, and the tasks-landed
    computation -- each degrading to a neutral value or a default plus one
    WARNING, so a fault in one signal takes out only the condition it feeds
    -- never the whole evaluation. All four are hand-editable operator
    inputs, so all four are validated rather than trusted; a `census:` block
    holding a quoted `max_interval_days: '10'` used to raise `TypeError` from
    inside `evaluate()` and defeat this promise (task 4085 amendment).
    That matters because the caller's fallback is coarse: the
    `evaluate` CLI's outermost catch-all turns any escaping exception into a
    blanket NO-FIRE, which would suppress the (a) max-interval backstop that
    exists precisely to survive a broken (b). `nightly.evaluate_census_step`
    documents this guarantee of this function; it is held here.
    """
    project_root = Path(project_root)
    now = now if now is not None else datetime.now(UTC)

    config = load_census_config(project_root)

    state_status, state = load_census_state(census_state_path(project_root))
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

    last_census_at, floor_anchor = _trigger_anchors(
        state, never_censused=never_censused, earliest_codebook_date=earliest_sighting
    )

    try:
        tasks_landed = compute_tasks_landed(state=state, status_fetcher=status_fetcher)
    except Exception as exc:  # noqa: BLE001 - a fail-safe decision must never crash the caller
        # `_bounded_repr(str(exc))`, not a bare `exc`: an escaping exception's
        # message is arbitrary (a chained fetch failure can carry a whole
        # get_statuses payload) and this WARNING is re-emitted as ONE nightly
        # journal line, exactly like every other message in this module. The
        # repr also escapes embedded newlines, so a multi-line exception text
        # cannot break the one-line guarantee either.
        logger.warning(
            "tasks-landed computation failed unexpectedly (%s) -- condition (b) "
            "fails safe (no fire); conditions (a)/(c) are unaffected: %s",
            type(exc).__name__,
            _bounded_repr(str(exc)),
        )
        # `None`, never `0` and never a re-raise: `evaluate()` already renders
        # None as "tasks-landed: delta unavailable ... -> N/A", so condition (b)
        # goes inert while the (a) max-interval backstop and (c) novelty-spike
        # still evaluate. Re-raising would let one fault in (b) discard the
        # whole decision via the CLI's outermost catch-all.
        tasks_landed = None

    return evaluate(
        now=now,
        last_census_at=last_census_at,
        floor_anchor=floor_anchor,
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
