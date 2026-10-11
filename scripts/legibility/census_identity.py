"""Who one census run is, and what it reads the project at.

A run's identity (its ``run_id`` and the basename its report and record
share), the previous run as census-state.json records it, and the commit
the run reads (``as_of_sha``). Contract: plans/census-incremental-prd.md
§4.2.

Stdlib-only; imports nothing from census.py.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


@dataclass(frozen=True)
class PriorCensus:
    """What census-state.json records about the previous run. A field is
    ``None`` when the state is missing, malformed, or predates it."""

    last_census_at: date | None = None
    run_id: str | None = None
    as_of_sha: str | None = None


def _state_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def prior_census(status: str, state: dict | None) -> PriorCensus:
    """Read :func:`census_trigger.load_census_state`'s result, which has
    already validated ``last_census_at``."""
    if status != "ok" or state is None:
        return PriorCensus()
    last_census_at = _state_text(state.get("last_census_at"))
    return PriorCensus(
        last_census_at=(
            datetime.fromisoformat(last_census_at).date() if last_census_at else None
        ),
        run_id=_state_text(state.get("last_census_run_id")),
        as_of_sha=_state_text(state.get("last_census_as_of_sha")),
    )


@dataclass(frozen=True)
class RunIdentity:
    """One census run's ``run_id`` and the basename its report and record
    share (plans/census-incremental-prd.md §4.2)."""

    run_id: str
    basename: str


def _identity_taken(identity: RunIdentity, plans_dir: Path, taken_run_ids) -> bool:
    return identity.run_id in taken_run_ids or any(
        (plans_dir / f"{identity.basename}{suffix}").exists() for suffix in (".md", ".json")
    )


def allocate_run_identity(
    *,
    project_id: str,
    day: date,
    plans_dir: Path,
    taken_run_ids: frozenset[str],
    limit: int = 1000,
) -> RunIdentity:
    """The first free identity for a census on *day*: ``census-<project>-
    <YYYYMMDD>`` / ``confusion-census-<YYYY-MM-DD>``, else the same with
    ``-2``, ``-3``, ... A candidate is taken when its report or record
    exists in *plans_dir* or its run_id is in *taken_run_ids*, so a second
    run never overwrites the first. Exhausting *limit* raises
    ``RuntimeError`` naming the directory."""
    for n in range(1, limit + 1):
        suffix = "" if n == 1 else f"-{n}"
        candidate = RunIdentity(
            run_id=f"census-{project_id}-{day:%Y%m%d}{suffix}",
            basename=f"confusion-census-{day.isoformat()}{suffix}",
        )
        if not _identity_taken(candidate, plans_dir, taken_run_ids):
            return candidate
    raise RuntimeError(
        f"census: no free run identity for {day.isoformat()} -- {limit} census "
        f"reports for that day already exist in {plans_dir}"
    )


_GIT_REV_PARSE_TIMEOUT_SECS = 30


def resolve_as_of_sha(project_root: Path | str) -> str:
    """The commit this census reads the project at: its ``HEAD``. Raises
    ``RuntimeError`` naming *project_root* and git's own error."""
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=_GIT_REV_PARSE_TIMEOUT_SECS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"git rev-parse HEAD failed in {project_root}: {exc}") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"git rev-parse HEAD failed in {project_root} (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()
