#!/usr/bin/env python3
"""Is a host's canonical build path and shared build cache in service?

Reads the verify observation log (``data/verify-logs/build-cache-observations.jsonl``)
for one host since a point in time, classifies each row healthy, degraded or
neutral per PRD C10 without ``expect_canon`` (so a ``real_path`` row is
neutral), and answers with the exit table of PRD C13. The PRD is
plans/canonical-build-path-shared-cache-prd.md.

The table is evaluated in order and the first match wins:

  74  the --observations file exists but cannot be read (a directory, or
      permission denied): the check has no input, so it judges nothing
  1   --canon-root differs from ``canon_root`` in the --status-json file
      (or that file is unreadable or lacks the key), or the latest row is
      degraded
  0   at least --min-runs healthy rows and the latest is healthy
  2   no row for the host since --since: nothing is producing observations
  75  otherwise (too few healthy rows yet, or the latest is neutral)
  64  usage error

One JSON line on stdout carries the verdict and the per-policy sums of runs,
hits, misses and cpu_secs; a one-line summary goes to stderr.
"""
from __future__ import annotations

import argparse
import enum
import json
import logging
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

ROW_TIME = "time"
ROW_HOST = "host"
ROW_BUILD_CACHE = "build_cache"
STATUS_CANON_ROOT = "canon_root"

EXIT_IN_SERVICE = 0
EXIT_DEGRADED = 1
EXIT_NO_OBSERVATIONS = 2
EXIT_USAGE = 64
EXIT_OBSERVATIONS_UNREADABLE = 74
EXIT_NOT_YET = 75

_LOG_PREFIX = "check_build_cache_in_service:"
_log = logging.getLogger("check_build_cache_in_service")


class Health(enum.Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    NEUTRAL = "neutral"


class Verdict(enum.Enum):
    CANON_MISMATCH = "canon_mismatch"
    DEGRADED = "degraded"
    IN_SERVICE = "in_service"
    NO_OBSERVATIONS = "no_observations"
    NOT_YET = "not_yet"
    OBSERVATIONS_UNREADABLE = "observations_unreadable"


EXIT_CODE_OF = {
    Verdict.CANON_MISMATCH: EXIT_DEGRADED,
    Verdict.DEGRADED: EXIT_DEGRADED,
    Verdict.IN_SERVICE: EXIT_IN_SERVICE,
    Verdict.NO_OBSERVATIONS: EXIT_NO_OBSERVATIONS,
    Verdict.NOT_YET: EXIT_NOT_YET,
    Verdict.OBSERVATIONS_UNREADABLE: EXIT_OBSERVATIONS_UNREADABLE,
}


@dataclass(frozen=True)
class Observation:
    time: datetime
    mode: str
    backend: str
    policy: str
    reason: str
    hits: int
    misses: int
    cpu_secs: float
    health: Health


def classify(build_cache: Mapping[str, object]) -> Health:
    backend = build_cache.get("backend")
    reason = build_cache.get("reason")
    if backend == "local_fallback" or reason == "read_only_server":
        return Health.DEGRADED
    cache_expected_and_used = backend == "shared"
    no_cache_expected = backend == "none" and build_cache.get("policy") == "incremental"
    if build_cache.get("mode") == "canon" and reason == "" and (
        cache_expected_and_used or no_cache_expected
    ):
        return Health.HEALTHY
    return Health.NEUTRAL


def parse_time(text: str) -> datetime:
    """An ISO-8601 instant; a naive value is read as UTC."""
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _observation_of(build_cache: Mapping[str, object], time: datetime) -> Observation | None:
    mode = build_cache.get("mode")
    backend = build_cache.get("backend")
    policy = build_cache.get("policy")
    reason = build_cache.get("reason")
    hits = build_cache.get("hits")
    misses = build_cache.get("misses")
    cpu_secs = build_cache.get("cpu_secs")
    if not (
        isinstance(mode, str)
        and isinstance(backend, str)
        and isinstance(policy, str)
        and isinstance(reason, str)
        and isinstance(hits, int)
        and isinstance(misses, int)
        and isinstance(cpu_secs, (int, float))
    ):
        return None
    return Observation(
        time=time,
        mode=mode,
        backend=backend,
        policy=policy,
        reason=reason,
        hits=hits,
        misses=misses,
        cpu_secs=float(cpu_secs),
        health=classify(build_cache),
    )


def _parse_row(line: str) -> tuple[str, Observation] | None:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict):
        return None
    host = record.get(ROW_HOST)
    stamp = record.get(ROW_TIME)
    build_cache = record.get(ROW_BUILD_CACHE)
    if not (isinstance(host, str) and isinstance(stamp, str) and isinstance(build_cache, dict)):
        return None
    try:
        time = parse_time(stamp)
    except ValueError:
        return None
    observation = _observation_of(build_cache, time)
    return None if observation is None else (host, observation)


def _host_rows(
    lines: Iterable[str], host: str, since: datetime
) -> tuple[list[Observation], int]:
    observations: list[Observation] = []
    malformed = 0
    for line in lines:
        if not line.strip():
            continue
        parsed = _parse_row(line)
        if parsed is None:
            malformed += 1
            continue
        row_host, observation = parsed
        if row_host == host and observation.time >= since:
            observations.append(observation)
    return observations, malformed


def read_observations(
    path: Path, host: str, since: datetime
) -> tuple[list[Observation], int]:
    """The host's rows at or after *since*, in file order, and the malformed-line count.

    An absent file holds no rows. Any other OSError (a directory, permission
    denied) propagates: the caller reports it, never as "no observations".
    """
    try:
        with path.open(encoding="utf-8", errors="replace") as lines:
            return _host_rows(lines, host, since)
    except FileNotFoundError:
        return [], 0


def latest_of(observations: Sequence[Observation]) -> Observation | None:
    """The row with the greatest time; on a tie, the later line."""
    if not observations:
        return None
    return max(reversed(observations), key=lambda observation: observation.time)


def canon_mismatch_reason(canon_root: str, status_json: Path) -> str | None:
    """Why the CANON in *status_json* is not *canon_root*, or None when it is."""
    try:
        status = json.loads(status_json.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"cannot read the canon_run.py status in {status_json}: {exc}"
    recorded = status.get(STATUS_CANON_ROOT) if isinstance(status, dict) else None
    if not isinstance(recorded, str):
        return f"{status_json} has no string {STATUS_CANON_ROOT!r} key"
    if os.path.normpath(recorded) != os.path.normpath(canon_root):
        return f"--canon-root {canon_root} differs from {STATUS_CANON_ROOT} {recorded} in {status_json}"
    return None


def decide(
    observations: Sequence[Observation],
    min_runs: int,
    canon_mismatch: str | None,
    observations_error: str | None,
) -> tuple[Verdict, str]:
    if observations_error is not None:
        return Verdict.OBSERVATIONS_UNREADABLE, observations_error
    if canon_mismatch is not None:
        return Verdict.CANON_MISMATCH, canon_mismatch
    latest = latest_of(observations)
    healthy = sum(1 for observation in observations if observation.health is Health.HEALTHY)
    if latest is not None and latest.health is Health.DEGRADED:
        return Verdict.DEGRADED, (
            f"the latest observation ({latest.time.isoformat()}) is degraded:"
            f" backend={latest.backend} reason={latest.reason!r}"
        )
    if latest is not None and latest.health is Health.HEALTHY and healthy >= min_runs:
        return Verdict.IN_SERVICE, f"{healthy} healthy observations, the latest healthy"
    if latest is None:
        return Verdict.NO_OBSERVATIONS, "no observation for the host in the window"
    return Verdict.NOT_YET, (
        f"{healthy} of {min_runs} healthy observations; the latest is {latest.health.value}"
    )


def per_policy_sums(observations: Iterable[Observation]) -> dict[str, dict[str, float | int]]:
    sums: dict[str, dict[str, float | int]] = {}
    for observation in observations:
        bucket = sums.setdefault(
            observation.policy, {"runs": 0, "hits": 0, "misses": 0, "cpu_secs": 0.0}
        )
        bucket["runs"] += 1
        bucket["hits"] += observation.hits
        bucket["misses"] += observation.misses
        bucket["cpu_secs"] += observation.cpu_secs
    return sums


def _latest_report(latest: Observation | None) -> dict[str, str] | None:
    if latest is None:
        return None
    return {
        "time": latest.time.isoformat(),
        "health": latest.health.value,
        "mode": latest.mode,
        "backend": latest.backend,
        "policy": latest.policy,
        "reason": latest.reason,
    }


class _UsageExitParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def _iso_instant(text: str) -> datetime:
    try:
        return parse_time(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not an ISO-8601 time: {text!r}") from exc


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1: {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = _UsageExitParser(
        prog="check_build_cache_in_service.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--since", type=_iso_instant, required=True)
    parser.add_argument("--min-runs", type=_positive_int, required=True)
    parser.add_argument("--canon-root")
    parser.add_argument("--status-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.canon_root is None) != (args.status_json is None):
        parser.error("--canon-root and --status-json are given together or not at all")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    observations_error = None
    try:
        observations, malformed = read_observations(args.observations, args.host, args.since)
    except OSError as exc:
        observations, malformed = [], 0
        observations_error = f"cannot read the observations in {args.observations}: {exc}"
    canon_mismatch = (
        None
        if args.canon_root is None
        else canon_mismatch_reason(args.canon_root, args.status_json)
    )
    verdict, reason = decide(observations, args.min_runs, canon_mismatch, observations_error)
    exit_code = EXIT_CODE_OF[verdict]
    counts = {
        health: sum(1 for observation in observations if observation.health is health)
        for health in Health
    }
    report = {
        "verdict": verdict.value,
        "exit_code": exit_code,
        "reason": reason,
        "host": args.host,
        "since": args.since.isoformat(),
        "observations": len(observations),
        "healthy": counts[Health.HEALTHY],
        "degraded": counts[Health.DEGRADED],
        "neutral": counts[Health.NEUTRAL],
        "malformed_lines": malformed,
        "latest": _latest_report(latest_of(observations)),
        "per_policy": per_policy_sums(observations),
    }
    print(json.dumps(report))
    _log.info(
        "%s %s (exit %d) for host %s since %s: %s",
        _LOG_PREFIX,
        verdict.value,
        exit_code,
        args.host,
        args.since.isoformat(),
        reason,
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
