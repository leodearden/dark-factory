"""The in-service check of scripts/deploy/check_build_cache_in_service.py.

Driven as a CLI over fixture observation and status files: one JSON line on
stdout, the exit code from the table in PRD C13
(plans/canonical-build-path-shared-cache-prd.md), with rows classified per
C10.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from cli_subprocess_timeout import cli_timeout_from_env
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

SCRIPT = (
    Path(__file__).resolve().parents[1] / "deploy" / "check_build_cache_in_service.py"
)
CLI_TIMEOUT = cli_timeout_from_env("CHECK_BUILD_CACHE_IN_SERVICE_TEST_TIMEOUT")

HOST = "leo-laptop"
OTHER_HOST = "leo-workstation"
CANON = "/opt/reify-canon/ws"
SINCE = datetime(2026, 11, 20, 0, 0, tzinfo=UTC)


def at(minutes: int) -> datetime:
    return SINCE + timedelta(minutes=minutes)


def row(
    time: datetime | str,
    host: str = HOST,
    mode: str = "canon",
    backend: str = "shared",
    policy: str = "shared",
    reason: str = "",
    hits: int = 0,
    misses: int = 0,
    cpu_secs: float = 0.0,
    **extra: Any,
) -> str:
    stamp = time if isinstance(time, str) else time.isoformat()
    record: dict[str, Any] = {
        "time": stamp,
        "host": host,
        "role": "task_verify",
        "task_id": "1",
        "build_cache": {
            "mode": mode,
            "backend": backend,
            "policy": policy,
            "reason": reason,
            "hits": hits,
            "misses": misses,
            "write_errors": 0,
            "uncached_machine_flags": 0,
            "detached_server": 0,
            "wall_secs": 1.0,
            "cpu_secs": cpu_secs,
        },
    }
    record.update(extra)
    return json.dumps(record)


def write_observations(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "build-cache-observations.jsonl"
    path.write_text("".join(f"{line}\n" for line in lines))
    return path


def write_status(tmp_path: Path, canon_root: str | None) -> Path:
    status: dict[str, Any] = {"enabled": True, "policy": "shared"}
    if canon_root is not None:
        status["canon_root"] = canon_root
    path = tmp_path / "canon-status.json"
    path.write_text(json.dumps(status))
    return path


def healthy_rows(count: int, start: int = 0) -> list[str]:
    return [row(at(start + i)) for i in range(count)]


def run_check(
    observations: Path,
    *extra: str,
    since: str | None = None,
    min_runs: str = "3",
    tz: str = "UTC",
) -> tuple[int, dict[str, Any], str]:
    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            str(SCRIPT),
            "--observations",
            str(observations),
            "--host",
            HOST,
            "--since",
            since if since is not None else SINCE.isoformat(),
            "--min-runs",
            min_runs,
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=CLI_TIMEOUT,
        env={**os.environ, "TZ": tz},
    )
    report = json.loads(proc.stdout) if proc.stdout.strip() else {}
    return proc.returncode, report, proc.stderr


class TestDegraded:
    def test_latest_local_fallback_is_degraded_despite_healthy_history(
        self, tmp_path: Path
    ) -> None:
        lines = [*healthy_rows(5), row(at(10), backend="local_fallback")]
        code, report, _ = run_check(write_observations(tmp_path, lines))
        assert code == 1
        assert report["verdict"] == "degraded"
        assert report["latest"]["backend"] == "local_fallback"

    def test_latest_read_only_server_is_degraded(self, tmp_path: Path) -> None:
        lines = [*healthy_rows(5), row(at(10), reason="read_only_server")]
        code, report, _ = run_check(write_observations(tmp_path, lines))
        assert code == 1
        assert report["verdict"] == "degraded"

    def test_canon_root_mismatch_is_degraded(self, tmp_path: Path) -> None:
        observations = write_observations(tmp_path, healthy_rows(5))
        status = write_status(tmp_path, "/opt/reify-canon/other")
        code, report, _ = run_check(
            observations, "--canon-root", CANON, "--status-json", str(status)
        )
        assert code == 1
        assert report["verdict"] == "canon_mismatch"

    def test_missing_status_file_fails_naming_it(self, tmp_path: Path) -> None:
        observations = write_observations(tmp_path, healthy_rows(5))
        status = tmp_path / "absent-status.json"
        code, report, _ = run_check(
            observations, "--canon-root", CANON, "--status-json", str(status)
        )
        assert code == 1
        assert report["verdict"] == "canon_mismatch"
        assert str(status) in report["reason"]

    def test_status_without_canon_root_fails_naming_it(self, tmp_path: Path) -> None:
        observations = write_observations(tmp_path, healthy_rows(5))
        status = write_status(tmp_path, None)
        code, report, _ = run_check(
            observations, "--canon-root", CANON, "--status-json", str(status)
        )
        assert code == 1
        assert report["verdict"] == "canon_mismatch"
        assert str(status) in report["reason"]
        assert "canon_root" in report["reason"]


class TestInService:
    def test_enough_healthy_rows_with_healthy_latest(self, tmp_path: Path) -> None:
        observations = write_observations(tmp_path, healthy_rows(3))
        status = write_status(tmp_path, CANON)
        code, report, _ = run_check(
            observations, "--canon-root", CANON, "--status-json", str(status)
        )
        assert code == 0
        assert report["verdict"] == "in_service"
        assert report["healthy"] == 3

    def test_backend_none_under_incremental_counts_as_healthy(
        self, tmp_path: Path
    ) -> None:
        lines = [row(at(i), backend="none", policy="incremental") for i in range(3)]
        code, report, _ = run_check(write_observations(tmp_path, lines))
        assert code == 0
        assert report["verdict"] == "in_service"


class TestNoObservations:
    def test_only_another_hosts_rows(self, tmp_path: Path) -> None:
        lines = [row(at(i), host=OTHER_HOST) for i in range(5)]
        code, report, _ = run_check(write_observations(tmp_path, lines))
        assert code == 2
        assert report["verdict"] == "no_observations"

    def test_only_rows_older_than_since(self, tmp_path: Path) -> None:
        lines = [row(at(-60 - i)) for i in range(5)]
        code, report, _ = run_check(write_observations(tmp_path, lines))
        assert code == 2
        assert report["verdict"] == "no_observations"

    def test_observations_file_absent(self, tmp_path: Path) -> None:
        code, report, _ = run_check(tmp_path / "absent.jsonl")
        assert code == 2
        assert report["verdict"] == "no_observations"


class TestObservationsUnreadable:
    def test_a_directory_is_an_io_failure_not_a_degraded_cache(
        self, tmp_path: Path
    ) -> None:
        directory = tmp_path / "observations-dir"
        directory.mkdir()
        code, report, _ = run_check(directory)
        assert code == 74
        assert report["verdict"] == "observations_unreadable"
        assert str(directory) in report["reason"]

    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-000 file")
    def test_permission_denied_is_an_io_failure(self, tmp_path: Path) -> None:
        observations = write_observations(tmp_path, healthy_rows(3))
        observations.chmod(0)
        try:
            code, report, _ = run_check(observations)
        finally:
            observations.chmod(0o600)
        assert code == 74
        assert report["verdict"] == "observations_unreadable"

    def test_beats_a_canon_mismatch(self, tmp_path: Path) -> None:
        directory = tmp_path / "observations-dir"
        directory.mkdir()
        status = write_status(tmp_path, "/opt/reify-canon/other")
        code, report, _ = run_check(
            directory, "--canon-root", CANON, "--status-json", str(status)
        )
        assert code == 74
        assert report["verdict"] == "observations_unreadable"


class TestNotYet:
    def test_too_few_healthy_rows(self, tmp_path: Path) -> None:
        code, report, _ = run_check(write_observations(tmp_path, healthy_rows(2)))
        assert code == 75
        assert report["verdict"] == "not_yet"

    def test_enough_healthy_but_latest_neutral_falls_through(
        self, tmp_path: Path
    ) -> None:
        lines = [*healthy_rows(5), row(at(10), mode="real_path", backend="none")]
        code, report, _ = run_check(write_observations(tmp_path, lines))
        assert code == 75
        assert report["verdict"] == "not_yet"
        assert report["neutral"] == 1


class TestFirstMatchWins:
    def test_degraded_latest_beats_too_few_healthy(self, tmp_path: Path) -> None:
        lines = [row(at(0)), row(at(1), backend="local_fallback")]
        code, report, _ = run_check(write_observations(tmp_path, lines))
        assert code == 1
        assert report["verdict"] == "degraded"

    def test_canon_mismatch_beats_no_rows(self, tmp_path: Path) -> None:
        observations = write_observations(tmp_path, [])
        status = write_status(tmp_path, "/opt/reify-canon/other")
        code, report, _ = run_check(
            observations, "--canon-root", CANON, "--status-json", str(status)
        )
        assert code == 1
        assert report["verdict"] == "canon_mismatch"


class TestReport:
    def test_per_policy_sums_only_this_hosts_window(self, tmp_path: Path) -> None:
        lines = [
            row(at(0), policy="shared", hits=10, misses=2, cpu_secs=1.5),
            row(at(1), policy="shared", hits=20, misses=3, cpu_secs=2.5),
            row(at(2), backend="none", policy="incremental", misses=7, cpu_secs=4.0),
            row(at(3), host=OTHER_HOST, policy="shared", hits=1000, misses=1000),
            row(at(-30), policy="shared", hits=500, misses=500, cpu_secs=99.0),
        ]
        _, report, _ = run_check(write_observations(tmp_path, lines))
        assert report["per_policy"] == {
            "shared": {"runs": 2, "hits": 30, "misses": 5, "cpu_secs": 4.0},
            "incremental": {"runs": 1, "hits": 0, "misses": 7, "cpu_secs": 4.0},
        }
        assert report["observations"] == 3

    def test_malformed_lines_are_counted_and_ignored(self, tmp_path: Path) -> None:
        broken = json.dumps({"time": at(4).isoformat(), "host": HOST})
        lines = [*healthy_rows(3), "not json at all", broken]
        code, report, _ = run_check(write_observations(tmp_path, lines))
        assert report["malformed_lines"] == 2
        assert report["observations"] == 3
        assert code == 0

    def test_naive_since_is_read_as_utc(self, tmp_path: Path) -> None:
        lines = [row(at(-5)), *healthy_rows(3)]
        naive_since = SINCE.replace(tzinfo=None).isoformat()
        code, report, _ = run_check(
            write_observations(tmp_path, lines), since=naive_since, tz="Pacific/Auckland"
        )
        assert report["observations"] == 3
        assert code == 0

    def test_naive_row_time_is_read_as_utc(self, tmp_path: Path) -> None:
        naive_before_since = at(-5).replace(tzinfo=None).isoformat()
        lines = [row(naive_before_since), *healthy_rows(3)]
        code, report, _ = run_check(
            write_observations(tmp_path, lines), tz="America/Los_Angeles"
        )
        assert report["observations"] == 3
        assert code == 0

    def test_stderr_carries_a_prefixed_summary(self, tmp_path: Path) -> None:
        _, _, stderr = run_check(write_observations(tmp_path, healthy_rows(3)))
        assert "check_build_cache_in_service:" in stderr


class TestUsage:
    def test_canon_root_without_status_json_is_a_usage_error(
        self, tmp_path: Path
    ) -> None:
        observations = write_observations(tmp_path, healthy_rows(3))
        code, _, _ = run_check(observations, "--canon-root", CANON)
        assert code == 64

    def test_min_runs_zero_is_a_usage_error(self, tmp_path: Path) -> None:
        observations = write_observations(tmp_path, healthy_rows(3))
        code, _, _ = run_check(observations, min_runs="0")
        assert code == 64

    def test_help_prints_the_exit_table(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-I", str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT,
        )
        assert proc.returncode == 0
        table_rows = [line.split()[0] for line in proc.stdout.splitlines() if line[:4].strip().isdigit()]
        assert table_rows == ["74", "1", "0", "2", "75", "64"]
