#!/usr/bin/env python3
"""Prune the shared build-cache store down to a size cap.

The rule is PRD C13's (plans/canonical-build-path-shared-cache-prd.md): least
recently used objects first, by the later of access and modification time,
until the store's total size is at or under --cap-bytes. Files modified less
than --min-age-secs ago are never removed, because rclone writes a PUT in place
under its final name; directories are never removed. After any removal the
store unit's main process gets one SIGHUP so rclone drops its directory cache;
without it rclone keeps serving a removed object as a truncated 200. That pid
is resolved before anything is removed, so a failing systemctl removes nothing.

It deletes by default, since a timer runs it; --check is the dry run. One JSON
line on stdout, logs on stderr. Exit 0 for a completed pass, even one still
over cap (reported as over_cap); exit 1 when the store root is missing or the
systemctl call fails, in which case nothing was removed.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

EXIT_OK = 0
EXIT_FAILED = 1

DEFAULT_MIN_AGE_SECS = 600.0
DEFAULT_RCLONE_UNIT = "build-cache-store.service"

_LOG_PREFIX = "build_cache_store_prune:"
_log = logging.getLogger("build_cache_store_prune")


class SystemctlError(RuntimeError):
    pass


@dataclass(frozen=True)
class StoreFile:
    path: str
    size: int
    mtime: float
    last_use: float


def scan(root: Path) -> list[StoreFile]:
    """Every regular file under *root*; symlinks are neither followed nor listed."""
    files: list[StoreFile] = []
    pending = [str(root)]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(entry.path)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                try:
                    stat = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                files.append(
                    StoreFile(
                        path=entry.path,
                        size=stat.st_size,
                        mtime=stat.st_mtime,
                        last_use=max(stat.st_atime, stat.st_mtime),
                    )
                )
    return files


def is_young(store_file: StoreFile, now: float, min_age_secs: float) -> bool:
    return now - store_file.mtime < min_age_secs


def choose_victims(
    files: Sequence[StoreFile], cap_bytes: int, now: float, min_age_secs: float
) -> list[StoreFile]:
    """The least recently used old-enough files whose removal brings the total under the cap."""
    total = sum(store_file.size for store_file in files)
    candidates = sorted(
        (store_file for store_file in files if not is_young(store_file, now, min_age_secs)),
        key=lambda store_file: (store_file.last_use, store_file.path),
    )
    victims: list[StoreFile] = []
    for candidate in candidates:
        if total <= cap_bytes:
            break
        victims.append(candidate)
        total -= candidate.size
    return victims


def remove(victims: Iterable[StoreFile]) -> None:
    for victim in victims:
        Path(victim.path).unlink(missing_ok=True)


def store_main_pid(unit: str) -> int:
    """The unit's MainPID as the user manager reports it; 0 when it is not running."""
    proc = subprocess.run(
        ["systemctl", "--user", "show", "-p", "MainPID", "--value", unit],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemctlError(
            f"systemctl --user show {unit} exited {proc.returncode}: {proc.stderr.strip()}"
        )
    try:
        return int(proc.stdout.strip())
    except ValueError as exc:
        raise SystemctlError(f"systemctl reported no MainPID for {unit}: {proc.stdout!r}") from exc


def signal_store(unit: str, pid: int) -> int | None:
    """SIGHUP the unit's main process *pid* so rclone drops its directory cache; the pid, or None."""
    if pid <= 0:
        _log.info("%s %s is not running; nothing to signal", _LOG_PREFIX, unit)
        return None
    try:
        os.kill(pid, signal.SIGHUP)
    except ProcessLookupError:
        _log.warning("%s %s main pid %d exited before the signal", _LOG_PREFIX, unit, pid)
        return None
    return pid


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must not be negative: {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_cache_store_prune.py",
        description="Remove least-recently-used store objects down to a size cap.",
    )
    parser.add_argument("store", type=Path)
    parser.add_argument("--cap-bytes", type=_non_negative_int, required=True)
    parser.add_argument("--min-age-secs", type=float, default=DEFAULT_MIN_AGE_SECS)
    parser.add_argument("--rclone-unit", default=DEFAULT_RCLONE_UNIT)
    parser.add_argument("--check", action="store_true", help="report what would go; remove nothing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.store.is_dir():
        _log.error("%s store root %s is not a directory", _LOG_PREFIX, args.store)
        print(json.dumps({"store": str(args.store), "error": "store root missing"}))
        return EXIT_FAILED

    now = time.time()
    files = scan(args.store)
    victims = choose_victims(files, args.cap_bytes, now, args.min_age_secs)
    signalled: int | None = None
    error: str | None = None
    if victims and not args.check:
        try:
            pid = store_main_pid(args.rclone_unit)
        except SystemctlError as exc:
            _log.error("%s %s; nothing was removed", _LOG_PREFIX, exc)
            error = str(exc)
            victims = []
        else:
            remove(victims)
            signalled = signal_store(args.rclone_unit, pid)

    total_bytes = sum(store_file.size for store_file in files)
    removed_bytes = sum(victim.size for victim in victims)
    remaining_bytes = total_bytes - removed_bytes
    report: dict[str, object] = {
        "store": str(args.store),
        "files": len(files),
        "total_bytes": total_bytes,
        "cap_bytes": args.cap_bytes,
        "removed": len(victims),
        "removed_bytes": removed_bytes,
        "remaining_bytes": remaining_bytes,
        "skipped_young": sum(1 for f in files if is_young(f, now, args.min_age_secs)),
        "over_cap": remaining_bytes > args.cap_bytes,
        "signalled": signalled,
        "check": args.check,
    }
    if error is not None:
        report["error"] = error
    exit_code = EXIT_OK if error is None else EXIT_FAILED

    print(json.dumps(report))
    _log.info(
        "%s %s %d of %d files (%d bytes); %d bytes remain against a cap of %d%s",
        _LOG_PREFIX,
        "would remove" if args.check else "removed",
        len(victims),
        len(files),
        removed_bytes,
        remaining_bytes,
        args.cap_bytes,
        " (still over cap)" if report["over_cap"] else "",
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
