#!/usr/bin/env python3
"""Carries the pinned FalkorDB recovery dumps and their sibling files through a host
migration, checking each dump's size and sha256 at every hop.

Why they are kept, and the retention rule: docs/falkordb-recovery-dumps.md.
"""

import argparse
import hashlib
import re
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

LIVE_FALKORDB_DIR = PurePosixPath("fused-memory/data/falkordb")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def _require_bare_file_name(name: str) -> None:
    if not name or "/" in name or name in (".", ".."):
        raise ValueError(f"a pinned dump name must be a bare file name, got {name!r}")


def _require_sha256_hex(sha256: str) -> None:
    if not _SHA256_HEX.fullmatch(sha256):
        raise ValueError(f"a pinned dump sha256 must be 64 lowercase hex chars, got {sha256!r}")


def _require_positive_size(size: int) -> None:
    if size <= 0:
        raise ValueError(f"a pinned dump size must be positive, got {size}")


def _require_relative_path(path: PurePosixPath) -> None:
    if path.is_absolute():
        raise ValueError(f"an archive path must be relative to the repo root, got {str(path)!r}")


def _require_no_parent_component(path: PurePosixPath) -> None:
    if ".." in path.parts:
        raise ValueError(f"an archive path must not contain '..', got {str(path)!r}")


def _require_outside_live_falkordb_dir(path: PurePosixPath) -> None:
    if path.parts[: len(LIVE_FALKORDB_DIR.parts)] == LIVE_FALKORDB_DIR.parts:
        raise ValueError(
            "an archive path must not be at or under the live FalkorDB dir "
            f"{str(LIVE_FALKORDB_DIR)!r}, got {str(path)!r}"
        )


@dataclass(frozen=True)
class PinnedDump:
    name: str
    size: int
    sha256: str
    pre_state_of: str

    def __post_init__(self) -> None:
        _require_bare_file_name(self.name)
        _require_sha256_hex(self.sha256)
        _require_positive_size(self.size)


def _require_dumps(path: str, dumps: tuple[PinnedDump, ...]) -> None:
    if not dumps:
        raise ValueError(f"an archive must pin at least one dump, {path!r} pins none")


def _require_unique_dump_names(path: str, dumps: tuple[PinnedDump, ...]) -> None:
    names = [dump.name for dump in dumps]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"archive {path!r} pins a dump name more than once: {', '.join(repeated)}")


@dataclass(frozen=True)
class ArchiveDir:
    path: str
    dumps: tuple[PinnedDump, ...]

    def __post_init__(self) -> None:
        posix_path = PurePosixPath(self.path)
        _require_relative_path(posix_path)
        _require_no_parent_component(posix_path)
        _require_outside_live_falkordb_dir(posix_path)
        _require_dumps(self.path, self.dumps)
        _require_unique_dump_names(self.path, self.dumps)


RECOVERY_ARCHIVES: tuple[ArchiveDir, ...] = (
    ArchiveDir(
        path="fused-memory/data/cgl-eta",
        dumps=(
            PinnedDump(
                name="falkordb-backup-2026-07-11T1002.rdb",
                size=655513026,
                sha256="149929c0fd9b12aa16055ebd60bf710bab21e152733ce7c5246d6df8affd80c8",
                pre_state_of="the 2026-07-11 CGL-eta canary2 apply (task 2273); "
                "the only pre-state of the 13 nodes it moved",
            ),
            PinnedDump(
                name="falkordb-backup-20260712T100121Z.rdb",
                size=594360506,
                sha256="d4cd227e456326fabddf8e97c846d3077e2e315af0a4782abf6a452b8f6b7ffc",
                pre_state_of="the 2026-07-12 CGL-eta cross-graph apply "
                "(task 2273; a pre-6616 cross-graph move)",
            ),
        ),
    ),
    ArchiveDir(
        path="fused-memory/data/cgl-iota",
        dumps=(
            PinnedDump(
                name="falkordb-dump-pre-consolidation-lastsave-1783890198.rdb",
                size=579321680,
                sha256="520e48321415d53eef1adfda5dae59e7ecdb0af3ed57e8686531421dcd57b36e",
                pre_state_of="the 2026-07-14 CGL-iota namespace-family consolidation "
                "(task 2275, esc-2275-2)",
            ),
        ),
    ),
    ArchiveDir(
        path="fused-memory/data/solar-ns-merge",
        dumps=(
            PinnedDump(
                name="falkordb-dump-pre-solar-merge-lastsave-1789559099.rdb",
                size=680226657,
                sha256="6412048223a6e80e0fa0fdd744a5f9e6adb422caef01fef0cd24d86ffb03fb7c",
                pre_state_of="the 2026-09-16 my_solar_challenge -> solar_challenge merge "
                "(solar_challenge task 98, esc-98-1)",
            ),
        ),
    ),
    ArchiveDir(
        path="data/backups/2026-07-02-ops-purge-prune",
        dumps=(
            PinnedDump(
                name="falkordb-dump.rdb",
                size=726146113,
                sha256="46dc407783418ece9c1dca8142db3b6e097b66ced29b34ddabec8a15ecebe627",
                pre_state_of="the 2026-07-02 ops purge/prune; "
                "the pre-state for the 2026-07-02..07-12 merges (pre-4986)",
            ),
        ),
    ),
)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Carry the pinned FalkorDB recovery dumps through a host migration.")
    commands = parser.add_subparsers(dest="command", required=True)
    copy = commands.add_parser(
        "copy", help="copy every archive dir from SOURCE to DEST, then check the dumps under DEST")
    copy.add_argument("source", type=Path, help="root holding the archive dirs at their repo paths")
    copy.add_argument("dest", type=Path, help="root to receive them at the same relative paths")
    check = commands.add_parser("check", help="check every pinned dump under ROOT")
    check.add_argument("root", type=Path, help="root holding the archive dirs at their repo paths")
    return parser.parse_args(argv)


def log(message: str) -> None:
    print(f"[recovery-archive] {message}", file=sys.stderr, flush=True)


def dump_problems(path: Path, dump: PinnedDump) -> list[str]:
    if not path.is_file():
        return [f"{path}: missing"]
    size = path.stat().st_size
    if size != dump.size:
        return [f"{path}: size {size} bytes, pinned {dump.size} bytes"]
    with path.open("rb") as dump_file:
        sha256 = hashlib.file_digest(dump_file, "sha256").hexdigest()
    if sha256 != dump.sha256:
        return [f"{path}: sha256 {sha256}, pinned {dump.sha256}"]
    return []


def check_root(root: Path, archives: Sequence[ArchiveDir]) -> list[str]:
    problems: list[str] = []
    for archive in archives:
        problems += _archive_problems(root, archive)
    return problems


def _archive_problems(root: Path, archive: ArchiveDir) -> list[str]:
    archive_dir = root / archive.path
    if not archive_dir.is_dir():
        return _logged_failures([f"{archive_dir}: archive directory missing"])
    problems: list[str] = []
    for dump in archive.dumps:
        found = dump_problems(archive_dir / dump.name, dump)
        if not found:
            log(f"ok {archive.path}/{dump.name} ({dump.size} bytes) - pre-state of {dump.pre_state_of}")
        problems += _logged_failures(found)
    return problems


def copy_archives(source: Path, dest: Path, archives: Sequence[ArchiveDir]) -> list[str]:
    refusal = _copy_refusal(source, dest, archives)
    if refusal:
        return _logged_failures([refusal])
    problems: list[str] = []
    for archive in archives:
        problems += _logged_failures(_copy_archive(source, dest, archive))
    dest_problems = check_root(dest, archives)
    if dest_problems:
        log("to tell a corrupt source from a bad copy, run: "
            f"python3 {Path(__file__).resolve()} check {source}")
    return problems + dest_problems


def _copy_refusal(source: Path, dest: Path, archives: Sequence[ArchiveDir]) -> str | None:
    resolved_dest = dest.resolve()
    if resolved_dest == source.resolve():
        return f"refusing to copy {source} onto itself ({dest})"
    for archive in archives:
        source_dir = source / archive.path
        if resolved_dest.is_relative_to(source_dir.resolve()):
            return f"refusing to copy into {dest}: it is inside the source archive {source_dir}"
    return None


def _copy_archive(source: Path, dest: Path, archive: ArchiveDir) -> list[str]:
    source_dir, dest_dir = source / archive.path, dest / archive.path
    if not source_dir.is_dir():
        return [f"{source_dir}: source archive directory missing"]
    try:
        shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
    except OSError as exc:
        return [f"{source_dir}: copy to {dest_dir} failed: {exc}"]
    log(f"copied {archive.path}")
    return []


def _logged_failures(problems: list[str]) -> list[str]:
    for problem in problems:
        log(f"FAILED {problem}")
    return problems


def main(argv: Sequence[str] | None = None, *,
         archives: Sequence[ArchiveDir] = RECOVERY_ARCHIVES) -> int:
    args = parse_args(argv)
    if args.command == "copy":
        problems = copy_archives(args.source, args.dest, archives)
    else:
        problems = check_root(args.root, archives)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
