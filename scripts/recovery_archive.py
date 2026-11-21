#!/usr/bin/env python3
"""Carries the pinned FalkorDB recovery dumps and their sibling files through a host
migration, checking each dump's size and sha256 at every hop.

Why they are kept, and the retention rule: docs/falkordb-recovery-dumps.md.
"""

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

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
