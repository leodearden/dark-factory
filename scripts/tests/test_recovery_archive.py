"""scripts/recovery_archive.py: carry the pinned FalkorDB recovery dumps through a host migration (task 7153)."""

import pytest
from recovery_archive import RECOVERY_ARCHIVES, ArchiveDir, PinnedDump

GOOD_SHA = "0" * 64


def _dump(name: str = "dump.rdb", size: int = 10, sha256: str = GOOD_SHA) -> PinnedDump:
    return PinnedDump(name=name, size=size, sha256=sha256, pre_state_of="a test apply")


@pytest.mark.parametrize("name", ["sub/dump.rdb", "..", ""])
def test_a_dump_name_must_be_a_bare_file_name(name):
    with pytest.raises(ValueError, match="bare file name") as raised:
        _dump(name=name)
    assert repr(name) in str(raised.value)


@pytest.mark.parametrize("sha256", ["0" * 63, "0" * 65, "A" * 64, "g" * 64])
def test_a_dump_sha256_must_be_64_lowercase_hex(sha256):
    with pytest.raises(ValueError, match="64 lowercase hex") as raised:
        _dump(sha256=sha256)
    assert sha256 in str(raised.value)


@pytest.mark.parametrize("size", [0, -1])
def test_a_dump_size_must_be_positive(size):
    with pytest.raises(ValueError, match="positive") as raised:
        _dump(size=size)
    assert str(size) in str(raised.value)


@pytest.mark.parametrize(
    ("path", "invariant"),
    [
        ("/home/leo/fused-memory/data/cgl-eta", "relative"),
        ("fused-memory/data/../data/cgl-eta", r"\.\."),
        ("fused-memory/data/falkordb", "live FalkorDB"),
        ("fused-memory/data/falkordb/old", "live FalkorDB"),
    ],
)
def test_an_archive_path_is_relative_and_outside_the_live_falkordb_dir(path, invariant):
    with pytest.raises(ValueError, match=invariant) as raised:
        ArchiveDir(path=path, dumps=(_dump(),))
    assert path in str(raised.value)


@pytest.mark.parametrize("path", ["fused-memory/data/falkordb-old", "fused-memory/data/cgl-eta"])
def test_a_sibling_of_the_live_falkordb_dir_is_an_archive_path(path):
    assert ArchiveDir(path=path, dumps=(_dump(),)).path == path


def test_an_archive_carries_at_least_one_dump():
    with pytest.raises(ValueError, match="at least one") as raised:
        ArchiveDir(path="data/backups/empty", dumps=())
    assert "data/backups/empty" in str(raised.value)


def test_an_archive_carries_each_dump_name_once():
    with pytest.raises(ValueError, match="more than once") as raised:
        ArchiveDir(path="data/backups/twice", dumps=(_dump(), _dump(size=11)))
    assert "dump.rdb" in str(raised.value)


def test_the_carried_set_is_the_q2_ruling_set():
    """Removing an entry is Leo's decision (docs/falkordb-recovery-dumps.md), so the set is pinned twice."""
    assert {f"{a.path}/{d.name}" for a in RECOVERY_ARCHIVES for d in a.dumps} == {
        "fused-memory/data/cgl-eta/falkordb-backup-2026-07-11T1002.rdb",
        "fused-memory/data/cgl-eta/falkordb-backup-20260712T100121Z.rdb",
        "fused-memory/data/cgl-iota/falkordb-dump-pre-consolidation-lastsave-1783890198.rdb",
        "fused-memory/data/solar-ns-merge/falkordb-dump-pre-solar-merge-lastsave-1789559099.rdb",
        "data/backups/2026-07-02-ops-purge-prune/falkordb-dump.rdb",
    }
    paths = [a.path for a in RECOVERY_ARCHIVES]
    assert len(paths) == len(set(paths))
    assert all(d.pre_state_of.strip() for a in RECOVERY_ARCHIVES for d in a.dumps)
