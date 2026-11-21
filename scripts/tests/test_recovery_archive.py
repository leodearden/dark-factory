"""scripts/recovery_archive.py: carry the pinned FalkorDB recovery dumps through a host migration (task 7153)."""

import hashlib
import os
import shutil
from pathlib import Path

import pytest
from recovery_archive import RECOVERY_ARCHIVES, ArchiveDir, PinnedDump, main

GOOD_SHA = "0" * 64
ARCH_A = "fused-memory/data/arch-a"
ARCH_B = "data/backups/arch-b"
DUMP_CONTENT = {
    f"{ARCH_A}/first.rdb": b"first dump bytes",
    f"{ARCH_A}/second.rdb": b"the second dump, a little longer",
    f"{ARCH_B}/dump.rdb": b"the third dump",
}
SIBLING_CONTENT = {
    f"{ARCH_A}/apply-report.json": b'{"moved": 13}',
    f"{ARCH_A}/FINDINGS.md": b"# findings",
    f"{ARCH_B}/logs/run.log": b"a log line",
}


def _pinned(repo_path: str) -> PinnedDump:
    content = DUMP_CONTENT[repo_path]
    name = Path(repo_path).name
    return PinnedDump(
        name=name,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        pre_state_of=f"the {name} test apply",
    )


ARCHIVES = tuple(
    ArchiveDir(path=archive, dumps=tuple(_pinned(p) for p in DUMP_CONTENT if p.startswith(f"{archive}/")))
    for archive in (ARCH_A, ARCH_B)
)


def _write_archives(root: Path) -> tuple[ArchiveDir, ...]:
    for repo_path, content in {**DUMP_CONTENT, **SIBLING_CONTENT}.items():
        (root / repo_path).parent.mkdir(parents=True, exist_ok=True)
        (root / repo_path).write_bytes(content)
    return ARCHIVES


def _flip_first_byte(path: Path) -> str:
    content = bytearray(path.read_bytes())
    content[0] ^= 0xFF
    path.write_bytes(bytes(content))
    return hashlib.sha256(content).hexdigest()


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


def test_check_passes_and_names_every_intact_dump_with_its_pre_state(tmp_path, capsys):
    archives = _write_archives(tmp_path)

    assert main(["check", str(tmp_path)], archives=archives) == 0

    stderr = capsys.readouterr().err
    for archive in archives:
        for dump in archive.dumps:
            assert f"{archive.path}/{dump.name}" in stderr
            assert dump.pre_state_of in stderr


def test_check_fails_on_a_missing_dump(tmp_path, capsys):
    archives = _write_archives(tmp_path)
    (tmp_path / ARCH_A / "second.rdb").unlink()

    assert main(["check", str(tmp_path)], archives=archives) == 1

    failed = [line for line in capsys.readouterr().err.splitlines() if "FAILED" in line]
    assert any(f"{ARCH_A}/second.rdb" in line and "missing" in line for line in failed)


@pytest.mark.parametrize("change", [b"extra", None])
def test_check_fails_on_a_dump_whose_size_differs(tmp_path, capsys, change):
    archives = _write_archives(tmp_path)
    dump_path = tmp_path / ARCH_B / "dump.rdb"
    pinned = DUMP_CONTENT[f"{ARCH_B}/dump.rdb"]
    altered = pinned + change if change else pinned[:-3]
    dump_path.write_bytes(altered)

    assert main(["check", str(tmp_path)], archives=archives) == 1

    failed = [line for line in capsys.readouterr().err.splitlines() if "FAILED" in line]
    assert any(
        f"{ARCH_B}/dump.rdb" in line and str(len(altered)) in line and str(len(pinned)) in line
        for line in failed
    )


def test_check_fails_on_a_same_size_dump_with_a_flipped_byte(tmp_path, capsys):
    archives = _write_archives(tmp_path)
    actual_sha = _flip_first_byte(tmp_path / ARCH_A / "first.rdb")
    pinned_sha = archives[0].dumps[0].sha256

    assert main(["check", str(tmp_path)], archives=archives) == 1

    failed = [line for line in capsys.readouterr().err.splitlines() if "FAILED" in line]
    assert any(
        f"{ARCH_A}/first.rdb" in line and actual_sha in line and pinned_sha in line
        for line in failed
    )


def test_check_reports_every_problem_not_just_the_first(tmp_path, capsys):
    archives = _write_archives(tmp_path)
    (tmp_path / ARCH_A / "first.rdb").unlink()
    _flip_first_byte(tmp_path / ARCH_B / "dump.rdb")

    assert main(["check", str(tmp_path)], archives=archives) == 1

    failed = [line for line in capsys.readouterr().err.splitlines() if "FAILED" in line]
    assert any(f"{ARCH_A}/first.rdb" in line for line in failed)
    assert any(f"{ARCH_B}/dump.rdb" in line for line in failed)


def test_check_fails_on_a_missing_archive_directory(tmp_path, capsys):
    archives = _write_archives(tmp_path)
    shutil.rmtree(tmp_path / ARCH_B)

    assert main(["check", str(tmp_path)], archives=archives) == 1

    failed = [line for line in capsys.readouterr().err.splitlines() if "FAILED" in line]
    assert any(ARCH_B in line for line in failed)


def test_check_without_a_root_is_a_usage_error():
    with pytest.raises(SystemExit) as raised:
        main(["check"], archives=ARCHIVES)
    assert raised.value.code == 2


OLD_MTIME_NS = 1_700_000_000 * 10**9


def _source_tree(root: Path) -> tuple[ArchiveDir, ...]:
    archives = _write_archives(root)
    for repo_path in (*DUMP_CONTENT, *SIBLING_CONTENT):
        os.utime(root / repo_path, ns=(OLD_MTIME_NS, OLD_MTIME_NS))
    for planted in (f"{ARCH_A}/__pycache__/helper.cpython-312.pyc",
                    "fused-memory/data/falkordb/dump.rdb",
                    "fused-memory/data/unlisted/x.rdb"):
        (root / planted).parent.mkdir(parents=True, exist_ok=True)
        (root / planted).write_bytes(b"not carried")
    return archives


def _snapshot(root: Path) -> dict[Path, tuple[bytes, int]]:
    return {path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def _copy(source: Path, dest: Path, archives: tuple[ArchiveDir, ...]) -> int:
    return main(["copy", str(source), str(dest)], archives=archives)


def test_copy_carries_exactly_the_named_archive_dirs_with_their_siblings(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    archives = _source_tree(src)

    assert _copy(src, dst, archives) == 0

    for repo_path in (*DUMP_CONTENT, *SIBLING_CONTENT):
        assert (dst / repo_path).read_bytes() == (src / repo_path).read_bytes()
        assert (dst / repo_path).stat().st_mtime_ns == OLD_MTIME_NS
    assert not (dst / ARCH_A / "__pycache__").exists()
    assert not (dst / "fused-memory/data/unlisted").exists()
    assert not (dst / "fused-memory/data/falkordb").exists()


def test_copy_over_an_earlier_copy_succeeds(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    archives = _source_tree(src)

    assert _copy(src, dst, archives) == 0
    assert _copy(src, dst, archives) == 0


def test_copy_checks_the_destination_and_names_a_corrupt_dump_there(tmp_path, capsys):
    src, dst = tmp_path / "src", tmp_path / "dst"
    archives = _source_tree(src)
    _flip_first_byte(src / ARCH_A / "first.rdb")

    assert _copy(src, dst, archives) == 1

    failed = [line for line in capsys.readouterr().err.splitlines() if "FAILED" in line]
    assert any(str(dst / ARCH_A / "first.rdb") in line and "sha256" in line for line in failed)


def test_copy_names_a_missing_source_archive_and_still_carries_the_others(tmp_path, capsys):
    src, dst = tmp_path / "src", tmp_path / "dst"
    archives = _source_tree(src)
    shutil.rmtree(src / ARCH_B)

    assert _copy(src, dst, archives) == 1

    failed = [line for line in capsys.readouterr().err.splitlines() if "FAILED" in line]
    assert any(str(src / ARCH_B) in line for line in failed)
    for repo_path in (*DUMP_CONTENT, *SIBLING_CONTENT):
        if repo_path.startswith(f"{ARCH_A}/"):
            assert (dst / repo_path).read_bytes() == (src / repo_path).read_bytes()


def test_copy_refuses_a_destination_that_is_the_source(tmp_path, capsys):
    src = tmp_path / "src"
    archives = _source_tree(src)
    same = tmp_path / "src" / ".." / "src"
    before = _snapshot(tmp_path)

    assert main(["copy", str(src), str(same)], archives=archives) == 1

    failed = [line for line in capsys.readouterr().err.splitlines() if "FAILED" in line]
    assert any(str(src) in line and str(same) in line for line in failed)
    assert _snapshot(tmp_path) == before


def test_copy_refuses_a_destination_inside_a_source_archive(tmp_path, capsys):
    src = tmp_path / "src"
    archives = _source_tree(src)
    inside = src / ARCH_A / "nested-dest"
    before = _snapshot(tmp_path)

    assert _copy(src, inside, archives) == 1

    failed = [line for line in capsys.readouterr().err.splitlines() if "FAILED" in line]
    assert any(str(inside) in line for line in failed)
    assert not inside.exists()
    assert _snapshot(tmp_path) == before
