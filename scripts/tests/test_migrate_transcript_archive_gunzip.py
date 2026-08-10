"""Tests for scripts/migrate_transcript_archive_gunzip.py — the one-off sweep
that decompresses the agent-transcript archive in place, leaving one plain,
greppable `.jsonl` corpus (task 3618, leaf α of
plans/transcript-preservation-seam-prd.md).

step-1: the CORROBORATE-BEFORE-DESTROY contract (INV-3). `gunzip_one(gz)`
decompresses to the sibling `.jsonl`, re-opens and reads the result back,
mirrors the `.gz` mtime, and only THEN unlinks the source. The mtime mirror is
a first-class assertion, not an incidental: `gc_agent_transcripts.scan_task_dirs`
derives each task dir's retention age from its NEWEST descendant mtime, so a
migration that stamped `now` on all 4,554 files would silently reset the whole
90-day retention window in a single pass.
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import os
import subprocess
import zlib
from pathlib import Path

import migrate_transcript_archive_gunzip as mig
import pytest

LOG_PREFIX = "migrate_transcript_archive_gunzip:"

DAY = 86_400
NOW = 1_000_000_000.0

# A distinctive PAST mtime, far enough from "now" that a migration which
# stamped the current clock instead of mirroring the source would be visibly
# wrong rather than merely imprecise.
ARCHIVED_MTIME = NOW - 30 * DAY


def _payload(n_records: int = 20) -> bytes:
    """Serialize a multi-record JSONL body — the exact bytes we compress and
    then expect back byte-for-byte after the round-trip."""
    return "".join(
        json.dumps({"type": "user", "seq": i, "pad": "x" * 50}) + "\n"
        for i in range(n_records)
    ).encode("utf-8")


def _write_gz(path: Path, payload: bytes, mtime: float = ARCHIVED_MTIME) -> Path:
    """Write *payload* as a valid gz at *path*, stamped with *mtime*.

    Mirrors the real writer (shared/src/shared/transcript_archive.py), which
    `os.utime`s each archived copy with the SOURCE transcript's mtime — which
    is why mirroring the `.gz` mtime onto the `.jsonl` is the correct thing for
    the migration to do.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        handle.write(payload)
    os.utime(path, (mtime, mtime))
    return path


# ---------------------------------------------------------------------------
# step-1: gunzip_one — round-trip exactness, mtime mirror, source removal
# ---------------------------------------------------------------------------

def test_gunzip_one_writes_byte_identical_plain_sibling(tmp_path):
    """(a) The produced `.jsonl` is BYTE-IDENTICAL to the pre-compression
    payload. DEFLATE is lossless by construction, so this is exact equality —
    not a tolerance."""
    payload = _payload()
    gz = _write_gz(tmp_path / "3618" / "enc" / "sess-a.jsonl.gz", payload)

    mig.gunzip_one(gz)

    plain = tmp_path / "3618" / "enc" / "sess-a.jsonl"
    assert plain.exists(), "the sibling .jsonl was not created"
    assert plain.read_bytes() == payload


def test_gunzip_one_mirrors_the_gz_mtime_onto_the_plain_file(tmp_path):
    """(b) The retention-age property `gc_agent_transcripts.scan_task_dirs`
    keys on survives the migration. Stamping `now` here would silently reset
    the entire 90-day retention window across the archive."""
    gz = _write_gz(tmp_path / "3618" / "enc" / "sess-b.jsonl.gz", _payload())
    gz_mtime = gz.stat().st_mtime

    mig.gunzip_one(gz)

    plain = tmp_path / "3618" / "enc" / "sess-b.jsonl"
    assert int(plain.stat().st_mtime) == int(gz_mtime)
    assert int(plain.stat().st_mtime) == int(ARCHIVED_MTIME)


def test_gunzip_one_unlinks_the_source_after_corroborating(tmp_path):
    """(c) The source `.gz` is gone once the plain twin is written, read back
    and stamped — the destroy half of corroborate-before-destroy."""
    gz = _write_gz(tmp_path / "3618" / "enc" / "sess-c.jsonl.gz", _payload())

    mig.gunzip_one(gz)

    assert not gz.exists(), "the source .gz should be unlinked after a good migration"
    assert (tmp_path / "3618" / "enc" / "sess-c.jsonl").exists()


def test_gunzip_one_migrates_the_nested_subagent_layout(tmp_path):
    """(d) The deeper `<sid>/subagents/agent-<hex>.jsonl.gz` layout migrates
    identically — the archive nests subagent transcripts one level below the
    main session file, and both shapes must be swept."""
    payload = _payload(5)
    gz = _write_gz(
        tmp_path / "3618" / "enc" / "sess-d" / "subagents" / "agent-1a2b.jsonl.gz",
        payload,
    )

    mig.gunzip_one(gz)

    plain = tmp_path / "3618" / "enc" / "sess-d" / "subagents" / "agent-1a2b.jsonl"
    assert plain.read_bytes() == payload
    assert int(plain.stat().st_mtime) == int(ARCHIVED_MTIME)
    assert not gz.exists()


# ---------------------------------------------------------------------------
# step-3: migrate_archive — idempotency and crash-resume (INV-7)
#
# The sweep must be safe to re-run. It walks ~4,554 files and an operator can
# kill it part-way, so every intermediate state it can be interrupted in has to
# be a state a later run recovers from correctly — including the nastiest one,
# where a run died between writing a twin and unlinking its source, leaving a
# HALF-WRITTEN twin next to a perfectly good .gz.
# ---------------------------------------------------------------------------

def _mtimes(root: Path) -> dict[Path, float]:
    """Snapshot every file's mtime under *root* — for proving a re-run is inert."""
    return {p: p.stat().st_mtime for p in sorted(root.rglob("*")) if p.is_file()}


def test_existing_good_twin_is_skipped_and_its_gz_unlinked(tmp_path):
    """(a) The resume case: a run killed between writing the twin and unlinking
    the source. The twin reads back cleanly, so it is trusted — classified
    `skipped`, NOT re-decompressed — and the redundant .gz is unlinked so the
    archive still converges to zero .gz."""
    payload = _payload()
    gz = _write_gz(tmp_path / "3618" / "enc" / "sess-a.jsonl.gz", payload)
    twin = tmp_path / "3618" / "enc" / "sess-a.jsonl"
    twin.write_bytes(payload)
    os.utime(twin, (ARCHIVED_MTIME, ARCHIVED_MTIME))
    before = twin.stat().st_mtime_ns

    summary = mig.migrate_archive(tmp_path, apply=True)

    assert summary["skipped"] == 1
    assert summary["migrated"] == 0
    assert summary["failed"] == 0
    # Not re-decompressed: an untouched mtime proves the good twin was trusted
    # rather than silently rewritten.
    assert twin.stat().st_mtime_ns == before
    assert twin.read_bytes() == payload
    assert not gz.exists()


def test_existing_unreadable_twin_is_re_gunzipped_from_the_authoritative_gz(tmp_path):
    """(b) A half-written twin must NEVER be trusted on mere existence. The .gz
    is the authoritative copy until corroborated, so an undecodable/partial twin
    is overwritten from it and only then is the source unlinked."""
    payload = _payload()
    gz = _write_gz(tmp_path / "3618" / "enc" / "sess-b.jsonl.gz", payload)
    twin = tmp_path / "3618" / "enc" / "sess-b.jsonl"
    # A partial prefix of the real payload, ending in a raw 0xFF — the shape a
    # killed write leaves behind: plausible-looking, but neither complete nor
    # decodable.
    twin.write_bytes(payload[: len(payload) // 2] + b"\xff")

    summary = mig.migrate_archive(tmp_path, apply=True)

    assert summary["migrated"] == 1
    assert summary["skipped"] == 0
    assert summary["failed"] == 0
    assert twin.read_bytes() == payload, "the bad twin was not rebuilt from the .gz"
    assert int(twin.stat().st_mtime) == int(ARCHIVED_MTIME)
    assert not gz.exists()


def test_second_run_over_a_migrated_tree_is_a_clean_no_op(tmp_path):
    """(c) Idempotency proper: re-running over a fully migrated tree migrates
    nothing, fails nothing, and does not touch a single file's mtime."""
    _write_gz(tmp_path / "3618" / "enc" / "sess-c.jsonl.gz", _payload())
    _write_gz(
        tmp_path / "3618" / "enc" / "sess-c" / "subagents" / "agent-1.jsonl.gz",
        _payload(3),
    )

    first = mig.migrate_archive(tmp_path, apply=True)
    assert first["migrated"] == 2

    before = _mtimes(tmp_path)
    second = mig.migrate_archive(tmp_path, apply=True)

    assert second["scanned"] == 0
    assert second["migrated"] == 0
    assert second["skipped"] == 0
    assert second["failed"] == 0
    assert _mtimes(tmp_path) == before, "a no-op re-run must not touch any file"


def test_plain_jsonl_without_a_gz_is_left_untouched(tmp_path):
    """(d) An already-plain transcript with no .gz at all is not the sweep's
    business — it is neither scanned, rewritten, nor re-stamped."""
    plain = tmp_path / "3618" / "enc" / "sess-d.jsonl"
    plain.parent.mkdir(parents=True)
    plain.write_bytes(_payload())
    os.utime(plain, (ARCHIVED_MTIME, ARCHIVED_MTIME))
    before = _mtimes(tmp_path)

    summary = mig.migrate_archive(tmp_path, apply=True)

    assert summary["scanned"] == 0
    assert summary["migrated"] == 0
    assert summary["skipped"] == 0
    assert _mtimes(tmp_path) == before


def test_summary_carries_distinct_counters_not_one_conflated_total(tmp_path):
    """The report separates CLASSIFICATION from ACTION (mirroring
    gc_agent_transcripts.build_gc_report): a run that skipped 1 and migrated 1
    must be distinguishable from one that migrated 2."""
    payload = _payload()
    _write_gz(tmp_path / "3618" / "enc" / "fresh.jsonl.gz", payload)
    _write_gz(tmp_path / "3618" / "enc" / "resumed.jsonl.gz", payload)
    twin = tmp_path / "3618" / "enc" / "resumed.jsonl"
    twin.write_bytes(payload)

    summary = mig.migrate_archive(tmp_path, apply=True)

    assert summary["scanned"] == 2
    assert summary["migrated"] == 1
    assert summary["skipped"] == 1
    assert summary["failed"] == 0
    assert summary["failed_paths"] == []


def test_absent_root_is_a_clean_empty_no_op(tmp_path):
    """An absent root is an empty sweep, not a crash."""
    summary = mig.migrate_archive(tmp_path / "does-not-exist", apply=True)

    assert summary["scanned"] == 0
    assert summary["migrated"] == 0
    assert summary["failed"] == 0


def test_non_directory_root_is_a_clean_empty_no_op(tmp_path):
    """A root that exists but is a FILE is an empty sweep, not a crash."""
    root = tmp_path / "not-a-dir"
    root.write_text("x")

    summary = mig.migrate_archive(root, apply=True)

    assert summary["scanned"] == 0
    assert summary["migrated"] == 0
    assert summary["failed"] == 0


# ---------------------------------------------------------------------------
# step-5: LOUD failure, source RETAINED (INV-3 / INV-4)
#
# The three corrupt-gzip builders below are RELOCATED from
# scripts/tests/test_legibility_inventory.py:101-136 (and its deliberate
# near-copy in test_legibility_digest.py), where step-16 deletes them. After
# this task the migration script is the ONLY component that still reads real
# gzip streams, so this module becomes the one place these shapes stay
# reachable — a move, not a net loss of coverage. That is also precisely what
# licenses the reader-side UNREADABLE_FILE_ERRORS tuple to narrow.
# ---------------------------------------------------------------------------

def _write_bad_magic_gz(path: Path) -> Path:
    """Write a file that is not gzip at all — decompression fails immediately
    with ``gzip.BadGzipFile`` (already an ``OSError``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this is not gzip at all\n")
    return path


def _write_truncated_gz(path: Path) -> Path:
    """Write the first half of a valid gz stream — the interrupted-write shape.

    Decompression reaches the end of the bytes before the end-of-stream marker
    and raises ``EOFError``, which is NOT an ``OSError``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = gzip.compress(_payload(200))
    path.write_bytes(blob[: len(blob) // 2])
    return path


def _write_corrupt_body_gz(path: Path) -> Path:
    """Write a gz whose DEFLATE body is damaged, so decompression raises
    ``zlib.error`` — a THIRD distinct shape, and also not an ``OSError``.

    Where a flipped byte lands decides which failure gzip reports: many flips
    still decode and only trip the trailing checksum (``gzip.BadGzipFile``,
    already an ``OSError``), a few truncate the bit stream (``EOFError``), and
    only a flip that makes the DEFLATE stream itself unparseable raises
    ``zlib.error``. So this probes for the first flip position that produces
    that shape rather than assuming any particular byte does. The probe runs
    against STDLIB ``gzip`` — never the code under test — so the helper stays
    valid regardless of how the migration normalizes its exceptions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = gzip.compress(_payload(200))
    for index in range(10, len(blob)):
        candidate = bytearray(blob)
        candidate[index] ^= 0xFF
        try:
            gzip.GzipFile(fileobj=io.BytesIO(bytes(candidate))).read()
        except zlib.error:
            path.write_bytes(bytes(candidate))
            return path
        except Exception:
            continue  # a different corruption shape — keep probing
    raise AssertionError("no single-byte flip produced a zlib.error body")


CORRUPTION_SHAPES = [
    pytest.param(_write_bad_magic_gz, id="bad-magic"),
    pytest.param(_write_truncated_gz, id="truncated"),
    pytest.param(_write_corrupt_body_gz, id="corrupt-body"),
]


@pytest.mark.parametrize("build_corrupt", CORRUPTION_SHAPES)
def test_uncorroborated_source_is_retained_not_destroyed(tmp_path, build_corrupt):
    """(a) The load-bearing half of INV-3: a file we could not read back is
    NEVER unlinked. Losing an unreadable archive entry is still data loss — the
    bytes may be partially recoverable by hand, and they are certainly not
    recoverable once deleted."""
    gz = build_corrupt(tmp_path / "3618" / "enc" / "broken.jsonl.gz")

    mig.migrate_archive(tmp_path, apply=True)

    assert gz.exists(), "an un-corroborated source .gz must survive the sweep"


@pytest.mark.parametrize("build_corrupt", CORRUPTION_SHAPES)
def test_failure_is_counted_and_named(tmp_path, build_corrupt):
    """(b) The machine-readable signal: counted in `failed` AND named in
    `failed_paths`, so an operator can act on the specific files."""
    gz = build_corrupt(tmp_path / "3618" / "enc" / "broken.jsonl.gz")

    summary = mig.migrate_archive(tmp_path, apply=True)

    assert summary["failed"] == 1
    assert summary["migrated"] == 0
    assert str(gz) in summary["failed_paths"]


@pytest.mark.parametrize("build_corrupt", CORRUPTION_SHAPES)
def test_failure_emits_a_structured_loud_warning(tmp_path, caplog, build_corrupt):
    """(c) INV-4 loud-over-silent: one greppable WARNING naming the path, so a
    failure is discoverable in the log and not merely a count."""
    caplog.set_level(logging.WARNING, logger="migrate_transcript_archive_gunzip")
    gz = build_corrupt(tmp_path / "3618" / "enc" / "broken.jsonl.gz")

    mig.migrate_archive(tmp_path, apply=True)

    warn_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        LOG_PREFIX in m and str(gz) in m for m in warn_msgs
    ), f"missing LOUD WARNING for the unreadable source; got {warn_msgs}"


@pytest.mark.parametrize("build_corrupt", CORRUPTION_SHAPES)
def test_one_bad_file_does_not_cost_its_healthy_siblings(tmp_path, build_corrupt):
    """(d) Best-effort: the sweep walks ~4,554 files, so a single unreadable
    entry must never abort the run and strand the rest."""
    payload = _payload()
    build_corrupt(tmp_path / "3618" / "enc" / "broken.jsonl.gz")
    before = _write_gz(tmp_path / "3618" / "enc" / "aaa-healthy.jsonl.gz", payload)
    after = _write_gz(tmp_path / "3618" / "enc" / "zzz-healthy.jsonl.gz", payload)

    summary = mig.migrate_archive(tmp_path, apply=True)

    # Healthy siblings on BOTH sides of the bad file in sorted-walk order, so
    # this fails whether the sweep aborts or merely skips the remainder.
    assert summary["migrated"] == 2
    assert summary["failed"] == 1
    for gz in (before, after):
        assert not gz.exists()
        assert mig.plain_sibling(gz).read_bytes() == payload


def test_partial_twin_is_removed_so_a_later_run_cannot_trust_it(tmp_path):
    """A failure must not leave debris a RESUME would mistake for a good twin.

    The corrupt-body shape fails part-way through decompression, so bytes are
    already on disk when it raises. Left behind, a later run's existence check
    would find a plausible twin — the exact half-written-file trap step-3(b)
    guards, reached here from the failure path instead."""
    gz = _write_corrupt_body_gz(tmp_path / "3618" / "enc" / "broken.jsonl.gz")

    mig.migrate_archive(tmp_path, apply=True)

    assert gz.exists()
    assert not mig.plain_sibling(gz).exists(), (
        "a partially-written .jsonl must be cleaned up, or a later resume run "
        "would trust it and unlink the authoritative .gz"
    )


def test_unwritable_destination_directory_retains_the_source(tmp_path):
    """A fourth, non-corruption failure shape: the source is perfectly good but
    the destination cannot be written (read-only dir / EACCES). The .gz must
    still survive and the failure must still be counted.

    PermissionError stands in for the whole OSError class here for the same
    reason gc_agent_transcripts' scan tests use it: it is a failure the guard
    genuinely has to survive rather than one swallowed earlier by an existence
    check."""
    enc = tmp_path / "3618" / "enc"
    gz = _write_gz(enc / "sess-a.jsonl.gz", _payload())
    enc.chmod(0o500)  # r-x: entries readable, but nothing new can be created
    try:
        summary = mig.migrate_archive(tmp_path, apply=True)

        assert summary["failed"] == 1
        assert summary["migrated"] == 0
        assert str(gz) in summary["failed_paths"]
        assert gz.exists()
    finally:
        enc.chmod(0o700)  # restore so tmp_path teardown can clean up


# ---------------------------------------------------------------------------
# step-7: the CLI contract, driven as a real subprocess
#
# Invoked via subprocess (the idiom at test_gc_agent_transcripts.py:417-432) so
# the stdout/stderr split is proven for real: stdout stays PURE JSON while the
# LOUD logs go to stderr, which is what lets an operator pipe the report into
# jq while still seeing failures.
#
# The headline contract is that DRY-RUN IS THE DEFAULT. This script deletes
# thousands of source files, so it follows the MUTATOR house convention
# (repair_wiped_metadata_files.py:924 and ~18 fused-memory migrations) rather
# than gc_agent_transcripts' --check sweep convention: mutation is opt-in.
# ---------------------------------------------------------------------------

SCRIPT = Path(__file__).parent.parent / "migrate_transcript_archive_gunzip.py"


def _run_cli(*args):
    """Drive the migration CLI as a real subprocess (inherits the parent env).

    stdout carries the machine-readable JSON summary; stderr carries the LOUD
    human log lines (basicConfig logs to stderr), so ``json.loads(stdout)`` sees
    pure JSON.
    """
    return subprocess.run(
        ["python3", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_cli_dry_run_is_the_default_and_mutates_nothing(tmp_path):
    """(a) THE headline safety property: with no --apply, a run over a real
    archive touches nothing at all, yet still reports what it WOULD do."""
    payload = _payload()
    gz = _write_gz(tmp_path / "3618" / "enc" / "sess-a.jsonl.gz", payload)
    before = _mtimes(tmp_path)

    result = _run_cli("--root", str(tmp_path))

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert gz.exists(), "the default run must NOT delete a source .gz"
    assert not mig.plain_sibling(gz).exists(), "the default run must NOT write"
    assert _mtimes(tmp_path) == before
    # ... and still projects the work accurately.
    summary = json.loads(result.stdout)
    assert summary["apply"] is False
    assert summary["scanned"] == 1
    assert summary["migrated"] == 1
    # LOUD: stderr says plainly that nothing was changed.
    assert LOG_PREFIX in result.stderr


def test_cli_apply_performs_the_migration(tmp_path):
    """(b) --apply opts into the mutation."""
    payload = _payload()
    gz = _write_gz(tmp_path / "3618" / "enc" / "sess-b.jsonl.gz", payload)

    result = _run_cli("--root", str(tmp_path), "--apply")

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert not gz.exists()
    assert mig.plain_sibling(gz).read_bytes() == payload
    summary = json.loads(result.stdout)
    assert summary["apply"] is True
    assert summary["migrated"] == 1


def test_cli_root_defaults_to_the_archive_root(tmp_path):
    """(c) --root overrides the default; the default is default_archive_root()
    — asserted against the parser rather than by sweeping the live archive."""
    parser = mig.build_parser()

    assert parser.parse_args([]).root == str(mig.default_archive_root())
    assert parser.parse_args(["--root", str(tmp_path)]).root == str(tmp_path)
    # Dry-run is the parser-level default, not just a runtime convention.
    assert parser.parse_args([]).apply is False
    assert parser.parse_args(["--apply"]).apply is True


def test_cli_stdout_is_pure_json(tmp_path):
    """(d) stdout parses as a SINGLE JSON object carrying the counters — the
    LOUD log lines must not contaminate it."""
    _write_gz(tmp_path / "3618" / "enc" / "sess-c.jsonl.gz", _payload())

    result = _run_cli("--root", str(tmp_path), "--apply")

    summary = json.loads(result.stdout)  # must not raise
    assert set(summary) >= {
        "root", "apply", "scanned", "migrated", "skipped", "failed", "failed_paths",
    }
    assert LOG_PREFIX not in result.stdout, "log lines leaked into the JSON stream"


def test_cli_exits_zero_when_nothing_failed(tmp_path):
    """(e) exit 0 on a clean run."""
    _write_gz(tmp_path / "3618" / "enc" / "sess-d.jsonl.gz", _payload())

    result = _run_cli("--root", str(tmp_path), "--apply")

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert json.loads(result.stdout)["failed"] == 0


def test_cli_exits_non_zero_when_any_file_failed(tmp_path):
    """(e) A non-zero failure count must be an EXIT CODE, not a warning buried
    in a log — this is a one-off migration whose failures an operator has to
    act on, which is why it diverges from gc_agent_transcripts' always-exit-0
    sweep posture.

    Pinned to the EXACT code, not merely non-zero: OPERATIONS.md §13 documents
    three distinct values an operator triages on, so a `!= 0` assertion would
    stay green while EXIT_FAILURES and EXIT_CONFLICTS swapped underneath the
    runbook."""
    _write_truncated_gz(tmp_path / "3618" / "enc" / "broken.jsonl.gz")
    _write_gz(tmp_path / "3618" / "enc" / "healthy.jsonl.gz", _payload())

    result = _run_cli("--root", str(tmp_path), "--apply")

    assert result.returncode == mig.EXIT_FAILURES, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    summary = json.loads(result.stdout)
    assert summary["failed"] == 1
    assert summary["migrated"] == 1  # the healthy sibling still migrated


def test_cli_absent_root_is_a_clean_no_op(tmp_path):
    """An absent root exits 0 with an empty summary rather than crashing."""
    result = _run_cli("--root", str(tmp_path / "does-not-exist"))

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert json.loads(result.stdout)["scanned"] == 0


# ---------------------------------------------------------------------------
# step-9: the dry run must be REAL validation, not a file count
#
# The dry run is the ONLY full-scale rehearsal an operator gets before being
# asked to run --apply against ~4,554 live files. If it merely counts .gz
# entries without reading them, "projected failed == 0" is structurally
# guaranteed and proves nothing about whether the archive is actually
# migratable — it would pass just as happily over a tree of corrupt files.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("build_corrupt", CORRUPTION_SHAPES)
def test_dry_run_detects_unreadable_sources(tmp_path, build_corrupt):
    """A dry run reads each source through, so real corruption is reported
    BEFORE a human is asked to pull the trigger — while still changing
    nothing."""
    gz = build_corrupt(tmp_path / "3618" / "enc" / "broken.jsonl.gz")
    _write_gz(tmp_path / "3618" / "enc" / "healthy.jsonl.gz", _payload())
    before = _mtimes(tmp_path)

    summary = mig.migrate_archive(tmp_path, apply=False)

    assert summary["scanned"] == 2
    assert summary["failed"] == 1
    assert str(gz) in summary["failed_paths"]
    assert summary["migrated"] == 1  # only the healthy one is projected
    # Still a pure projection: not one byte changed.
    assert _mtimes(tmp_path) == before


def test_dry_run_over_a_healthy_archive_projects_zero_failures(tmp_path):
    """The converse: a genuinely healthy archive projects failed == 0, so the
    gate distinguishes 'validated' from 'never looked'."""
    _write_gz(tmp_path / "3618" / "enc" / "a.jsonl.gz", _payload())
    _write_gz(tmp_path / "3618" / "enc" / "b" / "subagents" / "c.jsonl.gz", _payload(3))

    summary = mig.migrate_archive(tmp_path, apply=False)

    assert summary["scanned"] == 2
    assert summary["migrated"] == 2
    assert summary["failed"] == 0


# ---------------------------------------------------------------------------
# step-19: the RESUMED-SESSION conflict — never overwrite a twin that carries
# MORE data than the .gz
#
# Reachable on the DOCUMENTED operator path. OPERATIONS.md §13 instructs
# "Re-run step 3 after the redeploy to sweep" the transition-window residue,
# and any task in flight ACROSS that redeploy has BOTH forms on disk: the
# pre-redeploy process archived `<sid>.jsonl.gz`, then the post-redeploy
# producer hook wrote a plain `<sid>.jsonl` for the resumed, longer session.
#
# The twin fails `_twin_is_trustworthy` (its size does not match the .gz ISIZE)
# and so falls through to `gunzip_one`, which would overwrite the NEWER, LONGER
# transcript with stale .gz content, roll its mtime backwards (corrupting the
# retention age `gc_agent_transcripts.scan_task_dirs` derives from the newest
# descendant mtime), unlink the .gz, and report `migrated: 1` — silently.
#
# SIZE, NOT MTIME, IS THE DISCRIMINATOR. Both shapes that reach this point are
# NEWER than the archived .gz: a killed write stamped `now` because its
# `os.utime` mirror had not run yet, and a resumed session is genuinely newer.
# What separates them is the DIRECTION of size — a killed write leaves a strict
# PREFIX (shorter), a resumed append-only transcript leaves a strict SUPERSET
# (longer). Tests (b) and (c) below pin that distinction so a mtime-triggered
# rule cannot pass.
# ---------------------------------------------------------------------------

def _write_resumed_session_conflict(root: Path) -> tuple[Path, Path, bytes]:
    """Build the shape a session resumed across a fleet redeploy leaves behind.

    A SHORT archived `.gz` stamped `ARCHIVED_MTIME`, next to a plain twin whose
    bytes are a strict SUPERSET of it (the same records plus more — what an
    append-only resumed transcript looks like) stamped a NEWER mtime.

    Returns ``(gz, twin, twin_bytes)``.
    """
    archived = _payload(5)
    resumed = _payload(20)
    assert resumed.startswith(archived), "fixture must be a strict byte superset"

    gz = _write_gz(root / "3618" / "enc" / "sess.jsonl.gz", archived)
    twin = root / "3618" / "enc" / "sess.jsonl"
    twin.write_bytes(resumed)
    os.utime(twin, (ARCHIVED_MTIME + DAY, ARCHIVED_MTIME + DAY))
    return gz, twin, resumed


def test_resumed_session_twin_is_never_overwritten_by_the_stale_gz(tmp_path, caplog):
    """(a) THE REGRESSION. A twin carrying MORE data than the `.gz` is the
    authoritative copy; overwriting it from the archive is unrecoverable data
    loss. Both files are left exactly as they are and the operator is TOLD."""
    caplog.set_level(logging.WARNING, logger="migrate_transcript_archive_gunzip")
    gz, twin, resumed = _write_resumed_session_conflict(tmp_path)
    before_mtime_ns = twin.stat().st_mtime_ns

    summary = mig.migrate_archive(tmp_path, apply=True)

    assert twin.read_bytes() == resumed, "the resumed transcript was overwritten"
    # No mtime rollback: gc_agent_transcripts derives the retention age from the
    # newest descendant mtime, so stamping the .gz's older mtime here would age
    # a live task dir out early.
    assert twin.stat().st_mtime_ns == before_mtime_ns
    # BOTH copies retained: a size comparison proves the twin is longer, never
    # that it CONTAINS the .gz content, so the .gz is kept for adjudication.
    assert gz.exists(), "the .gz must be retained for the operator to adjudicate"
    assert summary["migrated"] == 0
    assert summary["skipped"] == 0
    assert summary["failed"] == 0
    assert summary["conflicts"] == 1
    assert str(gz) in summary["conflict_paths"]
    # LOUD: told, not left to be noticed.
    warn_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        LOG_PREFIX in m and str(gz) in m for m in warn_msgs
    ), f"missing LOUD WARNING for the conflicting twin; got {warn_msgs}"


def test_half_written_twin_is_still_rebuilt_even_though_it_is_newer(tmp_path):
    """(b) NON-REGRESSION, and the reason SIZE is the discriminator. The
    half-written twin at the committed contract above is ALSO newer than the
    `.gz` (its killed write stamped `now`; the mtime mirror never ran), so a
    "newer => conflict" rule would misclassify it and strand a truncated
    transcript. It is SHORTER, so it is still rebuilt from the authoritative
    source."""
    payload = _payload()
    gz = _write_gz(tmp_path / "3618" / "enc" / "sess-b.jsonl.gz", payload)
    twin = tmp_path / "3618" / "enc" / "sess-b.jsonl"
    twin.write_bytes(payload[: len(payload) // 2] + b"\xff")

    # The premise this test exists to pin: newer, but strictly SHORTER.
    assert twin.stat().st_mtime > gz.stat().st_mtime
    assert twin.stat().st_size < len(payload)

    summary = mig.migrate_archive(tmp_path, apply=True)

    assert summary["migrated"] == 1
    assert summary["conflicts"] == 0
    assert summary["conflict_paths"] == []
    assert twin.read_bytes() == payload, "the bad twin was not rebuilt from the .gz"
    assert int(twin.stat().st_mtime) == int(ARCHIVED_MTIME)
    assert not gz.exists()


def test_complete_but_unstamped_twin_is_still_skipped(tmp_path):
    """(c) NON-REGRESSION, the crash-resume window. A twin written completely
    but killed BEFORE its `os.utime` mirror has an mtime of `now` — newer than
    the `.gz`, yet exactly right. Corroboration is checked FIRST, so a newer
    mtime alone never blocks the resume path."""
    payload = _payload()
    gz = _write_gz(tmp_path / "3618" / "enc" / "sess-c.jsonl.gz", payload)
    twin = tmp_path / "3618" / "enc" / "sess-c.jsonl"
    twin.write_bytes(payload)  # no os.utime — mtime is `now`

    assert twin.stat().st_mtime > gz.stat().st_mtime
    assert twin.stat().st_size == len(payload)

    summary = mig.migrate_archive(tmp_path, apply=True)

    assert summary["skipped"] == 1
    assert summary["conflicts"] == 0
    assert summary["migrated"] == 0
    assert twin.read_bytes() == payload
    assert not gz.exists(), "a corroborated twin still converges the archive to zero .gz"


def test_dry_run_projects_the_conflict_without_touching_anything(tmp_path):
    """(d) The adjudication list is visible BEFORE `--apply`. Classification
    happens on the projection path too, so the operator sees the conflicts in
    the dry run rather than discovering them mid-sweep."""
    gz, twin, resumed = _write_resumed_session_conflict(tmp_path)
    before = _mtimes(tmp_path)

    summary = mig.migrate_archive(tmp_path, apply=False)

    assert summary["scanned"] == 1
    assert summary["conflicts"] == 1
    assert str(gz) in summary["conflict_paths"]
    assert summary["migrated"] == 0
    # A pure projection: not one byte, and not one mtime, changed.
    assert twin.read_bytes() == resumed
    assert gz.exists()
    assert _mtimes(tmp_path) == before


def test_cli_exits_non_zero_and_reports_conflicts(tmp_path):
    """(e) A conflict leaves a `.gz` on disk, which contradicts the runbook's
    `expect 0` confirmation — so it must carry an EXIT CODE, not just a counter
    the operator might not read. Distinct from the failure code: "needs
    adjudication" is not "is damaged", and the runbook triages the two
    differently, so this pins EXIT_CONFLICTS exactly rather than `!= 0`."""
    gz, twin, resumed = _write_resumed_session_conflict(tmp_path)

    result = _run_cli("--root", str(tmp_path), "--apply")

    assert result.returncode == mig.EXIT_CONFLICTS, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    summary = json.loads(result.stdout)
    assert summary["conflicts"] == 1, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert str(gz) in summary["conflict_paths"], (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert summary["failed"] == 0
    assert gz.exists()
    assert twin.read_bytes() == resumed


def test_cli_failures_outrank_conflicts_in_the_exit_code(tmp_path):
    """(e) When BOTH shapes are present in one sweep, EXIT_FAILURES wins.

    OPERATIONS.md §13 tells the operator to triage on the code, and the two
    shapes want opposite responses: a conflict is adjudicated by hand and then
    the run repeats, whereas a failure means a source could not be read at all
    and something is damaged. Surfacing the milder code when both are present
    would send the operator down the conflict path and leave the damaged file
    unexamined, so the precedence is part of the contract — and it is invisible
    to the two single-shape tests above, each of which would stay green with
    the precedence inverted."""
    gz, twin, resumed = _write_resumed_session_conflict(tmp_path)
    _write_truncated_gz(tmp_path / "3618" / "enc" / "broken.jsonl.gz")

    result = _run_cli("--root", str(tmp_path), "--apply")

    assert result.returncode == mig.EXIT_FAILURES, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    summary = json.loads(result.stdout)
    # Both counters non-zero: the code is a PRECEDENCE choice between two live
    # conditions, not an artefact of the conflict having gone unnoticed.
    assert summary["failed"] == 1, f"stdout={result.stdout!r}"
    assert summary["conflicts"] == 1, f"stdout={result.stdout!r}"
    # And both sources are still on disk, which is why step 4 of the runbook
    # reconciles its residue against `conflicts + failed`.
    assert gz.exists()
    assert (tmp_path / "3618" / "enc" / "broken.jsonl.gz").exists()
    assert twin.read_bytes() == resumed


# ---------------------------------------------------------------------------
# step-20: the DESTINATION side of corroborate-before-destroy
#
# step-1 pins that an unreadable SOURCE is never destroyed. This pins the other
# half: an unreadable source must not destroy the DESTINATION either. Writing
# `open(dest, 'wb')` truncates a pre-existing twin before a single source byte
# has been validated, and the failure path's `_discard_partial` then unlinks the
# remains — so a damaged `.gz` beside a healthy plain twin loses BOTH copies and
# the session's transcript is gone entirely.
#
# This is reachable on the DOCUMENTED operator path, not a contrived one.
# OPERATIONS.md §13 instructs a re-run after the fleet redeploy, which is
# exactly when both forms coexist (step-19). The resumed-session conflict guard
# cannot save it: `_gz_uncompressed_size` reads the last 4 bytes of a DAMAGED
# stream as an ISIZE trailer, which is an effectively uniform 32-bit number
# (~2e9 expected), so `_twin_supersedes_gz` almost always compares against
# garbage, returns False, and drops the file straight through to `gunzip_one`.
#
# The contract: when the source cannot be read, BOTH files survive untouched.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("build_corrupt", CORRUPTION_SHAPES)
def test_unreadable_source_never_destroys_a_pre_existing_twin(tmp_path, build_corrupt):
    """(a) THE REGRESSION. A damaged `.gz` beside a healthy plain twin must not
    cost the twin. Asserts BYTES and MTIME, because a destination write that
    truncates-then-refills would preserve neither."""
    gz = build_corrupt(tmp_path / "3618" / "enc" / "sess.jsonl.gz")
    twin = tmp_path / "3618" / "enc" / "sess.jsonl"
    # Deliberately a DIFFERENT length from the corruption fixtures' own
    # `_payload(200)` body, and shorter than it. The corrupt-body shape damages
    # only the DEFLATE stream, leaving a valid ISIZE trailer — so a twin that
    # happened to match that length would corroborate and take the skip path,
    # never reaching the destination-truncation bug this test pins. Shorter also
    # keeps `_twin_supersedes_gz` False, so all three shapes land on `gunzip_one`.
    live = _payload(137)
    twin.write_bytes(live)
    os.utime(twin, (ARCHIVED_MTIME + DAY, ARCHIVED_MTIME + DAY))
    before_mtime_ns = twin.stat().st_mtime_ns

    summary = mig.migrate_archive(tmp_path, apply=True)

    assert twin.exists(), (
        "the pre-existing plain twin was DESTROYED by a failed migration of its "
        "damaged .gz — both copies of the transcript are now gone"
    )
    assert twin.read_bytes() == live, "the live twin was truncated or overwritten"
    # gc_agent_transcripts derives the retention age from the newest descendant
    # mtime, so even a byte-identical rewrite that rolled the mtime would age a
    # live task dir out early.
    assert twin.stat().st_mtime_ns == before_mtime_ns
    # The source is damaged, so it is still retained and reported, exactly as
    # step-1 requires — this test tightens that contract, it does not relax it.
    assert gz.exists()
    assert summary["failed"] == 1
    assert str(gz) in summary["failed_paths"]
    assert summary["migrated"] == 0


@pytest.mark.parametrize("build_corrupt", CORRUPTION_SHAPES)
def test_failed_migration_leaves_no_temp_debris(tmp_path, build_corrupt):
    """(b) Whatever staging the fix uses must clean up after itself. A stray
    scratch file left in the archive root would be swept into the corpus every
    reader now walks plainly, and could be mistaken for a transcript."""
    build_corrupt(tmp_path / "3618" / "enc" / "sess.jsonl.gz")
    twin = tmp_path / "3618" / "enc" / "sess.jsonl"
    twin.write_bytes(_payload(137))  # see (a): must not match the fixture's ISIZE

    mig.migrate_archive(tmp_path, apply=True)

    survivors = sorted(p.name for p in (tmp_path / "3618" / "enc").iterdir())
    assert survivors == ["sess.jsonl", "sess.jsonl.gz"], (
        f"unexpected debris left in the archive dir: {survivors}"
    )


# ---------------------------------------------------------------------------
# Amendment: a source too small to carry an ISIZE trailer is never CORROBORATED
# BY OMISSION.
#
# `_gz_uncompressed_size` returns None below 8 bytes, and `_corroborate` skips
# the size check when it is given None — degrading the twin check to "is valid
# UTF-8", which nearly any plain file satisfies. That would declare the twin
# trustworthy and unlink the source without it ever having been decompressed or
# compared, a hole in the module's headline contract (docstring: "A file that
# cannot be corroborated is NEVER destroyed"). Nothing recoverable is lost — a
# file that small cannot hold even a complete gzip header — but the invariant
# has to hold by construction, not by luck.
# ---------------------------------------------------------------------------


def test_a_sub_isize_source_is_failed_not_skipped(tmp_path):
    """A `.gz` too short to carry a trailer, beside a perfectly healthy twin,
    must be counted `failed` with its source RETAINED — not `skipped` with the
    source unlinked on the strength of a check that never ran."""
    enc = tmp_path / "3618" / "enc"
    enc.mkdir(parents=True)
    gz = enc / "sess.jsonl.gz"
    gz.write_bytes(b"\x1f\x8b\x08\x00")  # gzip magic, truncated mid-header
    os.utime(gz, (ARCHIVED_MTIME, ARCHIVED_MTIME))
    assert mig._gz_uncompressed_size(gz) is None, "fixture must have no ISIZE"

    twin = enc / "sess.jsonl"
    live = _payload(137)
    twin.write_bytes(live)
    os.utime(twin, (ARCHIVED_MTIME + DAY, ARCHIVED_MTIME + DAY))
    before_mtime_ns = twin.stat().st_mtime_ns

    summary = mig.migrate_archive(tmp_path, apply=True)

    assert summary["failed"] == 1
    assert summary["skipped"] == 0
    assert summary["migrated"] == 0
    assert str(gz) in summary["failed_paths"]
    # The undecompressable source survives, so an operator can still look at it.
    assert gz.exists()
    # And the twin is untouched: unreadable source, nothing written.
    assert twin.read_bytes() == live
    assert twin.stat().st_mtime_ns == before_mtime_ns


def test_a_sub_isize_source_is_not_trusted_by_omission(tmp_path):
    """The unit-level statement of the same rule, so the intent survives a
    refactor of `migrate_archive`'s branch order: no ISIZE means no
    corroboration is possible, which means False — never a pass by default."""
    enc = tmp_path / "3618" / "enc"
    enc.mkdir(parents=True)
    gz = enc / "sess.jsonl.gz"
    gz.write_bytes(b"\x1f\x8b\x08\x00")
    twin = enc / "sess.jsonl"
    twin.write_bytes(_payload(137))  # readable, decodable — and irrelevant

    assert mig._twin_is_trustworthy(gz, twin) is False
    # Nor does it read as a conflict: a comparison that cannot be made must not
    # be reported as one (the two guards agree on the None case).
    assert mig._twin_supersedes_gz(gz, twin) is False
