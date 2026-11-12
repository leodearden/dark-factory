"""Tests for scripts/legibility/census_identity.py — one census run's
identity, the previous run's state, and the commit it reads
(plans/census-incremental-prd.md §4.2)."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from legibility import census_identity, census_state, state_docs
from legibility.census_identity import PriorCensus, RunIdentity
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_ID_DAY = date(2026, 10, 6)


def _git(repo, *args) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
    ).stdout


def _allocate(plans_dir, taken=frozenset(), **kwargs):
    return census_identity.allocate_run_identity(
        project_id="dark_factory", day=_ID_DAY, plans_dir=plans_dir,
        taken_run_ids=taken, **kwargs,
    )


def test_allocate_run_identity_takes_the_plain_names_when_free(tmp_path):
    assert _allocate(tmp_path) == RunIdentity(
        run_id="census-dark_factory-20261006", basename="confusion-census-2026-10-06",
    )


@pytest.mark.parametrize("existing", [".md", ".json"])
def test_allocate_run_identity_skips_a_basename_whose_report_or_record_exists(
    tmp_path, existing,
):
    (tmp_path / f"confusion-census-2026-10-06{existing}").write_text("x", encoding="utf-8")

    assert _allocate(tmp_path) == RunIdentity(
        run_id="census-dark_factory-20261006-2", basename="confusion-census-2026-10-06-2",
    )


def test_allocate_run_identity_skips_a_taken_run_id(tmp_path):
    identity = _allocate(tmp_path, frozenset({"census-dark_factory-20261006"}))

    assert identity.run_id == "census-dark_factory-20261006-2"
    assert identity.basename == "confusion-census-2026-10-06-2"


def test_allocate_run_identity_takes_the_first_free_suffix(tmp_path):
    (tmp_path / "confusion-census-2026-10-06.md").write_text("x", encoding="utf-8")

    identity = _allocate(tmp_path, frozenset({"census-dark_factory-20261006-2"}))

    assert identity == RunIdentity(
        run_id="census-dark_factory-20261006-3", basename="confusion-census-2026-10-06-3",
    )


def test_allocate_run_identity_raises_naming_the_directory_when_exhausted(tmp_path):
    for name in ("confusion-census-2026-10-06", "confusion-census-2026-10-06-2"):
        (tmp_path / f"{name}.md").write_text("x", encoding="utf-8")

    with pytest.raises(RuntimeError) as excinfo:
        _allocate(tmp_path, limit=2)

    assert str(tmp_path) in str(excinfo.value)


def test_resolve_as_of_sha_is_the_project_head(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "-c", "user.email=t@example.com", "-c", "user.name=T",
         "commit", "-q", "--allow-empty", "-m", "fixture")

    assert census_identity.resolve_as_of_sha(tmp_path) == _git(
        tmp_path, "rev-parse", "HEAD",
    ).strip()


def test_resolve_as_of_sha_raises_naming_a_root_that_is_no_repo(tmp_path):
    with pytest.raises(RuntimeError) as excinfo:
        census_identity.resolve_as_of_sha(tmp_path)

    assert str(tmp_path) in str(excinfo.value)


def _census_state(**fields) -> census_state.CensusState:
    values: dict = dict(
        last_census_at=None,
        last_census_run_id=None,
        last_census_report=None,
        last_census_as_of_sha=None,
        session_watermark=None,
    )
    values.update(fields)
    return census_state.CensusState(**values)


def test_prior_census_reads_a_loaded_state():
    prior = census_identity.prior_census(state_docs.Ok(_census_state(
        last_census_at=date(2026, 10, 1),
        last_census_run_id="census-dark_factory-20261001",
        last_census_report="plans/confusion-census-2026-10-01.md",
        last_census_as_of_sha="b" * 40,
        session_watermark=datetime(2026, 10, 1, tzinfo=UTC),
    )))

    assert prior == PriorCensus(
        last_census_at=date(2026, 10, 1),
        run_id="census-dark_factory-20261001",
        as_of_sha="b" * 40,
    )


def test_prior_census_of_a_legacy_state_carries_only_its_date():
    read = state_docs.Ok(_census_state(last_census_at=date(2026, 10, 1)))

    assert census_identity.prior_census(read) == PriorCensus(last_census_at=date(2026, 10, 1))


@pytest.mark.parametrize("read", [
    state_docs.Missing(Path("census-state.json")),
    state_docs.Malformed(Path("census-state.json"), "x"),
], ids=["missing", "malformed"])
def test_prior_census_without_a_usable_state_is_empty(read):
    assert census_identity.prior_census(read) == PriorCensus()


def test_prior_census_refuses_an_ok_from_a_second_copy_of_state_docs(monkeypatch):
    """One source file loaded as two modules has two Ok classes. Reading the
    other copy's Ok as "no previous run" would lose the taken run id."""
    spec = importlib.util.spec_from_file_location("state_docs_second_copy", state_docs.__file__)
    assert spec is not None and spec.loader is not None
    second_copy = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, second_copy)
    spec.loader.exec_module(second_copy)
    foreign_ok = second_copy.Ok(_census_state(last_census_run_id="census-dark_factory-20261001"))

    with pytest.raises(AssertionError):
        census_identity.prior_census(foreign_ok)
