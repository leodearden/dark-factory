"""Tests for scripts/legibility/census_identity.py — one census run's
identity, the previous run's state, and the commit it reads
(plans/census-incremental-prd.md §4.2)."""
from __future__ import annotations

import subprocess
from datetime import date

import pytest
from legibility import census_identity
from legibility.census_identity import PriorCensus, RunIdentity

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


def test_prior_census_reads_a_loaded_state():
    prior = census_identity.prior_census("ok", {
        "last_census_at": "2026-10-01",
        "last_census_run_id": "census-dark_factory-20261001",
        "last_census_as_of_sha": "b" * 40,
    })

    assert prior == PriorCensus(
        last_census_at=date(2026, 10, 1),
        run_id="census-dark_factory-20261001",
        as_of_sha="b" * 40,
    )


def test_prior_census_of_a_legacy_state_carries_only_its_date():
    assert census_identity.prior_census("ok", {"last_census_at": "2026-10-01"}) == PriorCensus(
        last_census_at=date(2026, 10, 1),
    )


@pytest.mark.parametrize(("status", "state"), [
    ("missing", None),
    ("malformed", None),
    ("ok", None),
])
def test_prior_census_without_a_usable_state_is_empty(status, state):
    assert census_identity.prior_census(status, state) == PriorCensus()
