"""Tests for scripts/legibility/census.py's OUTPUT-PATH creation — that a
census run creates the parent directory of every file it persists, so a
target project that does not share dark-factory's own directory layout is
censusable at all.

Regression origin: on 2026-08-03 a census of /home/leo/src/reify mined to
saturation (~12.5h, ~$100) and then died on its first output write with
``[Errno 2] No such file or directory:
'/home/leo/src/reify/plans/confusion-census-2026-08-02-payloads.json'`` —
reify simply has no ``plans/`` directory. Every persisted output of a run
is affected, not just that one file: ``report_path`` and the dry-run
payloads go through ``Path.write_text``, while ``codebook.dump`` and
``advance_census_state`` write atomically via
``tempfile.mkstemp(dir=os.path.dirname(path))``, which raises the same
``FileNotFoundError`` on a missing directory. dark-factory's own tree
happens to contain ``plans/`` and ``docs/legibility/``, which is the only
reason this was never hit before.

This file is deliberately SELF-CONTAINED (cross-test-module imports are
fragile under the repo-wide ``--import-mode=importlib`` addopts, see
scripts/tests/conftest.py) and deliberately SMALL: every test drives
``run_census`` with an EMPTY ``batch_source``, so it needs none of
test_legibility_census.py's digest/mining/codebook-merge fixtures. A run
that mines nothing still writes the report, dumps the codebook, advances
census-state and (under ``--dry-run-filing``) writes the payload JSON —
which is precisely and only the surface under test here.
"""
from __future__ import annotations

import json
from typing import Any

import census as mod

import config as config_mod

_DATE = "2026-08-03"


def _minimal_v2_codebook() -> dict:
    """A minimal well-formed v2 codebook: one entry, no candidates."""
    return {
        "version": 2,
        "entries": [
            {
                "id": "entry-a",
                "title": "Some confusion cluster",
                "severity": "high",
                "status": "open",
                "origin_phase": "implement",
                "manifested_phase": "merge",
                "sightings": [],
            }
        ],
        "candidates": [],
    }


def _run_census_kwargs(root, **overrides) -> dict[str, Any]:
    """Every ``run_census`` seam, wired to trivial inline fakes, with the
    four output paths under directories that do NOT exist in *root*.

    Mirrors the SHAPE of test_legibility_census.py's ``_run_census_kwargs``
    (including the load-bearing ``dict[str, Any]`` annotation — without it
    pyright re-reports this heterogeneous dict's value union once per
    parameter at every ``run_census(**kwargs)`` call site) but NOT its fake
    zoo: ``batch_source=[]`` mines nothing, so the only LLM call any test
    here makes is the headroom probe.
    """
    kwargs: dict[str, Any] = dict(
        batch_source=[],
        invoke=lambda prompt, model: "pong",
        verify_fn=lambda clusters, *, model: {"verified": [], "rejected": [], "fixed": []},
        synthesize_fn=lambda verified, *, model: "No novel clusters this census.",
        submit_fn=lambda payload: {"task_id": "t1"},
        escalate_fn=lambda **kw: None,
        status_fetcher=lambda: {"statuses": {}},
        commit=lambda **kw: None,
        codebook_dict=_minimal_v2_codebook(),
        config=config_mod.LegibilityConfig(
            project_id="target_project",
            project_root=str(root),
            escalation_port=8103,
            cwd_prefixes=[str(root)],
        ),
        project_root=str(root),
        project_id="target_project",
        # docs/legibility/ and plans/ both deliberately absent from *root* --
        # this is the layout of any project that is not dark-factory itself.
        codebook_path=root / "docs" / "legibility" / "confusion-codebook.yaml",
        census_state_path=root / "docs" / "legibility" / "census-state.json",
        report_path=root / "plans" / f"confusion-census-{_DATE}.md",
        date=_DATE,
        force=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_run_census_creates_missing_report_parent_dir(tmp_path):
    """*root* has no ``plans/``; the dated report must still be written."""
    kwargs = _run_census_kwargs(tmp_path)
    assert not (tmp_path / "plans").exists(), "precondition: no plans/ dir"

    outcome = mod.run_census(**kwargs)

    assert outcome.status == "done"
    assert kwargs["report_path"].exists()


def test_run_census_creates_missing_dry_run_payloads_parent_dir(tmp_path):
    """The EXACT proven failure: the 2026-08-03 reify census died writing
    ``<root>/plans/confusion-census-<date>-payloads.json`` into a project
    with no ``plans/`` directory, after the full mining spend."""
    payloads_path = tmp_path / "plans" / f"confusion-census-{_DATE}-payloads.json"
    kwargs = _run_census_kwargs(tmp_path, dry_run_payloads_path=payloads_path)
    assert not (tmp_path / "plans").exists(), "precondition: no plans/ dir"

    outcome = mod.run_census(**kwargs)

    assert outcome.status == "done"
    assert payloads_path.exists()
    # An empty list is the honest content for a run with no novel clusters.
    assert json.loads(payloads_path.read_text(encoding="utf-8")) == []


def test_run_census_creates_missing_codebook_and_state_parent_dirs(tmp_path):
    """The SECOND copy of the defect. ``codebook.dump`` and
    ``advance_census_state`` write atomically via
    ``tempfile.mkstemp(dir=os.path.dirname(path))``, which raises
    ``FileNotFoundError`` on a missing directory exactly as ``write_text``
    does. reify never reached this pair only because ``main()`` loads its
    config from ``<root>/docs/legibility/legibility.yaml``, so that
    directory necessarily existed; a target reached via ``--config``
    pointing elsewhere trips it at the same sunk cost."""
    kwargs = _run_census_kwargs(tmp_path)
    assert not (tmp_path / "docs" / "legibility").exists(), "precondition"

    outcome = mod.run_census(**kwargs)

    assert outcome.status == "done"
    assert kwargs["codebook_path"].exists()
    assert kwargs["census_state_path"].exists()


def test_run_census_creates_output_dirs_before_mining_begins(tmp_path):
    """FAIL-FAST: the parents exist before the first batch is pulled.

    Pins that an un-creatable output path can never again burn a full
    mining run — reify sank ~12.5h and ~$100 into mining and only then
    discovered its report directory did not exist.
    """
    report_path = tmp_path / "plans" / f"confusion-census-{_DATE}.md"
    observed = []

    class _RecordingBatchSource:
        def __iter__(self):
            observed.append(report_path.parent.exists())
            return iter(())

    kwargs = _run_census_kwargs(
        tmp_path, batch_source=_RecordingBatchSource(), report_path=report_path,
    )

    outcome = mod.run_census(**kwargs)

    assert outcome.status == "done"
    assert observed == [True], (
        "output parents must be created BEFORE mining pulls its first batch"
    )


def test_run_census_deferred_at_headroom_creates_no_output_dirs(tmp_path):
    """The DEFER branch stays side-effect-free: no empty directories are
    left behind in the target project. Matches the existing no-side-effects
    contract that test_legibility_census.py's
    ``test_run_census_defers_on_headroom_banner`` already pins."""
    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=lambda prompt, model: "You have reached your usage limit for this period.",
    )

    outcome = mod.run_census(**kwargs)

    assert outcome.status == "deferred"
    assert not (tmp_path / "plans").exists()
    assert not (tmp_path / "docs" / "legibility").exists()


def test_advance_census_state_creates_its_own_parent_dir(tmp_path):
    """The writer-side half of the guarantee, unit-tested directly.

    ``run_census``'s ``_ensure_output_parents`` buys FAIL-FAST but covers
    only that function's own four paths. ``advance_census_state`` is
    census-state.json's SOLE writer and is callable from anywhere, so it
    creates its own parent too (mirroring trickle_state's atomic writer) —
    otherwise ``tempfile.mkstemp(dir=<missing>)`` raises the very
    FileNotFoundError this task exists to remove, for any future caller
    that does not route through run_census.
    """
    state_path = tmp_path / "docs" / "legibility" / "census-state.json"
    assert not state_path.parent.exists()

    mod.advance_census_state(
        state_path,
        now_iso="2026-08-03T00:00:00+00:00",
        report_path="plans/confusion-census-2026-08-03.md",
        done_count=7,
    )

    assert json.loads(state_path.read_text(encoding="utf-8")) == {
        "last_census_at": "2026-08-03T00:00:00+00:00",
        "last_census_report": "plans/confusion-census-2026-08-03.md",
        "last_census_done_count": 7,
    }
