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
``census_state.advance`` write atomically through a temp file in the
target's own directory, which raises the same ``FileNotFoundError`` on a
missing directory unless the writer creates it. dark-factory's own tree
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

import itertools
import json
from typing import Any

import pytest
from legibility import census_run, embeddings, landing_gate, unlanded
from legibility import config as config_mod
from legibility_census_scaffold import narrowing, with_seams
from legibility_task_store_fake import FakeTaskStore
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_DATE = "2026-08-03"
_EMPTY_SYNTHESIS = json.dumps(
    {"new": [], "re_observed": [], "phase_refinements": [], "corrections": [], "notes": []},
)
_ONE_HOT_AXES = itertools.count()


def _distinct_one_hot_embed(texts, *, model):
    """An embed_fn under which no two texts are ever similar: each text gets
    a one-hot vector on an axis no earlier text was given."""
    axes = [next(_ONE_HOT_AXES) for _ in texts]
    width = max(axes, default=-1) + 1
    return [[1.0 if index == axis else 0.0 for index in range(width)] for axis in axes]


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


def _never_roll_back(**kwargs):
    raise AssertionError("roll_back must not run on a landed commit")


_done = narrowing(census_run.Done)
_deferred = narrowing(census_run.Deferred)


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
    effects = census_run.CensusEffects(
        batch_source=[],
        selection_of=lambda: None,
        invoke=lambda prompt, model: "pong",
        verify_fn=lambda clusters, *, model: {"verified": [], "rejected": [], "fixed": []},
        synthesize_fn=lambda brief, *, model: _EMPTY_SYNTHESIS,
        embedder=embeddings.Embedder(
            embed_fn=_distinct_one_hot_embed, cache_path=root / "embeddings.sqlite",
        ),
        submit_fn=lambda payload: {"task_id": "t1"},
        task_store=FakeTaskStore(),
        status_fetcher=lambda: {"statuses": {}},
        escalate_fn=lambda **kw: None,
        gate=lambda: landing_gate.GateVerdict(landing_gate.GateStatus.PASSED, ("true",), 0),
        commit=lambda **kw: unlanded.CommitResult(ok=True, sha="f" * 40, stderr="", attempts=1),
        roll_back=_never_roll_back,
    )
    effects, overrides = with_seams(effects, overrides)
    kwargs: dict[str, Any] = dict(
        effects=effects,
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
        run_id="census-target_project-20260803",
        as_of_sha="a" * 40,
        since=None,
        force=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_run_census_creates_missing_report_parent_dir(tmp_path):
    """*root* has no ``plans/``; the dated report must still be written."""
    kwargs = _run_census_kwargs(tmp_path)
    assert not (tmp_path / "plans").exists(), "precondition: no plans/ dir"

    outcome = census_run.run_census(**kwargs)

    outcome = _done(outcome)
    assert kwargs["report_path"].exists()


def test_run_census_creates_missing_dry_run_payloads_parent_dir(tmp_path):
    """The EXACT proven failure: the 2026-08-03 reify census died writing
    ``<root>/plans/confusion-census-<date>-payloads.json`` into a project
    with no ``plans/`` directory, after the full mining spend."""
    payloads_path = tmp_path / "plans" / f"confusion-census-{_DATE}-payloads.json"
    kwargs = _run_census_kwargs(tmp_path, dry_run_payloads_path=payloads_path)
    assert not (tmp_path / "plans").exists(), "precondition: no plans/ dir"

    outcome = census_run.run_census(**kwargs)

    outcome = _done(outcome)
    assert payloads_path.exists()
    # An empty list is the honest content for a run with no novel clusters.
    assert json.loads(payloads_path.read_text(encoding="utf-8")) == []


def test_run_census_creates_missing_codebook_and_state_parent_dirs(tmp_path):
    """The SECOND copy of the defect. ``codebook.dump`` and
    ``census_state.advance`` write atomically through a temp file in the
    target's own directory, which raises ``FileNotFoundError`` on a missing
    directory exactly as ``write_text`` does unless the writer creates it. reify never reached this pair only because ``main()`` loads its
    config from ``<root>/docs/legibility/legibility.yaml``, so that
    directory necessarily existed; a target reached via ``--config``
    pointing elsewhere trips it at the same sunk cost."""
    kwargs = _run_census_kwargs(tmp_path)
    assert not (tmp_path / "docs" / "legibility").exists(), "precondition"

    outcome = census_run.run_census(**kwargs)

    outcome = _done(outcome)
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

    outcome = census_run.run_census(**kwargs)

    outcome = _done(outcome)
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

    outcome = census_run.run_census(**kwargs)

    outcome = _deferred(outcome)
    assert not (tmp_path / "plans").exists()
    assert not (tmp_path / "docs" / "legibility").exists()
