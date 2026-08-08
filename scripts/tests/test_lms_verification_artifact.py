"""The committed health report must prove a live slate run actually happened.

PRD-MARKER:local-memory-models-eval serving

Task 3713 (LME-alpha) of `plans/local-memory-models-eval-prd.md`.

This module is the ANTI-FABRICATION GATE, and it is the reason the rest of the
substrate can be trusted.  Alpha's user-observable signal is "health script
output lists every candidate endpoint answering with valid output, and
nvidia-smi within budget".  Stated as prose in a README, that signal is only
ever provable by an agent's own narration of its own run.  Stated as this file,
it is provable by `pytest`.

Three properties make it unfakeable in the ways that actually matter:

1. The expected arm set is DERIVED from `arms.yaml`, never hardcoded here.  So
   adding a ninth arm to the slate without re-verifying it turns this suite red
   instead of silently shipping an unverified arm under a green build.  A
   hardcoded list would quietly keep passing — which is exactly the failure this
   whole file exists to prevent.
2. Every row is validated through the SAME pydantic `HealthReport` model the
   healthcheck writes with.  A hand-edited artifact that drifts from the schema
   is a parse failure, not a tolerated variant, so the artifact and the producer
   can never disagree about what a report is.
3. It is deliberately NOT `@pytest.mark.integration`.  The root addopts deselect
   that marker (`pyproject.toml`), so an integration-marked gate would be absent
   from the default suite `verify` runs — present in the tree, checked by
   nobody.  This runs offline against the committed file, in the default suite,
   every time.

It is written RED, before the live steps, and can only be greened by the run
having actually happened.  If an arm cannot be served within the measured
budget, the correct move is to escalate with the measurement — NOT to hand-write
a PASS row here.  A green suite reached that way would be a lie told to every
downstream task (eta, theta, iota) that reads this slate as verified.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

import lms_vram
from lms_healthcheck import REPORT_SCHEMA_VERSION, HealthReport
from lms_manifest import load_arms

_LMS_DIR = Path(__file__).resolve().parents[1] / 'local-model-serving'
ARTIFACT_PATH = _LMS_DIR / 'verification' / 'health-report.json'
MANIFEST_PATH = _LMS_DIR / 'arms.yaml'


@pytest.fixture(scope='module')
def raw_artifact() -> dict:
    """The committed artifact, parsed as JSON and nothing more.

    Separate from the schema-validated fixture so a malformed file reports
    "unparseable" rather than a confusing pydantic error 40 fields deep.
    """
    if not ARTIFACT_PATH.exists():
        pytest.fail(
            f'{ARTIFACT_PATH} does not exist. This artifact is the '
            "task's user-observable signal and is written by a LIVE run:\n"
            '    uv run --project shared python '
            'scripts/local-model-serving/lms_healthcheck.py --all '
            f'--output {ARTIFACT_PATH}\n'
            'Do NOT hand-write it to green this test.'
        )
    try:
        return json.loads(ARTIFACT_PATH.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - corrupt artifact
        pytest.fail(f'{ARTIFACT_PATH} is not valid JSON: {exc}')


@pytest.fixture(scope='module')
def report(raw_artifact: dict) -> HealthReport:
    """The artifact through the producer's own model.

    Validating with `HealthReport` rather than ad-hoc key checks is the point:
    the test cannot accept a shape the healthcheck would not emit.
    """
    return HealthReport.model_validate(raw_artifact)


def test_artifact_carries_every_top_level_report_section(raw_artifact: dict) -> None:
    """The step-15 report schema, checked on the raw JSON.

    Checked pre-validation so a MISSING section is named directly, instead of
    surfacing as one line of a pydantic error listing every other field too.
    """
    expected = {'schema_version', 'measured_at', 'gpu', 'arms', 'vram', 'overall'}
    missing = expected - set(raw_artifact)
    assert not missing, (
        f'{ARTIFACT_PATH} is missing report sections {sorted(missing)}; '
        'it was not produced by lms_healthcheck.run_healthcheck'
    )


def test_artifact_schema_version_matches_the_producer(report: HealthReport) -> None:
    """A stale artifact from an older report shape must not read as current.

    Without this, a schema change would leave the old artifact passing every
    other assertion here while describing a report format nothing emits.
    """
    assert report.schema_version == REPORT_SCHEMA_VERSION


def test_artifact_measured_at_is_timezone_aware(report: HealthReport) -> None:
    """A naive stamp makes a stale artifact indistinguishable from a fresh one.

    The artifact's whole job is to prove a run happened at a knowable time, so
    an unanchored timestamp defeats its purpose.
    """
    stamp = datetime.fromisoformat(report.measured_at)
    assert stamp.tzinfo is not None, (
        f'measured_at={report.measured_at!r} is timezone-naive; '
        'lms_healthcheck._now_iso emits aware UTC'
    )


def test_artifact_names_the_gpu_the_verdicts_belong_to(report: HealthReport) -> None:
    """Every verdict is relative to specific hardware.

    A report that does not say which card produced it cannot be checked against
    the budget it claims to have respected.
    """
    assert report.gpu.name.strip()
    assert report.gpu.driver_version.strip()
    assert report.gpu.total_mib > 0


def test_every_manifest_arm_has_a_passing_row(report: HealthReport) -> None:
    """The load-bearing assertion. Expected arms come from the MANIFEST.

    Deriving the expectation means a slate change without a re-run is caught.
    Hardcoding it here would let a ninth arm ship unverified under a green
    build, which is precisely the outcome this file exists to make impossible.
    """
    manifest = load_arms(MANIFEST_PATH)
    expected_ids = set(manifest.arm_ids())
    assert expected_ids, 'arms.yaml declares no arms; the manifest itself is broken'

    rows = {row.arm_id: row for row in report.arms}

    unverified = sorted(expected_ids - set(rows))
    assert not unverified, (
        f'arms {unverified} are declared in arms.yaml but carry NO row in '
        f'{ARTIFACT_PATH.name}. They were never verified; the slate is '
        'narrower than the PRD commissioned.'
    )

    failed = sorted(
        f'{arm_id}({rows[arm_id].reason}: {rows[arm_id].detail[:120]})'
        for arm_id in expected_ids
        if rows[arm_id].verdict != 'PASS'
    )
    assert not failed, (
        f'these arms did not answer with valid output: {failed}. Escalate with '
        'the arm id, the exact command, the nvidia-smi reading and the reason '
        'code — do not hand-edit a PASS row.'
    )


def test_report_carries_no_arm_absent_from_the_manifest(report: HealthReport) -> None:
    """A row for an arm the manifest does not declare is a drifted artifact.

    Either it was merged from a stale run or the manifest was narrowed after
    the run.  Both mean the artifact no longer describes THIS slate.
    """
    manifest = load_arms(MANIFEST_PATH)
    stray = sorted({row.arm_id for row in report.arms} - set(manifest.arm_ids()))
    assert not stray, (
        f'{ARTIFACT_PATH.name} carries rows for {stray}, which arms.yaml does '
        'not declare'
    )


def test_arm_rows_describe_the_arms_the_manifest_declares(report: HealthReport) -> None:
    """Each row's identity fields must match its manifest entry.

    A row claiming a different port or served_model_name than the manifest is
    the 2026-04-08 404 bug's signature: the probe measured SOMETHING, but not
    necessarily the arm the row is filed under.
    """
    manifest = load_arms(MANIFEST_PATH)
    for row in report.arms:
        arm = manifest.by_id(row.arm_id)
        assert row.served_model_name == arm.served_model_name, (
            f'{row.arm_id}: report says served_model_name='
            f'{row.served_model_name!r}, manifest says {arm.served_model_name!r}'
        )
        assert row.endpoint.startswith(arm.base_url), (
            f'{row.arm_id}: report endpoint {row.endpoint!r} is not on the '
            f"manifest's {arm.base_url!r} — a probe on the wrong port cannot "
            'attribute its result to this arm'
        )
        assert row.axis == arm.axis
        assert row.stack == arm.stack


def test_passing_rows_carry_a_real_measured_latency(report: HealthReport) -> None:
    """A PASS with zero latency was not measured over the wire.

    A synthesised row is the cheapest way to fake this artifact, and a
    zero-millisecond round trip to a model server is the tell.
    """
    unmeasured = sorted(
        row.arm_id for row in report.arms
        if row.verdict == 'PASS' and row.latency_ms <= 0.0
    )
    assert not unmeasured, (
        f'arms {unmeasured} report a PASS with latency_ms <= 0; a real probe '
        'over HTTP cannot take zero time'
    )


def test_vram_block_passes_within_the_recorded_budget(report: HealthReport) -> None:
    """nvidia-smi within budget — the second half of the PRD's stated signal.

    The SUBJECT of this check was corrected on 2026-08-06 (esc-3713-6, approved
    by the steward before the artifact existed).  It used to re-derive
    `used_mib <= nominal_ceiling_gib * 1024`, i.e. TOTAL card usage against PRD
    D10's nominal 19.5 GiB.  That charged every arm a second time for the
    ~7.3 GiB desktop+whisper baseline D10 had already subtracted, and it was not
    a big-arm technicality: a 9B AWQ measured 21.75 GiB total and failed while
    serving schema-constrained completions correctly.

    This is still the same INTERNAL-CONSISTENCY check — "the verdict and the
    numbers it was computed from disagree" — re-derived against the corrected
    subject, and it is strictly stronger than the version it replaces: the
    footprint, its live baseline and the live budget must all be present AND
    mutually coherent, where before only one comparison was re-run.
    """
    vram = report.vram
    assert vram.verdict == 'PASS', (
        f'VRAM verdict is {vram.verdict}: {vram.reason}. The arm took '
        f'{vram.arm_footprint_gib} GiB against the {vram.budget_gib} GiB free '
        'before it started.'
    )
    assert vram.arm_footprint_mib <= vram.budget_mib, (
        f'arm_footprint_mib={vram.arm_footprint_mib} exceeds the budget '
        f'{vram.budget_mib} MiB free at baseline even though the verdict says '
        'PASS; the verdict and the numbers it was computed from disagree'
    )
    assert vram.used_mib - vram.baseline_mib == vram.arm_footprint_mib, (
        f'used={vram.used_mib} minus baseline={vram.baseline_mib} is not the '
        f'reported footprint {vram.arm_footprint_mib}; the block was assembled, '
        'not measured'
    )
    # This assertion USED to read `total == used + free`, and that premise is
    # false about the instrument.  nvidia-smi reserves memory for the driver/ECC
    # that belongs to NEITHER `used` nor `free`: measured on this card,
    # `memory.reserved` is 455 MiB against a 454 MiB shortfall in the artifact.
    #
    # The producer already knew.  lms_vram documents it verbatim -- "used + free
    # never sums exactly to total (driver/ECC reserve ~450 MiB here)" -- and
    # tolerates it with _COHERENCE_TOLERANCE.  The two halves of this package
    # disagreed, and nothing could notice until an artifact existed to check.
    # The constant is IMPORTED rather than restated so they cannot drift apart
    # again.
    #
    # Bounded on BOTH sides, which the original was not: a NEGATIVE shortfall
    # (used + free exceeding total) is impossible from one reading and is the
    # signature of a block assembled from separate ones.  So this is a tighter
    # fabrication check than the identity it replaces, not a looser one -- it
    # rejects everything the old form rejected except the one case the old form
    # got wrong.
    shortfall = vram.total_mib - (vram.used_mib + vram.free_mib)
    assert 0 <= shortfall <= lms_vram._COHERENCE_TOLERANCE * vram.total_mib, (
        f'total={vram.total_mib} minus used={vram.used_mib} plus '
        f'free={vram.free_mib} leaves {shortfall} MiB unaccounted for. A small '
        'positive shortfall is the driver/ECC reserve; a negative or large one '
        'means the block was assembled from separate readings, or the fields '
        'were transposed'
    )


def test_vram_baseline_is_a_real_pre_start_reading(report: HealthReport) -> None:
    """The subtrahend must be measured, and measured BEFORE the arm.

    Required alongside the subject correction (esc-3713-6).  A zero baseline
    means the pre-start probe never ran, and subtracting it would credit the
    desktop's memory to the arm; a baseline at or above `used` means the reading
    was not taken before this run at all.  Either way the footprint below it
    would be fiction, and fiction in the flattering direction.
    """
    vram = report.vram
    assert vram.baseline_mib > 0, (
        f'baseline_mib={vram.baseline_mib}: no pre-start nvidia-smi reading '
        'stands behind this report'
    )
    assert vram.baseline_mib < vram.used_mib, (
        f'baseline_mib={vram.baseline_mib} is not below used_mib='
        f'{vram.used_mib}; the arm appears to have freed memory, so this '
        'baseline was not taken before this run'
    )
    assert 0 < vram.budget_mib <= vram.total_mib, (
        f'budget_mib={vram.budget_mib} is not a plausible free reading on a '
        f'{vram.total_mib} MiB card'
    )


def test_every_arm_actually_occupied_the_card(report: HealthReport) -> None:
    """A zero or negative footprint means the arm never started.

    A model server that loaded weights onto a GPU cannot take no VRAM, so this
    catches the artifact assembled from readings taken while nothing was
    running — the shape a fabricated report naturally takes.
    """
    assert report.vram.arm_footprint_mib > 0, (
        f'the merged vram block reports arm_footprint_mib='
        f'{report.vram.arm_footprint_mib}'
    )
    weightless = sorted(
        row.arm_id for row in report.arms if row.arm_footprint_mib <= 0
    )
    assert not weightless, (
        f'arms {weightless} report a footprint of zero or less; a model server '
        'holding weights on this GPU cannot take no VRAM'
    )


def test_each_arm_row_was_measured_at_a_knowable_time(report: HealthReport) -> None:
    """The slate is measured one arm at a time, so a single top-level stamp
    cannot say when any given arm was actually up."""
    for row in report.arms:
        stamp = datetime.fromisoformat(row.measured_at)
        assert stamp.tzinfo is not None, (
            f'{row.arm_id}: measured_at={row.measured_at!r} is timezone-naive'
        )


def test_vram_block_reports_both_budget_figures(report: HealthReport) -> None:
    """Both PRD D10's nominal ceiling AND this host's measured budget.

    Reporting only the nominal figure hides that this host has ~16.4 GiB free,
    not 19.5 — the deviation that forces Open Q3 to be resolved honestly.
    Reporting only the measured one drops the PRD's stated terms.
    """
    vram = report.vram
    assert vram.nominal_ceiling_gib > 0
    assert vram.operating_budget_gib > 0
    assert vram.operating_budget_gib < vram.nominal_ceiling_gib, (
        'the measured operating budget is not below PRD D10 nominal ceiling; '
        'if the desktop VRAM was freed for this run, say so explicitly rather '
        'than letting the artifact imply the PRD estimate held'
    )


def test_overall_verdict_agrees_with_its_parts(report: HealthReport) -> None:
    """`overall` must be derivable from the rows, not asserted independently.

    An artifact whose summary says PASS over failing parts is worse than one
    that fails: it is a green light no one would think to re-check.
    """
    derived = (
        'PASS'
        if all(row.verdict == 'PASS' for row in report.arms)
        and report.vram.verdict == 'PASS'
        else 'FAIL'
    )
    assert report.overall == derived, (
        f'overall={report.overall} but the rows and vram block derive {derived}'
    )
    assert report.overall == 'PASS'


def test_artifact_carries_the_delivered_check_marker(raw_artifact: dict) -> None:
    """The artifact is a committed file under scripts/, so the grep covers it.

    `test_lms_marker_contract.py` enumerates committed files; this asserts the
    marker lives in a real FIELD rather than a comment JSON cannot carry.
    """
    marker = 'PRD-MARKER:local-memory-models-eval serving'
    assert raw_artifact.get('prd_marker') == marker, (
        f'{ARTIFACT_PATH.name} must carry {marker!r} in a `prd_marker` field'
    )
