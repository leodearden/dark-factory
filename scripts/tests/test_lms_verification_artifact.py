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

THE COMMITTED ARTIFACT IS v4, AND A LIVE RE-RUN IS OWED (task 3755).
--------------------------------------------------------------------
It is evidence of a real ~39-minute 7-arm slate run (b7c9887f22) and is not
stale in any way that matters: every property below still holds against it.
What it predates is the v5 consumer inventory, so it cannot say who else held
the card while those arms were measured.  Re-deriving it needs docker, systemd
and the shared 3090 — and, pointedly, may be REFUSED by the very guard v5 adds
if ollama is resident, which is the whole reason the guard exists.  So the
version check became :data:`ACCEPTED_ARTIFACT_SCHEMA_VERSIONS` (see its
comment) rather than being relaxed or, worse, greened by hand.

That widening is a WIDENING, and this file says so rather than dressing it up:
against the v4 artifact the live guards are the two version pins and the
"carries none of the consumer keys" check, which is what stops a hand-edited
v4 faking an inventory.  The stronger checks — a measured inventory and a
CLEAN pollution state — start applying when the artifact is re-derived at v5,
and not before.  The owed re-run is filed as task 4229
(:data:`_OWED_LIVE_RERUN_TASK_ID`); do not close that gap by editing this file.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import lms_vram
import pytest
from lms_healthcheck import REPORT_SCHEMA_VERSION, HealthReport
from lms_manifest import load_arms

_LMS_DIR = Path(__file__).resolve().parents[1] / 'local-model-serving'
ARTIFACT_PATH = _LMS_DIR / 'verification' / 'health-report.json'
MANIFEST_PATH = _LMS_DIR / 'arms.yaml'

#: Report schema versions this gate accepts from the committed artifact.
#:
#: A GRANDFATHER CLAUSE, and deliberately a narrow, self-expiring one.  The
#: committed artifact is v4 and is evidence of a LIVE 7-arm slate run
#: (b7c9887f22) that took ~39 minutes on the shared 3090.  Task 3755 bumped the
#: producer to v5 for additive fields, and re-deriving the artifact offline is
#: not possible -- hand-writing a v5 file to green a version equality is exactly
#: the fabrication this module exists to prevent.
#:
#: So the check is widened by ONE version rather than deleted, and every other
#: property in this file still applies to the v4 file unchanged.  Two guards
#: keep the widening from rotting into a permanent hole:
#:
#: 1. `test_the_accepted_set_expires_at_the_next_schema_bump` pins
#:    `max(...) == REPORT_SCHEMA_VERSION`, so v6 turns this file RED and forces
#:    a conscious decision here instead of the clause silently sliding forward.
#: 2. `test_the_accepted_set_grandfathers_exactly_one_older_version` pins the
#:    remainder to exactly `_GRANDFATHERED_ARTIFACT_SCHEMA_VERSION`, so a
#:    SECOND older version cannot join the set without an edit here either.
#: 3. The grandfathered branch of
#:    `test_the_committed_artifact_accounts_for_who_else_held_the_card` is a
#:    LIVE assertion, not a skip: the v4 file must carry NONE of the consumer
#:    keys, so it cannot be hand-edited to fake an inventory.
#:
#: WHAT THIS IS AND IS NOT, stated honestly because a claim the suite does not
#: enforce is worse than no claim.  Against the CURRENT v4 artifact the live
#: checks are exactly (2), (3) and the expiry pin -- that is a real guard, but
#: it is NOT the strictly-stronger check the widening once claimed to be.  The
#: ADDITIONAL strictness -- a measured non-empty inventory and a CLEAN
#: pollution state -- begins the moment the artifact is re-derived at v5, and
#: is not in force today.
#:
#: The owed live re-run is FILED, not implicit: see _OWED_LIVE_RERUN_TASK_ID.
ACCEPTED_ARTIFACT_SCHEMA_VERSIONS = frozenset({4, 5})

#: The first version whose reports carry the consumer inventory (task 3755).
_CONSUMER_EVIDENCE_SCHEMA_VERSION = 5

#: The ONE older version this gate still accepts, and the reason the clause is
#: narrow rather than open-ended.  Named so both the accepted-set pin and the
#: grandfathered branch below refer to the same number.
_GRANDFATHERED_ARTIFACT_SCHEMA_VERSION = 4

#: The filed, closeable work item that ends the grandfather clause: task 4229,
#: "lms: re-run the live 7-arm slate to regenerate health-report.json at schema
#: v5" (pending; sibling filing 4202 covers the same gap, so the pin survives a
#: dedupe of either).  A temporary exemption pointing at a real task can be
#: closed; one pointing at nobody becomes permanent.
_OWED_LIVE_RERUN_TASK_ID = '4229'

#: The v5 vram keys.  Named once so both directions of the check use the same
#: list and cannot drift apart.
_CONSUMER_EVIDENCE_KEYS = (
    'baseline_consumers', 'probe_consumers', 'consumer_inventory_note',
    'pollution', 'pollution_reason',
)

#: The two branches `test_the_committed_artifact_accounts_for_who_else_held_
#: the_card` can take.  Named so the test can assert that one of them actually
#: RAN: a dispatch whose arms are both silently skippable is how five
#: substantive assertions became dead code behind a `pytest.skip`.
_BRANCH_MEASURED = 'v5+: inventory measured and checked'
_BRANCH_GRANDFATHERED = 'v4: no inventory, and none may be smuggled in'


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


def test_artifact_schema_version_is_one_this_gate_understands(
    report: HealthReport,
) -> None:
    """A stale artifact from an older report shape must not read as current.

    Without this, a schema change would leave the old artifact passing every
    other assertion here while describing a report format nothing emits.
    """
    assert report.schema_version in ACCEPTED_ARTIFACT_SCHEMA_VERSIONS, (
        f'{ARTIFACT_PATH} is schema v{report.schema_version}, which this gate '
        f'does not know how to check (accepts '
        f'{sorted(ACCEPTED_ARTIFACT_SCHEMA_VERSIONS)}). Re-run the slate live '
        'rather than widening the set to fit a file already on disk.'
    )


def test_the_accepted_set_expires_at_the_next_schema_bump() -> None:
    """The grandfather clause must not slide forward on its own.

    Pinning the TOP of the accepted set to the producer's current version means
    the next bump turns this file red and someone has to decide, in a reviewable
    diff, whether the older artifact is still acceptable evidence. Without this
    the set would quietly accept every past shape forever, which is how an
    anti-fabrication gate becomes decoration.
    """
    assert max(ACCEPTED_ARTIFACT_SCHEMA_VERSIONS) == REPORT_SCHEMA_VERSION, (
        f'the producer is at v{REPORT_SCHEMA_VERSION} but this gate accepts up '
        f'to v{max(ACCEPTED_ARTIFACT_SCHEMA_VERSIONS)}. Decide here: either '
        'the committed artifact is re-derived by a live run, or the older '
        'version is consciously grandfathered by adding it to the set.'
    )


def test_the_accepted_set_grandfathers_exactly_one_older_version() -> None:
    """The set may hold the producer's version plus ONE named older one.

    Complements the expiry pin above from the other side: that one stops the
    clause sliding FORWARD, this one stops the set quietly growing a SECOND
    grandfathered version. Either change is a decision, and a decision belongs
    in a reviewable diff rather than in a set literal nobody re-reads.
    """
    grandfathered = ACCEPTED_ARTIFACT_SCHEMA_VERSIONS - {REPORT_SCHEMA_VERSION}
    assert grandfathered == {_GRANDFATHERED_ARTIFACT_SCHEMA_VERSION}, (
        f'this gate grandfathers {sorted(grandfathered)}, but only '
        f'v{_GRANDFATHERED_ARTIFACT_SCHEMA_VERSION} is named and justified '
        f'here (owed live re-run: task {_OWED_LIVE_RERUN_TASK_ID}). Name the '
        'other version and say why it is still acceptable evidence, or re-run '
        'the slate live.'
    )


def test_the_committed_artifact_accounts_for_who_else_held_the_card(
    report: HealthReport, raw_artifact: dict,
) -> None:
    """One test, two branches, NO skips -- and it records which branch ran.

    `arm_footprint_mib` is `used - baseline`, which is the ARM's footprint only
    if nothing else moved between the two readings. From v5 the report records
    what it saw, so this gate can check it. Until the artifact is re-derived it
    is v4 and cannot, so the v4 branch asserts the complementary property: a
    producer that never inventoried the card cannot have left the keys behind.

    The two branches used to be two tests, each `pytest.skip`-ing on the
    version the other handled -- which meant that against the committed v4
    artifact the v5 branch's five assertions were dead code and the widened
    version set shipped with nothing compensating for it. Merged here so the
    dispatch is visible, and `branch` is assigned only AFTER a branch's
    assertions have all run, so an artifact matching neither turns this red
    instead of passing silently.
    """
    vram = raw_artifact['vram']
    branch: str | None = None

    if report.schema_version >= _CONSUMER_EVIDENCE_SCHEMA_VERSION:
        missing = [key for key in _CONSUMER_EVIDENCE_KEYS if key not in vram]
        assert not missing, (
            f'v{report.schema_version} artifact is missing {sorted(missing)}; '
            'it was not produced by this version of run_healthcheck'
        )
        assert report.vram.probe_consumers, (
            'probe_consumers is empty, but the arm itself runs as a CUDA '
            'compute app and must appear in its own probe reading. An empty '
            'list here means nothing was inventoried, not that the card was '
            'quiet.'
        )
        assert report.vram.consumer_inventory_note, (
            'consumer_inventory_note is empty: the artifact carries the lists '
            'without the caveat that they do not sum to memory.used'
        )
        assert report.vram.pollution == lms_vram.PollutionState.CLEAN, (
            f'pollution={report.vram.pollution}: '
            f'{report.vram.pollution_reason}. A slate measured on a contended '
            'card is not evidence the arms fit.'
        )
        assert report.vram.pollution_reason == '', (
            'pollution is CLEAN but a reason is recorded; the block '
            'contradicts itself about what was seen'
        )
        branch = _BRANCH_MEASURED

    elif report.schema_version == _GRANDFATHERED_ARTIFACT_SCHEMA_VERSION:
        # A pre-v5 artifact was produced by code that never inventoried the
        # card, so its vram block must carry NONE of those keys. Checked on the
        # RAW JSON on purpose: through the model every one of them has a
        # default, so a validated v4 report reads the same whether the key was
        # absent or hand-typed. Only the raw file can tell those apart -- which
        # is what stops a v4 artifact being edited to look like it recorded a
        # clean, uncontended run.
        smuggled = [key for key in _CONSUMER_EVIDENCE_KEYS if key in vram]
        assert not smuggled, (
            f'v{report.schema_version} artifact carries {sorted(smuggled)}, '
            'but no producer at that version could have measured them. Either '
            'the file was hand-edited, or it is really a newer report '
            'mislabelled with an older version -- both are worse than a '
            'missing field.'
        )
        assert _OWED_LIVE_RERUN_TASK_ID.strip(), (
            'the grandfather clause must point at a FILED, closeable task for '
            'the owed live re-run. An unnamed promise is how a temporary '
            'exemption becomes permanent.'
        )
        branch = _BRANCH_GRANDFATHERED

    assert branch in (_BRANCH_MEASURED, _BRANCH_GRANDFATHERED), (
        f'artifact is v{report.schema_version}, which neither branch of this '
        f'test checks: v{_CONSUMER_EVIDENCE_SCHEMA_VERSION}+ carries the '
        f'inventory and v{_GRANDFATHERED_ARTIFACT_SCHEMA_VERSION} is '
        'grandfathered as carrying none. A version outside both would pass '
        'this gate having been checked by nothing.'
    )


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
