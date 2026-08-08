"""Tests for lms_vram — the measured VRAM budget (task 3713, LME-alpha).

PRD-MARKER:local-memory-models-eval serving

Everything here is pure: the GPU is reached only through an injected runner, so
these tests are green on a box with no NVIDIA driver at all.

The point of the module under test is that the budget is a LIVE MEASUREMENT,
not the hard-coded 19-20 GiB of PRD D10.  Measured on this host 2026-08-05:
24576 MiB total, 7362 used, 16761 free -- whisper-writer holds 4050 MiB and the
KDE/X11 desktop the ~3.3 GB balance.  So the real operating budget is ~16.4
GiB, and the MoE arm's nominal ~17 GiB does not fit.  `test_arm_fits_*` below
is where that deviation stops being a README note and becomes behaviour.

Every probe-parse failure raises rather than returning a zero.  A silent 0 is
the dangerous answer: `used_mib = 0` reports a *passing* budget with maximal
headroom off a broken probe, which is precisely the reading an operator would
trust.
"""
import json
from datetime import datetime

import lms_manifest
import lms_vram
import pytest

# Verbatim `nvidia-smi --query-gpu=memory.total,memory.used,memory.free
# --format=csv,noheader,nounits` output captured on this host 2026-08-05.
MEASURED_CSV = '24576, 7362, 16761\n'


def _arm(**overrides):
    fields = {
        'arm_id': 'demo',
        'axis': 'llm',
        'reasoning': 'off',
        'stack': 'vllm',
        'image': 'vllm/vllm-openai:v0.26.0',
        'model_ref': 'org/model',
        'port': 8410,
        'served_model_name': 'demo',
        'structured_output_mode': 'json_schema',
        'est_vram_gib': 6.0,
    }
    fields.update(overrides)
    return lms_manifest.ArmEntry(**fields)


# ---------------------------------------------------------------------------
# (a) parse_nvidia_smi_csv
# ---------------------------------------------------------------------------


def test_parse_nvidia_smi_csv_on_the_measured_host_output():
    reading = lms_vram.parse_nvidia_smi_csv(MEASURED_CSV)

    assert reading.total_mib == 24576
    assert reading.used_mib == 7362
    assert reading.free_mib == 16761


def test_parse_exposes_gib_conversions():
    reading = lms_vram.parse_nvidia_smi_csv(MEASURED_CSV)

    assert reading.total_gib == pytest.approx(24.0, abs=0.01)
    assert reading.free_gib == pytest.approx(16.37, abs=0.01)
    assert reading.used_gib == pytest.approx(7.19, abs=0.01)


@pytest.mark.parametrize(
    ('label', 'text'),
    [
        ('empty', ''),
        ('whitespace only', '   \n\n'),
        # The caller forgot `noheader`.  Parsing row 2 anyway would hide a
        # command-line drift that changes what the numbers MEAN.
        ('header row present',
         'memory.total [MiB], memory.used [MiB], memory.free [MiB]\n24576, 7362, 16761\n'),
        ('non-numeric field', '24576, abc, 16761\n'),
        # nvidia-smi emits [N/A] for a field the driver cannot report.
        ('[N/A] value', '24576, [N/A], 16761\n'),
        ('too few fields', '24576, 7362\n'),
        ('too many fields', '24576, 7362, 16761, 99\n'),
        # A second GPU makes "the budget" ambiguous; this rig is single-GPU.
        ('two GPUs', '24576, 7362, 16761\n24576, 100, 24476\n'),
        ('units left in', '24576 MiB, 7362 MiB, 16761 MiB\n'),
    ],
)
def test_parse_nvidia_smi_csv_raises_rather_than_returning_zero(label, text):
    with pytest.raises(lms_vram.VramProbeError):
        lms_vram.parse_nvidia_smi_csv(text)


def test_parse_error_message_quotes_the_offending_output():
    with pytest.raises(lms_vram.VramProbeError) as excinfo:
        lms_vram.parse_nvidia_smi_csv('24576, [N/A], 16761\n')
    assert '[N/A]' in str(excinfo.value)


def test_parse_rejects_a_reading_whose_parts_do_not_add_up():
    """total must account for used + free; a wild mismatch means the fields
    were transposed or the query changed shape."""
    with pytest.raises(lms_vram.VramProbeError):
        lms_vram.parse_nvidia_smi_csv('24576, 7362, 1\n')


# ---------------------------------------------------------------------------
# probe_gpu — the GPU is reached only through an injected runner
# ---------------------------------------------------------------------------


def test_probe_gpu_shells_the_expected_query_through_the_injected_runner():
    recorded = []

    def runner(argv):
        recorded.append(argv)
        return MEASURED_CSV

    reading = lms_vram.probe_gpu(runner)

    assert reading.total_mib == 24576
    assert recorded == [[
        'nvidia-smi',
        '--query-gpu=memory.total,memory.used,memory.free',
        '--format=csv,noheader,nounits',
    ]]


def test_probe_gpu_wraps_a_runner_failure_in_the_typed_error():
    def runner(argv):
        raise FileNotFoundError('nvidia-smi')

    with pytest.raises(lms_vram.VramProbeError) as excinfo:
        lms_vram.probe_gpu(runner)
    assert 'nvidia-smi' in str(excinfo.value)


# ---------------------------------------------------------------------------
# (b) gpu_memory_utilization_for
# ---------------------------------------------------------------------------


def test_gpu_memory_utilization_is_the_budget_share_rounded_to_3dp():
    assert lms_vram.gpu_memory_utilization_for(16.0, 24.0) == 0.667
    assert lms_vram.gpu_memory_utilization_for(12.0, 24.0) == 0.5
    assert lms_vram.gpu_memory_utilization_for(24.0, 24.0) == 1.0


def test_gpu_memory_utilization_is_never_the_0_95_pod_era_default():
    """The pod-era entrypoint default (0.95 of a 96 GB card) would hand vLLM
    ~23 GiB of this 24 GiB card and evict whisper-writer, which Leo requires
    resident (PRD D10)."""
    assert lms_vram.gpu_memory_utilization_for(16.37, 24.0) < 0.95


@pytest.mark.parametrize(
    ('budget_gib', 'total_gib'),
    [
        (30.0, 24.0),   # budget exceeds the card
        (0.0, 24.0),
        (-1.0, 24.0),
        (16.0, 0.0),
        (16.0, -24.0),
    ],
)
def test_gpu_memory_utilization_raises_on_an_impossible_budget(budget_gib, total_gib):
    with pytest.raises(lms_vram.VramProbeError):
        lms_vram.gpu_memory_utilization_for(budget_gib, total_gib)


# ---------------------------------------------------------------------------
# (c) evaluate_budget — the ARM's footprint against the LIVE budget
# ---------------------------------------------------------------------------
#
# The subject of this verdict was corrected on 2026-08-06 (esc-3713-6).  It used
# to judge TOTAL card usage against PRD D10's nominal 19.5 GiB, which charged
# every arm a second time for the ~7.3 GiB desktop+whisper baseline D10 had
# already subtracted -- so a 9B AWQ that served correctly measured 21.75 GiB
# total and FAILED.  PRD l.165/l.192 derive 19.5 as the allowance *to the arm*.
#
# It now judges `used - baseline` against the free VRAM measured at that same
# baseline reading.  Both reference figures still travel with every verdict, and
# the safety margin is deliberately NOT re-added here: it is applied once, at
# allocation time, in `lms_serve._memory_share_for`.  Charging it twice would put
# a correctly-sized arm on a false knife edge.

_BASELINE_MIB = 7362
_BASELINE_FREE_MIB = 16761


def _budget(used_mib, total_mib=24576, baseline_mib=_BASELINE_MIB,
            baseline_free_mib=_BASELINE_FREE_MIB):
    return lms_vram.evaluate_budget(
        used_mib=used_mib,
        total_mib=total_mib,
        baseline_mib=baseline_mib,
        baseline_free_mib=baseline_free_mib,
    )


def test_evaluate_budget_reports_both_the_nominal_ceiling_and_the_measured_budget():
    verdict = _budget(used_mib=14000)

    assert verdict.nominal_ceiling_gib == pytest.approx(19.5)
    assert verdict.operating_budget_gib == pytest.approx(
        lms_vram.MEASURED_OPERATING_BUDGET_GIB
    )
    # The measured budget is materially BELOW the PRD's nominal ceiling; that
    # gap is the finding, so a report that showed only one figure would hide it.
    assert verdict.operating_budget_gib < verdict.nominal_ceiling_gib


def test_evaluate_budget_charges_the_arm_only_for_what_the_arm_took():
    """The correction, stated as arithmetic.

    22271 MiB used with a 7310 MiB baseline is a 14961 MiB arm -- the real
    `qwen3.5-9b` measurement from 2026-08-06.  Under the old total-usage subject
    that arm read as 21.75 GiB and failed; it took 14.61 GiB and fitted.
    """
    verdict = _budget(used_mib=22271, baseline_mib=7310, baseline_free_mib=16813)

    assert verdict.arm_footprint_mib == 14961
    assert verdict.arm_footprint_gib == pytest.approx(14.61, abs=0.01)
    assert verdict.budget_mib == 16813
    assert verdict.verdict == 'PASS'
    assert verdict.headroom_gib == pytest.approx((16813 - 14961) / 1024, abs=0.01)


def test_evaluate_budget_fails_when_the_arm_exceeds_the_live_free_reading():
    verdict = _budget(used_mib=24500, baseline_mib=1000, baseline_free_mib=16761)

    assert verdict.arm_footprint_mib == 23500
    assert verdict.verdict == 'FAIL'
    assert verdict.headroom_gib < 0
    assert 'budget' in verdict.reason.lower()


def test_evaluate_budget_uses_the_live_budget_not_the_frozen_constant():
    """A frozen 16.37 GiB budget would misattribute desktop drift to the arm.

    The two calls below differ ONLY in what was free when the baseline was
    taken.  If the constant were the operative number they would agree, and an
    arm run on an emptier card would be judged against capacity it did not have.
    """
    roomy = _budget(used_mib=20000, baseline_mib=4000, baseline_free_mib=20000)
    cramped = _budget(used_mib=20000, baseline_mib=4000, baseline_free_mib=15000)

    assert roomy.budget_mib == 20000
    assert cramped.budget_mib == 15000
    assert roomy.verdict == 'PASS'
    assert cramped.verdict == 'FAIL'


def test_evaluate_budget_does_not_re_add_the_safety_margin():
    """The margin is applied once, in the share -- not again in the verdict.

    Double-counting it would fail an arm that fits its allocation exactly, which
    is the knife edge esc-3713-6 explicitly ruled out.
    """
    verdict = _budget(used_mib=7362 + 16761, baseline_mib=7362,
                      baseline_free_mib=16761)

    assert verdict.arm_footprint_mib == 16761
    assert verdict.verdict == 'PASS'
    assert verdict.headroom_gib == pytest.approx(0.0, abs=0.01)


def test_evaluate_budget_carries_the_raw_numbers_it_was_computed_from():
    """The artifact's internal-consistency gate re-derives the verdict from
    these, so a verdict that dropped them could not be checked at all."""
    verdict = _budget(used_mib=14000)

    assert verdict.used_mib == 14000
    assert verdict.total_mib == 24576
    assert verdict.baseline_mib == _BASELINE_MIB
    assert verdict.used_mib - verdict.baseline_mib == verdict.arm_footprint_mib


@pytest.mark.parametrize(
    ('used_mib', 'total_mib', 'baseline_mib', 'baseline_free_mib'),
    [
        (-1, 24576, 7362, 16761),        # negative usage
        (7362, 0, 7362, 16761),          # no card
        (30000, 24576, 7362, 16761),     # used beyond the card
        (14000, 24576, 0, 16761),        # zero baseline: the probe was dead
        (14000, 24576, -5, 16761),       # negative baseline
        (14000, 24576, 15000, 16761),    # baseline above used: not the same run
        (14000, 24576, 7362, 0),         # no free VRAM at baseline
        (14000, 24576, 7362, 30000),     # free beyond the card
    ],
)
def test_evaluate_budget_raises_on_an_incoherent_reading(
    used_mib, total_mib, baseline_mib, baseline_free_mib,
):
    with pytest.raises(lms_vram.VramProbeError):
        lms_vram.evaluate_budget(
            used_mib=used_mib,
            total_mib=total_mib,
            baseline_mib=baseline_mib,
            baseline_free_mib=baseline_free_mib,
        )


# ---------------------------------------------------------------------------
# (d) arm_fits — the measured deviation from PRD D10, as behaviour
# ---------------------------------------------------------------------------


def test_arm_fits_is_false_for_the_moe_arm_against_the_measured_budget():
    """PRD line 127 budgets the MoE stretch arm at ~17 GiB against a nominal
    19-20 GiB.  Measured free VRAM here is 16.4 GiB, so it does not fit -- and
    the tooling must say so BEFORE a launch, not after an OOM."""
    moe = _arm(arm_id='moe-stretch', stack='llamacpp',
               structured_output_mode='json_object', est_vram_gib=17.0)

    assert lms_vram.arm_fits(moe, free_gib=16.4) is False


def test_arm_fits_is_true_for_a_dense_arm_against_the_measured_budget():
    dense = _arm(arm_id='qwen3.5-9b', est_vram_gib=6.0)

    assert lms_vram.arm_fits(dense, free_gib=16.4) is True


def test_arm_fits_would_have_been_true_under_the_nominal_prd_budget():
    """The same MoE arm fits the PRD's nominal ceiling.  Pinning both results
    keeps the deviation legible: this is a budget disagreement, not a bug."""
    moe = _arm(arm_id='moe-stretch', stack='llamacpp',
               structured_output_mode='json_object', est_vram_gib=17.0)

    assert lms_vram.arm_fits(moe, free_gib=19.5) is True


def test_arm_fits_reserves_a_safety_margin_so_an_exact_fit_is_refused():
    """An arm sized to the last byte of free VRAM OOMs on the first allocation
    spike; the margin is what makes "fits" mean "runs"."""
    exact = _arm(est_vram_gib=16.4)

    assert lms_vram.arm_fits(exact, free_gib=16.4) is False
    assert lms_vram.arm_fits(exact, free_gib=16.4 + lms_vram.SAFETY_MARGIN_GIB) is True


def test_arm_fits_explains_the_refusal():
    moe = _arm(arm_id='moe-stretch', stack='llamacpp',
               structured_output_mode='json_object', est_vram_gib=17.0)

    reason = lms_vram.arm_fit_reason(moe, free_gib=16.4)

    assert 'moe-stretch' in reason
    assert '17' in reason
    assert '16.4' in reason


def test_arm_fit_reason_is_empty_when_the_arm_fits():
    assert lms_vram.arm_fit_reason(_arm(est_vram_gib=6.0), free_gib=16.4) == ''


# ---------------------------------------------------------------------------
# (e) the per-arm baseline — measured LIVE, immediately before the arm starts
# ---------------------------------------------------------------------------
#
# esc-3713-6 made this binding: the baseline MUST be an nvidia-smi reading taken
# just before THAT arm started, never the frozen MEASURED_BASELINE_GIB.  A frozen
# baseline misattributes desktop drift to the arm, and it does so in the
# fabrication-relevant direction -- a desktop that shrank between the constant's
# capture and the run hands the arm a discount it did not earn.
#
# Recording it at `lms_ctl start` rather than accepting it as a healthcheck flag
# is the point: the number is produced by the start event itself, so it cannot be
# typed in after the fact to make a report fit.


def _reading(used_mib=7310, free_mib=16813, total_mib=24576):
    return lms_vram.GpuReading(
        total_mib=total_mib, used_mib=used_mib, free_mib=free_mib,
    )


def test_baseline_round_trips_through_the_recorded_file(tmp_path, monkeypatch):
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))

    lms_vram.record_baseline('qwen3.5-9b', _reading())
    restored = lms_vram.read_baseline('qwen3.5-9b')

    assert restored.used_mib == 7310
    assert restored.free_mib == 16813
    assert restored.total_mib == 24576


def test_baseline_is_written_per_arm_and_does_not_collide(tmp_path, monkeypatch):
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))

    lms_vram.record_baseline('qwen3.5-9b', _reading(used_mib=7310))
    lms_vram.record_baseline('phi-4-14b', _reading(used_mib=7900, free_mib=16223))

    assert lms_vram.read_baseline('qwen3.5-9b').used_mib == 7310
    assert lms_vram.read_baseline('phi-4-14b').used_mib == 7900


def test_reading_a_missing_baseline_raises_rather_than_defaulting(
    tmp_path, monkeypatch,
):
    """No baseline means nobody measured the card before this arm started.

    Defaulting to MEASURED_BASELINE_GIB here would silently reintroduce exactly
    the frozen number esc-3713-6 ruled out, and the artifact would look
    identical either way.
    """
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))

    with pytest.raises(lms_vram.VramProbeError, match='no baseline'):
        lms_vram.read_baseline('qwen3.5-9b')


def test_a_corrupt_baseline_file_raises(tmp_path, monkeypatch):
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))
    lms_vram.baseline_path('qwen3.5-9b').write_text('not json at all')

    with pytest.raises(lms_vram.VramProbeError):
        lms_vram.read_baseline('qwen3.5-9b')


def test_the_recorded_baseline_is_stamped_with_an_aware_utc_time(
    tmp_path, monkeypatch,
):
    """A baseline with no time on it cannot be told apart from last week's."""
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))

    lms_vram.record_baseline('qwen3.5-9b', _reading())
    payload = json.loads(lms_vram.baseline_path('qwen3.5-9b').read_text())

    stamped = datetime.fromisoformat(payload['measured_at'])
    assert stamped.tzinfo is not None


def test_read_baselines_takes_the_most_conservative_of_several(
    tmp_path, monkeypatch,
):
    """Probing several arms at once has several baselines; the LOWEST prior
    usage attributes the MOST memory to the arms, which is the reading that
    cannot flatter them."""
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))
    lms_vram.record_baseline('a', _reading(used_mib=7310, free_mib=16813))
    lms_vram.record_baseline('b', _reading(used_mib=9000, free_mib=15123))

    chosen = lms_vram.read_baselines(['a', 'b'])

    assert chosen.used_mib == 7310
    assert chosen.free_mib == 16813


def test_read_baselines_raises_when_asked_for_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))

    with pytest.raises(lms_vram.VramProbeError):
        lms_vram.read_baselines([])


def test_clearing_a_baseline_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))
    lms_vram.record_baseline('a', _reading())

    lms_vram.clear_baseline('a')
    lms_vram.clear_baseline('a')

    assert not lms_vram.baseline_path('a').exists()
