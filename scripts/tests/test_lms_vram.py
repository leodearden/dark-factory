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
import pytest

import lms_manifest
import lms_vram

# Verbatim `nvidia-smi --query-gpu=memory.total,memory.used,memory.free
# --format=csv,noheader,nounits` output captured on this host 2026-08-05.
MEASURED_CSV = '24576, 7362, 16761\n'


def _arm(**overrides):
    fields = {
        'arm_id': 'demo',
        'axis': 'llm',
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
# (c) evaluate_budget — reports BOTH figures
# ---------------------------------------------------------------------------


def test_evaluate_budget_reports_both_the_nominal_ceiling_and_the_measured_budget():
    verdict = lms_vram.evaluate_budget(used_mib=7362, total_mib=24576)

    assert verdict.nominal_ceiling_gib == pytest.approx(19.5)
    assert verdict.operating_budget_gib == pytest.approx(
        lms_vram.MEASURED_OPERATING_BUDGET_GIB
    )
    # The measured budget is materially BELOW the PRD's nominal ceiling; that
    # gap is the finding, so a report that showed only one figure would hide it.
    assert verdict.operating_budget_gib < verdict.nominal_ceiling_gib


def test_evaluate_budget_passes_when_usage_is_inside_the_budget():
    verdict = lms_vram.evaluate_budget(used_mib=7362, total_mib=24576)

    assert verdict.verdict == 'PASS'
    assert verdict.used_gib == pytest.approx(7.19, abs=0.01)
    assert verdict.headroom_gib == pytest.approx(19.5 - 7.19, abs=0.02)


def test_evaluate_budget_fails_when_usage_exceeds_the_budget():
    verdict = lms_vram.evaluate_budget(used_mib=21000, total_mib=24576)

    assert verdict.verdict == 'FAIL'
    assert verdict.headroom_gib < 0
    assert 'budget' in verdict.reason.lower()


def test_evaluate_budget_honours_an_explicit_budget_override():
    verdict = lms_vram.evaluate_budget(
        used_mib=17000, total_mib=24576, budget_gib=16.0,
    )

    assert verdict.verdict == 'FAIL'
    assert verdict.budget_gib == pytest.approx(16.0)
    # Both reference figures still travel with the verdict.
    assert verdict.nominal_ceiling_gib == pytest.approx(19.5)


def test_evaluate_budget_at_exactly_the_budget_passes():
    used_mib = int(round(19.5 * 1024))

    verdict = lms_vram.evaluate_budget(used_mib=used_mib, total_mib=24576)

    assert verdict.verdict == 'PASS'
    assert verdict.headroom_gib == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize(
    ('used_mib', 'total_mib'),
    [(-1, 24576), (7362, 0), (30000, 24576)],
)
def test_evaluate_budget_raises_on_an_incoherent_reading(used_mib, total_mib):
    with pytest.raises(lms_vram.VramProbeError):
        lms_vram.evaluate_budget(used_mib=used_mib, total_mib=total_mib)


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
