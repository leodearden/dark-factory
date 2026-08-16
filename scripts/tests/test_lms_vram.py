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
from datetime import UTC, datetime

import lms_manifest
import lms_vram
import pytest
from pydantic import ValidationError

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

    lms_vram.record_baseline('qwen3.5-9b', _reading(), consumers=[])
    restored = lms_vram.read_baseline_record('qwen3.5-9b').reading

    assert restored.used_mib == 7310
    assert restored.free_mib == 16813
    assert restored.total_mib == 24576


def test_baseline_is_written_per_arm_and_does_not_collide(tmp_path, monkeypatch):
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))

    lms_vram.record_baseline('qwen3.5-9b', _reading(used_mib=7310), consumers=[])
    lms_vram.record_baseline(
        'phi-4-14b', _reading(used_mib=7900, free_mib=16223), consumers=[],
    )

    assert lms_vram.read_baseline_record('qwen3.5-9b').reading.used_mib == 7310
    assert lms_vram.read_baseline_record('phi-4-14b').reading.used_mib == 7900


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
        lms_vram.read_baseline_record('qwen3.5-9b')


def test_a_corrupt_baseline_file_raises(tmp_path, monkeypatch):
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))
    lms_vram.baseline_path('qwen3.5-9b').write_text('not json at all')

    with pytest.raises(lms_vram.VramProbeError):
        lms_vram.read_baseline_record('qwen3.5-9b')


def test_the_recorded_baseline_is_stamped_with_an_aware_utc_time(
    tmp_path, monkeypatch,
):
    """A baseline with no time on it cannot be told apart from last week's."""
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))

    lms_vram.record_baseline('qwen3.5-9b', _reading(), consumers=[])
    payload = json.loads(lms_vram.baseline_path('qwen3.5-9b').read_text())

    stamped = datetime.fromisoformat(payload['measured_at'])
    assert stamped.tzinfo is not None


def test_read_baseline_records_takes_the_most_conservative_of_several(
    tmp_path, monkeypatch,
):
    """Probing several arms at once has several baselines; the LOWEST prior
    usage attributes the MOST memory to the arms, which is the reading that
    cannot flatter them."""
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))
    lms_vram.record_baseline('a', _reading(used_mib=7310, free_mib=16813), consumers=[])
    lms_vram.record_baseline('b', _reading(used_mib=9000, free_mib=15123), consumers=[])

    chosen = lms_vram.read_baseline_records(['a', 'b']).reading

    assert chosen.used_mib == 7310
    assert chosen.free_mib == 16813


def test_read_baseline_records_raises_when_asked_for_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))

    with pytest.raises(lms_vram.VramProbeError):
        lms_vram.read_baseline_records([])


def test_clearing_a_baseline_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))
    lms_vram.record_baseline('a', _reading(), consumers=[])

    lms_vram.clear_baseline('a')
    lms_vram.clear_baseline('a')

    assert not lms_vram.baseline_path('a').exists()


# ---------------------------------------------------------------------------
# (f) the non-arm GPU consumer inventory (task 3755)
# ---------------------------------------------------------------------------
#
# `--query-gpu=memory.used` says HOW MUCH of the card is held; it does not say
# BY WHOM.  On 2026-08-06 ollama (`/usr/local/lib/ollama/llama-server`, pid
# 905936) held 10314 MiB with qwen3:14b resident on keep_alive while a slate ran
# against the same card, and every budget verdict produced that day silently
# charged the arm for -- or credited it with -- memory it never touched.
#
# `--query-compute-apps=pid,process_name,used_memory` is what names the holders.
# It is an INVENTORY, not an accounting: it does not list the KDE/X11 graphics
# contexts at all, which is exactly why this host's operating budget is ~16.4
# GiB rather than 19.5.
#
# Verbatim `nvidia-smi --query-compute-apps=pid,process_name,used_memory
# --format=csv,noheader,nounits` output measured on this host:
#
#     7575, python, 4050                        <- whisper-writer (PRD D10)
#     905936, /usr/local/lib/ollama/llama-server, 10314

MEASURED_WHISPER_ROW = '7575, python, 4050'
MEASURED_OLLAMA_ROW = '905936, /usr/local/lib/ollama/llama-server, 10314'


def test_parse_compute_apps_on_the_measured_whisper_writer_row():
    consumers = lms_vram.parse_nvidia_smi_compute_apps_csv(
        MEASURED_WHISPER_ROW + '\n'
    )

    assert len(consumers) == 1
    assert consumers[0].pid == 7575
    assert consumers[0].process_name == 'python'
    assert consumers[0].used_mib == 4050


def test_parse_compute_apps_on_the_measured_two_consumer_output():
    """The exact reading that motivated this guard: whisper-writer AND ollama."""
    consumers = lms_vram.parse_nvidia_smi_compute_apps_csv(
        f'{MEASURED_WHISPER_ROW}\n{MEASURED_OLLAMA_ROW}\n'
    )

    assert [c.pid for c in consumers] == [7575, 905936]
    assert [c.process_name for c in consumers] == [
        'python', '/usr/local/lib/ollama/llama-server',
    ]
    assert [c.used_mib for c in consumers] == [4050, 10314]


def test_a_consumer_exposes_the_gib_conversion_in_the_module_convention():
    consumer = lms_vram.parse_nvidia_smi_compute_apps_csv(
        MEASURED_OLLAMA_ROW + '\n'
    )[0]

    assert consumer.used_gib == pytest.approx(10.07, abs=0.01)


@pytest.mark.parametrize(('label', 'text'), [('empty', ''), ('whitespace only', '  \n\n')])
def test_parse_compute_apps_reads_no_output_as_a_legitimately_empty_card(label, text):
    """Unlike the memory row, ZERO compute apps is a real state.

    nvidia-smi prints nothing at all when no process holds the card, and that is
    the reading a clean baseline is *supposed* to produce.  Raising here would
    make the healthiest possible slate the one this tooling refuses to record.
    """
    assert lms_vram.parse_nvidia_smi_compute_apps_csv(text) == []


@pytest.mark.parametrize(
    ('label', 'text'),
    [
        # The caller dropped `noheader`; row 1 is not a consumer.
        ('header row present',
         'pid, process_name, used_gpu_memory [MiB]\n7575, python, 4050\n'),
        ('non-integer pid', 'abc, python, 4050\n'),
        ('non-integer used_memory', '7575, python, lots\n'),
        # nvidia-smi emits [N/A] for a field the driver cannot report.
        ('[N/A] used_memory', '7575, python, [N/A]\n'),
        ('[N/A] pid', '[N/A], python, 4050\n'),
        # The caller dropped `nounits`; "4050 MiB" must not become 4050 by luck.
        ('units left in', '7575, python, 4050 MiB\n'),
        ('too few fields', '7575, python\n'),
        ('one field', '7575\n'),
        # A blank field is NOT the empty-output case: something was there and we
        # cannot say what.
        ('blank process_name', '7575, , 4050\n'),
        ('blank pid', ', python, 4050\n'),
        ('blank used_memory', '7575, python, \n'),
        ('negative used_memory', '7575, python, -1\n'),
        # One bad row poisons the whole inventory: a partial list read as
        # complete is what lets an unlisted holder pass for absent.
        ('one good row and one bad', '7575, python, 4050\nx, python, 99\n'),
    ],
)
def test_parse_compute_apps_raises_rather_than_coercing(label, text):
    with pytest.raises(lms_vram.VramProbeError):
        lms_vram.parse_nvidia_smi_compute_apps_csv(text)


def test_parse_compute_apps_error_quotes_the_offending_row():
    with pytest.raises(lms_vram.VramProbeError) as excinfo:
        lms_vram.parse_nvidia_smi_compute_apps_csv('7575, python, [N/A]\n')

    assert '[N/A]' in str(excinfo.value)


def test_parse_compute_apps_keeps_a_comma_inside_a_process_name():
    """Split on the FIRST and LAST comma, not on every comma.

    A process path may legitimately contain a comma.  Splitting naively would
    make such a row look like a field-count error and raise -- turning a real,
    possibly large, GPU holder into a refusal to record anything at all.
    """
    consumers = lms_vram.parse_nvidia_smi_compute_apps_csv(
        '4242, /opt/weird,path/llama-server, 2048\n'
    )

    assert len(consumers) == 1
    assert consumers[0].pid == 4242
    assert consumers[0].process_name == '/opt/weird,path/llama-server'
    assert consumers[0].used_mib == 2048


def test_a_consumer_is_frozen():
    consumer = lms_vram.GpuConsumer(pid=7575, process_name='python', used_mib=4050)

    with pytest.raises(ValidationError):
        consumer.used_mib = 0


def test_probe_gpu_consumers_shells_the_expected_query_through_the_injected_runner():
    recorded = []

    def runner(argv):
        recorded.append(argv)
        return f'{MEASURED_WHISPER_ROW}\n{MEASURED_OLLAMA_ROW}\n'

    consumers = lms_vram.probe_gpu_consumers(runner)

    assert [c.pid for c in consumers] == [7575, 905936]
    assert recorded == [[
        'nvidia-smi',
        '--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader,nounits',
    ]]


def test_probe_gpu_consumers_wraps_a_runner_failure_in_the_typed_error():
    def runner(argv):
        raise FileNotFoundError('nvidia-smi')

    with pytest.raises(lms_vram.VramProbeError) as excinfo:
        lms_vram.probe_gpu_consumers(runner)

    assert 'nvidia-smi' in str(excinfo.value)


# ---------------------------------------------------------------------------
# (g) the baseline pollution predicate — a positive ALLOWLIST
# ---------------------------------------------------------------------------
#
# `lms_ctl.start` records the baseline BEFORE `systemctl start`, so at that
# instant nothing of ours is on the card.  Anything sizeable is therefore a
# surprise, and a positive allowlist is both safe and maximally strict: it
# catches ollama AND anything nobody anticipated.
#
# The probe-time guard (section (i)) cannot use the same rule, because there the
# arm's own container is legitimately a NEW consumer and
# `--query-compute-apps=pid,process_name,used_memory` cannot tell a
# containerised vLLM `python` from any other `python`.  That asymmetry is
# deliberate, not an oversight.


def _consumer(pid=7575, process_name='python', used_mib=4050):
    return lms_vram.GpuConsumer(
        pid=pid, process_name=process_name, used_mib=used_mib,
    )


#: whisper-writer as measured on this host: resident since 2026-07-03, and
#: required resident by PRD D10.
WHISPER = _consumer()

#: ollama as measured 2026-08-06, qwen3:14b resident on keep_alive.  The
#: reading that motivated this whole guard.
OLLAMA = _consumer(
    pid=905936, process_name='/usr/local/lib/ollama/llama-server', used_mib=10314,
)


def test_the_pollution_floor_is_the_tasks_one_gib():
    """Below this a consumer is noise, not a finding."""
    assert lms_vram.POLLUTION_FLOOR_MIB == 1024


def test_the_expected_consumer_allowlist_is_whisper_writer_alone():
    (whisper,) = lms_vram.EXPECTED_CONSUMERS

    assert 'whisper' in whisper.label.lower()
    # Derived, not guessed: every whisper-writer reading recorded on this host
    # is 4050 MiB. 6144 sits far above that (reload drift) and far below the
    # 10314 MiB ollama holding that motivated the guard.
    assert whisper.ceiling_mib == 6144
    assert whisper.ceiling_mib > 4050
    assert whisper.ceiling_mib < 10314


@pytest.mark.parametrize(
    'process_name',
    [
        'python',
        'python3',
        '/usr/bin/python3',
        '/usr/bin/python',
        # Versioned interpreters, which nvidia-smi reports on plenty of hosts.
        # Omitting them made a routine whisper-writer restart under a venv or
        # after a distro interpreter bump refuse an arm on a clean card.
        'python3.11',
        '/usr/bin/python3.12',
        '/home/leo/.venvs/ww/bin/python3.13',
    ],
)
def test_the_whisper_writer_pattern_matches_a_bare_or_pathed_python(process_name):
    (whisper,) = lms_vram.EXPECTED_CONSUMERS

    assert whisper.matches(process_name)


@pytest.mark.parametrize(
    'process_name',
    [
        '/usr/local/lib/ollama/llama-server',
        'llama-server',
        'pythonish',
        '/opt/not-python',
        # The version suffix the pattern admits is `3`-only and must be a
        # complete version: widening it for `python3.12` must not admit a
        # different interpreter, nor a name that merely starts like one.
        'python2',
        'python3x',
    ],
)
def test_the_whisper_writer_pattern_does_not_match_anything_else(process_name):
    (whisper,) = lms_vram.EXPECTED_CONSUMERS

    assert not whisper.matches(process_name)


def test_whisper_writer_alone_at_its_measured_size_is_a_clean_baseline():
    assert lms_vram.unexpected_baseline_consumers([WHISPER]) == []


def test_an_empty_card_is_a_clean_baseline():
    assert lms_vram.unexpected_baseline_consumers([]) == []


def test_the_measured_ollama_holding_pollutes_the_baseline():
    """The exact 2026-08-06 reading: whisper-writer plus 10314 MiB of ollama."""
    offenders = lms_vram.unexpected_baseline_consumers([WHISPER, OLLAMA])

    assert offenders == [OLLAMA]


def test_a_stray_under_the_floor_is_noise_and_not_pollution():
    """A few hundred MiB cannot move a budget verdict, and flagging it would
    make the guard cry wolf until an operator learns to ignore it."""
    stray = _consumer(pid=4242, process_name='/usr/bin/glxgears', used_mib=512)

    assert lms_vram.unexpected_baseline_consumers([WHISPER, stray]) == []


def test_a_stranger_over_the_floor_pollutes_even_if_nobody_anticipated_it():
    """The allowlist is positive precisely so an UNANTICIPATED holder is caught.
    Nothing of ours is on the card at baseline, so there is no false-positive
    direction to protect."""
    stranger = _consumer(pid=4242, process_name='/opt/somebody/inference', used_mib=2048)

    assert lms_vram.unexpected_baseline_consumers([stranger]) == [stranger]


def test_an_oversized_python_cannot_ride_the_whisper_writer_allowlist():
    """Otherwise any containerised vLLM left over from a previous slate could
    sit in a 'clean' baseline under the name `python` and be subtracted from
    the next arm's footprint."""
    fat_python = _consumer(pid=4242, process_name='python', used_mib=9000)

    assert lms_vram.unexpected_baseline_consumers([fat_python]) == [fat_python]


def test_the_whisper_writer_ceiling_admits_a_reading_at_the_ceiling_exactly():
    at_ceiling = _consumer(used_mib=6144)
    over_ceiling = _consumer(used_mib=6145)

    assert lms_vram.unexpected_baseline_consumers([at_ceiling]) == []
    assert lms_vram.unexpected_baseline_consumers([over_ceiling]) == [over_ceiling]


# --- the ONE documented exception: a knowingly co-resident arm --------------
#
# `lms_ctl start --no-exclusive` leaves another arm's vLLM container on the
# card on purpose.  nvidia-smi reports it as `python` far over whisper-writer's
# ceiling -- the same undecidability `classify_pollution` documents -- so the
# strict rule would call the operator's own arm an intruder.


def test_a_coresident_arm_is_excused_only_when_an_arm_was_declared():
    """Same inventory, opposite verdicts: the excuse comes from the caller
    saying an arm is knowingly resident, never from the reading alone."""
    arm_container = _consumer(pid=41001, process_name='python', used_mib=9000)

    assert lms_vram.unexpected_baseline_consumers([arm_container]) == [arm_container]
    assert lms_vram.unexpected_baseline_consumers(
        [arm_container], coresident_arms=['phi-4-14b'],
    ) == []


def test_a_declared_coresident_arm_does_not_excuse_ollama():
    """The relaxation narrows to the still-decidable negative rule; the holder
    the whole guard exists for is refused either way."""
    offenders = lms_vram.unexpected_baseline_consumers(
        [WHISPER, OLLAMA], coresident_arms=['phi-4-14b'],
    )

    assert offenders == [OLLAMA]


def test_the_coresident_relaxation_still_ignores_sub_floor_noise():
    stray = _consumer(pid=4242, process_name='/usr/bin/glxgears', used_mib=512)

    assert lms_vram.unexpected_baseline_consumers(
        [stray], coresident_arms=['phi-4-14b'],
    ) == []


def test_the_excusing_arms_are_persisted_so_every_reader_applies_one_rule(
    tmp_path, monkeypatch,
):
    """A relaxation held only in the writer's memory would make the identical
    file mean two different things depending on who read it."""
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))
    arm_container = _consumer(pid=41001, process_name='python', used_mib=9000)

    lms_vram.record_baseline(
        'qwen3.5-9b', _reading(), consumers=[WHISPER, arm_container],
        coresident_arms=['phi-4-14b'],
    )
    record = lms_vram.read_baseline_record('qwen3.5-9b')

    assert record.coresident_arms == ['phi-4-14b']
    assert arm_container in record.consumers


def test_a_baseline_written_before_this_key_existed_excuses_nobody(
    tmp_path, monkeypatch,
):
    """Absent reads as the STRICT rule.  Unlike a missing `consumers` key it
    cannot flatter anything, so it raises nothing -- it can only make the guard
    stricter than the writer intended."""
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))
    path = tmp_path / 'qwen3.5-9b.json'
    path.write_text(json.dumps({
        'arm_id': 'qwen3.5-9b',
        'measured_at': '2026-08-06T12:00:00+00:00',
        'total_mib': 24576, 'used_mib': 7310, 'free_mib': 16813,
        'consumers': [WHISPER.model_dump(mode='json')],
    }) + '\n')

    assert lms_vram.read_baseline_record('qwen3.5-9b').coresident_arms == []


def test_recording_is_still_refused_when_only_ollama_is_the_offender(
    tmp_path, monkeypatch,
):
    """The data-integrity backstop keeps its teeth under the relaxation."""
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))

    with pytest.raises(lms_vram.PollutedBaselineError):
        lms_vram.record_baseline(
            'qwen3.5-9b', _reading(), consumers=[WHISPER, OLLAMA],
            coresident_arms=['phi-4-14b'],
        )

    assert not lms_vram.baseline_path('qwen3.5-9b').exists()


def test_polluted_baseline_error_is_a_vram_probe_error():
    """So every existing `except VramProbeError` handler still catches it, and
    a polluted baseline can never escape a caller that was already careful."""
    assert issubclass(lms_vram.PollutedBaselineError, lms_vram.VramProbeError)


def test_the_polluted_baseline_error_names_the_offending_consumer():
    """An operator reading only stderr must know WHICH process to deal with."""
    message = str(lms_vram.PollutedBaselineError([OLLAMA]))

    assert '905936' in message
    assert '/usr/local/lib/ollama/llama-server' in message
    assert '10314' in message


def test_the_polluted_baseline_error_carries_the_offenders_for_a_caller():
    error = lms_vram.PollutedBaselineError([OLLAMA])

    assert error.consumers == [OLLAMA]


# ---------------------------------------------------------------------------
# (h) the baseline store carries the inventory, and refuses a polluted one
# ---------------------------------------------------------------------------
#
# "Do not record a polluted baseline" is enforced AT THE WRITE, not at every
# read.  Enforcing at the write means no polluted file can exist for any later
# reader to pick up, and it matches the refusal already sited in `lms_ctl.start`
# for a failed pre-flight -- "a refused arm leaves no baseline behind for
# another arm's report to pick up".
#
# `consumers` is a REQUIRED keyword, not an optional one.  An optional inventory
# would let a caller record a baseline carrying no evidence about who else held
# the card, and downstream that absence reads as "clean".


def test_the_recorded_baseline_carries_the_consumer_inventory(tmp_path, monkeypatch):
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))

    lms_vram.record_baseline('qwen3.5-9b', _reading(), consumers=[WHISPER])
    record = lms_vram.read_baseline_record('qwen3.5-9b')

    assert record.reading.used_mib == 7310
    assert record.reading.free_mib == 16813
    assert record.reading.total_mib == 24576
    assert record.consumers == [WHISPER]


def test_an_empty_inventory_round_trips_as_an_empty_inventory(tmp_path, monkeypatch):
    """A card nobody holds is a real, and ideal, baseline state.  It must not
    be confused with "nobody looked"."""
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))

    lms_vram.record_baseline('qwen3.5-9b', _reading(), consumers=[])

    assert lms_vram.read_baseline_record('qwen3.5-9b').consumers == []


def test_the_recorded_record_is_stamped_with_an_aware_utc_time(tmp_path, monkeypatch):
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))

    lms_vram.record_baseline('qwen3.5-9b', _reading(), consumers=[WHISPER])
    record = lms_vram.read_baseline_record('qwen3.5-9b')

    assert record.measured_at.tzinfo is not None


def test_recording_a_polluted_baseline_is_refused_and_writes_nothing(
    tmp_path, monkeypatch,
):
    """The measured 2026-08-06 hazard, refused at the write.

    Not "written and flagged": a file on disk would be picked up by the next
    healthcheck as the number to subtract from an arm's footprint, and a flag
    inside it only helps a reader who thought to look.
    """
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))

    with pytest.raises(lms_vram.PollutedBaselineError) as excinfo:
        lms_vram.record_baseline(
            'qwen3.5-9b', _reading(), consumers=[WHISPER, OLLAMA],
        )

    assert '/usr/local/lib/ollama/llama-server' in str(excinfo.value)
    assert not lms_vram.baseline_path('qwen3.5-9b').exists()


def test_a_refused_baseline_does_not_clobber_a_previously_clean_one(
    tmp_path, monkeypatch,
):
    """The refusal is raised BEFORE any write, so a truncate-then-fail cannot
    destroy a good measurement and leave a half-file behind."""
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))
    lms_vram.record_baseline('qwen3.5-9b', _reading(used_mib=7310), consumers=[WHISPER])

    with pytest.raises(lms_vram.PollutedBaselineError):
        lms_vram.record_baseline(
            'qwen3.5-9b', _reading(used_mib=17000, free_mib=7123),
            consumers=[WHISPER, OLLAMA],
        )

    survivor = lms_vram.read_baseline_record('qwen3.5-9b')
    assert survivor.reading.used_mib == 7310
    assert survivor.consumers == [WHISPER]


def test_a_baseline_payload_with_no_consumers_key_raises_rather_than_reading_clean(
    tmp_path, monkeypatch,
):
    """A baseline written before this guard existed carries no evidence about
    who held the card.  Defaulting it to `[]` would make the one baseline we
    know NOTHING about look like the cleanest possible reading.

    Baselines live in $XDG_RUNTIME_DIR and are per-boot, so the staleness window
    is bounded and the fix is a single command -- which the message must name.
    """
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))
    path = lms_vram.baseline_path('qwen3.5-9b')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        'arm_id': 'qwen3.5-9b',
        'measured_at': '2026-08-06T12:00:00+00:00',
        'total_mib': 24576,
        'used_mib': 7310,
        'free_mib': 16813,
    }))

    with pytest.raises(lms_vram.VramProbeError, match='lms_ctl start'):
        lms_vram.read_baseline_record('qwen3.5-9b')


def test_reading_a_missing_baseline_record_raises(tmp_path, monkeypatch):
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))

    with pytest.raises(lms_vram.VramProbeError, match='no baseline'):
        lms_vram.read_baseline_record('qwen3.5-9b')


def test_read_baseline_records_returns_the_chosen_records_own_consumers(
    tmp_path, monkeypatch,
):
    """The reading and the inventory can never come from different moments.

    Picking the lowest-used READING from one arm and the inventory from another
    would describe a card that never existed -- and the mismatch would be
    invisible in the artifact, which is the worst property a record can have.
    """
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))
    stray = _consumer(pid=331, process_name='/usr/bin/glxgears', used_mib=512)
    lms_vram.record_baseline(
        'a', _reading(used_mib=7310, free_mib=16813), consumers=[WHISPER],
    )
    lms_vram.record_baseline(
        'b', _reading(used_mib=9000, free_mib=15123), consumers=[WHISPER, stray],
    )

    chosen = lms_vram.read_baseline_records(['a', 'b'])

    assert chosen.reading.used_mib == 7310
    assert chosen.reading.free_mib == 16813
    assert chosen.consumers == [WHISPER]


def test_read_baseline_records_raises_when_asked_for_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path))

    with pytest.raises(lms_vram.VramProbeError):
        lms_vram.read_baseline_records([])


def test_a_baseline_record_is_frozen():
    record = lms_vram.GpuBaseline(
        reading=_reading(),
        consumers=[WHISPER],
        measured_at=datetime.now(UTC),
    )

    with pytest.raises(ValidationError):
        record.consumers = []


# ---------------------------------------------------------------------------
# (i) probe-time pollution — a KNOWN-FOREIGN list plus baseline drift
# ---------------------------------------------------------------------------
#
# The baseline guard's positive allowlist CANNOT be reused here.  At probe time
# the arm's own container is legitimately a NEW consumer, and
# `--query-compute-apps=pid,process_name,used_memory` cannot tell a
# containerised vLLM `python` from any other `python` -- arms run as docker
# containers and nvidia-smi reports HOST pids, so the arm appears as an
# ordinary compute app.  A negative allowlist here would flag every healthy run.
#
# What IS decidable:
#   (i)  a newcomer matching a known-foreign process path, and
#   (ii) any change in a consumer that was already present at baseline -- which
#        is by construction non-arm, because the arm was not running then.
#
# Both DIRECTIONS of (ii) are pollution.  Growth over-charges the arm; a shrink
# or a vanish UNDER-charges it, and the flattering direction is precisely the
# one a fabricated artifact wants.


ARM_CONTAINER = _consumer(pid=910001, process_name='python', used_mib=12800)


def test_a_new_unrecognised_consumer_alone_is_clean_because_it_may_be_the_arm():
    """The arm itself is a newcomer at probe time and looks like any other
    `python`.  Flagging it would make every healthy run POLLUTED."""
    state, reason = lms_vram.classify_pollution([WHISPER], [WHISPER, ARM_CONTAINER])

    assert state is lms_vram.PollutionState.CLEAN
    assert reason == ''


def test_an_unchanged_baseline_with_no_newcomer_at_all_is_clean():
    state, reason = lms_vram.classify_pollution([WHISPER], [WHISPER])

    assert state is lms_vram.PollutionState.CLEAN
    assert reason == ''


def test_an_ollama_newcomer_over_the_floor_pollutes_the_probe():
    """The exact measured scenario: whisper-writer at 4050 and a 10314 MiB
    `/usr/local/lib/ollama/llama-server` that arrived while the arm ran."""
    state, reason = lms_vram.classify_pollution(
        [WHISPER], [WHISPER, ARM_CONTAINER, OLLAMA],
    )

    assert state is lms_vram.PollutionState.POLLUTED
    assert '905936' in reason
    assert '/usr/local/lib/ollama/llama-server' in reason
    assert '10314' in reason


def test_a_foreign_newcomer_under_the_floor_is_noise():
    tiny_ollama = _consumer(
        pid=905936, process_name='/usr/local/lib/ollama/llama-server', used_mib=64,
    )

    state, _ = lms_vram.classify_pollution([WHISPER], [WHISPER, tiny_ollama])

    assert state is lms_vram.PollutionState.CLEAN


def test_a_baseline_consumer_that_grew_past_the_floor_pollutes():
    """It was there before the arm started, so it is not the arm.  Its growth
    is charged to the arm by `used - baseline`."""
    fatter_whisper = _consumer(pid=7575, process_name='python', used_mib=6000)

    state, reason = lms_vram.classify_pollution([WHISPER], [fatter_whisper])

    assert state is lms_vram.PollutionState.POLLUTED
    assert '7575' in reason
    assert '4050' in reason and '6000' in reason


def test_a_baseline_consumer_that_shrank_pollutes_and_the_reason_says_why():
    """The FLATTERING direction, and the one a fabricated artifact wants.

    A consumer that shrank leaves `used - baseline` smaller than the arm truly
    took, so the arm gets a discount it did not earn.  Silence here would be a
    verdict that passes for the wrong reason.
    """
    thinner_whisper = _consumer(pid=7575, process_name='python', used_mib=1000)

    state, reason = lms_vram.classify_pollution([WHISPER], [thinner_whisper])

    assert state is lms_vram.PollutionState.POLLUTED
    assert 'under-charge' in reason.lower() or 'under-charg' in reason.lower()


def test_a_baseline_consumer_that_vanished_pollutes():
    state, reason = lms_vram.classify_pollution([WHISPER], [ARM_CONTAINER])

    assert state is lms_vram.PollutionState.POLLUTED
    assert '7575' in reason


def test_sub_floor_jitter_in_a_baseline_consumer_stays_clean():
    """whisper-writer's own allocation wobbles; a few hundred MiB cannot move a
    verdict, and flagging it would make the guard cry wolf."""
    jittered = _consumer(pid=7575, process_name='python', used_mib=4050 + 512)

    state, reason = lms_vram.classify_pollution([WHISPER], [jittered, ARM_CONTAINER])

    assert state is lms_vram.PollutionState.CLEAN
    assert reason == ''


def test_a_vanished_baseline_consumer_under_the_floor_is_noise():
    stray = _consumer(pid=331, process_name='/usr/bin/glxgears', used_mib=512)

    state, _ = lms_vram.classify_pollution([WHISPER, stray], [WHISPER])

    assert state is lms_vram.PollutionState.CLEAN


def test_the_reason_is_empty_exactly_when_the_state_is_clean():
    """So a consumer can branch on either and never disagree with itself."""
    cases = [
        ([WHISPER], [WHISPER, ARM_CONTAINER]),
        ([WHISPER], [WHISPER, ARM_CONTAINER, OLLAMA]),
        ([WHISPER], [ARM_CONTAINER]),
        ([], [ARM_CONTAINER]),
    ]
    for baseline, probe in cases:
        state, reason = lms_vram.classify_pollution(baseline, probe)
        assert (state is lms_vram.PollutionState.CLEAN) == (reason == '')


def test_classify_pollution_never_reports_the_unmeasured_sentinel():
    """UNMEASURED means nobody looked.  Anything that ran a classification did."""
    state, _ = lms_vram.classify_pollution([WHISPER], [WHISPER, OLLAMA])

    assert state is not lms_vram.PollutionState.UNMEASURED


def test_the_known_foreign_list_matches_ollama_and_no_containerised_arm():
    (ollama,) = lms_vram.KNOWN_FOREIGN_CONSUMERS

    assert ollama.matches('/usr/local/lib/ollama/llama-server')
    assert not ollama.matches('python')
    assert not ollama.matches('/usr/bin/python3')


# ---------------------------------------------------------------------------
# (j) the snapshot carries the inventory, so all three are ONE capture
# ---------------------------------------------------------------------------


MEASURED_IDENTITY_CSV = 'NVIDIA GeForce RTX 3090, 580.95.05\n'


def _snapshot_runner(compute_apps_csv):
    """A fake nvidia-smi answering each of the three narrow queries."""
    recorded = []

    def runner(argv):
        recorded.append(argv)
        if any(a.startswith('--query-compute-apps') for a in argv):
            return compute_apps_csv
        if any(a.startswith('--query-gpu=name') for a in argv):
            return MEASURED_IDENTITY_CSV
        return MEASURED_CSV

    return runner, recorded


def test_probe_gpu_snapshot_populates_the_consumer_inventory():
    runner, _ = _snapshot_runner(f'{MEASURED_WHISPER_ROW}\n{MEASURED_OLLAMA_ROW}\n')

    snapshot = lms_vram.probe_gpu_snapshot(runner)

    assert snapshot.reading.used_mib == 7362
    assert snapshot.identity.name == 'NVIDIA GeForce RTX 3090'
    assert [c.pid for c in snapshot.consumers] == [7575, 905936]


def test_probe_gpu_snapshot_shells_all_three_narrow_queries():
    """Three narrow queries, not one wide one: the strict memory parser -- the
    one whose failure would misreport the budget -- keeps its exact shape and
    its existing coverage."""
    runner, recorded = _snapshot_runner(MEASURED_WHISPER_ROW + '\n')

    lms_vram.probe_gpu_snapshot(runner)

    assert recorded == [
        ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
        [
            'nvidia-smi',
            '--query-gpu=memory.total,memory.used,memory.free',
            '--format=csv,noheader,nounits',
        ],
        [
            'nvidia-smi',
            '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader,nounits',
        ],
    ]


def test_a_snapshot_cannot_be_built_without_an_inventory():
    """The field is REQUIRED, so no call site can silently omit the evidence
    about who else held the card at the moment the reading was taken.  An
    optional inventory would read downstream as "nobody else was there"."""
    with pytest.raises(ValidationError):
        # The omission is the assertion: `consumers` is deliberately absent so
        # pydantic rejects the construction at runtime.  pyright sees the same
        # missing required argument statically, which is the very thing under
        # test, so the diagnostic is suppressed here rather than satisfied.
        lms_vram.GpuSnapshot(  # pyright: ignore[reportCallIssue]
            identity=lms_vram.GpuIdentity(
                name='NVIDIA GeForce RTX 3090', driver_version='580.95.05',
            ),
            reading=_reading(),
        )
