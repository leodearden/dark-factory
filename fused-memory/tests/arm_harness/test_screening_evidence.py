"""One arm's η screening evidence on disk, typed and validated (arm_harness/screening_evidence.py)."""

import json
from pathlib import Path

import pytest
from shared.memory_eval_metrics import canonical_json_text

from arm_harness._fakes import (
    EVIDENCE_STAMP,
    TAP_LISTEN_URL,
    call,
    command_record,
    health_report,
    llm_spec,
    screening_spec,
    slate_arm,
    write_arm_evidence,
)
from fused_memory.arm_harness.arm_spec import load_arm_spec
from fused_memory.arm_harness.metrics_record import IndexConfiguration, LlmMetricId
from fused_memory.arm_harness.screening_evidence import (
    LMS_CTL_EXIT_CARD_HELD,
    SCREENING_RUN_SHAPE,
    ArmCommands,
    ArmEvidencePaths,
    ScreeningEvidenceError,
    ScreeningStage,
    TapBinding,
    load_arm_commands,
    load_arm_evidence,
    load_vram_evidence,
    own_model_calls,
    write_arm_commands,
    write_screening_spec,
)
from fused_memory.arm_harness.slate import arm_endpoint

QWEN = slate_arm()


def _commands() -> ArmCommands:
    return ArmCommands(
        arm_id='qwen3.5-9b',
        tap=TapBinding(listen_url=TAP_LISTEN_URL, upstream_url='http://127.0.0.1:8410'),
        records=(
            command_record(ScreeningStage.START),
            command_record(ScreeningStage.SMOKE, exit_code=3, output_tail='FAIL positive'),
        ),
    )


def _write_health(tmp_path: Path, report: dict) -> Path:
    path = tmp_path / 'health.json'
    path.write_text(json.dumps(report))
    return path


# --- layout, records, the run shape --------------------------------------------------


def test_one_layout_names_every_evidence_file(tmp_path):
    paths = ArmEvidencePaths(root=tmp_path, arm_id='phi-4-14b')

    assert paths.spec == tmp_path / 'specs' / 'phi-4-14b.json'
    assert paths.commands == tmp_path / 'arms' / 'phi-4-14b' / 'commands.json'
    assert paths.health == tmp_path / 'arms' / 'phi-4-14b' / 'health.json'
    assert paths.smoke_calls == tmp_path / 'arms' / 'phi-4-14b' / 'smoke-calls.jsonl'
    assert paths.calls == tmp_path / 'arms' / 'phi-4-14b' / 'calls.jsonl'
    assert paths.runs == tmp_path / 'runs' / 'phi-4-14b'


def test_the_stages_are_a_fixed_vocabulary_in_sweep_order():
    assert [stage.value for stage in ScreeningStage] == [
        'release', 'start', 'wait-ready', 'smoke', 'run', 'healthcheck', 'stop', 'teardown',
    ]


def test_arm_commands_round_trip_as_canonical_json(tmp_path):
    path = tmp_path / 'arms' / 'qwen3.5-9b' / 'commands.json'
    commands = _commands()

    write_arm_commands(path, commands)

    assert load_arm_commands(path) == commands
    assert path.read_text() == canonical_json_text(commands.model_dump(mode='json'))


def test_the_screening_spec_round_trips(tmp_path):
    spec = screening_spec(QWEN)
    path = tmp_path / 'specs' / 'qwen3.5-9b.json'

    write_screening_spec(path, spec)

    assert load_arm_spec(path) == spec
    assert path.read_text() == canonical_json_text(spec.model_dump(mode='json'))


def test_the_screening_run_shape_is_prereg_section_8s():
    assert SCREENING_RUN_SHAPE.limit == 20
    assert SCREENING_RUN_SHAPE.concurrency == 3
    assert SCREENING_RUN_SHAPE.index_configuration is IndexConfiguration.WITH_INDICES


# --- α's VRAM reading ----------------------------------------------------------------


def test_reads_the_vram_block_and_the_one_arm_row(tmp_path):
    path = _write_health(tmp_path, health_report('qwen3.5-9b', top_level_entities_named=3))

    evidence = load_vram_evidence(path, 'qwen3.5-9b')

    assert evidence.row.arm_id == 'qwen3.5-9b'
    assert evidence.row.reasoning == 'on'
    assert evidence.row.verdict == 'PASS'
    assert evidence.row.top_level_entities_named == 3
    assert evidence.vram.verdict == 'PASS'
    assert evidence.vram.arm_footprint_mib == 16637
    assert evidence.vram.budget_mib == 18019
    assert evidence.vram.pollution == 'CLEAN'
    assert [c.process_name for c in evidence.vram.baseline_consumers] == ['python']


def _two_rows(report: dict) -> dict:
    return report | {'arms': report['arms'] * 2}


@pytest.mark.parametrize(
    ('report', 'invariant', 'offending'),
    [
        (health_report(schema_version=5), 'REPORT_SCHEMA_VERSION', '5'),
        (_two_rows(health_report()), 'exactly one arm row', '2'),
        (health_report() | {'arms': []}, 'exactly one arm row', '0'),
        (health_report('phi-4-14b'), 'arm_id', 'phi-4-14b'),
        (health_report(pollution='POLLUTED'), 'void', 'POLLUTED'),
        (health_report(baseline_consumers=()), 'whisper-writer', '[]'),
    ],
    ids=['stale-schema', 'two-rows', 'no-rows', 'other-arm', 'polluted', 'no-baseline-consumer'],
)
def test_refuses_a_vram_reading_that_measured_nothing_valid(tmp_path, report, invariant, offending):
    path = _write_health(tmp_path, report)

    with pytest.raises(ScreeningEvidenceError) as raised:
        load_vram_evidence(path, 'qwen3.5-9b')

    message = str(raised.value)
    assert str(path) in message
    assert invariant in message
    assert offending in message


def test_a_failed_vram_verdict_is_a_reading_not_a_refusal(tmp_path):
    path = _write_health(tmp_path, health_report(vram_verdict='FAIL', arm_footprint_mib=19000))

    assert load_vram_evidence(path, 'qwen3.5-9b').vram.verdict == 'FAIL'


def test_screening_evidence_errors_are_value_errors():
    assert issubclass(ScreeningEvidenceError, ValueError)


# --- one arm's evidence --------------------------------------------------------------


def test_loads_a_served_arm_with_its_run_calls_and_vram(tmp_path):
    paths = write_arm_evidence(tmp_path, QWEN)

    evidence = load_arm_evidence(paths, QWEN)

    assert evidence.served is True
    assert evidence.arm == QWEN
    assert evidence.spec == screening_spec(QWEN)
    assert evidence.commands.arm_id == 'qwen3.5-9b'
    assert evidence.vram is not None and evidence.vram.vram.verdict == 'PASS'
    assert [c.prompt_tokens for c in evidence.calls] == [1000, 1001, 1002, 1003, 1004]
    assert evidence.run is not None and evidence.run.spec == evidence.spec
    assert len(evidence.outcomes) == SCREENING_RUN_SHAPE.limit
    metric_ids = {record.metric.metric_id for record in evidence.records}
    assert LlmMetricId.EPISODE_LATENCY_P95 in metric_ids


def test_a_journal_left_beside_the_run_is_not_evidence(tmp_path):
    paths = write_arm_evidence(tmp_path, QWEN)
    (paths.runs / EVIDENCE_STAMP / 'journal').mkdir()

    assert load_arm_evidence(paths, QWEN).served is True


@pytest.mark.parametrize(('start_exit', 'wait_ready_exit'), [(4, 0), (0, 1)])
def test_an_unserved_arm_loads_without_smoke_run_or_health(tmp_path, start_exit, wait_ready_exit):
    paths = write_arm_evidence(
        tmp_path, QWEN, start_exit=start_exit, wait_ready_exit=wait_ready_exit
    )

    evidence = load_arm_evidence(paths, QWEN)

    assert evidence.served is False
    assert evidence.vram is None
    assert evidence.run is None
    assert evidence.calls == ()
    assert evidence.outcomes == ()
    assert evidence.records == ()


def _refused(paths: ArmEvidencePaths, *fragments: str) -> None:
    with pytest.raises(ScreeningEvidenceError) as raised:
        load_arm_evidence(paths, QWEN)
    message = str(raised.value)
    for fragment in fragments:
        assert fragment in message


def test_refuses_an_arm_whose_card_was_held(tmp_path):
    paths = write_arm_evidence(tmp_path, QWEN, start_exit=LMS_CTL_EXIT_CARD_HELD)

    _refused(paths, 'qwen3.5-9b', 'EXIT_CARD_HELD', str(LMS_CTL_EXIT_CARD_HELD))


def test_refuses_commands_with_no_start_record(tmp_path):
    records = (command_record(ScreeningStage.STOP),)
    paths = write_arm_evidence(tmp_path, QWEN, records=records, start_exit=1)

    _refused(paths, 'qwen3.5-9b', 'start')


def test_refuses_a_spec_for_another_arm(tmp_path):
    other = screening_spec(slate_arm(arm_id='phi-4-14b', served_model_name='phi-4-14b'))
    paths = write_arm_evidence(tmp_path, QWEN, spec=other)

    _refused(paths, 'phi-4-14b', 'qwen3.5-9b')


def test_refuses_a_spec_that_is_not_a_candidate(tmp_path):
    control = llm_spec(
        arm_id='qwen3.5-9b', arm_role='control', preregistration_sha=None,
        scratch_group_id='evalmem_lme_eta_qwen3_5_9b',
    )
    paths = write_arm_evidence(tmp_path, QWEN, spec=control)

    _refused(paths, 'candidate', 'control')


def test_refuses_a_tap_pointed_at_another_port(tmp_path):
    tap = TapBinding(listen_url=TAP_LISTEN_URL, upstream_url='http://127.0.0.1:8412')
    paths = write_arm_evidence(tmp_path, QWEN, tap=tap)

    _refused(paths, '8412', '8410')


def test_refuses_a_tap_that_was_not_the_specs_endpoint(tmp_path):
    tap = TapBinding(listen_url='http://127.0.0.1:8419', upstream_url=arm_endpoint(QWEN))
    paths = write_arm_evidence(tmp_path, QWEN, tap=tap)

    _refused(paths, 'http://127.0.0.1:8419', 'http://127.0.0.1:8418')


def test_refuses_a_served_arm_without_a_smoke_record(tmp_path):
    records = (
        command_record(ScreeningStage.START),
        command_record(ScreeningStage.WAIT_READY),
        command_record(ScreeningStage.RUN),
    )
    paths = write_arm_evidence(tmp_path, QWEN, records=records)

    _refused(paths, 'qwen3.5-9b', 'smoke')


def test_refuses_a_served_arm_without_a_health_reading(tmp_path):
    paths = write_arm_evidence(tmp_path, QWEN, write_health=False)

    _refused(paths, str(paths.health))


def test_refuses_a_served_arm_without_its_call_log(tmp_path):
    paths = write_arm_evidence(tmp_path, QWEN)
    paths.calls.unlink()

    _refused(paths, str(paths.calls))


def test_refuses_a_served_arm_without_its_smoke_call_log(tmp_path):
    paths = write_arm_evidence(tmp_path, QWEN)
    paths.smoke_calls.unlink()

    _refused(paths, str(paths.smoke_calls))


@pytest.mark.parametrize('stamps', [(), ('20261007T120000Z', '20261007T130000Z')])
def test_refuses_a_served_arm_without_exactly_one_run(tmp_path, stamps):
    paths = write_arm_evidence(tmp_path, QWEN, stamps=stamps)
    paths.runs.mkdir(parents=True, exist_ok=True)

    _refused(paths, str(paths.runs), str(len(stamps)))


def test_refuses_a_stamp_dir_with_no_run_manifest(tmp_path):
    paths = write_arm_evidence(tmp_path, QWEN)
    (paths.runs / EVIDENCE_STAMP / 'run.json').unlink()

    _refused(paths, 'run.json')


def test_refuses_a_run_of_a_different_spec(tmp_path):
    drifted = screening_spec(QWEN).model_copy(update={'scratch_group_id': 'evalmem_other'})
    paths = write_arm_evidence(tmp_path, QWEN, run_overrides={'spec': drifted})

    _refused(paths, 'qwen3.5-9b', 'spec')


@pytest.mark.parametrize(
    ('override', 'fragment'),
    [
        (
            {'settings_summary': {
                'concurrency': 4, 'index_configuration': 'with-indices', 'episode_timeout_s': 120.0,
            }},
            'concurrency',
        ),
        (
            {'settings_summary': {
                'concurrency': 3, 'index_configuration': 'embedding-only',
                'episode_timeout_s': 120.0,
            }},
            'index_configuration',
        ),
        ({'episode_ids': ('e01', 'e02', 'e03')}, 'limit'),
    ],
    ids=['concurrency', 'index-configuration', 'limit'],
)
def test_refuses_a_run_not_of_the_screening_shape(tmp_path, override, fragment):
    paths = write_arm_evidence(tmp_path, QWEN, run_overrides=override)

    _refused(paths, 'qwen3.5-9b', fragment)


def test_refuses_a_health_reading_taken_in_another_reasoning_mode(tmp_path):
    paths = write_arm_evidence(tmp_path, QWEN, health=health_report(reasoning='off'))

    _refused(paths, 'reasoning', "'off'", "'on'")


def test_refuses_an_own_model_call_whose_max_tokens_is_not_the_specs(tmp_path):
    calls = (call(), call(max_tokens=2048))
    paths = write_arm_evidence(tmp_path, QWEN, calls=calls)

    _refused(paths, 'max_tokens', '2048', '4096')


def test_refuses_a_smoke_call_whose_max_tokens_is_not_the_specs(tmp_path):
    paths = write_arm_evidence(tmp_path, QWEN, smoke_calls=(call(max_tokens=1024),))

    _refused(paths, str(paths.smoke_calls), 'max_tokens', '1024', '4096')


def test_other_models_calls_do_not_carry_the_max_tokens_premise(tmp_path):
    calls = (call(), call(model='text-embedding-3-small', max_tokens=None))
    paths = write_arm_evidence(tmp_path, QWEN, calls=calls, smoke_calls=calls)

    assert len(load_arm_evidence(paths, QWEN).calls) == 2


def test_own_model_calls_are_the_specs_served_model_only():
    own, embedder = call(), call(model='text-embedding-3-small')

    assert own_model_calls((own, embedder, own), screening_spec(QWEN)) == (own, own)
