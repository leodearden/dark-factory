"""η's sweep driver (scripts/local_memory_models_eval/screen_slate.py), fully offline."""

import json
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from _fm_helpers import load_script_module

from arm_harness._fakes import CODE_SHA, CORPUS_SHA, PREREG_SHA, SCREENING_PARAMS, slate_arm
from fused_memory.arm_harness.arm_spec import load_arm_spec
from fused_memory.arm_harness.screening_evidence import (
    SCREENING_RUN_SHAPE,
    ArmEvidencePaths,
    ScreeningStage,
    load_arm_commands,
    load_arm_evidence,
)
from fused_memory.arm_harness.slate import arm_endpoint, candidate_spec

LME_DIR = Path(__file__).parents[2] / 'scripts' / 'local_memory_models_eval'
QWEN = slate_arm()
PHI = slate_arm(
    arm_id='phi-4-14b', port=8412, served_model_name='phi-4-14b', reasoning='off',
    max_model_len=16384,
)
MOE = slate_arm(
    arm_id='moe-stretch', stack='llamacpp', port=8413, served_model_name='moe-stretch',
    structured_output_mode='json_object', quant='q4_k_xl', reasoning='off', max_model_len=16384,
)
SLATE = (QWEN, PHI, MOE)
UV = Path('/opt/uv/bin/uv')
ARM_STAGES = ['start', 'wait-ready', 'smoke', 'run', 'healthcheck', 'stop', 'teardown']


@pytest.fixture
def driver() -> ModuleType:
    return load_script_module(LME_DIR / 'screen_slate.py', mod_name='lme_screen_slate')


# --- the pure argv layer --------------------------------------------------------------


def _submit(driver: ModuleType, **overrides: Any) -> list[str]:
    kwargs = {
        'evidence_root': Path('evidence'),
        'pinned_checkout': Path('pinned'),
        'serving_root': Path('serving'),
        'corpus_manifest': Path('pinned/manifest.json'),
        'preregistration_inputs': Path('inputs.json'),
        'control_spec': Path('control-a.json'),
        'uv': Path('bin/uv'),
        'preregistration_sha': PREREG_SHA,
        'log_path': Path('logs/screen.log'),
    }
    return driver.submit_argv(**(kwargs | overrides))


def _flag(argv: Sequence[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_submit_is_a_transient_collected_unit_that_does_not_wait(driver, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    argv = _submit(driver)

    prefix = argv[: argv.index('--')]
    assert prefix[:2] == ['systemd-run', '--user']
    assert f'--unit={driver.UNIT_NAME}' in prefix
    assert driver.UNIT_NAME == 'lme-eta-screen'
    assert '--collect' in prefix
    assert '--wait' not in prefix
    working = [arg for arg in prefix if arg.startswith('--working-directory=')]
    assert len(working) == 1
    assert Path(working[0].split('=', 1)[1]).is_absolute()
    log = tmp_path / 'logs' / 'screen.log'
    assert prefix[prefix.index(f'StandardOutput=append:{log}') - 1] == '-p'
    assert prefix[prefix.index(f'StandardError=append:{log}') - 1] == '-p'


def test_submit_passes_the_api_key_by_name_only(driver, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-never-on-a-command-line')

    argv = _submit(driver)

    assert '--setenv=OPENAI_API_KEY' in argv
    assert not any('sk-never-on-a-command-line' in arg for arg in argv)
    assert not any(arg.startswith('--setenv=OPENAI_API_KEY=') for arg in argv)


def test_submit_resolves_every_payload_path_absolute(driver, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    argv = _submit(driver, ready_timeout=600.0, arms=('phi-4-14b',), tap_port=8419)

    payload = argv[argv.index('--') + 1:]
    assert Path(payload[0]).is_absolute()
    assert payload[1] == str((LME_DIR / 'screen_slate.py').resolve())
    assert payload[2] == '--in-unit'
    for flag, relative in [
        ('--evidence-root', 'evidence'),
        ('--pinned-checkout', 'pinned'),
        ('--serving-root', 'serving'),
        ('--corpus-manifest', 'pinned/manifest.json'),
        ('--preregistration-inputs', 'inputs.json'),
        ('--control-spec', 'control-a.json'),
        ('--uv', 'bin/uv'),
    ]:
        assert _flag(payload, flag) == str(tmp_path / relative)
    assert _flag(payload, '--preregistration-sha') == PREREG_SHA
    assert _flag(payload, '--tap-port') == '8419'
    assert _flag(payload, '--ready-timeout') == '600.0'
    assert _flag(payload, '--arm') == 'phi-4-14b'


def test_the_pinned_harness_runs_from_its_own_fused_memory_dir(driver, tmp_path):
    spec = tmp_path / 'specs' / 'qwen3.5-9b.json'

    argv = driver.harness_argv(UV, tmp_path / 'pinned', 'smoke', spec)

    assert argv == [
        str(UV), 'run', '--no-sync', 'python', 'scripts/local_memory_models_eval/harness.py',
        'smoke', '--arm-spec', str(spec),
    ]
    assert driver.harness_cwd(tmp_path / 'pinned') == tmp_path / 'pinned' / 'fused-memory'


def test_the_screening_run_takes_its_shape_from_the_one_constant(driver, tmp_path):
    pinned, evidence = tmp_path / 'pinned', tmp_path / 'evidence'
    spec, manifest = evidence / 'specs' / 'qwen3.5-9b.json', pinned / 'corpus_manifest.json'

    argv = driver.screening_run_argv(UV, pinned, spec, evidence, manifest)

    assert argv[:8] == driver.harness_argv(UV, pinned, 'run', spec)
    assert _flag(argv, '--manifest') == str(manifest)
    assert _flag(argv, '--out-root') == str(evidence / 'runs')
    assert _flag(argv, '--repo-root') == str(pinned)
    assert _flag(argv, '--limit') == str(SCREENING_RUN_SHAPE.limit)
    assert _flag(argv, '--concurrency') == str(SCREENING_RUN_SHAPE.concurrency)
    assert _flag(argv, '--index-configuration') == SCREENING_RUN_SHAPE.index_configuration.value
    assert (SCREENING_RUN_SHAPE.limit, SCREENING_RUN_SHAPE.concurrency) == (20, 3)
    assert SCREENING_RUN_SHAPE.index_configuration.value == 'with-indices'


def test_the_pinned_corpus_manifest_is_the_pinned_checkouts(driver, tmp_path):
    assert driver.pinned_corpus_manifest(tmp_path) == (
        tmp_path / 'fused-memory' / 'scripts' / 'local_memory_models_eval' / 'corpus_manifest.json'
    )


def test_lms_commands_run_alphas_clis_through_the_shared_project(driver, tmp_path):
    serving = tmp_path / 'serving'
    tools = serving / 'scripts' / 'local-model-serving'

    assert driver.lms_argv(UV, serving, 'start', 'qwen3.5-9b') == [
        str(UV), 'run', '--project', str(serving / 'shared'), 'python',
        str(tools / 'lms_ctl.py'), 'start', 'qwen3.5-9b',
    ]
    assert driver.lms_argv(UV, serving, 'stop-all')[-1] == 'stop-all'
    assert driver.lms_argv(UV, serving, 'wait-ready', 'phi-4-14b', '--timeout', '60')[-3:] == [
        'phi-4-14b', '--timeout', '60',
    ]
    assert driver.healthcheck_argv(UV, serving, 'phi-4-14b', tmp_path / 'health.json') == [
        str(UV), 'run', '--project', str(serving / 'shared'), 'python',
        str(tools / 'lms_healthcheck.py'), '--arm', 'phi-4-14b', '--output',
        str(tmp_path / 'health.json'),
    ]


# --- the sweep ------------------------------------------------------------------------


@dataclass
class TapSession:
    upstream: str
    log_name: str
    port: int
    closed: bool = False


@dataclass
class FakeTaps:
    sessions: list[TapSession] = field(default_factory=list)
    active: TapSession | None = None

    @contextmanager
    def __call__(self, upstream_url: str, *, log_path: Path, port: int) -> Iterator[str]:
        session = TapSession(upstream_url, Path(log_path).name, port)
        self.sessions.append(session)
        self.active = session
        try:
            yield f'http://127.0.0.1:{port}'
        finally:
            session.closed = True
            self.active = None


@dataclass
class Call:
    verb: str
    arm_id: str | None
    cwd: Path
    tap_log: str | None
    spec_written: bool


def _label(argv: Sequence[str]) -> tuple[str, str | None]:
    if argv[5].endswith('lms_ctl.py'):
        return argv[6], (argv[7] if len(argv) > 7 else None)
    if argv[5].endswith('lms_healthcheck.py'):
        return 'healthcheck', _flag(argv, '--arm')
    return argv[5], Path(_flag(argv, '--arm-spec')).stem


@dataclass
class RecordingRunner:
    driver: ModuleType
    taps: FakeTaps
    evidence_root: Path
    exits: dict[tuple[str, str | None], int] = field(default_factory=dict)
    raises: set[tuple[str, str | None]] = field(default_factory=set)
    calls: list[Call] = field(default_factory=list)

    def __call__(self, argv: Sequence[str], cwd: Path) -> Any:
        verb, arm_id = _label(argv)
        spec = ArmEvidencePaths(self.evidence_root, arm_id or '-').spec
        tap = self.taps.active
        self.calls.append(Call(verb, arm_id, cwd, tap.log_name if tap else None, spec.exists()))
        if (verb, arm_id) in self.raises:
            raise RuntimeError(f'{verb} {arm_id} blew up')
        code = self.exits.get((verb, arm_id), 0)
        return self.driver.CommandOutcome(exit_code=code, output_tail=f'{verb} said {code}')

    def verbs(self, arm_id: str) -> list[str]:
        return [call.verb for call in self.calls if call.arm_id == arm_id]


def _config(driver: ModuleType, tmp_path: Path) -> Any:
    return driver.SweepConfig(
        evidence_root=tmp_path / 'evidence',
        pinned_checkout=tmp_path / 'pinned',
        serving_root=tmp_path / 'serving',
        corpus_manifest=tmp_path / 'pinned' / 'corpus_manifest.json',
        uv=UV,
        code_sha=CODE_SHA,
        corpus_sha=CORPUS_SHA,
        preregistration_sha=PREREG_SHA,
        params=SCREENING_PARAMS,
        tap_port=8418,
        ready_timeout=600.0,
    )


@dataclass
class Sweep:
    config: Any
    runner: RecordingRunner
    taps: FakeTaps
    failures: tuple[str, ...]


def _sweep(driver: ModuleType, tmp_path: Path, slate=SLATE, **runner_kwargs: Any) -> Sweep:
    config = _config(driver, tmp_path)
    taps = FakeTaps()
    runner = RecordingRunner(driver, taps, config.evidence_root, **runner_kwargs)
    failures = driver.sweep(slate, config, runner=runner, tap_factory=taps)
    return Sweep(config, runner, taps, failures)


def _commands(sweep: Sweep, arm_id: str):
    return load_arm_commands(ArmEvidencePaths(sweep.config.evidence_root, arm_id).commands)


def test_one_release_then_each_arm_through_every_stage_in_order(driver, tmp_path):
    sweep = _sweep(driver, tmp_path)

    first = sweep.runner.calls[0]
    assert (first.verb, first.arm_id) == ('stop-all', None)
    assert [call.verb for call in sweep.runner.calls].count('stop-all') == 1
    arm_order = [c.arm_id for c in sweep.runner.calls[1:] if c.verb == 'start']
    assert arm_order == ['qwen3.5-9b', 'phi-4-14b', 'moe-stretch']
    for arm in SLATE:
        assert sweep.runner.verbs(arm.arm_id) == ARM_STAGES
        commands = _commands(sweep, arm.arm_id)
        assert [r.stage.value for r in commands.records] == ARM_STAGES
        assert all(r.exit_code == 0 for r in commands.records)
        assert commands.tap.listen_url == 'http://127.0.0.1:8418'
        assert commands.tap.upstream_url == arm_endpoint(arm)
    assert sweep.failures == ()
    release = json.loads((sweep.config.evidence_root / driver.RELEASE_RECORD_FILENAME).read_text())
    assert release['stage'] == ScreeningStage.RELEASE.value


def test_the_smoke_and_the_run_each_go_through_a_fresh_tap_session(driver, tmp_path):
    sweep = _sweep(driver, tmp_path, slate=(QWEN,))

    by_verb = {call.verb: call for call in sweep.runner.calls}
    assert by_verb['smoke'].tap_log == 'smoke-calls.jsonl'
    assert by_verb['run'].tap_log == 'calls.jsonl'
    assert by_verb['healthcheck'].tap_log is None
    assert by_verb['start'].tap_log is None
    assert [s.log_name for s in sweep.taps.sessions] == ['smoke-calls.jsonl', 'calls.jsonl']
    assert all(s.upstream == arm_endpoint(QWEN) and s.port == 8418 for s in sweep.taps.sessions)
    assert all(session.closed for session in sweep.taps.sessions)


def test_the_harness_runs_pinned_and_alphas_tools_from_the_serving_root(driver, tmp_path):
    sweep = _sweep(driver, tmp_path, slate=(QWEN,))

    cwds = {call.verb: call.cwd for call in sweep.runner.calls}
    pinned = tmp_path / 'pinned' / 'fused-memory'
    assert (cwds['smoke'], cwds['run'], cwds['teardown']) == (pinned, pinned, pinned)
    serving = tmp_path / 'serving'
    assert (cwds['start'], cwds['wait-ready'], cwds['healthcheck'], cwds['stop']) == (
        serving, serving, serving, serving,
    )


def test_the_candidate_spec_is_written_before_the_arm_is_smoked(driver, tmp_path):
    sweep = _sweep(driver, tmp_path, slate=(QWEN,))

    smoke = next(call for call in sweep.runner.calls if call.verb == 'smoke')
    assert smoke.spec_written
    written = load_arm_spec(ArmEvidencePaths(sweep.config.evidence_root, 'qwen3.5-9b').spec)
    assert written == candidate_spec(
        QWEN,
        base_url='http://127.0.0.1:8418/v1',
        code_sha=CODE_SHA,
        corpus_sha=CORPUS_SHA,
        preregistration_sha=PREREG_SHA,
        params=SCREENING_PARAMS,
    )


def test_a_failed_smoke_still_measures_the_other_three_gates(driver, tmp_path):
    sweep = _sweep(driver, tmp_path, slate=(QWEN,), exits={('smoke', 'qwen3.5-9b'): 3})

    assert sweep.runner.verbs('qwen3.5-9b') == ARM_STAGES
    smoke = _commands(sweep, 'qwen3.5-9b').record_of(ScreeningStage.SMOKE)
    assert smoke is not None and smoke.exit_code == 3
    assert smoke.output_tail == 'smoke said 3'


@pytest.mark.parametrize(
    ('failing', 'expected'),
    [('start', ['start', 'stop']), ('wait-ready', ['start', 'wait-ready', 'stop'])],
)
def test_an_unserved_arm_is_still_stopped_and_its_evidence_loads(driver, tmp_path, failing, expected):
    sweep = _sweep(driver, tmp_path, slate=(QWEN,), exits={(failing, 'qwen3.5-9b'): 4})

    assert sweep.runner.verbs('qwen3.5-9b') == expected
    assert [r.stage.value for r in _commands(sweep, 'qwen3.5-9b').records] == expected
    assert sweep.taps.sessions == []
    paths = ArmEvidencePaths(sweep.config.evidence_root, 'qwen3.5-9b')
    assert load_arm_evidence(paths, QWEN).served is False


def test_a_failed_run_is_still_torn_down(driver, tmp_path):
    sweep = _sweep(driver, tmp_path, slate=(QWEN,), exits={('run', 'qwen3.5-9b'): 1})

    assert sweep.runner.verbs('qwen3.5-9b') == ARM_STAGES


def test_an_exception_mid_arm_stops_it_closes_the_tap_and_the_sweep_goes_on(driver, tmp_path):
    sweep = _sweep(driver, tmp_path, slate=(QWEN, PHI), raises={('run', 'qwen3.5-9b')})

    assert sweep.runner.verbs('qwen3.5-9b') == [
        'start', 'wait-ready', 'smoke', 'run', 'stop', 'teardown',
    ]
    assert all(session.closed for session in sweep.taps.sessions)
    assert sweep.runner.verbs('phi-4-14b') == ARM_STAGES
    stages = [r.stage.value for r in _commands(sweep, 'qwen3.5-9b').records]
    assert stages == ['start', 'wait-ready', 'smoke', 'stop', 'teardown']
    assert sweep.failures == ('qwen3.5-9b',)


def test_an_arm_subset_runs_in_manifest_order(driver, tmp_path):
    config = _config(driver, tmp_path)
    taps = FakeTaps()
    runner = RecordingRunner(driver, taps, config.evidence_root)

    driver.sweep(SLATE, config, arms=('moe-stretch', 'qwen3.5-9b'), runner=runner, tap_factory=taps)

    started = [call.arm_id for call in runner.calls if call.verb == 'start']
    assert started == ['qwen3.5-9b', 'moe-stretch']


def test_an_unknown_arm_in_the_subset_is_refused_before_anything_runs(driver, tmp_path):
    config = _config(driver, tmp_path)
    runner = RecordingRunner(driver, FakeTaps(), config.evidence_root)

    with pytest.raises(ValueError, match='mistral'):
        driver.sweep(SLATE, config, arms=('mistral',), runner=runner, tap_factory=FakeTaps())

    assert runner.calls == []


def test_a_non_empty_evidence_root_is_refused_before_anything_runs(driver, tmp_path):
    config = _config(driver, tmp_path)
    config.evidence_root.mkdir(parents=True)
    (config.evidence_root / 'leftover').write_text('x')
    runner = RecordingRunner(driver, FakeTaps(), config.evidence_root)

    with pytest.raises(ValueError, match=str(config.evidence_root)):
        driver.sweep(SLATE, config, runner=runner, tap_factory=FakeTaps())

    assert runner.calls == []


# --- the real runner ------------------------------------------------------------------


def test_the_real_runner_strips_virtual_env_and_runs_in_the_given_cwd(driver, tmp_path, monkeypatch):
    monkeypatch.setenv('VIRTUAL_ENV', '/somewhere/else/.venv')
    probe = 'import os,sys; print(os.environ.get("VIRTUAL_ENV")); print(os.getcwd()); sys.exit(3)'

    outcome = driver.run_command([sys.executable, '-c', probe], tmp_path)

    assert outcome.exit_code == 3
    assert outcome.output_tail.splitlines() == ['None', str(tmp_path)]


def test_the_real_runner_keeps_only_the_tail_of_combined_output(driver, tmp_path):
    probe = 'import sys; print("e" * 50, file=sys.stderr); print("x" * 20000)'

    outcome = driver.run_command([sys.executable, '-c', probe], tmp_path)

    assert outcome.exit_code == 0
    assert len(outcome.output_tail) == driver.OUTPUT_TAIL_CHARS
    assert outcome.output_tail.rstrip('\n').endswith('x' * 100)
