#!/usr/bin/env python3
"""η's screening sweep: each arms.yaml LLM arm in turn, pinned harness behind the usage tap.

By default this submits one transient ``systemd --user`` unit and returns; ``--in-unit``
performs the sweep inside it. Provenance: plans/local-memory-models-eval-screening/README.md.
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from fused_memory.arm_harness.arm_spec import LlmArmSpec, LlmParams, load_arm_spec
from fused_memory.arm_harness.corpus import corpus_sha
from fused_memory.arm_harness.preregistration import load_preregistration_inputs
from fused_memory.arm_harness.screening_evidence import (
    SCREENING_RUN_SHAPE,
    ArmCommands,
    ArmEvidencePaths,
    CommandRecord,
    ScreeningStage,
    TapBinding,
    write_arm_commands,
    write_release_record,
    write_screening_spec,
)
from fused_memory.arm_harness.slate import (
    LOOPBACK_HOST,
    SlateArm,
    arm_endpoint,
    candidate_spec,
    load_llm_slate,
)
from fused_memory.arm_harness.usage_tap import usage_tap

UNIT_NAME = 'lme-eta-screen'
TAP_PORT = 8418
OUTPUT_TAIL_CHARS = 4000
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
CONTROLS = REPO_ROOT / 'plans' / 'local-memory-models-eval-controls'
DEFAULT_PREREGISTRATION_INPUTS = CONTROLS / 'preregistration-inputs.json'
DEFAULT_CONTROL_SPEC = CONTROLS / 'specs' / 'incumbent-generic-a.json'
HARNESS_SCRIPT = 'scripts/local_memory_models_eval/harness.py'
SERVING_TOOLS = Path('scripts') / 'local-model-serving'


@dataclass(frozen=True)
class CommandOutcome:
    exit_code: int
    output_tail: str


Runner = Callable[[Sequence[str], Path], CommandOutcome]
TapFactory = Callable[..., AbstractContextManager[str]]


def _absolute(path: Path | str) -> str:
    return os.path.abspath(path)


def submit_argv(
    *,
    evidence_root: Path,
    pinned_checkout: Path,
    serving_root: Path,
    corpus_manifest: Path,
    preregistration_inputs: Path,
    control_spec: Path,
    uv: Path,
    preregistration_sha: str,
    log_path: Path,
    tap_port: int = TAP_PORT,
    ready_timeout: float | None = None,
    arms: Sequence[str] = (),
) -> list[str]:
    log = _absolute(log_path)
    payload = [
        _absolute(sys.executable), str(SCRIPT_PATH), '--in-unit',
        '--evidence-root', _absolute(evidence_root),
        '--pinned-checkout', _absolute(pinned_checkout),
        '--serving-root', _absolute(serving_root),
        '--corpus-manifest', _absolute(corpus_manifest),
        '--preregistration-inputs', _absolute(preregistration_inputs),
        '--control-spec', _absolute(control_spec),
        '--uv', _absolute(uv),
        '--preregistration-sha', preregistration_sha,
        '--tap-port', str(tap_port),
    ]
    if ready_timeout is not None:
        payload += ['--ready-timeout', str(ready_timeout)]
    for arm_id in arms:
        payload += ['--arm', arm_id]
    return [
        'systemd-run', '--user', f'--unit={UNIT_NAME}', '--collect',
        f'--working-directory={REPO_ROOT}',
        '--setenv=OPENAI_API_KEY',
        '-p', f'StandardOutput=append:{log}',
        '-p', f'StandardError=append:{log}',
        '--', *payload,
    ]


def harness_cwd(pinned_checkout: Path) -> Path:
    return pinned_checkout / 'fused-memory'


def pinned_corpus_manifest(pinned_checkout: Path) -> Path:
    return harness_cwd(pinned_checkout) / Path(HARNESS_SCRIPT).parent / 'corpus_manifest.json'


def harness_argv(uv: Path, verb: str, spec_path: Path, *extra: str) -> list[str]:
    return [
        str(uv), 'run', '--no-sync', 'python', HARNESS_SCRIPT, verb,
        '--arm-spec', str(spec_path), *extra,
    ]


def screening_run_argv(
    uv: Path, pinned_checkout: Path, paths: ArmEvidencePaths, corpus_manifest: Path
) -> list[str]:
    return harness_argv(
        uv, 'run', paths.spec,
        '--manifest', str(corpus_manifest),
        '--out-root', str(paths.runs.parent),
        '--repo-root', str(pinned_checkout),
        '--limit', str(SCREENING_RUN_SHAPE.limit),
        '--concurrency', str(SCREENING_RUN_SHAPE.concurrency),
        '--index-configuration', SCREENING_RUN_SHAPE.index_configuration.value,
    )


def _serving_argv(uv: Path, serving_root: Path, tool: str, *args: str) -> list[str]:
    return [
        str(uv), 'run', '--no-sync', '--project', str(serving_root / 'shared'), 'python',
        str(serving_root / SERVING_TOOLS / tool), *args,
    ]


def lms_argv(
    uv: Path, serving_root: Path, verb: str, arm_id: str | None = None, *extra: str
) -> list[str]:
    subject = [] if arm_id is None else [arm_id]
    return _serving_argv(uv, serving_root, 'lms_ctl.py', verb, *subject, *extra)


def healthcheck_argv(uv: Path, serving_root: Path, arm_id: str, output: Path) -> list[str]:
    return _serving_argv(
        uv, serving_root, 'lms_healthcheck.py', '--arm', arm_id, '--output', str(output)
    )


def run_command(argv: Sequence[str], cwd: Path) -> CommandOutcome:
    env = {key: value for key, value in os.environ.items() if key != 'VIRTUAL_ENV'}
    tail = ''
    with subprocess.Popen(
        list(argv), cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors='replace',
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end='', flush=True)
            tail = (tail + line)[-OUTPUT_TAIL_CHARS:]
    return CommandOutcome(exit_code=process.returncode, output_tail=tail)


@dataclass(frozen=True)
class SweepConfig:
    evidence_root: Path
    pinned_checkout: Path
    serving_root: Path
    corpus_manifest: Path
    uv: Path
    code_sha: str
    corpus_sha: str
    preregistration_sha: str
    params: LlmParams
    tap_port: int = TAP_PORT
    ready_timeout: float | None = None

    @property
    def tap_listen_url(self) -> str:
        return f'http://{LOOPBACK_HOST}:{self.tap_port}'


def _record(
    runner: Runner, stage: ScreeningStage, argv: Sequence[str], cwd: Path
) -> CommandRecord:
    print(f'--- {stage.value}: {shlex.join(argv)}', flush=True)
    started_at = datetime.now(UTC)
    outcome = runner(argv, cwd)
    print(f'--- {stage.value} exited {outcome.exit_code}', flush=True)
    return CommandRecord(
        stage=stage,
        argv=tuple(argv),
        exit_code=outcome.exit_code,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        output_tail=outcome.output_tail,
    )


@contextmanager
def _tap_session(
    tap_factory: TapFactory, tap: TapBinding, log_path: Path, port: int
) -> Iterator[None]:
    with tap_factory(tap.upstream_url, log_path=log_path, port=port) as listen_url:
        if listen_url != tap.listen_url:
            raise RuntimeError(f'the usage tap listens at {listen_url}, not at {tap.listen_url}')
        yield


def _screen_arm(
    arm: SlateArm, config: SweepConfig, runner: Runner, tap_factory: TapFactory
) -> None:
    paths = ArmEvidencePaths(config.evidence_root, arm.arm_id)
    paths.arm_dir.mkdir(parents=True, exist_ok=True)
    tap = TapBinding(listen_url=config.tap_listen_url, upstream_url=arm_endpoint(arm))
    write_screening_spec(paths.spec, _arm_spec(arm, config, tap))
    records: list[CommandRecord] = []

    def stage(name: ScreeningStage, argv: Sequence[str], cwd: Path) -> int:
        records.append(_record(runner, name, argv, cwd))
        return records[-1].exit_code

    lms = partial(lms_argv, config.uv, config.serving_root)
    serving, harness_dir = config.serving_root, harness_cwd(config.pinned_checkout)
    timeout = () if config.ready_timeout is None else ('--timeout', str(config.ready_timeout))
    session = partial(_tap_session, tap_factory, tap, port=config.tap_port)
    run_attempted = False
    try:
        if stage(ScreeningStage.START, lms('start', arm.arm_id), serving) != 0:
            return
        wait_ready = lms('wait-ready', arm.arm_id, *timeout)
        if stage(ScreeningStage.WAIT_READY, wait_ready, serving) != 0:
            return
        with session(paths.smoke_calls):
            stage(ScreeningStage.SMOKE, harness_argv(config.uv, 'smoke', paths.spec), harness_dir)
        run_attempted = True
        run = screening_run_argv(config.uv, config.pinned_checkout, paths, config.corpus_manifest)
        with session(paths.calls):
            stage(ScreeningStage.RUN, run, harness_dir)
        health = healthcheck_argv(config.uv, serving, arm.arm_id, paths.health)
        stage(ScreeningStage.HEALTHCHECK, health, serving)
    finally:
        try:
            stage(ScreeningStage.STOP, lms('stop', arm.arm_id), serving)
            if run_attempted:
                teardown = harness_argv(config.uv, 'teardown', paths.spec)
                stage(ScreeningStage.TEARDOWN, teardown, harness_dir)
        finally:
            commands = ArmCommands(arm_id=arm.arm_id, tap=tap, records=tuple(records))
            write_arm_commands(paths.commands, commands)


def _arm_spec(arm: SlateArm, config: SweepConfig, tap: TapBinding) -> LlmArmSpec:
    return candidate_spec(
        arm,
        base_url=f'{tap.listen_url}/v1',
        code_sha=config.code_sha,
        corpus_sha=config.corpus_sha,
        preregistration_sha=config.preregistration_sha,
        params=config.params,
    )


def _selected(slate: Sequence[SlateArm], arms: Sequence[str]) -> tuple[SlateArm, ...]:
    if not arms:
        return tuple(slate)
    unknown = sorted(set(arms) - {arm.arm_id for arm in slate})
    if unknown:
        raise ValueError(f'unknown arm ids {unknown}; the slate is {[a.arm_id for a in slate]}')
    return tuple(arm for arm in slate if arm.arm_id in arms)


def _require_fresh(evidence_root: Path) -> None:
    if evidence_root.exists() and any(evidence_root.iterdir()):
        raise ValueError(
            f'{evidence_root} is not empty: each sweep writes a fresh evidence root, so a '
            're-screen needs a new one'
        )


def sweep(
    slate: Sequence[SlateArm],
    config: SweepConfig,
    *,
    arms: Sequence[str] = (),
    runner: Runner = run_command,
    tap_factory: TapFactory = usage_tap,
) -> tuple[str, ...]:
    """Screen the selected arms in manifest order; returns the arms whose sweep raised."""
    selected = _selected(slate, arms)
    _require_fresh(config.evidence_root)
    release = lms_argv(config.uv, config.serving_root, 'stop-all')
    write_release_record(
        config.evidence_root,
        _record(runner, ScreeningStage.RELEASE, release, config.serving_root),
    )
    failures: list[str] = []
    for arm in selected:
        try:
            _screen_arm(arm, config, runner, tap_factory)
        except Exception:
            traceback.print_exc()
            failures.append(arm.arm_id)
    return tuple(failures)


def _main_checkout() -> Path:
    common = subprocess.run(
        ['git', '-C', str(REPO_ROOT), 'rev-parse', '--path-format=absolute', '--git-common-dir'],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return Path(common).parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="η's screening sweep over the arms.yaml LLM slate (a systemd --user unit)"
    )
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--pinned-checkout', type=Path, required=True)
    parser.add_argument('--serving-root', type=Path, help='default: the main checkout')
    parser.add_argument('--corpus-manifest', type=Path, help="default: the pinned checkout's")
    parser.add_argument(
        '--preregistration-inputs', type=Path, default=DEFAULT_PREREGISTRATION_INPUTS
    )
    parser.add_argument('--control-spec', type=Path, default=DEFAULT_CONTROL_SPEC)
    parser.add_argument('--preregistration-sha', required=True)
    parser.add_argument('--uv', type=Path, help='default: uv on PATH')
    parser.add_argument('--tap-port', type=int, default=TAP_PORT)
    parser.add_argument('--ready-timeout', type=float)
    parser.add_argument('--arm', action='append', default=[], help='screen only these arms')
    parser.add_argument('--log', type=Path, help='default: <evidence-root>/../lme-eta-screen.log')
    parser.add_argument('--dry-run', action='store_true', help='print the submit argv only')
    parser.add_argument('--in-unit', action='store_true', help=argparse.SUPPRESS)
    return parser


def _in_unit(args: argparse.Namespace) -> int:
    inputs = load_preregistration_inputs(args.preregistration_inputs)
    control = load_arm_spec(args.control_spec)
    if not isinstance(control, LlmArmSpec):
        raise SystemExit(f'{args.control_spec} is not an LLM control spec')
    manifest_sha = corpus_sha(args.corpus_manifest.read_bytes())
    if manifest_sha != inputs.corpus_sha:
        raise SystemExit(
            f'{args.corpus_manifest} hashes to {manifest_sha}, not the controls\' corpus_sha '
            f'{inputs.corpus_sha}'
        )
    config = SweepConfig(
        evidence_root=args.evidence_root,
        pinned_checkout=args.pinned_checkout,
        serving_root=args.serving_root,
        corpus_manifest=args.corpus_manifest,
        uv=args.uv,
        code_sha=inputs.code_sha,
        corpus_sha=manifest_sha,
        preregistration_sha=args.preregistration_sha,
        params=control.params,
        tap_port=args.tap_port,
        ready_timeout=args.ready_timeout,
    )
    slate = load_llm_slate(args.serving_root / SERVING_TOOLS / 'arms.yaml')
    failures = sweep(slate, config, arms=tuple(args.arm))
    print(f'sweep finished; arms whose sweep raised: {list(failures) or "none"}', flush=True)
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.in_unit:
        return _in_unit(args)
    uv = args.uv or shutil.which('uv')
    if uv is None:
        raise SystemExit('uv is not on PATH; pass --uv')
    unit_argv = submit_argv(
        evidence_root=args.evidence_root,
        pinned_checkout=args.pinned_checkout,
        serving_root=args.serving_root or _main_checkout(),
        corpus_manifest=args.corpus_manifest or pinned_corpus_manifest(args.pinned_checkout),
        preregistration_inputs=args.preregistration_inputs,
        control_spec=args.control_spec,
        uv=Path(uv),
        preregistration_sha=args.preregistration_sha,
        log_path=args.log or Path(_absolute(args.evidence_root)).parent / f'{UNIT_NAME}.log',
        tap_port=args.tap_port,
        ready_timeout=args.ready_timeout,
        arms=tuple(args.arm),
    )
    print(shlex.join(unit_argv))
    if args.dry_run:
        return 0
    return subprocess.run(unit_argv, check=False).returncode


if __name__ == '__main__':
    sys.exit(main())
