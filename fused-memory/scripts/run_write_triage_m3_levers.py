#!/usr/bin/env python3
"""Replay the rated soak corpus through the shipped judge under each M3 lever arm (task 7519).

The only LLM edge of the M3 lever study. Each :class:`Arm` is a judge prompt,
model and reasoning effort, or a second pass over a parent arm's verdicts.
Each case goes to the SHIPPED ``write_triage_judge.judge_write`` with a
one-candidate slate: the parent its raters rated. The arm's prompt is held in
force by ``write_triage_judge_wording.judge_system_prompt``, and its model
and effort sit on a copy of the config. One row per case is appended to
``<arms-dir>/<arm>.jsonl``. Each row carries the judge's outcome and the
outcome deployed routing gives it (``write_triage_m3_replay_cases.py``), with
its cost and latency. ``score_write_triage_m3_levers.py`` scores the rows.

Arms run one after another, because the prompt override is process-global.
Calls within an arm run concurrently. A transport failure is retried with
doubling backoff. An unreadable answer, or a failure that outlives the
retries, gives a fail-open ``stored`` row, as triage would. Re-running
resumes: rows already written are kept, and rows from another cases file or
prompt are refused. Once the list-price spend recorded in ``--arms-dir``
reaches ``--budget-usd``, no further call is dispatched. Calls already in
flight still land, so the cap can be overshot by up to ``--concurrency``
calls. The run exits 0 only when every named arm is complete.

Usage
-----
Run from ``fused-memory/`` with the key loaded for this one command. The
replay takes many minutes, so run it detached and read its log::

    set -a; . /home/leo/src/dark-factory/.env; set +a
    setsid uv run python scripts/run_write_triage_m3_levers.py run \\
        --cases ../data/write-triage-m3-levers/cases.jsonl \\
        --arms psi-repro,f3p,f3p-repeat --budget-usd 60 \\
        > ../data/write-triage-m3-levers/run.log 2>&1 &
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import re
import sys
import time
import types
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import openai

from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_JUDGE,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    BandDecision,
    resolve_attach_guard,
)
from fused_memory.server.write_triage_judge import (
    JudgeOutputError,
    judge_write,
    resolve_judge_enabled,
)
from fused_memory.utils.script_loader import load_script

_SCRIPTS = Path(__file__).resolve().parent

if TYPE_CHECKING:
    import score_write_triage_pairs as _pairs
    import write_triage_judge_wording as _wording
    import write_triage_m3_replay_cases as _cases
else:
    _cases = load_script(_SCRIPTS / 'write_triage_m3_replay_cases.py', 'write_triage_m3_replay_cases')
    _pairs = load_script(_SCRIPTS / 'score_write_triage_pairs.py', 'score_write_triage_pairs')
    _wording = load_script(_SCRIPTS / 'write_triage_judge_wording.py', 'write_triage_judge_wording')

KIND_JUDGE = 'judge'
KIND_SECOND_PASS = 'second_pass'
KIND_TRANSFORM = 'transform'
ARM_KINDS = (KIND_JUDGE, KIND_SECOND_PASS, KIND_TRANSFORM)

SCOPE_ALL = 'all'
SCOPE_SOAK_JUDGE_BAND = 'soak-judge-band'
"""Judge every case the soak's own judge band held, whatever today's route."""

SHIPPED_MODEL = 'gpt-6.1-sol'
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 2.0
DEFAULT_ARMS_DIR = Path(__file__).resolve().parent.parent / 'calibration' / 'write_triage_m3_lever_arms'

#: Transport failures worth another attempt; anything else aborts the run.
TRANSIENT = (
    TimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)

#: The cosine a candidate is shown with when the soak could not measure it.
#: The judge prompt renders no cosine; the selector only needs a number.
_UNMEASURED_STORE_SCORE = 1.0

_SECOND_PASS_SOURCES = frozenset({OUTCOME_AMENDED, OUTCOME_RESTATED})
_EVAL_PROJECT_ID = 'write-triage-m3-levers'
_NAME = re.compile(r'[a-z0-9][a-z0-9-]*')


@dataclass(frozen=True)
class Arm:
    """One replay arm: a judge configuration, or a pass over a parent arm's rows."""

    name: str
    kind: str
    prompt: str | None
    model: str | None
    effort: str | None
    parent: str | None
    scope: str
    description: str


_SHIPPED_PROMPT = _wording.system_prompt(_wording.WORDING_SHIPPED)


def _shipped_arm(name: str, effort: str, description: str) -> Arm:
    return Arm(name, KIND_JUDGE, _SHIPPED_PROMPT, SHIPPED_MODEL, effort, None, SCOPE_ALL, description)


ARMS: tuple[Arm, ...] = (
    Arm('psi-repro', KIND_JUDGE, _wording.PSI_JUDGE_SYSTEM_PROMPT, SHIPPED_MODEL, 'low', None,
        SCOPE_SOAK_JUDGE_BAND,
        'the soak-era psi prompt and model; its agreement with the soak outcomes is the harness error'),
    _shipped_arm('f3p', 'low', "the shipped F3' prompt as deployed: the baseline every lever is read against"),
    _shipped_arm('f3p-repeat', 'low', "f3p again: the judge's run-to-run noise floor"),
    _shipped_arm('b1-sol-medium', 'medium', 'the shipped prompt at medium reasoning effort'),
    _shipped_arm('b2-sol-high', 'high', 'the shipped prompt at high reasoning effort'),
)


def arm_named(name: str, arms: Iterable[Arm] = ARMS) -> Arm:
    known = {arm.name: arm for arm in arms}
    if name not in known:
        raise ValueError(f'unknown arm {name!r}; known arms are {list(known)}')
    return known[name]


def validate_arms(arms: Sequence[Arm]) -> None:
    """Refuse a malformed arm table: names, kinds, scopes, prompts and parents."""
    seen: set[str] = set()
    for arm in arms:
        problem = _arm_problem(arm, seen)
        if problem:
            raise ValueError(f'arm {arm.name!r}: {problem}')
        seen.add(arm.name)


def _arm_problem(arm: Arm, seen: set[str]) -> str | None:
    if not _NAME.fullmatch(arm.name) or arm.name in seen:
        return 'the name must be unique, lowercase and filename-safe'
    if arm.kind not in ARM_KINDS or arm.scope not in (SCOPE_ALL, SCOPE_SOAK_JUDGE_BAND):
        return f'kind {arm.kind!r} or scope {arm.scope!r} is unknown'
    if (arm.kind == KIND_JUDGE) != (arm.parent is None):
        return 'a judge arm names no parent, and every other kind names one'
    if arm.parent is not None and arm.parent not in seen:
        return f'parent {arm.parent!r} is not an earlier arm'
    if arm.kind != KIND_TRANSFORM and not arm.prompt:
        return 'a judge or second-pass arm needs a prompt'
    return None


def arm_service(config: Any, arm: Arm) -> Any:
    """The duck-typed service ``judge_write`` reads: *config* with the arm's model and effort."""
    write_triage = config.write_triage.model_copy(update={
        'judge_model': arm.model, 'judge_reasoning_effort': arm.effort,
    })
    return types.SimpleNamespace(config=config.model_copy(update={'write_triage': write_triage}))


def cases_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class _Call:
    """One judged answer, or the failure that ended it."""

    outcome: str
    candidate_id: str | None
    attempts: int
    seconds: float
    usage: Any
    transport_failures: tuple[str, ...]
    error: BaseException | None


async def _ask(
    case: Any, service: Any, sleep: Callable[[float], Awaitable[Any]],
) -> _Call:
    features = case.features
    score = features.cos_parent if features.cos_parent is not None else _UNMEASURED_STORE_SCORE
    candidate = MemoryResult(
        id=features.target_id, content=case.parent_text, source_store=SourceStore.mem0,
        metadata={'store_score': score},
    )
    decision = BandDecision(
        outcome=OUTCOME_JUDGE, canonical_id=features.target_id, similarity=score,
        t_high=None, t_low=None,
    )
    failures: list[str] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = time.monotonic()
        try:
            verdict = await judge_write(
                memory_service=service, content=case.child_text, project_id=_EVAL_PROJECT_ID,
                decision=decision, candidates=[candidate],
            )
        except JudgeOutputError as exc:
            return _Call(OUTCOME_STORED, None, attempt, time.monotonic() - started, None,
                         tuple(failures), exc)
        except TRANSIENT as exc:
            failures.append(type(exc).__name__)
            if attempt == MAX_ATTEMPTS:
                return _Call(OUTCOME_STORED, None, attempt, time.monotonic() - started, None,
                             tuple(failures), exc)
            await sleep(BACKOFF_SECONDS * 2 ** (attempt - 1))
            continue
        return _Call(verdict.outcome, verdict.candidate_id, attempt, time.monotonic() - started,
                     verdict.usage, tuple(failures), None)
    raise AssertionError('unreachable: the last attempt always returns')


def _base_row(case: Any, arm: Arm, *, cases_sha256: str, prompt_sha256: str | None) -> dict[str, Any]:
    features = case.features
    return {
        'arm': arm.name, 'entry_id': features.entry_id, 'target_id': features.target_id,
        'route': str(features.route), 'model': arm.model, 'effort': arm.effort,
        'prompt_sha256': prompt_sha256, 'cases_sha256': cases_sha256,
        'judge_outcome': None, 'candidate_id': None, 'dispute_marked': None,
        'calls': 0, 'attempts': 0, 'seconds': None, 'input_tokens': None,
        'output_tokens': None, 'reasoning_tokens': None, 'usd': 0.0,
        'error_type': None, 'error': None, 'transport_failures': [],
    }


def _call_fields(call: _Call, model: str | None) -> dict[str, Any]:
    usage = call.usage
    price = _pairs.LIST_PRICES.get(model or '')
    usd = None if usage is None or price is None else price.usd(usage.input_tokens, usage.output_tokens)
    return {
        'calls': 1, 'attempts': call.attempts, 'seconds': round(call.seconds, 3),
        'input_tokens': None if usage is None else usage.input_tokens,
        'output_tokens': None if usage is None else usage.output_tokens,
        'reasoning_tokens': None if usage is None else usage.reasoning_tokens,
        'usd': 0.0 if usd is None else usd,
        'error_type': None if call.error is None else type(call.error).__name__,
        'error': None if call.error is None else str(call.error)[:500],
        'transport_failures': list(call.transport_failures),
    }


def _judged(case: Any, arm: Arm) -> bool:
    if arm.scope == SCOPE_SOAK_JUDGE_BAND:
        return case.features.soak_band == OUTCOME_JUDGE
    return case.features.route is _cases.Route.JUDGE


async def judge_case(
    case: Any, arm: Arm, service: Any, *, cases_sha256: str, prompt_sha256: str,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> dict[str, Any]:
    """One judge-arm row for *case*; the caller holds the arm's prompt in force."""
    row = _base_row(case, arm, cases_sha256=cases_sha256, prompt_sha256=prompt_sha256)
    guard = resolve_attach_guard(service)
    if not _judged(case, arm):
        return row | {'routed_outcome': _cases.routed_outcome(case.features, None, guard)}
    call = await _ask(case, service, sleep)
    routed = _cases.routed_outcome(
        dataclasses.replace(case.features, route=_cases.Route.JUDGE), call.outcome, guard,
    ) if arm.scope == SCOPE_SOAK_JUDGE_BAND else _cases.routed_outcome(case.features, call.outcome, guard)
    return row | _call_fields(call, arm.model) | {
        'judge_outcome': call.outcome, 'candidate_id': call.candidate_id, 'routed_outcome': routed,
    }


async def second_pass_case(
    case: Any, parent_row: Mapping[str, Any], arm: Arm, service: Any, *, cases_sha256: str,
    prompt_sha256: str, sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> dict[str, Any]:
    """A second pass over a parent row: is an attached amendment or restatement disputed?"""
    row = _base_row(case, arm, cases_sha256=cases_sha256, prompt_sha256=prompt_sha256) | {
        'judge_outcome': parent_row['judge_outcome'], 'candidate_id': parent_row['candidate_id'],
        'routed_outcome': parent_row['routed_outcome'], 'dispute_marked': False,
    }
    attached = parent_row['judge_outcome'] in _SECOND_PASS_SOURCES and (
        parent_row['routed_outcome'] == parent_row['judge_outcome']
    )
    if not attached:
        return row
    call = await _ask(case, service, sleep)
    return row | _call_fields(call, arm.model) | {
        'dispute_marked': call.error is None and call.outcome == OUTCOME_CONTESTED,
    }


@dataclass(frozen=True)
class RunResult:
    complete: tuple[str, ...]
    partial: tuple[str, ...]
    spent_usd: float

    @property
    def exit_code(self) -> int:
        return 0 if not self.partial else 2

    @property
    def message(self) -> str:
        if not self.partial:
            return f'every arm complete; ${self.spent_usd:.2f} recorded at list price'
        return (
            f'arms left partial: {", ".join(self.partial)}; ${self.spent_usd:.2f} recorded at'
            ' list price'
        )


def read_arm_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def arm_scope_cases(cases: Sequence[Any], arm: Arm) -> list[Any]:
    """The cases a complete *arm* holds a row for."""
    if arm.scope == SCOPE_SOAK_JUDGE_BAND:
        return [case for case in cases if case.features.soak_band == OUTCOME_JUDGE]
    return list(cases)


class _Spend:
    def __init__(self, recorded: float, budget: float) -> None:
        self.total = recorded
        self.budget = budget

    @property
    def exhausted(self) -> bool:
        return self.total >= self.budget


def _refuse_foreign(rows: Sequence[Mapping[str, Any]], arm: Arm, *, cases_sha256: str,
                    prompt_sha256: str | None) -> None:
    expected = {'cases_sha256': cases_sha256, 'prompt_sha256': prompt_sha256,
                'model': arm.model, 'effort': arm.effort}
    for row in rows:
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(
                    f'arm {arm.name!r}: row {row.get("entry_id")!r} has {key} {row.get(key)!r},'
                    f' but this run has {value!r}; move the arm file aside to start it afresh'
                )


async def run_arms(
    cases: Sequence[Any],
    arms: Sequence[Arm],
    *,
    config: Any,
    arms_dir: Path,
    budget_usd: float,
    cases_sha256: str,
    concurrency: int = 8,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> RunResult:
    """Run each arm in order, resuming from and appending to its arm file."""
    arms_dir.mkdir(parents=True, exist_ok=True)
    recorded = sum(row.get('usd') or 0.0 for path in sorted(arms_dir.glob('*.jsonl'))
                   for row in read_arm_rows(path))
    spend = _Spend(recorded, budget_usd)
    complete: list[str] = []
    partial: list[str] = []
    for arm in arms:
        done = await _run_arm(cases, arm, config=config, arms_dir=arms_dir, spend=spend,
                              cases_sha256=cases_sha256, concurrency=concurrency, sleep=sleep)
        (complete if done else partial).append(arm.name)
    return RunResult(tuple(complete), tuple(partial), spend.total)


async def _run_arm(
    cases: Sequence[Any], arm: Arm, *, config: Any, arms_dir: Path, spend: _Spend,
    cases_sha256: str, concurrency: int, sleep: Callable[[float], Awaitable[Any]],
) -> bool:
    if arm.kind not in (KIND_JUDGE, KIND_SECOND_PASS):
        raise ValueError(f'arm {arm.name!r}: kind {arm.kind!r} is not replayed by a judge call')
    service = arm_service(config, arm)
    if not resolve_judge_enabled(service):
        raise ValueError('write_triage.judge_enabled is false, so no arm can be replayed')
    path = arms_dir / f'{arm.name}.jsonl'
    scope = arm_scope_cases(cases, arm)
    parents = _complete_parent_rows(cases, arm, arms_dir)
    if parents is None:
        return False
    with _wording.judge_system_prompt(arm.prompt or '', label=arm.name) as prompt_sha:
        existing = read_arm_rows(path)
        _refuse_foreign(existing, arm, cases_sha256=cases_sha256, prompt_sha256=prompt_sha)
        written = {row['entry_id'] for row in existing}
        todo = [case for case in scope if case.features.entry_id not in written]
        gate = asyncio.Semaphore(concurrency)

        async def one(case: Any) -> None:
            async with gate:
                if spend.exhausted and _needs_call(case, arm, parents):
                    return
                if arm.kind == KIND_JUDGE:
                    row = await judge_case(case, arm, service, cases_sha256=cases_sha256,
                                           prompt_sha256=prompt_sha, sleep=sleep)
                else:
                    row = await second_pass_case(
                        case, parents[case.features.entry_id], arm, service,
                        cases_sha256=cases_sha256, prompt_sha256=prompt_sha, sleep=sleep,
                    )
                spend.total += row['usd']
                with path.open('a') as handle:
                    handle.write(json.dumps(row, sort_keys=True) + '\n')

        await asyncio.gather(*(one(case) for case in todo))
    rows = read_arm_rows(path)
    if {row['entry_id'] for row in rows} != {case.features.entry_id for case in scope}:
        return False
    _write_sorted(path, rows)
    return True


def _needs_call(case: Any, arm: Arm, parents: Mapping[str, Mapping[str, Any]]) -> bool:
    if arm.kind == KIND_JUDGE:
        return _judged(case, arm)
    parent = parents[case.features.entry_id]
    return parent['judge_outcome'] in _SECOND_PASS_SOURCES and (
        parent['routed_outcome'] == parent['judge_outcome']
    )


def _complete_parent_rows(
    cases: Sequence[Any], arm: Arm, arms_dir: Path,
) -> dict[str, dict[str, Any]] | None:
    """The parent arm's rows by entry id; ``{}`` for a judge arm; None while the parent is partial."""
    if arm.parent is None:
        return {}
    rows = {row['entry_id']: row for row in read_arm_rows(arms_dir / f'{arm.parent}.jsonl')}
    if set(rows) != {case.features.entry_id for case in cases}:
        return None
    return rows


def _write_sorted(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: row['entry_id'])
    scratch = path.with_suffix('.jsonl.tmp')
    scratch.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in ordered))
    os.replace(scratch, path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='replay the named arms, resuming any partial one')
    run.add_argument('--cases', type=Path, required=True, help='the scratch cases JSONL (with texts)')
    run.add_argument('--arms', required=True, help='comma-separated arm names, run in this order')
    run.add_argument('--budget-usd', type=float, required=True,
                     help='list-price spend cap over every arm file in --arms-dir')
    run.add_argument('--arms-dir', type=Path, default=DEFAULT_ARMS_DIR)
    run.add_argument('--concurrency', type=int, default=8)
    run.add_argument('--config', type=Path, default=None,
                     help='a fused-memory config file (sets CONFIG_PATH)')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.config is not None:
        os.environ['CONFIG_PATH'] = str(args.config)
    from fused_memory.config.schema import FusedMemoryConfig

    validate_arms(ARMS)
    arms = [arm_named(name.strip()) for name in args.arms.split(',') if name.strip()]
    result = asyncio.run(run_arms(
        _cases.read_cases(args.cases), arms, config=FusedMemoryConfig(), arms_dir=args.arms_dir,
        budget_usd=args.budget_usd, cases_sha256=cases_digest(args.cases),
        concurrency=args.concurrency,
    ))
    print(result.message, file=sys.stderr if result.exit_code else sys.stdout)
    return result.exit_code


if __name__ == '__main__':
    sys.exit(main())
