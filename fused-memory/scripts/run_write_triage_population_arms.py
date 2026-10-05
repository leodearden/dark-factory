#!/usr/bin/env python3
"""Run the π judge arms over a frozen write-triage population, and publish them.

Task 6151 (π), plans/write-triage-flip-readiness-prd.md §11 D14 and D16. Each
:class:`Arm` is one judge configuration (model, reasoning effort, slate width,
wording); each judge-band write of the snapshot that
``freeze_write_triage_population.py`` froze is judged once per arm, through
the SHIPPED ``write_triage_judge._call_llm`` and parser. A row follows ι's
per-case contract (``score_write_triage_pairs.py::JudgedCase``, C2''), so μ
scores the arm files with ι unchanged.

Usage
-----
Run from ``fused-memory/``. ``run`` is resumable: re-invoke it until it exits
0. Arms run one after another (the wording override is process-global);
calls within an arm run concurrently.

  uv run python scripts/run_write_triage_population_arms.py run \\
      --snapshot <out-root>/write-triage-population-<date>/snapshot.json \\
      --max-writes N --budget-usd 40

Rows land in ``arms/<arm>.jsonl`` beside the snapshot. A long run belongs
detached (``setsid … > log 2>&1``), polled by reading the log.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time
import types
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openai

from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.write_triage import OUTCOME_JUDGE, OUTCOME_STORED, JudgeUsage

# `_call_llm` is the one PRIVATE reach: an arm needs the raw text, the usage of
# an unparseable answer, and parse failures kept apart from transport failures,
# none of which `judge_write` returns. Task 5846 owns the public seam.
from fused_memory.server.write_triage_judge import (
    JudgeOutputError,
    _call_llm,
    build_judge_prompt,
    parse_judge_verdict,
    resolve_judge_field_chars,
    resolve_judge_provider,
    resolve_judge_timeout,
    select_judge_candidates,
)

_SCRIPTS = Path(__file__).resolve().parent


def _load_script(path: Path, mod_name: str) -> types.ModuleType:
    """Load a ``scripts/`` sibling by path, cached in ``sys.modules``."""
    cached = sys.modules.get(mod_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_freeze = _load_script(
    _SCRIPTS / 'freeze_write_triage_population.py', 'freeze_write_triage_population',
)
_wording = _load_script(_SCRIPTS / 'write_triage_judge_wording.py', 'write_triage_judge_wording')
_scorer = _load_script(_SCRIPTS / 'score_write_triage_pairs.py', 'score_write_triage_pairs')

logger = logging.getLogger(__name__)

#: The only provider the arms are priced and requested for.
_ARM_PROVIDER = 'openai'


@dataclass(frozen=True)
class Arm:
    """One judge configuration the population is run under."""

    model: str
    reasoning_effort: str | None
    width: int
    wording: str

    @property
    def name(self) -> str:
        """Unique and filename-safe, e.g. ``gpt-6.1-sol:low@20`` or ``gpt-4o-mini@5+pre-psi``."""
        effort = '' if self.reasoning_effort is None else f':{self.reasoning_effort}'
        wording = '' if self.wording == _wording.WORDING_SHIPPED else f'+{self.wording}'
        return f'{self.model}{effort}@{self.width}{wording}'


_SHIPPED = _wording.WORDING_SHIPPED
_PRE_PSI = _wording.WORDING_PRE_PSI

#: The ten arms, in cost order: gpt-4o-mini under both wordings, then the
#: three frontier arms under the shipped one, each at widths 5 and 20.
ARMS: tuple[Arm, ...] = (
    Arm('gpt-4o-mini', None, 5, _SHIPPED),
    Arm('gpt-4o-mini', None, 5, _PRE_PSI),
    Arm('gpt-4o-mini', None, 20, _SHIPPED),
    Arm('gpt-4o-mini', None, 20, _PRE_PSI),
    Arm('gpt-6-luna', 'low', 5, _SHIPPED),
    Arm('gpt-6-luna', 'low', 20, _SHIPPED),
    Arm('gpt-5.6-terra', 'none', 5, _SHIPPED),
    Arm('gpt-5.6-terra', 'none', 20, _SHIPPED),
    Arm('gpt-6.1-sol', 'low', 5, _SHIPPED),
    Arm('gpt-6.1-sol', 'low', 20, _SHIPPED),
)


def _as_memory_result(candidate: Mapping[str, Any]) -> MemoryResult:
    """A frozen slate record as the row the shipped selector reads (``store_score`` included)."""
    return MemoryResult(
        id=candidate['memory_id'],
        content=candidate.get('content') or '',
        source_store=SourceStore.mem0,
        metadata=dict(candidate.get('metadata') or {}),
    )


def _usage_row(usage: JudgeUsage | None) -> dict[str, int | None] | None:
    if usage is None:
        return None
    return {
        'prompt_tokens': usage.input_tokens,
        'completion_tokens': usage.output_tokens,
        'reasoning_tokens': usage.reasoning_tokens,
    }


def _usd(model: str, usage: Mapping[str, Any] | None) -> float | None:
    price = _scorer.LIST_PRICES.get(model)
    if price is None or usage is None:
        return None
    return price.usd(usage['prompt_tokens'], usage['completion_tokens'])


def _identity(write: Mapping[str, Any], arm: Arm, snapshot_sha256: str) -> dict[str, Any]:
    """What a row says about the arm and the write, before any answer."""
    return {
        'arm': arm.name,
        'judge_model': arm.model,
        'reasoning_effort': arm.reasoning_effort,
        'width': arm.width,
        'wording': arm.wording,
        'snapshot_sha256': snapshot_sha256,
        'memory_id': write['memory_id'],
        'project_id': write['project_id'],
        'category': write['category'],
        'recon_marker': write['recon_marker'],
        'declares_attach_keys': write['declares_attach_keys'],
        'band': write['band'],
        'band_winner_id': write['band_winner_id'],
        'attempts': 1,
        'transport_failures': [],
    }


def _answer(
    *,
    outcome: str,
    verdict_candidate_id: str | None,
    judged_candidate_id: str | None,
    raw_text: str | None,
    usage: dict[str, int | None] | None,
    model: str,
    seconds: float,
    failure: JudgeOutputError | None,
) -> dict[str, Any]:
    """What a row says about the answer."""
    return {
        'outcome': outcome,
        'verdict_candidate_id': verdict_candidate_id,
        'judged_candidate_id': judged_candidate_id,
        'raw_text': raw_text,
        'usage': usage,
        'usd': _usd(model, usage),
        'judge_seconds': seconds,
        'parse_failure': failure is not None,
        'failure': None if failure is None else f'{type(failure).__name__}: {failure}',
    }


def _fail_open(
    model: str,
    failure: JudgeOutputError,
    *,
    raw_text: str | None,
    usage: dict[str, int | None] | None,
    seconds: float,
) -> dict[str, Any]:
    """An answer that is not a verdict: ``stored``, naming nothing, kept as a parse failure."""
    return _answer(
        outcome=OUTCOME_STORED, verdict_candidate_id=None, judged_candidate_id=None,
        raw_text=raw_text, usage=usage, model=model, seconds=seconds, failure=failure,
    )


async def judge_for_arm(
    write: Mapping[str, Any],
    arm: Arm,
    *,
    service: Any,
    snapshot_sha256: str,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Judge one judge-band *write* under *arm*; return its case row.

    Selection and rendering are the shipped ones at the arm's width and the
    configured field cap. A :class:`JudgeOutputError` (an unparseable reply,
    or an incomplete response) is the fail-open ``stored`` row with
    ``parse_failure``; every other exception propagates to the runner. The
    caller holds the arm's wording in force.
    """
    provider = resolve_judge_provider(service)
    if provider != _ARM_PROVIDER:
        raise ValueError(f'the arms run on {_ARM_PROVIDER!r}; the config resolves {provider!r}')
    if write['band'] != OUTCOME_JUDGE:
        raise ValueError(f'write {write["memory_id"]} is in band {write["band"]!r}, not the judge')
    selected = select_judge_candidates(
        [_as_memory_result(c) for c in write['candidates']], arm.width,
        canonical_id=write['band_winner_id'],
    )
    prompt = build_judge_prompt(
        write['content'], selected, field_chars=resolve_judge_field_chars(service),
    )
    identity = _identity(write, arm, snapshot_sha256)
    started = clock()
    try:
        reply = await _call_llm(
            provider=provider, model=arm.model, prompt=prompt, memory_service=service,
            timeout=resolve_judge_timeout(service), reasoning_effort=arm.reasoning_effort,
        )
    except JudgeOutputError as exc:
        seconds = clock() - started
        return identity | _fail_open(arm.model, exc, raw_text=None, usage=None, seconds=seconds)
    seconds = clock() - started
    usage = _usage_row(reply.usage)
    try:
        verdict = parse_judge_verdict(reply.text, [c.id for c in selected])
    except JudgeOutputError as exc:
        return identity | _fail_open(
            arm.model, exc, raw_text=reply.text, usage=usage, seconds=seconds,
        )
    canonical_of = {c['memory_id']: c['canonical_id'] for c in write['candidates']}
    return identity | _answer(
        outcome=verdict.outcome,
        verdict_candidate_id=verdict.candidate_id,
        judged_candidate_id=(
            None if verdict.candidate_id is None else canonical_of[verdict.candidate_id]
        ),
        raw_text=reply.text, usage=usage, model=arm.model, seconds=seconds, failure=None,
    )


# --- the runner ----------------------------------------------------------------

#: Transport failures worth another attempt. Anything else (auth, a 400, a
#: mis-configuration) would recur on every write, so it aborts the run.
_TRANSIENT = (
    TimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)

#: The first retry's delay; each further retry doubles it.
_BACKOFF_SECONDS = 2.0


def judge_band_order(
    snapshot: Mapping[str, Any], snapshot_sha256: str,
) -> list[dict[str, Any]]:
    """The judge-band writes, ordered by ``sha256(f'{snapshot_sha256}:{memory_id}')``.

    Derived from the snapshot's own hash, so no seed is stored, and any prefix
    is a deterministic sample.
    """
    def rank(write: Mapping[str, Any]) -> str:
        return hashlib.sha256(f'{snapshot_sha256}:{write["memory_id"]}'.encode()).hexdigest()

    return sorted((w for w in snapshot['writes'] if w['band'] == OUTCOME_JUDGE), key=rank)


def arm_path(arms_dir: Path, arm_name: str) -> Path:
    return Path(arms_dir) / f'{arm_name}.jsonl'


def read_rows(path: Path) -> list[dict[str, Any]]:
    """The rows of one arm file, or ``[]`` when it does not exist yet."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def _rows_of_this_snapshot(
    arms_dir: Path, arm_name: str, snapshot_sha256: str,
) -> dict[str, dict[str, Any]]:
    rows = read_rows(arm_path(arms_dir, arm_name))
    foreign = sorted({str(row.get('snapshot_sha256')) for row in rows} - {snapshot_sha256})
    if foreign:
        raise ValueError(
            f'arm {arm_name} holds rows from another snapshot ({", ".join(foreign)}); '
            f'this run is snapshot {snapshot_sha256}',
        )
    return {row['memory_id']: row for row in rows}


def _spent_usd(arms_dir: Path) -> float:
    """List-price spend recorded in every arm file under *arms_dir*."""
    return sum(
        row.get('usd') or 0.0
        for path in sorted(Path(arms_dir).glob('*.jsonl'))
        for row in read_rows(path)
    )


@dataclass
class _Budget:
    """The run's shared spend tally against its hard cap."""

    spent: float
    cap: float
    refused: bool = False

    def allows_dispatch(self) -> bool:
        if self.spent >= self.cap:
            self.refused = True
            return False
        return True


async def _judge_with_retries(
    write: Mapping[str, Any],
    arm: Arm,
    *,
    service: Any,
    snapshot_sha256: str,
    sleep: Callable[[float], Awaitable[Any]],
    max_attempts: int,
) -> dict[str, Any] | None:
    """:func:`judge_for_arm`, retrying transient transport failures; ``None`` when exhausted."""
    failures: list[str] = []
    for attempt in range(max_attempts):
        try:
            row = await judge_for_arm(
                write, arm, service=service, snapshot_sha256=snapshot_sha256,
            )
        except _TRANSIENT as exc:
            failures.append(type(exc).__name__)
            if attempt + 1 < max_attempts:
                await sleep(_BACKOFF_SECONDS * 2 ** attempt)
            continue
        return row | {'attempts': attempt + 1, 'transport_failures': failures}
    logger.warning(
        'arm %s: write %s got no row after %d attempts (%s)',
        arm.name, write['memory_id'], max_attempts, ', '.join(failures),
    )
    return None


async def _run_arm(
    arm: Arm,
    todo: Sequence[Mapping[str, Any]],
    sink_path: Path,
    budget: _Budget,
    *,
    service: Any,
    snapshot_sha256: str,
    concurrency: int,
    sleep: Callable[[float], Awaitable[Any]],
    max_attempts: int,
) -> int:
    """Judge *todo* under *arm*, appending each row as it lands; return how many landed.

    A non-transient error stops further dispatch, lets in-flight calls finish
    and land, then propagates.
    """
    semaphore = asyncio.Semaphore(concurrency)
    aborted = asyncio.Event()
    landed = 0
    with sink_path.open('a', encoding='utf-8') as sink:

        async def one(write: Mapping[str, Any]) -> None:
            nonlocal landed
            async with semaphore:
                if aborted.is_set() or not budget.allows_dispatch():
                    return
                try:
                    row = await _judge_with_retries(
                        write, arm, service=service, snapshot_sha256=snapshot_sha256,
                        sleep=sleep, max_attempts=max_attempts,
                    )
                except Exception:
                    aborted.set()
                    raise
                if row is None:
                    return
                sink.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n')
                sink.flush()
                budget.spent += row['usd'] or 0.0
                landed += 1

        outcomes = await asyncio.gather(*(one(w) for w in todo), return_exceptions=True)
    errors = [o for o in outcomes if isinstance(o, BaseException)]
    if errors:
        raise errors[0]
    return landed


async def run_arms(
    snapshot: Mapping[str, Any],
    snapshot_sha256: str,
    arms_dir: Path,
    *,
    arms: Sequence[Arm],
    service: Any,
    max_writes: int | None,
    concurrency: int,
    budget_usd: float,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    max_attempts: int = 4,
) -> dict[str, Any]:
    """Run *arms* in order over the first *max_writes* of :func:`judge_band_order`.

    Resumable: a row already in ``arms_dir/<arm>.jsonl`` is never re-judged.
    Dispatch stops once the list-price spend of every arm file reaches
    *budget_usd*. Returns per-arm coverage of the run set, whether the budget
    refused a dispatch, and the spend.
    """
    too_wide = [arm.name for arm in arms if arm.width > snapshot['candidate_k']]
    if too_wide:
        raise ValueError(
            f'arms {", ".join(too_wide)} are wider than the frozen slates '
            f'(candidate_k={snapshot["candidate_k"]})',
        )
    run_set = judge_band_order(snapshot, snapshot_sha256)[:max_writes]
    arms_dir = Path(arms_dir)
    arms_dir.mkdir(parents=True, exist_ok=True)
    existing = {arm.name: _rows_of_this_snapshot(arms_dir, arm.name, snapshot_sha256) for arm in arms}
    budget = _Budget(spent=_spent_usd(arms_dir), cap=budget_usd)
    coverage: dict[str, dict[str, Any]] = {}
    for arm in arms:
        todo = [w for w in run_set if w['memory_id'] not in existing[arm.name]]
        logger.info('arm %s: %d of %d writes to judge', arm.name, len(todo), len(run_set))
        with _wording.judge_wording(arm.wording):
            landed = await _run_arm(
                arm, todo, arm_path(arms_dir, arm.name), budget, service=service,
                snapshot_sha256=snapshot_sha256, concurrency=concurrency, sleep=sleep,
                max_attempts=max_attempts,
            )
        covered = {r['memory_id'] for r in read_rows(arm_path(arms_dir, arm.name))}
        rows = sum(1 for w in run_set if w['memory_id'] in covered)
        coverage[arm.name] = {
            'complete': rows == len(run_set),
            'rows': rows,
            'missing': len(run_set) - rows,
            'written_now': landed,
        }
    return {'arms': coverage, 'budget_exhausted': budget.refused, 'spent_usd': budget.spent}


# --- CLI -----------------------------------------------------------------------

def _command_run(args: argparse.Namespace) -> int:
    from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415

    snapshot, snapshot_sha256 = _freeze.load_snapshot(args.snapshot)
    requested = set(args.arms or [arm.name for arm in ARMS])
    summary = asyncio.run(run_arms(
        snapshot, snapshot_sha256, Path(args.snapshot).parent / 'arms',
        arms=[arm for arm in ARMS if arm.name in requested],
        service=types.SimpleNamespace(config=FusedMemoryConfig()),
        max_writes=args.max_writes, concurrency=args.concurrency,
        budget_usd=args.budget_usd,
    ))
    print(json.dumps(summary, indent=2))
    gaps = [f'{name} missing {c["missing"]}' for name, c in summary['arms'].items()
            if not c['complete']]
    if summary['budget_exhausted'] or gaps:
        print(
            f'incomplete: budget_exhausted={summary["budget_exhausted"]}; '
            f'{"; ".join(gaps) or "every requested arm complete"}',
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='judge the run set under each arm (resumable)')
    run.add_argument('--config', default=None,
                     help='path to a fused-memory config file (sets CONFIG_PATH)')
    run.add_argument('--snapshot', type=Path, required=True,
                     help="a frozen snapshot.json; rows land in its directory's arms/")
    run.add_argument('--max-writes', dest='max_writes', type=int, default=None,
                     help='judge only this prefix of the judge-band order (default: all)')
    run.add_argument('--concurrency', type=int, default=8,
                     help='calls in flight within one arm (default: 8)')
    run.add_argument('--budget-usd', dest='budget_usd', type=float, default=40.0,
                     help='hard list-price cap over every arm file (default: 40.0)')
    run.add_argument('--arms', nargs='+', choices=[arm.name for arm in ARMS], default=None,
                     help='arms to run, in ARMS order whatever the order given (default: all)')
    run.set_defaults(handler=_command_run)
    args = parser.parse_args(argv)
    if getattr(args, 'config', None):
        os.environ['CONFIG_PATH'] = str(args.config)
    return args.handler(args)


if __name__ == '__main__':
    raise SystemExit(main())
