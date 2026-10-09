#!/usr/bin/env python3
"""Run the π judge arms over a frozen write-triage population, and publish them.

Task 6151 (π), plans/write-triage-flip-readiness-prd.md §11 D14 and D16. Each
:class:`Arm` is one judge configuration (model, reasoning effort, slate width,
wording); each judge-band write of the snapshot that
``freeze_write_triage_population.py`` froze is judged once per arm, through
the SHIPPED ``write_triage_judge._call_llm`` and parser. A row follows ι's
per-case contract (``score_write_triage_pairs.py::JudgedCase``, C2''), so μ
scores the arm files with ι unchanged. Task 6530 (π2, §12 D19) adds a
write-time mode: the same sample, each slate cut to the records created
before its write and its band re-decided.

Usage
-----
Run from ``fused-memory/``. ``run`` is resumable: re-invoke it until it exits
0. Arms run one after another (the wording override is process-global);
calls within an arm run concurrently.

  uv run python scripts/run_write_triage_population_arms.py run \\
      --snapshot <out-root>/write-triage-population-<date>/snapshot.json \\
      --max-writes N --budget-usd 40

Rows land in ``arms/<arm>.jsonl`` beside the snapshot. The write-time run
(π2) judges the same sample with its own arm table into ``arms-write-time/``:

  uv run python scripts/run_write_triage_population_arms.py run --slates write-time \\
      --snapshot <snapshot.json> --max-writes 658 --budget-usd 15

A long run belongs detached (``setsid … > log 2>&1``), polled by reading the
log. ``publish`` then refuses anything partial and writes the two committed
artifacts:

  uv run python scripts/run_write_triage_population_arms.py publish \\
      --snapshot <snapshot.json> \\
      --fixture-report shipped=<dir>/fixture-shipped.json \\
      --fixture-report pre-psi=<dir>/fixture-pre-psi.json

and ``publish-write-time`` does the same for the write-time run:

  uv run python scripts/run_write_triage_population_arms.py publish-write-time \\
      --snapshot <snapshot.json> --max-writes 658 --budget-usd 15
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import importlib.util
import json
import logging
import math
import os
import statistics
import sys
import time
import types
from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import openai

from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.write_triage import (
    OUTCOME_CONTESTED,
    OUTCOME_JUDGE,
    OUTCOME_STORED,
    TRIAGE_OUTCOMES,
    JudgeUsage,
    decide_band,
)

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
_PACKAGE_ROOT = _SCRIPTS.parent


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
_calibrate = _load_script(_SCRIPTS / 'calibrate_write_triage.py', 'calibrate_write_triage')

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


class Slates(StrEnum):
    """Which slate a write is judged on."""

    #: The slate as frozen 2026-10-05, records created after the write included (π).
    FROZEN = 'frozen'
    #: Only records created strictly before the write, band re-decided (π2, PRD §12 D19).
    WRITE_TIME = 'write-time'


#: π2's arms (PRD §12.4): the flip-decision arms at width 5; sol:low@20 is over budget.
WRITE_TIME_ARMS: tuple[Arm, ...] = tuple(
    arm for arm in ARMS
    if arm.name in {'gpt-4o-mini@5', 'gpt-5.6-terra:none@5', 'gpt-6.1-sol:low@5'}
)
#: Where each slate mode's arm files live, beside the snapshot.
ARMS_DIR_NAME: dict[Slates, str] = {Slates.FROZEN: 'arms', Slates.WRITE_TIME: 'arms-write-time'}
#: The arms each slate mode runs.
ARMS_OF: dict[Slates, tuple[Arm, ...]] = {Slates.FROZEN: ARMS, Slates.WRITE_TIME: WRITE_TIME_ARMS}


def _as_memory_result(candidate: Mapping[str, Any]) -> MemoryResult:
    """A frozen slate record as the row the shipped selector reads (``store_score`` included)."""
    return MemoryResult(
        id=candidate['memory_id'],
        content=candidate.get('content') or '',
        source_store=SourceStore.mem0,
        metadata=dict(candidate.get('metadata') or {}),
    )


def write_time_slate(
    write: Mapping[str, Any], *, t_high: float, t_low: float,
) -> dict[str, Any]:
    """*write* as it was at write time: a new dict, *write* untouched.

    A candidate is kept only when its ``created_at`` is an instant strictly
    before the write's; one whose instant cannot be parsed cannot be shown
    earlier, so it is dropped. The band, its winner and the similarity are
    then re-decided by the shipped ``decide_band`` at *t_high* / *t_low*.
    """
    written = _freeze.parse_created_at(write['created_at'])
    if written is None:
        raise ValueError(
            f'write {write["memory_id"]} has no aware created_at ({write["created_at"]!r}); '
            'the freeze excludes undated writes',
        )
    kept = [
        candidate for candidate in write['candidates']
        if (instant := _freeze.parse_created_at(candidate.get('created_at'))) is not None
        and instant < written
    ]
    decision = decide_band([_as_memory_result(c) for c in kept], t_high=t_high, t_low=t_low)
    return {
        **write,
        'slates': Slates.WRITE_TIME,
        'candidates': kept,
        'band': decision.outcome,
        'band_winner_id': decision.canonical_id,
        'similarity': decision.similarity,
    }


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


def _identity(
    write: Mapping[str, Any], arm: Arm, snapshot_sha256: str, *,
    field_chars: int, timeout_seconds: float,
) -> dict[str, Any]:
    """What a row says about the arm, the call settings and the write, before any answer."""
    return {
        'arm': arm.name,
        'judge_model': arm.model,
        'reasoning_effort': arm.reasoning_effort,
        'width': arm.width,
        'wording': arm.wording,
        'snapshot_sha256': snapshot_sha256,
        'field_chars': field_chars,
        'timeout_seconds': timeout_seconds,
        'memory_id': write['memory_id'],
        'project_id': write['project_id'],
        'category': write['category'],
        'recon_marker': write['recon_marker'],
        'declares_attach_keys': write['declares_attach_keys'],
        'band': write['band'],
        'band_winner_id': write['band_winner_id'],
        'slates': write['slates'],
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
    field_chars = resolve_judge_field_chars(service)
    timeout_seconds = resolve_judge_timeout(service)
    prompt = build_judge_prompt(write['content'], selected, field_chars=field_chars)
    identity = _identity(
        write, arm, snapshot_sha256, field_chars=field_chars, timeout_seconds=timeout_seconds,
    )
    started = clock()
    try:
        reply = await _call_llm(
            provider=provider, model=arm.model, prompt=prompt, memory_service=service,
            timeout=timeout_seconds, reasoning_effort=arm.reasoning_effort,
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


@dataclass(frozen=True)
class RunSet:
    """The writes a run judges under its slates, and the drawn writes those slates moved out."""

    slates: Slates
    sample_size: int
    writes: tuple[dict[str, Any], ...]
    left_judge_band: tuple[str, ...]


def draw_run_set(
    snapshot: Mapping[str, Any], snapshot_sha256: str, *,
    max_writes: int | None, slates: Slates,
) -> RunSet:
    """The first *max_writes* of the frozen judge-band order, each viewed under *slates*.

    A drawn write the view moves out of the judge band is listed, never
    replaced by the next write of the order, so every mode judges π's sample.
    """
    sample = judge_band_order(snapshot, snapshot_sha256)[:max_writes]
    match slates:
        case Slates.FROZEN:
            viewed = [{**write, 'slates': Slates.FROZEN} for write in sample]
        case Slates.WRITE_TIME:
            viewed = [
                write_time_slate(write, t_high=snapshot['t_high'], t_low=snapshot['t_low'])
                for write in sample
            ]
    return RunSet(
        slates=slates,
        sample_size=len(sample),
        writes=tuple(view for view in viewed if view['band'] == OUTCOME_JUDGE),
        left_judge_band=tuple(view['memory_id'] for view in viewed if view['band'] != OUTCOME_JUDGE),
    )


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
    """The run's list-price tally of landed rows, and the cap that stops dispatch.

    A call is dispatched only while the tally is under the cap, and counted
    only once its row lands. So calls already in flight when the cap is
    crossed still land (the overshoot is under concurrency x the costliest
    call), and a transient failure the provider billed is never counted.
    """

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
    slates: Slates = Slates.FROZEN,
) -> dict[str, Any]:
    """Run *arms* in order over the first *max_writes* of :func:`judge_band_order`.

    Each write is judged on its *slates* view (:func:`draw_run_set`), and only
    the arms of ``ARMS_OF[slates]`` may run.
    Resumable: a row already in ``arms_dir/<arm>.jsonl`` is never re-judged.
    Dispatch stops once the list-price spend of every arm file reaches
    *budget_usd*; calls in flight then still land, so the spend can pass it
    by up to *concurrency* calls (see :class:`_Budget`). Returns per-arm
    coverage of the run set, whether the budget refused a dispatch, the
    spend, the slates, the sample size and the drawn writes that left the band.
    """
    off_table = [arm.name for arm in arms if arm not in ARMS_OF[slates]]
    if off_table:
        raise ValueError(f'arms {", ".join(off_table)} are not run on {slates} slates')
    too_wide = [arm.name for arm in arms if arm.width > snapshot['candidate_k']]
    if too_wide:
        raise ValueError(
            f'arms {", ".join(too_wide)} are wider than the frozen slates '
            f'(candidate_k={snapshot["candidate_k"]})',
        )
    drawn = draw_run_set(snapshot, snapshot_sha256, max_writes=max_writes, slates=slates)
    run_set = drawn.writes
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
    return {
        'arms': coverage, 'budget_exhausted': budget.refused, 'spent_usd': budget.spent,
        'slates': slates, 'sample_size': drawn.sample_size,
        'left_judge_band': list(drawn.left_judge_band),
    }


# --- publish -------------------------------------------------------------------

_RUN_SET_ORDER = 'sha256(snapshot_sha256:memory_id)'
_PAIR_ORDER = 'sha256(snapshot_sha256:entry_id:target_id)'
#: The rater brief's per-text cap and truncation marker.
_PAIR_TEXT_CHARS = 4_000
_ATTACH_OUTCOMES = TRIAGE_OUTCOMES - {OUTCOME_STORED}
#: Provenance two fixture reports must share for their wordings to be compared.
_MATCHED_FIXTURE_PROVENANCE = ('field_chars', 'slate_mode', 'project_id', 'judge_model')

DEFAULT_POPULATION_OUT = _PACKAGE_ROOT / 'calibration' / 'write_triage_population.json'
DEFAULT_PAIRS_OUT = _PACKAGE_ROOT / 'calibration' / 'write_triage_pairs_to_rate.jsonl'
DEFAULT_ALREADY_RATED = _PACKAGE_ROOT / 'tests' / 'fixtures' / 'write_triage_pair_verdicts_seed.jsonl'
DEFAULT_WRITE_TIME_POPULATION_OUT = (
    _PACKAGE_ROOT / 'calibration' / 'write_triage_population_write_time.json'
)
DEFAULT_WRITE_TIME_PAIRS_OUT = (
    _PACKAGE_ROOT / 'calibration' / 'write_triage_pairs_to_rate_write_time.jsonl'
)
#: The append-only verdict corpus the raters' answers land in.
DEFAULT_VERDICT_CORPUS = _PACKAGE_ROOT / 'calibration' / 'write_triage_pair_verdicts.jsonl'

#: How the write-time artifact's slates were cut, as the artifact states it.
WRITE_TIME_RULE = (
    "Each write is judged on the candidates of its frozen slate whose created_at is strictly "
    "before the write's (an unparseable created_at cannot be shown earlier, so it is dropped); "
    "band, band winner and similarity are re-decided by the shipped decide_band at the "
    "snapshot's t_high and t_low. Where at least n candidates remain, the top n equal "
    "production's write-time top n among records still live at the freeze. The sample is the "
    "frozen judge-band prefix; a sampled write that leaves the judge band is listed, never "
    "replaced."
)


def _sha256(data: bytes | str) -> str:
    return hashlib.sha256(data.encode('utf-8') if isinstance(data, str) else data).hexdigest()


def repo_relative(path: Path) -> str:
    """*path* relative to the checkout that holds it (the first parent with ``.git``)."""
    resolved = Path(path).resolve()
    for parent in resolved.parents:
        if (parent / '.git').exists():
            return resolved.relative_to(parent).as_posix()
    raise ValueError(f'{path} is not inside a git checkout')


def _count(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(collections.Counter(row[key] for row in rows).items()))


def _validated_rows(
    arms: Sequence[Arm],
    arm_rows_by_name: Mapping[str, Sequence[Mapping[str, Any]]],
    snapshot_sha256: str,
    expected_ids: Collection[str],
    *,
    run_set: str,
) -> None:
    """Refuse, naming the arm, any of *arms* not covering exactly *expected_ids* once each.

    *run_set* describes the expected writes for the refusal message.
    """
    for arm in arms:
        if not arm_rows_by_name.get(arm.name):
            raise ValueError(f'arm {arm.name} has no rows to publish')
    for arm in arms:
        rows = arm_rows_by_name[arm.name]
        foreign = sorted({str(row.get('snapshot_sha256')) for row in rows} - {snapshot_sha256})
        if foreign:
            raise ValueError(f'arm {arm.name} holds rows from another snapshot: {foreign}')
        if len({row['memory_id'] for row in rows}) != len(rows):
            raise ValueError(f'arm {arm.name} judged a write more than once')
    for arm in arms:
        if {row['memory_id'] for row in arm_rows_by_name[arm.name]} != set(expected_ids):
            raise ValueError(f'arm {arm.name} does not cover {run_set}')


def _validated_run_set(
    snapshot: Mapping[str, Any], snapshot_sha256: str,
    arm_rows_by_name: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """The run set every arm covers, refusing anything that would read as partial."""
    first_arm_rows = arm_rows_by_name.get(ARMS[0].name) or []
    prefix = judge_band_order(snapshot, snapshot_sha256)[:len(first_arm_rows)]
    _validated_rows(
        ARMS, arm_rows_by_name, snapshot_sha256, {write['memory_id'] for write in prefix},
        run_set=f'the common prefix of the judge-band order, its first {len(prefix)} writes',
    )
    return prefix


def _validated_write_time_run_set(
    snapshot: Mapping[str, Any], snapshot_sha256: str,
    arm_rows_by_name: Mapping[str, Sequence[Mapping[str, Any]]],
    sample_size: int,
) -> RunSet:
    """The write-time run set every arm covers, refusing anything partial or on other slates."""
    run_set = draw_run_set(
        snapshot, snapshot_sha256, max_writes=sample_size, slates=Slates.WRITE_TIME,
    )
    _validated_rows(
        WRITE_TIME_ARMS, arm_rows_by_name, snapshot_sha256,
        {write['memory_id'] for write in run_set.writes},
        run_set=(
            f'the {len(run_set.writes)} of the first {run_set.sample_size} writes of the '
            'judge-band order still in the judge band at write time'
        ),
    )
    for arm in WRITE_TIME_ARMS:
        stray = sorted(
            {str(row.get('slates')) for row in arm_rows_by_name[arm.name]} - {Slates.WRITE_TIME},
        )
        if stray:
            raise ValueError(
                f'arm {arm.name} holds rows judged on {", ".join(stray)} slates, '
                f'not {Slates.WRITE_TIME}',
            )
    return run_set


def _has_later_candidate(write: Mapping[str, Any]) -> bool:
    written = _freeze.parse_created_at(write['created_at'])
    instants = (_freeze.parse_created_at(c.get('created_at')) for c in write['candidates'])
    return written is not None and any(i is not None and i > written for i in instants)


def _population_block(
    snapshot: Mapping[str, Any], snapshot_sha256: str, snapshot_path: Path,
    run_set: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    writes = snapshot['writes']
    frozen = sum(1 for write in writes if write['band'] == OUTCOME_JUDGE)
    return {
        'n_writes': len(writes),
        'n_judge_band': len(run_set),
        'n_judge_band_frozen': frozen,
        'judge_band_sample': None if len(run_set) == frozen else {
            'order': _RUN_SET_ORDER, 'size': len(run_set), 'of': frozen,
        },
        'projects': list(snapshot['projects']),
        'frozen_at': snapshot['frozen_at'],
        'snapshot_sha256': snapshot_sha256,
        'snapshot_path': repo_relative(snapshot_path),
        'by_category': _count(writes, 'category'),
        'by_project': _count(writes, 'project_id'),
        'by_band': _count(writes, 'band'),
        'recon_marker_writes': sum(1 for write in writes if write['recon_marker']),
        'recon_marker_run_set': sum(1 for write in run_set if write['recon_marker']),
        'declares_attach_keys_writes': sum(1 for w in writes if w['declares_attach_keys']),
        **_freeze.exclusions(snapshot),
        'judge_band_with_later_candidates': sum(1 for w in run_set if _has_later_candidate(w)),
    }


def _write_time_population_block(
    snapshot: Mapping[str, Any], snapshot_sha256: str, snapshot_path: Path, run_set: RunSet,
) -> dict[str, Any]:
    frozen = {write['memory_id']: write for write in snapshot['writes']}
    order = judge_band_order(snapshot, snapshot_sha256)
    sample = order[:run_set.sample_size]
    judged = run_set.writes
    leavers = [
        write_time_slate(frozen[memory_id], t_high=snapshot['t_high'], t_low=snapshot['t_low'])
        for memory_id in run_set.left_judge_band
    ]
    sizes = [len(write['candidates']) for write in judged]
    return {
        'slates': Slates.WRITE_TIME,
        'slates_rule': WRITE_TIME_RULE,
        'n_writes': len(snapshot['writes']),
        'n_judge_band_frozen': len(order),
        'judge_band_sample': {'order': _RUN_SET_ORDER, 'size': len(sample), 'of': len(order)},
        'n_judge_band_write_time': len(judged),
        'excluded_by_write_time_filter': {
            'count': len(leavers),
            'memory_ids': list(run_set.left_judge_band),
            'by_band': _count(leavers, 'band'),
        },
        'band_winner_changed': sum(
            1 for w in judged if w['band_winner_id'] != frozen[w['memory_id']]['band_winner_id']
        ),
        'writes_with_later_candidates': sum(1 for w in sample if _has_later_candidate(w)),
        'later_candidates_dropped': sum(
            len(frozen[w['memory_id']]['candidates']) - len(w['candidates']) for w in judged
        ),
        'write_time_slate_size': {
            'min': min(sizes), 'median': statistics.median(sizes), 'max': max(sizes),
        },
        'recon_marker_run_set': sum(1 for w in judged if w['recon_marker']),
        'declares_attach_keys_run_set': sum(1 for w in judged if w['declares_attach_keys']),
        'projects': list(snapshot['projects']),
        'frozen_at': snapshot['frozen_at'],
        'snapshot_sha256': snapshot_sha256,
        'snapshot_path': repo_relative(snapshot_path),
        't_high': snapshot['t_high'],
        't_low': snapshot['t_low'],
    }


def _one_value(arm: Arm, rows: Sequence[Mapping[str, Any]], key: str) -> Any:
    values = {row.get(key) for row in rows}
    if len(values) != 1:
        raise ValueError(f'arm {arm.name} rows disagree on {key}: {sorted(map(str, values))}')
    return values.pop()


def _arm_row(arm: Arm, rows: Sequence[Mapping[str, Any]], cases_path: Path) -> dict[str, Any]:
    seconds = _calibrate.summarize_distribution(
        [row['judge_seconds'] for row in rows if row['judge_seconds'] is not None],
    )
    failures = collections.Counter(name for row in rows for name in row['transport_failures'])
    priced = [row['usd'] for row in rows if row['usd'] is not None]
    return {
        'arm': arm.name,
        'model': arm.model,
        'provider': _ARM_PROVIDER,
        'reasoning_effort': arm.reasoning_effort,
        'width': arm.width,
        'wording': arm.wording,
        'system_prompt_sha256': _sha256(_wording.system_prompt(arm.wording)),
        'field_chars': _one_value(arm, rows, 'field_chars'),
        'timeout_seconds': _one_value(arm, rows, 'timeout_seconds'),
        'calls': len(rows),
        'parse_failures': sum(1 for row in rows if row['parse_failure']),
        'transport_failures': sum(failures.values()),
        'transport_failures_by_type': dict(sorted(failures.items())),
        'usd': round(math.fsum(priced), 6),
        'unpriced_calls': len(rows) - len(priced),
        'p50_seconds': seconds['median'],
        'p95_seconds': seconds['p95'],
        'outcomes': _count(rows, 'outcome'),
        'cases_path': repo_relative(cases_path),
        'cases_sha256': _sha256(cases_path.read_bytes()),
    }


def _spend_block(arm_rows: Sequence[Mapping[str, Any]], budget_usd: float) -> dict[str, Any]:
    return {
        'usd_total': round(math.fsum(row['usd'] for row in arm_rows), 6),
        'budget_usd': budget_usd,
        'list_prices_as_of': _scorer.LIST_PRICES_AS_OF,
        'list_prices_source': _scorer.LIST_PRICES_SOURCE,
    }


def _wording_side(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    attaches = sum(1 for row in rows if row['outcome'] in _ATTACH_OUTCOMES)
    contested = sum(1 for row in rows if row['outcome'] == OUTCOME_CONTESTED)
    return {
        'outcomes': _count(rows, 'outcome'),
        'attaches': attaches,
        'contested': contested,
        'contested_share_of_attaches': contested / attaches if attaches else None,
    }


def _wording_population(
    arm_rows_by_name: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Each non-shipped arm against its shipped twin on the same writes, keyed by its name.

    Label-free: contested shares and the paired discordance, with ι's exact
    McNemar p on the discordant counts.
    """
    readout: dict[str, Any] = {}
    for arm in ARMS:
        if arm.wording == _SHIPPED:
            continue
        twin = Arm(arm.model, arm.reasoning_effort, arm.width, _SHIPPED)
        shipped, other = arm_rows_by_name[twin.name], arm_rows_by_name[arm.name]
        contested_shipped = {r['memory_id'] for r in shipped if r['outcome'] == OUTCOME_CONTESTED}
        contested_other = {r['memory_id'] for r in other if r['outcome'] == OUTCOME_CONTESTED}
        only_shipped = len(contested_shipped - contested_other)
        only_other = len(contested_other - contested_shipped)
        readout[arm.name] = {
            'shipped_arm': twin.name,
            'other_arm': arm.name,
            _SHIPPED: _wording_side(shipped),
            arm.wording: _wording_side(other),
            'contested_only_shipped': only_shipped,
            f'contested_only_{arm.wording.replace("-", "_")}': only_other,
            # A private reach into ι, the one home of the flip-gate statistics, so its test
            # is reused rather than re-spelled. Promoting it is a follow-up of task 6151.
            'mcnemar_p': _scorer._mcnemar_exact(only_shipped, only_other),
        }
    return readout


def _fixture_side(report: Mapping[str, Any]) -> dict[str, Any]:
    provenance = report['provenance']
    duplicates = report['per_class']['duplicate']
    duplicates_contested = report['confusion']['duplicate']['contested']
    middle_band = report['production_shape']['middle_band']
    return {
        'duplicates_n': duplicates['n'],
        'duplicates_contested': duplicates_contested,
        'duplicates_contested_rate': (
            round(duplicates_contested / duplicates['n'], 4) if duplicates['n'] else None
        ),
        'false_contested': report['false_contested'],
        'middle_band': {key: middle_band[key] for key in ('n', 'correct', 'accuracy')},
        'pseudo_contradiction_n': report['per_class']['pseudo_contradiction']['n'],
        'pseudo_contradiction_contested': (
            report['confusion']['pseudo_contradiction']['contested']
        ),
        'field_chars': provenance['field_chars'],
        'judge_model': provenance['judge_model'],
        'judge_system_prompt_sha256': provenance['judge_system_prompt_sha256'],
    }


def _wording_fixture(
    fixture_reports: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any] | None:
    """Per wording, the fixture eval's readout, refused unless the two runs are matched."""
    if fixture_reports is None:
        return None
    if set(fixture_reports) != set(_wording.WORDINGS):
        raise ValueError(
            f'fixture reports are needed for exactly {list(_wording.WORDINGS)}, '
            f'got {sorted(fixture_reports)}',
        )
    for field in _MATCHED_FIXTURE_PROVENANCE:
        values = {w: r['provenance'].get(field) for w, r in fixture_reports.items()}
        if len({json.dumps(v) for v in values.values()}) != 1:
            raise ValueError(f'the fixture reports are not matched on {field}: {values}')
    for wording, report in fixture_reports.items():
        measured = report['provenance'].get('judge_system_prompt_sha256')
        expected = _sha256(_wording.system_prompt(wording))
        if measured != expected:
            raise ValueError(
                f'the {wording} fixture report measured prompt {measured}, '
                f'not the {wording} wording {expected}',
            )
    return {wording: _fixture_side(fixture_reports[wording]) for wording in _wording.WORDINGS}


def build_population_artifact(
    snapshot: Mapping[str, Any],
    snapshot_sha256: str,
    snapshot_path: Path,
    arm_rows_by_name: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    budget_usd: float,
    fixture_reports: Mapping[str, Mapping[str, Any]] | None = None,
    pairs_to_rate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The committed population artifact; refuses (``ValueError``) any partial arm.

    Every arm of :data:`ARMS` must cover the same prefix of
    :func:`judge_band_order` with rows of this snapshot, once each. Arm files
    are read from ``arms/`` beside *snapshot_path* for their path and digest.
    """
    run_set = _validated_run_set(snapshot, snapshot_sha256, arm_rows_by_name)
    arms_dir = Path(snapshot_path).parent / ARMS_DIR_NAME[Slates.FROZEN]
    arms = [
        _arm_row(arm, arm_rows_by_name[arm.name], arm_path(arms_dir, arm.name)) for arm in ARMS
    ]
    return {
        'population': _population_block(snapshot, snapshot_sha256, snapshot_path, run_set),
        'arms': arms,
        'spend': _spend_block(arms, budget_usd),
        'pairs_to_rate': None if pairs_to_rate is None else dict(pairs_to_rate),
        'wording_attribution': {
            'population': _wording_population(arm_rows_by_name),
            'fixture': _wording_fixture(fixture_reports),
        },
    }


def build_write_time_population_artifact(
    snapshot: Mapping[str, Any],
    snapshot_sha256: str,
    snapshot_path: Path,
    arm_rows_by_name: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    sample_size: int,
    budget_usd: float,
    pairs_to_rate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The committed π2 artifact; refuses (``ValueError``) any partial or mixed arm.

    Every arm of :data:`WRITE_TIME_ARMS` must cover, on write-time slates,
    exactly the writes of the first *sample_size* of :func:`judge_band_order`
    still in the judge band. Arm files are read from ``arms-write-time/``
    beside *snapshot_path* for their path and digest.
    """
    run_set = _validated_write_time_run_set(
        snapshot, snapshot_sha256, arm_rows_by_name, sample_size,
    )
    arms_dir = Path(snapshot_path).parent / ARMS_DIR_NAME[Slates.WRITE_TIME]
    arms = [
        _arm_row(arm, arm_rows_by_name[arm.name], arm_path(arms_dir, arm.name))
        for arm in WRITE_TIME_ARMS
    ]
    return {
        'population': _write_time_population_block(
            snapshot, snapshot_sha256, snapshot_path, run_set,
        ),
        'arms': arms,
        'spend': _spend_block(arms, budget_usd),
        'pairs_to_rate': None if pairs_to_rate is None else dict(pairs_to_rate),
    }


def _capped(text: str | None) -> str | None:
    if text is None or len(text) <= _PAIR_TEXT_CHARS:
        return text
    return f'{text[:_PAIR_TEXT_CHARS]}…[truncated, {len(text)} chars total]'


def _target_text(snapshot: Mapping[str, Any], write: Mapping[str, Any], target_id: str) -> str | None:
    on_slate = [c for c in write['candidates'] if c['memory_id'] == target_id]
    if on_slate:
        return on_slate[0]['content']
    return (snapshot['targets'].get(target_id) or {}).get('content')


def build_pairs_to_rate(
    snapshot: Mapping[str, Any],
    snapshot_sha256: str,
    arm_rows_by_name: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    already_rated: Collection[tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The blind rater rows, one per distinct judged (write, target) pair, and their stats.

    Rows carry texts only: no arm, outcome, band or score. Pairs already in
    *already_rated* are left out and counted; a target whose text is gone is
    kept as ``null`` and counted. Refuses (``ValueError``) every run
    :func:`build_population_artifact` refuses.
    """
    _validated_run_set(snapshot, snapshot_sha256, arm_rows_by_name)
    return _pair_rows(snapshot, snapshot_sha256, arm_rows_by_name, already_rated)


def build_write_time_pairs_to_rate(
    snapshot: Mapping[str, Any],
    snapshot_sha256: str,
    arm_rows_by_name: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    sample_size: int,
    already_rated: Collection[tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """:func:`build_pairs_to_rate` for the π2 run, refusing what its artifact refuses."""
    _validated_write_time_run_set(snapshot, snapshot_sha256, arm_rows_by_name, sample_size)
    return _pair_rows(snapshot, snapshot_sha256, arm_rows_by_name, already_rated)


def _pair_rows(
    snapshot: Mapping[str, Any],
    snapshot_sha256: str,
    arm_rows_by_name: Mapping[str, Sequence[Mapping[str, Any]]],
    already_rated: Collection[tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    writes = {write['memory_id']: write for write in snapshot['writes']}
    judged = {
        (row['memory_id'], row['judged_candidate_id'])
        for rows in arm_rows_by_name.values() for row in rows
        if row['band'] == OUTCOME_JUDGE and row['judged_candidate_id'] is not None
    }
    rated = judged & set(already_rated)
    pairs = sorted(
        judged - rated, key=lambda pair: _sha256(f'{snapshot_sha256}:{pair[0]}:{pair[1]}'),
    )
    rows = [{
        'entry_id': entry_id,
        'target_id': target_id,
        'entry_text': _capped(writes[entry_id]['content']),
        'target_text': _capped(_target_text(snapshot, writes[entry_id], target_id)),
    } for entry_id, target_id in pairs]
    return rows, {
        'n_pairs': len(rows),
        'excluded_already_rated': len(rated),
        'missing_target_text': sum(1 for row in rows if row['target_text'] is None),
        'order': _PAIR_ORDER,
    }


def load_rated_pairs(paths: Iterable[Path]) -> set[tuple[str, str]]:
    """Every (entry_id, target_id) a verdict file already holds."""
    return {
        (row['entry_id'], row['target_id'])
        for path in paths for row in read_rows(Path(path))
    }


def _rated_source(path: Path) -> dict[str, Any]:
    """A verdict file as it stood: an append-only file's later state keeps this prefix."""
    body = Path(path).read_bytes()
    return {
        'path': repo_relative(path),
        'rows': len(body.splitlines(keepends=True)),
        'sha256': _sha256(body),
    }


# --- CLI -----------------------------------------------------------------------

def _command_run(args: argparse.Namespace) -> int:
    from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415

    snapshot, snapshot_sha256 = _freeze.load_snapshot(args.snapshot)
    requested = set(args.arms or [arm.name for arm in ARMS_OF[args.slates]])
    summary = asyncio.run(run_arms(
        snapshot, snapshot_sha256, Path(args.snapshot).parent / ARMS_DIR_NAME[args.slates],
        arms=[arm for arm in ARMS if arm.name in requested],
        service=types.SimpleNamespace(config=FusedMemoryConfig()),
        max_writes=args.max_writes, concurrency=args.concurrency,
        budget_usd=args.budget_usd, slates=args.slates,
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


def _fixture_reports(values: Sequence[str]) -> dict[str, dict[str, Any]] | None:
    if not values:
        return None
    reports: dict[str, dict[str, Any]] = {}
    for value in values:
        wording, _, path = value.partition('=')
        if wording not in _wording.WORDINGS or not path:
            raise ValueError(f'--fixture-report takes WORDING=PATH with WORDING in '
                             f'{list(_wording.WORDINGS)}, got {value!r}')
        reports[wording] = json.loads(Path(path).read_text(encoding='utf-8'))
    return reports


def _write_staged(bodies: Mapping[Path, str]) -> None:
    """Write every body beside its path before moving any into place.

    A body that cannot be written therefore replaces none of the files.
    """
    staged = {Path(path): Path(path).with_name(f'{Path(path).name}.tmp') for path in bodies}
    try:
        for path, body in bodies.items():
            staged[Path(path)].write_text(body, encoding='utf-8')
        for path, temporary in staged.items():
            os.replace(temporary, path)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)


def _jsonl_body(rows: Iterable[Mapping[str, Any]]) -> str:
    return ''.join(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n' for row in rows)


def _json_body(artifact: Mapping[str, Any]) -> str:
    return json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + '\n'


def _command_publish(args: argparse.Namespace) -> int:
    snapshot, snapshot_sha256 = _freeze.load_snapshot(args.snapshot)
    arms_dir = Path(args.snapshot).parent / ARMS_DIR_NAME[Slates.FROZEN]
    arm_rows = {arm.name: read_rows(arm_path(arms_dir, arm.name)) for arm in ARMS}
    pairs, stats = build_pairs_to_rate(
        snapshot, snapshot_sha256, arm_rows,
        already_rated=load_rated_pairs(args.already_rated),
    )
    pairs_body = _jsonl_body(pairs)
    artifact = build_population_artifact(
        snapshot, snapshot_sha256, args.snapshot, arm_rows,
        budget_usd=args.budget_usd,
        fixture_reports=_fixture_reports(args.fixture_report),
        pairs_to_rate={
            **stats, 'path': repo_relative(args.pairs_out), 'sha256': _sha256(pairs_body),
        },
    )
    _write_staged({args.pairs_out: pairs_body, args.population_out: _json_body(artifact)})
    print(json.dumps({
        'population_out': str(args.population_out), 'pairs_out': str(args.pairs_out),
        'n_judge_band': artifact['population']['n_judge_band'],
        'n_pairs': stats['n_pairs'], 'usd_total': artifact['spend']['usd_total'],
    }, indent=2))
    return 0


def _command_publish_write_time(args: argparse.Namespace) -> int:
    snapshot, snapshot_sha256 = _freeze.load_snapshot(args.snapshot)
    arms_dir = Path(args.snapshot).parent / ARMS_DIR_NAME[Slates.WRITE_TIME]
    arm_rows = {arm.name: read_rows(arm_path(arms_dir, arm.name)) for arm in WRITE_TIME_ARMS}
    pairs, stats = build_write_time_pairs_to_rate(
        snapshot, snapshot_sha256, arm_rows, sample_size=args.max_writes,
        already_rated=load_rated_pairs(args.already_rated),
    )
    pairs_body = _jsonl_body(pairs)
    artifact = build_write_time_population_artifact(
        snapshot, snapshot_sha256, args.snapshot, arm_rows,
        sample_size=args.max_writes, budget_usd=args.budget_usd,
        pairs_to_rate={
            **stats, 'path': repo_relative(args.pairs_out), 'sha256': _sha256(pairs_body),
            'already_rated': [_rated_source(path) for path in args.already_rated],
        },
    )
    _write_staged({args.pairs_out: pairs_body, args.population_out: _json_body(artifact)})
    population = artifact['population']
    print(json.dumps({
        'population_out': str(args.population_out), 'pairs_out': str(args.pairs_out),
        'n_judge_band_write_time': population['n_judge_band_write_time'],
        'excluded_by_write_time_filter': population['excluded_by_write_time_filter']['count'],
        'n_pairs': stats['n_pairs'], 'usd_total': artifact['spend']['usd_total'],
    }, indent=2))
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
                     help="a frozen snapshot.json; rows land in its directory's arms/ "
                          '(arms-write-time/ under --slates write-time)')
    run.add_argument('--slates', type=Slates, choices=[s.value for s in Slates],
                     default=Slates.FROZEN.value,
                     help='judge each write on its frozen slate, or on the records created '
                          'before it (default: frozen)')
    run.add_argument('--max-writes', dest='max_writes', type=int, default=None,
                     help='judge only this prefix of the judge-band order (default: all)')
    run.add_argument('--concurrency', type=int, default=8,
                     help='calls in flight within one arm (default: 8)')
    run.add_argument('--budget-usd', dest='budget_usd', type=float, default=40.0,
                     help='list-price spend over every arm file at which dispatch stops; '
                          'calls in flight still land, so spend can pass it by up to '
                          '--concurrency calls (default: 40.0)')
    run.add_argument('--arms', nargs='+', choices=[arm.name for arm in ARMS], default=None,
                     help='arms to run, in ARMS order whatever the order given '
                          "(default: every arm of the --slates mode's table)")
    run.set_defaults(handler=_command_run)
    publish = commands.add_parser(
        'publish', help='refuse anything partial, then write the committed artifacts',
    )
    publish.add_argument('--snapshot', type=Path, required=True,
                         help="the frozen snapshot.json whose arms/ to publish")
    publish.add_argument('--population-out', dest='population_out', type=Path,
                         default=DEFAULT_POPULATION_OUT)
    publish.add_argument('--pairs-out', dest='pairs_out', type=Path, default=DEFAULT_PAIRS_OUT)
    publish.add_argument('--already-rated', dest='already_rated', type=Path, nargs='+',
                         default=[DEFAULT_ALREADY_RATED],
                         help='verdict files whose pairs are left out of the rater file')
    publish.add_argument('--fixture-report', dest='fixture_report', action='append',
                         default=[], metavar='WORDING=PATH',
                         help='a fixture eval report per wording; both or neither')
    publish.add_argument('--budget-usd', dest='budget_usd', type=float, default=40.0)
    publish.set_defaults(handler=_command_publish)
    write_time = commands.add_parser(
        'publish-write-time',
        help='refuse anything partial, then write the π2 write-time artifacts',
    )
    write_time.add_argument('--snapshot', type=Path, required=True,
                            help='the frozen snapshot.json whose arms-write-time/ to publish')
    write_time.add_argument('--max-writes', dest='max_writes', type=int, required=True,
                            help="the run's sample size, a prefix of the judge-band order "
                                 "(π's is 658)")
    write_time.add_argument('--population-out', dest='population_out', type=Path,
                            default=DEFAULT_WRITE_TIME_POPULATION_OUT)
    write_time.add_argument('--pairs-out', dest='pairs_out', type=Path,
                            default=DEFAULT_WRITE_TIME_PAIRS_OUT)
    write_time.add_argument('--already-rated', dest='already_rated', type=Path, nargs='+',
                            default=[DEFAULT_VERDICT_CORPUS, DEFAULT_ALREADY_RATED],
                            help='verdict files whose pairs are left out of the rater file '
                                 '(default: the verdict corpus and the seed)')
    write_time.add_argument('--budget-usd', dest='budget_usd', type=float, default=15.0)
    write_time.set_defaults(handler=_command_publish_write_time)
    args = parser.parse_args(argv)
    if getattr(args, 'config', None):
        os.environ['CONFIG_PATH'] = str(args.config)
    return args.handler(args)


if __name__ == '__main__':
    raise SystemExit(main())
