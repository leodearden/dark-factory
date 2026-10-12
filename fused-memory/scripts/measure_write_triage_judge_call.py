#!/usr/bin/env python3
"""Measure what one write-triage judge call costs at the configured field cap.

Task 6076. For each slate width in :data:`WIDTHS` it builds a worst-case slate
(every field over ``write_triage.judge_field_chars``, so every field is
elided) and calls the SHIPPED ``server/write_triage_judge.py::judge_write``
on it, sequentially, as production does per write.

* Input tokens are the provider's own count, carried on
  ``TriageJudgeVerdict.usage``; they include the system instructions. A call
  whose usage was not reported counts as unmeasured, never as 0.
* Seconds are wall time around the whole ``judge_write`` call, an upper bound
  on the provider span ``judge_timeout_seconds`` actually bounds. The probe
  raises its in-memory timeout to :data:`_MEASUREMENT_CEILING_SECONDS` so the
  tail is observed rather than clipped at the value being judged, and records
  the SHIPPED timeout as the bound.

Expected spend at the defaults is under USD 0.25 at gpt-4o-mini list price.
Fields are built from the calibration fixture's real prose, because repeated
characters tokenize far below real text.

Usage
-----
  uv run python scripts/measure_write_triage_judge_call.py
"""
from __future__ import annotations

import asyncio
import collections
import importlib.util
import json
import sys
import tempfile
import time
import types
import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.write_triage import (
    OUTCOME_JUDGE,
    BandDecision,
    TriageJudgeVerdict,
)
from fused_memory.server.write_triage_judge import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_prompt,
    judge_write,
    resolve_judge_field_chars,
    resolve_judge_model,
    resolve_judge_provider,
    resolve_judge_reasoning_effort,
    resolve_judge_timeout,
)

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _PACKAGE_ROOT / 'scripts'


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


_calibrate = _load_script(_SCRIPTS / 'calibrate_write_triage.py', 'calibrate_write_triage')

#: The slate widths measured: the shipped ``judge_candidate_count`` and the
#: wider slates the flip-readiness arms run.
WIDTHS = (5, 10, 20)

_MEASUREMENT_CEILING_SECONDS = 60.0

_DEFAULT_FIXTURE = _PACKAGE_ROOT / 'tests' / 'fixtures' / 'write_triage_calibration.jsonl'
_DEFAULT_REPORT_PATH = 'calibration/write_triage_judge_call_cost.json'
_SMOKE_REPORT_NAME = 'write_triage_judge_call_cost.smoke.json'

#: The call count behind the committed artifact the decision record cites.
DEFAULT_CALLS_PER_WIDTH = 20

_PROBE_PROJECT_ID = 'dark_factory'

#: Namespace for the slate's uuid5 ids, so a re-run renders identical ids.
_SLATE_ID_NAMESPACE = uuid.UUID('5f0c6b7e-6076-4a3e-9c1d-2b8f3e4a6076')


class CallSample(NamedTuple):
    """One timed judge call: its usage and verdict, or the failure it raised."""

    seconds: float
    input_tokens: int | None
    outcome: str | None
    error: str | None


def _field_over(texts: Sequence[str], offset: int, field_chars: int) -> str:
    """Texts joined by blank lines, from *offset* round the list, until over the cap."""
    parts: list[str] = []
    index = offset
    while len('\n\n'.join(parts)) <= field_chars:
        parts.append(texts[index % len(texts)])
        index += 1
    return '\n\n'.join(parts)


def worst_case_slate(
    texts: Sequence[str], width: int, field_chars: int,
) -> tuple[str, list[MemoryResult]]:
    """A new entry and *width* candidates, every field longer than *field_chars*.

    Field ``i`` (the entry is 0) starts at ``texts[i]``, so neighbouring
    fields differ. Cosines descend from 0.90 so the judge keeps the whole
    slate when ``judge_candidate_count`` is *width*.
    """
    usable = [text for text in texts if text]
    if not usable:
        raise ValueError('worst_case_slate needs at least one non-empty text')
    entry = _field_over(usable, 0, field_chars)
    candidates = [
        MemoryResult(
            id=str(uuid.uuid5(_SLATE_ID_NAMESPACE, f'{width}:{i}')),
            content=_field_over(usable, i + 1, field_chars),
            category=MemoryCategory.procedural_knowledge,
            source_store=SourceStore.mem0,
            metadata={'store_score': round(0.90 - i / 100, 2)},
        )
        for i in range(width)
    ]
    return entry, candidates


def configure_for_width(config: Any, width: int) -> None:
    """Show the judge the whole *width* slate, and let the tail run past the timeout.

    In memory only. ``judge_write`` trims the slate to
    ``judge_candidate_count``, so a wider slate needs a wider count.
    """
    config.write_triage.judge_candidate_count = width
    config.write_triage.judge_timeout_seconds = _MEASUREMENT_CEILING_SECONDS


async def live_judge_call(
    service: Any, entry: str, candidates: Sequence[MemoryResult],
) -> TriageJudgeVerdict:
    """One call to the SHIPPED judge, as the middle band makes it."""
    return await judge_write(
        memory_service=service,
        content=entry,
        project_id=_PROBE_PROJECT_ID,
        decision=BandDecision(
            outcome=OUTCOME_JUDGE,
            canonical_id=candidates[0].id,
            similarity=0.9,
            t_high=None,
            t_low=None,
        ),
        candidates=list(candidates),
    )


async def measure_width(
    service: Any,
    entry: str,
    candidates: Sequence[MemoryResult],
    *,
    calls: int,
    clock: Callable[[], float],
    call_judge: Callable[[Any, str, Sequence[MemoryResult]], Awaitable[TriageJudgeVerdict]],
) -> list[CallSample]:
    """Time *calls* sequential judge calls on one slate."""
    samples: list[CallSample] = []
    for _ in range(calls):
        started = clock()
        try:
            verdict = await call_judge(service, entry, candidates)
        except Exception as exc:  # noqa: BLE001 — a measurement counts failures
            samples.append(CallSample(clock() - started, None, None, type(exc).__name__))
            continue
        input_tokens = verdict.usage.input_tokens if verdict.usage is not None else None
        samples.append(CallSample(clock() - started, input_tokens, verdict.outcome, None))
    return samples


def summarize_width(
    width: int, samples: Sequence[CallSample], *, timeout_seconds: float,
) -> dict[str, Any]:
    """Per-width summary. Seconds count every call, a failed one included.

    Input tokens count only calls that answered and reported usage.
    """
    seconds = _calibrate.summarize_distribution([s.seconds for s in samples])
    tokens = [
        s.input_tokens for s in samples if s.error is None and s.input_tokens is not None
    ]
    return {
        'width': width,
        'calls': len(samples),
        'seconds': seconds,
        'p95_within_timeout': (
            None if seconds['p95'] is None else seconds['p95'] < timeout_seconds
        ),
        'input_tokens': _calibrate.summarize_distribution(tokens),
        'outcomes': dict(collections.Counter(s.outcome for s in samples if s.error is None)),
        'errors': dict(collections.Counter(s.error for s in samples if s.error is not None)),
    }


def probe_provenance(service: Any, *, calls_per_width: int) -> dict[str, Any]:
    """What the probe measured against, resolved before any in-memory change."""
    return {
        'field_chars': resolve_judge_field_chars(service),
        'judge_provider': resolve_judge_provider(service),
        'judge_model': resolve_judge_model(service),
        'judge_reasoning_effort': resolve_judge_reasoning_effort(service),
        'timeout_seconds': resolve_judge_timeout(service),
        'measurement_ceiling_seconds': _MEASUREMENT_CEILING_SECONDS,
        'widths': list(WIDTHS),
        'calls_per_width': calls_per_width,
    }


def _package_path(report_path: str) -> Path:
    path = Path(report_path)
    return path if path.is_absolute() else _PACKAGE_ROOT / path


def guard_committed_report(report_path: str, *, calls_per_width: int) -> Path:
    """The path to write: *report_path*, unless a smoke run would overwrite the artifact.

    The committed artifact is the :data:`DEFAULT_CALLS_PER_WIDTH` measurement
    the decision record and PRD C1 cite. A run of any other size aimed at it
    is redirected to a temp path with a warning, and still prints its report.
    """
    path = _package_path(report_path)
    committed = _package_path(_DEFAULT_REPORT_PATH)
    if calls_per_width == DEFAULT_CALLS_PER_WIDTH or path.resolve() != committed.resolve():
        return path
    redirected = Path(tempfile.gettempdir()) / _SMOKE_REPORT_NAME
    print(
        f'calls_per_width={calls_per_width} is not the committed '
        f'{DEFAULT_CALLS_PER_WIDTH}-call measurement at {committed}; writing '
        f'{redirected} instead. Pass --report-path to choose somewhere else.',
        file=sys.stderr,
    )
    return redirected


def build_report(rows: Sequence[dict[str, Any]], *, provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        'provenance': provenance,
        'widths': sorted(rows, key=lambda row: row['width']),
    }


async def _measure_all(
    service: Any, texts: Sequence[str], *, provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for width in WIDTHS:
        configure_for_width(service.config, width)
        entry, candidates = worst_case_slate(texts, width, provenance['field_chars'])
        samples = await measure_width(
            service, entry, candidates,
            calls=provenance['calls_per_width'],
            clock=time.perf_counter,
            call_judge=live_judge_call,
        )
        prompt = build_judge_prompt(entry, candidates, field_chars=provenance['field_chars'])
        rows.append({
            **summarize_width(width, samples, timeout_seconds=provenance['timeout_seconds']),
            'prompt_chars': len(JUDGE_SYSTEM_PROMPT) + len(prompt),
        })
    return rows


def main() -> int:
    import argparse  # noqa: PLC0415
    import os  # noqa: PLC0415

    from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=None,
                        help='Path to fused-memory config file (sets CONFIG_PATH)')
    parser.add_argument('--fixture', default=str(_DEFAULT_FIXTURE),
                        help='JSONL whose `content` fields supply the slate prose')
    parser.add_argument('--calls-per-width', dest='calls_per_width', type=int,
                        default=DEFAULT_CALLS_PER_WIDTH)
    parser.add_argument('--report-path', dest='report_path', default=_DEFAULT_REPORT_PATH,
                        help='package-relative unless absolute '
                             f'(default: {_DEFAULT_REPORT_PATH})')
    args = parser.parse_args()
    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)
    report_path = guard_committed_report(args.report_path, calls_per_width=args.calls_per_width)

    service = types.SimpleNamespace(config=FusedMemoryConfig())
    provenance = probe_provenance(service, calls_per_width=args.calls_per_width)
    texts = [str(record['content']) for record in _calibrate.load_fixture(args.fixture)]
    rows = asyncio.run(_measure_all(service, texts, provenance=provenance))

    report = {
        'generated_at': datetime.now(UTC).isoformat(timespec='seconds'),
        **build_report(rows, provenance=provenance),
    }
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
