#!/usr/bin/env python3
"""Run the π judge arms over a frozen write-triage population, and publish them.

Task 6151 (π), plans/write-triage-flip-readiness-prd.md §11 D14 and D16. Each
:class:`Arm` is one judge configuration (model, reasoning effort, slate width,
wording); each judge-band write of the snapshot that
``freeze_write_triage_population.py`` froze is judged once per arm, through
the SHIPPED ``write_triage_judge._call_llm`` and parser. A row follows ι's
per-case contract (``score_write_triage_pairs.py::JudgedCase``, C2''), so μ
scores the arm files with ι unchanged.
"""
from __future__ import annotations

import importlib.util
import sys
import time
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


_wording = _load_script(_SCRIPTS / 'write_triage_judge_wording.py', 'write_triage_judge_wording')
_scorer = _load_script(_SCRIPTS / 'score_write_triage_pairs.py', 'score_write_triage_pairs')

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
