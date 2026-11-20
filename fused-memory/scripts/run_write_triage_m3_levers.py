#!/usr/bin/env python3
"""Replay the rated soak corpus through the shipped judge under each M3 lever arm (task 7519).

The only LLM edge of the M3 lever study. Each :class:`Arm` is a judge prompt,
model and reasoning effort, a second pass over a parent arm's verdicts, or a
call-free transform of them.
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
import enum
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
from types import MappingProxyType
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
    JUDGE_EXEMPLARS,
    JUDGE_REPLY_SHAPE,
    JudgeExemplar,
    JudgeOutputError,
    judge_write,
    render_judge_exemplars,
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


class ArmKind(enum.StrEnum):
    """What an arm does with each case."""

    JUDGE = 'judge'
    SECOND_PASS = 'second_pass'
    TRANSFORM = 'transform'


class ArmScope(enum.StrEnum):
    """Which cases an arm holds a row for, and which of them it judges."""

    ALL = 'all'
    SOAK_JUDGE_BAND = 'soak-judge-band'
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
    """One replay arm: a judge configuration, or a pass over a parent arm's rows.

    A transform arm makes no call: ``transform`` reads each parent row and
    says whether the write is dispute-marked. An unknown kind or scope is
    refused at construction.
    """

    name: str
    kind: ArmKind
    prompt: str | None
    model: str | None
    effort: str | None
    parent: str | None
    scope: ArmScope
    description: str
    transform: Callable[[Mapping[str, Any]], bool] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'kind', ArmKind(self.kind))
        object.__setattr__(self, 'scope', ArmScope(self.scope))


def mark_refused_contest(parent_row: Mapping[str, Any]) -> bool:
    """Lever e1: a write the judge contested but the attach guard stored marks the record it disputes.

    Like F4's supersession mark, the write stays standalone, as the guard
    wants, and a reader of the disputed record sees the dispute.
    """
    return (
        parent_row['judge_outcome'] == OUTCOME_CONTESTED
        and parent_row['routed_outcome'] == OUTCOME_STORED
    )


_SHIPPED_PROMPT = _wording.system_prompt(_wording.WORDING_SHIPPED)


def _shipped_arm(name: str, effort: str, description: str) -> Arm:
    return Arm(name, ArmKind.JUDGE, _SHIPPED_PROMPT, SHIPPED_MODEL, effort, None, ArmScope.ALL,
               description)


# --- lever prompts ----------------------------------------------------------
#
# Drafted from the mechanism-level categories of the step-11 classification
# (calibration/write_triage_m3_hidden_conflicts.jsonl's dispute kinds) and the
# CONFLICT question of calibration/write_triage_rater_brief_v2.md, never from
# a case's text; ``refuse_corpus_text`` refuses any prompt that shares an
# 8-token shingle with one. Every exemplar is synthetic, on the shipped
# greenhouse candidate. Each prompt keeps the shipped vocabulary and reply
# shape, so the shipped parser reads its answers.

_GREENHOUSE = JUDGE_EXEMPLARS[0].candidate

_MIXED_DISPUTE = JudgeExemplar(
    entry=(
        'Glasshouse setpoints are entered in degrees F, and since the spring firmware '
        'update the night schedule is set in degrees C.'
    ),
    candidate=_GREENHOUSE,
    verdict='contests',
)
"""An entry that adds to the candidate and also says part of it is now different."""

_PASSING_DISPUTE = JudgeExemplar(
    entry=(
        'The irrigation timer now opens the valves at 06:00. It takes its readings from '
        'the greenhouse thermostat, which was recalibrated to Celsius last week.'
    ),
    candidate=_GREENHOUSE,
    verdict='contests',
)
"""An entry about something else that shows, in passing, that the candidate is outdated."""

LEVER_EXEMPLARS: Mapping[str, tuple[JudgeExemplar, ...]] = MappingProxyType({
    'a1-partial-dispute': (*JUDGE_EXEMPLARS, _MIXED_DISPUTE),
    'c1-dispute-marker': (JUDGE_EXEMPLARS[1], _MIXED_DISPUTE),
    'd1-stale-detail-check': (*JUDGE_EXEMPLARS, _PASSING_DISPUTE),
})
"""Each lever prompt's worked examples, as rendered by ``render_judge_exemplars``."""

A1_PARTIAL_DISPUTE_PROMPT = f"""\
You classify the RELATIONSHIP between a new memory entry and a small set of \
existing entries retrieved as its closest matches. You do not decide which \
entry is correct, and you do not merge, rewrite or rank them.

Answer with exactly one of these four words:

- "distinct" — no candidate makes the same core claim as the new entry. \
Shared wording alone neither makes a match nor rules one out.
- "restates" — the new entry asserts what a candidate already asserts and \
adds nothing a reader of that candidate would need: no new number, condition, \
cause, fix, caveat or newer state. A paraphrase that adds nothing restates; \
anything new makes it "amends", or "contests" if it says the candidate is \
wrong, outdated or different.
- "amends" — the new entry asserts what a candidate asserts AND adds \
something the candidate does not have: a detail, a scope, a later \
observation — and says nothing the candidate states, central or secondary, \
is wrong, outdated or different.
- "contests" — the new entry says that something a candidate states is \
wrong, outdated or different, explicitly or implicitly. That something may \
be the candidate's central claim or any secondary part of it, such as a \
status, a count, a measurement or where something lives. An entry that adds \
to a candidate AND disputes any part of it contests; it does not amend. It \
does not contest when it merely restates the candidate, or agrees with all \
of it and only adds. You are DETECTING a contradiction so a human or a \
downstream gate can adjudicate it; you are NOT deciding which side is true, \
and nothing you say here deletes or edits anything.

Find the candidate whose core claim the new entry shares (or, for \
"contests", disputes in any part), answer about THAT candidate, and name it \
by its id: the verdict is filed against the candidate you name and no other. \
Answer "distinct", naming none, only when no candidate qualifies.

Worked examples:

{render_judge_exemplars(LEVER_EXEMPLARS['a1-partial-dispute'])}

Reply with a bare JSON object and nothing else:

{JUDGE_REPLY_SHAPE}\
"""
"""Lever a1: the shipped prompt with the amends/contests boundary moved.

A dispute of any part of the candidate, central or secondary, contests even
when the entry also adds. 13 of the 14 amended-stratum M3 hits dispute a
secondary part of their parent.
"""

C1_DISPUTE_MARKER_PROMPT = f"""\
You check one new memory entry against one existing entry, the candidate. An \
earlier check found that the new entry restates or amends the candidate: it \
asserts what the candidate asserts, and may add to it. You do not decide \
which entry is correct, and you do not merge, rewrite or rank them.

Your one question: does any part of the new entry also say that something \
the candidate states is wrong, outdated or different? That something may be \
the candidate's central claim or a secondary detail, such as a status, a \
count, a measurement, where something lives, or a claim that something does \
not exist. The new entry need not mention the candidate, and it may make the \
point in passing.

- "contests" — it does: some part of the new entry disputes something the \
candidate states. You are DETECTING a dispute so a human or a downstream \
gate can adjudicate it; you are NOT deciding which side is true, and nothing \
you say here deletes or edits anything.
- "amends" — it does not: the new entry agrees with everything the \
candidate states.

The other two words of the vocabulary, "distinct" and "restates", do not \
answer this question; give one of the two words above. Name the candidate \
by its id, whichever answer you give.

Worked examples:

{render_judge_exemplars(LEVER_EXEMPLARS['c1-dispute-marker'])}

Reply with a bare JSON object and nothing else:

{JUDGE_REPLY_SHAPE}\
"""
"""Lever c1: a second pass over an attached amendment or restatement, asking only about a dispute.

A ``contests`` answer marks the write, and a marked write reads as contested.
"""

D1_STALE_DETAIL_CHECK_PROMPT = f"""\
You classify the RELATIONSHIP between a new memory entry and a small set of \
existing entries retrieved as its closest matches. You do not decide which \
entry is correct, and you do not merge, rewrite or rank them.

Answer with exactly one of these four words:

- "distinct" — no candidate makes the same core claim as the new entry. \
Shared wording alone neither makes a match nor rules one out.
- "restates" — the new entry asserts what a candidate already asserts and \
adds nothing a reader of that candidate would need: no new number, condition, \
cause, fix, caveat or newer state. A paraphrase that adds nothing restates; \
anything new makes it "amends", or "contests" if it says the candidate is \
wrong, outdated or different.
- "amends" — the new entry asserts what a candidate asserts AND adds \
something the candidate does not have: a detail, a scope, a later \
observation — without saying the candidate is wrong.
- "contests" — the new entry addresses the same claim or subject as a \
candidate and says it is wrong, outdated or different, explicitly or \
implicitly — not when it merely restates it, and not when it agrees and \
only adds to it. You are DETECTING a contradiction so a human or a \
downstream gate can adjudicate it; you are NOT deciding which side is true, \
and nothing you say here deletes or edits anything.

Check the details too. Before you answer anything but "contests", read each \
detail a candidate states that can go out of date: a status (pending, open, \
not yet filed, not yet decided), a location (the file, module or function \
where something lives), a measured figure (a duration, count, size or \
ratio), an absence or a limit ("there is no X", "only Y"), or a defect \
stated as current. If the new entry shows that any such detail has changed \
or reads differently, answer "contests" about that candidate. This holds \
when the detail is incidental to the candidate, when the new entry gives it \
only in passing, and when the new entry is mainly about something else; the \
new entry need not mention the candidate at all.

Find the candidate whose core claim the new entry shares (or, for \
"contests", contradicts), answer about THAT candidate, and name it by its \
id: the verdict is filed against the candidate you name and no other. \
Answer "distinct", naming none, only when no candidate qualifies.

Worked examples:

{render_judge_exemplars(LEVER_EXEMPLARS['d1-stale-detail-check'])}

Reply with a bare JSON object and nothing else:

{JUDGE_REPLY_SHAPE}\
"""
"""Lever d1: the shipped prompt plus a check of the candidate's time-sensitive details.

Named from the step-11 classification. 12 of the 14 amended-stratum M3 hits
dispute such a detail: a status, a location, a measured figure or an
absence. 10 of them do it in an aside, and 11 sit under a parent whose
subject is not the entry's.
"""

ARMS: tuple[Arm, ...] = (
    Arm('psi-repro', ArmKind.JUDGE, _wording.PSI_JUDGE_SYSTEM_PROMPT, SHIPPED_MODEL, 'low', None,
        ArmScope.SOAK_JUDGE_BAND,
        'the soak-era psi prompt and model; its agreement with the soak outcomes is the harness error'),
    _shipped_arm('f3p', 'low', "the shipped F3' prompt as deployed: the baseline every lever is read against"),
    _shipped_arm('f3p-repeat', 'low', "f3p again: the judge's run-to-run noise floor"),
    Arm('a1-partial-dispute', ArmKind.JUDGE, A1_PARTIAL_DISPUTE_PROMPT, SHIPPED_MODEL, 'low', None,
        ArmScope.ALL, 'the amends/contests boundary moved: a dispute of any part of a candidate contests'),
    Arm('c1-dispute-marker', ArmKind.SECOND_PASS, C1_DISPUTE_MARKER_PROMPT, SHIPPED_MODEL, 'low',
        'f3p', ArmScope.ALL,
        "a second pass over f3p's attached amendments and restatements; a dispute marks the write"),
    Arm('d1-stale-detail-check', ArmKind.JUDGE, D1_STALE_DETAIL_CHECK_PROMPT, SHIPPED_MODEL, 'low',
        None, ArmScope.ALL,
        "the shipped prompt plus a check of each candidate's time-sensitive details"),
    Arm('e1-mark-refused-contests', ArmKind.TRANSFORM, None, None, None, 'f3p', ArmScope.ALL,
        "f3p's contests the attach guard stored, marking the record each disputes",
        mark_refused_contest),
    Arm('d1-e1-combined', ArmKind.TRANSFORM, None, None, None, 'd1-stale-detail-check', ArmScope.ALL,
        'the recommended sequence: d1, then e1 over its verdicts', mark_refused_contest),
    Arm('a1-e1-combined', ArmKind.TRANSFORM, None, None, None, 'a1-partial-dispute', ArmScope.ALL,
        'the low-false-contest alternative: a1, then e1 over its verdicts', mark_refused_contest),
    _shipped_arm('b1-sol-medium', 'medium', 'the shipped prompt at medium reasoning effort'),
    _shipped_arm('b2-sol-high', 'high', 'the shipped prompt at high reasoning effort'),
)


def arm_named(name: str, arms: Iterable[Arm] = ARMS) -> Arm:
    known = {arm.name: arm for arm in arms}
    if name not in known:
        raise ValueError(f'unknown arm {name!r}; known arms are {list(known)}')
    return known[name]


def validate_arms(arms: Sequence[Arm]) -> None:
    """Refuse a malformed arm table: names, prompts, parents and transforms."""
    seen: set[str] = set()
    for arm in arms:
        problem = _arm_problem(arm, seen)
        if problem:
            raise ValueError(f'arm {arm.name!r}: {problem}')
        seen.add(arm.name)


def _arm_problem(arm: Arm, seen: set[str]) -> str | None:
    if not _NAME.fullmatch(arm.name) or arm.name in seen:
        return 'the name must be unique, lowercase and filename-safe'
    if (arm.kind is ArmKind.JUDGE) != (arm.parent is None):
        return 'a judge arm names no parent, and every other kind names one'
    if arm.parent is not None and arm.parent not in seen:
        return f'parent {arm.parent!r} is not an earlier arm'
    if arm.kind is not ArmKind.TRANSFORM and not arm.prompt:
        return 'a judge or second-pass arm needs a prompt'
    if (arm.kind is ArmKind.TRANSFORM) != (arm.transform is not None):
        return 'a transform arm, and only a transform arm, carries a transform function'
    if arm.kind is ArmKind.TRANSFORM and (arm.prompt or arm.model or arm.effort):
        return 'a transform arm makes no call, so it names no prompt, model or effort'
    return None


SHINGLE_TOKENS = 8
"""A prompt may share no run of this many tokens with any case text."""

_TOKEN = re.compile(r'\w+|[^\w\s]')


def _shingles(text: str) -> dict[tuple[str, ...], None]:
    """*text*'s runs of :data:`SHINGLE_TOKENS` lowercased word-or-symbol tokens, in order."""
    tokens = _TOKEN.findall(text.lower())
    return dict.fromkeys(
        tuple(tokens[start:start + SHINGLE_TOKENS])
        for start in range(len(tokens) - SHINGLE_TOKENS + 1)
    )


def refuse_corpus_text(arms: Iterable[Arm], cases: Iterable[_cases.ReplayCase]) -> None:
    """Refuse an arm whose prompt shares a token shingle with any case's write or parent text.

    The same corpus both motivates and scores the lever prompts, so a prompt
    carrying a case's wording would report a gain that does not generalise.
    """
    corpus: set[tuple[str, ...]] = set()
    for case in cases:
        corpus.update(_shingles(case.child_text))
        corpus.update(_shingles(case.parent_text))
    for arm in arms:
        shared = [shingle for shingle in _shingles(arm.prompt or '') if shingle in corpus]
        if shared:
            raise ValueError(
                f'arm {arm.name!r}: its prompt shares {len(shared)} {SHINGLE_TOKENS}-token'
                f' spans with the case texts, first {" ".join(shared[0])!r}'
            )


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
    case: _cases.ReplayCase, service: Any, sleep: Callable[[float], Awaitable[Any]],
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


def _blank_row(
    arm: Arm, *, entry_id: str, target_id: str, route: str, cases_sha256: str,
    prompt_sha256: str | None,
) -> dict[str, Any]:
    """Every key an arm row carries, for a write no call has been made on."""
    return {
        'arm': arm.name, 'entry_id': entry_id, 'target_id': target_id, 'route': route,
        'model': arm.model, 'effort': arm.effort,
        'prompt_sha256': prompt_sha256, 'cases_sha256': cases_sha256,
        'judge_outcome': None, 'candidate_id': None, 'dispute_marked': None,
        'calls': 0, 'attempts': 0, 'seconds': None, 'input_tokens': None,
        'output_tokens': None, 'reasoning_tokens': None, 'usd': 0.0,
        'error_type': None, 'error': None, 'transport_failures': [],
    }


def _base_row(
    case: _cases.ReplayCase, arm: Arm, *, cases_sha256: str, prompt_sha256: str | None,
) -> dict[str, Any]:
    features = case.features
    return _blank_row(
        arm, entry_id=features.entry_id, target_id=features.target_id, route=str(features.route),
        cases_sha256=cases_sha256, prompt_sha256=prompt_sha256,
    )


def _inherited(parent_row: Mapping[str, Any]) -> dict[str, Any]:
    """The verdicts a second-pass or transform row copies from its parent's row."""
    return {
        'judge_outcome': parent_row['judge_outcome'], 'candidate_id': parent_row.get('candidate_id'),
        'routed_outcome': parent_row['routed_outcome'],
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


def _in_scope(case: _cases.ReplayCase, arm: Arm) -> bool:
    """Does a complete *arm* hold a row for *case*?"""
    return arm.scope is ArmScope.ALL or case.features.soak_band == OUTCOME_JUDGE


def _replayed_features(case: _cases.ReplayCase, arm: Arm) -> _cases.CaseFeatures:
    """*case*'s features as *arm* routes them: a soak-judge-band arm judges every case it holds."""
    if arm.scope is ArmScope.SOAK_JUDGE_BAND:
        return dataclasses.replace(case.features, route=_cases.Route.JUDGE)
    return case.features


def _judged(case: _cases.ReplayCase, arm: Arm) -> bool:
    """Does a judge *arm* call the judge on *case*, one of the cases it holds a row for?"""
    return _replayed_features(case, arm).route is _cases.Route.JUDGE


def _second_pass_applies(parent_row: Mapping[str, Any]) -> bool:
    """Is the parent's write an attached amendment or restatement, which a second pass re-asks?"""
    return parent_row['judge_outcome'] in _SECOND_PASS_SOURCES and (
        parent_row['routed_outcome'] == parent_row['judge_outcome']
    )


async def judge_case(
    case: _cases.ReplayCase, arm: Arm, service: Any, *, cases_sha256: str, prompt_sha256: str,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> dict[str, Any]:
    """One judge-arm row for *case*; the caller holds the arm's prompt in force."""
    row = _base_row(case, arm, cases_sha256=cases_sha256, prompt_sha256=prompt_sha256)
    guard = resolve_attach_guard(service)
    features = _replayed_features(case, arm)
    if features.route is not _cases.Route.JUDGE:
        return row | {'routed_outcome': _cases.routed_outcome(features, None, guard)}
    call = await _ask(case, service, sleep)
    return row | _call_fields(call, arm.model) | {
        'judge_outcome': call.outcome, 'candidate_id': call.candidate_id,
        'routed_outcome': _cases.routed_outcome(features, call.outcome, guard),
    }


async def second_pass_case(
    case: _cases.ReplayCase, parent_row: Mapping[str, Any], arm: Arm, service: Any, *,
    cases_sha256: str, prompt_sha256: str,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> dict[str, Any]:
    """A second pass over a parent row: is an attached amendment or restatement disputed?"""
    row = _base_row(case, arm, cases_sha256=cases_sha256, prompt_sha256=prompt_sha256) | (
        _inherited(parent_row) | {'dispute_marked': False}
    )
    if not _second_pass_applies(parent_row):
        return row
    call = await _ask(case, service, sleep)
    return row | _call_fields(call, arm.model) | {
        'dispute_marked': call.error is None and call.outcome == OUTCOME_CONTESTED,
    }


def transform_row(
    parent_row: Mapping[str, Any], arm: Arm, *, cases_sha256: str,
) -> dict[str, Any]:
    """A transform arm's row: the parent's verdicts, marked by the arm's function, at no cost."""
    if arm.transform is None:
        raise ValueError(f'arm {arm.name!r} carries no transform')
    row = _blank_row(
        arm, entry_id=parent_row['entry_id'], target_id=parent_row['target_id'],
        route=parent_row['route'], cases_sha256=cases_sha256, prompt_sha256=None,
    )
    return row | _inherited(parent_row) | {
        'dispute_marked': bool(parent_row.get('dispute_marked')) or arm.transform(parent_row),
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


def arm_scope_cases(cases: Sequence[_cases.ReplayCase], arm: Arm) -> list[_cases.ReplayCase]:
    """The cases a complete *arm* holds a row for."""
    return [case for case in cases if _in_scope(case, arm)]


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
    cases: Sequence[_cases.ReplayCase],
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
    refuse_corpus_text(arms, cases)
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
    cases: Sequence[_cases.ReplayCase], arm: Arm, *, config: Any, arms_dir: Path, spend: _Spend,
    cases_sha256: str, concurrency: int, sleep: Callable[[float], Awaitable[Any]],
) -> bool:
    if arm.kind is ArmKind.TRANSFORM:
        return _run_transform_arm(cases, arm, arms_dir=arms_dir, cases_sha256=cases_sha256)
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

        async def one(case: _cases.ReplayCase) -> None:
            async with gate:
                if spend.exhausted and _needs_call(case, arm, parents):
                    return
                if arm.kind is ArmKind.JUDGE:
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


def _run_transform_arm(
    cases: Sequence[_cases.ReplayCase], arm: Arm, *, arms_dir: Path, cases_sha256: str,
) -> bool:
    parents = _complete_parent_rows(cases, arm, arms_dir)
    if parents is None:
        return False
    path = arms_dir / f'{arm.name}.jsonl'
    existing = read_arm_rows(path)
    _refuse_foreign(existing, arm, cases_sha256=cases_sha256, prompt_sha256=None)
    written = {row['entry_id'] for row in existing}
    rows = [*existing, *(
        transform_row(parents[case.features.entry_id], arm, cases_sha256=cases_sha256)
        for case in cases if case.features.entry_id not in written
    )]
    _write_sorted(path, rows)
    return True


def _needs_call(
    case: _cases.ReplayCase, arm: Arm, parents: Mapping[str, Mapping[str, Any]],
) -> bool:
    if arm.kind is ArmKind.JUDGE:
        return _judged(case, arm)
    return _second_pass_applies(parents[case.features.entry_id])


def _complete_parent_rows(
    cases: Sequence[_cases.ReplayCase], arm: Arm, arms_dir: Path,
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
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0] if __doc__ else None)
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
