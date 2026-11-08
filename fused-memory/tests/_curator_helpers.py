"""Test helpers shared by the TaskCurator test modules and the tests that inject a curator.

A plain module rather than conftest.py, for the reason ``_fm_helpers.py`` gives.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from typing import Any, NoReturn
from unittest.mock import AsyncMock, MagicMock

import pytest
from shared.cli_invoke import AgentResult

from fused_memory.backends.task_backend_types import MetadataLookupResult
from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig
from fused_memory.middleware.curator_types import (
    CandidateTask,
    CuratorDecision,
    PoolEntry,
    PoolWithheld,
    PreparedCandidate,
    empty_pool_sizes,
    is_combine_eligible_status,
)
from fused_memory.middleware.task_interceptor import TaskInterceptor


def make_config() -> FusedMemoryConfig:
    cfg = FusedMemoryConfig()
    cfg.curator = CuratorConfig()
    return cfg


def agent_result(structured: dict | None = None, output: str = '') -> AgentResult:
    """A successful LLM call whose structured verdict is *structured*."""
    return AgentResult(
        success=True,
        output=output,
        structured_output=structured,
        cost_usd=0.01,
    )


def pool_with_ids(*pairs: tuple[str, str]) -> list[PoolEntry]:
    """Build a pool with the given (task_id, status) pairs."""
    return [
        PoolEntry(
            task_id=tid,
            title='t',
            description='',
            details='',
            files_to_modify=[],
            module_keys=[],
            status=status,
            priority='medium',
            source='module',
            combine_eligible=is_combine_eligible_status(status),
        )
        for tid, status in pairs
    ]


def prepared(
    title: str,
    pool: Iterable[PoolEntry] = (),
    corpus_error: str | None = None,
    *,
    candidate: CandidateTask | None = None,
    withheld: PoolWithheld | None = None,
) -> PreparedCandidate:
    """A prepared bundle for *title* over *pool*; *corpus_error* marks a failed assembly.

    *candidate* replaces the title-only candidate when a test needs every
    field, and *withheld* is the census of what the caps kept out of *pool*.
    """
    entries = list(pool)
    return PreparedCandidate(
        candidate=candidate if candidate is not None else CandidateTask(title=title),
        pool=entries,
        pool_sizes={**empty_pool_sizes(), 'module': len(entries)},
        prompt_tokens=0,
        withheld=withheld,
        corpus_error=corpus_error,
    )


def batch_create_result(n: int) -> AgentResult:
    """A successful batch verdict that creates each of *n* candidates."""
    return agent_result({
        'decisions': [
            {'candidate_index': i, 'action': 'create', 'justification': f'c{i}'}
            for i in range(n)
        ],
    })


_BATCH_CANDIDATE_HEADING = '# Candidate batch_index='


def candidates_in(prompt: str) -> int:
    """How many candidates a batch *prompt* carries; 0 for a single-candidate prompt."""
    return prompt.count(_BATCH_CANDIDATE_HEADING)


def candidate_titles(prompt: str) -> list[str]:
    """The titles of the candidates a curator *prompt* asks about, in prompt order.

    Reads the line after each candidate heading, at either spec.
    """
    lines = prompt.splitlines()
    return [
        following.removeprefix('  title: ')
        for heading, following in zip(lines, lines[1:], strict=False)
        if heading.startswith('# Candidate ')
    ]


InvokeArm = AgentResult | BaseException | Callable[[dict[str, Any]], AgentResult]


def tripwire(reason: str) -> Callable[..., NoReturn]:
    """An invoke arm, or a side effect, that fails the test outright when reached.

    The pipeline decides every ``Exception`` its invoker raises, so an
    ``AssertionError`` there becomes a deferral the test may never notice.
    ``pytest.fail`` raises a ``BaseException``, which no ``except Exception``
    arm absorbs.
    """
    def fail(*_args: Any, **_kwargs: Any) -> NoReturn:
        pytest.fail(reason)

    return fail


def _answer(arm: InvokeArm, kwargs: dict[str, Any]) -> AgentResult:
    if isinstance(arm, BaseException):
        raise arm
    if isinstance(arm, AgentResult):
        return arm
    return arm(kwargs)


def invoke_by_role(*, single: InvokeArm, batch: InvokeArm) -> AsyncMock:
    """An ``invoke_with_cap_retry`` stand-in that answers each curator call by its ledger role.

    A ``'task_curator'`` call is answered by *single* and a
    ``'task_curator_batch'`` call by *batch*. An arm is returned when it is an
    ``AgentResult``, raised when it is an exception, and otherwise called with
    the call's keyword arguments, so a :func:`tripwire` arm fails the test.
    Any other role fails the test too.
    """
    arms = {'task_curator': single, 'task_curator_batch': batch}

    async def invoke(**kwargs: Any) -> AgentResult:
        role = kwargs.get('role')
        if role not in arms:
            pytest.fail(f'unexpected curator invocation role {role!r}')
        return _answer(arms[role], kwargs)

    return AsyncMock(side_effect=invoke)


def must_not_invoke() -> AsyncMock:
    """An ``invoke_with_cap_retry`` stand-in for paths that must never reach the LLM."""
    return AsyncMock(side_effect=tripwire('the curator must not call the LLM'))


def zot_agent_result() -> AgentResult:
    """An LLM call that hung with zero output until its timeout killed it."""
    return AgentResult(
        success=False, output='', subtype='error_empty_output',
        timed_out=True, turns=0, cost_usd=0.0, duration_ms=181_000,
    )


async def empty_corpus(*_args, **_kwargs) -> tuple[list[PoolEntry], dict[str, int], PoolWithheld]:
    """A ``TaskCurator._build_corpus`` stand-in that assembles an empty pool."""
    return [], empty_pool_sizes(), PoolWithheld()


def escalator_double() -> AsyncMock:
    """A curator escalator whose failure and degraded-streak reports file nothing."""
    escalator = AsyncMock()
    escalator.report_failure = AsyncMock(return_value=None)
    escalator.report_consecutive_degraded = AsyncMock(return_value=None)
    return escalator


def answer_each_with(
    curator: Any, verdict: Callable[[CandidateTask], Awaitable[CuratorDecision]],
) -> None:
    """Make *curator* judge each candidate the ticket worker sends it with *verdict*.

    ``prepare_candidate`` returns a bundle over an empty pool, and
    ``curate_prepared`` answers each bundle, in order, with
    ``await verdict(candidate)`` — the surface the worker drives.
    """
    async def prepare(candidate: CandidateTask, *_args: Any) -> PreparedCandidate:
        return prepared(candidate.title, candidate=candidate)

    async def curate_prepared(
        bundles: Sequence[PreparedCandidate], *_args: Any,
    ) -> list[CuratorDecision]:
        return [await verdict(bundle.candidate) for bundle in bundles]

    curator.prepare_candidate = AsyncMock(side_effect=prepare)
    curator.curate_prepared = AsyncMock(side_effect=curate_prepared)


def curator_returning(*decisions: CuratorDecision) -> MagicMock:
    """A curator double for ``TaskInterceptor(curator=...)``.

    Its ``curate_prepared`` hands out *decisions* in order, one per bundle.
    """
    curator = MagicMock()
    answers = iter(decisions)

    async def next_decision(_candidate: CandidateTask) -> CuratorDecision:
        return next(answers)

    answer_each_with(curator, next_decision)
    curator.note_created = MagicMock()
    curator.record_task = AsyncMock()
    curator.reembed_task = AsyncMock(return_value=None)
    curator.close = AsyncMock()
    return curator


def serve_existing_task(taskmaster: AsyncMock, task: Mapping[str, Any]) -> None:
    """Make *taskmaster* hold *task* as the one match of every metadata lookup.

    Its ``find_tasks_by_metadata`` answers with a one-row page for *task*, and
    its ``get_task`` returns *task*: the two reads the R4 idempotency check
    makes.
    """
    page: MetadataLookupResult = {
        'results': [{
            'task_id': int(task['id']),
            'title': task['title'],
            'status': task['status'],
            'priority': task.get('priority', 'medium'),
            'updated_at': '',
        }],
        'total': 1,
        'tasks_with_key': 1,
        'malformed_metadata_ids': [],
    }
    taskmaster.find_tasks_by_metadata = AsyncMock(return_value=page)
    taskmaster.get_task = AsyncMock(return_value=dict(task))


def hold_first_prepare(curator: Any) -> asyncio.Event:
    """Make *curator*'s next ``prepare_candidate`` call wait for the returned event.

    Every later call delegates straight to the original.
    """
    original = curator.prepare_candidate
    release = asyncio.Event()
    held = True

    async def prepare_candidate(*args: Any, **kwargs: Any) -> Any:
        nonlocal held
        if held:
            held = False
            await release.wait()
        return await original(*args, **kwargs)

    curator.prepare_candidate = prepare_candidate
    return release


async def submit_as_one_batch(
    interceptor: TaskInterceptor,
    curator: Any,
    submissions: Sequence[Mapping[str, Any]],
    *,
    project_root: str = '/project',
) -> list[str]:
    """Submit every entry of *submissions* so the curator worker judges them in one batch.

    The worker drains its queue only after the first ticket's prepare step
    returns, and that step is held until every submission is queued. The
    first submission must therefore carry a title, so that its prepare step
    reaches ``curator.prepare_candidate``. Returns the ticket ids in
    submission order.
    """
    release = hold_first_prepare(curator)
    ticket_ids: list[str] = []
    try:
        for submission in submissions:
            response = await interceptor.submit_task(project_root, **submission)
            assert 'ticket' in response, f'submit_task returned no ticket: {response!r}'
            ticket_ids.append(response['ticket'])
    finally:
        release.set()
    return ticket_ids
