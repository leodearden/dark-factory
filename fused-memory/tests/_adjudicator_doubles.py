"""A scripted stand-in for the adjudicator's model call, and the replies and SDK errors it answers with.

``ScriptedResponses`` satisfies ``fused_memory.llm.openai_responses.ResponsesCall``,
so the adjudicator and its calibration are driven through their real call path
with no network and no live model. ``EvalAnswerKey`` scripts it per eval case
for a calibration run.
"""
from __future__ import annotations

import asyncio
import collections
import json
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import httpx
import openai

from fused_memory.llm.openai_responses import ResponsesReply, TokenUsage
from fused_memory.reconciliation.consolidation_adjudicator import Decision, VerdictReasonCode
from fused_memory.reconciliation.consolidation_adjudicator_calibration import EvalCase

if TYPE_CHECKING:
    from openai.types.responses import ResponseFormatTextConfigParam

#: What one call answers: a reply, an exception to raise, or a function of the request.
Answer = ResponsesReply | BaseException | Callable[[Mapping[str, Any]], 'ResponsesReply | BaseException']

RESOLVED_MODEL = 'gpt-6.1-sol-2026-10-01'
USAGE = TokenUsage(input_tokens=1_000, output_tokens=100, reasoning_tokens=50)


class ScriptedResponses:
    """Records every request's kwargs and answers from *answers*.

    The answers are consumed one per call and the last one repeats, so a
    single answer serves every call. With *gate*, every call waits on that event first,
    so a test can hold calls in flight (or hang them). ``peak_in_flight`` is the
    most calls ever awaiting at once.
    """

    def __init__(self, first: Answer, *rest: Answer, gate: asyncio.Event | None = None):
        self._script: Sequence[Answer] = (first, *rest)
        self._gate = gate
        self.requests: list[dict[str, Any]] = []
        self.in_flight = 0
        self.peak_in_flight = 0

    def _answer_for(self, request: Mapping[str, Any]) -> ResponsesReply | BaseException:
        answer = self._script[min(len(self.requests) - 1, len(self._script) - 1)]
        if isinstance(answer, (ResponsesReply, BaseException)):
            return answer
        return answer(request)

    async def __call__(
        self,
        *,
        model: str,
        instructions: str,
        user_content: str,
        text_format: ResponseFormatTextConfigParam,
        max_output_tokens: int,
        timeout: float,
        reasoning_effort: str | None,
    ) -> ResponsesReply:
        request = {
            'model': model,
            'instructions': instructions,
            'user_content': user_content,
            'text_format': text_format,
            'max_output_tokens': max_output_tokens,
            'timeout': timeout,
            'reasoning_effort': reasoning_effort,
        }
        self.requests.append(request)
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            if self._gate is not None:
                await self._gate.wait()
            for _ in range(3):
                await asyncio.sleep(0)
            answer = self._answer_for(request)
        finally:
            self.in_flight -= 1
        if isinstance(answer, BaseException):
            raise answer
        return answer


def reply_text(
    text: str | None,
    *,
    resolved_model: str | None = RESOLVED_MODEL,
    usage: TokenUsage | None = USAGE,
) -> ResponsesReply:
    return ResponsesReply(
        text=text, incomplete=False, incomplete_reason=None,
        resolved_model=resolved_model, usage=usage,
    )


def reply_json(
    body: Mapping[str, Any],
    *,
    resolved_model: str | None = RESOLVED_MODEL,
    usage: TokenUsage | None = USAGE,
) -> ResponsesReply:
    return reply_text(json.dumps(body), resolved_model=resolved_model, usage=usage)


def incomplete_reply(reason: str = 'max_output_tokens') -> ResponsesReply:
    return ResponsesReply(
        text='{"decision": "pro', incomplete=True, incomplete_reason=reason,
        resolved_model=RESOLVED_MODEL, usage=USAGE,
    )


_REQUEST = httpx.Request('POST', 'https://api.openai.com/v1/responses')


def _status(code: int) -> httpx.Response:
    return httpx.Response(code, request=_REQUEST)


def rate_limited() -> openai.RateLimitError:
    return openai.RateLimitError('rate limited', response=_status(429), body=None)


def server_error(status: int = 502) -> openai.InternalServerError:
    return openai.InternalServerError('server error', response=_status(status), body=None)


def connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=_REQUEST)


def sdk_timeout() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=_REQUEST)


def bad_request() -> openai.BadRequestError:
    return openai.BadRequestError('bad request', response=_status(400), body=None)


class UnidentifiedCase(BaseException):
    """A request that matches no eval case, or several: a fault in the test, never an answer.

    A ``BaseException``, so the adjudicator's ``except Exception`` cannot turn it
    into a content failure and the run fails loudly instead.
    """


def verdict_reply(
    decision: Decision, keep: Sequence[str] = (), *, resolved_model: str | None = RESOLVED_MODEL,
) -> ResponsesReply:
    code = {
        Decision.proceed: VerdictReasonCode.same_lesson,
        Decision.drop: VerdictReasonCode.not_same_lesson,
        Decision.unresolved: VerdictReasonCode.insufficient_information,
    }[decision]
    return reply_json(
        {
            'decision': decision.value, 'keep_member_ids': list(keep), 'claim': None,
            'reason_code': code.value, 'reason': 'scripted',
        },
        resolved_model=resolved_model,
    )


def expected_reply(case: EvalCase) -> ResponsesReply:
    """The case's expectation: a proceed keeping its keep ids, else its first other decision."""
    if Decision.proceed in case.expected.decision_set:
        return verdict_reply(Decision.proceed, case.expected.keep_ids)
    decision = Decision.drop if Decision.drop in case.expected.decision_set else Decision.unresolved
    return verdict_reply(decision)


def keep_all_reply(case: EvalCase) -> ResponsesReply:
    return verdict_reply(Decision.proceed, [m.id for m in case.members])


class EvalAnswerKey:
    """Answers each eval case's request through *answer*, or through a scripted deviation.

    The case is identified from the rendered user content: every member id and
    the claim of exactly one case must appear in it. A case listed in
    *deviations* answers its first attempts from that script and later ones
    from *answer*. ``calls`` counts the requests per case id.
    """

    def __init__(
        self,
        cases: Sequence[EvalCase],
        *,
        answer: Callable[[EvalCase], ResponsesReply] = expected_reply,
        deviations: Mapping[str, Sequence[ResponsesReply | BaseException]] | None = None,
    ):
        self._cases = tuple(cases)
        self._answer = answer
        self._deviations = dict(deviations or {})
        self.calls: collections.Counter[str] = collections.Counter()

    def case_for(self, user_content: str) -> EvalCase:
        matches = [
            case for case in self._cases
            if case.claim in user_content and all(m.id in user_content for m in case.members)
        ]
        if len(matches) != 1:
            raise UnidentifiedCase(f'{len(matches)} eval cases match the request')
        return matches[0]

    def __call__(self, request: Mapping[str, Any]) -> ResponsesReply | BaseException:
        case = self.case_for(request['user_content'])
        attempt = self.calls[case.case_id]
        self.calls[case.case_id] += 1
        script = self._deviations.get(case.case_id, ())
        if attempt < len(script):
            return script[attempt]
        return self._answer(case)
