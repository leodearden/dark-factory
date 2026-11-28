"""The consolidation adjudicator (plans/consolidation-adjudicator-prd.md K3, K1's keep rule).

Public names of ``fused_memory.reconciliation.consolidation_adjudicator`` only.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import re
import subprocess
import sys
import types
from unittest.mock import AsyncMock, patch

import pytest
from _adjudicator_doubles import (
    RESOLVED_MODEL,
    USAGE,
    ScriptedResponses,
    bad_request,
    connection_error,
    forbidden,
    incomplete_reply,
    no_api_key,
    rate_limited,
    reply_json,
    reply_text,
    sdk_timeout,
    server_error,
    unauthorized,
)
from _sdk_client_stubs import sdk_client_double
from shared.testing_own_time import run_on_own_time
from shared.testing_virtual_clock import virtual_clock_test

from fused_memory.config.schema import (
    ConsolidationAutoConfig,
    FusedMemoryConfig,
    LLMConfig,
    LLMProvidersConfig,
    OpenAIProviderConfig,
)
from fused_memory.llm.list_prices import list_price
from fused_memory.reconciliation.consolidation_adjudicator import (
    INPUT_TEMPLATE,
    INSTRUCTIONS,
    MAX_OUTPUT_TOKENS,
    Adjudication,
    AdjudicationFailure,
    AdjudicationRequest,
    AdjudicatorSettings,
    Decision,
    ExcludedMember,
    FailureKind,
    Hazard,
    MemberInput,
    Verdict,
    VerdictInvalid,
    VerdictReasonCode,
    adjudicate,
    adjudicator_identity,
    keep_rule_holds,
    openai_model_call,
    output_schema,
    parse_verdict,
    render_request,
    text_format,
)
from fused_memory.reconciliation.consolidation_auto import (
    REASON_CODE_CLASSES,
    AutoReasonCode,
    ReasonCodeClass,
)

BANNER = AutoReasonCode.member_carries_correction_banner


def _member(member_id: str, text: str = 'a lesson', category: str = 'procedural_knowledge'):
    return MemberInput(id=member_id, category=category, text=text)


def _request(**overrides) -> AdjudicationRequest:
    fields = {
        'topic': 'cargo-fmt-not-gated',
        'claim': 'cargo fmt is not part of the verify gate.',
        'category': 'procedural_knowledge',
        'members': (_member('a'), _member('b'), _member('c')),
    }
    fields.update(overrides)
    return AdjudicationRequest(**fields)


def _settings(**overrides) -> AdjudicatorSettings:
    fields = {
        'model': 'gpt-6.1-sol',
        'reasoning_effort': 'low',
        'endpoint_host': 'api.openai.com',
        'timeout_seconds': 60.0,
        'field_chars': 4000,
        'member_max': 8,
        'member_min': 2,
    }
    fields.update(overrides)
    return AdjudicatorSettings(**fields)


def _no_claim_problems(claim: str, topic: str | None) -> list[str]:
    return []


def _reply(**overrides) -> str:
    body = {
        'decision': 'proceed',
        'keep_member_ids': ['a', 'b'],
        'claim': None,
        'reason_code': 'same_lesson',
        'reason': 'Both restate the lesson.',
    }
    body.update(overrides)
    return json.dumps(body)


class TestKeepRule:
    @pytest.mark.parametrize('proposed', range(2, 9))
    def test_truth_table_at_member_min_two(self, proposed: int) -> None:
        for kept in range(proposed + 1):
            expected = kept >= 2 and 3 * kept >= 2 * proposed
            assert keep_rule_holds(kept, proposed, member_min=2) is expected, (kept, proposed)

    @pytest.mark.parametrize(
        ('kept', 'proposed', 'holds'),
        [(2, 3, True), (2, 4, False), (4, 6, True), (5, 8, False)],
    )
    def test_the_named_examples(self, kept: int, proposed: int, holds: bool) -> None:
        assert keep_rule_holds(kept, proposed, member_min=2) is holds

    def test_member_min_three_refuses_two_of_two(self) -> None:
        assert keep_rule_holds(2, 2, member_min=3) is False


class TestRequestInvariants:
    def test_no_members(self) -> None:
        with pytest.raises(ValueError, match='member'):
            _request(members=())

    def test_a_repeated_member_id(self) -> None:
        with pytest.raises(ValueError, match="'a'"):
            _request(members=(_member('a'), _member('a')))

    def test_a_hazard_outside_the_adjudicator_class(self) -> None:
        with pytest.raises(ValueError, match='mixed_category'):
            _request(hazards=(Hazard(code=AutoReasonCode.mixed_category, subject_id='a'),))

    def test_a_hazard_about_a_member_not_shown(self) -> None:
        with pytest.raises(ValueError, match="'zz'"):
            _request(hazards=(Hazard(code=BANNER, subject_id='zz'),))

    def test_topic_may_be_absent(self) -> None:
        assert _request(topic=None).topic is None


def _block_bounds(rendered: str, member_id: str, next_id: str) -> tuple[int, int]:
    start = rendered.index(INPUT_TEMPLATE.member_header.format(id=member_id))
    end = rendered.index(INPUT_TEMPLATE.member_header.format(id=next_id))
    return start, end


class TestRender:
    def test_a_hazard_is_rendered_inside_its_subject_block(self) -> None:
        rendered = render_request(
            _request(hazards=(Hazard(code=BANNER, subject_id='b'),)), field_chars=4000,
        )
        line = INPUT_TEMPLATE.hazard_line.format(
            description=dict(INPUT_TEMPLATE.hazard_descriptions)[BANNER],
        )
        assert rendered.count(line) == 1
        start, end = _block_bounds(rendered, 'b', 'c')
        assert start < rendered.index(line) < end

    def test_a_member_without_a_hazard_shows_the_none_line(self) -> None:
        rendered = render_request(_request(), field_chars=4000)
        start, end = _block_bounds(rendered, 'a', 'b')
        assert INPUT_TEMPLATE.none_line in rendered[start:end]

    def test_members_appear_in_request_order(self) -> None:
        rendered = render_request(
            _request(members=(_member('c'), _member('a'), _member('b'))), field_chars=4000,
        )
        positions = [
            rendered.index(INPUT_TEMPLATE.member_header.format(id=i)) for i in ('c', 'a', 'b')
        ]
        assert positions == sorted(positions)

    def test_a_long_text_is_cut_and_marked(self) -> None:
        rendered = render_request(
            _request(members=(_member('a', 'x' * 50), _member('b'))), field_chars=10,
        )
        assert 'x' * 10 + INPUT_TEMPLATE.cut_marker in rendered
        assert 'x' * 11 not in rendered

    def test_a_text_at_the_limit_is_not_cut(self) -> None:
        rendered = render_request(
            _request(members=(_member('a', 'x' * 10), _member('b'))), field_chars=10,
        )
        assert INPUT_TEMPLATE.cut_marker not in rendered

    def test_the_excluded_section_appears_only_when_given(self) -> None:
        plain = render_request(_request(), field_chars=4000)
        assert INPUT_TEMPLATE.excluded_header not in plain
        excluded = ExcludedMember(
            id='e1', topic='other-topic', canonical_first_paragraph='Other topic teaches this.',
        )
        rendered = render_request(_request(excluded=(excluded,)), field_chars=4000)
        assert INPUT_TEMPLATE.excluded_header in rendered
        assert INPUT_TEMPLATE.excluded_member.format(
            id='e1', topic='other-topic', first_paragraph='Other topic teaches this.',
        ) in rendered

    def test_the_incumbent_section_appears_only_when_given(self) -> None:
        plain = render_request(_request(), field_chars=4000)
        assert INPUT_TEMPLATE.incumbent_header not in plain
        rendered = render_request(
            _request(incumbent_text='The incumbent canonical.'), field_chars=4000,
        )
        assert INPUT_TEMPLATE.incumbent_header in rendered
        assert 'The incumbent canonical.' in rendered

    def test_no_topic_line_without_a_topic(self) -> None:
        with_topic = render_request(_request(topic='t-slug'), field_chars=4000)
        without = render_request(_request(topic=None), field_chars=4000)
        line = INPUT_TEMPLATE.topic_line.format(topic='t-slug') + '\n'
        assert line in with_topic
        assert with_topic.replace(line, '', 1) == without

    def test_the_claim_and_category_are_shown(self) -> None:
        rendered = render_request(_request(), field_chars=4000)
        assert INPUT_TEMPLATE.claim_line.format(
            claim='cargo fmt is not part of the verify gate.',
        ) in rendered
        assert INPUT_TEMPLATE.category_line.format(category='procedural_knowledge') in rendered

    def test_every_adjudicator_class_code_has_a_description(self) -> None:
        described = dict(INPUT_TEMPLATE.hazard_descriptions)
        adjudicator_codes = {
            code for code, cls in REASON_CODE_CLASSES.items()
            if cls is ReasonCodeClass.ADJUDICATOR
        }
        assert adjudicator_codes
        assert adjudicator_codes <= set(described)
        assert all(described[code].strip() for code in adjudicator_codes)


class TestIdentity:
    def test_a_64_hex_sha256_stable_for_equal_settings(self) -> None:
        identity = adjudicator_identity(_settings())
        assert re.fullmatch('[0-9a-f]{64}', identity)
        assert adjudicator_identity(_settings()) == identity

    @pytest.mark.parametrize(
        ('field', 'value'),
        [
            ('model', 'gpt-6-luna'),
            ('reasoning_effort', 'medium'),
            ('reasoning_effort', None),
            ('endpoint_host', 'proxy.internal:8443'),
            ('endpoint_host', None),
            ('field_chars', 2000),
            ('member_max', 6),
            ('member_min', 3),
        ],
    )
    def test_a_hashed_setting_changes_the_identity(self, field: str, value: object) -> None:
        changed = dataclasses.replace(_settings(), **{field: value})
        assert adjudicator_identity(changed) != adjudicator_identity(_settings())

    def test_the_timeout_does_not_change_the_identity(self) -> None:
        changed = dataclasses.replace(_settings(), timeout_seconds=5.0)
        assert adjudicator_identity(changed) == adjudicator_identity(_settings())

    def test_configs_differing_in_one_hashed_leaf_differ(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv('CONFIG_PATH', str(tmp_path / 'missing.yaml'))
        base = FusedMemoryConfig()
        other = FusedMemoryConfig(
            consolidation_auto=ConsolidationAutoConfig(adjudicator_field_chars=2000),
        )
        assert adjudicator_identity(AdjudicatorSettings.from_config(base)) != (
            adjudicator_identity(AdjudicatorSettings.from_config(other))
        )


class TestParseVerdict:
    def test_a_valid_proceed(self) -> None:
        verdict = parse_verdict(_reply(), _request(), _no_claim_problems)
        assert verdict == Verdict(
            decision=Decision.proceed,
            keep_member_ids=('a', 'b'),
            claim=None,
            reason_code=VerdictReasonCode.same_lesson,
            reason='Both restate the lesson.',
        )
        for field in dataclasses.fields(verdict):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(verdict, field.name, None)

    @pytest.mark.parametrize(
        ('decision', 'code'),
        [('drop', 'not_same_lesson'), ('unresolved', 'conflicting_claims'),
         ('drop', 'same_referent_as_existing_topic')],
    )
    def test_a_valid_drop_or_unresolved(self, decision: str, code: str) -> None:
        verdict = parse_verdict(
            _reply(decision=decision, keep_member_ids=[], reason_code=code),
            _request(), _no_claim_problems,
        )
        assert verdict.decision == Decision(decision)
        assert verdict.keep_member_ids == ()
        assert verdict.reason_code == VerdictReasonCode(code)

    def test_a_replacement_claim_that_passes_is_kept(self) -> None:
        verdict = parse_verdict(
            _reply(claim='A better claim for the kept members.'), _request(), _no_claim_problems,
        )
        assert verdict.claim == 'A better claim for the kept members.'

    def test_the_claim_check_sees_the_claim_and_the_topic(self) -> None:
        seen: list[tuple[str, str | None]] = []

        def check(claim: str, topic: str | None) -> list[str]:
            seen.append((claim, topic))
            return []

        parse_verdict(_reply(claim='Replacement.'), _request(topic=None), check)
        assert seen == [('Replacement.', None)]

    def test_a_null_claim_never_consults_the_check(self) -> None:
        def check(claim: str, topic: str | None) -> list[str]:
            raise AssertionError('consulted')

        parse_verdict(_reply(claim=None), _request(), check)

    @pytest.mark.parametrize(
        ('text', 'reason'),
        [
            ('not json at all', 'JSON'),
            ('[1, 2]', 'object'),
            (json.dumps({'decision': 'proceed', 'keep_member_ids': ['a', 'b'], 'claim': None,
                         'reason_code': 'same_lesson'}), 'reason'),
            (_reply(confidence=0.9), 'confidence'),
            (_reply(decision='merge'), 'merge'),
            (_reply(reason_code='cross_category'), 'cross_category'),
            (_reply(keep_member_ids=['a', 'zz']), 'zz'),
            (_reply(keep_member_ids=['a', 'a']), "'a'"),
            (_reply(keep_member_ids='a'), 'keep_member_ids'),
            (_reply(decision='drop', reason_code='not_same_lesson'), 'keep_member_ids'),
            (_reply(decision='unresolved', keep_member_ids=[], claim='A claim.',
                    reason_code='conflicting_claims'), 'claim'),
            (_reply(reason=7), 'reason'),
        ],
        ids=[
            'non-json', 'array', 'missing-key', 'extra-key', 'decision-enum', 'reason-enum',
            'unshown-kept-id', 'repeated-kept-id', 'keep-not-a-list', 'keep-on-drop',
            'claim-on-unresolved', 'reason-not-a-string',
        ],
    )
    def test_an_invalid_reply_names_why(self, text: str, reason: str) -> None:
        with pytest.raises(VerdictInvalid, match=re.escape(reason)):
            parse_verdict(text, _request(), _no_claim_problems)

    def test_verdict_invalid_is_a_value_error(self) -> None:
        assert issubclass(VerdictInvalid, ValueError)

    def test_a_replacement_claim_the_check_rejects(self) -> None:
        def check(claim: str, topic: str | None) -> list[str]:
            return ['claim opens with a banner', 'claim is two lines']

        with pytest.raises(VerdictInvalid, match='claim opens with a banner'):
            parse_verdict(_reply(claim='[CORRECTION] x'), _request(), check)

    def test_a_verdict_cannot_keep_members_unless_it_proceeds(self) -> None:
        with pytest.raises(ValueError):
            Verdict(
                decision=Decision.drop, keep_member_ids=('a',), claim=None,
                reason_code=VerdictReasonCode.not_same_lesson, reason='r',
            )
        with pytest.raises(ValueError):
            Verdict(
                decision=Decision.unresolved, keep_member_ids=(), claim='c',
                reason_code=VerdictReasonCode.conflicting_claims, reason='r',
            )


class TestOutputSchema:
    def test_text_format_is_strict_json_schema(self) -> None:
        sent = text_format()
        assert set(sent) == {'type', 'name', 'strict', 'schema'}
        assert sent['type'] == 'json_schema'
        assert sent.get('strict') is True
        assert isinstance(sent['name'], str) and sent['name']
        assert sent['schema'] == output_schema()

    def test_strict_mode_requires_every_property_and_no_others(self) -> None:
        schema = output_schema()
        assert schema['type'] == 'object'
        assert schema['additionalProperties'] is False
        assert set(schema['required']) == set(schema['properties'])
        assert set(schema['properties']) == {
            'decision', 'keep_member_ids', 'claim', 'reason_code', 'reason',
        }

    def test_the_enums_come_from_the_vocabularies(self) -> None:
        properties = output_schema()['properties']
        assert properties['decision']['enum'] == [d.value for d in Decision]
        assert properties['reason_code']['enum'] == [c.value for c in VerdictReasonCode]

    def test_the_schema_is_json_serialisable(self) -> None:
        sent = text_format()
        assert json.loads(json.dumps(sent)) == sent

    def test_mutating_a_returned_format_reaches_neither_the_identity_nor_the_next_call(
        self,
    ) -> None:
        identity = adjudicator_identity(_settings())
        pristine = text_format()
        held = text_format()
        held['strict'] = False
        held['schema'].clear()
        output_schema()['required'].clear()
        assert text_format() == pristine
        assert adjudicator_identity(_settings()) == identity


def _words(text: str) -> set[str]:
    return set(re.findall(r'[a-z]+(?:_[a-z]+)*', text))


def _snake_case_words(text: str) -> set[str]:
    return {word for word in _words(text) if '_' in word}


class TestInstructions:
    def test_the_mixed_category_rule_is_gone(self) -> None:
        assert 'cross_category' not in INSTRUCTIONS

    def test_every_output_field_decision_and_reason_code_is_named(self) -> None:
        vocabulary = {
            *output_schema()['properties'],
            *(d.value for d in Decision),
            *(c.value for c in VerdictReasonCode),
        }
        assert vocabulary - _words(INSTRUCTIONS) == set()

    def test_every_snake_case_name_is_an_output_field_or_a_reason_code(self) -> None:
        known = {*output_schema()['properties'], *(c.value for c in VerdictReasonCode)}
        assert _snake_case_words(INSTRUCTIONS) - known == set()


_IMPORT_PROBE_BUDGET_S = 120.0


def _probe(body: str) -> subprocess.CompletedProcess[str]:
    return run_on_own_time([sys.executable, '-c', body], budget=_IMPORT_PROBE_BUDGET_S)


_EXECUTOR = 'fused_memory.reconciliation.consolidation_auto_executor'


class TestImportsNoExecutorCode:
    """K3: the adjudicator imports no executor code; nor does its calibration."""

    @pytest.mark.parametrize(
        'module',
        [
            'fused_memory.reconciliation.consolidation_adjudicator',
            'fused_memory.reconciliation.consolidation_adjudicator_calibration',
        ],
    )
    def test_importing_does_not_load_the_executor(self, module: str) -> None:
        result = _probe(
            'import importlib, sys\n'
            f'importlib.import_module({module!r})\n'
            f'assert {_EXECUTOR!r} not in sys.modules\n',
        )
        assert result.returncode == 0, result.stderr


def _body(**overrides) -> dict:
    return json.loads(_reply(**overrides))


async def _adjudicate(double: ScriptedResponses, **overrides):
    kwargs = {
        'settings': _settings(),
        'model_call': double,
        'claim_check': _no_claim_problems,
    }
    kwargs.update(overrides)
    request = kwargs.pop('request', _request())
    return await adjudicate(request, **kwargs)


class TestTheCall:
    @pytest.mark.asyncio
    async def test_one_call_with_the_settings_and_the_constants(self) -> None:
        double = ScriptedResponses(reply_json(_body()))
        settings = _settings(model='gpt-6-luna', reasoning_effort='high', field_chars=10)
        request = _request(members=(_member('a', 'y' * 40), _member('b')))
        await _adjudicate(double, settings=settings, request=request)
        assert double.requests == [{
            'model': 'gpt-6-luna',
            'reasoning_effort': 'high',
            'instructions': INSTRUCTIONS,
            'user_content': render_request(request, field_chars=10),
            'text_format': text_format(),
            'max_output_tokens': MAX_OUTPUT_TOKENS,
            'timeout': 60.0,
        }]

    @pytest.mark.asyncio
    async def test_a_valid_reply_is_an_adjudication_with_its_call_meta(self) -> None:
        double = ScriptedResponses(reply_json(_body()))
        result = await _adjudicate(double)
        assert isinstance(result, Adjudication)
        assert result.verdict.decision is Decision.proceed
        assert result.verdict.keep_member_ids == ('a', 'b')
        meta = result.meta
        assert meta.identity == adjudicator_identity(_settings())
        assert meta.resolved_model == RESOLVED_MODEL
        assert (meta.input_tokens, meta.output_tokens) == (USAGE.input_tokens, USAGE.output_tokens)
        price = list_price('gpt-6.1-sol')
        assert price is not None
        assert meta.cost_usd == pytest.approx(price.usd(USAGE.input_tokens, USAGE.output_tokens))
        assert meta.latency_s >= 0

    @pytest.mark.asyncio
    async def test_an_unpriced_model_costs_none(self) -> None:
        double = ScriptedResponses(reply_json(_body()))
        result = await _adjudicate(double, settings=_settings(model='no-such-model'))
        assert isinstance(result, Adjudication)
        assert result.meta.cost_usd is None

    @pytest.mark.asyncio
    async def test_missing_usage_costs_none(self) -> None:
        double = ScriptedResponses(reply_json(_body(), usage=None))
        result = await _adjudicate(double)
        assert isinstance(result, Adjudication)
        assert result.meta.cost_usd is None
        assert (result.meta.input_tokens, result.meta.output_tokens) == (None, None)


class TestFailures:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'error',
        [TimeoutError(), sdk_timeout(), connection_error(), rate_limited(),
         server_error(500), server_error(502), server_error(503),
         unauthorized(), forbidden(), no_api_key()],
        ids=['timeout', 'sdk-timeout', 'connection', '429', '500', '502', '503',
             '401', '403', 'no-api-key'],
    )
    async def test_a_transport_failure(self, error: BaseException) -> None:
        double = ScriptedResponses(error)
        result = await _adjudicate(double)
        assert isinstance(result, AdjudicationFailure)
        assert result.kind is FailureKind.TRANSPORT
        assert len(double.requests) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('answer', 'why'),
        [
            (bad_request(), 'BadRequestError'),
            (incomplete_reply('max_output_tokens'), 'max_output_tokens'),
            (reply_text('not json'), 'JSON'),
            (reply_text(None), 'JSON'),
            (reply_json({**_body(), 'confidence': 0.9}), 'confidence'),
            (reply_json(_body(keep_member_ids=['a', 'zz'])), 'zz'),
        ],
        ids=['400', 'incomplete', 'malformed', 'no-text', 'unknown-key', 'unshown-kept-id'],
    )
    async def test_a_content_failure_names_why(self, answer, why: str) -> None:
        double = ScriptedResponses(answer)
        result = await _adjudicate(double)
        assert isinstance(result, AdjudicationFailure)
        assert result.kind is FailureKind.CONTENT
        assert why in result.detail
        assert len(double.requests) == 1

    @pytest.mark.asyncio
    async def test_a_replacement_claim_the_check_rejects(self) -> None:
        def check(claim: str, topic: str | None) -> list[str]:
            return ['claim restates the topic slug']

        double = ScriptedResponses(reply_json(_body(claim='Replacement claim.')))
        result = await _adjudicate(double, claim_check=check)
        assert isinstance(result, AdjudicationFailure)
        assert result.kind is FailureKind.CONTENT
        assert 'claim restates the topic slug' in result.detail

    @pytest.mark.asyncio
    async def test_a_failed_reply_still_reports_its_call(self) -> None:
        result = await _adjudicate(ScriptedResponses(incomplete_reply()))
        assert isinstance(result, AdjudicationFailure)
        assert result.meta is not None
        assert result.meta.identity == adjudicator_identity(_settings())
        assert result.meta.input_tokens == USAGE.input_tokens

    @pytest.mark.asyncio
    async def test_cancellation_propagates(self) -> None:
        with pytest.raises(asyncio.CancelledError):
            await _adjudicate(ScriptedResponses(asyncio.CancelledError()))

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'fault', [KeyError('model'), TypeError('unexpected keyword'), RuntimeError('wrapper bug')],
        ids=['KeyError', 'TypeError', 'RuntimeError'],
    )
    async def test_a_fault_in_the_model_call_propagates_rather_than_scoring(
        self, fault: Exception,
    ) -> None:
        with pytest.raises(type(fault)):
            await _adjudicate(ScriptedResponses(fault))

    @pytest.mark.asyncio
    async def test_a_fault_in_the_claim_check_propagates(self) -> None:
        def check(claim: str, topic: str | None) -> list[str]:
            raise LookupError('claim rules unavailable')

        with pytest.raises(LookupError):
            await _adjudicate(
                ScriptedResponses(reply_json(_body(claim='Replacement claim.'))),
                claim_check=check,
            )


def _config(monkeypatch, tmp_path, *, openai_section, **auto) -> FusedMemoryConfig:
    monkeypatch.setenv('CONFIG_PATH', str(tmp_path / 'missing.yaml'))
    return FusedMemoryConfig(
        llm=LLMConfig(providers=LLMProvidersConfig(openai=openai_section)),
        consolidation_auto=ConsolidationAutoConfig(**auto),
    )


class TestFromConfig:
    def test_the_k7_leaves_member_min_and_the_endpoint_host(self, monkeypatch, tmp_path) -> None:
        config = _config(
            monkeypatch, tmp_path,
            openai_section=OpenAIProviderConfig(
                api_key='sk-x', api_url='https://proxy.example:8443/v1',
            ),
            adjudicator_model='gpt-6-luna', adjudicator_reasoning_effort=None,
            adjudicator_timeout_seconds=12.5, adjudicator_field_chars=1500,
            adjudicator_member_max=7, member_min=3,
        )
        assert AdjudicatorSettings.from_config(config) == AdjudicatorSettings(
            model='gpt-6-luna', reasoning_effort=None, endpoint_host='proxy.example:8443',
            timeout_seconds=12.5, field_chars=1500, member_max=7, member_min=3,
        )

    def test_no_provider_section_is_the_sdk_default_endpoint(self, monkeypatch, tmp_path) -> None:
        config = _config(monkeypatch, tmp_path, openai_section=None)
        assert AdjudicatorSettings.from_config(config).endpoint_host is None


def _sdk_response() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        output_text='{}', status='completed', incomplete_details=None,
        model='gpt-6.1-sol', usage=None,
    )


class TestOpenAIModelCall:
    @virtual_clock_test
    async def test_the_client_uses_the_provider_section_and_one_attempt(
        self, monkeypatch, tmp_path,
    ) -> None:
        config = _config(
            monkeypatch, tmp_path,
            openai_section=OpenAIProviderConfig(api_key='sk-x', api_url='https://h.example/v1'),
        )
        client = sdk_client_double()
        client.responses.create = AsyncMock(return_value=_sdk_response())
        with patch('openai.AsyncOpenAI', return_value=client) as ctor:
            reply = await openai_model_call(config)(
                model='gpt-6.1-sol', instructions='i', user_content='u',
                text_format=text_format(), max_output_tokens=MAX_OUTPUT_TOKENS,
                timeout=5.0, reasoning_effort='low',
            )
        ctor.assert_called_once_with(
            api_key='sk-x', base_url='https://h.example/v1', max_retries=0,
        )
        assert reply.text == '{}'
        assert client.responses.create.call_args.kwargs['text'] == {'format': text_format()}

    @virtual_clock_test
    async def test_a_blank_key_is_not_forwarded(self, monkeypatch, tmp_path) -> None:
        config = _config(
            monkeypatch, tmp_path,
            openai_section=OpenAIProviderConfig(api_key='', api_url='https://h.example/v1'),
        )
        client = sdk_client_double()
        client.responses.create = AsyncMock(return_value=_sdk_response())
        with patch('openai.AsyncOpenAI', return_value=client) as ctor:
            await openai_model_call(config)(
                model='m', instructions='i', user_content='u', text_format=text_format(),
                max_output_tokens=10, timeout=5.0, reasoning_effort=None,
            )
        ctor.assert_called_once_with(base_url='https://h.example/v1', max_retries=0)

    def test_an_overridden_ambient_base_url_is_reported(
        self, monkeypatch, tmp_path, caplog,
    ) -> None:
        config = _config(
            monkeypatch, tmp_path,
            openai_section=OpenAIProviderConfig(api_key='sk-x', api_url='https://h.example/v1'),
        )
        monkeypatch.setenv('OPENAI_BASE_URL', f'https://ambient-{tmp_path.name}.example/v1')
        with caplog.at_level('WARNING'):
            openai_model_call(config)
        assert any(
            'consolidation adjudicator' in record.getMessage()
            and 'OPENAI_BASE_URL' in record.getMessage()
            for record in caplog.records
        )
