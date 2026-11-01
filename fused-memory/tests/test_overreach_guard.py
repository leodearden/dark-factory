"""Tests for fused_memory.services.overreach_guard (task 6137).

The ruling-overreach guard's class vocabulary and judge core, per
docs/ruling-overreach-guard-design-2026-10-01/design.md §4 and §8.
"""
from __future__ import annotations

import dataclasses
from collections.abc import MutableMapping
from typing import Any

import pytest
from graphiti_core.prompts.models import Message
from pydantic import BaseModel

from fused_memory.models.enums import MemoryCategory
from fused_memory.services.overreach_guard import (
    CLASSIFIER_NAMES,
    EPISODE_CLASSIFIERS,
    GUARDED_CATEGORIES,
    HEAD_CHARS,
    JUDGE_SYSTEM_PROMPT,
    LABELS,
    OVERREACH,
    PROMPT_VERSION,
    EdgeJudgement,
    JudgeEdge,
    JudgeReplyError,
    build_judge_prompt,
    category_of,
    is_guarded_category,
    judge_edges,
    matching_classifiers,
    minted_edges,
    parse_judge_reply,
)

RULING_HEADED = 'RULING (Leo, 2026-09-01, esc-1-2): x'
PLAIN_DECISION = 'Task 848 (stale merge-lane cleanup) is done as of 2026-04-20 with commit bb7101'


class TestClassPredicate:
    @pytest.mark.parametrize('category', list(MemoryCategory))
    def test_only_decision_records_are_in_class(self, category: MemoryCategory) -> None:
        expected = category is MemoryCategory.decisions_and_rationale
        assert is_guarded_category(category) is expected

    def test_an_unknown_category_is_out_of_class(self) -> None:
        assert is_guarded_category(None) is False

    @pytest.mark.parametrize('category', list(MemoryCategory))
    def test_a_category_value_decodes_to_its_member(self, category: MemoryCategory) -> None:
        assert category_of(category.value) is category
        assert category_of(category) is category

    @pytest.mark.parametrize('value', [None, '', 'bogus', 'Decisions_And_Rationale', 7])
    def test_an_absent_or_unknown_value_decodes_to_none(self, value: Any) -> None:
        assert category_of(value) is None

    def test_the_class_is_exactly_the_decisions_category(self) -> None:
        assert frozenset({MemoryCategory.decisions_and_rationale}) == GUARDED_CATEGORIES

    def test_classifier_names_are_the_report_keys_in_order(self) -> None:
        assert CLASSIFIER_NAMES == (
            'category_decisions', 'header_ruling_paren', 'header_ruling',
            'ruling_lexeme_head', 'decision_anchor_head',
        )

    def test_episode_classifiers_are_keyed_in_name_order(self) -> None:
        assert tuple(EPISODE_CLASSIFIERS) == CLASSIFIER_NAMES

    def test_episode_classifiers_are_read_only(self) -> None:
        assert not isinstance(EPISODE_CLASSIFIERS, MutableMapping)

    @pytest.mark.parametrize('name', CLASSIFIER_NAMES)
    def test_each_classifier_is_a_category_content_predicate(self, name: str) -> None:
        holds = EPISODE_CLASSIFIERS[name](MemoryCategory.decisions_and_rationale, RULING_HEADED)
        assert holds is True

    def test_a_ruling_headed_decision_matches_all_five(self) -> None:
        matched = matching_classifiers(MemoryCategory.decisions_and_rationale, RULING_HEADED)
        assert matched == CLASSIFIER_NAMES

    def test_the_same_content_in_another_category_drops_only_the_category(self) -> None:
        matched = matching_classifiers(MemoryCategory.temporal_facts, RULING_HEADED)
        assert matched == CLASSIFIER_NAMES[1:]

    def test_a_plain_decision_record_matches_only_the_category(self) -> None:
        matched = matching_classifiers(MemoryCategory.decisions_and_rationale, PLAIN_DECISION)
        assert matched == ('category_decisions',)

    @pytest.mark.parametrize('name', ['ruling_lexeme_head', 'decision_anchor_head'])
    def test_a_lexeme_past_the_head_window_does_not_count(self, name: str) -> None:
        content = 'x' * 250 + ' ruling (Leo, 2026-09-01, esc-1-2)'
        assert HEAD_CHARS == 200
        assert EPISODE_CLASSIFIERS[name](MemoryCategory.decisions_and_rationale, content) is False

    def test_the_label_vocabulary(self) -> None:
        assert LABELS == (
            'holding', 'bookkeeping', 'context', 'overreach', 'misbound', 'unjudgeable',
        )
        assert OVERREACH == 'overreach'
        assert OVERREACH in LABELS


# --------------------------------------------------------------------------- #
# The judge core (design.md §8 leaf 2)
# --------------------------------------------------------------------------- #

EPISODE = 'EP-0000-mint'


def _row(uuid: str, episodes: list[str], *, source: str = 'S', target: str = 'T') -> dict:
    return {
        'uuid': uuid, 'fact': f'fact of {uuid}', 'source_name': source,
        'target_name': target, 'episodes': episodes,
    }


def _judge_edge(uuid: str) -> JudgeEdge:
    return JudgeEdge(uuid=uuid, fact=f'fact of {uuid}', source_name='S', target_name='T')


class TestMintedEdges:
    def test_keeps_exactly_the_rows_the_episode_minted(self) -> None:
        rows = [
            _row('minted', [EPISODE]),
            _row('corroborated', ['OTHER', EPISODE]),
            _row('no-episodes', []),
            _row('minted-then-corroborated', [EPISODE, 'LATER']),
        ]
        assert minted_edges(EPISODE, rows) == (
            _judge_edge('minted'), _judge_edge('minted-then-corroborated'),
        )

    def test_input_order_is_preserved(self) -> None:
        rows = [_row(uuid, [EPISODE]) for uuid in ('z', 'a', 'm')]
        assert [e.uuid for e in minted_edges(EPISODE, rows)] == ['z', 'a', 'm']

    def test_endpoint_names_are_carried(self) -> None:
        (edge,) = minted_edges(EPISODE, [_row('e', [EPISODE], source='Task 6128', target='ANGLE')])
        assert (edge.source_name, edge.target_name) == ('Task 6128', 'ANGLE')

    def test_a_judge_edge_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            _judge_edge('e').__setattr__('fact', 'other')


EDGES = (
    JudgeEdge(
        uuid='2789bcbf-0000-4000-8000-000000000001',
        fact='The angular gate for orient_exp is narrowed to accept ANGLE only.',
        source_name='orient_exp', target_name='ANGLE',
    ),
    JudgeEdge(
        uuid='4f99fbf2-0000-4000-8000-000000000002',
        fact='Narrowing the gate implies sin should not accept dimensionless arguments.',
        source_name='Task 6080', target_name='sin',
    ),
)


class TestBuildJudgePrompt:
    def test_the_whole_episode_body_is_included(self) -> None:
        body = 'RULING (Leo): ' + 'b' * 4970 + ' END-OF-BODY'
        assert len(body) > 4990
        assert body in build_judge_prompt(body, EDGES)

    def test_each_edge_appears_under_its_ordinal_key(self) -> None:
        lines = build_judge_prompt('body', EDGES).splitlines()
        for ordinal, edge in enumerate(EDGES, start=1):
            (line,) = [ln for ln in lines if ln.startswith(f'E{ordinal}:')]
            assert edge.fact in line
            assert edge.source_name in line
            assert edge.target_name in line

    def test_edge_uuids_never_reach_the_prompt(self) -> None:
        text = build_judge_prompt('body', EDGES)
        assert not any(edge.uuid in text for edge in EDGES)

    @pytest.mark.parametrize('label', LABELS)
    def test_the_rubric_names_every_label(self, label: str) -> None:
        assert label in JUDGE_SYSTEM_PROMPT

    def test_the_prompt_version_is_a_non_empty_string(self) -> None:
        assert isinstance(PROMPT_VERSION, str)
        assert PROMPT_VERSION.strip()


def _item(edge: str, label: str = 'holding', rationale: str = 'quotes "narrow to ANGLE ONLY"') -> dict:
    return {'edge': edge, 'label': label, 'rationale': rationale}


def _reply(*items: Any) -> dict:
    return {'judgements': list(items)}


class TestParseJudgeReply:
    def test_judgements_map_back_to_uuids_in_edge_order(self) -> None:
        reply = _reply(_item('E2', 'overreach', 'r2'), _item('E1', 'holding', 'r1'))
        assert parse_judge_reply(reply, EDGES) == (
            EdgeJudgement(edge_uuid=EDGES[0].uuid, label='holding', rationale='r1'),
            EdgeJudgement(edge_uuid=EDGES[1].uuid, label='overreach', rationale='r2'),
        )

    def test_extra_keys_inside_an_item_are_ignored(self) -> None:
        reply = _reply({**_item('E1'), 'confidence': 0.9}, _item('E2'))
        assert [j.label for j in parse_judge_reply(reply, EDGES)] == ['holding', 'holding']

    def test_the_reply_error_is_a_value_error(self) -> None:
        assert issubclass(JudgeReplyError, ValueError)

    @pytest.mark.parametrize('reply', [
        pytest.param(['E1'], id='not-a-mapping'),
        pytest.param({}, id='no-judgements'),
        pytest.param({'judgements': 'E1'}, id='judgements-not-a-list'),
        pytest.param(_reply('E1', _item('E2')), id='item-not-a-mapping'),
        pytest.param(_reply({'edge': 'E1', 'label': 'holding'}, _item('E2')), id='missing-field'),
        pytest.param(
            _reply({'edge': 'E1', 'label': 'holding', 'rationale': 3}, _item('E2')),
            id='non-str-field',
        ),
        pytest.param(_reply(_item('E1', label='maybe'), _item('E2')), id='label-outside-vocabulary'),
        pytest.param(_reply(_item('E1', rationale='   '), _item('E2')), id='blank-rationale'),
        pytest.param(_reply(_item('E1'), _item('E2'), _item('E9')), id='unknown-key'),
        pytest.param(_reply(_item('E1'), _item('E1'), _item('E2')), id='duplicate-key'),
        pytest.param(_reply(_item('E1')), id='offered-but-not-judged'),
    ])
    def test_a_malformed_reply_is_rejected(self, reply: Any) -> None:
        with pytest.raises(JudgeReplyError):
            parse_judge_reply(reply, EDGES)


class _FakeLLM:
    """Records every generate_response call and answers with a canned reply."""

    def __init__(self, reply: Any = None, error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def generate_response(
        self, messages: list[Message], response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None, **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append({
            'messages': messages, 'response_model': response_model,
            'max_tokens': max_tokens, **kwargs,
        })
        if self.error is not None:
            raise self.error
        return self.reply


def _enums(schema: Any) -> list[list[Any]]:
    if isinstance(schema, dict):
        found = [schema['enum']] if 'enum' in schema else []
        return found + [e for value in schema.values() for e in _enums(value)]
    if isinstance(schema, list):
        return [e for value in schema for e in _enums(value)]
    return []


class TestJudgeEdges:
    @pytest.mark.asyncio
    async def test_one_call_with_the_rubric_then_the_prompt(self) -> None:
        llm = _FakeLLM(reply=_reply(_item('E1'), _item('E2', 'overreach')))
        await judge_edges(llm, 'the episode body', EDGES)
        (call,) = llm.calls
        system, user = call['messages']
        assert isinstance(system, Message) and isinstance(user, Message)
        assert (system.role, system.content) == ('system', JUDGE_SYSTEM_PROMPT)
        assert (user.role, user.content) == ('user', build_judge_prompt('the episode body', EDGES))

    @pytest.mark.asyncio
    async def test_the_response_model_asks_for_judgements_in_the_label_vocabulary(self) -> None:
        llm = _FakeLLM(reply=_reply(_item('E1'), _item('E2')))
        await judge_edges(llm, 'body', EDGES)
        model = llm.calls[0]['response_model']
        assert isinstance(model, type) and issubclass(model, BaseModel)
        schema = model.model_json_schema()
        assert schema['properties']['judgements']['type'] == 'array'
        assert list(LABELS) in _enums(schema)

    @pytest.mark.asyncio
    async def test_returns_the_parsed_judgements(self) -> None:
        llm = _FakeLLM(reply=_reply(_item('E1', 'holding', 'r1'), _item('E2', 'overreach', 'r2')))
        judged = await judge_edges(llm, 'body', EDGES)
        assert judged == (
            EdgeJudgement(edge_uuid=EDGES[0].uuid, label='holding', rationale='r1'),
            EdgeJudgement(edge_uuid=EDGES[1].uuid, label='overreach', rationale='r2'),
        )

    @pytest.mark.asyncio
    async def test_no_edges_means_no_call(self) -> None:
        llm = _FakeLLM(reply=_reply())
        assert await judge_edges(llm, 'body', ()) == ()
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_a_malformed_reply_raises(self) -> None:
        llm = _FakeLLM(reply=_reply(_item('E1', label='maybe'), _item('E2')))
        with pytest.raises(JudgeReplyError):
            await judge_edges(llm, 'body', EDGES)

    @pytest.mark.asyncio
    async def test_an_llm_failure_propagates_to_the_caller(self) -> None:
        llm = _FakeLLM(error=RuntimeError('provider down'))
        with pytest.raises(RuntimeError, match='provider down'):
            await judge_edges(llm, 'body', EDGES)
