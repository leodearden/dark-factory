"""The link adjudicator (plans/write-triage-link-healing-prd.md H2, task 6184).

The mirror tests read the LIVE rater brief: the scale's one home in code is
``link_heal.Verdict``, and the brief is the document raters and the model read,
so the two must agree word for word (INV-10).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

import pytest

from fused_memory.maintenance.link_adjudicator import (
    RATER_BRIEF_PATH,
    TRUNCATION_MARKER,
    VERDICT_OUTPUT_SCHEMA,
    AdjudicationFailure,
    LinkPair,
    LinkVerdict,
    Shard,
    ShardReply,
    adjudicate_links,
    cap_text,
)
from fused_memory.maintenance.link_heal import (
    AGREEING_VERDICTS,
    BELONGS_VERDICTS,
    MISFILE_VERDICTS,
    Verdict,
)

_WORD_BULLET = re.compile(r'^- \*\*([A-Z]+)\*\*:', re.MULTILINE)
_CLASS_LINE = re.compile(r'^Belongs = (.*?)\. Wrong \(a misfile\) = (.*?)\.$', re.MULTILINE)
_MARKER_EXAMPLE = re.compile(r'`(…\[truncated, [^`]*\])`')
_UPPER_WORD = re.compile(r'\b[A-Z]{2,}\b')


def _brief() -> str:
    return RATER_BRIEF_PATH.read_text(encoding='utf-8')


def _brief_words() -> list[str]:
    return _WORD_BULLET.findall(_brief())


def _class_line() -> tuple[str, str]:
    match = _CLASS_LINE.search(_brief())
    assert match is not None, 'the brief lost its Belongs/Wrong line'
    return match.group(1), match.group(2)


def _verdicts_named(fragment: str) -> frozenset[Verdict]:
    return frozenset(Verdict(word) for word in _UPPER_WORD.findall(fragment))


class TestTheBriefIsTheScale:
    def test_the_brief_path_resolves_to_the_committed_brief(self):
        assert RATER_BRIEF_PATH.is_file(), RATER_BRIEF_PATH
        assert RATER_BRIEF_PATH.parts[-2:] == ('calibration', 'write_triage_rater_brief.md')

    def test_the_brief_lists_the_verdict_words_in_order(self):
        assert _brief_words() == [verdict.value for verdict in Verdict]

    def test_the_belongs_line_names_the_belongs_verdicts(self):
        belongs, _ = _class_line()
        assert _verdicts_named(belongs) == BELONGS_VERDICTS

    def test_the_misfile_line_names_the_misfile_verdicts(self):
        _, misfile = _class_line()
        assert _verdicts_named(misfile) == MISFILE_VERDICTS

    def test_agreeing_is_belongs_without_corrects(self):
        assert BELONGS_VERDICTS - {Verdict.CORRECTS} == AGREEING_VERDICTS

    def test_the_brief_marker_example_is_the_truncation_marker(self):
        examples = _MARKER_EXAMPLE.findall(_brief())
        assert examples == [TRUNCATION_MARKER.format(total='N')]


class TestVerdictOutputSchema:
    def _item(self) -> dict:
        verdicts = VERDICT_OUTPUT_SCHEMA['properties']['verdicts']
        assert verdicts['type'] == 'array'
        return verdicts['items']

    def test_the_verdict_enum_is_the_brief_word_list(self):
        assert self._item()['properties']['verdict']['enum'] == _brief_words()

    def test_each_item_requires_exactly_id_verdict_and_reason(self):
        item = self._item()
        assert sorted(item['required']) == ['id', 'reason', 'verdict']
        assert sorted(item['properties']) == ['id', 'reason', 'verdict']
        assert item['additionalProperties'] is False

    def test_the_top_level_requires_only_verdicts(self):
        assert VERDICT_OUTPUT_SCHEMA['type'] == 'object'
        assert VERDICT_OUTPUT_SCHEMA['required'] == ['verdicts']
        assert VERDICT_OUTPUT_SCHEMA['additionalProperties'] is False


class TestCapText:
    def test_a_text_within_the_cap_is_unchanged(self):
        text = 'x' * 40
        assert cap_text(text, 40) is text

    def test_a_longer_text_is_cut_and_marked_with_its_full_length(self):
        text = 'abcdefghij' * 5
        assert cap_text(text, 20) == text[:20] + TRUNCATION_MARKER.format(total=50)

    def test_an_already_capped_text_is_left_as_is(self):
        original = 'y' * 12345
        capped = original[:4000] + TRUNCATION_MARKER.format(total=len(original))
        assert len(capped) == 4031
        assert cap_text(capped, 4000) == capped


def _answer(shard: Shard, word_for: Callable[[str], str] = lambda _text: 'EXTENDS') -> dict:
    return {
        'verdicts': [
            {'id': item.item_id, 'verdict': word_for(item.child_text), 'reason': 'because'}
            for item in shard.items
        ],
    }


Script = Callable[[int, Shard], ShardReply]


def _valid(call: int, shard: Shard) -> ShardReply:
    return ShardReply(success=True, structured_output=_answer(shard))


class FakeAsker:
    """Records every Shard it is asked, and answers through *script*(call index, shard)."""

    def __init__(self, script: Script = _valid) -> None:
        self.shards: list[Shard] = []
        self._script = script

    async def __call__(self, shard: Shard) -> ShardReply:
        self.shards.append(shard)
        return self._script(len(self.shards) - 1, shard)


def _on_call(call_index: int, broken: Callable[[Shard], ShardReply]) -> Script:
    """Answer validly, except on *call_index*, which gets *broken*'s reply."""
    def script(call: int, shard: Shard) -> ShardReply:
        return broken(shard) if call == call_index else _valid(call, shard)
    return script


def _payload(edit: Callable[[dict], object]) -> Callable[[Shard], ShardReply]:
    def broken(shard: Shard) -> ShardReply:
        return ShardReply(success=True, structured_output=edit(_answer(shard)))
    return broken


def _pairs(count: int) -> list[LinkPair]:
    return [
        LinkPair(key=f'dark_factory:child-{index}', child_text=f'child {index}', parent_text=f'parent {index}')
        for index in range(count)
    ]


async def _adjudicate(
    pairs: list[LinkPair],
    ask: FakeAsker,
    *,
    shard_size: int = 2,
    field_chars: int = 4000,
    failure_streak: int = 3,
    on_failure_storm=None,
) -> list[LinkVerdict]:
    return await adjudicate_links(
        pairs,
        model='opus',
        shard_size=shard_size,
        field_chars=field_chars,
        failure_streak=failure_streak,
        ask=ask,
        on_failure_storm=on_failure_storm,
    )


def _failures(verdicts: list[LinkVerdict]) -> list[AdjudicationFailure | None]:
    return [verdict.failure for verdict in verdicts]


class TestSharding:
    @pytest.mark.asyncio
    async def test_pairs_go_in_shards_and_come_back_in_input_order(self):
        words = ['SAME', 'EXTENDS', 'SUBSUMED', 'CORRECTS', 'RELATED']
        word_for = {f'child {index}': word for index, word in enumerate(words)}

        def reversed_reply(call: int, shard: Shard) -> ShardReply:
            answer = _answer(shard, word_for.__getitem__)
            answer['verdicts'].reverse()
            return ShardReply(success=True, structured_output=answer)

        ask = FakeAsker(reversed_reply)
        pairs = _pairs(5)

        verdicts = await _adjudicate(pairs, ask)

        assert [len(shard.items) for shard in ask.shards] == [2, 2, 1]
        assert [verdict.key for verdict in verdicts] == [pair.key for pair in pairs]
        assert [verdict.verdict for verdict in verdicts] == [Verdict(word) for word in words]
        assert {verdict.model for verdict in verdicts} == {'opus'}
        assert all(shard.model == 'opus' for shard in ask.shards)
        assert all(verdict.reason == 'because' and not verdict.failed for verdict in verdicts)

    @pytest.mark.asyncio
    async def test_texts_reach_the_asker_capped(self):
        ask = FakeAsker()
        pair = LinkPair(key='k', child_text='c' * 50, parent_text='p' * 10)

        await _adjudicate([pair], ask, field_chars=20)

        (item,) = ask.shards[0].items
        assert item.child_text == cap_text('c' * 50, 20)
        assert item.parent_text == 'p' * 10

    @pytest.mark.asyncio
    async def test_item_ids_are_positional_and_never_the_key(self):
        ask = FakeAsker()
        pairs = _pairs(5)

        await _adjudicate(pairs, ask)

        assert [[item.item_id for item in shard.items] for shard in ask.shards] == [
            ['p1', 'p2'], ['p1', 'p2'], ['p1'],
        ]
        shown = json.dumps([[vars(item) for item in shard.items] for shard in ask.shards])
        assert not any(pair.key in shown for pair in pairs)

    @pytest.mark.asyncio
    async def test_duplicate_pair_keys_are_refused(self):
        pairs = [LinkPair('dup', 'a', 'b'), LinkPair('dup', 'c', 'd')]

        with pytest.raises(ValueError, match='dup'):
            await _adjudicate(pairs, FakeAsker())


def _drop_last(answer: dict) -> dict:
    answer['verdicts'].pop()
    return answer


def _repeat_first(answer: dict) -> dict:
    answer['verdicts'][1]['id'] = answer['verdicts'][0]['id']
    return answer


def _rename_first(answer: dict) -> dict:
    answer['verdicts'][0]['id'] = 'p99'
    return answer


class TestAShardReplyFailsWhole:
    @pytest.mark.parametrize(
        ('edit', 'failure'),
        [
            (_drop_last, AdjudicationFailure.OMITTED_ID),
            (_repeat_first, AdjudicationFailure.DUPLICATE_ID),
            (_rename_first, AdjudicationFailure.UNKNOWN_ID),
        ],
    )
    @pytest.mark.asyncio
    async def test_a_bad_id_set_fails_every_pair_of_its_shard(self, edit, failure):
        ask = FakeAsker(_on_call(1, _payload(edit)))

        verdicts = await _adjudicate(_pairs(5), ask)

        assert _failures(verdicts) == [None, None, failure, failure, None]
        assert all(verdict.verdict is None for verdict in verdicts if verdict.failed)
        assert all(verdict.verdict is Verdict.EXTENDS for verdict in verdicts if not verdict.failed)

    @pytest.mark.parametrize(
        'edit',
        [
            pytest.param(lambda answer: answer['verdicts'], id='not-a-dict'),
            pytest.param(lambda answer: '{"verdicts": [', id='undecodable-json-string'),
            pytest.param(lambda answer: {'answers': answer['verdicts']}, id='no-verdicts-list'),
            pytest.param(
                lambda answer: {'verdicts': [{**entry, 'verdict': 'MAYBE'} for entry in answer['verdicts']]},
                id='verdict-word-outside-the-scale',
            ),
            pytest.param(
                lambda answer: {'verdicts': [
                    {key: value for key, value in entry.items() if key != 'reason'}
                    for entry in answer['verdicts']
                ]},
                id='entry-without-reason',
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_a_malformed_reply_is_a_parse_failure(self, edit):
        ask = FakeAsker(_on_call(0, _payload(edit)))

        verdicts = await _adjudicate(_pairs(3), ask)

        assert _failures(verdicts) == [
            AdjudicationFailure.PARSE_FAILURE, AdjudicationFailure.PARSE_FAILURE, None,
        ]
        assert all(verdict.detail for verdict in verdicts if verdict.failed)

    @pytest.mark.asyncio
    async def test_a_valid_json_string_payload_is_accepted(self):
        ask = FakeAsker(_on_call(0, _payload(json.dumps)))

        verdicts = await _adjudicate(_pairs(2), ask)

        assert [verdict.verdict for verdict in verdicts] == [Verdict.EXTENDS, Verdict.EXTENDS]

    @pytest.mark.asyncio
    async def test_a_failed_cli_reply_fails_the_shard_with_its_detail(self):
        def failed(shard: Shard) -> ShardReply:
            return ShardReply(success=False, detail='subtype=error_max_turns')

        verdicts = await _adjudicate(_pairs(3), FakeAsker(_on_call(0, failed)))

        assert _failures(verdicts) == [
            AdjudicationFailure.CLI_FAILED, AdjudicationFailure.CLI_FAILED, None,
        ]
        assert verdicts[0].detail == 'subtype=error_max_turns'

    @pytest.mark.asyncio
    async def test_an_asker_that_raises_fails_the_shard_and_adjudicate_does_not_raise(self):
        def raising(shard: Shard) -> ShardReply:
            raise TimeoutError('cli hung')

        verdicts = await _adjudicate(_pairs(3), FakeAsker(_on_call(0, raising)))

        assert _failures(verdicts)[:2] == [AdjudicationFailure.CLI_FAILED] * 2
        assert 'TimeoutError' in verdicts[0].detail
        assert 'cli hung' in verdicts[0].detail
        assert verdicts[2].verdict is Verdict.EXTENDS


class TestShardFailureStorm:
    @pytest.mark.asyncio
    async def test_a_streak_stops_adjudicating_and_reports_once(self):
        def failing(call: int, shard: Shard) -> ShardReply:
            if call == 0:
                return ShardReply(success=False, detail='down')
            return ShardReply(success=True, structured_output=[])

        storms: list[dict] = []
        ask = FakeAsker(failing)

        verdicts = await _adjudicate(
            _pairs(8), ask, failure_streak=2, on_failure_storm=storms.append,
        )

        assert len(ask.shards) == 2
        assert len(storms) == 1
        (storm,) = storms
        assert storm['count'] == 2
        assert storm['threshold'] == 2
        assert storm['labels'] == ['cli_failed', 'parse_failure']
        assert storm['model'] == 'opus'
        assert storm['shards_total'] == 4
        assert storm['shards_failed'] == 2
        assert _failures(verdicts)[4:] == [AdjudicationFailure.NOT_ATTEMPTED] * 4
        assert all(verdict.verdict is None for verdict in verdicts)

    @pytest.mark.asyncio
    async def test_a_success_between_failures_resets_the_streak(self):
        def alternating(call: int, shard: Shard) -> ShardReply:
            if call % 2 == 0:
                return ShardReply(success=False, detail='flaky')
            return _valid(call, shard)

        storms: list[dict] = []
        ask = FakeAsker(alternating)

        verdicts = await _adjudicate(
            _pairs(8), ask, failure_streak=2, on_failure_storm=storms.append,
        )

        assert storms == []
        assert len(ask.shards) == 4
        assert _failures(verdicts) == [AdjudicationFailure.CLI_FAILED, AdjudicationFailure.CLI_FAILED, None, None] * 2
