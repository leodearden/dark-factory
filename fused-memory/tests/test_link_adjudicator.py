"""The link adjudicator (plans/write-triage-link-healing-prd.md H2, task 6184).

The mirror tests read the LIVE rater brief: the scale's one home in code is
``link_heal.Verdict``, and the brief is the document raters and the model read,
so the two must agree word for word (INV-10).
"""

from __future__ import annotations

import re

from fused_memory.maintenance.link_adjudicator import (
    RATER_BRIEF_PATH,
    TRUNCATION_MARKER,
    VERDICT_OUTPUT_SCHEMA,
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
        assert AGREEING_VERDICTS == BELONGS_VERDICTS - {Verdict.CORRECTS}

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
