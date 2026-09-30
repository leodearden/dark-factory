"""Tests of ``orchestrator/src/orchestrator/agents/code_quality.py``, which renders
the role prompts' code-quality block from the packaged normative doc.

The synthetic half proves ``section()`` and ``render()`` can FAIL: a missing,
empty or brace-carrying rendered section raises, naming the section, and the
parsers below raise on a hole in a numbered list.
"""

from __future__ import annotations

import importlib.resources
import os
import re
from pathlib import Path

import pytest
from _role_splice_contract import MARKDOWN_HEADING, assert_brace_free, assert_nonempty

from orchestrator.agents.code_quality import NORMATIVE_DOC, guidance, render, section

# Resolved from THIS FILE, never from the process CWD, so it holds from
# orchestrator/, the repo root and a ``.worktrees/<id>`` checkout alike.
REPO_ROOT = Path(__file__).resolve().parents[2]

_ANCHOR = '## The fourteen heuristics'
_TITLE = '## Code quality — judge against this definition'

#: A headline is ``<n>. **<text>**`` at COLUMN 0, so an indented continuation
#: line's own bold run (the real doc's heuristic 14 carries ``**No cheating**``)
#: is never read as a further headline.
_HEADLINE_RE = re.compile(r'^(\d{1,2})\. \*\*(.+?)\*\*', re.MULTILINE)

#: A bullet label is ``- **<text>**`` at COLUMN 0, for the same reason: a
#: bullet's continuation lines carry their own inline bold and are not items.
_BULLET_LABEL_RE = re.compile(r'^- \*\*(.+?)\*\*', re.MULTILINE)


def numbered_headlines(text: str, heading: str) -> list[str]:
    """Return the bold headlines of the numbered list under *heading*, in order.

    Raises :class:`ValueError` naming the observed numbers when they are not
    exactly ``1..n``: an item dropped from the middle of the list leaves a hole,
    and a hole is an error rather than a shorter answer.
    """
    matches = _HEADLINE_RE.findall(section(text, heading))
    numbers = [int(number) for number, _ in matches]
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError(
            f'the numbered list under {heading!r} is not numbered 1..n — '
            f'observed {numbers}. An item was dropped, renumbered, or lost its '
            'bold headline.'
        )
    return [headline for _, headline in matches]


def bold_item_labels(text: str, heading: str) -> list[str]:
    """Return the bold LABELS of the bullet list under *heading*, in order."""
    return _BULLET_LABEL_RE.findall(section(text, heading))


# ---------------------------------------------------------------------------
# Parser synthetics. A numbered bold item BEFORE the anchor and one AFTER the
# section's terminating ``## `` heading must both be excluded.
# ---------------------------------------------------------------------------

_WELL_FORMED = (
    '# Code quality\n\n'
    '## Definition\n\n'
    '1. **Not a heuristic.** A numbered bold item in an EARLIER section.\n\n'
    f'{_ANCHOR}\n\n'
    'Prose between the heading and the list, which this parser never reads.\n\n'
    '1. **Informative names.** A name says what the thing is and does.\n'
    '2. **Simple control flows.** Few decision points per unit.\n'
    '3. **SPOT — single point of truth.** Each fact lives in one place.\n\n'
    '## Two stances\n\n'
    '4. **Not a heuristic either.** A numbered bold item in a LATER section.\n'
)

_WITH_CONTINUATION = (
    f'{_ANCHOR}\n\n'
    '1. **Informative names.** A name says what the thing is and does.\n'
    '   **No cheating**: an indented continuation line has its own bold run.\n'
    '2. **Simple control flows.** Few decision points per unit.\n\n'
    '## Two stances\n'
)

_WITH_UNBOLDED_TAIL = (
    f'{_ANCHOR}\n\n'
    '1. **Informative names.** A name says what the thing is and does.\n'
    '2. A numbered line carrying no bold headline at all.\n\n'
    '## Two stances\n'
)

_WITH_UNBOLDED_MIDDLE = (
    f'{_ANCHOR}\n\n'
    '1. **Informative names.** A name says what the thing is and does.\n'
    '2. A numbered line carrying no bold headline at all.\n'
    '3. **Simple control flows.** Few decision points per unit.\n\n'
    '## Two stances\n'
)

_NON_CONTIGUOUS = (
    f'{_ANCHOR}\n\n'
    '1. **Informative names.** A name says what the thing is and does.\n'
    '2. **Simple control flows.** Few decision points per unit.\n'
    '4. **Small function scopes.** A function does one thing.\n\n'
    '## Two stances\n'
)

# The anchor DEMOTED below ``## ``, with the section that follows carrying
# numbered bold items of its own. A substring search would match the demoted
# heading, and the ``^## `` terminator would not close it.
_DEMOTED_ANCHOR = (
    f'#{_ANCHOR}\n\n'
    '1. **Informative names.** A name says what the thing is and does.\n\n'
    '## A later section\n\n'
    '2. **Not a heuristic.** A numbered bold item the slice must never reach.\n'
)

# Continuation lines sit at column 0 carrying their own inline bold.
_BULLET_SECTION = (
    '## Two stances\n\n'
    '- **Comments.** Aim for code that is clear with no or low comments, since\n'
    'a continuation line carries its own **inline bold** and is not an item.\n'
    "- **Tests.** Test access to a module's internals is an interface smell.\n\n"
    '## Do not steer by\n\n'
    '- **Raw line count.** A bullet in a LATER section.\n'
)


class TestSection:
    """``section()`` — the one slicer behind ``render()`` and both parsers."""

    _TEXT = (
        '# Title\n\n'
        '## A\n\n'
        'Body of A.\n\n'
        '### A sub-heading inside A\n\n'
        'Still body of A.\n\n'
        '## B\n\n'
        'Body of B, the last section.\n'
    )

    def test_returns_only_the_body_up_to_the_next_level_two_heading(self):
        assert section(self._TEXT, '## A').strip() == (
            'Body of A.\n\n### A sub-heading inside A\n\nStill body of A.'
        )

    def test_the_last_section_runs_to_the_end_of_the_text(self):
        assert section(self._TEXT, '## B').strip() == 'Body of B, the last section.'

    def test_a_missing_heading_raises_naming_it(self):
        with pytest.raises(ValueError, match=re.escape('## Nowhere')):
            section(self._TEXT, '## Nowhere')

    def test_a_demoted_heading_raises_rather_than_matching_as_a_substring(self):
        with pytest.raises(ValueError, match=re.escape('## B')):
            section(self._TEXT.replace('## B', '### B'), '## B')


class TestHeadlineParser:
    """The numbered-headline parser, against synthetic strings only."""

    def test_headlines_returned_in_document_order(self):
        assert numbered_headlines(_WELL_FORMED, _ANCHOR) == [
            'Informative names.',
            'Simple control flows.',
            'SPOT — single point of truth.',
        ]

    def test_indented_continuation_bold_run_is_not_a_headline(self):
        assert numbered_headlines(_WITH_CONTINUATION, _ANCHOR) == [
            'Informative names.',
            'Simple control flows.',
        ]

    def test_numbered_line_without_a_bold_headline_is_not_picked_up(self):
        assert numbered_headlines(_WITH_UNBOLDED_TAIL, _ANCHOR) == ['Informative names.']

    def test_unbolded_line_inside_the_list_is_caught_by_the_contiguity_check(self):
        # Not picking up an unbolded line is only safe because a DROPPED item
        # inside the list shows up as a hole in the numbering.
        with pytest.raises(ValueError, match=r'\[1, 3\]'):
            numbered_headlines(_WITH_UNBOLDED_MIDDLE, _ANCHOR)

    def test_section_stops_at_the_next_heading(self):
        headlines = numbered_headlines(_WELL_FORMED, _ANCHOR)
        assert 'Not a heuristic.' not in headlines
        assert 'Not a heuristic either.' not in headlines

    def test_missing_heading_raises_naming_the_heading_it_looked_for(self):
        text = _WELL_FORMED.replace(_ANCHOR, '## The fifteen heuristics')
        with pytest.raises(ValueError, match=re.escape(_ANCHOR)):
            numbered_headlines(text, _ANCHOR)

    def test_non_contiguous_numbering_raises_naming_the_observed_numbers(self):
        with pytest.raises(ValueError, match=r'\[1, 2, 4\]'):
            numbered_headlines(_NON_CONTIGUOUS, _ANCHOR)

    def test_one_renamed_headline_changes_the_parsed_list(self):
        drifted = _WELL_FORMED.replace(
            '2. **Simple control flows.**', '2. **Straightforward control flows.**',
        )
        assert numbered_headlines(drifted, _ANCHOR) != numbered_headlines(_WELL_FORMED, _ANCHOR)

    def test_a_demoted_heading_raises_rather_than_matching_as_a_substring(self):
        with pytest.raises(ValueError, match=re.escape(_ANCHOR)):
            numbered_headlines(_DEMOTED_ANCHOR, _ANCHOR)


class TestBoldItemLabels:
    """The bullet-label parser, against synthetic strings only."""

    def test_labels_returned_in_document_order(self):
        assert bold_item_labels(_BULLET_SECTION, '## Two stances') == ['Comments.', 'Tests.']

    def test_a_continuation_lines_inline_bold_is_not_a_label(self):
        assert 'inline bold' not in bold_item_labels(_BULLET_SECTION, '## Two stances')

    def test_section_stops_at_the_next_heading(self):
        assert bold_item_labels(_BULLET_SECTION, '## Do not steer by') == ['Raw line count.']

    def test_missing_heading_raises_naming_the_heading_it_looked_for(self):
        with pytest.raises(ValueError, match=re.escape('## Nowhere')):
            bold_item_labels(_BULLET_SECTION, '## Nowhere')

    def test_one_renamed_label_changes_the_parsed_list(self):
        drifted = _BULLET_SECTION.replace('- **Comments.**', '- **On comments.**')
        assert (
            bold_item_labels(drifted, '## Two stances')
            != bold_item_labels(_BULLET_SECTION, '## Two stances')
        )


# ---------------------------------------------------------------------------
# render() over a synthetic doc with the real doc's section layout. Each body
# is a distinctive sentence so presence and absence are unambiguous.
# ---------------------------------------------------------------------------

_PREAMBLE = '# Code quality\n\nPreamble sentence that is never rendered.\n'

_SECTIONS = {
    '## Definition': 'Definition sentence of the synthetic doc.',
    _ANCHOR: (
        'Heuristics intro sentence.\n\n'
        '1. **Informative names.** A name says what the thing is.\n'
        '2. **Simple control flows.** Few decision points per unit.'
    ),
    '## Two stances': (
        '- **Comments.** Stance sentence about comments.\n'
        '- **Tests.** Stance sentence about tests.'
    ),
    '## What to measure': 'Measurement sentence that is never rendered.',
    '## Do not steer by': '- **Raw line count.** Do-not-steer sentence.',
    '## Relationship to the design invariants': 'Relationship sentence that is never rendered.',
    '## Reach': 'Reach sentence that is never rendered.',
}

#: The doc sections render() carries, in the order it carries them.
_RENDERED_SOURCE_HEADINGS = ('## Definition', _ANCHOR, '## Two stances', '## Do not steer by')
_EXCLUDED_HEADINGS = ('## What to measure', '## Relationship to the design invariants', '## Reach')


def _doc(sections: dict[str, str]) -> str:
    return _PREAMBLE + ''.join(f'\n{heading}\n\n{body}\n' for heading, body in sections.items())


def _with(heading: str, body: str) -> str:
    return _doc({**_SECTIONS, heading: body})


def _without(heading: str) -> str:
    return _doc({h: b for h, b in _SECTIONS.items() if h != heading})


class TestRender:
    """``render(doc_text)`` — the pure seam, over synthetic docs only."""

    def test_opens_its_own_section_and_ends_with_exactly_one_newline(self):
        rendered = render(_doc(_SECTIONS))
        assert rendered.startswith('\n## ')
        assert rendered.endswith('\n')
        assert not rendered.endswith('\n\n')

    def test_carries_exactly_the_rendered_headings_in_order(self):
        assert re.findall(r'^## .*$', render(_doc(_SECTIONS)), re.MULTILINE) == [
            _TITLE,
            _ANCHOR,
            '## Two stances',
            '## Do not steer by',
        ]

    def test_the_definition_body_sits_verbatim_under_the_title(self):
        rendered = render(_doc(_SECTIONS))
        assert section(rendered, _TITLE).strip() == _SECTIONS['## Definition']

    @pytest.mark.parametrize('heading', _RENDERED_SOURCE_HEADINGS[1:])
    def test_each_rendered_section_body_is_verbatim(self, heading):
        assert section(render(_doc(_SECTIONS)), heading).strip() == _SECTIONS[heading]

    @pytest.mark.parametrize('heading', _EXCLUDED_HEADINGS)
    def test_an_excluded_section_body_is_absent(self, heading):
        assert _SECTIONS[heading] not in render(_doc(_SECTIONS))

    def test_the_preamble_is_absent(self):
        assert 'Preamble sentence' not in render(_doc(_SECTIONS))

    def test_editing_a_rendered_sentence_changes_the_render(self):
        edited = 'Stance sentence about comments, edited in the doc alone.'
        rendered = render(
            _with('## Two stances', _SECTIONS['## Two stances'].replace(
                'Stance sentence about comments.', edited,
            ))
        )
        assert edited in rendered

    @pytest.mark.parametrize('heading', _RENDERED_SOURCE_HEADINGS)
    def test_a_missing_rendered_section_raises_naming_it(self, heading):
        with pytest.raises(ValueError, match=re.escape(heading)):
            render(_without(heading))

    def test_a_whitespace_only_rendered_section_raises_naming_it(self):
        with pytest.raises(ValueError, match=re.escape('## Two stances')):
            render(_with('## Two stances', '   \n\t'))

    @pytest.mark.parametrize('brace', ['{', '}'])
    def test_a_brace_in_a_rendered_section_raises_naming_it(self, brace):
        with pytest.raises(ValueError, match=re.escape('## Do not steer by')):
            render(_with('## Do not steer by', f'- **Raw line count.** A literal {brace} here.'))

    def test_a_brace_in_an_excluded_section_does_not_raise(self):
        render(_with('## Reach', 'A literal { brace in a section that is never rendered.'))


class TestNormativeDocLocation:
    """Where the one copy of the doc lives."""

    def test_the_normative_doc_is_package_data(self):
        assert NORMATIVE_DOC.is_file()

    def test_docs_path_is_a_symlink_to_the_packaged_doc(self):
        """The one remaining second site of the doc. The link keeps every
        existing pointer to ``docs/code-quality.md`` resolving, and is relative
        so it resolves in every checkout location.
        """
        docs = REPO_ROOT / 'docs' / 'code-quality.md'
        assert docs.is_symlink()
        assert not os.path.isabs(os.readlink(docs))
        with importlib.resources.as_file(NORMATIVE_DOC) as packaged:
            assert docs.resolve() == packaged.resolve()


#: Substrings that would break one of the existing all-roles prompt scanners if
#: a doc edit introduced them into a rendered section.
_FORBIDDEN_IN_ANY_PROMPT_BLOCK = (
    'mcp__',
    'submit_review_verdict',
    'Output pure JSON',
    'produce a structured JSON review',
    ':!.task',
)

_FORMAT_REMEDY = (
    'Rewrite the doc line without braces: the block reaches '
    "roles.py::_REVIEWER_HEURISTICS_TEMPLATE's str.format() call."
)


@pytest.fixture(scope='module')
def block() -> str:
    return guidance()


class TestRenderedGuidanceShape:
    """The real rendered block's STRUCTURE: headings, numbering and labels.

    No sentence of doc prose is asserted; section bodies are free to change.
    """

    def test_is_nonempty(self, block):
        assert_nonempty(
            'guidance()', block,
            remedy='Restore the rendered sections of the packaged code_quality.md.',
        )

    def test_opens_its_own_section_and_ends_with_exactly_one_newline(self, block):
        assert block.startswith(MARKDOWN_HEADING)
        assert block.endswith('\n')
        assert not block.endswith('\n\n')

    def test_carries_exactly_the_rendered_headings_in_order(self, block):
        assert re.findall(r'^## .*$', block, re.MULTILINE) == [
            _TITLE,
            _ANCHOR,
            '## Two stances',
            '## Do not steer by',
        ]

    def test_carries_fourteen_numbered_heuristics(self, block):
        assert len(numbered_headlines(block, _ANCHOR)) == 14

    def test_carries_the_two_stances(self, block):
        assert bold_item_labels(block, '## Two stances') == ['Comments.', 'Tests.']

    def test_carries_the_four_do_not_steer_by_items(self, block):
        assert bold_item_labels(block, '## Do not steer by') == [
            'Raw line count.',
            'Average complexity.',
            'Line coverage under autouse stubs.',
            'Test count or test-to-code ratio.',
        ]

    def test_is_brace_free(self, block):
        assert_brace_free('guidance()', block, remedy=_FORMAT_REMEDY)

    @pytest.mark.parametrize('forbidden', _FORBIDDEN_IN_ANY_PROMPT_BLOCK)
    def test_cannot_break_the_all_roles_scanners(self, block, forbidden):
        assert forbidden not in block
