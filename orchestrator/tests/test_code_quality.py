"""Tests of ``orchestrator/src/orchestrator/agents/code_quality.py``, which renders
the role prompts' code-quality block from the packaged normative doc.

The synthetic half proves ``section()`` and ``render()`` can FAIL: a missing,
empty or brace-carrying rendered section raises, naming the section.
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

_TITLE = '## Code quality — judge against this definition'
_DEFINITION = '## Definition'
_HEURISTICS = '## The fourteen heuristics'
_STANCES = '## Two stances'
_DO_NOT_STEER = '## Do not steer by'

#: The rendered sections whose body is a ``- **Label.**`` bullet list.
_BULLET_SECTION_HEADINGS = (_STANCES, _DO_NOT_STEER)
#: The doc sections rendered verbatim under their own heading, in order.
_VERBATIM_SECTION_HEADINGS = (_HEURISTICS, *_BULLET_SECTION_HEADINGS)
#: The doc sections render() carries, in the order it carries them.
_RENDERED_SOURCE_HEADINGS = (_DEFINITION, *_VERBATIM_SECTION_HEADINGS)
#: The rendered block's headings: the title stands in for ``## Definition``.
_EXPECTED_RENDERED_HEADINGS = (_TITLE, *_VERBATIM_SECTION_HEADINGS)


class TestSection:
    """``section()`` — the slicer behind ``render()``."""

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


# ---------------------------------------------------------------------------
# render() over a synthetic doc with the real doc's section layout. Each body
# is a distinctive sentence so presence and absence are unambiguous.
# ---------------------------------------------------------------------------

_PREAMBLE = '# Code quality\n\nPreamble sentence that is never rendered.\n'

_SECTIONS = {
    _DEFINITION: 'Definition sentence of the synthetic doc.',
    _HEURISTICS: (
        'Heuristics intro sentence.\n\n'
        '1. **Informative names.** A name says what the thing is.\n'
        '2. **Simple control flows.** Few decision points per unit.'
    ),
    _STANCES: (
        '- **Comments.** Stance sentence about comments.\n'
        '- **Tests.** Stance sentence about tests.'
    ),
    '## What to measure': 'Measurement sentence that is never rendered.',
    _DO_NOT_STEER: '- **Raw line count.** Do-not-steer sentence.',
    '## Relationship to the design invariants': 'Relationship sentence that is never rendered.',
    '## Reach': 'Reach sentence that is never rendered.',
}

_EXCLUDED_HEADINGS = tuple(h for h in _SECTIONS if h not in _RENDERED_SOURCE_HEADINGS)


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
        assert tuple(re.findall(r'^## .*$', render(_doc(_SECTIONS)), re.MULTILINE)) == (
            _EXPECTED_RENDERED_HEADINGS
        )

    def test_the_definition_body_sits_verbatim_under_the_title(self):
        rendered = render(_doc(_SECTIONS))
        assert section(rendered, _TITLE).strip() == _SECTIONS[_DEFINITION]

    @pytest.mark.parametrize('heading', _VERBATIM_SECTION_HEADINGS)
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
            _with(_STANCES, _SECTIONS[_STANCES].replace(
                'Stance sentence about comments.', edited,
            ))
        )
        assert edited in rendered

    @pytest.mark.parametrize('heading', _RENDERED_SOURCE_HEADINGS)
    def test_a_missing_rendered_section_raises_naming_it(self, heading):
        with pytest.raises(ValueError, match=re.escape(heading)):
            render(_without(heading))

    def test_a_whitespace_only_rendered_section_raises_naming_it(self):
        with pytest.raises(ValueError, match=re.escape(_STANCES)):
            render(_with(_STANCES, '   \n\t'))

    @pytest.mark.parametrize('brace', ['{', '}'])
    def test_a_brace_in_a_rendered_section_raises_naming_it(self, brace):
        with pytest.raises(ValueError, match=re.escape(_DO_NOT_STEER)):
            render(_with(_DO_NOT_STEER, f'- **Raw line count.** A literal {brace} here.'))

    @pytest.mark.parametrize('heading', _EXCLUDED_HEADINGS)
    def test_a_brace_in_an_excluded_section_does_not_raise(self, heading):
        render(_with(heading, 'A literal { brace in a section that is never rendered.'))


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
    """The real rendered block's STRUCTURE: headings, heuristic numbering and list shape.

    No doc prose and no item label is asserted; section bodies and labels are free to change.
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
        assert tuple(re.findall(r'^## .*$', block, re.MULTILINE)) == _EXPECTED_RENDERED_HEADINGS

    def test_numbers_the_heuristics_one_to_fourteen(self, block):
        numbers = re.findall(r'^(\d+)\. \*\*', section(block, _HEURISTICS), re.MULTILINE)
        assert numbers == [str(n) for n in range(1, 15)]

    @pytest.mark.parametrize('heading', _BULLET_SECTION_HEADINGS)
    def test_each_bullet_section_is_a_bold_labelled_list(self, block, heading):
        assert re.search(r'^- \*\*', section(block, heading), re.MULTILINE), (
            f"{heading!r} carries no '- **Label.**' bullet at column 0: the "
            "section must stay a bold-labelled bullet list."
        )

    def test_is_brace_free(self, block):
        assert_brace_free('guidance()', block, remedy=_FORMAT_REMEDY)

    @pytest.mark.parametrize('forbidden', _FORBIDDEN_IN_ANY_PROMPT_BLOCK)
    def test_cannot_break_the_all_roles_scanners(self, block, forbidden):
        assert forbidden not in block
