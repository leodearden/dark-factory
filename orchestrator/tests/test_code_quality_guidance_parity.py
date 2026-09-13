"""Anti-drift parity guard: the fourteen code-quality heuristic HEADLINES in
``docs/code-quality.md`` vs the copies the reviewer and architect role prompts
now carry inline (task 5225).

A dispatched agent's system prompt cannot follow a cross-reference, so
``orchestrator/src/orchestrator/agents/roles.py`` carries the fourteen
headlines inline. That is a second copy of a list whose single normative home
is ``docs/code-quality.md`` (INV-9 ``one-fact-one-home``; the doc's own
preamble records that a restated copy went stale once already, task 3802).
This module is what keeps the inline copy provably DERIVED rather than a
second source.

WHAT THIS ASSERTS, AND NOTHING ELSE: the 14 bold HEADLINE TOKENS of a numbered
markdown list under the literal heading ``## The fourteen heuristics``, parsed
by ONE parser from BOTH sides and compared as an ordered list. It asserts
nothing whatever about the prose around them — the doc's agreed readings and
the prompt's surrounding instructions can both be rewritten word for word and
this module stays green (``TestHeadlineParser`` demonstrates that executably
rather than by assertion). It is an ordered list EQUALITY, not a substring
pin, so renaming, reordering, adding or dropping a headline on either side
fails loudly and names the divergence.

TASK 5192 — open when this landed — is adjudicating whether prompt-PROSE drift
guards earn their edit friction. This guard is over a STRUCTURED numbered list,
not prose, which is exactly why rewording every sentence on either side is a
no-op here. But if 5192 rules against structured pins too, THIS MODULE is in
scope for that ruling: delete it and leave the prompt block in place.

Built in two halves, in that order — the layout of
``orchestrator/tests/test_cited_test_class_drift.py``: unit tests of the pure
parser against synthetic strings and ``tmp_path`` files FIRST, then the wired
guard against the real doc and the real role prompts. The real tree is green on
arrival, so the synthetic half carries the whole burden of proving the
mechanism can FAIL.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from code_quality_headlines import (
    DOC_PATH,
    HEADLINE_SECTION_HEADING,
    doc_headlines,
    numbered_headlines,
)

# orchestrator/tests/test_code_quality_guidance_parity.py -> parents[0]=tests,
# parents[1]=orchestrator, parents[2]=repo root. Same idiom as
# orchestrator/tests/conftest.py's REPO_ROOT and test_cited_test_class_drift.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]

_ANCHOR = HEADLINE_SECTION_HEADING

# Structurally real synthetics: a numbered bold item BEFORE the anchor and one
# AFTER the section's terminating ``## `` heading, both of which must be
# excluded, around a short well-formed list.
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

# Mirrors the real doc's heuristic 14, whose continuation line carries its own
# ``**No cheating**`` bold run.
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


class TestHeadlineParser:
    """The pure parser, against synthetic strings and ``tmp_path`` files only.

    No test in this class reads the real doc or the real role prompts; that is
    the wired half's job. These are what prove the mechanism can FAIL.
    """

    def test_headlines_returned_in_document_order(self):
        assert numbered_headlines(_WELL_FORMED, _ANCHOR) == [
            'Informative names.',
            'Simple control flows.',
            'SPOT — single point of truth.',
        ]

    def test_indented_continuation_bold_run_is_not_a_headline(self):
        # The match is anchored at column 0, so heuristic 14's inline
        # ``**No cheating**`` continuation cannot masquerade as a 15th item.
        assert numbered_headlines(_WITH_CONTINUATION, _ANCHOR) == [
            'Informative names.',
            'Simple control flows.',
        ]

    def test_numbered_line_without_a_bold_headline_is_not_picked_up(self):
        assert numbered_headlines(_WITH_UNBOLDED_TAIL, _ANCHOR) == ['Informative names.']

    def test_unbolded_line_inside_the_list_is_caught_by_the_contiguity_check(self):
        # Not picking up an unbolded line is only safe because a DROPPED item
        # inside the list shows up as a hole in the numbering. This pins the
        # interaction between those two rules so neither can be relaxed alone.
        with pytest.raises(ValueError, match=r'\[1, 3\]'):
            numbered_headlines(_WITH_UNBOLDED_MIDDLE, _ANCHOR)

    def test_section_stops_at_the_next_heading(self):
        headlines = numbered_headlines(_WELL_FORMED, _ANCHOR)
        assert 'Not a heuristic.' not in headlines  # before the anchor
        assert 'Not a heuristic either.' not in headlines  # after the terminator

    def test_missing_heading_raises_naming_the_heading_it_looked_for(self):
        # Never ``[]`` and never a skip: a renamed heading is exactly the drift
        # this guard exists to catch (INV-2 structured-facts-at-failure).
        text = _WELL_FORMED.replace(_ANCHOR, '## The fifteen heuristics')
        with pytest.raises(ValueError, match=re.escape(_ANCHOR)):
            numbered_headlines(text, _ANCHOR)

    def test_non_contiguous_numbering_raises_naming_the_observed_numbers(self):
        with pytest.raises(ValueError, match=r'\[1, 2, 4\]'):
            numbered_headlines(_NON_CONTIGUOUS, _ANCHOR)

    def test_one_renamed_headline_makes_the_two_lists_unequal(self):
        # THE CAN-FAIL PROOF. The real tree is green on arrival, so without
        # this the wired half would pass identically if the parser silently
        # returned the same wrong answer (e.g. ``[]``) on both sides.
        drifted = _WELL_FORMED.replace(
            '2. **Simple control flows.**', '2. **Straightforward control flows.**',
        )
        assert numbered_headlines(drifted, _ANCHOR) != numbered_headlines(_WELL_FORMED, _ANCHOR)


class TestDocHeadlines:
    """``doc_headlines`` — the file-reading wrapper around the pure parser."""

    def test_doc_path_is_the_repo_root_code_quality_md(self):
        # Resolved from ``__file__``, so this holds inside a ``.worktrees/<id>``
        # checkout, which is where this test itself runs.
        assert DOC_PATH == _REPO_ROOT / 'docs' / 'code-quality.md'
        assert DOC_PATH.is_file()

    def test_missing_doc_raises_loudly_rather_than_skipping(self, tmp_path):
        # tmp_path holds no code-quality.md at all. A caller with no doc to
        # check against has no basis for asserting parity, and treating that as
        # "nothing to check" would disable the guard exactly when its
        # precondition breaks (exemplar:
        # shared/tests/test_architecture_doc_transition_parity.py::TestDocPath).
        with pytest.raises(OSError):
            doc_headlines(tmp_path / 'code-quality.md')

    def test_reads_a_tmp_path_doc_through_the_same_parser(self, tmp_path):
        doc = tmp_path / 'code-quality.md'
        doc.write_text(_WELL_FORMED, encoding='utf-8')
        assert doc_headlines(doc) == numbered_headlines(_WELL_FORMED, _ANCHOR)

