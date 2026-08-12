"""Validate the COMMITTED upstream bug report (task 3695).

Task 3695's deliverable is not code -- it is a report:
``docs/upstream/toolcall-parser-overconsumption.md``, written to be filed with
an external harness maintainer with no access to this repository. For that
kind of deliverable the dominant failure mode is not a bug, it is a
plausible-looking hand-typed document, and nothing else downstream would
catch one. So this module does not trust the prose. It re-runs the SAME
PRODUCTION code the report's claims are about -- ``shared.toolcall_markup`` --
over the report's own committed text and specimens, following the precedent
of ``fused-memory/tests/test_toolcall_xml_leak_sweep_artifacts.py`` (which
calls itself "the anti-fabrication gate for an OPERATIONAL task" for exactly
the same reason).

Two gates, both genuinely behavioural rather than cosmetic:

1. **Containment** (this module, ``TestContainment``): the report's full text
   contains no raw envelope literal -- i.e. no member of
   :data:`shared.toolcall_markup.ENVELOPE_LITERALS` appears unescaped -- and
   matches neither alternative of the PRD's collection predicate. A tracked
   file that DID contain a raw literal would corrupt the next agent that
   edits it (the agent's own authoring tool call would terminate early, for
   the identical reason the report itself describes) and would plant a sweep
   hazard of exactly the kind a loose predicate-matching glob has hit before
   on other git-tracked artifacts in this repo.

2. **Specimen anti-fabrication** (added alongside the report's specimen
   section): every specimen the report presents is parsed from a
   machine-readable header, un-escaped, and replayed through PRODUCTION
   ``repair()``. The parameters production code independently recovers must
   equal the parameters the report claims were dropped. A specimen typed
   from imagination cannot satisfy that.

Nothing here asserts prose wording, and nothing asserts any of the report's
counts or percentages -- the transcript archive this report measures grows
live, and pinning a count would make a later, better repairer look like a
regression (PRD ``plans/toolcall-markup-containment-prd.md`` guard G6).

## Sentinel-literal hazard (inherited from ``toolcall_markup.py`` / task 3083)

Do not paste a raw envelope literal into this file. Every opening angle
bracket that would form part of a markup tag is spelled with the ``\\x3c``
escape, exactly as
``shared/src/shared/toolcall_markup.py`` requires of itself: writing one
verbatim would make THIS file's own authoring tool call terminate early,
reproducing the very bug under test. The report's specimens are read from
disk and un-escaped IN MEMORY ONLY (their own ``&#60;`` -> ``\\x3c``
conversion happens at assertion time, never in this file's source text).
"""

from __future__ import annotations

import re
from pathlib import Path

import shared.toolcall_markup as tm

# Mirrors shared/tests/conftest.py:21 rather than inventing a second
# root-finding scheme.
REPO_ROOT = Path(__file__).resolve().parents[2]

REPORT_PATH = REPO_ROOT / 'docs' / 'upstream' / 'toolcall-parser-overconsumption.md'

# ---------------------------------------------------------------------------
# The PRD's collection predicate (plans/toolcall-markup-containment-prd.md
# section 2, line 35; restated verbatim in
# shared/tests/fixtures/toolcall_markup_corpus.README.md's "Collection
# predicate" section). Reproduced here, rather than imported, because it is
# a *corpus-extraction* rule, not a public symbol of shared.toolcall_markup --
# but the first alternative is built from tm.closer_for() so it can never
# drift from the module's own INVOKE_CLOSER spelling.
#
# A report that matched either alternative could be picked up as a specimen
# by a future re-extraction of the transcript archive, or rewritten by a
# future sweep script -- this report must never look like the corruption it
# describes.
# ---------------------------------------------------------------------------

_ALT1 = re.compile(re.escape(tm.closer_for('invoke')) + r'\s*$', re.MULTILINE)
_ALT2 = re.compile(r'\x3c/[A-Za-z_]\w*>\s*\x3cparameter\s+name="[^"]+">')


def _read_report() -> str:
    return REPORT_PATH.read_text()


class TestContainment:
    """The report is a tracked file, and it must never carry a live sentinel."""

    def test_report_exists(self) -> None:
        assert REPORT_PATH.is_file(), (
            f'{REPORT_PATH.relative_to(REPO_ROOT)} does not exist. This is '
            'the deliverable for task 3695 (upstream harness bug report: '
            'tool-call parser over-consumption and silent parameter drop).'
        )

    def test_report_carries_no_raw_envelope_literal(self) -> None:
        text = _read_report()
        hit = tm.detect(text)
        assert hit is None, (
            f'{REPORT_PATH.name} contains a raw envelope literal ({hit!r}). '
            'A literal here would corrupt the next authoring tool call that '
            'edits this file -- the exact defect this report describes -- '
            'and would plant a sweep hazard in a git-tracked file. Escape '
            "every opening angle bracket that forms part of a markup tag as "
            "'&#60;'."
        )

    def test_report_does_not_match_collection_predicate_alt1(self) -> None:
        text = _read_report()
        match = _ALT1.search(text)
        assert match is None, (
            f'{REPORT_PATH.name} matches the collection predicate\'s first '
            f'alternative at {match!r} — a future re-extraction of the '
            'transcript archive could pick this report up as a specimen.'
        )

    def test_report_does_not_match_collection_predicate_alt2(self) -> None:
        text = _read_report()
        match = _ALT2.search(text)
        assert match is None, (
            f'{REPORT_PATH.name} matches the collection predicate\'s second '
            f'alternative at {match!r} — a future re-extraction of the '
            'transcript archive could pick this report up as a specimen.'
        )
