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
from typing import NamedTuple

import pytest

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


# ---------------------------------------------------------------------------
# Specimen anti-fabrication gate
#
# Each specimen in the report carries a one-line, machine-readable HTML
# comment immediately above its fenced block: id / tool / param / supplied /
# schema / dropped. HTML comments are invisible in rendered markdown, so an
# upstream reader sees a clean document while this module gets an
# unambiguous parse target -- deliberately not a sidecar data file (which
# would break self-containment: the .md gets filed upstream on its own and
# a sidecar would stay behind) and not a parse of the visible prose table
# (which would drift on harmless formatting edits and would smuggle in the
# prose-pinning rule 5 forbids).
# ---------------------------------------------------------------------------

# Positional groups, not Python's named-group syntax -- deliberately,
# mirroring toolcall_markup.py's own _CLOSER_RE / _CANONICAL_OPENER_RE /
# _ECHO_OPENER_RE, none of which use named groups either. Spelling a named
# group out requires an opening angle bracket immediately followed by an
# identifier and, eventually, a closing one -- structurally the exact
# open-tag shape this module exists to keep out of its own source, even
# though it is regex syntax rather than an envelope literal.
#
# The comment delimiter is matched as the literal text '&#60;!--', NOT as an
# escaped-then-decoded real HTML comment opener: the report spells its own
# structural markup with the same '&#60;' convention as everything else in
# it (design decision -- see plan.json), so that is what is actually on
# disk. Matching a real, decoded comment opener here would never find it,
# and re-escaping this text back to search for one would recreate the exact
# literal this module exists to keep out of its own source for no benefit.
#
# Group order: id, tool, param, supplied, schema, dropped, block.
_SPECIMEN_HEADER_RE = re.compile(
    r'&#60;!--\s*specimen\s+'
    r'id="([^"]*)"\s+'
    r'tool="([^"]*)"\s+'
    r'param="([^"]*)"\s+'
    r'supplied="([^"]*)"\s+'
    r'schema="([^"]*)"\s+'
    r'dropped="([^"]*)"\s*'
    r'--&gt;\s*'
    r'```text\s*\n'
    r'(.*?)\n'
    r'```',
    re.DOTALL,
)


class _Specimen(NamedTuple):
    id: str
    tool: str
    param: str
    supplied: list[str]
    schema: list[str]
    dropped: frozenset[str]
    #: The fenced block's content, un-escaped ('&#60;' -> a real opening
    #: angle bracket) IN MEMORY ONLY. Never written back to any file.
    value: str


def _parse_list(raw: str) -> list[str]:
    return [item for item in raw.split(',') if item]


def _parse_specimens(text: str) -> list[_Specimen]:
    """Every (header, fenced block) pair in *text*, un-escaped. Pure."""
    specimens = []
    for match in _SPECIMEN_HEADER_RE.finditer(text):
        (
            specimen_id, tool, param, supplied_raw, schema_raw, dropped_raw, block,
        ) = match.groups()
        specimens.append(
            _Specimen(
                id=specimen_id,
                tool=tool,
                param=param,
                supplied=_parse_list(supplied_raw),
                schema=_parse_list(schema_raw),
                dropped=frozenset(_parse_list(dropped_raw)),
                value=block.replace('&#60;', '\x3c'),
            )
        )
    return specimens


def _report_specimens() -> list[_Specimen]:
    """Specimens parsed fresh from the committed report, or [] if absent.

    Never raises on a missing report: TestContainment.test_report_exists is
    the assertion that owns that failure mode. A collection-time exception
    here would instead abort collecting this WHOLE module, hiding every
    other test's result behind an import error.
    """
    if not REPORT_PATH.is_file():
        return []
    return _parse_specimens(_read_report())


# Parsed once at collection time, mirroring
# test_toolcall_xml_leak_sweep_artifacts.py's _REPORT_PATHS: a later pytest
# invocation re-imports this module and re-reads the (by-then-edited) file,
# so this reflects whatever is committed at the moment a run starts.
_SPECIMENS = _report_specimens()
_SPECIMEN_IDS = [s.id for s in _SPECIMENS]


class TestSpecimenAntiFabrication:
    """Every specimen the report presents must be a REAL corrupted call.

    Un-escapes each specimen's fenced block back to real text and replays it
    through PRODUCTION ``shared.toolcall_markup.repair()`` -- the exact call
    shape the report's own claims describe. A specimen typed from
    imagination, with a plausible-looking but wrong ``dropped`` list, cannot
    satisfy this: ``repair()`` independently re-derives what it would
    recover, and the report's claim must equal that exactly. This is the
    same anti-fabrication shape as
    ``fused-memory/tests/test_toolcall_xml_leak_sweep_artifacts.py``, applied
    to a hand-authored report instead of a captured sweep.
    """

    def test_specimens_were_parsed(self) -> None:
        # Deliberately NOT parametrized: a parametrize list built from zero
        # parsed specimens would collect zero test cases and report
        # vacuously green -- exactly the failure class
        # test_toolcall_xml_leak_sweep_artifacts.py's TestArtifactsExist
        # exists to rule out. Asserts on the MODULE-LEVEL _SPECIMENS list
        # itself, not a fresh _report_specimens() call: _SPECIMENS is the
        # exact list the two parametrize decorators below are built from,
        # so it is the thing whose emptiness would actually cause the
        # vacuously-green collection this guard exists to rule out.
        assert _SPECIMENS, (
            'no machine-readable specimen headers parsed from '
            f'{REPORT_PATH.name} -- the parametrized specimen tests below '
            'would collect zero cases and report vacuously green.'
        )

    def test_specimen_ids_are_unique(self) -> None:
        specimens = _report_specimens()
        ids = [s.id for s in specimens]
        assert len(set(ids)) == len(ids), f'duplicate specimen id in {ids}'

    @pytest.mark.parametrize('specimen', _SPECIMENS, ids=_SPECIMEN_IDS)
    def test_specimen_is_detected_as_corrupted(self, specimen: _Specimen) -> None:
        assert tm.detect(specimen.value) is not None, (
            f'{specimen.id}: un-escaped value carries no envelope literal '
            'that shared.toolcall_markup.detect() recognizes -- this is not '
            'actually a corrupted specimen.'
        )

    @pytest.mark.parametrize('specimen', _SPECIMENS, ids=_SPECIMEN_IDS)
    def test_production_repair_reproduces_the_claimed_drop(
        self, specimen: _Specimen
    ) -> None:
        result = tm.repair(
            specimen.value, specimen.param, specimen.schema, specimen.supplied
        )
        assert result is not None, (
            f'{specimen.id}: shared.toolcall_markup.repair() refused to '
            'repair this specimen -- the report claims a repairable '
            'specimen that production code does not independently agree is '
            'repairable.'
        )
        assert set(result.recovered) == specimen.dropped, (
            f'{specimen.id}: repair() recovered {sorted(result.recovered)}, '
            f'but the report claims dropped={sorted(specimen.dropped)}. The '
            "report's claim must equal what production code independently "
            'recovers.'
        )
        assert tm.detect(result.clean_value) is None, (
            f'{specimen.id}: repair() clean_value still trips detect()'
        )
        assert specimen.value.startswith(result.clean_value), (
            f'{specimen.id}: clean_value is not a prefix of the specimen '
            'value (invariant D5)'
        )
        for name, recovered_value in result.recovered.items():
            assert recovered_value in specimen.value, (
                f'{specimen.id}: recovered value for {name!r} is not a '
                'verbatim substring of the specimen value (invariant D5)'
            )
