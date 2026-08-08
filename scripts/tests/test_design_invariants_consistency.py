"""Drift guard over the design-invariant family's non-auto-extending sites.

Task 3802. ``docs/legibility/design-invariants.md`` is the single normative
copy of dark-factory's invariant family (INV-5, ``no-lockstep-duplication``,
applied to the invariants themselves). Several other committed artifacts
nonetheless restate parts of that family, and none of them auto-extend when an
invariant is added. MEASURED, not predicted: the G7 trigger-shape list in
``skills/prd/references/gates.md`` drifted TWICE (INV-6/INV-7 landed 2026-08-02
without it; the 2026-08-06 addendum added only INV-8's shape — see task 3811),
and ``CONTRIBUTING.md`` §6 restated all eight slugs twelve lines above its own
rule forbidding exactly that.

This module derives the family ONCE from the normative doc's headings and
cross-checks every other site against it. It never stores its own snapshot slug
list: a hardcoded constant here would be one more lock-step copy, stale on the
next invariant exactly like the prose sites were.

THE FOUR PINNED SITES (see ``PINNED_SITES`` for the machine-readable registry):
  * ``docs/legibility/design-invariants.md`` — SOURCE OF TRUTH. Its
    ``## INV-N `slug``` headings define the family; its live family-size claims
    are pinned inside marked spans.
  * ``docs/legibility/design-invariants-fixtures.md`` — one fixture section per
    invariant, plus a rehearsal verdict table pinned for COVERAGE (never for its
    rationale prose, which the doc itself declares a point-in-time snapshot).
  * ``skills/prd/references/gates.md`` — TWO independent enumerations: the
    family-inventory row (ordered) and the G7 trigger-shape fallback list (set).
  * ``CONTRIBUTING.md`` — pinned as an ABSENCE: it must restate NO slug at all.

LIVE FAMILY-SIZE CLAIMS ARE PINNED ONLY INSIDE HTML-COMMENT MARKED SPANS. This
is load-bearing, not stylistic: both docs also carry HISTORICAL range prose that
is correct as written and must stay unpinned — design-invariants.md's "INV-1..INV-5
encode the agent-legibility survey's cross-cutting root causes" (true: that is the
founding subset's provenance) and the fixtures doc's "INV-1..5 fixtures were seeded
2026-07-14; INV-6..7 ... 2026-08-02". A blanket "every INV-1..N range must equal
the family size" rule would land RED against factually-correct prose and could
only be greened by falsifying history. Markers make the live-vs-historical
distinction explicit in the doc and mechanically checkable, following the repo's
existing convention at CONTRIBUTING.md's ``lint-command-mirror`` block.

PLACEMENT IS LOAD-BEARING. ``scripts/tests/`` modules must import NO first-party
package — that is what lets ``uv run --project shared pytest scripts/tests/``
(``scripts/orchestrator.yaml``'s ``test_command``) satisfy them on a freshly
synced verify worktree. This module is stdlib-only (``re``, ``pathlib``) plus
``pytest``.

EXTRACTOR CONTRACT. Every extractor below raises a loud ``AssertionError``
naming its ``source`` rather than returning an empty list. An extractor that
silently yields nothing turns every downstream drift assertion green while
pinning nothing at all — strictly worse than no guard, because the check still
reports success. Extractors are unit-tested against HAND-WRITTEN fixture
markdown, never the live docs, so those tests stay stable under any future doc
edit; the live assertions re-read every committed artifact fresh.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

NORMATIVE_DOC = REPO_ROOT / "docs" / "legibility" / "design-invariants.md"
FIXTURES_DOC = REPO_ROOT / "docs" / "legibility" / "design-invariants-fixtures.md"
GATES_DOC = REPO_ROOT / "skills" / "prd" / "references" / "gates.md"
CONTRIBUTING_DOC = REPO_ROOT / "CONTRIBUTING.md"


# ---------------------------------------------------------------------------
# parse_invariant_headings — the family extractor, fixture-driven tests
# ---------------------------------------------------------------------------

# (a) Happy path. Three well-formed headings in document order.
_HEADINGS_HAPPY = """\
# Design invariants

Intro prose.

## INV-1 `a-slug`

**Rule**: something.

## INV-2 `b-slug`

**Rule**: something else.

## INV-3 `c-slug`
"""

# (b) Decoy immunity, every decoy shape the real doc actually carries:
# a `### INV-N `slug`` SUB-heading (the fixtures doc uses those for shapes), an
# inline prose mention of a number outside the family, a backticked slug inside
# a body paragraph (design-invariants.md's own §Census seam does this), and a
# `## INV-N` heading carrying no backticked slug at all.
_HEADINGS_DECOY = """\
# Design invariants

A gate checklist. Numeric aliases are prose convenience; INV-9 is not a thing.

## INV-1 `a-slug`

### INV-4 `sub-shape`

Body prose naming `b-slug` and `c-slug` mid-sentence, which is a reference,
not an enumeration.

## INV-2 `b-slug`

## INV-5

## INV-3 `c-slug`
"""

# (c) THE VACUITY HAZARD: a doc that no longer parses. Structurally plausible —
# it still talks about invariants — but carries zero matching headings, e.g.
# after a heading-level or backtick-style edit.
_HEADINGS_NONE = """\
# Design invariants

INV-1 contracts-machine-checked
INV-2 structured-facts-at-failure

### INV-1 `a-slug`
"""

# (d) Non-contiguous numbering — an invariant deleted without renumbering.
_HEADINGS_GAP = """\
## INV-1 `a-slug`

## INV-2 `b-slug`

## INV-4 `d-slug`
"""

# (e) Numbering that does not start at 1 — a doc split, or a truncated read.
_HEADINGS_OFFSET = """\
## INV-2 `b-slug`

## INV-3 `c-slug`
"""

# (f) Duplicate slug across two headings. Slugs are stable IDS referenced by G7
# waivers and `/review`'s invariant_findings; two headings sharing one makes
# every downstream by-slug lookup ambiguous.
_HEADINGS_DUP_SLUG = """\
## INV-1 `a-slug`

## INV-2 `a-slug`

## INV-3 `c-slug`
"""

# (g) Duplicate number — a bad merge that landed two INV-2 sections.
_HEADINGS_DUP_NUMBER = """\
## INV-1 `a-slug`

## INV-2 `b-slug`

## INV-2 `c-slug`
"""

_FIXTURE_SOURCE = "a hand-written fixture"


def test_parse_invariant_headings_returns_ordered_number_slug_pairs() -> None:
    """(a) Headings parse to ``(number, slug)`` pairs in document order."""
    assert parse_invariant_headings(_HEADINGS_HAPPY, source=_FIXTURE_SOURCE) == [
        (1, "a-slug"),
        (2, "b-slug"),
        (3, "c-slug"),
    ]


def test_parse_invariant_headings_ignores_every_decoy_shape() -> None:
    """(b) Sub-headings, prose mentions and slug-less headings are not family members.

    The decoys are measured, not hypothetical: the fixtures doc uses ``###``
    sub-headings per fixture shape, and design-invariants.md's Census seam
    references slugs in running prose. A heuristic keyed on "a backticked slug
    near an INV- token" would swallow all of them.
    """
    assert parse_invariant_headings(_HEADINGS_DECOY, source=_FIXTURE_SOURCE) == [
        (1, "a-slug"),
        (2, "b-slug"),
        (3, "c-slug"),
    ]


@pytest.mark.parametrize(
    ("markdown_text", "case", "expected_phrase"),
    [
        pytest.param(
            _HEADINGS_NONE, "zero headings", "no `## INV-N `slug`` headings", id="zero-headings"
        ),
        pytest.param(_HEADINGS_GAP, "non-contiguous", "contiguous", id="numbering-gap"),
        pytest.param(_HEADINGS_OFFSET, "does not start at 1", "contiguous", id="offset-start"),
        pytest.param(_HEADINGS_DUP_SLUG, "duplicate slug", "duplicate slug", id="duplicate-slug"),
        pytest.param(
            _HEADINGS_DUP_NUMBER, "duplicate number", "duplicate number", id="duplicate-number"
        ),
    ],
)
def test_parse_invariant_headings_fails_loudly(
    markdown_text: str, case: str, expected_phrase: str
) -> None:
    """(c-g) Every malformed family RAISES and names its ``source``.

    Zero headings is the vacuity hazard this whole module is built around: an
    extractor that returned ``[]`` would turn every downstream drift assertion
    into an empty-vs-empty comparison that passes while pinning nothing —
    strictly worse than no guard, because the suite still reports success. The
    remaining cases are the same failure one level down: a family with a gap, an
    offset start, or a duplicate id silently mis-describes what the other sites
    are being compared against.

    Naming ``source`` in the message is what makes a red run actionable: four
    different docs are parsed by this one extractor, so "which doc broke" is not
    recoverable from the traceback alone.
    """
    with pytest.raises(AssertionError) as excinfo:
        parse_invariant_headings(markdown_text, source=_FIXTURE_SOURCE)

    message = str(excinfo.value)
    assert _FIXTURE_SOURCE in message, f"{case}: message must name the source doc: {message!r}"
    assert expected_phrase in message, f"{case}: message must diagnose the defect: {message!r}"
