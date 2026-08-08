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

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

NORMATIVE_DOC = REPO_ROOT / "docs" / "legibility" / "design-invariants.md"
FIXTURES_DOC = REPO_ROOT / "docs" / "legibility" / "design-invariants-fixtures.md"
GATES_DOC = REPO_ROOT / "skills" / "prd" / "references" / "gates.md"
CONTRIBUTING_DOC = REPO_ROOT / "CONTRIBUTING.md"

# A family this small would mean the normative doc stopped parsing, not that
# dark-factory shrank its invariant list: eight are landed and none has ever been
# retired. The floor is a NON-VACUITY guard on every comparison downstream, kept
# well below the live count so a deliberate retirement does not go red spuriously.
_MINIMUM_FAMILY_SIZE = 5

# The one structural shape that defines family membership. `##` exactly (a `###`
# sub-heading is a fixture shape in the fixtures doc, not an invariant), a bare
# integer, and a backticked lowercase-kebab slug to end of line.
_HEADING_RE = re.compile(r"^## INV-(\d+) `([a-z0-9][a-z0-9-]*)`$", re.MULTILINE)


def parse_invariant_headings(md_text: str, *, source: str) -> list[tuple[int, str]]:
    """The ordered ``(number, slug)`` family declared by *md_text*'s headings.

    Every failure is a loud ``AssertionError`` naming *source* and the specific
    defect, never an empty list. That is the whole contract: an extractor that
    silently yields nothing turns every downstream drift assertion into an
    empty-vs-empty comparison that PASSES while pinning nothing — strictly worse
    than no guard, because the suite still reports success. *source* is named
    because one extractor parses several docs, so "which doc broke" is not
    recoverable from the traceback.

    Duplicates are checked BEFORE contiguity: a doubled number (1, 2, 2) is a
    duplicate, not a numbering gap, and the message a reader gets should say so.
    """
    pairs = [(int(number), slug) for number, slug in _HEADING_RE.findall(md_text)]
    assert pairs, (
        f"{source}: no `## INV-N `slug`` headings found at all (task 3802). This "
        f"guard derives the whole invariant family from those headings, so an "
        f"empty parse would silently turn every other site's drift check into a "
        f"vacuous pass. Either the heading shape changed (it must be `## INV-<n> "
        f"`<lower-kebab-slug>`` on its own line) or the wrong file was read."
    )

    numbers = [number for number, _ in pairs]
    slugs = [slug for _, slug in pairs]

    duplicate_numbers = sorted({n for n in numbers if numbers.count(n) > 1})
    assert not duplicate_numbers, (
        f"{source}: duplicate number(s) {duplicate_numbers} among the invariant "
        f"headings {pairs} (task 3802) — two sections claim the same INV-N alias, "
        f"so any by-number reference is ambiguous. Renumber one of them."
    )

    duplicate_slugs = sorted({s for s in slugs if slugs.count(s) > 1})
    assert not duplicate_slugs, (
        f"{source}: duplicate slug(s) {duplicate_slugs} among the invariant "
        f"headings {pairs} (task 3802). Slugs are the STABLE ids referenced by G7 "
        f"waivers, `/review`'s invariant_findings and the confusion census's "
        f"invariant_violated field, so two headings sharing one makes every "
        f"by-slug lookup ambiguous. Give each invariant its own slug."
    )

    expected = list(range(1, len(numbers) + 1))
    assert numbers == expected, (
        f"{source}: invariant numbers {numbers} are not contiguous from 1 "
        f"(expected {expected}) (task 3802). Numeric aliases are prose "
        f"convenience over a contiguous 1..N range; a gap or an offset start "
        f"means an invariant was removed or the doc was truncated mid-read, and "
        f"either way the family this guard compares every other site against is "
        f"not the one the doc means."
    )
    return pairs


def _repo_relative(path: Path) -> str:
    """A repo-relative label for assertion messages, so a red run names the file."""
    return str(path.relative_to(REPO_ROOT))


def canonical_family() -> list[tuple[int, str]]:
    """The ordered family, parsed fresh from the normative doc on EVERY call.

    Deliberately not a module constant or a cached snapshot: a stored slug list
    in this file would be one more lock-step copy — the very INV-5 violation this
    guard enforces — and would go stale on the next invariant exactly like the
    prose sites did. Parsing at call time is what makes the source of truth
    auto-extend: adding INV-9 to the doc immediately turns every unpinned site
    red until it is updated.
    """
    return parse_invariant_headings(
        NORMATIVE_DOC.read_text(encoding="utf-8"), source=_repo_relative(NORMATIVE_DOC)
    )


def canonical_slugs() -> list[str]:
    """The family's slugs in canonical (INV-1..INV-N) order."""
    return [slug for _, slug in canonical_family()]


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


# ---------------------------------------------------------------------------
# LIVE: the normative doc parses, and the fixtures doc's sections match it
# ---------------------------------------------------------------------------


def test_normative_doc_parses_to_a_contiguous_unique_family() -> None:
    """The real design-invariants.md yields a usable family — the vacuity floor.

    Contiguity and uniqueness are already enforced inside the extractor; what
    this adds is the SIZE floor. Every other assertion in this module compares
    some site against ``canonical_family()``, so a doc that stopped parsing (a
    heading-style edit, a move) would otherwise make them all pass while pinning
    nothing. The floor sits well below the eight landed invariants so a
    deliberate retirement does not go red spuriously.
    """
    family = canonical_family()

    assert len(family) >= _MINIMUM_FAMILY_SIZE, (
        f"{_repo_relative(NORMATIVE_DOC)} parsed to only {len(family)} invariant(s) "
        f"{family} (task 3802), below the non-vacuity floor of "
        f"{_MINIMUM_FAMILY_SIZE}. Every cross-site check in this module is "
        f"relative to this family, so a truncated parse would silently pin "
        f"nothing. Check the `## INV-N `slug`` heading shape in that doc."
    )


def test_fixtures_doc_sections_match_the_normative_family() -> None:
    """Every invariant has a calibration-fixture section, and no orphans exist.

    ``docs/legibility/design-invariants-fixtures.md`` carries one ``## INV-N
    `slug``` section per invariant. That correspondence does NOT auto-extend —
    it is hand-maintained — so a new invariant lands with no seeded violations to
    calibrate the G7 / Step-5.5 walk against, and a retired one leaves a section
    referencing a slug no gate will ever emit. Compared as an ordered list, since
    both docs number their sections and disagreeing order means one of them
    renumbered without the other.
    """
    normative = canonical_family()
    fixtures = parse_invariant_headings(
        FIXTURES_DOC.read_text(encoding="utf-8"), source=_repo_relative(FIXTURES_DOC)
    )

    missing = [pair for pair in normative if pair not in fixtures]
    extra = [pair for pair in fixtures if pair not in normative]
    assert not missing and not extra, (
        f"{_repo_relative(FIXTURES_DOC)} has drifted from "
        f"{_repo_relative(NORMATIVE_DOC)} (task 3802).\n"
        f"  MISSING fixture section(s) (invariant exists, nothing calibrates the "
        f"walk against it): {missing}\n"
        f"  ORPHAN fixture section(s) (fixture exists for no current invariant): "
        f"{extra}\n"
        f"  normative: {normative}\n"
        f"  fixtures:  {fixtures}\n"
        f"Add a `## INV-N `slug`` section with a PRD-leaf-shaped and a "
        f"code-snippet-shaped seeded violation for each missing invariant, or "
        f"delete the orphan section."
    )
    assert fixtures == normative, (
        f"{_repo_relative(FIXTURES_DOC)}'s sections carry the same invariants as "
        f"{_repo_relative(NORMATIVE_DOC)} but in a different ORDER (task 3802):\n"
        f"  normative: {normative}\n"
        f"  fixtures:  {fixtures}\n"
        f"One doc renumbered without the other."
    )
