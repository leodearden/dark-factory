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


def marked_span(md_text: str, name: str, *, source: str) -> str:
    """The text strictly between ``<!-- name:begin … -->`` and ``<!-- name:end -->``.

    The begin marker's own explanatory comment is EXCLUDED from the returned
    span: it names this test module's path and a task number, either of which a
    claim regex downstream could match, so a span that swallowed it would pin the
    comment instead of the prose.

    Loud on a missing, duplicated or inverted marker, never a ``''`` return. An
    empty span satisfies every claim check by containing no claims; a duplicated
    marker is the same failure one level down, since silently taking the first
    span leaves the second copy unpinned and free to drift.
    """
    begin_marker = f"<!-- {name}:begin"
    end_marker = f"<!-- {name}:end -->"

    begin_count = md_text.count(begin_marker)
    assert begin_count == 1, (
        f"{source}: expected exactly one `{begin_marker}` marker, found "
        f"{begin_count} (task 3802). This marker delimits a span whose contents "
        f"are pinned against the live invariant family. If it was deleted, "
        f"restore it around the claim it named; if it was duplicated, one of the "
        f"two spans is unpinned and free to drift."
    )
    end_count = md_text.count(end_marker)
    assert end_count == 1, (
        f"{source}: expected exactly one `{end_marker}` marker closing "
        f"`{begin_marker}`, found {end_count} (task 3802) — restore the closing "
        f"marker directly below the pinned claim."
    )

    begin_index = md_text.index(begin_marker)
    end_index = md_text.index(end_marker)
    assert begin_index < end_index, (
        f"{source}: the `{name}:end` marker precedes its `{name}:begin` marker "
        f"(task 3802) — the span is inverted, so it delimits everything except "
        f"the claim it was meant to pin. Swap them."
    )

    comment_close = md_text.index("-->", begin_index)
    assert comment_close < end_index, (
        f"{source}: the `{begin_marker}` comment is never closed with `-->` "
        f"before the `{name}:end` marker (task 3802)."
    )
    return md_text[comment_close + len("-->") : end_index]


_RANGE_CLAIM_RE = re.compile(r"INV-(\d+)\.\.(?:INV-)?(\d+)")

_CARDINAL_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

_CARDINAL_CLAIM_RE = re.compile(
    r"\b(" + "|".join(_CARDINAL_WORDS) + r"|\d+)\s+(?:invariants|ids|slugs)\b",
    re.IGNORECASE,
)


def assert_family_claims(
    span_text: str,
    family: list[tuple[int, str]],
    *,
    source: str,
    span_name: str,
    allow_no_claim: bool = False,
) -> None:
    """Every LIVE family-size claim inside a marked span must name the real size.

    Two claim shapes, both measured in the live docs: an ``INV-<a>..[INV-]<b>``
    range token, which must be ``1..len(family)``, and a ``<cardinal>
    invariants|ids|slugs`` phrase, which must name ``len(family)``.

    Applied ONLY inside marked spans, never file-wide. Both docs also carry
    HISTORICAL ranges that are correct as written ("INV-1..INV-5 encode the
    agent-legibility survey's cross-cutting root causes"), and a blanket rule
    could only be greened by falsifying them.

    A span carrying NEITHER claim shape is loud unless *allow_no_claim*. That
    guards the marker drifting off the sentence it was meant to wrap: a prose
    edit that moves the claim out of the span otherwise leaves this check green
    while the claim it names is no longer pinned at all.
    """
    size = len(family)
    found_a_claim = False

    for match in _RANGE_CLAIM_RE.finditer(span_text):
        found_a_claim = True
        low, high = int(match.group(1)), int(match.group(2))
        assert (low, high) == (1, size), (
            f"{source} span {span_name!r}: the range claim {match.group(0)!r} is "
            f"stale (task 3802) — the live family parsed from "
            f"{_repo_relative(NORMATIVE_DOC)} is INV-1..INV-{size}. Update the "
            f"sentence inside the marker, or move the marker if this range is a "
            f"HISTORICAL claim (a founding subset, a dated addendum) that is "
            f"correct as written and must not be pinned."
        )

    for match in _CARDINAL_CLAIM_RE.finditer(span_text):
        found_a_claim = True
        token = match.group(1).lower()
        # A spelled-out cardinal, else a bare integer — the regex admits only
        # those two shapes, so int() cannot raise here.
        claimed = _CARDINAL_WORDS[token] if token in _CARDINAL_WORDS else int(token)
        assert claimed == size, (
            f"{source} span {span_name!r}: the count claim {match.group(0)!r} is "
            f"stale (task 3802) — the live family parsed from "
            f"{_repo_relative(NORMATIVE_DOC)} has {size} invariants. Update the "
            f"count inside the marker, or de-number the sentence so it cannot go "
            f"stale again (the fix CONTRIBUTING.md §6 took)."
        )

    assert found_a_claim or allow_no_claim, (
        f"{source} span {span_name!r}: carries neither an `INV-<a>..<b>` range "
        f"nor a `<cardinal> invariants|ids|slugs` phrase (task 3802), so it pins "
        f"nothing. The marker has most likely drifted off the sentence it was "
        f"meant to wrap — move it back, or pass allow_no_claim=True at the call "
        f"site if this span is deliberately pinned for something else. Span "
        f"text: {span_text!r}"
    )


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


# ---------------------------------------------------------------------------
# marked_span / assert_family_claims — fixture-driven tests
#
# Live family-size claims (INV ranges, cardinal counts) are pinned ONLY inside
# explicit HTML-comment marked spans. See the module docstring: both docs also
# carry HISTORICAL range prose that is correct as written, so a blanket
# file-wide rule would land RED against facts and could only be greened by
# falsifying history.
# ---------------------------------------------------------------------------

# The two live family-size claims in the normative doc. Suffixed names rather
# than one repeated `inv-family-claim` marker: `marked_span` is loud on a
# DUPLICATE marker (an ambiguous span is how a second, unpinned copy hides), and
# that loudness is worth more than the cosmetic saving of a shared name.
_NORMATIVE_CLAIM_SPANS = ("inv-family-claim-intro", "inv-family-claim-census")

# (a) Happy path, modelled on the real doc: the begin marker carries an
# explanatory comment naming the pinning test, and that comment's own prose sits
# OUTSIDE the returned span.
_SPAN_HAPPY = """\
Some prose above.

<!-- inv-family-claim-intro:begin
     Pinned by scripts/tests/test_design_invariants_consistency.py — task 3802. -->
Numeric aliases INV-1..INV-8 are prose convenience only.
<!-- inv-family-claim-intro:end -->

Some prose below.
"""

_SPAN_NO_BEGIN = """\
Numeric aliases INV-1..INV-8 are prose convenience only.
<!-- inv-family-claim-intro:end -->
"""

_SPAN_NO_END = """\
<!-- inv-family-claim-intro:begin -->
Numeric aliases INV-1..INV-8 are prose convenience only.
"""

_SPAN_DUPLICATE_BEGIN = """\
<!-- inv-family-claim-intro:begin -->
Numeric aliases INV-1..INV-8 are prose convenience only.
<!-- inv-family-claim-intro:begin -->
A second, unpinned copy.
<!-- inv-family-claim-intro:end -->
"""

_SPAN_DUPLICATE_END = """\
<!-- inv-family-claim-intro:begin -->
Numeric aliases INV-1..INV-8 are prose convenience only.
<!-- inv-family-claim-intro:end -->
More prose.
<!-- inv-family-claim-intro:end -->
"""

_SPAN_INVERTED = """\
<!-- inv-family-claim-intro:end -->
Numeric aliases INV-1..INV-8 are prose convenience only.
<!-- inv-family-claim-intro:begin -->
"""

# A synthetic family standing in for the live one, so these tests keep asserting
# the same thing when a ninth invariant lands.
_FIXTURE_FAMILY = [(number, f"slug-{number}") for number in range(1, 9)]
_FIXTURE_SPAN_NAME = "a-fixture-span"


def test_marked_span_returns_only_the_text_between_the_markers() -> None:
    """(a) The begin comment's own prose is excluded; the wrapped claim is not."""
    span = marked_span(_SPAN_HAPPY, "inv-family-claim-intro", source=_FIXTURE_SOURCE)

    assert "Numeric aliases INV-1..INV-8 are prose convenience only." in span
    assert "Pinned by scripts/tests" not in span, (
        "the begin comment's explanatory prose must sit outside the span — it "
        f"names a test path and a task number a claim regex could match: {span!r}"
    )
    assert "prose above" not in span and "prose below" not in span


@pytest.mark.parametrize(
    ("markdown_text", "case", "expected_phrase"),
    [
        pytest.param(_SPAN_NO_BEGIN, "missing begin", "begin", id="missing-begin"),
        pytest.param(_SPAN_NO_END, "missing end", "end", id="missing-end"),
        pytest.param(_SPAN_DUPLICATE_BEGIN, "duplicate begin", "found 2", id="duplicate-begin"),
        pytest.param(_SPAN_DUPLICATE_END, "duplicate end", "found 2", id="duplicate-end"),
        pytest.param(_SPAN_INVERTED, "inverted", "precedes", id="inverted-markers"),
    ],
)
def test_marked_span_fails_loudly_on_a_broken_marker(
    markdown_text: str, case: str, expected_phrase: str
) -> None:
    """A missing, duplicated or inverted marker RAISES — never returns ''.

    Missing is the vacuity hazard: an empty span satisfies every claim check
    below by containing no claims to check. Duplicated is the same failure one
    level down — silently taking the first span leaves the second copy of the
    claim unpinned and free to drift, which is precisely the defect this module
    exists to catch.
    """
    with pytest.raises(AssertionError) as excinfo:
        marked_span(markdown_text, "inv-family-claim-intro", source=_FIXTURE_SOURCE)

    message = str(excinfo.value)
    assert _FIXTURE_SOURCE in message, f"{case}: message must name the source: {message!r}"
    assert "inv-family-claim-intro" in message, f"{case}: must name the marker: {message!r}"
    assert expected_phrase in message, f"{case}: must diagnose the defect: {message!r}"


def test_assert_family_claims_accepts_a_current_range_and_cardinal() -> None:
    """A span whose range and spelled-out count both name the live size passes."""
    assert_family_claims(
        "Numeric aliases INV-1..INV-8 are prose convenience; the eight ids above.",
        _FIXTURE_FAMILY,
        source=_FIXTURE_SOURCE,
        span_name=_FIXTURE_SPAN_NAME,
    )


@pytest.mark.parametrize(
    ("span_text", "case", "expected_phrase"),
    [
        pytest.param(
            "Numeric aliases INV-1..INV-7 are prose convenience only.",
            "stale range",
            "INV-1..INV-7",
            id="stale-range",
        ),
        pytest.param(
            "The slug vocabulary is *this* doc — the seven ids above.",
            "stale cardinal",
            "seven ids",
            id="stale-cardinal",
        ),
        pytest.param(
            "Walk the batch against all 7 invariants.",
            "stale numeric cardinal",
            "7 invariants",
            id="stale-numeric-cardinal",
        ),
        pytest.param(
            "The founding subset INV-2..INV-8 is not the family.",
            "range not starting at 1",
            "INV-2..INV-8",
            id="range-offset-start",
        ),
    ],
)
def test_assert_family_claims_rejects_a_stale_claim(
    span_text: str, case: str, expected_phrase: str
) -> None:
    """A range or cardinal disagreeing with the live family size RAISES.

    This is the drift the marked spans exist to catch: the claim reads fine in
    isolation and only becomes false when an invariant is added, which is exactly
    when nobody is looking at the sentence that mentions a count.
    """
    with pytest.raises(AssertionError) as excinfo:
        assert_family_claims(
            span_text, _FIXTURE_FAMILY, source=_FIXTURE_SOURCE, span_name=_FIXTURE_SPAN_NAME
        )

    message = str(excinfo.value)
    assert _FIXTURE_SOURCE in message, f"{case}: message must name the source: {message!r}"
    assert _FIXTURE_SPAN_NAME in message, f"{case}: message must name the span: {message!r}"
    assert expected_phrase in message, f"{case}: message must quote the claim: {message!r}"


def test_assert_family_claims_is_loud_on_a_span_carrying_no_claim() -> None:
    """A span with neither a range nor a cardinal RAISES by default.

    The failure mode this guards is a marker drifting off the sentence it was
    meant to wrap — a prose edit that moves the claim out of the span leaves the
    guard green while the claim it names is no longer pinned at all.
    """
    with pytest.raises(AssertionError) as excinfo:
        assert_family_claims(
            "A gate checklist, not an essay.",
            _FIXTURE_FAMILY,
            source=_FIXTURE_SOURCE,
            span_name=_FIXTURE_SPAN_NAME,
        )

    message = str(excinfo.value)
    assert _FIXTURE_SOURCE in message
    assert _FIXTURE_SPAN_NAME in message


def test_assert_family_claims_permits_a_claimless_span_when_explicitly_allowed() -> None:
    """``allow_no_claim=True`` is the only way to opt a span out of the floor.

    Explicit at the call site, so a span pinned for its slugs rather than its
    counts says so in code instead of passing vacuously by accident.
    """
    assert_family_claims(
        "A gate checklist, not an essay.",
        _FIXTURE_FAMILY,
        source=_FIXTURE_SOURCE,
        span_name=_FIXTURE_SPAN_NAME,
        allow_no_claim=True,
    )


def test_normative_doc_family_size_claims_are_current() -> None:
    """LIVE: design-invariants.md's two marked family-size claims name the real size.

    The doc states the family size twice in prose — "Numeric aliases INV-1..INV-N
    are prose convenience only" in the intro, and "the slug vocabulary is *this*
    doc — the <n> ids above" at the Census seam. Neither auto-extends. Both are
    wrapped in `inv-family-claim-*` markers so this check can find them without a
    content heuristic, and so a human editing the doc sees at the edit site that
    the sentence is pinned.

    The doc's HISTORICAL ranges are deliberately NOT wrapped and NOT checked:
    "INV-1..INV-5 encode the agent-legibility survey's cross-cutting root causes"
    and the INV-6..INV-7 / INV-8 provenance sentences are true as written, and a
    blanket file-wide range rule could only be greened by falsifying them.
    """
    family = canonical_family()
    text = NORMATIVE_DOC.read_text(encoding="utf-8")
    source = _repo_relative(NORMATIVE_DOC)

    for span_name in _NORMATIVE_CLAIM_SPANS:
        span = marked_span(text, span_name, source=source)
        assert_family_claims(span, family, source=source, span_name=span_name)


# ---------------------------------------------------------------------------
# gates.md — TWO independent enumerations, pinned differently on purpose
#
# Task 3811 (commit f29da1855b) deliberately preserved the two-site shape:
# gates.md:181 records that collapsing them was considered and REJECTED. The
# family-inventory row is read generically off each project's own normative doc
# by projects that HAVE adopted one; the trigger-shape list exists precisely for
# projects that have NOT, and is a hand-distilled illustrative set. So both are
# pinned, but not in the same way — see the two live tests below.
# ---------------------------------------------------------------------------

_GATES_TRIGGER_SPAN = "inv-trigger-shapes"

_SLUGS_HAPPY = "adds a fallback (`storm-escape-required`)? a contract in prose (`a-slug`)?"

# Every backticked non-slug shape the real docs carry near an enumeration:
# a numeric alias, a dotted metadata key, an angle-bracket placeholder, a
# CamelCase symbol, a path, and a call expression.
_SLUGS_DECOYS = (
    "See `INV-5`, `metadata.g7_waivers`, `G7 waiver: <slug>`, `FailureCategory`, "
    "`docs/legibility/design-invariants.md`, `_run()` and `Y` — none is a slug, "
    "but `real-slug` is."
)

# The measured shape of the live trigger-shape paragraph: `contracts-machine-checked`
# appears TWICE, once for "a tool without a declared filter/envelope convention"
# and once for "a contract in prose". Duplicates must survive extraction so the
# caller — not the extractor — decides whether order and multiplicity matter.
_SLUGS_DUPLICATED = (
    "a tool without a declared envelope (`contracts-machine-checked`)? "
    "a contract in prose (`contracts-machine-checked`)? "
    "a log-scrape (`structured-facts-at-failure`)?"
)

# A fixture table carrying BOTH project rows. The reify row's INV-SF slugs are
# the same lexical shape as dark-factory's, so an extractor keyed on "a table row
# with backticked slugs" would silently merge two projects' families — and the
# merged set would then never equal either one.
_FAMILY_ROW_TABLE = """\
| Project | Family |
|---|---|
| dark-factory | INV-1..3 — `a-slug`, `b-slug`, `c-slug` (the doc is normative) |
| reify | INV-SF-1..2 (silent-failure) — `undef-has-provenance`, `diagnostics-carry-codes` |
"""

_FAMILY_ROW_ABSENT = """\
| Project | Family |
|---|---|
| reify | INV-SF-1..2 (silent-failure) — `undef-has-provenance`, `diagnostics-carry-codes` |
"""

_FAMILY_ROW_DUPLICATED = """\
| dark-factory | INV-1..3 — `a-slug`, `b-slug`, `c-slug` |
| reify | INV-SF-1..2 — `undef-has-provenance` |
| dark-factory | INV-1..2 — `a-slug`, `b-slug` |
"""


def test_slugs_in_span_returns_backticked_slugs_in_document_order() -> None:
    assert slugs_in_span(_SLUGS_HAPPY) == ["storm-escape-required", "a-slug"]


def test_slugs_in_span_ignores_backticked_non_slug_tokens() -> None:
    """Numeric aliases, dotted keys, placeholders, CamelCase and paths are not slugs.

    All six shapes are measured in gates.md and design-invariants.md near an
    enumeration. An extractor keyed on "anything backticked" would fold them into
    the family and make the set comparison fail with a nonsense diff.
    """
    assert slugs_in_span(_SLUGS_DECOYS) == ["real-slug"]


def test_slugs_in_span_preserves_duplicates_in_order() -> None:
    """Duplicates survive extraction; the CALLER decides whether they matter.

    The live trigger-shape paragraph names `contracts-machine-checked` twice
    because two distinct trigger shapes map to that one invariant. Silently
    de-duplicating inside the extractor would hide that from the ordered
    family-row check, which legitimately must not tolerate a repeat.
    """
    assert slugs_in_span(_SLUGS_DUPLICATED) == [
        "contracts-machine-checked",
        "contracts-machine-checked",
        "structured-facts-at-failure",
    ]


def test_dark_factory_family_row_ignores_the_reify_row() -> None:
    """DECOY IMMUNITY: the sibling project's row has the same lexical shape.

    Merging the two families is the failure that would not announce itself — the
    combined set simply never equals dark-factory's, and the diff would blame the
    wrong rows.
    """
    row = dark_factory_family_row(_FAMILY_ROW_TABLE)

    assert slugs_in_span(row) == ["a-slug", "b-slug", "c-slug"]
    assert "undef-has-provenance" not in row


@pytest.mark.parametrize(
    ("markdown_text", "case", "expected_phrase"),
    [
        pytest.param(_FAMILY_ROW_ABSENT, "absent", "found 0", id="row-absent"),
        pytest.param(_FAMILY_ROW_DUPLICATED, "duplicated", "found 2", id="row-duplicated"),
    ],
)
def test_dark_factory_family_row_fails_loudly(
    markdown_text: str, case: str, expected_phrase: str
) -> None:
    """A missing or duplicated row RAISES — never returns ''.

    Absent is the vacuity hazard (an empty row yields an empty slug list, and
    empty-vs-a-real-family at least fails loudly — but empty-vs-empty would not,
    if the family ever failed to parse too). Duplicated means one of the two rows
    is unpinned and free to drift.
    """
    with pytest.raises(AssertionError) as excinfo:
        dark_factory_family_row(markdown_text)

    message = str(excinfo.value)
    assert "gates.md" in message, f"{case}: message must name the doc: {message!r}"
    assert expected_phrase in message, f"{case}: message must diagnose the defect: {message!r}"


def test_gates_family_row_lists_the_whole_family_in_order() -> None:
    """LIVE: gates.md's family-inventory row transcribes the family, in order.

    Pinned as an ORDERED list, unlike the trigger-shape span below: this row is a
    straight canonical-order transcription today, so ordering is free signal —
    a mis-ordered row means someone edited it by hand against a stale copy.
    ``assert_family_claims`` additionally pins the row's own `INV-1..N` range
    token, which is a second thing that does not auto-extend.
    """
    family = canonical_family()
    text = GATES_DOC.read_text(encoding="utf-8")
    source = _repo_relative(GATES_DOC)
    row = dark_factory_family_row(text)

    assert slugs_in_span(row) == canonical_slugs(), (
        f"{source}: the `| dark-factory |` family-inventory row has drifted from "
        f"{_repo_relative(NORMATIVE_DOC)} (task 3802).\n"
        f"  row:       {slugs_in_span(row)}\n"
        f"  normative: {canonical_slugs()}\n"
        f"Update the row to list every slug in canonical INV-1..N order. It is "
        f"illustrative — the doc is normative — but an illustration naming the "
        f"wrong family is worse than none."
    )
    assert_family_claims(
        row, family, source=source, span_name="the `| dark-factory |` family-inventory row"
    )


def test_gates_trigger_shape_list_covers_every_invariant() -> None:
    """LIVE: the G7 fallback trigger-shape list names every invariant at least once.

    A SET comparison, deliberately, and not an ordered or exact-multiset one. The
    paragraph is in NON-canonical order and legitimately names
    `contracts-machine-checked` TWICE — once for "a tool without a declared
    filter/envelope convention", once for "a contract in prose" — because two
    distinct trigger shapes map to that invariant. An ordered assertion would be
    RED-and-unfixable without rewriting normative gate prose this task is not
    scoped to author.

    A set still catches the defect this site has actually exhibited twice: an
    invariant added to the normative doc with no trigger shape appended here, so
    projects with no invariants file of their own screen against an incomplete
    list and never see the new failure mode.
    """
    text = GATES_DOC.read_text(encoding="utf-8")
    source = _repo_relative(GATES_DOC)
    span = marked_span(text, _GATES_TRIGGER_SPAN, source=source)

    listed = set(slugs_in_span(span))
    canonical = set(canonical_slugs())
    missing = sorted(canonical - listed)
    unknown = sorted(listed - canonical)

    assert not missing and not unknown, (
        f"{source}: the G7 trigger-shape fallback list inside the "
        f"`{_GATES_TRIGGER_SPAN}` marker has drifted from "
        f"{_repo_relative(NORMATIVE_DOC)} (task 3802).\n"
        f"  MISSING (invariant exists, no trigger shape screens for it): {missing}\n"
        f"  UNKNOWN (trigger shape names no current invariant): {unknown}\n"
        f"Append the new invariant's trigger shape to the G7 fallback list in "
        f"gates.md — the same instruction that paragraph's own follow-up already "
        f"carries. This list is what projects WITHOUT their own "
        f"design-invariants.md screen against, so a gap here is a gate that "
        f"silently stops covering a known failure mode."
    )
