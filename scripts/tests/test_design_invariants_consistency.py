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


# A backticked token whose ENTIRE content is the canonical slug shape. Requiring
# the closing backtick immediately after the run is what excludes every
# backticked non-slug the docs carry beside real slugs: `INV-5` (uppercase),
# `metadata.g7_waivers` (dot), `docs/legibility/...` (slash), `_run()`
# (underscore), `G7 waiver: <slug>` (space).
_SLUG_TOKEN_RE = re.compile(r"`([a-z0-9][a-z0-9-]*)`")

# The family-inventory row's own anchor. A row PREFIX rather than a marked span:
# it is already a stable, self-describing anchor, and the test asserts there is
# exactly one such line — which a content heuristic could not.
_FAMILY_ROW_PREFIX = "| dark-factory |"


def slugs_in_span(span_text: str) -> list[str]:
    """Backticked canonical-slug-shaped tokens, in document order, duplicates kept.

    Duplicates are deliberately NOT collapsed here. The live trigger-shape
    paragraph names one invariant twice because two distinct trigger shapes map
    to it, while the family-inventory row must not repeat a slug at all — so
    multiplicity is the CALLER's decision, and an extractor that de-duplicated
    would quietly take it away from both.

    Takes a span, not a whole document, precisely because slug-shaped tokens
    appear in ordinary prose too (gates.md discusses `no-lockstep-duplication` by
    name outside both enumerations). Delimiting the span is the caller's job.
    """
    return _SLUG_TOKEN_RE.findall(span_text)


def dark_factory_family_row(md_text: str) -> str:
    """The single ``| dark-factory |`` row of gates.md's family-inventory table.

    Loud when the row is absent or duplicated, never a ``''`` return. The row is
    matched by its LINE PREFIX rather than by content: the adjacent ``| reify |``
    row lists INV-SF slugs of exactly the same lexical shape, so any "table row
    containing backticked slugs" heuristic would merge two projects' families
    into a set that equals neither — and would then blame the wrong rows in its
    diff.
    """
    rows = [line for line in md_text.splitlines() if line.startswith(_FAMILY_ROW_PREFIX)]
    assert len(rows) == 1, (
        f"skills/prd/references/gates.md: expected exactly one line starting "
        f"`{_FAMILY_ROW_PREFIX}` in the G7 family-inventory table, found "
        f"{len(rows)} (task 3802). Absent means this guard would pin nothing; "
        f"duplicated means one of the two rows is unpinned and free to drift. "
        f"Matched rows: {rows!r}"
    )
    return rows[0]


def section_span(md_text: str, heading_prefix: str, *, source: str) -> str:
    """One markdown section: its ``## <heading_prefix>…`` line to the next ``## ``.

    Line-anchored on the heading rather than sliced by index of a substring, so a
    section name quoted inside a paragraph elsewhere cannot re-target the span.
    Stops at the next ``## `` specifically — not at the next heading of any level
    — so the section's own ``###`` sub-headings stay inside it.

    Loud on an absent or duplicated heading, never a ``''`` return. This
    extractor feeds ABSENCE assertions, where an empty span is the worst possible
    silent failure: it satisfies "contains no restated slug" by containing
    nothing at all.
    """
    lines = md_text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(f"## {heading_prefix}")]
    assert len(starts) == 1, (
        f"{source}: expected exactly one `## {heading_prefix}…` heading, found "
        f"{len(starts)} (task 3802). Absent means the section was renamed or "
        f"renumbered and this guard would pin nothing; duplicated means one of "
        f"the two copies is unchecked. Update the heading prefix constant in "
        f"scripts/tests/test_design_invariants_consistency.py, or de-duplicate "
        f"the section."
    )

    start = starts[0]
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")),
        len(lines),
    )
    return "\n".join(lines[start:end])


# A rehearsal verdict row, anchored on its Fixture ID cell. Matched by ROW SHAPE
# over the whole document rather than per-table: the live row set is split over a
# base table plus two dated addendum tables, so "parse the table" would silently
# cover a third of it.
_VERDICT_ROW_RE = re.compile(r"^\| `INV-(\d+)-(PRD|CODE)` \|")

# The legend's own cumulative-count claim — a second thing that does not
# auto-extend when an addendum walk adds rows.
_CUMULATIVE_ROWS_RE = re.compile(r"(\d+) rows cumulative")

_BACKTICKED_SLUG_RE = re.compile(r"^`([a-z0-9][a-z0-9-]*)`$")


def verdict_table_rows(md_text: str, *, source: str) -> list[tuple[int, str, str]]:
    """Every rehearsal verdict row as ``(number, shape, expected_slug)``.

    The Expected-slug cell is column 4. A row that is too short, or whose cell is
    empty or not a backticked slug, RAISES rather than yielding ``None``: a
    ``None`` would flow into the slug-equality check and surface as "expected
    `contracts-machine-checked`, got None" — a red with a misleading diagnosis
    pointing at the family rather than at the malformed row.
    """
    rows: list[tuple[int, str, str]] = []
    for line in md_text.splitlines():
        match = _VERDICT_ROW_RE.match(line)
        if match is None:
            continue
        cells = [cell.strip() for cell in line.split("|")]
        assert len(cells) >= 6, (
            f"{source}: verdict row {line!r} has {len(cells)} pipe-separated "
            f"cells, too few to carry an Expected-slug column (task 3802)."
        )
        slug_match = _BACKTICKED_SLUG_RE.match(cells[4])
        assert slug_match is not None, (
            f"{source}: the Expected-slug cell of verdict row "
            f"`INV-{match.group(1)}-{match.group(2)}` is {cells[4]!r}, not a "
            f"backticked slug (task 3802). That column is the one this guard "
            f"compares against the normative family, so an unreadable cell must "
            f"fail loudly rather than compare as None."
        )
        rows.append((int(match.group(1)), match.group(2), slug_match.group(1)))

    assert rows, (
        f"{source}: no `| `INV-N-PRD|CODE` |` verdict rows found at all (task "
        f"3802) — the coverage check would pass vacuously. Either the row shape "
        f"changed or the wrong file was read."
    )
    return rows


def assert_verdict_table_covers(
    md_text: str, family: list[tuple[int, str]], *, source: str
) -> None:
    """Every invariant is rehearsed in both shapes, with the right expected slug.

    COVERAGE only — never the Verdict/rationale column, which the doc's own
    "Snapshot caveat" declares a point-in-time transcription rather than a live
    pin on the G7 / Step-5.5 text.

    Unknown numbers are checked FIRST so a renumbering reports "INV-9 names no
    current invariant" rather than the per-invariant shape failure it also causes.
    """
    rows = verdict_table_rows(md_text, source=source)
    canonical = dict(family)

    unknown = sorted({number for number, _, _ in rows if number not in canonical})
    assert not unknown, (
        f"{source}: verdict row(s) for INV-{unknown} name no current invariant "
        f"(task 3802); the family is INV-1..INV-{len(family)}. Delete the orphan "
        f"rows, or renumber them if the fixture survived a family renumbering."
    )

    for number, slug in family:
        shapes = sorted(shape for row_number, shape, _ in rows if row_number == number)
        assert shapes == ["CODE", "PRD"], (
            f"{source}: INV-{number} `{slug}` has verdict rows {shapes}, expected "
            f"exactly one PRD row and one CODE row (task 3802). The rehearsal "
            f"table does not auto-extend: a new invariant lands with nothing "
            f"calibrating the G7 walk (PRD shape) or the /review Step-5.5 audit "
            f"(CODE shape) against it."
        )

    mismatched = [
        (number, shape, cell, canonical[number])
        for number, shape, cell in rows
        if cell != canonical[number]
    ]
    assert not mismatched, (
        f"{source}: verdict row(s) whose Expected-slug cell disagrees with "
        f"{_repo_relative(NORMATIVE_DOC)} (task 3802) — (number, shape, cell, "
        f"canonical): {mismatched}. The slug is what the gate must emit, so a "
        f"stale cell rehearses the wrong acceptance."
    )

    claims = _CUMULATIVE_ROWS_RE.findall(md_text)
    assert len(claims) == 1, (
        f"{source}: expected exactly one `<N> rows cumulative` legend claim, "
        f"found {len(claims)}: {claims} (task 3802)."
    )
    assert int(claims[0]) == len(rows), (
        f"{source}: the legend claims {claims[0]} rows cumulative but the tables "
        f"carry {len(rows)} (task 3802) — update the legend when an addendum "
        f"walk adds rows."
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


# ---------------------------------------------------------------------------
# CONTRIBUTING.md — pinned as an ABSENCE
#
# §6 restated all eight slugs twelve lines above its own rule saying not to
# ("it's the single normative copy ...; don't restate them elsewhere"), and
# design-invariants.md says the same at its head ("no restatement, per INV-5").
# The fix DELETES the duplicate rather than re-syncing it, so this site cannot
# go stale again; the assertion below is the machine-checkable form of the
# anti-restatement rule itself.
# ---------------------------------------------------------------------------

_CONTRIBUTING_SECTION = "6. Design invariants"

_SECTION_DOC = """\
# Contributing

## 5. Git workflow

Branch off main.

## 6. Design invariants & PRD gates

Every task in a batch is walked against the invariants.

### A sub-heading inside §6

Still inside section six.

## 7. Commit messages

Match the observed convention.
"""

_SECTION_DOC_DUPLICATED = """\
## 6. Design invariants & PRD gates

First copy.

## 7. Commit messages

## 6. Design invariants & PRD gates

Second copy, e.g. from a bad merge.
"""


def test_section_span_stops_at_the_next_top_level_heading() -> None:
    """The span covers the section's own body — sub-headings in, siblings out.

    Stopping at the next ``## `` (not at the next heading of any level) is what
    keeps §6's own sub-headings inside the span while excluding §5 and §7. A span
    that ran to EOF would make the absence assertions below pin the whole file
    under a section's name, which is a different — and much more brittle — claim
    than the one this test means to make.
    """
    span = section_span(_SECTION_DOC, _CONTRIBUTING_SECTION, source=_FIXTURE_SOURCE)

    assert "Every task in a batch is walked" in span
    assert "Still inside section six" in span, "a `###` sub-heading must not end the span"
    assert "Branch off main" not in span, "the span must not run backwards into §5"
    assert "Match the observed convention" not in span, "the span must stop at the next `## `"


@pytest.mark.parametrize(
    ("markdown_text", "heading_prefix", "case", "expected_phrase"),
    [
        pytest.param(
            _SECTION_DOC, "9. No such section", "absent", "found 0", id="section-absent"
        ),
        pytest.param(
            _SECTION_DOC_DUPLICATED,
            _CONTRIBUTING_SECTION,
            "duplicated",
            "found 2",
            id="section-duplicated",
        ),
    ],
)
def test_section_span_fails_loudly(
    markdown_text: str, heading_prefix: str, case: str, expected_phrase: str
) -> None:
    """A missing or duplicated heading RAISES — never returns ''.

    An empty span passes every absence assertion below by containing nothing to
    find, which is the vacuity failure in its purest form: the guard would report
    success precisely because it had stopped looking.
    """
    with pytest.raises(AssertionError) as excinfo:
        section_span(markdown_text, heading_prefix, source=_FIXTURE_SOURCE)

    message = str(excinfo.value)
    assert _FIXTURE_SOURCE in message, f"{case}: message must name the source: {message!r}"
    assert heading_prefix in message, f"{case}: message must name the heading: {message!r}"
    assert expected_phrase in message, f"{case}: message must diagnose the defect: {message!r}"


def test_contributing_does_not_restate_the_invariant_family() -> None:
    """LIVE: CONTRIBUTING.md points at the normative doc instead of copying it.

    Three assertions, one theme — this site is pinned as an ABSENCE:

    (a) WHOLE-FILE, zero canonical slugs. CONTRIBUTING.md §6 used to restate all
        eight, twelve lines above its own rule forbidding restatement. Re-syncing
        that list would have fixed today's contradiction and left the site free to
        drift again on the next invariant; deleting it means there is nothing left
        to drift. This assertion is the anti-restatement rule made machine-
        checkable — it fails the moment anyone re-introduces a copy, which a
        content-equality pin never would.

    (b) WHOLE-FILE, every `INV-<a>..<b>` range names the live family. This
        deliberately KEEPS the repo-layout bullet's "design-invariants.md
        (INV-1..INV-N, gates ...)" orienting hint alive rather than deleting it:
        unlike a slug list it is cheap to keep current, and now it is pinned.
        ``assert_family_claims`` also covers cardinal-count phrases file-wide,
        which is strictly stricter and cannot fire on anything but a genuinely
        stale count — the exact drift this task exists to stop.

    (c) §6 SPAN carries no `<cardinal> invariants` phrase (it said "the single
        normative copy of the eight invariants") and DOES carry the literal path
        `docs/legibility/design-invariants.md`, so the bullet stays a working
        pointer rather than becoming a dangling reference. A pointer that no
        longer names its target is how a deletion turns into an orphan.
    """
    family = canonical_family()
    text = CONTRIBUTING_DOC.read_text(encoding="utf-8")
    source = _repo_relative(CONTRIBUTING_DOC)

    restated = [slug for slug in canonical_slugs() if slug in text]
    assert not restated, (
        f"{source} restates {len(restated)} invariant slug(s) {restated} (task "
        f"3802). That doc's own §6 says of design-invariants.md: \"it's the "
        f"single normative copy; don't restate them elsewhere\", and the "
        f"normative doc says the same at its head (\"no restatement, per "
        f"INV-5\"). Replace the copy with a pointer to "
        f"{_repo_relative(NORMATIVE_DOC)} — a restated list here has already "
        f"gone stale once and cannot be kept current by policy alone."
    )

    assert_family_claims(text, family, source=source, span_name="the whole file")

    section = section_span(text, _CONTRIBUTING_SECTION, source=source)

    stale_count = _CARDINAL_CLAIM_RE.search(section)
    assert stale_count is None, (
        f"{source} §{_CONTRIBUTING_SECTION} states the family size in prose "
        f"({stale_count.group(0)!r} — task 3802). Nothing here needs a count: "
        f"de-number the sentence (\"the single normative copy\") so it cannot go "
        f"stale when an invariant is added."
    )

    assert _repo_relative(NORMATIVE_DOC) in section, (
        f"{source} §{_CONTRIBUTING_SECTION} no longer names "
        f"{_repo_relative(NORMATIVE_DOC)} (task 3802). This section is pinned as "
        f"an ABSENCE — it must not restate the invariants — which only works if "
        f"it still POINTS at the doc that does. Without the path, deleting the "
        f"restatement just turned the G7 bullet into a dangling reference."
    )


# ---------------------------------------------------------------------------
# The fixtures doc's rehearsal verdict table — pinned for COVERAGE only
#
# The doc carries an explicit "Snapshot caveat" declaring the Verdict column a
# point-in-time transcription of the G7 / Step-5.5 wording, NOT a live pin on
# those source docs, and tells readers to re-walk rather than trust the quoted
# rationale as current. So this guard asserts COVERAGE — every invariant present
# in both PRD and CODE shape, each row's Expected-slug cell equal to the
# canonical slug, the legend's cumulative count current — and never the prose.
# Coverage is the property that actually fails to auto-extend when an invariant
# is added, so it is the right thing to mechanize.
# ---------------------------------------------------------------------------

_SMALL_FAMILY = [(1, "a-slug"), (2, "b-slug")]

# Rows are split over THREE tables in the live doc (a base table plus two dated
# addenda), so the extractor must collect across all of them rather than parse
# "the" table. Modelled on that shape.
_VERDICT_HAPPY = """\
Acceptance: every fixture flags with the correct slug — 4 rows cumulative, all `Y`.

| Fixture ID | Shape | Invariant | Expected slug | Verdict | Match |
|---|---|---|---|---|---|
| `INV-1-PRD` | PRD | INV-1 a-slug | `a-slug` | G7's list fires | Y |
| `INV-1-CODE` | CODE | INV-1 a-slug | `a-slug` | Step 5.5 fires | Y |

### Addendum — later walk

| Fixture ID | Shape | Invariant | Expected slug | Verdict | Match |
|---|---|---|---|---|---|
| `INV-2-PRD` | PRD | INV-2 b-slug | `b-slug` | G7's list fires | Y |
| `INV-2-CODE` | CODE | INV-2 b-slug | `b-slug` | Step 5.5 fires | Y |
"""

_VERDICT_MISSING_CODE = _VERDICT_HAPPY.replace(
    "| `INV-2-CODE` | CODE | INV-2 b-slug | `b-slug` | Step 5.5 fires | Y |\n", ""
).replace("4 rows cumulative", "3 rows cumulative")

_VERDICT_WRONG_SLUG = _VERDICT_HAPPY.replace(
    "| `INV-2-PRD` | PRD | INV-2 b-slug | `b-slug` |",
    "| `INV-2-PRD` | PRD | INV-2 b-slug | `a-slug` |",
)

_VERDICT_UNKNOWN_NUMBER = _VERDICT_HAPPY.replace("| `INV-2-CODE` |", "| `INV-9-CODE` |")

_VERDICT_STALE_LEGEND = _VERDICT_HAPPY.replace("4 rows cumulative", "5 rows cumulative")

_VERDICT_TOO_FEW_COLUMNS = "| `INV-1-PRD` | PRD |\n"

_VERDICT_EMPTY_SLUG_CELL = "| `INV-1-PRD` | PRD | INV-1 a-slug |  | fires | Y |\n"


def test_verdict_table_rows_collects_rows_across_every_table() -> None:
    """Rows are collected by ROW SHAPE, not per-table — the live set is split in three."""
    assert verdict_table_rows(_VERDICT_HAPPY, source=_FIXTURE_SOURCE) == [
        (1, "PRD", "a-slug"),
        (1, "CODE", "a-slug"),
        (2, "PRD", "b-slug"),
        (2, "CODE", "b-slug"),
    ]


@pytest.mark.parametrize(
    ("markdown_text", "case"),
    [
        pytest.param(_VERDICT_TOO_FEW_COLUMNS, "too few columns", id="short-row"),
        pytest.param(_VERDICT_EMPTY_SLUG_CELL, "empty expected-slug cell", id="empty-slug-cell"),
        pytest.param("no table here at all\n", "no rows", id="no-rows"),
    ],
)
def test_verdict_table_rows_fails_loudly(markdown_text: str, case: str) -> None:
    """A malformed or absent row RAISES rather than yielding ``None`` or ``[]``.

    Yielding ``None`` for an unreadable Expected-slug cell would make the
    slug-equality check below compare ``None`` against a real slug — a red with a
    misleading diagnosis — and an empty row list would make coverage pass
    vacuously.
    """
    with pytest.raises(AssertionError) as excinfo:
        verdict_table_rows(markdown_text, source=_FIXTURE_SOURCE)

    assert _FIXTURE_SOURCE in str(excinfo.value), case


def test_assert_verdict_table_covers_accepts_a_complete_table() -> None:
    assert_verdict_table_covers(_VERDICT_HAPPY, _SMALL_FAMILY, source=_FIXTURE_SOURCE)


@pytest.mark.parametrize(
    ("markdown_text", "case", "expected_phrase"),
    [
        pytest.param(_VERDICT_MISSING_CODE, "missing CODE row", "CODE", id="missing-code-row"),
        pytest.param(_VERDICT_WRONG_SLUG, "wrong expected slug", "a-slug", id="wrong-slug"),
        pytest.param(_VERDICT_UNKNOWN_NUMBER, "unknown invariant", "9", id="unknown-number"),
        pytest.param(_VERDICT_STALE_LEGEND, "stale legend", "5", id="stale-legend-count"),
    ],
)
def test_assert_verdict_table_covers_rejects_drift(
    markdown_text: str, case: str, expected_phrase: str
) -> None:
    """Each drift shape the table can develop when the family changes fires."""
    with pytest.raises(AssertionError) as excinfo:
        assert_verdict_table_covers(markdown_text, _SMALL_FAMILY, source=_FIXTURE_SOURCE)

    message = str(excinfo.value)
    assert _FIXTURE_SOURCE in message, f"{case}: {message!r}"
    assert expected_phrase in message, f"{case}: message must quote the defect: {message!r}"


def test_fixtures_verdict_table_covers_every_invariant_in_both_shapes() -> None:
    """LIVE: the rehearsal table walks every invariant in both fixture shapes.

    Coverage only. The Verdict/rationale column is deliberately NOT pinned: the
    doc's own "Snapshot caveat" declares it a point-in-time transcription of the
    G7 and Step-5.5 text as it read on 2026-07-14, and instructs readers to
    re-walk the fixtures against the current wording rather than trust the quoted
    rationale. Pinning that prose would contradict the doc's stated contract and
    would go red on any unrelated gate-wording edit.
    """
    assert_verdict_table_covers(
        FIXTURES_DOC.read_text(encoding="utf-8"),
        canonical_family(),
        source=_repo_relative(FIXTURES_DOC),
    )


# ---------------------------------------------------------------------------
# Registry completeness — the SITE LIST is itself mechanized
#
# A drift guard whose site list is hand-maintained reproduces the exact defect it
# exists to prevent: it reads green while a newly created enumeration site sits
# unpinned. So `PINNED_SITES` is not trusted — it is checked against a scan. Any
# markdown under the repo root, docs/legibility/ or skills/ carrying at least
# `_ENUMERATION_THRESHOLD` DISTINCT canonical slugs must be registered.
#
# Registration means "this file's relationship to the family is pinned", NOT
# "this file enumerates": CONTRIBUTING.md is registered and carries ZERO slugs,
# because what is pinned there is the ABSENCE of a restatement.
#
# plans/ and docs/prds/ are excluded BY DESIGN, not by oversight. Their PRDs and
# capability manifests transcribe slugs as point-in-time G7 walk records (a scan
# measured fourteen such files at 5-7 slugs each); those records must NOT be
# updated when the family changes, so pinning them would force rewriting history.
# ---------------------------------------------------------------------------


def _write_fixture_site(directory: Path, name: str, slugs: list[str]) -> Path:
    """A throwaway markdown file enumerating `slugs`, for the scan's unit tests.

    Written under ``tmp_path`` rather than into the repo: the scan's own fixtures
    must not be discoverable BY the scan, or the live assertion would go red on
    this module's test data.
    """
    body = "\n".join(f"- `{slug}` — a restated entry" for slug in slugs)
    path = directory / name
    path.write_text(f"# fixture site\n\n{body}\n", encoding="utf-8")
    return path


_FIXTURE_SLUGS = [slug for _, slug in _FIXTURE_FAMILY]


def test_unregistered_enumeration_sites_is_empty_when_every_site_is_pinned(
    tmp_path: Path,
) -> None:
    """The green case: over-threshold files are all registered, so nothing is returned.

    Includes the two shapes that must NOT be reported — a registered file carrying
    zero slugs (CONTRIBUTING.md's post-fix state, pinned as an absence) and an
    unregistered file that mentions slugs but stays under the threshold.
    """
    enumerating = _write_fixture_site(tmp_path, "enumerating.md", _FIXTURE_SLUGS[:4])
    absence = _write_fixture_site(tmp_path, "absence.md", [])
    under = _write_fixture_site(tmp_path, "under-threshold.md", _FIXTURE_SLUGS[:3])

    registry = {str(enumerating): "enumerates the family", str(absence): "pinned as an absence"}

    assert (
        unregistered_enumeration_sites(
            [enumerating, absence, under], registry, _FIXTURE_FAMILY, threshold=4
        )
        == []
    )


def test_unregistered_enumeration_sites_reports_a_new_unpinned_site(tmp_path: Path) -> None:
    """A NEW enumeration site that nobody pinned is exactly what this scan is for."""
    registered = _write_fixture_site(tmp_path, "registered.md", _FIXTURE_SLUGS[:5])
    newcomer = _write_fixture_site(tmp_path, "newcomer.md", _FIXTURE_SLUGS[:4])

    registry = {str(registered): "enumerates the family"}

    assert unregistered_enumeration_sites(
        [registered, newcomer], registry, _FIXTURE_FAMILY, threshold=4
    ) == [str(newcomer)]


def test_unregistered_enumeration_sites_counts_distinct_slugs_only(tmp_path: Path) -> None:
    """Threshold is DISTINCT slugs: a doc quoting one slug six times is not an enumeration.

    Counting occurrences instead would flag every doc that discusses a single
    invariant in depth — noise that would train readers to add files to the
    registry to silence it, which is how a guard stops meaning anything.
    """
    repeated = _write_fixture_site(tmp_path, "repeated.md", [_FIXTURE_SLUGS[0]] * 6)

    assert unregistered_enumeration_sites([repeated], {}, _FIXTURE_FAMILY, threshold=4) == []


def test_unregistered_enumeration_sites_fails_loudly_on_an_empty_scan(tmp_path: Path) -> None:
    """An empty file list RAISES rather than returning ``[]``.

    ``[]`` from an empty scan is indistinguishable from "every site is pinned" —
    a broken glob would report the guard's strongest possible result while
    checking nothing at all.
    """
    with pytest.raises(AssertionError) as excinfo:
        unregistered_enumeration_sites([], {}, _FIXTURE_FAMILY, threshold=4)

    assert "no files" in str(excinfo.value).lower()
    assert tmp_path.exists()  # the fixture dir is unused here by design


def test_every_enumeration_site_is_pinned() -> None:
    """LIVE: no markdown file enumerates the family without being registered.

    This is the assertion that keeps `PINNED_SITES` honest. Without it the
    registry would be one more hand-maintained enumeration — the very shape this
    module exists to mechanize — and a newly written site could enumerate the
    family, drift on the next invariant, and never turn anything red.

    The scan is asserted NON-VACUOUS in two independent ways before its result is
    trusted: every registered site must actually be reachable by the globs (a
    typo'd root or a moved doc is otherwise silently unpinned), and the normative
    doc itself must clear the threshold (it defines every slug, so if IT does not
    register as an enumeration the reader, the slug family, or the threshold is
    broken).
    """
    family = canonical_family()
    scanned = _enumeration_scan_files()

    missing = sorted(site for site in PINNED_SITES if REPO_ROOT / site not in scanned)
    assert not missing, (
        f"registered site(s) {missing} are not reachable by the enumeration scan "
        f"(task 3802). Either the file moved — update PINNED_SITES — or the scan "
        f"roots {sorted(_SCAN_ROOTS)} no longer cover it, in which case a whole "
        f"tree of sites is unpinned and this guard is reading green over nothing."
    )

    assert NORMATIVE_DOC in scanned and (
        len({slug for _, slug in family if slug in NORMATIVE_DOC.read_text(encoding="utf-8")})
        >= _ENUMERATION_THRESHOLD
    ), (
        f"{_repo_relative(NORMATIVE_DOC)} defines every slug in the family yet "
        f"does not clear the {_ENUMERATION_THRESHOLD}-slug enumeration threshold "
        f"(task 3802) — the scan cannot detect an enumeration site at all, so its "
        f"empty result below would be vacuous."
    )

    unregistered = unregistered_enumeration_sites(
        scanned, PINNED_SITES, family, threshold=_ENUMERATION_THRESHOLD
    )
    assert not unregistered, (
        f"markdown file(s) {unregistered} restate {_ENUMERATION_THRESHOLD}+ "
        f"invariant slugs but are not in PINNED_SITES (task 3802). Every such "
        f"copy has to be updated by hand when an invariant is added, and every "
        f"one this repo has had drifted. Either pin the site here (add it to "
        f"PINNED_SITES with an assertion covering what it enumerates) or stop "
        f"enumerating there and point at {_repo_relative(NORMATIVE_DOC)} instead. "
        f"Note that plans/ and docs/prds/ are excluded on purpose — they record "
        f"point-in-time G7 walks that must not be retro-edited — so a new PRD "
        f"transcribing slugs will never appear here."
    )
