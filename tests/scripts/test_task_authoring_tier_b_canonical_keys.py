"""Contract: every key in docs/task-authoring.md §8's Tier-B Canonical column is Tier-A blessed.

Task 4303. The Tier-B table pairs each **canonical** metadata spelling with the
**aliases to avoid**, and its own preamble promises that the aliases "each still
emit ``code=unknown_key`` as a greppable drift signal until the caller is fixed
to use the canonical spelling". That promise is only meaningful if migrating to
the canonical spelling actually CLEARS the warning — which requires the
canonical to be on the Tier-A allowlist (``_BLESSED_METADATA_KEYS`` in
``shared/src/shared/task_metadata.py``). Nothing enforced that.

WHY A MACHINE CHECK, MEASURED NOT PREDICTED. On the base tree ``related_tasks``
was the SOLE violation — measured over the Canonical column, ``prd_path``,
``prd_task_label``, ``invariants`` and ``source_finding_id`` were all blessed
and ``related_tasks`` was not — so §8 told every author to migrate toward a
spelling that still minted a census line, and the drift signal could not
discriminate "used a deprecated alias" from "used the canonical spelling". That
is a CLASS defect, not a one-off: without this guard the next Tier-B row whose
canonical is unblessed silently recreates it, with no test failing.

The consequence is worse than a stray census line. The alias-vs-canonical
distinction is what an operator's ``grep code=unknown_key`` is FOR. When the
canonical warns too, the grep returns the compliant authors alongside the
drifting ones and stops being actionable — and the corpus-dominance evidence
that later blessing rulings rest on gets read off a contaminated count.

IMPORTS THE FROZENSET, NEVER REGEXES THE SOURCE. Borrowed wholesale from the
task-3780 guard next door, whose stated reason applies identically here:
``human_curator_gate`` enters the allowlist through the
``HUMAN_CURATOR_GATE_KEY`` symbol rather than a string literal, so a
source-parsing implementation would silently miss it and report phantom drift —
or be "fixed" by deleting a real entry.

A SEPARATE FILE FROM ``test_task_authoring_blessed_keys_drift.py``, deliberately.
That guard's docstring carries an explicit OUT OF SCOPE clause: it asserts
equality between two enumerations of the same allowlist, "makes NO assertion
about the surrounding prose", and states that "this guard must not be extended
to them" — naming the very paragraphs this table sits among. This is a different
invariant over a different region of the document, so it gets its own markers
and its own failure message.

PLACEMENT IS LOAD-BEARING, not incidental. ``dark-factory-orchestrator.yaml``'s
``test_command`` ends with ``uv run --project shared pytest tests/scripts/
scripts/tests/ --timeout=300``, so a guard here runs under FULL_SUITE and under
merge-role ``merge_verify_breadth: full``, AND runs under the ``shared`` project
so ``from shared.task_metadata import ...`` resolves.

OUT OF SCOPE. This pins a cross-artifact KEY-SET contract — which spellings the
table nominates as canonical, and whether the parser is silent for them. It
asserts NOTHING about prose wording, about the Aliases column, or about the
qualifying paragraphs below the table (the ``esc-3796-1`` two-classes ruling,
the ``related_memory_ids`` note, the ``stage1_finding_id`` note), all of which
sit deliberately OUTSIDE the marker pair and stay unpinned. Note in particular
that the blessed-ALIAS exception the table documents (``origin_finding_id``,
silent by design) is orthogonal to this invariant and is not contradicted by it:
this guard constrains the Canonical column only. Parser BEHAVIOUR for the
individual keys is pinned by ``shared/tests/test_task_metadata.py``; neither
implies the other.
"""
from __future__ import annotations

import pathlib
import re

import pytest
from shared.task_metadata import _BLESSED_METADATA_KEYS

REPO_ROOT = pathlib.Path(__file__).parents[2]

TASK_AUTHORING_PATH = REPO_ROOT / "docs" / "task-authoring.md"

# The full HTML-comment forms, not the bare slug: the begin literal is not a
# substring of the end literal (the `/` differs), so `.count()` on each is
# unambiguous. Distinct slug from the `tier-a-blessed-keys-mirror` pair higher
# up the same document — the two marker pairs are independent.
MARKER_BEGIN = "<!-- tier-b-canonical-keys -->"
MARKER_END = "<!-- /tier-b-canonical-keys -->"

# `key` — every backtick-quoted name in a cell. Deliberately matched globally
# per cell rather than treating a cell as one key: see `_canonical_keys`.
_BACKTICKED = re.compile(r"`([^`]+)`")

# The markdown header/separator rows, which carry no keys.
_SEPARATOR = re.compile(r"^\|[\s:|-]+\|$")


def _table_rows(markdown_text):
    """The markdown table rows between the Tier-B markers, header/separator dropped.

    Anchored on an EXPLICIT marker pair rather than positionally ("the table
    after the Tier-B heading"). A positional match quietly guards nothing the
    moment the section is renamed, reordered, or gains a second table; an
    explicit marker fails loudly instead, and the failure names what to restore.

    Every failure is a loud ``AssertionError`` naming the marker literal and the
    doc, never a ``[]``/``None`` return. That is the vacuity hazard and the whole
    point: an extractor that silently yields nothing turns the invariant
    downstream green while pinning nothing at all — strictly worse than having no
    guard, because the suite still reports success.
    """
    begin_count = markdown_text.count(MARKER_BEGIN)
    assert begin_count == 1, (
        f"expected exactly one {MARKER_BEGIN!r} marker in docs/task-authoring.md, "
        f"found {begin_count} (task 4303). This marker opens the Tier-B table "
        f"whose Canonical column must be Tier-A blessed. If it was deleted, "
        f"restore it immediately above that table; if it was duplicated, one of "
        f"the two tables is unpinned and free to drift."
    )
    end_count = markdown_text.count(MARKER_END)
    assert end_count == 1, (
        f"expected exactly one {MARKER_END!r} marker to close {MARKER_BEGIN!r} in "
        f"docs/task-authoring.md, found {end_count} (task 4303) — restore the "
        f"closing marker immediately below the Tier-B table, and ABOVE the "
        f"qualifying paragraphs, which are deliberately outside the marker."
    )

    begin_at = markdown_text.index(MARKER_BEGIN)
    end_at = markdown_text.index(MARKER_END)
    assert begin_at < end_at, (
        f"the Tier-B markers are INVERTED in docs/task-authoring.md: "
        f"{MARKER_END!r} appears before {MARKER_BEGIN!r} (task 4303). Swap them "
        f"back around the table — as written they delimit an empty span and this "
        f"guard would pin nothing."
    )

    marked = markdown_text[begin_at + len(MARKER_BEGIN):end_at]
    rows = [
        line.strip()
        for line in marked.splitlines()
        if line.strip().startswith("|") and not _SEPARATOR.match(line.strip())
    ]
    # Drop the header row (`| Canonical | Aliases to avoid |`) — it carries no
    # backticked names, so it contributes nothing, but dropping it explicitly
    # keeps the row count honest for the more-keys-than-rows assertion below.
    rows = [row for row in rows if _BACKTICKED.search(row)]

    assert rows, (
        f"the table between {MARKER_BEGIN!r} and {MARKER_END!r} in "
        f"docs/task-authoring.md has no key-bearing rows (task 4303) — this "
        f"invariant would pass vacuously against an empty table"
    )
    return rows


def _canonical_keys(markdown_text):
    """Every backtick-quoted name in the FIRST cell of each Tier-B row.

    EVERY name, not one per cell. The first row's canonical cell reads
    ``` `prd_path` + `prd_task_label` ``` — two keys joined by " + " inside a
    single cell — so a "one cell is one key" parse silently drops
    ``prd_task_label`` and the invariant stops covering it. That trap is pinned
    by ``test_canonical_cell_with_two_keys_yields_both`` below.

    Returns names in DOCUMENT ORDER (a list, not a set) so a caller can also see
    an accidental double-entry, which set membership alone cannot.
    """
    names = []
    for row in _table_rows(markdown_text):
        # `row` is `| canonical | aliases |`; split on the pipes and take the
        # first non-empty cell. Only the Canonical column is in scope — the
        # Aliases column is deliberately unpinned (they must stay UNblessed,
        # which is asserted as parser behaviour in
        # shared/tests/test_task_metadata.py, not here).
        cells = row.strip("|").split("|")
        names.extend(_BACKTICKED.findall(cells[0]))
    return names


# --------------------------------------------------------------------------
# Guard-the-guard: the extractor's own behaviour, on synthetic documents.
# --------------------------------------------------------------------------

_HAPPY_DOC = """\
Some preamble prose naming `zeta`, deliberately outside the marker.

<!-- tier-b-canonical-keys -->

| Canonical | Aliases to avoid |
|---|---|
| `alpha` + `beta` | `alfa`, `bta` |
| `gamma` | `gama` |

<!-- /tier-b-canonical-keys -->

Qualifying paragraphs mentioning `eta`, deliberately outside the marker and
therefore unpinned.
"""

_NO_BEGIN_MARKER_DOC = """\
| `alpha` | `alfa` |
<!-- /tier-b-canonical-keys -->
"""

_NO_END_MARKER_DOC = """\
<!-- tier-b-canonical-keys -->
| `alpha` | `alfa` |
"""

_DUPLICATE_MARKER_DOC = """\
<!-- tier-b-canonical-keys -->
| `alpha` | `alfa` |
<!-- /tier-b-canonical-keys -->

## A later section

<!-- tier-b-canonical-keys -->
| `gamma` | `gama` |
<!-- /tier-b-canonical-keys -->
"""

_INVERTED_MARKER_DOC = """\
<!-- /tier-b-canonical-keys -->
| `alpha` | `alfa` |
<!-- tier-b-canonical-keys -->
"""

_EMPTY_TABLE_DOC = """\
<!-- tier-b-canonical-keys -->

| Canonical | Aliases to avoid |
|---|---|

<!-- /tier-b-canonical-keys -->

Prose naming `alpha` outside the marker, which must NOT be picked up.
"""


def test_canonical_keys_extracts_the_marked_table():
    assert _canonical_keys(_HAPPY_DOC) == ["alpha", "beta", "gamma"]


def test_canonical_cell_with_two_keys_yields_both():
    """The " + " trap: one cell can name two canonical keys.

    ``| `prd_path` + `prd_task_label` |`` is a real row in the live table. A
    naive one-key-per-cell parse drops the second name and the invariant
    silently stops covering it — the exact shape of the bug this file exists to
    prevent, reintroduced inside the guard itself.
    """
    keys = _canonical_keys(_HAPPY_DOC)
    assert "beta" in keys, "the second key in a ' + '-joined canonical cell was dropped"
    assert len(keys) > len(_table_rows(_HAPPY_DOC)), (
        "extracted no more keys than there are rows — the ' + '-joined cell is "
        "not being split, so at least one canonical key is unpinned"
    )


def test_canonical_keys_ignores_names_outside_the_marker():
    """Prose above and below the marker is unpinned and must not leak in."""
    keys = _canonical_keys(_HAPPY_DOC)
    assert "zeta" not in keys
    assert "eta" not in keys
    # The Aliases column is out of scope too — those must stay UNblessed.
    assert "alfa" not in keys and "gama" not in keys


@pytest.mark.parametrize(
    "markdown_text, case",
    [
        (_NO_BEGIN_MARKER_DOC, "missing begin marker"),
        (_NO_END_MARKER_DOC, "missing end marker"),
        (_DUPLICATE_MARKER_DOC, "duplicated marker pair"),
        (_INVERTED_MARKER_DOC, "inverted markers"),
        (_EMPTY_TABLE_DOC, "marked span holds no key-bearing rows"),
    ],
)
def test_canonical_keys_fails_loudly_on_a_broken_marker(markdown_text, case):
    """A broken marker must raise, never return an empty list.

    A silent ``[]`` would turn the invariant below green while pinning nothing —
    the vacuity failure that makes a guard worse than no guard, because the suite
    still reports success.
    """
    with pytest.raises(AssertionError):
        _canonical_keys(markdown_text)


# --------------------------------------------------------------------------
# The invariant.
# --------------------------------------------------------------------------


def test_tier_b_marker_pair_is_present_and_non_empty():
    """The live document carries the marker pair around a non-empty table."""
    keys = _canonical_keys(TASK_AUTHORING_PATH.read_text(encoding="utf-8"))
    assert keys, (
        "no canonical keys extracted from the Tier-B table in "
        "docs/task-authoring.md (task 4303) — the invariant below would pass "
        "vacuously"
    )


def test_tier_b_canonical_column_has_no_duplicate_names():
    """A spelling nominated as canonical twice is a table error, invisible to set logic."""
    documented = _canonical_keys(TASK_AUTHORING_PATH.read_text(encoding="utf-8"))
    duplicates = sorted({name for name in documented if documented.count(name) > 1})
    assert not duplicates, (
        f"docs/task-authoring.md's Tier-B Canonical column names {duplicates} "
        f"more than once (task 4303) — a spelling cannot be the canonical target "
        f"of two different rows; the membership assertion cannot see this"
    )


def test_every_tier_b_canonical_key_is_tier_a_blessed():
    """Migrating to a documented canonical spelling must actually clear the warning.

    Reads BOTH sides live: the table from the committed markdown, the allowlist
    by IMPORTING the frozenset (see the module docstring for why never by
    regexing the source).

    MEASURED NON-VACUITY, not assumed. This guard's RED in development was
    scaffolding-shaped — the markers did not exist yet — which proves the
    extractor RUNS, not that the invariant BITES. So it was falsified by hand at
    the commit that added the markers: with the markers in place,
    ``'related_tasks'`` was temporarily removed from ``_BLESSED_METADATA_KEYS``
    and this test FAILED with ``AssertionError: ... nominates ['related_tasks']
    as canonical spelling(s), but they are NOT in _BLESSED_METADATA_KEYS``;
    restoring the entry returned all 11 tests in this file to green. The removal
    was never committed. So the guard demonstrably catches the real bug it was
    written for, rather than only its own missing scaffolding.

    Also measured at that commit: the extractor pulls 5 canonical keys from 4
    table rows, confirming the ``+``-joined first cell really is split on the
    LIVE document and not only on the synthetic one.
    """
    documented = _canonical_keys(TASK_AUTHORING_PATH.read_text(encoding="utf-8"))

    # NON-VACUITY, the code side. `_canonical_keys` already refuses an empty
    # table; an empty frozenset would pass just as vacuously.
    assert _BLESSED_METADATA_KEYS, (
        "_BLESSED_METADATA_KEYS is empty (task 4303) — this invariant would pass "
        "vacuously"
    )

    unblessed = sorted(set(documented) - _BLESSED_METADATA_KEYS)
    assert not unblessed, (
        f"docs/task-authoring.md §8's Tier-B table nominates {unblessed} as "
        f"canonical spelling(s), but they are NOT in _BLESSED_METADATA_KEYS in "
        f"shared/src/shared/task_metadata.py (task 4303).\n"
        f"An author who follows §8 and migrates to one of these will STILL mint "
        f"a code=unknown_key census line, and the table's own preamble — that "
        f"the aliases warn 'until the caller is fixed to use the canonical "
        f"spelling' — becomes false. Worse, the drift signal stops "
        f"discriminating: `grep code=unknown_key` then returns the compliant "
        f"authors alongside the drifting ones and is no longer actionable.\n"
        f"Remedy: bless the key (add it to _BLESSED_METADATA_KEYS with a "
        f"rationale comment AND mirror it into the Tier-A listing inside the "
        f"`tier-a-blessed-keys-mirror` fence — those two are pinned to each "
        f"other by test_task_authoring_blessed_keys_drift.py and must move in "
        f"one commit), or stop nominating it as canonical."
    )
