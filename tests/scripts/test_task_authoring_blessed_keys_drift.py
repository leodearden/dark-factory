"""Mirror contract: docs/task-authoring.md §8 restates ``_BLESSED_METADATA_KEYS`` exactly.

Task 3780. The Tier-A listing in ``docs/task-authoring.md`` enumerates the same
allowlist that ``shared/src/shared/task_metadata.py`` enumerates in code. Nothing
kept the two in step until this guard, and the listing is the copy a task author
actually reads before deciding whether a key they are about to write will
manufacture ``unknown_key`` census noise.

WHY A MACHINE CHECK, MEASURED NOT PREDICTED. Task 4372 (landed 2026-08-17, the
day before this guard) blessed ``source_finding_id`` and ``stage1_finding_id`` in
commit ``4722efeb16``, then had to mirror both into the doc BY HAND in a separate
follow-up commit ``ca71e0e12a``, whose message states the hazard outright: "The
listing in docs/task-authoring.md is hand-maintained prose with no sync test, so
it drifts silently if not mirrored by hand." The follow-up landed; the next one
might not, and nothing would have failed. The same file already demonstrates
where that ends — the frozenset's own header comment said "the 39 load-bearing
...keys" while the frozenset held 42, having gone silently stale across two
independent blessings (tasks 3697 and 4372) before task 3780 dropped the numeral.

The drift is not cosmetic. UNDER-listing is the dangerous direction: an author
consulting the doc sees a key absent, concludes it is unblessed, and either
renames a machine-written key under ``x_`` (forking the vocabulary against every
sibling task and blinding its live reader) or files a redundant blessing task.
OVER-listing sends them the other way — they write a key the parser still
censures, and the census line they were told would not appear, appears.

IMPORTS THE FROZENSET, NEVER REGEXES THE SOURCE. ``human_curator_gate`` enters
the set through the ``HUMAN_CURATOR_GATE_KEY`` symbol rather than a string
literal, so a source-parsing implementation of this guard would silently miss it
and report a phantom drift — or, worse, be "fixed" by deleting the real entry
from the doc.

PLACEMENT IS LOAD-BEARING, not incidental. ``dark-factory-orchestrator.yaml``'s
``test_command`` ends with ``uv run --project shared pytest tests/scripts/
scripts/tests/ --timeout=300``, so a guard here runs under FULL_SUITE and under
merge-role ``merge_verify_breadth: full``, AND runs under the ``shared`` project
so ``from shared.task_metadata import ...`` resolves.

OUT OF SCOPE. This asserts equality between two committed enumerations of the
same machine-consumed allowlist. It makes NO assertion about the surrounding
prose — the qualifying paragraphs below the fence (the finding-provenance
exception, the ``cross_repo`` explanation, the ``execution_class`` note) are
deliberately outside the marker and unpinned, and this guard must not be extended
to them. Parser BEHAVIOUR is pinned by ``shared/tests/test_task_metadata.py``;
neither implies the other.
"""
from __future__ import annotations

import pathlib

import pytest
from shared.task_metadata import _BLESSED_METADATA_KEYS

REPO_ROOT = pathlib.Path(__file__).parents[2]

TASK_AUTHORING_PATH = REPO_ROOT / "docs" / "task-authoring.md"

# The full HTML-comment forms, not the bare slug: the begin literal is not a
# substring of the end literal (the `/` differs), so `.count()` on each is
# unambiguous.
MIRROR_BEGIN = "<!-- tier-a-blessed-keys-mirror -->"
MIRROR_END = "<!-- /tier-a-blessed-keys-mirror -->"

_FENCE = "```"


def _documented_blessed_keys(markdown_text):
    """The comma-separated key names in the fence delimited by the mirror markers.

    Anchored on an EXPLICIT marker pair rather than positionally ("the fenced
    block after the Tier-A heading"). A positional match quietly guards nothing
    the moment the section is renamed, reordered, or gains a second fence; an
    explicit marker fails loudly instead, and the failure names what to restore.

    Every failure is a loud ``AssertionError`` naming the marker literal and the
    doc, never a ``[]``/``None`` return. That is the vacuity hazard and the whole
    point of this function: an extractor that silently yields nothing turns the
    drift assertion downstream green while pinning nothing at all — strictly
    worse than having no guard, because the suite still reports success.

    Returns the names in DOCUMENT ORDER (a list, not a set) so the caller can
    also detect an accidental double-entry, which set equality alone cannot see.
    """
    begin_count = markdown_text.count(MIRROR_BEGIN)
    assert begin_count == 1, (
        f"expected exactly one {MIRROR_BEGIN!r} marker in docs/task-authoring.md, "
        f"found {begin_count} (task 3780). This marker opens the fenced Tier-A "
        f"listing that mirrors _BLESSED_METADATA_KEYS in "
        f"shared/src/shared/task_metadata.py. If it was deleted, restore it "
        f"immediately above that fence; if it was duplicated, one of the two "
        f"listings is unpinned and free to drift."
    )
    end_count = markdown_text.count(MIRROR_END)
    assert end_count == 1, (
        f"expected exactly one {MIRROR_END!r} marker to close {MIRROR_BEGIN!r} in "
        f"docs/task-authoring.md, found {end_count} (task 3780) — restore the "
        f"closing marker immediately below the fenced Tier-A listing, and above "
        f"the explanatory paragraphs, which are deliberately outside the marker."
    )

    begin_at = markdown_text.index(MIRROR_BEGIN)
    end_at = markdown_text.index(MIRROR_END)
    assert begin_at < end_at, (
        f"the Tier-A mirror markers are INVERTED in docs/task-authoring.md: "
        f"{MIRROR_END!r} appears before {MIRROR_BEGIN!r} (task 3780). Swap them "
        f"back around the fenced listing — as written they delimit an empty span "
        f"and this guard would pin nothing."
    )

    marked = markdown_text[begin_at + len(MIRROR_BEGIN):end_at]

    # Strip the fence delimiter lines; everything between them is the listing.
    # Keeping this tolerant (bare names, wrapped across lines, no per-line
    # trailing-comma discipline, stray backticks defended against) is deliberate:
    # the listing is prose formatting, and a brittle parser here would report a
    # formatting reflow as a vocabulary drift.
    body_lines = [
        line for line in marked.splitlines() if not line.strip().startswith(_FENCE)
    ]
    body = " ".join(body_lines)
    names = [name.strip().strip("`").strip() for name in body.split(",")]
    names = [name for name in names if name]

    assert names, (
        f"the fenced listing between {MIRROR_BEGIN!r} and {MIRROR_END!r} in "
        f"docs/task-authoring.md is empty (task 3780) — this mirror invariant "
        f"would pass vacuously against an empty allowlist"
    )
    return names


_HAPPY_DOC = """\
This is `_BLESSED_METADATA_KEYS` in `shared/src/shared/task_metadata.py`:

<!-- tier-a-blessed-keys-mirror -->
```
alpha, beta, gamma_delta,
epsilon
```
<!-- /tier-a-blessed-keys-mirror -->

Some qualifying prose mentioning `zeta` and `eta`, deliberately outside the
marker and therefore unpinned.
"""

_HAPPY_KEYS = ["alpha", "beta", "gamma_delta", "epsilon"]

_NO_BEGIN_MARKER_DOC = """\
```
alpha, beta
```
<!-- /tier-a-blessed-keys-mirror -->
"""

_NO_END_MARKER_DOC = """\
<!-- tier-a-blessed-keys-mirror -->
```
alpha, beta
```
"""

_DUPLICATE_MARKER_DOC = """\
<!-- tier-a-blessed-keys-mirror -->
```
alpha, beta
```
<!-- /tier-a-blessed-keys-mirror -->

## A later section

<!-- tier-a-blessed-keys-mirror -->
```
gamma, delta
```
<!-- /tier-a-blessed-keys-mirror -->
"""

_INVERTED_MARKER_DOC = """\
<!-- /tier-a-blessed-keys-mirror -->
```
alpha, beta
```
<!-- tier-a-blessed-keys-mirror -->
"""

_EMPTY_FENCE_DOC = """\
<!-- tier-a-blessed-keys-mirror -->
```
```
<!-- /tier-a-blessed-keys-mirror -->
"""

_DECOY_DOC = """\
```
decoy_before, another_decoy
```

<!-- tier-a-blessed-keys-mirror -->
```
alpha, beta, gamma_delta,
epsilon
```
<!-- /tier-a-blessed-keys-mirror -->

```
decoy_after
```
"""


def test_documented_blessed_keys_extracts_the_marked_fence():
    """(a) Only the marked fence's names are returned, in document order."""
    assert _documented_blessed_keys(_HAPPY_DOC) == _HAPPY_KEYS


def test_documented_blessed_keys_ignores_unmarked_fences():
    """(b) Sibling fenced blocks outside the marker are never extracted.

    §8 already carries other fenced blocks (the `grep` recipes above the Tier-A
    heading, among others). An extractor keyed on "the next fence" rather than on
    the marker would pick one up, and the resulting drift report would name shell
    fragments as metadata keys — a confusing failure whose obvious "fix" is to
    edit the wrong block.
    """
    assert _documented_blessed_keys(_DECOY_DOC) == _HAPPY_KEYS


@pytest.mark.parametrize(
    "markdown_text,case",
    [
        (_NO_BEGIN_MARKER_DOC, "opening marker deleted"),
        (_NO_END_MARKER_DOC, "closing marker deleted"),
        (_DUPLICATE_MARKER_DOC, "marker pair duplicated"),
        (_INVERTED_MARKER_DOC, "markers inverted"),
        (_EMPTY_FENCE_DOC, "marked fence emptied"),
    ],
)
def test_documented_blessed_keys_fails_loudly_on_a_broken_marker(markdown_text, case):
    """(c) A missing, duplicated, inverted or emptied marker RAISES — never [].

    All five are the same failure at different depths: each would otherwise
    silently reduce this guard to a tautology while the suite kept reporting
    green. The message must tell a human what to restore and where, so the
    assertions check the marker literal and the doc path are both named.
    """
    with pytest.raises(AssertionError) as excinfo:
        _documented_blessed_keys(markdown_text)

    message = str(excinfo.value)
    assert "tier-a-blessed-keys-mirror" in message, case
    assert "task-authoring.md" in message, case


def test_task_authoring_marker_pair_is_present_and_non_empty():
    """The LIVE doc carries the marker pair around a non-empty listing.

    Separated from the equality assertion below on purpose. If the marker is ever
    deleted outright, this fails with "restore the marker" rather than with a
    43-key drift report, which would send a reader off editing the listing
    instead of the markup that went missing.
    """
    documented = _documented_blessed_keys(
        TASK_AUTHORING_PATH.read_text(encoding="utf-8")
    )
    assert documented, "the marked Tier-A listing extracted as empty (task 3780)"


def test_task_authoring_tier_a_listing_has_no_duplicate_names():
    """A name listed twice is a real defect that SET equality cannot see.

    The listing is hand-wrapped prose, so a copy-paste during a reflow can
    plausibly double an entry. Left unchecked it would survive every future run
    of the equality assertion below, since {a, b} == {a, b} regardless.
    """
    documented = _documented_blessed_keys(
        TASK_AUTHORING_PATH.read_text(encoding="utf-8")
    )
    duplicates = sorted({name for name in documented if documented.count(name) > 1})
    assert not duplicates, (
        f"docs/task-authoring.md's Tier-A listing names {duplicates} more than "
        f"once (task 3780) — remove the duplicate entries; the set-equality "
        f"assertion cannot see them"
    )


def test_task_authoring_tier_a_listing_mirrors_the_blessed_frozenset():
    """The documented Tier-A listing must equal ``_BLESSED_METADATA_KEYS`` as a set.

    Reads BOTH sides live: the doc from the committed markdown, the allowlist by
    IMPORTING the frozenset (see the module docstring for why never by regexing
    the source).

    Reports ``missing`` and ``extra`` separately because the two have opposite
    remedies and opposite consequences for a task author — under-listing tells
    them a blessed key is unblessed, over-listing tells them an unblessed key is
    safe to write.

    MEASURED RED at the ORIGINAL base (2026-08-18): the marker pair did not
    exist, and the listing was missing ``execution_class``. Both legs of that RED
    were this task's own doing — verified mechanically beforehand that the listing
    was otherwise exactly ``sorted(_BLESSED_METADATA_KEYS)`` (42 names, 42 unique,
    empty symmetric difference), so this guard was sound before it was needed.

    That "otherwise exact" property did NOT survive the rebase onto main, and the
    difference is the guard earning its keep on its first contact with real drift:
    task 3122 blessed ``files_tagged_empty`` in commit ``0197216782`` and never
    mirrored it into the doc, so main's listing was silently under-listing it —
    exactly the hand-maintained-prose failure this guard exists to catch, landed
    independently of this task and found only because the guard ran. The mirror
    entry was added in this branch's post-rebase reconcile commit; the blessing
    itself is main's, not this task's.
    """
    documented = _documented_blessed_keys(
        TASK_AUTHORING_PATH.read_text(encoding="utf-8")
    )

    # NON-VACUITY, the code side. `_documented_blessed_keys` already refuses an
    # empty doc listing; an empty frozenset would pass just as vacuously.
    assert _BLESSED_METADATA_KEYS, (
        "_BLESSED_METADATA_KEYS is empty (task 3780) — this mirror invariant "
        "would pass vacuously"
    )

    missing = sorted(_BLESSED_METADATA_KEYS - set(documented))
    extra = sorted(set(documented) - _BLESSED_METADATA_KEYS)
    assert not missing and not extra, (
        f"docs/task-authoring.md's Tier-A listing has drifted from "
        f"_BLESSED_METADATA_KEYS in shared/src/shared/task_metadata.py "
        f"(task 3780).\n"
        f" MISSING from the doc (blessed in code, absent from the listing): "
        f"{missing}\n"
        f" EXTRA in the doc (listed as blessed, but still censured by the "
        f"parser): {extra}\n"
        f"MISSING is the dangerous direction: an author consulting the doc sees "
        f"the key absent, concludes it is unblessed, and either x_-renames a "
        f"machine-written key — forking the vocabulary against every sibling "
        f"task and blinding its live reader — or files a redundant blessing "
        f"task. EXTRA sends them the other way: they write a key the parser "
        f"still censures. If you just blessed or unblessed a key, mirror the "
        f"change into the fence inside the `tier-a-blessed-keys-mirror` marker."
    )
