"""Shared marker-span extraction for the docs/task-authoring.md guards.

Task 4999, extracted from task 3780's and task 4303's guards. Flagged by
task 4303's amendment pass (reviewer_comprehensive suggestion 2) and
deliberately deferred there — task 4303's lock set covered only
``test_task_authoring_tier_b_canonical_keys.py``, and migrating that one
copy alone would have left both copies in existence AND added a
single-caller indirection, strictly worse than the status quo. Recorded in
CHANGELOG.md under the task-4303 entry as "Deliberately deferred".

Two guards over docs/task-authoring.md each pin a DIFFERENT invariant
across a DIFFERENT marked region of the same document:

* ``test_task_authoring_blessed_keys_drift.py`` (task 3780) — the
  ``tier-a-blessed-keys-mirror`` fence mirrors ``_BLESSED_METADATA_KEYS``.
* ``test_task_authoring_tier_b_canonical_keys.py`` (task 4303) — the
  ``tier-b-canonical-keys`` table's Canonical column must be Tier-A
  blessed.

Both anchor on an EXPLICIT HTML-comment marker pair rather than matching
positionally ("the fence after the Tier-A heading"), for the same reason:
a positional match quietly guards nothing the moment the section is
renamed, reordered, or gains a sibling; an explicit marker fails loudly
instead and names what to restore. Both also carried the SAME three
assertions — exactly one begin marker, exactly one end marker, begin
before end — and the same slice. That machinery lives here now; each
guard keeps its OWN extraction of the marked span's content (a
comma-separated fence vs. a markdown table's first column) and its OWN
messaging for that part, which are genuinely different and must not be
merged — only the marker-span plumbing itself was ever a byte-for-byte
copy.

Every failure below is a loud ``AssertionError`` naming the marker literal
and the document — never a silent ``""`` return. That is the vacuity
hazard this module exists to close: an extractor that silently yields
nothing turns the invariant downstream green while pinning nothing at
all, which is strictly worse than having no guard, because the suite
still reports success.

Importable by NAME (``from task_authoring_marker_span import
marked_span``) because ``tests/scripts/conftest.py`` puts this directory
on ``sys.path``; pytest's ``--import-mode=importlib`` deliberately does
not do that for you.
"""
from __future__ import annotations


def marked_span(
    markdown_text: str,
    begin: str,
    end: str,
    *,
    doc_path: str,
    task: str,
    label: str,
    content: str,
) -> str:
    """Return the text strictly between one *begin* ... *end* marker pair.

    *begin* and *end* must each occur EXACTLY ONCE in *markdown_text*, and
    *begin* must occur strictly before *end* — otherwise this raises
    ``AssertionError`` rather than returning ``""`` or silently picking one
    of several occurrences, either of which would let the caller's guard
    pin nothing while still reporting green.

    *doc_path* and *task* are folded into every failure message so a
    reader knows which document and which guard to fix. *label* names the
    marker pair for the inversion message (e.g. ``"Tier-A mirror"``);
    *content* describes what the pair delimits (e.g. ``"the fenced Tier-A
    listing"``), used to tell a reader where to restore a deleted or
    duplicated marker.

    Does NOT check the returned span for emptiness — an empty span is a
    valid (if useless) result here. Each caller's OWN extraction is what is
    positioned to say what "empty" means for its content (no names parsed
    vs. no key-bearing rows), and to raise loudly for that.
    """
    begin_count = markdown_text.count(begin)
    assert begin_count == 1, (
        f"expected exactly one {begin!r} marker in {doc_path}, found "
        f"{begin_count} ({task}). This marker opens {content}. If it was "
        f"deleted, restore it immediately above; if it was duplicated, one "
        f"of the two copies is unpinned and free to drift."
    )
    end_count = markdown_text.count(end)
    assert end_count == 1, (
        f"expected exactly one {end!r} marker to close {begin!r} in "
        f"{doc_path}, found {end_count} ({task}) — restore the closing "
        f"marker immediately below {content}, and above any explanatory "
        f"prose, which is deliberately outside the marker."
    )

    begin_at = markdown_text.index(begin)
    end_at = markdown_text.index(end)
    assert begin_at < end_at, (
        f"the {label} markers are INVERTED in {doc_path}: {end!r} appears "
        f"before {begin!r} ({task}). Swap them back around {content} — as "
        f"written they delimit an empty span and this guard would pin "
        f"nothing."
    )

    return markdown_text[begin_at + len(begin):end_at]
