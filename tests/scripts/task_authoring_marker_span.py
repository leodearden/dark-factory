"""Shared marker-span extraction for the docs/task-authoring.md guards.

Task 4999, extracted from task 3780's and task 4303's guards, which had each
grown a byte-for-byte copy of the machinery below. (Deliberately deferred
during task 4303 for want of a lock on the sibling guard — see CHANGELOG.md's
task-4303 entry.)

Two guards over docs/task-authoring.md each pin a DIFFERENT invariant across a
DIFFERENT marked region of the same document:

* ``test_task_authoring_blessed_keys_drift.py`` (task 3780) — the
  ``tier-a-blessed-keys-mirror`` fence mirrors ``_BLESSED_METADATA_KEYS``.
* ``test_task_authoring_tier_b_canonical_keys.py`` (task 4303) — the
  ``tier-b-canonical-keys`` table's Canonical column must be Tier-A blessed.

WHY AN EXPLICIT MARKER PAIR, not a positional match ("the fence after the
Tier-A heading"): a positional match quietly guards nothing the moment the
section is renamed, reordered, or gains a sibling; an explicit marker fails
loudly instead and names what to restore.

WHY EVERY FAILURE HERE IS A LOUD ``AssertionError`` naming the marker literal
and the document, never a silent ``""`` return: an extractor that silently
yields nothing turns the invariant downstream green while pinning nothing at
all, which is strictly worse than having no guard, because the suite still
reports success. That vacuity hazard is what this module exists to close.

Only the marker-span plumbing is shared. Each guard keeps its OWN extraction of
the span's content — a comma-separated fence vs. a markdown table's first
column — and its own messaging for that part; those are genuinely different and
must not be merged.

Importable by NAME (``from task_authoring_marker_span import marked_span``)
because ``tests/scripts/conftest.py`` puts this directory on ``sys.path``;
pytest's ``--import-mode=importlib`` deliberately does not do that for you.
"""
from __future__ import annotations


def marked_span(
    markdown_text: str,
    begin: str,
    end: str,
    *,
    doc_path: str,
    task: str,
    delimits: str,
) -> str:
    """Return the text strictly between one *begin* ... *end* marker pair.

    *begin* and *end* must each occur EXACTLY ONCE in *markdown_text*, and
    *begin* must occur strictly before *end* — otherwise this raises
    ``AssertionError`` rather than returning ``""`` or silently picking one
    of several occurrences, either of which would let the caller's guard
    pin nothing while still reporting green.

    PRECONDITION, asserted below: neither marker literal may be a substring of
    the other, since both are located by ``.count()``/``.index()`` over the
    same text. Pass the full HTML-comment forms (``<!-- slug -->`` and
    ``<!-- /slug -->``, which differ by the ``/``), never the bare slug shared
    by both.

    *doc_path* and *task* are folded into every failure message so a reader
    knows which document and which guard to fix. *delimits* describes what the
    marker pair encloses (e.g. ``"the Tier-B table"``), and tells a reader
    where to restore a deleted or duplicated marker.

    Does NOT check the returned span for emptiness — an empty span is a
    valid (if useless) result here. Each caller's OWN extraction is what is
    positioned to say what "empty" means for its content (no names parsed
    vs. no key-bearing rows), and to raise loudly for that.
    """
    assert begin not in end and end not in begin, (
        f"{begin!r} and {end!r} overlap as substrings, so counting them "
        f"cannot tell one marker from the other ({task}). Pass the full "
        f"HTML-comment forms, not the bare slug."
    )

    begin_count = markdown_text.count(begin)
    assert begin_count == 1, (
        f"expected exactly one {begin!r} marker in {doc_path}, found "
        f"{begin_count} ({task}). This marker opens {delimits}. If it was "
        f"deleted, restore it immediately above; if it was duplicated, one "
        f"of the two copies is unpinned and free to drift."
    )
    end_count = markdown_text.count(end)
    assert end_count == 1, (
        f"expected exactly one {end!r} marker to close {begin!r} in "
        f"{doc_path}, found {end_count} ({task}) — restore the closing "
        f"marker immediately below {delimits}, and above any explanatory "
        f"prose, which is deliberately outside the marker."
    )

    begin_at = markdown_text.index(begin)
    end_at = markdown_text.index(end)
    assert begin_at < end_at, (
        f"the markers are INVERTED in {doc_path}: {end!r} appears before "
        f"{begin!r} ({task}). Swap them back around {delimits} — as written "
        f"they delimit an empty span and this guard would pin nothing."
    )

    return markdown_text[begin_at + len(begin):end_at]
