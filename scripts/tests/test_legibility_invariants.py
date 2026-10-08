"""Tests for scripts/legibility/invariants.py — a project's invariant slugs read
from its design-invariants doc (plans/census-incremental-prd.md D10)."""
from __future__ import annotations

import logging

from legibility import invariants
from legibility.invariants import Invariant

_PLAIN_DOC = """\
# Design invariants

Intro prose.

## INV-1 `a-slug`

**Rule**: something.

## INV-2 `b-slug`

## INV-3 `c-slug`
"""

# Mirrors reify's doc: three families, each numbered from 1.
_REIFY_DOC = """\
# Reify design invariants

## INV-SF-1 `undef-has-provenance`

## INV-SF-2 `units-are-explicit`

## INV-AD-1 `angle-crossings-explicit`

## INV-PD-2 `pose-frames-named`
"""

_DECOY_DOC = """\
# Design invariants

A slug mentioned in prose: `prose-slug`. INV-9 is not a heading.

## INV-1 `a-slug`

### INV-4 `sub-shape`

## INV-5

## INV-sf-1 `lowercase-family`

## INV-6 `trailing-text` and more words

## INV-2 `b-slug`
"""

_NO_HEADINGS_DOC = """\
# Design invariants

INV-1 contracts-machine-checked

### INV-1 `a-slug`
"""


def _project_with_doc(tmp_path, text):
    doc = tmp_path / invariants.DOC_RELPATH
    doc.parent.mkdir(parents=True)
    doc.write_text(text, encoding='utf-8')
    return tmp_path


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING]


# ---------------------------------------------------------------------------
# parse_headings
# ---------------------------------------------------------------------------

def test_plain_headings_parse_in_document_order():
    assert invariants.parse_headings(_PLAIN_DOC) == (
        Invariant(family=None, number=1, slug='a-slug'),
        Invariant(family=None, number=2, slug='b-slug'),
        Invariant(family=None, number=3, slug='c-slug'),
    )


def test_family_prefixed_headings_parse_with_their_family():
    assert invariants.parse_headings(_REIFY_DOC) == (
        Invariant(family='SF', number=1, slug='undef-has-provenance'),
        Invariant(family='SF', number=2, slug='units-are-explicit'),
        Invariant(family='AD', number=1, slug='angle-crossings-explicit'),
        Invariant(family='PD', number=2, slug='pose-frames-named'),
    )


def test_decoy_shapes_are_not_headings():
    assert invariants.parse_headings(_DECOY_DOC) == (
        Invariant(family=None, number=1, slug='a-slug'),
        Invariant(family=None, number=2, slug='b-slug'),
    )


# ---------------------------------------------------------------------------
# read_slugs
# ---------------------------------------------------------------------------

def test_read_slugs_returns_the_doc_slugs_in_order(tmp_path):
    root = _project_with_doc(tmp_path, _PLAIN_DOC)
    assert invariants.read_slugs(root) == ('a-slug', 'b-slug', 'c-slug')


def test_read_slugs_keeps_the_first_of_a_duplicated_slug(tmp_path):
    doc = _PLAIN_DOC + '\n## INV-4 `a-slug`\n\n## INV-5 `d-slug`\n'
    root = _project_with_doc(tmp_path, doc)
    assert invariants.read_slugs(root) == ('a-slug', 'b-slug', 'c-slug', 'd-slug')


def test_read_slugs_reads_a_family_prefixed_doc(tmp_path):
    root = _project_with_doc(tmp_path, _REIFY_DOC)
    assert invariants.read_slugs(str(root)) == (
        'undef-has-provenance',
        'units-are-explicit',
        'angle-crossings-explicit',
        'pose-frames-named',
    )


def test_a_project_without_the_doc_has_no_slugs_and_no_warning(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    assert invariants.read_slugs(tmp_path) == ()
    assert _warnings(caplog) == []


def test_a_doc_with_no_headings_has_no_slugs_and_warns_once(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    root = _project_with_doc(tmp_path, _NO_HEADINGS_DOC)

    assert invariants.read_slugs(root) == ()

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert str(root / invariants.DOC_RELPATH) in warnings[0].getMessage()
