"""Tests for ``scripts/inline_suppression_scan.py``: what the scan recognises.

One subject: which suppression sites the scan finds in a source, and which
tracked files of a tree it reads to find them.  The kind table in
``scripts/inline_suppression_kinds.py`` is observable only through the scan, so
its behaviour is pinned here rather than in a module of its own.  Fixture trees
come from ``inline_suppression_fixtures``, whose docstring says why they are real
git repositories.
"""

from pathlib import Path

import pytest
from inline_suppression_fixtures import run_git, sites_in, track_fixture_tree
from inline_suppression_kinds import Kind
from inline_suppression_refusal import InstrumentFailure
from inline_suppression_scan import scan_source, scan_tree

# ---------------------------------------------------------------------------
# Layer 1 — the kind table and the COMMENT-token scan.


def test_scan_finds_every_kind_the_table_declares():
    """All five kinds, so a row nobody detects turns this red.

    Driven off ``Kind`` itself rather than a hand-written list: a sixth row
    added to the table (the PRD names ``pytest.mark.skip`` and ``shellcheck
    disable`` as the expected extensions) with no detection pattern would
    otherwise pass a list that never mentioned it.
    """
    source = '\n'.join(
        [
            'a = 1  # type: ignore[arg-type]',
            'b = 2  # noqa: E402',
            'c = 3  # pyright: ignore[reportArgumentType]',
            'd = 4  # pragma: no cover',
            'e = 5  # nosec',
        ]
    )

    assert {site.kind for site in sites_in(source)} == set(Kind)


def test_site_carries_path_line_kind_codes_and_the_stripped_physical_line():
    source = 'x = 1\n\n    # noqa: E402\n'

    (site,) = sites_in(source, path='pkg/mod.py')

    assert site.path == 'pkg/mod.py'
    assert site.line == 3
    assert site.kind is Kind.NOQA
    assert site.codes == ('E402',)
    assert site.text == '# noqa: E402'


def test_bracketed_codes_are_extracted_in_source_order():
    """Order is the SOURCE's here; sorting is the key's job, not the scan's."""
    (mypy_site,) = sites_in('a = 1  # type: ignore[attr-defined, arg-type]')
    (pyright_site,) = sites_in('a = 1  # pyright: ignore[reportArgumentType]')

    assert mypy_site.codes == ('attr-defined', 'arg-type')
    assert pyright_site.codes == ('reportArgumentType',)


def test_noqa_codes_are_extracted_with_or_without_a_space_after_the_colon():
    (listed,) = sites_in('a = 1  # noqa: E402,F401')
    (tight,) = sites_in('a = 1  # noqa:E402')

    assert listed.codes == ('E402', 'F401')
    assert tight.codes == ('E402',)


def test_noqa_is_read_case_insensitively_as_ruff_reads_it():
    """Ruff suppresses on ``# NOQA: F401``, so a site the scan missed would be invisible to the gate."""
    (upper,) = sites_in('import os  # NOQA: F401')
    (mixed,) = sites_in('import os  # NoQa')

    assert (upper.kind, upper.codes) == (Kind.NOQA, ('F401',))
    assert (mixed.kind, mixed.codes) == (Kind.NOQA, ())


def test_type_and_pyright_ignores_stay_case_sensitive_as_their_tools_are():
    assert sites_in('a = 1  # TYPE: IGNORE') == []
    assert sites_in('a = 1  # PYRIGHT: IGNORE') == []


def test_noqa_code_extraction_stops_at_the_first_token_that_is_not_a_code():
    """The live tree's dominant shape: a code, then a dash, then prose.

    Measured over this repository 2026-09-19 — 232 distinct ``noqa`` tails, of
    which the prose-carrying ones (``# noqa: F401  — the binding IS the
    wiring``, ``# noqa: E402  (import after path fix)``) are common enough that
    splitting the whole tail on whitespace manufactures 'the', 'never', 'a' and
    'IS' as rule codes.  Stopping at the first non-code token reproduces ruff's
    own reading and yields 40 distinct codes over the tree, every one of them
    real.
    """
    (dashed,) = sites_in('a = 1  # noqa: F401  — the binding IS the wiring')
    (parenthesised,) = sites_in('a = 1  # noqa: E402  (import after path fix)')

    assert dashed.codes == ('F401',)
    assert parenthesised.codes == ('E402',)


def test_a_kebab_case_code_is_a_code_only_in_first_position():
    """The union of the two consumers' own grammars, and nothing wider.

    ruff recognises a ``<letters><digits>`` code anywhere in the list;
    ``fused-memory/scripts/check_bare_magicmock_config.py`` anchors its
    kebab-case code immediately after ``noqa:``.  Admitting kebab-case
    ANYWHERE would read the tail of ``# noqa: F401  re-export shim`` as a
    second code, which is neither consumer's rule (9 such sites measured over
    this repository, 2026-09-19).
    """
    (first_party,) = sites_in('a = 1  # noqa: bare-magicmock — deliberate')
    (invented,) = sites_in('a = 1  # noqa: bare-something-else')
    (prose,) = sites_in('a = 1  # noqa: F401  re-export shim')

    assert first_party.codes == ('bare-magicmock',)
    assert invented.codes == ('bare-something-else',)
    assert prose.codes == ('F401',)


def test_bare_markers_carry_no_codes():
    """Absence of codes is an empty tuple, never None — a Site always has a
    codes tuple, so no consumer of it needs a null check."""
    bare_noqa, bare_ignore, pragma, nosec = sites_in(
        '\n'.join(
            [
                'a = 1  # noqa',
                'b = 2  # type: ignore',
                'c = 3  # pragma: no cover',
                'd = 4  # nosec',
            ]
        )
    )

    assert bare_noqa.codes == ()
    assert bare_ignore.codes == ()
    assert pragma.codes == ()
    assert nosec.codes == ()


def test_a_suppression_inside_a_string_literal_is_not_a_site():
    """THE REASON THIS IS ``tokenize`` AND NOT A REGEX.

    A regex over source text cannot tell a comment from a string that merely
    mentions one, and this repository is full of the latter — every test that
    asserts on a marker, and this very module.  Same argument as
    ``scripts/merge_lane_metrics.py::_comment_lines``.
    """
    source = '\n'.join(
        [
            'DOC = "# type: ignore[arg-type]"',
            "OTHER = '# noqa: E402'",
            'TRIPLE = """',
            '# pyright: ignore[reportAny]',
            '"""',
        ]
    )

    assert sites_in(source) == []


def test_the_word_nanosecond_does_not_register_as_nosec():
    """``nosec`` is a word, not a substring — 'nanosecond' contains it."""
    assert sites_in('a = 1  # 5 nanoseconds is the budget') == []
    assert sites_in('a = 1  # nanosecond') == []
    assert sites_in('a = 1  # nosecret') == []


def test_one_comment_carrying_a_suppression_and_a_disposition_is_one_site():
    """D6's shape: the disposition rides in the SAME comment token.

    The Site's text is the whole stripped physical line — the disposition
    included — because that is what D7 digests, so editing the disposition
    changes the key just as editing the code does.
    """
    (site,) = sites_in('    value = call()  # type: ignore[attr-defined]  # debt: task 5601')

    assert site.kind is Kind.TYPE_IGNORE
    assert site.codes == ('attr-defined',)
    assert site.text == 'value = call()  # type: ignore[attr-defined]  # debt: task 5601'


def test_one_comment_carrying_two_kinds_yields_one_site_per_kind():
    comment_sites = sites_in('a = 1  # type: ignore[arg-type]  # noqa: E402')

    assert [(site.kind, site.codes) for site in comment_sites] == [
        (Kind.TYPE_IGNORE, ('arg-type',)),
        (Kind.NOQA, ('E402',)),
    ]


def test_scan_source_yields_every_comment_not_only_the_suppressing_ones():
    """The disposition-without-a-suppression violation (scenario 7's second
    half) is a property of a comment that produced NO Site, so the scan has to
    hand back the plain comments too."""
    comments = scan_source(
        'a = 1  # debt: task 5\nb = 2  # noqa: E402\n', path='m.py'
    )

    assert [(comment.line, comment.text, len(comment.sites)) for comment in comments] == [
        (1, '# debt: task 5', 0),
        (2, '# noqa: E402', 1),
    ]


# ---------------------------------------------------------------------------
# Layer 1 — tracked-file enumeration, the byte prefilter, and the failure
# polarity that makes an unreadable file loud instead of invisible.


def test_only_tracked_files_are_scanned(tmp_path: Path):
    """An untracked marker-bearing file is invisible.

    ``git ls-files``, never a filesystem walk: a scratch copy, a gitignored
    watcher artifact or a half-written module must not be able to turn the gate
    red on nothing but local working-tree state.  The cost is stated rather
    than hidden — a new file that has not been ``git add``ed is not scanned
    either, and an author running this locally before staging gets a green
    verdict on content the gate will see once it is staged.
    """
    track_fixture_tree(tmp_path, {'tracked.py': 'a = 1  # noqa: E402\n'})
    (tmp_path / 'untracked.py').write_text('b = 2  # noqa: F401\n', encoding='utf-8')

    scan = scan_tree(tmp_path)

    assert [site.path for site in scan.sites] == ['tracked.py']


def test_scan_counts_the_files_it_enumerated_and_the_files_it_tokenized(tmp_path: Path):
    """THE COUNTED WORK THE ≤10 s BUDGET RESTS ON.

    The prefilter is the whole reason the scan fits the budget, and the only
    honest way to assert it on a loaded box is to count files rather than
    seconds — see
    ``test_inline_suppression_ratchet.py::test_the_live_scan_tokenizes_exactly_the_marker_bearing_files``.
    A file with no marker byte substring
    is ENUMERATED and not TOKENIZED.
    """
    track_fixture_tree(
        tmp_path,
        {
            'marked.py': 'a = 1  # noqa: E402\n',
            'plain.py': 'b = 2\nc = 3\n',
            'also_plain.py': '"""Just a docstring."""\n',
        },
    )

    scan = scan_tree(tmp_path)

    assert scan.files_enumerated == 3
    assert scan.files_tokenized == 1


def test_the_prefilter_admits_a_file_whose_only_marker_is_an_upper_case_noqa(tmp_path: Path):
    track_fixture_tree(tmp_path, {'shouty.py': 'import os  # NOQA: F401\n'})

    scan = scan_tree(tmp_path)

    assert scan.files_tokenized == 1
    assert [(site.kind, site.codes) for site in scan.sites] == [
        (Kind.NOQA, ('F401',))
    ]


def test_a_tracked_file_that_cannot_be_tokenized_is_an_instrument_failure(tmp_path: Path):
    """BOUNDARY SCENARIO 8 — never skipped, and never a violation.

    A file the scanner cannot read is not evidence about that file's
    suppressions; it is evidence that the instrument is broken.  Skipping it
    would make the count read LOW, which is the direction that reports a breach
    as a clean tree (INV-11 ``no-silent-fail-soft``).  Exit 2, and the message
    names the file so the operator knows which one to open.
    """
    track_fixture_tree(tmp_path, {'broken.py': 'value = (  # noqa: E402\n'})

    with pytest.raises(InstrumentFailure) as caught:
        scan_tree(tmp_path)

    assert 'broken.py' in str(caught.value)


def test_a_tracked_file_with_broken_indentation_is_an_instrument_failure(tmp_path: Path):
    """The trio is ``(TokenError, IndentationError, SyntaxError)``, not
    ``TokenError`` alone — a file whose dedent matches no outer level raises
    the second and would otherwise escape as a bare traceback.

    A mismatched DEDENT specifically, not merely a surprising indent: measured
    against this interpreter, ``tokenize`` accepts an over-indented line
    without complaint (the parser rejects it later, but the tokenizer does not),
    so a fixture built on one would assert nothing.
    """
    track_fixture_tree(
        tmp_path, {'bad_indent.py': 'if True:\n      a = 1  # noqa: E402\n   b = 2\n'}
    )

    with pytest.raises(InstrumentFailure) as caught:
        scan_tree(tmp_path)

    assert 'bad_indent.py' in str(caught.value)


def test_a_tracked_file_that_is_not_utf8_is_an_instrument_failure(tmp_path: Path):
    """Decoding is its own step with its own catch, ahead of tokenize."""
    track_fixture_tree(tmp_path, {'placeholder.py': 'a = 1\n'})
    (tmp_path / 'undecodable.py').write_bytes(b'a = 1  # noqa: E402\nb = "\xff\xfe"\n')
    run_git(['add', '-A', '-f'], cwd=tmp_path)

    with pytest.raises(InstrumentFailure) as caught:
        scan_tree(tmp_path)

    assert 'undecodable.py' in str(caught.value)


def test_a_broken_file_with_no_marker_substring_is_never_read(tmp_path: Path):
    """The prefilter's REACH, stated rather than discovered later.

    A file carrying no marker byte substring cannot hold a suppression, so it
    is never decoded and never tokenized — and therefore a syntactically broken
    one is not an instrument failure either.  That is honest: this scanner
    never claimed to parse the tree, only to find its markers, and the bytes it
    did read prove there is nothing here to find.
    """
    track_fixture_tree(tmp_path, {'broken.py': 'value = (\n'})

    scan = scan_tree(tmp_path)

    assert scan.files_enumerated == 1
    assert scan.files_tokenized == 0
    assert scan.sites == ()


def test_enumeration_refuses_rather_than_returning_an_empty_corpus(tmp_path: Path):
    """An empty corpus and a clean corpus are INDISTINGUISHABLE downstream.

    Only one of them is good news, so a failed ``git ls-files`` raises instead
    of degrading to ``[]`` — the argument
    ``scripts/audit_manifest_descriptor_drift.py::ManifestDiscoveryUnavailable``
    makes, and it is sharper here because an empty scan would seed or tighten a
    baseline to nothing and silently open the gate.
    """
    not_a_repo = tmp_path / 'loose'
    not_a_repo.mkdir()

    with pytest.raises(InstrumentFailure):
        scan_tree(not_a_repo)


def test_enumeration_refuses_when_git_cannot_be_run_at_all(tmp_path: Path, monkeypatch):
    """A missing ``git`` is an OSError, not a non-zero return — a refusal that
    only checked the return code would let this one through as a clean scan."""
    track_fixture_tree(tmp_path, {'a.py': 'x = 1  # noqa: E402\n'})
    monkeypatch.setenv('PATH', str(tmp_path / 'empty-bin'))

    with pytest.raises(InstrumentFailure):
        scan_tree(tmp_path)


def test_a_tracked_path_whose_worktree_file_is_gone_is_passed_over(tmp_path: Path):
    """``git ls-files`` reads the INDEX, so it lists a file deleted from the
    worktree.  That is an ordinary mid-edit state, not a broken instrument."""
    track_fixture_tree(tmp_path, {'kept.py': 'a = 1  # noqa: E402\n', 'gone.py': 'b = 2\n'})
    (tmp_path / 'gone.py').unlink()

    scan = scan_tree(tmp_path)

    assert [site.path for site in scan.sites] == ['kept.py']
