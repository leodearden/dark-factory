"""Tests for ``scripts/inline_suppression_key.py``: D7's multiset key.

One subject: the key an unowned site contributes to the ratchet's multiset —
``(kind, sorted codes, digest of the stripped line)`` and no path — and its
rendering.  ``plans/inv12-exceptions-owned-or-ratified-prd.md``, D7.
"""

from collections import Counter
from pathlib import Path

from inline_suppression_fixtures import sites_in
from inline_suppression_key import SuppressionKey, key_for
from inline_suppression_kinds import Kind
from inline_suppressions import Scope, SuppressionClass

# ---------------------------------------------------------------------------
# D7 — the multiset key: (kind, sorted codes, digest of the stripped line),
# and deliberately NO path.


def _keys(source: str, *, path: str = 'm.py') -> list:
    return [key_for(site) for site in sites_in(source, path=path)]


def test_the_same_marker_line_in_two_files_yields_one_key():
    """NO PATH IN THE KEY, which is what makes D7 a multiset and not a
    per-file count."""
    line = 'value = call()  # type: ignore[attr-defined]\n'

    assert _keys(line, path='pkg/a.py') == _keys(line, path='other/deeply/nested/b.py')


def test_renaming_and_splitting_a_file_changes_no_key(tmp_path: Path):
    """BOUNDARY SCENARIO 4 — 40 grandfathered markers survive a rename and a
    split, at the multiset level.

    Under the per-file counts the brief originally proposed, splitting this
    repository's 391-marker file would make every moved marker NEW and
    regenerating the baseline the only exit — the one operation the whole
    ratchet exists to prevent anybody performing casually.
    """
    del tmp_path
    lines = [f'v{index} = call()  # type: ignore[attr-defined]' for index in range(40)]

    before = Counter(_keys('\n'.join(lines) + '\n', path='big.py'))
    after = Counter(
        _keys('\n'.join(lines[:17]) + '\n', path='renamed/part_one.py')
        + _keys('\n'.join(lines[17:]) + '\n', path='renamed/part_two.py')
    )

    assert before == after
    assert sum(before.values()) == 40


def test_editing_the_marker_line_changes_its_key():
    """BOUNDARY SCENARIO 3 — a touch is visible, and editing only the CODE is
    a touch.

    This is D7's whole conversion mechanism: a grandfathered marker stays
    grandfathered until somebody edits the line it sits on, at which point it
    becomes a new key and needs a disposition.
    """
    original = _keys('value = call()  # type: ignore[arg-type]')
    recoded = _keys('value = call()  # type: ignore[attr-defined]')
    renamed = _keys('other = call()  # type: ignore[arg-type]')

    assert original != recoded
    assert original != renamed


def test_editing_an_unrelated_line_of_the_same_statement_does_not():
    """The stated LIMIT of "touched", rather than a hidden one.

    The key digests one physical line, so a multi-line call whose marker rides
    on the closing line is untouched by an edit to its first argument.  D7
    records this as a cost accepted knowingly, so it belongs in a test rather
    than in a reader's discovery.
    """
    before = _keys('value = call(\n    first,\n)  # type: ignore[arg-type]\n')
    after = _keys('value = call(\n    SECOND,\n)  # type: ignore[arg-type]\n')

    assert before == after


def test_reindenting_the_marker_line_does_not_change_its_key():
    """The digest is of the STRIPPED line — moving a statement into an ``if``
    is not an edit to the suppression."""
    flush = _keys('value = call()  # type: ignore[arg-type]')
    indented = _keys('if True:\n        value = call()  # type: ignore[arg-type]\n')

    assert flush == indented


def test_codes_are_sorted_at_construction():
    """``[b, a]`` and ``[a, b]`` are the same suppression.

    Asserted on the constructor rather than on two source lines, because two
    source lines spelling the codes in different orders differ in their DIGEST
    too — which would make the test pass for the wrong reason.
    """
    digest = 'abcdef012345'
    kind = Kind.TYPE_IGNORE

    one = SuppressionKey(kind=kind, codes=('b', 'a'), digest=digest)
    other = SuppressionKey(kind=kind, codes=('a', 'b'), digest=digest)

    assert one == other
    assert one.codes == ('a', 'b')


def test_two_kinds_on_one_line_are_two_distinct_keys():
    """Same digest, different kind — so a line silencing two tools owes two
    dispositions' worth of accounting, not one."""
    first, second = _keys('a = 1  # type: ignore[arg-type]  # noqa: E402')

    assert first != second
    assert first.digest == second.digest
    assert {first.kind, second.kind} == {
        Kind.TYPE_IGNORE,
        Kind.NOQA,
    }


def test_a_rendered_key_is_a_string_that_no_other_key_shares():
    """THE RENDERING CONTRACT ``SuppressionKey.render`` states.

    A string, because JSON object keys are strings by the format's definition.
    Distinct for two keys that differ only in their digest, because the digest
    IS the line's identity and the unowned multiset the ratchet compares is
    keyed by the rendering, so one that dropped it would merge two different
    lines into one count.  And never
    spelled like a ratified class key, so a reader cannot take one for the
    other: the ``@`` that separates a class key's scope is checked present in a
    real ``SuppressionClass`` rendering and absent from this one, so a change to
    the separator on either side breaks this test.
    """
    key = key_for(sites_in('a = 1  # noqa: E402')[0])
    kind = Kind.NOQA
    one = SuppressionKey(kind=kind, codes=('E402',), digest='abcdef012345')
    other = SuppressionKey(kind=kind, codes=('E402',), digest='543210fedcba')
    class_key = SuppressionClass(
        kind=kind, code='E402', scope=Scope.ANY
    )

    assert isinstance(key.render(), str)
    assert one.render() != other.render()
    assert '@' in class_key.render()
    assert '@' not in key.render()
