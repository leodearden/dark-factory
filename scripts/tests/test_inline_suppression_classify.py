"""Tests for ``scripts/inline_suppression_classify.py``: ownership under a class table.

One subject: how each suppression site is owned — by an inline disposition, by
a ratified class, or not at all — and which disposition faults the
classification reports.  ``plans/inv12-exceptions-owned-or-ratified-prd.md``,
D6 and D9, boundary scenarios 2, 7 and 10.  No test patches a module global:
each states the class table it classifies against, empty unless it builds a
real ``{row: Policy(...)}`` one.
"""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from inline_suppression_classify import Ownership, Scope, SuppressionClass, classify
from inline_suppression_consumers import ConsumerModel
from inline_suppression_fixtures import RUFF_CONFIG, track_fixture_tree
from inline_suppression_kinds import Kind
from inline_suppression_scan import scan_tree
from shared.governed_exceptions import INLINE_MARKER_FORMS, Policy

# ---------------------------------------------------------------------------
# Layer 3 — classification: owned, ratified by class, or unowned.


def _classify(
    tmp_path: Path,
    files: Mapping[str, str],
    *,
    classes: Mapping[SuppressionClass, Policy] = MappingProxyType({}),
):
    """Scan and classify a fixture tree against *classes* in one step."""
    track_fixture_tree(tmp_path, files)
    scan = scan_tree(tmp_path)
    return classify(
        scan, ConsumerModel(tmp_path), classes=classes
    )


def test_a_site_with_a_well_formed_disposition_is_owned(tmp_path: Path):
    """BOUNDARY SCENARIO 2 — all three of D6's forms, and none contributes a
    key to the unowned multiset."""
    result = _classify(
        tmp_path,
        {
            'pyproject.toml': RUFF_CONFIG,
            'm.py': (
                'a = 1  # noqa: E402  # debt: task 5601\n'
                'b = 2  # noqa: F401  # debt: ticket tkt_0RTCC80EM92A7WD08D6RF6ZZPY\n'
                'c = 3  # noqa: B006  # ratified: inv12-day-one-test-doubles\n'
            ),
        },
    )

    assert result.violations == ()
    assert result.counts == {}
    assert [entry.ownership for entry in result.classified] == [
        Ownership.DEBT,
        Ownership.DEBT,
        Ownership.POLICY,
    ]


def test_a_site_with_no_disposition_is_unowned_and_contributes_a_key(tmp_path: Path):
    result = _classify(
        tmp_path, {'pyproject.toml': RUFF_CONFIG, 'm.py': 'a = 1  # noqa: E402\n'}
    )

    assert [entry.ownership for entry in result.classified] == [
        Ownership.UNOWNED
    ]
    assert sum(result.counts.values()) == 1


def test_a_marker_that_does_not_parse_is_a_violation_naming_its_comment(tmp_path: Path):
    """BOUNDARY SCENARIO 7, first half.

    Present-but-broken is the case worth being loud about: the author plainly
    meant to disposition something, and the entry is silently undisposed until
    somebody is told.  Exit 1 and not 2 — it is a fault at the SITE, fixed by
    an agent, which is what ``MalformedDisposition``'s own docstring records.
    """
    result = _classify(
        tmp_path,
        {'pyproject.toml': RUFF_CONFIG, 'm.py': 'a = 1  # noqa: E402  # debt: soon\n'},
    )

    (violation,) = result.violations
    assert violation.path == 'm.py'
    assert violation.line == 1
    rendered = violation.render()
    for form in INLINE_MARKER_FORMS:
        assert form in rendered


def test_a_disposition_on_a_line_with_no_suppression_is_a_violation(tmp_path: Path):
    """BOUNDARY SCENARIO 7, second half — and it CANNOT come from the parser.

    ``parse_disposition_marker`` returns a perfectly valid ``Debt(TaskRef(5))``
    here; its docstring says so outright and delegates this violation to the
    scanner, because deciding it needs the kind table.  So the detection is
    "this comment yielded a Disposition and no Site", never a parser outcome.

    THE PREFILTER'S REACH BOUNDS THIS FINDING, and the bound is asserted rather
    than left to be discovered.  A file carrying no marker byte substring at
    all is never decoded, so a stray disposition alone in such a file is
    invisible.  That is the honest limit and it lands on the harmless side: in
    a file with no suppressions there is nothing anywhere for the marker to
    answer for, so it is inert prose.  The dangerous case — an author who
    believes a REAL marker in this file is now dispositioned when it is not —
    is exactly the case that IS caught, because that file carries a marker.
    Widening the prefilter to the disposition keywords would close the gap only
    by putting a second copy of D6's grammar in the scanner, which is the one
    thing ``shared.governed_exceptions`` exists to prevent.
    """
    result = _classify(
        tmp_path,
        {
            'pyproject.toml': RUFF_CONFIG,
            'm.py': 'a = 1  # noqa: E402  # debt: task 5601\nx = 2  # debt: task 5\n',
            'markerless.py': 'y = 3  # debt: task 7\n',
        },
    )

    (violation,) = result.violations
    assert violation.path == 'm.py'
    assert violation.line == 2
    assert result.counts == {}


def test_a_site_matching_a_ratified_class_is_policy_by_reference(tmp_path: Path):
    """BOUNDARY SCENARIO 10 — outside the unowned multiset, and COUNTED under
    its class so blanket policy stays visible in the report."""
    row = SuppressionClass(
        kind=Kind.NOQA,
        code='E402',
        scope=Scope.ANY,
    )

    result = _classify(
        tmp_path,
        {'pyproject.toml': RUFF_CONFIG, 'm.py': 'a = 1  # noqa: E402\n'},
        classes={row: Policy('inv12-day-one')},
    )

    (entry,) = result.classified
    assert entry.ownership is Ownership.CLASS
    assert entry.suppression_class == row
    assert result.counts == {}


def test_class_scope_matches_src_tests_or_any(tmp_path: Path):
    """A path is ``tests`` iff one of its COMPONENTS is ``tests``.

    Verified complete for this repository: every tracked test module lives
    under a ``tests`` component, so the rule needs no filename pattern beside
    it.  ``src`` is then simply "not tests", which keeps the two scopes a
    partition rather than two independent predicates that could both miss.
    """
    files = {
        'pyproject.toml': RUFF_CONFIG,
        'pkg/mod.py': 'a = 1  # noqa: E402\n',
        'pkg/tests/test_mod.py': 'b = 2  # noqa: E402\n',
    }
    for scope, expected in (
        (Scope.ANY, {'pkg/mod.py', 'pkg/tests/test_mod.py'}),
        (Scope.SRC, {'pkg/mod.py'}),
        (Scope.TESTS, {'pkg/tests/test_mod.py'}),
    ):
        row = SuppressionClass(
            kind=Kind.NOQA, code='E402', scope=scope
        )
        tree = tmp_path / scope.value
        tree.mkdir()
        result = _classify(tree, files, classes={row: Policy('inv12-day-one')})

        covered = {
            entry.site.path
            for entry in result.classified
            if entry.ownership is Ownership.CLASS
        }
        assert covered == expected, scope


def test_a_suppression_class_renders_d9s_published_key():
    row = SuppressionClass(
        kind=Kind.NOQA,
        code='E402',
        scope=Scope.TESTS,
    )

    assert row.render() == 'noqa[E402]@tests'
