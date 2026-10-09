"""Boundary tests for ``scripts/inline_suppressions.py`` — task 5601, PRD scenarios 1-11.

WHAT IS UNDER TEST.  The inline-suppression scanner: D9's ratified classes and
the ``--check`` / ``--seed`` / ``--tighten`` / ``--json`` verbs with the 0/1/2
exit ladder the PRD Contract fixes.
``plans/inv12-exceptions-owned-or-ratified-prd.md``, boundary-test sketch rows
1-11.  What the scan recognises, D7's key and D8's consumer model are the
subjects of ``test_inline_suppression_scan.py``, ``test_inline_suppression_key.py``
and ``test_inline_suppression_consumers.py``.

HOW IT IS TESTED.  Almost every test builds a throwaway git repository under
``tmp_path`` and calls ``inline_suppressions.main([...])`` IN-PROCESS,
capturing stdout/stderr with ``capsys``: the verbs' whole contract is an exit
code plus rendered lines, and an in-process call gets both without paying a
subprocess per scenario.  The one exception is deliberate: the import-failure
test (Contract: "an ImportError is 2, never 1") needs a real interpreter,
because the fault it pins happens before ``main`` exists.

Fixture trees come from ``inline_suppression_fixtures``, whose docstring says
why they are real git repositories and how they stay xdist-safe.

NO WALL-CLOCK ASSERTION APPEARS IN THIS MODULE, on purpose: the PRD's ≤10 s
scan budget is pinned as COUNTED WORK, and
:func:`test_the_live_scan_tokenizes_exactly_the_marker_bearing_files` says why.
Subprocesses therefore get a generous ``timeout=`` rather than a clock guard.
"""

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import inline_suppressions
import pytest
from inline_suppression_fixtures import (
    RUFF_CONFIG,
    SUBPROCESS_TIMEOUT_SECS,
    run_git,
    track_fixture_tree,
    write_files,
)
from inline_suppression_kinds import KIND_SPECS, Kind
from inline_suppression_scan import scan_tree
from shared.governed_exceptions import INLINE_MARKER_FORMS, Policy
from shared.ratchet import BASELINE_README, SCHEMA_VERSION, load

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / 'scripts' / 'inline_suppressions.py'

#: Where a fixture tree's baseline lives.  The real one is
#: ``scripts/inline_suppression_baseline.json`` and is task 5607's to seed; fixtures
#: keep theirs inside ``tmp_path`` so no test can reach the committed file.
_BASELINE_NAME = 'inline_suppression_baseline.json'


def _write_fixture_tree(
    root: Path, files: Mapping[str, str], *, baseline: bool = False
) -> Path:
    """Track *files* in a git repo at *root*, and return its baseline path.

    The tree itself is :func:`inline_suppression_fixtures.track_fixture_tree`'s.
    With ``baseline=True`` the scanner's own ``--seed`` verb seeds the returned
    path from this very tree, so a fixture baseline can never drift from the key
    format the scanner emits.  The path is returned either way: the tests that
    need an ABSENT baseline, or a deliberately corrupt one, need somewhere to
    point ``--baseline`` at just as much as the seeded ones do.
    """
    track_fixture_tree(root, files)

    baseline_path = root / _BASELINE_NAME
    if baseline:
        seeded = inline_suppressions.main(
            ['--seed', '--root', str(root), '--baseline', str(baseline_path)]
        )
        assert seeded == 0, f'fixture --seed failed with exit {seeded}'
    return baseline_path


def _check(
    root: Path,
    baseline_path: Path,
    *paths: str,
    classes: Mapping[inline_suppressions.SuppressionClass, Policy] = (
        inline_suppressions.RATIFIED_SUPPRESSION_CLASSES
    ),
) -> int:
    """Run ``--check`` over *root* against *classes*, optionally scoped to *paths*."""
    return inline_suppressions.main(
        ['--check', '--root', str(root), '--baseline', str(baseline_path), *paths],
        classes=classes,
    )


def _python_env_without_shared() -> dict[str, str]:
    """An environment in which ``import shared`` cannot resolve.

    ``PYTHONPATH`` is emptied and ``PYTHONNOUSERSITE`` is set — and NEITHER IS
    SUFFICIENT, which is why :data:`_NO_SITE` exists beside this.  Measured from
    ``/tmp`` with both applied: ``import shared`` still resolved, to
    ``<worktree>/shared/src/shared/__init__.py``, because the workspace member is
    installed EDITABLE and the ``.pth`` entry that does it lives in the venv's own
    ``site-packages``, which neither knob touches.
    """
    env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
    env['PYTHONNOUSERSITE'] = '1'
    return env


#: The interpreter flag that actually hides an editable install: ``-S`` skips
#: site processing altogether, so no ``.pth`` file is read.  Measured: the same
#: subprocess is rc=0 ``RESOLVED`` without it and rc=1
#: ``ModuleNotFoundError: No module named 'shared'`` with it.  The scanner only
#: ever imports stdlib plus ``shared``, so nothing else is lost.
_NO_SITE = ('-S',)


def _run_script(
    args: list[str],
    *,
    cwd: Path,
    script: Path,
    env: dict[str, str] | None = None,
    flags: tuple[str, ...] = (),
):
    """Run *script* as a real subprocess with ``sys.executable`` and *flags*."""
    return subprocess.run(
        [sys.executable, *flags, str(script), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECS,
        env=env,
    )


# ---------------------------------------------------------------------------
# Layer 3 — classification: owned, ratified by class, or unowned.


def _classify(
    tmp_path: Path,
    files: Mapping[str, str],
    *,
    classes: Mapping[inline_suppressions.SuppressionClass, Policy] = MappingProxyType({}),
):
    """Scan and classify a fixture tree against *classes* in one step."""
    _write_fixture_tree(tmp_path, files)
    scan = scan_tree(tmp_path)
    return inline_suppressions.classify(
        scan, inline_suppressions.ConsumerModel(tmp_path), classes=classes
    )


def test_the_shipped_class_table_is_empty(tmp_path: Path):
    """D9 — the valve is the OPERATOR's, and it ships shut.

    The rows are ruled by task 5603 and applied by task 5609; an implementer adding one here
    would be ratifying a blanket exception on the operator's behalf, which is
    the one thing D9 reserves.
    """
    del tmp_path
    assert dict(inline_suppressions.RATIFIED_SUPPRESSION_CLASSES) == {}


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
        inline_suppressions.Ownership.DEBT,
        inline_suppressions.Ownership.DEBT,
        inline_suppressions.Ownership.POLICY,
    ]


def test_a_site_with_no_disposition_is_unowned_and_contributes_a_key(tmp_path: Path):
    result = _classify(
        tmp_path, {'pyproject.toml': RUFF_CONFIG, 'm.py': 'a = 1  # noqa: E402\n'}
    )

    assert [entry.ownership for entry in result.classified] == [
        inline_suppressions.Ownership.UNOWNED
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
    row = inline_suppressions.SuppressionClass(
        kind=Kind.NOQA,
        code='E402',
        scope=inline_suppressions.Scope.ANY,
    )

    result = _classify(
        tmp_path,
        {'pyproject.toml': RUFF_CONFIG, 'm.py': 'a = 1  # noqa: E402\n'},
        classes={row: Policy('inv12-day-one')},
    )

    (entry,) = result.classified
    assert entry.ownership is inline_suppressions.Ownership.CLASS
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
        (inline_suppressions.Scope.ANY, {'pkg/mod.py', 'pkg/tests/test_mod.py'}),
        (inline_suppressions.Scope.SRC, {'pkg/mod.py'}),
        (inline_suppressions.Scope.TESTS, {'pkg/tests/test_mod.py'}),
    ):
        row = inline_suppressions.SuppressionClass(
            kind=Kind.NOQA, code='E402', scope=scope
        )
        tree = tmp_path / scope.value
        tree.mkdir()
        result = _classify(tree, files, classes={row: Policy('inv12-day-one')})

        covered = {
            entry.site.path
            for entry in result.classified
            if entry.ownership is inline_suppressions.Ownership.CLASS
        }
        assert covered == expected, scope


def test_a_suppression_class_renders_d9s_published_key():
    row = inline_suppressions.SuppressionClass(
        kind=Kind.NOQA,
        code='E402',
        scope=inline_suppressions.Scope.TESTS,
    )

    assert row.render() == 'noqa[E402]@tests'


# ---------------------------------------------------------------------------
# Layer 4 — the ratchet and the verbs.


def _revise(root: Path, files: Mapping[str, str], *, removing: tuple[str, ...] = ()) -> None:
    """Evolve an already-tracked fixture tree and re-stage the result.

    The ratchet's whole subject matter is a tree that CHANGED after a baseline
    was seeded, so every scenario from here down needs a second edit against a
    live index.  ``git add -A`` stages deletions too, which is what makes the
    rename half of scenario 4 a real rename rather than a copy.
    """
    for relative in removing:
        (root / relative).unlink()
    write_files(root, files)
    run_git(['add', '-A', '-f'], cwd=root)


def _grandfathered(count: int) -> str:
    """*count* distinct lines, each carrying one undisposed suppression.

    Distinct rather than identical on purpose: identical lines collapse to one
    D7 key with a multiplicity, which is a different scenario (5 and 9) from
    the many-keys-survive-a-rename one this feeds.
    """
    return ''.join(f'a{index} = {index}  # type: ignore[arg-type]\n' for index in range(count))


def test_a_new_undisposed_suppression_is_one_violation_line(tmp_path: Path, capsys):
    """BOUNDARY SCENARIO 1 — the gate's headline case.

    The line has to carry everything an agent needs to act without opening the
    PRD: where it is, what kind it is, its codes, why it is a finding, and how
    to spell the fix.  The accepted forms are asserted against
    ``INLINE_MARKER_FORMS`` itself rather than against retyped strings, so a
    later edit to the published grammar cannot leave this message behind.
    """
    baseline = _write_fixture_tree(tmp_path, {'m.py': 'a = 1\n'}, baseline=True)
    _revise(tmp_path, {'m.py': 'a = 1  # type: ignore[arg-type]\n'})
    capsys.readouterr()

    assert _check(tmp_path, baseline) == 1

    (violation,) = capsys.readouterr().err.strip().splitlines()
    assert violation.startswith('m.py:1:')
    assert 'type: ignore' in violation
    assert 'arg-type' in violation
    for form in INLINE_MARKER_FORMS:
        assert form in violation


def test_the_same_suppression_with_a_disposition_is_green(tmp_path: Path):
    """BOUNDARY SCENARIO 2 — the escape hatch the gate exists to push authors
    towards, asserted on the very marker scenario 1 rejects."""
    baseline = _write_fixture_tree(tmp_path, {'m.py': 'a = 1\n'}, baseline=True)
    _revise(tmp_path, {'m.py': 'a = 1  # type: ignore[arg-type]  # debt: task 5601\n'})

    assert _check(tmp_path, baseline) == 0


def test_editing_a_grandfathered_markers_code_is_a_new_violation(tmp_path: Path, capsys):
    """BOUNDARY SCENARIO 3 — D7's conversion mechanism, at the gate.

    The marker is not new and the file is not new; only the CODE on the line
    changed, which is exactly the edit that ought to make an author own what
    they are silencing.  The digest is of the whole stripped line, so the old
    key stops being claimed and the new one is in excess.
    """
    baseline = _write_fixture_tree(
        tmp_path, {'m.py': 'a = 1  # type: ignore[arg-type]\n'}, baseline=True
    )
    _revise(tmp_path, {'m.py': 'a = 1  # type: ignore[attr-defined]\n'})
    capsys.readouterr()

    assert _check(tmp_path, baseline) == 1
    assert 'attr-defined' in capsys.readouterr().err


def test_renaming_and_splitting_a_file_of_grandfathered_markers_is_green(tmp_path: Path):
    """BOUNDARY SCENARIO 4 — the pathless key, end to end.

    Forty grandfathered markers move to two new filenames and the gate does not
    notice, because no key mentions a path.  This is the property that keeps an
    ordinary refactor out of the baseline: a key set that moved with the code
    needs no diff to the committed file at all.
    """
    lines = _grandfathered(40).splitlines(keepends=True)
    baseline = _write_fixture_tree(tmp_path, {'big.py': ''.join(lines)}, baseline=True)
    _revise(
        tmp_path,
        {'moved/first.py': ''.join(lines[:17]), 'moved/second.py': ''.join(lines[17:])},
        removing=('big.py',),
    )

    assert _check(tmp_path, baseline) == 0


def test_several_new_suppressions_print_one_violation_each(tmp_path: Path, capsys):
    baseline = _write_fixture_tree(tmp_path, {'m.py': 'a = 1\n'}, baseline=True)
    _revise(
        tmp_path,
        {
            'm.py': 'a = 1  # type: ignore[arg-type]\n',
            'n.py': 'b = 2  # pyright: ignore[reportArgumentType]\n',
        },
    )
    capsys.readouterr()

    assert _check(tmp_path, baseline) == 1

    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 2
    assert {line.split(':')[0] for line in lines} == {'m.py', 'n.py'}


def test_a_clean_whole_tree_run_labels_its_green_clean(tmp_path: Path, capsys):
    """Three greens exist and they are not interchangeable, so the report says
    which one this is.  ``clean`` is the only one that means the whole tree was
    measured against a real baseline and nothing was in excess."""
    baseline = _write_fixture_tree(
        tmp_path, {'m.py': 'a = 1  # type: ignore[arg-type]\n'}, baseline=True
    )
    capsys.readouterr()

    assert _check(tmp_path, baseline) == 0

    report = capsys.readouterr().out
    assert 'clean' in report
    assert 'partial' not in report


def _dead_marker_violation(
    tmp_path: Path,
    capsys,
    source: str,
    *,
    classes: Mapping[inline_suppressions.SuppressionClass, Policy] = (
        inline_suppressions.RATIFIED_SUPPRESSION_CLASSES
    ),
) -> str:
    """Seed a marker-free tree, add *source*, and return its one violation line."""
    baseline = _write_fixture_tree(
        tmp_path, {'pyproject.toml': RUFF_CONFIG, 'm.py': 'a = 1\n'}, baseline=True
    )
    _revise(tmp_path, {'m.py': source})
    capsys.readouterr()

    assert _check(tmp_path, baseline, classes=classes) == 1

    (violation,) = capsys.readouterr().err.strip().splitlines()
    return violation


@pytest.mark.parametrize(
    'source',
    [
        'a = 1  # noqa: PLC0415\n',
        'a = 1  # pragma: no cover\n',
    ],
)
def test_a_new_marker_no_tool_reads_is_rejected_for_deletion(
    tmp_path: Path, capsys, source: str
):
    """BOUNDARY SCENARIO 9 — D8's rejection arm.

    Both halves of the population D8 names: a ``noqa`` whose code the nearest
    config does not select (PLC0415 is the tree's largest such inflow) and a kind
    no configured tool reads at all.  The remedy is DELETION, so the line says
    so and publishes no disposition forms — offering an author a fix that does
    not work is worse than offering none.
    """
    violation = _dead_marker_violation(tmp_path, capsys, source)

    assert 'delete this marker' in violation
    assert 'no tool reads it' in violation
    for form in INLINE_MARKER_FORMS:
        assert form not in violation


@pytest.mark.parametrize(
    'source',
    [
        'a = 1  # noqa: PLC0415  # debt: task 5601\n',
        'a = 1  # pragma: no cover  # debt: task 5601\n',
    ],
)
def test_a_disposition_does_not_rescue_a_marker_no_tool_reads(
    tmp_path: Path, capsys, source: str
):
    """BOUNDARY SCENARIO 9's sharp edge — D8 accepts NO disposition.

    The exit code and the reason are both unchanged from the undispositioned
    case, which is the whole content of the claim: a dead marker is not debt to
    be owned, it is a line to be removed, and an author who dispositions one has
    answered a question nobody asked.  This falls out of the fixed
    consumer-first classification order rather than from a special case.
    """
    violation = _dead_marker_violation(tmp_path, capsys, source)

    assert 'delete this marker' in violation
    assert 'no tool reads it' in violation


def test_a_ratified_class_does_not_rescue_a_marker_no_tool_reads(tmp_path: Path, capsys):
    """BOUNDARY SCENARIO 9 against D9's valve, which is the other thing that
    could plausibly rescue a site and equally does not.

    The operator's valve rules on suppressions a tool HONOURS; a row matching a
    marker nothing reads would ratify a no-op, so the consumer check running
    first makes the row inert rather than making it a widening.
    """
    row = inline_suppressions.SuppressionClass(
        kind=Kind.NOQA,
        code='PLC0415',
        scope=inline_suppressions.Scope.ANY,
    )

    violation = _dead_marker_violation(
        tmp_path, capsys, 'a = 1  # noqa: PLC0415\n', classes={row: Policy('inv12-day-one')}
    )

    assert 'no tool reads it' in violation


def test_a_grandfathered_dead_marker_is_green_and_still_counted(
    tmp_path: Path, capsys
):
    """D8's "grandfathered dead markers stay counted" — the complement that
    keeps the gate a RATCHET rather than a sweep.

    Rejecting every dead marker outright would red the whole tree on the day the
    gate lands, so the baseline holds the existing ones.  What must NOT happen is
    that they vanish from the report: the population is the thing D11's sweep
    tightens, and a number nobody can see never shrinks.
    """
    baseline = _write_fixture_tree(
        tmp_path,
        {'pyproject.toml': RUFF_CONFIG, 'm.py': 'a = 1  # noqa: PLC0415\n'},
        baseline=True,
    )
    capsys.readouterr()

    assert _check(tmp_path, baseline) == 0

    report = capsys.readouterr().out
    assert '1 unowned in 1 key' in report


def test_a_second_copy_of_a_grandfathered_dead_line_is_in_excess(tmp_path: Path, capsys):
    """The multiset is a multiset: the baseline permits ONE of that line.

    Identical lines share one D7 key, so the only thing separating the
    grandfathered copy from a new one is multiplicity — which is exactly what
    ``excess`` measures, and why the key carries a count rather than a flag.
    """
    baseline = _write_fixture_tree(
        tmp_path,
        {'pyproject.toml': RUFF_CONFIG, 'm.py': 'a = 1  # noqa: PLC0415\n'},
        baseline=True,
    )
    _revise(tmp_path, {'m.py': 'a = 1  # noqa: PLC0415\na = 1  # noqa: PLC0415\n'})
    capsys.readouterr()

    assert _check(tmp_path, baseline) == 1

    lines = capsys.readouterr().err.strip().splitlines()
    assert [line.split(':')[1] for line in lines] == ['1', '2']


def test_with_no_baseline_yet_a_whole_tree_of_undisposed_markers_is_advisory(
    tmp_path: Path, capsys
):
    """BOUNDARY SCENARIO 11 — the unseeded state, which is every tree until
    task 5607's cutover lands.

    D12 makes baseline ABSENCE a legitimate state, so the gate is green and says
    it is enforcing nothing.  The label has to be its own word rather than a
    silent ``clean``: a reader who sees ``clean`` over a tree of undisposed
    markers concludes the scanner is broken, and a reader who sees nothing at all
    concludes the gate is live when it is not.  The message names the step that
    seeds the baseline, so the next question is answered in the same line.
    """
    baseline = _write_fixture_tree(tmp_path, {'m.py': _grandfathered(5)})
    capsys.readouterr()

    assert not baseline.exists()
    assert _check(tmp_path, baseline) == 0

    report = capsys.readouterr().out
    assert 'advisory' in report
    assert '--seed' in report


def test_a_scoped_check_over_a_clean_scope_is_partial_and_ignores_the_rest(
    tmp_path: Path, capsys
):
    """The label exists because a scoped green is a WEAKER claim, and the one
    place that matters is the finding it did not look for.

    Scoped ``--check`` is sound in one direction only, which is why D12 keeps it:
    dropping sites can lower a key's current count and so only ever UNDER-reports
    excess — it can never manufacture a violation.  Slack is the mirror and is
    therefore UNSOUND from a partial view (every unscanned baseline key reads as
    headroom), so the report declines to put a number on it rather than printing
    one that is wrong.
    """
    baseline = _write_fixture_tree(
        tmp_path, {'pkg/kept.py': 'a = 1  # type: ignore[arg-type]\n'}, baseline=True
    )
    _revise(tmp_path, {'other/fresh.py': 'b = 2  # type: ignore[attr-defined]\n'})
    capsys.readouterr()

    assert _check(tmp_path, baseline, 'pkg') == 0

    report = capsys.readouterr()
    assert 'partial' in report.out
    assert 'clean' not in report.out
    assert 'slack n/a' in report.out
    assert report.err == ''

    assert _check(tmp_path, baseline) == 1


def test_a_scope_that_matches_no_tracked_file_is_refused(tmp_path: Path, capsys):
    """The one way a scan reaches an empty corpus with nothing broken at all: a
    mistyped path on the documented early-feedback run.

    The corpus refusal one step earlier states the argument — an empty scan and
    a clean scan are the same violation count, and only one of them is good news
    — and narrowing a healthy corpus to nothing lands in exactly that state.  The
    report line does print ``0 of 0``, which is a tell only a reader who looks
    will catch; this module chooses the refusal over the tell.
    """
    baseline = _write_fixture_tree(
        tmp_path, {'pkg/m.py': 'a = 1  # type: ignore[arg-type]\n'}, baseline=True
    )
    capsys.readouterr()

    assert _check(tmp_path, baseline, 'pkg_typo') == 2

    report = capsys.readouterr()
    assert 'pkg_typo' in report.err
    assert 'clean' not in report.out
    assert 'partial' not in report.out


@pytest.mark.parametrize('damage', ['truncated', 'not-json', 'wrong-schema'])
def test_a_baseline_that_exists_but_cannot_be_read_is_never_green(
    tmp_path: Path, capsys, damage: str
):
    """THE ANTI-FAIL-SOFT CASE, and the whole reason absence is detected by an
    explicit existence check instead of by catching ``BaselineUnusable``.

    ``shared.ratchet.load`` collapses absent, undecodable, unparseable, misshapen
    and wrong-schema into ONE refusal, because to the kernel's callers they mean
    one thing.  This consumer is the one place where they do not: absence is a
    legitimate unseeded state and everything else is a broken instrument.  Reaching
    the advisory path by catching that refusal would report a corrupt baseline as
    a clean tree — exactly the silent fail-soft the kernel's own docstring says an
    empty baseline causes (INV-11).
    """
    baseline = _write_fixture_tree(
        tmp_path, {'m.py': 'a = 1  # type: ignore[arg-type]\n'}, baseline=True
    )
    seeded = baseline.read_text(encoding='utf-8')
    if damage == 'truncated':
        baseline.write_text(seeded[: len(seeded) // 2], encoding='utf-8')
    elif damage == 'not-json':
        baseline.write_text('this is not a baseline\n', encoding='utf-8')
    else:
        baseline.write_text(seeded.replace('"schema_version": 1', '"schema_version": 99'))
    capsys.readouterr()

    assert _check(tmp_path, baseline) == 2

    report = capsys.readouterr()
    assert 'advisory' not in report.out
    assert 'clean' not in report.out
    assert str(baseline) in report.err


def test_a_baseline_measured_under_other_params_is_refused_by_every_verb(
    tmp_path: Path, capsys
):
    """The kernel's OTHER reachable refusal, and structurally unlike the three
    damage cases above: this file is well-formed and loads cleanly, and the
    refusal arrives later, out of ``excess``/``tighten``.

    The params block is what makes two baselines two measurements of the same
    thing, so changing the scanned kinds or the digest width makes every count a
    count of something else — and the only way out is the re-seed the verbs
    exist to stop anyone performing casually.  ``--tighten`` matters most here:
    rewriting the file under the new params WOULD be that re-seed, performed by
    accident and reported as a forward click, so the bytes are asserted
    unchanged.  It is also the branch the next edit to ``KIND_SPECS`` or the
    digest width will actually hit.
    """
    baseline = _write_fixture_tree(tmp_path, {'m.py': _grandfathered(2)}, baseline=True)
    raw = json.loads(baseline.read_text(encoding='utf-8'))
    raw['params']['digest_hex'] = 10
    baseline.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding='utf-8')
    unchanged = baseline.read_bytes()
    capsys.readouterr()

    for verb in ('--check', '--json', '--tighten'):
        code = inline_suppressions.main(
            [verb, '--root', str(tmp_path), '--baseline', str(baseline)]
        )
        report = capsys.readouterr()

        assert code == 2, (verb, report.out, report.err)
        assert 'digest_hex' in report.err, verb
        assert report.out == '', verb

    assert baseline.read_bytes() == unchanged


def _seed(root: Path, baseline_path: Path, *paths: str) -> int:
    """Run ``--seed`` over *root*, optionally scoped to *paths*."""
    return inline_suppressions.main(
        ['--seed', '--root', str(root), '--baseline', str(baseline_path), *paths]
    )


def test_seed_writes_the_unowned_multiset_under_the_kernels_preamble(tmp_path: Path):
    """``--seed`` is task 5607's one call, so what it writes has to be reviewable.

    The kernel owns the file's shape and its preamble — the paragraph stating
    that the only legal diff is a DELETION is re-emitted on every write, which is
    the only handle a reviewer has on a file keyed by content digests.  What this
    scanner owns is the CONTENT: exactly the unowned multiset, and a params block
    naming the kinds it swept and the digest width D7 fixes.
    """
    files = {
        'pyproject.toml': RUFF_CONFIG,
        'm.py': _grandfathered(3) + 'b = 2  # noqa: E402  # debt: task 5601\n',
    }
    baseline = _write_fixture_tree(tmp_path, files)

    assert _seed(tmp_path, baseline) == 0

    raw = json.loads(baseline.read_text(encoding='utf-8'))
    assert raw['_README'] == BASELINE_README
    assert raw['schema_version'] == SCHEMA_VERSION
    written = load(baseline)
    scan = scan_tree(tmp_path)
    expected = inline_suppressions.classify(
        scan,
        inline_suppressions.ConsumerModel(tmp_path),
        classes=inline_suppressions.RATIFIED_SUPPRESSION_CLASSES,
    ).counts
    assert dict(written.counts) == expected
    assert len(written.counts) == 3
    assert set(written.params) == {'kinds', 'key_scheme', 'digest_hex'}
    assert written.params['kinds'] == tuple(kind.value for kind in Kind)
    assert written.params['digest_hex'] == 12
    assert written.complete is True


def test_a_baseline_just_seeded_makes_the_same_tree_clean(tmp_path: Path, capsys):
    """The round trip, which is the only thing that proves the two halves agree.

    A key the seed writes and a key the check computes are produced by the same
    code, so equality is unsurprising; what this catches is a params block or a
    schema version that differs between the write and the read, which turns every
    later run into an exit 2 nobody can explain.
    """
    baseline = _write_fixture_tree(tmp_path, {'m.py': _grandfathered(4)})
    assert _seed(tmp_path, baseline) == 0
    capsys.readouterr()

    assert _check(tmp_path, baseline) == 0
    assert 'clean' in capsys.readouterr().out


def test_seeding_over_an_existing_baseline_is_refused_and_changes_nothing(
    tmp_path: Path, capsys
):
    """BOUNDARY SCENARIO 6 — and it is asserted on BYTES, not on mtime.

    ``shared.ratchet.dump`` is deliberately unpoliced against whatever already
    sits at its path: seeding a new baseline and carrying an honestly incomplete
    one across a file boundary are both legitimate, and neither survives a writer
    that refuses unfamiliar keys.  So regenerating an existing baseline from a
    fresh scan is the one call that widens the gate, and closing that hole is this
    consumer's job rather than the kernel's.
    """
    baseline = _write_fixture_tree(
        tmp_path, {'m.py': 'a = 1  # type: ignore[arg-type]\n'}, baseline=True
    )
    before = baseline.read_bytes()
    _revise(tmp_path, {'m.py': 'a = 1  # type: ignore[arg-type]\nb = 2  # nosec\n'})
    capsys.readouterr()

    assert _seed(tmp_path, baseline) == 2

    assert baseline.read_bytes() == before
    assert str(baseline) in capsys.readouterr().err


def test_a_scoped_seed_is_refused_before_any_scan_work(tmp_path: Path, capsys):
    """A baseline seeded from part of a tree makes every unscanned suppression a
    fresh violation, so the scope is refused rather than honoured.

    REFUSED BEFORE THE SCAN, which is asserted the only way it can be asserted
    from outside: the tree also holds a file that cannot be tokenized, and that
    file is an exit 2 of its own with a message naming it.  A refusal that came
    after the scan would report the file instead of the scope.
    """
    baseline = _write_fixture_tree(
        tmp_path,
        {'pkg/m.py': 'a = 1  # type: ignore[arg-type]\n', 'broken.py': 'a = (  # nosec\n'},
    )
    capsys.readouterr()

    assert _seed(tmp_path, baseline, 'pkg') == 2

    error = capsys.readouterr().err
    assert 'scoped' in error
    assert 'broken.py' not in error
    assert not baseline.exists()


def _tighten(root: Path, baseline_path: Path, *paths: str) -> int:
    """Run ``--tighten`` over *root*, optionally scoped to *paths*."""
    return inline_suppressions.main(
        ['--tighten', '--root', str(root), '--baseline', str(baseline_path), *paths]
    )


def test_removing_a_marker_becomes_slack_that_tighten_takes_away(tmp_path: Path, capsys):
    """BOUNDARY SCENARIO 5 — the ratchet's forward click.

    Slack is not a cosmetic figure: because two identical lines share one D7 key,
    an un-tightened baseline lets an identical line straight back in where one was
    removed.  The headroom is a standing invitation nobody meant to leave open,
    which is why ``--check`` reports it and D11's sweep files a task to spend it.
    """
    lines = _grandfathered(3).splitlines(keepends=True)
    baseline = _write_fixture_tree(tmp_path, {'m.py': ''.join(lines)}, baseline=True)
    _revise(tmp_path, {'m.py': ''.join(lines[:2])})
    capsys.readouterr()

    assert _check(tmp_path, baseline) == 0
    assert 'slack 1' in capsys.readouterr().out

    before = set(load(baseline).counts)
    assert _tighten(tmp_path, baseline) == 0

    reported = capsys.readouterr().out
    (gone,) = before - set(load(baseline).counts)
    assert gone in reported

    assert _check(tmp_path, baseline) == 0
    assert 'slack 0' in capsys.readouterr().out


def test_a_second_tighten_over_an_unchanged_tree_writes_the_same_bytes(
    tmp_path: Path, capsys
):
    """Idempotent, and asserted on BYTES so a rewrite that reordered keys or
    re-rendered a float would show up.

    ``tighten`` is the pointwise minimum, so this follows from the arithmetic
    rather than from a guard — which is the property worth pinning, because a
    baseline that churned on every run would make its own review impossible.
    """
    lines = _grandfathered(3).splitlines(keepends=True)
    baseline = _write_fixture_tree(tmp_path, {'m.py': ''.join(lines)}, baseline=True)
    _revise(tmp_path, {'m.py': ''.join(lines[:2])})

    assert _tighten(tmp_path, baseline) == 0
    once = baseline.read_bytes()
    assert _tighten(tmp_path, baseline) == 0
    capsys.readouterr()

    assert baseline.read_bytes() == once


def test_tighten_never_adds_a_key_so_it_is_not_a_way_to_go_green(tmp_path: Path):
    """The no-add-key property, from the consumer's side.

    ``tighten``'s result is a pointwise minimum, so a key the baseline does not
    hold has multiplicity 0 there and the minimum of anything and 0 is 0 — the
    property is structural rather than a check anyone can forget.  What this test
    pins is the consequence that matters at the gate: an agent facing a red run
    cannot clear it by tightening.
    """
    baseline = _write_fixture_tree(
        tmp_path, {'m.py': 'a = 1  # type: ignore[arg-type]\n'}, baseline=True
    )
    _revise(
        tmp_path,
        {'m.py': 'a = 1  # type: ignore[arg-type]\nb = 2  # type: ignore[attr-defined]\n'},
    )
    before = set(load(baseline).counts)

    assert _tighten(tmp_path, baseline) == 0

    assert set(load(baseline).counts) == before
    assert _check(tmp_path, baseline) == 1


def test_a_scoped_tighten_is_refused_and_leaves_the_baseline_alone(tmp_path: Path, capsys):
    """The widening the kernel's arithmetic cannot refuse, refused here.

    A scoped scan is honestly ``complete=True`` for its scope, so it passes
    ``_require_comparable`` and the pointwise minimum goes through — writing
    ``baseline ∩ scope`` and silently deleting every key outside the paths.  That
    is a one-command gate-widening, so the verb refuses the scope outright.
    """
    baseline = _write_fixture_tree(
        tmp_path,
        {
            'pkg/kept.py': 'a = 1  # type: ignore[arg-type]\n',
            'other/kept.py': 'b = 2  # type: ignore[attr-defined]\n',
        },
        baseline=True,
    )
    before = baseline.read_bytes()
    capsys.readouterr()

    assert _tighten(tmp_path, baseline, 'pkg') == 2

    assert baseline.read_bytes() == before
    assert 'scoped' in capsys.readouterr().err


#: One tree exercising every block the report publishes: an unowned marker of
#: each pyright spelling, a debt, a policy by inline ratification, both flavours
#: of dead marker, and a file the prefilter drops.
_REPORT_TREE = {
    'pyproject.toml': RUFF_CONFIG,
    'plain.py': 'nothing = "here"\n',
    'm.py': (
        'a = 1  # type: ignore[arg-type]\n'
        'b = 2  # noqa: E402  # debt: task 5601\n'
        'c = 3  # noqa: F401  # ratified: inv12-day-one-test-doubles\n'
        'd = 4  # pragma: no cover\n'
        'e = 5  # noqa: PLC0415\n'
        'g = 7  # pyright: ignore[reportArgumentType]\n'
    ),
}


def _json_text(
    root: Path,
    baseline_path: Path,
    capsys,
    *paths: str,
    classes: Mapping[inline_suppressions.SuppressionClass, Policy] = (
        inline_suppressions.RATIFIED_SUPPRESSION_CLASSES
    ),
) -> str:
    """Run ``--json`` against *classes* and return its raw stdout, asserting exit 0.

    The buffer is drained first: a fixture seeded through ``--seed`` has already
    printed its own report line, and ``json.loads`` of the two concatenated fails
    on the first character with nothing to say about why.
    """
    capsys.readouterr()
    code = inline_suppressions.main(
        ['--json', '--root', str(root), '--baseline', str(baseline_path), *paths],
        classes=classes,
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    return captured.out


def _json_report(
    root: Path,
    baseline_path: Path,
    capsys,
    *paths: str,
    classes: Mapping[inline_suppressions.SuppressionClass, Policy] = (
        inline_suppressions.RATIFIED_SUPPRESSION_CLASSES
    ),
) -> dict:
    """The parsed ``--json`` report, run against *classes*."""
    return json.loads(_json_text(root, baseline_path, capsys, *paths, classes=classes))


def test_json_publishes_every_block_the_register_reads(tmp_path: Path, capsys):
    """``--json`` is task 5602's only data source, so nothing downstream re-scans.

    The totals are asserted as WHOLE collections rather than by spot-checking a
    key, because the failure this guards against is a block that quietly stops
    being emitted — which a membership test passes right up until the consumer
    reads a missing key.  Every kind appears even at zero, for the same reason:
    ``nosec`` reading 0 is a measurement, and a schema where it vanishes makes
    "no bandit markers" and "the scanner forgot bandit" the same output.
    """
    baseline = _write_fixture_tree(tmp_path, _REPORT_TREE)
    report = _json_report(tmp_path, baseline, capsys)

    assert report['schema_version'] == inline_suppressions.REPORT_SCHEMA_VERSION == 1
    assert set(report['params']) == {'kinds', 'key_scheme', 'digest_hex'}
    assert report['status'] == 'advisory'
    assert report['files_enumerated'] == 2
    assert report['files_tokenized'] == 1
    assert report['sites'] == 6
    assert report['kind_totals'] == {
        'noqa': 3,
        'nosec': 0,
        'pragma: no cover': 1,
        'pyright: ignore': 1,
        'type: ignore': 1,
    }
    assert report['consumers'] == {'first-party': 0, 'none': 2, 'pyright': 2, 'ruff': 2}
    assert report['ownership'] == {'class': 0, 'debt': 1, 'policy': 1, 'unowned': 4}
    assert report['by_kind_code'] == [
        {'kind': 'noqa', 'code': 'E402', 'sites': 1},
        {'kind': 'noqa', 'code': 'F401', 'sites': 1},
        {'kind': 'noqa', 'code': 'PLC0415', 'sites': 1},
        {'kind': 'pragma: no cover', 'code': None, 'sites': 1},
        {'kind': 'pyright: ignore', 'code': 'reportArgumentType', 'sites': 1},
        {'kind': 'type: ignore', 'code': 'arg-type', 'sites': 1},
    ]


def test_json_names_every_debt_owner_and_every_ratified_id_with_its_sites(
    tmp_path: Path, capsys
):
    """The two blocks that exist for a CONSUMER rather than for a reader.

    Task 5602's closed-world check asks whether every inline ``ratified:`` id names a
    real ratification row, so it needs the ids AND the sites citing each one — an
    id with no sites to point at is a finding it cannot report.  The debt block
    is the same shape for the same reason: the owner has to be followable back to
    a task or a ticket without re-scanning the tree.
    """
    baseline = _write_fixture_tree(tmp_path, _REPORT_TREE)
    report = _json_report(tmp_path, baseline, capsys)

    assert report['debt'] == [
        {'path': 'm.py', 'line': 2, 'kind': 'noqa', 'codes': ['E402'], 'owner': 'task 5601'}
    ]
    assert report['ratified'] == [
        {
            'id': 'inv12-day-one-test-doubles',
            'sites': [{'path': 'm.py', 'line': 3, 'kind': 'noqa', 'codes': ['F401']}],
        }
    ]


def test_json_publishes_the_resolved_ruff_lists_rather_than_the_params_block(
    tmp_path: Path, capsys
):
    """``ConsumerModel.resolved``'s audit trail, kept out of ``params`` as
    ``scripts/inline_suppression_key.py::key_params`` records.

    The consumer model's answer for a ``noqa`` depends entirely on these two
    lists, so a reader has to be able to see what the model actually read —
    including the known meta-prefix limit, which is only visible as an absence.
    They are NOT in ``params``: a params mismatch is exit 2 whose only exit is
    re-seeding, so putting them there would punish an ordinary reviewable
    pyproject edit by demanding the one operation nobody should perform casually.
    """
    files = dict(_REPORT_TREE)
    files['pkg/pyproject.toml'] = '[tool.ruff.lint]\nselect = ["F"]\nignore = []\n'
    files['pkg/mod.py'] = 'h = 8  # noqa: F401\n'
    baseline = _write_fixture_tree(tmp_path, files)

    report = _json_report(tmp_path, baseline, capsys)

    assert report['ruff_config'] == [
        {'pyproject': 'pkg/pyproject.toml', 'select': ['F'], 'ignore': []},
        {
            'pyproject': 'pyproject.toml',
            'select': ['E', 'F', 'UP', 'B', 'SIM', 'I'],
            'ignore': ['E501'],
        },
    ]
    assert 'select' not in report['params']


def test_json_counts_a_class_ratified_site_under_its_class(tmp_path: Path, capsys):
    """BOUNDARY SCENARIO 10 in the report — blanket policy stays VISIBLE.

    A class row moves sites out of the unowned multiset, which is exactly the
    move that could hide a growing population behind one operator ruling.  So the
    report counts them under the rendered class key, and D11's sweep can see how
    much each row is carrying.
    """
    row = inline_suppressions.SuppressionClass(
        kind=Kind.TYPE_IGNORE,
        code='arg-type',
        scope=inline_suppressions.Scope.ANY,
    )
    baseline = _write_fixture_tree(tmp_path, _REPORT_TREE)

    report = _json_report(tmp_path, baseline, capsys, classes={row: Policy('inv12-day-one')})

    assert report['classes'] == [
        {
            'class': 'type: ignore[arg-type]@any',
            'sites': [{'path': 'm.py', 'line': 1, 'kind': 'type: ignore', 'codes': ['arg-type']}],
        }
    ]
    assert report['ownership']['class'] == 1


def test_json_carries_the_verdict_a_check_would_reach(tmp_path: Path, capsys):
    """The report reader sees the gate's answer without running the gate.

    Without this a consumer would have to invoke ``--check`` as well, and then
    reconcile two scans of a tree that may have changed between them.
    """
    baseline = _write_fixture_tree(
        tmp_path,
        {
            'm.py': _grandfathered(3),
            'n.py': 'y = 8  # type: ignore[arg-type]  # debt: soon\n',
        },
        baseline=True,
    )
    _revise(tmp_path, {'m.py': _grandfathered(2) + 'z = 9  # type: ignore[no-any-return]\n'})

    report = _json_report(tmp_path, baseline, capsys)

    assert report['status'] == 'clean'
    assert set(report['baseline']) == {'path', 'present', 'excess', 'slack'}
    assert report['baseline']['path'] == str(baseline)
    assert report['baseline']['present'] is True
    assert list(report['baseline']['excess'].values()) == [1]
    assert list(report['baseline']['slack'].values()) == [1]
    located = [(entry['path'], entry['line']) for entry in report['violations']]
    assert located == [('n.py', 1), ('m.py', 3)]


def test_json_exits_zero_even_when_the_gate_would_be_red(tmp_path: Path, capsys):
    """A REPORT verb, not a gate: a consumer parsing the report must not also be
    gated by it.

    Only a broken instrument makes ``--json`` non-zero, which is what lets task 5602
    read a tree that is currently in breach — the state it most needs to read.
    """
    baseline = _write_fixture_tree(tmp_path, {'m.py': 'a = 1\n'}, baseline=True)
    _revise(tmp_path, {'m.py': 'a = 1  # type: ignore[arg-type]\n'})

    report = _json_report(tmp_path, baseline, capsys)

    assert report['violations'] != []
    assert _check(tmp_path, baseline) == 1


@pytest.mark.parametrize('verb', ['--seed', '--tighten'])
def test_json_is_mutually_exclusive_with_the_writing_verbs(tmp_path: Path, capsys, verb: str):
    """An argparse error rather than a silent precedence rule.

    ``--json --seed`` has two defensible readings (report then write; write then
    report) and no way for the caller to say which they meant, so the CLI refuses
    instead of picking one.
    """
    baseline = _write_fixture_tree(tmp_path, _REPORT_TREE)

    with pytest.raises(SystemExit) as raised:
        inline_suppressions.main(
            ['--json', verb, '--root', str(tmp_path), '--baseline', str(baseline)]
        )

    assert raised.value.code == 2
    assert 'not allowed with' in capsys.readouterr().err


def test_two_json_runs_over_one_tree_emit_identical_bytes(tmp_path: Path, capsys):
    """Determinism, asserted on BYTES, plus the sort that makes it hold.

    A report that reordered between runs would make every consumer's diff noise,
    and the reason it cannot is structural: every mapping is dumped with sorted
    keys and every list is sorted before it is dumped.
    """
    baseline = _write_fixture_tree(tmp_path, _REPORT_TREE)

    first = _json_text(tmp_path, baseline, capsys)
    second = _json_text(tmp_path, baseline, capsys)

    assert first == second
    report = json.loads(first)
    assert list(report) == sorted(report)
    assert list(report['kind_totals']) == sorted(report['kind_totals'])
    assert report['by_kind_code'] == sorted(
        report['by_kind_code'], key=lambda row: (row['kind'], row['code'] or '')
    )


def test_a_missing_shared_import_is_exit_two_and_never_exit_one(tmp_path: Path):
    """The Contract's "an ImportError is 2, never 1", which only a real process
    can prove.

    Every other test here calls ``main([...])`` in-process, and none of them could
    catch this: the fault it pins happens before ``main`` is reachable.  A
    top-level ``from shared… import …`` raises while the module is still
    executing, so ``main`` is never defined, the ``__main__`` block never runs,
    and Python's own uncaught-exception exit is 1 — the exact code the Contract
    reserves for a FINDING.  A gate that reported a broken environment as an
    INV-12 breach would send an agent to fix code that was never the problem.

    The scanner's whole ``inline_suppression*.py`` family is copied into
    ``tmp_path``: its sibling strata resolve from ``sys.path[0]``, which is the
    copied entry script's own directory, so copying that script alone would fail
    its first sibling import with an unrelated exit 1.  The ``<repo>/shared/src``
    its bootstrap resolves from ``__file__`` does not exist there either, and the
    run gets ``-S`` because the workspace member is editable-installed in the
    venv and ``PYTHONPATH`` alone cannot hide it.  A generous subprocess
    ``timeout`` rather than any wall-clock assertion, per this module's docstring.
    """
    for source in sorted((REPO_ROOT / 'scripts').glob('inline_suppression*.py')):
        (tmp_path / source.name).write_text(source.read_text(encoding='utf-8'), encoding='utf-8')
    copied = tmp_path / SCRIPT.name

    completed = _run_script(
        ['--check', '--root', str(tmp_path), '--baseline', str(tmp_path / _BASELINE_NAME)],
        cwd=tmp_path,
        script=copied,
        env=_python_env_without_shared(),
        flags=_NO_SITE,
    )

    assert completed.returncode == 2, completed.stderr
    assert 'shared' in completed.stderr
    assert 'uv run --project shared' in completed.stderr
    assert 'Traceback' not in completed.stderr


# ---------------------------------------------------------------------------
# The live tree — the merge gate's actual enforcement point.

#: Where task 5607 seeds the committed baseline.  Absent until that cutover lands,
#: which is what makes the enforcing guard below a skip rather than a red.
LIVE_BASELINE = REPO_ROOT / 'scripts' / _BASELINE_NAME


def _live_report(capsys, *extra: str) -> dict:
    """The ``--json`` report for THIS repository."""
    capsys.readouterr()
    code = inline_suppressions.main(['--json', '--root', str(REPO_ROOT), *extra])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    return json.loads(captured.out)


def test_the_live_tree_reports_the_signal_this_scanner_exists_to_produce(capsys):
    """The NON-VACUITY FLOOR, because "nothing was read" and "nothing was wrong"
    are otherwise the same output.

    Four kinds are non-zero in this repository, and that is the task's
    user-observable deliverable: pyright and ruff are both declared gates here
    and ``# type: ignore`` / ``# noqa`` / ``# pragma: no cover`` /
    ``# pyright: ignore`` all appear, so a scan that silently read nothing fails
    all four and is loud.

    THE FIFTH KIND IS ASSERTED AS A KEY, NOT AS A ZERO.  ``nosec`` reads 0 only
    because bandit is not installed, which is a fact about this corpus on this
    day: the first legitimate ``# nosec`` anyone writes would turn an equality
    here into a red pointing at this scanner's test rather than at the new
    marker.  What is worth holding is ``_tally``'s contract that every kind gets
    a row even at zero — a schema omitting ``nosec`` would make "no bandit
    markers" and "the scanner stopped looking for them" the same output — and
    that is a statement about the report's SHAPE, which no corpus can falsify.
    The four assertions above already carry the whole non-vacuity floor.
    """
    report = _live_report(capsys)

    assert report['files_enumerated'] >= 1000
    assert report['sites'] >= 1000
    for kind in ('type: ignore', 'noqa', 'pragma: no cover', 'pyright: ignore'):
        assert report['kind_totals'][kind] > 0, kind
    assert 'nosec' in report['kind_totals']
    assert report['consumers']['none'] > 0
    assert report['ruff_config'] != []


def test_the_live_tree_carries_no_disposition_faults(tmp_path: Path, capsys):
    """THE OTHER HALF OF THE GATE, and the half no baseline can ever grandfather.

    ``_verdict`` reports ``classification.violations`` on the ADVISORY path too:
    a marker that does not parse, or a disposition on a line that suppresses
    nothing, is a fault at its site whatever a baseline holds.  The ratchet
    grandfathers KEYS and a fault is not a key, so seeding cannot turn one
    green — which is why this test carries none of the seeded guard's skip, and
    why it is not one more line in the non-vacuity floor above: a named red says
    which of the two subjects broke.

    The run is pinned ADVISORY by pointing it at a baseline that cannot exist,
    so the violations it reports are exactly the disposition faults, before the
    seed and after it alike, and never ratchet excess.
    """
    report = _live_report(capsys, '--baseline', str(tmp_path / _BASELINE_NAME))

    assert report['status'] == 'advisory'
    assert report['violations'] == [], (
        'disposition fault(s) in this repository:\n  '
        + '\n  '.join(f'{fault["path"]}:{fault["line"]}: {fault["reason"]}' for fault in report['violations'])
        + '\nA disposition marker quoted inside a `#` comment is a permanent violation that no '
        'disposition can cover. Quote example markers in a DOCSTRING or another string '
        'literal, which this scanner deliberately does not read, and never in a `#` comment.'
    )


def test_the_live_scan_tokenizes_exactly_the_marker_bearing_files(capsys):
    """THE ≤10 s BUDGET, PINNED AS COUNTED WORK AND NEVER AS A CLOCK.

    The budget fits only because of the prefilter, so the honest guard is that
    the prefilter is doing its job: strictly fewer files are tokenized than
    enumerated, and the ones that are are exactly those whose raw bytes carry a
    marker substring.  An ``assert elapsed < 10`` would instead be a new flake on
    the box the merge gate runs on — measured, this scan costs about 5 CPU-seconds
    and took 6.5-6.8 s of wall clock at load 105 on 32 cores.
    ``orchestrator/tests/test_merge_lane_ratchet.py`` ruled the same way for the
    sibling ratchet: count WORK, never wall-clock.

    The expected count is re-derived from the PUBLIC kind table rather than read
    off the scanner's private prefilter constant, so the two can disagree; a
    marker added to the table with no prefilter byte is exactly what that catches.
    A tracked path whose worktree file is gone is passed over here for the same
    reason the scanner passes over it: ``git ls-files`` reads the index.
    """
    specs = tuple(KIND_SPECS.values())
    listed = run_git(['ls-files', '-z', '--', '*.py'], cwd=REPO_ROOT).stdout
    tracked = {path for path in listed.split('\0') if path}
    carrying = 0
    for relative in tracked:
        try:
            raw = (REPO_ROOT / relative).read_bytes()
        except FileNotFoundError:
            continue
        carrying += any(
            spec.marker.encode('utf-8') in (raw.lower() if spec.folds_case else raw)
            for spec in specs
        )

    report = _live_report(capsys)

    assert report['files_enumerated'] == len(tracked)
    assert report['files_tokenized'] < report['files_enumerated']
    assert report['files_tokenized'] == carrying


def test_the_live_tree_passes_the_gate_once_the_baseline_is_seeded(capsys):
    """The merge gate's actual assertion — ENFORCING iff the baseline exists.

    It skips rather than reds before task 5607's cutover because D12 makes
    baseline absence a legitimate state, and a guard that failed on the
    pre-cutover tree would block every task until that cutover landed.  Everything about the gate MECHANISM
    is covered hermetically by the fixture-tree tests above, so the skip loses no
    coverage of this module's behaviour — only of this repository's compliance,
    which is not yet a thing to be compliant with.
    """
    if not LIVE_BASELINE.exists():
        pytest.skip(
            f'{LIVE_BASELINE.relative_to(REPO_ROOT)} does not exist yet: the baseline is '
            'seeded once, on main, with `--seed` (task 5607). Until then every run is '
            'advisory by design (D12), and the gate mechanism is covered hermetically by '
            'the fixture-tree tests in this module.'
        )
    capsys.readouterr()

    assert inline_suppressions.main(['--check', '--root', str(REPO_ROOT)]) == 0, (
        capsys.readouterr().err
    )
