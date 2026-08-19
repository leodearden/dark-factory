"""The single findable home for this suite's steward test-scaffolding invariants.

Consolidation lineage 3461 → 3514 → 3551 → 3647.  Task 3461 merged the two
near-identical ``_make_steward`` copies from ``test_suggestion_triage.py`` and
``test_workflow_state_machine_boundary.py``; task 3514 folded in the two that
remained (``test_out_of_band_routing.py``'s, and ``test_steward.py``'s
five-fixture graph); task 3551 examined
``test_workflow_escalated_steward_stall.py``'s ``_make_steward_config`` and left
it standing, propagating only its sandboxed ``project_root`` recipe.

Each of those tasks recorded its findings in PROSE, and each subsequent task
re-litigated them from scratch — which is the evidence that prose alone is not a
durable record.  Task 3647 therefore turns the two standing rulings into
CHECKABLE invariants, matching the ethos the suite already states elsewhere
("Enforced, not merely documented" — ``test_out_of_band_routing.py``'s
``_REVIEW_PROJECT_ROOT`` block, ``conftest.py``'s ``make_steward`` worktree
guard).  Three concerns live here, deliberately in ONE module because the
lineage's actual failure mode is that they keep getting scattered and re-derived:

1. :class:`TestAssertSandboxedProjectRoot` — the contract of
   ``_orch_helpers.assert_sandboxed_project_root``, the shared assertion that
   replaced two drifted inline copies of the sandbox block.
2. ``TestNoInlineSandboxedProjectRootAsserts`` — an AST recurrence guard that no
   module re-implements that block inline (added in step-3).
3. ``TestStewardConstructionSitesAreCensused`` — an AST census guard pinning the
   sanctioned steward-construction sites, each with a recorded reason (step-5).

Modelled on ``test_git_repo_isolation_guard.py``, which is exactly this shape for
the esc-3072-3 incident class: helper unit tests, plus an AST recurrence guard
with synthetic-source detector self-tests, plus a liveness assertion, all in one
module.
"""
from __future__ import annotations

import ast
import functools
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple
from unittest.mock import MagicMock

import pytest
from _orch_helpers import assert_sandboxed_project_root


class TestAssertSandboxedProjectRoot:
    """The full contract of ``_orch_helpers.assert_sandboxed_project_root``.

    Four clauses, each pinned by at least one test below: the value must be a
    real ``Path``, it must be a CREATED directory, it must not be the sandbox
    root itself, and it must resolve strictly below that root.  Clause 2 is
    pinned BOTH ways ``is_dir()`` can fail (a path that does not exist, and one
    that exists as a regular file), and clause 4's ``.resolve()`` calls are
    pinned by the symlink case — without it, a regression that dropped them
    would leave every other test here green while a link under the sandbox
    pointing outside it sailed through.

    Clauses 2 and 3 are the DRIFT RECONCILIATION.  The copy in
    ``test_conftest_helpers.py`` never carried the ``.is_dir()`` clause, and
    NEITHER copy carried the ``!= tmp_path`` strictness clause — ``is_relative_to``
    returns ``True`` for a path against itself, so it alone does not mean
    "strictly below".  Folding the copies onto this helper therefore strengthens
    them rather than levelling them down to the laxest copy, which is the usual
    failure mode when deduplicating drifted assertions.

    Every rejection is an ``AssertionError`` specifically — a sandbox escape must
    read as a test FAILURE, not an error — and every message names the offending
    value, so a report identifies the bad root without a debugger.
    """

    def test_accepts_a_created_directory_strictly_below_the_sandbox(self, tmp_path):
        """The happy path: exactly what ``make_steward`` produces (``tmp_path / 'project'``).

        Returns ``None``; the helper is used for its raise, like its
        ``assert_isolated_git_repo`` sibling.
        """
        project_root = tmp_path / 'project'
        project_root.mkdir()

        assert assert_sandboxed_project_root(project_root, tmp_path) is None

    @pytest.mark.parametrize(
        ('bad', 'type_name'),
        [
            (MagicMock(), 'MagicMock'),
            ('/tmp/project', 'str'),
        ],
        ids=['magicmock', 'str'],
    )
    def test_rejects_a_non_path(self, bad, type_name, tmp_path):
        """Clause 1. A ``MagicMock`` child silently satisfies every ``/``-join the
        steward performs without ever producing a directory, so a mock root
        cannot be caught downstream — only here.  A ``str`` is rejected for the
        same reason in reverse: it has no ``.is_dir()`` / ``.is_relative_to`` to
        check the remaining clauses with.
        """
        with pytest.raises(AssertionError, match=type_name):
            assert_sandboxed_project_root(bad, tmp_path)

    def test_rejects_a_path_that_was_never_created(self, tmp_path):
        """Clause 2 — the one ``test_conftest_helpers.py``'s copy drifted away from.

        The retired ``Path('/tmp/fake-project')`` literal was never created by
        anything, and a dangling ``project_root`` is a latent ``cwd=`` failure the
        moment a test stops patching the invoke seam.
        """
        never_created = tmp_path / 'never-created'

        with pytest.raises(AssertionError, match='never-created'):
            assert_sandboxed_project_root(never_created, tmp_path)

    def test_rejects_a_regular_file(self, tmp_path):
        """Clause 2, the OTHER way ``is_dir()`` fails: the path exists, but is
        not a directory.

        Distinct from the never-created case above and just as plausible — a
        test that has already written ``tmp_path / 'project'`` as a file, then
        hands it over as a root.  ``exists()`` would accept it; ``is_dir()`` is
        what rejects it, so this pins the clause's actual spelling rather than
        the weaker one it could drift to.
        """
        regular_file = tmp_path / 'project'
        regular_file.write_text('not a directory', encoding='utf-8')

        with pytest.raises(AssertionError, match='CREATED directory'):
            assert_sandboxed_project_root(regular_file, tmp_path)

    def test_rejects_a_symlink_under_the_sandbox_pointing_outside_it(self, tmp_path):
        """Clause 4's ``.resolve()`` calls, on BOTH sides — the load-bearing part.

        A symlink created under the sandbox but pointing outside it is
        lexically contained (``link.is_relative_to(sandbox)`` is ``True``) and
        ``is_dir()`` follows it, so clauses 1-3 all pass.  Only resolving both
        sides catches it — and everything the steward writes through such a root
        lands outside the directory pytest's retention sweep reclaims, which is
        the exact escape this helper exists to stop.

        Without this case a regression that dropped ``.resolve()`` (say
        ``resolved = project_root``) would leave every other test in this class
        green.
        """
        sandbox = tmp_path / 'sandbox'
        sandbox.mkdir()
        outside = tmp_path / 'elsewhere'
        outside.mkdir()
        link = sandbox / 'project'
        link.symlink_to(outside, target_is_directory=True)

        # Clauses 1-3 genuinely pass: this is a real, existing, non-root Path.
        assert link.is_dir()
        assert link.is_relative_to(sandbox)

        with pytest.raises(AssertionError, match='elsewhere'):
            assert_sandboxed_project_root(link, sandbox)

    def test_rejects_the_sandbox_root_itself(self, tmp_path):
        """Clause 3 — the strictness clause NEITHER existing copy carried.

        ``Path.is_relative_to`` returns ``True`` for a path against itself, so
        the two folded copies would both have accepted the sandbox root.  This
        is the literal reading of "strictly below", and it matches the spelling
        ``conftest.py``'s ``make_steward`` worktree guard already uses
        (``resolved == root or not resolved.is_relative_to(root)``).
        """
        with pytest.raises(AssertionError, match='strictly below'):
            assert_sandboxed_project_root(tmp_path, tmp_path)

    def test_rejects_an_existing_directory_outside_the_sandbox(self, tmp_path):
        """Clause 4 — the retired ``/tmp/project`` and ``/tmp/fake-project`` failure mode.

        Those literals pointed OUTSIDE the test sandbox, so anything the steward
        wrote relative to ``config.project_root`` escaped pytest's ``tmp_path``
        retention sweep.  The sandbox root here is a SUB-directory of ``tmp_path``
        so the offending root is a real, existing directory (clauses 1-3 all
        pass) and this test can only be satisfied by clause 4 — and nothing is
        created outside the directory pytest reclaims.
        """
        sandbox = tmp_path / 'sandbox'
        sandbox.mkdir()
        outside = tmp_path / 'elsewhere'
        outside.mkdir()

        with pytest.raises(AssertionError, match='elsewhere'):
            assert_sandboxed_project_root(outside, sandbox)

    def test_rejection_message_names_the_sandbox_it_was_checked_against(self, tmp_path):
        """Every message names BOTH the offending value and the root it was
        checked against — a report saying only "not under tmp_path" does not say
        WHICH tmp_path, and the two folded call sites live in different modules.
        """
        outside = tmp_path / 'elsewhere'
        outside.mkdir()
        sandbox = tmp_path / 'sandbox'
        sandbox.mkdir()

        with pytest.raises(AssertionError) as excinfo:
            assert_sandboxed_project_root(outside, sandbox)

        message = str(excinfo.value)
        assert str(outside) in message, f'message must name the offending root: {message}'
        assert str(sandbox) in message, f'message must name the sandbox root: {message}'

    def test_a_non_path_rejection_is_an_assertion_error_not_an_attribute_error(
        self, tmp_path,
    ):
        """Clause ordering is load-bearing: the ``isinstance`` clause must run
        FIRST.  A helper that reached for ``.is_dir()`` on a ``str`` would raise
        ``AttributeError``, which pytest reports as an ERROR rather than a
        failure and buries the actual diagnosis.
        """
        with pytest.raises(AssertionError):
            assert_sandboxed_project_root('/tmp/project', tmp_path)


# ===========================================================================
# Recurrence guard: nobody re-implements the sandbox block inline (task 3647)
# ===========================================================================

_TESTS_DIR = Path(__file__).parent
_HELPER_NAME = 'assert_sandboxed_project_root'

# The canonical OWNER of the pattern, excluded from the sweep by construction:
# `_orch_helpers.py` is where the block is supposed to live, so matching it
# there would be the guard flagging the fix.  Same shape as
# `test_git_repo_isolation_guard.py`'s single `_TARGET_MODULE` scoping.
_CANONICAL_OWNER = '_orch_helpers.py'

# A sweep that silently parses nothing reads as "zero recurrences".  The tests
# tree holds 532 parseable modules as of task 3647, so this floor sits ~25%
# below the real count: close enough that dropping a meaningful fraction of the
# tree trips it, far enough that ordinary growth and pruning does not.  A floor
# several times below the real size (this was 100) is nearly as blind as no
# floor at all — it would have let a glob regression discard 80% of the tree
# while both liveness tests stayed green.  The tree only grows, so raise this
# when it drifts; do NOT lower it to silence a failure.
_MIN_MODULES_SWEPT = 400

# Co-locating the two sweeping tests on one xdist worker was MEASURED and
# REJECTED, recorded here so it is not re-tried.  `_scan_tests_tree` is a
# per-PROCESS cache, so an `xdist_group` tag would let the second guard read the
# first one's parse and halve this module's CPU.  It also SERIALISES the two
# heaviest tests onto one worker: measured 66s wall for the module grouped vs
# 44s ungrouped.  The suite runs `-n auto` on a many-core host where the binding
# constraint is the critical path, not CPU, so the two guards are left free to
# run in parallel and each pay their own scan.


def _mentions(node: ast.AST, name: str) -> bool:
    """True if *name* appears anywhere under *node* as a Name id or attribute."""
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id == name:
            return True
        if isinstance(n, ast.Attribute) and n.attr == name:
            return True
    return False


def _asserts_isinstance_path(func: ast.AST) -> bool:
    """Signal (ii): an ``assert isinstance(X, Path)`` somewhere in *func*.

    This is the DISCRIMINATING conjunct — see `_inline_sandbox_asserts`.
    """
    for node in ast.walk(func):
        if not isinstance(node, ast.Assert):
            continue
        for call in ast.walk(node.test):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == 'isinstance'
                and len(call.args) >= 2
                and isinstance(call.args[1], ast.Name)
                and call.args[1].id == 'Path'
            ):
                return True
    return False


def _asserts_is_relative_to_tmp_path(func: ast.AST) -> bool:
    """Signal (iii): a NON-NEGATED ``assert ....is_relative_to(<...tmp_path...>)``.

    Both qualifiers are load-bearing, not incidental strictness:

    * non-negated — ``assert not x.is_relative_to(y)`` asserts the OPPOSITE
      invariant (that two paths are disjoint), which this helper does not own;
    * ``tmp_path`` inside the call — a containment assertion against some other
      root (``worktree_base``, ``xdg_home``, …) is a different invariant.
    """
    for node in ast.walk(func):
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            continue
        for call in ast.walk(test):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == 'is_relative_to'
                and _mentions(call, 'tmp_path')
            ):
                return True
    return False


def _inline_sandbox_asserts(tree: ast.Module) -> list[str]:
    """Functions in *tree* that re-implement the sandboxed-project-root block.

    A hit satisfies ALL THREE signals:

    (i)   mentions ``project_root``;
    (ii)  asserts ``isinstance(X, Path)``;
    (iii) has a non-negated ``is_relative_to(...tmp_path...)`` assert.

    The CONJUNCTION is the design, and signal (ii) is what discriminates.
    Measured over the whole tests tree at the time this guard landed:

    * all three signals → exactly 2 hits, both the real copies, zero false
      positives;
    * dropping signal (ii) → 3 hits, re-admitting
      ``test_scheduler_state.py::test_bare_config_project_root_isolated_to_tmp``,
      which asserts a DERIVED snapshot path is under ``tmp_path`` and checks
      ``project_root`` itself by ``==``, not containment.  It is a genuinely
      different assertion and must not be flagged.

    Two further near-misses are excluded by signal (iii) rather than (ii), and
    are recorded here so a future reader does not "simplify" that signal either:

    * ``test_mcp_lifecycle.py::test_neither_path_is_under_a_sample_project_root``
      — ``assert not queue_dir.is_relative_to(project_root)``: NEGATED, and
      against ``project_root`` / ``xdg_home`` rather than ``tmp_path``;
    * ``test_verify_preexisting_main_break.py::test_real_git_probe_lifecycle``
      — ``worktree_arg.is_relative_to(git_ops.worktree_base)``: a different
      containment root, no ``tmp_path`` in the call.

    A guard with false positives gets weakened or deleted by the next author, so
    every relaxation of this rule costs one of those exclusions.
    """
    offenders: list[str] = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not _mentions(func, 'project_root'):
            continue
        if not _asserts_isinstance_path(func):
            continue
        if not _asserts_is_relative_to_tmp_path(func):
            continue
        offenders.append(f'{func.name}:{func.lineno}')
    return offenders


class _TreeScan(NamedTuple):
    """The parsed tests tree, plus what the scan could NOT parse.

    ``unparseable`` exists so a module can never vanish from both guards
    silently: a swallowed ``SyntaxError`` shrinks the swept set, and a shrunken
    swept set is indistinguishable from "found nothing wrong".  The liveness
    assertions check it is empty rather than trusting the count alone.
    """

    modules: tuple[tuple[Path, ast.Module], ...]
    unparseable: tuple[str, ...]


@functools.cache
def _scan_tests_tree() -> _TreeScan:
    """Read and ``ast.parse`` EVERY ``*.py`` under this tests tree, once.

    Deliberately unfiltered: both AST guards in this file sweep from here, and
    each applies its own exclusions at use time.  The recurrence guard drops the
    canonical owner; the census guard drops nothing, so a steward built in a
    helper module could never be silently sanctioned by a shared exclusion it
    was not the subject of.

    COST, and why the consumers are shaped the way they are.  Reading and
    parsing the 532-module tree is by far the most expensive thing in this file,
    and it is paid PER PROCESS: under ``pytest-xdist`` a ``functools.cache``
    buys nothing across workers, so every test that triggers a fresh full-tree
    scan lands on some worker and pays for it again.  With six sweeping tests
    that cost the suite ~80s of CPU to compute the same two answers.  Hence:
    every derived sweep is cached (``_recurrence_swept_modules``,
    ``_census_by_module``), and each guard exposes ONE test that asserts its
    offender list, its liveness floor and its exclusions together.  Splitting
    those assertions back into a test apiece is not free and not more rigorous —
    it re-triggers the scan on a fresh worker for each one.  Add a case to the
    detector self-tests (synthetic sources, no scan) instead.
    """
    parsed: list[tuple[Path, ast.Module]] = []
    unparseable: list[str] = []
    for path in sorted(_TESTS_DIR.rglob('*.py')):
        try:
            parsed.append((path, ast.parse(path.read_text(encoding='utf-8'))))
        except SyntaxError as exc:  # pragma: no cover - asserted on, not swallowed
            unparseable.append(f'{path.relative_to(_TESTS_DIR)}: {exc}')
    return _TreeScan(tuple(parsed), tuple(unparseable))


@functools.cache
def _recurrence_swept_modules() -> tuple[tuple[Path, ast.Module], ...]:
    """Everything the RECURRENCE guard scans: all modules bar the canonical owner."""
    return tuple(
        (p, t) for p, t in _scan_tests_tree().modules if p.name != _CANONICAL_OWNER
    )


class TestNoInlineSandboxedProjectRootAsserts:
    """No module re-implements the sandboxed-``project_root`` block inline.

    The whole point of task 3647: task 3551 propagated the block by COPY, and by
    3647 the two copies had drifted apart (one had lost the ``.is_dir()``
    clause; neither had the strictness clause).  Documenting "use the helper"
    is what the lineage already tried three times.  This makes it checkable.
    """

    def test_no_module_reimplements_the_block(self) -> None:
        """The whole recurrence guard, in ONE full-tree scan.

        Liveness, the canonical-owner exclusion and the offender list are
        asserted together and IN THAT ORDER, so a broken sweep reports itself as
        a broken sweep rather than as "zero recurrences".  They are not split
        into a test apiece on purpose — see ``_scan_tests_tree`` for the cost.
        """
        swept = _recurrence_swept_modules()

        # -- liveness first: a sweep that inspects nothing reads as coverage.
        assert not _scan_tests_tree().unparseable, (
            f'modules under {_TESTS_DIR} failed to parse and so were silently '
            f'absent from this sweep — fix them or this guard is blind to them: '
            f'{list(_scan_tests_tree().unparseable)}'
        )
        assert len(swept) >= _MIN_MODULES_SWEPT, (
            f'the recurrence sweep found only {len(swept)} modules under '
            f'{_TESTS_DIR} — expected at least {_MIN_MODULES_SWEPT}. A guard '
            f'that parses nothing reports zero recurrences and reads as coverage.'
        )

        # -- the canonical owner is where the block is SUPPOSED to live.
        assert (_TESTS_DIR / _CANONICAL_OWNER).is_file(), (
            f'{_CANONICAL_OWNER} must exist — it owns {_HELPER_NAME}'
        )
        assert all(p.name != _CANONICAL_OWNER for p, _ in swept), (
            f'{_CANONICAL_OWNER} must be excluded: it is the canonical owner of '
            f'the pattern, so flagging it there would be the guard flagging the fix'
        )

        offenders: list[str] = []
        for path, tree in swept:
            offenders.extend(
                f'{path.relative_to(_TESTS_DIR)}::{hit}'
                for hit in _inline_sandbox_asserts(tree)
            )

        assert not offenders, (
            'Inline re-implementation(s) of the sandboxed-project_root block.\n'
            'Each of these asserts by hand that a project_root is a real Path '
            'under tmp_path. That block has drifted before: task 3551 spread it '
            'by copy, and by task 3647 one copy had lost its .is_dir() clause '
            'and NEITHER carried the strictly-below clause (Path.is_relative_to '
            'returns True for a path against itself).\n'
            f'Fix: call {_HELPER_NAME}(<root>, tmp_path) from _orch_helpers, '
            'which owns all four clauses and their rationale. If your assertion '
            'is genuinely a different invariant, it should not be matching all '
            "three of this guard's signals — see _inline_sandbox_asserts.\n"
            f'Offenders: {offenders}'
        )

    # -- detector self-tests: synthetic sources, so this module never self-trips --
    #
    # Kept inside string literals deliberately. The sweep above parses THIS
    # module too; real assert statements here would make the guard flag itself.

    def test_the_detector_matches_the_full_three_signal_shape(self) -> None:
        """Positive sample: a detector that silently stops matching is worse
        than no detector, because it reads as coverage."""
        tree = ast.parse(
            'def test_copy(make_steward, tmp_path):\n'
            '    project_root = make_steward().config.project_root\n'
            '    assert isinstance(project_root, Path)\n'
            '    assert project_root.resolve().is_relative_to(tmp_path.resolve())\n'
        )

        assert _inline_sandbox_asserts(tree) != []

    def test_the_detector_ignores_a_function_without_project_root(self) -> None:
        """Negative sample for signal (i)."""
        tree = ast.parse(
            'def test_worktree(steward, tmp_path):\n'
            '    wt = steward.worktree\n'
            '    assert isinstance(wt, Path)\n'
            '    assert wt.resolve().is_relative_to(tmp_path.resolve())\n'
        )

        assert _inline_sandbox_asserts(tree) == []

    def test_the_detector_ignores_a_function_without_the_isinstance_assert(self) -> None:
        """Negative sample for signal (ii) — the discriminating conjunct.

        This is the shape of the real near-miss in ``test_scheduler_state.py``:
        a DERIVED path asserted under ``tmp_path``, with ``project_root``
        checked by equality rather than containment.
        """
        tree = ast.parse(
            'def test_derived(config, tmp_path):\n'
            '    assert config.project_root == tmp_path.resolve()\n'
            '    snapshot = Path(config.project_root) / "data" / "state.json"\n'
            '    assert snapshot.is_relative_to(tmp_path.resolve())\n'
        )

        assert _inline_sandbox_asserts(tree) == []

    def test_the_detector_ignores_a_negated_is_relative_to(self) -> None:
        """Negative sample for signal (iii), negation half — the real near-miss
        in ``test_mcp_lifecycle.py`` asserts the OPPOSITE invariant."""
        tree = ast.parse(
            'def test_disjoint(project_root, tmp_path):\n'
            '    assert isinstance(project_root, Path)\n'
            '    assert not project_root.is_relative_to(tmp_path)\n'
        )

        assert _inline_sandbox_asserts(tree) == []

    def test_the_detector_ignores_containment_against_another_root(self) -> None:
        """Negative sample for signal (iii), ``tmp_path`` half — the real
        near-miss in ``test_verify_preexisting_main_break.py`` checks
        containment under ``worktree_base``."""
        tree = ast.parse(
            'def test_other_root(project_root, git_ops, tmp_path):\n'
            '    assert isinstance(project_root, Path)\n'
            '    assert project_root.is_relative_to(git_ops.worktree_base)\n'
        )

        assert _inline_sandbox_asserts(tree) == []


# ===========================================================================
# Census guard: DECISION 1 — the steward-construction split is PERMANENT
# ===========================================================================

class _Sanctioned(NamedTuple):
    """One adjudicated steward-construction module: how many sites, and why.

    *sites* is deliberately a COUNT, not just a flag.  Sanctioning a module
    wholesale would pre-approve every FUTURE construction it grows — someone
    could add a fourth idiom inside an already-listed module and the census
    would stay green, which is the exact silent appearance this guard exists to
    stop.  Pinning the count means a new construction in a sanctioned module
    still trips, and still forces its own adjudication.  Line numbers are
    deliberately NOT pinned: they churn on every unrelated edit above the site,
    and the count already carries the signal.

    *reason* is a POINTER, not a restatement.  The ruling itself is owned by
    ``conftest.make_steward.__doc__`` — the same rationale living in three
    copies is the drift task 3647 exists to end, and re-stating it here would
    just re-create it at a fourth address with nothing asserting the copies
    agree.
    """

    sites: int
    reason: str


# Every module allowed to construct a steward outside conftest's `make_steward`
# factory, keyed by path RELATIVE TO the tests tree (not basename: the sweep is
# a full rglob, so basename keys would let a same-named module in a `fixtures/`
# subdir inherit a sanction it was never adjudicated for).
#
# This mapping IS the adjudication record: task 3647 ruled the split permanent,
# and the census below is what makes that ruling checkable rather than one more
# prose restatement the next consolidation task re-litigates (3461, 3514 and
# 3551 each wrote it down; each successor re-derived it from scratch anyway).
#
# Adding an entry is a DECISION, not a formality — see the failure message.
_SANCTIONED_STEWARD_CONSTRUCTION: dict[str, _Sanctioned] = {
    'conftest.py': _Sanctioned(
        sites=1,
        reason=(
            "the canonical `make_steward` fixture-factory itself — the suite's "
            'one steward factory, and the thing every other site should be using'
        ),
    ),
    'test_workflow_escalated_steward_stall.py': _Sanctioned(
        # Two sites: the `_CapFiringSteward` subclass declaration, and its
        # construction inside `_make_real_steward_factory`.
        sites=2,
        reason=(
            '`_CapFiringSteward`: the PERMANENT exception, examined by task 3551 '
            'and ruled permanent by 3647. Its three structural reasons '
            '(SUBCLASS / `config_dir=` / late-bound worktree callback) are owned '
            'by `conftest.make_steward.__doc__` — read them there, do not '
            'restate them here'
        ),
    ),
    'test_verdict_servers_integration_gate.py': _Sanctioned(
        sites=1,
        reason=(
            '`_build_steward_for_triage`: builds a real `TaskSteward` against a '
            'REAL `OrchestratorConfig` and a real on-disk meta-root, which '
            "`make_steward`'s `spec_set` MagicMock cannot supply. Rationale and "
            'the 2488-postdates-3514 history are owned by '
            '`conftest.make_steward.__doc__`; whether `make_steward` should grow '
            'a `config=` passthrough is left open and filed as ticket '
            'tkt_0RSMX59FSJ27QWSS9VKBYRFMFG (task 3647 owned the ADJUDICATION, '
            'not the redesign)'
        ),
    ),
    'test_workflow_e2e.py': _Sanctioned(
        sites=1,
        reason=(
            'a known FALSE POSITIVE of the suffix rule, and the worked example '
            'of its cost. `_SpyStewardFactory.Steward()` is a nested STUB class '
            'that merely ends in `Steward`; it subclasses nothing and is not a '
            '`TaskSteward` at all, so there is no factory to fold it onto. It is '
            'adjudicated here rather than special-cased in the detector because '
            'the rule matches by NAME and any type-aware exclusion would have to '
            'resolve imports — see `_steward_construction_sites` for why that '
            'trade is deliberate'
        ),
    ),
}


def _steward_construction_sites(tree: ast.Module) -> list[str]:
    """Steward constructions in *tree*: ``<lineno> (<what>)`` for each.

    Three shapes, all structural:

    * an ``ast.Call`` whose func is an ``ast.Name`` ending in ``Steward``
      (``TaskSteward(...)``, ``_CapFiringSteward(...)``);
    * an ``ast.Call`` whose func is an ``ast.Attribute`` whose ``.attr`` ends in
      ``Steward`` (``harness.TaskSteward(...)``, ``_SpyStewardFactory.Steward()``).
      Matching only the Name form would leave the attribute form invisible, so a
      fourth idiom could appear silently through a module-qualified or nested
      name — exactly what this census exists to prevent;
    * an ``ast.ClassDef`` with a base ending in ``Steward`` — a subclass is a
      second steward SHAPE even before it is instantiated, and the standing
      exception is exactly that.  Both the Name and Attribute base forms count,
      which is why the Call branch has to handle both too: the asymmetry would
      be an oversight, not a decision.

    The rule matches by NAME, and that cuts BOTH ways.  Recording both
    directions, because a reader who knows only one will "simplify" the other
    away:

    * FALSE NEGATIVES, all deliberate — ``make_steward`` (lowercase ``s``) does
      not match, so the canonical factory's ~250 call sites are correctly not
      swept in, which is what keeps this rule cheap and stable; nor does
      ``isinstance(x, TaskSteward)`` or a ``-> TaskSteward`` annotation, since a
      bare Name outside a call is not a construction.  A steward built through
      an alias (``S = TaskSteward; S(...)``) or a factory-returned class would
      also slip through; nothing in the tree does that today.
    * FALSE POSITIVES — any identifier ending in ``Steward``, called anywhere,
      matches even when it is not a ``TaskSteward``. ``test_workflow_e2e.py``'s
      ``_SpyStewardFactory.Steward()`` is a live instance: a nested stub class
      that subclasses nothing.  Discriminating by TYPE rather than name would
      mean resolving imports across the tree, which is a far more fragile rule
      than an allowlist entry; so the trade is to accept the false positive and
      adjudicate it, which costs one line and leaves a record.
    """
    sites: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            called = (
                node.func.id if isinstance(node.func, ast.Name)
                else node.func.attr if isinstance(node.func, ast.Attribute)
                else None
            )
            if called is not None and called.endswith('Steward'):
                sites.append(f'{node.lineno} (constructs {called})')
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                name = (
                    base.id if isinstance(base, ast.Name)
                    else base.attr if isinstance(base, ast.Attribute)
                    else None
                )
                if name is not None and name.endswith('Steward'):
                    sites.append(f'{node.lineno} (class {node.name} subclasses {name})')
    return sites


@functools.cache
def _census_by_module() -> Mapping[str, tuple[str, ...]]:
    """``{module path relative to the tests tree: (site, ...)}``, computed once.

    Cached, and returned read-only, because the full-tree walk is the expensive
    half of this module: see ``_scan_tests_tree`` for the cost and for why the
    tests that consume this are collapsed rather than split.
    """
    census: dict[str, tuple[str, ...]] = {}
    for path, tree in _scan_tests_tree().modules:
        sites = _steward_construction_sites(tree)
        if sites:
            census[str(path.relative_to(_TESTS_DIR))] = tuple(sites)
    return MappingProxyType(census)


class TestStewardConstructionSitesAreCensused:
    """Every steward built outside ``make_steward`` is sanctioned, with a reason.

    This is the ENFORCEMENT of DECISION 1 — the ruling that the
    steward-construction split is permanent.  The ruling itself, and every
    structural reason behind it, is owned by ``conftest.make_steward.__doc__``
    and is deliberately not restated here or in the allowlist below; the module
    docstring above records why the lineage's prose kept failing.

    What the census adds is teeth in BOTH directions: a new construction cannot
    appear silently, and its author must either use the factory or write down
    why they cannot — the adjudication 3461, 3514 and 3551 each had to redo from
    scratch.
    """

    def test_every_steward_construction_site_is_sanctioned(self) -> None:
        """The whole census, in ONE full-tree scan.

        Liveness, the unsanctioned-module check, the per-module site COUNT and
        allowlist staleness are asserted together and in that order: a census
        that read nothing sanctions everything, so it must report itself as
        broken before it reports "all clear".  Not split into a test apiece on
        purpose — see ``_scan_tests_tree`` for the cost.
        """
        census = _census_by_module()

        # -- liveness first, in both directions: did we read the tree, and does
        #    the detector still match anything at all?
        assert not _scan_tests_tree().unparseable, (
            f'modules under {_TESTS_DIR} failed to parse and so were silently '
            f'absent from this census — a module that cannot be read cannot be '
            f'censused: {list(_scan_tests_tree().unparseable)}'
        )
        assert len(_scan_tests_tree().modules) >= _MIN_MODULES_SWEPT, (
            f'the census parsed only {len(_scan_tests_tree().modules)} modules '
            f'under {_TESTS_DIR} — expected at least {_MIN_MODULES_SWEPT}. A '
            f'census that reads nothing sanctions everything.'
        )
        assert census, (
            'the census found NO steward construction anywhere, not even '
            "conftest.py's `make_steward` — the detector has stopped matching, "
            'so this guard is vacuously green'
        )

        # -- a module nobody has adjudicated at all.
        unsanctioned = {
            module: sites
            for module, sites in census.items()
            if module not in _SANCTIONED_STEWARD_CONSTRUCTION
        }
        assert not unsanctioned, (
            'Unsanctioned steward-construction site(s).\n'
            'This suite has ONE steward factory: the `make_steward` fixture in '
            'conftest.py (task 3461 merged two copies into it, task 3514 folded '
            'in the two that remained). A construction outside it is a fourth '
            'idiom of the kind this census exists to stop appearing silently.\n'
            'Fix, and it is a real choice between two options:\n'
            '  (a) fold the site onto `make_steward` — extend that fixture '
            'rather than adding a factory beside it; or\n'
            '  (b) if it structurally cannot fold, add the module to '
            '_SANCTIONED_STEWARD_CONSTRUCTION in this file with the REASON '
            'recorded, the way the standing exceptions are recorded there. That '
            'adjudication is this guard\'s whole purpose — an entry with no '
            'reason defeats it.\n'
            f'Unsanctioned: {dict(unsanctioned)}'
        )

        # -- a NEW site inside an already-sanctioned module. Sanctioning a
        #    module wholesale would pre-approve it, which is the same silent
        #    appearance one directory over.
        miscounted = {
            module: (sites, _SANCTIONED_STEWARD_CONSTRUCTION[module].sites)
            for module, sites in census.items()
            if len(sites) != _SANCTIONED_STEWARD_CONSTRUCTION[module].sites
        }
        assert not miscounted, (
            'Steward-construction site COUNT changed in an already-sanctioned '
            'module.\n'
            'The allowlist sanctions a fixed number of sites per module, not the '
            'module wholesale, precisely so a new construction added beside a '
            'sanctioned one still has to be adjudicated rather than inheriting '
            "its neighbour's sanction.\n"
            'Fix: fold the new site onto `make_steward`, or bump the module\'s '
            '`sites=` count and extend its `reason` to cover what you added. If '
            'the count went DOWN, a site was folded or removed — lower the count '
            'to match, so the entry keeps its teeth.\n'
            f'{{module: (found, sanctioned)}}: {miscounted}'
        )

        # -- an entry naming a module that no longer builds a steward is rot: it
        #    silently pre-sanctions whatever that module does next. Same shape as
        #    `test_git_repo_isolation_guard.py`'s check that every
        #    `_SELF_INITIALISING_HELPERS` entry still names a live helper.
        stale = sorted(set(_SANCTIONED_STEWARD_CONSTRUCTION) - set(census))
        assert not stale, (
            f'_SANCTIONED_STEWARD_CONSTRUCTION names {stale}, which no longer '
            f'construct a steward. Remove the entries — a stale sanction '
            f'pre-approves whatever that module builds next, unexamined.'
        )

    def test_the_detector_matches_all_three_construction_shapes(self) -> None:
        """Self-test over synthetic source: a detector that silently stops
        matching reads as coverage.  Kept in string literals so the census
        scanning this module does not self-trip."""
        tree = ast.parse(
            'class _MySteward(TaskSteward):\n'
            '    pass\n'
            '\n'
            'def _build(worktree):\n'
            '    return TaskSteward(task_id="1", worktree=worktree)\n'
        )

        sites = _steward_construction_sites(tree)

        assert len(sites) == 2, sites
        assert any('subclasses TaskSteward' in site for site in sites), sites
        assert any('constructs TaskSteward' in site for site in sites), sites

    def test_the_detector_matches_an_attribute_form_construction(self) -> None:
        """The third shape, and the reason it is not optional: matching only the
        bare-Name call would let a module-qualified or nested construction —
        ``harness.TaskSteward(...)``, ``_Factory.Steward()`` — appear silently,
        which is exactly what this census exists to prevent.  A live instance
        exists in the tree (``test_workflow_e2e.py``), so this is not
        hypothetical."""
        tree = ast.parse(
            'def _build(worktree):\n'
            '    return harness.TaskSteward(task_id="1", worktree=worktree)\n'
        )

        sites = _steward_construction_sites(tree)

        assert len(sites) == 1, sites
        assert 'constructs TaskSteward' in sites[0], sites

    def test_the_detector_ignores_the_canonical_factory_and_non_constructions(self) -> None:
        """Negative self-test: ``make_steward`` is lowercase and must not match,
        nor must a type annotation or an ``isinstance`` check that merely NAMES
        the class.  This is what keeps the rule cheap across ~250 call sites."""
        tree = ast.parse(
            'def test_something(make_steward) -> TaskSteward:\n'
            '    steward = make_steward()\n'
            '    assert isinstance(steward, TaskSteward)\n'
            '    return steward\n'
        )

        assert _steward_construction_sites(tree) == []
