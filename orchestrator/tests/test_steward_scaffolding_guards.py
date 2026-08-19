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

from unittest.mock import MagicMock

import pytest
from _orch_helpers import assert_sandboxed_project_root


class TestAssertSandboxedProjectRoot:
    """The full contract of ``_orch_helpers.assert_sandboxed_project_root``.

    Four clauses, each pinned by at least one test below: the value must be a
    real ``Path``, it must be a CREATED directory, it must not be the sandbox
    root itself, and it must resolve strictly below that root.

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
