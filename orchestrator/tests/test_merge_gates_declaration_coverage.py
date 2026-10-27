"""Pure coverage of ``declaration_coverage``: declared entries vs landed paths.

The predicate is the git-free converse of the plan-files gate's coverage
rule, so these cases pin the shared exact + directory arms, the repo-root
guard, and the documented boundary that a rename is not resolved.
"""

from __future__ import annotations

from orchestrator.merge_lane.gates import DeclarationCoverage, declaration_coverage


class TestExactArm:
    def test_landed_path_equal_to_declared_entry_is_covered(self) -> None:
        assert declaration_coverage(['src/a.py'], ['src/a.py']) == DeclarationCoverage(
            undeclared=(), undelivered=(),
        )

    def test_dot_slash_entry_is_normalized(self) -> None:
        assert declaration_coverage(['./src/a.py'], ['src/a.py']) == DeclarationCoverage(
            undeclared=(), undelivered=(),
        )

    def test_redundant_separator_entry_is_normalized(self) -> None:
        assert declaration_coverage(['src//a.py'], ['src/a.py']) == DeclarationCoverage(
            undeclared=(), undelivered=(),
        )


class TestDirectoryArm:
    def test_bare_directory_entry_covers_paths_beneath_it(self) -> None:
        assert declaration_coverage(
            ['src/pkg'], ['src/pkg/m.py', 'src/pkg/sub/n.py'],
        ) == DeclarationCoverage(undeclared=(), undelivered=())

    def test_trailing_slash_directory_entry_covers_paths_beneath_it(self) -> None:
        assert declaration_coverage(
            ['src/pkg/'], ['src/pkg/m.py', 'src/pkg/sub/n.py'],
        ) == DeclarationCoverage(undeclared=(), undelivered=())

    def test_sibling_prefix_is_not_covered(self) -> None:
        assert declaration_coverage(
            ['src/pkg'], ['src/pkg/m.py', 'src/pkgx/b.py'],
        ) == DeclarationCoverage(undeclared=('src/pkgx/b.py',), undelivered=())


class TestUndelivered:
    def test_undelivered_entry_is_reported_as_written(self) -> None:
        assert declaration_coverage(
            ['./docs/x.md', 'src/a.py'], ['src/a.py'],
        ) == DeclarationCoverage(undeclared=(), undelivered=('./docs/x.md',))

    def test_directory_entry_with_a_landing_beneath_is_delivered(self) -> None:
        assert declaration_coverage(
            ['src/pkg/', 'src/b.py'], ['src/pkg/m.py'],
        ) == DeclarationCoverage(undeclared=(), undelivered=('src/b.py',))


class TestDegenerateInputs:
    def test_empty_declaration_leaves_every_path_undeclared_sorted(self) -> None:
        assert declaration_coverage(
            [], ['src/z.py', 'src/a.py'],
        ) == DeclarationCoverage(undeclared=('src/a.py', 'src/z.py'), undelivered=())

    def test_empty_paths_leave_every_declared_entry_undelivered(self) -> None:
        assert declaration_coverage(
            ['src/z.py', '', 'src/a.py'], [],
        ) == DeclarationCoverage(undeclared=(), undelivered=('src/a.py', 'src/z.py'))

    def test_empty_string_entry_is_ignored(self) -> None:
        assert declaration_coverage(
            ['', 'src/a.py'], ['src/a.py'],
        ) == DeclarationCoverage(undeclared=(), undelivered=())

    def test_duplicates_appear_once(self) -> None:
        assert declaration_coverage(
            ['docs/x.md', 'docs/x.md'], ['src/a.py', 'src/a.py'],
        ) == DeclarationCoverage(undeclared=('src/a.py',), undelivered=('docs/x.md',))


class TestRepoRootGuard:
    def test_dot_entry_covers_nothing(self) -> None:
        assert declaration_coverage(['.'], ['a.py']) == DeclarationCoverage(
            undeclared=('a.py',), undelivered=('.',),
        )

    def test_dot_slash_entry_covers_nothing(self) -> None:
        assert declaration_coverage(['./'], ['a.py']) == DeclarationCoverage(
            undeclared=('a.py',), undelivered=('./',),
        )


class TestRenameBoundary:
    def test_rename_is_not_resolved(self) -> None:
        assert declaration_coverage(
            ['old/a.py'], ['new/a.py', 'old/a.py'],
        ) == DeclarationCoverage(undeclared=('new/a.py',), undelivered=())

    def test_renamed_declared_path_reads_as_undelivered_plus_undeclared(self) -> None:
        assert declaration_coverage(['old/a.py'], ['new/a.py']) == DeclarationCoverage(
            undeclared=('new/a.py',), undelivered=('old/a.py',),
        )
