"""Tests for ProjectPrefixRegistry — multi-project prefix discovery + collision handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from fused_memory.middleware.project_prefix_registry import (
    DARK_FACTORY_ROOT,
    ProjectPrefixRegistry,
)


def _mkproj(parent: Path, name: str, dirs: list[str]) -> Path:
    """Create a fake project under *parent* with the given top-level *dirs*."""
    root = parent / name
    root.mkdir()
    for d in dirs:
        (root / d).mkdir()
    return root


# ---------------------------------------------------------------------------
# from_roots — basic discovery
# ---------------------------------------------------------------------------


class TestFromRootsBasic:
    def test_empty_root_list_yields_empty_registry(self):
        r = ProjectPrefixRegistry.from_roots([])
        assert r.project_to_root == {}
        assert r.project_to_prefixes == {}
        assert r.prefix_to_project == {}
        assert not r  # __bool__ → False

    def test_single_project_with_one_unique_dir(self, tmp_path):
        p = _mkproj(tmp_path, 'reify', ['crates'])
        r = ProjectPrefixRegistry.from_roots([str(p)])
        assert r.project_to_root == {'reify': str(p.resolve())}
        assert r.project_to_prefixes == {'reify': ('crates/',)}
        assert r.prefix_to_project == {'crates/': 'reify'}
        assert bool(r) is True

    def test_hyphenated_dir_gets_underscore_alias(self, tmp_path):
        p = _mkproj(tmp_path, 'dark-factory', ['fused-memory'])
        r = ProjectPrefixRegistry.from_roots([str(p)])
        assert r.project_to_prefixes['dark_factory'] == ('fused-memory/', 'fused_memory/')
        assert r.prefix_to_project['fused-memory/'] == 'dark_factory'
        assert r.prefix_to_project['fused_memory/'] == 'dark_factory'

    def test_generic_dirs_are_filtered_out(self, tmp_path):
        # src/, tests/, docs/, scripts/, hooks/, review/ should all be skipped.
        p = _mkproj(
            tmp_path, 'project',
            ['src', 'tests', 'docs', 'scripts', 'hooks', 'review', 'real_subproject'],
        )
        r = ProjectPrefixRegistry.from_roots([str(p)])
        assert r.project_to_prefixes['project'] == ('real_subproject/',)

    def test_hidden_dirs_filtered_out(self, tmp_path):
        p = _mkproj(tmp_path, 'project', ['.git', '.venv', 'real'])
        r = ProjectPrefixRegistry.from_roots([str(p)])
        assert r.project_to_prefixes['project'] == ('real/',)

    def test_files_at_top_level_ignored(self, tmp_path):
        p = _mkproj(tmp_path, 'project', ['real_dir'])
        (p / 'README.md').write_text('hi')
        (p / 'pyproject.toml').write_text('[project]')
        r = ProjectPrefixRegistry.from_roots([str(p)])
        assert r.project_to_prefixes['project'] == ('real_dir/',)


# ---------------------------------------------------------------------------
# from_roots — collisions
# ---------------------------------------------------------------------------


class TestFromRootsCollisions:
    def test_shared_prefix_dropped_from_both_projects(self, tmp_path):
        """When two projects both have a `tools/` dir, that prefix is dropped."""
        a = _mkproj(tmp_path, 'reify', ['crates', 'tools'])
        b = _mkproj(tmp_path, 'autopilot-video', ['pipeline', 'tools'])
        r = ProjectPrefixRegistry.from_roots([str(a), str(b)])
        # tools/ removed from both project_to_prefixes entries
        assert 'tools/' not in r.prefix_to_project
        assert 'tools/' not in r.project_to_prefixes['reify']
        assert 'tools/' not in r.project_to_prefixes['autopilot_video']
        # Unique prefixes survive
        assert 'crates/' in r.prefix_to_project
        assert r.prefix_to_project['crates/'] == 'reify'
        assert 'pipeline/' in r.prefix_to_project
        assert r.prefix_to_project['pipeline/'] == 'autopilot_video'

    def test_underscore_alias_collision_treated_independently(self, tmp_path):
        """fused-memory/ and fused_memory/ are independent prefix entries —
        a collision on one doesn't drop the other.
        """
        # Project A has 'fused-memory' (hyphen alias generated) — yields both fused-memory/ and fused_memory/
        a = _mkproj(tmp_path, 'a', ['fused-memory'])
        # Project B has only 'fused_memory' (no hyphen → no alias) — yields just fused_memory/
        b = _mkproj(tmp_path, 'b', ['fused_memory'])
        r = ProjectPrefixRegistry.from_roots([str(a), str(b)])
        # fused_memory/ collides → dropped from both
        assert 'fused_memory/' not in r.prefix_to_project
        # fused-memory/ unique to a → survives
        assert r.prefix_to_project.get('fused-memory/') == 'a'

    def test_duplicate_project_id_keeps_first(self, tmp_path):
        """Two roots resolving to the same project_id: first wins; second skipped."""
        a = _mkproj(tmp_path, 'project_x', ['unique_a'])
        # Different parent, same name → same project_id.
        sub = tmp_path / 'sub'
        sub.mkdir()
        b = _mkproj(sub, 'project_x', ['unique_b'])
        r = ProjectPrefixRegistry.from_roots([str(a), str(b)])
        assert r.project_to_root == {'project_x': str(a.resolve())}
        # Only first project's prefixes registered.
        assert r.prefix_to_project == {'unique_a/': 'project_x'}


# ---------------------------------------------------------------------------
# from_roots — rename-stable declared-id resolution
# ---------------------------------------------------------------------------


class TestFromRootsRenameStable:
    """A directory whose basename differs from the project_id declared in its
    dark-factory-orchestrator.yaml (the exact shape a project-dir rename
    produces) registers — and owns its prefixes — under the DECLARED id, so
    the path-scope guard stops spuriously rejecting the renamed project's
    paths under the wrong basename-derived id.
    """

    def test_renamed_dir_owns_prefixes_under_declared_id(self, tmp_path):
        # basename `solar-challenge` → basename-derived id 'solar_challenge',
        # but the manifest declares 'my_solar_challenge'.
        root = _mkproj(tmp_path, 'solar-challenge', ['firmware'])
        (root / 'dark-factory-orchestrator.yaml').write_text(
            'project_id: "my_solar_challenge"\n',
        )
        r = ProjectPrefixRegistry.from_roots([str(root)])

        assert r.is_known('my_solar_challenge') is True
        assert r.root_for_project('my_solar_challenge') == str(root.resolve())
        assert r.project_for_prefix('firmware/') == 'my_solar_challenge'
        # The stale basename-derived id must NOT appear anywhere.
        assert r.is_known('solar_challenge') is False
        assert r.project_for_prefix('firmware/') != 'solar_challenge'

    def test_no_manifest_still_keys_by_basename(self, tmp_path):
        """Back-compat: a project that declares no id keys by basename."""
        root = _mkproj(tmp_path, 'plain-proj', ['firmware'])
        r = ProjectPrefixRegistry.from_roots([str(root)])
        assert r.is_known('plain_proj') is True
        assert r.project_for_prefix('firmware/') == 'plain_proj'


# ---------------------------------------------------------------------------
# Lookups + helpers
# ---------------------------------------------------------------------------


class TestRegistryLookups:
    @pytest.fixture
    def registry(self, tmp_path):
        a = _mkproj(tmp_path, 'reify', ['crates'])
        b = _mkproj(tmp_path, 'dark-factory', ['fused-memory'])
        return ProjectPrefixRegistry.from_roots([str(a), str(b)])

    def test_project_for_prefix(self, registry):
        assert registry.project_for_prefix('crates/') == 'reify'
        assert registry.project_for_prefix('fused-memory/') == 'dark_factory'
        assert registry.project_for_prefix('fused_memory/') == 'dark_factory'
        assert registry.project_for_prefix('unknown/') is None

    def test_root_for_project(self, registry, tmp_path):
        assert registry.root_for_project('reify') == str((tmp_path / 'reify').resolve())
        assert registry.root_for_project('dark_factory') == str(
            (tmp_path / 'dark-factory').resolve(),
        )
        assert registry.root_for_project('nope') is None

    def test_is_known(self, registry):
        assert registry.is_known('reify')
        assert registry.is_known('dark_factory')
        assert not registry.is_known('autopilot_video')

    def test_all_prefixes_sorted(self, registry):
        prefixes = registry.all_prefixes()
        assert prefixes == tuple(sorted(prefixes))
        # Spot-check membership.
        assert 'crates/' in prefixes
        assert 'fused-memory/' in prefixes


# ---------------------------------------------------------------------------
# project_for_path — exact leading-path-component ownership lookup
# ---------------------------------------------------------------------------


class TestProjectForPath:
    """Tests for ProjectPrefixRegistry.project_for_path.

    Unlike project_for_prefix (exact prefix-string lookup) or the guard's
    find_paths (regex-over-prose), this resolves an arbitrary file path to
    its owning project via leading-path-component matching — the CERTAIN
    signal that check_files_for_scope (task 2206) relies on for rejection.
    """

    @pytest.fixture
    def registry(self, tmp_path):
        a = _mkproj(tmp_path, 'reify', ['crates'])
        b = _mkproj(tmp_path, 'dark-factory', ['fused-memory', 'orchestrator'])
        return ProjectPrefixRegistry.from_roots([str(a), str(b)])

    def test_owning_project_for_simple_path(self, registry):
        assert registry.project_for_path('orchestrator/foo.py') == 'dark_factory'

    def test_hyphenated_prefix_and_underscore_alias_both_resolve(self, registry):
        assert registry.project_for_path('fused-memory/src/x.py') == 'dark_factory'
        assert registry.project_for_path('fused_memory/x.py') == 'dark_factory'

    def test_exact_dir_with_no_trailing_content(self, registry):
        assert registry.project_for_path('crates') == 'reify'

    def test_unowned_leading_component_returns_none(self, registry):
        assert registry.project_for_path('unknown_dir/x') is None

    def test_empty_string_returns_none(self, registry):
        assert registry.project_for_path('') is None

    def test_none_ish_returns_none(self, registry):
        """An actual None (defensive — callers should only pass str) is None-ish too."""
        assert registry.project_for_path(None) is None

    def test_startswith_but_different_component_returns_none(self, registry):
        """'cratesfoo/x' merely starts with the string 'crates' but is a
        DIFFERENT leading component — must not match the 'crates/' prefix."""
        assert registry.project_for_path('cratesfoo/x') is None

    def test_prefix_mid_path_returns_none(self, registry):
        """A prefix appearing as a non-leading segment must not match —
        leading-component only, not substring/regex."""
        assert registry.project_for_path('vendor/crates/x') is None

    def test_leading_dot_slash_is_stripped(self, registry):
        assert registry.project_for_path('./orchestrator/x') == 'dark_factory'

    def test_longest_prefix_wins(self):
        """When more than one registered prefix leads file_path, the longest wins.

        Hand-built (not via from_roots/_mkproj): real directory-derived
        prefixes are always single path components, so two of them can never
        both be a leading run of the same file_path — this exercises the
        tie-break directly.
        """
        registry = ProjectPrefixRegistry(
            prefix_to_project={'a/': 'short_owner', 'a/b/': 'long_owner'},
        )
        assert registry.project_for_path('a/b/c.py') == 'long_owner'


# ---------------------------------------------------------------------------
# project_for_path — ABSOLUTE regime, resolved against known project roots
# (task 3109)
#
# Before task 3109 an absolute path could never match a registered prefix
# (prefixes are top-level dir names; an absolute path leads with '/'), so it
# always came back None = "unowned", silently disarming the path-scope guard
# for exactly the file shape an agent produces after reading a foreign repo.
# ---------------------------------------------------------------------------


class TestProjectForPathAbsolute:
    """Absolute paths resolve against ``project_to_root``, root-authoritatively.

    Two disjoint regimes: absolute candidates are matched against known
    project ROOTS (component boundary, longest-root-wins); relative
    candidates keep the untouched leading-path-component prefix scan.
    """

    @pytest.fixture
    def registry(self, tmp_path):
        a = _mkproj(tmp_path, 'reify', ['crates'])
        b = _mkproj(tmp_path, 'dark-factory', ['fused-memory', 'orchestrator', 'docs'])
        return ProjectPrefixRegistry.from_roots([str(a), str(b)])

    def test_project_for_path_absolute_under_known_root_resolves_owner(self, registry):
        """The incident-shaped absolute paths all resolve to dark_factory.

        Includes a path under a _GENERIC_DIRS directory ('docs/') and a
        repo-root-level file ('README.md'), pinning the ROOT-AUTHORITATIVE
        semantics: an absolute path under a known root is owned even when its
        remainder matches NO registered prefix. The mirrored reify direction
        pins that the fix is symmetric, not dark-factory-special.
        """
        df = registry.root_for_project('dark_factory')
        reify = registry.root_for_project('reify')
        assert df is not None and reify is not None

        assert registry.project_for_path(
            f'{df}/orchestrator/src/orchestrator/git_ops.py') == 'dark_factory'
        assert registry.project_for_path(
            f'{df}/fused-memory/src/fused_memory/middleware/task_interceptor.py',
        ) == 'dark_factory'
        # _GENERIC_DIRS member — unowned in its RELATIVE spelling, owned here.
        assert registry.project_for_path(f'{df}/docs/task-authoring.md') == 'dark_factory'
        assert registry.project_for_path('docs/task-authoring.md') is None
        # Repo-root-level file — no leading prefix at all.
        assert registry.project_for_path(f'{df}/README.md') == 'dark_factory'

        assert registry.project_for_path(f'{reify}/crates/foo.rs') == 'reify'

    def test_longest_root_wins_for_nested_roots(self):
        """Nested roots (a project root and a worktree root beneath it) are real.

        Hand-built so dict insertion order puts the SHORTER root first — a
        first-match scan would deterministically return 'outer'. Mirrors the
        relative scan's ``test_longest_prefix_wins`` tiebreak.
        """
        registry = ProjectPrefixRegistry(
            project_to_root={'outer': '/a', 'inner': '/a/b'},
        )
        assert registry.project_for_path('/a/b/c.py') == 'inner'
        assert registry.project_for_path('/a/z.py') == 'outer'

    def test_absolute_sibling_root_prefix_does_not_match(self, registry):
        """A sibling root sharing a string prefix must not match (component boundary)."""
        df = registry.root_for_project('dark_factory')
        assert registry.project_for_path(f'{df}-old/orchestrator/x.py') is None

    def test_absolute_path_equal_to_root_returns_owner(self, registry):
        """Mirrors test_exact_dir_with_no_trailing_content for the absolute case."""
        df = registry.root_for_project('dark_factory')
        assert registry.project_for_path(df) == 'dark_factory'

    def test_absolute_path_under_no_known_root_returns_none(self, registry):
        """Genuinely unclassifiable stays unowned — even with a registered
        prefix ('orchestrator/') appearing inside the path."""
        assert registry.project_for_path('/var/tmp/somewhere/orchestrator/x.py') is None

    def test_empty_or_slash_only_root_never_owns(self):
        """Degenerate roots must not turn startswith('' + '/') into a universal match."""
        registry = ProjectPrefixRegistry(
            project_to_root={'bogus': '/', 'other': ''},
        )
        assert registry.project_for_path('/anything/x.py') is None

    def test_relative_path_behaviour_unchanged(self, registry):
        """Regression block: task 3109's ABSOLUTE work left these unchanged.

        Scoped to 3109 on purpose — it does NOT claim the relative regime is
        untouched in general. Task 4156 added lexical normalisation there, and
        every assertion below survives it unchanged (none of these spellings
        carries a '..' segment or a redundant separator). See
        TestRelativePathNormalisation for what 4156 did change.
        """
        assert registry.project_for_path('orchestrator/foo.py') == 'dark_factory'
        assert registry.project_for_path('crates') == 'reify'
        assert registry.project_for_path('./orchestrator/x') == 'dark_factory'
        # task 1494: component boundary + non-leading segment
        assert registry.project_for_path('cratesfoo/x') is None
        assert registry.project_for_path('vendor/crates/x') is None
        # task 2434: generic-dir denylist
        assert registry.project_for_path('docs/task-authoring.md') is None
        assert registry.project_for_path('') is None
        assert registry.project_for_path(None) is None

    def test_default_registry_resolves_absolute_dark_factory_path(self):
        """DARK_FACTORY_ROOT is now classification-load-bearing under default().

        Single-project deployments (no known_project_roots configured) get
        absolute-path ownership from the built-in root constant. A checkout at
        a different path simply fails to match — fail-OPEN to the pre-3109
        verdict, never a false rejection.

        The positive path is DERIVED from the constant rather than hardcoded:
        the constant's own comment invites an operator to update it for a
        differently-located checkout, which must not break a test whose
        semantics (default registry owns absolute paths under its own root;
        unrelated roots stay unowned) remain true. The negative case keeps an
        unrelated literal root on purpose.
        """
        registry = ProjectPrefixRegistry.default()
        assert registry.project_for_path(
            f'{DARK_FACTORY_ROOT}/orchestrator/x.py') == 'dark_factory'
        assert registry.project_for_path('/home/leo/src/reify/crates/x.rs') is None


# ---------------------------------------------------------------------------
# Absolute-path NORMALISATION (task 3109 amendment).
#
# `_owner_for_absolute_path` compares lexically, so unnormalised spellings of
# a real path defeat the comparison in BOTH directions unless normalised
# first: a '..' segment produces a FALSE rejection (a sibling-repo file
# matching the root by bare startswith), duplicate separators produce a MISSED
# rejection.  os.path.normpath fixes both purely lexically — no symlink
# resolution, no stat — so the deliberate no-I/O property is preserved.
# ---------------------------------------------------------------------------


class TestAbsolutePathNormalisation:
    @pytest.fixture
    def registry(self, tmp_path):
        a = _mkproj(tmp_path, 'reify', ['crates'])
        b = _mkproj(tmp_path, 'dark-factory', ['fused-memory', 'orchestrator', 'docs'])
        return ProjectPrefixRegistry.from_roots([str(a), str(b)])

    def test_parent_segment_does_not_produce_false_ownership(self, registry):
        """'<df_root>/../reify/crates/x.rs' is a REIFY file, not a DF one.

        Without normalisation this matched the dark-factory root by bare
        ``startswith`` and rejected a legitimately-local file — a FALSE
        rejection, the one failure direction the docstrings promise never
        happens.
        """
        df = registry.root_for_project('dark_factory')
        assert df is not None
        escaped = f'{df}/../reify/crates/x.rs'
        assert registry.project_for_path(escaped) == 'reify'
        assert registry.project_for_path(escaped) != 'dark_factory'

    def test_parent_segment_escaping_all_known_roots_is_unowned(self, registry):
        df = registry.root_for_project('dark_factory')
        assert df is not None
        assert registry.project_for_path(f'{df}/../../elsewhere/x.py') is None

    def test_duplicate_separators_still_resolve_owner(self, registry):
        """'//' inside the path must not silently disarm the guard."""
        df = registry.root_for_project('dark_factory')
        assert df is not None
        head, _, tail = df.rpartition('/')
        assert registry.project_for_path(
            f'{head}//{tail}/orchestrator/x.py') == 'dark_factory'
        assert registry.project_for_path(
            f'{df}//orchestrator//x.py') == 'dark_factory'

    def test_trailing_slash_on_root_spelling_resolves_owner(self, registry):
        df = registry.root_for_project('dark_factory')
        assert df is not None
        assert registry.project_for_path(f'{df}/') == 'dark_factory'

    def test_default_registry_normalises_both_directions(self):
        """The same two shapes on the built-in single-project registry."""
        registry = ProjectPrefixRegistry.default()
        head, _, tail = DARK_FACTORY_ROOT.rpartition('/')
        assert registry.project_for_path(
            f'{head}//{tail}/orchestrator/x.py') == 'dark_factory'
        # A sibling-repo file reached via '..' must NOT be claimed.
        assert registry.project_for_path(
            f'{DARK_FACTORY_ROOT}/../reify/crates/x.rs') is None

    def test_unnormalised_root_in_hand_built_registry_still_matches(self):
        """Roots are normalised too — from_roots resolves them, but a
        hand-built or config-sourced registry need not be clean."""
        registry = ProjectPrefixRegistry(
            project_to_root={'proj': '/a//b/c/../'},
        )
        assert registry.project_for_path('/a/b/x.py') == 'proj'
        assert registry.project_for_path('/a/b') == 'proj'
        assert registry.project_for_path('/a/bc/x.py') is None

    def test_tilde_path_resolves_like_its_absolute_spelling(self, registry, monkeypatch):
        """'~/dark-factory/orchestrator/x.py' is the same file as its absolute
        spelling — an agent that has been shelling around emits it that way.

        HOME is derived from the RESOLVED root so the expansion lines up with
        what ``from_roots`` stored (matters if tmp_path is ever a symlink).
        """
        df = registry.root_for_project('dark_factory')
        assert df is not None
        monkeypatch.setenv('HOME', str(Path(df).parent))

        assert registry.project_for_path(
            '~/dark-factory/orchestrator/x.py') == 'dark_factory'
        assert registry.project_for_path('~/reify/crates/x.rs') == 'reify'
        assert registry.project_for_path('~/unknown-proj/x.py') is None
        # Bare '~' expands to the HOME dir, which is under no known root.
        assert registry.project_for_path('~') is None

    def test_known_unhandled_spellings_fail_open(self, registry, monkeypatch):
        """Documented residue: spellings that stay UNOWNED by design.

        Each needs something this resolver deliberately does not have — a base
        directory for parent-relative paths, an NSS lookup for '~user', or a
        POSIX interpretation of a leading '//'. All three degrade fail-OPEN (a
        missed rejection, never a false one). Pinned so the gap is visible to
        the next reader rather than implied covered by the docstrings.
        """
        df = registry.root_for_project('dark_factory')
        assert df is not None
        monkeypatch.setenv('HOME', str(Path(df).parent))

        # Parent-relative: unresolvable without a base directory.
        assert registry.project_for_path('../dark-factory/orchestrator/x.py') is None
        # '~user/' would need an NSS/passwd lookup (I/O) to expand.
        assert registry.project_for_path('~nobody/dark-factory/orchestrator/x.py') is None
        # POSIX reserves a leading '//'; normpath preserves it by design.
        assert registry.project_for_path(f'/{df}/orchestrator/x.py') is None

    def test_roots_longest_first_is_precomputed_and_ordered(self, registry):
        """The per-lookup work is hoisted to a cached, ordered table.

        Longest-root-wins now lives in the ORDERING (first match wins), and
        degenerate roots are filtered once at build time rather than on every
        lookup of the hot submit path.
        """
        pairs = registry._roots_longest_first
        assert pairs is registry._roots_longest_first, 'expected a cached table'
        lengths = [len(root) for root, _ in pairs]
        assert lengths == sorted(lengths, reverse=True)
        assert {pid for _, pid in pairs} == {'reify', 'dark_factory'}
        assert all(root and not root.endswith('/') for root, _ in pairs)

        degenerate = ProjectPrefixRegistry(
            project_to_root={'bogus': '/', 'blank': '', 'dots': '/a/..'},
        )
        assert degenerate._roots_longest_first == ()


# ---------------------------------------------------------------------------
# Relative-path NORMALISATION (task 4156).
#
# The sibling of TestAbsolutePathNormalisation above, for the OTHER regime.
# The relative prefix scan compares just as lexically as the absolute one, so
# unnormalised spellings defeat it in the SAME two directions: a '..' segment
# produces a FALSE ownership ('orchestrator/../elsewhere/x.py' matched the
# prefix 'orchestrator/' by bare startswith and was claimed for dark_factory),
# and the mirrored case produces a MISSED one ('foo/../orchestrator/x.py' IS
# orchestrator/x.py but matched nothing).  The absolute regime already fixed
# both with os.path.normpath at :417; this regime now makes the identical
# call, so the two read as symmetric rather than merely both-patched — still
# purely lexical, no symlink resolution, no stat, no CWD.
# ---------------------------------------------------------------------------


class TestRelativePathNormalisation:
    @pytest.fixture
    def registry(self, tmp_path):
        a = _mkproj(tmp_path, 'reify', ['crates'])
        b = _mkproj(tmp_path, 'dark-factory', ['fused-memory', 'orchestrator', 'docs'])
        return ProjectPrefixRegistry.from_roots([str(a), str(b)])

    def test_parent_segment_does_not_produce_false_ownership(self, registry):
        """A '..' segment that escapes a registered prefix must not be claimed.

        Each escaped path is asserted against its already-correct CONTROL
        spelling, so the test states the EQUIVALENCE ('orchestrator/../x' is
        the same file as 'x') rather than merely the negation.
        """
        assert registry.project_for_path('somewhere-else/x.py') is None  # control
        assert registry.project_for_path('orchestrator/../somewhere-else/x.py') is None

        assert registry.project_for_path('elsewhere/y.py') is None  # control
        assert registry.project_for_path('./orchestrator/../elsewhere/y.py') is None

        # Normalises to '.', which is under no registered prefix.
        assert registry.project_for_path('orchestrator/..') is None

        # Escapes past the notional base entirely — normalises to '../...'.
        assert registry.project_for_path('orchestrator/../../elsewhere/x.py') is None

    def test_parent_segment_does_not_misattribute_across_projects(self, registry):
        """'orchestrator/../crates/foo.rs' is a REIFY file, not a dark-factory one.

        The relative twin of test_parent_segment_does_not_produce_false_ownership
        in TestAbsolutePathNormalisation: without normalisation this matched the
        prefix 'orchestrator/' by bare startswith and hard-rejected a
        legitimately-local file — a FALSE rejection, the one failure direction
        the module docstrings promise never happens.
        """
        escaped = 'orchestrator/../crates/foo.rs'
        assert registry.project_for_path(escaped) == 'reify'
        assert registry.project_for_path(escaped) != 'dark_factory'

    def test_parent_segment_back_into_a_prefix_resolves_owner(self, registry):
        """The mirrored MISS: 'foo/../orchestrator/x.py' IS orchestrator/x.py.

        Symmetric normalisation necessarily closes both directions, so the fix
        also creates rejections that previously did not fire. Pinned explicitly
        so the widened-rejection behaviour is a recorded decision rather than an
        unnoticed side effect — the analogue of the duplicate-separator miss
        test_duplicate_separators_still_resolve_owner closed for the absolute
        regime.
        """
        assert registry.project_for_path('foo/../orchestrator/x.py') == 'dark_factory'
        assert registry.project_for_path('orchestrator/x.py') == 'dark_factory'  # control

    def test_known_unhandled_spellings_fail_open(self, registry):
        """Documented residue: a LEADING '..' stays UNOWNED by design.

        The relative regime is given no base directory, so '../foo/x.py' is
        genuinely unresolvable — leaving it unowned is a missed rejection, never
        a false one, the same fail-open direction _owner_for_absolute_path
        documents. Pinned so the gap stays visible to the next reader rather
        than implied covered by the docstrings.

        This block is also the regression guard on HOW the normalisation is
        implemented: it fails if a future implementer reaches for
        Path.resolve()/os.path.realpath, resolves against the process CWD, or
        strips leading '..' segments — each of which would make ownership depend
        on where the interceptor happens to be running.
        """
        assert registry.project_for_path('../dark-factory/orchestrator/x.py') is None
        assert registry.project_for_path('../crates/x.rs') is None
        assert registry.project_for_path('..') is None
        # Normalises to '..' — the residue is about the RESULT, not the spelling.
        assert registry.project_for_path('a/../..') is None

    def test_redundant_separators_still_resolve_owner(self, registry):
        """Symmetry with the absolute regime, pinned so the two cannot drift."""
        assert registry.project_for_path('orchestrator//x.py') == 'dark_factory'
        assert registry.project_for_path('orchestrator/./x.py') == 'dark_factory'

    def test_default_registry_normalises_relative_parent_segments(self):
        """The same shapes on the built-in single-project registry.

        Single-project deployments (no known_project_roots configured) reach
        project_for_path through default(), so they must not keep the defect.
        """
        registry = ProjectPrefixRegistry.default()
        assert registry.project_for_path(
            'orchestrator/../somewhere-else/x.py') is None
        assert registry.project_for_path(
            './orchestrator/../elsewhere/y.py') is None


# ---------------------------------------------------------------------------
# Edge: non-existent root does not crash
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_nonexistent_root_yields_empty_prefixes_for_that_project(self, tmp_path):
        ghost = tmp_path / 'does_not_exist'
        r = ProjectPrefixRegistry.from_roots([str(ghost)])
        # The project_id is registered (we resolve the path even if it doesn't exist),
        # but with no prefixes.
        assert 'does_not_exist' in r.project_to_root
        assert r.project_to_prefixes['does_not_exist'] == ()

    def test_root_pointing_at_a_file_yields_empty_prefixes(self, tmp_path):
        f = tmp_path / 'just_a_file'
        f.write_text('not a dir')
        r = ProjectPrefixRegistry.from_roots([str(f)])
        assert r.project_to_prefixes['just_a_file'] == ()


# ---------------------------------------------------------------------------
# default() — built-in, filesystem-independent dark_factory registry.
#
# Folds in the retired back-compat shim's hard-coded constants (task 2208 /
# PRD D2) so the guard always classifies dark-factory paths even when no
# known_project_roots are configured.
# ---------------------------------------------------------------------------


class TestDefaultRegistry:
    _EXPECTED_PREFIXES = (
        'orchestrator/',
        'fused-memory/',
        'fused_memory/',
        'mem0/',
        'graphiti/',
        'dashboard/',
    )

    def test_default_registry_is_truthy(self):
        assert bool(ProjectPrefixRegistry.default()) is True

    def test_default_registry_carries_dark_factory_prefixes(self):
        r = ProjectPrefixRegistry.default()
        assert r.project_to_prefixes['dark_factory'] == self._EXPECTED_PREFIXES
        for prefix in self._EXPECTED_PREFIXES:
            assert prefix in r.all_prefixes()

    def test_project_for_prefix_resolves_dark_factory(self):
        r = ProjectPrefixRegistry.default()
        assert r.project_for_prefix('orchestrator/') == 'dark_factory'

    def test_project_for_path_resolves_dark_factory(self):
        r = ProjectPrefixRegistry.default()
        assert r.project_for_path('fused-memory/src/x.py') == 'dark_factory'
        assert r.project_for_path('mem0/foo.py') == 'dark_factory'

    def test_root_for_project(self):
        r = ProjectPrefixRegistry.default()
        assert r.root_for_project('dark_factory') == '/home/leo/src/dark-factory'

    def test_is_known(self):
        assert ProjectPrefixRegistry.default().is_known('dark_factory')

    def test_module_exposes_moved_constants(self):
        from fused_memory.middleware import project_prefix_registry as mod

        assert mod.DARK_FACTORY_PROJECT_ID == 'dark_factory'
        assert mod.DARK_FACTORY_ROOT == '/home/leo/src/dark-factory'
        assert mod.DARK_FACTORY_PATH_PREFIXES == self._EXPECTED_PREFIXES
