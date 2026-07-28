"""Tests for the multi-project path-scope guard.

Covers behaviour beyond the dark_factory-only default registry: per-project
prefix registries, suggested_project derivation, multi-project mismatches,
empty-registry short-circuit, and the prompt-only ``check_text_for_scope``
branch. (The back-compat shim's own test module was retired alongside the
shim in task 2208 / PRD D2 — see
``test_project_prefix_registry.py::TestDefaultRegistry`` for the folded-in
dark_factory constants coverage.)
"""

from __future__ import annotations

from pathlib import Path

from fused_memory.middleware.path_scope_guard import (
    PathGuardVerdict,
    check_candidate_for_scope,
    check_files_for_scope,
    check_text_for_scope,
    find_paths,
    is_routing_override,
)
from fused_memory.middleware.project_prefix_registry import ProjectPrefixRegistry
from fused_memory.middleware.task_curator import CandidateTask


def _mkproj(parent: Path, name: str, dirs: list[str]) -> Path:
    root = parent / name
    root.mkdir()
    for d in dirs:
        (root / d).mkdir()
    return root


def _two_project_registry(tmp_path: Path) -> ProjectPrefixRegistry:
    """Reify (crates/, gui/) + dark-factory (fused-memory/, orchestrator/)."""
    a = _mkproj(tmp_path, 'reify', ['crates', 'gui'])
    b = _mkproj(tmp_path, 'dark-factory', ['fused-memory', 'orchestrator'])
    return ProjectPrefixRegistry.from_roots([str(a), str(b)])


def _candidate(
    title: str = '',
    description: str = '',
    details: str = '',
    files_to_modify: list[str] | None = None,
) -> CandidateTask:
    """Build a real CandidateTask with only the fields the guard reads."""
    return CandidateTask(
        title=title,
        description=description,
        details=details,
        files_to_modify=files_to_modify or [],
        priority='medium',
    )


# ---------------------------------------------------------------------------
# find_paths — generalised regex
# ---------------------------------------------------------------------------


class TestFindPaths:
    def test_empty_text_returns_empty(self):
        assert find_paths('', ('foo/',)) == []

    def test_empty_prefixes_returns_empty(self):
        assert find_paths('lots of text', ()) == []

    def test_single_match(self):
        assert find_paths('see crates/x.rs', ('crates/',)) == ['crates/']

    def test_dedup_and_order(self):
        """RE-POINTED by task 3120: the single-bare-segment shape is now
        deliberately NOT a match.

        ``crates/a`` has no right-context that looks like a path segment (no
        further '/', no file extension), so under the right-boundary contract
        it no longer lexes as a path.  This is a synthetic fixture with no
        witness in the live corpus.  Dedup-and-order over *genuine* multi-hit
        text is retained in
        ``TestFindPathsRightBoundary::test_dedup_and_order_retained``.
        """
        result = find_paths('crates/a then gui/b then crates/c', ('crates/', 'gui/'))
        assert result == []

    def test_word_boundary_no_match_on_suffix(self):
        # "supercrates/" must not match "crates/"
        assert find_paths('supercrates/x.rs', ('crates/',)) == []

    # ------------------------------------------------------------------
    # New boundary contract: '/' and '.' are NOT valid left boundaries
    # ------------------------------------------------------------------

    def test_slash_preceded_no_match(self):
        """A prefix immediately preceded by '/' must NOT match (task-1494)."""
        assert find_paths('a/corpus/x', ('corpus/',)) == []

    def test_deep_nested_slash_no_match(self):
        """A multi-segment path that passes *through* the prefix must not match."""
        assert find_paths('repo/test/corpus/expr.txt', ('corpus/',)) == []

    def test_leading_prefix_still_matches(self):
        """RE-POINTED by task 3120: the single-bare-segment shape is now
        deliberately NOT a match.

        The LEFT boundary is still satisfied here (start-of-string), but the
        RIGHT boundary is not: ``x`` is neither followed by another '/' nor
        does it carry a file extension.  A synthetic fixture with no witness
        in the live corpus.  Leading-position matching over a genuine path is
        retained in
        ``TestFindPathsRightBoundary::test_accepted_right_context_further_slash``.
        """
        assert find_paths('corpus/x', ('corpus/',)) == []

    def test_space_preceded_still_matches(self):
        """RE-POINTED by task 3120: the single-bare-segment shape is now
        deliberately NOT a match.

        A space is still a valid LEFT boundary; the failure is on the right —
        ``x`` does not look like a path segment.  A synthetic fixture with no
        witness in the live corpus.  Space-preceded matching over a genuine
        path is retained in ``TestFindPaths::test_single_match``
        ('see crates/x.rs').
        """
        assert find_paths('see corpus/x', ('corpus/',)) == []

    def test_dot_preceded_no_match(self):
        """A prefix immediately preceded by '.' must NOT match (task-1494).

        '.' is excluded from the left-boundary class specifically to prevent a
        dotted-namespace or dotted-package form like ``pkg.corpus/foo`` from
        triggering a leading-prefix match.  Note that ``./corpus/x`` is already
        covered by the '/' exclusion (the char immediately before ``corpus/`` is
        '/'), so '.' only adds value for the standalone-dotted-name case.
        """
        # Contrived single-char prefix: a.corpus/x
        assert find_paths('a.corpus/x', ('corpus/',)) == []
        # More realistic: a package/namespace separator before the prefix
        assert find_paths('pkg.corpus/grammar.js', ('corpus/',)) == []

    def test_relative_path_prefix_dot_slash_not_a_boundary(self):
        """'./corpus/x' is already excluded by the '/' rule (char before 'corpus/' is '/').

        The '.' in the boundary class is NOT responsible for this case — it only
        adds value for purely-dotted forms like 'pkg.corpus/'.  This test
        documents that ./... is handled by '/' exclusion and guards that we
        don't accidentally re-introduce it.
        """
        assert find_paths('./corpus/x', ('corpus/',)) == []


# ---------------------------------------------------------------------------
# New right-boundary contract (task 3120): the token AFTER '<prefix>/' must
# look like a path segment — either another '/' follows, or it carries a file
# extension.  Without this, any English slash-construction whose left half
# happens to name a registered top-level directory ("not a backend/timeout
# error", "tools/call") lexed as a path.
# ---------------------------------------------------------------------------


class TestFindPathsRightBoundary:
    """Task 3120: a registered prefix only matches when what follows it looks
    like a path, not when the '/' is English punctuation.

    Every negative fixture below is a MEASURED false positive taken from live
    corpus text, not a synthetic construction.
    """

    # ------------------------------------------------------------------
    # REJECTED: '/' used as English punctuation
    # ------------------------------------------------------------------

    def test_english_slash_backend_timeout(self):
        assert (
            find_paths(
                'get_memory_by_id returned found=false, not a backend/timeout error',
                ('backend/',),
            )
            == []
        )

    def test_english_slash_corpus_compile(self):
        assert find_paths('already hosts corpus/compile helpers', ('corpus/',)) == []

    def test_english_slash_archive_pause(self):
        assert (
            find_paths(
                'or (b) explicitly archive/pause its reconciliation cadence',
                ('archive/',),
            )
            == []
        )

    def test_english_slash_tools_call(self):
        assert find_paths('sends a JSON-RPC tools/call envelope', ('tools/',)) == []

    def test_english_slash_research_status(self):
        assert (
            find_paths(
                'uncommitted plans/*.md/.txt research/status docs',
                ('research/',),
            )
            == []
        )

    # ------------------------------------------------------------------
    # ACCEPTED right context A: another '/' follows
    # ------------------------------------------------------------------

    def test_accepted_right_context_further_slash(self):
        """A multi-segment path still matches — including at start-of-string."""
        assert find_paths('crates/reify-eval/src/engine_edit.rs', ('crates/',)) == [
            'crates/'
        ]
        assert find_paths('backend/v2/api.py', ('backend/',)) == ['backend/']

    # ------------------------------------------------------------------
    # ACCEPTED right context B: a file extension follows
    # ------------------------------------------------------------------

    def test_accepted_right_context_file_extension(self):
        assert find_paths('gui/package.json', ('gui/',)) == ['gui/']
        assert find_paths('see crates/x.rs', ('crates/',)) == ['crates/']

    def test_accepted_right_context_multi_dot_stem(self):
        """A multi-dot stem ('a.b.txt') still reads as a file."""
        assert find_paths('gui/a.b.txt', ('gui/',)) == ['gui/']

    def test_accepted_right_context_extension_is_case_insensitive(self):
        """An upper-case extension is still an extension."""
        assert find_paths('crates/x.RS', ('crates/',)) == ['crates/']

    def test_dedup_and_order_retained(self):
        """Dedup + first-seen ordering still hold over genuine path text.

        Retains the coverage dropped when ``TestFindPaths::test_dedup_and_order``
        was re-pointed to the new contract.
        """
        result = find_paths(
            'crates/a.rs then gui/b.json then crates/c/d.rs',
            ('crates/', 'gui/'),
        )
        assert result == ['crates/', 'gui/']

    # ------------------------------------------------------------------
    # REJECTED right context: end-of-token
    # ------------------------------------------------------------------

    def test_end_of_token_is_not_an_accepted_right_context(self):
        """Design decision 2 (task 3120): a bare trailing '<prefix>/' does NOT match.

        This is the 28% bare-MENTION class — "see fused-memory/", "the
        'corpus/' dir".  Admitting end-of-token as a right context would
        re-admit almost all of the noise this change exists to remove, so it
        is deliberately excluded even though it looks like a regression in
        isolation.

        No real protection is lost: the FILES-certain check
        (``check_files_for_scope`` -> ``registry.project_for_path``) still
        hard-rejects a genuinely DECLARED foreign file, and that path is
        untouched by this change.
        """
        assert find_paths('crates/', ('crates/',)) == []
        assert find_paths("the quoted token 'corpus/' here", ('corpus/',)) == []

    # ------------------------------------------------------------------
    # KNOWN FAIL-OPEN residue, deliberately not closed
    # ------------------------------------------------------------------

    def test_known_fail_open_residue_leading_dot_file(self):
        """MISSED detection, never a false one — pinned so the gap stays visible.

        'docs/.gitignore' has an empty stem plus a >6-char extension, so it
        satisfies neither right-context alternative.  Widening the classes to
        admit it would start re-admitting the punctuation shapes above.
        """
        assert find_paths('docs/.gitignore', ('docs/',)) == []

    def test_known_fail_open_residue_glob_spelling(self):
        """MISSED detection, never a false one — pinned so the gap stays visible.

        The glob metacharacter in 'plans/*.md' is in neither the segment class
        nor the extension class.  Same conservative direction the guard
        already takes for '../other-repo/x.py' and '~user/...' in
        ``project_for_path``.
        """
        assert find_paths('plans/*.md', ('plans/',)) == []


# ---------------------------------------------------------------------------
# check_candidate_for_scope
# ---------------------------------------------------------------------------


class TestCheckCandidateForScope:
    def test_empty_registry_always_ok(self):
        # No registry → guard is a no-op.
        empty = ProjectPrefixRegistry.from_roots([])
        c = _candidate(title='Edit fused-memory/X')
        v = check_candidate_for_scope(c, 'reify', empty)
        assert v.outcome == 'ok'
        assert v.matched_paths == ()

    def test_own_project_paths_are_ok(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        c = _candidate(title='Refactor crates/foo')
        v = check_candidate_for_scope(c, 'reify', registry)
        assert v.outcome == 'ok'

    def test_other_project_paths_rejected(self, tmp_path):
        # Task 3120: the multi-segment spelling is deliberate — this test
        # exercises the ROUTING/verdict machinery, not the single-bare-segment
        # lexer shape (whose right-boundary contract lives in
        # TestFindPathsRightBoundary).
        registry = _two_project_registry(tmp_path)
        c = _candidate(title='Edit fused-memory/src/x.py')
        v = check_candidate_for_scope(c, 'reify', registry)
        assert v.outcome == 'rejection'
        assert v.matched_paths == ('fused-memory/',)
        assert v.suggested_project == 'dark_factory'
        assert v.error_type == 'DarkFactoryPathScopeViolation'

    def test_reverse_direction_caught(self, tmp_path):
        """The reify→dark-factory direction the original guard missed."""
        registry = _two_project_registry(tmp_path)
        c = _candidate(title='Update crates/widget.rs')
        v = check_candidate_for_scope(c, 'dark_factory', registry)
        assert v.outcome == 'rejection'
        assert v.suggested_project == 'reify'

    def test_multiple_other_projects_no_single_suggestion(self, tmp_path):
        """When mismatches span >1 project, suggested_project is None."""
        # Add a third project so a candidate can mention paths from two others.
        c_root = _mkproj(tmp_path, 'cthird', ['cthird_dir'])
        registry = ProjectPrefixRegistry.from_roots([
            str(_mkproj(tmp_path, 'reify', ['crates'])),
            str(_mkproj(tmp_path, 'dark-factory', ['fused-memory'])),
            str(c_root),
        ])
        # Task 3120: the multi-segment/extensioned spelling is deliberate —
        # this test exercises the ambiguous-suggestion ROUTING path, not the
        # single-bare-segment lexer shape.
        c = _candidate(
            title='wat',
            description='Edit fused-memory/src/x.py and crates/y.rs',
        )
        v = check_candidate_for_scope(c, 'cthird', registry)
        assert v.outcome == 'rejection'
        assert set(v.matched_paths) == {'fused-memory/', 'crates/'}
        assert v.suggested_project is None  # ambiguous

    def test_files_to_modify_scanned(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        c = _candidate(
            title='generic title',
            files_to_modify=['fused-memory/src/x.py'],
        )
        v = check_candidate_for_scope(c, 'reify', registry)
        assert v.outcome == 'rejection'
        assert v.suggested_project == 'dark_factory'

    def test_unknown_prefix_is_silently_ignored(self, tmp_path):
        """A path prefix that's not in the registry doesn't trigger rejection."""
        registry = _two_project_registry(tmp_path)
        c = _candidate(title='Look at random/path/here.py')
        v = check_candidate_for_scope(c, 'reify', registry)
        assert v.outcome == 'ok'

    # ------------------------------------------------------------------
    # Nested-path boundary fix (task-1494)
    # ------------------------------------------------------------------

    def _know_live_reify_registry(self, tmp_path: Path) -> ProjectPrefixRegistry:
        """Know-live (corpus/, tools/) + reify (crates/) registry."""
        kl = _mkproj(tmp_path, 'know-live', ['corpus', 'tools'])
        reify = _mkproj(tmp_path, 'reify', ['crates'])
        return ProjectPrefixRegistry.from_roots([str(kl), str(reify)])

    def test_nested_corpus_path_under_reify_is_ok(self, tmp_path):
        """A candidate under 'reify' citing a path that passes THROUGH '/corpus/'
        must NOT be rejected — nested segment is not a leading prefix (task-1494)."""
        registry = self._know_live_reify_registry(tmp_path)
        c = _candidate(details='vendor/tree-sitter-x/test/corpus/expr.txt')
        v = check_candidate_for_scope(c, 'reify', registry)
        assert v.outcome == 'ok'
        assert v.matched_paths == ()
        assert v.suggested_project is None

    def test_nested_tools_path_under_reify_is_ok(self, tmp_path):
        """A path passing through '/tools/' must not trigger know_live (task-1494)."""
        registry = self._know_live_reify_registry(tmp_path)
        c = _candidate(details='repo/scripts/tools/gen.sh')
        v = check_candidate_for_scope(c, 'reify', registry)
        assert v.outcome == 'ok'
        assert v.matched_paths == ()

    def test_bare_leading_corpus_under_reify_is_rejection(self, tmp_path):
        """A BARE leading 'corpus/' reference (not preceded by '/') filed under reify
        must still be rejected and suggest know_live (regression guard)."""
        registry = self._know_live_reify_registry(tmp_path)
        c = _candidate(title='Edit corpus/wordlist.txt')
        v = check_candidate_for_scope(c, 'reify', registry)
        assert v.outcome == 'rejection'
        assert v.suggested_project == 'know_live'

    def test_bare_leading_tools_under_reify_is_rejection(self, tmp_path):
        """A BARE leading 'tools/' reference filed under reify must still be rejected
        and suggest know_live (regression guard)."""
        registry = self._know_live_reify_registry(tmp_path)
        c = _candidate(title='Update tools/gen.sh')
        v = check_candidate_for_scope(c, 'reify', registry)
        assert v.outcome == 'rejection'
        assert v.suggested_project == 'know_live'

    def test_project_root_prefixed_path_under_reify_is_ok(self, tmp_path):
        """A project-root-prefixed path (e.g. 'know-live/corpus/x') filed under reify
        must NOT trigger a rejection: 'corpus/' is mid-path (preceded by '/'), so the
        tightened lookbehind does NOT match it.

        This is the intentional task-1494 tradeoff — the guard now detects only
        BARE LEADING references.  A mis-filing that specifies the full project-root
        path ('know-live/corpus/wordlist.txt') is a false-negative here; it relies
        on downstream LLM Stage-2 routing rather than the regex guard.

        Explicitly documented so the narrowing is locked in and understood, rather
        than appearing as an accidental gap.  Compare with
        test_bare_leading_corpus_under_reify_is_rejection: once 'know-live/' is
        prepended, the guard yields 'ok'.
        """
        registry = self._know_live_reify_registry(tmp_path)
        # Full project-root path: 'know-live/corpus/x' — 'corpus/' is mid-path
        c = _candidate(details='Edit know-live/corpus/wordlist.txt to add entries')
        v = check_candidate_for_scope(c, 'reify', registry)
        assert v.outcome == 'ok'
        assert v.matched_paths == ()
        assert v.suggested_project is None


# ---------------------------------------------------------------------------
# check_text_for_scope (prompt-only path)
# ---------------------------------------------------------------------------


class TestCheckTextForScope:
    def test_none_text_is_ok(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        v = check_text_for_scope(None, 'reify', registry)
        assert v.outcome == 'ok'

    def test_text_with_other_project_path_rejected(self, tmp_path):
        # Task 3120: the multi-segment spelling is deliberate — this test
        # exercises the prompt-only ROUTING branch, not the
        # single-bare-segment lexer shape.
        registry = _two_project_registry(tmp_path)
        v = check_text_for_scope('please patch fused-memory/src/foo.py', 'reify', registry)
        assert v.outcome == 'rejection'
        assert v.suggested_project == 'dark_factory'

    def test_empty_registry_short_circuits(self):
        empty = ProjectPrefixRegistry.from_roots([])
        v = check_text_for_scope('fused-memory/X', 'reify', empty)
        assert v.outcome == 'ok'

    # ------------------------------------------------------------------
    # 'memory'/'deploy' generic-dirs denylist (task 2434)
    # ------------------------------------------------------------------

    def test_generic_memory_and_deploy_dirs_not_owned(self, tmp_path):
        """A project with top-level memory/ and deploy/ dirs must NOT have
        either registered as an owned prefix — both are common scratch/config
        dir names across projects (task 2434 over-fire fix)."""
        reify = _mkproj(tmp_path, 'reify', ['crates', 'memory', 'deploy'])
        registry = ProjectPrefixRegistry.from_roots([str(reify)])
        assert 'memory/' not in registry.all_prefixes()
        assert 'deploy/' not in registry.all_prefixes()
        assert 'memory/' not in registry.prefix_to_project
        assert 'deploy/' not in registry.prefix_to_project
        # Deliberate coverage-loss tradeoff (task 2434): the CERTAIN
        # files-check (project_for_path / check_files_for_scope) also no
        # longer classifies concrete files under these generic dirs as
        # owned by any project. Pin that here so it's an explicit, intended
        # behavior rather than a silent gap.
        assert registry.project_for_path('memory/foo.py') is None
        assert registry.project_for_path('deploy/x.service') is None
        # Positive-coverage pin: denylisting 'memory'/'deploy' must not
        # collaterally drop a genuine, non-generic top-level dir like
        # 'crates/' from the registry.
        assert 'crates/' in registry.all_prefixes()
        assert registry.project_for_path('crates/foo.rs') == 'reify'

    def test_prose_mention_of_memory_edges_is_ok(self, tmp_path):
        """Prose referencing 'memory/edges' (not a path) must not trigger a
        scope_violation advisory now that 'memory' is a generic dir."""
        reify = _mkproj(tmp_path, 'reify', ['crates', 'memory', 'deploy'])
        registry = ProjectPrefixRegistry.from_roots([str(reify)])
        v = check_text_for_scope(
            're-cite the existing memory/edges', 'dark_factory', registry,
        )
        assert v.outcome == 'ok'


# ---------------------------------------------------------------------------
# check_files_for_scope — CERTAIN classifier for concrete metadata.files
# ---------------------------------------------------------------------------


class TestCheckFilesForScope:
    """Unit tests for check_files_for_scope — the CERTAIN files classifier.

    Unlike check_candidate_for_scope / check_text_for_scope (regex-over-prose,
    heuristic), this classifies each concrete file via
    ProjectPrefixRegistry.project_for_path (exact leading-path-component
    match).  Used by the interceptor's FILES-certain check (task 2206) to
    hard-reject a submission whose metadata.files name a path under a KNOWN
    other project's tree.
    """

    def test_files_all_in_submitting_project_are_ok(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        v = check_files_for_scope(['crates/foo.rs'], 'reify', registry)
        assert v.outcome == 'ok'
        assert v.matched_paths == ()

    def test_file_under_another_project_is_rejected(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        v = check_files_for_scope(['fused-memory/src/x.py'], 'reify', registry)
        assert v.outcome == 'rejection'
        assert v.matched_paths == ('fused-memory/src/x.py',)
        assert v.suggested_project == 'dark_factory'

    def test_empty_files_is_ok(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        v = check_files_for_scope([], 'reify', registry)
        assert v.outcome == 'ok'

    def test_none_files_is_ok(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        v = check_files_for_scope(None, 'reify', registry)
        assert v.outcome == 'ok'

    def test_empty_registry_is_ok(self):
        empty = ProjectPrefixRegistry.from_roots([])
        v = check_files_for_scope(['fused-memory/src/x.py'], 'reify', empty)
        assert v.outcome == 'ok'

    def test_unowned_leading_component_is_ok(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        v = check_files_for_scope(['random/thing.py'], 'reify', registry)
        assert v.outcome == 'ok'

    def test_files_spanning_two_other_projects_no_single_suggestion(self, tmp_path):
        """When mismatched files span >1 other project, suggested_project is None."""
        c_root = _mkproj(tmp_path, 'cthird', ['cthird_dir'])
        registry = ProjectPrefixRegistry.from_roots([
            str(_mkproj(tmp_path, 'reify', ['crates'])),
            str(_mkproj(tmp_path, 'dark-factory', ['fused-memory'])),
            str(c_root),
        ])
        v = check_files_for_scope(
            ['fused-memory/x.py', 'crates/y.rs'], 'cthird', registry,
        )
        assert v.outcome == 'rejection'
        assert set(v.matched_paths) == {'fused-memory/x.py', 'crates/y.rs'}
        assert v.suggested_project is None


# ---------------------------------------------------------------------------
# all_files_foreign_owner — cross-repo (all-foreign) classifier (task 3004)
# ---------------------------------------------------------------------------


class TestAllFilesForeignOwner:
    """Unit tests for all_files_foreign_owner — the cross-repo classifier.

    Distinct from check_files_for_scope (which rejects on ANY foreign file):
    this returns the single foreign owner ONLY when a submission is ENTIRELY
    foreign under one owner with NO locally-owned file — the reify-task 5308
    cross-repo deliverable shape (task 3004).  Files with no registered owner
    stay neutral (conservative, matching _aggregate_owner_mismatches).
    """

    def test_all_files_single_foreign_owner_returns_owner(self, tmp_path):
        # (a) every file under dark_factory (orchestrator/), filed under reify.
        from fused_memory.middleware.path_scope_guard import all_files_foreign_owner

        registry = _two_project_registry(tmp_path)
        owner = all_files_foreign_owner(
            ['orchestrator/src/x.py', 'orchestrator/tests/y.py'],
            'reify', registry,
        )
        assert owner == 'dark_factory'

    def test_mixed_local_and_foreign_returns_none(self, tmp_path):
        # (b) one reify-owned + one dark_factory file → NOT all-foreign;
        # existing check_files_for_scope rejection semantics are preserved.
        from fused_memory.middleware.path_scope_guard import all_files_foreign_owner

        registry = _two_project_registry(tmp_path)
        owner = all_files_foreign_owner(
            ['crates/foo.rs', 'orchestrator/x.py'],
            'reify', registry,
        )
        assert owner is None

    def test_all_unowned_returns_none(self, tmp_path):
        # (c) files with no registered owner stay neutral → None.
        from fused_memory.middleware.path_scope_guard import all_files_foreign_owner

        registry = _two_project_registry(tmp_path)
        owner = all_files_foreign_owner(
            ['random/thing.py', 'other/local.py'],
            'reify', registry,
        )
        assert owner is None

    def test_all_local_to_submitting_project_returns_none(self, tmp_path):
        # (c') files owned by the submitting project → None.
        from fused_memory.middleware.path_scope_guard import all_files_foreign_owner

        registry = _two_project_registry(tmp_path)
        owner = all_files_foreign_owner(
            ['crates/foo.rs', 'gui/app.rs'],
            'reify', registry,
        )
        assert owner is None

    def test_empty_files_or_registry_returns_none(self, tmp_path):
        # (d) empty files or empty registry → None.
        from fused_memory.middleware.path_scope_guard import all_files_foreign_owner

        registry = _two_project_registry(tmp_path)
        assert all_files_foreign_owner([], 'reify', registry) is None
        assert all_files_foreign_owner(None, 'reify', registry) is None
        empty = ProjectPrefixRegistry.from_roots([])
        assert all_files_foreign_owner(['orchestrator/x.py'], 'reify', empty) is None

    def test_files_span_two_foreign_owners_returns_none(self, tmp_path):
        # (e) foreign files split across TWO owners → None (single-owner req).
        from fused_memory.middleware.path_scope_guard import all_files_foreign_owner

        c_root = _mkproj(tmp_path, 'cthird', ['cthird_dir'])
        registry = ProjectPrefixRegistry.from_roots([
            str(_mkproj(tmp_path, 'reify', ['crates'])),
            str(_mkproj(tmp_path, 'dark-factory', ['fused-memory'])),
            str(c_root),
        ])
        owner = all_files_foreign_owner(
            ['fused-memory/x.py', 'cthird_dir/y.py'],
            'reify', registry,
        )
        assert owner is None

    def test_unregistered_filer_all_foreign_returns_none(self, tmp_path):
        # (f) FILER-REGISTRATION gate (task 3004 / esc-3004 NARROW decision):
        # an UNREGISTERED filer whose files are ALL foreign under a single
        # owner does NOT qualify for the cross-repo allow+tag path — it returns
        # None here and falls through to check_files_for_scope's hard reject,
        # preserving task-2206's anti-bypass guard. Cross-repo is a
        # relationship between two REGISTERED projects.
        from fused_memory.middleware.path_scope_guard import all_files_foreign_owner

        registry = _two_project_registry(tmp_path)  # registers reify + dark_factory
        assert registry.is_known('reify') is True
        assert registry.is_known('outsider') is False
        # Same all-foreign single-owner file set that returns 'dark_factory' for
        # a REGISTERED filer ('reify') must return None for an UNREGISTERED one.
        owner = all_files_foreign_owner(
            ['orchestrator/src/x.py', 'orchestrator/tests/y.py'],
            'outsider', registry,
        )
        assert owner is None


# ---------------------------------------------------------------------------
# Task 3109: ABSOLUTE metadata.files entries are classified identically to
# their repo-relative spelling.
#
# Before this task, project_for_path matched only registered RELATIVE prefixes
# as a leading path component, so an absolute path ('/home/.../orchestrator/
# x.py') could never match any prefix and came back None = "unowned" —
# silently disarming BOTH the FILES-certain hard reject (check_files_for_scope
# returned 'ok') and the task-3004 cross-repo tagger (all_files_foreign_owner
# returned None). Absolute paths are the shape an agent produces after
# actually reading a foreign repo, i.e. exactly the incident case.
# ---------------------------------------------------------------------------


class TestAbsoluteForeignPaths:
    """Absolute foreign paths must produce the same verdict as relative ones.

    Absolute paths are derived from ``registry.root_for_project(...)`` rather
    than re-joined from ``tmp_path`` by hand, so the strings line up with the
    ``Path(...).resolve()``-normalised roots ``from_roots`` stores (matters if
    pytest's tmp_path is ever a symlink).
    """

    @staticmethod
    def _abs_incident_files(registry) -> list[str]:
        """The four incident-shaped dark-factory files, absolute spelling."""
        df = registry.root_for_project('dark_factory')
        assert df is not None
        return [
            f'{df}/orchestrator/src/orchestrator/git_ops.py',
            f'{df}/orchestrator/src/orchestrator/merge_worker.py',
            f'{df}/fused-memory/src/fused_memory/middleware/task_interceptor.py',
            f'{df}/fused-memory/src/x.py',
        ]

    _REL_INCIDENT_FILES = [
        'orchestrator/src/orchestrator/git_ops.py',
        'orchestrator/src/orchestrator/merge_worker.py',
        'fused-memory/src/fused_memory/middleware/task_interceptor.py',
        'fused-memory/src/x.py',
    ]

    def test_check_files_for_scope_rejects_absolute_foreign_paths(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        abs_files = self._abs_incident_files(registry)

        verdict = check_files_for_scope(abs_files, 'reify', registry)

        assert verdict.outcome == 'rejection', (
            f'Absolute foreign paths must hard-reject, got: {verdict!r}'
        )
        assert verdict.suggested_project == 'dark_factory'
        assert list(verdict.matched_paths) == abs_files

        # The two spellings must never diverge again: the absolute verdict has
        # to match the repo-relative verdict for the same logical files.
        rel_verdict = check_files_for_scope(
            self._REL_INCIDENT_FILES, 'reify', registry,
        )
        assert verdict.outcome == rel_verdict.outcome
        assert verdict.suggested_project == rel_verdict.suggested_project

    def test_all_files_foreign_owner_tags_absolute_cross_repo(self, tmp_path):
        """Task-3004's allow-and-tag path must fire for the all-absolute shape."""
        from fused_memory.middleware.path_scope_guard import all_files_foreign_owner

        registry = _two_project_registry(tmp_path)
        abs_files = self._abs_incident_files(registry)

        assert all_files_foreign_owner(abs_files, 'reify', registry) == 'dark_factory'

    def test_absolute_paths_under_own_root_are_ok(self, tmp_path):
        from fused_memory.middleware.path_scope_guard import all_files_foreign_owner

        registry = _two_project_registry(tmp_path)
        reify = registry.root_for_project('reify')
        assert reify is not None
        own_files = [f'{reify}/crates/widget.rs', f'{reify}/crates/other.rs']

        assert check_files_for_scope(own_files, 'reify', registry).outcome == 'ok'
        assert all_files_foreign_owner(own_files, 'reify', registry) is None

    def test_absolute_path_under_no_known_root_is_ok(self, tmp_path):
        """An absolute path under NO registered root stays unowned (fail-open).

        Deliberate boundary: 'unowned stays unowned'. The dir is created but
        never passed to from_roots, so nothing in the registry can claim it —
        even though 'crates/' IS a registered prefix in its relative spelling.
        """
        from fused_memory.middleware.path_scope_guard import all_files_foreign_owner

        registry = _two_project_registry(tmp_path)
        stranger = _mkproj(tmp_path, 'unknown-proj', ['crates'])
        files = [str(stranger / 'crates' / 'x.rs')]

        assert check_files_for_scope(files, 'reify', registry).outcome == 'ok'
        assert all_files_foreign_owner(files, 'reify', registry) is None

    def test_mixed_absolute_foreign_and_local_is_hard_reject(self, tmp_path):
        """Task-2206's mixed-scope hard reject survives the absolute spelling.

        An absolute foreign path must not become an all-foreign escape hatch
        when a locally-owned file is also declared.
        """
        from fused_memory.middleware.path_scope_guard import all_files_foreign_owner

        registry = _two_project_registry(tmp_path)
        df = registry.root_for_project('dark_factory')
        files = [f'{df}/fused-memory/src/x.py', 'crates/local_widget.rs']

        verdict = check_files_for_scope(files, 'reify', registry)
        assert verdict.outcome == 'rejection', f'got: {verdict!r}'
        assert all_files_foreign_owner(files, 'reify', registry) is None


# ---------------------------------------------------------------------------
# Verdict.to_error_dict
# ---------------------------------------------------------------------------


class TestVerdictErrorDict:
    def test_ok_verdict_yields_empty(self):
        v = PathGuardVerdict(outcome='ok')
        assert v.to_error_dict() == {}

    def test_rejection_with_suggested_project(self):
        v = PathGuardVerdict(
            outcome='rejection',
            project_id='reify',
            matched_paths=('fused-memory/',),
            suggested_project='dark_factory',
        )
        d = v.to_error_dict()
        assert d['error_type'] == 'DarkFactoryPathScopeViolation'
        assert d['project_id'] == 'reify'
        assert d['matched_paths'] == ['fused-memory/']
        assert d['suggested_project'] == 'dark_factory'
        assert 'dark_factory' in d['error']

    def test_rejection_without_suggested_project(self):
        v = PathGuardVerdict(
            outcome='rejection',
            project_id='reify',
            matched_paths=('fused-memory/', 'crates/'),
            suggested_project=None,
        )
        d = v.to_error_dict()
        assert d['suggested_project'] is None
        # Error message should hint at manual routing.
        assert 'manually' in d['error'] or 'manual' in d['error']


# ---------------------------------------------------------------------------
# is_routing_override
# ---------------------------------------------------------------------------


class TestIsRoutingOverride:
    """Unit tests for the is_routing_override(reason) predicate.

    Pins the strip-semantics: whitespace-only is NOT a valid override.
    """

    def test_non_empty_reason_returns_true(self):
        assert is_routing_override('owner asserted ownership') is True

    def test_single_word_reason_returns_true(self):
        assert is_routing_override('cross-cutting') is True

    def test_empty_string_returns_false(self):
        assert is_routing_override('') is False

    def test_none_returns_false(self):
        assert is_routing_override(None) is False

    def test_whitespace_only_returns_false(self):
        """Whitespace-only reason must NOT count as an override (strip semantics)."""
        assert is_routing_override('   ') is False

    def test_reason_with_leading_trailing_whitespace_is_true(self):
        """A reason that is non-empty after stripping is valid."""
        assert is_routing_override('  owner confirmed  ') is True
