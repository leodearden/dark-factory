"""Contract for the shared cache-buster helpers in `_cache_buster_helpers`.

`index.html` version-stamps every `/static/redux/*` asset with a single
shared `?v=NN`, and that number is the only thing standing between a fixed
`.js` file and a browser that already holds the broken copy.  The guard in
`test_index_html.py` used to check that number against a HARDCODED FLOOR,
which failed the one case that actually bites: task/3490's rebase dropped its
own `43 -> 44` bump because main had already released 44, so the branch
merged carrying main's number while still shipping modified JSX — a URL
browsers had already cached now served different bytes, and the floor was
green throughout.

The rule is therefore base-RELATIVE, and this file owns its contract in three
layers so each is independently constructible:

  * EXTRACTION — pure text.  Which tags carry a cache-buster at all, and what
    "the" version of a body is.
  * DECISION — pure literals.  `cache_buster_violation` over a
    `ReduxBaseState`, so the failing truth-table rows can be pinned without a
    repo that exhibits them.
  * GIT — throwaway `tmp_path` repos.  `resolve_redux_base_state` against
    synthetic histories, including a literal replay of the task/3490 shape.

The split exists because on any healthy branch `merge-base(main, HEAD)` is
main's tip and no redux asset is modified, so a base-relative rule is
VACUOUSLY green against the real repo.  A genuine RED for this guard can only
be constructed, never observed here — the same shape `test_jsx_source_helpers.py`
uses to pin `_dashboard_helpers` against synthetic sources.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from _cache_buster_helpers import (
    INDEX_HTML_REL,
    REDUX_ASSET_DIR,
    ReduxBaseState,
    cache_buster_violation,
    redux_cache_buster_versions,
    resolve_redux_base_state,
    sole_cache_buster_version,
)

_CHARTS = 'dashboard/src/dashboard/static/redux/charts.jsx'

# A body shaped like the real index.html: several versioned /static/redux/*
# tags, one unpkg CDN tag carrying an SRI hash and NO ?v=, and a non-redux
# /static/ reference.  The latter two must not be counted.
_UNIFORM_BODY = """
<!doctype html>
<html>
  <head>
    <link rel="icon" href="/static/favicon.svg">
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"
            integrity="sha384-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            crossorigin="anonymous"></script>
    <script src="/static/redux/spark_path.js?v=49"></script>
    <script src="/static/redux/graph_layout.js?v=49"></script>
    <link rel="stylesheet" href="/static/redux/app.css?v=49">
  </head>
  <body><script type="text/babel" src="/static/redux/app.jsx?v=49"></script></body>
</html>
"""

_MIXED_BODY = _UNIFORM_BODY.replace('graph_layout.js?v=49', 'graph_layout.js?v=48')

_UNVERSIONED_BODY = """
<!doctype html>
<html>
  <head>
    <link rel="icon" href="/static/favicon.svg">
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"
            integrity="sha384-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            crossorigin="anonymous"></script>
    <script src="/static/redux/spark_path.js"></script>
  </head>
</html>
"""


class TestReduxCacheBusterVersions:
    """Which tags count as versioned redux assets, pinned against literals."""

    def test_uniform_body_yields_the_single_shared_version(self) -> None:
        """Every `/static/redux/*?v=` tag contributes; the CDN and favicon do not.

        The unpkg tag carries an SRI hash whose base64 payload contains digits,
        and `/static/favicon.svg` sits under `/static/` — both are near misses
        that a looser pattern would sweep in.
        """
        assert redux_cache_buster_versions(_UNIFORM_BODY) == {49}

    def test_partial_bump_yields_both_versions(self) -> None:
        """A body mid-bump reports every distinct version it carries."""
        assert redux_cache_buster_versions(_MIXED_BODY) == {48, 49}

    def test_body_with_no_versioned_tag_yields_the_empty_set(self) -> None:
        """No `?v=` anywhere means no versions — not a crash, and not a default."""
        assert redux_cache_buster_versions(_UNVERSIONED_BODY) == set()


class TestSoleCacheBusterVersion:
    """"The" version of a body — defined only when the body carries exactly one."""

    def test_returns_the_int_for_a_uniform_body(self) -> None:
        """A uniform body has a single version, returned as an int."""
        assert sole_cache_buster_version(_UNIFORM_BODY) == 49

    def test_raises_when_no_versioned_tag_is_present(self) -> None:
        """An unversioned body RAISES rather than returning a sentinel.

        This is load-bearing.  Any sentinel — `0`, `None`, `-1` — would make
        every downstream freshness comparison pass VACUOUSLY against an empty
        or restructured index.html, which is a permanent false GREEN: the one
        state where the guard is most needed would be the one state it cannot
        report.  Same reasoning `test_jsx_source_helpers.py` gives for
        `extract_function_body` raising rather than returning `''`.
        """
        with pytest.raises(ValueError):
            sole_cache_buster_version(_UNVERSIONED_BODY)

    def test_raises_when_the_body_carries_mixed_versions(self) -> None:
        """A mid-bump body has no single version, so there is nothing to compare.

        The message must name the observed set, so the reader learns WHICH
        tags disagree without reopening index.html.
        """
        with pytest.raises(ValueError) as exc:
            sole_cache_buster_version(_MIXED_BODY)

        assert '48' in str(exc.value) and '49' in str(exc.value), (
            f'the mixed-version error must name the observed versions, got {exc.value!r}'
        )


class TestAssetPathConstants:
    """The repo-root-relative paths the git layer hands to `git show`/`git diff`."""

    def test_redux_asset_dir_is_the_real_served_directory(self) -> None:
        """Pinned so moving the asset dir breaks HERE, loudly.

        The git layer passes this as a pathspec; a stale value would silently
        match nothing, leaving `changed_assets` permanently empty and the
        freshness rule permanently exempt — a false GREEN that no other test
        in this suite would notice.
        """
        assert REDUX_ASSET_DIR == 'dashboard/src/dashboard/static/redux'

    def test_index_html_rel_is_the_versioned_file(self) -> None:
        """`git show <sha>:<path>` needs the repo-root-relative spelling."""
        assert INDEX_HTML_REL == 'dashboard/src/dashboard/static/redux/index.html'


def _base(
    base_version: int | None,
    changed_assets: tuple[str, ...] = (),
) -> ReduxBaseState:
    """A `ReduxBaseState` with the two fields under test varied and the rest fixed."""
    return ReduxBaseState(
        base_ref='main',
        base_commit='0123456789abcdef',
        base_version=base_version,
        changed_assets=changed_assets,
    )


class TestCacheBusterViolationWhenAssetsChanged:
    """FRESH rule: modify an existing redux asset and the number must go UP.

    "Existing" is what makes the demand fair — the URL is already in browser
    caches, so serving different bytes at the same `?v=` is the exact pairing
    the guard exists to make impossible.
    """

    def test_equal_version_with_a_changed_asset_is_a_violation(self) -> None:
        """THE TASK/3490 SHAPE — the case the old hardcoded floor could not catch.

        3490's rebase dropped its own `43 -> 44` bump because main had already
        released 44, so the branch carried main's number while still shipping
        modified JSX.  `>= 45` was satisfied throughout.  Base-relative, it is
        not: the base already RELEASED this number.
        """
        violation = cache_buster_violation(45, _base(45, (_CHARTS,)))

        assert violation is not None, (
            'a branch that modifies a released redux asset without bumping past the '
            "base's version must be reported — this is the task/3490 failure."
        )
        assert '45' in violation, (
            f'the message must name the version at issue, got {violation!r}'
        )
        assert _CHARTS in violation, (
            'the message must name the changed asset, so the reader learns WHICH '
            f'file went stale rather than only that a number is wrong; got {violation!r}'
        )

    def test_lower_version_with_a_changed_asset_is_a_violation(self) -> None:
        """A number that went DOWN is worse than one that stood still."""
        assert cache_buster_violation(44, _base(45, (_CHARTS,))) is not None

    def test_higher_version_with_a_changed_asset_is_clean(self) -> None:
        """The ordinary correct bump: asset touched, version strictly newer."""
        assert cache_buster_violation(46, _base(45, (_CHARTS,))) is None


class TestCacheBusterViolationWhenNoAssetChanged:
    """The EXEMPTION: a branch touching no redux asset is never forced to bump.

    Without this, every unrelated task in the repo would owe a cache-buster
    bump it has no reason to make, and bumps would stop meaning anything.
    """

    def test_equal_version_with_no_changed_asset_is_clean(self) -> None:
        """No asset modified, number unchanged — nothing browsers hold went stale."""
        assert cache_buster_violation(45, _base(45)) is None

    def test_higher_version_with_no_changed_asset_is_clean(self) -> None:
        """Bumping without touching an asset is wasteful, not wrong; not our rule."""
        assert cache_buster_violation(46, _base(45)) is None

    def test_lower_version_with_no_changed_asset_is_a_violation(self) -> None:
        """MONOTONIC rule — and it is NOT redundant with the freshness rule.

        This is the base-relative replacement for the retired `assert v >= 45`
        floor, and it is the only rule that fires here: `changed_assets` is
        empty, so freshness is exempt.  A branch that reverts the number
        re-points every asset URL at bytes browsers already cached from an
        earlier release, whether or not that branch touched an asset itself.
        """
        violation = cache_buster_violation(44, _base(49))

        assert violation is not None, (
            'a cache-buster that moves BACKWARDS versus the merge base must be '
            'reported even when the branch modified no redux asset — this is the '
            'monotonic rule that replaces the retired hardcoded floor.'
        )
        assert '44' in violation and '49' in violation, (
            f'the message must name both versions, got {violation!r}'
        )


class TestCacheBusterViolationWithNoBaseline:
    """`base_version is None` — the base commit predates the `?v=` scheme.

    There is genuinely nothing to compare against, and a missing baseline must
    not manufacture a failure: that would turn "we cannot measure" into "you
    did something wrong", which is a false red on work that is not at fault.
    """

    def test_missing_baseline_with_a_changed_asset_is_clean(self) -> None:
        assert cache_buster_violation(45, _base(None, (_CHARTS,))) is None

    def test_missing_baseline_with_no_changed_asset_is_clean(self) -> None:
        assert cache_buster_violation(45, _base(None)) is None


# ---------------------------------------------------------------------------
# The git layer, pinned against throwaway repos under tmp_path.
#
# `_init_repo` is copied from tests/scripts/test_basetemp_git_isolation.py:55
# rather than imported, for the reason that module states about its own copy
# (itself taken from orchestrator/tests/test_git_repo_isolation_guard.py): it
# is not importable from the dashboard suite.  The suite-wide
# `_df_git_ceiling_at_basetemp` fixture pins git's upward repo walk inside the
# pytest basetemp, so a repo built here cannot escape into a live worktree.
# ---------------------------------------------------------------------------

_CHARTS_REL = f'{REDUX_ASSET_DIR}/charts.jsx'


def _init_repo(path: Path) -> Path:
    """``git init`` a fresh repo at *path* with one commit.  Creates its own
    target, so it cannot escape into an enclosing repo.
    """
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init', '-b', 'main'], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ['git', 'config', 'user.email', 'test@test.com'], cwd=path, check=True, capture_output=True,
    )
    subprocess.run(
        ['git', 'config', 'user.name', 'Test'], cwd=path, check=True, capture_output=True,
    )
    (path / 'README.md').write_text('# sentinel\n')
    subprocess.run(['git', 'add', '-A'], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ['git', 'commit', '-m', 'initial'], cwd=path, check=True, capture_output=True,
    )
    return path


def _git(repo: Path, *args: str) -> str:
    """Run git in *repo*, never inheriting the process cwd."""
    return subprocess.run(
        ['git', *args],
        cwd=repo, capture_output=True, text=True, check=True, timeout=60,
    ).stdout


def _write_index(repo: Path, version: int) -> None:
    """Write index.html with a handful of `?v=<version>` tags, mirroring the real one."""
    (repo / INDEX_HTML_REL).write_text(
        '<!doctype html>\n<html><head>\n'
        '<link rel="icon" href="/static/favicon.svg">\n'
        f'<script src="/static/redux/spark_path.js?v={version}"></script>\n'
        f'<script src="/static/redux/graph_layout.js?v={version}"></script>\n'
        f'<script type="text/babel" src="/static/redux/charts.jsx?v={version}"></script>\n'
        '</head></html>\n'
    )


def _released_repo(tmp_path: Path, version: int = 45) -> Path:
    """A repo whose `main` has released the redux tree at *version*.

    HEAD is left on a fresh branch `work` off main, so `merge-base(main, HEAD)`
    is main's tip — the shape the guard sees on a real task branch.
    """
    repo = _init_repo(tmp_path / 'repo')
    (repo / REDUX_ASSET_DIR).mkdir(parents=True)
    _write_index(repo, version)
    (repo / _CHARTS_REL).write_text('function Chart() { return null; }\n')
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-m', f'release v={version}')
    _git(repo, 'checkout', '-b', 'work')
    return repo


class TestResolveReduxBaseStateAgainstSyntheticRepos:
    """`resolve_redux_base_state` against constructed histories.

    Constructed because a real branch cannot exhibit the failure: on a healthy
    branch the merge base IS main's tip and nothing under the redux directory
    differs, so every base-relative rule is vacuously satisfied.
    """

    def test_task_3490_replay_reports_the_stale_asset(self, tmp_path: Path) -> None:
        """THE HEADLINE: main released v=45; the branch edits charts.jsx and
        does NOT bump — end to end, that must be reported.

        This is literally what happened on task/3490: the rebase dropped its
        own `43 -> 44` bump because main had already released 44, so the branch
        carried main's number while still modifying shipped JSX.  The old
        `assert v >= 45` floor was green for the entire ride.
        """
        repo = _released_repo(tmp_path, version=45)
        (repo / _CHARTS_REL).write_text('function Chart() { return "fixed"; }\n')
        _git(repo, 'commit', '-am', 'fix charts, forget the bump')

        state = resolve_redux_base_state(repo)

        assert state is not None, 'a repo with a main ref must resolve a base state'
        assert state.base_version == 45, (
            f'the base released 45, got base_version={state.base_version!r}'
        )
        assert state.changed_assets == (_CHARTS_REL,), (
            f'charts.jsx was modified in place versus the base, got {state.changed_assets!r}'
        )
        assert cache_buster_violation(45, state) is not None, (
            'the task/3490 shape must be REPORTED end to end: an asset modified '
            'in place while the cache-buster still reads the number the base '
            'already released.'
        )

    def test_a_bump_alongside_the_asset_change_is_clean(self, tmp_path: Path) -> None:
        """The correct discharge: touch the asset AND bump past the base."""
        repo = _released_repo(tmp_path, version=45)
        (repo / _CHARTS_REL).write_text('function Chart() { return "fixed"; }\n')
        _write_index(repo, 46)
        _git(repo, 'commit', '-am', 'fix charts and bump to 46')

        state = resolve_redux_base_state(repo)

        assert state is not None
        assert state.base_version == 45
        assert cache_buster_violation(46, state) is None, (
            'bumping past the base while changing an asset is exactly the '
            'behaviour the guard asks for and must not be flagged.'
        )

    def test_a_change_outside_the_redux_dir_is_not_a_trigger(self, tmp_path: Path) -> None:
        """The EXEMPTION that keeps this guard from taxing every unrelated task."""
        repo = _released_repo(tmp_path, version=45)
        (repo / 'README.md').write_text('# sentinel, edited\n')
        _git(repo, 'commit', '-am', 'unrelated work')

        state = resolve_redux_base_state(repo)

        assert state is not None
        assert state.changed_assets == (), (
            f'nothing under {REDUX_ASSET_DIR} changed, got {state.changed_assets!r}'
        )

    def test_an_uncommitted_edit_already_counts(self, tmp_path: Path) -> None:
        """The trigger set is the WORKING TREE, not HEAD.

        Diffing base..HEAD would leave the guard exempt for the whole of an
        implementer's edit/verify loop and only fire after the commit — the
        slowest possible moment to learn a bump is owed.  This asset is never
        `git add`ed and must still be seen.
        """
        repo = _released_repo(tmp_path, version=45)
        (repo / _CHARTS_REL).write_text('function Chart() { return "wip"; }\n')

        state = resolve_redux_base_state(repo)

        assert state is not None
        assert state.changed_assets == (_CHARTS_REL,), (
            'an uncommitted working-tree edit must already be reported, so the '
            'implementer sees the demand during the edit/verify loop rather '
            f'than only after committing; got {state.changed_assets!r}'
        )

    def test_index_html_is_excluded_from_its_own_trigger_set(self, tmp_path: Path) -> None:
        """index.html carries the version, so it cannot be its own trigger.

        Including it would make every bump demand a further bump, and a branch
        editing only inline markup would owe a bump for a change no cached
        asset URL can serve.
        """
        repo = _released_repo(tmp_path, version=45)
        _write_index(repo, 46)
        _git(repo, 'commit', '-am', 'bump only')

        state = resolve_redux_base_state(repo)

        assert state is not None
        assert state.changed_assets == (), (
            f'index.html must not appear in its own trigger set, got {state.changed_assets!r}'
        )

    def test_an_added_asset_is_not_a_trigger(self, tmp_path: Path) -> None:
        """A brand-new file has a brand-new URL that was never in any cache.

        Forcing a bump for it would be a false positive, and the convention
        already allows it — commit 9e93e849dc registered
        tasks_offline_banner.js at the then-current v=46.
        """
        repo = _released_repo(tmp_path, version=45)
        (repo / REDUX_ASSET_DIR / 'brand_new.js').write_text('window.DF_NEW = {};\n')
        _git(repo, 'add', '-A')
        _git(repo, 'commit', '-m', 'add a new asset')

        state = resolve_redux_base_state(repo)

        assert state is not None
        assert state.changed_assets == (), (
            'an ADDED asset gets a URL no browser has cached, so it must not '
            f'force a bump; got {state.changed_assets!r}'
        )


class TestResolveReduxBaseStateWhenNoBaseRefResolves:
    """No base ref means NOT MEASURABLE — the caller skips, it does not fail."""

    def test_returns_none_when_no_candidate_ref_exists(self, tmp_path: Path) -> None:
        """A repo whose only branch is `trunk` resolves nothing, and that is fine.

        An sdist install or a shallow clone has no `main` either.  Failing
        there would be a false red on a tree that is not broken; returning
        `None` lets the caller degrade to a visible pytest skip.
        """
        repo = _init_repo(tmp_path / 'repo')
        _git(repo, 'branch', '-m', 'main', 'trunk')

        assert resolve_redux_base_state(repo) is None, (
            'with no main and no origin/main there is nothing to measure against, '
            'so the resolver must return None rather than raise.'
        )

    def test_falls_through_to_the_second_candidate_ref(self, tmp_path: Path) -> None:
        """`base_refs` is tried in order; a later ref rescues a missing earlier one.

        A checkout that only ever fetched the remote has `origin/main` and no
        local `main`.
        """
        repo = _released_repo(tmp_path, version=45)
        main_sha = _git(repo, 'rev-parse', 'main').strip()
        _git(repo, 'update-ref', 'refs/remotes/origin/main', main_sha)
        _git(repo, 'branch', '-D', 'main')

        state = resolve_redux_base_state(repo, base_refs=('main', 'origin/main'))

        assert state is not None, (
            'the local main is gone but origin/main resolves, so the second '
            'candidate ref must be tried.'
        )
        assert state.base_ref == 'origin/main', (
            f'the resolved ref must be recorded for the failure message, got {state.base_ref!r}'
        )
        assert state.base_version == 45


# ---------------------------------------------------------------------------
# The WIRING proof: that test_index_html.py actually consumes the evaluator,
# and that the hardcoded floor is gone.
#
# Both are proved BEHAVIOURALLY — import the guard and call it with synthetic
# arguments — never by grepping source text or asserting on docstring prose.
# A source-grep proof is a documentation meta-test: any refactor breaks it and
# any copy-pasted call satisfies it, so it cannot distinguish a guard that
# WORKS from one that merely mentions the right identifier.
#
# The imports are function-scoped, not module-scoped, and that is deliberate.
# The suite's cross-module convention (orchestrator/tests/test_verify_scope_kappa.py:49
# and siblings) imports HELPERS and FIXTURES at module level — never `test_*`
# names, because a module-level `from x import test_y` binds `test_y` as an
# attribute of THIS module and pytest then collects and runs it here a second
# time, against the real fixtures.
# ---------------------------------------------------------------------------

# A body carrying every asset test_redux_cache_buster_bumped requires present,
# at a uniform — and deliberately TINY — version.  v=1 is far below the `>= 45`
# floor that used to live in that test, which is what makes the floor's
# retirement observable rather than merely asserted.
_REQUIRED_ASSETS = (
    'graph_layout.js',
    'prd_grouping.js',
    'task_status_counts.js',
    'runtime_format.js',
    'orch_filter.js',
    'esc_flow_layout.js',
    'memory_evals_fmt.js',
    'spark_path.js',
)


def _synthetic_index(version: int) -> str:
    """An index.html body at a uniform *version* carrying every required asset."""
    tags = '\n'.join(
        f'<script src="/static/redux/{name}?v={version}"></script>'
        for name in _REQUIRED_ASSETS
    )
    return f'<!doctype html>\n<html><head>\n{tags}\n</head></html>\n'


class TestGuardIsWiredToTheEvaluator:
    """`test_index_html.py`'s freshness guard consumes `cache_buster_violation`."""

    def test_guard_fails_on_the_task_3490_shape(self) -> None:
        """Fed a stale state, the real guard RAISES, and names the stale asset.

        This is the end-to-end proof: not that the module imports the helper,
        but that a state the evaluator rejects actually fails the test that
        gates the merge.
        """
        from test_index_html import test_redux_cache_buster_is_newer_than_merge_base as guard

        base = ReduxBaseState(
            base_ref='main',
            base_commit='0123456789abcdef',
            base_version=45,
            changed_assets=(_CHARTS_REL,),
        )

        with pytest.raises(AssertionError) as exc:
            guard(_synthetic_index(45), base)

        assert _CHARTS_REL in str(exc.value), (
            'the guard must surface the evaluator\'s message, which names the '
            f'stale asset — got {exc.value!r}'
        )

    def test_guard_passes_when_the_bump_landed(self) -> None:
        """A version strictly newer than the base discharges the demand."""
        from test_index_html import test_redux_cache_buster_is_newer_than_merge_base as guard

        base = ReduxBaseState(
            base_ref='main',
            base_commit='0123456789abcdef',
            base_version=45,
            changed_assets=(_CHARTS_REL,),
        )

        guard(_synthetic_index(46), base)

    def test_guard_passes_when_no_redux_asset_changed(self) -> None:
        """The exemption survives the wiring: no asset touched, no bump owed."""
        from test_index_html import test_redux_cache_buster_is_newer_than_merge_base as guard

        base = ReduxBaseState(
            base_ref='main',
            base_commit='0123456789abcdef',
            base_version=45,
            changed_assets=(),
        )

        guard(_synthetic_index(45), base)

    def test_guard_skips_when_no_base_state_resolved(self) -> None:
        """An unresolvable base SKIPS — not a silent pass, and not a red.

        A silent pass would hide that the guard never ran; a red would blame a
        tree that is not broken.  A skip stays visible in the pytest output,
        which is the only outcome that tells the truth here.
        """
        from test_index_html import test_redux_cache_buster_is_newer_than_merge_base as guard

        with pytest.raises(pytest.skip.Exception):
            guard(_synthetic_index(45), None)


class TestHardcodedFloorIsRetired:
    """`test_redux_cache_buster_bumped` no longer pins an absolute version."""

    def test_uniform_body_at_a_tiny_version_passes(self) -> None:
        """A uniform `?v=1` body carrying every required asset must PASS.

        Under the old `assert v >= 45` this fails, so this test is a genuine
        RED that goes green only once the floor is retired — and it pins the
        retirement BEHAVIOURALLY, without asserting on any prose.

        The floor was dead weight: index.html shipped v=49 while the floor
        still read 45, four releases stale.  The base-relative MONOTONIC rule
        next door is a self-maintaining superset of what it protected against,
        and seven sibling modules keep their own `min(versions) >= N`
        anti-revert pins regardless.
        """
        from test_index_html import test_redux_cache_buster_bumped as bumped

        bumped(_synthetic_index(1))

    def test_mixed_versions_still_fail(self) -> None:
        """Retiring the floor must not cost the UNIFORMITY check.

        `test_redux_cache_buster_bumped` remains the sole home of that check —
        every sibling module asserts only its own floor — so a regression here
        would go unnoticed everywhere else.
        """
        from test_index_html import test_redux_cache_buster_bumped as bumped

        mixed = _synthetic_index(49).replace('graph_layout.js?v=49', 'graph_layout.js?v=48')

        with pytest.raises(AssertionError):
            bumped(mixed)

    def test_a_missing_required_asset_still_fails(self) -> None:
        """And it must not cost the presence checks either."""
        from test_index_html import test_redux_cache_buster_bumped as bumped

        without_spark = _synthetic_index(49).replace(
            '<script src="/static/redux/spark_path.js?v=49"></script>\n', ''
        )

        with pytest.raises(AssertionError):
            bumped(without_spark)
