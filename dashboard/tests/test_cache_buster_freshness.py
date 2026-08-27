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

import pytest
from _cache_buster_helpers import (
    INDEX_HTML_REL,
    REDUX_ASSET_DIR,
    ReduxBaseState,
    cache_buster_violation,
    redux_cache_buster_versions,
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
