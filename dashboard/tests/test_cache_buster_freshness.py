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
    redux_cache_buster_versions,
    sole_cache_buster_version,
)

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
