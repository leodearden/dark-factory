"""Shared helpers for the `/static/redux/*?v=` cache-buster freshness guard.

`index.html` version-stamps every `/static/redux/*` asset with one shared
`?v=NN`.  That number is the whole cache key: if a `.js` file's bytes change
while its URL does not, a browser holding the old copy never fetches the new
one.  The guard that enforces this lives in `test_index_html.py`; this module
holds the parts of it that are worth testing on their own.

Three layers, deliberately separated (see `test_cache_buster_freshness.py`
for the contract and the reasoning):

  * EXTRACTION — `redux_cache_buster_versions` / `sole_cache_buster_version`.
    Pure text, no I/O.
  * DECISION — `ReduxBaseState` / `cache_buster_violation`.  Pure literals.
  * GIT — `resolve_redux_base_state`.  The only layer that shells out.

The split is what makes the rule testable at all: on a healthy branch
`merge-base(main, HEAD)` is main's tip and no redux asset is modified, so a
base-relative rule is vacuously green against the real repo.  Every failing
case has to be constructed.
"""

from __future__ import annotations

import re

# Repo-root-relative, which is the form both `git show <sha>:<path>` and
# `git diff -- <pathspec>` require.  Pinned in the contract file so moving the
# asset directory breaks loudly rather than leaving the pathspec silently
# matching nothing.
REDUX_ASSET_DIR = 'dashboard/src/dashboard/static/redux'
INDEX_HTML_REL = f'{REDUX_ASSET_DIR}/index.html'

# Copied verbatim from test_index_html.py's inline findall so the uniformity
# check and the freshness check can never drift on WHICH tags count.  The
# `/static/redux/` prefix is what excludes the unpkg CDN script tags (which
# carry an SRI hash full of digits and no `?v=`) and `/static/favicon.svg`.
_VERSION_RE = re.compile(r'/static/redux/[^"?]+\?v=(\d+)')


def redux_cache_buster_versions(body: str) -> set[int]:
    """Every distinct `?v=` version carried by a `/static/redux/*` tag in *body*.

    Returns the empty set when the body carries no versioned redux tag at all
    — an honest "nothing to compare", which `sole_cache_buster_version` turns
    into an error rather than a default.
    """
    return {int(v) for v in _VERSION_RE.findall(body)}


def sole_cache_buster_version(body: str) -> int:
    """The single shared cache-buster version of *body*.

    Raises `ValueError` when the body carries no versioned redux tag, or
    carries more than one distinct version.  Raising rather than returning a
    sentinel is load-bearing: any sentinel would let every downstream
    freshness comparison pass vacuously against an empty or restructured
    index.html, which is a permanent false GREEN.
    """
    versions = redux_cache_buster_versions(body)
    if not versions:
        raise ValueError(
            'no /static/redux/*?v= cache-buster tag found in the given body, so '
            'there is no version to compare — index.html has either been '
            'restructured or was not served.'
        )
    if len(versions) > 1:
        raise ValueError(
            f'mixed /static/redux/*?v= cache-buster versions: {sorted(versions)} — '
            'there is no single version to compare against; bump all of them '
            'uniformly to the same value.'
        )
    return next(iter(versions))
