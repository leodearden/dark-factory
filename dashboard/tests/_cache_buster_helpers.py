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
from dataclasses import dataclass

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


@dataclass(frozen=True)
class ReduxBaseState:
    """What `merge-base(main, HEAD)` says about the redux assets, for comparison.

    * `base_ref` / `base_commit` — which ref resolved and the sha it landed on,
      carried purely so a failure message can be acted on without opening git.
    * `base_version` — the single shared `?v=` in index.html AT THE BASE, or
      `None` when the base predates the file or the scheme (nothing to
      compare against, which is not the same as a violation).
    * `changed_assets` — repo-root-relative paths under the redux directory
      that existed at base and now hold DIFFERENT bytes.  index.html itself is
      excluded: it is the file carrying the version, so including it would
      make every bump its own trigger.
    """

    base_ref: str
    base_commit: str
    base_version: int | None
    changed_assets: tuple[str, ...]


def cache_buster_violation(head_version: int, base: ReduxBaseState) -> str | None:
    """The cache-buster rule.  Returns an operator-facing message, or `None`.

    Two rules, evaluated against `merge-base(main, HEAD)` rather than main's
    tip — so a branch's result stays a function of the BRANCH, and an
    unrelated merge landing on main cannot turn an in-flight branch red on its
    own.  (The merge lane rebases before merging, so at the moment this
    actually gates a merge the merge base IS main's tip and the two coincide.)

      * MONOTONIC, unconditional: `head_version >= base.base_version`.
      * FRESH, only when a redux asset was modified in place:
        `head_version > base.base_version`.

    A missing `base_version` short-circuits to `None`: "not measurable" must
    never be reported as "you did something wrong".
    """
    if base.base_version is None:
        return None

    short = base.base_commit[:12]

    if head_version < base.base_version:
        return (
            f'index.html cache-buster went BACKWARDS: {head_version}, but '
            f'{base.base_ref} at merge base {short} already serves '
            f'{base.base_version}. Every /static/redux/* URL would be re-pointed '
            'at bytes browsers already cached from an earlier release. Bump '
            f'every /static/redux/*?v= in {INDEX_HTML_REL} to at least '
            f'{base.base_version}.'
        )

    if base.changed_assets and head_version <= base.base_version:
        listed = ', '.join(base.changed_assets)
        return (
            f'index.html cache-buster is {head_version}, but {base.base_ref} at '
            f'merge base {short} ALREADY RELEASED {base.base_version} — so this '
            'branch changes the CONTENT of assets whose URL is unchanged, and '
            'an already-open browser keeps serving the copy it cached. '
            f'Modified since the merge base: {listed}. Bump every '
            f'/static/redux/*?v= in {INDEX_HTML_REL} to at least '
            f'{base.base_version + 1}. (A version number is only a cache key '
            'if it is strictly newer than every version already released; a '
            'rebase that drops your own bump because main got there first '
            'leaves exactly this state — task/3490.)'
        )

    return None
