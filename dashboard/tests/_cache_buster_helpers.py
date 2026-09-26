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
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

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


def _git(
    repo_root: Path, *args: str
) -> subprocess.CompletedProcess[str]:
    """Run git in *repo_root*, never inheriting the process cwd.

    The cwd matters: the verify lane runs `cd dashboard && uv run pytest
    tests/`, so the process cwd is the SUBPROJECT, not the repo root, and a
    git call that inherited it would resolve pathspecs against the wrong
    directory.  Follows the `_git` convention at
    `scripts/tests/test_lms_marker_contract.py::_git`, minus `check=True` —
    callers here decide what a non-zero exit means, and the two answers
    differ (see `resolve_redux_base_state`).
    """
    return subprocess.run(
        ['git', *args],
        cwd=repo_root, capture_output=True, text=True, timeout=60,
    )


def resolve_redux_base_state(
    repo_root: Path,
    base_refs: Sequence[str] = ('main', 'origin/main'),
) -> ReduxBaseState | None:
    """What the merge base says about the redux assets, or `None` if unmeasurable.

    Returns `None` — do NOT raise — when no candidate ref resolves.  "No main
    ref" means an sdist install or a shallow clone, i.e. "not measurable
    here", not "the tree is bad"; failing there would be a false red, so the
    caller degrades to a visible pytest skip instead.

    The asymmetry after that point is deliberate: once `merge-base` has
    succeeded we know we are in a real checkout, so a subsequent `git diff`
    failure is a BROKEN tree and raises with the captured stderr rather than
    degrading into an exempt state — loud over silent fail-soft.

    `changed_assets` is measured against the WORKING TREE, not HEAD, so an
    implementer sees the demand during the edit/verify loop rather than only
    after committing.  `--diff-filter=M` restricts it to a path that existed
    at base and now holds different bytes, which is precisely "an asset URL
    browsers already cached now serves something else": an ADDED asset has a
    URL no browser has cached and must not force a bump.  `--no-renames`
    keeps a rename reported as add+delete so it cannot masquerade as a
    modification.
    """
    base_sha = None
    base_ref = None
    for ref in base_refs:
        proc = _git(repo_root, 'merge-base', ref, 'HEAD')
        if proc.returncode == 0 and proc.stdout.strip():
            base_sha = proc.stdout.strip()
            base_ref = ref
            break
    if base_sha is None or base_ref is None:
        return None

    shown = _git(repo_root, 'show', f'{base_sha}:{INDEX_HTML_REL}')
    base_version: int | None = None
    if shown.returncode == 0:
        try:
            base_version = sole_cache_buster_version(shown.stdout)
        except ValueError:
            # The base predates the `?v=` scheme, or was mid-bump.  There is no
            # single number to compare against, which is not a violation.
            base_version = None

    diffed = _git(
        repo_root,
        'diff', '--name-only', '--no-renames', '--diff-filter=M',
        base_sha, '--', REDUX_ASSET_DIR,
    )
    if diffed.returncode != 0:
        raise RuntimeError(
            f'git diff against merge base {base_sha} failed in {repo_root} after '
            f'merge-base({base_ref}, HEAD) had already succeeded, so this is a '
            f'broken checkout rather than an unmeasurable one: {diffed.stderr.strip()}'
        )

    changed_assets = tuple(
        sorted(
            line
            for line in diffed.stdout.splitlines()
            if line and line != INDEX_HTML_REL
        )
    )

    return ReduxBaseState(
        base_ref=base_ref,
        base_commit=base_sha,
        base_version=base_version,
        changed_assets=changed_assets,
    )
