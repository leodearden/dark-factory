"""Hermetic synthetic fixtures for test_eval_task_sampler.

Kept in a uniquely-named sibling module (not ``conftest``) per the
orchestrator tests' import convention (see ``conftest`` docstring), so it can
be imported directly from test files without a ``sys.modules['conftest']``
clash with sibling subprojects.

``make_candidate`` builds a :class:`CompletedTaskCandidate` that lands in a
requested ``(repo, kind, path)`` cell, self-checking against the real
classifiers by default so a synthetic corpus can never silently misclassify
out from under the sampling tests.
"""

from __future__ import annotations

from orchestrator.evals.task_sampler import (
    CompletedTaskCandidate,
    classify_kind,
    classify_path,
    repo_of,
)

_REPO_PROJECT = {'df': 'dark_factory', 'reify': 'reify'}
_REPO_ROOT = {'df': '/home/leo/src/dark-factory', 'reify': '/home/leo/src/reify'}

# Titles chosen so classify_kind deterministically returns the intended kind
# and carry NO hard-blocker token (so path='simple' stays simple).
_KIND_TITLE = {
    'bugfix': 'Bug: fix incorrect result',
    'feature': 'Add capability to the module',
    'refactor': 'Refactor: extract helper',
}


def make_candidate(
    task_id: object,
    *,
    repo: str = 'df',
    kind: str = 'feature',
    path: str = 'full',
    verify: bool = True,
    **overrides: object,
) -> CompletedTaskCandidate:
    """Build a candidate in the ``(repo, kind, path)`` cell.

    ``complexity`` is derived from *path* (``'simple'`` → ``'simple'``, else
    ``'high'``); title from *kind*. Any field can be forced via *overrides*.
    With ``verify=True`` (default) the built candidate is asserted to classify
    into exactly ``(repo, kind, path)``.
    """
    complexity = 'simple' if path == 'simple' else 'high'
    fields = dict(
        task_id=str(task_id),
        project=_REPO_PROJECT[repo],
        project_root=_REPO_ROOT[repo],
        title=_KIND_TITLE[kind],
        description='',
        complexity=complexity,
        modules=[],
        pre_commit=f'pre{task_id}',
        post_commit=f'post{task_id}',
        merge_sha=f'merge{task_id}',
    )
    fields.update(overrides)
    cand = CompletedTaskCandidate(**fields)  # type: ignore[arg-type]
    if verify:
        got = (repo_of(cand), classify_kind(cand), classify_path(cand))
        assert got == (repo, kind, path), (
            f'make_candidate({task_id}) classified {got}, expected {(repo, kind, path)}'
        )
    return cand


def rich_corpus() -> list[CompletedTaskCandidate]:
    """22 candidates across 7 cells, one cell deliberately over-supplied.

    ``('df', 'feature', 'full')`` holds 10 of the 22 so the round-robin
    fairness test can assert an over-supplied cell does NOT dominate a bounded
    [10,14] selection while six other cells each have supply (2 apiece).
    """
    cands: list[CompletedTaskCandidate] = []
    n = 0
    for _ in range(10):
        n += 1
        cands.append(make_candidate(f't{n}', repo='df', kind='feature', path='full'))
    other_cells = [
        ('df', 'bugfix', 'full'),
        ('df', 'refactor', 'full'),
        ('df', 'feature', 'simple'),
        ('reify', 'bugfix', 'full'),
        ('reify', 'feature', 'full'),
        ('reify', 'refactor', 'full'),
    ]
    for repo, kind, path in other_cells:
        for _ in range(2):
            n += 1
            cands.append(make_candidate(f't{n}', repo=repo, kind=kind, path=path))
    return cands
