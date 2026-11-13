"""The stream_touch incident layout behind task 7515's filer-first attribution tests.

The filer shares ``crates/`` with reify and ``research/`` with know-live, so the
global prefix table drops both as collisions. It grows ``sessions/`` and
``corpus/`` after the registry is built, while mission-control and know-live
own those names, so the registry is a stale snapshot. dark-factory alone owns
``orchestrator/``.
"""

from __future__ import annotations

from pathlib import Path

from fused_memory.middleware.project_prefix_registry import ProjectPrefixRegistry

FILER_PROJECT = 'stream_touch'
TASK_27_FILES = [
    'crates/touch-session/src/plan.rs',
    'crates/touch-session/tests/plan.rs',
    'sessions/plans/lp-starter-v1.toml',
]


def _mkproj(parent: Path, name: str, dirs: list[str]) -> Path:
    root = parent / name
    root.mkdir()
    for d in dirs:
        (root / d).mkdir()
    return root


def stale_snapshot_registry(parent: Path) -> ProjectPrefixRegistry:
    """Lay the five incident projects out under *parent*; return the stale registry.

    The filer's root is ``registry.root_for_project(FILER_PROJECT)``.
    """
    filer = _mkproj(parent, 'stream-touch', ['crates', 'research'])
    roots = [
        filer,
        _mkproj(parent, 'reify', ['crates']),
        _mkproj(parent, 'mission-control', ['sessions']),
        _mkproj(parent, 'know-live', ['corpus', 'research']),
        _mkproj(parent, 'dark-factory', ['orchestrator']),
    ]
    registry = ProjectPrefixRegistry.from_roots([str(r) for r in roots])
    (filer / 'sessions').mkdir()
    (filer / 'corpus').mkdir()
    return registry
