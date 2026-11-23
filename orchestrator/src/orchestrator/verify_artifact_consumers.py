"""Which tests consume a behavioural artifact, and which modules own those tests.

A behavioural artifact is a non-``.py``/``.rs`` file whose content is
behaviour: a script the tests execute, a ``SKILL.md`` loaded verbatim into an
agent prompt, or the operational yaml the test suites bind to. A diff touching
one is not trivial, so ``orchestrator/src/orchestrator/verify_plan.py`` routes
it to the consumers listed here.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase


@dataclass(frozen=True)
class ArtifactConsumers:
    """Artifact *globs* (``fnmatchcase``, ``*`` crosses ``/``) and the tests that consume them.

    *test_files* are worktree-root-relative collectable tests; *full_suites*
    are module prefixes whose whole suite consumes the artifact.
    """

    globs: tuple[str, ...]
    test_files: tuple[str, ...] = ()
    full_suites: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.globs:
            raise ValueError(f'{self!r} names no artifact glob')
        if not self.test_files and not self.full_suites:
            raise ValueError(f'{self!r} names no consumer test file or full suite')


@dataclass(frozen=True)
class ConsumerDemand:
    """What one module must run because *artifacts* were touched."""

    artifacts: tuple[str, ...]
    test_files: tuple[str, ...]
    full_suite: bool


BEHAVIOURAL_ARTIFACTS: tuple[ArtifactConsumers, ...] = (
    ArtifactConsumers(
        globs=('skills/spawn/*.sh',),
        test_files=(
            'orchestrator/tests/test_session_hooks.py',
            'orchestrator/tests/test_session_hooks_bash.py',
            'scripts/tests/test_legibility_inventory.py',
            'tests/scripts/test_spawn_claude.py',
        ),
    ),
    ArtifactConsumers(
        globs=('skills/unblock-auto/SKILL.md', 'skills/escalation-watcher-auto/SKILL.md'),
        test_files=(
            'orchestrator/tests/test_harness_watcher_supervisor.py',
            'orchestrator/tests/test_skill_prompt.py',
        ),
    ),
    ArtifactConsumers(
        globs=('dark-factory-orchestrator.yaml',),
        full_suites=('tests/scripts',),
    ),
)


def _owning_prefix(path: str, module_prefixes: Iterable[str]) -> str | None:
    """The longest prefix in *module_prefixes* that *path* lies under, else None."""
    owners = [p for p in module_prefixes if path.startswith(p + '/')]
    return max(owners, key=len, default=None)


def consumer_demand(
    files: Sequence[str],
    module_prefixes: Sequence[str],
    table: Sequence[ArtifactConsumers] = BEHAVIOURAL_ARTIFACTS,
) -> dict[str, ConsumerDemand]:
    """Per owning module prefix, the consumers of the artifacts *files* touches.

    A consumer that no prefix in *module_prefixes* owns is dropped, and a
    module left with no demand is absent from the result.
    """
    artifacts: dict[str, set[str]] = {}
    tests: dict[str, set[str]] = {}
    full: set[str] = set()
    for entry in table:
        touched = [f for f in files if any(fnmatchcase(f, g) for g in entry.globs)]
        if not touched:
            continue
        for path in entry.test_files:
            owner = _owning_prefix(path, module_prefixes)
            if owner is not None:
                tests.setdefault(owner, set()).add(path)
                artifacts.setdefault(owner, set()).update(touched)
        for prefix in entry.full_suites:
            if prefix in module_prefixes:
                full.add(prefix)
                artifacts.setdefault(prefix, set()).update(touched)
    return {
        prefix: ConsumerDemand(
            artifacts=tuple(sorted(touched)),
            test_files=tuple(sorted(tests.get(prefix, ()))),
            full_suite=prefix in full,
        )
        for prefix, touched in artifacts.items()
    }
