"""Tests for orchestrator.verify_artifact_consumers — behavioural-artifact routing.

A behavioural artifact is a non-.py/.rs file whose content is behaviour (a
script tests execute, a SKILL.md loaded verbatim into an agent prompt, the
operational yaml the suites bind to). ``consumer_demand`` maps a diff's
touched artifacts to the registered modules whose tests consume them (task 5296).
"""

from __future__ import annotations

import re
from fnmatch import fnmatchcase
from pathlib import Path

import pytest

from orchestrator.verify_artifact_consumers import (
    BEHAVIOURAL_ARTIFACTS,
    ArtifactConsumers,
    ConsumerDemand,
    consumer_demand,
)
from orchestrator.verify_plan import FileKind, classify_file

REPO_ROOT = Path(__file__).resolve().parents[2]

REGISTRY_PREFIXES = ['orchestrator', 'scripts', 'tests/scripts']

SPAWN_CONSUMERS = {
    'tests/scripts': {'tests/scripts/test_spawn_claude.py'},
    'scripts': {'scripts/tests/test_legibility_inventory.py'},
    'orchestrator': {
        'orchestrator/tests/test_session_hooks.py',
        'orchestrator/tests/test_session_hooks_bash.py',
    },
}

SKILL_PROMPT_CONSUMERS = (
    'orchestrator/tests/test_harness_watcher_supervisor.py',
    'orchestrator/tests/test_skill_prompt.py',
)


class TestConsumerDemandMechanism:
    """The lookup, driven by synthetic tables so each rule is isolated."""

    def test_no_touched_file_matches_any_glob(self) -> None:
        table = (ArtifactConsumers(globs=('a/*.sh',), test_files=('a/tests/test_a.py',)),)
        assert consumer_demand(['README.md', 'b/x.sh'], ['a'], table) == {}

    def test_test_files_route_to_the_longest_owning_prefix(self) -> None:
        table = (ArtifactConsumers(
            globs=('art/run.sh',),
            test_files=('tests/scripts/test_a.py', 'scripts/tests/test_b.py', 'tests/test_c.py'),
        ),)
        demand = consumer_demand(['art/run.sh'], ['scripts', 'tests', 'tests/scripts'], table)
        assert demand == {
            'tests/scripts': ConsumerDemand(
                artifacts=('art/run.sh',), test_files=('tests/scripts/test_a.py',), full_suite=False,
            ),
            'scripts': ConsumerDemand(
                artifacts=('art/run.sh',), test_files=('scripts/tests/test_b.py',), full_suite=False,
            ),
            'tests': ConsumerDemand(
                artifacts=('art/run.sh',), test_files=('tests/test_c.py',), full_suite=False,
            ),
        }

    def test_test_files_and_artifacts_are_sorted_and_deduplicated(self) -> None:
        table = (
            ArtifactConsumers(globs=('art/*.sh',), test_files=('m/tests/test_z.py', 'm/tests/test_a.py')),
            ArtifactConsumers(globs=('art/b.sh',), test_files=('m/tests/test_a.py',)),
        )
        demand = consumer_demand(['art/b.sh', 'art/a.sh'], ['m'], table)
        assert demand == {'m': ConsumerDemand(
            artifacts=('art/a.sh', 'art/b.sh'),
            test_files=('m/tests/test_a.py', 'm/tests/test_z.py'),
            full_suite=False,
        )}

    def test_full_suites_entry_demands_exactly_that_prefix(self) -> None:
        table = (ArtifactConsumers(globs=('root.yaml',), full_suites=('m',)),)
        demand = consumer_demand(['root.yaml'], ['m', 'm/sub', 'other'], table)
        assert demand == {'m': ConsumerDemand(artifacts=('root.yaml',), test_files=(), full_suite=True)}

    def test_unowned_consumers_are_dropped_and_empty_modules_absent(self) -> None:
        table = (ArtifactConsumers(
            globs=('art.sh',),
            test_files=('gone/tests/test_x.py',),
            full_suites=('also-gone',),
        ),)
        assert consumer_demand(['art.sh'], ['m'], table) == {}

    def test_an_owned_consumer_survives_beside_a_dropped_one(self) -> None:
        table = (ArtifactConsumers(
            globs=('art.sh',), test_files=('gone/tests/test_x.py', 'm/tests/test_y.py'),
        ),)
        assert consumer_demand(['art.sh'], ['m'], table) == {'m': ConsumerDemand(
            artifacts=('art.sh',), test_files=('m/tests/test_y.py',), full_suite=False,
        )}

    def test_two_artifacts_demanding_one_module_union(self) -> None:
        table = (
            ArtifactConsumers(globs=('a.sh',), test_files=('m/tests/test_a.py',)),
            ArtifactConsumers(globs=('b.md',), test_files=('m/tests/test_b.py',)),
        )
        demand = consumer_demand(['b.md', 'a.sh'], ['m'], table)
        assert demand == {'m': ConsumerDemand(
            artifacts=('a.sh', 'b.md'),
            test_files=('m/tests/test_a.py', 'm/tests/test_b.py'),
            full_suite=False,
        )}

    def test_a_prefix_is_not_owner_of_a_sibling_sharing_its_spelling(self) -> None:
        table = (ArtifactConsumers(globs=('a.sh',), test_files=('mod-extra/tests/test_a.py',)),)
        assert consumer_demand(['a.sh'], ['mod'], table) == {}

    @pytest.mark.parametrize(
        ('path', 'matches'),
        [
            ('skills/spawn/spawn-claude.sh', True),
            ('skills/spawn/hooks/stop.sh', True),
            ('skills/spawn/SKILL.md', False),
            ('skills/spawn/hooks/README.md', False),
            ('Skills/spawn/x.sh', False),
        ],
    )
    def test_globs_are_fnmatchcase_with_star_crossing_slash(self, path: str, matches: bool) -> None:
        table = (ArtifactConsumers(globs=('skills/spawn/*.sh',), test_files=('m/tests/test_a.py',)),)
        assert bool(consumer_demand([path], ['m'], table)) is matches

    def test_empty_globs_raise_naming_the_entry(self) -> None:
        with pytest.raises(ValueError, match=re.escape("'m/tests/test_named.py'")):
            ArtifactConsumers(globs=(), test_files=('m/tests/test_named.py',))

    def test_no_consumers_raise_naming_the_entry(self) -> None:
        with pytest.raises(ValueError, match=re.escape("'named/glob.sh'")):
            ArtifactConsumers(globs=('named/glob.sh',), test_files=(), full_suites=())


def _on_disk_matches(glob: str) -> list[str]:
    literal = re.split(r'[*?\[]', glob, maxsplit=1)[0]
    if literal == glob:
        return [glob] if (REPO_ROOT / glob).is_file() else []
    base = REPO_ROOT / literal.rpartition('/')[0]
    return [
        rel for rel in (p.relative_to(REPO_ROOT).as_posix() for p in base.rglob('*') if p.is_file())
        if fnmatchcase(rel, glob)
    ]


class TestBehaviouralArtifactsTable:
    """Guards that catch a renamed artifact, consumer test or consumer module."""

    @pytest.mark.parametrize('glob', [g for entry in BEHAVIOURAL_ARTIFACTS for g in entry.globs])
    def test_every_glob_matches_a_file_on_disk(self, glob: str) -> None:
        assert _on_disk_matches(glob), f'{glob!r} matches no file under {REPO_ROOT}'

    @pytest.mark.parametrize('path', [p for entry in BEHAVIOURAL_ARTIFACTS for p in entry.test_files])
    def test_every_consumer_test_exists_and_is_collectable(self, path: str) -> None:
        assert (REPO_ROOT / path).is_file(), path
        assert classify_file(path, None) is FileKind.COLLECTABLE_TEST

    @pytest.mark.parametrize('prefix', [p for entry in BEHAVIOURAL_ARTIFACTS for p in entry.full_suites])
    def test_every_full_suite_prefix_is_a_registered_module(self, prefix: str) -> None:
        assert (REPO_ROOT / prefix / 'orchestrator.yaml').is_file(), prefix


class TestBehaviouralArtifactsAcceptance:
    """The real table, against the real registry's prefixes."""

    @pytest.mark.parametrize('artifact', ['skills/spawn/spawn-claude.sh', 'skills/spawn/hooks/stop.sh'])
    def test_spawn_scripts_demand_every_module_that_executes_them(self, artifact: str) -> None:
        demand = consumer_demand([artifact], REGISTRY_PREFIXES)
        assert set(demand) == set(SPAWN_CONSUMERS)
        for prefix, expected_tests in SPAWN_CONSUMERS.items():
            assert expected_tests <= set(demand[prefix].test_files), prefix
            assert demand[prefix].artifacts == (artifact,)
            assert demand[prefix].full_suite is False

    @pytest.mark.parametrize(
        'artifact', ['skills/unblock-auto/SKILL.md', 'skills/escalation-watcher-auto/SKILL.md'],
    )
    def test_auto_skill_prompts_demand_only_the_prompt_loader_tests(self, artifact: str) -> None:
        assert consumer_demand([artifact], REGISTRY_PREFIXES) == {'orchestrator': ConsumerDemand(
            artifacts=(artifact,), test_files=SKILL_PROMPT_CONSUMERS, full_suite=False,
        )}

    def test_root_yaml_demands_only_the_tests_scripts_full_suite(self) -> None:
        assert consumer_demand(['dark-factory-orchestrator.yaml'], REGISTRY_PREFIXES) == {
            'tests/scripts': ConsumerDemand(
                artifacts=('dark-factory-orchestrator.yaml',), test_files=(), full_suite=True,
            ),
        }

    @pytest.mark.parametrize('prose', ['README.md', 'skills/foo.md'])
    def test_prose_demands_nothing(self, prose: str) -> None:
        assert consumer_demand([prose], REGISTRY_PREFIXES) == {}
