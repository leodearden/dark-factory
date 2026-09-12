"""Repo-invariant guard: no orchestrator config the loader reads repeats a mapping key.

THE DEFECT, stated once. A YAML document may declare the same mapping key twice
— most consequentially a second top-level ``verify_env:`` block appended below
an existing one. Both blocks read as intentional, and a diff that adds the
second is TEXTUALLY CLEAN TO MERGE: it touches no line the first block owns, so
no merge conflict, no reviewer prompt, nothing. But PyYAML's SafeLoader keeps
only the LAST occurrence and reports nothing at all, so every key declared in
the earlier block silently ceases to exist in the effective config. Task 4635
hit exactly this: a second ``verify_env:`` carrying ``DF_REQUIRE_SANDBOX_TESTS``
landed below the one carrying the merge-leg ``PYTEST_XDIST_AUTO_NUM_WORKERS``
pin, and the pin — the live arm of an A/B experiment — vanished from the running
fleet with no signal anywhere.

WHY A TEST AND NOT A STRICTER LOADER. The remedy deliberately lives at TEST
time. Teaching ``orchestrator/src/orchestrator/config.py`` to reject a duplicate
key would turn the next occurrence into a failed orchestrator restart — a dead
fleet — where a red test turns it into a merge-gate failure on the branch that
introduced it. Catch it where it is cheap to fix, not where it is expensive to
survive.

THE SCOPE IS THE FILE SET THE LOADER ACTUALLY READS, which is three classes of
file and nothing else:

  * ``orchestrator/src/orchestrator/config.py::_load_defaults`` — the
    package-bundled ``defaults.yaml``;
  * ``orchestrator/src/orchestrator/config.py::YamlSettingsSource`` — the
    project config, ``dark-factory-orchestrator.yaml``;
  * ``orchestrator/src/orchestrator/config.py::_discover_module_configs`` —
    every discovered ``<prefix>/orchestrator.yaml``.

Only in those does a duplicate key degrade a RUNNING orchestrator silently. A
repo-wide ``*.yaml`` sweep would also pass today, but it would pull capability
manifests, docker-compose files and PRD fixtures into a guard whose failure mode
is orchestrator misconfiguration.
"""
from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from orchestrator.config import ModuleConfig, OrchestratorConfig

REPO_ROOT = Path(__file__).parents[2]

ROOT_CONFIG_PATH = REPO_ROOT / 'dark-factory-orchestrator.yaml'


@dataclass(frozen=True)
class DuplicateKey:
    """One repeated mapping key, as data rather than as a rendered message.

    A caller that must aggregate findings across files, count them, or render
    them into a failure message should not have to parse prose back out of a
    formatted string to do it.
    """

    path: Path
    key: object
    first_line: int
    duplicate_line: int


class _DuplicateKeyError(yaml.constructor.ConstructorError):
    """Raised at the first repeated key in a document, carrying both key nodes' marks."""

    def __init__(self, key: object, first_mark: Any, duplicate_mark: Any) -> None:
        super().__init__(
            context=f'while constructing a mapping that already declares {key!r}',
            context_mark=first_mark,
            problem=f'found a duplicate key {key!r}',
            problem_mark=duplicate_mark,
        )
        self.key = key
        self.first_mark = first_mark
        self.duplicate_mark = duplicate_mark


class _NoDuplicateKeysLoader(yaml.SafeLoader):
    """A SafeLoader that raises on a repeated mapping key instead of keeping the last.

    A SUBCLASS, and never a mutation of ``yaml.SafeLoader`` itself: registering
    the constructor on the shared class would change the behaviour of every
    other yaml consumer in the same pytest process — including the production
    loaders under test here, which must keep parsing the way they do in the
    fleet for this file's reproduction tests to mean anything.

    The pure-Python loader rather than ``CSafeLoader``: node ``start_mark``
    line numbers are what every finding is built from, and the C loader's
    speed argument is about hot paths, not about parsing eleven small files
    once.
    """


def _construct_mapping_rejecting_duplicates(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode
) -> dict[Any, Any]:
    """Walk the key nodes for a repeat, then delegate the actual construction."""
    seen: dict[Any, yaml.nodes.Node] = {}
    for key_node, _value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, Hashable):
            # Not ours to report: the delegate below raises the proper
            # "found unhashable key" ConstructorError for this shape.
            continue
        if key in seen:
            raise _DuplicateKeyError(key, seen[key].start_mark, key_node.start_mark)
        seen[key] = key_node
    return yaml.SafeLoader.construct_mapping(loader, node)


_NoDuplicateKeysLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_rejecting_duplicates,
)


def duplicate_keys(path: Path) -> list[DuplicateKey]:
    """Report the first repeated mapping key in *path*, at any nesting level.

    Returns at most one finding per file — the loader stops at the first
    repeat, which is the earliest possible stop inside one document and is
    enough to turn the sweep red and name a file to fix.

    Any OTHER ``yaml.YAMLError`` deliberately PROPAGATES. An orchestrator
    config that does not parse at all is a louder defect than one with a
    duplicate key, and swallowing it into a green empty list would report
    exactly the silence this guard exists to remove.
    """
    try:
        yaml.load(path.read_text(), _NoDuplicateKeysLoader)
    except _DuplicateKeyError as exc:
        return [
            DuplicateKey(
                path=path,
                key=exc.key,
                first_line=exc.first_mark.line + 1,
                duplicate_line=exc.duplicate_mark.line + 1,
            )
        ]
    return []


def _yaml_file(tmp_path: Path, text: str) -> Path:
    """Write *text* to a scratch YAML file and return its path."""
    path = tmp_path / 'config.yaml'
    path.write_text(text)
    return path


def test_a_repeated_top_level_key_is_reported_with_both_line_numbers(tmp_path: Path) -> None:
    """The simplest shape: one key declared twice at the top level."""
    path = _yaml_file(
        tmp_path,
        'project_root: /one\n'
        'max_concurrent_tasks: 4\n'
        'project_root: /two\n',
    )

    findings = duplicate_keys(path)

    assert len(findings) == 1, f'expected exactly one finding, got {findings!r}'
    finding = findings[0]
    assert finding.path == path
    assert finding.key == 'project_root'
    assert (finding.first_line, finding.duplicate_line) == (1, 3), (
        'line numbers must be 1-based and name the FIRST occurrence and the '
        f'duplicate, in that order; got {finding!r}'
    )


def test_a_key_repeated_inside_a_nested_mapping_is_reported(tmp_path: Path) -> None:
    """The "at any level" clause: the duplicate need not be top-level to bite."""
    path = _yaml_file(
        tmp_path,
        'verify_env:\n'
        '  PYTEST_XDIST_AUTO_NUM_WORKERS: "8"\n'
        '  DF_REQUIRE_SANDBOX_TESTS: "1"\n'
        '  PYTEST_XDIST_AUTO_NUM_WORKERS: "16"\n',
    )

    findings = duplicate_keys(path)

    assert len(findings) == 1, f'expected exactly one finding, got {findings!r}'
    finding = findings[0]
    assert finding.key == 'PYTEST_XDIST_AUTO_NUM_WORKERS'
    assert (finding.first_line, finding.duplicate_line) == (2, 4)


def test_a_clean_multi_level_document_yields_no_findings(tmp_path: Path) -> None:
    """No false positives: the same key name under DIFFERENT parents is fine."""
    path = _yaml_file(
        tmp_path,
        'verify_env:\n'
        '  PYTEST_XDIST_AUTO_NUM_WORKERS: "8"\n'
        'roles:\n'
        '  implementer:\n'
        '    model: opus\n'
        '  architect:\n'
        '    model: opus\n',
    )

    assert duplicate_keys(path) == []


def test_a_second_verify_env_block_is_silently_last_wins_and_is_reported(tmp_path: Path) -> None:
    """Task 4635's exact shape, with the silent-degradation mechanism pinned alongside.

    Two assertions, and the first is the point: it records PyYAML's OBSERVED
    behaviour on these bytes, so the guard below is demonstrably protecting
    against something real rather than against a theory about a parser.
    """
    text = (
        'verify_env:\n'
        '  PYTEST_XDIST_AUTO_NUM_WORKERS: "8"\n'
        'max_concurrent_tasks: 4\n'
        'verify_env:\n'
        '  DF_REQUIRE_SANDBOX_TESTS: "1"\n'
    )
    path = _yaml_file(tmp_path, text)

    loaded = yaml.safe_load(text)
    assert loaded['verify_env'] == {'DF_REQUIRE_SANDBOX_TESTS': '1'}, (
        'the mechanism under guard: PyYAML keeps only the LAST block, so the '
        f'earlier one is gone without a word; got {loaded["verify_env"]!r}'
    )

    findings = duplicate_keys(path)

    assert len(findings) == 1, f'expected exactly one finding, got {findings!r}'
    finding = findings[0]
    assert finding.key == 'verify_env'
    assert (finding.first_line, finding.duplicate_line) == (1, 4)


_MUTANT_MARKER_KEY = 'DF_DUPLICATE_KEY_GUARD_MARKER'


def _declared_verify_env(path: Path) -> dict[str, str]:
    """The ``verify_env`` mapping as the file DECLARES it, read through the strict loader.

    Through ``_NoDuplicateKeysLoader`` rather than ``yaml.safe_load``, and that
    is redundant enforcement of the same invariant at a second site rather than
    an accident: if the root config ever regains a duplicate key, this helper
    RAISES instead of quietly handing back the surviving block — which would
    otherwise leave the survival test comparing the survivor against itself and
    agreeing.
    """
    document = yaml.load(path.read_text(), _NoDuplicateKeysLoader) or {}
    return document.get('verify_env') or {}


def _with_second_verify_env_block(text: str) -> str:
    """Append a SECOND top-level ``verify_env:`` block, reproducing task 4635's shape.

    A TEXT transform, never a yaml round trip: this is applied to a copy of
    ``dark-factory-orchestrator.yaml``, ~1400 lines of load-bearing comments a
    parse-and-dump would erase — the same reason
    ``scripts/merge-pytest-n-ab-switch.sh`` edits that file by line. Appending
    below the existing block is also precisely how the real defect arrived:
    it conflicts with nothing.
    """
    return text.rstrip('\n') + f'\n\nverify_env:\n  {_MUTANT_MARKER_KEY}: "1"\n'


def test_every_verify_env_key_declared_in_the_root_config_survives_into_the_effective_config(
    root_config: OrchestratorConfig,
) -> None:
    """Nothing else in this repo reads the EFFECTIVE verify_env, and that was the gap.

    Task 4635's duplicate ``verify_env:`` block was invisible precisely because
    no test ever compared what the yaml DECLARES against what the production
    loader ends up serving. This is that comparison, through the real loader,
    against this worktree's own config.

    KEYS ONLY, NEVER VALUES, and the distinction is load-bearing rather than
    cautious: ``scripts/merge-pytest-n-ab-switch.sh`` legitimately flips
    ``PYTEST_XDIST_AUTO_NUM_WORKERS`` between arms and commits the result, and
    ``config.py::YamlSettingsSource._expand_env_vars`` rewrites any ``${VAR}``
    value between file and effective config — so a value pin would go red on a
    sanctioned operator action. A whole entry VANISHING is exactly where the
    duplicate-key defect bites, and keys are what catch it.

    A SUPERSET, not an equality, for the same reason: ``config.py::load_config``
    folds ``effective_verify_env`` back in for sccache, so the effective mapping
    may legitimately carry keys the file never declared.
    """
    declared = _declared_verify_env(ROOT_CONFIG_PATH)

    assert declared, (
        f'{ROOT_CONFIG_PATH} declares an EMPTY verify_env, so this test would '
        'pass while comparing nothing. Either the block was deleted, or it was '
        'renamed and this guard is now reading the wrong key'
    )
    missing = sorted(set(declared) - set(root_config.verify_env))
    assert not missing, (
        f'{sorted(missing)!r} are declared in {ROOT_CONFIG_PATH.name} but absent '
        'from the effective config the orchestrator serves. The usual cause is a '
        'SECOND top-level `verify_env:` block: PyYAML keeps only the last one and '
        'says nothing. Remedy: fold the blocks into a single mapping'
    )


def test_a_second_top_level_verify_env_block_silently_drops_the_earlier_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The negative control: prove the test above is guarding a real failure.

    A guard that has never been seen to fail is a guard nobody knows the
    polarity of. This reproduces task 4635's defect against the PRODUCTION
    loader — on a mutated COPY, never on the tracked file — and asserts that
    the real loader really does drop the earlier block's keys without a word.

    The mutant is built as a TEXT transform for the reason
    ``scripts/merge-pytest-n-ab-switch.sh`` gives for its own line-based edit of
    the same file: it is ~1400 lines of load-bearing comments that a parse-and-
    dump round trip would destroy.

    ``ORCH_CONFIG_PATH`` is re-pointed BEFORE the config is constructed, which
    is the whole point — ``OrchestratorConfig.settings_customise_sources`` reads
    that env var at construction time, so a config built earlier would still be
    describing the tracked file.
    """
    declared = _declared_verify_env(ROOT_CONFIG_PATH)
    assert declared, f'{ROOT_CONFIG_PATH} declares an empty verify_env; nothing to lose'

    mutant = tmp_path / ROOT_CONFIG_PATH.name
    mutant.write_text(_with_second_verify_env_block(ROOT_CONFIG_PATH.read_text()))

    monkeypatch.setenv('ORCH_CONFIG_PATH', str(mutant))
    effective = OrchestratorConfig(project_root=REPO_ROOT).verify_env

    assert _MUTANT_MARKER_KEY in effective, (
        f'the mutant copy at {mutant} was not the file the loader read — '
        f'effective verify_env is {effective!r}. Without this check the '
        'assertion below would pass vacuously against the pydantic defaults, '
        'which is the reports-green-while-checking-something-else failure this '
        'whole directory exists to prevent'
    )
    survivors = sorted(set(declared) & set(effective))
    assert not survivors, (
        f'expected the second `verify_env:` block to shadow the first entirely, '
        f'but {survivors!r} survived — the reproduction no longer reproduces, so '
        'the guard above may be passing for a reason other than the one claimed'
    )
    assert [f.key for f in duplicate_keys(mutant)] == ['verify_env'], (
        'the detector must flag the very shape the production loader just '
        'swallowed, or it would not have caught task 4635 either'
    )


def test_every_orchestrator_config_the_loader_reads_has_no_duplicate_keys(
    discover_module_configs: Callable[[], dict[str, ModuleConfig]],
) -> None:
    """The guard itself: no config the orchestrator loads repeats a mapping key.

    Aggregated across the whole set rather than stopping at the first offender,
    so one red run names every file that needs folding.
    """
    findings = [
        finding
        for path in _orchestrator_config_paths(discover_module_configs)
        for finding in duplicate_keys(path)
    ]

    assert not findings, (
        'duplicate mapping key(s) in orchestrator config(s) the loader reads:\n'
        + '\n'.join(
            f'  {finding.path.relative_to(REPO_ROOT)}: {finding.key!r} declared at '
            f'line {finding.first_line} and again at line {finding.duplicate_line}'
            for finding in findings
        )
        + '\nPyYAML keeps only the LAST occurrence and reports nothing, so every key '
        'in the earlier block is silently absent from the effective config. Remedy: '
        'fold the blocks into a single mapping.'
    )


def test_the_swept_set_is_the_set_the_loader_actually_reads(
    discover_module_configs: Callable[[], dict[str, ModuleConfig]],
) -> None:
    """The scope-honesty assertion, because a sweep over an empty set reports green.

    The expectation is DERIVED from the same production walk the sweep uses, not
    written down as a roster: a literal list or count of today's module configs
    rots on the next one added, and this directory has already recorded
    hard-coded counts of directory contents going stale as a measured defect.
    """
    prefixes = list(discover_module_configs())
    paths = _orchestrator_config_paths(discover_module_configs)

    assert prefixes, (
        'the production module-config walk found NO module configs, which would '
        'make the sweep above pass while checking almost nothing. Either discovery '
        'broke, or every <prefix>/orchestrator.yaml in the repo was removed'
    )
    assert ROOT_CONFIG_PATH in paths, (
        f'{ROOT_CONFIG_PATH} is the file YamlSettingsSource reads and the one task '
        '4635 actually regressed; a sweep that skips it guards nothing that matters'
    )
    assert DEFAULTS_PATH in paths, (
        f'{DEFAULTS_PATH} is the package-bundled layer _load_defaults reads, and a '
        'duplicate key there degrades every project this orchestrator serves'
    )
    assert sorted(set(paths) - {ROOT_CONFIG_PATH, DEFAULTS_PATH}) == sorted(
        REPO_ROOT / prefix / 'orchestrator.yaml' for prefix in prefixes
    ), (
        'the swept module configs must be exactly one <prefix>/orchestrator.yaml per '
        f'discovered prefix ({prefixes!r}), or the sweep has drifted from the set '
        '_discover_module_configs actually registers'
    )
    assert len(paths) == len(prefixes) + 2, (
        f'expected {len(prefixes)} module configs plus the root config and '
        f'defaults.yaml, got {len(paths)} paths — a duplicate entry would make the '
        'sweep read one file twice and report on it twice'
    )
    missing = [path for path in paths if not path.is_file()]
    assert not missing, (
        f'{missing!r} do not exist, so duplicate_keys was never going to read them. '
        'A swept path that is not a file is a silently empty leg of this guard'
    )
