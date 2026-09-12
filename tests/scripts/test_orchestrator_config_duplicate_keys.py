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

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parents[2]


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
