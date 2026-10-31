"""The one home for the workspace pytest-config discovery walk and its two anti-vacuity guards.

IMPORT ME, DO NOT COPY ME. A guard that needs to know which pyproject.toml files
declare ``[tool.pytest.ini_options]`` calls :func:`discovered_pytest_configs`.

This module holds NO test functions of its own and imports no test file. It
resolves by bare name because tests/scripts/conftest.py puts this directory on
sys.path. Its own tests live in ``test_pytest_config_discovery.py`` beside it.
"""
from __future__ import annotations

import pathlib
import tomllib

REPO_ROOT = pathlib.Path(__file__).parents[2]

# ``'.'`` names the ROOT pyproject.toml, the convention
# ``test_merge_gate_parallelism_config.py::_declared_addopts`` already
# established: the root is a genuinely different file from any member's, and
# naming it this way lets it read as one more entry in the sweep instead of a
# special case bolted onto it.
ROOT_CONFIG_NAME = '.'

# A FLOOR, not an equality — 8 at authorship (the repo root plus all seven
# workspace members) — so a member added later is swept with no edit anywhere,
# while a discovery that rots and finds fewer still fails loudly. Same idiom, and
# the same reason, as ``test_marker_registration_drift.py::_MIN_EXPECTED_TEST_FILES``.
MIN_EXPECTED_PYTEST_CONFIGS = 8


def _pytest_ini_options(pyproject: pathlib.Path) -> dict | None:
    """``[tool.pytest.ini_options]`` for *pyproject*, or ``None`` if it declares none.

    Shape copied from ``test_pytest_workspace_collection.py::_pytest_ini_options``,
    with ONE deliberate difference: that helper returns ``{}`` both for an absent
    file and for a config that declares no pytest table, which is right for a
    caller reading a single key out of it. This module DISCOVERS configs by the
    presence of that table, so the two cases have to be distinguishable — an
    empty-but-present table would otherwise be skipped by discovery rather than
    reported by a sweep as a config missing its keys, which is a vacuity hole
    exactly where the sweeps can least afford one.
    """
    if not pyproject.exists():
        return None
    data = tomllib.loads(pyproject.read_text(encoding='utf-8'))
    pytest_table = data.get('tool', {}).get('pytest', {})
    if 'ini_options' not in pytest_table:
        return None
    return pytest_table['ini_options']


def discovered_pytest_configs(
    repo_root: pathlib.Path = REPO_ROOT, *, min_expected: int = MIN_EXPECTED_PYTEST_CONFIGS,
) -> dict[str, dict]:
    """Every pyproject.toml under *repo_root* declaring ``[tool.pytest.ini_options]``.

    DISCOVERED from ``[tool.uv.workspace].members`` rather than hardcoded, so a
    new member joins every sweep with no edit. That is the whole point: the
    defect the sweeps guard against is a config that nobody remembered to hold to
    a policy, and a hand-written list is the same class of forgetting.

    Both anti-vacuity guards live HERE rather than in each caller, because they
    are properties of DISCOVERY and every sweep depends on them.
    """
    root_pyproject = repo_root / 'pyproject.toml'
    root_data = tomllib.loads(root_pyproject.read_text(encoding='utf-8'))
    members = root_data.get('tool', {}).get('uv', {}).get('workspace', {}).get('members', [])

    # A typo'd key here would make the loop below iterate nothing and turn every
    # sweep into a silent pass.
    assert members, (
        '[tool.uv.workspace].members is empty or missing in the root '
        f'pyproject.toml at {repo_root} — discovery reads it to find which '
        'subprojects to check, so every sweep would otherwise pass vacuously.'
    )

    configs: dict[str, dict] = {}
    for name in [ROOT_CONFIG_NAME, *members]:
        ini_options = _pytest_ini_options(repo_root / name / 'pyproject.toml')
        if ini_options is not None:
            configs[name] = ini_options

    assert len(configs) >= min_expected, (
        f'only discovered {len(configs)} pytest configs '
        f'({sorted(configs)}), expected at least {min_expected} (task 5442). '
        'The sweep may be looking in the wrong place — it reads the root '
        'pyproject.toml plus one per [tool.uv.workspace].members entry, with '
        f'{ROOT_CONFIG_NAME!r} naming the root. A config that stopped being '
        'discovered is a config no sweep checks any more.'
    )
    return configs
