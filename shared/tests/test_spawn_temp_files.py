"""Tests for shared.spawn_temp_files — the names of the temp files a Claude CLI
spawn is handed by argv path, and the startup sweep that reclaims the ones a
SIGKILLed spawner never unlinked.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from pathlib import Path

import pytest

from shared.spawn_temp_files import (
    MCP_CONFIG_TEMP_NAME,
    SPAWN_TEMP_NAMES,
    SYSPROMPT_TEMP_NAME,
    SpawnTempName,
)


def test_spawn_temp_names_are_exactly_the_two_kinds() -> None:
    assert SPAWN_TEMP_NAMES == (SYSPROMPT_TEMP_NAME, MCP_CONFIG_TEMP_NAME)


@pytest.mark.parametrize('kind', SPAWN_TEMP_NAMES, ids=lambda k: k.prefix)
def test_matches_a_real_mkstemp_name(kind: SpawnTempName, tmp_path: Path) -> None:
    fd, path = tempfile.mkstemp(prefix=kind.prefix, suffix=kind.suffix, dir=tmp_path)
    os.close(fd)
    assert kind.matches(Path(path).name), path


@pytest.mark.parametrize('kind', SPAWN_TEMP_NAMES, ids=lambda k: k.prefix)
def test_matches_its_own_mkstemp_output_in_the_default_tempdir(
    kind: SpawnTempName, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    fd, path = kind.mkstemp()
    os.close(fd)
    assert Path(path).parent == tmp_path
    assert kind.matches(Path(path).name), path


def test_names_are_the_ones_already_on_hosts() -> None:
    """Legacy orphans on hosts carry these names; keeping them keeps those reclaimable."""
    assert SYSPROMPT_TEMP_NAME.prefix == 'sysprompt_'
    assert SYSPROMPT_TEMP_NAME.suffix == '.txt'
    assert MCP_CONFIG_TEMP_NAME.prefix == 'mcp_'
    assert MCP_CONFIG_TEMP_NAME.suffix == '.json'


@pytest.mark.parametrize('name', [
    'mcp_del7442.py',
    'mcp_config.json',
    'sysprompt_abcd1234.txt.bak',
    'startup_probe_sysprompt_abcd1234.txt',
    'sysprompt_ABCD1234.txt',
    'mcp_abcd12345.json',
])
def test_no_kind_matches_a_name_mkstemp_would_not_have_made(name: str) -> None:
    assert not any(kind.matches(name) for kind in SPAWN_TEMP_NAMES)


def test_spawn_temp_name_is_frozen() -> None:
    frozen_field = 'prefix'
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(SYSPROMPT_TEMP_NAME, frozen_field, 'other_')
