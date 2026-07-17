"""Tests for dashboard.config — canonical-vs-legacy escalation URL discovery.

``_discover_escalation_urls`` must prefer ``dark-factory-orchestrator.yaml``
(the canonical top-level orchestrator config name) over the legacy spellings
left behind by the 7-project migration (commit 97f0bd9f85), falling back to
a legacy spelling only when the canonical file is absent — and logging a
WARNING naming the project whenever that fallback is used.
"""

from __future__ import annotations

import logging

import pytest
import yaml

from dashboard.config import _discover_escalation_urls

_LOGGER_NAME = 'dashboard.config'


def _write_yaml(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


class TestCanonicalConfigPreferred:
    """Scope 1: canonical name wins; legacy spellings are a warned fallback."""

    def test_canonical_only_resolves_with_no_warning(self, tmp_path, caplog):
        """A root with only the canonical config resolves, with zero warnings."""
        root = tmp_path / 'myproj'
        _write_yaml(root / 'dark-factory-orchestrator.yaml', {'escalation': {'port': 9101}})

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = _discover_escalation_urls([root])

        assert result == {'myproj': 'http://127.0.0.1:9101/mcp'}
        assert [r for r in caplog.records if r.name == _LOGGER_NAME] == []

    def test_canonical_wins_over_legacy(self, tmp_path, caplog):
        """When both canonical and legacy configs exist, canonical's port wins."""
        root = tmp_path / 'myproj'
        _write_yaml(root / 'dark-factory-orchestrator.yaml', {'escalation': {'port': 9101}})
        _write_yaml(root / 'orchestrator.yaml', {'escalation': {'port': 9202}})

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = _discover_escalation_urls([root])

        assert result == {'myproj': 'http://127.0.0.1:9101/mcp'}
        assert [r for r in caplog.records if r.name == _LOGGER_NAME] == []

    @pytest.mark.parametrize(
        'legacy_rel_path',
        ['orchestrator.yaml', 'orchestrator-config.yaml', 'orchestrator/config.yaml'],
    )
    def test_legacy_spelling_resolves_with_warning(self, tmp_path, caplog, legacy_rel_path):
        """A root with only a legacy spelling still resolves, but warns naming the project."""
        root = tmp_path / 'myproj'
        _write_yaml(root / legacy_rel_path, {'escalation': {'port': 9202}})

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = _discover_escalation_urls([root])

        assert result == {'myproj': 'http://127.0.0.1:9202/mcp'}
        records = [r for r in caplog.records if r.name == _LOGGER_NAME]
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        assert 'myproj' in records[0].getMessage()


class TestNoUrlWarnsNamingRoot:
    """Scope 2: a root that yields no escalation URL at all warns, naming the root."""

    def test_no_config_at_all_warns_naming_root(self, tmp_path, caplog):
        """A root with no config file of any spelling yields no entry, and warns."""
        root = tmp_path / 'myproj'
        root.mkdir()

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = _discover_escalation_urls([root])

        assert result == {}
        records = [r for r in caplog.records if r.name == _LOGGER_NAME]
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        assert str(root) in records[0].getMessage()

    def test_canonical_without_port_warns_naming_root(self, tmp_path, caplog):
        """Canonical config present but missing escalation.port still yields no entry.

        This isolates the no-URL warning from the legacy-fallback warning —
        no legacy file exists here, so exactly one (the no-URL) warning fires.
        """
        root = tmp_path / 'myproj'
        _write_yaml(root / 'dark-factory-orchestrator.yaml', {'escalation': {'host': '127.0.0.1'}})

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = _discover_escalation_urls([root])

        assert result == {}
        records = [r for r in caplog.records if r.name == _LOGGER_NAME]
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        assert str(root) in records[0].getMessage()
