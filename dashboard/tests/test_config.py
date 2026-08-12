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

from dashboard.config import DashboardConfig, _discover_escalation_urls

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

    @pytest.mark.parametrize(
        ('earlier_raw', 'expect_parse_warning', 'expect_mapping_warning'),
        [
            (yaml.safe_dump({'escalation': {'host': '127.0.0.1'}}), False, False),
            ('escalation: [unclosed', True, False),
            ('escalation: not-a-dict', False, True),
        ],
        ids=['portless-earlier', 'unparseable-earlier', 'non-mapping-earlier'],
    )
    def test_legacy_fallthrough_past_bad_earlier_spelling(
        self, tmp_path, caplog, earlier_raw, expect_parse_warning, expect_mapping_warning
    ):
        """A no-URL earlier legacy spelling does not halt the fallback loop.

        With the canonical config absent, ``_discover_root_escalation_url``
        iterates ``_LEGACY_CONFIG_NAMES`` in order. A portless, unparseable,
        or non-mapping-``escalation:`` config at an EARLIER spelling
        (``orchestrator.yaml``) must not stop iteration nor be treated as
        authoritative — resolution continues to a LATER legacy spelling
        (``orchestrator/config.yaml``) that carries a valid port, still
        emitting the 'please migrate' nudge naming the project. Guards
        against a regression that stopped after the first legacy candidate
        returned None.
        """
        root = tmp_path / 'myproj'
        # Earlier spelling (first in _LEGACY_CONFIG_NAMES): yields no URL.
        earlier = root / 'orchestrator.yaml'
        earlier.parent.mkdir(parents=True, exist_ok=True)
        earlier.write_text(earlier_raw)
        # Later spelling (last in _LEGACY_CONFIG_NAMES): valid port.
        _write_yaml(root / 'orchestrator/config.yaml', {'escalation': {'port': 9202}})

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = _discover_escalation_urls([root])

        # Resolved via the LATER legacy spelling, not the bad earlier one.
        assert result == {'myproj': 'http://127.0.0.1:9202/mcp'}
        messages = [r.getMessage() for r in caplog.records if r.name == _LOGGER_NAME]
        migrate_msgs = [m for m in messages if 'please migrate' in m]
        assert len(migrate_msgs) == 1
        assert 'myproj' in migrate_msgs[0]
        assert str(root / 'orchestrator/config.yaml') in migrate_msgs[0]
        # Unparseable earlier config additionally logs one parse-failure warning.
        assert sum('Failed to read' in m for m in messages) == (1 if expect_parse_warning else 0)
        # Non-mapping-escalation earlier config logs one mapping-type warning.
        assert sum('not a mapping' in m for m in messages) == (1 if expect_mapping_warning else 0)
        # The root DID resolve — no no-URL warning fired.
        assert not any('No escalation URL discovered' in m for m in messages)


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

    def test_portless_canonical_is_authoritative_over_valid_legacy(self, tmp_path, caplog):
        """A portless canonical config is NOT masked by a valid legacy config.

        The canonical file existing at all makes it authoritative: it must
        not silently fall through to a legacy spelling's port just because
        the canonical config itself is misconfigured. This asserts the
        no-URL warning (naming the root) fires instead of the legacy-
        fallback warning (naming the project) and the legacy port is
        ignored entirely.
        """
        root = tmp_path / 'myproj'
        _write_yaml(root / 'dark-factory-orchestrator.yaml', {'escalation': {'host': '127.0.0.1'}})
        _write_yaml(root / 'orchestrator.yaml', {'escalation': {'port': 9202}})

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = _discover_escalation_urls([root])

        assert result == {}
        records = [r for r in caplog.records if r.name == _LOGGER_NAME]
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        assert str(root) in records[0].getMessage()
        assert 'please migrate' not in records[0].getMessage()

    def test_malformed_canonical_is_authoritative_over_valid_legacy(self, tmp_path, caplog):
        """A malformed canonical config is NOT masked by a valid legacy config.

        Mirrors ``test_portless_canonical_is_authoritative_over_valid_legacy``
        for the ``_read_escalation_url`` parse-failure branch: the canonical
        file's mere presence — even unparseable — makes it authoritative, so
        resolution never falls through to the legacy port. Two WARNINGs fire
        (the parse-failure naming the file, plus the no-URL one naming the
        root); neither is the legacy-fallback warning, and the legacy port
        is ignored entirely.
        """
        root = tmp_path / 'myproj'
        _write_yaml(root / 'orchestrator.yaml', {'escalation': {'port': 9202}})
        (root / 'dark-factory-orchestrator.yaml').write_text('escalation: [unclosed')

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = _discover_escalation_urls([root])

        assert result == {}
        records = [r for r in caplog.records if r.name == _LOGGER_NAME]
        assert len(records) == 2
        assert all(r.levelno == logging.WARNING for r in records)
        messages = [r.getMessage() for r in records]
        assert sum('Failed to read' in m for m in messages) == 1
        assert sum('No escalation URL discovered' in m for m in messages) == 1
        assert not any('please migrate' in m for m in messages)


class TestNonMappingEscalationSection:
    """Scope 3: a non-mapping ``escalation:`` value, or a non-mapping YAML
    root, is malformed configuration. Both map to the same warn-and-
    return-None path as a YAML parse error — never to an exception —
    regardless of whether the wrong-typed value is truthy (``'x'``, ``5``,
    ``true``, ``[a, b]``) or falsy (``[]``).
    """

    @pytest.mark.parametrize(
        ('raw_body', 'type_name'),
        [
            ('escalation: not-a-dict', 'str'),
            ('escalation: 5', 'int'),
            ('escalation: true', 'bool'),
            ('escalation: [a, b]', 'list'),
            ('escalation: []', 'list'),
        ],
        ids=['str', 'int', 'bool', 'nonempty-list', 'empty-list'],
    )
    def test_non_mapping_escalation_warns_and_skips(
        self, tmp_path, caplog, raw_body, type_name
    ):
        """A non-mapping ``escalation:`` value warns and yields no URL — never raises.

        Covers both truthy wrong-typed values (which previously raised
        ``AttributeError`` out of ``_discover_escalation_urls``) and the
        falsy ``[]`` case (which previously degraded silently, with no
        file-naming warning at all — only the caller's generic no-URL
        warning fired).
        """
        root = tmp_path / 'myproj'
        canonical_path = root / 'dark-factory-orchestrator.yaml'
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        canonical_path.write_text(raw_body)

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = _discover_escalation_urls([root])

        assert result == {}
        records = [r for r in caplog.records if r.name == _LOGGER_NAME]
        assert len(records) == 2
        assert all(r.levelno == logging.WARNING for r in records)
        messages = [r.getMessage() for r in records]
        mapping_msgs = [
            m
            for m in messages
            if 'not a mapping' in m and str(canonical_path) in m and type_name in m
        ]
        assert len(mapping_msgs) == 1
        assert sum('No escalation URL discovered' in m for m in messages) == 1
        assert not any('please migrate' in m for m in messages)

    @pytest.mark.parametrize(
        ('raw_body', 'type_name'),
        [
            ('just-a-string', 'str'),
            ('- a\n- b', 'list'),
        ],
        ids=['scalar-root', 'list-root'],
    )
    def test_non_mapping_yaml_root_warns_and_skips(
        self, tmp_path, caplog, raw_body, type_name
    ):
        """A YAML document whose top-level value is not a mapping warns and yields no URL.

        Pins that the pre-existing root guard becomes diagnostic — naming
        the file and the offending type — rather than silently swallowing
        the file.
        """
        root = tmp_path / 'myproj'
        canonical_path = root / 'dark-factory-orchestrator.yaml'
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        canonical_path.write_text(raw_body)

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = _discover_escalation_urls([root])

        assert result == {}
        records = [r for r in caplog.records if r.name == _LOGGER_NAME]
        assert len(records) == 2
        assert all(r.levelno == logging.WARNING for r in records)
        messages = [r.getMessage() for r in records]
        mapping_msgs = [
            m
            for m in messages
            if 'not a mapping' in m and str(canonical_path) in m and type_name in m
        ]
        assert len(mapping_msgs) == 1

    def test_non_mapping_canonical_is_authoritative_over_valid_legacy(self, tmp_path, caplog):
        """A non-mapping canonical config is NOT masked by a valid legacy config.

        Mirrors ``test_malformed_canonical_is_authoritative_over_valid_legacy``
        for the type-check branch: the canonical file's mere presence — even
        with a wrong-typed ``escalation:`` — makes it authoritative, so
        resolution never falls through to the legacy port.
        """
        root = tmp_path / 'myproj'
        _write_yaml(root / 'orchestrator.yaml', {'escalation': {'port': 9202}})
        (root / 'dark-factory-orchestrator.yaml').write_text('escalation: not-a-dict')

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = _discover_escalation_urls([root])

        assert result == {}
        records = [r for r in caplog.records if r.name == _LOGGER_NAME]
        assert len(records) == 2
        assert all(r.levelno == logging.WARNING for r in records)
        messages = [r.getMessage() for r in records]
        assert sum('not a mapping' in m for m in messages) == 1
        assert sum('No escalation URL discovered' in m for m in messages) == 1
        assert not any('please migrate' in m for m in messages)

    def test_startup_survives_a_malformed_project_config(self, tmp_path, caplog):
        """``DashboardConfig()`` construction survives a malformed known-project config.

        The blast-radius regression: one operator typo in one *known*
        project's config must not abort dashboard startup for the process
        as a whole. Exercised at the real entry point
        (``DashboardConfig.__post_init__``), not just the private discovery
        helper — a future try/except added only around the helper's callers
        would leave a helper-only suite green over a still-broken startup.
        """
        main_root = tmp_path / 'main'
        broken_root = tmp_path / 'broken'
        broken_canonical = broken_root / 'dark-factory-orchestrator.yaml'
        broken_canonical.parent.mkdir(parents=True, exist_ok=True)
        broken_canonical.write_text('escalation: not-a-dict')

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            cfg = DashboardConfig(project_root=main_root, known_project_roots=[broken_root])

        assert cfg.escalation_urls == {}


class TestMultiRootDiscovery:
    """A single call spans multiple roots with independent outcomes."""

    def test_mixed_resolved_and_unresolved_roots(self, tmp_path, caplog):
        """One resolvable root and one unresolvable root in the same call.

        The resolvable root's URL is returned; the unresolvable root is
        simply absent from the result dict, with exactly one no-URL
        warning — naming only the unresolvable root, not the resolvable one.
        """
        good_root = tmp_path / 'goodproj'
        _write_yaml(good_root / 'dark-factory-orchestrator.yaml', {'escalation': {'port': 9101}})
        bad_root = tmp_path / 'badproj'
        bad_root.mkdir()

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = _discover_escalation_urls([good_root, bad_root])

        assert result == {'goodproj': 'http://127.0.0.1:9101/mcp'}
        records = [r for r in caplog.records if r.name == _LOGGER_NAME]
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        assert str(bad_root) in records[0].getMessage()
        assert str(good_root) not in records[0].getMessage()
