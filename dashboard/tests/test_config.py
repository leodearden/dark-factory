"""Tests for dashboard.config — escalation URL discovery, plus the curator
ledger path (``TestCuratorEventsDbPath``).

Discovery reads only ``dark-factory-orchestrator.yaml``, the canonical
top-level orchestrator config name. Task 7138 retired the legacy spellings
left behind by the 7-project migration (commit 97f0bd9f85): a stray
``orchestrator.yaml``, ``orchestrator-config.yaml`` or
``orchestrator/config.yaml`` is never read, with or without the canonical
file beside it.
"""

from __future__ import annotations

import dataclasses
import logging

import pytest
import yaml

from dashboard.config import (
    DashboardConfig,
    _discover_escalation_urls,
    discover_root_escalation_url,
)

_LOGGER_NAME = 'dashboard.config'


def _write_yaml(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


def _write_raw(path, text: str) -> None:
    """Write literal (possibly malformed) YAML text — for cases ``_write_yaml``
    can't produce, e.g. unparseable or wrong-typed-at-the-root documents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class TestCanonicalConfigOnly:
    """Scope 1: only the canonical name is read; legacy spellings are never consulted."""

    def test_canonical_only_resolves_with_no_warning(self, tmp_path, caplog):
        """A root with only the canonical config resolves, with zero warnings."""
        root = tmp_path / 'myproj'
        _write_yaml(root / 'dark-factory-orchestrator.yaml', {'escalation': {'port': 9101}})

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = _discover_escalation_urls([root])

        assert result == {'myproj': 'http://127.0.0.1:9101/mcp'}
        assert [r for r in caplog.records if r.name == _LOGGER_NAME] == []

    def test_stray_legacy_file_beside_canonical_is_ignored(self, tmp_path, caplog):
        """A stray legacy file beside the canonical one is never read."""
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
    def test_legacy_spelling_alone_is_not_discovered(self, tmp_path, caplog, legacy_rel_path):
        """A retired legacy spelling with no canonical file beside it is never read."""
        root = tmp_path / 'myproj'
        legacy_path = root / legacy_rel_path
        _write_yaml(legacy_path, {'escalation': {'port': 9202}})

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            url = discover_root_escalation_url(root)
            result = _discover_escalation_urls([root])

        assert url is None
        assert result == {}
        records = [r for r in caplog.records if r.name == _LOGGER_NAME]
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        assert 'No escalation URL discovered' in records[0].getMessage()
        assert str(root) in records[0].getMessage()
        assert 'dark-factory-orchestrator.yaml' in records[0].getMessage()
        assert not any(str(legacy_path) in r.getMessage() for r in caplog.records)


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

        Exactly one warning, the no-URL one, fires.
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

    def test_portless_canonical_ignores_stray_legacy_port(self, tmp_path, caplog):
        """A portless canonical config never takes a stray legacy file's port.

        The misconfigured canonical config is the real problem, so the no-URL
        warning naming the root fires and the legacy port is ignored entirely.
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

    def test_malformed_canonical_ignores_stray_legacy_port(self, tmp_path, caplog):
        """A malformed canonical config never takes a stray legacy file's port.

        Mirrors ``test_portless_canonical_ignores_stray_legacy_port``
        for the ``_read_escalation_url`` parse-failure branch. Two WARNINGs
        fire (the parse-failure naming the file, plus the no-URL one naming
        the root), and the legacy port is ignored entirely.
        """
        root = tmp_path / 'myproj'
        _write_yaml(root / 'orchestrator.yaml', {'escalation': {'port': 9202}})
        _write_raw(root / 'dark-factory-orchestrator.yaml', 'escalation: [unclosed')

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = _discover_escalation_urls([root])

        assert result == {}
        records = [r for r in caplog.records if r.name == _LOGGER_NAME]
        assert len(records) == 2
        assert all(r.levelno == logging.WARNING for r in records)
        messages = [r.getMessage() for r in records]
        assert sum('Failed to read' in m for m in messages) == 1
        assert sum('No escalation URL discovered' in m for m in messages) == 1


class TestNonMappingEscalationSection:
    """Scope 3: a non-mapping ``escalation:`` value, a non-mapping YAML
    root, or a wrong-typed ``host``/``port`` leaf inside an otherwise
    well-formed ``escalation:`` mapping, is malformed configuration. All
    three map to the same warn-and-return-None path as a YAML parse error
    — never to an exception, and never to a silently-malformed URL —
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
        _write_raw(canonical_path, raw_body)

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

    @pytest.mark.parametrize(
        ('raw_body', 'type_name'),
        [
            ('just-a-string', 'str'),
            ('- a\n- b', 'list'),
            ('[]', 'list'),
        ],
        ids=['scalar-root', 'list-root', 'empty-list-root'],
    )
    def test_non_mapping_yaml_root_warns_and_skips(
        self, tmp_path, caplog, raw_body, type_name
    ):
        """A YAML document whose top-level value is not a mapping warns and yields no URL.

        Pins that the pre-existing root guard becomes diagnostic — naming
        the file and the offending type — rather than silently swallowing
        the file. Includes the FALSY ``[]`` root: a naive ``safe_load(f) or
        {}`` coalesce would silently rewrite an empty-list document into
        ``{}`` before the mapping check ever saw it, reopening exactly the
        truthy/falsy asymmetry this module already closed for
        ``escalation:`` values (see the class docstring) one level up, at
        the whole-document level.
        """
        root = tmp_path / 'myproj'
        canonical_path = root / 'dark-factory-orchestrator.yaml'
        _write_raw(canonical_path, raw_body)

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

    def test_non_mapping_canonical_ignores_stray_legacy_port(self, tmp_path, caplog):
        """A non-mapping canonical config never takes a stray legacy file's port.

        Mirrors ``test_malformed_canonical_ignores_stray_legacy_port``
        for the type-check branch: a wrong-typed ``escalation:`` yields no
        URL, and the legacy port is ignored entirely.
        """
        root = tmp_path / 'myproj'
        _write_yaml(root / 'orchestrator.yaml', {'escalation': {'port': 9202}})
        _write_raw(root / 'dark-factory-orchestrator.yaml', 'escalation: not-a-dict')

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = _discover_escalation_urls([root])

        assert result == {}
        records = [r for r in caplog.records if r.name == _LOGGER_NAME]
        assert len(records) == 2
        assert all(r.levelno == logging.WARNING for r in records)
        messages = [r.getMessage() for r in records]
        assert sum('not a mapping' in m for m in messages) == 1
        assert sum('No escalation URL discovered' in m for m in messages) == 1

    @pytest.mark.parametrize(
        ('escalation_value', 'field_name', 'type_name'),
        [
            ({'host': ['a'], 'port': 9101}, 'escalation.host', 'list'),
            ({'host': '127.0.0.1', 'port': [9101]}, 'escalation.port', 'list'),
        ],
        ids=['bad-host-type', 'bad-port-type'],
    )
    def test_non_mapping_leaf_warns_and_skips(
        self, tmp_path, caplog, escalation_value, field_name, type_name
    ):
        """A wrong-typed ``host`` or ``port`` LEAF warns and yields no URL — never a bad URL.

        ``escalation:`` here is itself a well-formed mapping (unlike the
        other tests in this class); only a leaf inside it is wrong-typed.
        Without validating the leaves, a wrong-typed ``host`` or ``port``
        previously formatted straight into the URL string with no
        diagnostic at all — e.g. ``host: [a]`` silently produced
        ``"http://['a']:9101/mcp"``, a malformed URL that would only fail
        much later, at request time, with an opaque connection error
        instead of a startup warning naming the file.
        """
        root = tmp_path / 'myproj'
        canonical_path = root / 'dark-factory-orchestrator.yaml'
        _write_yaml(canonical_path, {'escalation': escalation_value})

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = _discover_escalation_urls([root])

        assert result == {}
        records = [r for r in caplog.records if r.name == _LOGGER_NAME]
        assert len(records) == 2
        assert all(r.levelno == logging.WARNING for r in records)
        messages = [r.getMessage() for r in records]
        leaf_msgs = [
            m for m in messages if field_name in m and str(canonical_path) in m and type_name in m
        ]
        assert len(leaf_msgs) == 1
        assert sum('No escalation URL discovered' in m for m in messages) == 1

    def test_startup_survives_a_malformed_project_config(self, tmp_path, caplog):
        """``DashboardConfig()`` construction survives a malformed known-project config.

        The blast-radius regression: one operator typo in one *known*
        project's config must not abort dashboard startup for the process
        as a whole. Exercised at the real entry point
        (``DashboardConfig.__post_init__``), not just the private discovery
        helper — a future try/except added only around the helper's callers
        would leave a helper-only suite green over a still-broken startup.

        ``main_root`` carries a VALID config so the assertion can tell
        "survived and skipped the bad root" apart from "gave up on
        everything": against an all-broken fixture, both an implementation
        that aborts the whole discovery loop on the first malformed root
        and one that unconditionally returns ``{}`` would still pass a bare
        ``== {}`` assertion.
        """
        main_root = tmp_path / 'main'
        _write_yaml(main_root / 'dark-factory-orchestrator.yaml', {'escalation': {'port': 9101}})
        broken_root = tmp_path / 'broken'
        broken_canonical = broken_root / 'dark-factory-orchestrator.yaml'
        _write_raw(broken_canonical, 'escalation: not-a-dict')

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            cfg = DashboardConfig(project_root=main_root, known_project_roots=[broken_root])

        # The good root still resolves despite the broken sibling.
        assert cfg.escalation_urls == {'main': 'http://127.0.0.1:9101/mcp'}
        messages = [r.getMessage() for r in caplog.records if r.name == _LOGGER_NAME]
        mapping_msgs = [
            m for m in messages if 'not a mapping' in m and str(broken_canonical) in m
        ]
        assert len(mapping_msgs) == 1


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


class TestSameBasenameRoots:
    """Two configured roots sharing a basename keep distinct labels and URLs."""

    def _roots(self, tmp_path):
        a = tmp_path / 'a' / 'proj'
        b = tmp_path / 'b' / 'proj'
        _write_yaml(a / 'dark-factory-orchestrator.yaml', {'escalation': {'port': 9101}})
        _write_yaml(b / 'dark-factory-orchestrator.yaml', {'escalation': {'port': 9102}})
        return a, b

    def test_each_root_is_labelled_by_its_shortest_unique_suffix(self, tmp_path):
        a, b = self._roots(tmp_path)
        config = DashboardConfig(project_root=a, known_project_roots=[b])

        assert config.project_labels.of(a.resolve()) == 'a/proj'
        assert config.project_labels.of(b.resolve()) == 'b/proj'
        assert config.project_labels.roots == config.all_project_roots

    def test_the_label_table_is_built_once_per_config(self, tmp_path):
        a, b = self._roots(tmp_path)
        config = DashboardConfig(project_root=a, known_project_roots=[b])

        assert config.project_labels is config.project_labels

    def test_a_replaced_roster_is_labelled_afresh(self, tmp_path):
        a, b = self._roots(tmp_path)
        config = DashboardConfig(project_root=a, known_project_roots=[b])
        assert config.project_labels.of(a.resolve()) == 'a/proj'

        narrowed = dataclasses.replace(config, known_project_roots=[])

        assert narrowed.project_labels.roots == (a.resolve(),)
        assert narrowed.project_labels.of(a.resolve()) == 'proj'

    def test_each_root_keeps_its_own_escalation_url(self, tmp_path):
        a, b = self._roots(tmp_path)
        expected = {
            'a/proj': 'http://127.0.0.1:9101/mcp',
            'b/proj': 'http://127.0.0.1:9102/mcp',
        }

        assert _discover_escalation_urls([a, b]) == expected
        assert DashboardConfig(project_root=a, known_project_roots=[b]).escalation_urls == expected


class TestCuratorEventsDbPath:
    """``curator_events_db`` resolves where fused-memory writes the curator's
    invocation ledger: ``<reconciliation.data_dir>/curator_events.db``.
    """

    def test_env_override_relocates_curator_ledger(self, tmp_path, monkeypatch):
        monkeypatch.setenv('RECONCILIATION_DATA_DIR', str(tmp_path / 'recon'))

        cfg = DashboardConfig(project_root=tmp_path / 'proj')

        assert cfg.curator_events_db == tmp_path / 'recon' / 'curator_events.db'

    @pytest.mark.parametrize('env_value', [None, ''], ids=['unset', 'empty'])
    def test_env_unset_or_empty_falls_back_to_project_root(
        self, tmp_path, monkeypatch, env_value,
    ):
        if env_value is None:
            monkeypatch.delenv('RECONCILIATION_DATA_DIR', raising=False)
        else:
            monkeypatch.setenv('RECONCILIATION_DATA_DIR', env_value)

        cfg = DashboardConfig(project_root=tmp_path / 'proj')

        assert cfg.curator_events_db == (
            cfg.project_root / 'data' / 'reconciliation' / 'curator_events.db'
        )
