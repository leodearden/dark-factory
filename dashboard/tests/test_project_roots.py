"""Tests for ``DashboardConfig.all_project_roots``, the one walk over every
configured project root.
"""

from __future__ import annotations

import logging

from dashboard.config import DashboardConfig

_LOGGER_NAME = 'dashboard.config'
_NO_URL_MARKER = 'No escalation URL discovered for project root'


def _dirs(tmp_path, *names):
    paths = [tmp_path / name for name in names]
    for path in paths:
        path.mkdir(parents=True)
    return paths


class TestAllProjectRoots:
    def test_primary_first_then_known_roots_in_configured_order(self, tmp_path):
        a, b, c = _dirs(tmp_path, 'a', 'b', 'c')
        config = DashboardConfig(project_root=a, known_project_roots=[b, c])
        assert config.all_project_roots == (a.resolve(), b.resolve(), c.resolve())

    def test_a_repeated_root_keeps_its_first_position(self, tmp_path):
        a, b, c = _dirs(tmp_path, 'a', 'b', 'c')
        config = DashboardConfig(project_root=a, known_project_roots=[b, a, c, b])
        assert config.all_project_roots == (a.resolve(), b.resolve(), c.resolve())

    def test_a_symlinked_alias_is_the_same_root(self, tmp_path):
        (real,) = _dirs(tmp_path, 'real')
        alias = tmp_path / 'alias'
        alias.symlink_to(real, target_is_directory=True)
        config = DashboardConfig(project_root=alias, known_project_roots=[real])
        assert config.all_project_roots == (real.resolve(),)

    def test_a_primary_only_config_names_just_the_primary(self, tmp_path):
        (a,) = _dirs(tmp_path, 'a')
        config = DashboardConfig(project_root=a)
        assert config.all_project_roots == (a.resolve(),)

    def test_the_roster_is_a_tuple_so_no_caller_can_mutate_it(self, tmp_path):
        a, b = _dirs(tmp_path, 'a', 'b')
        config = DashboardConfig(project_root=a, known_project_roots=[b])
        assert isinstance(config.all_project_roots, tuple)

    def test_escalation_url_discovery_probes_each_distinct_root_once(self, tmp_path, caplog):
        a, b = _dirs(tmp_path, 'a', 'b')
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            DashboardConfig(project_root=a, known_project_roots=[a, b, b])
        no_url = [
            r.getMessage()
            for r in caplog.records
            if r.name == _LOGGER_NAME and _NO_URL_MARKER in r.getMessage()
        ]
        assert len(no_url) == 2, no_url
        assert sum(str(a.resolve()) in m for m in no_url) == 1
        assert sum(str(b.resolve()) in m for m in no_url) == 1
