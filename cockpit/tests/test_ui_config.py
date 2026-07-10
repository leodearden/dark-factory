"""Tests for cockpit.ui_config — the cockpit's own UI-state persistence.

cockpit-ui.json (at fleet_root/cockpit-ui.json) is the cockpit's ONLY write
target in C5a (PRD §2/§5 pure-consumer discipline: the view never mutates
registry records). Fail-soft is a hard constraint: a missing or malformed
config file must never raise, only degrade to defaults with a logged
warning.
"""

from __future__ import annotations

import logging

from orchestrator import session_registry as sr


class TestLoadUiConfig:
    def test_absent_file_returns_documented_defaults(self, tmp_path):
        from cockpit.ui_config import CockpitUIConfig, load_ui_config

        cfg = load_ui_config(tmp_path)

        assert cfg == CockpitUIConfig(selected_slug=None, poll_interval=1.5)

    def test_malformed_file_falls_back_to_defaults_with_warning(self, tmp_path, caplog):
        from cockpit.ui_config import CockpitUIConfig, load_ui_config, ui_config_path

        path = ui_config_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{not valid json')

        with caplog.at_level(logging.WARNING):
            cfg = load_ui_config(tmp_path)

        assert cfg == CockpitUIConfig()
        assert any('cockpit-ui.json' in record.message for record in caplog.records)


class TestSaveUiConfig:
    def test_writes_to_fleet_root_cockpit_ui_json(self, tmp_path):
        from cockpit.ui_config import CockpitUIConfig, save_ui_config, ui_config_path

        cfg = CockpitUIConfig(selected_slug='some-slug', poll_interval=2.0)

        save_ui_config(cfg, tmp_path)

        expected_path = sr.fleet_root(tmp_path) / 'cockpit-ui.json'
        assert ui_config_path(tmp_path) == expected_path
        assert expected_path.is_file()

    def test_round_trips_through_load(self, tmp_path):
        from cockpit.ui_config import CockpitUIConfig, load_ui_config, save_ui_config

        cfg = CockpitUIConfig(selected_slug='unblock-df-2085-4242', poll_interval=2.5)

        save_ui_config(cfg, tmp_path)
        loaded = load_ui_config(tmp_path)

        assert loaded == cfg
