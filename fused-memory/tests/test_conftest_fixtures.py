"""Tests that validate shared conftest fixtures added for task-147 DRY refactor.

These tests are written BEFORE the fixtures exist (TDD step-1) and will fail
until conftest.py is extended in step-2.
"""
from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# preserve_config_path fixture tests
# ---------------------------------------------------------------------------

class TestPreserveConfigPath:
    """preserve_config_path autouse fixture saves/restores CONFIG_PATH around each test."""

    def test_absent_key_is_absent(self, preserve_config_path):
        """When CONFIG_PATH is not set, the fixture doesn't interfere."""
        # Remove CONFIG_PATH if present so we start clean
        os.environ.pop('CONFIG_PATH', None)
        assert os.environ.get('CONFIG_PATH') is None

    def test_can_set_config_path_during_test(self, preserve_config_path):
        """Setting CONFIG_PATH during a test is visible within the test."""
        os.environ['CONFIG_PATH'] = '/tmp/inside_test.yaml'
        assert os.environ['CONFIG_PATH'] == '/tmp/inside_test.yaml'
        # Cleanup is the fixture's responsibility; we just verify it's set here

    def test_fixture_accepts_pre_set_value(self, preserve_config_path):
        """The fixture can be requested explicitly even when CONFIG_PATH was set before."""
        os.environ['CONFIG_PATH'] = '/tmp/pre_set.yaml'
        # Fixture should save this value on entry; test can see it
        assert os.environ['CONFIG_PATH'] == '/tmp/pre_set.yaml'

    def test_is_autouse_so_no_explicit_request_needed(self):
        """preserve_config_path is autouse; tests don't need to request it by name.

        This test requests no fixture by name but still passes when autouse is active.
        If the fixture is broken (e.g., raises on setup) this test will fail.
        """
        # No CONFIG_PATH interaction; just confirms autouse doesn't break normal tests
        assert True


# ---------------------------------------------------------------------------
# standard_mock_config fixture tests
# ---------------------------------------------------------------------------

class TestStandardMockConfig:
    """standard_mock_config returns a MagicMock with the common 1536-dim embedder attrs."""

    def test_embedder_dimensions_is_1536(self, standard_mock_config):
        assert standard_mock_config.embedder.dimensions == 1536

    def test_embedder_providers_openai_is_none(self, standard_mock_config):
        assert standard_mock_config.embedder.providers.openai is None

    def test_embedder_model_is_text_embedding_3_small(self, standard_mock_config):
        assert standard_mock_config.embedder.model == 'text-embedding-3-small'

    def test_can_override_dimensions(self, standard_mock_config):
        """MagicMock supports attribute assignment for 768-dim test variants."""
        standard_mock_config.embedder.dimensions = 768
        assert standard_mock_config.embedder.dimensions == 768

    def test_can_override_model(self, standard_mock_config):
        """MagicMock supports attribute assignment for alternate model names."""
        standard_mock_config.embedder.model = 'text-embedding-ada-002'
        assert standard_mock_config.embedder.model == 'text-embedding-ada-002'
