"""Tests for fused_memory.maintenance._utils.override_config_path context manager."""

import pytest


class TestOverrideConfigPath:
    """override_config_path sets/restores CONFIG_PATH env var around a block."""

    def test_sets_config_path(self, monkeypatch):
        """Sets CONFIG_PATH when config_path is provided and it was previously unset."""
        from fused_memory.maintenance._utils import override_config_path

        monkeypatch.delenv('CONFIG_PATH', raising=False)

        with override_config_path('/some/path/config.yaml'):
            import os
            assert os.environ.get('CONFIG_PATH') == '/some/path/config.yaml'

    def test_restores_original_value(self, monkeypatch):
        """Restores the previous CONFIG_PATH value after the block exits."""
        from fused_memory.maintenance._utils import override_config_path

        monkeypatch.setenv('CONFIG_PATH', '/original/config.yaml')

        with override_config_path('/new/path/config.yaml'):
            import os
            assert os.environ.get('CONFIG_PATH') == '/new/path/config.yaml'

        import os
        assert os.environ.get('CONFIG_PATH') == '/original/config.yaml'

    def test_removes_env_var_when_previously_unset(self, monkeypatch):
        """Removes CONFIG_PATH after block when it was not set before entry."""
        from fused_memory.maintenance._utils import override_config_path

        monkeypatch.delenv('CONFIG_PATH', raising=False)

        with override_config_path('/some/path/config.yaml'):
            pass  # set during block

        import os
        assert 'CONFIG_PATH' not in os.environ

    def test_noop_when_config_path_is_none(self, monkeypatch):
        """Does not touch CONFIG_PATH at all when config_path is None."""
        import os

        from fused_memory.maintenance._utils import override_config_path

        # Case 1: env var already set — should remain unchanged
        monkeypatch.setenv('CONFIG_PATH', '/existing/config.yaml')
        with override_config_path(None):
            assert os.environ.get('CONFIG_PATH') == '/existing/config.yaml'
        assert os.environ.get('CONFIG_PATH') == '/existing/config.yaml'

        # Case 2: env var not set — should remain absent
        monkeypatch.delenv('CONFIG_PATH', raising=False)
        with override_config_path(None):
            assert 'CONFIG_PATH' not in os.environ
        assert 'CONFIG_PATH' not in os.environ

    def test_restores_on_exception(self, monkeypatch):
        """Restores CONFIG_PATH even when the body raises an exception."""
        from fused_memory.maintenance._utils import override_config_path

        monkeypatch.delenv('CONFIG_PATH', raising=False)

        with pytest.raises(RuntimeError, match='boom'), override_config_path('/some/path/config.yaml'):
            raise RuntimeError('boom')

        import os
        assert 'CONFIG_PATH' not in os.environ
