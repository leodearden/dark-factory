"""Tests verifying the public API contract of the dark-factory-shared package."""

from __future__ import annotations

from pathlib import Path


class TestTopLevelImports:
    """Verify that all public symbols are importable from the top-level shared namespace."""

    def test_top_level_imports(self):
        from shared import (
            AccountConfig,
            AccountState,
            AgentResult,
            SessionBudgetExhausted,
            UsageCapConfig,
            UsageGate,
            invoke_claude_agent,
            invoke_with_cap_retry,
        )

        assert AgentResult is not None
        assert invoke_claude_agent is not None
        assert invoke_with_cap_retry is not None
        assert UsageGate is not None
        assert AccountState is not None
        assert SessionBudgetExhausted is not None
        assert AccountConfig is not None
        assert UsageCapConfig is not None


class TestModuleLevelAll:
    """Verify that each submodule defines __all__ with exactly the expected symbols."""

    def test_cli_invoke_all(self):
        from shared import cli_invoke

        assert hasattr(cli_invoke, '__all__'), 'cli_invoke must define __all__'
        assert set(cli_invoke.__all__) == {
            'AgentResult',
            'invoke_claude_agent',
            'invoke_with_cap_retry',
        }

    def test_usage_gate_all(self):
        from shared import usage_gate

        assert hasattr(usage_gate, '__all__'), 'usage_gate must define __all__'
        assert set(usage_gate.__all__) == {
            'UsageGate',
            'AccountState',
            'SessionBudgetExhausted',
        }

    def test_config_models_all(self):
        from shared import config_models

        assert hasattr(config_models, '__all__'), 'config_models must define __all__'
        assert set(config_models.__all__) == {
            'AccountConfig',
            'UsageCapConfig',
        }


class TestInitAllCompleteness:
    """Verify that shared.__all__ covers the union of all module __all__ entries."""

    def test_init_all_covers_all_module_symbols(self):
        import shared
        from shared import cli_invoke, config_models, usage_gate

        union = set(cli_invoke.__all__) | set(usage_gate.__all__) | set(config_models.__all__)
        assert set(shared.__all__) == union, (
            f'shared.__all__ must equal union of submodule __all__.\n'
            f'Missing from shared.__all__: {union - set(shared.__all__)}\n'
            f'Extra in shared.__all__: {set(shared.__all__) - union}'
        )

    def test_no_private_symbols_in_any_all(self):
        import shared
        from shared import cli_invoke, config_models, usage_gate

        for module, name in [
            (shared, 'shared'),
            (cli_invoke, 'cli_invoke'),
            (usage_gate, 'usage_gate'),
            (config_models, 'config_models'),
        ]:
            private = [s for s in module.__all__ if s.startswith('_')]
            assert private == [], (
                f'{name}.__all__ must not contain private symbols: {private}'
            )


class TestPEP561:
    """Verify PEP 561 py.typed marker is present."""

    def test_pep561_py_typed(self):
        py_typed = (
            Path(__file__).resolve().parent.parent / 'src' / 'shared' / 'py.typed'
        )
        assert py_typed.exists(), (
            f'PEP 561 marker missing: {py_typed}\n'
            'Create an empty shared/src/shared/py.typed file.'
        )


class TestPackageMetadata:
    """Verify importlib.metadata picks up correct package metadata."""

    def test_package_metadata(self):
        import importlib.metadata as meta

        m = meta.metadata('dark-factory-shared')
        assert m['Name'] == 'dark-factory-shared'
        assert m['Version'] == '0.1.0'

        requires = m.get_all('Requires-Dist') or []
        dist_names = {r.split('>=')[0].split('[')[0].strip() for r in requires}
        assert 'httpx' in dist_names, f'httpx missing from Requires-Dist: {requires}'
        assert 'pydantic' in dist_names, f'pydantic missing from Requires-Dist: {requires}'

    def test_shared_version_attribute(self):
        import shared

        assert hasattr(shared, '__version__'), 'shared must expose __version__'
        assert shared.__version__ == '0.1.0'


class TestEditableInstallLocation:
    """Verify the editable install resolves to the local src/ tree."""

    def test_editable_install_location(self):
        import importlib.metadata as meta
        import importlib.util

        # The package must be discoverable by importlib.metadata
        dist = meta.distribution('dark-factory-shared')
        assert dist is not None

        # The shared package must load from somewhere under shared/src
        spec = importlib.util.find_spec('shared')
        assert spec is not None, 'shared package not importable'
        assert spec.origin is not None, 'shared package has no origin (namespace package?)'

        origin = Path(spec.origin).resolve()
        # Must be under a 'src/shared' directory
        assert 'src' in origin.parts, (
            f'shared loaded from unexpected location: {origin}\n'
            'Expected path under shared/src/shared/'
        )
