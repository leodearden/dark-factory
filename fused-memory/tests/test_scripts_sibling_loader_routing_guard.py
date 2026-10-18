"""Source guard: no script under ``fused-memory/scripts/`` builds its own
``spec_from_file_location`` loader. Sibling scripts load through
``fused_memory.utils.script_loader.load_script``.

The scope is discovered, every ``*.py`` under ``scripts/``, because the forks
this guard prevents arrived in NEW files (tasks 5845, 6184 and 6151), which a
fixed list would not have covered. The ``tests/`` side has its own guard,
``tests/test_script_loader_routing_guard.py``, with a different remedy.

NOT integration-marked: this file only parses source.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
from _ast_guard import calls_named, parse_python_module

SCRIPTS_ROOT = pathlib.Path(__file__).parent.parent / 'scripts'

LOADER_FACTORY = 'spec_from_file_location'

# (scripts-root-relative module, TOP-LEVEL function name) pairs that may keep
# a local loader. Per function, so an ordinary fork beside an exempt one in the
# same file is still caught.
EXEMPT_LOADERS = frozenset(
    {
        # Same policy as load_script; migration owned by pending task 6569.
        ('census_memory_metadata.py', '_load_probe_module'),
        ('retro_stamp_topics.py', '_load_probe_module'),
        ('normalize_topic_slugs.py', 'load_census_module'),
        ('local_memory_models_eval/harness.py', 'load_probe_module'),
        # Different policy; owned by the 5941 follow-up (agent-followup-5941).
        # Reuses the cache only when it has a FusedMemoryClient attribute.
        ('normalize_topic_slugs.py', 'load_mcp_client_module'),
        # Reuses the cache only when it has a FusedMemoryClient attribute.
        ('migrate_task_metadata_to_x_namespace.py', '_load_sibling_client'),
        # Raises FixtureError rather than ImportError when there is no spec.
        ('bake_off_storage_shape.py', '_load_sibling_script'),
        # Deliberately execs a fresh copy and never registers it.
        ('cgl_eta_auto_apply_impl.py', '_load_migrate'),
    }
)

SHARED_LOADER_IMPORT = 'from fused_memory.utils.script_loader import load_script'


def _discovered_modules() -> list[pathlib.Path]:
    return sorted(
        path for path in SCRIPTS_ROOT.rglob('*.py') if '__pycache__' not in path.parts
    )


def _module_key(path: pathlib.Path) -> str:
    return path.relative_to(SCRIPTS_ROOT).as_posix()


def _top_level_functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]


def _unrouted_loader_lines(path: pathlib.Path) -> list[int]:
    if LOADER_FACTORY not in path.read_text():
        return []
    tree = parse_python_module(path)
    module_key = _module_key(path)
    exempt_calls = {
        id(call)
        for fn in _top_level_functions(tree)
        if (module_key, fn.name) in EXEMPT_LOADERS
        for call in calls_named(fn, LOADER_FACTORY)
    }
    return sorted(
        call.lineno for call in calls_named(tree, LOADER_FACTORY) if id(call) not in exempt_calls
    )


@pytest.mark.parametrize('path', _discovered_modules(), ids=_module_key)
def test_scripts_load_siblings_through_the_shared_loader(path):
    lines = _unrouted_loader_lines(path)
    assert not lines, (
        f'{_module_key(path)} calls {LOADER_FACTORY} at line(s) {lines}. Load a sibling '
        f"script with `{SHARED_LOADER_IMPORT}`, then `load_script(<path>, '<existing "
        f"sys.modules key>')`. A forked loader carries its own caching policy, which "
        f'drifts from the one home in fused_memory/utils/script_loader.py.'
    )


@pytest.mark.parametrize(
    ('module_key', 'function_name'),
    sorted(EXEMPT_LOADERS),
    ids=lambda value: value,
)
def test_every_exemption_names_one_live_loader(module_key, function_name):
    path = SCRIPTS_ROOT / module_key
    assert path.exists(), f'exempt module {module_key} no longer exists'
    matches = [
        fn for fn in _top_level_functions(parse_python_module(path)) if fn.name == function_name
    ]
    assert len(matches) == 1, (
        f'{module_key} defines {len(matches)} top-level functions named {function_name}; '
        f'an exemption must name exactly one'
    )
    assert calls_named(matches[0], LOADER_FACTORY), (
        f'{module_key}::{function_name} no longer calls {LOADER_FACTORY}; drop its exemption'
    )
