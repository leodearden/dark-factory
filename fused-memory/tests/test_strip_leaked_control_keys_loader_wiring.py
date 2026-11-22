"""The two scripts that borrow ``FusedMemoryClient`` load it from the right file
under the shared ``sys.modules`` key.

Each loader is one ``load_script(<path>, <key>)`` call, so that pair is all
that can be wrong in it, and a wrong path fails only when an operator runs the
script. ``load_script``'s reuse policy has its own tests in
``test_script_loader.py``; these tests pin only the pair.

``test_strip_leaked_control_keys.py`` registers the key at collection, so each
test runs with it unregistered: a cached entry would let a wrong path pass.
"""
from __future__ import annotations

import sys
import types
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from _fm_helpers import load_script_module

SCRIPTS_DIR = Path(__file__).parent.parent / 'scripts'
CLIENT_SCRIPT = SCRIPTS_DIR / 'strip_leaked_control_keys.py'
CLIENT_KEY = 'strip_leaked_control_keys'

normalize_topic_slugs = load_script_module(
    SCRIPTS_DIR / 'normalize_topic_slugs.py', mod_name='normalize_topic_slugs',
)
migrate_task_metadata = load_script_module(
    SCRIPTS_DIR / 'migrate_task_metadata_to_x_namespace.py',
    mod_name='migrate_task_metadata_to_x_namespace',
)


def assert_loaded_from_the_client_script(module: types.ModuleType) -> None:
    assert module.__file__ is not None
    assert Path(module.__file__).resolve() == CLIENT_SCRIPT.resolve()


@pytest.fixture
def client_key_unregistered() -> Iterator[None]:
    saved = sys.modules.pop(CLIENT_KEY, None)
    yield
    sys.modules.pop(CLIENT_KEY, None)
    if saved is not None:
        sys.modules[CLIENT_KEY] = saved


@pytest.mark.parametrize(
    'load_client_class',
    [normalize_topic_slugs.load_mcp_client_class, migrate_task_metadata._load_sibling_client],
    ids=['normalize_topic_slugs', 'migrate_task_metadata_to_x_namespace'],
)
@pytest.mark.usefixtures('client_key_unregistered')
def test_the_client_class_comes_from_the_sibling_script_under_the_shared_key(
    load_client_class: Callable[[], type],
) -> None:
    client_cls = load_client_class()

    client_module = sys.modules[CLIENT_KEY]
    assert_loaded_from_the_client_script(client_module)
    assert client_cls is client_module.FusedMemoryClient


@pytest.mark.usefixtures('client_key_unregistered')
def test_normalize_topic_slugs_reads_the_client_scripts_default_server() -> None:
    client_module = normalize_topic_slugs.load_mcp_client_module()

    assert_loaded_from_the_client_script(client_module)
    assert isinstance(client_module.DEFAULT_SERVER, str)
