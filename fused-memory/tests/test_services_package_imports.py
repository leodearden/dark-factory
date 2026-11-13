"""Importing the services package, or a module that imports only services leaves, must not load services/memory_service.py (task 6635)."""

from pathlib import Path

import pytest
from _import_closure import SRC, modules_loaded_by_importing

_PACKAGE = SRC / 'fused_memory'


@pytest.mark.parametrize(
    ('module', 'source_file'),
    [
        pytest.param(
            'fused_memory.services',
            _PACKAGE / 'services' / '__init__.py',
            id='services-package',
        ),
        pytest.param(
            'fused_memory.services.durable_queue',
            _PACKAGE / 'services' / 'durable_queue.py',
            id='durable_queue',
        ),
        pytest.param(
            'fused_memory.services.topic_anchor',
            _PACKAGE / 'services' / 'topic_anchor.py',
            id='topic_anchor',
        ),
        pytest.param(
            'fused_memory.middleware.dead_letter_escalator',
            _PACKAGE / 'middleware' / 'dead_letter_escalator.py',
            id='dead_letter_escalator',
        ),
    ],
)
def test_importing_it_does_not_load_memory_service(module, source_file):
    probe = modules_loaded_by_importing(module, ('fused_memory.services.memory_service',))

    assert Path(probe.file).resolve() == source_file
    assert probe.loaded == ()
