"""pyright unresolved-import output specimens shared by the verify tests.

Lives in ``_pyright_import_fixtures.py`` (not ``conftest.py``) to follow the
``_xdist_crash_fixtures.py`` pattern: importable from any test file without
``sys.modules['conftest']`` collisions across subprojects.

Consumers: test_verify_classify.py (the predicates, unit level) and
test_verify_pyright_interpreter_misresolution.py (end to end through
``run_verification``).
"""

from __future__ import annotations


def unresolved_import_lines(*modules: str) -> str:
    """Render *modules* in pyright's REAL unresolved-import line shape.

    Captured verbatim during the task-3367 reproduction, e.g.::

        /repo/src/fused_memory/server.py:12:8 - error: Import "pytest" could
        not be resolved (reportMissingImports)
    """
    return '\n'.join(
        f'  /repo/src/fused_memory/server.py:{i + 8}:8 - error: Import "{mod}" could not be '
        f'resolved (reportMissingImports)'
        for i, mod in enumerate(modules)
    )


# Task 6127 / esc-6021-3: mirrors the measured 6021 attempt-1 laptop type log
# (since pruned from data/verify-logs). 164 reportMissingImports over seven
# top-level modules, pytest NOT among them, because the cold venv was still
# being populated; plus a consequential non-import error. The dotted names are
# the log's own surviving spellings.
PARTIALLY_PROVISIONED_TYPE_OUTPUT = (
    unresolved_import_lines(
        'graphiti_core.driver.falkordb_driver', 'graphiti_core', 'qdrant_client.models',
        'openai', 'falkordb.asyncio', 'mem0', 'anthropic', 'redis',
    )
    + '\n'
    + '  /repo/src/fused_memory/backends/mem0_client.py:9:18 - error: "AsyncMemory" '
    'is unknown import symbol (reportAttributeAccessIssue)\n'
    + '171 errors, 1 warning, 0 informations\n'
)
