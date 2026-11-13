"""``paged_read.py`` is the graphiti-free home of the paginated read primitive.

Two guards hold that shape: importing it loads neither graphiti nor the FalkorDB
client, and graphiti_client's old spellings are the very objects defined here,
so no second copy can grow back there.
"""

from pathlib import Path

import pytest
from _import_closure import SRC, modules_loaded_by_importing

from fused_memory.backends import graphiti_client, paged_read


def test_the_paged_read_module_loads_no_graphiti_and_no_falkordb():
    probe = modules_loaded_by_importing(
        'fused_memory.backends.paged_read',
        (
            'graphiti_core',
            'graphiti_core.driver.falkordb_driver',
            'fused_memory.backends.graphiti_client',
            'falkordb',
        ),
    )

    assert Path(probe.file).resolve() == SRC / 'fused_memory' / 'backends' / 'paged_read.py'
    assert probe.loaded == ()


@pytest.mark.parametrize(
    ('graphiti_client_name', 'paged_read_name'),
    [
        ('PagedRead', 'PagedRead'),
        ('_paged_ro_query', 'paged_ro_query'),
        ('_DEFAULT_READ_PAGE_SIZE', 'DEFAULT_READ_PAGE_SIZE'),
        ('_RESULTSET_SIZE', 'RESULTSET_SIZE'),
        ('INCOMPLETE_STRUCTURAL_KINDS', 'INCOMPLETE_STRUCTURAL_KINDS'),
        ('INCOMPLETE_KINDS', 'INCOMPLETE_KINDS'),
    ],
)
def test_the_names_graphiti_client_reads_with_are_the_paged_read_objects(
    graphiti_client_name, paged_read_name
):
    assert getattr(graphiti_client, graphiti_client_name) is getattr(
        paged_read, paged_read_name
    )


@pytest.mark.parametrize(
    ('graphiti_client_name', 'paged_read_name'),
    [
        ('_MAX_READ_PAGES', 'MAX_READ_PAGES'),
        ('_census_count', 'census_count'),
        ('INCOMPLETE_STRUCTURAL_REFUSAL', 'INCOMPLETE_STRUCTURAL_REFUSAL'),
        ('INCOMPLETE_PAGE_CAP', 'INCOMPLETE_PAGE_CAP'),
        ('INCOMPLETE_CENSUS_UNAVAILABLE', 'INCOMPLETE_CENSUS_UNAVAILABLE'),
        ('INCOMPLETE_SHORT_READ', 'INCOMPLETE_SHORT_READ'),
    ],
)
def test_graphiti_client_re_exports_only_the_paged_read_objects(
    graphiti_client_name, paged_read_name
):
    """These names exist in graphiti_client only for old importers; each row goes when its re-export does (task 9363)."""
    assert getattr(graphiti_client, graphiti_client_name) is getattr(
        paged_read, paged_read_name
    )
