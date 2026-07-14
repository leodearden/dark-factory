"""Tests for fused_memory.server.manifest_stamping.stamp_capability_manifests.

Unit-level coverage for the commit_planning manifest-stamping helper (PRD γ,
plans/capability-delivered-checks-prd.md): sidecar discovery, α-loader
validation, task_id stamping, and the mechanical (grep/script only)
delivered_checks copy into producer task metadata. Uses tmp_path sidecars and
a mocked task_interceptor — no DB/backend involved (that's covered by the
commit_planning integration tests in test_task_tools.py).
"""

from unittest.mock import AsyncMock

import pytest

from fused_memory.server.manifest_stamping import stamp_capability_manifests


@pytest.mark.asyncio
async def test_no_prd_metadata_returns_none(tmp_path):
    """A batch with no prd_path/prd_task_label metadata is a complete no-op."""
    task_interceptor = AsyncMock()
    ids = ['1']
    tasks_data = [{'id': '1', 'metadata': {'files': ['a.py']}}]

    result = await stamp_capability_manifests(
        project_root=str(tmp_path),
        ids=ids,
        tasks_data=tasks_data,
        task_interceptor=task_interceptor,
    )

    assert result is None
    task_interceptor.update_task.assert_not_called()


@pytest.mark.asyncio
async def test_sidecar_missing_on_disk_returns_none(tmp_path):
    """prd_path/prd_task_label are present but the derived sidecar file doesn't exist."""
    task_interceptor = AsyncMock()
    ids = ['1']
    tasks_data = [
        {
            'id': '1',
            'metadata': {
                'prd_path': 'plans/foo-prd.md',
                'prd_task_label': 'alpha',
            },
        },
    ]

    result = await stamp_capability_manifests(
        project_root=str(tmp_path),
        ids=ids,
        tasks_data=tasks_data,
        task_interceptor=task_interceptor,
    )

    assert result is None
    task_interceptor.update_task.assert_not_called()
