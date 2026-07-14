"""Tests for fused_memory.server.manifest_stamping.stamp_capability_manifests.

Unit-level coverage for the commit_planning manifest-stamping helper (PRD γ,
plans/capability-delivered-checks-prd.md): sidecar discovery, α-loader
validation, task_id stamping, and the mechanical (grep/script only)
delivered_checks copy into producer task metadata. Uses tmp_path sidecars and
a mocked task_interceptor — no DB/backend involved (that's covered by the
commit_planning integration tests in test_task_tools.py).
"""

import json
from unittest.mock import AsyncMock

import pytest
import yaml

from fused_memory.server.manifest_stamping import stamp_capability_manifests

_HAPPY_PATH_SIDECAR_YAML = """\
prd: plans/foo-prd.md
schema_version: 1
tasks:
  - label: alpha
    task_id: null
    title: Do the thing
    capabilities:
      - name: grep_check
        binding: 'grep for the marker'
        verdict: PASS
        delivered_check:
          kind: grep
          pattern: 'TODO(alpha)'
          expect: absent
          paths:
            - src/foo.py
      - name: script_check
        binding: 'run the checker script'
        verdict: PASS
        delivered_check:
          kind: script
          script: scripts/check_alpha.sh
          args: ['--strict']
          timeout_secs: 30
      - name: manual_check
        binding: 'eyeball the UI'
        verdict: PASS
        delivered_check:
          kind: manual
          reason: 'no automated check available'
"""


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
async def test_happy_path_stamps_file_and_copies_mechanical_checks(tmp_path):
    """Valid sidecar: task_id is stamped to disk; only grep+script checks copy to metadata."""
    plans_dir = tmp_path / 'plans'
    plans_dir.mkdir()
    sidecar_path = plans_dir / 'foo-prd.capability-manifest.yaml'
    sidecar_path.write_text(_HAPPY_PATH_SIDECAR_YAML, encoding='utf-8')

    task_interceptor = AsyncMock()
    task_interceptor.update_task = AsyncMock(return_value={'success': True})
    ids = ['101']
    tasks_data = [
        {
            'id': '101',
            'metadata': {
                'prd_path': 'plans/foo-prd.md',
                'prd_task_label': 'alpha',
                'files': ['src/foo.py'],
            },
        },
    ]

    report = await stamp_capability_manifests(
        project_root=str(tmp_path),
        ids=ids,
        tasks_data=tasks_data,
        task_interceptor=task_interceptor,
        agent_id='claude-test',
    )

    assert report == {
        'path': 'plans/foo-prd.capability-manifest.yaml',
        'stamped': ['alpha'],
        'missing_labels': [],
        'errors': [],
    }

    reloaded = yaml.safe_load(sidecar_path.read_text(encoding='utf-8'))
    assert reloaded['tasks'][0]['label'] == 'alpha'
    assert reloaded['tasks'][0]['task_id'] == 101

    task_interceptor.update_task.assert_called_once()
    call = task_interceptor.update_task.call_args
    assert call.args[0] == '101'
    assert call.args[1] == str(tmp_path)
    assert call.kwargs['agent_id'] == 'claude-test'
    assert 'metadata_mode' not in call.kwargs
    assert 'append' not in call.kwargs

    payload = json.loads(call.kwargs['metadata'])
    checks = payload['delivered_checks']
    assert len(checks) == 2
    by_kind = {c['kind']: c for c in checks}
    assert set(by_kind) == {'grep', 'script'}
    assert by_kind['grep']['name'] == 'grep_check'
    assert by_kind['grep']['pattern'] == 'TODO(alpha)'
    assert by_kind['grep']['expect'] == 'absent'
    assert by_kind['grep']['paths'] == ['src/foo.py']
    assert by_kind['script']['name'] == 'script_check'
    assert by_kind['script']['script'] == 'scripts/check_alpha.sh'
    assert by_kind['script']['args'] == ['--strict']
    assert by_kind['script']['timeout_secs'] == 30


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
