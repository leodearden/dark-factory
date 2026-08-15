"""Tests for fused_memory.server.manifest_stamping.stamp_capability_manifests.

Unit-level coverage for the commit_planning manifest-stamping helper (PRD γ,
plans/capability-delivered-checks-prd.md): sidecar discovery, α-loader
validation, task_id stamping, and the mechanical (grep/script only)
delivered_checks copy into producer task metadata. Uses tmp_path sidecars and
a mocked task_interceptor — no DB/backend involved (that's covered by the
commit_planning integration tests in test_task_tools.py).
"""

import json
import logging
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

_MALFORMED_SIDECAR_YAML = """\
prd: plans/bad-prd.md
schema_version: 1
tasks:
  - label: beta
    task_id: null
    title: Broken task
    capabilities:
      - name: broken_check
        binding: 'grep for something'
        verdict: PASS
        delivered_check:
          kind: grep
          pattern: 'something'
"""

_MISSING_LABEL_AND_MANUAL_SIDECAR_YAML = """\
prd: plans/bar-prd.md
schema_version: 1
tasks:
  - label: alpha
    task_id: null
    title: Mechanical task
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
  - label: beta
    task_id: null
    title: Manual-only task
    capabilities:
      - name: manual_check
        binding: 'eyeball the UI'
        verdict: PASS
        delivered_check:
          kind: manual
          reason: 'no automated check available'
"""


def _mechanical_sidecar_yaml(prd_stem: str, labels: list[str]) -> str:
    """Render a valid capability-manifest sidecar YAML string for tests.

    One task entry per label in ``labels``, in order, each with
    ``task_id: null`` and a single MECHANICAL (``grep``) ``delivered_check``
    — never ``manual`` — so every generated fixture both parses against
    ``shared.capability_manifest.parse_capability_manifest`` and reaches the
    step-5 ``update_task`` call for each label. An invalid fixture here
    would silently divert a test that's supposed to exercise the
    containment guard / multi-sidecar tie-break / rejected-write branch
    into the already-covered malformed-sidecar branch instead — verified
    ad-hoc against the real α-loader when this helper was authored (see
    pre-1 in plan.json), not re-asserted as a standing test here.
    """
    task_blocks = []
    for label in labels:
        task_blocks.append(
            f'  - label: {label}\n'
            f'    task_id: null\n'
            f'    title: Mechanical task {label}\n'
            f'    capabilities:\n'
            f'      - name: grep_check_{label}\n'
            f"        binding: 'grep for the {label} marker'\n"
            f'        verdict: PASS\n'
            f'        delivered_check:\n'
            f'          kind: grep\n'
            f"          pattern: 'TODO({label})'\n"
            f'          expect: absent\n'
            f'          paths:\n'
            f'            - src/{label}.py\n'
        )
    return f'prd: plans/{prd_stem}-prd.md\nschema_version: 1\ntasks:\n' + ''.join(task_blocks)


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

    # The atomic temp+rename write leaves no stray .tmp file behind on the
    # success path either.
    leftovers = [p for p in plans_dir.iterdir() if p.name.endswith('.tmp')]
    assert leftovers == []

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


@pytest.mark.asyncio
async def test_malformed_sidecar_is_fail_soft(tmp_path):
    """A sidecar that fails α validation never raises; nothing is stamped or written."""
    plans_dir = tmp_path / 'plans'
    plans_dir.mkdir()
    sidecar_path = plans_dir / 'bad-prd.capability-manifest.yaml'
    sidecar_path.write_text(_MALFORMED_SIDECAR_YAML, encoding='utf-8')

    task_interceptor = AsyncMock()
    ids = ['202']
    tasks_data = [
        {
            'id': '202',
            'metadata': {
                'prd_path': 'plans/bad-prd.md',
                'prd_task_label': 'beta',
            },
        },
    ]

    report = await stamp_capability_manifests(
        project_root=str(tmp_path),
        ids=ids,
        tasks_data=tasks_data,
        task_interceptor=task_interceptor,
    )

    assert report is not None
    assert report['path'] == 'plans/bad-prd.capability-manifest.yaml'
    assert report['stamped'] == []
    assert len(report['errors']) == 1
    assert 'expect' in report['errors'][0]

    # The sidecar on disk is byte-identical — no partial task_id stamp.
    assert sidecar_path.read_text(encoding='utf-8') == _MALFORMED_SIDECAR_YAML
    task_interceptor.update_task.assert_not_called()


@pytest.mark.asyncio
async def test_missing_label_and_manual_only(tmp_path):
    """A batch label absent from the sidecar is reported; a manual-only label
    still gets its task_id stamped but writes no metadata."""
    plans_dir = tmp_path / 'plans'
    plans_dir.mkdir()
    sidecar_path = plans_dir / 'bar-prd.capability-manifest.yaml'
    sidecar_path.write_text(_MISSING_LABEL_AND_MANUAL_SIDECAR_YAML, encoding='utf-8')

    task_interceptor = AsyncMock()
    task_interceptor.update_task = AsyncMock(return_value={'success': True})
    ids = ['301', '302', '303']
    tasks_data = [
        {
            'id': '301',
            'metadata': {'prd_path': 'plans/bar-prd.md', 'prd_task_label': 'alpha'},
        },
        {
            'id': '302',
            'metadata': {'prd_path': 'plans/bar-prd.md', 'prd_task_label': 'beta'},
        },
        {
            'id': '303',
            'metadata': {'prd_path': 'plans/bar-prd.md', 'prd_task_label': 'zeta'},
        },
    ]

    report = await stamp_capability_manifests(
        project_root=str(tmp_path),
        ids=ids,
        tasks_data=tasks_data,
        task_interceptor=task_interceptor,
    )

    assert report == {
        'path': 'plans/bar-prd.capability-manifest.yaml',
        'stamped': ['alpha', 'beta'],
        'missing_labels': ['zeta'],
        'errors': [],
    }

    reloaded = yaml.safe_load(sidecar_path.read_text(encoding='utf-8'))
    by_label = {t['label']: t['task_id'] for t in reloaded['tasks']}
    assert by_label == {'alpha': 301, 'beta': 302}

    task_interceptor.update_task.assert_called_once()
    call = task_interceptor.update_task.call_args
    assert call.args[0] == '301'


@pytest.mark.asyncio
async def test_write_failure_mid_stamp_leaves_original_sidecar_intact(tmp_path, monkeypatch):
    """A write/replace *exception* between temp-write and os.replace (e.g. a
    disk error — the path where the ``finally`` unlink runs, unlike a hard
    process kill) must never corrupt the tracked sidecar: os.replace is
    atomic, so the original file is left byte-identical and parseable, and
    no stray .tmp file lingers on disk for this exception path."""
    plans_dir = tmp_path / 'plans'
    plans_dir.mkdir()
    sidecar_path = plans_dir / 'foo-prd.capability-manifest.yaml'
    sidecar_path.write_text(_HAPPY_PATH_SIDECAR_YAML, encoding='utf-8')

    def _boom(*args, **kwargs):
        raise OSError('simulated crash between temp-write and replace')

    monkeypatch.setattr(
        'fused_memory.server.manifest_stamping.os.replace',
        _boom,
    )

    task_interceptor = AsyncMock()
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
    )

    assert report is not None
    assert report['stamped'] == []
    assert len(report['errors']) == 1
    assert 'failed to stamp/write' in report['errors'][0]

    # Original sidecar is untouched and still parseable.
    assert sidecar_path.read_text(encoding='utf-8') == _HAPPY_PATH_SIDECAR_YAML
    yaml.safe_load(sidecar_path.read_text(encoding='utf-8'))

    # No stray temp file left behind in the sidecar's directory.
    leftovers = [p for p in plans_dir.iterdir() if p.name.endswith('.tmp')]
    assert leftovers == []

    task_interceptor.update_task.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('traversal_style', ['relative_dotdot', 'absolute'])
async def test_containment_refuses_traversal_prd_path(tmp_path, caplog, traversal_style):
    """A derived sidecar path that resolves outside project_root must be
    refused by the containment guard — both a relative '../' escape and an
    absolute prd_path reach the guard (via different pathlib routes) and
    must both be blocked. With no safe candidate surviving, the call
    returns None (the documented no-sidecar no-op contract), the escaping
    file is never read or written, and the ONLY observable is a server-side
    WARNING log — there is no report to attach the refusal to."""
    project_root = tmp_path / 'proj'
    (project_root / 'plans').mkdir(parents=True)
    outside_dir = tmp_path / 'outside'
    outside_dir.mkdir()
    evil_path = outside_dir / 'evil-prd.capability-manifest.yaml'
    evil_text = _mechanical_sidecar_yaml('evil', ['alpha'])
    evil_path.write_text(evil_text, encoding='utf-8')

    if traversal_style == 'relative_dotdot':
        prd_path = '../outside/evil-prd.md'
    else:
        prd_path = str(outside_dir / 'evil-prd.md')

    task_interceptor = AsyncMock()
    ids = ['1']
    tasks_data = [
        {
            'id': '1',
            'metadata': {'prd_path': prd_path, 'prd_task_label': 'alpha'},
        },
    ]

    with caplog.at_level(logging.WARNING, logger='fused_memory.server.manifest_stamping'):
        result = await stamp_capability_manifests(
            project_root=str(project_root),
            ids=ids,
            tasks_data=tasks_data,
            task_interceptor=task_interceptor,
        )

    assert result is None
    task_interceptor.update_task.assert_not_called()

    # Load-bearing security assertion: the escaping file was never read or
    # written back.
    assert evil_path.read_text(encoding='utf-8') == evil_text

    warning_records = [r for r in caplog.records if r.levelname == 'WARNING']
    assert len(warning_records) == 1, (
        f'Expected exactly 1 WARNING; got {len(warning_records)}: '
        f'{[r.getMessage() for r in warning_records]}'
    )
    message = warning_records[0].getMessage()
    assert 'resolve outside project_root' in message
    assert 'evil-prd.capability-manifest.yaml' in message


@pytest.mark.asyncio
async def test_containment_refusal_is_reported_when_a_safe_sidecar_also_exists(tmp_path):
    """The *other* half of the containment guard (lines 199-203): when at
    least one in-root sidecar survives, there IS a report to attach the
    refusal to, so it surfaces as a report['errors'] entry instead of only
    a log line. Neither this test nor test_containment_refuses_traversal_
    prd_path substitutes for the other — they exercise the guard's two
    structurally distinct exits."""
    project_root = tmp_path / 'proj'
    (project_root / 'plans').mkdir(parents=True)
    outside_dir = tmp_path / 'outside'
    outside_dir.mkdir()
    evil_path = outside_dir / 'evil-prd.capability-manifest.yaml'
    evil_text = _mechanical_sidecar_yaml('evil', ['alpha'])
    evil_path.write_text(evil_text, encoding='utf-8')

    good_path = project_root / 'plans' / 'good-prd.capability-manifest.yaml'
    good_path.write_text(_mechanical_sidecar_yaml('good', ['beta']), encoding='utf-8')

    task_interceptor = AsyncMock()
    task_interceptor.update_task = AsyncMock(return_value={'success': True})
    ids = ['1', '2']
    tasks_data = [
        {
            'id': '1',
            'metadata': {'prd_path': '../outside/evil-prd.md', 'prd_task_label': 'alpha'},
        },
        {
            'id': '2',
            'metadata': {'prd_path': 'plans/good-prd.md', 'prd_task_label': 'beta'},
        },
    ]

    report = await stamp_capability_manifests(
        project_root=str(project_root),
        ids=ids,
        tasks_data=tasks_data,
        task_interceptor=task_interceptor,
    )

    assert report['path'] == 'plans/good-prd.capability-manifest.yaml'
    assert report['stamped'] == ['beta']
    assert report['missing_labels'] == []

    assert len(report['errors']) == 1
    error_entry = report['errors'][0]
    assert 'resolved outside project_root, refused' in error_entry
    assert '../outside/evil-prd.capability-manifest.yaml' in error_entry

    # The escaping file was never read or written back...
    assert evil_path.read_text(encoding='utf-8') == evil_text
    # ...while the safe sidecar was stamped.
    reloaded = yaml.safe_load(good_path.read_text(encoding='utf-8'))
    assert reloaded['tasks'][0]['task_id'] == 2

    # The escaping task's label was excluded from label_to_task_id
    # entirely, not merely skipped at write time.
    task_interceptor.update_task.assert_called_once()
    assert task_interceptor.update_task.call_args.args[0] == '2'


@pytest.mark.asyncio
async def test_multiple_sidecars_processes_lexicographically_first(tmp_path):
    """An unexpected second distinct sidecar in the same batch (lines 186,
    193-198) is processed deterministically: the lexicographically-first
    rel path wins, not the batch-first one. The batch here deliberately
    lists the lexicographically-LAST sidecar first — if it listed them in
    lexicographic order, the test would pass on insertion order alone and
    would not pin existing_rel_paths.sort() at all."""
    plans_dir = tmp_path / 'plans'
    plans_dir.mkdir()
    alpha_path = plans_dir / 'alpha-prd.capability-manifest.yaml'
    alpha_text = _mechanical_sidecar_yaml('alpha', ['alabel'])
    alpha_path.write_text(alpha_text, encoding='utf-8')
    zeta_path = plans_dir / 'zeta-prd.capability-manifest.yaml'
    zeta_text = _mechanical_sidecar_yaml('zeta', ['zlabel'])
    zeta_path.write_text(zeta_text, encoding='utf-8')

    task_interceptor = AsyncMock()
    task_interceptor.update_task = AsyncMock(return_value={'success': True})
    # Batch order is REVERSED vs lexicographic order: zeta first, alpha second.
    ids = ['9', '8']
    tasks_data = [
        {
            'id': '9',
            'metadata': {'prd_path': 'plans/zeta-prd.md', 'prd_task_label': 'zlabel'},
        },
        {
            'id': '8',
            'metadata': {'prd_path': 'plans/alpha-prd.md', 'prd_task_label': 'alabel'},
        },
    ]

    report = await stamp_capability_manifests(
        project_root=str(tmp_path),
        ids=ids,
        tasks_data=tasks_data,
        task_interceptor=task_interceptor,
    )

    assert report['path'] == 'plans/alpha-prd.capability-manifest.yaml'
    assert report['stamped'] == ['alabel']
    assert report['missing_labels'] == []

    assert len(report['errors']) == 1
    error_entry = report['errors'][0]
    assert 'multiple capability-manifest sidecars matched this batch' in error_entry
    assert 'plans/zeta-prd.capability-manifest.yaml' in error_entry

    reloaded = yaml.safe_load(alpha_path.read_text(encoding='utf-8'))
    assert reloaded['tasks'][0]['task_id'] == 8

    # The ignored sidecar is never mutated.
    assert zeta_path.read_text(encoding='utf-8') == zeta_text

    task_interceptor.update_task.assert_called_once()
    assert task_interceptor.update_task.call_args.args[0] == '8'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'rejection_resp',
    [
        {'success': False, 'error': 'status_via_update_task', 'task_id': '2'},
        {'error': 'backlog exceeded', 'error_type': 'ReconciliationBacklogExceeded'},
        {},
    ],
    ids=['write_authority', 'backlog_no_success_key', 'bare_empty_dict'],
)
async def test_rejected_update_task_write_is_recorded_not_silently_dropped(
    tmp_path, caplog, rejection_resp
):
    """The interceptor_write_succeeded(resp) rejection branch (lines
    353-364), parametrized over the three rejection shapes that helper's
    own docstring documents (task_interceptor.py:5840-5861) — each defeats
    a different clause of its boolean expression: the write-authority
    shape defeats `resp.get('success', True)`, the no-success-key backlog
    shape defeats `not resp.get('error')`, and the bare {} defeats
    `bool(resp)`. A rejected write must be recorded loudly, not silently
    dropped — but it must NOT roll back the sidecar stamp already
    committed to disk (that asymmetry is itself worth pinning: the two
    writes — sidecar stamp, task metadata — can diverge)."""
    plans_dir = tmp_path / 'plans'
    plans_dir.mkdir()
    sidecar_path = plans_dir / 'good-prd.capability-manifest.yaml'
    sidecar_path.write_text(_mechanical_sidecar_yaml('good', ['beta']), encoding='utf-8')

    task_interceptor = AsyncMock()
    task_interceptor.update_task = AsyncMock(return_value=rejection_resp)
    ids = ['2']
    tasks_data = [
        {
            'id': '2',
            'metadata': {'prd_path': 'plans/good-prd.md', 'prd_task_label': 'beta'},
        },
    ]

    with caplog.at_level(logging.WARNING, logger='fused_memory.server.manifest_stamping'):
        report = await stamp_capability_manifests(
            project_root=str(tmp_path),
            ids=ids,
            tasks_data=tasks_data,
            task_interceptor=task_interceptor,
        )

    assert report['path'] == 'plans/good-prd.capability-manifest.yaml'
    # The rejection must NOT retroactively empty stamped.
    assert report['stamped'] == ['beta']
    assert report['missing_labels'] == []

    # A rejected metadata write does NOT roll back the already-committed
    # sidecar stamp.
    reloaded = yaml.safe_load(sidecar_path.read_text(encoding='utf-8'))
    assert reloaded['tasks'][0]['task_id'] == 2

    assert len(report['errors']) == 1
    error_entry = report['errors'][0]
    assert 'update_task rejected delivered_checks write' in error_entry
    assert "'beta'" in error_entry
    assert 'task 2' in error_entry

    warning_records = [r for r in caplog.records if r.levelname == 'WARNING']
    assert any('rejected delivered_checks write' in r.getMessage() for r in warning_records)
