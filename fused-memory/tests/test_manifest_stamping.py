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
import subprocess
from unittest.mock import AsyncMock

import pytest
import yaml

from fused_memory.server.manifest_stamping import stamp_capability_manifests

# Hand-written rather than built from _mechanical_sidecar_yaml below: this
# fixture mixes grep + script + manual capabilities on a single label, and
# test_happy_path_stamps_file_and_copies_mechanical_checks asserts exact
# values (script args/timeout, grep pattern/paths) the helper's
# single-grep-capability shape doesn't produce.
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
      - name: path_check
        binding: 'the strand test module exists'
        verdict: PASS
        delivered_check:
          kind: path
          expect: present
          paths:
            - orchestrator/tests/test_workflow_merge_gating_strand.py
      - name: manual_check
        binding: 'eyeball the UI'
        verdict: PASS
        delivered_check:
          kind: manual
          reason: 'no automated check available'
"""

# Hand-written: carries a task-level `note:` (durable provenance,
# shared.capability_manifest.ManifestTask.note) alongside a `verdict: OPEN`
# capability row that has no delivered_check (OPEN records an undecided
# binding, not a measured absence — see ManifestCapability's docstring) plus
# one ordinary mechanical grep capability. Exercises the claim
# ManifestTask's own docstring makes about this module's step-4 write-back:
# a DECLARED field (note) survives yaml.safe_dump(raw, ...) where a YAML
# comment would not. See test_round_trip_preserves_task_level_note_and_open_verdict.
_NOTE_AND_OPEN_VERDICT_SIDECAR_YAML = """\
prd: plans/note-prd.md
schema_version: 1
tasks:
  - label: gamma
    task_id: null
    title: Task carrying a note and an OPEN verdict
    note: "SPLIT 2026-08-19. The original gamma row was one task across four servers; this leaf carries fused-memory."
    capabilities:
      - name: open_check
        binding: 'decision deferred to this leaf'
        verdict: OPEN
      - name: grep_check
        binding: 'grep for the marker'
        verdict: PASS
        delivered_check:
          kind: grep
          pattern: 'TODO(gamma)'
          expect: absent
          paths:
            - src/gamma.py
"""

# Hand-written and deliberately INVALID (missing the grep check's required
# `expect` field) — not a duplicate of the helper's always-valid output, so
# not a helper candidate.
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


def _mechanical_sidecar_yaml(
    prd_stem: str, labels: list[str], manual_labels: list[str] | None = None
) -> str:
    """Render a valid capability-manifest sidecar YAML string for tests.

    One task entry per label in ``labels``, in order, each with
    ``task_id: null`` and a single MECHANICAL (``grep``) ``delivered_check``,
    plus one task entry per label in ``manual_labels`` (if given) with a
    single ``manual`` ``delivered_check`` instead. Every generated fixture
    parses against ``shared.capability_manifest.parse_capability_manifest``;
    an invalid one would silently divert a test that's supposed to exercise
    the containment guard / multi-sidecar tie-break / rejected-write branch
    into the already-covered malformed-sidecar branch instead.
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
    for label in manual_labels or []:
        task_blocks.append(
            f'  - label: {label}\n'
            f'    task_id: null\n'
            f'    title: Manual-only task {label}\n'
            f'    capabilities:\n'
            f'      - name: manual_check_{label}\n'
            f"        binding: 'eyeball the UI'\n"
            f'        verdict: PASS\n'
            f'        delivered_check:\n'
            f'          kind: manual\n'
            f"          reason: 'no automated check available'\n"
        )
    return f'prd: plans/{prd_stem}-prd.md\nschema_version: 1\ntasks:\n' + ''.join(task_blocks)


# Built from the helper above — it's a near-duplicate of the helper's
# default one-mechanical-label shape plus one manual-only label.
# _HAPPY_PATH_SIDECAR_YAML and _MALFORMED_SIDECAR_YAML above stay
# hand-written since they aren't (see the comments there).
_MISSING_LABEL_AND_MANUAL_SIDECAR_YAML = _mechanical_sidecar_yaml(
    'bar', ['alpha'], manual_labels=['beta']
)


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
    """Valid sidecar: task_id is stamped to disk; every MECHANICAL check copies
    to metadata (grep + script + path) and only manual is dropped."""
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
    # Deliberately updated from 2/{grep,script} when kind='path' was added
    # (task 4743): leaving them would silently assert that path checks are
    # NOT stamped, and an unstamped check is one the delta gate never sees.
    assert len(checks) == 3
    by_kind = {c['kind']: c for c in checks}
    assert set(by_kind) == {'grep', 'script', 'path'}
    assert by_kind['grep']['name'] == 'grep_check'
    assert by_kind['grep']['pattern'] == 'TODO(alpha)'
    assert by_kind['grep']['expect'] == 'absent'
    assert by_kind['grep']['paths'] == ['src/foo.py']
    assert by_kind['script']['name'] == 'script_check'
    assert by_kind['script']['script'] == 'scripts/check_alpha.sh'
    assert by_kind['script']['args'] == ['--strict']
    assert by_kind['script']['timeout_secs'] == 30
    assert by_kind['path']['name'] == 'path_check'
    assert by_kind['path']['expect'] == 'present'
    assert by_kind['path']['paths'] == [
        'orchestrator/tests/test_workflow_merge_gating_strand.py'
    ]


@pytest.mark.asyncio
async def test_round_trip_preserves_task_level_note_and_open_verdict(tmp_path):
    """Task 4489 (follow-up from 4471, ticket tkt_0RSNVJT1ZNWKS5Y7BM5F7A2QAD):
    the round-trip counterpart to shared's LOAD-only
    TestLoader::test_load_sidecar_with_task_level_note. ManifestTask's
    docstring claims a DECLARED `note` field survives this module's step-4
    ``yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)`` write-back
    where a YAML comment would not — that claim can only be asserted here,
    on the stamping side, since shared/ has no write path of its own. Stamp
    a sidecar carrying a task-level `note:` and a `verdict: OPEN` capability
    row through the real helper and confirm both survive byte-for-byte
    (as decoded values) in the rewritten file, unstamped fields included.
    """
    plans_dir = tmp_path / 'plans'
    plans_dir.mkdir()
    sidecar_path = plans_dir / 'note-prd.capability-manifest.yaml'
    sidecar_path.write_text(_NOTE_AND_OPEN_VERDICT_SIDECAR_YAML, encoding='utf-8')

    task_interceptor = AsyncMock()
    task_interceptor.update_task = AsyncMock(return_value={'success': True})
    ids = ['401']
    tasks_data = [
        {
            'id': '401',
            'metadata': {
                'prd_path': 'plans/note-prd.md',
                'prd_task_label': 'gamma',
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
        'path': 'plans/note-prd.capability-manifest.yaml',
        'stamped': ['gamma'],
        'missing_labels': [],
        'errors': [],
    }

    reloaded = yaml.safe_load(sidecar_path.read_text(encoding='utf-8'))

    # Whole-document equality against the fixture (task_id patched from
    # null to the stamped value) is both shorter and strictly stronger than
    # spot-checking individual leaves: it also covers doc-level keys (prd,
    # schema_version), the task's title, and the grep row's full
    # delivered_check body, none of which a narrower per-field check would
    # re-read after the rewrite.
    expected = yaml.safe_load(_NOTE_AND_OPEN_VERDICT_SIDECAR_YAML)
    expected['tasks'][0]['task_id'] = 401
    assert reloaded == expected

    # Named check for the specific claim this test exists to verify: the
    # task-level `note:` survives the safe_dump write-back verbatim.
    # Subsumed by the whole-document equality above; kept as documentation
    # of intent, derived from the parsed fixture rather than retyped so it
    # can't silently desync from it.
    assert reloaded['tasks'][0]['note'] == expected['tasks'][0]['note']

    # An OPEN row has no delivered_check, so only the sibling grep check
    # copies into metadata.delivered_checks — mirrors
    # test_happy_path_stamps_file_and_copies_mechanical_checks's manual-check
    # exclusion, one row over.
    task_interceptor.update_task.assert_called_once()
    call = task_interceptor.update_task.call_args
    assert call.args[0] == '401'
    payload = json.loads(call.kwargs['metadata'])
    checks = payload['delivered_checks']
    assert len(checks) == 1
    assert checks[0]['name'] == 'grep_check'
    assert checks[0]['kind'] == 'grep'


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

    warning_records = [
        r
        for r in caplog.records
        if r.levelname == 'WARNING' and r.name == 'fused_memory.server.manifest_stamping'
    ]
    assert len(warning_records) == 1, (
        f'Expected exactly 1 WARNING; got {len(warning_records)}: '
        f'{[r.getMessage() for r in warning_records]}'
    )
    message = warning_records[0].getMessage()
    assert 'resolve outside project_root' in message
    assert 'evil-prd.capability-manifest.yaml' in message


@pytest.mark.asyncio
async def test_containment_refusal_is_reported_when_a_safe_sidecar_also_exists(tmp_path):
    """The *other* half of the containment guard (the errors-entry exit in
    `_stamp_capability_manifests_impl` step 2): when at least one in-root
    sidecar survives, there IS a report to attach the refusal to, so it
    surfaces as a report['errors'] entry instead of only a log line.
    Neither this test nor test_containment_refuses_traversal_prd_path
    substitutes for the other — they exercise the guard's two structurally
    distinct exits."""
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

    assert report is not None
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
    """An unexpected second distinct sidecar in the same batch (the
    multi-sidecar tie-break in `_stamp_capability_manifests_impl` step 2)
    is processed deterministically: the lexicographically-first rel path
    wins, not the batch-first one. The batch here deliberately lists the
    lexicographically-LAST sidecar first — if it listed them in
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

    assert report is not None
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
    """The interceptor_write_succeeded(resp) rejection branch (the last
    check in `_stamp_capability_manifests_impl` step 5's per-label loop),
    parametrized over the three rejection shapes documented in
    `interceptor_write_succeeded`'s own docstring in
    `fused_memory.middleware.task_interceptor` — each defeats a different
    clause of its boolean expression: the write-authority shape defeats
    `resp.get('success', True)`, the no-success-key backlog shape defeats
    `not resp.get('error')`, and the bare {} defeats `bool(resp)`. A
    rejected write must be recorded loudly, not silently dropped — but it
    must NOT roll back the sidecar stamp already committed to disk (that
    asymmetry is itself worth pinning: the two writes — sidecar stamp,
    task metadata — can diverge)."""
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

    assert report is not None
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

    warning_records = [
        r
        for r in caplog.records
        if r.levelname == 'WARNING' and r.name == 'fused_memory.server.manifest_stamping'
    ]
    assert any('rejected delivered_checks write' in r.getMessage() for r in warning_records)


@pytest.mark.asyncio
async def test_rejected_write_does_not_block_sibling_labels(tmp_path):
    """A rejection is per-label and must not abort the remaining
    `for task in doc.tasks` iterations. The rejection branch is the last
    statement in the loop body, so continuation holds by construction today
    — which is exactly why an accidental `return report` / `break` added
    there would go unnoticed without this test. Complements step-4 (which
    pins that a rejection is *recorded*) by pinning that it is
    *contained*."""
    plans_dir = tmp_path / 'plans'
    plans_dir.mkdir()
    sidecar_path = plans_dir / 'multi-prd.capability-manifest.yaml'
    sidecar_path.write_text(
        _mechanical_sidecar_yaml('multi', ['first', 'second']), encoding='utf-8'
    )

    task_interceptor = AsyncMock()
    task_interceptor.update_task = AsyncMock(
        side_effect=[
            {'success': False, 'error': 'status_via_update_task', 'task_id': '11'},
            {'success': True},
        ]
    )
    ids = ['11', '12']
    tasks_data = [
        {
            'id': '11',
            'metadata': {'prd_path': 'plans/multi-prd.md', 'prd_task_label': 'first'},
        },
        {
            'id': '12',
            'metadata': {'prd_path': 'plans/multi-prd.md', 'prd_task_label': 'second'},
        },
    ]

    report = await stamp_capability_manifests(
        project_root=str(tmp_path),
        ids=ids,
        tasks_data=tasks_data,
        task_interceptor=task_interceptor,
    )

    # The second label was still attempted after the first was rejected.
    assert task_interceptor.update_task.call_count == 2
    assert [
        c.args[0] for c in task_interceptor.update_task.call_args_list
    ] == ['11', '12']

    assert report is not None
    assert report['stamped'] == ['first', 'second']
    assert report['missing_labels'] == []

    reloaded = yaml.safe_load(sidecar_path.read_text(encoding='utf-8'))
    by_label = {t['label']: t['task_id'] for t in reloaded['tasks']}
    assert by_label == {'first': 11, 'second': 12}

    # The successful sibling must not be tarred by its neighbour's failure.
    assert len(report['errors']) == 1
    assert 'first' in report['errors'][0]
    assert 'second' not in report['errors'][0]


# ---------------------------------------------------------------------------
# Delivered-check POLARITY: the stamper's REFUSE-TO-COPY path
# (task 3500, step-15 RED / step-16 GREEN)
#
# Same lint as commit_planning's gate, OPPOSITE contract. commit_planning is a
# synchronous gate whose caller is a live agent that can repair a descriptor
# and re-commit, so it REJECTS the batch. This helper is contractually
# never-raising and must not block the status flip that called it, so the
# worst it may do is REFUSE TO COPY the offending check — leaving a dependent
# ungated (the pre-gate status quo) rather than blocked forever (the wedge
# this task exists to prevent).
# ---------------------------------------------------------------------------

#: Committed into the polarity fixture tree, so an `expect=present` grep for
#: it is already green at the authoring tree -> vacuous_present.
_POL_LANDED = 'LandedSymbol'
#: Never committed, so an `expect=present` grep for it is the healthy
#: forward-looking shape the gate must leave alone.
_POL_FUTURE = 'FutureSymbol'
#: Committed into BOTH fixture files, so an `expect=absent` grep for it is
#: healthy under the 2x2 but over-broad for a task owning only the seed.
_POL_WIDE = 'WidelyUsedSymbol'
_POL_SEED_REL = 'src/seeded.py'
_POL_OTHER_REL = 'src/other.py'

# NOTE every pattern above is a bare identifier. `git grep -E` is POSIX
# EXTENDED regex, so the neighbouring fixtures' 'TODO(alpha)' pattern actually
# searches for the literal `TODOalpha` — a capture group, not parentheses.
# A polarity fixture whose pattern did that would silently test the wrong cell.


def _polarity_git_repo(root):
    """git init *root* and commit the two-file polarity seed tree."""
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ['git', 'init', '-b', 'main', str(root)],
        check=True, capture_output=True, text=True,
    )
    for args in (
        ('config', 'user.email', 'polarity-test@example.com'),
        ('config', 'user.name', 'Polarity Test'),
    ):
        subprocess.run(
            ['git', '-C', str(root), *args], check=True, capture_output=True, text=True,
        )
    seed = root / _POL_SEED_REL
    seed.parent.mkdir(parents=True, exist_ok=True)
    seed.write_text(
        f'class {_POL_LANDED}:\n    pass\n\n{_POL_WIDE} = 1\n', encoding='utf-8',
    )
    (root / _POL_OTHER_REL).write_text(f'use({_POL_WIDE})\n', encoding='utf-8')
    subprocess.run(
        ['git', '-C', str(root), 'add', _POL_SEED_REL, _POL_OTHER_REL],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ['git', '-C', str(root), 'commit', '-m', 'seed the authoring tree'],
        check=True, capture_output=True, text=True,
    )
    return root


def _polarity_sidecar_yaml(prd_stem, label, checks):
    """Render a sidecar whose single *label* declares *checks*.

    *checks* is a list of ``(name, pattern, expect, paths)`` tuples; ``paths``
    may be ``None`` to omit the key (a whole-tree grep).
    """
    blocks = []
    for name, pattern, expect, paths in checks:
        block = (
            f'      - name: {name}\n'
            f"        binding: 'polarity fixture {name}'\n"
            f'        verdict: PASS\n'
            f'        delivered_check:\n'
            f'          kind: grep\n'
            f"          pattern: '{pattern}'\n"
            f'          expect: {expect}\n'
        )
        if paths:
            block += '          paths:\n' + ''.join(f'            - {p}\n' for p in paths)
        blocks.append(block)
    return (
        f'prd: plans/{prd_stem}-prd.md\n'
        f'schema_version: 1\n'
        f'tasks:\n'
        f'  - label: {label}\n'
        f'    task_id: null\n'
        f'    title: Polarity fixture {label}\n'
        f'    capabilities:\n' + ''.join(blocks)
    )


async def _stamp_polarity(root, sidecar_yaml, *, prd_stem, label, files, task_id='401'):
    """Write the sidecar, run the stamper against *root*, return (report, interceptor)."""
    plans_dir = root / 'plans'
    plans_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = plans_dir / f'{prd_stem}-prd.capability-manifest.yaml'
    sidecar_path.write_text(sidecar_yaml, encoding='utf-8')

    task_interceptor = AsyncMock()
    task_interceptor.update_task = AsyncMock(return_value={'success': True})
    report = await stamp_capability_manifests(
        project_root=str(root),
        ids=[task_id],
        tasks_data=[
            {
                'id': task_id,
                'metadata': {
                    'prd_path': f'plans/{prd_stem}-prd.md',
                    'prd_task_label': label,
                    'files': files,
                },
            },
        ],
        task_interceptor=task_interceptor,
    )
    # The sidecar exists and carries the label, so the stamper always returns a
    # report here. Pinning it once keeps every caller's subscript legible, and
    # turns a regression to the None no-op path into a named failure rather
    # than a TypeError at the first subscript.
    assert report is not None, 'stamper returned the no-op None for a live sidecar'
    return report, task_interceptor, sidecar_path


def _copied_checks(task_interceptor):
    call = task_interceptor.update_task.call_args
    return json.loads(call.kwargs['metadata'])['delivered_checks']


@pytest.mark.asyncio
async def test_stamper_refuses_vacuous_check_but_copies_the_clean_sibling(tmp_path):
    """A label mixing one vacuous and one clean check copies ONLY the clean one.

    Refusal is per CHECK, not per label: dropping the whole label would strip
    a sound gate along with the unsound one.
    """
    root = _polarity_git_repo(tmp_path / 'proj')
    report, interceptor, sidecar_path = await _stamp_polarity(
        root,
        _polarity_sidecar_yaml('mix', 'alpha', [
            ('vacuous_grep', _POL_LANDED, 'present', [_POL_SEED_REL]),
            ('clean_grep', _POL_FUTURE, 'present', [_POL_SEED_REL]),
        ]),
        prd_stem='mix', label='alpha', files=[_POL_SEED_REL],
    )

    interceptor.update_task.assert_called_once()
    copied = _copied_checks(interceptor)
    assert [c['name'] for c in copied] == ['clean_grep']

    # (b) The refusal is named loudly, with enough to act on.
    joined = ' '.join(report['errors'])
    assert 'plans/mix-prd.capability-manifest.yaml' in joined
    assert 'alpha' in joined
    assert 'vacuous_grep' in joined
    assert 'vacuous_present' in joined

    # (c) A refused check NEVER rolls back the task_id stamp step 4 already
    # committed to disk.
    assert report['stamped'] == ['alpha']
    reloaded = yaml.safe_load(sidecar_path.read_text(encoding='utf-8'))
    assert reloaded['tasks'][0]['task_id'] == 401


@pytest.mark.asyncio
async def test_stamper_all_refused_label_calls_no_update_task(tmp_path):
    """When a label's ONLY check is refused, update_task is not called at all."""
    root = _polarity_git_repo(tmp_path / 'proj')
    report, interceptor, _ = await _stamp_polarity(
        root,
        _polarity_sidecar_yaml('solo', 'alpha', [
            ('only_grep', _POL_LANDED, 'present', [_POL_SEED_REL]),
        ]),
        prd_stem='solo', label='alpha', files=[_POL_SEED_REL],
    )

    interceptor.update_task.assert_not_called()
    assert report['stamped'] == ['alpha']
    joined = ' '.join(report['errors'])
    assert 'only_grep' in joined
    assert 'vacuous_present' in joined


@pytest.mark.asyncio
async def test_stamper_warn_tier_check_is_copied_and_reported(tmp_path):
    """A warn-tier check IS copied and surfaces under polarity_warnings."""
    root = _polarity_git_repo(tmp_path / 'proj')
    report, interceptor, _ = await _stamp_polarity(
        root,
        _polarity_sidecar_yaml('warned', 'alpha', [
            # Whole-tree expect=absent: healthy under the 2x2 (it still
            # matches, so it can go green) but also matches src/other.py,
            # which this task does not declare.
            ('wide_grep', _POL_WIDE, 'absent', None),
        ]),
        prd_stem='warned', label='alpha', files=[_POL_SEED_REL],
    )

    interceptor.update_task.assert_called_once()
    assert [c['name'] for c in _copied_checks(interceptor)] == ['wide_grep']
    assert report['errors'] == []
    warnings = report.get('polarity_warnings')
    assert warnings, f'an over-broad absent check must be reported, got {report!r}'
    assert [(w['name'], w['code'], w['severity']) for w in warnings] == [
        ('wide_grep', 'absent_overbroad', 'warn'),
    ]


@pytest.mark.asyncio
async def test_stamper_clean_sidecar_report_has_no_polarity_warnings_key(tmp_path):
    """A fully clean sidecar's report keeps its exact 4-key shape.

    Protects the five exact-dict-equality assertions on this report
    (test_manifest_stamping.py:182,:319; test_delivered_checks_e2e.py:435,:697;
    test_task_tools.py:2514).
    """
    root = _polarity_git_repo(tmp_path / 'proj')
    report, interceptor, _ = await _stamp_polarity(
        root,
        _polarity_sidecar_yaml('clean', 'alpha', [
            ('clean_grep', _POL_FUTURE, 'present', [_POL_SEED_REL]),
        ]),
        prd_stem='clean', label='alpha', files=[_POL_SEED_REL],
    )

    assert report == {
        'path': 'plans/clean-prd.capability-manifest.yaml',
        'stamped': ['alpha'],
        'missing_labels': [],
        'errors': [],
    }
    assert [c['name'] for c in _copied_checks(interceptor)] == ['clean_grep']


@pytest.mark.asyncio
async def test_stamper_non_git_root_copies_everything_and_logs(tmp_path, caplog):
    """Infra fail-open: an unevaluable check is still copied, and reported.

    The lint cannot reach a verdict on a non-git root. Refusing to copy would
    convert an availability failure into the very wedge this gate prevents, so
    every check is copied — but the failure is recorded at WARNING rather than
    passed over in silence.

    The report itself keeps its 4-key shape: an errored disposition on a
    non-git root is the DEFAULT state of every pre-existing fixture in this
    file, so routing it into `errors`/`polarity_warnings` would break the five
    exact-dict assertions decision 7 exists to protect (filed as esc-3500-2).
    """
    root = tmp_path / 'proj'  # deliberately NOT a git repo
    with caplog.at_level(logging.WARNING):
        report, interceptor, _ = await _stamp_polarity(
            root,
            _polarity_sidecar_yaml('nogit', 'alpha', [
                ('unevaluable_grep', _POL_LANDED, 'present', [_POL_SEED_REL]),
            ]),
            prd_stem='nogit', label='alpha', files=[_POL_SEED_REL],
        )

    interceptor.update_task.assert_called_once()
    assert [c['name'] for c in _copied_checks(interceptor)] == ['unevaluable_grep']
    assert report == {
        'path': 'plans/nogit-prd.capability-manifest.yaml',
        'stamped': ['alpha'],
        'missing_labels': [],
        'errors': [],
    }
    polarity_warnings = [
        r for r in caplog.records
        if r.levelname == 'WARNING' and 'unevaluable_grep' in r.getMessage()
    ]
    assert polarity_warnings, (
        'an unevaluable check must be loudly recorded, not silently passed; '
        f'got {[r.getMessage() for r in caplog.records]}'
    )
