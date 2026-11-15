"""A PRD's signal mark: its sidecar and H1 title read off disk, and the mark built from them.

The reader half reads real files under a tmp project root; the mark half
builds views with ``build_prd_view`` over a small tree and those documents
(PRD ``plans/dashboard-prd-lanes-prd.md`` §4.7).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from shared.capability_manifest import (
    CapabilityManifestDoc,
    parse_capability_manifest,
    sidecar_path,
)

from dashboard.data import prd_signal
from dashboard.data.census import build_census
from dashboard.data.datum import Datum, DatumState
from dashboard.data.prd_attribution import lineage_entry
from dashboard.data.prd_signal import clear_prd_documents, read_prd_documents
from dashboard.data.prd_view import (
    SIGNAL_CAUSE_TEXT,
    PrdDocument,
    PrdView,
    PrdViewInputs,
    SidecarUnread,
    SignalCause,
    TaskActivity,
    build_prd_view,
)

NOW = datetime(2026, 11, 5, 9, 0, tzinfo=UTC)
KEY = 'plans/x-prd.md'
OTHER = 'plans/y-prd.md'


def sidecar_yaml(prd: str, *, labels: tuple[str, ...] = ('gate',), leaves: str = '[gate]') -> str:
    blocks = ''.join(
        f'  - label: {label}\n    task_id: {number}\n    capabilities: []\n'
        for number, label in enumerate(labels, start=10)
    )
    leaf_line = '' if leaves == '' else f'signal_leaves: {leaves}\n'
    return f'prd: {prd}\nschema_version: 1\ntasks:\n{blocks}{leaf_line}'


def write(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def write_sidecar(root: Path, key: str, text: str) -> Path:
    return write(root, sidecar_path(key), text)


def bump_mtime(path: Path) -> None:
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


@pytest.fixture(autouse=True)
def _forget_documents():
    clear_prd_documents()
    yield
    clear_prd_documents()


@pytest.fixture
def uncached(monkeypatch):
    """Every request reads the disk again, so only the mtime cache can answer twice."""
    monkeypatch.setattr(prd_signal, 'PRD_DOCUMENTS_TTL_SECONDS', 0.0)


async def documents_of(root: Path, *keys: str) -> dict[str, PrdDocument]:
    datum = await read_prd_documents(str(root), keys, now=NOW)
    assert datum.value is not None, datum.reason
    return dict(datum.value)


async def test_a_sidecar_is_loaded_into_its_manifest_document(tmp_path):
    write_sidecar(tmp_path, KEY, sidecar_yaml(KEY))

    document = (await documents_of(tmp_path, KEY))[KEY]

    assert isinstance(document.sidecar, CapabilityManifestDoc)
    assert document.sidecar.prd == KEY
    assert document.sidecar.signal_leaves == ['gate']
    assert [task.task_id for task in document.sidecar.tasks] == [10]


async def test_the_datum_is_fresh_at_now_and_holds_exactly_the_keys_asked_for(tmp_path):
    write_sidecar(tmp_path, KEY, sidecar_yaml(KEY))
    write_sidecar(tmp_path, OTHER, sidecar_yaml(OTHER))

    datum = await read_prd_documents(str(tmp_path), (KEY,), now=NOW)

    assert datum.state is DatumState.FRESH
    assert datum.as_of == NOW
    assert datum.reason is None
    assert datum.value is not None
    assert set(datum.value) == {KEY}


async def test_a_missing_sidecar_reads_as_no_sidecar(tmp_path):
    write(tmp_path, KEY, '# X\n')

    document = (await documents_of(tmp_path, KEY))[KEY]

    assert isinstance(document.sidecar, SidecarUnread)
    assert document.sidecar.cause is SignalCause.NO_SIDECAR


@pytest.mark.parametrize(
    ('text', 'expected_detail'),
    [
        pytest.param(
            sidecar_yaml(KEY, leaves='[nowhere]'),
            "signal_leaves label 'nowhere' names no task block",
            id='unknown-signal-label',
        ),
        pytest.param(
            sidecar_yaml(KEY, leaves='[]'),
            'signal_leaves must be non-empty when present',
            id='empty-signal-leaves',
        ),
        pytest.param('prd: [unclosed\n  - : :\n', 'while parsing', id='malformed-yaml'),
        pytest.param('', 'CapabilityManifestDoc', id='empty-document'),
    ],
)
async def test_a_sidecar_that_fails_to_load_is_invalid_with_its_error_text(
    tmp_path, text, expected_detail,
):
    write_sidecar(tmp_path, KEY, text)

    document = (await documents_of(tmp_path, KEY))[KEY]

    assert isinstance(document.sidecar, SidecarUnread)
    assert document.sidecar.cause is SignalCause.SIDECAR_INVALID
    assert document.sidecar.detail is not None
    assert expected_detail in document.sidecar.detail


async def test_a_sidecar_path_that_is_not_a_readable_file_is_invalid_with_the_os_error(tmp_path):
    (tmp_path / sidecar_path(KEY)).mkdir(parents=True)

    document = (await documents_of(tmp_path, KEY))[KEY]

    assert isinstance(document.sidecar, SidecarUnread)
    assert document.sidecar.cause is SignalCause.SIDECAR_INVALID
    assert document.sidecar.detail is not None
    assert 'IsADirectoryError' in document.sidecar.detail


async def test_one_bad_sidecar_does_not_stop_the_others_from_reading(tmp_path):
    write_sidecar(tmp_path, KEY, 'prd: [unclosed\n')
    write_sidecar(tmp_path, OTHER, sidecar_yaml(OTHER))

    documents = await documents_of(tmp_path, KEY, OTHER)

    assert isinstance(documents[KEY].sidecar, SidecarUnread)
    assert isinstance(documents[OTHER].sidecar, CapabilityManifestDoc)


async def test_the_title_is_the_first_h1_line_with_its_marker_stripped(tmp_path):
    write(
        tmp_path, KEY,
        'Front matter\n#not-a-heading\n## A section\n#  Lanes by need  \nbody\n# Later heading\n',
    )

    document = (await documents_of(tmp_path, KEY))[KEY]

    assert document.title == 'Lanes by need'


@pytest.mark.parametrize(
    'prd_text',
    [pytest.param(None, id='prd-file-missing'), pytest.param('## Only sections\n', id='no-h1')],
)
async def test_the_title_is_none_without_an_h1(tmp_path, prd_text):
    if prd_text is not None:
        write(tmp_path, KEY, prd_text)
    write_sidecar(tmp_path, KEY, sidecar_yaml(KEY))

    document = (await documents_of(tmp_path, KEY))[KEY]

    assert document.title is None
    assert isinstance(document.sidecar, CapabilityManifestDoc)


async def test_a_root_that_is_not_a_directory_is_unknown_rather_than_every_sidecar_missing(
    tmp_path,
):
    missing_root = tmp_path / 'unmounted'

    datum = await read_prd_documents(str(missing_root), (KEY,), now=NOW)

    assert datum.state is DatumState.UNKNOWN
    assert datum.value is None
    assert datum.reason is not None
    assert str(missing_root) in datum.reason


async def test_an_unchanged_sidecar_is_served_the_same_parsed_document(tmp_path, uncached):
    write(tmp_path, KEY, '# X\n')
    write_sidecar(tmp_path, KEY, sidecar_yaml(KEY))

    first = (await documents_of(tmp_path, KEY))[KEY]
    second = (await documents_of(tmp_path, KEY))[KEY]

    assert second.sidecar is first.sidecar
    assert second.title == first.title == 'X'


async def test_a_rewritten_sidecar_is_read_again(tmp_path, uncached):
    path = write_sidecar(tmp_path, KEY, sidecar_yaml(KEY))
    first = (await documents_of(tmp_path, KEY))[KEY]

    path.write_text(sidecar_yaml(KEY, labels=('gate', 'tail'), leaves='[tail]'), encoding='utf-8')
    bump_mtime(path)
    second = (await documents_of(tmp_path, KEY))[KEY]

    assert isinstance(first.sidecar, CapabilityManifestDoc)
    assert isinstance(second.sidecar, CapabilityManifestDoc)
    assert first.sidecar.signal_leaves == ['gate']
    assert second.sidecar.signal_leaves == ['tail']


async def test_a_rewritten_prd_file_reads_its_new_title(tmp_path, uncached):
    path = write(tmp_path, KEY, '# Old title\n')
    assert (await documents_of(tmp_path, KEY))[KEY].title == 'Old title'

    path.write_text('# New title\n', encoding='utf-8')
    bump_mtime(path)

    assert (await documents_of(tmp_path, KEY))[KEY].title == 'New title'


async def test_a_changed_key_set_reads_the_new_key(tmp_path):
    write_sidecar(tmp_path, KEY, sidecar_yaml(KEY))
    write_sidecar(tmp_path, OTHER, sidecar_yaml(OTHER))
    write(tmp_path, OTHER, '# Other\n')

    first = await documents_of(tmp_path, KEY)
    second = await documents_of(tmp_path, KEY, OTHER)

    assert set(first) == {KEY}
    assert set(second) == {KEY, OTHER}
    assert second[OTHER].title == 'Other'
    assert isinstance(second[OTHER].sidecar, CapabilityManifestDoc)
    assert second[KEY].sidecar is first[KEY].sidecar


# ---------------------------------------------------------------------------
# The mark: built by the view from each row's document and the store
# ---------------------------------------------------------------------------

ROOT = '/srv/project'
AS_OF = NOW - timedelta(seconds=5)
BOUND = 60


def done_as(kind: str) -> dict[str, Any]:
    return {'prd_path': KEY, 'done_provenance': {'kind': kind, 'stamped_at': AS_OF.isoformat()}}


TREE: Mapping[int, tuple[str, Mapping[str, Any]]] = MappingProxyType({
    1: ('done', done_as('merged')),
    2: ('done', done_as('already_done')),
    3: ('pending', {'prd_path': KEY}),
    4: ('cancelled', {'prd_path': KEY}),
    5: ('pending', {'prd_path': OTHER}),
    6: ('in-progress', {'prd_path': KEY}),
})
ABSENT = 99


def fresh(value: Any) -> Datum[Any]:
    return Datum(value, AS_OF, DatumState.FRESH, None, BOUND)


def manifest(*blocks: Mapping[str, Any], leaves: list[str] | None) -> CapabilityManifestDoc:
    raw: dict[str, Any] = {
        'prd': KEY,
        'schema_version': 1,
        'tasks': [{**block, 'capabilities': []} for block in blocks],
    }
    if leaves is not None:
        raw['signal_leaves'] = leaves
    return parse_capability_manifest(raw)


def bound(label: str, task_id: int) -> dict[str, Any]:
    return {'label': label, 'task_id': task_id}


def signal_view(documents: Mapping[str, PrdDocument]) -> PrdView:
    status_map = {task_id: status for task_id, (status, _) in TREE.items()}
    rows = [
        {'id': task_id, 'status': status, 'dependencies': [], 'metadata': dict(metadata)}
        for task_id, (status, metadata) in TREE.items()
        if status not in {'done', 'cancelled'}
    ]
    lineage = {task_id: lineage_entry(metadata, ROOT) for task_id, (_, metadata) in TREE.items()}
    inputs = PrdViewInputs(
        project_root=ROOT,
        status_map=status_map,
        census=fresh(build_census(status_map)),
        rows=fresh(rows),
        lineage=fresh(MappingProxyType(lineage)),
        escalations=fresh(MappingProxyType({})),
        activity=fresh(TaskActivity(MappingProxyType({}), MappingProxyType({}), frozenset(), None)),
        documents=fresh(MappingProxyType(dict(documents))),
        liveness=fresh(MappingProxyType({})),
    )
    datum = build_prd_view(inputs, now=NOW)
    assert datum.value is not None, datum.reason
    return datum.value


def row_wire(sidecar: CapabilityManifestDoc | SidecarUnread, title: str | None = None) -> dict:
    view = signal_view({KEY: PrdDocument(title=title, sidecar=sidecar)})
    rows = view.to_wire()['prds']
    assert isinstance(rows, list)
    [row] = [row for row in rows if row['key'] == KEY]
    return row


def leaf(label: str, task_id: int | None, status: str | None, done_kind: str | None = None):
    return {'label': label, 'task_id': task_id, 'status': status, 'done_kind': done_kind}


def test_every_leaf_done_is_delivered_and_each_leaf_shows_its_done_kind():
    signal = row_wire(manifest(bound('a', 1), bound('b', 2), leaves=['a', 'b']))['signal']

    assert signal == {
        'mark': 'delivered',
        'cause': None,
        'reason': None,
        'leaves': [leaf('a', 1, 'done', 'merged'), leaf('b', 2, 'done', 'already_done')],
    }


def test_some_leaves_done_none_cancelled_is_partial():
    signal = row_wire(manifest(bound('a', 1), bound('c', 3), leaves=['a', 'c']))['signal']

    assert signal['mark'] == 'partial'
    assert signal['cause'] is None
    assert signal['leaves'] == [leaf('a', 1, 'done', 'merged'), leaf('c', 3, 'pending')]


def test_no_leaf_done_none_cancelled_is_not_yet():
    signal = row_wire(manifest(bound('c', 3), bound('f', 6), leaves=['c', 'f']))['signal']

    assert signal['mark'] == 'not_yet'
    assert signal['leaves'] == [leaf('c', 3, 'pending'), leaf('f', 6, 'in-progress')]


def test_only_the_named_leaves_count_toward_the_mark():
    sidecar = manifest(bound('a', 1), bound('c', 3), leaves=['a'])

    assert row_wire(sidecar)['signal']['mark'] == 'delivered'


_UNKNOWN_SIGNALS: dict[str, tuple[CapabilityManifestDoc | SidecarUnread, SignalCause]] = {
    'no-sidecar': (SidecarUnread(SignalCause.NO_SIDECAR, None), SignalCause.NO_SIDECAR),
    'names-no-leaf': (manifest(bound('a', 1), leaves=None), SignalCause.NAMES_NO_LEAF),
    'unbound-label': (
        manifest({'label': 'a', 'task_id': None}, leaves=['a']), SignalCause.UNBOUND_LABEL,
    ),
    'leaf-not-in-store': (
        manifest(bound('a', ABSENT), leaves=['a']), SignalCause.LEAF_NOT_IN_STORE,
    ),
    'stale-binding': (manifest(bound('a', 5), leaves=['a']), SignalCause.STALE_BINDING),
    'external-leaf': (
        manifest({'label': 'a', 'external_task_id': 'reify:12'}, leaves=['a']),
        SignalCause.EXTERNAL_LEAF,
    ),
    'cancelled-beside-done': (
        manifest(bound('a', 1), bound('d', 4), leaves=['a', 'd']), SignalCause.LEAF_CANCELLED,
    ),
}


@pytest.mark.parametrize(
    ('sidecar', 'cause'), list(_UNKNOWN_SIGNALS.values()), ids=list(_UNKNOWN_SIGNALS),
)
def test_each_unknown_signal_names_its_cause_and_its_text(sidecar, cause):
    signal = row_wire(sidecar)['signal']

    assert signal['mark'] == 'unknown'
    assert signal['cause'] == cause.value
    assert signal['reason'].startswith(SIGNAL_CAUSE_TEXT[cause])


def test_a_sidecar_that_fails_to_load_reads_unknown_with_the_validation_error():
    error = "ValidationError: signal_leaves label 'x' names no task block"
    signal = row_wire(SidecarUnread(SignalCause.SIDECAR_INVALID, error))['signal']

    assert signal['mark'] == 'unknown'
    assert signal['cause'] == 'sidecar_invalid'
    assert error in signal['reason']


def test_a_prd_naming_no_signal_leaf_reads_exactly_that_and_no_other_cause():
    signal = row_wire(manifest(bound('a', 1), leaves=None))['signal']

    assert signal['reason'] == 'PRD names no signal leaf'
    assert set(SIGNAL_CAUSE_TEXT) == set(SignalCause)


def test_an_unbound_leaf_is_served_with_no_task():
    signal = row_wire(manifest({'label': 'a', 'task_id': None}, leaves=['a']))['signal']

    assert signal['leaves'] == [leaf('a', None, None)]


def test_a_stale_binding_names_the_other_prd_the_bound_task_belongs_to():
    signal = row_wire(manifest(bound('a', 5), leaves=['a']))['signal']

    assert signal['cause'] == 'stale_binding'
    assert OTHER in signal['reason']
    assert signal['leaves'] == [leaf('a', 5, 'pending')]


def test_a_cancelled_leaf_beside_done_ones_is_never_delivered():
    signal = row_wire(manifest(bound('a', 1), bound('d', 4), leaves=['a', 'd']))['signal']

    assert signal['mark'] == 'unknown'
    assert signal['reason'].startswith('signal leaf cancelled')


def test_the_first_failing_leaf_in_signal_leaves_order_names_the_cause():
    sidecar = manifest(
        bound('d', 4), {'label': 'u', 'task_id': None}, bound('a', 1), leaves=['u', 'd', 'a'],
    )
    signal = row_wire(sidecar)['signal']

    assert signal['cause'] == 'unbound_label'
    assert [served['label'] for served in signal['leaves']] == ['u', 'd', 'a']


def test_a_rows_title_is_its_documents_title_else_its_key():
    sidecar = SidecarUnread(SignalCause.NO_SIDECAR, None)

    assert row_wire(sidecar, title='Lanes by need')['title'] == 'Lanes by need'
    assert row_wire(sidecar, title=None)['title'] == KEY


def test_signal_coverage_counts_rows_whose_loaded_sidecar_names_a_leaf_over_all_rows():
    named = PrdDocument(title=None, sidecar=manifest(bound('a', 1), leaves=['a']))

    view = signal_view({KEY: named})

    assert {row.key for row in view.prds} == {KEY, OTHER}
    assert view.to_wire()['signal_coverage'] == {'named': 1, 'of': 2}


async def test_a_new_key_set_that_fails_serves_the_roots_last_good_documents_stale(tmp_path):
    root = tmp_path / 'project'
    write_sidecar(root, KEY, sidecar_yaml(KEY))
    await documents_of(root, KEY)
    root.rename(tmp_path / 'unmounted')

    datum = await read_prd_documents(str(root), (KEY, OTHER), now=NOW)

    assert datum.state is DatumState.STALE
    assert datum.value is not None
    assert set(datum.value) == {KEY}
    assert str(root) in (datum.reason or '')
