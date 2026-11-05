"""The PRD documents a view reads: each open-work PRD's sidecar and H1 title, off disk."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from shared.capability_manifest import CapabilityManifestDoc, sidecar_path

from dashboard.data import prd_signal
from dashboard.data.datum import DatumState
from dashboard.data.prd_signal import clear_prd_documents, read_prd_documents
from dashboard.data.prd_view import PrdDocument, SidecarUnread, SignalCause

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
