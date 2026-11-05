"""The ``metadata.declared_scope`` record: v1 bands, record invariants and the write-authority rule."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.declared_scope import DeclaredScope, derive_declared_scope, preserves_stamp

_TINY = 'fix it'


def _files(count: int) -> list[str]:
    return [f'pkg/module_{index}.py' for index in range(count)]


@pytest.mark.parametrize(
    ('file_count', 'description', 'details', 'band'),
    [
        (0, _TINY, '', 'undeclared'),
        (0, 'd' * 4000, 't' * 2000, 'undeclared'),
        (1, 'd' * 5000, 't' * 999, 'narrow'),
        (2, 'd' * 3000, 't' * 3000, 'moderate'),
        (1, 'd' * 6000, '', 'moderate'),
        (3, _TINY, '', 'moderate'),
        (5, _TINY, '', 'moderate'),
        (6, _TINY, '', 'broad'),
        (6, 'd' * 3000, 't' * 3000, 'broad'),
    ],
)
def test_v1_band_table(file_count: int, description: str, details: str, band: str) -> None:
    record = derive_declared_scope(_files(file_count), description, details, basis='submit')

    assert record.band == band


@pytest.mark.parametrize('basis', ['submit', 'combine'])
def test_the_record_carries_its_counts_basis_and_rule(basis: str) -> None:
    record = derive_declared_scope(_files(2), 'describe', 'the details', basis=basis)

    assert (record.files, record.text_chars, record.basis, record.rule) == (
        2, len('describe') + len('the details'), basis, 'v1',
    )


def test_a_bare_str_is_not_a_file_list() -> None:
    with pytest.raises(TypeError):
        derive_declared_scope('pkg/module.py', _TINY, '', basis='submit')


def _fields(**overrides: object) -> dict[str, object]:
    return {'band': 'narrow', 'files': 1, 'text_chars': 10, 'basis': 'submit', 'rule': 'v1', **overrides}


@pytest.mark.parametrize(
    'fields',
    [
        pytest.param(_fields(x_note='extra'), id='unknown-field'),
        pytest.param(_fields(files=-1), id='negative-files'),
        pytest.param(_fields(text_chars=-1), id='negative-text-chars'),
        pytest.param(_fields(band='tiny'), id='band-outside-literals'),
        pytest.param(_fields(basis='plan'), id='basis-outside-literals'),
        pytest.param(_fields(rule='v2'), id='rule-outside-literals'),
    ],
)
def test_a_malformed_record_is_refused(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        DeclaredScope(**fields)


@pytest.mark.parametrize('missing', ['band', 'files', 'text_chars', 'basis', 'rule'])
def test_every_field_is_required(missing: str) -> None:
    fields = _fields()
    del fields[missing]

    with pytest.raises(ValidationError):
        DeclaredScope(**fields)


@pytest.mark.parametrize('field', ['band', 'files', 'text_chars', 'basis', 'rule'])
def test_is_frozen(field: str) -> None:
    record = DeclaredScope(**_fields())

    with pytest.raises(ValidationError):
        setattr(record, field, 'moderate')


def test_a_band_that_disagrees_with_its_counts_is_refused_by_name() -> None:
    with pytest.raises(ValidationError) as exc_info:
        DeclaredScope(**_fields(band='narrow', files=9, text_chars=12))

    message = str(exc_info.value)
    assert "'narrow'" in message
    assert 'files=9' in message
    assert 'text_chars=12' in message


_STAMP = derive_declared_scope(_files(1), _TINY, '', basis='submit').model_dump()
_ALTERED = {**_STAMP, 'band': 'broad', 'files': 6}
_OTHER = {'x_note': 'kept'}


def _blob(stamp: dict[str, object] | None) -> dict[str, object]:
    return _OTHER if stamp is None else {**_OTHER, 'declared_scope': stamp}


@pytest.mark.parametrize(
    ('stored', 'payload', 'replace', 'preserved'),
    [
        pytest.param(None, None, False, True, id='merge-unstamped-untouched'),
        pytest.param(None, _STAMP, False, False, id='merge-adds-a-stamp'),
        pytest.param(_STAMP, None, False, True, id='merge-leaves-the-stamp-alone'),
        pytest.param(_STAMP, _STAMP, False, True, id='merge-echoes-the-stamp'),
        pytest.param(_STAMP, _ALTERED, False, False, id='merge-alters-the-stamp'),
        pytest.param(None, None, True, True, id='replace-unstamped-untouched'),
        pytest.param(None, _STAMP, True, False, id='replace-adds-a-stamp'),
        pytest.param(_STAMP, None, True, False, id='replace-drops-the-stamp'),
        pytest.param(_STAMP, _STAMP, True, True, id='replace-carries-the-stamp'),
        pytest.param(_STAMP, _ALTERED, True, False, id='replace-alters-the-stamp'),
    ],
)
def test_preserves_stamp_truth_table(
    stored: dict[str, object] | None,
    payload: dict[str, object] | None,
    replace: bool,
    preserved: bool,
) -> None:
    assert preserves_stamp(_blob(stored), _blob(payload), replace=replace) is preserved
