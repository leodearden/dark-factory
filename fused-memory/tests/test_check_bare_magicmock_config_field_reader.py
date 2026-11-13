"""Tests for check_bare_magicmock_config.py::dataclass_field_names.

The stdlib-only checker reads a dataclass's field names from its defining
source with ``ast`` instead of importing the class (task 7268).  These tests
drive that reader through ``tmp_path`` source files only, and hold it to
``dataclasses.fields`` on an exec'd copy of the same source.
"""

import dataclasses
import textwrap
from pathlib import Path

import pytest
from _fm_helpers import load_script_module

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'check_bare_magicmock_config.py'

_checker = load_script_module(SCRIPT_PATH, mod_name='check_bare_magicmock_config')


def _write(path: Path, source: str) -> Path:
    path.write_text(textwrap.dedent(source), encoding='utf-8')
    return path


def test_reads_exactly_the_fields_dataclasses_fields_reports(tmp_path):
    path = _write(
        tmp_path / 'verdict.py',
        """\
        import typing
        from dataclasses import KW_ONLY, InitVar, dataclass, field
        from typing import ClassVar


        @dataclass
        class Verdict:
            passed: bool
            summary: str = ''
            logs: list[str] = field(default_factory=list)
            derived: int = field(init=False, default=0)
            registry: ClassVar[dict[str, int]] = {}
            qualified: typing.ClassVar[int] = 0
            seed: InitVar[int] = 0
            _: KW_ONLY
            strict: bool = False
            (paren): int = 0
            plain = 3

            class Nested:
                inner: int

            def label(self) -> str:
                return self.summary
        """,
    )
    ns: dict[str, object] = {'__name__': 'verdict_fixture'}
    exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec', dont_inherit=True), ns)
    verdict = ns['Verdict']
    assert dataclasses.is_dataclass(verdict)
    oracle = {f.name for f in dataclasses.fields(verdict)}

    names = _checker.dataclass_field_names(path, 'Verdict')

    assert isinstance(names, frozenset)
    assert names == {'passed', 'summary', 'logs', 'derived', 'strict'}
    assert names == oracle


def test_a_class_that_is_not_top_level_is_reported_missing(tmp_path):
    path = _write(
        tmp_path / 'nested.py',
        """\
        class Other:
            passed: bool


        def build():
            class Verdict:
                passed: bool

            return Verdict
        """,
    )

    with pytest.raises(ValueError) as excinfo:
        _checker.dataclass_field_names(path, 'Verdict')

    assert 'Verdict' in str(excinfo.value)
    assert str(path) in str(excinfo.value)


def test_a_class_defined_twice_at_top_level_is_refused(tmp_path):
    path = _write(
        tmp_path / 'twice.py',
        """\
        class Verdict:
            passed: bool


        class Verdict:
            summary: str
        """,
    )

    with pytest.raises(ValueError) as excinfo:
        _checker.dataclass_field_names(path, 'Verdict')

    assert 'Verdict' in str(excinfo.value)
    assert str(path) in str(excinfo.value)
    assert 'found 2' in str(excinfo.value)


def test_a_class_with_base_classes_is_refused(tmp_path):
    path = _write(
        tmp_path / 'derived.py',
        """\
        class Base:
            x: int


        class Verdict(Base):
            passed: bool
        """,
    )

    with pytest.raises(ValueError) as excinfo:
        _checker.dataclass_field_names(path, 'Verdict')

    assert 'Verdict' in str(excinfo.value)
    assert 'Base' in str(excinfo.value)
