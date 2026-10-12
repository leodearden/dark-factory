"""Tests for ``scripts/_write_triage_probe_common.py`` (task 5790).

The unit under test is the ref-independent plumbing the write_triage flip
gate's two probes share: item 1's ``check_write_triage_attach_target.py`` and
item 5's ``check_write_triage_attach_consumption.py``. Each probe's own suite
still drives the probe end to end; this file pins the shared seams once, so a
drift between the two copies they replaced cannot come back unnoticed.

Only the module's PUBLIC names are exercised. The imports are real and
in-process, so every test that imports a tree gets a uniquely named top-level
package, its own copy of ``sys.path`` and a ``sys.modules`` cleanup.
"""
from __future__ import annotations

import inspect
import logging
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import _write_triage_probe_common as common
import pytest

_FATAL = common.ImportMode.FATAL
_NON_FATAL = common.ImportMode.NON_FATAL


@pytest.fixture
def package(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A top-level package name no other test, and no installed tree, uses."""
    monkeypatch.setattr(sys, 'path', list(sys.path))
    name = f'wtpc_{uuid4().hex}'
    yield name
    for key in [k for k in sys.modules if k == name or k.startswith(f'{name}.')]:
        del sys.modules[key]


def _lay_module(root: Path, package: str, body: str) -> str:
    """Write ``<root>/<package>/mod.py`` holding *body*; return its dotted name."""
    (root / package).mkdir(parents=True)
    (root / package / '__init__.py').write_text('')
    (root / package / 'mod.py').write_text(body)
    return f'{package}.mod'


#: Each way a tree can fail to yield the module, and the words its report uses.
_FAILURES = {
    'not-a-directory': 'not a directory',
    'absent': 'cannot import',
    'exits': 'SystemExit',
    'outside-root': 'outside --src-root',
}


def _failing_tree(kind: str, tmp_path: Path, package: str) -> tuple[Path, list[Path]]:
    """``(src_root, extra_paths)`` that fail to import ``<package>.mod`` as *kind*."""
    if kind == 'not-a-directory':
        return tmp_path / 'no-such-tree', []
    src_root = tmp_path / 'src'
    src_root.mkdir()
    if kind == 'exits':
        _lay_module(src_root, package, 'raise SystemExit(0)\n')
    if kind == 'outside-root':
        # A SIBLING whose name extends the root's: the substitution a string
        # prefix test would wave through.
        installed = tmp_path / 'src-installed'
        _lay_module(installed, package, 'VALUE = 1\n')
        return src_root, [installed]
    return src_root, []


class TestIsInside:
    def test_a_descendant_is_inside(self, tmp_path: Path) -> None:
        assert common.is_inside(tmp_path / 'a' / 'b', tmp_path)

    def test_the_root_itself_is_inside(self, tmp_path: Path) -> None:
        assert common.is_inside(tmp_path, tmp_path)

    def test_a_sibling_extending_the_roots_name_is_not_inside(
        self, tmp_path: Path,
    ) -> None:
        root = tmp_path / 'src'
        assert not common.is_inside(tmp_path / 'src-installed' / 'x', root)


class TestImportFromTreeFatal:
    def test_returns_the_module_from_the_tree(
        self, tmp_path: Path, package: str,
    ) -> None:
        name = _lay_module(tmp_path, package, 'VALUE = 42\n')

        module = common.import_from_tree(name, tmp_path, mode=_FATAL)

        assert module.VALUE == 42
        assert common.is_inside(Path(str(module.__file__)), tmp_path)

    @pytest.mark.parametrize('kind', sorted(_FAILURES))
    def test_a_tree_that_cannot_yield_the_module_is_unverifiable(
        self, kind: str, tmp_path: Path, package: str,
    ) -> None:
        src_root, extra_paths = _failing_tree(kind, tmp_path, package)

        with pytest.raises(common.Unverifiable, match=_FAILURES[kind]):
            common.import_from_tree(
                f'{package}.mod', src_root, mode=_FATAL, extra_paths=extra_paths,
            )


class TestImportFromTreeNonFatal:
    @pytest.mark.parametrize('kind', sorted(_FAILURES))
    def test_a_tree_that_cannot_yield_the_module_warns_and_returns_none(
        self,
        kind: str,
        tmp_path: Path,
        package: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        src_root, extra_paths = _failing_tree(kind, tmp_path, package)
        name = f'{package}.mod'

        with caplog.at_level(logging.WARNING, logger=common.PROBE_LOGGER_NAME):
            module = common.import_from_tree(
                name, src_root, mode=_NON_FATAL, extra_paths=extra_paths,
            )

        assert module is None
        warned = [
            record.getMessage()
            for record in caplog.records
            if record.name == common.PROBE_LOGGER_NAME
            and record.levelno >= logging.WARNING
        ]
        assert any(name in message for message in warned), warned


@pytest.mark.parametrize('mode', list(common.ImportMode))
def test_ctrl_c_in_the_module_body_belongs_to_the_operator(
    mode: common.ImportMode, tmp_path: Path, package: str,
) -> None:
    name = _lay_module(tmp_path, package, 'raise KeyboardInterrupt\n')

    with pytest.raises(KeyboardInterrupt):
        common.import_from_tree(name, tmp_path, mode=mode)


class TestRequire:
    @staticmethod
    def _module() -> ModuleType:
        module = ModuleType('wtpc_required')
        module.__dict__.update(present='here', nothing=None)
        return module

    def test_returns_the_attribute(self) -> None:
        assert common.require(self._module(), 'present') == 'here'

    @pytest.mark.parametrize('name', ['absent', 'nothing'])
    def test_a_missing_or_none_attribute_is_unverifiable(self, name: str) -> None:
        with pytest.raises(common.Unverifiable, match=f'wtpc_required exposes no {name}'):
            common.require(self._module(), name)


class TestParentKey:
    def test_the_modules_own_key_wins(self) -> None:
        assert common.parent_key(SimpleNamespace(PARENT_ID_KEY='x_parent')) == 'x_parent'

    @pytest.mark.parametrize(
        'module',
        [SimpleNamespace(), SimpleNamespace(PARENT_ID_KEY=''), SimpleNamespace(PARENT_ID_KEY=7)],
        ids=['absent', 'empty', 'not-a-str'],
    )
    def test_falls_back_to_parent_id(self, module: SimpleNamespace) -> None:
        assert common.parent_key(module) == 'parent_id'


def _every_parameter_kind(a, /, b, *args, c, **kwargs) -> None:
    """One parameter of each kind, in declaration order."""


class TestUsableParameters:
    def test_keeps_positional_and_keyword_parameters_in_order(self) -> None:
        names = [p.name for p in common.usable_parameters(_every_parameter_kind)]

        assert names == ['a', 'b', 'c']

    def test_an_uninspectable_object_has_none(self) -> None:
        assert common.usable_parameters(1) == []


class TestNamesATarget:
    @pytest.mark.parametrize(
        ('name', 'expected'),
        [
            ('attach_target_id', True),
            ('target', True),
            ('candidate_id', True),
            # The band's vocabulary, deliberately NOT a target name: a
            # `canonical_id` that merely interpolates the band canonical
            # once passed the gate while binding nothing.
            ('canonical_id', False),
            ('footer_note', False),
            ('verdict_words', False),
        ],
    )
    def test_only_names_that_designate_the_attach_target(
        self, name: str, expected: bool,
    ) -> None:
        param = inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)

        assert common.names_a_target(param) is expected


def _report(
    evaluate: Callable[[list[str]], int], capsys: pytest.CaptureFixture[str],
) -> tuple[int, list[str]]:
    """``(rc, stdout lines)`` of one ``report_fail_closed`` run."""
    rc = common.report_fail_closed(evaluate)
    return rc, capsys.readouterr().out.splitlines()


def _probe_logger() -> logging.Logger:
    return logging.getLogger(common.PROBE_LOGGER_NAME)


class TestReportFailClosed:
    def test_a_verdict_is_written_as_reported(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        def evaluate(out: list[str]) -> int:
            out.extend(['line one', 'line two'])
            return common.EXIT_OK

        rc = common.report_fail_closed(evaluate)

        assert rc == common.EXIT_OK
        assert capsys.readouterr().out == 'line one\nline two\n'

    def test_an_unverifiable_invariant_fails_closed(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        def evaluate(out: list[str]) -> int:
            raise common.Unverifiable('why')

        rc, lines = _report(evaluate, capsys)

        assert rc == common.EXIT_FAIL
        assert 'FAIL  UNVERIFIABLE: why' in lines
        assert any('Failing closed' in line for line in lines)

    @pytest.mark.parametrize(
        ('escape', 'named'),
        [(SystemExit(0), 'SystemExit'), (KeyboardInterrupt(), 'KeyboardInterrupt')],
        ids=['system-exit-0', 'ctrl-c'],
    )
    def test_an_escape_from_the_evaluation_fails_closed(
        self,
        escape: BaseException,
        named: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # SystemExit(0) out of the ref's code once exited 0 having printed
        # nothing, which the gate read as PASS.
        def evaluate(out: list[str]) -> int:
            raise escape

        rc, lines = _report(evaluate, capsys)

        assert rc == common.EXIT_FAIL
        verdicts = [line for line in lines if line.startswith('FAIL  UNVERIFIABLE')]
        assert verdicts, lines
        assert named in '\n'.join(lines)

    def test_warnings_come_last_prefixed_and_once_each(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        def evaluate(out: list[str]) -> int:
            _probe_logger().warning('w1')
            _probe_logger().warning('w1')
            _probe_logger().warning('w2')
            _probe_logger().info('quiet')
            out.append('body')
            return common.EXIT_OK

        _, lines = _report(evaluate, capsys)

        assert lines[-3:] == ['body', 'WARN  w1', 'WARN  w2']
        assert not any('quiet' in line for line in lines)

    def test_warnings_follow_an_unverifiable_verdict(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        def evaluate(out: list[str]) -> int:
            _probe_logger().warning('w1')
            raise common.Unverifiable('why')

        _, lines = _report(evaluate, capsys)

        assert lines[-1] == 'WARN  w1'
        assert lines.index('FAIL  UNVERIFIABLE: why') < lines.index('WARN  w1')

    def test_the_collector_lives_exactly_one_run(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        handlers_before = list(_probe_logger().handlers)
        propagate_before = _probe_logger().propagate

        _report(lambda out: common.EXIT_OK, capsys)
        _probe_logger().warning('between')
        _, lines = _report(lambda out: common.EXIT_OK, capsys)

        assert not any('between' in line for line in lines)
        assert _probe_logger().handlers == handlers_before
        assert _probe_logger().propagate is propagate_before

    def test_a_non_fatal_import_failure_reaches_the_report(
        self,
        tmp_path: Path,
        package: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        name = f'{package}.mod'

        def evaluate(out: list[str]) -> int:
            common.import_from_tree(name, tmp_path, mode=_NON_FATAL)
            return common.EXIT_OK

        _, lines = _report(evaluate, capsys)

        assert any(line.startswith('WARN  ') and name in line for line in lines), lines
