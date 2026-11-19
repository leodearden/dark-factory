"""The pytest plugin verify injects into a target's test legs as ``-p _df_defect_plugin``.

It is the only module in a directory with no ``__init__.py``, and it imports the
standard library and pytest only: verify puts that directory LAST on the
target's ``PYTHONPATH``, so it is importable in a target that has no ``shared``
installed and can shadow none of the target's own modules
(plans/defect-tests-prd.md decision 19).

It registers the ``defect(ref)`` and ``criterion(id)`` markers additively, so a
target running ``--strict-markers`` accepts them and keeps its own.

THE COLLECTION MAP. When the run is given ``--junitxml=<report>``, the plugin
writes ``<report>.tests.json`` (``COLLECTION_MAP_SUFFIX`` appended to the
report's absolute path). It is a JSON object keyed by node id, holding one entry
per collected item that carries a well-formed ``defect`` or ``criterion`` mark::

    {"<nodeid>": {"defects": ["<ref>", ...], "criterion": "<id>" | null}}

A mark is well-formed when it has exactly one positional str argument and no
keywords, the rule of ``shared/src/shared/defect_enumerator.py::_read_ref``;
other shapes are left out, and the static scan reports them. ``defects`` holds
every well-formed defect ref in ``item.iter_markers`` order, recorded verbatim:
consumers apply ``shared.defects.parse_ref``. ``criterion`` is the closest
well-formed criterion mark. The file is written whenever a junit report is
requested, ``{}`` included, so an absent map means "not recorded" and ``{}``
means "nothing marked".

THE OUTCOME FILE. When the run is given ``--df-defect-outcomes=<path>`` (only
proof legs are), the plugin writes ``<path>``, a JSON object with one record per
test node id, plus one per collector whose collection errored, keyed by the
collector's node id; a consumer resolves a test to it through the test's
enclosing collector::

    {"<nodeid>": {"side_outcome": "<SideOutcome value>", "exc_type": "<module.qualname>" | null,
                  "module": str | null, "symbol": str | null, "owner_type": "<module.qualname>" | null}}

``side_outcome`` takes the values of ``shared.defects.SideOutcome`` (never
``not_collected``, which only a consumer can observe). The first phase whose
report is not passed decides it: a failed setup or teardown is an error, a failed
call is ``failed``, a skip in any phase (an xfail included) is ``skipped``, and
all phases passing is ``passed``. ``exc_type`` is set for failed and error
records only. The names are read from the exception's attributes, never its
text: ``ModuleNotFoundError`` gives ``module``; any other ``ImportError`` gives
``module`` and ``symbol`` (``name_from``); ``AttributeError`` gives ``symbol``,
``owner_type`` (the type of the object it was raised on) and, when that object
is a module, ``module``. A collection error raised by pytest as
``Collector.CollectError`` is read through its ``__cause__``; a conftest import
failure is not unwrapped. An error is ``error_attributed`` only when its class's
names are complete (a module for ``ModuleNotFoundError``, a module and a symbol
for ``ImportError``, a symbol for ``AttributeError``), else
``error_unattributed``. ``error_attributed`` means only that a complete
structural name was recorded: the proof stage decides whether that name is
absent at base and defined at tip.

WHO WRITES, AND WHEN. The writer is the one process that is not an xdist worker:
the run itself, or the xdist controller (a worker is a process whose config
carries ``workerinput``). A worker ships its part through
``config.workeroutput``; the writer merges every part at
``pytest_testnodedown`` and writes once, at its own ``pytest_sessionfinish``,
atomically (a temporary file in the target directory, then ``os.replace``). A
write that fails is reported as one line on stderr and never fails the run.

NODE IDS are pytest's own, relative to the run's rootdir. A consumer that stores
one qualifies it as ``NodeId(module_root, path, name)``.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import types
from collections.abc import Generator, Iterable, Iterator
from typing import Any

import pytest

DEFECT_MARKER = 'defect'
CRITERION_MARKER = 'criterion'
COLLECTION_MAP_SUFFIX = '.tests.json'

_WORKEROUTPUT_PREFIX = '_df_defect_plugin.'


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        '--df-defect-outcomes',
        default=None,
        metavar='path',
        help='write every test\'s side outcome to this JSON file (defect-test proofs).',
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line('markers', f'{DEFECT_MARKER}(ref): this test reproduces the declared defect <ref>.')
    config.addinivalue_line('markers', f'{CRITERION_MARKER}(id): this test evidences acceptance criterion <id>.')
    xmlpath = getattr(config.option, 'xmlpath', None)
    if xmlpath:
        target = os.path.abspath(os.path.expanduser(xmlpath)) + COLLECTION_MAP_SUFFIX
        config.pluginmanager.register(_CollectionMap(config, 'collection_map', target))
    outcomes = config.getoption('df_defect_outcomes')
    if outcomes:
        config.pluginmanager.register(_OutcomeRecorder(config, 'outcomes', os.path.abspath(outcomes)))


class _HandOff:
    """One output file: xdist workers ship their part, the writer merges the parts and writes the file."""

    def __init__(self, config: pytest.Config, name: str, target: str) -> None:
        self._config = config
        self._key = _WORKEROUTPUT_PREFIX + name
        self._target = target
        self.payload: dict[str, Any] = {}

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node: Any, error: object) -> None:
        self.payload.update(getattr(node, 'workeroutput', {}).get(self._key, {}))

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        if hasattr(self._config, 'workerinput'):
            _ship_to_controller(self._config, self._key, self.payload)
        else:
            _write_atomically(self._target, self.payload)


def _ship_to_controller(config: pytest.Config, key: str, payload: dict[str, Any]) -> None:
    """xdist sets ``workeroutput`` beside ``workerinput`` and sends it to the controller after sessionfinish."""
    vars(config)['workeroutput'][key] = payload


class _CollectionMap(_HandOff):
    def pytest_collection_finish(self, session: pytest.Session) -> None:
        for item in session.items:
            entry = _map_entry(item)
            if entry is not None:
                self.payload[item.nodeid] = entry


class _OutcomeRecorder(_HandOff):
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item: pytest.Item, call: pytest.CallInfo[None]) -> Generator[None, Any, None]:
        outcome = yield
        decided = self.payload.get(item.nodeid)
        if decided is None or decided['side_outcome'] == 'passed':
            self.payload[item.nodeid] = _phase_record(outcome.get_result(), call)

    def pytest_exception_interact(self, node: Any, call: pytest.CallInfo[Any], report: Any) -> None:
        if isinstance(report, pytest.CollectReport) and call.excinfo is not None:
            self.payload[node.nodeid] = _error_record(call.excinfo.value, failed=False)


def _phase_record(report: pytest.TestReport, call: pytest.CallInfo[None]) -> dict[str, Any]:
    if report.skipped:
        return _record('skipped')
    if report.passed:
        return _record('passed')
    exc = call.excinfo.value if call.excinfo is not None else None
    return _error_record(exc, failed=report.when == 'call')


def _error_record(exc: BaseException | None, *, failed: bool) -> dict[str, Any]:
    if isinstance(exc, pytest.Collector.CollectError) and exc.__cause__ is not None:
        exc = exc.__cause__
    if exc is None:
        return _record('failed' if failed else 'error_unattributed')
    module, symbol, owner_type = _structural_names(exc)
    if failed:
        side_outcome = 'failed'
    elif _names_complete(exc, module, symbol):
        side_outcome = 'error_attributed'
    else:
        side_outcome = 'error_unattributed'
    return _record(side_outcome, _qualified(type(exc)), module, symbol, owner_type)


def _structural_names(exc: BaseException) -> tuple[str | None, str | None, str | None]:
    if isinstance(exc, ModuleNotFoundError):
        return exc.name, None, None
    if isinstance(exc, ImportError):
        return exc.name, getattr(exc, 'name_from', None), None
    if isinstance(exc, AttributeError) and exc.name is not None:
        owner = exc.obj
        module = owner.__name__ if isinstance(owner, types.ModuleType) else None
        return module, exc.name, _qualified(type(owner))
    return None, None, None


def _names_complete(exc: BaseException, module: str | None, symbol: str | None) -> bool:
    if isinstance(exc, ModuleNotFoundError):
        return module is not None
    if isinstance(exc, ImportError):
        return module is not None and symbol is not None
    return isinstance(exc, AttributeError) and symbol is not None


def _qualified(cls: type) -> str:
    return f'{cls.__module__}.{cls.__qualname__}'


def _record(
    side_outcome: str,
    exc_type: str | None = None,
    module: str | None = None,
    symbol: str | None = None,
    owner_type: str | None = None,
) -> dict[str, Any]:
    return {
        'side_outcome': side_outcome,
        'exc_type': exc_type,
        'module': module,
        'symbol': symbol,
        'owner_type': owner_type,
    }


def _map_entry(item: pytest.Item) -> dict[str, Any] | None:
    defects = list(_well_formed_refs(item.iter_markers(DEFECT_MARKER)))
    criterion = next(_well_formed_refs(item.iter_markers(CRITERION_MARKER)), None)
    if not defects and criterion is None:
        return None
    return {'defects': defects, 'criterion': criterion}


def _well_formed_refs(marks: Iterable[pytest.Mark]) -> Iterator[str]:
    for mark in marks:
        if len(mark.args) == 1 and not mark.kwargs and isinstance(mark.args[0], str):
            yield mark.args[0]


def _write_atomically(target: str, payload: dict[str, Any]) -> None:
    directory = os.path.dirname(target)
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=f'.{os.path.basename(target)}.', suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, sort_keys=True)
            os.replace(tmp, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
    except OSError as exc:
        sys.stderr.write(f'_df_defect_plugin: could not write {target}: {exc}\n')
