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
from collections.abc import Iterable, Iterator
from typing import Any

import pytest

DEFECT_MARKER = 'defect'
CRITERION_MARKER = 'criterion'
COLLECTION_MAP_SUFFIX = '.tests.json'

_WORKEROUTPUT_PREFIX = '_df_defect_plugin.'


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line('markers', f'{DEFECT_MARKER}(ref): this test reproduces the declared defect <ref>.')
    config.addinivalue_line('markers', f'{CRITERION_MARKER}(id): this test evidences acceptance criterion <id>.')
    xmlpath = getattr(config.option, 'xmlpath', None)
    if xmlpath:
        target = os.path.abspath(os.path.expanduser(xmlpath)) + COLLECTION_MAP_SUFFIX
        config.pluginmanager.register(_CollectionMap(config, 'collection_map', target))


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
            self._config.workeroutput[self._key] = self.payload  # type: ignore[attr-defined]
        else:
            _write_atomically(self._target, self.payload)


class _CollectionMap(_HandOff):
    def pytest_collection_finish(self, session: pytest.Session) -> None:
        for item in session.items:
            entry = _map_entry(item)
            if entry is not None:
                self.payload[item.nodeid] = entry


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
