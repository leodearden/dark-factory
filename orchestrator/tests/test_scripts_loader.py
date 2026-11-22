"""Contract of :func:`_scripts_loader.load_script`, the by-path loader for ``scripts/``.

``scripts/merge_lane_throughput.py`` is the fixture: it imports the real
``_checkout`` binder, which edits sys.path at import, so it exercises every
side effect the loader must contain.
"""

from __future__ import annotations

import sys

import pytest
from _scripts_loader import load_script


def test_load_executes_the_script_and_registers_it_under_the_given_name() -> None:
    name = '_mlt_loader_probe'
    try:
        module = load_script('merge_lane_throughput.py', name)
        assert sys.modules[name] is module
        assert callable(module.compute_mixes)
    finally:
        sys.modules.pop(name, None)


def test_load_leaves_import_resolution_as_it_found_it() -> None:
    name = '_mlt_loader_containment_probe'
    path_before = list(sys.path)
    binder_before = '_checkout' in sys.modules
    try:
        load_script('merge_lane_throughput.py', name)
        assert sys.path == path_before
        assert ('_checkout' in sys.modules) == binder_before
    finally:
        sys.modules.pop(name, None)


def test_a_failed_load_leaves_no_module_registered() -> None:
    name = '_missing_script_probe'
    try:
        with pytest.raises(FileNotFoundError):
            load_script('no_such_script.py', name)
        assert name not in sys.modules
    finally:
        sys.modules.pop(name, None)
