"""A relative fused-memory data path is anchored at this checkout, never the CWD.

Task 5462.  These tests assert values only: nothing is created under
``CHECKOUT_ROOT``.
"""

import builtins
import importlib.util
import io
import os
from pathlib import Path
from types import ModuleType

import pytest

import fused_memory
from fused_memory.config import checkout_anchor
from fused_memory.config.checkout_anchor import (
    CHECKOUT_ROOT,
    anchor_at_checkout,
    holds_fused_memory_checkout,
)


@pytest.fixture
def installed_outside_a_checkout(tmp_path) -> ModuleType:
    """The module as a non-editable install imports it: from site-packages, with no checkout above."""
    site_packages = tmp_path / 'venv' / 'lib' / 'python3.13' / 'site-packages'
    installed = site_packages / 'fused_memory' / 'config' / 'checkout_anchor.py'
    installed.parent.mkdir(parents=True)
    installed.write_text(Path(checkout_anchor.__file__).read_text())
    spec = importlib.util.spec_from_file_location('installed_checkout_anchor', installed)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkout_root_is_the_checkout_holding_the_imported_package():
    assert CHECKOUT_ROOT.is_absolute()
    assert Path(fused_memory.__file__).resolve().is_relative_to(CHECKOUT_ROOT / 'fused-memory')
    assert (CHECKOUT_ROOT / 'fused-memory' / 'config' / 'config.yaml').is_file()
    assert checkout_anchor.ANCHOR_HOLDS_CHECKOUT is True


def test_only_a_root_holding_the_fused_memory_pyproject_holds_a_checkout(tmp_path):
    assert holds_fused_memory_checkout(tmp_path) is False

    (tmp_path / 'fused-memory').mkdir()
    assert holds_fused_memory_checkout(tmp_path) is False

    (tmp_path / 'fused-memory' / 'pyproject.toml').write_text('')
    assert holds_fused_memory_checkout(tmp_path) is True


def test_a_relative_path_is_joined_onto_the_checkout_whatever_the_cwd(
    monkeypatch, tmp_path, tmp_path_factory
):
    monkeypatch.chdir(tmp_path)
    from_tmp = anchor_at_checkout('./data/reconciliation')

    monkeypatch.chdir(tmp_path_factory.mktemp('elsewhere'))
    from_elsewhere = anchor_at_checkout('./data/reconciliation')

    assert from_tmp == str(CHECKOUT_ROOT / 'data' / 'reconciliation')
    assert from_elsewhere == from_tmp


def test_anchoring_a_relative_path_does_not_stat_resolve_or_open(monkeypatch):
    """Coroutines reach anchor_at_checkout through server/main.py::resolve_data_dir, so a call must not touch the filesystem (INV-8)."""

    def no_io(*_a, **_k):
        raise AssertionError('anchor_at_checkout touched the filesystem at call time')

    with monkeypatch.context() as patched:
        patched.setattr(Path, 'is_file', no_io)
        patched.setattr(Path, 'exists', no_io)
        patched.setattr(Path, 'stat', no_io)
        patched.setattr(Path, 'resolve', no_io)
        patched.setattr(os, 'stat', no_io)
        patched.setattr(os, 'lstat', no_io)
        patched.setattr(io, 'open', no_io)
        patched.setattr(builtins, 'open', no_io)
        anchored = anchor_at_checkout('./data/reconciliation')

    assert anchored == str(CHECKOUT_ROOT / 'data' / 'reconciliation')


def test_an_absolute_path_is_returned_byte_identical(tmp_path):
    absolute = str(tmp_path / 'abs')
    non_normalised = f'{tmp_path}/x/'

    assert anchor_at_checkout(absolute) == absolute
    assert anchor_at_checkout(non_normalised) == non_normalised


def test_a_relative_path_is_refused_when_the_anchor_holds_no_checkout(
    installed_outside_a_checkout, tmp_path
):
    """A non-editable install puts the anchor inside site-packages' parents.

    Joining onto it would fork an empty ledger there, so the refusal names the
    anchor and the offending path instead.
    """
    with pytest.raises(RuntimeError) as refusal:
        installed_outside_a_checkout.anchor_at_checkout('./data/reconciliation')

    message = str(refusal.value)
    assert str((tmp_path / 'venv' / 'lib').resolve()) in message
    assert './data/reconciliation' in message


def test_an_absolute_path_needs_no_checkout(installed_outside_a_checkout, tmp_path):
    absolute = str(tmp_path / 'abs')

    assert installed_outside_a_checkout.anchor_at_checkout(absolute) == absolute
