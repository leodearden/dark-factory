"""A relative fused-memory data path is anchored at this checkout, never the CWD.

Task 5462.  These tests assert values only: nothing is created under
``CHECKOUT_ROOT``.
"""

from pathlib import Path

import pytest

import fused_memory
from fused_memory.config import checkout_anchor
from fused_memory.config.checkout_anchor import CHECKOUT_ROOT, anchor_at_checkout


def test_checkout_root_is_the_checkout_holding_the_imported_package():
    assert CHECKOUT_ROOT.is_absolute()
    assert Path(fused_memory.__file__).resolve().is_relative_to(CHECKOUT_ROOT / 'fused-memory')
    assert (CHECKOUT_ROOT / 'fused-memory' / 'config' / 'config.yaml').is_file()


def test_a_relative_path_is_joined_onto_the_checkout_whatever_the_cwd(
    monkeypatch, tmp_path, tmp_path_factory
):
    monkeypatch.chdir(tmp_path)
    from_tmp = anchor_at_checkout('./data/reconciliation')

    monkeypatch.chdir(tmp_path_factory.mktemp('elsewhere'))
    from_elsewhere = anchor_at_checkout('./data/reconciliation')

    assert from_tmp == str(CHECKOUT_ROOT / 'data' / 'reconciliation')
    assert from_elsewhere == from_tmp


def test_an_absolute_path_is_returned_byte_identical(tmp_path):
    absolute = str(tmp_path / 'abs')
    non_normalised = f'{tmp_path}/x/'

    assert anchor_at_checkout(absolute) == absolute
    assert anchor_at_checkout(non_normalised) == non_normalised


def test_a_relative_path_is_refused_when_the_anchor_holds_no_checkout(monkeypatch, tmp_path):
    """A non-editable install puts the anchor inside site-packages' parents.

    Joining onto it would fork an empty ledger there, so the refusal names the
    anchor, the offending path and the remedy instead.
    """
    monkeypatch.setattr(checkout_anchor, 'CHECKOUT_ROOT', tmp_path)

    with pytest.raises(RuntimeError) as refusal:
        anchor_at_checkout('./data/reconciliation')

    message = str(refusal.value)
    assert str(tmp_path) in message
    assert './data/reconciliation' in message
    assert 'absolute' in message


def test_an_absolute_path_needs_no_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(checkout_anchor, 'CHECKOUT_ROOT', tmp_path)
    absolute = str(tmp_path / 'abs')

    assert anchor_at_checkout(absolute) == absolute
