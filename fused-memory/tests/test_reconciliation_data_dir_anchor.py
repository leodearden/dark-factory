"""``reconciliation.data_dir`` is absolute from config load, anchored at the checkout.

Task 5462.  A relative value used to resolve against whatever CWD a process
ran in, so an operator script launched from the wrong directory silently
created a fresh, empty ledger.  These tests assert values only: nothing is
created under ``CHECKOUT_ROOT``.
"""

from fused_memory.config.checkout_anchor import CHECKOUT_ROOT
from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig


def test_the_schema_default_is_anchored_at_the_checkout(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert ReconciliationConfig().data_dir == str(CHECKOUT_ROOT / 'data' / 'reconciliation')


def test_a_relative_value_names_the_same_directory_from_every_cwd(
    monkeypatch, tmp_path, tmp_path_factory
):
    monkeypatch.chdir(tmp_path)
    from_tmp = ReconciliationConfig(data_dir='rel/recon').data_dir

    monkeypatch.chdir(tmp_path_factory.mktemp('elsewhere'))
    from_elsewhere = ReconciliationConfig(data_dir='rel/recon').data_dir

    assert from_tmp == str(CHECKOUT_ROOT / 'rel' / 'recon')
    assert from_elsewhere == from_tmp


def test_an_absolute_value_passes_through_byte_identical(tmp_path):
    absolute = str(tmp_path / 'abs')

    assert ReconciliationConfig(data_dir=absolute).data_dir == absolute


def test_an_operator_script_in_the_wrong_cwd_still_finds_the_checkout_ledger(
    monkeypatch, tmp_path_factory
):
    """The tracked YAML's relative default, loaded from an unrelated CWD."""
    monkeypatch.delenv('RECONCILIATION_DATA_DIR')
    monkeypatch.chdir(tmp_path_factory.mktemp('elsewhere'))

    config = FusedMemoryConfig()

    assert config.taskmaster is not None
    assert config.reconciliation.data_dir == str(CHECKOUT_ROOT / 'data' / 'reconciliation')


def test_escalation_queue_dir_is_not_anchored_at_load():
    """``reconciliation/harness.py::ReconciliationHarness._start_escalation_server``
    anchors it at ``explore_codebase_root`` instead."""
    assert ReconciliationConfig(escalation_queue_dir='esc').escalation_queue_dir == 'esc'
