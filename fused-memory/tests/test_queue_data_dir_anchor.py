"""``queue.data_dir`` is absolute from config load, anchored at the checkout.

Task 10210.  The directory holds ``write_queue.db``, the planned-episode
registry and the overreach shadow sidecar.  A relative value used to resolve
against whatever CWD a process ran in, so a process launched from the wrong
directory silently opened a parallel, empty durable write queue.  These tests
assert values only: nothing is created under ``CHECKOUT_ROOT``.
"""

from fused_memory.config.checkout_anchor import CHECKOUT_ROOT
from fused_memory.config.schema import FusedMemoryConfig, QueueConfig


def test_the_schema_default_is_anchored_at_the_checkout(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert QueueConfig().data_dir == str(CHECKOUT_ROOT / 'data' / 'queue')


def test_a_relative_value_names_the_same_directory_from_every_cwd(
    monkeypatch, tmp_path, tmp_path_factory
):
    monkeypatch.chdir(tmp_path)
    from_tmp = QueueConfig(data_dir='rel/queue').data_dir

    monkeypatch.chdir(tmp_path_factory.mktemp('elsewhere'))
    from_elsewhere = QueueConfig(data_dir='rel/queue').data_dir

    assert from_tmp == str(CHECKOUT_ROOT / 'rel' / 'queue')
    assert from_elsewhere == from_tmp


def test_an_absolute_value_passes_through_byte_identical(tmp_path):
    absolute = str(tmp_path / 'abs')

    assert QueueConfig(data_dir=absolute).data_dir == absolute


def test_a_process_in_the_wrong_cwd_still_finds_the_checkout_write_queue(
    monkeypatch, tmp_path_factory
):
    """The tracked YAML's relative default, loaded from an unrelated CWD."""
    monkeypatch.delenv('QUEUE_DATA_DIR')
    monkeypatch.chdir(tmp_path_factory.mktemp('elsewhere'))

    config = FusedMemoryConfig()

    assert config.taskmaster is not None
    assert config.queue.data_dir == str(CHECKOUT_ROOT / 'data' / 'queue')
