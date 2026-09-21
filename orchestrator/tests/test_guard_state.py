"""Tests for the ``orchestrator.guard_state`` durable-guard primitive (task 5352).

Covers:
  step-1: the ``collections.abc`` contract both facades must satisfy, driven
          purely in-memory (``path=None``) — this is the interface the nine
          existing call sites already speak, so the wiring steps can swap the
          facades in without editing a single one of them.
"""

from __future__ import annotations

from collections.abc import MutableMapping, MutableSet
from datetime import timedelta

from orchestrator.guard_state import PersistentMap, PersistentSet

_TTL = timedelta(days=1)


# ---------------------------------------------------------------------------
# step-1 — PersistentSet speaks MutableSet
# ---------------------------------------------------------------------------

class TestPersistentSetSemantics:
    """The four set-shaped guards use only ``add`` / ``discard`` / ``in`` /
    ``len`` / ``sorted`` / ``update`` / ``==``; pin exactly those."""

    def test_is_a_mutable_set(self):
        assert isinstance(PersistentSet(None, ttl=_TTL), MutableSet)

    def test_add_then_membership(self):
        s = PersistentSet(None, ttl=_TTL)
        assert 'esc-1' not in s
        s.add('esc-1')
        assert 'esc-1' in s

    def test_discard_removes(self):
        s = PersistentSet(None, ttl=_TTL)
        s.add('esc-1')
        s.discard('esc-1')
        assert 'esc-1' not in s

    def test_discard_of_absent_key_is_a_noop(self):
        """``_maybe_promote_blocker``'s done-clear arm discards unconditionally."""
        s = PersistentSet(None, ttl=_TTL)
        s.discard('never-added')
        assert len(s) == 0

    def test_len_counts_entries(self):
        s = PersistentSet(None, ttl=_TTL)
        assert len(s) == 0
        s.add('a')
        s.add('b')
        assert len(s) == 2

    def test_add_is_idempotent_for_membership_and_length(self):
        s = PersistentSet(None, ttl=_TTL)
        s.add('a')
        s.add('a')
        assert len(s) == 1

    def test_sorted_iterates_keys(self):
        """``_watch_for_escalation`` builds its ``--exclude-id`` argv this way."""
        s = PersistentSet(None, ttl=_TTL)
        s.update(['esc-2', 'esc-1'])
        assert sorted(s) == ['esc-1', 'esc-2']

    def test_update_inserts_every_element(self):
        """``_mark_coalesce_derailed`` marks a whole train in one call."""
        s = PersistentSet(None, ttl=_TTL)
        s.update(['a', 'b'])
        assert 'a' in s
        assert 'b' in s

    def test_equals_plain_set_in_both_directions(self):
        """``test_merge_queue_coalesce.py`` asserts ``== {'m1','m2'}`` against
        the real attribute, so the facade must compare equal to a plain set.

        Satisfied by inheriting ``collections.abc.Set.__eq__`` — no ``__eq__``
        is written on the facade.
        """
        s = PersistentSet(None, ttl=_TTL)
        s.update(['m1', 'm2'])
        plain = {'m1', 'm2'}
        assert s == plain
        # Bound to a local rather than written inline so the reflected
        # direction — the one that exercises `set.__eq__` returning
        # NotImplemented and Python retrying ours — is not a lint-flagged
        # Yoda condition.
        assert plain == s

    def test_empty_equals_plain_empty_set(self):
        assert PersistentSet(None, ttl=_TTL) == set()


# ---------------------------------------------------------------------------
# step-1 — PersistentMap speaks MutableMapping
# ---------------------------------------------------------------------------

class TestPersistentMapSemantics:
    """The five map-shaped guards use only ``[k]`` / ``[k] = v`` / ``[k] += 1``
    / ``get`` / ``pop`` / ``in`` / ``len`` / ``==``; pin exactly those."""

    def test_is_a_mutable_mapping(self):
        assert isinstance(PersistentMap(None, ttl=_TTL), MutableMapping)

    def test_setitem_then_getitem(self):
        m = PersistentMap(None, ttl=_TTL)
        m['k'] = 1
        assert m['k'] == 1

    def test_stores_str_values_too(self):
        """``open_fix_tasks[fp] = task_id`` stores a task id, not a count."""
        m = PersistentMap(None, ttl=_TTL)
        m['fp'] = 'fix-1'
        assert m['fp'] == 'fix-1'

    def test_get_returns_default_for_missing(self):
        """``self._retry_counts.get(esc_id, 0)`` is the steward's ladder read."""
        m = PersistentMap(None, ttl=_TTL)
        assert m.get('missing', 0) == 0

    def test_augmented_increment(self):
        """``self._red_advance_counts[fp] += 1`` is the offline lane's idiom."""
        m = PersistentMap(None, ttl=_TTL)
        m['k'] = 1
        m['k'] += 1
        assert m['k'] == 2

    def test_pop_removes_and_returns_then_defaults(self):
        """The resolve-path cleanup pops with a default and must not raise."""
        m = PersistentMap(None, ttl=_TTL)
        m['k'] = 7
        assert m.pop('k', None) == 7
        assert 'k' not in m
        assert m.pop('k', None) is None

    def test_membership_and_len(self):
        m = PersistentMap(None, ttl=_TTL)
        assert len(m) == 0
        m['k'] = 1
        assert 'k' in m
        assert len(m) == 1
        del m['k']
        assert 'k' not in m
        assert len(m) == 0

    def test_iterates_keys(self):
        m = PersistentMap(None, ttl=_TTL)
        m['b'] = 1
        m['a'] = 2
        assert sorted(m) == ['a', 'b']

    def test_equals_plain_dict_in_both_directions(self):
        """``assert worker.open_fix_tasks == {}`` in test_offline_lane.py
        depends on this; ``Mapping.__eq__`` supplies it."""
        m = PersistentMap(None, ttl=_TTL)
        empty: dict[str, str] = {}
        assert m == empty
        assert empty == m
        m['k'] = 'v'
        filled = {'k': 'v'}
        assert m == filled
        assert filled == m
