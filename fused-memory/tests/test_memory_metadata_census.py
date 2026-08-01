"""Tests for :mod:`fused_memory.services.memory_metadata_census` — the
observability half of the Mem0 metadata vocabulary contract (task 3195,
leaf β of ``docs/prds/memory-metadata-vocabulary.md``).

Two contracts live here, both from PRD V1 / INV-4:

* the grep-anchored **census line**, which is how a warn-mode violation is
  actually observed (in warn mode nothing raises, so the log line IS the
  signal — if it is wrong, the whole warn tier is silent);
* the **unknown-key storm escape**, so "a drifting writer flooding unknown
  keys is heard, not logged into oblivion".

The clock is injected everywhere; these tests never sleep.
"""

import logging

import pytest

_CENSUS_LOGGER = 'fused_memory.services.memory_metadata_census'


def _violation(key='weird_key', code='unknown_key', message='m', fatal=False):
    from fused_memory.memory_metadata import MetadataViolation

    return MetadataViolation(key=key, code=code, message=message, fatal=fatal)


class TestCensusLine:
    """The grep anchor ``memory_metadata.schema_warning``.

    Ported from ``backends/sqlite_task_backend.py:853`` ``_emit_schema_warning``,
    one domain over.
    """

    def _emit(self, violations, *, project_id: str = 'dark_factory',
              agent_id: str | None = 'claude-x'):
        # `agent_id: str | None` mirrors emit_schema_warnings' own signature —
        # the absent-agent_id placeholder case below passes None deliberately.
        from fused_memory.services.memory_metadata_census import emit_schema_warnings

        emit_schema_warnings(violations, project_id=project_id, agent_id=agent_id)

    def test_one_violation_emits_exactly_one_warning_with_every_field(self, caplog):
        """The anchor token AND each k=v pair are asserted SEPARATELY.

        A single "does the whole line look right" assertion would let a
        future format tweak keep the anchor while silently dropping a field,
        which is exactly the failure that makes a census unusable months
        later when someone tries to grep it.

        NOTE: ``memory_metadata.schema_warning`` is CONTRACT-FIXED. It is
        what the enforce-gate census greps for. It must never be renamed.
        """
        caplog.set_level(logging.WARNING, logger=_CENSUS_LOGGER)
        self._emit(
            [_violation(key='weird_key', code='unknown_key')],
            project_id='dark_factory',
            agent_id='claude-task-3195',
        )
        records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(records) == 1
        message = records[0].getMessage()
        assert 'memory_metadata.schema_warning' in message
        assert 'project_id=dark_factory' in message
        assert 'agent_id=claude-task-3195' in message
        assert 'code=unknown_key' in message
        assert 'key=weird_key' in message

    def test_one_line_per_violation(self, caplog):
        caplog.set_level(logging.WARNING, logger=_CENSUS_LOGGER)
        self._emit([
            _violation(key='a', code='unknown_key'),
            _violation(key='b', code='unknown_key'),
            _violation(key='topic', code='invalid_topic_slug', fatal=True),
        ])
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 3

    def test_no_violations_emits_nothing(self, caplog):
        caplog.set_level(logging.WARNING, logger=_CENSUS_LOGGER)
        self._emit([])
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_absent_agent_id_renders_a_stable_placeholder(self, caplog):
        """`agent_id=None` must not render as the bare string 'None'.

        'None' is indistinguishable from a writer that literally set that
        string, which would corrupt the per-writer aggregation the storm
        detector keys on. A distinct placeholder keeps "unset" greppable.
        """
        caplog.set_level(logging.WARNING, logger=_CENSUS_LOGGER)
        self._emit([_violation()], agent_id=None)
        message = caplog.records[0].getMessage()
        assert 'agent_id=None' not in message
        assert 'agent_id=<unset>' in message

    def test_fatal_violation_also_censuses_under_warn_mode(self, caplog):
        """Nothing is silently swallowed.

        Under warn mode a fatal violation does not reject — so if it did not
        also census, the single most enforcement-relevant class of violation
        would be the one class that left no trace at all.
        """
        caplog.set_level(logging.WARNING, logger=_CENSUS_LOGGER)
        self._emit([_violation(key='topic', code='invalid_topic_slug', fatal=True)])
        message = caplog.records[0].getMessage()
        assert 'memory_metadata.schema_warning' in message
        assert 'code=invalid_topic_slug' in message


class _FakeClock:
    """Injected monotonic clock — deterministic, and no test ever sleeps."""

    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class TestUnknownKeyStormDetector:
    """Per-(project_id, agent_id) rolling-window counter.

    A storm is a DRIFTING WRITER, not global volume: with a 1,627-key live
    baseline a global counter would fire on healthy fleet traffic and would
    never identify the culprit.
    """

    def _detector(self, clock, *, threshold=5, window_seconds=300):
        from fused_memory.services.memory_metadata_census import UnknownKeyStormDetector

        return UnknownKeyStormDetector(
            threshold=threshold, window_seconds=window_seconds, time_fn=clock
        )

    def test_below_threshold_never_fires(self):
        clock = _FakeClock()
        detector = self._detector(clock, threshold=5)
        for _ in range(4):
            assert detector.record('p', 'a', ['k']) is False

    def test_fires_exactly_on_the_crossing_call(self):
        clock = _FakeClock()
        detector = self._detector(clock, threshold=5)
        results = [detector.record('p', 'a', ['k']) for _ in range(5)]
        assert results == [False, False, False, False, True]

    def test_does_not_re_fire_after_crossing(self):
        """Returning True forever would mint a fresh escalation per write.

        The filer's open-escalation dedup is the second line of defence, but
        a detector that latched True would hammer it on every single write.
        """
        clock = _FakeClock()
        detector = self._detector(clock, threshold=3)
        assert [detector.record('p', 'a', ['k']) for _ in range(3)][-1] is True
        for _ in range(10):
            assert detector.record('p', 'a', ['k']) is False

    def test_each_key_counts_as_one_warn(self):
        """The census emits one line per KEY, so the counter must too."""
        clock = _FakeClock()
        detector = self._detector(clock, threshold=3)
        assert detector.record('p', 'a', ['k1', 'k2']) is False
        assert detector.record('p', 'a', ['k3']) is True

    def test_writers_do_not_aggregate(self):
        """A different agent_id must not push another writer over the line."""
        clock = _FakeClock()
        detector = self._detector(clock, threshold=3)
        assert detector.record('p', 'agent-a', ['k']) is False
        assert detector.record('p', 'agent-b', ['k']) is False
        assert detector.record('p', 'agent-b', ['k']) is False
        # agent-a still has 1 of 3 — agent-b's traffic did not count for it.
        assert detector.record('p', 'agent-a', ['k']) is False

    def test_projects_do_not_aggregate(self):
        clock = _FakeClock()
        detector = self._detector(clock, threshold=3)
        assert detector.record('proj-a', 'a', ['k']) is False
        assert detector.record('proj-b', 'a', ['k']) is False
        assert detector.record('proj-b', 'a', ['k']) is False
        assert detector.record('proj-a', 'a', ['k']) is False

    def test_stale_warns_drop_out_of_the_window(self):
        """A slow trickle must never accumulate into a false storm."""
        clock = _FakeClock()
        detector = self._detector(clock, threshold=3, window_seconds=300)
        assert detector.record('p', 'a', ['k']) is False
        assert detector.record('p', 'a', ['k']) is False
        clock.advance(301)
        # The first two are now stale, so this is warn 1 of 3, not 3 of 3.
        assert detector.record('p', 'a', ['k']) is False
        assert detector.record('p', 'a', ['k']) is False
        assert detector.record('p', 'a', ['k']) is True

    def test_empty_key_list_is_a_no_op(self):
        clock = _FakeClock()
        detector = self._detector(clock, threshold=1)
        assert detector.record('p', 'a', []) is False

    def test_a_writer_that_falls_silent_leaves_no_residue(self):
        """Process-lifetime state must not grow without bound.

        A writer's stale timestamps are pruned only when that SAME writer
        records again, so without a sweep every distinct writer that ever
        drifted keeps a dict entry (plus a window of floats) for the life of
        the process. agent_id is free-form per-task text in this fleet, so
        that key space is effectively unbounded in a long-lived MCP server.
        """
        clock = _FakeClock()
        detector = self._detector(clock, threshold=3, window_seconds=300)
        detector._sweep_every = 2

        detector.record('p', 'transient-writer', ['k'])
        assert ('p', 'transient-writer') in detector._warns

        clock.advance(301)
        # Two records by a DIFFERENT writer, so the sweep is what evicts the
        # silent one — not its own prune loop, which never runs again.
        detector.record('p', 'steady-writer', ['k'])
        detector.record('p', 'steady-writer', ['k'])

        assert ('p', 'transient-writer') not in detector._warns
        assert ('p', 'steady-writer') in detector._warns, 'the live writer must survive'

    def test_the_sweep_never_evicts_the_writer_that_triggered_it(self):
        """Its deque was appended at `now`, so it is never stale."""
        clock = _FakeClock()
        detector = self._detector(clock, threshold=3, window_seconds=300)
        detector._sweep_every = 1

        detector.record('p', 'a', ['k'])
        detector.record('p', 'a', ['k'])
        assert len(detector._warns[('p', 'a')]) == 2
        # And the counting contract still holds across the sweep.
        assert detector.record('p', 'a', ['k']) is True

    def test_eviction_clears_the_firing_latch_so_a_recurrence_is_heard(self):
        """A drained writer is no longer over the line.

        Leaving it in `_firing` after evicting it from `_warns` would create a
        latch nothing can ever clear, permanently silencing a writer that
        drifts, is fixed, and later drifts again.
        """
        clock = _FakeClock()
        detector = self._detector(clock, threshold=2, window_seconds=300)
        detector._sweep_every = 2

        assert detector.record('p', 'a', ['k', 'k2']) is True
        assert ('p', 'a') in detector._firing

        clock.advance(301)
        detector.record('p', 'other', ['k'])
        detector.record('p', 'other', ['k'])
        assert ('p', 'a') not in detector._firing

        assert detector.record('p', 'a', ['k', 'k2']) is True, 'must fire again'

    def test_defaults_to_monotonic_not_wall_clock(self):
        """Wall-clock would let an NTP step corrupt the window."""
        import time

        from fused_memory.services.memory_metadata_census import UnknownKeyStormDetector

        detector = UnknownKeyStormDetector(threshold=5, window_seconds=300)
        assert detector._time_fn is time.monotonic


class TestFileUnknownKeyStormEscalation:
    """Direct port of ``middleware/candidate_key_escalation.py``.

    Same optional-import guard, same stable anchor, same open-escalation
    dedup, same hard never-raises contract.
    """

    def _file(self, tmp_path, *, project_id='dark_factory', agent_id='claude-x',
              keys=('weird_key', 'other_key')):
        from fused_memory.services.memory_metadata_census import (
            file_unknown_key_storm_escalation,
        )

        return file_unknown_key_storm_escalation(
            str(tmp_path), project_id=project_id, agent_id=agent_id, keys=list(keys)
        )

    def test_files_one_escalation_naming_the_writer_and_the_keys(self, tmp_path, monkeypatch):
        import fused_memory.services.memory_metadata_census as census

        submitted = []

        class _Queue:
            def __init__(self, path):
                self.path = path

            def get_by_task(self, task_id, status=None):
                return []

            def make_id(self, task_id):
                return f'esc-{task_id}-1'

            def submit(self, esc):
                submitted.append(esc)
                return esc.id

        monkeypatch.setattr(census, 'HAS_ESCALATION', True)
        monkeypatch.setattr(census, 'EscalationQueue', _Queue)

        esc_id = self._file(tmp_path, project_id='dark_factory', agent_id='claude-drifter')
        assert len(submitted) == 1
        esc = submitted[0]
        assert esc_id == esc.id
        # The summary must identify WHO is drifting — an escalation that
        # only says "a storm happened" leaves the operator to go find the
        # writer by hand, which is the whole job.
        assert 'dark_factory' in esc.summary
        assert 'claude-drifter' in esc.summary
        blob = f'{esc.summary}\n{esc.detail}'
        assert 'weird_key' in blob
        assert 'other_key' in blob

    def test_dedups_against_an_already_open_escalation(self, tmp_path, monkeypatch):
        """A persistent drift must not mint a fresh escalation per restart."""
        import fused_memory.services.memory_metadata_census as census

        submitted = []

        class _Existing:
            id = 'esc-memory-metadata-unknown-key-storm-1'

        class _Queue:
            def __init__(self, path):
                pass

            def get_by_task(self, task_id, status=None):
                return [_Existing()]

            def make_id(self, task_id):
                return 'esc-new-should-not-be-used'

            def submit(self, esc):  # pragma: no cover — must not be reached
                submitted.append(esc)
                return esc.id

        monkeypatch.setattr(census, 'HAS_ESCALATION', True)
        monkeypatch.setattr(census, 'EscalationQueue', _Queue)

        esc_id = self._file(tmp_path)
        assert submitted == []
        assert esc_id == _Existing.id

    def test_uses_a_stable_greppable_per_writer_anchor(self, tmp_path, monkeypatch):
        """One greppable family prefix, scoped down to the writer.

        The three call sites (dedup query / id mint / stored task_id) must
        agree — a mismatch would make the dedup query look for an anchor
        nothing is ever filed under, silently disabling it.
        """
        import fused_memory.services.memory_metadata_census as census

        seen = {}

        class _Queue:
            def __init__(self, path):
                pass

            def get_by_task(self, task_id, status=None):
                seen['queried'] = task_id
                return []

            def make_id(self, task_id):
                seen['minted'] = task_id
                return f'esc-{task_id}-1'

            def submit(self, esc):
                seen['task_id'] = esc.task_id
                return esc.id

        monkeypatch.setattr(census, 'HAS_ESCALATION', True)
        monkeypatch.setattr(census, 'EscalationQueue', _Queue)

        self._file(tmp_path, project_id='dark_factory', agent_id='claude-x')
        expected = 'memory-metadata-unknown-key-storm-dark-factory-claude-x'
        assert seen['queried'] == expected
        assert seen['minted'] == expected
        assert seen['task_id'] == expected
        # The family prefix stays intact, so one grep still finds the series.
        assert expected.startswith('memory-metadata-unknown-key-storm')

    def test_two_different_writers_get_two_escalations(self, tmp_path, monkeypatch):
        """A global anchor would mask every writer after the first.

        The escalation's whole job is to name WHICH writer is drifting. With
        one shared anchor, writer A's still-open escalation absorbs writer
        B's crossing and B survives only in an INFO log line — the operator
        sees one culprit named and no signal that anyone else crossed.
        """
        import fused_memory.services.memory_metadata_census as census

        open_by_task = {}

        class _Existing:
            def __init__(self, esc_id):
                self.id = esc_id

        class _Queue:
            def __init__(self, path):
                pass

            def get_by_task(self, task_id, status=None):
                found = open_by_task.get(task_id)
                return [found] if found else []

            def make_id(self, task_id):
                return f'esc-{task_id}-1'

            def submit(self, esc):
                open_by_task[esc.task_id] = _Existing(esc.id)
                return esc.id

        monkeypatch.setattr(census, 'HAS_ESCALATION', True)
        monkeypatch.setattr(census, 'EscalationQueue', _Queue)

        first = self._file(tmp_path, agent_id='claude-drifter-a')
        second = self._file(tmp_path, agent_id='claude-drifter-b')

        # Both writers must actually get an id: a `None` here would mean the
        # filing degraded to a log line, and the per-writer assertions below
        # would then be vacuous rather than failing.
        assert first is not None
        assert second is not None
        assert first != second, 'the second writer must not be folded into the first'
        assert len(open_by_task) == 2
        assert 'claude-drifter-a' in first
        assert 'claude-drifter-b' in second

        # ...while a REPEAT from the same writer still dedups, so one writer
        # cannot flood the queue.
        assert self._file(tmp_path, agent_id='claude-drifter-a') == first
        assert len(open_by_task) == 2

    def test_anchor_slugs_unsafe_writer_ids(self):
        """The anchor becomes a `.seq` FILENAME via `make_id`.

        agent_id is free-form text, so path separators, spaces and unbounded
        length all have to be neutralised before it names a file.
        """
        from fused_memory.services.memory_metadata_census import writer_anchor_task_id

        anchor = writer_anchor_task_id('Proj/Ect', 'agent id/../etc')
        assert '/' not in anchor
        assert ' ' not in anchor
        assert anchor == 'memory-metadata-unknown-key-storm-proj-ect-agent-id-etc'

        long_anchor = writer_anchor_task_id('p', 'a' * 500)
        assert len(long_anchor) < 150

    def test_absent_agent_id_gets_its_own_anchor(self):
        """`None` must not collide with a writer literally named 'unknown'."""
        from fused_memory.services.memory_metadata_census import writer_anchor_task_id

        assert writer_anchor_task_id('p', None).endswith('-unset')

    def test_returns_none_and_never_raises_without_the_escalation_package(
        self, tmp_path, monkeypatch, caplog
    ):
        """This runs on the LIVE memory write path.

        A raise here would turn a census warning into a lost memory — the
        write would fail because the *complaint about* the write failed.
        Asserted by calling inside a try that fails the test on ANY
        exception, rather than by pytest.raises-style narrowing.
        """
        import fused_memory.services.memory_metadata_census as census

        caplog.set_level(logging.DEBUG, logger=_CENSUS_LOGGER)
        monkeypatch.setattr(census, 'HAS_ESCALATION', False)
        try:
            result = self._file(tmp_path)
        except Exception as exc:  # pragma: no cover — the contract being pinned
            pytest.fail(f'must never raise on the live write path, got {exc!r}')
        assert result is None
        assert caplog.records, 'a degraded no-op must still leave a trace'

    def test_returns_none_and_never_raises_on_queue_io_failure(self, tmp_path, monkeypatch):
        import fused_memory.services.memory_metadata_census as census

        class _Queue:
            def __init__(self, path):
                pass

            def get_by_task(self, task_id, status=None):
                return []

            def make_id(self, task_id):
                return 'esc-x-1'

            def submit(self, esc):
                raise OSError('disk on fire')

        monkeypatch.setattr(census, 'HAS_ESCALATION', True)
        monkeypatch.setattr(census, 'EscalationQueue', _Queue)

        try:
            result = self._file(tmp_path)
        except Exception as exc:  # pragma: no cover — the contract being pinned
            pytest.fail(f'queue I/O failure must not propagate, got {exc!r}')
        assert result is None

    def test_a_dedup_read_failure_falls_through_to_filing(self, tmp_path, monkeypatch):
        """Best-effort dedup: a read failure must not suppress the escalation.

        Mirrors the precedent — failing closed here would mean a broken
        queue read silently swallows the storm signal entirely.
        """
        import fused_memory.services.memory_metadata_census as census

        submitted = []

        class _Queue:
            def __init__(self, path):
                pass

            def get_by_task(self, task_id, status=None):
                raise OSError('cannot read queue')

            def make_id(self, task_id):
                return 'esc-x-1'

            def submit(self, esc):
                submitted.append(esc)
                return esc.id

        monkeypatch.setattr(census, 'HAS_ESCALATION', True)
        monkeypatch.setattr(census, 'EscalationQueue', _Queue)

        assert self._file(tmp_path) == 'esc-x-1'
        assert len(submitted) == 1
