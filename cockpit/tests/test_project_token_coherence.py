"""The cross-leg invariant task 3812 exists to protect: the cockpit's SCORER
key and its PICKER name are the same string.

Five separate places produce or consume a project token -- registry_reader's
two readers, load_priorities' weights table, priority.score()'s lookup, and
weight_editor's picker -- and the bug this task fixes is not in any one of
them: it is what happens when two of them disagree. A weight an operator sets
on the name the picker offered then keys a bucket the scorer never looks up,
and NOTHING reports it. The failure is silent by construction, which is why
it needs a test that spans the legs rather than one per leg.

Each leg has its own unit tests (test_registry_reader.py, test_priority.py,
test_weight_editor.py). This file pins only the property that emerges from
their composition, over ONE real on-disk fleet root whose session and
decision rows are spelled DIFFERENTLY from each other and from the weights
file -- the realistic case, since all three are written by different
producers at different times.
"""

from __future__ import annotations

from datetime import UTC, datetime

from orchestrator import session_registry as sr

_NOW = datetime(2026, 7, 7, tzinfo=UTC)

# Three DIFFERENT raw spellings of the one project, one per producer:
# the spawn path writes the session, a C8 watcher writes the decision, and
# a human hand-edits priorities.yaml.
_SESSION_SPELLING = 'dark-factory'
_DECISION_SPELLING = 'df'
_WEIGHTS_SPELLING = 'dark_factory'

_CANONICAL = 'dark_factory'


def _seed_fleet(tmp_path):
    """Write one awaiting-input session and one OPEN decision, spelled apart."""
    sr.write_record(
        sr.SessionRecord(
            session_slug='unblock-df-2085-4242',
            status=sr.Status.AWAITING_INPUT,
            title='unblock:df#2085 slug',
            role='unblock',
            project=_SESSION_SPELLING,
            task_id='2085',
            start_ts='2026-07-07T00:00:00+00:00',
            question=sr.Question(text='Which port?', asked_at='2026-07-07T00:00:00+00:00'),
        ),
        root=tmp_path,
    )
    assert sr.write_decision(
        sr.DecisionRecord(
            id='dec-1',
            project=_DECISION_SPELLING,
            text='Which port?',
            filed_at='2026-07-07T00:00:00+00:00',
        ),
        root=tmp_path,
    )


def _write_weights(tmp_path, weight):
    path = tmp_path / 'priorities.yaml'
    path.write_text(f'project_weights:\n  {_WEIGHTS_SPELLING}: {weight}\n')
    return path


class TestOperatorWeightAppliesToBothRowKinds:
    def test_a_single_weight_raises_the_session_row_and_the_decision_row(self, tmp_path):
        """The whole point, end to end: ONE weights entry, spelled its own
        way, actually applies to BOTH queue rows even though each row's
        record is spelled differently on disk."""
        from cockpit.panes.decision_queue import order_queue
        from cockpit.priority import load_priorities
        from cockpit.registry_reader import scan_decisions, scan_sessions

        _seed_fleet(tmp_path)
        records = scan_sessions(tmp_path)
        decisions = scan_decisions(tmp_path)

        baseline = load_priorities(_write_weights(tmp_path, 1.0))
        weighted = load_priorities(_write_weights(tmp_path, 9.0))
        # sanity: the two differ ONLY in the project weight, so any score
        # change below is attributable to it and nothing else.
        assert baseline.defaults == weighted.defaults
        assert baseline.project_weights == {_CANONICAL: 1.0}
        assert weighted.project_weights == {_CANONICAL: 9.0}

        before = {i.key: i for i in order_queue(decisions, records, baseline, _NOW)}
        after = {i.key: i for i in order_queue(decisions, records, weighted, _NOW)}

        assert set(before) == {'decision:dec-1', 'session:unblock-df-2085-4242'}
        for key in before:
            assert after[key].score > before[key].score, (
                f'{key} did not respond to the project weight -- '
                f'its project token and the weights key have drifted apart'
            )

    def test_both_rows_carry_the_canonical_token(self, tmp_path):
        """The mechanism behind the test above: both row kinds reach the
        scorer under ONE token, so one weights key can reach both."""
        from cockpit.panes.decision_queue import order_queue
        from cockpit.priority import load_priorities
        from cockpit.registry_reader import scan_decisions, scan_sessions

        _seed_fleet(tmp_path)

        items = order_queue(
            scan_decisions(tmp_path),
            scan_sessions(tmp_path),
            load_priorities(_write_weights(tmp_path, 2.0)),
            _NOW,
        )

        assert len(items) == 2
        assert {i.project for i in items} == {_CANONICAL}


class TestPickerAndScorerKeyAgree:
    def test_the_picker_offers_exactly_the_name_the_weights_table_is_keyed_by(self, tmp_path):
        """The durable guard: no future change can move the scorer key and
        the picker apart again without failing here.

        known_projects unions all three producers -- scanned sessions,
        scanned decisions, and the weights table's own keys -- so if ANY of
        them stopped folding, this returns two names for one project and an
        operator gets offered a second, dead weight field.
        """
        from cockpit.panes.weight_editor import known_projects
        from cockpit.priority import load_priorities
        from cockpit.registry_reader import scan_decisions, scan_sessions

        _seed_fleet(tmp_path)
        priorities = load_priorities(_write_weights(tmp_path, 3.0))

        offered = known_projects(
            scan_sessions(tmp_path),
            scan_decisions(tmp_path),
            priorities.project_weights,
        )

        assert offered == [_CANONICAL]
        # ...and the offered name IS the key the scorer looks up.
        assert list(priorities.project_weights) == offered
