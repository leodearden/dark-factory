"""Argument validation for `consolidate_memories` (task 3133).

Pure unit tests: no server, no service, no event loop. The helper under
test is the op's FIRST step and its only zero-write one — everything after
it either writes a canonical, patches a retained peer or deletes a victim.
That position is the whole point: an argument set that cannot be executed
safely has to be refused while refusing still costs nothing.

Two properties are load-bearing here and pinned as such:

* EVERY offender is named in one error, never the first one found. The op
  is irreversible, so a caller that fixes one id, re-runs, and trips the
  next one has re-run a destructive operation to learn a fact the first
  call already knew (the same no-short-circuit bar `validate_memory_metadata`
  holds).
* A non-empty `supersedes` REQUIRES `run_id`. It becomes the tombstone's
  `deleting_run_id` — the field that answers "which run killed this
  record?" — and an unattributable delete must be refused before the
  canonical is written, not discovered after it exists.
"""

import pytest

from fused_memory.server.consolidation import validate_consolidate_args
from fused_memory.topic_slug import TOPIC_SLUG_MAX_LEN

_A = '11111111-1111-4111-8111-111111111111'
_B = '22222222-2222-4222-8222-222222222222'
_C = '33333333-3333-4333-8333-333333333333'

_TOPIC = 'memory-consolidation'
_CONTENT = 'The consolidated single-claim canonical.'
_RUN = 'run-abc123'


def _call(**overrides):
    """Validate a valid arg set with *overrides* applied."""
    kwargs = {
        'canonical_content': _CONTENT,
        'topic': _TOPIC,
        'supersedes': [_A, _B],
        'retain': None,
        'run_id': _RUN,
    }
    kwargs.update(overrides)
    return validate_consolidate_args(**kwargs)


class TestValidArgs:
    def test_valid_args_pass_and_normalize(self):
        err, supersedes, retain = _call()

        assert err is None
        assert supersedes == [_A, _B]
        assert retain == []

    def test_legacy_scalar_supersedes_is_accepted_and_normalized(self):
        """PRD D2's read tolerance, inherited — never re-parsed here.

        81 live records carry a SCALAR `supersedes`; `normalize_supersedes`
        is the single home of that tolerance, so a caller passing the legacy
        shape gets a 1-element list rather than a rejection.
        """
        err, supersedes, retain = _call(supersedes=_A)

        assert err is None
        assert supersedes == [_A]

    def test_retain_only_call_is_valid(self):
        """The ratified Option C default arm: tag peers, delete nothing."""
        err, supersedes, retain = _call(supersedes=None, retain=[_A, _B])

        assert err is None
        assert supersedes == []
        assert retain == [_A, _B]


class TestCanonicalContent:
    @pytest.mark.parametrize('bad', ['', '   ', '\n\t '])
    def test_blank_content_is_refused(self, bad):
        err, _, _ = _call(canonical_content=bad)

        assert err is not None
        assert err['error_type'] == 'ValidationError'
        assert 'canonical_content' in err['error']

    @pytest.mark.parametrize('bad', [None, 42, ['text'], {'text': 'x'}])
    def test_non_string_content_is_refused(self, bad):
        err, _, _ = _call(canonical_content=bad)

        assert err is not None
        assert err['error_type'] == 'ValidationError'
        assert 'canonical_content' in err['error']


class TestTopic:
    @pytest.mark.parametrize(
        'bad',
        [
            'memory_consolidation',  # snake_case — 98 live values look like this
            'Memory-Consolidation',  # uppercase
            'memory-consolidation\n',  # trailing newline (`$` would let this pass)
            '-leading-hyphen',
            'trailing-hyphen-',
            'double--hyphen',
            '',
            None,
            7,
        ],
    )
    def test_malformed_topic_is_refused(self, bad):
        err, _, _ = _call(topic=bad)

        assert err is not None
        assert err['error_type'] == 'ValidationError'
        assert 'topic' in err['error']

    def test_over_length_topic_is_refused(self):
        err, _, _ = _call(topic='a' * (TOPIC_SLUG_MAX_LEN + 1))

        assert err is not None
        assert 'topic' in err['error']

    def test_topic_error_names_the_rule_and_its_home(self):
        """The caller has to be able to find the namespace, not guess it.

        `topic` is ONE namespace shared with `ProceduralTopicCluster.topic_id`
        (PRD D4); a message that only says "invalid" sends the caller off to
        re-derive a regex that already has a single home.
        """
        err, _, _ = _call(topic='memory_consolidation')
        assert err is not None

        text = err['error'] + err.get('hint', '')
        assert 'fused_memory.topic_slug' in text
        assert 'hyphen' in text.lower()


class TestIdShapes:
    def test_every_malformed_supersede_is_named_in_one_error(self):
        """No short-circuit: the op is irreversible.

        A caller that fixes one id and re-runs has re-run a destructive
        operation to learn something the first call already knew.
        """
        err, _, _ = _call(supersedes=[_A, '873889a1', 12345, _B])

        assert err is not None
        assert err['error_type'] == 'ValidationError'
        assert '873889a1' in err['error']
        assert '12345' in err['error']
        assert 'supersedes' in err['error']

    def test_every_malformed_retain_id_is_named_in_one_error(self):
        err, _, _ = _call(supersedes=None, retain=['84147843', None, _A])

        assert err is not None
        assert '84147843' in err['error']
        assert 'retain' in err['error']

    def test_both_arms_report_together(self):
        """One call, one complete answer — across arms too."""
        err, _, _ = _call(supersedes=['873889a1'], retain=['84147843'])

        assert err is not None
        assert '873889a1' in err['error']
        assert '84147843' in err['error']

    def test_id_error_carries_the_resolve_the_full_uuid_hint(self):
        err, _, _ = _call(supersedes=['873889a1'])
        assert err is not None

        assert 'get_memory_by_id' in err.get('hint', '')


class TestArmPresence:
    @pytest.mark.parametrize(
        ('supersedes', 'retain'),
        [(None, None), ([], []), (None, []), ([], None)],
    )
    def test_no_arm_at_all_is_refused(self, supersedes, retain):
        """An empty arm is an ABSENT arm.

        Mirrors `_update_memory_arm_presence_error`: without this, a caller
        whose supersedes list computed to `[]` gets a success envelope for a
        consolidation that folded nothing.
        """
        err, _, _ = _call(supersedes=supersedes, retain=retain, run_id=None)

        assert err is not None
        assert err['error_type'] == 'ValidationError'
        assert 'supersedes' in err['error']
        assert 'retain' in err['error']


class TestArmOverlap:
    def test_an_id_in_both_arms_is_refused_by_name(self):
        """Delete-and-retain is not a resolvable instruction.

        Picking either arm silently would make the op's own report wrong
        about what it did to that record.
        """
        err, _, _ = _call(supersedes=[_A, _B], retain=[_B, _C])

        assert err is not None
        assert err['error_type'] == 'ValidationError'
        assert _B in err['error']
        assert _A not in err['error']

    def test_every_overlapping_id_is_named(self):
        err, _, _ = _call(supersedes=[_A, _B], retain=[_A, _B])
        assert err is not None

        assert _A in err['error']
        assert _B in err['error']


class TestDeleteArmRequiresRunId:
    """Task Details item 6 — the tombstone precondition.

    `deleting_run_id` names the run that KILLED a record, distinct from the
    victim's own `metadata.run_id` naming the run that WROTE it. There is no
    ambient run id at the MCP boundary and both fallbacks are worse than
    refusing: `''` mints tombstones answering "who deleted this?" with
    silence, and a session id puts a non-run identifier in a field auditors
    read as a run id.
    """

    @pytest.mark.parametrize('missing', [None, '', '   '])
    def test_delete_arm_without_run_id_is_refused(self, missing):
        err, _, _ = _call(run_id=missing)

        assert err is not None
        assert err['error_type'] == 'ValidationError'
        assert 'run_id' in err['error']

    def test_the_refusal_explains_that_the_delete_would_be_unattributable(self):
        err, _, _ = _call(run_id=None)
        assert err is not None

        text = err['error'] + err.get('hint', '')
        assert 'deleting_run_id' in text
        assert 'tombstone' in text.lower()

    def test_retain_only_call_needs_no_run_id(self):
        """The retain arm never deletes, so nothing becomes unattributable."""
        err, supersedes, retain = _call(supersedes=None, retain=[_A], run_id=None)

        assert err is None
        assert supersedes == []
        assert retain == [_A]

    def test_the_run_id_check_is_refused_alongside_other_offenders(self):
        """It is a collected violation, not a separate early gate."""
        err, _, _ = _call(supersedes=['873889a1'], run_id=None)
        assert err is not None

        assert '873889a1' in err['error']
        assert 'run_id' in err['error']
