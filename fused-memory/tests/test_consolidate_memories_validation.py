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

from fused_memory.server.consolidation import (
    ProposalLimits,
    validate_consolidate_args,
)
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


class TestIntraArmDuplicates:
    """A repeated id is refused BY NAME, naming both slots.

    Refused rather than de-duplicated for the same reason the overlap check
    above refuses rather than picking an arm: silently rewriting the caller's
    set would make the op's own report describe a request nobody made.

    The load-bearing consequence of tolerating one is not the double delete
    call — it is that `tombstones_written` and `tombstones_expected` would
    BOTH count the repeat while the recon ledger's five-part identity
    (project, kind, memory_id, '', '') collapses it to a single row. The pair
    the envelope advertises as its audit-trail proof would then overstate the
    ledger: an INFERRED count, in the op whose whole deliverable is
    corroborated ones.
    """

    def test_a_repeated_supersede_is_refused_naming_both_slots(self):
        """The VALUE alone is not actionable — both renders are identical, so
        a caller editing the list needs to know which slot to delete."""
        err, _, _ = _call(supersedes=[_A, _B, _A])

        assert err is not None
        assert err['error_type'] == 'ValidationError'
        assert 'supersedes[2]' in err['error']
        assert 'supersedes[0]' in err['error']
        assert err['hint']

    def test_a_repeated_retain_id_is_refused_naming_both_slots(self):
        """Both arms, symmetrically: a repeat in `retain` would tag one peer
        twice and count it twice in `retained`."""
        err, _, _ = _call(supersedes=None, retain=[_A, _A], run_id=None)

        assert err is not None
        assert 'retain[1]' in err['error']
        assert 'retain[0]' in err['error']

    def test_every_repeat_is_named_not_just_the_first(self):
        """The no-short-circuit bar this module opens with. The op is
        irreversible, so a caller must never have to re-run a destructive
        call to learn the offender the first call already knew about."""
        err, _, _ = _call(supersedes=[_A, _A, _B, _B])
        assert err is not None

        assert 'supersedes[1]' in err['error']
        assert 'supersedes[3]' in err['error']

    def test_a_repeat_in_each_arm_reports_together(self):
        err, _, _ = _call(supersedes=[_A, _A], retain=[_B, _B])
        assert err is not None

        assert 'supersedes[1]' in err['error']
        assert 'retain[1]' in err['error']

    def test_values_that_merely_HASH_alike_are_not_a_repeat(self):
        """`True == 1` and `hash(True) == hash(1)` in Python, as do `1` and
        `1.0`, so a value-keyed dict would call these a repeat — naming two
        members that render DIFFERENTLY, which contradicts the whole reason
        this check reports by index. Neither is a valid id, so the call is
        refused either way; what must not happen is an UNTRUE problem line,
        because that is the one thing a caller cannot act on."""
        err, _, _ = _call(supersedes=[1, True])

        assert err is not None
        assert 'repeats' not in err['error']
        # ...but they ARE both named as malformed, which is the true report.
        assert 'supersedes[0]' in err['error']
        assert 'supersedes[1]' in err['error']

    def test_an_unhashable_member_does_not_raise(self):
        """The validator's entire job is to refuse WITHOUT raising: a raise
        is flattened by `@mcp_tool_errors` and loses the hint."""
        err, _, _ = _call(supersedes=[{'a': 1}, {'a': 1}, ['x'], ['x']])

        assert err is not None
        assert err['error_type'] == 'ValidationError'

    def test_distinct_ids_are_untouched_by_the_check(self):
        """The discriminator: a clean call must not acquire a duplicate
        problem, and its arms must come back exactly as supplied."""
        err, supersedes, retain = _call(supersedes=[_A, _B], retain=[_C])

        assert err is None
        assert supersedes == [_A, _B]
        assert retain == [_C]


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


_SHORT = 'Stage one keeps reproposing the identical cluster'          # 7 words
_OK = 'Stage one keeps reproposing the identical unaddressed cluster'  # 8 words
_LIMITS = ProposalLimits(claim_max_chars=200, member_min=2, member_max=20)


def _ids(n: int) -> list[str]:
    """*n* distinct well-formed UUIDs, deterministic so failures are readable."""
    return [f'{i:08x}-0000-4000-8000-000000000000' for i in range(n)]


def _propose(**overrides):
    """Validate a well-formed PROPOSAL arg set with *overrides* applied.

    The proposal shape carries no `canonical_content` (task gamma's tool has
    not written one yet — it is proposing that the cluster be consolidated at
    all) and no delete arm, so `supersedes` and `run_id` stay absent.
    """
    kwargs = {
        'canonical_content': None,
        'topic': _TOPIC,
        'supersedes': None,
        'retain': _ids(5),
        'run_id': None,
        'claim': _OK,
        'limits': _LIMITS,
    }
    kwargs.update(overrides)
    return validate_consolidate_args(**kwargs)


class TestOpArmIsUnchanged:
    """The five existing parameters and the op arm's behaviour are untouched.

    `limits=None` selects the op shape, which is what `server/tools.py`'s
    `consolidate_memories` passes by omission — so the tool needs no edit and
    stays out of this task's file set.
    """

    def test_a_valid_op_call_still_passes_unchanged(self):
        assert _call() == (None, [_A, _B], [])

    def test_a_refused_op_call_is_byte_identical_to_today(self):
        """Explicit `limits=None` must be indistinguishable from omitting it."""
        assert _call(topic='memory_consolidation') == validate_consolidate_args(
            canonical_content=_CONTENT,
            topic='memory_consolidation',
            supersedes=[_A, _B],
            retain=None,
            run_id=_RUN,
            claim=None,
            limits=None,
        )

    def test_a_claim_without_limits_is_refused_by_name(self):
        """Fails CLOSED on a mis-wired caller rather than silently ignoring the
        claim — a silently dropped claim is a silently skipped cap, which is
        the whole reason the caps exist."""
        err, _, _ = _call(claim=_OK)

        assert err is not None
        assert err['error_type'] == 'ValidationError'
        assert 'claim' in err['error']
        assert 'limits' in err['error']


class TestProposalArm:
    """`limits is not None` selects the proposal shape (PRD C1).

    Checked at the EMIT boundary, so the LLM that wrote a mis-shaped claim can
    fix it in-turn rather than having a downstream executor discover it.
    """

    def test_a_well_formed_proposal_passes(self):
        err, supersedes, retain = _propose()

        assert err is None, err
        assert supersedes == []
        assert retain == _ids(5)

    def test_canonical_content_is_not_required(self):
        """The op arm's own requirement must not leak into this one: a
        proposal has no canonical text yet, by construction."""
        err, _, _ = _propose(canonical_content=None)

        assert err is None, err

    def test_a_missing_claim_is_refused(self):
        err, _, _ = _propose(claim=None)

        assert err is not None
        assert 'claim' in err['error']

    def test_a_malformed_topic_is_still_refused(self):
        """The topic rule is shared by both arms — one namespace, one check."""
        err, _, _ = _propose(topic='memory_consolidation')

        assert err is not None
        assert 'topic' in err['error']

    # --- claim shape: one executing test per rule ---

    def test_a_multi_line_claim_is_refused(self):
        claim = 'Stage one keeps reproposing\nthe identical unaddressed cluster'
        err, _, _ = _propose(claim=claim)

        assert err is not None
        assert 'claim' in err['error']
        assert 'single line' in err['error']

    def test_a_claim_over_the_cap_is_refused(self):
        claim = _OK + ' ' + 'z' * (_LIMITS.claim_max_chars - len(_OK))
        assert len(claim) == _LIMITS.claim_max_chars + 1

        err, _, _ = _propose(claim=claim)

        assert err is not None
        assert str(_LIMITS.claim_max_chars) in err['error']

    def test_a_claim_exactly_at_the_cap_is_accepted(self):
        """The bound is INCLUSIVE — an off-by-one here silently costs the
        caller a character of the only text that reaches the canonical."""
        claim = _OK + ' ' + 'z' * (_LIMITS.claim_max_chars - len(_OK) - 1)
        assert len(claim) == _LIMITS.claim_max_chars

        err, _, _ = _propose(claim=claim)

        assert err is None, err

    def test_a_bracketed_claim_is_refused(self):
        """The B6 shape, and what makes a correction-banner body unusable as a
        claim: a banner opens with `[CORRECTION ...]`, so a claim lifted from
        one would carry the banner into the canonical's first paragraph."""
        claim = '[CORRECTION 2026-09-08] Stage one keeps reproposing the cluster'
        err, _, _ = _propose(claim=claim)

        assert err is not None
        assert 'claim' in err['error']

    @pytest.mark.parametrize(
        'claim',
        [
            'INDEX of the dashboard js test substrate cluster records',
            'CANONICAL: memory consolidation ratchet grew once per pass here',
        ],
    )
    def test_a_label_shaped_claim_is_refused(self, claim):
        """A claim is an ASSERTION, not a heading. A label-shaped opening reads
        as a title in the canonical's first paragraph, which is the position
        PRD §2 measured the retrieval property out of."""
        err, _, _ = _propose(claim=claim)

        assert err is not None
        assert 'claim' in err['error']

    @pytest.mark.parametrize(
        'claim',
        [
            'Indexing the cluster is cheaper than reading every duplicate record',
            'Canonicalisation of the topic happened before the gate was ever filed',
        ],
    )
    def test_an_ordinary_claim_beginning_with_those_words_is_accepted(self, claim):
        """The `\\b` and the case-sensitivity are the point: the rule refuses the
        LABEL `INDEX —`, not every sentence that starts with the same letters."""
        err, _, _ = _propose(claim=claim)

        assert err is None, err

    def test_a_claim_that_only_echoes_the_slug_is_refused(self):
        """A claim built entirely from the topic's own words and stopwords
        asserts nothing the slug did not already say, so the canonical it
        would open carries no claim at all."""
        claim = (
            'The memory consolidation is a consolidation of the memory for all '
            'of these'
        )
        err, _, _ = _propose(claim=claim)

        assert err is not None
        assert 'claim' in err['error']

    def test_a_claim_reusing_slug_words_with_real_content_is_accepted(self):
        """Reusing the topic's words is normal and must not be punished — only
        saying NOTHING ELSE is refused."""
        claim = (
            'The memory consolidation ratchet grew by one canonical per '
            'reconciliation pass until gate 3200'
        )
        err, _, _ = _propose(claim=claim)

        assert err is None, err

    def test_a_seven_word_claim_is_refused_and_eight_is_accepted(self):
        """The boundary, both sides, so an off-by-one cannot pass unnoticed."""
        short_err, _, _ = _propose(claim=_SHORT)
        ok_err, _, _ = _propose(claim=_OK)

        assert short_err is not None
        assert ok_err is None, ok_err

    # --- member-count range ---

    @pytest.mark.parametrize('count', [1, 21])
    def test_a_member_count_outside_the_range_is_refused(self, count):
        err, _, _ = _propose(retain=_ids(count))

        assert err is not None
        assert str(count) in err['error']
        assert 'retain' in err['error']

    @pytest.mark.parametrize('count', [2, 20])
    def test_the_member_count_bounds_are_inclusive(self, count):
        err, _, _ = _propose(retain=_ids(count))

        assert err is None, err

    def test_the_limits_are_read_from_the_argument_not_a_default(self):
        """ProposalLimits carries NO defaults: the numbers have exactly one
        home (ConsolidationAutoConfig) and a default here would be a second."""
        err, _, _ = _propose(retain=_ids(5), limits=ProposalLimits(200, 6, 20))

        assert err is not None
        assert '5' in err['error']

    # --- the no-short-circuit bar ---

    def test_every_offender_is_named_in_one_error(self):
        """PRD C1's reason for checking at the emit boundary: the LLM fixes its
        shape IN-TURN, which it can only do if one refusal names every fault."""
        claim = 'INDEX\n' + 'x' * 300
        err, _, _ = _propose(claim=claim, retain=_ids(1))

        assert err is not None
        assert 'single line' in err['error'], err['error']
        assert str(_LIMITS.claim_max_chars) in err['error'], err['error']
        assert 'INDEX' in err['error'], err['error']
        assert '8' in err['error'], err['error']
        assert 'retain' in err['error'], err['error']

    def test_the_hint_names_the_shape_rules(self):
        """PRD D3: the emitting stage needs something actionable, not just a
        list of what it got wrong."""
        err, _, _ = _propose(claim=_SHORT)

        assert err is not None
        assert err['hint']
        assert 'claim' in err['hint']
