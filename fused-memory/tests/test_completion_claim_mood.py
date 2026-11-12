"""Unit tests for the completion-claim gate's non-assertive-mood filter (task 6677).

A completion marker only makes a claim when the writer ASSERTS it. Quoted
text, modal forms, conditional/interrogative/imperative scopes, present-tense
temporal clauses and attributive participles mention a task's landing without
asserting it, and the reify escalations behind this task
(esc-unverified-claim-5471-*, -7407-3, -8246-5, -6020-2, -5023-2) were each
tagged for exactly that confusion.

Only the public :func:`extract_completion_claims` is driven here.
"""

from __future__ import annotations

import pytest

from fused_memory.services.completion_claim_gate import (
    CompletionClaim,
    extract_completion_claims,
)

_KNOWN = frozenset({'reify', 'dark_factory', 'solar_challenge', 'solar_challenge_platform'})

# Verbatim from reify memory 4e6fb418. The quotation spans the '...' clause
# boundary, so the clause holding 'shipped' has no closing quote of its own.
_ESC_5471_4_MEMORY = (
    '- esc-unverified-claim-5471-3: "#5471 owns the shipped Bool-balance ... '
    'examples" while 5471 was pending.'
)

# Verbatim from solar_challenge_platform memory 04a1706d.
_ESC_536_1_MEMORY = (
    'solar_challenge_platform test convention (task 536, on branch task/536 until merged): '
    'give a test SetpointCommand an expiry measured from the clock that judges it, via '
    'tests/control/command_expiry.py::optimiser_expiry(issued_at), which is issued_at + 2 x '
    'DEFAULT_OPTIMISER_INTERVAL_SECONDS (the 3600 s life optimiser/translate.py::to_setpoint '
    "gives), e.g. optimiser_expiry(clock()) for a ControlPlane's clock or "
    'optimiser_expiry(HARNESS_CLOCK_START) for a conformance harness. Never use a far-future '
    'constant such as 2030-01-01: a ControlPlane on the real clock rejects it from that date. '
    'To test I5 eviction without sleeping out the hour, give the plane a SteppableClock and '
    'step it one REFRESH_INTERVAL past the expiry, as the task 536 cases in '
    'tests/control/test_modbus_binding.py and tests/control/test_p3_sim_conformance.py do.'
)

# Verbatim from solar_challenge memory 0d38599e.
_ESC_1053_1_MEMORY = (
    "solar_challenge correction (task 1053 implementer, 2026-11-02): task 894's decision-15 "
    'step split, merged to main at c6a4195, is what deleted load._try_richardsonpy_profile; '
    'task 1053 deletes and renames nothing. On branch task/1053 (holds once merged), '
    "load._simulated_household_kw's Elexon fallback warning reads \"the household model "
    'failed; falling back to Elexon deterministic profile: <type>: <message>", pinned as '
    'literals in two tests in tests/unit/test_load.py and in the forked child of '
    'tests/unit/test_load_reproducibility.py, while docs/prds/rng-hand-off.md C7 '
    'deliberately keeps quoting the old "richardsonpy stochastic generation failed" text as '
    'a historical record.'
)


def _extract(text: str, default_project_id: str = 'reify') -> list[CompletionClaim]:
    return extract_completion_claims(
        text,
        default_project_id=default_project_id,
        known_project_ids=_KNOWN,
    )


def _triples(
    text: str, default_project_id: str = 'reify',
) -> list[tuple[str, str, str]]:
    return [
        (claim.kind, claim.subject, claim.ref)
        for claim in _extract(text, default_project_id)
    ]


class TestQuotationsAreMentions:

    @pytest.mark.parametrize(
        'text',
        [
            pytest.param('the memory said "task 5422 has landed" earlier', id='straight'),
            pytest.param('the memory said “task 5422 has landed” earlier', id='curly'),
            pytest.param(_ESC_5471_4_MEMORY, id='esc-unverified-claim-5471-4'),
            pytest.param(
                'which had asserted reify task 5638 "was re-filed into '
                "dark_factory's task tree as ticket tkt_0RRRC5AASJ9Z630VP4PCN9H376\"",
                id='quoted-filing',
            ),
        ],
    )
    def test_a_quoted_marker_is_not_a_claim(self, text):
        assert _triples(text) == []

    @pytest.mark.parametrize(
        'text',
        [
            'evidence="task 3016 merged 1a925afbf4"',
            'status:"task 3016 merged"',
            pytest.param('status: "task 3016 merged"', id='yaml-spaced'),
            pytest.param('{"status": "task 3016 merged"}', id='json'),
        ],
    )
    def test_a_key_value_literal_is_still_read(self, text):
        assert _triples(text) == [('applied_work', 'task', '3016')]

    def test_an_unpaired_quote_does_not_cross_a_newline(self):
        assert _triples('note "x\ntask 5422 has landed') == [
            ('applied_work', 'task', '5422'),
        ]

    @pytest.mark.parametrize(
        'text',
        [
            pytest.param(
                'the 12" pipe is fine; task 5422 landed; he said "ok"', id='semicolon',
            ),
            pytest.param(
                'the 12" pipe is fine. Task 5422 landed. He said "ok"', id='sentence-end',
            ),
            pytest.param(
                'the 12" pipe is fine? task 5422 landed! he said "ok"', id='question-mark',
            ),
        ],
    )
    def test_a_stray_quote_does_not_swallow_a_later_clause(self, text):
        assert _triples(text) == [('applied_work', 'task', '5422')]

    def test_terminal_punctuation_inside_a_quotation_keeps_it_a_quotation(self):
        assert _triples('the memory said "task 5422 has landed." earlier') == []

    def test_backticks_do_not_delimit_a_quotation(self):
        assert _triples('task 5422 was left `cancelled`') == [
            ('disposition', 'task', '5422'),
        ]

    def test_an_apostrophe_does_not_open_a_quotation(self):
        assert _triples("task 5422's fix has been applied") == [
            ('applied_work', 'task', '5422'),
        ]


class TestModalFormsAreNotClaims:

    @pytest.mark.parametrize(
        'text',
        [
            'task 5422 must have landed by now',
            'task 5422 should have landed yesterday',
            'task 5422 would have been merged without the conflict',
            'task 5422 could have shipped earlier',
            'task 5422 may have been applied twice',
            'task 5422 must be merged first',
            'task 5422 will have landed by Friday',
            "task 5422 won't have shipped",
            'task 5422 can be closed as duplicate',
            pytest.param("task 5422 can't be merged yet", id='cant'),
            pytest.param('task 5422 can’t have landed', id='cant-curly'),
            pytest.param('task 5422 won’t be merged', id='wont-curly'),
            pytest.param('task 5422 shouldn’t be filed as tkt_0RRRC5AASJ9Z630VP4PCN9H376',
                         id='shouldnt-curly-filing'),
            pytest.param("task 5422 must've landed by now", id='must-ve'),
            pytest.param('task 5422 would’ve been merged', id='would-ve-curly'),
            pytest.param('task 5422 must have been fully merged', id='adverb-fully'),
            pytest.param('task 5422 should have been successfully merged',
                         id='adverb-successfully'),
            pytest.param('task 5422 is supposed to be fully merged', id='intention-adverb'),
        ],
    )
    def test_a_modal_governed_marker_is_not_a_claim(self, text):
        assert _triples(text) == []

    @pytest.mark.parametrize(
        'text',
        [
            pytest.param(
                'task 5422 has landed and will be reviewed', id='modal-after-barrier',
            ),
            pytest.param(
                'task 5422 landed; it would have been faster otherwise',
                id='modal-in-next-clause',
            ),
        ],
    )
    def test_a_modal_elsewhere_leaves_the_claim(self, text):
        assert _triples(text) == [('applied_work', 'task', '5422')]


class TestNonAssertiveScopes:

    @pytest.mark.parametrize(
        'text',
        [
            pytest.param(
                "check #8246's status for whether it has landed)",
                id='esc-unverified-claim-8246-5',
            ),
            pytest.param(
                "Merge-stall triage (check #8246's status for whether it has landed on main)",
                id='imperative-after-paren',
            ),
            pytest.param(
                'if #6020 has landed, ROUTE THROUGH IT rather than re-implementing '
                'the Applied case',
                id='esc-unverified-claim-6020-2',
            ),
            pytest.param(
                "the task text says to add an event only 'if #5455 has landed'",
                id='if-inside-single-quotes',
            ),
            pytest.param('unless task 5422 is merged first', id='unless'),
            pytest.param('in case task 5422 has shipped', id='in-case'),
            pytest.param('verify task 5422 was merged', id='imperative-verify'),
            pytest.param('Next: confirm #5422 has shipped', id='imperative-after-colon'),
            pytest.param('- check that task 5422 landed', id='imperative-bullet'),
        ],
    )
    def test_a_non_veridical_or_imperative_scope_is_not_a_claim(self, text):
        assert _triples(text) == []

    @pytest.mark.parametrize(
        'text',
        [
            pytest.param(
                'Doing it BEFORE #7407 stamps the PRD SHIPPED is the correct order',
                id='esc-unverified-claim-7407-3',
            ),
            pytest.param(
                'Task 1371 can be unblocked once Task 1374 is cancelled', id='once',
            ),
            pytest.param(
                'should NOT be marked superseded until task 791 is merged', id='until',
            ),
            pytest.param(
                'a #5023 cite would rot silently the day #5023 is cancelled',
                id='esc-unverified-claim-5023-2',
            ),
            pytest.param('when task 5422 has landed, re-run the sweep', id='when'),
            pytest.param(
                'On branch task/1053 (holds once it is merged), the warning reads X.',
                id='once-it-is-merged',
            ),
            pytest.param(
                'On branch task/1053 (holds once it has merged), the warning reads X.',
                id='once-it-has-merged',
            ),
            pytest.param(
                'Test convention (task 536, on branch task/536 until it is merged): '
                'use optimiser_expiry.',
                id='until-it-is-merged',
            ),
        ],
    )
    def test_a_present_tense_temporal_scope_is_not_a_claim(self, text):
        assert _triples(text) == []

    @pytest.mark.parametrize(
        ('text', 'project'),
        [
            pytest.param(
                'Test convention (task 536, on branch task/536 until merged): '
                'use optimiser_expiry.',
                'solar_challenge_platform',
                id='esc-unverified-claim-536-1',
            ),
            pytest.param(
                _ESC_536_1_MEMORY, 'solar_challenge_platform',
                id='esc-unverified-claim-536-1-record',
            ),
            pytest.param(
                'On branch task/1053 (holds once merged), the warning reads X.',
                'solar_challenge',
                id='esc-unverified-claim-1053-1',
            ),
            pytest.param(
                'On branch task/1053 (true when merged), the warning reads X.',
                'solar_challenge',
                id='when-merged',
            ),
            pytest.param(
                'On branch task/1053 (true until merged), the warning reads X.',
                'solar_challenge',
                id='until-merged',
            ),
            pytest.param(
                'Task 758 adds tests/_schema_state.py::index_shapes for this '
                '(holds once merged)',
                'solar_challenge_platform',
                id='esc-unverified-claim-758-2',
            ),
            pytest.param(
                'Coordination for task 728 (telemetry_hh compression), from task '
                "1378's plan, branch task/1378 (holds once merged)",
                'solar_challenge_platform',
                id='esc-unverified-claim-1378-2',
            ),
            pytest.param('task 5422 stays on its branch till merged', 'reify', id='till'),
            pytest.param(
                'task 5422 stays on its branch as soon as merged', 'reify', id='as-soon-as',
            ),
            pytest.param('task 5422 is rebased before merged', 'reify', id='before'),
            pytest.param('task 5422 reopens the day merged', 'reify', id='the-day'),
            pytest.param(
                'the guard on branch task/5422 applies whenever landed', 'reify',
                id='whenever',
            ),
            pytest.param(
                'the guard on branch task/5422 holds after merged', 'reify', id='after',
            ),
            pytest.param('task 5422 (holds once fully merged)', 'reify', id='adverb'),
            pytest.param('task 5422 stays open until cancelled', 'reify', id='disposition'),
            pytest.param('task 5422 runs once queued', 'reify', id='filing'),
            pytest.param('The freeze on task 5422 is until merged', 'reify', id='copular-until'),
            pytest.param(
                'the branch task/5422 guard is when merged', 'reify', id='copular-when',
            ),
            pytest.param(
                'the branch task/5422 guard was after merged', 'reify', id='copular-after',
            ),
        ],
    )
    def test_a_verbless_temporal_clause_is_not_a_claim(self, text, project):
        assert _triples(text, project) == []

    def test_the_1053_record_does_not_claim_its_branch_task(self):
        # Not '== []': the record also states task 894's true landing, which the gate misses.
        claimed = {(subject, ref) for _, subject, ref in _triples(_ESC_1053_1_MEMORY, 'solar_challenge')}
        assert ('task', '1053') not in claimed

    @pytest.mark.parametrize(
        ('text', 'ref'),
        [
            pytest.param('After task 5 landed, the suite went green.', '5', id='after-past'),
            pytest.param('When task 5 was merged, the queue halted.', '5', id='when-past'),
            pytest.param('once task 5 landed we re-ran the sweep', '5', id='once-past'),
            pytest.param(
                'recorded before task 3445 landed tests/x.py (+1 test)', '3445',
                id='before-past',
            ),
            pytest.param(
                'Task 3815 (if-then-else returning a Solid) shipped via geometry.rs', '3815',
                id='hyphen-attached-cue',
            ),
            pytest.param(
                'if task 6 is pending, note that task 5 has landed', '5',
                id='scope-ends-at-comma',
            ),
            pytest.param(
                'Task 5 has landed. Check the dashboard.', '5',
                id='imperative-in-next-clause',
            ),
            pytest.param('Task 536 merged in abc1234.', '536', id='declarative-merged-in'),
            pytest.param('#5422 landed in abc123', '5422', id='declarative-hash-landed-in'),
            pytest.param(
                'task 5422 was merged and holds once merged', '5422',
                id='verbless-clause-in-the-same-clause',
            ),
            pytest.param(
                'Task 5 was once merged into main but reverted', '5',
                id='adverbial-once-after-was',
            ),
            pytest.param(
                'task 5422 had once landed before the revert', '5422',
                id='adverbial-once-after-had',
            ),
            pytest.param('task 5422 has once been merged', '5422', id='adverbial-once-after-has'),
        ],
    )
    def test_an_asserted_landing_outside_any_scope_is_still_a_claim(self, text, ref):
        assert _triples(text) == [('applied_work', 'task', ref)]

    def test_a_participle_after_a_temporal_preposition_still_binds_forward(self):
        text = (
            'They are named saved_fleet_presets and saved_fleet_preset_named, after '
            "cancelled task 359's branch (commit 64dfb4a), and take α's Database in "
            'place of a path'
        )
        assert _triples(text, 'solar_challenge') == [('disposition', 'task', '359')]


_ESC_5471_1 = (
    '(3) PRD2 ε #5471 owns the shipped Bool-balance and stock-size examples, '
    'so a new Bool exemplar here would duplicate it'
)


class TestAttributiveMarkersBindForwardOnly:

    @pytest.mark.parametrize(
        'text',
        [
            pytest.param(_ESC_5471_1, id='esc-unverified-claim-5471-1'),
            pytest.param(
                'task 5578 is merge-deferred with a queued carrier branch',
                id='queued-carrier',
            ),
        ],
    )
    def test_an_attributive_marker_does_not_claim_an_earlier_ref(self, text):
        assert _triples(text) == []

    def test_an_attributive_marker_does_not_add_applied_work(self):
        text = (
            'task 5317 was EXPANDED to also own wiring the Manifold route into '
            'the shipped `reify build` CLI'
        )
        assert [t for t in _triples(text) if t[0] == 'applied_work'] == []

    def test_an_attributive_marker_leaves_the_filing_claim(self):
        text = 'dark_factory task 3846 was filed to give recon a landed-on-main check'
        assert _triples(text) == [('filing_dispatch', 'task', '3846')]

    def test_an_attributive_marker_still_binds_forward(self):
        assert _triples('the merged commit abc1234 fixed it') == [
            ('applied_work', 'commit', 'abc1234'),
        ]

    @pytest.mark.parametrize(
        ('text', 'ref'),
        [
            pytest.param(
                "Task 4105's landed fix deviates from its task text", '4105',
                id='possessive-is-not-attributive',
            ),
            pytest.param(
                "task 5422's de-flake fix has been applied", '5422',
                id='possessive-copula',
            ),
        ],
    )
    def test_a_possessive_lead_still_claims_its_owner(self, text, ref):
        assert _triples(text) == [('applied_work', 'task', ref)]
