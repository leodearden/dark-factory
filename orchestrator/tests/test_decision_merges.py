"""Tests for decision_registry's refile merges: how a second filing folds into an existing DecisionRecord.

Covers the never-downgrade severity rank, merge_decision_enrichment (a
cross-queue refile fills only what is empty, and never forges a queue/id
pair) and merge_same_queue_refile (a same-queue refile takes the watcher's
fields verbatim). Both keep the cockpit-owned custody fields of the first
filing.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pytest
from _decision_fixtures import make_decision, two_queues

from orchestrator import decision_registry as dr


@pytest.mark.parametrize(
    ('a', 'b', 'expected'),
    [
        # Both directions of every adjacent pair: the helper is symmetric,
        # so argument order must never decide the answer.
        ('', 'info', 'info'),
        ('info', '', 'info'),
        ('info', 'blocking', 'blocking'),
        ('blocking', 'info', 'blocking'),
        ('blocking', 'critical', 'critical'),
        ('critical', 'blocking', 'critical'),
        ('critical', 'urgent', 'urgent'),
        ('urgent', 'critical', 'urgent'),
        # ...and the non-adjacent pair that matters most in practice: a
        # second watcher's 'info' must not pull a 'critical' record down.
        ('critical', 'info', 'critical'),
        ('info', 'critical', 'critical'),
    ],
)
def test_max_decision_severity_takes_the_higher_of_the_two(a: str, b: str, expected: str) -> None:
    """The never-downgrade primitive: '' < info < blocking < critical < urgent.

    This deliberately does NOT reuse escalation/src/escalation/queue.py's
    ``_SEVERITY_RANK``, which is ``{'info': 0, 'blocking': 1}`` only: there,
    'critical' and 'urgent' fall to the ``.get(x, 0)`` default and rank
    INFO-tier. Reusing it would make the never-downgrade helper itself
    perform the exact downgrade this task exists to prevent -- a second
    watcher's 'info' would tie-or-beat a first watcher's 'critical'.
    (Importing it is impossible anyway: this module is stdlib-only.)
    """
    assert dr._max_decision_severity(a, b) == expected


@pytest.mark.parametrize('value', ['', 'info', 'blocking', 'critical', 'urgent'])
def test_max_decision_severity_of_an_equal_pair_is_that_value(value: str) -> None:
    """A tie returns the value itself, so a re-file never churns the field."""
    assert dr._max_decision_severity(value, value) == value


@pytest.mark.parametrize(
    ('existing', 'incoming'),
    [
        ('critical', 'criticl'),  # typo'd incoming must not displace
        ('critical', 'CRITICAL'),  # case variant is NOT a known severity
        ('urgent', 'whatever'),
        ('info', 'nonsense'),
    ],
)
def test_max_decision_severity_unknown_never_displaces_a_recognised_one(
    existing: str,
    incoming: str,
) -> None:
    """An UNRECOGNISED severity ranks unknown-lowest and never raises.

    A typo'd or case-variant severity from a second watcher (escalation
    models.KNOWN_SEVERITIES is lowercase-only and rejects case variants
    outright) must not be able to downgrade a recognised one. It ranks
    alongside the unset sentinel rather than blowing up, because this helper
    sits on a watcher's filing path and must never crash the watch loop.
    """
    assert dr._max_decision_severity(existing, incoming) == existing


def test_max_decision_severity_two_unknowns_returns_existing() -> None:
    """Ambiguous both-unknown case resolves to ``existing``: never downgrade,
    never churn the file with an equally-meaningless replacement.
    """
    assert dr._max_decision_severity('mystery-a', 'mystery-b') == 'mystery-a'


def test_max_decision_severity_unknown_loses_to_a_recognised_incoming() -> None:
    """The other direction: an unknown EXISTING value does not block a real
    incoming severity from landing, since unknown ranks at the bottom.
    """
    assert dr._max_decision_severity('bogus', 'blocking') == 'blocking'


@pytest.mark.parametrize(
    ('name', 'kept', 'offered'),
    [
        ('text', 'Adopt the reify plan?', 'reify?'),
        ('task_id', '5914', '9999'),
        ('escalation_id', 'esc-5914-1', 'esc-other-7'),
        ('session_id', 'watcher-df-1', 'watcher-recon-2'),
        ('options', ['yes', 'no'], ['maybe']),
    ],
)
def test_merge_decision_enrichment_never_clobbers_a_non_empty_field(
    name: str,
    kept: object,
    offered: object,
) -> None:
    """MODE-2 rule: a second watcher ENRICHES, it never overwrites.

    Two watchers can surface the SAME underlying human gate through two
    different queues (the observed esc-5914-1 shape, task 3528 requirement
    (b)). The second one to file must not replace what the first wrote --
    the cockpit row would otherwise flip to whichever watcher happened to
    write last.
    """
    existing = make_decision(**{name: kept})
    incoming = make_decision(**{name: offered})

    merged = dr.merge_decision_enrichment(existing, incoming)

    assert getattr(merged, name) == kept


@pytest.mark.parametrize(
    ('name', 'empty', 'offered'),
    [
        ('text', '', 'reify?'),
        ('task_id', None, '5914'),
        ('escalation_id', None, 'esc-5914-1'),
        ('session_id', None, 'watcher-recon-2'),
        ('options', None, ['yes', 'no']),
    ],
)
def test_merge_decision_enrichment_fills_an_empty_field(
    name: str,
    empty: object,
    offered: object,
) -> None:
    """The other half of the rule -- and what makes this ENRICHMENT rather
    than a no-op: a field the first filer left empty/None IS filled from the
    second filer, so the collapsed row is strictly better informed than
    either watcher's view alone.
    """
    existing = make_decision(**{name: empty})
    incoming = make_decision(**{name: offered})

    merged = dr.merge_decision_enrichment(existing, incoming)

    assert getattr(merged, name) == offered


@pytest.mark.parametrize(
    ('name', 'kept', 'offered'),
    [
        ('filed_at', '2026-07-07T00:00:00+00:00', '2026-08-08T00:00:00+00:00'),
        ('state', 'open', 'answered'),
        ('manual_boost', 5, 0),
    ],
)
def test_merge_decision_enrichment_keeps_custody_fields_with_the_first_filer(
    name: str,
    kept: object,
    offered: object,
) -> None:
    """CUSTODY fields always stay with the EXISTING record.

    A second watcher must be able to neither reset an operator's cockpit
    boost (manual_boost is the C5 human's, not a watcher's), nor re-open or
    close the record (state transitions are update_decision_state's job),
    nor restamp when the gate was first surfaced (filed_at drives queue
    age). These are the fields where "last writer wins" would silently
    destroy work done by someone else.
    """
    existing = make_decision(**{name: kept})
    incoming = make_decision(**{name: offered})

    merged = dr.merge_decision_enrichment(existing, incoming)

    assert getattr(merged, name) == kept


@pytest.mark.parametrize(
    ('existing_sev', 'incoming_sev', 'expected'),
    [
        ('info', 'urgent', 'urgent'),  # upgrade lands
        ('critical', 'info', 'critical'),  # downgrade refused
        ('', 'blocking', 'blocking'),  # unset is enriched
        ('blocking', 'blocking', 'blocking'),  # tie is stable
    ],
)
def test_merge_decision_enrichment_never_downgrades_severity(
    existing_sev: str,
    incoming_sev: str,
    expected: str,
) -> None:
    """severity is _max_decision_severity(existing, incoming), both ways.

    The record must end up carrying the MOST urgent view any watcher has of
    the gate: an incoming 'urgent' upgrades an existing 'info', and an
    incoming 'info' leaves an existing 'critical' alone.
    """
    existing = make_decision(severity=existing_sev)
    incoming = make_decision(severity=incoming_sev)

    assert dr.merge_decision_enrichment(existing, incoming).severity == expected


def test_merge_decision_enrichment_keeps_id_and_project_from_existing() -> None:
    """``id`` is the JOIN KEY -- the whole reason these two records are being
    merged at all -- so it, and the project it is scoped to, come from the
    existing record and are never taken from the incoming one.
    """
    existing = make_decision(id='esc-5914-1', project='df')
    incoming = make_decision(id='esc-5914-1', project='other-project')

    merged = dr.merge_decision_enrichment(existing, incoming)

    assert merged.id == 'esc-5914-1'
    assert merged.project == 'df'


def test_merge_decision_enrichment_is_pure() -> None:
    """Neither argument may be mutated in place.

    The helper is deliberately side-effect-free so it stays trivially
    testable in isolation from the CLI verb, and so a caller holding the
    pre-merge record (e.g. for a log line naming what changed) still sees
    what it read.
    """
    existing = make_decision(id='esc-5914-1', text='first?', severity='info')
    incoming = make_decision(id='esc-5914-1', text='second?', severity='urgent')
    before_existing = existing.to_dict()
    before_incoming = incoming.to_dict()

    merged = dr.merge_decision_enrichment(existing, incoming)

    assert merged is not existing
    assert merged is not incoming
    assert existing.to_dict() == before_existing
    assert incoming.to_dict() == before_incoming


def test_merge_decision_enrichment_touches_only_the_documented_fields() -> None:
    """Whole-record comparison, following the
    test_set_decision_escalations_dir_preserves_every_other_field idiom: a
    merge that differs from `existing` ONLY in severity must be expressible
    as a single dataclasses.replace, so a field silently drifting in is a
    hard failure rather than an unasserted gap.
    """
    existing = make_decision(id='esc-5914-1', severity='info')
    # Same in every field EXCEPT severity, which outranks.
    incoming = dataclasses.replace(existing, severity='urgent')

    merged = dr.merge_decision_enrichment(existing, incoming)

    assert merged == dataclasses.replace(existing, severity='urgent')


def test_merge_same_queue_refile_holds_the_cockpit_owned_fields() -> None:
    """The three COCKPIT-owned fields survive a same-queue re-file at any state.

    Sibling of test_merge_decision_enrichment_keeps_custody_fields_with_the_
    first_filer: the custody set is the same one, because custody does not
    depend on which queue re-filed. ``filed_at`` is queue age (it drives the
    cockpit's ordering, and a watcher restart is not news about it),
    ``manual_boost`` is the operator's C5 field (set_manual_boost's), and
    ``state`` is the operator's / reaper's disposition
    (update_decision_state's) -- so a watcher restart must not restamp the
    age, reset the boost, or RE-OPEN a row the human already dropped.

    ``id``/``project`` are asserted unchanged as the caller's PRECONDITION,
    not as a field this helper forces: unlike merge_decision_enrichment (which
    rebuilds from *existing* and so pins them structurally), this helper
    rebuilds from *incoming*, and is only ever reached from
    _run_write_decision after ``existing.project == project`` and with the id
    as the on-disk file key. Stating it here is what makes a future caller
    that widens those preconditions fail loudly rather than silently reassign
    a live row to another project.
    """
    existing = make_decision(
        id='esc-5914-1',
        project='df',
        filed_at='2026-07-07T00:00:00+00:00',
        state=dr.DecisionState.DROPPED,
        manual_boost=7,
    )
    incoming = make_decision(
        id='esc-5914-1',
        project='df',
        filed_at='2026-08-19T00:00:00+00:00',
        state=dr.DecisionState.OPEN,
        manual_boost=0,
    )

    merged = dr.merge_same_queue_refile(existing, incoming)

    assert merged.filed_at == '2026-07-07T00:00:00+00:00'
    assert merged.state == dr.DecisionState.DROPPED
    assert merged.manual_boost == 7
    assert merged.id == 'esc-5914-1'
    assert merged.project == 'df'


def test_merge_same_queue_refile_takes_the_watcher_owned_fields_verbatim() -> None:
    """The WATCHER-owned half lands exactly as filed -- downgrades and empties too.

    This is the half that keeps the watcher the sole authority on its own
    escalation, and the one place this helper deliberately differs from
    merge_decision_enrichment: there a second watcher may only FILL fields
    the first left empty and may never downgrade severity
    (_max_decision_severity), because two watchers are two views of one gate.
    Here there is only ONE view -- the same watcher's, revised -- so freezing
    the first values would strand stale prose and a stale severity in the
    cockpit queue forever.
    """
    queue = '/queues/orch'
    existing = make_decision(
        text='the original prose',
        severity='critical',
        task_id='5914',
        session_id='watcher-df-1',
        escalation_id='esc-5914-1',
        options=['yes', 'no'],
        escalations_dir=queue,
    )
    incoming = make_decision(
        text='the rephrased prose',
        severity='info',
        task_id=None,
        session_id=None,
        escalation_id='esc-5914-2',
        options=None,
        escalations_dir=queue,
    )

    merged = dr.merge_same_queue_refile(existing, incoming)

    assert merged.text == 'the rephrased prose'
    assert merged.severity == 'info'  # a DOWNGRADE lands (unlike enrichment)
    assert merged.task_id is None  # ...and so does an EMPTYING
    assert merged.session_id is None
    assert merged.escalation_id == 'esc-5914-2'
    assert merged.options is None
    assert merged.escalations_dir == queue


@pytest.mark.parametrize(
    'state',
    [
        dr.DecisionState.ANSWERED,
        dr.DecisionState.DROPPED,
        'deferred-by-hand',
    ],
)
def test_merge_same_queue_refile_preserves_an_unrecognized_state_verbatim(
    state: str,
) -> None:
    """``state`` is copied as an opaque str, never coerced through DecisionState.

    Mirrors DecisionState's own documented additive-safe contract:
    DecisionRecord.state is a plain ``str`` with NO from_dict coercion, so an
    unrecognized value must round-trip rather than raise -- or, here, be
    silently reset to 'open'. A disposition some future writer adds is
    therefore held back by this helper for free, instead of needing this
    module to be taught about it first.
    """
    existing = make_decision(state=state)
    incoming = make_decision(state=dr.DecisionState.OPEN)

    assert dr.merge_same_queue_refile(existing, incoming).state == state


def test_merge_same_queue_refile_is_pure() -> None:
    """Neither argument may be mutated in place.

    Mirrors test_merge_decision_enrichment_is_pure. The helper is
    deliberately side-effect-free -- including of LOGGING, which stays in the
    CLI verb at the policy boundary -- so it is trivially testable in
    isolation, and so a caller holding the pre-merge record (e.g. to name the
    held-back state in its divergence warning) still sees what it read.
    """
    existing = make_decision(id='esc-5914-1', text='first?', state='dropped')
    incoming = make_decision(id='esc-5914-1', text='second?', state='open')
    before_existing = existing.to_dict()
    before_incoming = incoming.to_dict()

    merged = dr.merge_same_queue_refile(existing, incoming)

    assert merged is not existing
    assert merged is not incoming
    assert existing.to_dict() == before_existing
    assert incoming.to_dict() == before_incoming


def test_same_queue_refile_and_enrichment_agree_on_the_custody_field_set() -> None:
    """THE ANTI-DIVERGENCE INVARIANT: one custody set, two merge branches.

    After task 3872 the write-decision upsert has two merge arms --
    merge_decision_enrichment (cross-queue) and merge_same_queue_refile
    (same-queue) -- and the load-bearing shared rule is WHICH fields are
    cockpit-owned. They differ only in how the WATCHER-owned half is taken
    (fill-if-empty + severity-max vs. verbatim). Add a fourth custody field
    to one and forget the other and the regression is silent: every
    field-specific test above still passes.

    Arranged so the two halves are separable. *existing* leaves every
    fill-if-empty field empty (text/task_id/session_id/record_slug/
    escalation_id/options/severity), so BOTH helpers take the watcher-owned half from *incoming*
    and the two runs can only differ on custody. The queue axis is
    neutralized (equal normalized ``escalations_dir`` on both), which is also
    the precondition of the same-queue arm.

    ``id``/``project``/``escalations_dir`` land in the set because both
    records agree on them, not because either helper had to choose -- that
    agreement is exactly the caller's precondition for this arm, so the set
    is spelled out in full rather than filtered down to the three fields that
    genuinely differ.
    """
    queue = '/queues/orch'
    existing = make_decision(
        text='',
        task_id=None,
        session_id=None,
        record_slug='',
        escalation_id=None,
        options=None,
        severity='',
        filed_at='2026-07-07T00:00:00+00:00',
        state=dr.DecisionState.DROPPED,
        manual_boost=7,
        escalations_dir=queue,
        closing_evidence='the operator dropped it: superseded by esc-5914-2',
        closed_at='2026-08-01T00:00:00+00:00',
        audit_outcomes=(dr.AuditOutcome(
            verdict=dr.AuditVerdict.OK, note='', audited_at='2026-08-02T00:00:00+00:00',
            policy_class='landing_provenance',
        ),),
    )
    incoming = make_decision(
        text="the watcher's current view",
        task_id='5914',
        session_id='watcher-df-1',
        escalation_id='esc-5914-1',
        options=['yes', 'no'],
        severity='critical',
        filed_at='2026-08-19T00:00:00+00:00',
        state=dr.DecisionState.OPEN,
        manual_boost=0,
        escalations_dir=queue,
    )

    def _kept_from_existing(merged: dr.DecisionRecord) -> set[str]:
        return {
            f.name
            for f in dataclasses.fields(dr.DecisionRecord)
            if getattr(merged, f.name) == getattr(existing, f.name)
        }

    enriched = _kept_from_existing(dr.merge_decision_enrichment(existing, incoming))
    refiled = _kept_from_existing(dr.merge_same_queue_refile(existing, incoming))

    assert enriched == refiled
    assert refiled == {
        'id',
        'project',
        'filed_at',
        'state',
        'manual_boost',
        'closing_evidence',
        'closed_at',
        'audit_outcomes',
        'escalations_dir',
    }


def test_merge_decision_enrichment_fills_an_empty_queue_stamp(tmp_path: Path) -> None:
    """The case that matters most right after task 3640's back-fill.

    An existing record with ``escalations_dir=''`` is the legacy/unstamped
    population 3640 had to back-fill. When a watcher files against that id
    carrying a REAL queue, the merge takes it: that is precisely the
    "append/update its own queue's escalations_dir entry" the task asks for,
    and it turns a project-only-scoped record into a properly queue-scoped
    one.
    """
    orch, _recon = two_queues(tmp_path)
    existing = make_decision(escalations_dir='')
    incoming = make_decision(escalations_dir=dr.normalize_escalations_dir(orch))

    merged = dr.merge_decision_enrichment(existing, incoming)

    assert merged.escalations_dir == dr.normalize_escalations_dir(orch)


def test_merge_decision_enrichment_keeps_a_conflicting_queue_stamp_and_warns(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two DIFFERENT real queues: FIRST-writer-wins, loudly.

    The field is a scalar (task 3640 hard-committed that, adding
    UNKNOWN_QUEUE as a third scalar state), so a cross-queue MODE-2 collapse
    genuinely cannot record both queues. Keeping the EXISTING stamp makes
    the outcome DETERMINISTIC (the first filer's queue) rather than racy
    (whichever watcher wrote last), which is strictly better -- but it is
    still information loss, so it must never be silent: a WARNING names the
    decision id and BOTH queue paths.
    """
    orch, recon = two_queues(tmp_path)
    existing = make_decision(id='esc-5914-1', escalations_dir=dr.normalize_escalations_dir(orch))
    incoming = make_decision(id='esc-5914-1', escalations_dir=dr.normalize_escalations_dir(recon))

    with caplog.at_level(logging.WARNING):
        merged = dr.merge_decision_enrichment(existing, incoming)

    assert merged.escalations_dir == dr.normalize_escalations_dir(orch)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings
    assert any(
        'esc-5914-1' in r.getMessage()
        and str(dr.normalize_escalations_dir(orch)) in r.getMessage()
        and str(dr.normalize_escalations_dir(recon)) in r.getMessage()
        for r in warnings
    )


def test_merge_decision_enrichment_does_not_let_the_unknown_sentinel_displace_a_real_queue(
    tmp_path: Path,
) -> None:
    """UNKNOWN_QUEUE loses to a real queue in BOTH directions.

    Existing ``<unknown>`` + incoming real queue -> take the real one: that
    is a genuine enrichment, since it makes an otherwise permanently
    unreapable record reapable again.

    Existing real queue + incoming ``<unknown>`` -> keep the real one:
    the reverse would make a reapable record unreapable, which no reaper
    could ever undo.

    Belt-and-braces given the CLI verb already refuses the sentinel at its
    boundary -- but merge_decision_enrichment is a public pure helper and
    must be correct on its own terms.
    """
    orch, _recon = two_queues(tmp_path)
    real = dr.normalize_escalations_dir(orch)

    from_unknown = dr.merge_decision_enrichment(
        make_decision(escalations_dir=dr.UNKNOWN_QUEUE),
        make_decision(escalations_dir=real),
    )
    to_unknown = dr.merge_decision_enrichment(
        make_decision(escalations_dir=real),
        make_decision(escalations_dir=dr.UNKNOWN_QUEUE),
    )

    assert from_unknown.escalations_dir == real
    assert to_unknown.escalations_dir == real


def test_merge_decision_enrichment_does_not_let_the_unknown_sentinel_displace_the_empty_queue(
    tmp_path: Path,
) -> None:
    """The third direction: '' must not be DOWNGRADED to ``<unknown>`` either.

    The two states are not interchangeable "not a real queue" spellings, and
    ranking them by truthiness (the obvious `if incoming_queue:` reading) gets
    it backwards. A record stamped '' is the LEGACY population and still
    closes, via the reaper's project-only fallback. A record stamped
    ``<unknown>`` is refused BY NAME by every reaper, so it can only ever be
    closed by a human. Adopting the sentinel therefore takes a reapable record
    and makes it permanently unreapable -- the mirror image of the '' -> real
    enrichment above, run backwards, and no reaper can undo it.

    ``<unknown>`` means "investigated, could not determine" and is legitimate
    ONLY from the back-fill that did the investigating (task 3640,
    set_decision_escalations_dir). A merge has investigated nothing, so it
    never gets to acquire the sentinel from the other filer.

    Unreachable through the CLI verb today, which refuses the sentinel on
    input -- but merge_decision_enrichment is public and must be correct on
    its own terms. Note the 144-case exhaustive test does NOT cover this: its
    predicate asserts the (queue, id) pair is vouched for, and ('', id) ->
    ('<unknown>', id) can still be perfectly vouched for while losing
    reapability.
    """
    to_unknown = dr.merge_decision_enrichment(
        make_decision(escalations_dir='', escalation_id=None),
        make_decision(escalations_dir=dr.UNKNOWN_QUEUE, escalation_id='esc-1'),
    )
    from_unknown = dr.merge_decision_enrichment(
        make_decision(escalations_dir=dr.UNKNOWN_QUEUE, escalation_id=None),
        make_decision(escalations_dir='', escalation_id='esc-1'),
    )

    assert to_unknown.escalations_dir == ''
    # ...and the id does not travel without its queue, per the pair rule.
    assert to_unknown.escalation_id is None
    # The reverse was already right and stays right: '' is not an "upgrade"
    # over an investigated ``<unknown>``, it is just a different unknown.
    assert from_unknown.escalations_dir == dr.UNKNOWN_QUEUE
    assert from_unknown.escalation_id is None


def test_merge_decision_enrichment_treats_equivalent_queue_spellings_as_one(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Both sides are NORMALIZED before comparing, never raw-string-compared.

    A dotted / trailing-slash spelling of the SAME queue must not read as a
    cross-queue conflict -- that is the fail-open mistake
    normalize_escalations_dir exists to prevent, and the reaper's axis-2
    guard already normalizes both sides for exactly this reason. No warning
    is emitted, because nothing conflicts.
    """
    orch, _recon = two_queues(tmp_path)
    existing = make_decision(escalations_dir=dr.normalize_escalations_dir(orch))
    incoming = make_decision(escalations_dir=f'{orch}/./')

    with caplog.at_level(logging.WARNING):
        merged = dr.merge_decision_enrichment(existing, incoming)

    assert merged.escalations_dir == dr.normalize_escalations_dir(orch)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_merge_decision_enrichment_never_forges_a_queue_id_pair(tmp_path: Path) -> None:
    """``escalations_dir`` and ``escalation_id`` must move as ONE PAIR.

    Merging them by INDEPENDENT rules lets the survivor carry a (queue, id)
    combination that existed on NEITHER filer's record: here the queue would
    be adopted from *incoming* while the id stays *existing*'s, yielding
    ``(orch, 'esc-3036-1')`` -- a pair `orch` never vouched for.

    That is fail-CLOSED, the one direction the reaper must never take.
    ``_run_reap_decisions._status`` JOINS on exactly this pair: axis 2
    compares the stamp, then ``read_escalation_status(escalations_dir,
    decision.escalation_id)`` resolves the id INSIDE that queue -- and
    ``esc-<taskid>-<n>`` ids are unique only WITHIN a queue, so the forged
    pair resolves against an unrelated escalation that merely shares the id.
    That is verbatim the incident _run_reap_decisions' own docstring records:
    an unrelated RESOLVED orchestrator escalation silently closing a still
    PENDING recon blocking gate, invisible in the cockpit for ~7 days.

    The correct outcome is to keep *existing*'s own pair: the record is left
    exactly as (un)reapable as it already was -- a visible cockpit row, which
    is the fail-OPEN direction _run_reap_decisions' asymmetry-of-harm
    paragraph prescribes.
    """
    orch, _recon = two_queues(tmp_path)
    existing = make_decision(escalations_dir='', escalation_id='esc-3036-1')
    incoming = make_decision(
        escalations_dir=dr.normalize_escalations_dir(orch),
        escalation_id='esc-9999-1',
    )

    merged = dr.merge_decision_enrichment(existing, incoming)

    assert (merged.escalations_dir, merged.escalation_id) != (
        dr.normalize_escalations_dir(orch),
        'esc-3036-1',
    )
    assert merged.escalations_dir == ''
    assert merged.escalation_id == 'esc-3036-1'


def test_merge_decision_enrichment_warns_when_it_refuses_an_ambiguous_queue_upgrade(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Refusing the upgrade is right, but it must never be SILENT.

    Same shape as ..._never_forges_a_queue_id_pair: the record keeps its
    undetermined queue and therefore stays unreapable. That is deliberate,
    not an oversight, so it is logged for the same never-silent-information
    -loss reason the conflicting-stamp WARNING already exists -- naming the
    decision and the incoming queue it declined to adopt, so an operator can
    see which record is being held and why.
    """
    orch, _recon = two_queues(tmp_path)
    existing = make_decision(id='esc-3036-1', escalations_dir='', escalation_id='esc-3036-1')
    incoming = make_decision(
        id='esc-3036-1',
        escalations_dir=dr.normalize_escalations_dir(orch),
        escalation_id='esc-9999-1',
    )

    with caplog.at_level(logging.WARNING):
        merged = dr.merge_decision_enrichment(existing, incoming)

    assert merged.escalations_dir == ''
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        'esc-3036-1' in r.getMessage()
        and str(dr.normalize_escalations_dir(orch)) in r.getMessage()
        for r in warnings
    )


def test_merge_decision_enrichment_does_not_borrow_an_escalation_id_across_a_queue_conflict(
    tmp_path: Path,
) -> None:
    """The OTHER direction of the same defect: the queue stays, the id must not move.

    First-writer-wins keeps `orch`, but filling the empty ``escalation_id``
    from the recon filer hands `orch`'s stamp an id from RECON's namespace.
    The damage is worse than a mismatch: today ``reap_answered_decisions``
    SKIPS a decision with no escalation_id outright (``if not
    escalation_id: continue``), so this record is inert and safe. Borrowing
    an id turns a record the reaper never touched into one it will resolve
    against the wrong namespace -- manufacturing the fail-CLOSED join out of
    nothing.
    """
    orch, recon = two_queues(tmp_path)
    existing = make_decision(
        escalations_dir=dr.normalize_escalations_dir(orch),
        escalation_id=None,
    )
    incoming = make_decision(
        escalations_dir=dr.normalize_escalations_dir(recon),
        escalation_id='esc-9999-1',
    )

    merged = dr.merge_decision_enrichment(existing, incoming)

    assert merged.escalations_dir == dr.normalize_escalations_dir(orch)
    assert merged.escalation_id is None


@pytest.mark.parametrize('incoming_id', [None, 'esc-a', 'esc-b'])
@pytest.mark.parametrize('incoming_queue_key', ['empty', 'unknown', 'orch', 'recon'])
@pytest.mark.parametrize('existing_id', [None, 'esc-a', 'esc-b'])
@pytest.mark.parametrize('existing_queue_key', ['empty', 'unknown', 'orch', 'recon'])
def test_merge_decision_enrichment_pair_is_always_vouched_for(
    tmp_path: Path,
    existing_queue_key: str,
    existing_id: str | None,
    incoming_queue_key: str,
    incoming_id: str | None,
) -> None:
    """THE INVARIANT, over the whole 12x12 input space.

    The two focused cases above are instances; this is the general rule they
    are instances of, asserted exhaustively so no future merge rule can
    reintroduce a forged pair through a combination nobody thought to test.

    Predicate: the merged ``(escalations_dir, escalation_id)`` is either
    id-less (inert -- the reaper skips it) or is a pair some INPUT record
    actually vouched for. Never a synthesis of one filer's queue with the
    other filer's id, because ``read_escalation_status`` is a JOIN on that
    pair and ids are unique only within a queue.

    Cheap to run exhaustively: merge_decision_enrichment is pure, so this is
    144 in-memory calls.
    """
    orch, recon = two_queues(tmp_path)
    queues = {
        'empty': '',
        'unknown': dr.UNKNOWN_QUEUE,
        'orch': dr.normalize_escalations_dir(orch),
        'recon': dr.normalize_escalations_dir(recon),
    }
    existing = make_decision(
        escalations_dir=queues[existing_queue_key], escalation_id=existing_id
    )
    incoming = make_decision(
        escalations_dir=queues[incoming_queue_key], escalation_id=incoming_id
    )

    merged = dr.merge_decision_enrichment(existing, incoming)

    merged_queue = dr.normalize_escalations_dir(merged.escalations_dir)
    assert merged.escalation_id is None or any(
        dr.normalize_escalations_dir(side.escalations_dir) == merged_queue
        and side.escalation_id == merged.escalation_id
        for side in (existing, incoming)
    )
