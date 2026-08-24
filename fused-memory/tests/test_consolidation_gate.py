"""Tests for the consolidation-gate brief + closure predicate (task 3112).

Covers the two defects the task owns: the gate-filing instruction that
prescribed no end-state shape (Defect 1 — :func:`render_end_state_brief` /
:func:`render_consolidation_gate_section`), and the absence of any mechanical
refusal to close a gate task over a malformed cluster (Defect 2 —
:func:`evaluate_closure`).

Assertions are pinned to runtime return values (the verdict dataclass, the
builder's returned dict) and to stable load-bearing substrings within the
rendered prose — NOT verbatim prompt-text equality — mirroring the
``test_recon_self_model.py`` / ``test_predicate_contradiction.py`` convention.
"""

from __future__ import annotations

import dataclasses

import pytest

from fused_memory import memory_metadata
from fused_memory.reconciliation import consolidation_gate
from fused_memory.reconciliation.consolidation_gate import (
    GATE_METADATA_KEY,
    ClosureVerdict,
    evaluate_closure,
    render_end_state_brief,
)


class TestRenderEndStateBrief:
    """Defect 1's payload: the end-state shape a filed gate must carry.

    Load-bearing-token assertions only — the prose is free to change, the
    shape it prescribes is not.
    """

    def test_returns_non_empty_str(self):
        brief = render_end_state_brief()
        assert isinstance(brief, str)
        assert brief.strip()

    def test_mandates_the_option_c_shape(self):
        """N short single-claim peers sharing one topic, exactly one canonical."""
        brief = render_end_state_brief()
        assert 'metadata.topic' in brief
        assert 'metadata.canonical' in brief
        # The canonical must itself be SHORT — an index/summary claim, not a
        # concatenation of the cluster.  This is the property PRD §3 measured
        # the inversion on, so the brief has to say it in so many words.
        assert 'short' in brief.lower()
        assert 'index' in brief.lower()

    def test_retires_the_appendix_end_state_by_name(self):
        """The '1 canonical + 1 appendix' absorbing target is explicitly dropped.

        Naming ``appendix`` is what stops a reader mistaking WHICH target is
        being retired — an unnamed retirement is indistinguishable from prose.
        """
        brief = render_end_state_brief()
        assert 'appendix' in brief.lower()
        assert 'retired' in brief.lower()

    def test_points_at_the_landed_op_and_its_ratified_retain_arm(self):
        """The brief and the executable op cannot prescribe different end states."""
        brief = render_end_state_brief()
        assert 'consolidate_memories' in brief
        assert 'retain' in brief

    def test_public_surface_is_exported(self):
        assert 'render_end_state_brief' in consolidation_gate.__all__
        assert 'GATE_METADATA_KEY' in consolidation_gate.__all__

    def test_gate_key_is_composed_from_the_shared_experimental_prefix(self):
        """Composed, never re-spelled — the CONTESTED_METADATA_KEY precedent."""
        assert GATE_METADATA_KEY.startswith(memory_metadata.EXPERIMENTAL_KEY_PREFIX)
        assert GATE_METADATA_KEY != memory_metadata.EXPERIMENTAL_KEY_PREFIX


# --------------------------------------------------------------------------- #
# Closure-predicate fixtures.  Payload shape mirrors what
# ``MemoryService.get_memories_by_metadata`` returns: {'id', 'created_at',
# 'metadata'} (see services/topic_anchor.py::select_canonical_payload, which
# documents the same shape).
# --------------------------------------------------------------------------- #

_TOPIC = 'recon-consolidation-gate-demo'


def _member(mid, *, canonical=None, topic=_TOPIC, supersedes=None, **extra_meta):
    """One scrolled topic member."""
    meta = dict(extra_meta)
    if topic is not None:
        meta['topic'] = topic
    if canonical is not None:
        meta['canonical'] = canonical
    if supersedes is not None:
        meta['supersedes'] = supersedes
    return {'id': mid, 'created_at': '2026-08-24T00:00:00+00:00', 'metadata': meta}


def _uuid(n):
    return f'00000000-0000-4000-8000-{n:012d}'


def _closure(members, gate_block=None, **kwargs):
    """Call ``evaluate_closure`` with a complete, available scroll by default.

    The production signature deliberately requires the completeness arguments
    (a caller must not be able to forget to disclose truncation); this helper
    supplies them so the membership tests stay about membership.
    """
    kwargs.setdefault('scroll_total', len(members))
    kwargs.setdefault('scroll_truncated', False)
    kwargs.setdefault('scroll_available', True)
    return evaluate_closure(
        gate_block if gate_block is not None else {'topic': _TOPIC},
        members=members,
        **kwargs,
    )


def _codes(verdict):
    return [r['code'] for r in verdict.reasons]


def _well_formed_cluster(n):
    """The canonical Option-C end state: N peers, exactly one canonical."""
    members = [_member(_uuid(1), canonical=True)]
    members += [_member(_uuid(i)) for i in range(2, n + 1)]
    return members


class TestClosureMembership:
    """Fix (1) — the membership test, and the reason the whole task is correct.

    Under PRD §3 Option C the end state IS N same-topic peers under one
    canonical, so a live same-topic peer is a legitimate member and can never
    be a refusal on membership grounds.  The prior plan's predicate flagged
    every such peer, which made every CORRECTLY executed consolidation
    permanently uncloseable; these tests go red if that is reintroduced.
    """

    @pytest.mark.parametrize('n', [1, 2, 3, 12])
    def test_well_formed_option_c_cluster_closes(self, n):
        verdict = _closure(_well_formed_cluster(n))
        assert verdict.closed is True
        assert verdict.exit_code == 0
        assert list(verdict.reasons) == []
        assert verdict.topic == _TOPIC

    def test_peer_count_alone_never_refuses(self):
        """A surviving peer that is neither canonical nor superseded is FINE."""
        peer = _uuid(7)
        members = [
            _member(_uuid(1), canonical=True, supersedes=[_uuid(99)]),
            _member(peer),
        ]
        verdict = _closure(members)
        assert verdict.closed is True
        # The peer must not be named anywhere in the verdict — not as a defect,
        # not as a warning.  This is the exact mutual exclusion that made the
        # prior plan's predicate unshippable.
        assert peer not in repr(verdict.reasons)

    def test_zero_canonicals_refuses(self):
        verdict = _closure([_member(_uuid(1)), _member(_uuid(2))])
        assert verdict.closed is False
        assert verdict.exit_code != 0
        assert 'no_canonical' in _codes(verdict)

    def test_multiple_canonicals_refuses_naming_both(self):
        """3198's per-topic uniqueness ships warn-mode-first, so this is
        REACHABLE through ordinary writes, not theoretical."""
        members = [
            _member(_uuid(1), canonical=True),
            _member(_uuid(2), canonical=True),
            _member(_uuid(3)),
        ]
        verdict = _closure(members)
        assert verdict.closed is False
        assert 'multiple_canonicals' in _codes(verdict)
        named = [r for r in verdict.reasons if r['code'] == 'multiple_canonicals']
        assert set(named[0]['ids']) == {_uuid(1), _uuid(2)}

    def test_canonical_is_strict_bool_not_truthy(self):
        """``meta.get('canonical') is True`` — an int 1 is NOT canonical.

        Mirrors services/topic_anchor.py::select_canonical_payload's identity
        check, so the closure predicate, the write-side uniqueness rule and the
        read-side anchor cannot disagree about what 'canonical' means.
        """
        verdict = _closure([_member(_uuid(1), canonical=1), _member(_uuid(2))])
        assert verdict.closed is False
        assert 'no_canonical' in _codes(verdict)

    def test_truthy_canonical_does_not_join_the_strict_one(self):
        """One strict True plus one truthy-but-not-True is ONE canonical."""
        members = [_member(_uuid(1), canonical=True), _member(_uuid(2), canonical=1)]
        verdict = _closure(members)
        assert verdict.closed is True


class TestClosureVerdictContract:
    def test_verdict_is_a_frozen_dataclass(self):
        verdict = _closure(_well_formed_cluster(2))
        assert dataclasses.is_dataclass(verdict)
        assert dataclasses.fields(ClosureVerdict)
        with pytest.raises(dataclasses.FrozenInstanceError):
            verdict.closed = False

    def test_closed_and_exit_code_can_never_disagree(self):
        closed = _closure(_well_formed_cluster(3))
        assert closed.closed is True and closed.exit_code == 0
        refused = _closure([_member(_uuid(1))])
        assert refused.closed is False and refused.exit_code != 0

    def test_message_is_non_empty_on_both_arms(self):
        assert _closure(_well_formed_cluster(2)).message.strip()
        assert _closure([_member(_uuid(1))]).message.strip()
