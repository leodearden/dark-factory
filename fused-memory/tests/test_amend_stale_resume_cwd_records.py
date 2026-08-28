"""Tests for scripts/amend_stale_resume_cwd_records.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution -- mirrors the pattern in
test_tag_cgl_eta_rehome_scope.py / test_prune_recon_cycle_summaries.py.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).parent.parent / 'scripts' / 'amend_stale_resume_cwd_records.py'
)


def _load_module() -> types.ModuleType:
    """Load amend_stale_resume_cwd_records.py from its file path.

    The module is registered in sys.modules under its name so that
    reflection-based decorators (e.g. @dataclass) work correctly.
    """
    mod_name = 'amend_stale_resume_cwd_records'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # required for @dataclass __module__ lookup
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()

STALE_ID = '6403e96b-f1af-403a-9513-59f007ed6d39'
WARNING_ID = 'd007aa46-5800-455c-af3c-32d8fd8445b2'


class TestAmendTargetsContract:
    """The write payload itself -- the DATA this script would send.

    Deliberately a tight substring contract over the load-bearing FACTS the
    corpus needs, NOT a prose pin: nothing here asserts docstrings, comments,
    or the wording of the surrounding sentences, so the texts stay editable.
    """

    def test_two_targets_in_load_bearing_order(self):
        # Order is semantic, not cosmetic: d007aa46 will assert that 6403e96b
        # was corrected, so it must never be written first (see step-11/12).
        assert [t.memory_id for t in _mod.AMEND_TARGETS] == [STALE_ID, WARNING_ID]

    def test_write_attribution_constants_exist(self):
        assert isinstance(_mod.WRITE_SOURCE, str) and _mod.WRITE_SOURCE
        assert isinstance(_mod.WRITE_REASON, str) and _mod.WRITE_REASON
        assert isinstance(_mod.AMENDED_SENTINEL, str) and _mod.AMENDED_SENTINEL

    def test_sentinel_present_in_every_replacement_text(self):
        # The sentinel is what carries idempotency, so it must be in BOTH
        # replacement texts -- a target whose new content lacks it would be
        # rewritten on every re-run.
        for target in _mod.AMEND_TARGETS:
            assert _mod.AMENDED_SENTINEL in target.new_content

    def test_sentinel_absent_from_every_preimage(self):
        # If a pre-image already contained the sentinel the classifier could
        # never reach 'amend' -- the guard would skip the very record it is
        # meant to correct.
        for target in _mod.AMEND_TARGETS:
            assert _mod.AMENDED_SENTINEL not in target.expected_preimage

    def test_each_replacement_differs_from_its_own_preimage(self):
        for target in _mod.AMEND_TARGETS:
            assert target.new_content != target.expected_preimage

    def test_stale_record_replacement_carries_the_correcting_facts(self):
        new_content = _mod.AMEND_TARGETS[0].new_content
        # The real April incident and the commit that fixed it are PRESERVED
        # -- this is an amend, not a retraction.
        assert 'e001dd3746' in new_content
        # The contrary measurement that supersedes the inferred cause.
        assert '2.1.236' in new_content
        assert '2026-08-19' in new_content
        assert 'plans/session-resume-eligibility-seam-prd.md' in new_content
        # What is still genuinely open is stated rather than resolved by
        # assertion -- nobody measured the April-era CLI.
        assert 'UNDETERMINED' in new_content

    def test_hygiene_warning_replacement_retires_the_stale_status_clause(self):
        warning = _mod.AMEND_TARGETS[1]
        # The clause being retired is present in the pre-image ...
        assert 'STILL UNCORRECTED' in warning.expected_preimage
        # ... and gone from the replacement.
        assert 'STILL UNCORRECTED' not in warning.new_content

    def test_hygiene_warning_replacement_keeps_its_measured_evidence(self):
        # The record's value is its MEASUREMENTS; the amend retires one status
        # sentence, it does not rewrite the finding.
        new_content = _mod.AMEND_TARGETS[1].new_content
        assert '0.786' in new_content
        assert '0.692' in new_content
        assert 'e001dd3746' in new_content

    def test_preimages_are_nonempty_strings(self):
        for target in _mod.AMEND_TARGETS:
            assert isinstance(target.expected_preimage, str)
            assert target.expected_preimage.strip()


class TestAmendTargetShape:
    def test_target_is_frozen(self):
        target = _mod.AMEND_TARGETS[0]
        with pytest.raises(dataclasses.FrozenInstanceError):
            target.memory_id = 'mutated'  # type: ignore[misc]


class TestClassifyAmendTarget:
    """The pure classifier: what a live read means for one target.

    Inputs are ``get_memory_by_id``-shaped envelopes -- ``{'found': True,
    'content': ...}`` on a hit, ``{'found': False}`` on a genuine miss, and
    ``{'error', 'error_type'}`` with ``found`` ABSENT on a backend failure.
    That three-way shape is the tool's documented no-silent-fail contract, and
    the classifier must preserve the distinction rather than collapse it.
    """

    def test_exact_preimage_classifies_amend(self):
        target = _mod.AMEND_TARGETS[0]
        decision = _mod.classify_amend_target(
            target, {'found': True, 'content': target.expected_preimage},
        )
        assert decision['action'] == 'amend'
        assert decision['id'] == target.memory_id

    def test_already_amended_content_classifies_skip(self):
        target = _mod.AMEND_TARGETS[0]
        decision = _mod.classify_amend_target(
            target, {'found': True, 'content': target.new_content},
        )
        assert decision['action'] == 'skip:already_amended'

    def test_sentinel_is_checked_before_the_preimage_comparison(self):
        # A re-run reads back content that no longer equals the pre-image (it
        # equals the replacement). If the pre-image check ran first, every
        # re-run would report a mismatch REFUSAL and an operator would have to
        # decide whether the corpus had been tampered with. The sentinel test
        # must therefore win -- assert it with text that is neither the
        # pre-image nor the exact replacement, so only ordering can explain it.
        target = _mod.AMEND_TARGETS[0]
        drifted = f'prefix added later. {target.new_content} suffix added later.'
        assert drifted != target.expected_preimage
        assert drifted != target.new_content
        decision = _mod.classify_amend_target(target, {'found': True, 'content': drifted})
        assert decision['action'] == 'skip:already_amended'

    def test_unrecognised_content_classifies_preimage_mismatch(self):
        target = _mod.AMEND_TARGETS[0]
        decision = _mod.classify_amend_target(
            target, {'found': True, 'content': 'something else entirely'},
        )
        assert decision['action'] == 'refuse:preimage_mismatch'

    def test_genuine_miss_classifies_not_found(self):
        target = _mod.AMEND_TARGETS[0]
        decision = _mod.classify_amend_target(target, {'found': False})
        assert decision['action'] == 'refuse:not_found'

    def test_backend_error_classifies_read_error_not_not_found(self):
        # 'found' is ABSENT on an error envelope. Unknown is not absent: a
        # timed-out read must never be reported as a vanished record.
        target = _mod.AMEND_TARGETS[0]
        decision = _mod.classify_amend_target(
            target, {'error': 'read timed out', 'error_type': 'TimeoutError'},
        )
        assert decision['action'] == 'refuse:read_error'

    def test_read_error_and_not_found_are_distinct_actions(self):
        target = _mod.AMEND_TARGETS[0]
        missing = _mod.classify_amend_target(target, {'found': False})
        errored = _mod.classify_amend_target(
            target, {'error': 'boom', 'error_type': 'TimeoutError'},
        )
        assert missing['action'] != errored['action']

    def test_classifier_is_pure_over_both_targets(self):
        # Same input twice -> same answer, and no target is special-cased.
        for target in _mod.AMEND_TARGETS:
            fetched = {'found': True, 'content': target.expected_preimage}
            first = _mod.classify_amend_target(target, fetched)
            second = _mod.classify_amend_target(target, fetched)
            assert first == second == {'id': target.memory_id, 'action': 'amend'}
