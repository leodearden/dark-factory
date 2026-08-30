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
from unittest.mock import AsyncMock

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
    """The target contract: what this script now PINS, and what it must not carry.

    The correction these targets describe LANDED on 2026-08-30 through a
    Stage-1 memory consolidation, not through this script -- so the targets are
    VERIFY-ONLY and there is no write payload left to assert. What is pinned
    instead is the live post-correction text, tightly: a handful of
    load-bearing facts, never the wording of the surrounding sentences.
    """

    def test_two_targets_in_load_bearing_order(self):
        # Order is semantic, not cosmetic: d007aa46's text asserts that
        # 6403e96b was corrected, so it must never be written first. The
        # ordering guard is retained even though neither target writes today.
        assert [t.memory_id for t in _mod.AMEND_TARGETS] == [STALE_ID, WARNING_ID]

    def test_sentinel_is_the_landed_correction_marker(self):
        # Re-pointed from the never-applied 'amended in place by task 4610' to
        # the substring the LANDED texts actually carry. This one constant is
        # what makes both targets classify skip:already_amended -- reusing the
        # idempotency path rather than adding a parallel branch.
        assert _mod.AMENDED_SENTINEL == 'Stage-1 memory consolidation, task 4610'

    def test_sentinel_present_in_every_preimage(self):
        # The inverse of what this suite asserted before the correction landed.
        # Both records now carry the marker, so a healthy read of either is a
        # skip -- which is exactly why the run is a zero-write exit-0 no-op.
        for target in _mod.AMEND_TARGETS:
            assert _mod.AMENDED_SENTINEL in target.expected_preimage

    def test_every_target_is_verify_only(self):
        # No replacement text and no metadata payload: there is nothing this
        # script could write at either target, by construction rather than by
        # a branch that happens to guard it.
        for target in _mod.AMEND_TARGETS:
            assert target.new_content is None
            assert target.metadata_patch is None

    @pytest.mark.parametrize(
        'name', ['_STALE_REPLACEMENT', '_WARNING_REPLACEMENT', '_WARNING_PRESERVED_PREFIX'],
    )
    def test_the_stale_write_payloads_are_deleted_not_merely_bypassed(self, name):
        # Re-pointing the sentinel alone would leave these 2026-08-28-authored
        # rewrites in the module as a loaded gun: unreachable today, but fired
        # at a landed correction by any future editor who "fixes" the sentinel
        # or adds a third target. Assert the hazard is GONE.
        assert not hasattr(_mod, name)

    def test_reversion_preimage_pins_the_original_bare_claim(self):
        assert _mod.REVERSION_PREIMAGE == (
            'Broken Claude CLI --resume due to sessions being per-project-directory'
        )

    def test_reversion_preimage_is_not_any_live_preimage(self):
        # The property the reversion guard actually depends on: a healthy
        # record can never EQUAL the reverted text, so the guard cannot fire
        # on a corrected corpus.
        for target in _mod.AMEND_TARGETS:
            assert _mod.REVERSION_PREIMAGE != target.expected_preimage

    def test_reversion_preimage_IS_quoted_inside_both_live_preimages(self):
        # MEASURED, and the whole reason classify_amend_target compares the
        # reversion text by EQUALITY and never by substring: both landed
        # corrections quote the original bare sentence verbatim (A inside its
        # SUPERSEDED preamble, B inside its measured-scores paragraph). A
        # substring test would classify both healthy records as reverted and
        # exit 1 on a perfectly correct corpus.
        for target in _mod.AMEND_TARGETS:
            assert _mod.REVERSION_PREIMAGE in target.expected_preimage

    def test_corrected_record_preimage_carries_the_superseding_facts(self):
        preimage = _mod.AMEND_TARGETS[0].expected_preimage
        # The landed text leads with the supersession marker ...
        assert 'SUPERSEDED' in preimage
        # ... and carries the measurement that supersedes the inferred cause.
        assert '2.1.236' in preimage
        assert 'cwd-AGNOSTIC' in preimage

    def test_hygiene_warning_preimage_has_retired_its_stale_status_clause(self):
        warning = _mod.AMEND_TARGETS[1].expected_preimage
        # The clause this task existed to retire is gone from the live record.
        assert 'STILL UNCORRECTED' not in warning
        # Replaced by one recording the correction.
        assert 'was ALSO corrected in place on 2026-08-30' in warning

    def test_hygiene_warning_preimage_still_carries_the_full_story(self):
        # The nuance the retired _STALE_REPLACEMENT would have put into target
        # A survives verbatim HERE -- which is why nothing was lost by not
        # writing it: the April symptom was REAL, only its cause was inferred,
        # and the April-vs-August question is still open.
        warning = _mod.AMEND_TARGETS[1].expected_preimage
        assert '0.786' in warning
        assert '0.692' in warning
        assert 'e001dd3746' in warning
        assert 'UNDETERMINED' in warning
        assert 'plans/session-resume-eligibility-seam-prd.md' in warning


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


def _decision(memory_id: str, action: str) -> dict:
    return {'id': memory_id, 'action': action}


class TestBuildAmendReport:
    """The report is the artifact an operator diffs and gates on."""

    def test_report_carries_the_expected_top_level_keys(self):
        report = _mod.build_amend_report(
            [_decision(STALE_ID, 'amend'), _decision(WARNING_ID, 'amend')],
            applied_ids=set(),
            dry_run=True,
            generated_at='2026-08-28T00:00:00+00:00',
        )
        assert set(report) == {
            'dry_run', 'generated_at', 'targets', 'totals', 'changes',
        }

    def test_changes_follow_the_two_real_targets_in_order(self):
        report = _mod.build_amend_report(
            [_decision(STALE_ID, 'amend'), _decision(WARNING_ID, 'amend')],
            applied_ids=set(),
            dry_run=True,
            generated_at='2026-08-28T00:00:00+00:00',
        )
        assert [c['id'] for c in report['changes']] == [STALE_ID, WARNING_ID]

    def test_changes_preserve_input_order_rather_than_sorting_by_id(self):
        # Order is SEMANTIC here (it encodes the precondition chain), so the
        # report must NOT sort by id the way the sibling sweeps' reports do.
        #
        # The two REAL ids cannot detect that: 6403e96b sorts BEFORE d007aa46
        # ('6' < 'd'), so their natural sort order already coincides with the
        # semantic order and an accidental sorted() would be invisible. Hence
        # synthetic ids in deliberately reverse-sorted order -- the only shape
        # in which sorting and order-preservation give different answers.
        reversed_ids = ['zzz-last-alphabetically', 'aaa-first-alphabetically']
        assert reversed_ids != sorted(reversed_ids)
        report = _mod.build_amend_report(
            [_decision(mid, 'amend') for mid in reversed_ids],
            applied_ids=set(),
            dry_run=True,
            generated_at='2026-08-28T00:00:00+00:00',
        )
        assert [c['id'] for c in report['changes']] == reversed_ids

    def test_one_change_entry_per_decision_including_skips_and_refusals(self):
        # Unlike the sibling sweeps (whose reports list only the records they
        # touched), every target here is load-bearing: an operator must see
        # what happened to BOTH, including the ones that were refused.
        report = _mod.build_amend_report(
            [
                _decision(STALE_ID, 'skip:already_amended'),
                _decision(WARNING_ID, 'refuse:preimage_mismatch'),
            ],
            applied_ids=set(),
            dry_run=False,
            generated_at='2026-08-28T00:00:00+00:00',
        )
        assert [c['id'] for c in report['changes']] == [STALE_ID, WARNING_ID]
        assert [c['action'] for c in report['changes']] == [
            'skip:already_amended', 'refuse:preimage_mismatch',
        ]
        assert all(set(c) >= {'id', 'action', 'applied'} for c in report['changes'])

    def test_totals_count_amended_skipped_and_refused(self):
        report = _mod.build_amend_report(
            [
                _decision(STALE_ID, 'amend'),
                _decision(WARNING_ID, 'refuse:precondition_failed'),
            ],
            applied_ids={STALE_ID},
            dry_run=False,
            generated_at='2026-08-28T00:00:00+00:00',
        )
        assert report['totals']['amended'] == 1
        assert report['totals']['refused'] == 1
        assert report['totals']['skipped'] == 0

    def test_totals_count_skips_separately_from_refusals(self):
        report = _mod.build_amend_report(
            [
                _decision(STALE_ID, 'skip:already_amended'),
                _decision(WARNING_ID, 'skip:already_amended'),
            ],
            applied_ids=set(),
            dry_run=False,
            generated_at='2026-08-28T00:00:00+00:00',
        )
        assert report['totals']['skipped'] == 2
        assert report['totals']['refused'] == 0
        assert report['totals']['amended'] == 0

    def test_dry_run_never_marks_anything_applied(self):
        # The safety property: a dry-run report must not claim a write.
        report = _mod.build_amend_report(
            [_decision(STALE_ID, 'amend'), _decision(WARNING_ID, 'amend')],
            applied_ids=set(),
            dry_run=True,
            generated_at='2026-08-28T00:00:00+00:00',
        )
        assert report['dry_run'] is True
        assert all(c['applied'] is False for c in report['changes'])
        assert report['totals']['amended'] == 0

    def test_applied_flag_tracks_the_applied_id_set(self):
        report = _mod.build_amend_report(
            [_decision(STALE_ID, 'amend'), _decision(WARNING_ID, 'amend')],
            applied_ids={STALE_ID},
            dry_run=False,
            generated_at='2026-08-28T00:00:00+00:00',
        )
        by_id = {c['id']: c for c in report['changes']}
        assert by_id[STALE_ID]['applied'] is True
        assert by_id[WARNING_ID]['applied'] is False

    def test_report_is_json_serialisable_without_a_default_hook(self):
        import json as _json
        report = _mod.build_amend_report(
            [_decision(STALE_ID, 'amend')],
            applied_ids={STALE_ID},
            dry_run=False,
            generated_at='2026-08-28T00:00:00+00:00',
        )
        _json.dumps(report)  # no default= fallback: primitives only

    def test_generated_at_and_targets_are_echoed(self):
        report = _mod.build_amend_report(
            [_decision(STALE_ID, 'amend'), _decision(WARNING_ID, 'amend')],
            applied_ids=set(),
            dry_run=True,
            generated_at='2026-08-28T12:34:56+00:00',
        )
        assert report['generated_at'] == '2026-08-28T12:34:56+00:00'
        assert report['targets'] == 2


@pytest.fixture(autouse=True)
def _neutralise_store_mutation_preflight(monkeypatch):
    """Keep this MOCK-unit suite independent of the REAL ``~/.mem0``.

    ``run(..., apply=True)`` runs a fail-closed capability preflight before it
    writes. That probe touches the real filesystem, so without this fixture
    every ``--apply`` test would pass or fail according to whether the machine
    running pytest happens to be able to write mem0's history directory -- and
    it genuinely cannot inside an agent sandbox, which is the whole reason the
    guard exists. This suite is deliberately MOCK-unit (an AsyncMock memory
    service, no live Qdrant), so the environment must not be an input to it.

    ``TestApplyStoreMutationPreflight`` re-rigs this per test -- to refuse, to
    record, or to pass -- so the guard's own behaviour is still pinned
    explicitly rather than assumed away.

    Deliberately NOT ``raising=False``: if the guard is ever removed from the
    script this fixture must break loudly rather than silently no-op.
    """
    monkeypatch.setattr(_mod, 'assert_store_mutation_allowed', lambda **_kw: None)


def _memory_service(contents: dict[str, str] | None = None) -> AsyncMock:
    """AsyncMock MemoryService whose get_memory_by_id serves *contents*.

    Defaults to serving each target's exact pre-image, i.e. the corpus as it
    actually stands today: both records stale and amendable.
    """
    if contents is None:
        contents = {t.memory_id: t.expected_preimage for t in _mod.AMEND_TARGETS}

    service = AsyncMock()

    async def _get(*, project_id, memory_id):  # noqa: ARG001
        if memory_id not in contents:
            return None
        return {'id': memory_id, 'content': contents[memory_id], 'metadata': {}}

    service.get_memory_by_id.side_effect = _get
    service.update_memory.return_value = {'status': 'updated', 'store': 'mem0'}
    return service


def _write_capable_targets() -> tuple:
    """Two SYNTHETIC write-capable targets standing in for AMEND_TARGETS.

    Both real targets are verify-only (``new_content is None``) because the
    correction they describe already landed, so nothing live exercises the
    retained write machinery -- the capability preflight, the pre-write
    re-corroboration, the precondition chain, the write-error handling. These
    stand-ins restore an amendable pre-image and a replacement text carrying
    the sentinel, while keeping the REAL memory ids and the REAL ordering so
    every id/order assertion below still means exactly what it did.

    Pinning that machinery against stand-ins rather than deleting it is
    deliberate: the blocking defect was in the pinned DATA, and ~400 lines of
    tested write path is a far larger and riskier thing to remove than the two
    constants that were actually dangerous. What makes keeping it SAFE is
    ``TestVerifyOnlyTargetsAreNeverWritten`` below -- the real targets cannot
    reach this path at all.
    """
    return (
        _mod.AmendTarget(
            memory_id=STALE_ID,
            expected_preimage='a stale record, pre-correction',
            new_content=f'a corrected record -- {_mod.AMENDED_SENTINEL}',
            metadata_patch={'x_corrected_by_task': '4610'},
        ),
        _mod.AmendTarget(
            memory_id=WARNING_ID,
            expected_preimage='a hygiene warning saying the first is STILL UNCORRECTED',
            new_content=f'a hygiene warning, status retired -- {_mod.AMENDED_SENTINEL}',
            metadata_patch={'x_corrected_by_task': '4610'},
        ),
    )


@pytest.fixture
def write_capable_targets(monkeypatch):
    """Swap AMEND_TARGETS for write-capable stand-ins for one test."""
    targets = _write_capable_targets()
    monkeypatch.setattr(_mod, 'AMEND_TARGETS', targets)
    return targets


class TestRunDryRun:
    """Dry run is the DEFAULT, and it must be inert."""

    @pytest.mark.asyncio
    async def test_dry_run_reads_each_target_once(self):
        service = _memory_service()
        await _mod.run(service, project_id='dark_factory', apply=False)
        assert service.get_memory_by_id.await_count == len(_mod.AMEND_TARGETS)

    @pytest.mark.asyncio
    async def test_dry_run_writes_nothing(self):
        # The safety property that makes the default mode runnable from
        # anywhere, including a sandboxed task worktree.
        service = _memory_service()
        await _mod.run(service, project_id='dark_factory', apply=False)
        service.update_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dry_run_never_deletes(self):
        # There is no delete arm in this script at all: the April incident is
        # preserved, not retracted. Pinned here and again on the apply path.
        service = _memory_service()
        await _mod.run(service, project_id='dark_factory', apply=False)
        service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dry_run_does_not_probe_write_capability(self, monkeypatch):
        # Nothing is going to be written, so no capability is needed. Probing
        # anyway would make the read-only report unobtainable from exactly the
        # sandboxed environments that most need to run it first.
        calls = []
        monkeypatch.setattr(
            _mod, 'assert_store_mutation_allowed',
            lambda **kw: calls.append(kw),
        )
        await _mod.run(_memory_service(), project_id='dark_factory', apply=False)
        assert calls == []

    @pytest.mark.asyncio
    async def test_dry_run_over_the_real_corpus_reports_both_already_corrected(self):
        # The corpus as it actually stands: both records carry the landed
        # correction, so the honest report is two skips and nothing applied.
        report = await _mod.run(
            _memory_service(), project_id='dark_factory', apply=False,
        )
        assert report['dry_run'] is True
        assert [c['action'] for c in report['changes']] == [
            'skip:already_amended', 'skip:already_amended',
        ]
        assert all(c['applied'] is False for c in report['changes'])
        assert _mod.resolve_exit_code(report) == 0

    @pytest.mark.asyncio
    async def test_dry_run_over_an_amendable_corpus_reports_amend_but_applies_nothing(
        self, write_capable_targets,  # noqa: ARG002
    ):
        # The same path with something actually to do: the report says 'amend'
        # and the applied flags stay False. This is what makes the dry run a
        # PREVIEW rather than a status readout.
        report = await _mod.run(
            _memory_service(), project_id='dark_factory', apply=False,
        )
        assert [c['action'] for c in report['changes']] == ['amend', 'amend']
        assert all(c['applied'] is False for c in report['changes'])

    @pytest.mark.asyncio
    async def test_dry_run_passes_the_project_id_through_to_each_read(self):
        service = _memory_service()
        await _mod.run(service, project_id='dark_factory', apply=False)
        for call in service.get_memory_by_id.await_args_list:
            assert call.kwargs['project_id'] == 'dark_factory'

    @pytest.mark.asyncio
    async def test_dry_run_reads_the_two_target_ids(self):
        service = _memory_service()
        await _mod.run(service, project_id='dark_factory', apply=False)
        read_ids = [
            c.kwargs['memory_id'] for c in service.get_memory_by_id.await_args_list
        ]
        assert read_ids == [STALE_ID, WARNING_ID]


@pytest.mark.usefixtures('write_capable_targets')
class TestRunApply:
    """The --apply write path: exactly two content amends, no deletes."""

    @pytest.mark.asyncio
    async def test_apply_writes_once_per_target(self):
        service = _memory_service()
        await _mod.run(service, project_id='dark_factory', apply=True)
        assert service.update_memory.await_count == len(_mod.AMEND_TARGETS)

    @pytest.mark.asyncio
    async def test_apply_sends_each_targets_own_replacement_text(self):
        service = _memory_service()
        await _mod.run(service, project_id='dark_factory', apply=True)
        sent = {
            c.kwargs['memory_id']: c.kwargs['content']
            for c in service.update_memory.await_args_list
        }
        for target in _mod.AMEND_TARGETS:
            assert sent[target.memory_id] == target.new_content

    @pytest.mark.asyncio
    async def test_apply_attributes_every_write_to_this_sweep(self):
        # The amendment storm alarm reads _source. A bulk rewrite under the
        # default 'mcp_tool' source would look exactly like the runaway
        # rewrite that alarm exists to catch.
        #
        # agent_id is the SECOND attribution channel and a different consumer:
        # the metadata_patch on these writes runs the vocabulary check at the
        # service seam, which keys its census lines and unknown-key storm
        # buckets by agent_id. Left unset it defaults to None, so this run's
        # rows would read as anonymous while _source named the sweep.
        service = _memory_service()
        await _mod.run(service, project_id='dark_factory', apply=True)
        for call in service.update_memory.await_args_list:
            assert call.kwargs['_source'] == _mod.WRITE_SOURCE
            assert call.kwargs['agent_id'] == _mod.WRITE_SOURCE
            assert call.kwargs['reason'] == _mod.WRITE_REASON
            assert call.kwargs['project_id'] == 'dark_factory'

    @pytest.mark.asyncio
    async def test_apply_never_deletes_anything(self):
        # The whole point of this task: the April incident is PRESERVED. A
        # delete would retire a measured historical observation.
        service = _memory_service()
        await _mod.run(service, project_id='dark_factory', apply=True)
        service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_apply_passes_no_metadata_delete_keys(self):
        # Content amend plus an additive metadata patch only -- nothing on
        # this path removes an existing key.
        service = _memory_service()
        await _mod.run(service, project_id='dark_factory', apply=True)
        for call in service.update_memory.await_args_list:
            assert call.kwargs.get('metadata_delete_keys') is None

    @pytest.mark.asyncio
    async def test_apply_reports_both_amended(self):
        report = await _mod.run(
            _memory_service(), project_id='dark_factory', apply=True,
        )
        assert report['dry_run'] is False
        assert [c['action'] for c in report['changes']] == ['amend', 'amend']
        assert all(c['applied'] is True for c in report['changes'])
        assert report['totals']['amended'] == 2


@pytest.mark.usefixtures('write_capable_targets')
class TestPreWriteRecorroboration:
    """Each write is corroborated against a read of its OWN record, just before.

    run()'s batch pass classifies BOTH targets before either is written, so by
    the time the second write is issued its pre-image is older than a full
    write (a Qdrant round-trip plus a re-embed of a ~2KB replacement). A
    curator sitting or a recon Stage-1/2 consolidation landing in that window
    is exactly what the pre-image guard exists to refuse, and the batch read
    cannot see it -- so the guard would have clobbered the very race it claims
    to catch. These tests pin the re-read that closes that reasoning gap.
    """

    @staticmethod
    def _service_racing_on_the_second_target(new_content: str) -> AsyncMock:
        """Both pre-images, until d007aa46 is rewritten DURING 6403e96b's write."""
        contents = {t.memory_id: t.expected_preimage for t in _mod.AMEND_TARGETS}
        service = AsyncMock()

        async def _get(*, project_id, memory_id):  # noqa: ARG001
            if memory_id not in contents:
                return None
            return {'id': memory_id, 'content': contents[memory_id], 'metadata': {}}

        async def _update(*, memory_id, **kwargs):  # noqa: ARG001
            if memory_id == STALE_ID:
                contents[WARNING_ID] = new_content
            return {'status': 'updated', 'store': 'mem0'}

        service.get_memory_by_id.side_effect = _get
        service.update_memory.side_effect = _update
        return service

    @pytest.mark.asyncio
    async def test_record_edited_during_an_earlier_write_is_refused_not_clobbered(self):
        service = self._service_racing_on_the_second_target(
            'a curator rewrote this record while the first amendment was in flight',
        )
        report = await _mod.run(service, project_id='dark_factory', apply=True)

        written = [
            c.kwargs['memory_id'] for c in service.update_memory.await_args_list
        ]
        assert written == [STALE_ID]  # the moved record was NOT overwritten
        by_id = {c['id']: c for c in report['changes']}
        assert by_id[WARNING_ID]['action'] == 'refuse:preimage_mismatch'
        assert by_id[WARNING_ID]['applied'] is False
        assert _mod.resolve_exit_code(report) == 1

    @pytest.mark.asyncio
    async def test_record_amended_by_someone_else_in_the_window_is_skipped(self):
        # The same race, benign: the concurrent write landed OUR correction.
        # Nothing to do, and it is not a refusal -- a record already carrying
        # the sentinel satisfies anything asserting it was corrected.
        service = self._service_racing_on_the_second_target(
            f'already corrected -- {_mod.AMENDED_SENTINEL}',
        )
        report = await _mod.run(service, project_id='dark_factory', apply=True)

        written = [
            c.kwargs['memory_id'] for c in service.update_memory.await_args_list
        ]
        assert written == [STALE_ID]
        by_id = {c['id']: c for c in report['changes']}
        assert by_id[WARNING_ID]['action'] == 'skip:already_amended'
        assert _mod.resolve_exit_code(report) == 0

    @pytest.mark.asyncio
    async def test_every_write_is_immediately_preceded_by_a_read_of_that_record(self):
        # The property itself: no write is ever issued on a pre-image read
        # before some other write happened.
        service = _memory_service()
        served = service.get_memory_by_id.side_effect
        calls: list[tuple[str, str]] = []

        async def _tracked_get(*, project_id, memory_id):
            calls.append(('read', memory_id))
            return await served(project_id=project_id, memory_id=memory_id)

        async def _tracked_update(*, memory_id, **kwargs):  # noqa: ARG001
            calls.append(('write', memory_id))
            return {'status': 'updated'}

        service.get_memory_by_id.side_effect = _tracked_get
        service.update_memory.side_effect = _tracked_update
        await _mod.run(service, project_id='dark_factory', apply=True)

        assert ('write', WARNING_ID) in calls  # the path under test ran
        for index, (kind, memory_id) in enumerate(calls):
            if kind == 'write':
                assert index > 0
                assert calls[index - 1] == ('read', memory_id)

    @pytest.mark.asyncio
    async def test_a_failed_re_read_refuses_that_target_and_blocks_the_next(self):
        # The re-read is I/O, so it can fail where the batch read succeeded.
        # It must be reported like any other read failure -- not propagated
        # out of run(), and not treated as permission to write blind.
        service = _memory_service()
        served = service.get_memory_by_id.side_effect
        seen: dict[str, int] = {}

        async def _flaky_get(*, project_id, memory_id):
            seen[memory_id] = seen.get(memory_id, 0) + 1
            if memory_id == STALE_ID and seen[memory_id] > 1:
                raise RuntimeError('qdrant timed out on the pre-write re-read')
            return await served(project_id=project_id, memory_id=memory_id)

        service.get_memory_by_id.side_effect = _flaky_get
        report = await _mod.run(service, project_id='dark_factory', apply=True)

        service.update_memory.assert_not_awaited()
        assert [c['action'] for c in report['changes']] == [
            'refuse:read_error', 'refuse:precondition_failed',
        ]
        assert _mod.resolve_exit_code(report) == 1


@pytest.mark.usefixtures('write_capable_targets')
class TestApplyStoreMutationPreflight:
    """The guard's own behaviour, re-rigged per test rather than assumed away.

    The autouse fixture neutralises the probe for every other test; this class
    puts it back so the fail-closed property is pinned explicitly.
    """

    @pytest.mark.asyncio
    async def test_preflight_runs_before_any_write(self, monkeypatch):
        order: list[str] = []
        monkeypatch.setattr(
            _mod, 'assert_store_mutation_allowed',
            lambda **_kw: order.append('preflight'),
        )
        service = _memory_service()

        async def _update(**kwargs):  # noqa: ARG001
            order.append('write')
            return {'status': 'updated'}

        service.update_memory.side_effect = _update
        await _mod.run(service, project_id='dark_factory', apply=True)
        assert order[0] == 'preflight'
        assert 'write' in order

    @pytest.mark.asyncio
    async def test_preflight_is_probed_once_per_run_not_per_write(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            _mod, 'assert_store_mutation_allowed', lambda **kw: calls.append(kw),
        )
        await _mod.run(_memory_service(), project_id='dark_factory', apply=True)
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_preflight_names_the_operation_it_gates(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            _mod, 'assert_store_mutation_allowed', lambda **kw: calls.append(kw),
        )
        await _mod.run(_memory_service(), project_id='dark_factory', apply=True)
        assert '--apply' in calls[0]['operation']

    @pytest.mark.asyncio
    async def test_refused_preflight_performs_zero_writes(self, monkeypatch):
        def _refuse(**_kw):
            raise _mod.StoreMutationUnavailable('sandboxed: cannot write ~/.mem0')

        monkeypatch.setattr(_mod, 'assert_store_mutation_allowed', _refuse)
        service = _memory_service()
        await _mod.run(service, project_id='dark_factory', apply=True)
        service.update_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_refused_preflight_surfaces_a_refusal_not_a_traceback(
        self, monkeypatch,
    ):
        # A sandboxed operator must get the report back and see WHY, rather
        # than a stack trace they have to interpret.
        def _refuse(**_kw):
            raise _mod.StoreMutationUnavailable('sandboxed: cannot write ~/.mem0')

        monkeypatch.setattr(_mod, 'assert_store_mutation_allowed', _refuse)
        report = await _mod.run(
            _memory_service(), project_id='dark_factory', apply=True,
        )
        assert [c['action'] for c in report['changes']] == [
            'refuse:store_unavailable', 'refuse:store_unavailable',
        ]
        assert all(c['applied'] is False for c in report['changes'])
        assert report['totals']['refused'] == 2


@pytest.mark.usefixtures('write_capable_targets')
class TestPreconditionOrdering:
    """d007aa46 asserts that 6403e96b was corrected, so it must never be
    written unless that is TRUE.

    If the two writes were independent and the first failed, the corpus would
    carry a hygiene warning saying "corrected" while the top-ranked stale
    record still misleads -- strictly worse than the status quo, and invisible
    to anyone who trusts the warning. Making the second conditional on the
    first is the only ordering under which a partial failure leaves the corpus
    honest.
    """

    @staticmethod
    def _actions(report):
        return {c['id']: c['action'] for c in report['changes']}

    @pytest.mark.asyncio
    async def test_error_envelope_on_first_write_blocks_the_second(self):
        service = _memory_service()

        async def _update(**kwargs):
            if kwargs['memory_id'] == STALE_ID:
                return {'error': 'nope', 'error_type': 'Mem0UpdateNotAuthorized'}
            return {'status': 'updated'}

        service.update_memory.side_effect = _update
        report = await _mod.run(service, project_id='dark_factory', apply=True)

        written = [c.kwargs['memory_id'] for c in service.update_memory.await_args_list]
        assert written == [STALE_ID]
        assert self._actions(report)[WARNING_ID] == 'refuse:precondition_failed'

    @pytest.mark.asyncio
    async def test_raised_first_write_blocks_the_second_and_is_reported(self):
        service = _memory_service()

        async def _update(**kwargs):
            if kwargs['memory_id'] == STALE_ID:
                raise RuntimeError('qdrant unreachable')
            return {'status': 'updated'}

        service.update_memory.side_effect = _update
        # Reported, not propagated.
        report = await _mod.run(service, project_id='dark_factory', apply=True)

        written = [c.kwargs['memory_id'] for c in service.update_memory.await_args_list]
        assert written == [STALE_ID]
        actions = self._actions(report)
        assert actions[STALE_ID].startswith('refuse:')
        assert actions[WARNING_ID] == 'refuse:precondition_failed'

    @pytest.mark.asyncio
    async def test_preimage_mismatch_on_first_target_blocks_all_writes(self):
        # The corpus changed underneath us: somebody else already edited
        # 6403e96b. Nothing may be written at all.
        service = _memory_service({
            STALE_ID: 'a different correction somebody else wrote',
            WARNING_ID: _mod.AMEND_TARGETS[1].expected_preimage,
        })
        report = await _mod.run(service, project_id='dark_factory', apply=True)

        service.update_memory.assert_not_awaited()
        actions = self._actions(report)
        assert actions[STALE_ID] == 'refuse:preimage_mismatch'
        assert actions[WARNING_ID] == 'refuse:precondition_failed'

    @pytest.mark.asyncio
    async def test_missing_first_target_blocks_the_second(self):
        service = _memory_service({
            WARNING_ID: _mod.AMEND_TARGETS[1].expected_preimage,
        })
        report = await _mod.run(service, project_id='dark_factory', apply=True)

        service.update_memory.assert_not_awaited()
        actions = self._actions(report)
        assert actions[STALE_ID] == 'refuse:not_found'
        assert actions[WARNING_ID] == 'refuse:precondition_failed'

    @pytest.mark.asyncio
    async def test_happy_path_still_writes_the_second(self):
        # The guard must not be so eager that it blocks the case it exists to
        # protect.
        service = _memory_service()
        report = await _mod.run(service, project_id='dark_factory', apply=True)

        written = [c.kwargs['memory_id'] for c in service.update_memory.await_args_list]
        assert written == [STALE_ID, WARNING_ID]
        assert self._actions(report)[WARNING_ID] == 'amend'

    @pytest.mark.asyncio
    async def test_already_amended_first_target_satisfies_the_precondition(self):
        # A skip is not a failure: if 6403e96b already carries the correction
        # then the claim d007aa46 makes about it is TRUE, so the second write
        # is licensed. This is the partial-failure re-run case.
        service = _memory_service({
            STALE_ID: _mod.AMEND_TARGETS[0].new_content,
            WARNING_ID: _mod.AMEND_TARGETS[1].expected_preimage,
        })
        report = await _mod.run(service, project_id='dark_factory', apply=True)

        written = [c.kwargs['memory_id'] for c in service.update_memory.await_args_list]
        assert written == [WARNING_ID]
        actions = self._actions(report)
        assert actions[STALE_ID] == 'skip:already_amended'
        assert actions[WARNING_ID] == 'amend'


@pytest.mark.usefixtures('write_capable_targets')
class TestIdempotency:
    """Re-running after a successful (or partial) apply must be safe.

    This is what lets an operator re-run without first working out how far the
    previous attempt got.
    """

    @staticmethod
    def _amended_corpus() -> dict[str, str]:
        return {t.memory_id: t.new_content for t in _mod.AMEND_TARGETS}

    @pytest.mark.asyncio
    async def test_second_apply_writes_nothing(self):
        service = _memory_service(self._amended_corpus())
        await _mod.run(service, project_id='dark_factory', apply=True)
        service.update_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_second_apply_reports_both_skipped_and_unapplied(self):
        report = await _mod.run(
            _memory_service(self._amended_corpus()),
            project_id='dark_factory', apply=True,
        )
        assert [c['action'] for c in report['changes']] == [
            'skip:already_amended', 'skip:already_amended',
        ]
        assert all(c['applied'] is False for c in report['changes'])
        assert report['totals']['skipped'] == 2
        assert report['totals']['amended'] == 0

    @pytest.mark.asyncio
    async def test_all_skip_run_does_not_probe_write_capability(self, monkeypatch):
        # Nothing is going to be written, so no capability is needed. This is
        # what lets an operator confirm from a sandbox that a previous apply
        # completed, without the probe refusing the run.
        calls = []
        monkeypatch.setattr(
            _mod, 'assert_store_mutation_allowed', lambda **kw: calls.append(kw),
        )
        await _mod.run(
            _memory_service(self._amended_corpus()),
            project_id='dark_factory', apply=True,
        )
        assert calls == []

    @pytest.mark.asyncio
    async def test_partially_amended_corpus_writes_only_what_is_still_stale(self):
        # The resume-after-partial-failure case: the first write landed, the
        # second did not.
        service = _memory_service({
            STALE_ID: _mod.AMEND_TARGETS[0].new_content,
            WARNING_ID: _mod.AMEND_TARGETS[1].expected_preimage,
        })
        report = await _mod.run(service, project_id='dark_factory', apply=True)

        written = [c.kwargs['memory_id'] for c in service.update_memory.await_args_list]
        assert written == [WARNING_ID]
        assert report['totals']['amended'] == 1
        assert report['totals']['skipped'] == 1

    @pytest.mark.asyncio
    async def test_third_run_after_completion_is_a_stable_no_op(self):
        # Idempotency must be stable, not merely true once: two consecutive
        # re-runs over an amended corpus give the same answer.
        corpus = self._amended_corpus()
        first = await _mod.run(
            _memory_service(corpus), project_id='dark_factory', apply=True,
        )
        second = await _mod.run(
            _memory_service(corpus), project_id='dark_factory', apply=True,
        )
        assert first['changes'] == second['changes']
        assert first['totals'] == second['totals']


class TestVerifyOnlyTargetsAreNeverWritten:
    """The property that makes keeping the retired write arm SAFE.

    Both real targets carry ``new_content is None``. Even under ``--apply``,
    even with every guard neutralised, they must reach no write at all -- not
    "are currently skipped by the sentinel branch", but cannot be written.
    Without this, a future target reintroduces a silent clobber the moment
    somebody re-points the sentinel.
    """

    @pytest.mark.asyncio
    async def test_apply_over_the_real_corpus_writes_nothing(self):
        service = _memory_service()
        await _mod.run(service, project_id='dark_factory', apply=True)
        service.update_memory.assert_not_awaited()
        service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_apply_over_the_real_corpus_does_not_probe_write_capability(
        self, monkeypatch,
    ):
        # Nothing to write means no capability is needed, so an operator can
        # run the verifier under --apply from a sandbox and still get an
        # answer rather than a fail-closed refusal.
        calls = []
        monkeypatch.setattr(
            _mod, 'assert_store_mutation_allowed', lambda **kw: calls.append(kw),
        )
        await _mod.run(_memory_service(), project_id='dark_factory', apply=True)
        assert calls == []

    @pytest.mark.asyncio
    async def test_apply_over_the_real_corpus_is_a_clean_zero_write_no_op(self):
        report = await _mod.run(
            _memory_service(), project_id='dark_factory', apply=True,
        )
        assert [c['action'] for c in report['changes']] == [
            'skip:already_amended', 'skip:already_amended',
        ]
        assert all(c['applied'] is False for c in report['changes'])
        assert report['totals']['amended'] == 0
        assert report['totals']['refused'] == 0
        assert _mod.resolve_exit_code(report) == 0

    @pytest.mark.asyncio
    async def test_an_amendable_verify_only_target_is_skipped_not_written(
        self, monkeypatch,
    ):
        # THE regression guard. A verify-only target whose pre-image matches
        # exactly -- so the classifier says 'amend' -- must still not be
        # written. The apply path must treat `new_content is None` as
        # unwritable, never pass content=None to update_memory.
        target = _mod.AmendTarget(
            memory_id=STALE_ID,
            expected_preimage='content with no sentinel in it at all',
        )
        monkeypatch.setattr(_mod, 'AMEND_TARGETS', (target,))
        service = _memory_service()

        # It really does classify 'amend' -- the skip below is the apply
        # path's doing, not the classifier declining to match.
        assert _mod.classify_amend_target(
            target, {'found': True, 'content': target.expected_preimage},
        )['action'] == 'amend'

        report = await _mod.run(service, project_id='dark_factory', apply=True)

        service.update_memory.assert_not_awaited()
        service.delete_memory.assert_not_awaited()
        assert [c['action'] for c in report['changes']] == ['skip:verify_only']
        assert report['changes'][0]['applied'] is False

    @pytest.mark.asyncio
    async def test_a_verify_only_skip_is_clean_not_a_refusal(self):
        # skip:verify_only belongs to the skip family, NOT to ERROR_OUTCOMES:
        # a verify-only target that matches its pre-image is a correct result,
        # and exiting 1 on it would make the verifier unusable.
        assert 'skip:verify_only' not in _mod.ERROR_OUTCOMES

    @pytest.mark.asyncio
    async def test_a_verify_only_target_needs_no_write_capability(self, monkeypatch):
        # The amendable set is computed BEFORE the preflight (step-14's
        # ordering), so an all-verify-only run probes nothing.
        calls = []
        monkeypatch.setattr(
            _mod, 'assert_store_mutation_allowed', lambda **kw: calls.append(kw),
        )
        monkeypatch.setattr(_mod, 'AMEND_TARGETS', (
            _mod.AmendTarget(
                memory_id=STALE_ID,
                expected_preimage='content with no sentinel in it at all',
            ),
        ))
        await _mod.run(_memory_service(), project_id='dark_factory', apply=True)
        assert calls == []


class TestBuildParser:
    def test_dry_run_is_the_default(self):
        # THE safety property of the CLI surface: an operator who forgets the
        # flag gets a report, not a corpus mutation.
        args = _mod.build_parser().parse_args([])
        assert args.apply is False

    def test_apply_is_opt_in(self):
        args = _mod.build_parser().parse_args(['--apply'])
        assert args.apply is True

    def test_project_id_defaults_to_dark_factory(self):
        args = _mod.build_parser().parse_args([])
        assert args.project_id == 'dark_factory'

    def test_project_id_is_overridable(self):
        args = _mod.build_parser().parse_args(['--project-id', 'reify'])
        assert args.project_id == 'reify'


class TestResolveExitCode:
    """An operator must be able to gate on the exit code without parsing."""

    @staticmethod
    def _report(*actions, dry_run=False, applied=()):
        decisions = [
            _decision(f'id-{i}', action) for i, action in enumerate(actions)
        ]
        return _mod.build_amend_report(
            decisions, applied_ids=set(applied), dry_run=dry_run,
            generated_at='2026-08-28T00:00:00+00:00',
        )

    def test_clean_dry_run_exits_zero(self):
        assert _mod.resolve_exit_code(self._report('amend', 'amend', dry_run=True)) == 0

    def test_fully_applied_run_exits_zero(self):
        report = self._report('amend', 'amend', applied=('id-0', 'id-1'))
        assert _mod.resolve_exit_code(report) == 0

    def test_all_skipped_run_exits_zero(self):
        report = self._report('skip:already_amended', 'skip:already_amended')
        assert _mod.resolve_exit_code(report) == 0

    def test_any_refusal_exits_one(self):
        report = self._report('amend', 'refuse:precondition_failed', applied=('id-0',))
        assert _mod.resolve_exit_code(report) == 1

    @pytest.mark.parametrize('action', sorted(_mod.ERROR_OUTCOMES))
    def test_every_named_error_outcome_exits_one(self, action):
        assert _mod.resolve_exit_code(self._report(action)) == 1

    def test_error_outcomes_is_exactly_the_refuse_namespace(self):
        # Named ONCE so the report and the exit code cannot drift on what
        # "clean" means. Every member is a refusal and nothing else is.
        assert all(a.startswith('refuse:') for a in _mod.ERROR_OUTCOMES)

    def test_exit_code_and_report_totals_agree_on_what_is_clean(self):
        # The agreement itself, not two independently-maintained lists.
        for action in sorted(_mod.ERROR_OUTCOMES):
            report = self._report(action)
            assert report['totals']['refused'] == 1
            assert _mod.resolve_exit_code(report) == 1

    def test_every_refusal_the_code_can_emit_is_named_in_error_outcomes(self):
        # Guards the drift this set exists to prevent: a new refuse:* outcome
        # added to the script but not to ERROR_OUTCOMES would exit 0 and read
        # to an automated caller as success.
        import re
        source = SCRIPT_PATH.read_text()
        emitted = set(re.findall(r"'(refuse:[a-z_]+)'", source))
        assert emitted <= set(_mod.ERROR_OUTCOMES)
