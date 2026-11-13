"""Unit tests for orchestrator.scope_drift: scope requests and drift reports (task 3090)."""

from __future__ import annotations

import dataclasses

import pytest

from orchestrator.pin_reservation import Blocker, BlockerKind
from orchestrator.scope_drift import (
    InvalidScopeRequest,
    ScopeBoundary,
    ScopeDriftReason,
    ScopeDriftReport,
    ScopeRequest,
    build_scope_drift_report,
    pending_scope_paths,
)

LOCK_DEPTH = 12


class TestScopeRequest:
    def test_accepts_a_repo_relative_file_path(self):
        request = ScopeRequest('orchestrator/src/orchestrator/foo.py', 'plan omitted it')
        assert request.path == 'orchestrator/src/orchestrator/foo.py'
        assert request.reason == 'plan omitted it'

    def test_normalizes_a_leading_dot_slash(self):
        request = ScopeRequest('./orchestrator/src/orchestrator/foo.py', 'plan omitted it')
        assert request.path == 'orchestrator/src/orchestrator/foo.py'

    @pytest.mark.parametrize(
        'path',
        [
            '/abs/orchestrator/foo.py',
            'orchestrator/../secrets/foo.py',
            '../foo.py',
            '.task/plan.json',
            'orchestrator/src/orchestrator',
            '',
        ],
    )
    def test_rejects_an_invalid_path_naming_it(self, path):
        with pytest.raises(InvalidScopeRequest) as excinfo:
            ScopeRequest(path, 'plan omitted it')
        assert excinfo.value.field == 'path'
        assert repr(path) in str(excinfo.value)

    @pytest.mark.parametrize('reason', ['', '   ', '\n\t'])
    def test_rejects_an_empty_reason(self, reason):
        with pytest.raises(InvalidScopeRequest) as excinfo:
            ScopeRequest('pkg/a.py', reason)
        assert excinfo.value.field == 'reason'
        assert repr(reason) in str(excinfo.value)

    @pytest.mark.parametrize('field_name', [f.name for f in dataclasses.fields(ScopeRequest)])
    def test_is_frozen(self, field_name):
        request = ScopeRequest('pkg/a.py', 'plan omitted it')
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(request, field_name, 'other')

    def test_mapping_round_trip(self):
        request = ScopeRequest('pkg/a.py', 'plan omitted it')
        assert ScopeRequest.from_mapping(request.to_mapping()) == request

    @pytest.mark.parametrize(
        ('mapping', 'field'),
        [
            ({'reason': 'plan omitted it'}, 'path'),
            ({'path': 'pkg/a.py'}, 'reason'),
            ({'path': 7, 'reason': 'plan omitted it'}, 'path'),
            ({'path': 'pkg/a.py', 'reason': None}, 'reason'),
        ],
    )
    def test_from_mapping_rejects_a_missing_or_non_string_field(self, mapping, field):
        with pytest.raises(InvalidScopeRequest) as excinfo:
            ScopeRequest.from_mapping(mapping)
        assert excinfo.value.field == field


class TestPendingScopePaths:
    def test_drops_declared_paths_and_duplicates_keeping_order(self):
        requests = [
            ScopeRequest('pkg/c.py', 'r'),
            ScopeRequest('pkg/a.py', 'r'),
            ScopeRequest('pkg/b.py', 'r'),
            ScopeRequest('pkg/c.py', 'again'),
        ]
        assert pending_scope_paths(requests, ['pkg/a.py']) == ('pkg/c.py', 'pkg/b.py')

    def test_plan_entries_are_compared_after_dot_slash_normalization(self):
        requests = [ScopeRequest('pkg/a.py', 'r'), ScopeRequest('pkg/b.py', 'r')]
        assert pending_scope_paths(requests, ['./pkg/a.py']) == ('pkg/b.py',)

    def test_empty_when_every_request_is_declared(self):
        requests = [ScopeRequest('pkg/a.py', 'r')]
        assert pending_scope_paths(requests, ['pkg/a.py', 'pkg/z.py']) == ()

    def test_empty_without_requests(self):
        assert pending_scope_paths([], ['pkg/a.py']) == ()


def _report(requested, contested=()):
    return build_scope_drift_report(
        requested,
        contested,
        lock_depth=LOCK_DEPTH,
        boundary=ScopeBoundary.EXECUTE,
        role='implementer',
        iteration=2,
        task_id='42',
    )


class TestBuildScopeDriftReport:
    def test_all_acquired_is_scope_expanded(self):
        report = _report(['pkg/a.py', 'pkg/b.py'])
        assert report.reason is ScopeDriftReason.IMPLEMENT_SCOPE_EXPANDED
        assert report.requested == ('pkg/a.py', 'pkg/b.py')
        assert report.acquired == ('pkg/a.py', 'pkg/b.py')
        assert report.contested == ()

    def test_a_contested_path_maps_to_its_holder(self):
        report = _report(
            ['pkg/a.py', 'pkg/b.py'],
            [Blocker('pkg/b.py', '777', BlockerKind.HELD)],
        )
        assert report.reason is ScopeDriftReason.IMPLEMENT_BLAST_RADIUS_LOCK_CONFLICT
        assert report.acquired == ('pkg/a.py',)
        assert report.contested == (('pkg/b.py', '777'),)

    def test_several_holders_record_the_lowest_in_blocker_order(self):
        report = _report(
            ['pkg/b.py'],
            [
                Blocker('pkg/b.py', '9', BlockerKind.PARKED),
                Blocker('pkg/b.py', '12', BlockerKind.HELD),
            ],
        )
        assert report.contested == (('pkg/b.py', '12'),)

    def test_a_path_deriving_no_lock_key_is_acquired(self):
        report = _report(['pkg/subdir'], [Blocker('pkg/subdir', '5', BlockerKind.HELD)])
        assert report.acquired == ('pkg/subdir',)
        assert report.contested == ()
        assert report.reason is ScopeDriftReason.IMPLEMENT_SCOPE_EXPANDED

    def test_event_data_shape(self):
        report = _report(
            ['pkg/a.py', 'pkg/b.py'],
            [Blocker('pkg/b.py', '777', BlockerKind.HELD)],
        )
        assert report.event_data() == {
            'reason': 'implement_blast_radius_lock_conflict',
            'boundary': 'execute',
            'role': 'implementer',
            'iteration': 2,
            'requested': ['pkg/a.py', 'pkg/b.py'],
            'acquired': ['pkg/a.py'],
            'contested': {'pkg/b.py': '777'},
        }

    @pytest.mark.parametrize(
        'field_name', [f.name for f in dataclasses.fields(ScopeDriftReport)]
    )
    def test_report_is_frozen(self, field_name):
        report = _report(['pkg/a.py'])
        assert isinstance(report, ScopeDriftReport)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(report, field_name, 'other')


class TestMergedWith:
    def test_a_later_acquisition_clears_an_earlier_contest(self):
        earlier = _report(['pkg/b.py'], [Blocker('pkg/b.py', '777', BlockerKind.HELD)])
        later = _report(['pkg/b.py', 'pkg/c.py'], [Blocker('pkg/c.py', '9', BlockerKind.HELD)])
        merged = earlier.merged_with(_report(['pkg/b.py'])).merged_with(later)
        assert merged.acquired == ('pkg/b.py',)
        assert merged.contested == (('pkg/c.py', '9'),)
        assert merged.requested == ('pkg/b.py', 'pkg/c.py')
        assert merged.reason is ScopeDriftReason.IMPLEMENT_BLAST_RADIUS_LOCK_CONFLICT

    def test_an_earlier_contest_survives_a_later_unrelated_grant(self):
        earlier = _report(['pkg/b.py'], [Blocker('pkg/b.py', '777', BlockerKind.HELD)])
        merged = earlier.merged_with(_report(['pkg/a.py']))
        assert merged.acquired == ('pkg/a.py',)
        assert merged.contested == (('pkg/b.py', '777'),)
        assert merged.reason is ScopeDriftReason.IMPLEMENT_BLAST_RADIUS_LOCK_CONFLICT

    def test_the_later_boundary_and_iteration_win(self):
        earlier = _report(['pkg/b.py'], [Blocker('pkg/b.py', '777', BlockerKind.HELD)])
        later = dataclasses.replace(
            _report(['pkg/a.py']), boundary=ScopeBoundary.AMENDMENT, iteration=5,
        )
        merged = earlier.merged_with(later)
        assert merged.boundary is ScopeBoundary.AMENDMENT
        assert merged.iteration == 5


class TestScopeDriftReason:
    def test_the_reason_is_the_enum_and_serializes_to_its_string(self):
        report = _report(['pkg/a.py'])
        assert report.reason is ScopeDriftReason.IMPLEMENT_SCOPE_EXPANDED
        assert report.event_data()['reason'] == 'implement_scope_expanded'
        assert type(report.event_data()['reason']) is str

    @pytest.mark.parametrize(
        ('reason', 'contested'),
        [
            (ScopeDriftReason.IMPLEMENT_SCOPE_EXPANDED, (('pkg/b.py', '7'),)),
            (ScopeDriftReason.IMPLEMENT_BLAST_RADIUS_LOCK_CONFLICT, ()),
        ],
    )
    def test_an_implement_reason_that_disagrees_with_contested_is_refused(
        self, reason, contested
    ):
        with pytest.raises(ValueError, match=reason.value):
            ScopeDriftReport(
                reason=reason, boundary=ScopeBoundary.EXECUTE, role='implementer',
                iteration=1, requested=('pkg/b.py',), acquired=(), contested=contested,
            )

    def test_an_unknown_reason_string_is_refused(self):
        with pytest.raises(ValueError):
            ScopeDriftReason('made_up')


class TestScopeBoundary:
    def test_an_unknown_boundary_string_is_refused(self):
        with pytest.raises(ValueError):
            dataclasses.replace(_report(['pkg/a.py']), boundary='made_up')

    def test_a_boundary_string_is_coerced_to_the_enum(self):
        report = dataclasses.replace(_report(['pkg/a.py']), boundary='debug_fix')
        assert report.boundary is ScopeBoundary.DEBUG_FIX


class TestFromEventData:
    @pytest.mark.parametrize(
        'report',
        [
            _report(['pkg/a.py']),
            _report(['pkg/a.py', 'pkg/b.py'], [Blocker('pkg/b.py', '777', BlockerKind.HELD)]),
        ],
        ids=['granted', 'contested'],
    )
    def test_round_trips_event_data(self, report):
        assert ScopeDriftReport.from_event_data(report.event_data()) == report

    @pytest.mark.parametrize(
        ('field_name', 'value'),
        [
            ('reason', 'made_up'),
            ('boundary', None),
            ('role', 7),
            ('iteration', '2'),
            ('iteration', True),
            ('requested', 'pkg/a.py'),
            ('acquired', [3]),
            ('contested', [['pkg/b.py', '777']]),
            ('contested', {'pkg/b.py': 777}),
        ],
    )
    def test_refuses_a_malformed_field(self, field_name, value):
        data = _report(['pkg/a.py']).event_data()
        data[field_name] = value
        with pytest.raises(ValueError):
            ScopeDriftReport.from_event_data(data)

    def test_refuses_a_non_mapping(self):
        with pytest.raises(ValueError):
            ScopeDriftReport.from_event_data(['reason'])
