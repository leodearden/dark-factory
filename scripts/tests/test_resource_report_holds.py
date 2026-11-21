"""The hold analysis: scripts/resource_report_holds.py.

PRD ``plans/per-consumer-resource-recording-prd.md`` leaf 2, contract C2 and
the §7 rows "Hold before go-live" and "Hold mapping, both shapes". Consumers
are built with ``sampler.consumers``, verify tags with
``shared.verify_scope_tag``, and the stores with resource_report_fixtures.
"""
from __future__ import annotations

from pathlib import Path

import _checkout as _checkout
import pytest
import resource_report_holds as holds
from sampler.consumers import Consumer, Role
from shared.verify_scope_tag import scope_tag_for

APP = '/user.slice/user-1000.slice/user@1000.service/app.slice'


class TestHoldMappingBothShapes:
    def test_the_unit_shape_maps_service_agents_and_verify_but_no_project(self, tmp_path: Path):
        root = tmp_path / 'solar-challenge'
        unit = 'orchestrator-my-solar-challenge.service'

        mapping = holds.hold_consumers(f'{APP}/{unit}', root)

        assert mapping.service == Consumer(Role.SERVICE, unit)
        assert mapping.agents == Consumer(Role.AGENTS, unit)
        assert mapping.project is None
        assert mapping.verify == Consumer(Role.VERIFY, scope_tag_for(root))
        assert mapping.unmapped_reason is None

    def test_the_unit_stem_and_the_verify_slug_are_not_assumed_equal(self, tmp_path: Path):
        root = tmp_path / 'solar-challenge'

        mapping = holds.hold_consumers(f'{APP}/orchestrator-my-solar-challenge.service', root)

        assert mapping.verify is not None
        assert mapping.verify.key is not None
        assert mapping.verify.key.startswith('solar-challenge-')
        assert not mapping.verify.key.startswith('my-solar-challenge')

    def test_the_slice_shape_keys_service_agents_and_project_by_the_slice(self, tmp_path: Path):
        root = tmp_path / 'dark-factory'
        project_slice = 'df-dark_factory.slice'

        mapping = holds.hold_consumers(f'{APP}/df.slice/{project_slice}', root)

        assert mapping.service == Consumer(Role.SERVICE, project_slice)
        assert mapping.agents == Consumer(Role.AGENTS, project_slice)
        assert mapping.project == Consumer(Role.PROJECT, project_slice)
        assert mapping.verify == Consumer(Role.VERIFY, scope_tag_for(root))
        assert mapping.verify.key is not None
        assert mapping.verify.key.startswith('dark-factory-')
        assert mapping.unmapped_reason is None

    def test_reify_s_unit_never_maps_its_legacy_slice(self, tmp_path: Path):
        mapping = holds.hold_consumers(f'{APP}/orchestrator-reify.service', tmp_path / 'reify')

        assert mapping.project is None
        assert Consumer(Role.PROJECT, 'reify.slice') not in {
            mapping.service, mapping.agents, mapping.project
        }

    def test_an_unresolved_own_cgroup_maps_only_verify(self, tmp_path: Path):
        root = tmp_path / 'dark-factory'

        mapping = holds.hold_consumers('', root)

        assert (mapping.service, mapping.agents, mapping.project) == (None, None, None)
        assert mapping.verify == Consumer(Role.VERIFY, scope_tag_for(root))
        assert mapping.unmapped_reason is not None
        assert 'unresolved' in mapping.unmapped_reason

    def test_a_hold_event_without_an_own_cgroup_maps_only_verify(self, tmp_path: Path):
        root = tmp_path / 'dark-factory'

        mapping = holds.hold_consumers(None, root)

        assert (mapping.service, mapping.agents, mapping.project) == (None, None, None)
        assert mapping.verify == Consumer(Role.VERIFY, scope_tag_for(root))
        assert mapping.unmapped_reason is not None
        assert 'no own cgroup' in mapping.unmapped_reason

    @pytest.mark.parametrize(
        ('own_cgroup', 'segment'),
        [
            (f'{APP}/orchestrator-watchdog.service', 'orchestrator-watchdog.service'),
            (f'{APP}/orchestrator-dark-factory.service/daemon', 'daemon'),
        ],
    )
    def test_a_last_segment_that_is_neither_shape_is_unmapped_by_name(
        self, tmp_path: Path, own_cgroup: str, segment: str
    ):
        root = tmp_path / 'dark-factory'

        mapping = holds.hold_consumers(own_cgroup, root)

        assert (mapping.service, mapping.agents, mapping.project) == (None, None, None)
        assert mapping.verify == Consumer(Role.VERIFY, scope_tag_for(root))
        assert mapping.unmapped_reason is not None
        assert segment in mapping.unmapped_reason
