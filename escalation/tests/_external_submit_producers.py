"""The calls other repositories make to ``escalation submit``.

The category census walks only this repository, so each such caller's exact
call is pinned here as data, located by ``<repo>:<path>::<symbol>``.
Where the caller bounds a field's width, the entry sits at that bound rather
than at a filed call's value, so a length cap that would refuse the caller's
real calls fails the test.
``test_submit_cli.py`` runs every entry through ``escalation.submit.main``, and
``test_category_contract.py`` derives its external-category pins from them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExternalSubmitProducer:
    locator: str
    task: str
    severity: str
    category: str
    agent_role: str
    summary: str
    detail: str

    def submit_argv(self, queue_dir: Path) -> list[str]:
        return [
            'submit',
            '--queue-dir', str(queue_dir),
            '--task', self.task,
            '--severity', self.severity,
            '--category', self.category,
            '--agent-role', self.agent_role,
            '--summary', self.summary,
            '--detail', self.detail,
        ]


_SCP_LONGEST_LANE = 'base-image-end-of-life-weekly'
_SCP_RUN_DIR = f'/home/leo/.local/state/scp-validation/{_SCP_LONGEST_LANE}/20261108T021300Z'
_SCP_REASON_AT_FULL_SUMMARY_WIDTH = 'tests failed: ' + ', '.join(
    f'tests.test_base_image_end_of_life::{test}'
    for test in (
        'test_the_dockerfile_pulls_each_scheduled_runtime',
        'test_a_pin_is_out_of_support_on_a_day_by_its_own_runtimes_schedule',
    )
)

EXTERNAL_SUBMIT_PRODUCERS: tuple[ExternalSubmitProducer, ...] = (
    ExternalSubmitProducer(
        locator=(
            'solar-challenge-platform:src/solar_challenge_platform/ci/cadence_alert.py'
            '::EscalationCli.submit_argv'
        ),
        task=f'validation-cadence-{_SCP_LONGEST_LANE}',
        severity='urgent',
        category='validation_lane_red',
        agent_role='harness-scp-validation-cadence',
        summary=f'Validation lane {_SCP_LONGEST_LANE} failed: {_SCP_REASON_AT_FULL_SUMMARY_WIDTH}',
        detail='\n'.join((
            f'Run dir: {_SCP_RUN_DIR}',
            f'Log: {_SCP_RUN_DIR}/lane.log',
            f'Result: {_SCP_RUN_DIR}/result.json',
            'Verdict: red',
            f'Reason: {_SCP_REASON_AT_FULL_SUMMARY_WIDTH}',
            'Exit code: 1',
            'Git sha: 42f39035ea5be0981ba8260ea2bc5a945495a4e6',
            'Start: 2026-11-08T02:13:00.561207+00:00',
            'End: 2026-11-08T02:13:06.204518+00:00',
            'Argv: uv run --extra dev pytest tests/test_base_image_end_of_life.py -m slow'
            f' --junitxml={_SCP_RUN_DIR}/junit.xml',
            f'Journal: journalctl --user -u scp-validation@{_SCP_LONGEST_LANE}.service',
        )),
    ),
)
