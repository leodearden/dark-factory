"""The calls other repositories make to ``escalation submit``.

The category census walks only this repository, so each such caller's exact
call is pinned here as data, named ``<repo>:<path>::<symbol>``.
``test_submit_cli.py`` runs every entry through ``escalation.submit.main``, and
``test_category_contract.py`` derives its external-category pins from them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExternalSubmitProducer:
    producer: str
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


_SCP_RUN_DIR = '/home/leo/.local/state/scp-validation/slow-lane/20261109T031700Z'

EXTERNAL_SUBMIT_PRODUCERS: tuple[ExternalSubmitProducer, ...] = (
    ExternalSubmitProducer(
        producer=(
            'solar-challenge-platform:src/solar_challenge_platform/ci/cadence_alert.py'
            '::EscalationCli.submit_argv'
        ),
        task='validation-cadence-slow-lane',
        severity='urgent',
        category='validation_lane_red',
        agent_role='harness-scp-validation-cadence',
        summary=(
            'Validation lane slow-lane failed: it gave no signal: pvlib: pass: 6 tests ran'
            ' timescale: no_signal: a test skipped instead of running [...]'
        ),
        detail='\n'.join((
            f'Run dir: {_SCP_RUN_DIR}',
            f'Log: {_SCP_RUN_DIR}/lane.log',
            f'Result: {_SCP_RUN_DIR}/result.json',
            'Verdict: no_signal',
            'Reason: pvlib: pass: 6 tests ran',
            'timescale: no_signal: a test skipped instead of running, so the lane gave no signal',
            'Exit code: 0',
            'Git sha: edb8a581a71ef19a8751e406fc278360c6b1bb82',
            'Start: 2026-11-09T03:17:00.468604+00:00',
            'End: 2026-11-09T03:36:59.686944+00:00',
            'Argv: uv run --extra dev python -m solar_challenge_platform.ci.slow_lane'
            f' --artifact-dir {_SCP_RUN_DIR}/artifacts',
            'Journal: journalctl --user -u scp-validation@slow-lane.service',
        )),
    ),
)
