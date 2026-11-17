"""The single declaration of the user timers this repo expects armed on its host, each with its installer.

Slots and rationale per timer: OPERATIONS.md §12 "The nightly ladder".
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

__all__ = ['ARMED_ELSEWHERE', 'EXPECTED_TIMERS', 'ExpectedTimer', 'InstallerArg']

_TIMER_SUFFIX = '.timer'
_TEMPLATE_SUFFIX = '@.timer'


class InstallerArg(enum.Enum):
    """What an installer takes as its one argument."""

    NONE = 'none'
    PROJECT_ID = 'project_id'
    PROJECT_ROOT = 'project_root'


@dataclass(frozen=True)
class ExpectedTimer:
    unit_file: str
    installer: str
    installer_arg: InstallerArg = InstallerArg.NONE

    def __post_init__(self) -> None:
        if not self.unit_file.endswith(_TIMER_SUFFIX):
            raise ValueError(f'expected timer unit_file {self.unit_file!r} does not end in {_TIMER_SUFFIX!r}')
        takes_project = self.installer_arg is not InstallerArg.NONE
        if self.is_template != takes_project:
            raise ValueError(
                f'a template timer (unit_file ending {_TEMPLATE_SUFFIX!r}) takes a project argument exactly when '
                f'its installer needs one, but {self.unit_file!r} has installer_arg={self.installer_arg.name}'
            )

    @property
    def is_template(self) -> bool:
        return self.unit_file.endswith(_TEMPLATE_SUFFIX)

    def unit_name(self, project_id: str) -> str:
        """The concrete unit name systemd answers for; a template is bound to *project_id*'s instance."""
        basename = Path(self.unit_file).name
        if not self.is_template:
            return basename
        if not project_id:
            raise ValueError(f'template timer {self.unit_file!r} needs a non-empty project_id to name its instance')
        return basename.replace(_TEMPLATE_SUFFIX, f'@{project_id}{_TIMER_SUFFIX}')

    def installer_argv(self, repo_root: Path, project_id: str) -> tuple[str, ...]:
        installer = str(repo_root / self.installer)
        if self.installer_arg is InstallerArg.PROJECT_ID:
            return (installer, project_id)
        if self.installer_arg is InstallerArg.PROJECT_ROOT:
            return (installer, str(repo_root))
        return (installer,)


EXPECTED_TIMERS: tuple[ExpectedTimer, ...] = (
    ExpectedTimer('scripts/shared-repo-git-maintenance.timer', 'scripts/install-shared-repo-git-maintenance-timer.sh'),
    ExpectedTimer('scripts/fused-memory-link-heal.timer', 'scripts/install-link-heal-timer.sh'),
    ExpectedTimer('scripts/transcript-secret-sweep.timer', 'scripts/install-transcript-secret-sweep-timer.sh'),
    ExpectedTimer(
        'scripts/legibility-trickle@.timer', 'scripts/legibility/install-trickle-timer.sh', InstallerArg.PROJECT_ID,
    ),
    ExpectedTimer('scripts/fused-memory-memory-eval.timer', 'scripts/install-memory-eval-timer.sh'),
    ExpectedTimer('scripts/fused-memory-flag-marker-sweep.timer', 'scripts/install-flag-marker-sweep-timer.sh'),
    ExpectedTimer(
        'scripts/quality-snapshot@.timer', 'scripts/install-quality-snapshot-timer.sh', InstallerArg.PROJECT_ROOT,
    ),
    ExpectedTimer(
        'scripts/reclaim-orphaned-worktrees.timer', 'scripts/install-reclaim-orphaned-worktrees-timer.sh',
    ),
    ExpectedTimer(
        'scripts/legibility-transcript-check@.timer',
        'scripts/legibility/install-transcript-check-timer.sh',
        InstallerArg.PROJECT_ID,
    ),
    ExpectedTimer(
        'scripts/legibility-trickle-health@.timer',
        'scripts/legibility/install-trickle-health-timer.sh',
        InstallerArg.PROJECT_ID,
    ),
    ExpectedTimer(
        'scripts/memory-metadata-coverage-census.timer', 'scripts/install-memory-metadata-coverage-census-timer.sh',
    ),
    ExpectedTimer('scripts/return-brief.timer', 'scripts/install-return-brief-timer.sh'),
    ExpectedTimer('scripts/exception-owner-sweep.timer', 'scripts/install-exception-owner-sweep-timer.sh'),
    ExpectedTimer('scripts/fused-memory-duplicate-audit.timer', 'scripts/install-duplicate-audit-timer.sh'),
    ExpectedTimer('dashboard/dark-factory-load-sampler.timer', 'scripts/install-load-sampler.sh'),
)

ARMED_ELSEWHERE: Mapping[str, str] = MappingProxyType({
    'scripts/orchestrator-watchdog.timer': 'installed and enabled by scripts/setup-host.sh §5',
    'dashboard/dark-factory-dashboard-watchdog.timer': 'installed and enabled by scripts/setup-host.sh §7',
    'scripts/deploy/laptop-battery-guard.timer': (
        'a sudo-installed system unit, on laptop hosts only, via scripts/deploy/install_laptop_battery_guard.sh'
    ),
})
