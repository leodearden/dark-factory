"""Per-run instrument checks (PRD §Instrument validation): is this run's measurement valid?

Every check returns a ``CheckResult``. A failed one names the invariant and its
offending values (INV-2). Only the two git checks touch the outside world, and a
git failure becomes a failed result rather than an exception.
"""

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from fused_memory.arm_harness.arm_spec import EmbeddingArmSpec, LlmArmSpec
from fused_memory.arm_harness.metrics_record import LlmMetricId, MetricsRecord
from fused_memory.arm_harness.replay import EpisodeOutcome

PREREGISTRATION_DOC_PATH = 'plans/local-memory-models-eval-preregistration.md'


class InstrumentCheckId(StrEnum):
    CODE_SHA_MATCHES_CHECKOUT = 'code-sha-matches-checkout'
    PREREGISTRATION_SHA = 'preregistration-sha'
    TOKEN_COST_ACCOUNTING = 'token-cost-accounting'
    REFERENCE_NONEMPTY = 'reference-nonempty'
    ARM_CONFIG_SYMMETRY = 'arm-config-symmetry'
    SINGLE_CODE_SHA = 'single-code-sha'
    ENDPOINT_CONFORMANCE = 'endpoint-conformance'
    VALIDATOR_NEGATIVE_CONTROL = 'validator-negative-control'
    INDEX_CONFIGURATION = 'index-configuration'


@dataclass(frozen=True)
class CheckResult:
    check_id: InstrumentCheckId
    passed: bool
    detail: str
    offenders: tuple[str, ...]


def check_passed(check_id: InstrumentCheckId, detail: str) -> CheckResult:
    return CheckResult(check_id=check_id, passed=True, detail=detail, offenders=())


def check_failed(
    check_id: InstrumentCheckId, detail: str, offenders: Sequence[str]
) -> CheckResult:
    return CheckResult(check_id=check_id, passed=False, detail=detail, offenders=tuple(offenders))


class _GitFailed(Exception):
    """A git command exited non-zero; the git checks turn this into a failed result."""


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ['git', '-C', str(repo_root), *args], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise _GitFailed(
            f'git {" ".join(args)} exited {completed.returncode}: {completed.stderr.strip()}'
        )
    return completed.stdout


def _dirty_paths(repo_root: Path) -> tuple[str, ...]:
    changed = _git(repo_root, 'diff', '--name-only', '-z', 'HEAD')
    untracked = _git(repo_root, 'ls-files', '--others', '--exclude-standard', '-z')
    return tuple(sorted({path for path in (changed + untracked).split('\0') if path}))


def check_code_sha_matches_checkout(
    spec: LlmArmSpec | EmbeddingArmSpec, repo_root: Path
) -> CheckResult:
    check_id = InstrumentCheckId.CODE_SHA_MATCHES_CHECKOUT
    invariant = f'code_sha must be the clean HEAD of {repo_root}'
    try:
        head_sha = _git(repo_root, 'rev-parse', 'HEAD').strip()
        dirty = _dirty_paths(repo_root)
    except _GitFailed as error:
        return check_failed(check_id, f'{invariant}: {error}', ())
    problems: list[str] = []
    offenders: list[str] = []
    if head_sha != spec.code_sha:
        problems.append(f'spec code_sha {spec.code_sha} != HEAD {head_sha}')
        offenders.append(head_sha)
    if dirty:
        problems.append(f'the tree is dirty: {list(dirty)}')
        offenders.extend(dirty)
    if problems:
        return check_failed(check_id, f'{invariant}: {"; ".join(problems)}', offenders)
    return check_passed(check_id, f'HEAD {head_sha} is clean and matches the spec')


def check_preregistration_sha(
    spec: LlmArmSpec | EmbeddingArmSpec, repo_root: Path
) -> CheckResult:
    check_id = InstrumentCheckId.PREREGISTRATION_SHA
    sha = spec.preregistration_sha
    if sha is None:
        return check_passed(
            check_id, f'{spec.arm_role} arm {spec.arm_id!r} carries no preregistration'
        )
    try:
        _git(repo_root, 'cat-file', '-e', f'{sha}:{PREREGISTRATION_DOC_PATH}')
    except _GitFailed as error:
        return check_failed(
            check_id,
            f'candidate arm {spec.arm_id!r}: preregistration_sha {sha} must carry '
            f'{PREREGISTRATION_DOC_PATH}: {error}',
            (sha,),
        )
    return check_passed(check_id, f'{sha} carries {PREREGISTRATION_DOC_PATH}')


def check_token_cost_accounting(
    spec: LlmArmSpec, records: Sequence[MetricsRecord]
) -> CheckResult:
    check_id = InstrumentCheckId.TOKEN_COST_ACCOUNTING
    values = {record.metric.metric_id: record.metric.value for record in records}
    required = [LlmMetricId.TOKENS_PER_EPISODE]
    if spec.pricing is not None:
        required.append(LlmMetricId.USD_PER_EPISODE)
    offenders = [metric_id for metric_id in required if not values.get(metric_id)]
    if offenders:
        found = {metric_id: values.get(metric_id) for metric_id in offenders}
        return check_failed(
            check_id,
            f'arm {spec.arm_id!r}: token accounting (and cost, on a metered arm) must be '
            f'present and non-zero; got {found}',
            offenders,
        )
    return check_passed(check_id, f'arm {spec.arm_id!r}: tokens and cost are accounted')


def check_reference_nonempty(reference: Sequence[EpisodeOutcome]) -> CheckResult:
    check_id = InstrumentCheckId.REFERENCE_NONEMPTY
    invariant = 'the reference graph must hold ok episodes with entities'
    if not reference:
        return check_failed(check_id, f'{invariant}: the reference has no outcomes', ())
    ok = [outcome for outcome in reference if outcome.ok]
    if not ok:
        failed_ids = [outcome.episode_id for outcome in reference]
        return check_failed(check_id, f'{invariant}: every reference episode failed', failed_ids)
    entities = sum(len(outcome.entity_names) for outcome in ok)
    if entities == 0:
        ok_ids = [outcome.episode_id for outcome in ok]
        return check_failed(check_id, f'{invariant}: {len(ok)} ok episodes hold 0 entities', ok_ids)
    return check_passed(check_id, f'{len(ok)} ok reference episodes hold {entities} entities')
