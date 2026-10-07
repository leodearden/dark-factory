"""The η survivor rule of plans/local-memory-models-eval-preregistration.md §8: four absolute gates per arm over its screening evidence; survivors are the arms that pass all four."""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator
from shared.memory_eval_metrics import canonical_json_text

from fused_memory.arm_harness.arm_spec import ArmId, ContentSha, GitSha
from fused_memory.arm_harness.frozen_model import FrozenModel
from fused_memory.arm_harness.metrics_record import LlmMetricId
from fused_memory.arm_harness.preregistration import LatencyEnvelope, PreregistrationInputs
from fused_memory.arm_harness.replay_types import EpisodeOutcome
from fused_memory.arm_harness.screening_evidence import (
    ArmEvidence,
    ReportedEvidence,
    ScreeningEvidenceError,
    ScreeningStage,
    reported_evidence,
)
from fused_memory.arm_harness.slate import LocalLlmStack, ReasoningMode, SlateArm
from fused_memory.arm_harness.usage_tap import CallRecord

SURVIVOR_CAP = 3
SCREENING_VERDICT_SCHEMA_VERSION = 1
SCREENING_VERDICT_FILENAME = 'screening-verdict.json'
_LISTED_REJECTIONS = 5


class GateId(StrEnum):
    CONFORMANCE_SMOKE = 'conformance-smoke'
    VRAM_FIT = 'vram-fit'
    CONTEXT_FIT = 'context-fit'
    THROUGHPUT_FLOOR = 'throughput-floor'


class GateVerdict(StrEnum):
    PASS = 'PASS'
    FAIL = 'FAIL'
    UNMEASURED = 'UNMEASURED'


class GateUnit(StrEnum):
    MIB = 'MiB'
    TOKENS = 'tokens'
    MS = 'ms'


class GateResult(FrozenModel):
    gate: GateId
    verdict: GateVerdict
    value: float | None = None
    bound: float | None = None
    margin: float | None = None
    unit: GateUnit | None = None
    detail: str

    @model_validator(mode='after')
    def _margin_is_bound_minus_value(self) -> Self:
        derived = None if self.value is None or self.bound is None else self.bound - self.value
        if self.margin != derived:
            raise ValueError(
                f'{self.gate.value}: margin {self.margin} must be bound {self.bound} - value '
                f'{self.value} = {derived}'
            )
        return self


def _gate(gate: GateId, verdict: GateVerdict, detail: str, **measured: object) -> GateResult:
    fields = {'gate': gate, 'verdict': verdict, 'detail': detail, **measured}
    return GateResult.model_validate(fields)


def _measured(
    gate: GateId, passed: bool, value: float, bound: float, unit: GateUnit, detail: str
) -> GateResult:
    verdict = GateVerdict.PASS if passed else GateVerdict.FAIL
    return _gate(
        gate, verdict, detail, value=value, bound=bound, margin=bound - value, unit=unit
    )


def _unserved_detail(evidence: ArmEvidence) -> str:
    for stage in (ScreeningStage.START, ScreeningStage.WAIT_READY):
        record = evidence.commands.record_of(stage)
        if record is None:
            return f'arm not served: no {stage.value} was recorded, so nothing was measured'
        if record.exit_code != 0:
            return (
                f'arm not served: {stage.value} exited {record.exit_code}, '
                'so nothing was measured'
            )
    return 'arm not served, so nothing was measured'


def _unserved(gate: GateId, evidence: ArmEvidence) -> GateResult:
    return _gate(gate, GateVerdict.UNMEASURED, _unserved_detail(evidence))


def conformance_gate(evidence: ArmEvidence) -> GateResult:
    gate = GateId.CONFORMANCE_SMOKE
    if not evidence.served:
        return _unserved(gate, evidence)
    smoke = evidence.commands.record_of(ScreeningStage.SMOKE)
    if smoke is None:
        return _gate(gate, GateVerdict.UNMEASURED, 'no smoke was recorded')
    if smoke.exit_code == 0:
        return _gate(gate, GateVerdict.PASS, 'harness.py smoke --arm-spec exited 0')
    return _gate(
        gate,
        GateVerdict.FAIL,
        f'harness.py smoke --arm-spec exited {smoke.exit_code}: {smoke.output_tail}',
    )


def vram_gate(evidence: ArmEvidence) -> GateResult:
    gate = GateId.VRAM_FIT
    if not evidence.served:
        return _unserved(gate, evidence)
    if evidence.vram is None:
        return _gate(gate, GateVerdict.UNMEASURED, 'no VRAM reading was taken')
    reading = evidence.vram.vram
    return _measured(
        gate,
        reading.verdict == 'PASS',
        reading.arm_footprint_mib,
        reading.budget_mib,
        GateUnit.MIB,
        f'lms_vram.evaluate_budget {reading.verdict}: {reading.reason}',
    )


def _unreported_detail(
    model_id: str, unreported: Sequence[CallRecord], own: Sequence[CallRecord]
) -> str:
    kinds = sorted(Counter((call.status, call.error_excerpt or '') for call in unreported).items())
    listed = '; '.join(
        f'{count}x status {status}: {excerpt}'
        for (status, excerpt), count in kinds[:_LISTED_REJECTIONS]
    )
    reported = [call.prompt_tokens for call in own if call.prompt_tokens is not None]
    longest = f'{max(reported)} tokens' if reported else 'none'
    return (
        f'{len(unreported)} of {len(own)} {model_id!r} calls report no prompt length, so the '
        f'longest prompt is unknown (longest of the {len(reported)} reported: {longest}): '
        f'{listed}'
    )


def context_gate(evidence: ArmEvidence) -> GateResult:
    gate = GateId.CONTEXT_FIT
    if not evidence.served:
        return _unserved(gate, evidence)
    model_id = evidence.spec.model_id
    bound = evidence.arm.max_model_len - evidence.spec.params.max_tokens
    own = evidence.own_model_calls
    unreported = [call for call in own if not call.succeeded or call.prompt_tokens is None]
    if unreported:
        detail = _unreported_detail(model_id, unreported, own)
        return _gate(gate, GateVerdict.UNMEASURED, detail, bound=bound, unit=GateUnit.TOKENS)
    if not own:
        detail = f'no {model_id!r} call reported a prompt length'
        return _gate(gate, GateVerdict.UNMEASURED, detail, bound=bound, unit=GateUnit.TOKENS)
    longest = max(call.prompt_tokens or 0 for call in own)
    return _measured(
        gate,
        longest <= bound,
        longest,
        bound,
        GateUnit.TOKENS,
        f'longest of {len(own)} server-reported prompts against max_model_len '
        f'{evidence.arm.max_model_len} - max_tokens {evidence.spec.params.max_tokens}',
    )


def throughput_gate(evidence: ArmEvidence, envelope: LatencyEnvelope) -> GateResult:
    gate = GateId.THROUGHPUT_FLOOR
    if not evidence.served:
        return _unserved(gate, evidence)
    bound = envelope.p95_bound_ms
    p95 = evidence.metric_value(LlmMetricId.EPISODE_LATENCY_P95)
    if p95 is None:
        ok = sum(1 for outcome in evidence.outcomes if outcome.ok)
        fastest = min((outcome.duration_ms for outcome in evidence.outcomes), default=None)
        fastest_text = 'none attempted' if fastest is None else f'{fastest:.0f} ms'
        detail = (
            f'no ok episode, so no warm p95 under load: {ok}/{len(evidence.outcomes)} '
            f'episodes ok; fastest attempted {fastest_text}'
        )
        return _gate(gate, GateVerdict.FAIL, detail, bound=bound, unit=GateUnit.MS)
    return _measured(
        gate,
        envelope.admits(p95),
        p95,
        bound,
        GateUnit.MS,
        f'warm episode-latency p95 under load, strictly below {bound:.0f} ms to pass',
    )


class ArmScreening(FrozenModel):
    arm_id: ArmId
    stack: LocalLlmStack
    reasoning: ReasoningMode
    served: bool
    gates: tuple[GateResult, ...]
    survives: bool
    reported: ReportedEvidence

    @model_validator(mode='after')
    def _survival_is_the_four_gates(self) -> Self:
        gate_ids = tuple(gate.gate for gate in self.gates)
        if gate_ids != tuple(GateId):
            raise ValueError(
                f'arm {self.arm_id!r}: gates must be exactly {[g.value for g in GateId]} in '
                f'order, got {[g.value for g in gate_ids]}'
            )
        passed = all(gate.verdict is GateVerdict.PASS for gate in self.gates)
        if self.survives != passed:
            raise ValueError(
                f'arm {self.arm_id!r}: survives {self.survives} must be {passed}, whether all '
                'four gates PASS'
            )
        return self


def screen_arm(
    evidence: ArmEvidence,
    envelope: LatencyEnvelope,
    reference_outcomes: Sequence[EpisodeOutcome],
) -> ArmScreening:
    gates = (
        conformance_gate(evidence),
        vram_gate(evidence),
        context_gate(evidence),
        throughput_gate(evidence, envelope),
    )
    return ArmScreening(
        arm_id=evidence.arm.arm_id,
        stack=evidence.arm.stack,
        reasoning=evidence.arm.reasoning,
        served=evidence.served,
        gates=gates,
        survives=all(gate.verdict is GateVerdict.PASS for gate in gates),
        reported=reported_evidence(evidence, reference_outcomes),
    )


class ScreeningCap(FrozenModel):
    cap: int = Field(ge=1)
    candidates: int = Field(ge=0)
    binds: bool

    @model_validator(mode='after')
    def _binds_iff_candidates_exceed_cap(self) -> Self:
        if self.binds != (self.candidates > self.cap):
            raise ValueError(
                f'binds {self.binds} must say whether {self.candidates} candidates exceed the '
                f'cap of {self.cap}'
            )
        return self


class ScreeningOutcome(StrEnum):
    PROCEED_TO_THETA = 'proceed-to-theta'
    NEGATIVE_VERDICT = 'negative-verdict'


class ScreeningVerdict(FrozenModel):
    schema_version: Literal[1]
    preregistration_sha: GitSha
    code_sha: GitSha
    corpus_sha: ContentSha
    envelope: LatencyEnvelope
    cap: ScreeningCap
    arms: tuple[ArmScreening, ...]
    survivors: tuple[ArmId, ...]
    outcome: ScreeningOutcome

    @model_validator(mode='after')
    def _survivors_and_outcome_follow_the_arms(self) -> Self:
        surviving = tuple(arm.arm_id for arm in self.arms if arm.survives)
        if self.survivors != surviving:
            raise ValueError(
                f'survivors {list(self.survivors)} must be the surviving arms {list(surviving)}'
            )
        if self.cap.candidates != len(self.arms):
            raise ValueError(
                f'cap.candidates {self.cap.candidates} must count the {len(self.arms)} arms'
            )
        expected = (
            ScreeningOutcome.PROCEED_TO_THETA if surviving else ScreeningOutcome.NEGATIVE_VERDICT
        )
        if self.outcome is not expected:
            raise ValueError(
                f'outcome {self.outcome.value!r} must be {expected.value!r} for '
                f'{len(surviving)} survivor(s)'
            )
        return self


def _require_evidence_is_slate(
    slate: Sequence[SlateArm], evidence_by_arm: Mapping[str, ArmEvidence]
) -> None:
    slate_ids = {arm.arm_id for arm in slate}
    missing = sorted(slate_ids - set(evidence_by_arm))
    extra = sorted(set(evidence_by_arm) - slate_ids)
    if missing or extra:
        raise ScreeningEvidenceError(
            f'the screening evidence must cover exactly the slate: missing {missing}, '
            f'extra {extra}'
        )


def _require_one(name: str, values: Iterable[str | None]) -> str:
    distinct = sorted({str(value) for value in values})
    if len(distinct) != 1:
        raise ScreeningEvidenceError(f'the screened arms must share one {name}; got {distinct}')
    return distinct[0]


def _require_matches_controls(name: str, screened: str, controls: str) -> None:
    if screened != controls:
        raise ScreeningEvidenceError(
            f'the screened arms ran at {name} {screened} but the controls at {controls}: '
            f'prereg §1 pins candidates to the controls\' {name}'
        )


def _require_cap_admits(arms: Sequence[ArmScreening]) -> None:
    passers = [arm.arm_id for arm in arms if arm.survives]
    if len(passers) > SURVIVOR_CAP:
        raise ScreeningEvidenceError(
            f'{len(passers)} arms pass all four gates {passers} but at most {SURVIVOR_CAP} '
            'advance, and no ranking rule is pre-registered'
        )


def derive_screening_verdict(
    slate: Sequence[SlateArm],
    evidence_by_arm: Mapping[str, ArmEvidence],
    inputs: PreregistrationInputs,
    reference_outcomes: Sequence[EpisodeOutcome],
) -> ScreeningVerdict:
    _require_evidence_is_slate(slate, evidence_by_arm)
    evidence = tuple(evidence_by_arm[arm.arm_id] for arm in slate)
    code_sha = _require_one('code_sha', (e.spec.code_sha for e in evidence))
    _require_matches_controls('code_sha', code_sha, inputs.code_sha)
    corpus_sha = _require_one('corpus_sha', (e.spec.corpus_sha for e in evidence))
    _require_matches_controls('corpus_sha', corpus_sha, inputs.corpus_sha)
    preregistration_sha = _require_one(
        'preregistration_sha', (e.spec.preregistration_sha for e in evidence)
    )
    arms = tuple(screen_arm(e, inputs.envelope, reference_outcomes) for e in evidence)
    _require_cap_admits(arms)
    survivors = tuple(arm.arm_id for arm in arms if arm.survives)
    outcome = (
        ScreeningOutcome.PROCEED_TO_THETA if survivors else ScreeningOutcome.NEGATIVE_VERDICT
    )
    return ScreeningVerdict(
        schema_version=SCREENING_VERDICT_SCHEMA_VERSION,
        preregistration_sha=preregistration_sha,
        code_sha=code_sha,
        corpus_sha=corpus_sha,
        envelope=inputs.envelope,
        cap=ScreeningCap(cap=SURVIVOR_CAP, candidates=len(arms), binds=len(arms) > SURVIVOR_CAP),
        arms=arms,
        survivors=survivors,
        outcome=outcome,
    )


def serialize_screening_verdict(verdict: ScreeningVerdict) -> str:
    return canonical_json_text(verdict.model_dump(mode='json'))


def load_screening_verdict(path: Path | str) -> ScreeningVerdict:
    return ScreeningVerdict.model_validate_json(Path(path).read_text())
