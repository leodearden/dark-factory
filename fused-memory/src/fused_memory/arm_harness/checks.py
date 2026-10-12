"""Runnable serving-dependent checks: endpoint smoke, index-configuration reality, control variance.

Each is a thin composition of the lower harness modules; the CLI runs them against
live serving (η, ι, ζ). The smoke always carries a negative control, so a validator
that accepts everything can never report a passing endpoint.
"""

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from graphiti_core.llm_client import LLMClient
from graphiti_core.prompts.models import Message
from pydantic import BaseModel, ValidationError
from redis.exceptions import ResponseError

from fused_memory.arm_harness.arm_backend import audited_arm_client
from fused_memory.arm_harness.arm_config import llm_arm_config
from fused_memory.arm_harness.arm_spec import LlmArmSpec
from fused_memory.arm_harness.comparison import (
    check_arm_config_symmetry,
    check_single_code_sha,
    require_comparable_runs,
)
from fused_memory.arm_harness.conformance import (
    ConformanceCounts,
    ResponseValidator,
    validate_response,
)
from fused_memory.arm_harness.instrument_checks import (
    CheckResult,
    InstrumentCheckId,
    check_failed,
    check_passed,
    check_reference_nonempty,
    check_token_cost_accounting,
)
from fused_memory.arm_harness.metrics_record import IndexConfiguration, MetricsRecord
from fused_memory.arm_harness.replay_types import EpisodeOutcome
from fused_memory.arm_harness.run_manifest import RunManifest
from fused_memory.arm_harness.scratch_guard import GuardCheckpoint, require_scratch_name
from fused_memory.config.schema import FusedMemoryConfig

# --- endpoint conformance smoke (boundary row 1) -------------------------------------


class SmokeEntity(BaseModel):
    name: str


class SmokeEntities(BaseModel):
    """Nested on purpose: its JSON schema carries ``$defs``/``$ref``, the llama.cpp #21228 trap."""

    entities: list[SmokeEntity]


SMOKE_MESSAGES: tuple[Message, ...] = (
    Message(role='system', content='Extract every named entity. Reply with JSON only.'),
    Message(role='user', content='Alice met Bob in Paris.'),
)


@dataclass(frozen=True)
class SmokeVerdict:
    positive: CheckResult
    negative_control: CheckResult
    counts: ConformanceCounts

    @property
    def passed(self) -> bool:
        return self.positive.passed and self.negative_control.passed


async def smoke_endpoint(
    spec: LlmArmSpec,
    base: FusedMemoryConfig,
    *,
    validator: ResponseValidator = validate_response,
) -> SmokeVerdict:
    """One schema-constrained request through the arm's audited client, plus a negative control."""
    client, ledger = audited_arm_client(spec, llm_arm_config(spec, base), validator=validator)
    positive = await _structured_response_probe(spec, client)
    return SmokeVerdict(
        positive=positive, negative_control=_negative_control(validator), counts=ledger.snapshot()
    )


async def _structured_response_probe(spec: LlmArmSpec, client: LLMClient) -> CheckResult:
    check_id = InstrumentCheckId.ENDPOINT_CONFORMANCE
    endpoint = (
        f'arm {spec.arm_id!r} at {spec.serving.base_url} ({spec.structured_output_mode})'
    )
    try:
        await client.generate_response(list(SMOKE_MESSAGES), response_model=SmokeEntities)
    except Exception as error:  # every failure is the verdict here, named by its class
        detail = f'{endpoint} gave no schema-valid response: {type(error).__name__}: {error}'
        return check_failed(check_id, detail, (type(error).__name__,))
    return check_passed(check_id, f'{endpoint} returned a schema-valid {SmokeEntities.__name__}')


def _negative_control(validator: ResponseValidator) -> CheckResult:
    check_id = InstrumentCheckId.VALIDATOR_NEGATIVE_CONTROL
    try:
        validator({'entities': [{'nom': 'Alice'}]}, SmokeEntities)
    except ValidationError:
        return check_passed(check_id, 'the validator rejected a deliberately off-schema payload')
    return check_failed(
        check_id,
        'the validator ACCEPTED a deliberately off-schema payload, so a passing smoke proves nothing',
        (),
    )


# --- index-configuration reality (boundary row 6) ------------------------------------

INDEX_PROBE_ATTEMPTS = 20
INDEX_PROBE_INTERVAL_S = 0.25
"""Twenty probes, nineteen sleeps between them: at most 4.75s spent waiting for the index."""

SEED_CYPHER = 'CREATE (:Entity {uuid: $uuid, name: $name})'
PROBE_CYPHER = "CALL db.idx.fulltext.queryNodes('Entity', $token) YIELD node RETURN node.uuid"
CLEANUP_CYPHER = 'MATCH (n:Entity {uuid: $uuid}) DELETE n'


class ProbeCleanupError(RuntimeError):
    """The index probe's seeded node could not be removed, so the scratch graph still holds it."""

    def __init__(self, graph_name: str, node_uuid: str) -> None:
        self.graph_name = graph_name
        self.node_uuid = node_uuid
        super().__init__(
            f'the index probe left Entity {node_uuid!r} in {graph_name!r}: its cleanup failed, '
            'so topology reads and retrieval on that graph will see it until it is deleted'
        )


class ProbeGraph(Protocol):
    """The slice of a falkordb ``AsyncGraph`` the index probe uses."""

    async def query(self, q: str, params: dict[str, Any] | None = None) -> Any: ...

    async def ro_query(self, q: str, params: dict[str, Any] | None = None) -> Any: ...


@dataclass(frozen=True)
class _ProbeAnswer:
    rows: int
    error: ResponseError | None


async def check_index_configuration(
    graph: ProbeGraph,
    graph_name: str,
    expected: IndexConfiguration,
    *,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
) -> CheckResult:
    """Whether the scratch graph's fulltext index exists in fact, as ``expected`` claims.

    The probe seeds one uniquely-named Entity (an ``evalmem_``-only write) and removes it;
    a failed removal raises ``ProbeCleanupError`` naming the node left behind.
    """
    require_scratch_name(graph_name, checkpoint=GuardCheckpoint.INDEX_PROBE)
    token = f'evalmemprobe{uuid.uuid4().hex}'
    node_uuid = str(uuid.uuid4())
    await graph.query(SEED_CYPHER, {'uuid': node_uuid, 'name': token})
    try:
        answer = await _probe_fulltext(graph, token, sleep)
    finally:
        await _remove_probe_node(graph, graph_name, node_uuid)
    return _index_verdict(graph_name, expected, answer)


async def _remove_probe_node(graph: ProbeGraph, graph_name: str, node_uuid: str) -> None:
    try:
        await graph.query(CLEANUP_CYPHER, {'uuid': node_uuid})
    except Exception as error:
        raise ProbeCleanupError(graph_name, node_uuid) from error


async def _probe_fulltext(
    graph: ProbeGraph, token: str, sleep: Callable[[float], Awaitable[object]]
) -> _ProbeAnswer:
    """Polls until a row arrives. A server-side ``ResponseError`` ends the poll.

    FalkorDB answers a fulltext query on an unindexed label with 0 rows, not an error
    (measured on graph module 4.18.0), so an error says nothing about the index.
    """
    rows = 0
    for attempt in range(INDEX_PROBE_ATTEMPTS):
        if attempt:
            await sleep(INDEX_PROBE_INTERVAL_S)
        try:
            response = await graph.ro_query(PROBE_CYPHER, {'token': token})
        except ResponseError as error:
            return _ProbeAnswer(rows=0, error=error)
        rows = len(response.result_set)
        if rows:
            break
    return _ProbeAnswer(rows=rows, error=None)


def _index_verdict(
    graph_name: str, expected: IndexConfiguration, answer: _ProbeAnswer
) -> CheckResult:
    check_id = InstrumentCheckId.INDEX_CONFIGURATION
    if answer.error is not None:
        detail = (
            f'fulltext probe on {graph_name!r} raised {type(answer.error).__name__}: '
            f'{answer.error}; a failed probe cannot confirm {expected.value}'
        )
        return check_failed(check_id, detail, (graph_name,))
    observed = f'fulltext probe on {graph_name!r} gave {answer.rows} row(s)'
    found = answer.rows > 0
    if expected is IndexConfiguration.WITH_INDICES and not found:
        detail = f'{observed}; the with-indices configuration does not differ in fact'
        return check_failed(check_id, detail, (graph_name,))
    if expected is IndexConfiguration.EMBEDDING_ONLY and found:
        detail = f'{observed}; the embedding-only configuration answers fulltext, so indices exist'
        return check_failed(check_id, detail, (graph_name,))
    return check_passed(check_id, f'{observed}, as {expected.value} expects')


# --- control variance (boundary row 4) -----------------------------------------------


def control_variance_check(
    runs: Sequence[RunManifest],
    records_by_arm: Mapping[str, Sequence[MetricsRecord]],
    *,
    reference: Sequence[EpisodeOutcome] | None = None,
) -> tuple[CheckResult, ...]:
    """Symmetry and one code sha across the control runs, token accounting per arm, and the reference.

    ``records_by_arm`` is keyed by arm id, so the runs must be comparable, one per arm,
    before any record is looked up (``RunComparabilityError`` otherwise).
    """
    require_comparable_runs(runs)
    per_arm = tuple(
        check_token_cost_accounting(_llm_spec(run), records_by_arm.get(run.spec.arm_id, ()))
        for run in runs
    )
    reference_checks = () if reference is None else (check_reference_nonempty(reference),)
    return (
        check_arm_config_symmetry(runs),
        check_single_code_sha(runs),
        *per_arm,
        *reference_checks,
    )


def _llm_spec(run: RunManifest) -> LlmArmSpec:
    if not isinstance(run.spec, LlmArmSpec):
        raise TypeError(f'arm {run.spec.arm_id!r} is a {run.spec.axis} arm; controls are LLM arms')
    return run.spec
