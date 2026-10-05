"""Shared arm-harness test doubles: valid spec builders and public-Protocol fakes."""

import asyncio
import dataclasses
import subprocess
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fused_memory.arm_harness.arm_spec import EmbeddingArmSpec, LlmArmSpec
from fused_memory.arm_harness.instrument_checks import PREREGISTRATION_DOC_PATH
from fused_memory.arm_harness.run_manifest import RunManifest
from fused_memory.backends.llm_token_usage import (
    AttributingTokenUsageTracker,
    TokenMeasurement,
    measure_llm_tokens,
)

CODE_SHA = 'a' * 40
CORPUS_SHA = 'b' * 64
PREREG_SHA = 'c' * 40
UNREACHABLE_BASE_URL = 'http://127.0.0.1:9/v1'
PROTECTED_GRAPHS = (
    'dark_factory',
    'reify',
    'know_live',
    'solar_challenge_platform',
    'autopilot_video',
    'pump_web_ui',
    'my_solar_challenge',
    'probe_e1_master',
    '_probe',
)
"""The live graphs of plans/local-memory-models-eval-prd.md §Hazards, as test inputs only."""


def llm_spec(*, base_url: str = UNREACHABLE_BASE_URL, **overrides) -> LlmArmSpec:
    """A local-stack candidate LLM arm; override any field by keyword."""
    data = {
        'arm_id': 'qwen3-8b-vllm',
        'axis': 'llm',
        'model_id': 'qwen3-8b',
        'serving': {'stack': 'vllm', 'base_url': base_url},
        'client_class': 'openai_generic',
        'structured_output_mode': 'json_schema',
        'params': {'temperature': 0.0, 'max_tokens': 4096},
        'pricing': None,
        'code_sha': CODE_SHA,
        'corpus_sha': CORPUS_SHA,
        'preregistration_sha': PREREG_SHA,
        'scratch_group_id': 'evalmem_qwen3_8b',
        'arm_role': 'candidate',
    }
    return LlmArmSpec.model_validate(data | overrides)


def incumbent_control_spec(**overrides) -> LlmArmSpec:
    """The metered incumbent control arm (stack 'openai', priced, no prereg sha)."""
    data = {
        'arm_id': 'incumbent-ctrl-a',
        'model_id': 'gpt-4.1-mini',
        'serving': {'stack': 'openai', 'base_url': 'https://api.openai.com/v1'},
        'pricing': {'usd_per_mtok_input': 0.4, 'usd_per_mtok_output': 1.6},
        'preregistration_sha': None,
        'scratch_group_id': 'evalmem_ctrl_a',
        'arm_role': 'control',
    }
    return llm_spec(**(data | overrides))


def embedding_spec(*, base_url: str = UNREACHABLE_BASE_URL, **overrides) -> EmbeddingArmSpec:
    """A local-stack candidate embedding arm; override any field by keyword."""
    data = {
        'arm_id': 'bge-m3',
        'axis': 'embedding',
        'model_id': 'BAAI/bge-m3',
        'serving': {'stack': 'tei', 'base_url': base_url},
        'embedding_dim': 1024,
        'code_sha': CODE_SHA,
        'corpus_sha': CORPUS_SHA,
        'preregistration_sha': PREREG_SHA,
        'scratch_group_id': 'evalmem_bge_m3',
        'arm_role': 'candidate',
    }
    return EmbeddingArmSpec.model_validate(data | overrides)


STARTED_AT = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
FINISHED_AT = datetime(2026, 10, 5, 13, 30, tzinfo=UTC)


def run_manifest_for(spec: LlmArmSpec | EmbeddingArmSpec, **overrides) -> RunManifest:
    """A complete run of ``spec`` over three episodes; override any field by keyword."""
    data = {
        'schema_version': 1,
        'spec': spec,
        'settings_summary': {
            'concurrency': 4,
            'index_configuration': 'with-indices',
            'episode_timeout_s': 120.0,
        },
        'effective_embedder': {'model': 'text-embedding-3-small', 'dimensions': 1536},
        'graphiti_max_coroutines': 5,
        'graphiti_semaphore_limit': 20,
        'episode_ids': ('e1', 'e2', 'e3'),
        'incomplete': False,
        'abort': None,
        'check_results': (),
        'started_at': STARTED_AT,
        'finished_at': FINISHED_AT,
    }
    return RunManifest.model_validate(data | overrides)


def fake_add_result(
    episode_uuid: str,
    entity_names: tuple[str, ...] = (),
    edges: tuple[tuple[str, str, str], ...] = (),
) -> SimpleNamespace:
    """A minimal AddEpisodeResults-shaped value: episode.uuid, nodes, edges (by node name)."""
    nodes = [SimpleNamespace(uuid=f'node-{name}', name=name) for name in entity_names]
    entity_edges = [
        SimpleNamespace(
            uuid=f'edge-{source}-{relation}-{target}',
            source_node_uuid=f'node-{source}',
            target_node_uuid=f'node-{target}',
            name=relation,
            episodes=[episode_uuid],
        )
        for source, relation, target in edges
    ]
    return SimpleNamespace(
        episode=SimpleNamespace(uuid=episode_uuid), nodes=nodes, edges=entity_edges
    )


Behaviour = Callable[[dict[str, Any]], Awaitable[Any]]


async def succeed(call: dict[str, Any]) -> Any:
    return fake_add_result(f'replay-{call["name"]}', entity_names=('alice', 'bob'))


async def fail(call: dict[str, Any]) -> Any:
    raise RuntimeError(f'extraction failed for {call["name"]}')


async def hang(call: dict[str, Any]) -> Any:
    await asyncio.Event().wait()


def slow(behaviour: Behaviour, seconds: float) -> Behaviour:
    async def delayed(call: dict[str, Any]) -> Any:
        await asyncio.sleep(seconds)
        return await behaviour(call)

    return delayed


class FakeArmGraph:
    """A public-``ArmGraph`` double: per-episode-name behaviours, recorded calls, overlap.

    ``usage`` is recorded on an attributing tracker inside each add_episode, so the
    real ``measure_llm_tokens`` window credits it to that episode.
    """

    def __init__(
        self,
        behaviours: Mapping[str, Behaviour] | None = None,
        *,
        default: Behaviour = succeed,
        usage: tuple[int, int] | None = (30, 10),
        llm_client: Any = None,
        search_results: Callable[[str], list[Any]] | None = None,
    ) -> None:
        self._behaviours = dict(behaviours or {})
        self._default = default
        self._usage = usage
        self.llm_client = llm_client or SimpleNamespace(
            token_tracker=AttributingTokenUsageTracker()
        )
        self._search_results = search_results or (lambda query: [])
        self.add_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.events: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def add_episode(self, **call: Any) -> Any:
        self.add_calls.append(call)
        self.events.append('add_episode')
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            result = await self._behaviours.get(call['name'], self._default)(call)
            if self._usage is not None:
                self.llm_client.token_tracker.record('extract_nodes', *self._usage)
            return result
        finally:
            self.in_flight -= 1

    async def search(
        self, query: str, group_ids: list[str] | None = None, num_results: int = 10
    ) -> list[Any]:
        self.search_calls.append(
            {'query': query, 'group_ids': group_ids, 'num_results': num_results}
        )
        self.events.append('search')
        return self._search_results(query)[:num_results]

    def token_probe(self) -> AbstractAsyncContextManager[TokenMeasurement]:
        return measure_llm_tokens(self.llm_client)


class RecordingJournal:
    """A journal double recording each log call's keyword arguments, in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def log_write_op(self, **kwargs: Any) -> None:
        self.calls.append(('log_write_op', kwargs))

    async def log_backend_op(self, **kwargs: Any) -> None:
        self.calls.append(('log_backend_op', kwargs))


def git(root: Path, *args: str) -> str:
    """Run real git in ``root`` with a throwaway identity; returns stripped stdout."""
    completed = subprocess.run(
        ['git', '-C', str(root), '-c', 'user.name=T', '-c', 'user.email=t@e.example',
         '-c', 'commit.gpgsign=false', *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@dataclasses.dataclass(frozen=True)
class PreregRepo:
    root: Path
    without_prereg: str
    with_prereg: str


def make_prereg_repo(root: Path) -> PreregRepo:
    """A real two-commit repo at ``root``: the second (HEAD) commit adds the preregistration doc."""
    root.mkdir(parents=True)
    git(root, 'init', '-q', '-b', 'main')
    (root / 'seed.txt').write_text('seed\n')
    git(root, 'add', '-A')
    git(root, 'commit', '-q', '--no-verify', '-m', 'seed')
    without_prereg = git(root, 'rev-parse', 'HEAD')
    doc = root / PREREGISTRATION_DOC_PATH
    doc.parent.mkdir(parents=True)
    doc.write_text('# preregistration\n')
    git(root, 'add', '-A')
    git(root, 'commit', '-q', '--no-verify', '-m', 'prereg')
    with_prereg = git(root, 'rev-parse', 'HEAD')
    return PreregRepo(root=root, without_prereg=without_prereg, with_prereg=with_prereg)
