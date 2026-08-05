"""Tests for lms_manifest — the arms.yaml contract surface (task 3713, LME-alpha).

PRD-MARKER:local-memory-models-eval serving

`arms.yaml` is the single contract every tool in this directory and every
downstream consumer (eta screening, theta full runs, iota embedding arms)
derives from.  So its loader validates loudly and typed: a malformed or
self-contradictory manifest must raise `ArmManifestError`, never fall back to a
default or silently drop an arm.  A silently-dropped arm is the worst failure
mode available here — the health report would show every *remaining* arm green
and the eval would quietly run a narrower slate than the PRD commissioned.

Fixture manifests are built inline under tmp_path.  ONLY the slate-coverage
tests read the committed `arms.yaml`, so a validation-rule test can never be
accidentally greened (or reddened) by an unrelated edit to the real manifest.
"""
from pathlib import Path

import pytest
import yaml

import lms_manifest

_COMMITTED_MANIFEST = (
    Path(__file__).resolve().parents[1] / 'local-model-serving' / 'arms.yaml'
)

# The PRD slate, transcribed from plans/local-memory-models-eval-prd.md lines
# 122-137.  Held here as literals rather than derived from the manifest, so this
# file is an independent statement of what the PRD commissioned: deriving the
# expectation from the artifact under test would make the assertion vacuous.
_PRD_DENSE_LLM_ARMS = {'qwen3.5-9b', 'mistral-small-3.2-24b', 'phi-4-14b'}
_PRD_MOE_ARM = 'moe-stretch'
_PRD_EMBEDDING_DIMS = {
    'qwen3-embedding-0.6b': 1024,
    'granite-embedding-english-r2': 768,
    'qwen3-embedding-4b': 2560,
    'gte-modernbert-base': 768,
}
# PRD line 134: the Qwen3-Embedding family requires a query-side instruct
# prefix; granite and gte take none.  A dropped prefix silently degrades every
# retrieval number iota later reports, so the manifest must carry it.
_PRD_PREFIXED_EMBEDDING_ARMS = {'qwen3-embedding-0.6b', 'qwen3-embedding-4b'}


# ---------------------------------------------------------------------------
# inline fixture builders
# ---------------------------------------------------------------------------


def _llm_arm(**overrides):
    arm = {
        'arm_id': 'demo-llm',
        'axis': 'llm',
        'stack': 'vllm',
        'image': 'vllm/vllm-openai:v0.11.0',
        'model_ref': 'Qwen/Qwen3.5-9B-AWQ',
        'quant': 'awq',
        'port': 8410,
        'served_model_name': 'demo-llm',
        'structured_output_mode': 'json_schema',
        'est_vram_gib': 6.0,
        'max_model_len': 32768,
    }
    arm.update(overrides)
    return arm


def _embedding_arm(**overrides):
    arm = {
        'arm_id': 'demo-embed',
        'axis': 'embedding',
        'stack': 'vllm',
        'image': 'vllm/vllm-openai:v0.11.0',
        'model_ref': 'ibm-granite/granite-embedding-english-r2',
        'quant': 'none',
        'port': 8414,
        'served_model_name': 'demo-embed',
        'structured_output_mode': 'none',
        'est_vram_gib': 1.0,
        'max_model_len': 8192,
        'dims': 768,
    }
    arm.update(overrides)
    return arm


def _write_manifest(tmp_path, arms, **top_level):
    payload = {'port_block': [8410, 8417], 'arms': arms}
    payload.update(top_level)
    path = tmp_path / 'arms.yaml'
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


# ---------------------------------------------------------------------------
# happy path: typed records with the fields the downstream tasks consume
# ---------------------------------------------------------------------------


def test_load_arms_returns_typed_records_with_the_full_field_set(tmp_path):
    path = _write_manifest(tmp_path, [_llm_arm(), _embedding_arm()])

    manifest = lms_manifest.load_arms(path)

    llm = manifest.by_id('demo-llm')
    assert llm.arm_id == 'demo-llm'
    assert llm.axis == 'llm'
    assert llm.stack == 'vllm'
    assert llm.image == 'vllm/vllm-openai:v0.11.0'
    assert llm.model_ref == 'Qwen/Qwen3.5-9B-AWQ'
    assert llm.quant == 'awq'
    assert llm.port == 8410
    assert llm.served_model_name == 'demo-llm'
    assert llm.structured_output_mode == 'json_schema'
    assert llm.est_vram_gib == pytest.approx(6.0)
    assert llm.max_model_len == 32768
    assert llm.dims is None
    assert llm.query_prefix is None

    embed = manifest.by_id('demo-embed')
    assert embed.axis == 'embedding'
    assert embed.dims == 768


def test_base_url_binds_127_0_0_1_never_localhost(tmp_path):
    """127.0.0.1 explicitly: `localhost` can resolve to ::1 while the server
    listens on IPv4 only (scripts/run_vllm_eval.py:505-512)."""
    path = _write_manifest(tmp_path, [_llm_arm(port=8413)])

    arm = lms_manifest.load_arms(path).by_id('demo-llm')

    assert arm.base_url == 'http://127.0.0.1:8413'
    assert 'localhost' not in arm.base_url


def test_query_prefix_is_optional_and_round_trips(tmp_path):
    path = _write_manifest(
        tmp_path,
        [_embedding_arm(query_prefix='Instruct: retrieve\nQuery: ')],
    )

    arm = lms_manifest.load_arms(path).by_id('demo-embed')

    assert arm.query_prefix == 'Instruct: retrieve\nQuery: '


# ---------------------------------------------------------------------------
# accessors used by eta / theta / iota
# ---------------------------------------------------------------------------


def test_by_id_returns_the_arm_and_raises_for_an_unknown_id(tmp_path):
    path = _write_manifest(tmp_path, [_llm_arm(), _embedding_arm()])
    manifest = lms_manifest.load_arms(path)

    assert manifest.by_id('demo-embed').arm_id == 'demo-embed'

    with pytest.raises(lms_manifest.ArmManifestError) as excinfo:
        manifest.by_id('no-such-arm')
    assert 'no-such-arm' in str(excinfo.value)


def test_by_axis_partitions_the_slate_and_is_empty_for_an_unused_axis(tmp_path):
    path = _write_manifest(
        tmp_path,
        [_llm_arm(), _llm_arm(arm_id='demo-llm-2', port=8411,
                              served_model_name='demo-llm-2'),
         _embedding_arm()],
    )
    manifest = lms_manifest.load_arms(path)

    assert [a.arm_id for a in manifest.by_axis('llm')] == ['demo-llm', 'demo-llm-2']
    assert [a.arm_id for a in manifest.by_axis('embedding')] == ['demo-embed']

    with pytest.raises(lms_manifest.ArmManifestError):
        manifest.by_axis('not-an-axis')


def test_arm_ids_preserves_manifest_order(tmp_path):
    path = _write_manifest(
        tmp_path,
        [_embedding_arm(), _llm_arm()],
    )

    assert lms_manifest.load_arms(path).arm_ids() == ['demo-embed', 'demo-llm']


# ---------------------------------------------------------------------------
# loud typed rejections — never a silent skip, never a default
# ---------------------------------------------------------------------------


def test_duplicate_arm_id_raises(tmp_path):
    path = _write_manifest(tmp_path, [_llm_arm(), _llm_arm(port=8411)])

    with pytest.raises(lms_manifest.ArmManifestError) as excinfo:
        lms_manifest.load_arms(path)
    assert 'demo-llm' in str(excinfo.value)


def test_duplicate_port_raises(tmp_path):
    """Two arms on one port is the 2026-04-08 404 bug's precondition
    (scripts/run_vllm_eval.py:541-553): a probe lands on whichever unit
    happens to hold the port and a DIFFERENT model answers."""
    path = _write_manifest(
        tmp_path,
        [_llm_arm(), _llm_arm(arm_id='demo-llm-2',
                              served_model_name='demo-llm-2')],
    )

    with pytest.raises(lms_manifest.ArmManifestError) as excinfo:
        lms_manifest.load_arms(path)
    assert '8410' in str(excinfo.value)


def test_unknown_stack_raises(tmp_path):
    path = _write_manifest(tmp_path, [_llm_arm(stack='ollama')])

    with pytest.raises(lms_manifest.ArmManifestError) as excinfo:
        lms_manifest.load_arms(path)
    assert 'ollama' in str(excinfo.value)


def test_unknown_axis_raises(tmp_path):
    path = _write_manifest(tmp_path, [_llm_arm(axis='reranker')])

    with pytest.raises(lms_manifest.ArmManifestError) as excinfo:
        lms_manifest.load_arms(path)
    assert 'reranker' in str(excinfo.value)


def test_embedding_arm_without_dims_raises(tmp_path):
    """iota compares arms by retrieval quality at their native dims; an arm with
    no declared dims cannot be checked against what the server actually returns,
    so the dimensionality mismatch would surface as a quality regression."""
    arm = _embedding_arm()
    del arm['dims']
    path = _write_manifest(tmp_path, [arm])

    with pytest.raises(lms_manifest.ArmManifestError) as excinfo:
        lms_manifest.load_arms(path)
    assert 'dims' in str(excinfo.value)


def test_llamacpp_arm_declaring_json_schema_raises(tmp_path):
    """ggml-org/llama.cpp#21228: llama.cpp silently falls back to UNCONSTRAINED
    output on a pydantic $ref/$defs schema.  A manifest claiming json_schema for
    an llamacpp arm therefore asserts a capability that does not exist, and the
    eval would credit constrained decoding it never had."""
    path = _write_manifest(
        tmp_path,
        [_llm_arm(stack='llamacpp', structured_output_mode='json_schema')],
    )

    with pytest.raises(lms_manifest.ArmManifestError) as excinfo:
        lms_manifest.load_arms(path)
    message = str(excinfo.value)
    assert 'llamacpp' in message
    assert '21228' in message


def test_llamacpp_arm_with_json_object_is_accepted(tmp_path):
    path = _write_manifest(
        tmp_path,
        [_llm_arm(stack='llamacpp', structured_output_mode='json_object',
                  quant='iq4_xs')],
    )

    arm = lms_manifest.load_arms(path).by_id('demo-llm')

    assert arm.stack == 'llamacpp'
    assert arm.structured_output_mode == 'json_object'


@pytest.mark.parametrize(
    'missing',
    ['arm_id', 'axis', 'stack', 'image', 'model_ref', 'port',
     'served_model_name', 'structured_output_mode', 'est_vram_gib'],
)
def test_missing_required_field_raises(tmp_path, missing):
    arm = _llm_arm()
    del arm[missing]
    path = _write_manifest(tmp_path, [arm])

    with pytest.raises(lms_manifest.ArmManifestError) as excinfo:
        lms_manifest.load_arms(path)
    assert missing in str(excinfo.value)


def test_missing_file_raises_typed_error(tmp_path):
    with pytest.raises(lms_manifest.ArmManifestError) as excinfo:
        lms_manifest.load_arms(tmp_path / 'absent.yaml')
    assert 'absent.yaml' in str(excinfo.value)


def test_empty_arms_list_raises(tmp_path):
    """A manifest with no arms would make every downstream sweep pass
    vacuously — the health check would report zero failures over zero arms."""
    path = _write_manifest(tmp_path, [])

    with pytest.raises(lms_manifest.ArmManifestError):
        lms_manifest.load_arms(path)


def test_non_mapping_document_raises(tmp_path):
    path = tmp_path / 'arms.yaml'
    path.write_text('- just\n- a\n- list\n')

    with pytest.raises(lms_manifest.ArmManifestError):
        lms_manifest.load_arms(path)


def test_unknown_field_raises_rather_than_being_ignored(tmp_path):
    """A typo'd key (`quantization:` for `quant:`) must not be silently
    dropped — that reads as "the default applied" and mis-serves the arm."""
    path = _write_manifest(tmp_path, [_llm_arm(quantisation='awq')])

    with pytest.raises(lms_manifest.ArmManifestError) as excinfo:
        lms_manifest.load_arms(path)
    assert 'quantisation' in str(excinfo.value)


# ---------------------------------------------------------------------------
# the committed manifest covers exactly the PRD slate
# ---------------------------------------------------------------------------


def test_committed_manifest_loads():
    assert _COMMITTED_MANIFEST.exists(), _COMMITTED_MANIFEST
    manifest = lms_manifest.load_arms(_COMMITTED_MANIFEST)
    assert len(manifest.arms) == 8


def test_committed_manifest_covers_the_prd_llm_slate():
    manifest = lms_manifest.load_arms(_COMMITTED_MANIFEST)
    llm_arms = manifest.by_axis('llm')

    assert {a.arm_id for a in llm_arms} == _PRD_DENSE_LLM_ARMS | {_PRD_MOE_ARM}

    dense = [a for a in llm_arms if a.arm_id in _PRD_DENSE_LLM_ARMS]
    assert len(dense) == 3
    assert all(a.stack == 'vllm' for a in dense)
    assert all(a.structured_output_mode == 'json_schema' for a in dense)

    moe = manifest.by_id(_PRD_MOE_ARM)
    assert moe.stack == 'llamacpp'
    assert moe.structured_output_mode == 'json_object'


def test_committed_manifest_covers_the_prd_embedding_slate():
    manifest = lms_manifest.load_arms(_COMMITTED_MANIFEST)
    embedding_arms = manifest.by_axis('embedding')

    assert {a.arm_id for a in embedding_arms} == set(_PRD_EMBEDDING_DIMS)
    assert {a.arm_id: a.dims for a in embedding_arms} == _PRD_EMBEDDING_DIMS
    assert {
        a.arm_id for a in embedding_arms if a.query_prefix
    } == _PRD_PREFIXED_EMBEDDING_ARMS


def test_committed_manifest_ports_are_unique_and_inside_the_reserved_block():
    manifest = lms_manifest.load_arms(_COMMITTED_MANIFEST)
    ports = [a.port for a in manifest.arms]

    assert len(set(ports)) == len(ports)
    low, high = manifest.port_block
    assert all(low <= p <= high for p in ports)
    # Ports already in service on this host: 8002 fused-memory, 8102 escalation.
    assert not (low <= 8002 <= high)
    assert not (low <= 8102 <= high)


def test_committed_manifest_arms_declare_a_footprint_the_3090_could_hold():
    """24576 MiB total, so any single arm declaring more than the whole card is
    a transcription error, not a stretch goal."""
    manifest = lms_manifest.load_arms(_COMMITTED_MANIFEST)

    assert all(0 < a.est_vram_gib < 24.0 for a in manifest.arms)
