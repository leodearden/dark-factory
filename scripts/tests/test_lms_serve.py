"""Tests for lms_serve.build_launch_argv (task 3713, LME-alpha).

PRD-MARKER:local-memory-models-eval serving

Pure argv construction, one case per stack.  No container is started and no GPU
is touched: the GPU reading is injected, so these run anywhere.

Two assertions here are deliberately NEGATIVE, because the thing that must not
happen is what costs us:

  * `--gpu-memory-utilization` must never be the 0.95 pod-era default.  That
    figure came from dedicated 96 GB eval pods (docs/vllm-eval-status.md:1037);
    on this shared 24 GB card it would hand vLLM ~23 GiB and evict
    whisper-writer, which Leo requires resident (PRD D10).
  * the llamacpp arm's argv must carry NO grammar / json-schema / guided
    decoding flag.  llama.cpp silently falls back to unconstrained output on
    $ref/$defs schemas (ggml-org/llama.cpp#21228), so a constrained-decoding
    flag there would claim a capability that does not exist -- and the eval
    would credit it.  Encoding that structurally is the only way it survives a
    later well-meaning edit.
"""
import pytest

import lms_manifest
import lms_serve
import lms_vram

MEASURED_GPU = lms_vram.GpuReading(total_mib=24576, used_mib=7362, free_mib=16761)

_CONSTRAINED_DECODING_TOKENS = (
    'grammar', 'json-schema', 'json_schema', 'guided', 'gbnf',
)

_CHAT_ONLY_FLAGS = (
    '--chat-template', '--tool-call-parser', '--enable-auto-tool-choice',
    '--runner generate',
)


def _arm(**overrides):
    fields = {
        'arm_id': 'demo-llm',
        'axis': 'llm',
        'stack': 'vllm',
        'image': 'vllm/vllm-openai:v0.26.0',
        'model_ref': 'QuantTrio/Qwen3.5-9B-AWQ',
        'quant': 'awq',
        'port': 8410,
        'served_model_name': 'demo-llm',
        'structured_output_mode': 'json_schema',
        'est_vram_gib': 6.0,
        'max_model_len': 32768,
        'max_num_seqs': 8,
    }
    fields.update(overrides)
    return lms_manifest.ArmEntry(**fields)


def _embedding_arm(**overrides):
    return _arm(**{
        'arm_id': 'demo-embed',
        'axis': 'embedding',
        'model_ref': 'ibm-granite/granite-embedding-english-r2',
        'quant': 'none',
        'port': 8415,
        'served_model_name': 'demo-embed',
        'structured_output_mode': 'none',
        'est_vram_gib': 1.0,
        'max_model_len': 8192,
        'max_num_seqs': 32,
        'dims': 768,
        **overrides,
    })


def _llamacpp_arm(**overrides):
    return _arm(**{
        'arm_id': 'moe-stretch',
        'stack': 'llamacpp',
        'image': 'ghcr.io/ggml-org/llama.cpp:server-cuda',
        'model_ref': 'unsloth/Qwen3.6-35B-A3B-GGUF',
        'quant': 'iq4_xs',
        'port': 8413,
        'served_model_name': 'moe-stretch',
        'structured_output_mode': 'json_object',
        'est_vram_gib': 14.0,
        'max_model_len': 16384,
        'gguf_file': 'Qwen3.6-35B-A3B-IQ4_XS.gguf',
        **overrides,
    })


def _value_after(argv, flag):
    return argv[argv.index(flag) + 1]


# ---------------------------------------------------------------------------
# shared container shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'arm_factory', [_arm, _embedding_arm, _llamacpp_arm], ids=['vllm-llm', 'vllm-embed', 'llamacpp'],
)
def test_every_stack_builds_a_foreground_docker_run_with_explicit_gpu_access(arm_factory):
    arm = arm_factory()

    argv = lms_serve.build_launch_argv(arm, MEASURED_GPU)

    assert argv[:2] == ['docker', 'run']
    assert '--rm' in argv
    # Docker's default runtime here is runc (`docker info`: "Default Runtime:
    # runc"), so GPU access must be requested per-run.
    assert '--gpus' in argv
    assert _value_after(argv, '--gpus') == 'all'
    # NEVER detached: systemd Type=exec supervises this process, and a
    # detached `docker run` exits immediately, which systemd reads as the
    # service having stopped while the container keeps the GPU.
    assert '-d' not in argv
    assert '--detach' not in argv


@pytest.mark.parametrize(
    'arm_factory', [_arm, _embedding_arm, _llamacpp_arm], ids=['vllm-llm', 'vllm-embed', 'llamacpp'],
)
def test_container_name_matches_what_the_unit_will_docker_stop(arm_factory):
    """`lms-arm@.service` runs `ExecStop=/usr/bin/docker stop lms-arm-%i`; if
    the name here drifted, stopping the unit would leave the container — and
    its VRAM — alive."""
    arm = arm_factory()

    argv = lms_serve.build_launch_argv(arm, MEASURED_GPU)

    assert _value_after(argv, '--name') == f'lms-arm-{arm.arm_id}'
    assert _value_after(argv, '--name') == arm.container_name


@pytest.mark.parametrize(
    'arm_factory', [_arm, _embedding_arm, _llamacpp_arm], ids=['vllm-llm', 'vllm-embed', 'llamacpp'],
)
def test_host_port_comes_from_the_manifest(arm_factory):
    arm = arm_factory()

    argv = lms_serve.build_launch_argv(arm, MEASURED_GPU)

    assert _value_after(argv, '-p') == f'{arm.port}:{lms_serve.CONTAINER_PORT}'


@pytest.mark.parametrize(
    'arm_factory', [_arm, _embedding_arm, _llamacpp_arm], ids=['vllm-llm', 'vllm-embed', 'llamacpp'],
)
def test_argv_hardcodes_no_absolute_host_path_outside_the_hf_cache_or_repo(arm_factory):
    """A stray absolute path would make the unit unreproducible on any other
    checkout and would silently bind something nobody reviewed."""
    argv = lms_serve.build_launch_argv(arm_factory(), MEASURED_GPU)

    allowed_roots = (
        str(lms_serve.HOST_HF_CACHE),
        lms_serve.CONTAINER_HF_CACHE,
        lms_serve.CONTAINER_DATA_DIR,
        str(lms_serve.REPO_ROOT),
    )
    for element in argv:
        for piece in element.split(':'):
            if piece.startswith('/'):
                assert piece.startswith(allowed_roots), (
                    f'{piece!r} in {element!r} is an absolute host path outside '
                    f'the HF cache and the repo'
                )


@pytest.mark.parametrize(
    'arm_factory', [_arm, _embedding_arm, _llamacpp_arm], ids=['vllm-llm', 'vllm-embed', 'llamacpp'],
)
def test_image_is_the_last_docker_argument_before_the_server_args(arm_factory):
    arm = arm_factory()

    argv = lms_serve.build_launch_argv(arm, MEASURED_GPU)

    assert arm.image in argv
    # Everything after the image is passed to the server, not to docker.
    assert argv.index(arm.image) < len(argv) - 1


def test_image_digest_is_used_when_pinned():
    """Step 19 resolves each tag to a digest; a digest pin is what makes a
    re-run serve the same BITS, not merely the same tag."""
    arm = _arm(image_digest='sha256:' + 'a' * 64)

    argv = lms_serve.build_launch_argv(arm, MEASURED_GPU)

    expected = f'{arm.image.split(":")[0]}@sha256:{"a" * 64}'
    assert expected in argv
    assert arm.image not in argv


# ---------------------------------------------------------------------------
# vLLM — dense LLM arms
# ---------------------------------------------------------------------------


def test_vllm_llm_argv_carries_the_manifest_serving_contract():
    arm = _arm()

    argv = lms_serve.build_launch_argv(arm, MEASURED_GPU)

    assert _value_after(argv, '--model') == arm.model_ref
    # served-model-name is what the health check verifies /v1/models against,
    # and what beta/eta will send as the model id.
    assert _value_after(argv, '--served-model-name') == arm.served_model_name
    assert _value_after(argv, '--max-model-len') == str(arm.max_model_len)
    assert _value_after(argv, '--max-num-seqs') == str(arm.max_num_seqs)
    assert _value_after(argv, '--quantization') == 'awq'
    assert _value_after(argv, '--runner') == 'generate'
    assert _value_after(argv, '--port') == str(lms_serve.CONTAINER_PORT)


def test_vllm_gpu_memory_utilization_is_derived_from_the_measured_reading():
    arm = _arm()

    argv = lms_serve.build_launch_argv(arm, MEASURED_GPU)

    expected = lms_vram.gpu_memory_utilization_for(
        MEASURED_GPU.free_gib, MEASURED_GPU.total_gib,
    )
    assert _value_after(argv, '--gpu-memory-utilization') == str(expected)


def test_vllm_gpu_memory_utilization_is_never_the_0_95_pod_era_default():
    argv = lms_serve.build_launch_argv(_arm(), MEASURED_GPU)

    value = float(_value_after(argv, '--gpu-memory-utilization'))
    assert value < 0.95
    assert 0 < value <= 1


def test_vllm_gpu_memory_utilization_tracks_a_different_reading():
    """It is a live measurement, not a constant: free VRAM after whisper-writer
    exits yields a larger share."""
    roomier = lms_vram.GpuReading(total_mib=24576, used_mib=3300, free_mib=20800)

    tight = float(_value_after(
        lms_serve.build_launch_argv(_arm(), MEASURED_GPU), '--gpu-memory-utilization'))
    loose = float(_value_after(
        lms_serve.build_launch_argv(_arm(), roomier), '--gpu-memory-utilization'))

    assert loose > tight


def test_vllm_arm_without_quantization_omits_the_flag():
    """`--quantization none` is not a thing; an fp16 arm must simply not pass it."""
    argv = lms_serve.build_launch_argv(_arm(quant='none'), MEASURED_GPU)

    assert '--quantization' not in argv


def test_vllm_mounts_the_hf_cache_read_write():
    argv = lms_serve.build_launch_argv(_arm(), MEASURED_GPU)

    assert '-v' in argv
    assert _value_after(argv, '-v') == (
        f'{lms_serve.HOST_HF_CACHE}:{lms_serve.CONTAINER_HF_CACHE}'
    )


# ---------------------------------------------------------------------------
# vLLM — embedding arms
# ---------------------------------------------------------------------------


def test_vllm_embedding_argv_uses_the_pooling_runner():
    arm = _embedding_arm()

    argv = lms_serve.build_launch_argv(arm, MEASURED_GPU)

    assert _value_after(argv, '--runner') == 'pooling'
    assert _value_after(argv, '--model') == arm.model_ref
    assert _value_after(argv, '--served-model-name') == arm.served_model_name


def test_vllm_embedding_argv_carries_no_chat_completion_only_flags():
    argv = lms_serve.build_launch_argv(_embedding_arm(), MEASURED_GPU)

    joined = ' '.join(argv)
    for flag in _CHAT_ONLY_FLAGS:
        assert flag not in joined, f'{flag} has no meaning for a pooling arm'


# ---------------------------------------------------------------------------
# llama.cpp — the MoE arm
# ---------------------------------------------------------------------------


def test_llamacpp_argv_serves_the_pinned_gguf_from_the_hf_cache():
    arm = _llamacpp_arm()

    argv = lms_serve.build_launch_argv(arm, MEASURED_GPU)

    model_path = _value_after(argv, '-m')
    assert model_path.startswith(lms_serve.CONTAINER_HF_CACHE)
    assert model_path.endswith(arm.gguf_file or '')
    assert arm.arm_id in model_path
    assert _value_after(argv, '-c') == str(arm.max_model_len)
    assert _value_after(argv, '--alias') == arm.served_model_name
    assert _value_after(argv, '--port') == str(lms_serve.CONTAINER_PORT)


def test_llamacpp_argv_offloads_every_layer_to_the_gpu():
    argv = lms_serve.build_launch_argv(_llamacpp_arm(), MEASURED_GPU)

    assert '-ngl' in argv
    assert int(_value_after(argv, '-ngl')) >= 99


def test_llamacpp_argv_carries_no_constrained_decoding_flag():
    """ggml-org/llama.cpp#21228 — llama.cpp silently falls back to
    UNCONSTRAINED output on $ref/$defs schemas.  This arm runs json_object
    plus a hard client-side validator (epsilon's); claiming constrained
    decoding here would let the eval credit a capability it never had."""
    argv = lms_serve.build_launch_argv(_llamacpp_arm(), MEASURED_GPU)

    joined = ' '.join(argv).lower()
    for token in _CONSTRAINED_DECODING_TOKENS:
        assert token not in joined, (
            f'{token!r} appears in the llama.cpp argv; see llama.cpp#21228'
        )


def test_llamacpp_arm_without_a_pinned_gguf_file_raises():
    """The committed manifest's moe-stretch is a TBD-Q3 placeholder until
    step 22 pins a quant that fits the measured budget.  Launching it must
    fail loudly rather than mounting a path that does not exist."""
    arm = _llamacpp_arm(gguf_file=None)

    with pytest.raises(lms_serve.ArmLaunchError) as excinfo:
        lms_serve.build_launch_argv(arm, MEASURED_GPU)
    assert 'gguf_file' in str(excinfo.value)


# ---------------------------------------------------------------------------
# TEI — the per-arm embedding fallback (PRD Open Q1)
# ---------------------------------------------------------------------------


def test_tei_argv_uses_model_id_and_port():
    arm = _embedding_arm(
        stack='tei',
        image='ghcr.io/huggingface/text-embeddings-inference:86-1.7',
    )

    argv = lms_serve.build_launch_argv(arm, MEASURED_GPU)

    assert _value_after(argv, '--model-id') == arm.model_ref
    assert _value_after(argv, '--port') == str(lms_serve.CONTAINER_PORT)
    # TEI reads its cache from /data, not the HF cache path vLLM uses.
    assert _value_after(argv, '-v') == (
        f'{lms_serve.HOST_HF_CACHE}:{lms_serve.CONTAINER_DATA_DIR}'
    )
    assert '--model' not in argv


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def test_unknown_stack_raises_a_typed_error():
    """Reached via model_construct because the manifest's Literal makes an
    unknown stack unrepresentable — which is the point: the defensive branch
    still has to exist and still has to be typed."""
    arm = lms_manifest.ArmEntry.model_construct(
        arm_id='weird', axis='llm', stack='ollama',
        image='x', model_ref='y', port=8410, served_model_name='weird',
        structured_output_mode='json_object', est_vram_gib=1.0,
        quant='none', max_model_len=8192, max_num_seqs=1,
        dims=None, query_prefix=None, gguf_file=None, image_digest=None,
        fallback_stack=None, notes=None,
    )

    with pytest.raises(lms_serve.ArmLaunchError) as excinfo:
        lms_serve.build_launch_argv(arm, MEASURED_GPU)
    assert 'ollama' in str(excinfo.value)


def test_placeholder_arm_raises_rather_than_launching_a_literal_tbd():
    arm = _llamacpp_arm(model_ref='TBD-Q3-pick-a-gguf', quant='TBD-Q3')

    with pytest.raises(lms_serve.ArmLaunchError) as excinfo:
        lms_serve.build_launch_argv(arm, MEASURED_GPU)
    assert 'TBD' in str(excinfo.value)


# ---------------------------------------------------------------------------
# the committed slate builds
# ---------------------------------------------------------------------------


def test_every_committed_non_placeholder_arm_builds_an_argv():
    manifest = lms_manifest.load_arms()

    built = []
    for arm in manifest.arms:
        if arm.is_placeholder:
            continue
        argv = lms_serve.build_launch_argv(arm, MEASURED_GPU)
        assert argv[:2] == ['docker', 'run']
        built.append(arm.arm_id)

    # Derived from the manifest, never a literal count.  This assertion used to
    # read `== 7` ("8 arms minus the TBD-Q3 placeholder") and step 22's Open Q3
    # resolution turned it red for the RIGHT reason -- but a count is the wrong
    # shape: it goes stale on every slate change and says nothing about WHICH
    # arm failed to build.
    expected = [a.arm_id for a in manifest.arms if not a.is_placeholder]
    assert built == expected
