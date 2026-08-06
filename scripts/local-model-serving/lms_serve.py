"""Build and exec one arm's container launch command.

PRD-MARKER:local-memory-models-eval serving

Task 3713 (LME-alpha) of `plans/local-memory-models-eval-prd.md`.

`lms-arm@.service` runs `lms_serve.py %i`.  This module resolves that instance
name against `arms.yaml`, checks the arm against MEASURED free VRAM, and then
`execvp`s `docker run` so systemd supervises the real process rather than a
shell wrapper.

Why containers.  Measured on this host: none of vLLM, llama.cpp or TEI is
installed (`which vllm llama-server llama-cli text-embeddings-router` all miss;
only `/usr/local/bin/ollama` exists), so all three have to come from nothing
either way.  Docker with the NVIDIA container runtime is already wired
(`/etc/docker/daemon.json` registers `nvidia`; `docker run --gpus all
nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi -L` lists the 3090 from this
user), and the repo's entire vLLM prior art is image-based.  Docker's default
runtime here is `runc`, so `--gpus all` is passed per run.

`build_launch_argv` is pure and total: no environment reads, no container
started, no GPU touched.  That is what lets every flag decision below be
asserted offline.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import lms_vram
from lms_manifest import ArmEntry, ArmManifestError, load_arms

#: Every stack listens on the same port INSIDE its container; only the host
#: port varies, and it comes from the manifest.  One less thing to desynchronise
#: between the unit, the launcher and the health check.
CONTAINER_PORT = 8000

HOST_HF_CACHE = Path.home() / '.cache' / 'huggingface'
CONTAINER_HF_CACHE = '/root/.cache/huggingface'
#: TEI reads its model cache from /data, not from an HF_HOME path.
CONTAINER_DATA_DIR = '/data'

#: GGUF quants live inside the HF cache tree rather than a directory of their
#: own, so ONE bind mount serves every stack and no launch argv ever names a
#: host path the reviewer has not already seen.
GGUF_SUBDIR = 'lms-gguf'

REPO_ROOT = Path(__file__).resolve().parents[2]

#: llama.cpp splits `-c` across `--parallel` slots, so one slot is what gives a
#: request the arm's full declared context.  Alpha only has to prove the
#: endpoint answers; eta/theta can revisit concurrency with measurements.
LLAMACPP_PARALLEL_SLOTS = 1

#: Offload every layer.  A partial offload silently moves the model onto the
#: CPU and turns a throughput comparison into a measurement of RAM bandwidth.
LLAMACPP_GPU_LAYERS = 99


class ArmLaunchError(Exception):
    """This arm cannot be turned into a launch command."""


def host_gguf_dir(arm: ArmEntry) -> Path:
    """Where `lms_fetch_weights` puts this arm's pinned quant."""
    return HOST_HF_CACHE / GGUF_SUBDIR / arm.arm_id


def container_gguf_path(arm: ArmEntry) -> str:
    if not arm.gguf_file:
        raise ArmLaunchError(
            f'arm {arm.arm_id!r} is stack=llamacpp but pins no gguf_file. '
            'Serving without one would mount a path that does not exist; pin '
            'the quant in arms.yaml first (PRD Open Q3)'
        )
    return f'{CONTAINER_HF_CACHE}/{GGUF_SUBDIR}/{arm.arm_id}/{arm.gguf_file}'


def image_ref(arm: ArmEntry) -> str:
    """Prefer the resolved digest: a tag can be re-pushed, a digest cannot."""
    if not arm.image_digest:
        return arm.image
    last_segment = arm.image.rsplit('/', 1)[-1]
    repository = arm.image.rsplit(':', 1)[0] if ':' in last_segment else arm.image
    return f'{repository}@{arm.image_digest}'


def _docker_preamble(arm: ArmEntry, mount: str) -> list[str]:
    return [
        'docker', 'run', '--rm',
        # Must match `lms-arm@.service`'s `ExecStop=docker stop lms-arm-%i`.
        '--name', arm.container_name,
        # Per-run because the daemon's default runtime is runc, not nvidia.
        '--gpus', 'all',
        '-v', mount,
        '-p', f'{arm.port}:{CONTAINER_PORT}',
    ]


def _memory_share_for(arm: ArmEntry, gpu: lms_vram.GpuReading) -> float:
    """vLLM's ``--gpu-memory-utilization`` for this arm.

    Never the 0.95 pod-era default: that came from dedicated 96 GB pods
    (docs/vllm-eval-status.md:1037) and here would evict whisper-writer, which
    Leo requires resident (PRD D10).

    The share is what vLLM takes FOR ITSELF -- confirmed by measurement, not
    read off the docs: derived 0.682 of a 24576 MiB card, the qwen3-embedding
    -0.6b container's resident delta was 16603 MiB against the 16761 MiB the
    share allows.  vLLM then sizes its paged KV cache to fill whatever is left
    after weights, so this one number is also the arm's KV bound.

    That splits the two axes, because they want opposite things:

    * A **generate** arm needs KV -- it is the context window.  Its share comes
      from the live free reading, so an emptier card buys real headroom.
    * A **pooling** arm cannot read a KV cache at all, yet vLLM still hands a
      DECODER-based embedder one sized to fill the share.  Measured on
      qwen3-embedding-0.6b: weights 1.12 GiB, overhead 0.42 GiB, KV cache
      14.56 GiB (136,272 tokens) -- a 0.6B model taking 16.2 GiB, all but
      1.5 GiB of it unusable.  Encoder embedders (granite-r2, gte-modernbert)
      never showed it, which is why the slate hid the effect until the two
      decoder arms ran.  So a pooling arm is held to the footprint it declared
      in ``arms.yaml``, plus the same safety margin the pre-flight uses.

    Bounding it here rather than via ``--kv-cache-memory-bytes`` keeps
    ``est_vram_gib`` load-bearing: the pre-flight (:func:`lms_vram.arm_fits`)
    already admits an arm on the strength of that number, and an arm that then
    ignored it would make the pre-flight's arithmetic false as soon as two arms
    were considered together.  An arm whose declared footprint is genuinely too
    small now fails LOUDLY at startup instead of silently eating the card.
    """
    if arm.axis == 'embedding':
        return lms_vram.gpu_memory_utilization_for(
            arm.est_vram_gib + lms_vram.SAFETY_MARGIN_GIB, gpu.total_gib,
        )
    return lms_vram.gpu_memory_utilization_for(gpu.free_gib, gpu.total_gib)


def _vllm_argv(arm: ArmEntry, gpu: lms_vram.GpuReading) -> list[str]:
    argv = _docker_preamble(arm, f'{HOST_HF_CACHE}:{CONTAINER_HF_CACHE}')
    argv += ['-e', f'HF_HOME={CONTAINER_HF_CACHE}']
    argv.append(image_ref(arm))
    argv += [
        '--model', arm.model_ref,
        '--served-model-name', arm.served_model_name,
        '--runner', 'pooling' if arm.axis == 'embedding' else 'generate',
    ]
    if arm.quant and arm.quant != 'none':
        argv += ['--quantization', arm.quant]
    argv += [
        '--max-model-len', str(arm.max_model_len),
        '--max-num-seqs', str(arm.max_num_seqs),
        '--gpu-memory-utilization', str(_memory_share_for(arm, gpu)),
        '--host', '0.0.0.0',
        '--port', str(CONTAINER_PORT),
    ]
    # No structured-outputs backend flag: xgrammar is vLLM's default for JSON
    # schemas, and the flag's spelling has moved across releases. Step 19
    # verifies the backend on the real image instead of guessing here.
    return argv


def _llamacpp_argv(arm: ArmEntry, gpu: lms_vram.GpuReading) -> list[str]:
    del gpu  # llama.cpp sizes itself from -ngl and -c, not from a memory share.
    model_path = container_gguf_path(arm)
    argv = _docker_preamble(arm, f'{HOST_HF_CACHE}:{CONTAINER_HF_CACHE}')
    argv.append(image_ref(arm))
    argv += [
        '-m', model_path,
        '-ngl', str(LLAMACPP_GPU_LAYERS),
        '-c', str(arm.max_model_len),
        '--parallel', str(LLAMACPP_PARALLEL_SLOTS),
        '--alias', arm.served_model_name,
        '--host', '0.0.0.0',
        '--port', str(CONTAINER_PORT),
    ]
    # DELIBERATELY no grammar / json-schema / guided-decoding flag.  llama.cpp
    # silently falls back to UNCONSTRAINED output on the $ref/$defs schemas
    # graphiti really emits (ggml-org/llama.cpp#21228), so a flag here would
    # claim a capability that does not exist and the eval would credit it.
    # This arm runs json_object plus a hard client-side validator.
    return argv


def _tei_argv(arm: ArmEntry, gpu: lms_vram.GpuReading) -> list[str]:
    del gpu  # TEI has no memory-share knob.
    argv = _docker_preamble(arm, f'{HOST_HF_CACHE}:{CONTAINER_DATA_DIR}')
    argv += ['-e', f'HUGGINGFACE_HUB_CACHE={CONTAINER_DATA_DIR}']
    argv.append(image_ref(arm))
    argv += [
        '--model-id', arm.model_ref,
        '--hostname', '0.0.0.0',
        '--port', str(CONTAINER_PORT),
    ]
    return argv


def build_launch_argv(arm: ArmEntry, gpu: lms_vram.GpuReading) -> list[str]:
    """The full `docker run ...` argv for one arm.

    Pure: same arm and same GPU reading always give the same argv, so every
    flag decision is assertable without a container, a GPU or a network.
    """
    if arm.is_placeholder:
        raise ArmLaunchError(
            f'arm {arm.arm_id!r} still carries TBD placeholders '
            f'(model_ref={arm.model_ref!r}, image={arm.image!r}, '
            f'quant={arm.quant!r}). Resolve the PRD open question that owns it '
            'before launching, rather than serving a literal "TBD"'
        )

    builders = {
        'vllm': _vllm_argv,
        'llamacpp': _llamacpp_argv,
        'tei': _tei_argv,
    }
    builder = builders.get(arm.stack)
    if builder is None:
        raise ArmLaunchError(
            f'arm {arm.arm_id!r} declares unknown stack {arm.stack!r}; '
            f'known stacks are {sorted(builders)}'
        )
    return builder(arm, gpu)


def main(argv: list[str] | None = None) -> int:
    """systemd entry point: `lms_serve.py <arm_id> [--dry-run]`."""
    args = list(sys.argv[1:] if argv is None else argv)
    dry_run = '--dry-run' in args
    positional = [a for a in args if not a.startswith('-')]
    if len(positional) != 1:
        print(
            'usage: lms_serve.py <arm_id> [--dry-run]   '
            '(arm_id is the systemd instance name, %i)',
            file=sys.stderr,
        )
        return 2
    arm_id = positional[0]

    try:
        arm = load_arms().by_id(arm_id)
    except ArmManifestError as exc:
        print(f'lms_serve: {exc}', file=sys.stderr)
        return 2

    try:
        gpu = lms_vram.probe_gpu()
    except lms_vram.VramProbeError as exc:
        # Refusing here is the point: launching blind on a shared card is how
        # whisper-writer gets evicted.
        print(f'lms_serve: cannot read GPU state: {exc}', file=sys.stderr)
        return 3

    if not lms_vram.arm_fits(arm, gpu.free_gib):
        print(
            f'lms_serve: refusing to start — {lms_vram.arm_fit_reason(arm, gpu.free_gib)}',
            file=sys.stderr,
        )
        return 4

    try:
        launch_argv = build_launch_argv(arm, gpu)
    except ArmLaunchError as exc:
        print(f'lms_serve: {exc}', file=sys.stderr)
        return 5

    print(f'lms_serve: {" ".join(launch_argv)}', flush=True)
    if dry_run:
        return 0
    os.execvp(launch_argv[0], launch_argv)


if __name__ == '__main__':  # pragma: no cover - process entry point
    raise SystemExit(main())
