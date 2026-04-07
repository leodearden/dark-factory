#!/usr/bin/env python3
"""Run a single vLLM eval on a RunPod GPU pod.

Creates pod → waits for vLLM → runs eval → terminates pod.
Pod is ALWAYS terminated in the finally block, even on errors.

The eval config (image, GPU type, pod sizing, model env vars) comes from
orchestrator.evals.configs.VLLM_EVAL_CONFIGS — single source of truth.
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, "/home/leo/src/runpod-toolkit")
sys.path.insert(0, "/home/leo/src/dark-factory/orchestrator/src")
from runpod_toolkit.config import RunPodConfig
from runpod_toolkit.compute import RunPodClient, PodStatus
from orchestrator.evals.configs import get_config_by_name, VLLM_EVAL_CONFIGS, EvalConfig

RUNPOD_KEY = "rpa_VLRVNJ8HB5CH7MQZL9WW2XPQBQO18V3PMA1H1BSM11niy2"
SSH_KEY = os.path.expanduser("~/.ssh/id_runpod")
SSH_PUBKEY = open(SSH_KEY + ".pub").read().strip()
PROJECT_ROOT = "/home/leo/src/dark-factory"

# Fallback GPU types when neither the config nor --gpu-type provides one.
GPU_TYPES = [
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA H200",
]

# RunPod-targetable config names: those with an image set in configs.py.
RUNPOD_CONFIG_NAMES = [c.name for c in VLLM_EVAL_CONFIGS if c.image is not None]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for_vllm(port, timeout=900):
    """Poll vLLM health endpoint until ready. Timeout 15 min for large model loads."""
    import urllib.request
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                log(f"vLLM healthy on port {port}")
                return True
        except Exception:
            pass
        time.sleep(10)
    return False


def run_eval(config_name, task_name, local_port, datacenter="US-NC-1",
             use_volume=True, image_override=None, gpu_type_override=None):
    cfg: EvalConfig | None = get_config_by_name(config_name)
    if cfg is None:
        log(f"ERROR: unknown eval config '{config_name}'")
        return 1
    if cfg.image is None and image_override is None:
        log(f"ERROR: config '{config_name}' has no image — not a RunPod config")
        return 1

    task_path = f"src/orchestrator/evals/tasks/{task_name}.json"

    image = image_override or cfg.image
    gpu_count = cfg.gpu_count
    container_disk = cfg.container_disk_gb
    hf_model = cfg.env_overrides["MODEL_NAME"]

    # Build pod env vars from config env_overrides (filter to vLLM entrypoint vars)
    extra_env = {}
    for k in ("TOOL_CALL_PARSER", "QUANTIZATION", "TP_SIZE",
              "MAX_MODEL_LEN", "GPU_MEMORY_UTIL", "MAX_NUM_SEQS",
              "ENFORCE_EAGER"):
        if k in cfg.env_overrides:
            extra_env[k] = cfg.env_overrides[k]

    if use_volume:
        # Volume approach: base image + volume with pre-downloaded weights
        volume_id = "obxma9bf1b"
        datacenter = "US-NC-1"  # volume is in this DC
        container_disk = 50  # just need space for runtime, not weights
    else:
        # Baked image (or :latest with HF download): use config's image and disk sizing
        volume_id = None
        datacenter = None  # let RunPod auto-select

    config = RunPodConfig(api_key=RUNPOD_KEY)
    client = RunPodClient(config)

    pod = None
    tunnel_proc = None
    exit_code = 1

    try:
        # 1. Create GPU pod
        env_vars = {
            "PUBLIC_KEY": SSH_PUBKEY,
            "MODEL_NAME": hf_model,
            **extra_env,
        }

        # Resolve GPU type list:
        #   --gpu-type override > config.gpu_type > GPU_TYPES fallback list
        if gpu_type_override:
            gpu_types_to_try = [gpu_type_override]
        elif cfg.gpu_type:
            gpu_types_to_try = [cfg.gpu_type]
        else:
            gpu_types_to_try = GPU_TYPES

        for gpu_type in gpu_types_to_try:
            try:
                log(f"Trying {gpu_count}× {gpu_type} in {datacenter}...")
                pod = client.create_pod(
                    name=f"eval-{config_name}",
                    gpu_type=gpu_type,
                    image=image,
                    gpu_count=gpu_count,
                    container_disk_gb=container_disk,
                    datacenter=datacenter,
                    network_volume_id=volume_id,
                    env_vars=env_vars,
                )
                log(f"Pod created: {pod.id} ({gpu_type})")
                break
            except Exception as e:
                log(f"  {gpu_type}: {e}")
                pod = None
                continue

        if not pod:
            log("ERROR: No GPU type available in this datacenter")
            return 1

        # 2. Wait for SSH
        log("Waiting for pod + SSH...")
        pod = client.wait_for_pod(
            pod_id=pod.id,
            status=PodStatus.RUNNING,
            timeout=3600,  # 60 min — large baked images (100+ GB) can pull slowly
            poll_interval=15,
            wait_for_ssh=True,
        )
        log(f"SSH available: {pod.ssh_host}:{pod.ssh_port}")

        # 3. Start SSH tunnel
        log(f"Starting SSH tunnel (localhost:{local_port} → pod:8000)")
        tunnel_proc = subprocess.Popen(
            [
                "ssh", "-N",
                "-L", f"{local_port}:localhost:8000",
                "-i", SSH_KEY,
                "-o", "StrictHostKeyChecking=no",
                "-o", "ServerAliveInterval=30",
                "-o", "ServerAliveCountMax=3",
                f"root@{pod.ssh_host}",
                "-p", str(pod.ssh_port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        time.sleep(3)

        # 4. Wait for vLLM health
        # Timeout scales with container_disk as a rough proxy for model size.
        health_timeout = max(900, container_disk * 12)  # ~12s per GB
        log(f"Waiting for vLLM to load model (timeout {health_timeout//60} min)...")
        if not wait_for_vllm(local_port, timeout=health_timeout):
            # SSH in to check what's happening before giving up
            log("Health timeout — checking pod status via SSH...")
            try:
                check = subprocess.run(
                    ["ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
                     "-o", "ConnectTimeout=5",
                     f"root@{pod.ssh_host}", "-p", str(pod.ssh_port),
                     "ps aux | grep vllm | head -3; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; ls -lh /root/.cache/huggingface/hub/ 2>/dev/null | tail -5"],
                    capture_output=True, text=True, timeout=15,
                )
                log(f"Pod diagnostics:\n{check.stdout}")
            except Exception as e:
                log(f"Could not diagnose: {e}")
            log(f"ERROR: vLLM did not become healthy within {health_timeout//60} min")
            return 1

        # 5. Run eval
        log(f"Running eval: config={config_name} task={task_name}")
        env = os.environ.copy()
        # Load .env for CLAUDE_OAUTH_TOKEN_G
        dotenv_path = os.path.join(PROJECT_ROOT, ".env")
        if os.path.exists(dotenv_path):
            with open(dotenv_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        env[k.strip()] = v.strip()

        oauth_token = env.get("CLAUDE_OAUTH_TOKEN_G", "")
        if not oauth_token:
            log("WARNING: CLAUDE_OAUTH_TOKEN_G not found in .env")

        env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token

        result = subprocess.run(
            [
                "uv", "run", "orchestrator", "eval",
                "--task", task_path,  # relative to cwd (orchestrator/)
                "--config-name", config_name,
                "--vllm-url", f"http://localhost:{local_port}",
                "--force",
            ],
            cwd=os.path.join(PROJECT_ROOT, "orchestrator"),
            env=env,
            timeout=3600,  # 1 hour max per eval
        )

        exit_code = result.returncode
        if exit_code == 0:
            log(f"EVAL PASSED: {config_name} / {task_name}")
        else:
            log(f"EVAL FAILED (exit {exit_code}): {config_name} / {task_name}")

    except subprocess.TimeoutExpired:
        log(f"EVAL TIMEOUT: {config_name} / {task_name}")
    except Exception as e:
        log(f"ERROR: {e}")
    finally:
        # ALWAYS clean up
        if tunnel_proc:
            log("Killing SSH tunnel...")
            tunnel_proc.terminate()
            try:
                tunnel_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel_proc.kill()

        if pod:
            log(f"TERMINATING POD {pod.id}...")
            try:
                ok = client.terminate_pod(pod.id)
                log(f"Pod terminated: {ok}")
            except Exception as e:
                log(f"WARNING: Failed to terminate pod {pod.id}: {e}")
                log(f"MANUAL CLEANUP NEEDED: runpodctl remove pod {pod.id}")

    return exit_code


def main():
    parser = argparse.ArgumentParser(description="Run vLLM eval on RunPod")
    parser.add_argument("--config", required=True, choices=RUNPOD_CONFIG_NAMES,
                        help="vLLM eval config name from orchestrator.evals.configs")
    parser.add_argument("--task", default="df_task_12")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--datacenter", default="US-NC-1")
    parser.add_argument("--no-volume", action="store_true",
                        help="Use baked Docker image instead of volume")
    parser.add_argument("--image", default=None,
                        help="Override Docker image (default: from config)")
    parser.add_argument("--gpu-type", default=None,
                        help="Force a specific RunPod GPU type id (e.g. 'NVIDIA H200 NVL'). "
                             "When set, overrides the config's gpu_type and the GPU_TYPES fallback list.")
    args = parser.parse_args()

    exit_code = run_eval(args.config, args.task, args.port, args.datacenter,
                         use_volume=not args.no_volume, image_override=args.image,
                         gpu_type_override=args.gpu_type)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
