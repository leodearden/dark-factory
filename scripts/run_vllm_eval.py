#!/usr/bin/env python3
"""Run a single vLLM eval on a RunPod GPU pod.

Creates pod → waits for vLLM → runs eval → terminates pod.
Pod is ALWAYS terminated in the finally block, even on errors.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, "/home/leo/src/runpod-toolkit")
from runpod_toolkit.config import RunPodConfig
from runpod_toolkit.compute import RunPodClient, PodStatus

RUNPOD_KEY = "rpa_VLRVNJ8HB5CH7MQZL9WW2XPQBQO18V3PMA1H1BSM11niy2"
SSH_KEY = os.path.expanduser("~/.ssh/id_runpod")
SSH_PUBKEY = open(SSH_KEY + ".pub").read().strip()
PROJECT_ROOT = "/home/leo/src/dark-factory"

# Model → (docker_tag, eval_config_name, gpu_count, container_disk_gb, extra_env)
# container_disk must be ≥ 2.5× model size (HF download uses temp + final copies)
MODELS = {
    "devstral-small":   ("devstral-small",   "devstral-small-2505-q6", 1, 250, {}),
    "qwen3-coder-next": ("qwen3-coder-next", "qwen3-coder-next-fp8-new",  1, 400, {"DTYPE": "float8", "TOOL_CALL_PARSER": "qwen3_coder"}),
    "reap-139b":        ("reap-139b",        "reap-139b-nvfp4",       1, 350, {}),
    "reap-172b":        ("reap-172b",        "reap-172b-nvfp4",       1, 430, {}),
    "minimax-m25":      ("minimax-m25",      "minimax-m25-fp8",       2, 570, {"DTYPE": "float8"}),
}

# GPU types to try in order (96GB → 80GB, cheapest viable first)
GPU_TYPES = [
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
    "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 PCIe",
]


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


def run_eval(model_key, task_name, local_port, datacenter="US-NC-1", use_volume=True):
    docker_tag, config_name, gpu_count, container_disk, extra_env = MODELS[model_key]
    task_path = f"src/orchestrator/evals/tasks/{task_name}.json"

    # HF model ID for MODEL_NAME env var
    HF_MODELS = {
        "devstral-small": "mistralai/Devstral-Small-2505",
        "qwen3-coder-next": "Qwen/Qwen3-Coder-Next",
        "reap-139b": "cerebras/MiniMax-M2.5-REAP-139B-A10B",
        "reap-172b": "cerebras/MiniMax-M2.5-REAP-172B-A10B",
        "minimax-m25": "MiniMaxAI/MiniMax-M2.5",
    }

    if use_volume:
        # Volume approach: base image + volume with pre-downloaded weights
        image = "leosiriusdawn/runpod-vllm:latest"
        volume_id = "obxma9bf1b"
        datacenter = "US-NC-1"  # volume is in this DC
        container_disk = 50  # just need space for runtime, not weights
    else:
        # Base image + HF download: container starts fast, model downloads inside
        # Much faster than baked 100GB+ images which timeout on RunPod pull
        image = "leosiriusdawn/runpod-vllm:latest"
        volume_id = None
        datacenter = None  # let RunPod auto-select
        # Need enough container disk for model weights + runtime
        container_disk = container_disk + 30  # extra headroom for download

    config = RunPodConfig(api_key=RUNPOD_KEY)
    client = RunPodClient(config)

    pod = None
    tunnel_proc = None
    exit_code = 1

    try:
        # 1. Create GPU pod
        env_vars = {
            "PUBLIC_KEY": SSH_PUBKEY,
            "MODEL_NAME": HF_MODELS[model_key],
            **extra_env,
        }
        if gpu_count > 1:
            env_vars["TP_SIZE"] = str(gpu_count)

        # Try GPU types in order until one is available
        for gpu_type in GPU_TYPES:
            try:
                log(f"Trying {gpu_count}× {gpu_type} in {datacenter}...")
                pod = client.create_pod(
                    name=f"eval-{model_key}",
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
            timeout=1800,  # 30 min — model loading can be slow
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
        # Timeout scales with model size: ~20 min download + ~10 min load
        # container_disk is a rough proxy for model size
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
            log(f"EVAL PASSED: {model_key} / {task_name}")
        else:
            log(f"EVAL FAILED (exit {exit_code}): {model_key} / {task_name}")

    except subprocess.TimeoutExpired:
        log(f"EVAL TIMEOUT: {model_key} / {task_name}")
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
    parser.add_argument("--model", required=True, choices=MODELS.keys())
    parser.add_argument("--task", default="df_task_12")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--datacenter", default="US-NC-1")
    parser.add_argument("--no-volume", action="store_true",
                        help="Use baked Docker image instead of volume")
    args = parser.parse_args()

    exit_code = run_eval(args.model, args.task, args.port, args.datacenter,
                         use_volume=not args.no_volume)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
