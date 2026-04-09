#!/usr/bin/env python3
"""Run one or more vLLM evals on a single RunPod GPU pod.

Creates pod → waits for vLLM healthy → runs N tasks sequentially against the
same vLLM endpoint → terminates pod. The pod is ALWAYS terminated in the
finally block, even on errors or KeyboardInterrupt.

Each task is evaluated against its own ``pre_task_commit`` (the commit
immediately before the original task was completed). Before any pod is
created, every task's baseline is preflighted: a throwaway worktree at
``pre_task_commit`` is checked out and the spec's lint+typecheck commands
are run. If the baseline is dirty under strict policy (the default), the
launcher refuses to run and tells the user which spec to fix. This avoids
burning GPU budget on baselines whose verify loops are pre-poisoned.

The eval config (image, GPU type, pod sizing, model env vars) comes from
``orchestrator.evals.configs.VLLM_EVAL_CONFIGS`` — single source of truth.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

sys.path.insert(0, "/home/leo/src/runpod-toolkit")
sys.path.insert(0, "/home/leo/src/dark-factory/orchestrator/src")
from runpod_toolkit.config import RunPodConfig  # type: ignore[import]
from runpod_toolkit.compute import RunPodClient, PodStatus  # type: ignore[import]
from orchestrator.evals.configs import (
    get_config_by_name,
    VLLM_EVAL_CONFIGS,
    EvalConfig,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SSH_KEY = os.path.expanduser("~/.ssh/id_runpod")
with open(SSH_KEY + ".pub") as _fh:
    SSH_PUBKEY = _fh.read().strip()

PROJECT_ROOT = Path("/home/leo/src/dark-factory")
ORCHESTRATOR_DIR = PROJECT_ROOT / "orchestrator"
TASKS_DIR = ORCHESTRATOR_DIR / "src/orchestrator/evals/tasks"
RESULTS_DIR = ORCHESTRATOR_DIR / "src/orchestrator/evals/results"

# Per-task log files for concurrent runs land here. /var/tmp survives reboot
# (unlike /tmp which is tmpfs); chosen so a mid-batch crash leaves logs
# behind for forensics.
EVAL_LOG_DIR = Path("/var/tmp/dark-factory-evals")


def _load_runpod_api_key() -> str:
    """Load RUNPOD_API_KEY from environment or PROJECT_ROOT/.env.

    Never hardcode the key in this file — a previous version embedded it
    as a constant and shipped it to a public GitHub repo, requiring a key
    rotation. Always source from .env (gitignored).
    """
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key
    dotenv_path = PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        with open(dotenv_path) as f:
            for raw in f:
                line = raw.strip()
                if line.startswith("RUNPOD_API_KEY="):
                    return line.split("=", 1)[1].strip()
    raise RuntimeError(
        "RUNPOD_API_KEY not found. Add `RUNPOD_API_KEY=rpa_...` to "
        f"{PROJECT_ROOT}/.env or export it in your shell."
    )

# H200 variants to try in priority order when a config requests H200.
H200_VARIANTS = [
    "NVIDIA H200",      # RunPod id for H200 SXM (141 GB)
    "NVIDIA H200 NVL",  # H200 NVL (143 GB)
]

# Fallback GPU types when neither the config nor --gpu-type provides one.
GPU_TYPES = [
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA H200",
]

# RunPod-targetable config names: those with an image set in configs.py.
RUNPOD_CONFIG_NAMES = [c.name for c in VLLM_EVAL_CONFIGS if c.image is not None]

# Workstation config names: those WITHOUT an image (no RunPod pod; direct URL).
WORKSTATION_CONFIG_NAMES = [c.name for c in VLLM_EVAL_CONFIGS if c.image is None]

# All vLLM config names (RunPod + workstation).
ALL_VLLM_CONFIG_NAMES = [c.name for c in VLLM_EVAL_CONFIGS]


# ---------------------------------------------------------------------------
# Internal exceptions
# ---------------------------------------------------------------------------


class PodBringupFailed(RuntimeError):
    """Pod creation, SSH, or vLLM health check failed."""


class TaskSpecNotFound(FileNotFoundError):
    """Eval task spec JSON does not exist."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PodHandle:
    """Live pod + SSH tunnel + vLLM endpoint state."""

    pod: Any  # runpod_toolkit Pod
    tunnel_proc: subprocess.Popen | None
    client: RunPodClient
    vllm_url: str
    local_port: int
    config_name: str


@dataclass
class BaselineStatus:
    """Result of preflighting a task's pre_task_commit."""

    task_id: str
    pre_task_commit: str
    head_matches: bool
    lint_clean: bool
    typecheck_clean: bool
    error: str | None = None

    @property
    def is_clean(self) -> bool:
        return self.head_matches and self.lint_clean and self.typecheck_clean


@dataclass
class EvalSummary:
    """One row in the multi-task summary table."""

    task_id: str
    config_name: str
    status: str  # "done" | "blocked" | "timeout" | "crashed" | "unknown"
    outcome: str | None
    cost_usd: float | None
    duration_s: float | None
    tests_pass: bool | None
    lint_clean: bool | None
    typecheck_clean: bool | None
    run_id: str | None
    result_path: Path | None
    error: str | None = None
    log_path: Path | None = None  # set in concurrent mode; None when streaming to TTY

    @classmethod
    def crashed(
        cls, task_id: str, config_name: str, exc: BaseException
    ) -> "EvalSummary":
        return cls(
            task_id=task_id,
            config_name=config_name,
            status="crashed",
            outcome=None,
            cost_usd=None,
            duration_s=None,
            tests_pass=None,
            lint_clean=None,
            typecheck_clean=None,
            run_id=None,
            result_path=None,
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Environment + spec loading
# ---------------------------------------------------------------------------


def build_eval_env() -> dict[str, str]:
    """Build the env dict passed to ``orchestrator eval`` subprocesses.

    Loads ``.env`` for ``CLAUDE_OAUTH_TOKEN_G`` and exposes it as
    ``CLAUDE_CODE_OAUTH_TOKEN``. Built once per launcher invocation; token
    rotation mid-run is not supported.
    """
    env = os.environ.copy()
    dotenv_path = PROJECT_ROOT / ".env"
    if dotenv_path.exists():
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
    return env


def orchestrator_config_for_spec(spec: dict) -> Path | None:
    """Locate the ``orchestrator eval --config`` YAML for a task spec.

    The eval CLI requires ``--config`` (or ``ORCH_CONFIG_PATH``) so it
    knows which project's ``project_root`` and ``fused_memory.project_id``
    to use. Different task families target different projects:
        df_task_*    → /home/leo/src/dark-factory/orchestrator/config.yaml
        reify_task_* → /home/leo/src/reify/orchestrator.yaml

    Resolves by checking known config-file names under ``spec['project_root']``.
    Returns ``None`` if nothing matches; the caller should fall back to the
    inherited ``ORCH_CONFIG_PATH`` env var.
    """
    project_root = spec.get("project_root")
    if not project_root:
        return None
    root = Path(project_root)
    candidates = [
        root / "orchestrator.yaml",
        root / "orchestrator" / "config.yaml",
        root / "orchestrator" / "orchestrator.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_task_spec(task_id: str) -> dict:
    """Load an eval task spec by id (e.g. ``df_task_12``)."""
    path = TASKS_DIR / f"{task_id}.json"
    if not path.exists():
        raise TaskSpecNotFound(
            f"Task spec not found: {path}. "
            f"Available: {sorted(p.stem for p in TASKS_DIR.glob('df_task_*.json'))}"
        )
    with open(path) as f:
        return json.load(f)


def resolve_task_ids(args: argparse.Namespace) -> list[str]:
    """Collapse ``--task`` / ``--tasks`` / ``--all-tasks`` into a list."""
    if args.all_tasks:
        # Glob both df_task_* (dark-factory project) and reify_task_*
        # (reify project). Both spec families live in the same dir but
        # carry their own ``project_root`` field.
        return sorted(
            p.stem
            for p in TASKS_DIR.glob("*_task_*.json")
        )
    if args.tasks:
        return [t.strip() for t in args.tasks.split(",") if t.strip()]
    if args.task:
        return [args.task]
    raise ValueError("No task selection — use --task, --tasks, or --all-tasks")


# ---------------------------------------------------------------------------
# Baseline preflight (B3)
# ---------------------------------------------------------------------------


def preflight_baseline(
    spec: dict, *, repo_root: Path | None = None
) -> BaselineStatus:
    """Verify a task's ``pre_task_commit`` is lint+typecheck clean.

    Creates a throwaway worktree at ``pre_task_commit``, runs the spec's
    ``setup_commands`` (so the venv exists), then runs the spec's lint and
    typecheck commands. Tests are deliberately skipped (slow + flaky on dev
    box). The throwaway worktree is removed before returning even on error.

    The worktree is created against ``spec['project_root']`` (e.g.
    ``/home/leo/src/reify`` for reify_task_*), falling back to the
    launcher's ``PROJECT_ROOT`` for backwards compatibility. An empty
    ``lint`` or ``typecheck`` command is treated as "no such step" and
    silently reported clean — reify tasks legitimately ship with an empty
    typecheck command.
    """
    task_id = spec["id"]
    sha = spec["pre_task_commit"]
    verify = spec.get("verify_commands")
    if verify is None:
        return BaselineStatus(
            task_id=task_id,
            pre_task_commit=sha,
            head_matches=False,
            lint_clean=False,
            typecheck_clean=False,
            error="spec is missing verify_commands",
        )
    lint_cmd = verify.get("lint", "")
    type_cmd = verify.get("typecheck", "")

    if repo_root is None:
        repo_root = Path(spec.get("project_root", PROJECT_ROOT))

    wt = repo_root / ".eval-worktrees" / task_id / f"preflight-{uuid4().hex[:8]}"
    wt.parent.mkdir(parents=True, exist_ok=True)

    try:
        try:
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(wt), sha],
                cwd=str(repo_root),
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            return BaselineStatus(
                task_id=task_id,
                pre_task_commit=sha,
                head_matches=False,
                lint_clean=False,
                typecheck_clean=False,
                error=f"git worktree add failed: {e.stderr.strip()}",
            )

        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(wt),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        head_ok = head == sha

        for setup in spec.get("setup_commands", []):
            log(f"  preflight setup: {setup}")
            subprocess.run(
                setup,
                shell=True,
                executable="/bin/bash",
                cwd=str(wt),
                timeout=600,
                check=False,
            )

        if lint_cmd:
            log(f"  preflight lint: {lint_cmd}")
            lint_rc = subprocess.run(
                lint_cmd,
                shell=True,
                executable="/bin/bash",
                cwd=str(wt),
                timeout=300,
            ).returncode
            lint_clean = lint_rc == 0
        else:
            log("  preflight lint: <empty cmd, skipped>")
            lint_clean = True

        if type_cmd:
            log(f"  preflight typecheck: {type_cmd}")
            type_rc = subprocess.run(
                type_cmd,
                shell=True,
                executable="/bin/bash",
                cwd=str(wt),
                timeout=900,
            ).returncode
            type_clean = type_rc == 0
        else:
            log("  preflight typecheck: <empty cmd, skipped>")
            type_clean = True

        return BaselineStatus(
            task_id=task_id,
            pre_task_commit=sha,
            head_matches=head_ok,
            lint_clean=lint_clean,
            typecheck_clean=type_clean,
        )
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(wt)],
            cwd=str(repo_root),
            capture_output=True,
        )


# ---------------------------------------------------------------------------
# vLLM health
# ---------------------------------------------------------------------------


def _pick_free_port() -> int:
    """Ask the OS for a free TCP port on 127.0.0.1.

    Used as the default for ``--port`` so parallel ``run_vllm_eval.py``
    invocations don't collide on a static port. The race window between
    closing the probe socket and ssh binding is tiny, and
    ``ExitOnForwardFailure=yes`` on the ssh tunnel is the safety net —
    see ``_build_ssh_tunnel_argv``.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_ssh_tunnel_argv(
    *,
    local_port: int,
    ssh_host: str,
    ssh_port: int,
    ssh_key: str,
) -> list[str]:
    """Build the ``ssh -N -L ...`` argv for the vLLM tunnel.

    ``ExitOnForwardFailure=yes`` is critical: without it, a port-bind
    collision (two ``run_vllm_eval.py`` processes racing for the same
    local port) leaves ssh running with no forward. The launcher's
    subsequent ``wait_for_vllm`` probe then hits whatever else is
    listening on that port — the winner's tunnel, in the 2026-04-08
    matrix retry #3 failure — and reports healthy. Every
    ``/v1/messages`` request then routes to the wrong pod and returns
    404 "model not found". See ``docs/plan-vllm-404-bug-fix.md``.

    We also bind explicitly to ``127.0.0.1`` (IPv4) and forward to the
    pod's ``127.0.0.1:8000``. Without an explicit bind, ssh's ``-L``
    can fall back to ``::1`` if 127.0.0.1:<port> is already in use,
    but the launcher's health probe uses ``urllib`` which resolves
    ``localhost`` to 127.0.0.1 first and never tries ``::1``. The
    mismatch silently routes the health probe to whatever else is
    listening on 127.0.0.1:<port> (e.g. the dark-factory escalation MCP
    server, which returns 404), making vLLM look hung.
    """
    return [
        "ssh",
        "-N",
        "-L",
        f"127.0.0.1:{local_port}:127.0.0.1:8000",
        "-i",
        ssh_key,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "ExitOnForwardFailure=yes",
        f"root@{ssh_host}",
        "-p",
        str(ssh_port),
    ]


def wait_for_vllm(
    port: int,
    expected_model: str,
    timeout: int = 900,
    tunnel_proc: subprocess.Popen | None = None,
) -> bool:
    """Poll vLLM until /health is 200 AND expected_model is in /v1/models.

    If *tunnel_proc* is provided, each poll iteration checks whether the
    SSH tunnel process is still alive.  A dead tunnel means vLLM is
    unreachable — fail fast instead of polling a dead port for hours.

    The model-list check catches the case where the health probe has
    landed on a sibling tunnel (e.g. if ``ExitOnForwardFailure=yes`` is
    ever stripped from the ssh invocation and two parallel launchers
    race on the same port). Without it, a sibling's pod serving a
    different model would falsely pass the ``/health`` probe and every
    subsequent ``/v1/messages`` request would 404. See the 2026-04-08
    404 bug in ``docs/plan-vllm-404-bug-fix.md``.
    """
    import urllib.request

    health_url = f"http://127.0.0.1:{port}/health"
    models_url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        # Fail fast if the SSH tunnel died.
        if tunnel_proc is not None and tunnel_proc.poll() is not None:
            log(
                f"SSH tunnel died (rc={tunnel_proc.returncode}) — "
                "vLLM is unreachable, aborting health wait"
            )
            return False
        try:
            if urllib.request.urlopen(health_url, timeout=5).status == 200:
                resp = urllib.request.urlopen(models_url, timeout=5)
                payload = json.loads(resp.read())
                served = {m.get("id", "") for m in payload.get("data", [])}
                if expected_model in served:
                    log(
                        f"vLLM healthy on port {port} "
                        f"(serving {expected_model})"
                    )
                    return True
                log(
                    f"Port {port} reachable but /v1/models lists "
                    f"{sorted(served)}, not {expected_model!r} — "
                    "probable sibling-tunnel collision; continuing to wait"
                )
        except Exception:
            pass
        time.sleep(10)
    return False


def vllm_healthy(port: int, timeout: int = 5, *, base_url: str | None = None) -> bool:
    """Single-shot health probe used at the top of each task iteration."""
    import urllib.request

    url = f"{base_url}/health" if base_url else f"http://127.0.0.1:{port}/health"
    try:
        resp = urllib.request.urlopen(url, timeout=timeout)
        return resp.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pod lifecycle
# ---------------------------------------------------------------------------


def bring_up_pod(cfg: EvalConfig, args: argparse.Namespace) -> PodHandle:
    """Create pod → wait for SSH → open tunnel → wait for vLLM healthy.

    Raises PodBringupFailed on any error. On partial failure, attempts to
    clean up whatever was created before re-raising.
    """
    image = args.image or cfg.image
    gpu_count = cfg.gpu_count
    container_disk = cfg.container_disk_gb
    hf_model = cfg.env_overrides["MODEL_NAME"]
    use_volume = args.volume

    extra_env = {
        k: cfg.env_overrides[k]
        for k in (
            "TOOL_CALL_PARSER",
            "QUANTIZATION",
            "TP_SIZE",
            "PP_SIZE",
            "MAX_MODEL_LEN",
            "GPU_MEMORY_UTIL",
            "MAX_NUM_SEQS",
            "ENFORCE_EAGER",
            "OVERRIDE_GENERATION_CONFIG",
            "MOE_BACKEND",
            "VLLM_TEST_FORCE_FP8_MARLIN",
            "VLLM_USE_FLASHINFER_MOE_FP4",
        )
        if k in cfg.env_overrides
    }

    if use_volume:
        volume_id = "obxma9bf1b"
        datacenter = args.datacenter or "US-NC-1"  # volume is in US-NC-1
        container_disk = 50
    else:
        volume_id = None
        datacenter = args.datacenter  # None = any DC

    if args.gpu_type:
        gpu_types_to_try = [args.gpu_type]
    elif cfg.gpu_type:
        gpu_types_to_try = H200_VARIANTS if cfg.gpu_type == "NVIDIA H200" else [cfg.gpu_type]
    else:
        gpu_types_to_try = GPU_TYPES

    config = RunPodConfig(api_key=_load_runpod_api_key())
    client = RunPodClient(config)
    pod = None
    tunnel_proc: subprocess.Popen | None = None

    try:
        env_vars = {
            "PUBLIC_KEY": SSH_PUBKEY,
            "MODEL_NAME": hf_model,
            **extra_env,
        }

        for gpu_type in gpu_types_to_try:
            try:
                log(f"Trying {gpu_count}× {gpu_type} in {datacenter}...")
                pod = client.create_pod(
                    name=f"eval-{cfg.name}",
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
            raise PodBringupFailed("No GPU type available in this datacenter")

        log("Waiting for pod + SSH...")
        pod = client.wait_for_pod(
            pod_id=pod.id,
            status=PodStatus.RUNNING,
            timeout=3600,
            poll_interval=15,
            wait_for_ssh=True,
        )
        log(f"SSH available: {pod.ssh_host}:{pod.ssh_port}")

        log(f"Starting SSH tunnel (127.0.0.1:{args.port} → pod:8000)")
        tunnel_proc = subprocess.Popen(
            _build_ssh_tunnel_argv(
                local_port=args.port,
                ssh_host=pod.ssh_host,
                ssh_port=pod.ssh_port,
                ssh_key=SSH_KEY,
            ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        time.sleep(3)

        # Generous health timeout: 120 min floor + 30s/GB scaling. Big
        # models on 2× H200 can spend 30+ min on HF download alone, plus
        # 5–15 min on vLLM startup compile. Previous formula
        # (max(900, container_disk * 12)) gave only 48 min for 240 GB and
        # bit us on minimax-m25-nvfp4-new and reap-172b-nvfp4-gb10-new in
        # matrix run #2. Prefer waiting too long over false-killing a
        # healthy slow-loading pod.
        health_timeout = max(7200, container_disk * 30)
        log(
            f"Waiting for vLLM to load model "
            f"(timeout {health_timeout // 60} min)..."
        )
        if not wait_for_vllm(
            args.port,
            expected_model=hf_model,
            timeout=health_timeout,
            tunnel_proc=tunnel_proc,
        ):
            log("Health timeout — checking pod status via SSH...")
            try:
                check = subprocess.run(
                    [
                        "ssh",
                        "-i",
                        SSH_KEY,
                        "-o",
                        "StrictHostKeyChecking=no",
                        "-o",
                        "ConnectTimeout=5",
                        f"root@{pod.ssh_host}",
                        "-p",
                        str(pod.ssh_port),
                        "ps aux | grep vllm | head -3; "
                        "nvidia-smi --query-gpu=memory.used,memory.total "
                        "--format=csv,noheader; "
                        "ls -lh /root/.cache/huggingface/hub/ 2>/dev/null | tail -5",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                log(f"Pod diagnostics:\n{check.stdout}")
            except Exception as e:
                log(f"Could not diagnose: {e}")
            raise PodBringupFailed(
                f"vLLM did not become healthy within {health_timeout // 60} min"
            )

        return PodHandle(
            pod=pod,
            tunnel_proc=tunnel_proc,
            client=client,
            vllm_url=f"http://127.0.0.1:{args.port}",
            local_port=args.port,
            config_name=cfg.name,
        )

    except Exception:
        # Clean up any partial state before re-raising as PodBringupFailed.
        partial = PodHandle(
            pod=pod,
            tunnel_proc=tunnel_proc,
            client=client,
            vllm_url=f"http://127.0.0.1:{args.port}",
            local_port=args.port,
            config_name=cfg.name,
        )
        tear_down_pod(partial)
        raise


def reuse_pod(cfg: EvalConfig, args: argparse.Namespace) -> PodHandle:
    """Attach to an existing pod by ID — skip creation, skip teardown.

    Fetches SSH info from the API, opens a fresh tunnel, waits for vLLM
    healthy, and returns a PodHandle with ``pod=None`` so tear_down_pod
    is a no-op.
    """
    hf_model = cfg.env_overrides["MODEL_NAME"]
    config = RunPodConfig(api_key=_load_runpod_api_key())
    client = RunPodClient(config)

    log(f"Reusing existing pod {args.pod_id}...")
    pod = client.get_pod(args.pod_id)
    if pod.ssh_host is None or pod.ssh_port is None:
        raise PodBringupFailed(
            f"Pod {args.pod_id} has no SSH info — is it RUNNING?"
        )
    log(f"SSH available: {pod.ssh_host}:{pod.ssh_port}")

    log(f"Starting SSH tunnel (127.0.0.1:{args.port} → pod:8000)")
    tunnel_proc = subprocess.Popen(
        _build_ssh_tunnel_argv(
            local_port=args.port,
            ssh_host=pod.ssh_host,
            ssh_port=pod.ssh_port,
            ssh_key=SSH_KEY,
        ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    time.sleep(3)

    log("Waiting for vLLM health check...")
    if not wait_for_vllm(args.port, expected_model=hf_model, timeout=60, tunnel_proc=tunnel_proc):
        tunnel_proc.terminate()
        raise PodBringupFailed(
            f"vLLM not healthy on reused pod {args.pod_id}"
        )

    log(f"vLLM healthy on port {args.port} (serving {hf_model})")
    # Return with pod=None so tear_down_pod skips termination.
    return PodHandle(
        pod=None,
        tunnel_proc=tunnel_proc,
        client=client,
        vllm_url=f"http://127.0.0.1:{args.port}",
        local_port=args.port,
        config_name=cfg.name,
    )


def _connect_workstation(cfg: EvalConfig, args: argparse.Namespace) -> PodHandle:
    """Connect directly to a workstation vLLM endpoint — no pod, no tunnel.

    Used for workstation configs (image=None) where vLLM is already running
    on a local or LAN machine (e.g. leo-workstation:8000). The returned
    PodHandle has pod=None and tunnel_proc=None so tear_down_pod is a no-op.
    """
    import urllib.request
    import urllib.parse

    vllm_url = args.vllm_url
    parsed = urllib.parse.urlparse(vllm_url)
    port = parsed.port or 8000

    # Health check
    health_url = f"{vllm_url}/health"
    log(f"Checking vLLM health at {health_url}...")
    try:
        resp = urllib.request.urlopen(health_url, timeout=10)
        if resp.status != 200:
            raise PodBringupFailed(f"vLLM health check returned {resp.status}")
    except Exception as e:
        raise PodBringupFailed(
            f"vLLM not reachable at {health_url}: {e}"
        ) from e

    # Model check
    hf_model = cfg.env_overrides["MODEL_NAME"]
    models_url = f"{vllm_url}/v1/models"
    try:
        resp = urllib.request.urlopen(models_url, timeout=10)
        payload = json.loads(resp.read())
        served = {m.get("id", "") for m in payload.get("data", [])}
        if hf_model not in served:
            raise PodBringupFailed(
                f"vLLM serving {sorted(served)}, expected {hf_model!r}"
            )
    except PodBringupFailed:
        raise
    except Exception as e:
        raise PodBringupFailed(f"Could not list models at {models_url}: {e}") from e

    log(f"vLLM healthy at {vllm_url} (serving {hf_model})")

    # Store the port for vllm_healthy() single-shot probes during task runs.
    # Override args.port so run_one_task's health probe uses the workstation
    # port, not the auto-picked free port meant for SSH tunnels.
    args.port = port

    return PodHandle(
        pod=None,
        tunnel_proc=None,
        client=None,  # type: ignore[arg-type]
        vllm_url=vllm_url,
        local_port=port,
        config_name=cfg.name,
    )


def tear_down_pod(handle: PodHandle | None) -> None:
    """Idempotent pod + tunnel teardown. Safe with None or partial handles."""
    if handle is None:
        return

    if handle.tunnel_proc is not None:
        log("Killing SSH tunnel...")
        try:
            handle.tunnel_proc.terminate()
            try:
                handle.tunnel_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                handle.tunnel_proc.kill()
        except Exception as e:
            log(f"WARNING: SSH tunnel teardown failed: {e}")

    if handle.pod is not None:
        log(f"TERMINATING POD {handle.pod.id}...")
        try:
            ok = handle.client.terminate_pod(handle.pod.id)
            log(f"Pod terminated: {ok}")
        except Exception as e:
            log(f"WARNING: Failed to terminate pod {handle.pod.id}: {e}")
            log(f"MANUAL CLEANUP NEEDED: runpodctl remove pod {handle.pod.id}")


# ---------------------------------------------------------------------------
# Per-task execution
# ---------------------------------------------------------------------------


def find_new_result_file(
    task_id: str,
    config_name: str,
    since_mtime: float,
    exclude_paths: set[Path] | None = None,
) -> Path | None:
    """Glob results/ for a result file newer than *since_mtime*.

    Returns the newest matching file, or None if no new file appeared.
    Any path in *exclude_paths* is treated as pre-existing — useful when
    multiple concurrent tasks share the same results directory and ext4's
    1-second mtime granularity makes the watermark insufficient on its own.
    """
    if not RESULTS_DIR.exists():
        return None
    excluded = exclude_paths or set()
    pattern = f"{task_id}__{config_name}__*.json"
    candidates = [
        p
        for p in RESULTS_DIR.glob(pattern)
        if p.stat().st_mtime > since_mtime and p not in excluded
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_result_file(path: Path) -> dict:
    """Load a result JSON; return ``{outcome, metrics, run_id}``."""
    with open(path) as f:
        data = json.load(f)
    return {
        "outcome": data.get("outcome"),
        "metrics": data.get("metrics", {}) or {},
        "run_id": data.get("run_id"),
    }


def _open_task_log(task_id: str, config_name: str):
    """Open a per-task log file under EVAL_LOG_DIR.

    Returns ``(path, file_handle)``. The handle is opened for writing and the
    caller is responsible for closing it. A 6-char uuid is appended to avoid
    sub-second collisions when the same (config, task) pair is re-run.
    """
    EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = uuid4().hex[:6]
    path = EVAL_LOG_DIR / f"eval-{config_name}-{task_id}-{timestamp}-{suffix}.log"
    return path, open(path, "w")


def run_one_task(
    spec: dict,
    cfg: EvalConfig,
    handle: PodHandle,
    env: dict[str, str],
    args: argparse.Namespace,
) -> EvalSummary:
    """Run one ``orchestrator eval`` subprocess against the live pod.

    Captures the results-dir mtime watermark before the call and locates the
    new result file by ``(task_id, config_name)`` glob + watermark afterwards.
    """
    task_id = spec["id"]
    config_name = cfg.name

    # Fail fast if the SSH tunnel died.
    if handle.tunnel_proc is not None and handle.tunnel_proc.poll() is not None:
        log(
            f"SKIP: SSH tunnel dead (rc={handle.tunnel_proc.returncode}) "
            f"before {task_id}"
        )
        return EvalSummary(
            task_id=task_id,
            config_name=config_name,
            status="crashed",
            outcome=None,
            cost_usd=None,
            duration_s=None,
            tests_pass=None,
            lint_clean=None,
            typecheck_clean=None,
            run_id=None,
            result_path=None,
            error="SSH tunnel dead before task start",
        )

    if not vllm_healthy(handle.local_port, base_url=handle.vllm_url):
        log(
            f"SKIP: vLLM endpoint unhealthy before {task_id} — "
            f"likely the model crashed mid-batch"
        )
        return EvalSummary(
            task_id=task_id,
            config_name=config_name,
            status="crashed",
            outcome=None,
            cost_usd=None,
            duration_s=None,
            tests_pass=None,
            lint_clean=None,
            typecheck_clean=None,
            run_id=None,
            result_path=None,
            error="vLLM endpoint unhealthy at task start",
        )

    pre_files = (
        {p: p.stat().st_mtime for p in RESULTS_DIR.glob(f"{task_id}__{config_name}__*.json")}
        if RESULTS_DIR.exists()
        else {}
    )
    pre_max_mtime = max(pre_files.values(), default=0.0)

    task_path = f"src/orchestrator/evals/tasks/{task_id}.json"

    # In concurrent mode redirect subprocess output to a per-task log file so
    # the launcher TTY only carries one-liners. In sequential mode keep
    # streaming live so single-task / gate runs are unchanged.
    log_path: Path | None = None
    log_fh = None
    stdout_target = None
    stderr_target = None
    if args.concurrency > 1:
        log_path, log_fh = _open_task_log(task_id, config_name)
        stdout_target = log_fh
        stderr_target = subprocess.STDOUT
        log(f"START: task={task_id} → {log_path}")
    else:
        log(f"Running eval: config={config_name} task={task_id}")

    # Each task may target a different orchestrator config (df_* uses
    # dark-factory's, reify_* uses reify's). Resolve per-task and pass via
    # explicit --config so concurrent tasks for different projects don't
    # race on a shared ORCH_CONFIG_PATH env var.
    orch_config = orchestrator_config_for_spec(spec)
    eval_cmd = [
        "uv",
        "run",
        "orchestrator",
        "eval",
        "--task",
        task_path,
        "--config-name",
        config_name,
        "--vllm-url",
        handle.vllm_url,
        "--force",
    ]
    if orch_config is not None:
        eval_cmd.extend(["--config", str(orch_config)])
    if args.orch_timeout_min is not None:
        eval_cmd.extend(["--timeout", str(args.orch_timeout_min)])

    t0 = time.monotonic()

    try:
        result = subprocess.run(
            eval_cmd,
            cwd=str(ORCHESTRATOR_DIR),
            env=env,
            timeout=args.task_timeout_min * 60,
            stdout=stdout_target,
            stderr=stderr_target,
        )
        rc = result.returncode
    except subprocess.TimeoutExpired:
        log(
            f"EVAL TIMEOUT: {task_id} exceeded "
            f"{args.task_timeout_min} min subprocess limit"
        )
        rc = -1
    finally:
        if log_fh is not None:
            log_fh.close()

    duration_s = time.monotonic() - t0

    new_path = find_new_result_file(
        task_id,
        config_name,
        pre_max_mtime,
        exclude_paths=set(pre_files.keys()),
    )
    if new_path is None:
        return EvalSummary(
            task_id=task_id,
            config_name=config_name,
            status="crashed",
            outcome=None,
            cost_usd=None,
            duration_s=duration_s,
            tests_pass=None,
            lint_clean=None,
            typecheck_clean=None,
            run_id=None,
            result_path=None,
            error=(
                f"orchestrator subprocess rc={rc} but no new result file "
                f"under {RESULTS_DIR} matching {task_id}__{config_name}__*.json"
            ),
            log_path=log_path,
        )

    data = parse_result_file(new_path)
    metrics = data["metrics"]
    outcome = data["outcome"] or "unknown"
    return EvalSummary(
        task_id=task_id,
        config_name=config_name,
        status=outcome,
        outcome=outcome,
        cost_usd=metrics.get("cost_usd"),
        duration_s=duration_s,
        tests_pass=metrics.get("tests_pass"),
        lint_clean=metrics.get("lint_clean"),
        typecheck_clean=metrics.get("typecheck_clean"),
        run_id=data["run_id"],
        result_path=new_path,
        log_path=log_path,
    )


# ---------------------------------------------------------------------------
# Summary + exit
# ---------------------------------------------------------------------------


def log_one_summary(s: EvalSummary) -> None:
    """Per-task one-line log. Replaces the misleading 'EVAL PASSED' log."""
    cost = f"${s.cost_usd:.2f}" if s.cost_usd is not None else "$?"
    dur = f"{s.duration_s:.0f}s" if s.duration_s is not None else "?s"
    if s.status == "done":
        log(
            f"EVAL DONE: task={s.task_id} config={s.config_name} "
            f"outcome={s.outcome} cost={cost} duration={dur} "
            f"tests={s.tests_pass} lint={s.lint_clean} type={s.typecheck_clean}"
        )
    elif s.status in ("blocked", "timeout"):
        log(
            f"EVAL FAILED: task={s.task_id} config={s.config_name} "
            f"outcome={s.outcome} cost={cost} duration={dur} "
            f"tests={s.tests_pass} lint={s.lint_clean} type={s.typecheck_clean}"
        )
    else:
        log(
            f"EVAL CRASHED: task={s.task_id} config={s.config_name} "
            f"error={s.error}"
        )


def print_summary_table(summaries: list[EvalSummary]) -> None:
    """Pretty per-batch summary at the end of the run."""
    if not summaries:
        log("No tasks ran.")
        return

    bar = "=" * 110
    sep = "-" * 110
    print()
    print(bar)
    config = summaries[0].config_name
    print(f"RUN SUMMARY  config={config}  tasks={len(summaries)}")
    print(sep)
    print(
        f"{'task_id':<22} {'outcome':<10} {'cost':>9} {'dur(s)':>8}  "
        f"{'tests':<5} {'lint':<5} {'type':<5}  {'log':<40}"
    )
    total_cost = 0.0
    total_dur = 0.0
    counts: dict[str, int] = {}
    for s in summaries:
        counts[s.status] = counts.get(s.status, 0) + 1
        cost = f"${s.cost_usd:.2f}" if s.cost_usd is not None else "  $?"
        dur = f"{s.duration_s:.0f}" if s.duration_s is not None else "?"
        tests = (
            "PASS" if s.tests_pass else ("FAIL" if s.tests_pass is False else "?")
        )
        lint = (
            "PASS" if s.lint_clean else ("FAIL" if s.lint_clean is False else "?")
        )
        typ = (
            "PASS"
            if s.typecheck_clean
            else ("FAIL" if s.typecheck_clean is False else "?")
        )
        outcome = s.outcome or s.status
        log_col = str(s.log_path) if s.log_path is not None else "(stdout)"
        print(
            f"{s.task_id:<22} {outcome:<10} {cost:>9} {dur:>8}  "
            f"{tests:<5} {lint:<5} {typ:<5}  {log_col}"
        )
        if s.cost_usd is not None:
            total_cost += s.cost_usd
        if s.duration_s is not None:
            total_dur += s.duration_s
    print(sep)
    counts_str = ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))
    print(
        f"{len(summaries)} tasks: {counts_str}   "
        f"total cost ${total_cost:.2f}   total dur {total_dur / 60:.0f}m"
    )
    print(bar)


def compute_exit_code(summaries: list[EvalSummary]) -> int:
    """Exit 0 only if every task ended with outcome=done."""
    if not summaries:
        return 1
    return 0 if all(s.status == "done" for s in summaries) else 1


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one or more vLLM evals on a single RunPod pod.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        required=True,
        choices=ALL_VLLM_CONFIG_NAMES,
        help="vLLM eval config name from orchestrator.evals.configs",
    )

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--task",
        default=None,
        help="Single task id (backwards-compat alias for --tasks X)",
    )
    g.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task ids, e.g. df_task_12,df_task_13,df_task_18",
    )
    g.add_argument(
        "--all-tasks",
        action="store_true",
        help="Run every df_task_*.json in evals/tasks/",
    )

    # Default: OS-assigned free port (resolved in main() via
    # _pick_free_port). This replaces the old hardcoded 8200 default
    # that collided when multiple run_vllm_eval.py ran in parallel —
    # see docs/plan-vllm-404-bug-fix.md. ``ExitOnForwardFailure=yes``
    # on the ssh tunnel is the safety net if --port is set explicitly
    # and collides.
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help=(
            "Local port for the ssh tunnel to vLLM. Default: an "
            "OS-assigned free port (avoids collision when multiple "
            "run_vllm_eval.py run in parallel)."
        ),
    )
    p.add_argument("--datacenter", default=None,
                   help="RunPod datacenter id (default: any available)")
    p.add_argument(
        "--volume",
        action="store_true",
        help="Attach the shared network volume (locks DC to US-NC-1 unless --datacenter overrides)",
    )
    p.add_argument(
        "--image",
        default=None,
        help="Override Docker image (default: from config)",
    )
    p.add_argument(
        "--gpu-type",
        default=None,
        help="Force a specific RunPod GPU type id",
    )

    p.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help=(
            "Number of tasks to run in parallel against the same pod. "
            "Default 5 matches MAX_NUM_SEQS=5 in the entrypoint. "
            "Concurrent mode redirects per-task subprocess output to "
            "/var/tmp/dark-factory-evals/. Ctrl-C waits for in-flight "
            "tasks to drain."
        ),
    )

    p.add_argument(
        "--verify-baseline-clean",
        default="strict",
        choices=["strict", "warn", "skip"],
        help=(
            "strict: refuse dirty baselines (default); "
            "warn: log and proceed; skip: don't preflight at all"
        ),
    )

    p.add_argument(
        "--task-timeout-min",
        type=int,
        default=70,
        help=(
            "Outer subprocess timeout in minutes (the launcher kills the "
            "orchestrator subprocess after this). Should be > "
            "--orch-timeout-min so the orchestrator hits its own timeout "
            "first and produces a clean result file."
        ),
    )

    p.add_argument(
        "--orch-timeout-min",
        type=int,
        default=None,
        help=(
            "Inner orchestrator-internal eval timeout in minutes (passed to "
            "`orchestrator eval --timeout`). When unset, the orchestrator "
            "uses the per-task spec value (default 60 min). Set this much "
            "higher (e.g. 360) so 'would-be-done' runs have time to finish."
        ),
    )

    p.add_argument(
        "--stop-on-first-failure",
        action="store_true",
        help="Abort the multi-task loop on first failed task; pod still terminated",
    )

    p.add_argument(
        "--pod-id",
        default=None,
        help=(
            "Reuse an existing RunPod pod instead of creating a new one. "
            "Skips pod creation and pod termination — the pod is left "
            "running after the eval finishes. Useful for the test-debug "
            "cycle: kill -9 a running launcher to keep the pod alive, "
            "then re-run with --pod-id to reattach."
        ),
    )

    p.add_argument(
        "--vllm-url",
        default=None,
        help=(
            "Direct vLLM endpoint URL (e.g. http://leo-workstation:8000). "
            "Used for workstation configs (image=None) to bypass RunPod "
            "pod lifecycle entirely. Required for workstation configs; "
            "auto-detected from config name when unset."
        ),
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.port is None:
        args.port = _pick_free_port()
        log(f"Auto-picked local port: {args.port}")

    if args.concurrency < 1:
        log(f"ERROR: --concurrency must be >= 1, got {args.concurrency}")
        return 1
    if args.concurrency > 5:
        log(
            f"WARN: --concurrency={args.concurrency} > 5 may exhaust the "
            f"reviewer cap (each eval fires 5 sonnet reviewers; "
            f"{args.concurrency} concurrent tasks = "
            f"{args.concurrency * 5} simultaneous reviewer calls)"
        )

    cfg = get_config_by_name(args.config)
    if cfg is None:
        log(f"ERROR: unknown config '{args.config}'")
        return 1

    is_workstation = cfg.image is None

    # Workstation configs require --vllm-url (or a sensible default).
    if is_workstation:
        if args.vllm_url is None:
            log(
                f"ERROR: config '{args.config}' is a workstation config "
                f"(image=None) — --vllm-url is required "
                f"(e.g. --vllm-url http://leo-workstation:8000)"
            )
            return 1
        log(f"Workstation mode: direct connection to {args.vllm_url}")
    else:
        # Fail fast if RunPod credentials are missing — don't burn preflight
        # time only to discover the key is unset right before pod creation.
        try:
            _load_runpod_api_key()
        except RuntimeError as e:
            log(f"ERROR: {e}")
            return 1

    task_ids = resolve_task_ids(args)
    log(f"Run plan: config={args.config} tasks={task_ids}")

    # Preflight every spec BEFORE pod cold-start. Refuse cheaply.
    specs: list[dict] = []
    for tid in task_ids:
        try:
            spec = load_task_spec(tid)
        except TaskSpecNotFound as e:
            log(f"ERROR: {e}")
            return 1

        if args.verify_baseline_clean != "skip":
            log(f"Preflighting baseline for {tid}...")
            status = preflight_baseline(spec)
            if not status.is_clean:
                if args.verify_baseline_clean == "strict":
                    log(
                        f"REFUSING: baseline dirty for {tid}: "
                        f"head_matches={status.head_matches} "
                        f"lint={status.lint_clean} typecheck={status.typecheck_clean}"
                    )
                    log(f"  pre_task_commit={status.pre_task_commit}")
                    if status.error:
                        log(f"  preflight error: {status.error}")
                    log(
                        f"  Fix: cherry-pick a cleanup commit on top of "
                        f"{status.pre_task_commit[:10]}, then update "
                        f"{tid}.json's pre_task_commit field. Repeat for any "
                        f"sibling tasks sharing this baseline."
                    )
                    log(
                        "  To bypass for one-off debugging, pass "
                        "--verify-baseline-clean=skip"
                    )
                    return 2
                else:
                    log(
                        f"WARN: baseline dirty for {tid}, continuing: "
                        f"lint={status.lint_clean} type={status.typecheck_clean}"
                    )
            else:
                log(f"  baseline clean: {tid}")
        specs.append(spec)

    env = build_eval_env()
    handle: PodHandle | None = None
    summaries: list[EvalSummary] = []

    try:
        try:
            if is_workstation:
                handle = _connect_workstation(cfg, args)
            elif args.pod_id:
                handle = reuse_pod(cfg, args)
            else:
                handle = bring_up_pod(cfg, args)
        except PodBringupFailed as e:
            log(f"FATAL: pod bringup failed: {e}")
            return 1

        if args.concurrency == 1:
            # Sequential path — preserves byte-identical behavior of the
            # single-task gate-eval CI path. subprocess output streams live
            # to the launcher TTY.
            for spec in specs:
                try:
                    summary = run_one_task(spec, cfg, handle, env, args)
                except Exception as e:
                    summary = EvalSummary.crashed(spec["id"], cfg.name, e)
                    log(f"CRASH in task {spec['id']}: {e}")

                summaries.append(summary)
                log_one_summary(summary)

                if args.stop_on_first_failure and summary.status != "done":
                    log(
                        "--stop-on-first-failure tripped; "
                        "aborting remaining tasks"
                    )
                    break
        else:
            # Concurrent path — windowed submission against a thread pool.
            # In-flight tasks always drain naturally; --stop-on-first-failure
            # just stops submitting NEW work, and KeyboardInterrupt is honored
            # via the executor's wait=True shutdown semantics.
            log(
                f"Running {len(specs)} tasks at "
                f"concurrency={args.concurrency} (per-task logs in "
                f"{EVAL_LOG_DIR})"
            )
            remaining: list[dict] = list(specs)
            in_flight: dict = {}  # Future → spec

            with ThreadPoolExecutor(
                max_workers=args.concurrency,
                thread_name_prefix="eval-task",
            ) as executor:
                while remaining or in_flight:
                    # Refill the window up to max concurrency.
                    while remaining and len(in_flight) < args.concurrency:
                        spec = remaining.pop(0)
                        fut = executor.submit(
                            run_one_task, spec, cfg, handle, env, args
                        )
                        in_flight[fut] = spec

                    # Wait for at least one to finish.
                    done, _ = wait(
                        in_flight.keys(), return_when=FIRST_COMPLETED
                    )
                    for fut in done:
                        spec = in_flight.pop(fut)
                        try:
                            summary = fut.result()
                        except Exception as e:
                            summary = EvalSummary.crashed(
                                spec["id"], cfg.name, e
                            )
                            log(f"CRASH in task {spec['id']}: {e}")
                        summaries.append(summary)
                        log_one_summary(summary)

                        if (
                            args.stop_on_first_failure
                            and summary.status != "done"
                        ):
                            log(
                                "--stop-on-first-failure tripped; draining "
                                f"queue ({len(remaining)} not yet submitted, "
                                f"{len(in_flight)} still in flight will "
                                "finish naturally)"
                            )
                            remaining = []
                            # Do NOT break — keep collecting in_flight
                            # completions so their summaries are recorded.
    finally:
        tear_down_pod(handle)

    print_summary_table(summaries)
    return compute_exit_code(summaries)


if __name__ == "__main__":
    sys.exit(main())
