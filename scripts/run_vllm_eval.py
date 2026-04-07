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
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

sys.path.insert(0, "/home/leo/src/runpod-toolkit")
sys.path.insert(0, "/home/leo/src/dark-factory/orchestrator/src")
from runpod_toolkit.config import RunPodConfig
from runpod_toolkit.compute import RunPodClient, PodStatus
from orchestrator.evals.configs import (
    get_config_by_name,
    VLLM_EVAL_CONFIGS,
    EvalConfig,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUNPOD_KEY = "rpa_VLRVNJ8HB5CH7MQZL9WW2XPQBQO18V3PMA1H1BSM11niy2"
SSH_KEY = os.path.expanduser("~/.ssh/id_runpod")
SSH_PUBKEY = open(SSH_KEY + ".pub").read().strip()

PROJECT_ROOT = Path("/home/leo/src/dark-factory")
ORCHESTRATOR_DIR = PROJECT_ROOT / "orchestrator"
TASKS_DIR = ORCHESTRATOR_DIR / "src/orchestrator/evals/tasks"
RESULTS_DIR = ORCHESTRATOR_DIR / "src/orchestrator/evals/results"

# Fallback GPU types when neither the config nor --gpu-type provides one.
GPU_TYPES = [
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA H200",
]

# RunPod-targetable config names: those with an image set in configs.py.
RUNPOD_CONFIG_NAMES = [c.name for c in VLLM_EVAL_CONFIGS if c.image is not None]


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
        return sorted(p.stem for p in TASKS_DIR.glob("df_task_*.json"))
    if args.tasks:
        return [t.strip() for t in args.tasks.split(",") if t.strip()]
    if args.task:
        return [args.task]
    raise ValueError("No task selection — use --task, --tasks, or --all-tasks")


# ---------------------------------------------------------------------------
# Baseline preflight (B3)
# ---------------------------------------------------------------------------


def preflight_baseline(
    spec: dict, *, repo_root: Path = PROJECT_ROOT
) -> BaselineStatus:
    """Verify a task's ``pre_task_commit`` is lint+typecheck clean.

    Creates a throwaway worktree at ``pre_task_commit``, runs the spec's
    ``setup_commands`` (so the venv exists), then runs the spec's lint and
    typecheck commands. Tests are deliberately skipped (slow + flaky on dev
    box). The throwaway worktree is removed before returning even on error.
    """
    task_id = spec["id"]
    sha = spec["pre_task_commit"]
    verify = spec.get("verify_commands", {})
    lint_cmd = verify.get("lint")
    type_cmd = verify.get("typecheck")

    if not lint_cmd or not type_cmd:
        return BaselineStatus(
            task_id=task_id,
            pre_task_commit=sha,
            head_matches=False,
            lint_clean=False,
            typecheck_clean=False,
            error="spec is missing verify_commands.lint or verify_commands.typecheck",
        )

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

        log(f"  preflight lint: {lint_cmd}")
        lint_rc = subprocess.run(
            lint_cmd,
            shell=True,
            executable="/bin/bash",
            cwd=str(wt),
            timeout=300,
        ).returncode

        log(f"  preflight typecheck: {type_cmd}")
        type_rc = subprocess.run(
            type_cmd,
            shell=True,
            executable="/bin/bash",
            cwd=str(wt),
            timeout=900,
        ).returncode

        return BaselineStatus(
            task_id=task_id,
            pre_task_commit=sha,
            head_matches=head_ok,
            lint_clean=(lint_rc == 0),
            typecheck_clean=(type_rc == 0),
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


def wait_for_vllm(port: int, timeout: int = 900) -> bool:
    """Poll vLLM /health until it returns 200 or *timeout* elapses."""
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


def vllm_healthy(port: int, timeout: int = 5) -> bool:
    """Single-shot health probe used at the top of each task iteration."""
    import urllib.request

    try:
        resp = urllib.request.urlopen(
            f"http://localhost:{port}/health", timeout=timeout
        )
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
    use_volume = not args.no_volume

    extra_env = {
        k: cfg.env_overrides[k]
        for k in (
            "TOOL_CALL_PARSER",
            "QUANTIZATION",
            "TP_SIZE",
            "MAX_MODEL_LEN",
            "GPU_MEMORY_UTIL",
            "MAX_NUM_SEQS",
            "ENFORCE_EAGER",
        )
        if k in cfg.env_overrides
    }

    if use_volume:
        volume_id = "obxma9bf1b"
        datacenter = "US-NC-1"
        container_disk = 50
    else:
        volume_id = None
        datacenter = None

    if args.gpu_type:
        gpu_types_to_try = [args.gpu_type]
    elif cfg.gpu_type:
        gpu_types_to_try = [cfg.gpu_type]
    else:
        gpu_types_to_try = GPU_TYPES

    config = RunPodConfig(api_key=RUNPOD_KEY)
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

        log(f"Starting SSH tunnel (localhost:{args.port} → pod:8000)")
        tunnel_proc = subprocess.Popen(
            [
                "ssh",
                "-N",
                "-L",
                f"{args.port}:localhost:8000",
                "-i",
                SSH_KEY,
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ServerAliveCountMax=3",
                f"root@{pod.ssh_host}",
                "-p",
                str(pod.ssh_port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        time.sleep(3)

        health_timeout = max(900, container_disk * 12)
        log(
            f"Waiting for vLLM to load model "
            f"(timeout {health_timeout // 60} min)..."
        )
        if not wait_for_vllm(args.port, timeout=health_timeout):
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
            vllm_url=f"http://localhost:{args.port}",
            local_port=args.port,
            config_name=cfg.name,
        )

    except Exception:
        # Clean up any partial state before re-raising as PodBringupFailed.
        partial = PodHandle(
            pod=pod,
            tunnel_proc=tunnel_proc,
            client=client,
            vllm_url=f"http://localhost:{args.port}",
            local_port=args.port,
            config_name=cfg.name,
        )
        tear_down_pod(partial)
        raise


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
) -> Path | None:
    """Glob results/ for a result file newer than *since_mtime*.

    Returns the newest matching file, or None if no new file appeared.
    """
    if not RESULTS_DIR.exists():
        return None
    pattern = f"{task_id}__{config_name}__*.json"
    candidates = [
        p for p in RESULTS_DIR.glob(pattern) if p.stat().st_mtime > since_mtime
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

    if not vllm_healthy(handle.local_port):
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
    log(f"Running eval: config={config_name} task={task_id}")
    t0 = time.monotonic()

    try:
        result = subprocess.run(
            [
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
            ],
            cwd=str(ORCHESTRATOR_DIR),
            env=env,
            timeout=args.task_timeout_min * 60,
        )
        rc = result.returncode
    except subprocess.TimeoutExpired:
        log(
            f"EVAL TIMEOUT: {task_id} exceeded "
            f"{args.task_timeout_min} min subprocess limit"
        )
        rc = -1

    duration_s = time.monotonic() - t0

    new_path = find_new_result_file(task_id, config_name, pre_max_mtime)
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

    bar = "=" * 78
    sep = "-" * 78
    print()
    print(bar)
    config = summaries[0].config_name
    print(f"RUN SUMMARY  config={config}  tasks={len(summaries)}")
    print(sep)
    print(
        f"{'task_id':<22} {'outcome':<10} {'cost':>9} {'dur(s)':>8}  "
        f"{'tests':<5} {'lint':<5} {'type':<5}"
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
        print(
            f"{s.task_id:<22} {outcome:<10} {cost:>9} {dur:>8}  "
            f"{tests:<5} {lint:<5} {typ:<5}"
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
        choices=RUNPOD_CONFIG_NAMES,
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

    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--datacenter", default="US-NC-1")
    p.add_argument(
        "--no-volume",
        action="store_true",
        help="Use baked Docker image instead of volume",
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
        default=1,
        choices=[1],
        help="v1 only supports 1; future: parallel task fanout",
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
        help="Per-task subprocess timeout in minutes",
    )

    p.add_argument(
        "--stop-on-first-failure",
        action="store_true",
        help="Abort the multi-task loop on first failed task; pod still terminated",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cfg = get_config_by_name(args.config)
    if cfg is None or cfg.image is None:
        log(f"ERROR: config '{args.config}' is not a vLLM/RunPod config")
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
            handle = bring_up_pod(cfg, args)
        except PodBringupFailed as e:
            log(f"FATAL: pod bringup failed: {e}")
            return 1

        for spec in specs:
            try:
                summary = run_one_task(spec, cfg, handle, env, args)
            except Exception as e:
                summary = EvalSummary.crashed(spec["id"], cfg.name, e)
                log(f"CRASH in task {spec['id']}: {e}")

            summaries.append(summary)
            log_one_summary(summary)

            if args.stop_on_first_failure and summary.status != "done":
                log("--stop-on-first-failure tripped; aborting remaining tasks")
                break
    finally:
        tear_down_pod(handle)

    print_summary_table(summaries)
    return compute_exit_code(summaries)


if __name__ == "__main__":
    sys.exit(main())
