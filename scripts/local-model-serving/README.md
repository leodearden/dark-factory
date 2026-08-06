# Local model serving substrate (LME-α)

PRD-MARKER:local-memory-models-eval serving

Task 3713 (**α**) of [`plans/local-memory-models-eval-prd.md`](../../plans/local-memory-models-eval-prd.md).

Serves the eval's candidate model endpoints as `systemd --user` units, one arm
at a time, on the single RTX 3090 this box also uses as a desktop. Its whole
job is to make **η** (LLM screening), **θ** (LLM full runs) and **ι**
(embedding arms) able to point at a local endpoint and trust what comes back.

α guarantees *an endpoint that answers correctly*. It does not run the eval, own
the client class (that is **β**), or own the production-grade output validator
(that is **ε**).

---

## The contract: `arms.yaml`

One manifest is the whole contract surface. `lms_serve`, `lms_ctl`,
`lms_fetch_weights`, `lms_healthcheck` and every downstream consumer read from
it and nowhere else, which is what makes it structurally impossible for a tool
to serve one thing while a probe measures another.

```python
import sys; sys.path.insert(0, 'scripts/local-model-serving')
from lms_manifest import load_arms

arm = load_arms().by_id('qwen3.5-9b')
arm.base_url             # 'http://127.0.0.1:8410'  — OpenAI-compatible root
arm.served_model_name    # 'qwen3.5-9b'             — the `model` field to send
arm.structured_output_mode   # 'json_schema' | 'json_object' | 'none'
arm.dims                 # embedding arms only: the vector length ι compares at
load_arms().by_axis('embedding')   # the four embedding arms
```

Read those four fields; do not hardcode a port, a model id or a dimension.
`load_arms()` raises a typed `ArmManifestError` on anything malformed rather
than half-loading — a manifest that dropped the arm it could not parse would
let a report show every *remaining* arm green, which reads as "the slate is
healthy" when the slate is quietly narrower than the PRD commissioned.

`base_url` is always `127.0.0.1`, never `localhost`: the latter can resolve to
`::1` while the server listens on IPv4 only, which presents as a dead arm
(`scripts/run_vllm_eval.py:505-512`).

### The slate

| arm_id | axis | stack | port | structured output | dims | est. VRAM |
|---|---|---|---|---|---|---|
| `qwen3.5-9b` | llm | vllm | 8410 | `json_schema` | — | 6.0 GiB |
| `mistral-small-3.2-24b` | llm | vllm | 8411 | `json_schema` | — | 14.0 GiB |
| `phi-4-14b` | llm | vllm | 8412 | `json_schema` | — | 9.0 GiB |
| `moe-stretch` | llm | llamacpp | 8413 | `json_object` | — | 17.0 GiB ⚠ |
| `qwen3-embedding-0.6b` | embedding | vllm | 8414 | — | 1024 | 2.0 GiB |
| `granite-embedding-english-r2` | embedding | vllm | 8415 | — | 768 | 1.0 GiB |
| `qwen3-embedding-4b` | embedding | vllm | 8416 | — | 2560 | 9.0 GiB |
| `gte-modernbert-base` | embedding | vllm | 8417 | — | 768 | 1.0 GiB |

**Ports 8410–8417 are reserved for this rig**, one per arm, bound to loopback
only. The block was chosen to clear what already listens on this host — 8002
(fused-memory MCP) and 8102 (escalation server). `lms_manifest` refuses a
manifest whose arm declares a port outside the block, and refuses two arms
sharing one: a shared port is the precondition for the 2026-04-08 bug where a
stale unit answered a probe and mis-attributed an entire run's metrics.

⚠ `moe-stretch` does not currently fit — see *Open Q3* below.

---

## Operator recipe

```bash
cd /home/leo/src/dark-factory

# 1. Install the unit template (idempotent; enables and starts nothing).
scripts/local-model-serving/install-lms-units.sh

# 2. Fetch the image and weights for an arm, in transient systemd --user
#    units so a multi-GB download survives your shell closing.
uv run --project shared python scripts/local-model-serving/lms_fetch_weights.py \
    --arm qwen3.5-9b
journalctl --user -u lms-fetch-qwen3.5-9b -f      # follow it

# 3. Start it.  Refuses BEFORE any systemctl call if it will not fit.
uv run --project shared python scripts/local-model-serving/lms_ctl.py \
    start qwen3.5-9b
uv run --project shared python scripts/local-model-serving/lms_ctl.py \
    wait-ready qwen3.5-9b

# 4. Prove it answers with VALID output, not merely 200 OK.
uv run --project shared python scripts/local-model-serving/lms_healthcheck.py \
    --arm qwen3.5-9b

# 5. Stop it before starting the next arm.  One arm at a time.
uv run --project shared python scripts/local-model-serving/lms_ctl.py \
    stop qwen3.5-9b
```

Useful variants: `lms_ctl active` / `stop-all`; `lms_fetch_weights --all`
(non-placeholder arms) or `--images-only`; `lms_healthcheck --active`,
`--all`, and `--output <path>` to write the JSON artifact.

Logs are the journal: `journalctl --user -u lms-arm@qwen3.5-9b.service -f`.

### One arm at a time

`lms_ctl start` is **exclusive by default** — it refuses while another arm's
unit is running. The PRD's funnel never needs the slate up simultaneously, and
on one 24 GB card two arms is an OOM. `--no-exclusive` exists for the case
where you have checked the arithmetic yourself.

`Restart=no` in the unit template is deliberate: a dead arm stays dead rather
than thrashing the GPU in a restart loop while you read the journal.

---

## The health check is the point

"The endpoint is up" is not the question. `lms_healthcheck` asks whether the arm
can do the thing the eval depends on, and it is deliberately harder to pass than
a `/health` probe:

- The probe payload is a **nested** pydantic model, so its `model_json_schema()`
  emits `$defs`/`$ref` — the shape graphiti really emits, and the exact shape
  llama.cpp silently mishandles. A flat stand-in schema would hand a PASS to an
  arm that cannot serve the eval.
- The **same** model that generates the request schema validates the response
  client-side. For a `json_object`-only arm, that leg is the only thing between
  an unconstrained fallback returning prose and a green row.
- A completion counts only after `/v1/models` lists the arm's
  `served_model_name`.
- Embedding vectors are checked for declared length, finiteness, and a non-zero
  L2 norm. An all-zero vector is finite and the right shape and would silently
  turn every retrieval score ι computes into noise.

Exit codes: `0` all green · `1` an arm failed · `2` manifest/arm-id error ·
`3` every arm passed but VRAM is over budget · `4` the GPU probe failed, so
**no report was produced** · `5` `--active` found nothing running.

Codes 3, 4 and 5 are separate on purpose. A budget failure and a model failure
have different fixes. A broken `nvidia-smi` must never degrade into a report
whose VRAM block reads `used 0 MiB, headroom 19.5 GiB` — a passing budget with
maximal headroom is the most trustworthy-looking wrong answer this rig can
emit. And `--active` with nothing running must not exit 0, or a wrapper script
would certify a slate nobody probed.

---

## VRAM budget

PRD D10 budgets "~19–20GB (24GB − ~4GB whisper-writer)". **That is a ceiling,
not this host's operating budget.** Measured 2026-08-05 with the desktop in its
ordinary state:

```
$ nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits
24576, 7362, 16761
```

| | MiB | GiB |
|---|---|---|
| card total | 24576 | 24.0 |
| whisper-writer (pid 7575, resident since 2026-07-03) | 4050 | 3.96 |
| KDE/X11 desktop — Xorg, plasmashell, kwin_x11, obs, ~40 clients | ~3312 | ~3.23 |
| **free: the real operating budget** | **16761** | **≈16.4** |

`--query-compute-apps` lists only whisper-writer; the desktop's graphics
contexts hold the other ~3.3 GB and appear in no compute-app listing at all,
which is why the PRD's arithmetic came out ~3 GiB high.

Both figures are reported on every health run and travel together in the JSON
artifact (`vram.nominal_ceiling_gib` = 19.5, `vram.operating_budget_gib`
= 16.37). Showing only the nominal ceiling would assert capacity this host does
not have; showing only the measured budget would hide the deviation from D10.
The `vram.verdict` is judged against the **nominal** ceiling, because that is
how the PRD states its user-observable signal.

### How `--gpu-memory-utilization` is derived

vLLM's cap is computed per launch from the live reading, never hardcoded:

```
--gpu-memory-utilization = round(free_gib / total_gib, 3)
                         = round(16.37 / 24.0, 3) = 0.682
```

Explicitly **not** the `0.95` that `docs/vllm-eval-status.md:1037` uses: that
figure came from dedicated 96 GB eval pods. On this shared card 0.95 hands vLLM
~23 GiB and evicts whisper-writer, which PRD D10 requires resident.

### Per-arm fit

`lms_vram.arm_fits(arm, free_gib)` requires `est_vram_gib + 0.5 GiB` of margin
against **measured free VRAM**, and both `lms_ctl start` and `lms_serve` refuse
before any side effect if it fails. The margin is what makes "fits" mean
"runs" — an arm sized to the last free byte OOMs on the first allocation spike
(CUDA graph capture, a sampler warmup buffer, a long prompt's KV).

If the desktop is logged out, free VRAM rises by ~3.2 GiB and arms that refuse
today will start. Nothing here assumes a fixed budget; it is re-measured on
every launch.

---

## llama.cpp #21228 — the MoE arm is `json_object`-only

llama.cpp's server silently falls back to **unconstrained** output when handed a
JSON schema containing `$ref`/`$defs` ([ggml-org/llama.cpp#21228]) — it reports
success and returns whatever the model felt like. graphiti's real extraction
schemas are exactly that shape.

So the MoE arm declares `structured_output_mode: json_object`, and
`lms_manifest` **rejects** any `llamacpp` arm claiming `json_schema`: the claim
is false by construction, and a manifest that permitted it would let a
downstream task believe a constraint that is not being enforced. The launch argv
for that arm carries no grammar / json-schema / guided-decoding flag either —
structural, so nobody can later add one "helpfully".

Division of labour:

| | owner |
|---|---|
| an endpoint that answers, and a probe proving it answers *validly* | **α** (here) |
| forcing the right client class per arm (`llm.client_class`, base_url) | **β** |
| the production-grade client-side output validator in the arm runner | **ε** |

α's client-side validation is a health check, not that validator. It exists so
an unconstrained arm cannot show up green here; ε owns the version the eval runs
against.

[ggml-org/llama.cpp#21228]: https://github.com/ggml-org/llama.cpp/issues/21228

---

## Verification artifact

`verification/health-report.json` is written by a live run
(`lms_healthcheck --all --output ...`) and committed. It carries
`schema_version`, an aware-UTC `measured_at`, the GPU identity (which card,
which driver — every verdict is relative to specific hardware), one row per arm,
and the VRAM block. `scripts/tests/test_lms_verification_artifact.py` requires a
`PASS` row for every arm in `arms.yaml` plus a passing VRAM block, so the test
can only be greened by the run having actually happened.

---

## Open questions

Two of the PRD's tactical open questions are delegated to this task and are
resolved from live measurement, not from the docs.

### Open Q1 — embedding serving stack

> TEI vs vLLM-pooling vs in-process for batch re-embeds. Decide in α/ι.

**Status: pending live measurement.** All four embedding arms currently declare
`stack: vllm` (the pooling runner), with TEI recorded per-arm as the fallback in
`fallback_stack`. `lms_serve` already builds a TEI argv, so switching an arm is a
manifest edit, not a code change.

*To be filled in by step 22: which arms load under the vLLM pooling runner,
which needed TEI, and the measured query latency behind the choice. ι owns the
batch-re-embed half of the question.*

### Open Q3 — MoE stretch-arm model and quant

> Which of the two models, GGUF quant level, client-side validator placement.
> Decide in α/η.

**Status: unresolved — the arm is a placeholder.** `moe-stretch` carries `TBD-`
values, so `ArmEntry.is_placeholder` is true and every tool refuses to launch or
fetch it rather than 404ing on a literal `TBD-Q3` model id and recording that as
an arm failure.

The measured constraint: its nominal ~17 GiB does **not** fit this host's
~16.4 GiB operating budget, though it would have fitted PRD D10's nominal 19.5.
The resolution must therefore be one of — a smaller quant, a smaller MoE, or an
explicit "requires the desktop VRAM freed" caveat — chosen against the measured
figure.

*To be filled in by step 22: the selected model, quant, image, measured resident
footprint, and which of the three routes was taken. Validator placement is ε's.*
