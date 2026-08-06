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

## Provisioned runtimes (measured, step 19)

Both serving images are pinned in `arms.yaml` by **tag and digest**. A tag can be
re-pushed under you; a digest cannot, and `lms_serve.image_ref()` prefers the
digest whenever one is present — so a re-run serves the same bits, not merely the
same name.

| stack | pinned tag | digest | measured version |
|---|---|---|---|
| vLLM | `vllm/vllm-openai:v0.26.0` | `sha256:ffb2d59b…4abf52` | vLLM **0.26.0**, torch 2.11.0+cu130 |
| llama.cpp | `ghcr.io/ggml-org/llama.cpp:server-cuda-b10276` | `sha256:48a88af7…17dd2` | `llama-server --version` → **10276 (6ea215d17)** |

vLLM 0.26.0 satisfies the PRD's "≥0.26" requirement as a measurement, not a
claim. The build-number tag is used for llama.cpp rather than the floating
`server-cuda`; both resolved to the same digest when measured (2026-08-06), and
the numbered one stays legible after `server-cuda` moves on.

TEI is deliberately **not** pulled yet. It is only needed if Open Q1 sends an
embedding arm to the fallback stack, which step 22 decides from measurement.

### What the live smoke actually proved

Run as a transient `systemd --user` unit on the reserved port 8410, against
`HuggingFaceTB/SmolLM2-135M-Instruct` — a non-slate stand-in, since step 19
downloads no slate weights:

- **GPU passthrough into the container.** `torch.cuda.is_available()` → `True`,
  `NVIDIA GeForce RTX 3090` from inside the image. Docker's default runtime here
  is `runc`, so this confirms the per-run `--gpus all` in `_docker_preamble` is
  doing the work.
- **Port plumbing and model identity.** `/v1/models` answered 200 on 8410 with
  `id: "lms-smoke"` — the `--served-model-name`, not the HF repo id. That is
  precisely the field `check_model_identity` matches on, so the identity leg that
  guards against the 2026-04-08 404 bug is verified end to end.
- **Structured outputs on a `$defs`/`$ref` schema.** The committed
  `build_llm_probe_request` body — whose schema genuinely carries `$defs` and
  `$ref` — was accepted by vLLM 0.26's structured-outputs path, and the returned
  text was constrained to exactly the nested probe shape. This is the capability
  the whole eval rests on, and it is now measured rather than assumed.
- **Coexistence.** whisper-writer held its 4050 MiB throughout, and stopping the
  unit released every byte the arm had taken (free VRAM returned to 16840 MiB).

Two findings worth carrying forward rather than burying:

1. `hf-internal-testing/tiny-random-LlamaForCausalLM` is **not** usable as a
   plumbing stand-in on this stack: its `head_dim=4` trips flex_attention's
   `NYI: embedding dimension … must be at least 16`. That is a property of the
   toy model, not of the image — but it costs a wasted 60 s start, so the note
   is here.
2. The 135M stand-in hit `PROBE_MAX_TOKENS` (512) and its truncated-but-
   structurally-correct output was reported as `not_json`. Constrained decoding
   worked; the model simply looped. A real slate arm should not, but if step 23
   sees `not_json` on an arm whose content is valid JSON up to the cut, the cause
   is the token budget and the reason code deserves to distinguish
   `finish_reason == "length"` — fix it there, with the offline test first.

---

## End-to-end chain smoke (measured, step 20)

Step 19 proved the *image*. Step 20 proved the **committed chain**, run for real
on the cheapest arm on the slate — `granite-embedding-english-r2`, 149M, 573 MB
of weights — before committing to the full slate's downloads. Every command was
the committed one; nothing was hand-run:

```
lms_fetch_weights --arm granite-embedding-english-r2 --weights-only   # transient unit, 11 s
lms_ctl start granite-embedding-english-r2                            # pre-flight → systemctl
lms_ctl wait-ready granite-embedding-english-r2                       # ready in 123 s
lms_healthcheck --arm granite-embedding-english-r2                    # PASS, exit 0
lms_ctl stop granite-embedding-english-r2
```

| measurement | value |
|---|---|
| weight fetch (transient unit, cold) | 11 s / 573 MB |
| unit start → `/v1/models` serving the arm | **123 s** (`wait-ready`, 5 s poll) |
| — container start + vLLM import | 38 s |
| — engine-core boot to "Starting to load model" | 50 s |
| — weight load | 1.6 s (0.29 GiB) |
| — profile, KV cache, warmup | 22.0 s (16.6 s of it `torch.compile`) |
| **resident VRAM while serving** | **796 MiB** (7309 → 8105 MiB used) |
| embeddings probe latency | 192 ms |
| VRAM after `stop` | 7309 MiB — **exactly** the pre-start baseline |
| whisper-writer throughout | 4050 MiB, undisturbed |

What this establishes that step 19 could not: the unit template instantiates
against a real `arms.yaml` arm, the pre-flight admits an arm that fits, the
launcher's argv serves the manifest's `served_model_name` (so `wait-ready`'s
identity leg matches), port 8415 of the reserved block is reachable, the
embeddings probe validates a real 768-dim vector, and `ExecStop=docker stop`
plus `--rm` release **every** byte and leave no container behind.

### Two findings for step 23

1. **`--gpu-memory-utilization` behaves differently per runner.** The pooling
   runner treated the derived `0.684` as a *ceiling* and took 796 MiB — nowhere
   near the 16.8 GiB that fraction of the card would allow. The **generate**
   runner sizes its KV cache to actually fill that budget, so the three vLLM LLM
   arms will behave nothing like this arm did. Do not read "the embedding arm
   only took 796 MiB" as evidence the derivation is conservative; step 23 must
   measure each LLM arm's resident footprint on its own.
2. **The 123 s start is mostly fixed cost, not model size.** Only 1.6 s of it
   was weights. A 14 GiB AWQ arm adds load time on top of the same ~110 s floor,
   which is why the unit's `TimeoutStartSec=900` is not generous — it is roughly
   right, and `wait-ready`'s default 900 s timeout matches it deliberately.

### Defect found and fixed by this run

`lms_fetch_weights` echoed the argv it was about to submit **including the real
`--setenv=HF_TOKEN=hf_…` value**, putting the secret on the operator's terminal
and into any transcript or log capturing it. Fixed with `redact_argv()`, which
returns a printable copy with the value masked and the flag left visible, while
the executed argv keeps the real token — the two are deliberately different
objects, and `test_redact_argv_does_not_mutate_the_argv_that_gets_executed`
exists because a redaction that leaked into the executed argv would turn every
download anonymous and fail only later, only on the gated repos.

---

## RESOLVED: the budget verdict's subject (esc-3713-6, 2026-08-06)

Two budget notions in this package disagreed about their **subject**, and the
disagreement failed arms that were serving correctly:

| check | subject | ceiling | `qwen3.5-9b` |
|---|---|---|---|
| `arm_fits` (pre-flight) | the **arm's** footprint | measured free, 16.4 GiB | 6.0 + 0.5 <= 16.37 -> **admits** |
| `evaluate_budget` (old) | **total** card usage | PRD nominal, 19.5 GiB | 21.75 > 19.5 -> **FAILed** |

An arm passed the pre-flight and then failed the verdict having done nothing
wrong. The PRD's own arithmetic says which subject is intended:

> l.192 — GPU headroom measured (24GB − ~4GB whisper-writer)
> l.165 — whisper-writer stays resident: all capacity math against ~19–20GB, not 24GB

19.5 GiB is what D10 derives as available **to the arm**. Applying it to *total*
usage charged the arm for the 7.3 GiB desktop+whisper baseline a second time —
and the desktop's ~3.2 GiB was never in D10's model at all. It was not a
big-arm technicality either: measurement showed **every** LLM arm failing, a 9B
AWQ included, because a generate arm legitimately takes the free-derived share
and that share plus the baseline exceeds 19.5 regardless of model size.

The reviewer approved the correction (esc-3713-6) *before* the artifact existed,
which is why re-pointing an assertion inside the anti-fabrication gate is
legitimate here and would not have been on the agent's own initiative.

### What changed

1. **`evaluate_budget` judges `used − baseline` against the live free reading**
   taken at that same baseline. `arm_footprint_mib <= budget_mib`.
2. **The baseline is live and per-arm.** `lms_ctl start` records the nvidia-smi
   reading it just pre-flighted on, between the refusal path and the
   `systemctl start`, into `$XDG_RUNTIME_DIR/lms-baselines/<arm>.json`
   (`LMS_BASELINE_DIR` overrides). A missing baseline produces **no report at
   all** — the same stance as a dead GPU probe. `MEASURED_BASELINE_GIB` and
   `MEASURED_OPERATING_BUDGET_GIB` remain as documented reference values and
   are never subtracted: a frozen baseline misattributes desktop drift to the
   arm, in the direction that flatters it.
3. **The safety margin is applied once**, at allocation time — not re-added in
   the verdict, which would double-charge it.
4. **The generate branch of `_memory_share_for` now reserves that margin too.**
   It was deriving the share from the *whole* free reading, so the arms with the
   largest allocations were the only ones sized to the last byte of free VRAM —
   precisely what `SAFETY_MARGIN_GIB` exists to prevent, on a card that must
   keep whisper-writer resident. This is also what makes the verdict satisfiable
   by construction: the share bounds the footprint at `free − margin`, so
   `footprint <= free` holds with headroom rather than by luck.
5. **PRD D10's 19.5 GiB stays in every report as a reported, non-gating field**,
   so the deviation this task measured remains legible.

The gate assertion moved with it, and got stronger rather than weaker: it now
re-derives `footprint == used − baseline`, requires `0 < baseline < used`, and
requires a positive footprint for every row — three checks where there was one.

### RESOLVED alongside: the pooling arms' KV balloon

vLLM sizes its paged KV cache to fill whatever `--gpu-memory-utilization`
allows, and does so for a **decoder** model under `--runner pooling` even though
a pooling model can never read one. `qwen3-embedding-0.6b` was handed 0.682
(16.37 GiB) and filled it: 1.12 GiB of weights and **14.56 GiB of unusable KV**.
`_memory_share_for` now bounds a pooling arm to `est_vram_gib +
SAFETY_MARGIN_GIB`. Bounding via the share rather than
`--kv-cache-memory-bytes` keeps `est_vram_gib` load-bearing for the pre-flight,
so a too-small declaration fails loudly at startup instead of silently eating
the card — which is how `0.6b` (2.0 -> 3.0) and `4b` (9.0 -> 10.0) were caught.

---

## Live slate run (measured, step 23 — 2026-08-06, IN PROGRESS)

Every arm run through the committed chain, one at a time, on a host whose
baseline was confirmed clean first (7302 MiB used, whisper-writer the only
compute app at 4050 MiB, no ollama model resident — esc-3713-7). Six of eight
arms answer correctly and fit; two carry real, diagnosed defects.

| arm | ready | resident (footprint) | budget | probe | verdict |
|---|---|---|---|---|---|
| `granite-embedding-english-r2` | 116 s | 762 MiB | 16814 MiB | 606 ms | **PASS** |
| `gte-modernbert-base` | 114 s | 787 MiB | 16839 MiB | 102 ms | **PASS** |
| `qwen3-embedding-0.6b` | 155 s | 3581 MiB | 16839 MiB | 706 ms | **PASS** |
| `qwen3-embedding-4b` | 158 s | 10113 MiB | 16839 MiB | 828 ms | **PASS** |
| `qwen3.5-9b` | 451 s | 14441 MiB | 16847 MiB | 2849 ms | **PASS** |
| `phi-4-14b` | 196 s | 15456 MiB | 16831 MiB | 3054 ms | **PASS** |
| `mistral-small-3.2-24b` | — | — | — | — | **start refused** |
| `moe-stretch` | 31 s | 14611 MiB | 16820 MiB | 5600 ms | **`empty_completion`** |

Every arm released **every** byte on stop: the card returned to 7275–7310 MiB
after each one, and whisper-writer was undisturbed throughout.

The subject correction is what makes the LLM arms reportable at all. Under the
old total-usage reading `qwen3.5-9b` measured 21.75 GiB and failed; it takes
14.10 GiB of the 16.45 GiB that was free before it started. The generate-branch
safety margin is visible in the numbers too — the same arm's footprint fell from
14961 MiB to 14441 MiB, and every LLM arm now lands ~1.4 GiB clear of its budget
instead of at the last byte.

### Two defects the live run found

1. **`mistral-small-3.2-24b` — the manifest declared the wrong quant.**
   `jeffcookio/Mistral-Small-3.2-24B-Instruct-2506-awq-sym` is named `awq` but
   its own `config.json` says `quant_method: compressed-tensors` (format
   `pack-quantized`, 4 bits). vLLM 0.26 refused the model outright:
   `Quantization method specified in the model config (compressed-tensors) does
   not match the quantization method specified in the 'quantization' argument
   (awq)`. That refusal is the right behaviour — a mismatch guessed past would
   serve numbers the eval would then attribute to a model it did not run.
   `arms.yaml` is corrected to `compressed-tensors` from the measurement; the
   arm has not yet been re-run under the correction.

2. **`moe-stretch` — the probe's token budget is spent on thinking.**
   The arm loads and serves (31 s to ready, 14611 MiB, 104 t/s), and llama.cpp's
   own log shows it generating the full `PROBE_MAX_TOKENS` (512) and stopping on
   length: `eval time = 4904.60 ms / 512 tokens`. `message.content` came back
   empty, so the probe reported `empty_completion`. **Root-caused and resolved —
   see *Reasoning mode* below.** It was never a token-budget question: the arm
   was being served with thinking force-enabled by a stack default nobody chose.

Neither is a budget failure and neither is hand-editable. `health-report.json`
is therefore **not yet committed**: the anti-fabrication gate stays red until
the slate is re-run under the corrected probe contract.

---

## Reasoning mode — declared, never inherited (esc-3713-10)

Diagnosed live on 2026-08-06. Two findings, and the second is the serious one.

### The mode was being set by accident, differently per arm

`moe-stretch`'s empty completion was not a budget problem. `/apply-template`
shows the mechanism directly: gemma-4's own chat template defaults
`enable_thinking` to **false** and suppresses reasoning by pre-closing the
thought channel (`<|turn>model\n<|channel>thought\n<channel|>`) — and llama.cpp
b10276 **overrides that to true**. 509 of the 512 tokens went into an
unterminated thought; `message.reasoning_content` held 1912 chars and
`message.content` was `''`.

Meanwhile `qwen3.5-9b`'s template defaults thinking **on** and ends the prompt
with an *open* `<think>` tag — and vLLM, with no `--reasoning-parser`, fills the
xgrammar bitmask from token 0, so the model is told to think and then
structurally prevented from doing so.

So the two thinking-capable arms were running in **opposite modes, by accident,
and nothing recorded it** — a confound sitting inside the eval's own independent
variable. `arms.yaml` now declares `reasoning:` per arm (`lms_manifest` refuses
an LLM arm that omits it) and the report row carries the mode it was measured
in. `phi-4-14b` and `mistral-small-3.2-24b` have no thinking construct at all.

| arm | mode | tokens | latency | probe entities found |
|---|---|---|---|---|
| `moe-stretch` | off | 236–276 | 2.3–2.9 s | **4/4** |
| `moe-stretch` | on | 1807 (84% thought) | 16.5–17.2 s | **4/4** |
| `qwen3.5-9b` | off | 18–39 | 0.35 s warm | **0/4** |
| `qwen3.5-9b` | on + `--reasoning-parser qwen3` | 2639 | ~35 s | **4/4** |

Reasoning is therefore **model-dependent, not generally good**: it buys gemma-4
nothing at 6.4× the wall clock, and is the difference between an extraction and
an empty object for qwen3.5. Which mode each arm is measured in is **ζ's**
pre-registered call (task 3719), expressed by setting these manifest fields —
not a per-run choice at η or θ.

### The probe was passing empty extractions

`qwen3.5-9b`'s committed **PASS** was a pass on
`{"entities": [], "summary": "No entities extracted from the provided text."}`.
`ProbeExtraction` accepts it because an empty list is a valid
`list[ProbeEntity]` — so the client-side validator this package calls its
safeguard could not tell a correct extraction from a refusal to extract, and
every non-reasoning qwen configuration returned zero entities.

`verify_llm_response` now applies an **extraction floor**: the completion must
name at least `PROBE_MIN_ENTITIES` (3) of the four entities `PROBE_TEXT`
contains, or it fails with `extraction_missed_entities`. Deliberately a floor
and not a score — ranking extraction quality is η's job on δ's real-episode
corpus; this only answers α's own question, whether the endpoint can do the
thing at all.

### Two smaller defects fixed alongside

* **`COMPLETION_TRUNCATED` was unreachable on vLLM.** llama.cpp returns
  `content: ''`, vLLM with a reasoning parser returns `content: null`. Only the
  empty string reached the truncation branch, so a vLLM arm that spent its whole
  budget reasoning reported `malformed_response` — blaming the harness's own cap
  on a broken body. `_extract_reasoning` now reads both stacks' spellings
  (`reasoning_content` and `reasoning`) and the null path is checked for
  truncation before it is called malformed.
* **`PROBE_MAX_TOKENS` is now derived, not chosen.** It mirrors the product's own
  `llm.max_tokens` (`fused-memory/config/config.yaml:15`, 4096) — the output
  budget the memory subsystem actually gives its LLM, asserted against that file
  by test so the two cannot drift. The old 512 was a runaway watchdog that had
  quietly become a capability gate. Runaway containment is unchanged:
  `COMPLETION_TIMEOUT_S` still bounds the request at 180 s.

### Known, not yet fixed

Every latency in the slate table above is a **single-sample cold** measurement —
`qwen3.5-9b` measured 2849 ms cold and ~350 ms warm, a 12× difference. These
numbers are not comparable across arms and are **not** ζ's p95-under-load
envelope metric. Tracked separately.

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

**Status: RESOLVED by measurement, step 22 (2026-08-06) — vLLM pooling for all
four. TEI is not needed and was never pulled.**

Every arm was started through the committed chain and probed with the committed
health check. All four loaded under the vLLM pooling runner and returned a valid
vector of their declared length:

| arm | load | probe | resident VRAM | probe verdict |
|---|---|---|---|---|
| `granite-embedding-english-r2` | 123 s | 192 ms | 796 MiB | PASS |
| `gte-modernbert-base` | 122 s | 286 ms | 788 MiB | PASS |
| `qwen3-embedding-0.6b` | 143 s | 700 ms | **16603 MiB** | PASS |
| `qwen3-embedding-4b` | 142 s | 580 ms | **16078 MiB** | PASS |

Uniformity was the preferred outcome and measurement allowed it: one stack across
all four removes a serving-stack confound from ι's query-latency comparison, so
the latency column above is comparable as it stands. `fallback_stack: tei` stays
in the manifest as the recorded fallback, unused.

#### The resident-VRAM split is architectural, and it matters

The two ~790 MiB arms are **encoder** (BERT-family) models. The two ~16 GiB arms
are **decoder** models that happen to emit embeddings — and vLLM gives a decoder a
paged KV cache sized to fill `--gpu-memory-utilization`, whether or not a pooling
model can ever use it. vLLM's own accounting for `qwen3-embedding-0.6b`:

```
weights 1.12 GiB · peak activation 0.30 GiB · non-torch 0.05 GiB
CUDA graph 0.07 GiB · KV cache 14.56 GiB   (136,272 tokens)
```

A 0.6B model with a declared `est_vram_gib: 2.0` took 16.2 GiB, of which 14.56
GiB is KV cache it cannot use. This is **not** an arm failure — the endpoint
answers correctly, which is what Q1 asked — but it makes `est_vram_gib`
decorative for these arms and is the direct cause of the budget conflict recorded
below. vLLM names the lever itself: `--kv-cache-memory=<bytes>`.

### Open Q3 — MoE stretch-arm model and quant

> Which of the two models, GGUF quant level, client-side validator placement.
> Decide in α/η.

**Status: RESOLVED by measurement, step 22 (2026-08-06) —
`unsloth/gemma-4-26B-A4B-it-qat-GGUF`, quant `UD-Q4_K_XL`, 13.27 GiB.**

The decision is arithmetic, not preference. Real GGUF file sizes, read from the
HF API rather than estimated:

| candidate | quant | real size | fits 16.4 GiB? |
|---|---|---|---|
| Qwen3.6-35B-A3B | UD-IQ4_XS | **16.51 GiB** | no — weights alone exceed the budget |
| Qwen3.6-35B-A3B | UD-IQ3_S | 12.74 GiB | yes, at a real quality cost |
| gemma-4-26B-A4B-it | UD-IQ4_XS | 12.66 GiB | yes |
| gemma-4-26B-A4B-it **qat** | UD-Q4_K_XL | **13.27 GiB** | yes ← **pinned** |

The PRD's ~17 GiB IQ4 estimate was for Qwen, and it is real: its smallest true
4-bit quant is 16.51 GiB of weights before a single KV byte. That **would** have
fitted D10's nominal 19.5 GiB and does not fit the measured 16.4 — exactly the
deviation this task was asked to resolve honestly rather than inherit.

Gemma was chosen over dropping Qwen to IQ3 because its QAT weights are
quantization-aware *trained* at 4 bits by Google, so Q4_K_XL is near-lossless
rather than a post-training approximation: **the arm that fits is also the one
that does not trade quality to fit.** Gemma's sliding-window attention helps
again on KV — 25 of its 30 layers use a 1024-token window, so 16k context costs
~0.82 GiB of KV instead of the ~3.8 GiB a full-attention model of this shape
would need. `est_vram_gib` is set to 14.5 on that basis.

The file is downloaded and staged (14,249,047,104 bytes). Validator placement
remains ε's.
