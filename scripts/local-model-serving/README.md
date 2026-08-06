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

## OPEN: the budget verdict's subject is miscalibrated (blocks step 23)

Measured in step 22, unresolved here on purpose — it is a design question, not a
bug to be quietly patched under an anti-fabrication gate.

Every arm measured so far answers correctly, and the **LLM** arms nonetheless
fail the VRAM verdict, because two budget notions in this package disagree
about their subject (the *embedding* arms listed here failed for a second,
independent reason that has since been fixed — see below):

| check | subject | ceiling | `qwen3-embedding-0.6b` |
|---|---|---|---|
| `arm_fits` (pre-flight) | the **arm's** footprint | measured free, 16.37 GiB | 2.0 + 0.5 ≤ 16.37 → **admits** |
| `evaluate_budget` (verdict) | **total** card usage | PRD nominal, 19.5 GiB | 23.35 > 19.5 → **FAILs** |

An arm therefore passes the pre-flight and then fails the verdict, having done
nothing wrong. The PRD's own arithmetic says which subject is intended:

> l.192 — GPU headroom measured (24GB − ~4GB whisper-writer)
> l.165 — whisper-writer stays resident: all capacity math against ~19–20GB, not 24GB

19.5 GiB is what D10 derives as available **to the arm** (card minus
whisper-writer). Applying it to *total* usage charges the arm for the 7.15 GiB
desktop+whisper baseline a second time — and the desktop's ~3.2 GiB was not in
D10's model at all, which is the same omission this README's *VRAM budget*
section already records.

Under the total-usage reading the real per-arm allowance collapses to
19.5 − 7.15 = **12.35 GiB**, which would knock `mistral-small-3.2-24b` (14 GiB of
weights) and `moe-stretch` (13.27 GiB) off the slate on a technicality, while
both fit the 16.4 GiB the card actually has free.

Two changes were needed. **The second has since landed**, and landing it
changed the shape of the first — so read the two parts below in order.

### RESOLVED: the pooling arms' KV balloon (landed 2026-08-06)

vLLM sizes its paged KV cache to fill whatever `--gpu-memory-utilization`
allows, and it does that for a **decoder** model served under `--runner
pooling` even though a pooling model can never read a KV cache. Derived from
free VRAM, `qwen3-embedding-0.6b` was handed 0.682 (16.37 GiB) and filled it:
1.12 GiB of weights, 0.42 GiB of overhead, and **14.56 GiB of unusable KV**.

`lms_serve._memory_share_for` now bounds a **pooling** arm to
`est_vram_gib + SAFETY_MARGIN_GIB`, while a **generate** arm keeps the
free-VRAM-derived share — the two axes want opposite things, because for an
LLM arm the KV cache *is* the context window. Bounding via the share rather
than `--kv-cache-memory-bytes` keeps `est_vram_gib` load-bearing: the
pre-flight already admits an arm on the strength of that number, so an arm that
then ignored it made the pre-flight's arithmetic false as soon as two arms were
considered together.

Measured before → after, same chain, same probe:

| arm | resident before | verdict before | resident after | verdict after |
|---|---|---|---|---|
| `qwen3-embedding-0.6b` | 16603 MiB | FAIL, 23.35 GiB | **3602 MiB** | **PASS**, 10.66 GiB |
| `qwen3-embedding-4b` | 16078 MiB | FAIL, 23.35 GiB | **10102 MiB** | **PASS**, 17.00 GiB |

Both still answer with a valid vector of their declared length (685 ms / 783 ms).
**Both now pass under the existing total-usage semantics**, so this half needed
no gate change at all — it was a bug, and it is fixed.

It also made `est_vram_gib` honest for these two arms, which measurement showed
were *under*-declared: the bound is real, so a too-small declaration now fails
loudly at startup (`ValueError: To serve at least one request with the model's
max seq len (8192), 0.88 GiB KV cache is needed, ... available 0.86 GiB`,
naming the 8032-token ceiling it could actually serve) instead of silently
eating the card. `0.6b` moved 2.0 → 3.0 and `4b` 9.0 → 10.0 on that evidence.

### STILL OPEN: the verdict's subject, now purely an LLM-arm question

What remains is **not** a KV bug and cannot be fixed the same way, because an
LLM arm's KV cache is load-bearing. Measured on the cheapest LLM arm:

| arm | resident | total card | probe | vram verdict |
|---|---|---|---|---|
| `qwen3.5-9b` | 14961 MiB | **21.75 GiB** | PASS, 3045 ms | **FAIL** (> 19.5) |

So the conflict is real and survives the KV fix: this arm serves
schema-constrained completions correctly and still fails, because the
free-derived share (16.37 GiB) plus the 7.31 GiB desktop+whisper baseline
exceeds the nominal 19.5 GiB ceiling. It is not specific to the big arms —
a 9B AWQ declaring `est_vram_gib: 6.0` already blows it.

The fix is the subject correction: **`evaluate_budget` should judge the arm's
footprint** (`used − baseline`) against the measured operating budget, not
total usage against the nominal ceiling. `lms_vram.MEASURED_BASELINE_GIB`
(7.19) already exists for this. `test_lms_verification_artifact.py`'s
`used_mib <= nominal_ceiling` assertion moves with it — deliberately called
out, because changing an assertion inside the anti-fabrication gate is exactly
the move that needs a reviewer, not an agent acting alone. (Its stated purpose
is *internal consistency* — "the verdict and the numbers it was computed from
disagree" — so the corrected form is the same check against the corrected
subject rather than a weaker one. That reading still wants a second pair of
eyes, which is why it is written here and not applied.)

Until that lands, `lms_healthcheck` exits 3 (budget) rather than 0 for the LLM
arms, and `verification/health-report.json` cannot honestly be greened. Hand-
editing a PASS row to get there is precisely what the gate exists to prevent.

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
