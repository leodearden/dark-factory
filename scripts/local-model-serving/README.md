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

| arm_id | axis | stack | port | structured output | reasoning | dims | declared / MEASURED VRAM |
|---|---|---|---|---|---|---|---|
| `qwen3.5-9b` | llm | vllm | 8410 | `json_schema` | **on** (`qwen3` parser) | — | 12.0 / **14507 MiB** |
| `phi-4-14b` | llm | vllm | 8412 | `json_schema` | off | — | 9.0 ⚠ / **15519 MiB** |
| `moe-stretch` | llm | llamacpp | 8413 | `json_object` | off | — | 14.5 / **14604 MiB** |
| `qwen3-embedding-0.6b` | embedding | vllm | 8414 | — | — | 1024 | 3.0 / **3584 MiB** |
| `granite-embedding-english-r2` | embedding | vllm | 8415 | — | — | 768 | 1.0 / **789 MiB** |
| `qwen3-embedding-4b` | embedding | vllm | 8416 | — | — | 2560 | 10.0 / **10114 MiB** |
| `gte-modernbert-base` | embedding | vllm | 8417 | — | — | 768 | 1.0 / **788 MiB** |

Port 8411 is **retired, not reassigned**: `mistral-small-3.2-24b` was dropped
from the slate on 2026-08-06 (see *The dropped arm* below). Reusing the port
would make every artifact keyed on it ambiguous.

⚠ `phi-4-14b` declares 9.0 GiB and measured **15519 MiB resident** — it is the
binding VRAM row on the slate, with only 1.37 GiB of headroom, despite carrying
the smallest declared footprint of the three LLM arms. `est_vram_gib` is the
*admission floor* `arm_fits` gates on, not the resident figure (a generate arm
sizes its KV cache to fill the share derived from free VRAM), but 9.0 is still
an understatement — its weight-only figure was not captured, so it is left
uncorrected rather than guessed at. `qwen3.5-9b` had the same defect and was
corrected 6.0 → 12.0 from vLLM's reported 11.21 GiB of weights.

**Ports 8410–8417 are reserved for this rig**, one per arm, bound to loopback
only. The block was chosen to clear what already listens on this host — 8002
(fused-memory MCP) and 8102 (escalation server). `lms_manifest` refuses a
manifest whose arm declares a port outside the block, and refuses two arms
sharing one: a shared port is the precondition for the 2026-04-08 bug where a
stale unit answered a probe and mis-attributed an entire run's metrics.

`moe-stretch` fits comfortably: Open Q3 resolved to gemma-4-26B-A4B QAT at
UD-Q4_K_XL (13.27 GiB of weights) precisely because it does, where Qwen3.6-35B-
A3B's smallest true 4-bit quant does not. See *Open Q3* below.

---

## Operator recipe

```bash
cd /home/leo/src/dark-factory

# 1. Install the unit template (idempotent; enables and starts nothing).
#    Fails unless systemd will ACTUALLY apply it — see below.
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

### The whole slate: `lms_slate_run.py`

Steps 1-5 above are the per-arm primitives. To run the WHOLE slate, don't type
them seven times — that is how the 2026-08-06 run was driven, and it is why no
compliant invocation for it existed anywhere in the repo afterwards.

```bash
uv run --project shared python scripts/local-model-serving/lms_slate_run.py
journalctl --user -u lms-slate-run -f      # follow it
```

That submits ONE transient `systemd --user` unit for the whole sweep (PRD
hazard 11: long runs go in transient units, never bare background shells — a
~30 minute slate through a background shell dies with your session, losing
every arm measured so far). Inside the unit it drives exactly the steps above,
**one arm at a time**: start / wait-ready / `--arm ... --output <part>` / stop.
Each arm is stopped even when its probe fails, because `lms_ctl start` refuses
rather than evicting — one arm left running would turn into six spurious
refusals. For the same reason the sweep opens with `lms_ctl stop-all`: an arm
left running by an *earlier* session would otherwise refuse all seven starts.
That touches `lms-arm@` units only, so whisper-writer is unaffected — but if
you have an arm up deliberately, the driver will stop it.

It is **resumable**. Each arm's report lands in `--parts-dir` as
`<arm_id>.json`, and a re-run skips any arm that already has a **passing**
part, so a sweep that died at arm six re-measures one arm and not seven. A part
whose row is a FAIL is *not* reused: the healthcheck writes its report before
returning a non-zero exit, so a failed arm does leave a valid file behind, and
reusing it would hand you a byte-identical artifact still carrying the stale
FAIL row after you had fixed the arm. Fixing an arm and re-running therefore
re-measures exactly the arms that failed. A part is also dropped when the
manifest's `served_model_name` for that arm no longer matches the one in the
part — but an `arms.yaml` edit that changes only `model_ref` or `quant` is
invisible in a report row, so use `--force` after one. `--force` re-measures
everything regardless. A re-measured arm's old part is **removed first**, so a
re-measure that fails leaves no part at all and the merge refuses by name
rather than quietly folding the previous run's row into this slate. Pass
`--parts-dir` explicitly to resume into a directory
from a previous run; the default lives under `$XDG_RUNTIME_DIR` and does not
survive a reboot.

`--dry-run` prints the compliant command and runs nothing — useful for reading
the exact invocation without a card. It is a submit-layer flag: combined with
`--in-unit` it is rejected rather than ignored, since `--in-unit` *is* the
sweep. `--ready-timeout` forwards to `lms_ctl wait-ready` (default 900s).

The artifact is written by `lms_healthcheck --merge`, never by the driver.
What that merge refuses is a slate **missing rows**, which is narrower than
"a failed sweep": an arm that never started or never came ready leaves no part,
`merge_reports`' manifest-coverage check then writes nothing and names the
uncovered arms, and the previously committed artifact is left intact. An arm
that came up and *failed its probe* is a different case — it leaves a part
carrying a FAIL row, so the slate is covered, the merge succeeds, and
`verification/health-report.json` is **overwritten with a red but complete
artifact**. That is the intended outcome (a red slate is a measurement), but do
not read a red sweep as one that cannot have touched the committed file.

**A manifest carrying TBD placeholder arms cannot be swept**, and the driver
refuses up front rather than discovering it 30 minutes in. Neither half of the
slate works for such an arm: `lms_ctl start` refuses a placeholder (exit 4),
and `lms_healthcheck --arm` cannot cover it either — the report needs the VRAM
baseline that only `lms_ctl start` writes, so it exits 8 having written
nothing — while the merge requires a row for every manifest arm. A hand-run
`lms_healthcheck --all` hits the same wall. Resolve the PRD open question that
owns the arm, or drop it from `arms.yaml`, before running the slate. All seven
arms are non-placeholder today.

### One arm at a time

`lms_ctl start` is **exclusive by default** — it refuses while another arm's
unit is running. The PRD's funnel never needs the slate up simultaneously, and
on one 24 GB card two arms is an OOM. `--no-exclusive` exists for the case
where you have checked the arithmetic yourself.

`--no-exclusive` also narrows the polluted-baseline refusal below, and only for
the arms systemd reports running: those may hold the card at baseline, are
recorded in the baseline's inventory alongside the arm ids that excused them,
and any *drift* in them during the run still marks the report POLLUTED. ollama
and anything else `KNOWN_FOREIGN_CONSUMERS` names is refused either way. On an
otherwise idle card the flag changes nothing about pollution — there is nobody
to excuse.

`Restart=no` in the unit template is deliberate: a dead arm stays dead rather
than thrashing the GPU in a restart loop while you read the journal.

### The install gate is the EFFECTIVE configuration, not file presence

`install-lms-units.sh` no longer claims success on "the file landed". After the
`daemon-reload` it runs [`scripts/check_lms_unit_parity.py`](../check_lms_unit_parity.py),
which compares the installed copy against the committed template *and asks
systemd what it would actually apply* — `systemctl --user show -p
WorkingDirectory -p DropInPaths lms-arm@probe.service`. A non-clean answer
fails the install with a non-zero exit and a `[lms_unit_parity]` report naming
the finding.

The installer tells those findings apart rather than describing all of them as a
drop-in: exit 1 points you at whichever of `[override]` / `[effective]` /
`[drift]` / `[vanished]` / `[unverifiable]` fired, exit 2 says the unit is not
where the installer just put it, and a checker that is *missing from the
checkout* or could not be run at all reports itself — with the command to run by
hand — instead of blaming an override it never looked for. Each of those still
FAILS the install: one that could not verify the effective configuration has
established nothing, which is the state this gate exists to stop reading as
success.

File presence was never the claim an operator needed, because it is blind to
the one thing that actually redirects a unit. `systemctl --user edit` never
touches the unit file; it writes `lms-arm@.service.d/override.conf` beside it,
and systemd merges that over the unit at load time. So a drop-in can pin
`WorkingDirectory` somewhere else while the installed file stays *byte-identical*
to the committed template — and the old installer reported success and left the
override in place, run after run.

**Every applying drop-in is named by absolute path, and none of them is
removed.** That is deliberate (task 3750): the drop-in observed on this host was
load-bearing while its worktree was unmerged, so deleting it blindly would have
pointed every arm at a directory with no launcher. Removal has a correct owner
already — `scripts/remove-lms-arm-worktree-dropin.sh` — which gates it behind
preconditions an installer does not check.

The same checker runs **warn-only** from `scripts/setup-host.sh`, so an override
that appears *between* installs is still surfaced on the next host bring-up
rather than waiting for someone to reinstall. It warns there instead of failing
because bring-up must not be bricked by state we deliberately refuse to auto-fix.
Exit 2 ("not installed on this host") is reported as benign info, not drift.

### If the install reports an override

1. See what systemd actually merged:

   ```bash
   systemctl --user cat lms-arm@qwen3.5-9b.service
   systemctl --user show -p WorkingDirectory -p DropInPaths \
       lms-arm@qwen3.5-9b.service
   ```

   `cat` shows the unit followed by every drop-in that applies to it;
   `show` reports the *resolved* values, which is the claim being checked.

2. If it is the known task-3713 worktree drop-in, remove it with its safety
   preconditions:

   ```bash
   scripts/remove-lms-arm-worktree-dropin.sh
   ```

   It refuses unless the launcher is present at
   `scripts/local-model-serving/lms_serve.py` in the main checkout, the template
   is still installed, and the launcher compiles — the three things that make
   removal safe rather than a way to break every arm at once.

3. Re-run `install-lms-units.sh`. It is idempotent, and it will now say so
   affirmatively: on the clean path it states the committed template *is* the
   effective configuration and names the resolved `WorkingDirectory`.

The failure mode this exists to prevent: a drop-in pinning `WorkingDirectory` at
a worktree that has since landed but is still on disk. The arms keep starting,
keep answering, and keep serving a **frozen tree** — while every operator, every
report and every reinstall says they are running merged main. Nothing about the
endpoint looks wrong; only the code behind it is stale.

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
**no report was produced** · `5` `--active` found nothing running · `6` `--merge`
refused to combine the inputs · `7` **the VRAM measurement is polluted** — a
non-arm process moved on the card, so nothing was learned about the arm ·
`8` **no usable baseline** — none was recorded for this arm, or the one on disk
predates the consumer inventory; re-take it with `lms_ctl start <arm>`.

Every one of these is separate on purpose, because each has a different fix.
A budget failure and a model failure are not the same problem. A broken
`nvidia-smi` must never degrade into a report whose VRAM block reads
`used 0 MiB, headroom 19.5 GiB` — a passing budget with maximal headroom is the
most trustworthy-looking wrong answer this rig can emit. `--active` with nothing
running must not exit 0, or a wrapper script would certify a slate nobody
probed. And **7 is not 3**: 3 says the arm is genuinely too big and something
should be stopped, whereas 7 says the arithmetic is void — stopping an arm in
response would mean acting on a number nobody measured. A polluted run is never
`0` either, however healthy the block's own `verdict` looks. **8 is not 4** for
the same reason inverted: the GPU answered fine and it is the *baseline store*
that has nothing usable in it, so `4`'s "go debug nvidia-smi" would be the wrong
errand — and the case is routine, not exotic, because a pre-guard baseline can
still be sitting in `$XDG_RUNTIME_DIR` from the current boot.

`lms_ctl start` has its own pair for the same reason: **4** the arm does not fit
this card (use a smaller arm), **5** another process is holding the card (free
it and start the same arm again). Collapsing those into one code would send an
operator to shrink an arm that fits perfectly well.

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

### Non-arm GPU consumers (task 3755)

`arm_footprint_mib` is `used − baseline`, which is the **arm's** footprint only
if nothing else moved on the card between those two readings. Up to schema v4
nothing recorded whether that held, so this happened and left no trace.

Measured 2026-08-06, during a slate run:

| pid | process | MiB | what it is |
|---|---|---|---|
| 7575 | `python` | 4050 | whisper-writer — PRD D10 requires it resident |
| 905936 | `/usr/local/lib/ollama/llama-server` | 10314 | ollama, `qwen3:14b` held on `keep_alive` |

`ollama.service` is a **persistent unit on this host**. It wakes on any request
to `:11434`, holds its model for the `keep_alive` window, and nothing about
starting an arm consults it. Ten gigabytes appearing between the baseline and
the probe are charged straight to the arm by the subtraction.

The guard, from v5 on:

- **`lms_ctl start` refuses to record a polluted baseline** (exit 5, no file
  written). At that moment no arm of ours is running, so the check is a positive
  **allowlist** — whisper-writer under a 6144 MiB ceiling, and nothing else over
  ~1 GiB. Anything unanticipated is caught, not just ollama.
- **`lms_healthcheck` classifies the probe reading** and reports
  `vram.pollution` plus both inventories in the artifact (exit 7). Here the rule
  has to be different: arms are docker containers, `nvidia-smi` reports **host**
  pids, and `--query-compute-apps=pid,process_name,used_memory` cannot tell a
  containerised vLLM `python` from any other `python`. The same allowlist would
  flag every healthy run. So probe-time pollution is a **known-foreign match**
  (the `/usr/local/lib/ollama/` path, which no arm wears) plus **drift in any
  consumer already present at baseline** — such a consumer is non-arm by
  construction, because the arm was not running then.
- **Drift is pollution in BOTH directions.** Growth over-charges the arm; a
  shrink or a vanish *under*-charges it. The flattering direction is the one a
  fabricated artifact wants, so it fails too.
- **The baseline capture is bracketed**: `lms_ctl start` reads the inventory,
  then the memory reading, then the inventory again, and refuses if the two
  inventories disagree by more than the floor. `nvidia-smi` cannot answer both
  questions in one call, so pairing them is a claim about a *window*. Its
  silent failure is the flattering one: a holder resident for the reading and
  gone by the inventory inflates the baseline, `assert_clean_baseline` sees a
  spotless card, and by probe time the mover is in neither reading — so nothing
  downstream can tell. A co-resident arm appears in both inventories and is
  therefore never a mover.

This replaces an operator discipline — "check `nvidia-smi` before you measure" —
that was never written down and that ε/θ/ι had no way to inherit.

Four limits, stated rather than implied:

- **The bracket narrows the window; it does not abolish it.** A process that
  both arrives and leaves between the first inventory and the reading is
  invisible to both, and no arrangement of separate `nvidia-smi` calls can see
  it. What is left is a sub-second race against a consumer that vanishes as
  fast as it appeared, rather than the minutes-long `keep_alive` holding that
  motivated the guard. Probe time is deliberately *not* bracketed: the arm
  itself is legitimately allocating there, so refusing on any movement would
  fail healthy runs.

- **The inventory is not an accounting.** `--query-compute-apps` lists only CUDA
  compute applications; the ~3.3 GiB of KDE/X11 graphics contexts appear in it
  nowhere. That is exactly why the operating budget is ~16.4 GiB and not 19.5.
  The entries name *who else held the card*; they do not sum to `memory.used`.
  A clean baseline on this host really does show 3312 MiB used and **zero**
  compute apps.
- **Nothing here stops or polices ollama.** The tools refuse to measure, and say
  what to free. ollama releases its model on `keep_alive` expiry by itself.
- **The residual case.** A probe-time newcomer that matches no foreign pattern
  and is not the arm cannot be told apart from `--query-compute-apps` alone. It
  is recorded verbatim in `vram.probe_consumers` as the audit trail rather than
  silently attributed to anybody. Resolving it exactly would need `docker top`
  or cgroup introspection to map host pids back to `lms-arm-<arm_id>`, which
  buys a docker dependency and a resolver whose own failure would mark every
  real run polluted.

The allowlist and the foreign list are **module constants with no environment
override** (`lms_vram.EXPECTED_CONSUMERS`, `lms_vram.KNOWN_FOREIGN_CONSUMERS`).
An env var could silence the guard invisibly; changing what this host is
expected to run should be a code change with a reviewable diff.

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

## Live slate run (measured, step 23 — 2026-08-06, COMPLETE)

Every arm run through the committed chain, one at a time, 30 m 22 s of live host
time on a card confirmed clean at 7205 MiB idle beforehand (whisper-writer the
only compute app at 4050 MiB, no ollama model resident). **7 of 7 PASS.**

| arm | ready | resident (footprint) | budget | probe | entities | verdict |
|---|---|---|---|---|---|---|
| `granite-embedding-english-r2` | 101 s | 789 MiB | 16922 MiB | 129 ms | — | **PASS** |
| `gte-modernbert-base` | 111 s | 788 MiB | 16922 MiB | 115 ms | — | **PASS** |
| `qwen3-embedding-0.6b` | 116 s | 3584 MiB | 16921 MiB | 507 ms | — | **PASS** |
| `qwen3-embedding-4b` | 157 s | 10114 MiB | 16911 MiB | 630 ms | — | **PASS** |
| `qwen3.5-9b` | 438 s | 14507 MiB | 16910 MiB | 43501 ms | 4/4 (4 top-level) | **PASS** |
| `phi-4-14b` | 171 s | 15519 MiB | 16918 MiB | 2854 ms | 3/4 (2 top-level) | **PASS** |
| `moe-stretch` | 20 s | 14604 MiB | 16915 MiB | 2465 ms | 4/4 (4 top-level) | **PASS** |

Every arm released **every** byte on stop (card back to 7198–7217 MiB within the
first 3 s poll), whisper-writer held 4050 MiB undisturbed throughout, and no two
arms were ever co-resident.

`phi-4-14b` is the row the extraction floor was ruled on: `FalkorDB` appears
**only as an attribute value** and `Leo` **only in the free-text summary**, which
the floor does not scan. So it captures 3 of 4 and promotes 2 — both numbers are
in the row, the floor passes it, and `top_level_entities_named` records the
representation difference for η without α judging it.

### Latency here is a COLD single sample — do not rank arms on it

`qwen3.5-9b` measured 2849 ms cold and ~350 ms warm at `reasoning: off` — a 12×
gap. These numbers are one measurement each, taken on the first request after
load, and they are **not** ζ's p95-under-load envelope metric. Task **3781**
fixes the instrument (warm the engine, not the prefix cache; report cold and warm
separately). The one exception is `qwen3.5-9b` at `reasoning: on`: 43.5 s cold vs
41.0 s warm, because that cost is genuine generation rather than load.

### The dropped arm

`mistral-small-3.2-24b` was commissioned by the PRD and **dropped on 2026-08-06**
(Leo's ruling, esc-3713-10) after live measurement proved it unservable here.

Its declared quant *was* wrong, and that defect is fixed and verified: `awq` →
`compressed-tensors`, measured from the weights' own `config.json`; vLLM 0.26
then accepted the model and resolved `max_model_len 16384`. The arm still never
reached weight loading, for an unrelated reason — it is a **vision**-language
model (`mistral3.py`, Pixtral tower), vLLM unconditionally sizes a multimodal
encoder budget at startup by pushing a dummy `[IMG]` prompt through
`PixtralProcessor`, and the quantized repo's repacked tokenizer encodes it to
**zero** image tokens against a text count of one:

```
ValueError: Mismatch in `image` token count between text and `input_ids`.
Got ids=[0] and text=[1].
```

transformers separately warns that repo needs `fix_mistral_regex=True`, pointing
at the same object. Root cause is the quantizer's tokenizer, not the quant and
not the weights — the card never moved (7212 → 7221 MiB).

Re-admitting the arm needs a different quantized repo or an upstream tokenizer
fix, **not** a flag suppressing the multimodal path: the eval would then be
measuring a model configured differently from the one the PRD costed. The PRD
text correction is tracked as task **3804**.

**Consequence, flagged rather than absorbed:** the LLM slate is three arms, so
η's survivor funnel narrows 3 → at most 3 and has near-nil selectivity on that
axis. Whether that warrants re-opening the slate is Leo's call, recorded on
tasks 3720 and 3804.

### Embedding vectors are NOT uniformly normalized — ι must not assume they are

Measured L2 norms on the probe query:

| arm | L2 norm |
|---|---|
| `qwen3-embedding-0.6b` | 1.0000 |
| `qwen3-embedding-4b` | 1.0000 |
| `granite-embedding-english-r2` | **30.36** |
| `gte-modernbert-base` | **37.09** |

All four pass the health check, whose floor is only `1e-6` (a zero vector carries
no direction and would make every similarity undefined). But **half the embedding
slate returns non-unit vectors**, so anything that treats a raw dot product as
cosine similarity — a standard optimisation when vectors are assumed
pre-normalized — will score those two arms on a completely different scale from
the Qwen pair. Cosine proper is unaffected; Qdrant normalises on insert under
`Cosine` distance but **not** under `Dot`. ι must normalise explicitly or pin the
distance metric, and say which in its report.

## Verification artifact

`verification/health-report.json` is written by a live run
(`lms_healthcheck --all --output ...`) and committed. It carries
`schema_version`, an aware-UTC `measured_at`, the GPU identity (which card,
which driver — every verdict is relative to specific hardware), one row per arm,
and the VRAM block. `scripts/tests/test_lms_verification_artifact.py` requires a
`PASS` row for every arm in `arms.yaml` plus a passing VRAM block, so the test
can only be greened by the run having actually happened.

**The committed file is `schema_version: 4`; the producer is at 5.** It is
evidence of a real ~39-minute 7-arm run and every other property of the gate
still holds against it — what it predates is the v5 consumer inventory, so it
cannot say who else held the card while those arms were measured. Re-deriving
it needs docker, systemd and the shared 3090, and may itself be *refused* by the
v5 guard if ollama is resident, which is the whole point of the guard. So the
gate's version check is `ACCEPTED_ARTIFACT_SCHEMA_VERSIONS = {4, 5}` — a narrow,
self-expiring grandfather clause, and a *widening* of the equality it replaced
rather than a strengthening of it. What still runs live against today's v4 file
is: the v4 block must carry **none** of the five consumer keys (so it cannot be
hand-edited to fake an inventory), the set may grandfather exactly one named
older version, and the set expires at the next schema bump. The *additional*
strictness — a v5 artifact must carry a measured inventory and a `CLEAN`
pollution state — begins the moment the artifact is re-derived at v5, and is not
in force today. That re-run is filed as task **4229**; do not close it by editing
the artifact.

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
| `qwen3-embedding-0.6b` | 143 s | 700 ms | ~~16603 MiB~~ **3584 MiB** | PASS |
| `qwen3-embedding-4b` | 142 s | 580 ms | ~~16078 MiB~~ **10114 MiB** | PASS |

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
