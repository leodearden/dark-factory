# confusion census 2026-09-20

Project: dark_factory

## Saturation

- batches: 6
- stop reason: saturated
  - batch 0: dup_rate=0.95 (total=20, succeeded=19, failed=1, saturated=True)
  - batch 1: dup_rate=0.89 (total=20, succeeded=18, failed=2, saturated=False)
  - batch 2: dup_rate=0.95 (total=20, succeeded=20, failed=0, saturated=True)
  - batch 3: dup_rate=0.84 (total=20, succeeded=19, failed=1, saturated=False)
  - batch 4: dup_rate=0.95 (total=20, succeeded=20, failed=0, saturated=True)
  - batch 5: dup_rate=0.90 (total=20, succeeded=20, failed=0, saturated=True)

## Origin x Manifestation Matrix

| origin \ manifested | architect | implement | unknown |
| --- | --- | --- | --- |
| implement | 0 | 4 | 0 |
| unknown | 1 | 2 | 2 |

## Synthesis

Evidence gathering is complete. Writing the synthesis now.

Synthesis complete. Everything below was checked against the archived transcripts under `data/orchestrator/agent-transcripts/`, the codebook, the source on main, the task store and the prior census reports; where the verifier's framing disagrees with that evidence, the correction is stated explicitly.

**Date:** 2026-09-20
**Method:** periodic census per `plans/confusion-reduction-prd.md` §5 (η): stratified-random saturation mining (Sonnet) over session digests, per-finding verification against current main (Sonnet), then this synthesis (Fable). Nine findings reached synthesis. This document adds a read of five archived orchestrator transcripts, the digest detector source, the routing-guard source, the 09-15 feasibility study report, the sandbox hook audit, and the codebook. Every mechanism claim below names the evidence it rests on.
**Companion artifact:** `docs/legibility/confusion-codebook.yaml`. Dispositions in §5 are inputs to the merger.
**Run notes:** twelfth completed periodic census, at the PRD's 5-day hard floor after 2026-09-15. A report dated today already existed on disk when this synthesis ran, committed at 04:19 BST as `cee5b7880a`, reading "No novel, verified confusion clusters this census" with two verify calls and both mining batches saturated (dup_rate 1.00 and 0.95); `census-state.json` already reads 2026-09-20. This synthesis therefore belongs to a second run on the same day, and its nine findings are a different verified set from that earlier run's empty one. Previous corpora: 09-15 (1 verified finding), 09-10 (2), 09-05 (1), 08-31 (1), 08-26 (1), 08-21 (2), 08-16 (3), 08-10 (1), 08-05 (0), 07-31 (15 / 4 clusters), 07-24 (52). Saturation statistics and filed-task ids are appended by the runner outside this synthesis. The synthesis sandbox refused every write and most `find`/`sed`/`awk` invocations, so evidence is limited to greps and reads; two live probes of the Bash hook were also refused, which is noted where it matters.

### Corpus

- **9 verified findings, 7 sessions, 9 sightings, 6 clusters.** Five sessions are orchestrated task sessions with archived transcripts; two (the 09-15 study lead and a trickle-coder run) live outside the sandbox and are read here only through the codebook and the study's own report.
- **Four of the nine findings are already in the codebook under the same session id.** Session `fbd7b22b` (task 5580) is recorded on `cand-20260919-4` (the stray `skim` token) and `cand-20260919-5` (the em-dash), and session `38a8f2d8` (the 09-15 study lead) on `cand-20260915-5` (watcher-loop hours) and `cand-20260915-6` (the second forecast line). The verifier presented all four as novel. The two python -c flattening sightings and the self-correction sighting match existing entries by shape (`entry-cand-20260722-6`, `entry-cand-20260722-16`) though not by session. Only the fused-memory connectivity finding has no codebook presence of any kind: no entry or candidate mentions "Unable to connect".
- **Role stamps, from the briefings in the transcripts.** Sessions `62f084a1` (task 5560), `e05f42c4` (4195), `fbd7b22b` (5580) and `bc5a9c30` (3895) each open with the TDD implementer briefing; `7a9d5656` (5650) opens with the TDD architect briefing. The verifier's `implement` and `architect` stamps hold. Its two `unknown × unknown` stamps resolve differently: the trickle-coder session is `ops × ops` per the existing entry it matches; the study session is an interactive `/team` run with no factory phase, and stays `unknown`.
- **Sessions and windows (UTC):** 3895 on 09-16 12:49 to 13:53; 4195 on 09-17 17:09 to 18:26; 5560 on 09-18 07:33 to 08:08; 5650 on 09-19 17:xx to 18:29; 5580 on 09-19 23:33 to 09-20 01:08. Tasks 3895 and 4195 are done (merged `8ba7cf72` and `890be056`); 5560 and 5580 are in-progress; 5650 is pending.

### Executive summary (observations)

1. **The three Bash "flattening" sightings share one discriminator the verifier did not name.** In each of the two python -c sessions, the agent issued its multi-line `python3 -c "…"` probe six and ten times respectively; exactly one call per session was flattened, and it was the only one that also contained a `git status` or `git log` subcommand. The heredoc commit in session `fbd7b22b` that arrived with `skim` inserted before `git status` and `git log` was likewise collapsed onto one line. A PreToolUse hook on Bash, `~/.claude/hooks/skim-rewrite.sh`, is documented in `plans/os-sandbox-claude-home-writer-audit.md` as a stateless command-rewrite filter, and it rewrote this synthesis's own `head`, `cat` and `git status` calls into `skim … --mode=pseudo` invocations. Whether the rewriter is what joins the lines was not directly tested: the sandbox refused the live probe.
2. **Two of the four already-filed candidates carry cause text the transcript contradicts.** The em-dash candidate blames model-generated Unicode; the em-dash was a verbatim copy of `verify.py` line 8473 on main and sat inside the intended string. The script failed because its `"""`-delimited old-text literal embedded the docstring's own closing `"""` on line 13, ending the literal early. The `skim` candidate calls the layer "unidentified"; it is the operator-installed hook above.
3. **The fused-memory outage in the 5650 architect session lasted at most 27 seconds of a 300-record session and was reported, not lost.** A search and an add_memory succeeded at 18:27:50; two add_memory calls at 18:28:07 and 18:28:17 each failed in under a second with "Unable to connect"; two escalation-server calls at 18:28:47 and 18:28:53 succeeded; the agent wrote "the plan is already finalized, so nothing is lost" and filed the fact through `escalate_info`. The watchdog had redeployed fused-memory at 17:58:25, thirty minutes earlier. What was wrong at 18:28 is not verified.
4. **Every self-inflicted error in this corpus was self-corrected within 30 seconds.** Relative-path guard crash: 6 s. Bare-filename open: 2.5 s. Em-dash script: 27 s. Flattened probe in 5560: the session's next action. The cost is turns, not outcomes.

### Origin × manifestation matrix

The runner's matrix renders the verifier's stamps. The table below carries the refined stamps this synthesis establishes (§Corpus): the trickle-coder sighting moves from `unknown × unknown` to `ops × ops`; everything else is unchanged.

| origin \ manifested | prd | architect | implement | verify | review | merge | recon | ops | unknown | **total** |
|---|---|---|---|---|---|---|---|---|---|---|
| implement | · | · | 3 | · | · | · | · | · | · | **3** |
| ops | · | · | · | · | · | · | · | 1 | · | **1** |
| unknown | · | 1 | 3 | · | · | · | · | · | 1 | **5** |
| **total** | **0** | **1** | **6** | **0** | **0** | **0** | **0** | **1** | **1** | **9** |

Readings, observational. Six of nine sightings manifest in implement sessions, all within one tool call of their cause. Five origins stay `unknown` because the cause sits in the tool layer (the Bash hook, the MCP transport) or outside the factory (an interactive study), not in a factory phase. The `merge` and `verify` columns are zero for an eleventh consecutive cycle, and the PRD's motivating architect/implement→merge hypothesis remains untested by the twelve post-07-24 corpora, which now total 22 findings.

### 1. Verified clusters

#### 1.1 Compound Bash commands containing `git status` / `git log` reach the shell with their newlines collapsed and a `skim` token inserted (3 sightings: sessions `62f084a1`, `e05f42c4`, `fbd7b22b`)

**The trace, session `62f084a1` (task 5560, implementer, 2026-09-18).** Six tool_use records carry a `python3 -c "` argument with real newlines (JSON `\n`) at 07:33:21, 07:34:14, 07:34:44, 07:35:19, 08:00:41 and 08:07:22. One `IndentationError: unexpected indent` appears in the transcript, at 08:07:54, for the 08:07:22 call. That call, "Final state check", was `git log --oneline -3 && git status --short && echo "=== plan status ===" && python3 -c "<four-line script>"`. The interpreter's echo shows the script as one physical line with a leading space. The five earlier probes contained no git subcommand. The session's other two errored results are an "Exit code 1" at 07:33:43 and a tool timeout at 08:04:52, neither an IndentationError.

**The trace, session `e05f42c4` (task 4195, implementer, 2026-09-17).** Ten newline-bearing `python3 -c "` inputs between 17:09:28 and 18:25:10; one IndentationError, at 18:25:43, for the 18:25:10 call "Confirm all steps done and tree clean": `python3 -c "<six-line script>" && echo "--- working tree ---" && git status --short && echo "(clean above …)"`. It is the session's only errored tool result. The nine earlier probes contained no git subcommand.

**The trace, session `fbd7b22b` (task 5580, implementer, 2026-09-19 23:54:40).** The model's input begins `git add -- . && git status --short && git commit -q -m "$(cat <<'EOF'`. Bash's error line reads `git add -- . && skim git status --short && git commit -q -m "$(cat <<'EOF' … EOF )" && skim git log --oneline -1`: the heredoc body on one line, `skim` before both `git status` and `git log`, exit 127 on a syntax error. The agent's input contained no `skim`.

**What is known about the rewriter.** `plans/os-sandbox-claude-home-writer-audit.md` line 68 inventories `~/.claude/settings.json`'s hooks: a `PreToolUse` hook matching `Bash` runs `~/.claude/hooks/skim-rewrite.sh`, which `exec`s `skim rewrite --hook`, "a stateless stdin→stdout command-rewrite filter". `skim` resolves to `/home/leo/.cargo/bin/skim`. In this synthesis session the same layer rewrote `head -30 <file>; cat <file>` into `skim <file> --mode=pseudo --max-lines 30, skim <file> --mode=pseudo` and refused a bare `git status --short` for approval, so the hook is live on the same host the orchestrated sessions ran on. The synthesis could not read the hook script or settings file (outside the allowed directories) and its live test of a multi-line `python3 -c` command was refused, so the line-joining itself is an inference from the 3-of-3 versus 0-of-14 split above, not a reproduction.

**Codebook state.** `entry-cand-20260722-6` ("Bash tool's static-analysis pre-check rejects multi-line heredoc/multi-command scripts…") holds roughly twenty flattening sightings since 07-30; five of its notes (07-30 onward) already remark on an inserted `skim` alongside the flattening, none names the hook, and the entry title still describes a pre-check rejection. `cand-20260919-4` records the 5580 commit sighting with cause "unidentified command-rewriting/interception layer".

#### 1.2 A heredoc edit script embedded the target docstring's own closing `"""` inside a `"""` literal; the reported error was the first non-ASCII character after the literal ended (1 sighting, session `fbd7b22b`)

At 00:59:48 the implementer ran `python3 - <<'PYEOF'` with `old = """    Reports ``passed`` … Never raises.\n    """\n"""`. The literal ends at the embedded `    """` on line 13; the `"""` on line 14 opens a new literal that line 15's `new = """` closes, leaving the rest of line 15 and line 16 as bare code. The traceback at 01:00:05 points at line 16, `flake). Otherwise ``failed`` if any attempt produced a genuine red — a real`, with `SyntaxError: invalid character '—' (U+2014)`. That line, em-dash included, is `orchestrator/src/orchestrator/verify.py` line 8473 on main, copied verbatim. At 01:00:15 the agent re-issued the edit with `'''` delimiters and without the closing quotes; it printed `ok` at 01:00:49 and the harness recorded the diff.

**Correction to the codebook.** `cand-20260919-5` titles this "Unicode em-dash (U+2014) in heredoc-passed Python code strings" and blames model-generated em-dashes that "heredoc delivery does not normalize". The em-dash was source text, was meant to be inside a string, and would have parsed there. The verifier's framing (the literal terminated early) is the one the script supports; its specific claim that "the em-dash [is] outside its triple-quoted string" is right, and the reason is the embedded closing quotes.

#### 1.3 fused-memory `add_memory` failed twice with "Unable to connect" inside a 27-second window of an otherwise connected architect session (1 sighting, session `7a9d5656`)

The 5650 architect's fused-memory calls, all to `http://127.0.0.1:8002/mcp` per `.mcp.json`: search at 18:27:03 (ok), add_memory and search at 18:27:50 (ok, result at 18:27:54), add_memory at 18:28:07 (error in 0.3 s), add_memory at 18:28:17 (error in 0.7 s). The two errors are the session's only errored tool results. Escalation-server calls (`127.0.0.1:8102`) at 18:28:47 and 18:28:53 succeeded. The agent's text at 18:28:28: "Memory writes are failing — the fused-memory server is unreachable. The plan is already finalized, so nothing is lost, but that's worth reporting." It then filed two `escalate_info` records. `data/fused-memory/last_redeploy_fused_memory.json` records a watchdog redeploy at 17:58:25 UTC the same day; the next liveness restart is stamped 02:26:02 on 09-20. Nothing read here shows what the server was doing at 18:28. The MCP server answered this synthesis's five `get_task` calls from the main checkout, so the verifier's "from task worktrees" framing is not evidenced: the worktree uses the same URL, and the same session had just written through it.

#### 1.4 Ad-hoc probes in the 3895 implementer session assumed a path shape twice, and corrected each within seconds (2 sightings, session `bc5a9c30`)

- **12:57:35 → 12:58:44.** A `python3 - <<'PY'` AST walk opened `'test_bake_off_storage_shape.py'` by bare name; `FileNotFoundError`. The re-issued call at 12:58:46 prefixes `cd …/.worktrees/3895/fused-memory/tests &&` and its result at 12:58:54 carries no error. The transcript records, for every Bash call, both the model's command and a `wireToolInputs` copy prefixed with a `cd <dir> &&` of the harness's tracked cwd; the two calls issued at 12:57:33 and 12:57:35 carry different prefixes (`.worktrees/3895` and `…/fused-memory/tests`). The effective cwd is therefore not readable from the model's command text, which is the condition under which a bare filename is a guess.
- **13:07:47 → 13:08:12.** The agent probed the routing guard it was itself authoring (task 3895's `files` list is exactly `fused-memory/tests/test_script_loader_routing_guard.py`) with `pathlib.Path('tests/_scratch_guard_probe.py')`; `_module_key` raised `ValueError: 'tests/_scratch_guard_probe.py' is not in the subpath of '…/fused-memory/tests'`. At 13:08:18 the probe was rebuilt as `g.TESTS_ROOT / '_scratch_guard_probe.py'` and ran. On main today `_module_key` is unchanged: `return path.relative_to(TESTS_ROOT).as_posix()`, with a one-line docstring and no path-shape check. The task merged as `8ba7cf72`.

Both are re-sightings of the codebook's worktree-cwd / relative-path family (it already holds sightings from worktrees 2998, 3515 and 4561 of the same shape).

#### 1.5 The 09-15 feasibility study lead double-counted watcher-loop hours for one project and compared against one of two forecast lines (1 sighting, session `38a8f2d8`)

`plans/price-estimation-skill-feasibility-2026-09-15.md` records both errors and their correction in its own text (§3, lines 305 to 311, and the seat table at line 743): the lead's first draft claimed a 5–9× human overrun; the critic showed 62 of 69 session-hours in the project directory were three unattended `/escalation-watcher` loops and that the forecast had a second human line, after which attended time was at or under forecast. The codebook already carries this as two candidates, `cand-20260915-5` and `cand-20260915-6`, both dated 09-15 with this session id, stamped `architect × review` and `prd × review` respectively for what is one sighting in one interactive study session. This synthesis stamps the sighting `unknown × unknown`: a `/team` study is not a factory phase.

#### 1.6 The digest's self-correction counter fired on the trickle coder's own JSON answer (1 sighting, session `b203a05c`)

The digest excerpt reads `## Self-Corrections … (turn 22) [that's wrong] {"matches": [{"entry_id": "watcher-loop-harness-mismatch", … "candidates": [{"title": …`. `scripts/legibility/digest.py::iter_self_corrections` scans assistant text blocks only, by design ("native-carrier scoping"), and `that's wrong` is the first of `SELF_CORRECTION_PATTERNS`. A trickle coder's answer is an assistant text block, so the scoping that excludes tool results and Write inputs does not exclude a meta-session whose output quotes correction markers from the material it is coding.

**Correction to the verifier.** Its claim that "every previously filed instance of this recursive-ingestion bug was observed landing in the 'User Corrections' bucket" is contradicted by the codebook: `entry-cand-20260722-16` carries a 2026-08-02 sighting (session `6a527d51`) whose note reads "The self_correct signal detector fired on the phrase 'I was wrong' found inside a nested candidate's evidence_quote field within an embedded trickle-coder JSON blob". The bucket-agnostic observation is right and was already on record.

### 2. Overlap with the codebook, in one table

| finding | session | codebook state before this census |
|---|---|---|
| python -c flattened (5560) | `62f084a1` | shape on `entry-cand-20260722-6`; session absent |
| python -c flattened (4195) | `e05f42c4` | shape on `entry-cand-20260722-6`; session absent |
| `skim` in commit | `fbd7b22b` | `cand-20260919-4`, same session |
| em-dash SyntaxError | `fbd7b22b` | `cand-20260919-5`, same session |
| add_memory unreachable | `7a9d5656` | none |
| `_module_key` relative path | `bc5a9c30` | shape only (worktree path family) |
| bare-filename open | `bc5a9c30` | shape only (worktree path family) |
| watcher hours + forecast line | `38a8f2d8` | `cand-20260915-5` and `-6`, same session |
| self_correct on coder JSON | `b203a05c` | shape on `entry-cand-20260722-16` (08-02 sighting) |

Eight of nine findings re-observe recorded shapes; four re-observe recorded sessions. The verifier's novelty screen did not consult the codebook's `candidates:` block by session id.

### 3. Observations the runner may act on

These are observations, not rulings.

- **The flattening entry's title and cause do not describe what its sightings show.** Twenty-plus sightings, the three here included, are newline collapses in commands that pass through the Bash PreToolUse rewriter, not pre-check rejections. The discriminator (a `git status`/`git log`/`cat`/`head` subcommand in the same compound command) is checkable on every sighting the entry already holds. A direct reproduction needs a session allowed to run a multi-line `python3 -c` alongside `git status`; this one was not.
- **Two 09-19 candidates need their cause text replaced, not extended** (§1.1, §1.2): one names an unidentified layer that the audit doc identifies, the other blames Unicode for an unbalanced literal.
- **The fused-memory blip is the only novel shape and has one sighting.** Nothing here shows a worktree-specific cause. What would discriminate it is the fused-memory server's own log for 18:28:00 to 18:28:20 on 09-19, which this synthesis could not read.
- **Two candidates for one study sighting carry inconsistent phase stamps** (§1.5).

### 4. What this census did not verify

The hook script's contents and whether `skim rewrite --hook` joins lines; the fused-memory server's state at 18:28 on 09-19; the trickle-coder transcript for `b203a05c` (outside the sandbox; the digest excerpt is the verifier's); the outcome of the retried commit in session `fbd7b22b` beyond the fact that task 5580 reached review at 04:39 on 09-20.

### 5. Codebook dispositions (inputs to the merger)

- `entry-cand-20260722-6`: add sightings `62f084a1` (2026-09-18, dark_factory, unknown × implement) and `e05f42c4` (2026-09-17, implement × implement), each noting the co-located `git status`/`git log` subcommand and the 1-of-6 and 1-of-10 split. Retitle to the observed mechanism: compound Bash commands that the PreToolUse rewriter touches reach the shell with newlines collapsed. Point the cause at `~/.claude/hooks/skim-rewrite.sh` as documented in `plans/os-sandbox-claude-home-writer-audit.md`, marked as inferred until reproduced.
- `cand-20260919-4`: fold into `entry-cand-20260722-6` as a sighting rather than promote; replace "unidentified command-rewriting/interception layer" with the hook reference.
- `cand-20260919-5`: keep as its own candidate; retitle to "heredoc edit script embeds the target docstring's closing `\"\"\"` inside a `\"\"\"` literal", replace the Unicode cause with §1.2, keep the sighting, add the 27-second self-correction.
- New candidate: fused-memory MCP `add_memory` returns "Unable to connect" transiently while the same session's earlier writes and later escalation calls succeed; sighting `7a9d5656` (2026-09-19, unknown × architect); cause unverified.
- Worktree relative-path family: add sightings `bc5a9c30` ×2 (2026-09-16, implement × implement) with the 2.5 s and 6 s corrections.
- `cand-20260915-5` and `cand-20260915-6`: no new sighting; restamp both to unknown × unknown for session `38a8f2d8`, and note they describe one sighting.
- `entry-cand-20260722-16`: add sighting `b203a05c` (ops × ops) noting the `self_correct` counter and the assistant-text carrier.
- No entry retires.


## Filed Tasks

_none filed._

## Cost

invoke calls: sonnet miner=120, sonnet verify=12, fable synthesis=1, haiku headroom-probe=4
