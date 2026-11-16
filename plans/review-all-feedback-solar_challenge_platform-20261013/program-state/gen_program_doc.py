"""Program doc (report-format §Artefact 3) generated from the record, rulings.json and filings.json; the lead edits nothing by hand after."""
import json, sys, pathlib, re, collections
sc = pathlib.Path(sys.argv[1]); rec = json.load(open(sc/'out/review-all-solar_challenge_platform-20261013.json'))
rul = json.load(open(sc/'rulings.json')); fil = json.load(open(sc/'filings.json'))
syn = rec['synthesis']; fk = {f['canonical']: f for f in rec['findings']}; pr = {p['canonical']: p for p in rec['prior_rows']}
F = lambda c: fk.get(c) or pr.get(c) or {}
RUN = rec['method']['run_id']; SHA = rec['method']['as_of_sha']
# S-ids = deliberation item numbers; item 8 is routed, not a stream; S27 = invariants document leaf
ITEM = {s['stream']: s['item'] for s in rul if isinstance(s['item'], int) and s['item'] <= 17}
ITEM.update({'Declared package strata': 18, 'Device last-seen has one reader': 19, 'Dispatch composition out of control/': 20, 'Governed ledgers on one mixin and reader': 21,
             'Money quantum and wire decimal grammar': 22, 'Ops reads through public seams': 23, 'Settings domains and deployment profile': 24, 'Simulated fleet identity': 25, 'Time grid and UTC home': 26})
DEPS = {1: [26, 22, 21], 2: [1, 9], 3: [24], 4: [26, 9], 5: [26], 6: [26], 7: [1], 9: [], 10: [1, 21], 11: [23, 17, 13], 12: [14, 6, 5], 13: [9, 26], 14: [24], 15: [2, 24], 16: [1, 21], 17: [9],
        18: [], 19: [], 20: [3, 24], 21: [9], 22: [], 23: [19, 6, 2], 24: [], 25: [], 26: [], 27: []}
def wave(i, memo={}):
    if i in memo: return memo[i]
    memo[i] = 1 + max([wave(d) for d in DEPS[i]], default=0); return memo[i]
slug = lambda n: re.sub(r'[^a-z0-9]+', '-', n.lower()).strip('-')
rows = []
for st in syn['streams']:
    if st['name'] not in ITEM or ITEM[st['name']] == 8: continue
    i = ITEM[st['name']]; r = next((x for x in rul if x['item'] == i), None)
    rows.append((i, st, r))
rows.sort(key=lambda x: x[0])
o = [f"# /review-all Remediation Program — solar_challenge_platform — 2026-10-14", "",
     f"Program of 26 PRD streams (25 from the synthesis plus one invariants-document leaf) acting on `plans/{RUN}.md` (record `plans/{RUN}.json`, run `{RUN}`, pinned at `{SHA}`). This document is the authoritative G4 seam map and shared convention set for every `/prd` session in the program. Each stream's session MUST read this file before authoring. Deliberation rulings were made by Leo on 2026-10-14 with the run's lead (`skills/review-all/references/deliberation.md`); they are recorded under Resolved decisions and are not relitigated unless a finding's evidence changed.", "",
     f"Findings are cited by key (`fk-…`), never by display id or array position. Every key resolves in the record's `findings` (this run) or `prior_rows` (the siblings `hotspot-survey-solar_challenge_platform-20261013` and `review-solar_challenge_platform-20261013`, same tree). Stream ids are the deliberation item numbers (S8 is a routing, not a stream).", "",
     "## Streams", "", "Waves follow the upstream-dependency graph (a stream's wave = 1 + max of its upstream waves). Mode: `agent` = one seat of a lead-sequenced `/team` run; `spawn` = an interactive `/prd` session. Upstream deps are seam dependencies: a stream's tasks that touch an upstream seam declare dependencies on the upstream stream's leaf for it (held leaves per Shared conventions).", "",
     "| ID | Slug | Scope (one line) | Finding keys | Mode | Size/risk | Wave | Upstream deps |", "|---|---|---|---|---|---|---|---|"]
for i, st, r in rows:
    keys = [F(c).get('key') or f"?{c[:30]}" for c in st['canonicals']]
    o.append(f"| S{i} | {slug(st['name'])} | {st['remedy'][:260].replace('|', '/')}… | {len(keys)} keys: {', '.join(keys)} | {st['mode']} | {st['size']}/{st['risk']} | {wave(i)} | {', '.join('S%d' % d for d in DEPS[i]) or '—'} |")
o.append(f"| S27 | design-invariants-document | Create docs/legibility/design-invariants.md listing the ten accepted invariant candidates in checkable form, each pointing at the owning stream's enforcing test; no second list anywhere (INV-9) | 0 keys (ruling R19) | agent | S/low | 1 | — |")
b8 = next(st for st in syn['streams'] if st['name'] == 'Board requirements without a mechanism')
o += ["", "## S8 — Board requirements routed to capability PRDs (ruling R8; no quality stream)", "", "| Finding key | Anchor | Severity | Statement gist | Route to |", "|---|---|---|---|---|"]
ROUTE = [('alert', 'alerting PRD (tasks 440-491 family)'), ('fault', 'alerting PRD (fault/off emulation)'), ('onboard', 'new capability manifest: member onboarding + consent renewals (req 10)'), ('role', 'approve-issue-gate PRD (roles + actor audit, req 18)'),
         ('approv', 'approve-issue-gate PRD'), ('backup', 'ops PRD (backup/restore + system description, req 23)'), ('health', 'ops PRD (deep health)'), ('monthly', 'new capability manifest: timed monthly cycle (req 12-14)'), ('fleet', 'new capability manifest: fleet-wide configuration (req 8)'),
         ('recompute', 'new capability manifest: independent recompute + export reconciliation'), ('dispatch', 'new capability manifest: dispatch policy'), ('later-phase', 'req 17 later-phase pages: no route this program')]
for c in b8['canonicals']:
    f = F(c); text = (f.get('statement') or '').lower() + ' ' + f.get('anchor', '').lower()
    route = next((r for k, r in ROUTE if k in text), 'capability manifest to be named at /prd time')
    o.append(f"| `{f.get('key')}` | `{f.get('anchor','')[:70]}` | {f.get('severity')} | {(f.get('statement') or '')[:140].replace('|','/')}… | {route} |")
o += ["| `fk-8b142ca219bf` | `timeseries settlement: >1 publisher of a metric per half-hour` | medium | S6 lands only the D7 stop-gap (task 1169); the capability half — billing selects the property's designated certified meter, seed stops publishing registers from an INVERTER device — is capability work | P-Fleet / metering capability PRD (added 2026-10-14 from S6 Q1); key stays open after 1169 |"]
o += ["", "Each `/prd` session that owns one of these PRDs stamps the routed keys on the tasks it files (`x_finding_key`, `x_finding_run = " + RUN + "`); the keys stay `open` in the record until then.", ""]
filed = [f for f in fil if f['disposition'].startswith('filed:')]; notf = [f for f in fil if not f['disposition'].startswith('filed:')]
by_step = collections.Counter(f['step'] for f in fil)
o += ["## Mechanical findings already filed (Phase 5)", "",
      f"{len(filed)} mechanical findings carry a task after the contract §8 dedup protocol ({by_step.get('filed', 0)} new tasks, {by_step.get('filed (curator combined)', 0)} curator-combined into existing tasks, {by_step.get('2', 0)} matched to live tasks at step 2); {len(notf)} weakened candidates stay `open` in the record. A stream whose remedy subsumes one of those tasks declares a dependency on it or cancels it with `x_acceptance_reason` naming the subsuming leaf; it never files a second task for the key.", "",
      "| key | task | step | sources |", "|---|---|---|---|"]
o += [f"| `{f['key']}` | {f['ticket_or_task']} | {f['step']} | {', '.join(f.get('sources', []))} |" for f in filed]
o += ["", "Supersedes done tasks (the finding re-verified present after the task landed): " + ", ".join("`fk-598e0f0aadce`→136, `fk-160bca1a028e`→200, `fk-41e7dcec9005`→126, `fk-d5d6b9349097`→150".split(', ')) + ".", ""]
o += ["## Seam ownership (G4 — authoritative)", "", "| Seam / artefact | Owner stream | Consumers (do NOT redefine) |", "|---|---|---|"]
for i, st, r in rows:
    for s in st['seam_candidates']:
        o.append(f"| {s.replace('|','/')} | S{i} | per the stream's PRD; every other stream imports downward |")
o += ["| docs/legibility/design-invariants.md | S27 | every stream points at it for its invariant id |", "| migrations chain (CHAIN_HEAD, numbering) | S9 | every stream that adds a migration lands it through S9's convention; second to land renumbers |", "| runtime/roles/bill.py | S2 (billing/push owns bill_account); S15 (providers_from_env); runtime keeps argparse + exit codes | S3 composes; never imports ci |", "| the leaf time module | S26 | everyone imports downward |", ""]
o += ["## Design invariants promoted (ruling R19)", "", "The project had no invariants document; S27 creates `docs/legibility/design-invariants.md` as the single list. Each candidate is enforced by a test the owning stream writes; this doc carries only the pointer.", "", "| # | Invariant (gist) | Heuristic | Owning stream (enforcing test) |", "|---|---|---|---|"]
OWN = {1: 'S18', 2: 'S12', 3: 'S13', 4: 'S11 (gate on generated output)', 5: 'S21 + S10', 6: 'S7', 7: 'S6', 8: 'S9', 9: 'S17', 10: 'repo grep gate (S27 lands it)'}
for n, ic in enumerate(syn['invariant_candidates'], 1):
    o.append(f"| {n} | {ic['statement'][:160].replace('|','/')} | {ic['heuristic'][:60]} | {OWN[n]} |")
o += ["", "## Resolved decisions (do not relitigate)", ""]
for r in rul:
    o.append(f"### R{r['item']} — {r['stream']} ({r['date']}; ruled_by: {r['ruled_by']})\n\n- Ruling: {r['ruling']}\n- Options considered: {'; '.join(r['options_considered'])}\n- Consequence that decided it: {r['consequence']}\n")
o += ["### Contradiction resolutions carried from the synthesis", ""]
for n, c in enumerate(syn['contradictions'], 1):
    o.append(f"- C{n} (between {', '.join(F(b).get('key', b) for b in c['between'])}): {c['resolution_or_question'][:400]}")
# author-pass amendments: the lead applies each seat's returned rulings, actions and cross-reference lines here (one editor per document)
res_dir = sc / 'prd-results'
if res_dir.exists():
    o += ["", "## Author-pass amendments (lead-applied from the seats' returned blocks)", "", "Each /prd seat returned open questions, decompose-time actions and cross-reference lines; the lead ruled on the questions and records everything here. These amend the stream rows, seam map and conventions above; where they conflict with an earlier line, the amendment wins.", ""]
    for p in sorted(res_dir.glob('S*.json'), key=lambda p: int(p.stem[1:])):
        d = json.load(open(p))
        o.append(f"### {d['stream']} — `{d['prd_path']}` ({d.get('leaves_estimate', '?')} leaves)")
        for k, title in (("lead_rulings_on_open_questions", "Lead rulings on the seat's open questions"), ("revisions_to_earlier_rulings", "Revisions to earlier amendments"), ("lead_actions_now", "Actions taken by the lead now"), ("lead_actions_at_decompose", "Actions for the decompose pass"), ("cross_reference_lines", "Cross-references applied")):
            if d.get(k):
                o.append(f"- **{title}:**")
                o += [f"  - {x}" for x in d[k]]
        if d.get('seam_dependencies'): o.append(f"- **Seam dependencies:** {d['seam_dependencies']}")
        o.append("")
o += ["", "## Shared conventions (every session)", "",
      f"- Read this program doc first, then the stream's row, its finding keys in `plans/{RUN}.json`, and the `## Mechanical filings` rows for its keys (handed verbatim in the seat brief).",
      "- PRD path: `docs/prds/review-all-<slug>.md` with its `.capability-manifest.yaml` (this project's convention); the sidecar PASS|FAIL|OPEN per `decompose-filing-gotchas`.",
      f"- Every filed task carries `metadata.x_finding_key` (list of the keys it closes), `metadata.x_finding_run = \"{RUN}\"`, `metadata.source = \"review-all\"`, severity → priority, `metadata.files` file-level; `planning_mode=True` → `add_dependency` → one `commit_planning` per batch.",
      "- Contract §8 dedup per key immediately before filing, regardless of what the Mechanical filings table says (step 1 forensic lookup on tasks.db `json_each(metadata,'$.x_finding_key')`, then search_tasks ≥ 0.6 read with get_task).",
      "- An acceptance is a `cancelled` task with `metadata.x_acceptance_reason` (never `deferred`).",
      f"- Freshness clause: anchors were verified at `{SHA[:10]}`; before building a task on one, `git diff {SHA[:10]}..main -- <path>` and re-read the symbol.",
      "- Held leaf: a leaf that depends on a later stream's contract is filed in planning mode with `metadata.x_seam_wait = {stream, contract}`, wired into its own batch's dependencies and EXCLUDED from that batch's `commit_planning`; the owning stream's session adds the dependency on its filed leaf, commits it and re-stamps the sidecar.",
      "- One editor per document: a `/prd` seat never edits this program doc or another stream's PRD; it returns its FILED row and cross-reference lines as a structured block and the lead applies them.",
      "- Migrations: resolve the number from CHAIN_HEAD at implementation time (second to land renumbers); shared PG enums need `postgresql.ENUM(create_type=False)`; prefer NOT NULL (pre-release, no real data).",
      "- Trigger chain: the gate for this run is filed by the lead when the first stream files; every session adds its `high`/`critical` tasks as dependencies of the gate id recorded in FILED; `commit_planning` of the gate only when the FILED table is complete (human act).",
      "- `git commit --only <paths>`; never a direct commit of anything but PRD/manifest/sidecar/program files; the orchestrator lands code.", "",
      "## Triage briefing", "",
      "- Three findings of this run are also correctness defects the orchestrator may meet before the streams land: the portal consent scope (`fk-acec2b429476`, S13), the Dockerfile uid (`fk-0ef6e8bf8461`, S14) and the console enum crash path (`fk-6e809699ddc6`, S11). A fix-forward task that touches them must stamp the key.",
      "- The billing golden lane is held for task 177; S1/S2/S16 change billing arithmetic homes and must keep the golden byte-exact or re-bless under Docker with the reason recorded.",
      "- S18's strata test starts with an allowlist of the five known violations; a stream that removes one deletes its entry (do not file 'allowlist stale' as a regression).",
      "- The 8 weakened mechanical candidates left `open` are listed in the record's filings with `not-filed`; do not re-file them from the task title alone.", "",
      "## FILED — program status", "", "Appended by the lead from the rows the decompose seats return. Trigger chain: interim human gate = task 1065 (deferred; metadata.trigger_chain {role: human_gate, skill: review-all, run_id: review-all-solar_challenge_platform-20261013}; milestone delayed 259200 s); every stream adds its high/critical leaves as its dependencies; commit_planning on 1065 is the human's act once this table is complete.", "", "| Stream | PRD | Task ids (label) | Finding keys | Held leaves | Gate deps | Notes |", "|---|---|---|---|---|---|---|"]
filed_dir = sc / 'filed'
if filed_dir.exists():
    for p in sorted(filed_dir.glob('S*.json'), key=lambda p: int(p.stem[1:])):
        d = json.load(open(p))
        ids = ", ".join(f"{f['task_id']} ({f['label']}{', ' + f['note'] if f.get('note') else ''})" for f in d['filed'])
        keys = sorted({k for f in d['filed'] for k in f.get('x_finding_key', [])})
        o.append(f"| {d['stream']} | `{d['prd_path']}` | {ids} | {', '.join('`'+k+'`' for k in keys) or 'none'} | {', '.join(d.get('held_leaves') or []) or 'none'} | {', '.join(d.get('gate_dependencies_added') or []) or 'none'} | {'; '.join(d.get('lead_rulings') or [])[:400]} |")
o += ["", "Cancelled as accepted into program leaves (x_acceptance_reason names the leaf): 789 (S22), 1051, 995, 996, 999, 1000 (S24), 359, 423, 928, 793, 936, 778, 922, 735, 1021, 1035, 1045 (S26), 1028 (S9), 1027, 285 (S20), 824, 568, 1063, 965, 685, 496, 863, 944, 1033 (S23), 570 (S21), 673, 684 (S17), 567, 931 (S14), 1001, 395, 390, 544 (S3). Amended in place: 1013 (S19 beta), 1044 (S25 beta), 517 (S26 kappa), 1029 (S23 gamma), 533 (re-scoped to the per-shard advisory lock). Held deferred for their streams: 582 (S13), 584 (S6). Keys stamped on pre-existing tasks: 210, 213, 446, 451, 454, 518. Cross-PRD edges added: 612/613 -> 1103; 596 -> 1123; 595/468/648 -> 1130 + 1138; 477 -> 1143. Done tasks 443 and 484 had delivered_checks repathed ahead of S14's moves. Held leaves released so far: 1133 -> 1126 (S17 beta2); 1118 -> 1150 + 1151 (S3 delta/epsilon); 1116 -> 1168 + 1173 + 1177 + 1178 (S4 eta/iota/lambda/mu); 1119 -> 1184 (S6 kappa); 1205 -> 267 (S1 epsilon, 2026-10-15).", "", "Lead actions 2026-10-15 (resumed session after the 2026-10-14 lead hit its context limit): cancelled as accepted — 895, 893, 853 (S4), 1201, 592 (S13), 1210 (S10 acceptance record), 192, 329, 280, 879 (S1); amended in place — 194, 267, 365, 519, 422, 555 (S1), 516, 1020 (S10), 493 (S13), 582 re-scoped (S13); keys stamped on pre-existing tasks — 734 (fk-f87240eea84a), 699 (fk-643fb20ec130 encoder half); cross-PRD edges — 511 -> 1168, 601 -> 1173 + 1205, 602 -> 1203, 603 -> 1204, 617 -> 1192, 618 -> 1193, 619 -> 1197 + 1198, 620 -> 1196, 972 -> 582, 521 -> 555, 930 -> 1162, 194/1209 -> 298, 1202 -> 899; held deferred for their streams — 295, 1046 (S2), 803, 811 (S16).", "", "Lead actions 2026-10-15, second round (S2, S7, S11, S12, S15, S16 decomposed; FILED table complete): cancelled as accepted — 803, 811 (S16), 939, 554, 1049, 1043, 717, 786, 825 (S11), 708, 514 (S12); amended in place — 295, 301, 1046 (S2), 1047 (S11), 809 (S12), 1025 (S15); keys stamped on pre-existing tasks — 476 (fk-43579bc83b3d), 1016 (fk-e507cf236576), 308 (fk-8e67094cbf75), 637 (fk-97dcf1114cc0, fk-29ac1fb9951f), 641 (fk-29ac1fb9951f, fk-fc2f2705caa3), 466 (fk-fc2f2705caa3); cross-PRD edges — 459 -> 1235, 460 -> 1230 + 1235 + 301, 637 -> 1235, 639 -> 1232, 644 -> 1235 + 1277, 663 -> 1232, 697 -> 1258, 605 -> 1258, 752 -> 1271, 972 -> 1269, 342 -> 1272, 471/472 -> 1259, 984 -> 1263, 466 -> 1276 + 1277, 641 -> 1277, 655 -> 1277, 1059/1222 -> 1025; held leaf released — 1117 -> 1232 (S23 beta2 on S2 beta1); delivered_checks repaired — 1205 (narrowed), 493 (widened), 459, 460, 655 (repathed); kept for Leo — 785 (branch has commits; 1262/1264 subsume), 752 (OPERATOR consent read), 582 vs 617 (basis-branch owner), 382 (per-task node_modules).", "", "Leo 2026-10-15 evening: gate 1065 committed (pending, 67 deps); 785 cancelled naming 1262 + 1264; 394 cancelled naming 1141 (esc-394-2 closed); ruling — OPERATOR may READ consent (S7 theta 1271; 752 kept); ruling sessions spawned for S15 Q3, S16 Q1, 582 vs 617 and 382 (briefs under ~/.claude/spawn-briefs/review-all-scp-2026-10-15/). The briefing refreshes above are filed as deterministic always-escalates human gates (run /review-briefing interactively on main when each fires): 1287 (S3, after 1153), 1288 (S13, after 1197), 1289 (S1, after 555), 1290 (S2, after 1235), 1291 (S11, after 1239 + 1249), 1292 (S11, after 1255/1258/1262/1264), 1293 (S12, after 1274), 1294 (S16, after 1236).", "", "Leo 2026-10-16 rulings (sessions rule:solar_challenge_platform#1279/#582/#1236; the #382 session is still open): S15 Q3 -> staging rung (R15 wording amended; S24 1077/1084 carry it; follow-up task 1304 staging banner, deps 1077 + 1239); 582 vs 617 -> A-narrow (582 owns the lawful-basis table and branch; 617/618 rewritten; 1200 -> 582; 617's edges to 442/461 removed); S16 Q1 -> option B (new leaf delta2 = 1299, not a gate dep). For Leo: amend ~/mission-control/consulting/solar-challenge/decisions.md (2026-09-29 cbs_billing clause superseded: CONTRACT, provisional) and put the billing Art 6 basis to the board with the DPIA.", "", "Leo 2026-10-16, task 382 (session rule:solar_challenge_platform#382): merge-verify frontend gate, tests drive the install, fail-closed; 382 re-scoped in place; 1298 filed (option B on dark_factory 5653); briefing gate 1305 after 382. The kept-for-Leo list above is now empty. Open caveat for Leo/lead: verify fallback scoping runs no pytest for a .py diff with no collectable test file (noted on 1268).", ""]
out = pathlib.Path('/home/leo/src/solar-challenge-platform/plans/review-all-program-solar_challenge_platform-2026-10-14.md')
out.write_text("\n".join(o)); print('wrote', out, out.stat().st_size, 'bytes; streams', len(rows) + 1)
for i, st, r in rows: print(f"S{i:<3d} w{wave(i)} {st['mode']:5s} {st['size']}/{st['risk']:6s} {slug(st['name'])}")
