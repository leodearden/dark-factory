"""One /prd seat brief per stream: program-doc row, rulings, finding keys with FULL statements/measurements/proposals, the Mechanical filings rows verbatim, seam rows, freshness clause."""
import json, sys, pathlib, re
sc = pathlib.Path(sys.argv[1]); rec = json.load(open(sc/'out/review-all-solar_challenge_platform-20261013.json'))
rul = {r['item']: r for r in json.load(open(sc/'rulings.json'))}; fil = {f['key']: f for f in json.load(open(sc/'filings.json'))}
syn = rec['synthesis']; fk = {f['canonical']: f for f in rec['findings']}; pr = {p['canonical']: p for p in rec['prior_rows']}
prior_full = {p['canonical']: p for p in json.load(open(sc/'prior_open.json'))['open']}
F = lambda c: fk.get(c) or pr.get(c) or {}
RUN = rec['method']['run_id']; SHA = rec['method']['as_of_sha']; MAIN = '/home/leo/src/solar-challenge-platform'
PROG = f'{MAIN}/plans/review-all-program-solar_challenge_platform-2026-10-14.md'
prog = pathlib.Path(PROG).read_text()
rows = {m.group(1): m.group(0) for m in re.finditer(r'^\| (S\d+) \|.*$', prog, re.M)}
seams = [l for l in prog.splitlines() if l.startswith('| ') and ' | S' in l and 'per the stream' in l]
ITEM = json.load(open(sc/'stream_items.json'))
out = sc/'prd-briefs'; out.mkdir(exist_ok=True)
for name, i in ITEM.items():
    st = next(s for s in syn['streams'] if s['name'] == name); r = rul.get(i)
    sid = f'S{i}'; slug = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')
    o = [f"# /prd seat brief — {sid} {name} ({st['mode']} mode)", "",
         f"Program: `{PROG}` — READ IT FIRST, in full (G4 seam map, rulings, shared conventions). Run `{RUN}`, record `{MAIN}/plans/{RUN}.json`, report `{MAIN}/plans/{RUN}.md`. Pinned tree `{SHA}`; main has moved — FRESHNESS CLAUSE: before building a task on any anchor below, `git diff {SHA[:10]}..main -- <path>` and re-read the symbol.", "",
         "## Your stream row (program doc)", "", rows.get(sid, '(row missing)'), "",
         "## Ruling (do not relitigate)", "", (f"R{r['item']}: {r['ruling']}\n\nOptions considered: {'; '.join(r['options_considered'])}\nConsequence that decided it: {r['consequence']}" if r else "(uncontested stream: approved as proposed under R18)"), "",
         f"## Remedy (synthesis)\n\n{st['remedy']}\n\n**Deletable:** {st['deletable']}\n\n**Seam candidates (G4):** {', '.join(st['seam_candidates']) or 'none'}", ""]
    if st['open_choices']:
        o.append("## Open choices — RULED above; listed so the trade-offs are visible")
        for oc in st['open_choices']:
            o.append(f"- {oc['question']}")
            for op in oc['options']: o.append(f"  - {op['option']} — {op['tradeoff']} — long-term: {op['long_term']}")
        o.append("")
    o.append("## Finding keys this stream acts on (full statements)"); o.append("")
    for c in st['canonicals']:
        f = F(c); full = prior_full.get(c, {})
        o.append(f"### `{f.get('key')}` — {f.get('severity')} — {f.get('verdict') or 'prior'} — `{f.get('anchor')}` — tags {', '.join(f.get('tags') or full.get('tags') or [])} — class {f.get('class')} — source {f.get('source_run') or 'this run'}")
        o.append(f"- statement: {f.get('statement') or full.get('statement')}")
        if full.get('review_statement'): o.append(f"- the /review sibling's statement: {full['review_statement']}")
        for m in f.get('measurements') or []: o.append(f"- measurement: {m.get('name')} = {m.get('value')} (rule: {m.get('rule')})")
        if f.get('next_change'): o.append(f"- next change: {f['next_change']}")
        prop = f.get('proposal') or full.get('proposal')
        if prop: o.append(f"- proposal (input, not instruction): {prop}")
        if f.get('verdict') == 'weakened': o.append(f"- WEAKENED by the skeptic: {f.get('verdict_notes')}")
        if f.get('critic_disputes_severity'): o.append(f"- critic disputes severity: {f['critic_disputes_severity']}")
        o.append("")
    o += ["## Mechanical filings rows for these keys (verbatim from the report; re-run contract §8 step 1 before filing regardless)", "", "| key | sources | dedup step | ticket/task | disposition |", "|---|---|---|---|---|"]
    any_ = False
    for c in st['canonicals']:
        k = F(c).get('key'); fl = fil.get(k)
        if fl: any_ = True; o.append(f"| `{k}` | {', '.join(fl.get('sources', []))} | {fl['step']} | {fl['ticket_or_task']} | {fl['disposition']}{(' — ' + fl['note']) if fl.get('note') else ''} |")
    if not any_: o.append("| — | — | — | — | none of this stream's keys was filed mechanically |")
    o += ["", "## Seam rows naming this stream", ""] + [l for l in seams if f"| {sid} |" in l or f"| {sid} " in l] + ["",
          "## What you return (structured block, no commits, no edits to the program doc or another stream's PRD)", "",
          "AUTHOR pass: write `docs/prds/review-all-<slug>.md` and `docs/prds/review-all-<slug>.capability-manifest.yaml` (this project's convention) under the /prd skill's gates G1-G6 with this brief as the folded-in context; return {prd_path, manifest_path, open_questions (only those the brief and rulings cannot answer), seam_dependencies_on_other_streams, cross_reference_lines}. Do not decompose or file anything in the author pass.",
          "DECOMPOSE pass (only when the lead sends the filed ids you depend on): decompose under planning_mode, stamp metadata.x_finding_key (list) / x_finding_run / source=review-all on every task, add your high/critical leaves as dependencies of the trigger-chain gate id the lead gives you, run commit_planning on your own batch (held leaves excluded), write the stamped sidecar, return your FILED row and cross-reference lines.", ""]
    (out/f'{sid}-{slug}.md').write_text("\n".join(o))
print('briefs:', sorted(p.name for p in out.iterdir()))
