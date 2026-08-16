# Tool-call parser over-consumption and silent parameter drop

## Summary

When a model closes a string-valued tool-call parameter with the wrong
closing tag, the parser does not raise an error. Instead it keeps scanning
forward past the intended value until it finds some other closing tag it
does recognize, and treats everything in between — every other parameter the
call was going to pass — as part of the first parameter's value. Those
intervening parameters are never delivered to the tool: not defaulted, not
logged, not surfaced as an error, simply discarded. We can reproduce this
deterministically, we have measured it happening in roughly 1 out of every
400 tool calls in our own transcripts, and for the large majority of the
corrupted calls we can deterministically reconstruct exactly which arguments
were dropped and what they contained — which is itself the strongest
evidence that the parser could raise an error here instead of silently
guessing.

> **Notation.** Every angle bracket below that would form part of a markup
> tag is written as the HTML entity `&#60;` instead of a literal `<`. This is
> not a style choice: the defect described in this report is triggered by a
> literal `<` appearing inside a string argument of a tool call, including
> the tool call that would be used to author this very report. Writing the
> raw character here would reproduce the bug in the act of reporting it —
> the authoring call would itself terminate early and silently drop its own
> remaining arguments, corrupting this document before it could even be
> saved. That this report cannot safely be written with literal angle
> brackets is itself part of the evidence for how the defect behaves.

## Over-consumption

Tool calls are serialized with each parameter's value wrapped in a pair of
tags: an opening tag naming the parameter, e.g. `&#60;parameter
name="description">`, and a matching closing tag, `&#60;/parameter>`. Models
occasionally close a parameter with the wrong tag. Two variants of this are
common: echoing the parameter's own name instead of using the generic
closer (`&#60;/description>` instead of `&#60;/parameter>`), and, having
drifted into that dialect, continuing to open and close the *following*
parameters the same way (`&#60;priority>medium&#60;/priority>` instead of
the canonical form). Both are a model-side formatting mistake, and this
report is not about fixing the model — see "What we're asking for" below.

What is under the parser's control is what happens next. When the parser
does not find the closing tag it expects, it does not treat this as a syntax
error. It **over-consumes**: it keeps scanning forward through the text
until it finds *some* closing tag it does recognize — a later well-formed
`&#60;/parameter>`, or, failing that, the tool call's own closing
`&#60;/invoke>` — and it treats everything between the original mis-close
and that terminator as though it were still part of the first parameter's
value.

The practical effect: a corrupted parameter's value ends up containing the
literal, still-tagged text of one or more *other* parameters, appended after
whatever the model actually intended that first parameter to contain.

## Silent parameter drop

Over-consumption alone would just produce an ugly value. What makes it a
serious defect is what happens to the parameters it swallows: every one of
them is **silently discarded**. Not defaulted to a documented fallback, not
logged, not surfaced to the caller or the model as an error of any kind —
the tool call simply proceeds as though those arguments were never supplied.

This is a substantially worse failure mode than a parse error would be, and
the difference is the entire point of this report. A parse error is loud: it
fails at the call site, with the offending text in hand, and whoever is
driving the call finds out immediately. A silently dropped argument is
quiet: the call *succeeds* — frequently with some other, unrelated piece of
code substituting a plausible-looking default for the now-missing parameter
— and the actual consequence surfaces, if it ever does, arbitrarily far
downstream, disconnected from its cause. We have repeatedly had to
reconstruct, well after the fact, which of several plausible default values
was silently substituted for an argument the model had actually supplied.

## Specimens

Four specimens, reconstructed from our own tool-call logs. The payload text
inside each value below has been replaced with a neutral stand-in rather
than quoted verbatim from the original call; what *is* reproduced exactly is
the markup structure — the tags, the mis-close, the parameter names, and
which of them end up dropped. The structure is the evidence this report
rests on, not the payload prose, and the structure is what the
machine-readable header above each block below is checked against. The tool
names below — `submit_task`, `add_memory`, `update_memory` — are our own
internal MCP tools; they appear only to illustrate the shape of the defect,
not because they matter to the harness. Each specimen also carries a
one-line HTML comment immediately above its block recording the call it
came from in a structured form (`id`, `tool`, the parameter whose value
absorbed the tail, the argument names that arrived intact, that tool's full
parameter set, and the names that were dropped); the comment is invisible in
a rendered view of this document and exists so the structural claim — that a
given piece of markup actually drops the named parameters — is independently
checkable against the parsed text, not just asserted.

This reconstruction applies only to the four specimens below. The Incidence,
Repairability, and Reproducibility figures later in this report are direct
measurements over the transcript archive, not reconstructions, and this
caveat does not extend to them.

| Specimen | Tool call | Corrupted parameter | Keys received | Keys dropped |
|---|---|---|---|---|
| S1 | `submit_task` | `description` | `project_root`, `title`, `description` | `priority`, `agent_id`, `metadata` |
| S2 | `submit_task` | `description` | `project_root`, `title`, `description` | `priority` (intended `low`) |
| S3 | `add_memory` | `content` | `content`, `project_id`, `category`, `agent_id` | none — `content` was the last parameter |
| S4 | `update_memory` | `content` | `memory_id`, `store`, `project_id`, `content` | `agent_id` |

### S1 — dialect blend

The model closes `description` by echoing its name (`&#60;/description>`
instead of `&#60;/parameter>`), then drifts into that same name-echoing
dialect for two more parameters, and finally blends the two dialects on a
third: `&#60;metadata">` closes with a stray quote immediately before the
bracket, on both its opening and closing tag. That stray quote is direct
evidence the model is interpolating between the canonical
`&#60;parameter name="X">` form and the name-echoing `&#60;X>` form, not
cleanly using either one.

&#60;!-- specimen id="S1" tool="submit_task" param="description" supplied="project_root,title,description" schema="project_root,prompt,title,description,details,dependencies,priority,metadata,tag,planning_mode,routing_override_reason,task_kind,agent_id" dropped="priority,agent_id,metadata" --&gt;
```text
Investigate the elevated error rate on the ingest pipeline and file a fix.&#60;/description>
&#60;priority>medium&#60;/priority>
&#60;agent_id>reviewer-bot-04&#60;/agent_id>
&#60;metadata">{"source": "weekly-audit"}&#60;/metadata">
&#60;/invoke>
```

### S2 — unterminated canonical opener

Here the drift is only partial: the model still opens `priority` the
canonical way, `&#60;parameter name="priority">`, but the call ends before
any closing tag at all — there is nothing left to consume once the intended
value ("low") runs out. The parser has no next terminator to over-consume
*to*, so the entire remainder of the text, starting from the mis-close,
becomes `description`'s value and `priority` never arrives.

&#60;!-- specimen id="S2" tool="submit_task" param="description" supplied="project_root,title,description" schema="project_root,prompt,title,description,details,dependencies,priority,metadata,tag,planning_mode,routing_override_reason,task_kind,agent_id" dropped="priority" --&gt;
```text
Retry the failed webhook delivery; this can likely be resolved automatically).&#60;/description>
&#60;parameter name="priority">low
```

### S3 — the invisible case

`content` is mis-closed exactly the same way as in the other specimens, but
here it happens to be the **last** parameter of the call. There is nothing
after it to over-consume, so nothing is dropped and the call behaves
correctly by accident. This specimen is included deliberately: it shows the
defect is purely positional. The same model mistake, on the same tool, is
completely invisible whenever the mis-closed parameter happens to be listed
last, and harmful whenever it isn't — nothing about the mistake itself
signals which case a given call falls into.

&#60;!-- specimen id="S3" tool="add_memory" param="content" supplied="content,project_id,category,agent_id" schema="content,project_id,category,agent_id,session_id,metadata,dual_write" dropped="" --&gt;
```text
This memory is being stored verbatim, matching the schema by design.&#60;/content>
&#60;/invoke>
```

### S4 — mis-closed with the generic tag

The opposite drift from the others: this time the model closes `content`
with the *generic* `&#60;/parameter>` closer rather than echoing its name,
then opens the next parameter canonically. The generic closer is exactly
what the parser is supposed to expect, on the wrong parameter — which is
enough by itself to send it looking further ahead, past `agent_id`, for the
next tag it recognizes.

&#60;!-- specimen id="S4" tool="update_memory" param="content" supplied="memory_id,store,project_id,content" schema="memory_id,store,project_id,content,metadata_patch,metadata_delete_keys,metadata_mode,reason,agent_id,session_id,metadata" dropped="agent_id" --&gt;
```text
The memory has been re-scoped).&#60;/parameter>
&#60;parameter name="agent_id">escalation-watcher-l2
```

## Incidence

Measured on 2026-08-05 against an archive of our own agent transcripts: 334
of 128,066 tool calls were corrupted this way — **0.26%**. A tool call
counted as corrupted if any one of its string-valued parameters matched,
verbatim: `&#60;/invoke>\s*$`, or
`&#60;/[A-Za-z_]\w*>\s*&#60;parameter\s+name="[^"]+">` (both written here
with the same escaping as the rest of this report; read `&#60;` as a literal
opening angle bracket). The whitespace requirement in the second pattern is
deliberate — it is what keeps ordinary prose that happens to quote a closing
tag out of the count, since a real over-consumption always mis-closes and
then continues directly into the next tag with nothing but whitespace in
between.

## Repairability

Replaying all 334 corrupted calls from that same archive through a
schema-validated, deterministic repairer: **308 repair cleanly (92.2%)**,
recovering 194 dropped parameters. The remaining 26 (7.8%) are
doubly-corrupted — more than one mis-close in the same value — and are
correctly refused rather than guessed, because a second mis-close makes the
boundary between recovered arguments ambiguous.

This is the load-bearing argument of this report. A downstream consumer,
working only from the malformed text and the tool's parameter schema, can
deterministically reconstruct the arguments the model actually supplied for
the overwhelming majority of corrupted calls. If a repairer with no access
to the original call can do this reliably, the parser itself — which has
strictly more context, including the exact position where parsing actually
diverged — could have raised a parse error at that point instead of silently
guessing a wrong terminator and discarding everything past it.

## What was actually lost

Of the 194 parameters recovered from those 334 calls, these seven were
dropped most often, across our own MCP tools: `category` ×70, `project_id`
×32, `rationale` ×25, `agent_id` ×18, `suggested_action` ×13, `issues` ×10,
`priority` ×5 — 173 of the 194, with the remaining 21 spread thinly across
other, lower-frequency parameters not itemized here.

These are not abstract slots. A dropped `category` or `project_id` on a
memory-write call means the record was filed under whatever fallback value
the receiving code substitutes for a missing argument, not the value
actually supplied — and nothing marks it as a fallback. A dropped `priority`
means a task runs at the wrong urgency indefinitely. A dropped `rationale`
means the reasoning behind a decision is gone, while some other field
elsewhere claims a reasoning that doesn't belong to it. None of this trips
any error path; all of it is a plausible, silently wrong value sitting where
a correct one should be.

## Reproducibility

A second, independent measurement on 2026-08-09 — a later, larger snapshot
of the same transcript archive, 5,704 transcript files (up from roughly
4,400 at the time of the 2026-08-05 measurement) — found 504 corrupted
calls: 443 repaired (87.9%), 61 unrepairable, 245 parameters recovered. The
same parameters dominate, at larger counts: `category` ×110, `project_id`
×44, `agent_id` ×25, `issues` ×16, `rationale` ×13, `suggested_action` ×12,
`priority` ×5 — 225 of the 245, with the remaining 20 spread thinly across
other, lower-frequency parameters not itemized here.

The repair rate moved from 92.2% to 87.9% between the two measurements, for
two identified and unremarkable reasons rather than one: the underlying
archive grew (more calls, of the same shape), and in the interim the
repairer itself was tightened to refuse a small class of cases it had
previously accepted — exactly 3 of the original 334 calls, re-checked under
the newer repairer, moved from repaired to unrepairable. Both measurements
are reported here, dated, rather than presenting either alone as a single
current figure: two independent snapshots showing the same shape, at
different points in time and different sample sizes, is stronger evidence
that this is a persistent property of the defect than either measurement
would be alone.

## What we're asking for

When the parser fails to find the closing tag it expects for a parameter, it
should raise a parse error identifying the parameter and the tag it found
instead — not consume forward and silently fold the intervening text into
that parameter's value.

To be precise about where the fault lies: the *originating* mistake is on
the model's side. It emits a closing tag in the wrong form — usually by
echoing the parameter's own name instead of using the generic closer, and
occasionally by blending the two forms together, as in specimen S1 above.
This report is not asking the harness to accept that wrong form as valid
input; that would treat a symptom as though it were a feature. It is asking
the harness to stop *amplifying* a model-side formatting mistake into a
silent loss of data the harness itself had no part in causing.

That distinction is the entire severity argument here. A model emitting a
slightly wrong closing tag is a mistake that can be improved at the model or
prompt level over time, and imperfectly, at that. A parser that responds to
it by discarding unrelated, correctly-formed arguments with no diagnostic of
any kind is a defect that only the parser's own maintainers can fix, and it
will keep costing real, silently-wrong data for as long as it goes unfixed
— regardless of how good the model gets, since the failure only needs to
happen once per call to lose everything after it.

## Scope of this report

We have already implemented deterministic detection and recovery for this
failure on our own side, downstream of the parser: a corrupted value can be
identified reliably, and, as shown above, the parameters it swallowed can be
recovered without guessing in the large majority of cases. That containment
is not what this report is about, and we are not blocked by this issue day
to day.

We are filing this because the amplification into silent data loss happens
*at* the parser, upstream of anything a downstream consumer can fix. No
amount of downstream detection or repair prevents the loss from happening in
the first place — it only cleans up after it, for the calls where cleanup
turns out to be possible at all. Fixing this at the source is the only way
to stop it from happening.
