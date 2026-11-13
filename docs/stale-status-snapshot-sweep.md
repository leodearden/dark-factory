# Stale status-snapshot edge sweep

## Purpose

This doc is the single home for the design rationale of Reconciliation Stage
1's deterministic stale status-snapshot edge sweep (task 2613). The sweep
retires VALID (`invalid_at IS NULL`) Graphiti edges that assert a task is
active/pending/blocked/stalled/in-progress once that assertion is
contradicted by the task's live status. The code keeps contracts and points
here as `docs/stale-status-snapshot-sweep.md §"<heading>"`. Edit the
rationale in this doc, not in the code.

Figures that move as the corpus grows, such as corpus sizes, read costs and
recall censuses, live in the `plans/` artifacts cited below, never here.
Refresh them there. The few figures quoted here are point-in-time history: the
result of a past run, or a timing of code that no longer exists. Each is
labelled with its run or task, and none is ever refreshed.

Stage 1 (`MemoryConsolidator`) had no deterministic sweep keeping these edges
in sync with the tasks they describe. Once a task reached a terminal status
(done/cancelled) its snapshot edges went stale and nothing retired them. The
prior approach, a manual LLM `search` sweep, was unreliable: run b1408864
found only 17 of an estimated 23+ stale edges. The sweep replaces it with a
deterministic direct lookup. It follows the deterministic-lookup-over-
semantic-search precedent of task 1680 (`count_memories_by_metadata`) and
task 2107 (`degenerate_task_node_sweep`).

## Layout

- `fused-memory/src/fused_memory/reconciliation/status_snapshot_extraction.py`
  is the pure lexical layer. It holds the marker alternations, the five
  anchored patterns and their one builder, the clause-break walk, the two
  subjecthood guards, and the `extract_*` entry points. It imports only the
  standard library and `task_filter`.
- `fused-memory/src/fused_memory/reconciliation/stale_status_snapshot_edge_sweep.py`
  is selection and orchestration. It holds the two selection rules, the
  superseding write, the corpus read, the per-project lock and both public
  sweeps. It imports exactly the three extraction entry points it calls.
- The extraction module never imports the sweep. That is pinned in a fresh
  interpreter by
  `fused-memory/tests/reconciliation/test_status_snapshot_extraction.py::test_extraction_module_imports_nothing_from_the_sweep`.
- Callers: `reconciliation/stages/memory_consolidator.py` runs the periodic
  sweep in Stage 1, and `reconciliation/targeted.py` runs the per-task sweep
  after a done/cancelled/deferred transition.
- Suites: `tests/reconciliation/test_status_snapshot_extraction.py` and
  `tests/reconciliation/test_stale_status_snapshot_edge_sweep.py`. The
  pinned plural-enumeration shapes live in
  `tests/reconciliation/plural_enum_shapes.py`, and the cost-counting `str`
  in `tests/reconciliation/counting_str.py`.

## Standing invariants

These cut across every extraction path and both selection rules. Re-check
all of them before changing any single path.

- **Fail-safe direction.** Under-selection self-heals next cycle, or is
  caught by Stage 2. Over-selection is unrecoverable, since an invalidated
  Graphiti edge is never re-validated. Every extraction path is
  deliberately biased toward missing a snapshot over wrongly retiring one.
- **Marker-addition rule.** EVERY extraction path must anchor its marker
  adjacently to whatever it describes. The whole-fact gate
  (`SNAPSHOT_STATUS_RE`) only decides whether a fact is worth considering at
  all. It matches a marker ANYWHERE in the fact, so it can never be what
  separates a status snapshot from an unrelated fact. The aggregate list path
  was once left resting on the gate alone and had to be re-anchored
  (`LIST_INTRODUCER_RE`) once 'blocked' widened the gate. The task-2885
  repro: 4 stale edges survived a blocked->done transition (invalidated=0)
  because the gate short-circuited before the `INACTIVE_TASK_STATUSES`
  cross-reference ever ran. Any marker added to `_STATUS_MARKER_ALT` must
  therefore be re-checked against every path individually, not just the
  individual form. `COUNT_QUANTITY_RE` is the one existing consumer that is
  NOT safe by default under such a widening; see
  §"Aggregate-list form". (task 3042)
- **Transitive markers.** 'blocked' and 'stalled' are also common transitive
  verbs. 'Task 5 blocked the merge queue' is a permanently-true historical
  fact, not a status snapshot. So both are matched only in copula/article
  form, never as a bare adjective; see §"Marker classes and the compound
  prefix". Same hazard and remedy as task 2824's transitive-verb caveat on
  `task_filter.PRESENT_TENSE_COMPLETION_RE`.
- **Closed-class connective vs status noun.** `INDIVIDUAL_SNAPSHOT_RE`
  admits only a closed-class connective: copula or article, never the
  preposition 'in', which would let the marker bind to a noun the task is
  merely located in. `SNAPSHOT_STATUS_PHRASE_RE` is the one path that DOES
  permit an open-class gap and the 'in' preposition, and only because it also
  requires the marker be immediately followed by the literal noun 'status'.
  That status-noun requirement keeps 'Task N is in the active branch' out
  while still reaching 'Task N is in a blocked status ...'. (task 3042)
- **Digits only from inside an anchored span.** A bare `\d+` contributes an
  id only from inside an already-detected, marker-anchored span: an
  aggregate list segment, or a plural enumeration's `ids` capture group
  (task 3079). Dates, commit hashes and ports in ordinary prose are neither.
  This is invariant (d) of §"Plural-enumeration form".
- **Invalidate only on positively known.** See §"Selection rules". An
  aggregate or plural edge is invalidated as a whole the moment any ONE
  referenced id contradicts it. Neither rule ever fires on an id whose census
  value is absent or outside the closed `shared.task_statuses.TaskStatus`
  vocabulary.
- **Best-effort.** Mirrors `degenerate_task_node_sweep`. An
  edge-enumeration or status cross-reference (`get_statuses`) failure ends
  this cycle's sweep early, and the sweep self-heals next cycle. A per-edge
  invalidation failure does not end it. See §"Sweep execution".
- **'blocked' is not inactive.** 'blocked' is deliberately NOT added to
  `INACTIVE_TASK_STATUSES`, which stays `{done, cancelled}`. That is what
  preserves invalidate-only-on-positively-terminal for a
  genuinely-still-blocked task. A stale blocked assertion IS retired before
  its task goes terminal (task 3037), but by a SEPARATE second rule keyed on
  the blocked-scoped id set; see §"Selection rules".
- **Two families, one builder.** Every extraction path exists in TWO
  families, union and blocked-only, built by the same
  `_build_snapshot_patterns`, and the blocked result is always a SUBSET of
  the union result. A future path or guard must be added inside the builder,
  never as a family-specific special case. A blocked id the union family
  cannot reach would let the sweep select an edge on an id it never put into
  its `get_statuses` census. See §"Pattern families and the subset
  property". (task 3037)

## Marker classes and the compound prefix

The status markers are split by part of speech.
`_ADJECTIVE_MARKER_ALT` ('active', 'pending', 'in progress') holds adjectives
only. `_TRANSITIVE_MARKER_ALT` ('blocked', 'stalled') holds markers that are
also common transitive verbs, so they are matched only in copula/article
form (tasks 3042, 3079).

`_STATUS_MARKER_ALT` is the union of both classes. It is derived from the
two constants rather than hand-written, so the gate (`SNAPSHOT_STATUS_RE`)
and the anchored matchers can never drift apart again. Before task 3042 they
were hand-copied duplicates, and a marker present in one but not the other
silently suppressed a whole class of edges: the gate short-circuits before
the `INACTIVE_TASK_STATUSES` cross-reference in
`select_stale_status_snapshot_edges` ever runs. It is used ONLY by the gate
and by the status-noun-anchored `SNAPSHOT_STATUS_PHRASE_RE`. Both are safe
against the transitive-verb reading: the gate merely admits the fact for
consideration, and the phrase form requires the literal noun 'status' after
the marker.

`SNAPSHOT_STATUS_RE` is the whole-fact gate. A fact that asserts none of
these markers anywhere is not a status-snapshot edge the sweep concerns
itself with at all. 'Task 5 is done' was already terminal at write time.

`_COPULA_ALT` is the closed-class connective shared by both arms of
`INDIVIDUAL_SNAPSHOT_RE`: optional for the adjective arm, mandatory for the
transitive-capable arm. `_ADVERB_ALT` admits closed-class adverbs between the
copula and the marker, so 'Task 5 is currently blocked' and 'is still
blocked' match as readily as 'Task 5 is blocked'. It mirrors the
`(?:been|already|now|fully)\s+` slot in
`task_filter.PRESENT_TENSE_COMPLETION_RE`. It is deliberately an explicit
closed-class list rather than a `\w+` gap: an open-class gap would re-admit
the very readings the two arms exist to exclude.

### The compound prefix

`_COMPOUND_PREFIX` admits an optional hyphenated-compound modifier
immediately before the marker. 'pairwise-stalled' then anchors to its copula
exactly as bare 'stalled' does ('Tasks A, B and C are pairwise-stalled', the
finding's verbatim fact).

It is bounded to a single hyphenated token. `\w+` cannot match whitespace
or punctuation, so it admits one compound modifier and nothing else. Were it
allowed to span whitespace, it would re-admit the multi-word readings the
mandatory-copula arms exist to exclude ('Tasks A and B are awaiting a
blocked merge': the MERGE is blocked).

That bound is on the gap's WIDTH, NOT on its lexical class. 'un-', 'non-',
'not-' and 'previously-' are each one `\w+-` token, so an unguarded prefix
re-admits the negation and past-exit readings that `_ADVERB_ALT`'s
closed-class discipline refuses for free on the bare marker:

    'Task 5 is un-blocked.'                -> {5}
    'Task 5 is previously-blocked.'        -> {5}
    "Task 5's status is un-blocked."       -> {5}
    'Tasks 1020 and 1030 are un-blocked.'  -> {1020, 1030}

Each is a permanently-true historical or negated assertion. The sweep would
retire a still-true edge the moment any referenced id went terminal, which
is unrecoverable over-selection.

The remedy mirrors `_GAP_EXCLUDED_ALT` exactly. The prefix stays open-class,
because the observed modifier vocabulary ('pairwise-', 'merge-', 'self-',
'auto-') is not closed and an unlisted but innocent modifier should not
silently drop a genuine snapshot. The closed class of inverting prefixes is
subtracted with a negative lookahead. The lookahead sits BEFORE the optional
group, so it is evaluated whether or not a prefix is present. No marker
begins with one of these tokens, so the prefix-less shape ('is blocked') is
unaffected.

The privative 'in-' belongs to the same subtracted class but cannot be a
bare 'in' entry in that alternation. The lookahead is evaluated at the
marker's own start position, and `in[-\s]?progress` is itself a MARKER, so
'in' in the main list would refuse 'Task 5 is in-progress' along with 'Task 5
is in-active'. It gets a second, narrower lookahead that fires only when
'in-' is NOT the head of 'in-progress'. The spaced spelling ('is in
progress') is untouched either way, since that lookahead requires a literal
hyphen. (task 3079)

The prefix is used only in the adjacency-anchored arms: individual, genitive
and plural enumeration. The phrase form needs no equivalent, since its marker
is pinned on the far side by the literal noun 'status'. (task 3079)

## The five extraction paths

Each path anchors its status marker directly to what it describes rather
than resting on the whole-fact gate alone. All five are compiled by
`_build_snapshot_patterns`. `extract_snapshot_edge_task_ids`'s docstring
gives the order in which they compose.

| constant | shape |
|---|---|
| `INDIVIDUAL_SNAPSHOT_RE` | 'Task N is/was <marker>', 'Task N <marker>' |
| `GENITIVE_STATUS_RE` | "Task N's status is <marker>" (task 3079) |
| `SNAPSHOT_STATUS_PHRASE_RE` | 'Task N ... in <marker> status' |
| `PLURAL_ENUM_SNAPSHOT_RE` | 'Tasks A, B and C are <marker>' (task 3079) |
| `LIST_INTRODUCER_RE` plus list segments | '<marker> tasks: [...]', '<marker> tasks are [...]' |

### Individual form

The individual form anchors the status marker directly to its own task
reference, so an incidental status word elsewhere in the fact is never
attributed to it ('Task 142 landed on the active branch' -> set()). It is
built from `TASK_REF_RE.pattern` rather than a hand-copied duplicate, so the
two stay in sync if the shared task-reference grammar changes. That grammar
covers 'task N', '#N', 'df N' and 'task/N'; task 3403 admitted the slash
form.

Two arms, split by the marker's part of speech. The adjective arm takes an
OPTIONAL copula, so verb-less 'Task 5 pending' matches. The
transitive-capable arm requires a MANDATORY copula or article, so 'Task 5 is
blocked' matches but the bare verb 'Task 5 blocked the merge queue' does
not. Only closed-class function words (copula, adverb, article) may sit
between reference and marker; notably NOT the preposition 'in', which would
bind the marker to a following noun. (tasks 2613, 3042)

Individual-form extraction used to run against a `COUNT_QUANTITY_RE`-stripped
copy of the WHOLE fact. So a verb-less snapshot like 'Task 5 in progress' or
'Task 5 pending' had its own digit stripped out first (it matches
`\d+\s+in[-\s]?progress` / `\d+\s+pending`) before id extraction ever ran: a
silent false negative. Anchoring the marker to its own reference fixed that,
and `COUNT_QUANTITY_RE` is now applied only to an already list-scoped
aggregate segment, never to the whole fact. (task 2613)

### Genitive form

"Task N's status is <marker>" is the canonical shape Graphiti writes for a
per-task status snapshot. Before task 3079 EVERY path missed it while the
whole-fact gate fired, the "matched the gate yet went undetected" symptom.
`INDIVIDUAL_SNAPSHOT_RE`'s `\s*` cannot cross the intervening "'s status ";
`SNAPSHOT_STATUS_PHRASE_RE` requires 'in <marker> status', with the marker
BEFORE the noun, whereas here the order is inverted; and
`LIST_INTRODUCER_RE` requires `<marker> tasks[:\[]`. So the miss rate was
structural, not incidental.

Both apostrophes are admitted: ASCII U+0027 and the typographic U+2019 that
most prose writers, LLM writers included, actually emit.

Two properties keep this path cheap:

- No trailing noun-phrase lookahead is needed, unlike the phrase form.
  Requiring the literal noun 'status' to be IMMEDIATELY followed by the
  copula already excludes the modifier-head hazard that lookahead exists
  for. In "Task 5's status report is pending review", 'report' sits between
  'status' and 'is', so nothing matches.
- No `_GAP_EXCLUDED_ALT` equivalent is needed either. Reusing the
  closed-class `_ADVERB_ALT` means negation and past-exit qualifiers ('is no
  longer', 'is not', 'was previously') are refused for free, because those
  tokens simply are not in the alternation. That is the same asymmetry the
  individual form has. Only the open-class gap of the phrase form has to buy
  that protection back explicitly.

The copula is MANDATORY here, so the whole marker alternation, including the
transitive-capable 'blocked', is safe: there is no bare-verb reading of "Task
5's status is blocked".

### Status-phrase form

The permissive path. It admits an open-class gap of up to 3 words between
the task reference and the preposition 'in' ('is deliberately parked in
...'), but ONLY when the marker is immediately followed by the literal noun
'status'. The gap is lazy (`{0,3}?`) and bounded, and cannot cross a clause
boundary: '.', ',' and ';' are not `\w`, and the gap's leading `\s+` cannot
absorb them. So 'Task 5 is done. Task 9 is in blocked status' anchors only to
task 9.

An earlier comment claimed the 'status' noun meant the open-class gap "costs
no precision". That was an overstatement a future maintainer would have
relied on. The status-noun requirement removes most but not all binding
ambiguity. Four precision residuals have been identified. (1), (2) and (4)
are now EXCLUDED, and (3) alone still binds and is pinned by a
documented-behaviour test rather than claimed away. (task 3042)

1. **A following head noun**, where 'status' is a modifier rather than the
   phrase head: 'Task 142 is tracked in the pending status report' is a
   pending status REPORT, not a pending task. Excluded by requiring the
   marker+status span to end the noun phrase: the trailing lookahead accepts
   end of string, punctuation, or a preposition or subordinator such as
   'due', 'under' or 'since'. 'status report' is common phrasing in this
   repo's memory corpus, so this was worth closing rather than merely
   documenting.
2. **Negation and past-exit qualifiers** absorbed by the gap invert the
   assertion outright: 'Task N is no longer in blocked status', 'was
   previously in blocked status', 'is not in blocked status', 'was briefly in
   blocked status, then merged'. Each is a permanently-true historical fact,
   precisely the blocked->unblocked transition that writes such an edge in
   the first place. That is the transition class of the task-2885 repro, so
   it is not exotic phrasing. Excluded by `_GAP_EXCLUDED_ALT`, a negative
   lookahead applied per gap word. The asymmetry that made this
   phrase-form-only: the individual form's closed-class connective already
   refuses the same semantics for free ('Task N is no longer blocked' ->
   set()), because 'not', 'no', 'longer' and 'never' were never in
   `_ADVERB_ALT`. An open-class gap forfeits that protection, which is the
   standing cost of the permissive path.
3. **A gap-internal nominal** that is the real subject of the phrase: 'Task 5
   depends on work in blocked status'. The WORK is blocked, not task 5. This
   still yields {5} and is NOT fixed: distinguishing it needs to know that
   'work' is the subject, which is beyond lexical matching at this altitude.
   It is pinned by a test so the trade-off stays visible. Widening the gap
   beyond `{0,3}` words would make it strictly worse.
4. **An intervening task reference** absorbed by the gap re-subjects the
   assertion onto a different task. 'Task 5 blocks task 9 in blocked status'
   and 'Task 5 supersedes task 9 in pending status' both yielded {5}. That
   is strictly worse than a plain over-selection, because `re.finditer` is
   non-overlapping: the wrong match consumed the inner 'task 9', so the
   CORRECT id was silently never extracted either. One match, both error
   directions at once. The bug needed BOTH a gap of at most 3 words and a
   word-initial reference token: 'df 9' was affected identically, '#9' never
   was (the gap's `\w+` cannot match '#'), and a 4+-word gap ('Task 5
   depends on task 9 in active status') was already out of reach. Excluded
   by `_GAP_NO_TASK_REF`. It is kept separate from `_GAP_EXCLUDED_ALT`
   because that list is about words which INVERT the assertion and this one
   is about words which RE-SUBJECT it. Its vocabulary mirrors
   `TASK_REF_RE`'s own ('task', 'df', digits); the third alternative, a
   literal '#', needs no exclusion. Its `\w*\b` tail makes it deliberately
   WIDER than `TASK_REF_RE`: 'tasks' is refused as a gap word even though
   the plural is not itself a reference. The cost is under-selection on
   shapes like 'Task 5 blocks tasks 9 in blocked status' (now set(),
   previously the wrong {5}), pinned by a test.

### Plural-enumeration form

'Tasks A, B and C are <marker>' (task 3079). Before this path such an edge
yielded NO ids at all, not merely the first. `TASK_REF_RE` anchors
`\btask\b` and so does not match the plural head 'Tasks', and even if it did,
the enumeration tail carries no reference token of its own.

Four anchoring properties, mirroring the adjacency discipline of every other
path:

- (a) The copula is MANDATORY; there is no optional-copula arm. It is the same
  remedy as `INDIVIDUAL_SNAPSHOT_RE`'s transitive arm, and it refuses the
  transitive-verb reading 'Tasks 1020, 1030, and 1031 blocked task 5'. Since
  the copula is mandatory, the whole marker alternation is safe here.
- (b) The enumeration must be the copula's SUBJECT. That is established
  jointly by plural agreement (`_PLURAL_COPULA_ALT`) and the absence of a
  governing preposition (`_ENUM_PREP_WORDS`); see §"Plural-enumeration
  subjecthood guard". Adjacency does NOT establish this reading: in 'The
  merge of tasks 1020 and 1030 is blocked' the marker IS adjacent to the
  copula, but the copula's subject is the outer head noun and the MERGE is
  what is blocked.
- (c) The marker must sit immediately after the enumeration and its copula,
  with only closed-class function words (adverb, 'all', article) in between.
  So 'Tasks 1020 and 1030 were merged into the active branch' does not
  match: the BRANCH is active, not the tasks. This is necessary but NOT
  sufficient for the (b) reading.
- (d) Bare digits are collected from the `ids` capture group ONLY, never from
  the whole fact. See the digits invariant in §"Standing invariants".

On (d): the aggregate path strips `COUNT_QUANTITY_RE` spans from its segment
before collecting digits, because a colon or bracket segment is free text
that really can contain '...and 3 pending others'. This path deliberately
omits the strip. `_ENUM_IDS_ALT`'s alphabet admits no count noun, so the
strip could never fire usefully, while it WOULD misfire on the
reference-token separator: '1020 task 1030' matches `COUNT_QUANTITY_RE`'s
`\d+\s+tasks?`, and stripping it would silently drop the legitimate leading
id. A count phrase adjacent to a real enumeration ('tasks 1020 and 1030 plus
3 pending others are pending') instead breaks the copula adjacency and kills
the match outright. That is under-selection, and it is pinned by a test.

A rejected enumeration suppresses EVERY anchored path inside its span, not
just the plural match. An enumeration may REPEAT the reference token, a shape
the plural path supports as a positive ('Tasks 1020, task 1030 and task 1031
are pending'). Suppressing only the plural match left its prepositional-
complement counterpart reaching the individual arm on its own, which
over-selected the trailing id:

    'Reviews for tasks 1020, task 1030 and task 1031 are pending.' -> {1031}
    'Statuses of the tasks 1020 and task 1030 are blocked.'        -> {1030}

A rejected span is, by construction, a region established to be a
preposition's complement, so NO id inside it is the copula's subject,
whichever pattern found it. The drop is scoped to the span, not the fact, so
a genuine snapshot in a later clause survives. (task 3079)

#### The enumeration tail

`_ENUM_IDS_ALT` matches '1020, 1030, and 1031', '1020 and 1030', '1020, task
1030 and task 1031'. It requires TWO OR MORE ids by construction: the
repeated group is `+`, not `*`. After a plural noun a lone number is more
likely a count than an id, and the well-formed singular shape already routes
through `INDIVIDUAL_SNAPSHOT_RE`.

The separator `_ENUM_SEP_ALT` must absorb the OXFORD COMMA (', and'), not
merely ',' or 'and' individually: a single-token separator silently misses
the finding's verbatim fact. Its alphabet is deliberately closed: digits,
whitespace, ',', 'and'/'&'/'/', and an optional repeated 'task' or '#'
reference token. That is what lets the ids group be scanned for bare digits
directly.

The separator's LEADING element, a comma or whitespace, is mandatory and
cannot match empty. That is what actually enforces two-or-more ids. With
every element optional, the repeated group matched between two adjacent
digits of a SINGLE number, so '1020' parsed as '102' + '0' and 'Tasks 1020
are pending' wrongly satisfied a rule meant to need two ids.

#### Possessive quantifiers (ReDoS)

Every quantifier in the enumeration tail is POSSESSIVE (task 3079). Written
greedily, the separator's TRAILING `\s*` overlaps its own LEADING `\s*,\s*` /
`\s+`. So a whitespace run between two ids can be apportioned between those
elements in (len(run) + 1) distinct ways. Under `_ENUM_IDS_ALT`'s `+`
repetition that is (w + 1) ** n parses, and ALL of them get explored whenever
the overall match ultimately FAILS. That is the common case, since most
enumerations in prose are not followed by a copula and a status marker.
Task 3079 timed the code before the fix on a non-matching fact. A list of 20
ids separated by ', ' took seconds, and one of only NINE ids separated by
',\n    ' took longer still. The base scales with whitespace-run WIDTH, so a
newline-indented list wedges the sweep at a much lower id count than a flat
2**n reading suggests. After the fix every shape parses in well under a
millisecond. The pre-fix timings and the current regression budget sit beside
the test that pins them:
`fused-memory/tests/reconciliation/test_status_snapshot_extraction.py::TestPluralEnumerationPerformance`.

Possessiveness is safe here because it changes only HOW a span is parsed,
never WHETHER it matches. Every possessive element is followed by something
that can never match whitespace: each `\s*+` / `\s++` is followed by ',', a
letter, '#' or `\d++`, and `\d++` is followed by whitespace or ','. So no
quantifier can ever hold a character a later element needs, and the accepted
enumeration alphabet (including '1020, # 1030' with an internal space after
the reference token) is unchanged. Possessive quantifiers require Python >=
3.11, which is this package's floor (`fused-memory/pyproject.toml`).

Pinned by `tests/reconciliation/test_status_snapshot_extraction.py::TestPluralEnumerationPerformance`.

### Aggregate-list form

`LIST_INTRODUCER_RE` detects the start of an aggregate list segment:
'<marker> tasks [', '<marker> tasks are [', '<marker> tasks:' or '<marker>
tasks are:'. Its `open` group records which delimiter opened the segment
('[' vs ':'), so `_list_segment` can decide how to find the segment's end.

It used to be a bare `\btasks?\b...` with no marker of its own, leaving the
whole aggregate path anchored to nothing but the whole-fact gate. Widening
that gate with 'blocked' therefore routed straight around the
transitive-verb hardening of `INDIVIDUAL_SNAPSHOT_RE`. 'Task 5 blocked these
tasks: 142, 148' and 'Task 5 blocked the merge of tasks [142, 148]' both
yielded {142, 148}. Each returned set() before 'blocked' joined the gate, so
the gate was the only thing holding them back. Those are permanently-true
historical facts, so the sweep would retire them the moment 142 or 148 went
terminal. (task 3042)

The marker is now required IMMEDIATELY before the list noun, the same
anchoring principle the individual and phrase forms use: adjacency to the
thing described, not mere co-occurrence somewhere in the fact. 'The active
pending tasks are [...]', 'active/in-progress tasks: ...' and 'Blocked tasks:
...' all still match. The two facts above no longer do, because the
intervening open-class words ('these', 'the merge of') break adjacency.

Two residuals, both in the fail-safe direction:

- Under-selection: a genuine aggregate whose marker is not adjacent to the
  list noun ('the tasks in the merge queue are [142, 148] and remain
  blocked') is not extracted.
- Ambiguity: the exact adjacency 'Task 5 blocked tasks: 142, 148', with no
  intervening word, still matches in the UNION family. It is genuinely
  ambiguous between the transitive-verb reading and the status-list reading
  'Blocked tasks: [...]', a shape the sweep must keep. Requiring adjacency
  narrows the hazard to that one collision rather than every '... blocked ...
  tasks:' fact. The BLOCKED family drops it; see §"The blocked family's list
  guard".

#### Counting phrases

`COUNT_QUANTITY_RE` strips 'N <count-noun>' spans ('8 tasks', '148
cancelled', '3 pending') from an aggregate list segment before bare-digit
extraction, so an embedded count phrase ('...tasks: 142, 148, and 200 total
remain') never contributes a spurious id.

Its alternation must be kept in step with `_STATUS_MARKER_ALT`. It is the
one `_STATUS_MARKER_ALT` consumer that is NOT safe by default under a marker
widening, and so the concrete reason the marker-addition rule exists. The
gate merely admits a fact for consideration, the phrase form requires the
literal noun 'status' after the marker, and the list introducer requires
adjacency to the list noun. But a marker missing HERE silently turns a count
phrase into a task id: 'Stalled tasks: 1020, 1030, and 3 stalled others'
would yield a spurious 3. 'stalled' was added for task 3079. It is
deliberately a separate literal list rather than an interpolation of
`_STATUS_MARKER_ALT`: it also carries non-marker count nouns ('tasks',
'done', 'total', 'deferred'), so the two sets overlap without either
containing the other.

#### The segment boundary

A colon-introduced segment has no closing delimiter of its own, and a bracket
segment may be unterminated. Both are closed by
`task_filter.py::STRICT_CLAUSE_BOUNDARY_RE`, the fail-safe-strict alphabet
`[.;\n!?]`. It is NOT `task_filter.py::WIDE_CLAUSE_BOUNDARY_RE`, which
narrowed its dot to `\.(?!\w)` at task 3403 so dotted technical tokens stop
shattering a sentence. The two started out identical; the canonical
rationale for keeping them apart lives next to `STRICT_CLAUSE_BOUNDARY_RE`.
The stake here: the segment is one from which BARE DIGITS are harvested as
task ids, and that path ends in `memory_service.update_edge(invalid_at=...)`,
a real Graphiti edge retirement. A longer segment absorbs more incidental
digits, which is over-selection. Copying the widening for symmetry would
trade a bounded soft-block for a permanent wrong invalidation.

#### The blocked family's list guard

`_list_is_governed_by_task_ref` drops a list segment whose introducer is
preceded, in the same clause, by an explicit task reference: 'Task 5 blocked
tasks: 142, 148' reads as a transitive verb taking that reference as its
subject, not as the status list 'Blocked tasks: 142, 148'. Only the BLOCKED
family runs it (task 3037).

`LIST_INTRODUCER_RE`'s adjacency requirement narrowed the transitive/
status-list collision to exactly one shape, and task 3042 logged that
survivor as a deliberate ambiguity residual. It was tolerable under the
TERMINAL rule alone: a permanently-true historical fact was only wrongly
retired once one of its listed ids reached done/cancelled, which is rare and
late. The BLOCKED rule retires on ANY positively known non-'blocked' status,
so the same residual fires on essentially the next cycle, and additionally
emits a superseding fact attributed to the wrong subject. That is a material
amplification in the over-selection direction, so the blocked family trades
it for under-selection.

The union family keeps the old behaviour deliberately. Narrowing it too
would change the task-2613 terminal rule, whose residual is unchanged in
both likelihood and consequence. Neither choice can violate the
blocked-subset-of-union invariant, because this guard only ever REMOVES ids
from the blocked family.

It is clause-scoped for the same reason the plural guard is: a reference in
an earlier sentence governs nothing here, so 'Task 5 is blocked. Blocked
tasks: 142, 148' keeps its aggregate.

Cost (fail-safe): a GENUINE aggregate that follows a task reference in its
own clause ('Task 5 is waiting on blocked tasks: 142, 148') is no longer
reachable by the blocked family. Its edge is not retired by the blocked
rule; the terminal rule still retires it when a listed id goes
done/cancelled, exactly as before task 3037.

## Pattern families and the subset property

All five paths are compiled by ONE builder and instantiated TWICE (task
3037): the UNION family, from which the five exported constants are bound,
and a BLOCKED family narrowed to the literal 'blocked' marker.
`extract_blocked_assertion_task_ids` runs the blocked family to report WHICH
ids a fact asserts as blocked. The union extractor structurally cannot make
that distinction: 'Task 5 is pending' and 'Task 5 is currently blocked' both
yield {5}. And the blocked selection rule requires it, since applied to union
ids it would retire an active/pending snapshot the moment its task moved to
'review'.

Hand-copying five blocked-only patterns would have recreated exactly the
drift surface task 3042 closed for the MARKER constants: a hardening applied
to one family and silently missed on the other. Deriving both from one
builder makes that drift structurally impossible. Every guard tasks 2613,
3042, 3079, 3403 and 4149 bought (transitive-verb, negation and past-exit,
intervening task reference, prepositional-complement subjecthood,
possessive quantifiers, token-internal-break narrowing) is written once and
applies to both families. `_extract_ids` likewise holds the single shared
ALGORITHM, so the families cannot drift in algorithm the way hand-copied
patterns would have let them drift in grammar.

The blocked marker is the literal 'blocked' ALONE, not the whole transitive
class. 'stalled' is not a member of the closed `TaskStatus` vocabulary, so
there is no census value a 'stalled' assertion could be compared against.
The blocked rule keys on "positively known and != 'blocked'", a test that is
only meaningful for a marker with a corresponding status value. Because
'blocked' is transitive-capable, it is supplied as the transitive
alternation, and the individual form's adjective arm is omitted: the
mandatory copula or article is exactly what keeps 'Task 5 blocked the merge
queue' out. The arms are assembled conditionally because an empty
alternative would match the empty string and turn every bare task reference
into a snapshot assertion.

A guard that only ever NARROWS one family is the one permitted asymmetry,
since it cannot break the subset property. It is still expressed as a builder
PARAMETER carried on `_SnapshotPatterns`, never as a branch on family
identity inside `_extract_ids`. `reject_list_after_task_ref` is the only
such parameter today. The two rules retire an edge on very different
triggers, so a residual tolerable under one can be intolerable under the
other.

### What actually makes the subset property hold

It is NOT pattern nesting, which is what the docs claimed until it was
measured false. Two mechanisms, both in `_ids_within_union`:

1. **Threading.** The union family's prepositional-complement
   `rejected_spans` are threaded into the narrower family's call. A rejected
   span is, by construction, a region established to be a PREPOSITION'S
   COMPLEMENT, so no id inside it is the copula's subject, whichever family
   and whichever pattern found it. That is verbatim the argument for
   applying a rejection across all three anchored paths within a family; it
   extends across the two FAMILIES for exactly the same reason, and that is
   where it was missed. `rejected_spans` is computed per call from a
   family's OWN `plural_enum` matches, and the blocked family's `plural_enum`
   needs the literal 'blocked' marker. So an enumeration the union family
   rejected on a 'pending', 'active', 'in progress' or 'stalled' marker was
   never built by the blocked family, suppressed nothing there, and left its
   individual arm free to claim a repeated 'task N' reference inside a region
   already ruled out:

       'Dependencies for tasks 1020 and task 1030 are pending in blocked
        status'  ->  union set(), blocked {1030}   (measured before the fix)

   An edge asserting the DEPENDENCIES are pending was retired, and a status
   attributed to a subject the fact never made a claim about.
2. **Intersection at the seam.** The narrower family's ids are intersected
   with the union family's. With threading in place this is MEASURED to be a
   no-op across the whole generated `_SUBSET_PROPERTY_CORPUS` (the counts sit
   beside that constant in the extraction suite), so it is NOT the fix. It makes the containment true BY CONSTRUCTION rather than by
   argument, at the cost of one set intersection on the small minority of
   edges that clear the `_BLOCKED_MARKER_RE` pre-gate. It exists because the
   original "the marker alternations nest, therefore the results nest"
   argument is UNSOUND for regex families in general: `finditer` is
   non-overlapping, so narrowing an alternation can delete an earlier match
   and thereby UNSHADOW a later one the wider family never reported. The
   intersection bounds that whole class of surprise in the safe direction.

The general warning worth carrying forward: ALGORITHM DATA computed inside
`_extract_ids` is PER-FAMILY, and must be threaded explicitly whenever it can
only NARROW a family's result. Sharing the pattern builder does not carry
it. The one piece of per-family data that is already safe and must be LEFT
ALONE is `reject_list_after_task_ref`, True for the blocked family only: it
makes that family reject MORE, which preserves the subset property.
`extract_marker_task_ids` (task 4851) passes the same seam, so its per-marker
ids are also a subset of the union.

The subset property is tested against a corpus GENERATED from axes, not
hand-picked:
`tests/reconciliation/test_status_snapshot_extraction.py::_SUBSET_PROPERTY_CORPUS`. A property ranging over the same inputs as the
examples is not a property test; it is the examples again.

## Plural-enumeration subjecthood guard

Two independently necessary constraints together establish that the
enumeration is the copula's SUBJECT rather than a preposition's complement.
Adjacency to the copula does NOT establish subjecthood; that was exactly the
defect. (task 3079)

1. **Plural agreement.** If the enumeration really is the subject, the
   copula must agree with it, so 'is', 'was' and 'remains' can only belong to
   some OTHER, singular subject ('The merge of tasks A and B is blocked': the
   MERGE is blocked). 'Tasks A and B is pending' is malformed English, so
   nothing legitimate is lost. The shared `_COPULA_ALT` is left untouched for
   the other paths; only this path narrows, via `_PLURAL_COPULA_ALT`.
2. **Preposition guard.** Agreement alone is NOT sufficient. When the outer
   head is itself PLURAL, the plural copula agrees with THAT head and the
   over-selection survives: 'Dependencies for tasks A and B are blocked',
   'The merges of tasks A and B are blocked', 'Reviews of tasks A, B, and C
   are pending'. All are ordinary phrasings in this repo's memory corpus,
   and all were verified still over-selecting under a copula-only fix.
   Agreement is a general grammatical constraint with no vocabulary to
   maintain; the preposition check covers the plural-outer-head residue
   agreement cannot see. They ship together.

The guard runs in PYTHON against the text preceding the match, not as an
in-pattern lookbehind. Python lookbehind is fixed-width, and a fixed offset
is defeated by a single intervening determiner. It scans the whole CLAUSE:
reject when a listed preposition appears anywhere between the last clause
break and the enumeration. Both properties are load-bearing, because the gap
between the preposition and the list noun is an OPEN class (determiners,
quantifiers, bare adjectives, possessives, participles, and stacks of them)
that neither a fixed offset nor any bound can cover. Each of these
over-selects under a narrower spelling (a fixed-width lookbehind pinned
before `\btasks\b`, a closed determiner slot, the review's suggested bound of
two arbitrary words), and every outer head here is plural, so agreement does
not save them either:

    'Statuses of the tasks 1020 and 1030 are blocked.'         (1 word)
    "Reviews of Leo's tasks 1020 and 1030 are pending."        (1 word)
    'Notes about a few tasks 1020 and 1030 are pending.'       (2 words)
    'Statuses of quite a few tasks 1020 and 1030 are pending.' (3 words)

Clause scoping closes that class outright rather than moving its boundary.
It is the safe direction to err in: every case where it disagrees with a
bounded form is one the clause form REJECTS.

`\b`-anchoring of the preposition words is retained, which is what keeps
'Migration tasks' and 'Verification tasks' matching: the 'on' inside
'Migration' is not `\b`-preceded. The clause start is located by a reverse
scan (§"Clause breaks") rather than by splitting the prefix into clauses, so
the cost is one bounded scan of text the regex engine has already walked.

### Residuals

ONE residual points the WRONG way. The preposition vocabulary
(`_ENUM_PREP_WORDS`) is a closed list, so an UNLISTED preposition does not
cause a miss: the guard simply fails to fire, the plural path matches, and
the ids are selected. That is over-selection. An earlier note called this
"fail-safe under-selection", which is what made the gap read as acceptable
(corrected, task 3079). So the list is a liability, not a safety margin. It
was widened once already, with from/by/under/within/around/through/via/
during/concerning, after the initial list was found to admit exactly the
shapes it exists to refuse ('Dependencies from tasks 1020 and 1030 are
blocked.'). It is one flat tuple, not inlined into the pattern, so the
regression test can parametrize over it directly: an entry added without a
plural-head case fails at once
(`tests/reconciliation/test_status_snapshot_extraction.py::TestExtractSnapshotEdgeTaskIds::test_every_guarded_preposition_is_exercised_against_a_plural_head`).

The converse residual IS fail-safe: a genuine subject-position enumeration
sharing a clause with a listed preposition is missed. Its main real-world
class is the sentence-initial ADVERBIAL PREAMBLE, since a comma does not end
the clause:

    'As of 2026-08-09, tasks 1020 and 1030 are pending.'      -> set()
    'In the merge queue, tasks 1020 and 1030 are pending.'    -> set()

Date-stamped and scope-setting preambles are common in this corpus, so this
is a real recall cost on the plural path, not a hypothetical one. It is
pinned by
`tests/reconciliation/test_status_snapshot_extraction.py::TestExtractSnapshotEdgeTaskIds::test_adverbial_preamble_is_a_documented_under_selection`.
Every candidate tightening trades back toward unrecoverable over-selection,
so it needed measurement against the real edge corpus first.

### The task-3949 measurement and verdict

That measurement was run against every populated FalkorDB graph the store
reports, discovered rather than hardcoded, because a hardcoded project list
is a coverage claim nothing checks. The figures live in the artifact, never
here: `plans/plural-enum-guard-recall-report.md` (per-graph table, totals)
and `plans/plural-enum-guard-recall-report.json`. An earlier draft
hand-copied that table inline, and every row went stale within days as the
corpus grew.

What the measurement FOUND is the durable part. The recall cost of this
guard on the real corpus is ZERO edges. Only two live facts reach the guard
at all, and both are genuine prepositional complements it is right to
reject, such as 'Task 3949 mentions that blockers for downstream,
still-unmerged tasks 1020 and 1030 are pending' (the BLOCKERS are pending).
Neither triages as an adverbial preamble, the only rejection class that
costs recall. Discount for the observer effect: both edges were written on
the first measurement day by task 3949's own session, so the corpus still
offers zero organic matches.

What the plural path loses, it loses at the REGEX, not at the guard. The
artifact's 'tasks <n>' column is a LEXICAL SUPERSET that already includes the
two full matches. The remainder carry 'tasks <digits>' but no status marker,
no copula, or broken copula-marker adjacency ('Tasks 1752 and 1753 are
related to ...', 'Both tasks 1920 and 1921 edit ...').

Both candidate tightenings were re-validated against the full precision
parametrization, as the finding required before shipping either:

- (a) A plural or capitalized head before the preposition. It re-opens none
  of the pinned precision and suppression shapes, but does NOT recover 'As of
  <date>, tasks ...', the finding's OWN motivating shape, because 'of' there
  follows the capitalized sentence-initial 'As'.
- (b) Restarting the backward scan after a preposition-free comma. It
  recovers all three preamble shapes, but RE-OPENS the pinned over-selection
  'Blockers for down-stream, still-unmerged tasks 1020 and 1030 are pending.',
  where the comma is intra-clause. It fails the required re-validation
  outright, and on live data too: it re-opens one of the two real corpus
  rejections above, which is the same shape.

VERDICT: NOT TIGHTENED. Zero measured recall loss to buy back, against
nonzero unrecoverable over-selection risk.

The verdict is only as good as its re-runnability, and a rejection census is
a point-in-time fact. Re-run it as the corpus grows:

    cd fused-memory && uv run python scripts/measure_plural_enum_guard_recall.py

The candidate re-validation is mechanical, in
`fused-memory/tests/test_measure_plural_enum_guard_recall.py`. It
parametrizes over the shared pinned corpora in
`tests/reconciliation/plural_enum_shapes.py`, which the extraction suite
parametrizes off too, so a shape appended there re-validates both
candidates. `_plural_enum_ids` takes its guard as a parameter solely so the
probe can run the IDENTICAL arm under a candidate guard instead of
re-implementing it. A hand-copied arm would keep measuring the old spelling
after the production arm gained a step, and report a reassuring zero while
doing it. It is a REPORTING seam, not an extension point: no production
caller may pass a different guard. Its `pattern` parameter defaults to the
union family's plural arm, so the probe's one-argument call keeps measuring
exactly what it always measured; `_extract_ids` passes its own family's arm
explicitly, because the blocked family narrows it.

## Clause breaks

SENTENCE-FINAL punctuation ends the span a preposition can govern, so the
plural guard and the blocked list guard look no further back than the last
break. That keeps 'Reviews for X. Tasks 1020 and 1030 are pending.' selected:
the enumeration opens a NEW sentence and really is its copula's subject.

A break requires TWO things together. The character must be a member of
`CLAUSE_BREAK_CHARS`, AND the occurrence must plausibly END a sentence, that
is, not be TOKEN-INTERNAL (`is_token_internal_break`). Every member of the
class gets that second, occurrence-level test (task 4149 for '.', task 4851
for ';', '!' and '?').

The class is deliberately minimal ('.', ';', '!', '?' and nothing else),
because the two directions are not symmetric at either level, class or
occurrence. Membership is a verified precision requirement (task 3949
embargo). Omitting a character from the class, or disqualifying an
occurrence via the flanking test, can only LENGTHEN the scanned clause, so
either can only ever cost under-selection. Admitting a character or
occurrence that does not actually end a sentence truncates the scan and
re-opens the over-selection the guard exists to close. So a character earns
a place in the class only by being able to end a sentence, and an occurrence
counts as a break only by plausibly ending one.

### Excluded from the class

Each was verified over-selecting when it was treated as a break (task 3079):

    ':'          'Statuses of the following: tasks 1020 and 1030 are pending.'
    '(' ')'      'Reviews of the following (still open): tasks 1020 and 1030
                  are pending.'
    '"' '“' '”'  'Statuses of the "next" tasks 1020 and 1030 are pending.'
    ','          'Statuses of the following, tasks 1020 and 1030, are pending.'
    newline      'Reviews for  the\n  tasks 1020 and 1030 are pending.'

A colon after a governing preposition no more ends that preposition's
government than a comma does; it typically introduces the very complement
the preposition governs. A parenthetical or quoted aside is an interpolation
INSIDE the clause, not a new one. The comma and newline exclusions are
load-bearing for the same reason. Colon-preambled genuine snapshots are
unaffected ('Note: tasks 1020 and 1030 are pending.' still extracts): they
carry no listed preposition, so the longer clause gives the guard nothing to
fire on.

### Token-internal occurrences

Excluded as breaks because such an occurrence ends no sentence. Each was
measured over-selecting when treated as a break (task 4149):

    filename extension     'Reviews for verify_cmd.py tasks 1020 and 1030
                            are pending.' -> {1020, 1030}; the
                            extension-free control 'Reviews for verify_cmd
                            tasks 1020 and 1030 are pending.' -> set() is
                            the measured minimal difference.
    version string         'Statuses of the v1.2 tasks 1020 and 1030 are
                            pending.'
    dotted module path     'Statuses of tasks in df.core tasks 1020 and
                            1030 are pending.'
    dotted section number  'Reviews for section 4.2.1 tasks 1020 and 1030
                            are pending.'

The same holds for a token-internal ';', '?' or '!' (a URL query '?', a
path or branch ';') and for the ';' that ends an HTML character entity
('&lt;', '&#60;') (task 4851). The entity case needs its own rule because
such a ';' is usually followed by a quote or paren rather than an
alphanumeric, so the flanking test alone would leave it breaking the clause.
Every excluded shape is pinned in
`tests/reconciliation/plural_enum_shapes.py::PRECISION_GUARD_SHAPES`.

A prefix-FINAL break, with nothing to its right within the prefix, has no
right flank and so is never alnum-flanked. It stays a break regardless,
which keeps the common 'X.Tasks 1020 and 1030 are pending.' shape selected.

Cost (fail-safe): a genuine sentence break with no following space AND a
following alphanumeric ('Reviews for the branch are done.Then tasks 1020 and
1030 are pending.', or the same with '?') reads as token-internal. The
backward scan then extends past the governing preposition and suppresses a
real snapshot; the edge is simply not retired this cycle. Pinned by
`tests/reconciliation/test_status_snapshot_extraction.py::TestExtractSnapshotEdgeTaskIds::test_token_internal_break_narrowing_costs_only_under_selection`.

### Cost of the walk

The walk's cost property is "no slicing, O(len(prefix)) total scan work". A
time-based test could not detect a quadratic regression at the input size it
shipped with, because the regression's extra cost is C-level slice memcpy.
So the performance tests count characters touched instead
(`tests/reconciliation/counting_str.py::CountingStr`). (task 4149)

## Selection rules

`select_stale_status_snapshot_edges` selects an edge when EITHER rule fires.

1. **Terminal rule** (task 2613). Any union-marker id resolves to a
   positively terminal status (`INACTIVE_TASK_STATUSES`, done/cancelled).
   Because it reads the UNION of every marker, it covers
   active/pending/in-progress/stalled assertions too.
2. **Blocked-assertion rule** (task 3037). Any blocked-asserted id resolves
   to a positively known status other than 'blocked'. A blocked assertion is
   contradicted by ANY non-blocked status, not merely a terminal one. Before
   this rule, 'Task 2848 remains blocked as of 2026-07-22' survived a
   blocked->pending unblock and stayed live until the task eventually reached
   done/cancelled.

In both rules an aggregate or plural edge is selected as a whole the moment
ANY single referenced id contradicts it: the snapshot as asserted no longer
holds.

Why rule 2 is a SEPARATE rule keyed on a blocked-scoped id set, rather than
'blocked' joining `INACTIVE_TASK_STATUSES`. That widening would make rule 1
retire every ACTIVE/PENDING snapshot whose task merely became blocked. And
because rule 1 reads union ids, a blocked-status trigger there could fire on
an id some OTHER marker contributed. Rule 2 reads only the ids the fact
asserts as blocked, which is exactly its scope. The two rules stay disjoint
in what they read: rule 1 reads union ids against `INACTIVE_TASK_STATUSES`,
rule 2 reads blocked ids against "positively known and != 'blocked'".

Positively known. `is_blocked_assertion_contradicted` checks membership of
the closed `TaskStatus` vocabulary, frozen at import as
`_KNOWN_TASK_STATUSES`. A naive `status != 'blocked'` would read every
unrecognised value as a contradiction, so a malformed, renamed or sentinel
census entry ('unknown', '', a future status this build has not heard of)
would RETIRE a live edge. Treating it as unknown costs at most one cycle of
under-selection. `None`, an id absent from the census, is likewise unknown,
which is what makes a transient `get_statuses` gap self-healing. The same
policy is adopted elsewhere, for example by
`orchestrator/tests/test_config_census_ignore.py`'s citation-liveness audit.

The general rule (asserted status != live status, for every marker) was
measured by task 4851 and NOT adopted; the verdict is in
`plans/stale-status-snapshot-sweep-report.md`. 'active' and 'stalled' have no
`TaskStatus` counterpart, hence no contradiction test, so they stay
terminal-rule-only.

The precomputed `edge_ids` / `blocked_edge_ids` maps exist so a caller that
already extracted ids (the sweep, which needs them to build its census)
does not re-run the extraction pipeline a second time per edge (task 2613,
extended to the blocked family by task 3037).

## The superseding write

After a successful blocked-rule invalidation, the sweep writes ONE
resulting-state-only `temporal_facts` memory per contradicted task per cycle
(`build_supersede_fact`), so the graph records what replaced the retired
assertion instead of merely losing it. (task 3037)

### The template

`fused-memory/src/fused_memory/reconciliation/stale_status_snapshot_edge_sweep.py::_SUPERSEDE_FACT_TEMPLATE`
is RESULTING-STATE-ONLY, and every word of it is load-bearing. Two
independent hazards rule out the wordings a reader reaches for first. Both
were measured on the extractor and on `task_filter`'s write guards, and both
are pinned by
`tests/reconciliation/test_stale_status_snapshot_edge_sweep.py::TestBuildSupersedeFact`.

1. 'Task {N} was blocked and is now pending' is FORBIDDEN, twice over. The
   sweep's OWN extractor re-extracts it (measured -> {2848}), so the
   superseding fact would satisfy the very selection rule that produced it:
   next cycle the sweep would retire it and write a fresh copy, forever. It is
   also MIXED TEMPORAL FRAMING, which
   `fused-memory/src/fused_memory/reconciliation/prompts/stage1.py::STAGE1_SYSTEM_PROMPT`
   §"Resulting-state-only temporal_facts" forbids for temporal_facts, because
   Graphiti atomization can re-emit the stale 'was blocked' fragment as a
   co-current edge, reintroducing exactly the assertion the sweep just
   retired.
2. 'Task {N} is no longer blocked' is also forbidden, but for only ONE of
   those reasons, which is why it is worth naming separately. It is
   extractor-SAFE (measured -> set()), so an author checking only the
   self-churn hazard would conclude it is fine. It is PRIOR-STATE framing,
   which the same resulting-state-only rule refuses. Unlike (1), NOTHING
   executable rejects this wording: all four of `task_filter`'s write-norm
   predicates pass it (asserted in
   `TestBuildSupersedeFact::test_prior_state_negation_is_extractor_safe_but_still_not_the_template`).
   The prohibition is a prose norm, so the template's compliance rests on
   review, not on a guard.

The chosen template states the CURRENT state and nothing else, with an
explicit 'As of <date>' stamp so it reads as a point-in-time observation
rather than a standing claim about a live task-table field (the distinction
`task_filter.frames_live_task_status_as_current_fact` enforces). The
timestamp is passed in, never read from a clock, so the fact carries the same
instant the sweep used as the edge's `invalid_at`, and a replayed or
back-dated run stays reproducible.

The sweep writes in-process via `memory_service.add_memory`, and so BYPASSES
the MCP-boundary enforcement in
`fused-memory/src/fused_memory/server/tools.py::add_memory` (and its
`add_episode` sibling) that would otherwise reject a `recon-stage-*`
writer's malformed framing. Compliance therefore rests entirely on the
template and its tests; there is no second line of defence downstream.

### One write per task, and the in-cycle retry

One fact per contradicted TASK per cycle, not per edge: a contradicted task
typically has several blocked-assertion edges, one per phrasing Graphiti
extracted, and writing per edge would emit the identical fact several times
in one cycle.

The write is gated on a SUCCESSFUL invalidation, and scoped to ids the
BLOCKED rule contributed. Gating on success bounds the write to exactly ONE
per contradicted task over the fact's whole lifetime: an invalidated edge
drops out of the valid-edge read, so it cannot re-present next cycle. Writing
on FAILURE would repeat the same fact every cycle for as long as the
invalidation kept failing, and meanwhile leave the graph asserting both that
the task is blocked (the edge is still valid) and that it is not.

The per-task dedupe marker means WRITTEN, not ATTEMPTED. Those coincide
except when a write fails, and there the distinction is the whole point. The
sibling edges of the same task are the only retry this fact can ever get,
since an invalidated edge does not re-enumerate next cycle, so a transient
failure must not consume the task's one-write-per-cycle budget.

### The per-cycle ceiling

`stale_status_snapshot_edge_sweep.py::_MAX_SUPERSEDE_WRITES_PER_CYCLE` caps
superseding writes per cycle (task 3037). Each write is a real Graphiti write
plus an embedding round-trip, awaited SERIALLY inside the per-stale-edge
invalidation loop. Its count is bounded only by how many distinct
blocked-asserted tasks are contradicted in the cycle. On the FIRST cycle
after it shipped, every blocked assertion accumulated since the sweep was
written became eligible at once. An unbounded number of sequential network
writes would blow the cycle's liveness budget (§"Extraction cost as a
liveness property") by minutes on the drain cycle.

The ceiling is sized to keep the worst-case added latency in the same order
as the enumeration cost already budgeted for, while sitting far above steady
state: once the backlog has drained, a cycle supersedes roughly one fact
per task unblocked since the previous cycle, single digits.

NO SILENT CAP: truncation is tallied into `stats['supersede_skipped']` and
logged at WARNING with the count. And it IS a loss, not a deferral: the edges
are still invalidated, so they do not re-enumerate next cycle, and the
skipped facts are never written. That is the same trade the sweep already
takes on a supersede FAILURE. The correctness-bearing half of the step,
retiring the contradicted assertion, always completes; only the record of
what replaced it is best-effort.

The ceiling counts ATTEMPTS, not successes, so a cycle whose writes all fail
cannot spin through an unbounded number of them via the in-cycle retry
either.

## Sweep execution

### Best-effort

The sweep mirrors `degenerate_task_node_sweep.sweep_degenerate_task_nodes`.
A transient backend error enumerating, cross-referencing (`get_statuses`) or
invalidating (`update_edge`) is caught, logged and tallied into
`stats['errors']`. An enumeration or cross-reference failure aborts the rest
of the cycle's sweep, since there is nothing left to act on, and returns the
stats gathered so far. A per-edge `update_edge` failure does NOT abort the
loop.

A superseding-write failure is tallied into `stats['supersede_errors']`
instead, and leaves the loop running. It degrades only OBSERVABILITY: by the
time it runs the edge is already invalidated, so the graph is consistent,
just without the fact recording what replaced the retired assertion. The
missing fact is not retried next cycle, because the invalidated edge no
longer enumerates. So it is a permanently missing superseding fact, not a
corrupted edge or a repeating failure. Its one retry is in-cycle; see
§"The superseding write".

`errors` stays scoped to the enumerate, cross-reference and invalidate paths.
That keeps the identity `invalidated == candidate_edges - errors` exactly
true: every selected edge is either invalidated or tallied there, by the ONE
counted invalidation loop. The identity is the structural guard against a
future second invalidation site going uncounted, and it is pinned by a test.
So a non-invalidation failure mode must never be folded into `errors`; the
superseding write has its own key for that reason (task 3037). The two read
keys (`enumeration_complete`, `enumeration_incomplete_kind`) describe the
read, not the counted funnel, so they leave the identity intact too. An
EMPIRICAL incompleteness does not touch `errors`: the sweep proceeds on what
it fetched, and the cycle merely says the corpus was partial.

### Read completeness

`enumeration_complete` is TRI-STATE because three outcomes are genuinely
distinct (task 4386). The only predicate a caller may gate on is `is True`;
`is not False` would admit the unknown cases and let a cycle that never
looked pass as one that looked and found everything. The stats are seeded
UNKNOWN, not True, so every early return reports "no corpus observed" rather
than a fabricated clean read. `enumeration_incomplete_kind` surfaces the
backend's stable `INCOMPLETE_*` discriminator, never `PagedRead.reason`,
whose wording the backend documents as unstable diagnostic prose that
consumers would otherwise parse.

The `enumerate_*` reads never raise; they report incompleteness as a value.
So the fail-closed structural guard (`apply_incompleteness_policy`) is
applied in `_read_corpus`, or a page-capped read would be taken for the whole
corpus and every edge the missing pages carry scanned as absent: a silently
clean cycle that retires nothing. It sits inside the same `try`, so a
structural incompleteness lands in the one handler. An EMPIRICAL
incompleteness only warns, through the sweep's injected logger so it surfaces
with the rest of the cycle's diagnostics. Completeness is reported even when
the policy aborts the read: a cycle that reported only errors=1 would leave
the operator reconstructing the cause from logs, the exact reconstruction
the signal exists to remove.

The Stage-1 stat keys (`STATUS_SNAPSHOT_ENUMERATION_*_STAT_KEY`) are exported
symbols, not inline literals at producer, consumer and test, matching the
`task_count_snapshot_cadence.SNAPSHOT_PRUNE_ENUMERATION_OK_STAT_KEY`
precedent: otherwise a rename has to be found by grep across five files,
which is how a stat key ends up half-renamed and a consumer silently reads
`None`. They are spelled as whole literals, not built from a prefix, so the
shipped key stays greppable from a log line (task 4386).

### The per-task sweep

`sweep_stale_status_snapshot_edges_for_task` (task 4851) is a LATENCY
complement to the periodic sweep, not a coverage one. It reads
`enumerate_valid_edges_mentioning(str(task_id))`, a digit-substring SUPERSET,
and keeps only the edges whose extracted ids contain the task. Both sweeps
share one core, so the one-pass extraction discipline holds for it too.

Its cost depends on the id. A multi-digit id reads a handful of rows; a one-
or two-digit id, or one that also occurs in dates, reads a sizeable fraction
of the corpus (row counts in `plans/falkordb-resultset-cap-audit.md`). That
is accepted because it is bounded by the periodic sweep's full read, runs off
any request path, and never overlaps another sweep of the project, so a
cascade of transitions queues rather than stacks.

### The per-project lock

One superseding fact per contradicted task holds only if each sweep READS
after the previous one's retirements: a retired edge stops enumerating, and
that is the whole dedup. Overlapping sweeps of one project would both read
the same still-valid edge, both retire it and both write its facts. A
cancellation cascade starts one targeted sweep per task without awaiting
any, and a full cycle's Stage-1 sweep can start beside them.

So every sweep holds `_sweep_lock_for(project_id)`. It is keyed by the
canonical project id, as the graph itself is, and by event loop, because an
`asyncio.Lock` binds to the first loop that waits on it. It serializes within
one process only, which is where both callers run. (task 4851)

## Known residuals

Deliberate, and all fail-safe under-selection unless noted.

- task 3042: a blocked-status assertion using neither a closed-class
  connective nor the literal 'status' noun ("Task N remains parked awaiting
  adjudication") is not extracted.
- task 3042: the phrase form's status-noun requirement narrows but does not
  eliminate binding ambiguity; see precision residual (3) in
  §"Status-phrase form". That one is over-selection.
- task 3079: the genitive path requires the literal noun 'status', so "Task
  N's state is pending" is not extracted.
- task 3079: the plural preposition guard has one residual pointed the WRONG
  way (an unlisted preposition over-selects) and one fail-safe residual (the
  adverbial preamble); see §"Plural-enumeration subjecthood guard". The
  fail-safe one is MEASURED at zero cost on the live corpus (task 3949).
- task 3037: the blocked family drops any aggregate list whose introducer
  follows a task reference in the same clause; see §"The blocked family's
  list guard".
- tasks 4149 / 4851: a genuine '.', ';', '!' or '?' sentence break
  immediately followed by an alphanumeric ('done.Then') reads as
  token-internal; see §"Token-internal occurrences".
- task 4851, measured: four lexical classes of genuine snapshot sit outside
  the closed-class copula/article anchoring and are not extracted: a relative
  clause ('task N, which is pending'), an attributive pre-modifier or
  predicate nominal ('Blocked task N ...'), a gerund ('due to being
  pending'), and a secondary or coordinated predicate ('is filed and
  pending'). Each would need its own precision pass before widening. Triage
  and samples: `plans/stale-status-snapshot-sweep-report.md`.

## Ruled-out hypotheses

Two hypotheses were investigated and RULED OUT for the task-2613 miss rate.
Do not re-open them.

1. **Edge traversal or scope.** `GraphitiBackend.get_all_valid_edges` runs an
   unfiltered `MATCH` (no entity-type, label or episode filter). Reproduced
   end-to-end, scanning every missed edge and dropping it at extraction:
   every gap was lexical, not a query-scope gap.

   AMENDED, task 4340 (2026-08-17). The lexical findings STAND; what does not
   stand is the scope of the ruling-out. Those grounds covered query FILTERS
   only and were silent about server-side TRUNCATION. Task 4340 measured a
   FalkorDB `RESULTSET_SIZE` cap silently truncating that very query to
   roughly HALF the valid-edge corpus on dark_factory (exact figures in
   `plans/falkordb-resultset-cap-audit.md`, the one place they are
   recorded). So at the time of the task-2613 investigation the sweep saw
   about half the edges, and the miss rate was computed against a truncated
   denominator. The read is paginated as of task 4340 and the truncation is
   gone.

   RE-MEASURED, task 4851, against the complete corpus. The metric's
   definition and every figure live in
   `plans/stale-status-snapshot-sweep-report.md` and nowhere else. The
   durable finding: at steady state the shipped rules select approximately
   nothing, because the sweep keeps up. The facts that still name a
   now-terminal task are an UPPER bound on misses, and by-eye triage finds
   them mostly NOT status snapshots at all. The genuine misses are the
   fail-safe lexical classes under §"Known residuals". Do not re-open the
   LEXICAL hypothesis; do not treat the task-2613 rate as measured on a whole
   corpus.
2. **A shared blind spot with `task_count_verification`.** That function is
   an aggregate census-vs-tree consistency check (`task_filter`), not a
   per-task edge sweep at all. Its 'healthy' report was correct, not blind.

## Extraction cost as a liveness property

The sweep calls `extract_snapshot_edge_task_ids_by_marker_class` once per
valid edge, from an unguarded comprehension with no per-edge timeout, over
the whole group's edge set (tens of thousands of edges; current figure in
`plans/stale-status-snapshot-sweep-report.md`). Extraction cost is therefore
a whole-cycle LIVENESS property: one pathological fact stalls the entire
reconciliation cycle. It is not a micro-optimisation, and it is why the
extraction regexes carry performance tests (task 3079).

The corpus keeps growing, and the task-4340 pagination removed a truncation
that had hidden about half of it, so the argument only gets STRONGER over
time. The read that feeds it is bounded. Its cost per full enumeration, and
what it adds to each cycle over the old truncated read, are in the MEASURED
COST section of `plans/falkordb-resultset-cap-audit.md`. (task 4340)

Adding the second (blocked) pattern family did NOT add a second per-edge pass
(task 3037). The per-edge extraction is exactly ONE entry into the pipeline:
`extract_snapshot_edge_task_ids_by_marker_class` evaluates the whole-fact gate
once and returns both id sets, and the sweep's comprehension is its only
caller. That is pinned by
`tests/reconciliation/test_stale_status_snapshot_edge_sweep.py::TestSweepStaleStatusSnapshotEdgesBlockedRuleAndCounters::test_extraction_runs_exactly_once_per_edge`,
which counts pipeline entries
rather than wall-clock (`CountingStr.reads`). The function takes
`str | None` because it is where a missing fact is normalised: callers read
it straight off an untyped edge dict, and an `or ''` at each call site would
cost an extra truthiness test per edge, which that probe would read as a
second pass.

What keeps the added family's cost off the hot path is the cheap literal
`_BLOCKED_MARKER_RE` pre-gate. The overwhelming majority of edges carry no
'blocked' token, so they pay one extra substring scan and never run the
blocked family's five anchored patterns. Only edges that really mention
'blocked' pay for the second family, and for those the work is irreducible,
since the two families answer different questions.

The clause scans of both guards use a reverse `rfind` plus
`search(..., pos)` rather than slicing, so they add no allocation to this
hot path (task 3079).
