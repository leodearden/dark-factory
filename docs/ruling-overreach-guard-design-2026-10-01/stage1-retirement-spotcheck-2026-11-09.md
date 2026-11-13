# Stage-1-invalidated flagged edges: blind spot-check (gate 6218, 2026-11-09)

Sample: 10 dark_factory + 10 reify edges, drawn at random (seed 6218) from
`flag_handoffs` edges that now carry `invalid_at` without `expired_at`
(the fused-memory `update_edge` shape, i.e. recon Stage 1). Adjudicated under
design.md §2.6 against the episode text, without reading the judge's rationale.
The table names each edge by its uuid prefix; the episode bodies are not committed (as §2.9).

| # | graph | edge | my label | note |
|---|---|---|---|---|
| 0 | df | 4199d608 | overreach (ii/iii) | import direction from a deferred, not-adopted partition |
| 1 | df | c1b63ac5 | overreach (ii) | inverted: record says the literal check cannot hold |
| 2 | df | f61c3ce1 | overreach (ii) | γ2 observed via _RealLocal; run_local "cannot be wrapped" |
| 3 | df | 37f17d30 | overreach (iii) | rejected option (wrap in AtomicConnection) stated as fact |
| 4 | df | 54862206 | overreach (ii), borderline | measurement_params() is the instrument's builder, not "from the wrapper" |
| 5 | df | 4c20c923 | overreach (ii) | record: no navigable link, no cross-tab navigation |
| 6 | df | 1ad88a77 | overreach (iii) | rejected partition's "transport module" stated as present |
| 7 | df | 064b83aa | holding | faithful to the decision + its code-quality.md ground |
| 8 | df | ec422e0c | overreach (ii) | garble: "takes priority over" for "takes priority, desc_cap, details_cap keywords" |
| 9 | df | 01f1ef75 | overreach (i) | adds "task record as well"; decision is commit message only |
| 10 | reify | db593c72 | overreach (ii) | misattributes cause to main 3e8f753316 (that is where the fix landed) |
| 11 | reify | 5a4ee584 | overreach (ii) | inverted import direction |
| 12 | reify | cbcca21e | overreach (ii) | garble: above-baseline is red, below is STALE |
| 13 | reify | ef195e93 | overreach (ii), borderline | "when compared to Geometry" invented from a list adjacency |
| 14 | reify | 243dc1c2 | borderline (context?) | a re-export is implied to exist today; the decision removes the reliance |
| 15 | reify | 8f92e0ba | overreach (ii) | self-contradictory garble (finite vs all-NaN) |
| 16 | reify | f6f6bbdc | overreach (ii) | "excluding gui/" contradicts the decision (gui/ is NOT excluded) |
| 17 | reify | 072452ea | overreach (ii) | record says the CLI seam is UNPINNED |
| 18 | reify | 520f9818 | overreach (ii) | the nine edges were wired by the ruling, not in #9571's scope |
| 19 | reify | 54fe51e7 | overreach (ii) | assemble_hex_p1_stiffness lives in analytical_validation |

Tally: 18 overreach (2 borderline), 1 holding, 1 borderline. Stage 1's
retirement precision on flagged edges ≈ 0.90 (18/20; Wilson ≈ [0.70–0.97]).
Single adjudicator, n=20: a spot-check, not a pre-registered measurement.
