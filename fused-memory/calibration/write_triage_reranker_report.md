# Write-triage reranker arms (rho1)

Rendered from the JSON report beside this file by `scripts/eval_write_triage_reranker.py`.

**Best:** bge-reranker-v2-m3 — rank-1 0.381, p95 0.468 s, qualified: yes (p95 ceiling 3.0 s)

| arm | class | status | skip reason | rank-1 | rank-5 | AUC | p50 s | p95 s | cost/write USD | device | VRAM MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cosine | baseline | measured | n/a | 0.190 (16/84) | 0.381 (32/84) | 0.460 | n/a | n/a | n/a | n/a | n/a |
| qwen3-reranker-0.6b | local_cross_encoder | measured | n/a | 0.202 (17/84) | 0.571 (48/84) | 0.241 | 0.694 | 1.998 | 0.00000 | cuda:0 (NVIDIA GeForce RTX 3090) | 3568 |
| mxbai-rerank-base-v2 | local_cross_encoder | measured | n/a | 0.238 (20/84) | 0.583 (49/84) | 0.401 | 0.434 | 1.096 | 0.00000 | cuda:0 (NVIDIA GeForce RTX 3090) | 1793 |
| bge-reranker-v2-m3 | local_cross_encoder | measured | n/a | 0.381 (32/84) | 0.631 (53/84) | 0.669 | 0.276 | 0.468 | 0.00000 | cuda:0 (NVIDIA GeForce RTX 3090) | 1183 |
| gpt-4o-mini-pairwise | llm_pairwise | measured | n/a | 0.190 (16/84) | 0.452 (38/84) | 0.405 | 0.711 | 1.441 | 0.00216 | remote:api.openai.com | n/a |
| jina-reranker | hosted_api | skipped | no_credential | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| voyage-rerank | hosted_api | skipped | no_credential | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| cohere-rerank | hosted_api | skipped | no_credential | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| jev-choice | jev_choice | skipped | no_credential | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Caveats

- Local cross-encoder arms score with each model's published template and default instruction; the LLM pairwise arm uses this script's own same-claim prompt, so the two classes are not prompted alike.
- The cosine baseline orders each slate by store_score, production's attach signal. It is reported for comparison and is never eligible for best.
- AUC is rank-based (Mann-Whitney, ties count half) over each case's canonical-or-alias pair score as the arm scored it on the retrieved slate. 9 hard negative(s) reached their slate; read the value beside that sample size.
- Latency is per-slate wall time on the recorded device after one untimed warm-up slate; remote arms include the network round trip.
- A local arm's cost is the per-call charge (zero), not amortised hardware or power.
- pairs_over_max_length is a lower bound on truncation: it counts entry plus candidate tokens without the template or special tokens. qwen3-reranker-0.6b at 8192 tokens: 0 pair(s) over; mxbai-rerank-base-v2 at 8192 tokens: 0 pair(s) over; bge-reranker-v2-m3 at 1024 tokens: 336 pair(s) over.
- jina-reranker was skipped (no_credential): JINA_API_KEY unset
- voyage-rerank was skipped (no_credential): VOYAGE_API_KEY unset
- cohere-rerank was skipped (no_credential): COHERE_API_KEY / CO_API_KEY unset
- jev-choice was skipped (no_credential): TYPESAFE_API_KEY unset

## Provenance

- `fixture_path`: "tests/fixtures/write_triage_calibration.jsonl"
- `record_count`: 104
- `canonical_aliases_path`: "tests/fixtures/write_triage_calibration.canonical_aliases.json"
- `canonical_aliases_count`: 3
- `project_id`: "reify"
- `candidate_k`: 20
- `retrieval_call`: "fused_memory.server.write_triage::retrieve_candidates"
- `limit`: null
- `libraries`: {"torch": "2.14.0", "sentence-transformers": "6.1.0", "transformers": "5.17.0"}
- `case_count`: 84
- `canonical_absent`: 21
- `degraded_retrievals`: 0
- `self_retrieved`: 0
- `p95_ceiling_seconds`: 3.0
- `max_arm_spend_usd`: 2.0
- `local_batch_size`: 4
- `vram_cap_gib`: 8.0
- `pairwise_concurrency`: 20
