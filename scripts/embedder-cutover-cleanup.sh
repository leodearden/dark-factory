#!/usr/bin/env bash
# Fail-loud placeholder for the θ cleanup action of plans/local-embedder-cutover-prd.md.
# θ is filed as a deterministic task whose before_done runs this script with
# `--profile live`; submit_task requires the script to exist and be executable at
# filing time. δ replaces this file with the real implementation, which delegates
# to its `reembed` cleanup subcommand. θ depends on λ (and transitively on δ), so
# this placeholder runs only if the dependency order was violated.
set -euo pipefail
echo "EMBEDDER-CUTOVER-CLEANUP-STUB: scripts/embedder-cutover-cleanup.sh is still the placeholder committed at decompose; δ of plans/local-embedder-cutover-prd.md has not replaced it. Refusing to touch the overlay, the drop-in or the backups (args: $*)." >&2
exit 1
