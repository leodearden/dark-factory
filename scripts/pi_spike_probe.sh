#!/usr/bin/env bash
# pi backend empirical spike — throwaway reproducer (task 2458 / T2 of
# plans/harness-backend-reconnect-pi-prd.md). Produces the observed evidence behind
# plans/pi-spike-findings.md. NOT part of the product; safe to delete.
#
# It installs the pinned pi packages into a scratch dir (never the repo — node_modules/
# is gitignored) and runs real headless pi invocations, printing the outputs the findings
# doc cites. Requires: node>=22, npm, and a provider key (GEMINI_API_KEY is easiest; the
# repo .env has one). Usage:
#     GEMINI_API_KEY=... scripts/pi_spike_probe.sh [workdir]
# or, from the dark-factory checkout with keys in .env:
#     set -a; source .env; set +a; scripts/pi_spike_probe.sh
set -euo pipefail

PI_AGENT_VERSION="0.80.6"      # @earendil-works/pi-coding-agent (PRD pin 0.80.x)
PI_MCP_ADAPTER_VERSION="2.11.0" # pi-mcp-adapter (PRD pin 2.11.x)
PROVIDER="${PI_SPIKE_PROVIDER:-google}"
MODEL="${PI_SPIKE_MODEL:-gemini-3-flash}"

WORK="${1:-$(mktemp -d /tmp/pi-spike.XXXXXX)}"
mkdir -p "$WORK"; cd "$WORK"
echo "== workdir: $WORK  provider=$PROVIDER model=$MODEL =="

# ---------------------------------------------------------------------------
# Install (throwaway).
# ---------------------------------------------------------------------------
[ -f package.json ] || echo '{"name":"pi-spike","private":true,"version":"1.0.0"}' > package.json
npm install --no-fund --no-audit \
  "@earendil-works/pi-coding-agent@${PI_AGENT_VERSION}" \
  "pi-mcp-adapter@${PI_MCP_ADAPTER_VERSION}" >/dev/null 2>&1
PI="$WORK/node_modules/.bin/pi"
echo "pi version: $("$PI" --version)"

pi_run() { # args...  -> stdout to $1.json, stderr to $1.err, prints exit code
  local tag="$1"; shift
  local rc=0
  timeout 150 "$PI" --provider "$PROVIDER" --model "$MODEL" --mode json -p "$@" \
    </dev/null >"$tag.json" 2>"$tag.err" || rc=$?
  echo "  [$tag] exit=$rc  stdout_bytes=$(wc -c <"$tag.json")"
}

# ---------------------------------------------------------------------------
# Q1 — exit-code semantics: success vs runtime-API-error vs CLI-usage-error.
# ---------------------------------------------------------------------------
echo "== Q1 exit codes =="
rm -rf s_ok; mkdir -p s_ok
pi_run q1_ok --session-dir s_ok --no-tools "Reply with exactly the word: PONG"
python3 - q1_ok.json <<'PY'
import json,sys
term=None
for l in open(sys.argv[1]):
    l=l.strip()
    if not l: continue
    o=json.loads(l)
    if o.get('type') in ('turn_end','agent_end'):
        term=o.get('message') or (o.get('messages') or [{}])[-1]
print("  success run -> terminal stopReason:", (term or {}).get('stopReason'))
PY
# runtime API error (bad model id) still exits 0 with stopReason:error in-stream:
timeout 60 "$PI" --provider "$PROVIDER" --model no-such-model-xyz --mode json -p --no-tools "hi" \
  </dev/null >q1_apierr.json 2>q1_apierr.err && echo "  api-error run exit=0 (as expected)"
grep -o '"stopReason":"error"' q1_apierr.json | head -1 | sed 's/^/  api-error in-stream: /'
# CLI usage error -> exit 1, stderr, no JSON:
set +e; "$PI" -p --this-is-not-a-flag "hi" </dev/null >q1_usage.json 2>q1_usage.err; echo "  usage-error exit=$?"; set -e
sed 's/^/  usage-error stderr: /' q1_usage.err | head -1

# ---------------------------------------------------------------------------
# Q2 — cost aggregation: per-message usage.cost.total (USD), NOT cumulative.
#      (a multi-turn/tool run shows the per-turn split summing to the total.)
# ---------------------------------------------------------------------------
echo "== Q2 cost (per-turn, summed) =="
mkdir -p toolwork; echo "the secret token is SPIKE-42" > toolwork/secret.txt
rm -rf s_cost; mkdir -p s_cost
pi_run q2 --session-dir s_cost \
  "Use your tools to read toolwork/secret.txt and tell me the secret token."
python3 - q2.json <<'PY'
import json,sys
turns=[]
for l in open(sys.argv[1]):
    l=l.strip()
    if not l: continue
    o=json.loads(l)
    if o.get('type')=='turn_end':
        u=o['message'].get('usage',{}); turns.append((u.get('totalTokens'), u.get('cost',{}).get('total')))
print("  per-turn (tokens,cost):", turns)
print("  SUM cost.total:", sum(c for _,c in turns if c))
PY

# ---------------------------------------------------------------------------
# Q3 — direct-tools generated name = <serverKey '-'->'_'>_<toolName>.
#      Stand up a stdio MCP server keyed "spike-demo" with tool "echo_it";
#      expected direct name: spike_demo_echo_it.
# ---------------------------------------------------------------------------
echo "== Q3 direct-tools name =="
cat > demo_mcp_server.mjs <<'JS'
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { ListToolsRequestSchema, CallToolRequestSchema } from "@modelcontextprotocol/sdk/types.js";
const server = new Server({ name: "spike-demo-server", version: "0.0.1" }, { capabilities: { tools: {} } });
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [{ name: "echo_it", description: "Echo back text (spike probe).",
    inputSchema: { type: "object", properties: { text: { type: "string" } }, required: ["text"] } }] }));
server.setRequestHandler(CallToolRequestSchema, async (req) =>
  ({ content: [{ type: "text", text: `ECHO:${req.params.arguments?.text ?? ""}` }] }));
await server.connect(new StdioServerTransport());
JS
rm -rf piwork; mkdir -p piwork
cat > piwork/.mcp.json <<EOF
{ "mcpServers": { "spike-demo": {
    "command": "node", "args": ["$WORK/demo_mcp_server.mjs"], "directTools": true } } }
EOF
( cd piwork && "$PI" install -l --approve "$WORK/node_modules/pi-mcp-adapter" >/dev/null 2>&1 )
# Warm the direct-tools metadata cache. `init` alone only does config discovery/import; the
# tool-metadata cache (~/.pi/agent/mcp-cache.json) is populated when the adapter first CONNECTS.
# So trigger one proxy search (connects to spike-demo + caches), THEN the direct-tools call works.
node "$WORK/node_modules/pi-mcp-adapter/cli.js" init >/dev/null 2>&1 || true
( cd piwork && timeout 150 "$PI" --provider "$PROVIDER" --model "$MODEL" --mode json -p --approve \
    --session-dir sWarm "Call mcp with search=echo to list matching tools, then stop." \
    </dev/null >q3_warm.json 2>q3_warm.err || true )
( cd piwork && timeout 150 "$PI" --provider "$PROVIDER" --model "$MODEL" --mode json -p --approve \
    --tools spike_demo_echo_it --session-dir sD \
    "Call spike_demo_echo_it with text=direct-mode-test and report the result." \
    </dev/null >q3.json 2>q3.err || true )
grep -o '"toolName": *"spike_demo_echo_it"' piwork/q3.json | head -1 \
  | sed 's/^/  observed direct tool name: /' || echo "  (fell back to proxy mcp — cache not warmed)"

# ---------------------------------------------------------------------------
# Q6 — bash-tool children inherit pi's env/namespace (=> bwrap wraps the tree).
# ---------------------------------------------------------------------------
echo "== Q6 child env inheritance =="
rm -rf s_env; mkdir -p s_env
SPIKE_MARKER="pi-child-inherit-OK-7723" timeout 150 "$PI" --provider "$PROVIDER" --model "$MODEL" \
  --mode json -p --approve --tools bash --session-dir s_env \
  'Run bash: printf "MARKER=%s\n" "$SPIKE_MARKER" ; report its stdout.' \
  </dev/null >q6.json 2>q6.err || true
grep -o 'MARKER=pi-child-inherit-OK-7723' q6.json | head -1 | sed 's/^/  bash child saw: /' \
  || echo "  (marker not found)"

# ---------------------------------------------------------------------------
# Q4/Q7 — no decode-forced structured output, no native turn/budget cap.
# ---------------------------------------------------------------------------
echo "== Q4/Q7 absent flags (help + dist grep) =="
"$PI" --help 2>&1 | grep -iE 'max-turns|max-steps|max-budget|output-schema|json-schema' \
  && echo "  ^ FOUND (unexpected)" || echo "  no --max-turns/--max-budget/--output-schema in --help (as expected)"

echo "== done. Findings written up in plans/pi-spike-findings.md. workdir=$WORK =="
