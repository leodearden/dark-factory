// prd_grouping.js — pure PRD-grouping-view logic for the Tasks tab's
// per-project "group by PRD" view (tab_tasks.jsx).
//
// This is a plain-JS module: no JSX, no Babel. It is loaded two ways:
//   - In the browser, via a classic `<script src="/static/redux/prd_grouping.js">`
//     tag (like graph_layout.js), which assigns `window.DF_PRD_GROUPING`.
//   - In node (no package.json in this repo, so this file resolves as
//     CommonJS), via `require`/`import` for the `node --test` suite under
//     dashboard/tests/js/.
//
// Both export paths are guarded so this file has no effect outside the
// environment it's actually running in.
//
// index.html loads this file (classic script, before the Babel JSX tags) so
// `window.DF_PRD_GROUPING` is defined before tab_tasks.jsx executes its
// top-level destructure of it.
//
// orderPrdGroups takes graph_layout.js's computeTiers as an INJECTED
// parameter rather than importing/reading window.DF_GRAPH_LAYOUT internally
// — see the plan's design decisions for why (keeps this module independently
// unit-testable and avoids a hard cross-module load-order/require coupling).

// ── Derive a PRD box title from its path/ref ──
// basename (the substring after the last '/'), with a trailing '-prd.md'
// stripped, else a trailing '.md' stripped, else returned as-is.
function prdTitle(prdPath) {
  if (!prdPath) return prdPath;
  const base = prdPath.includes('/') ? prdPath.slice(prdPath.lastIndexOf('/') + 1) : prdPath;
  if (base.endsWith('-prd.md')) return base.slice(0, base.length - '-prd.md'.length);
  if (base.endsWith('.md')) return base.slice(0, base.length - '.md'.length);
  return base;
}

const API = { prdTitle };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = API;
}
if (typeof window !== 'undefined') {
  window.DF_PRD_GROUPING = API;
}
