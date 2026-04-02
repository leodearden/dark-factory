/**
 * Shared Chart.js utilities for the Dark Factory dashboard.
 *
 * Provides:
 *  - CHART_PALETTE   — 10-color array reused across every chart partial
 *  - getOrDestroyChart(canvasId) — safely retrieve a canvas, destroying any
 *    prior Chart.js instance to avoid stale-chart buildup during HTMX morphs
 */

/* exported CHART_PALETTE, getOrDestroyChart */

var CHART_PALETTE = [
  '#60a5fa', '#34d399', '#fbbf24', '#f87171', '#a78bfa',
  '#fb923c', '#38bdf8', '#4ade80', '#e879f9', '#94a3b8'
];

/**
 * Look up a <canvas> by ID, destroy any existing Chart.js instance attached to
 * it, and return the bare element ready for a fresh chart.
 *
 * Returns null when the element is not in the DOM (guard for conditional
 * Jinja blocks that omit the canvas when data is empty).
 */
function getOrDestroyChart(canvasId) {
  var el = document.getElementById(canvasId);
  if (!el) return null;
  var existing = Chart.getChart(el);
  if (existing) existing.destroy();
  return el;
}
