/* Escalation Analytics tab — Origin / Lifespan / Workflow lifecycle panels.
 *
 * No JS test runner in this project (see scheduler_drawer.jsx comment).
 * Wiring contracts are verified via Python source-assertion tests in
 * dashboard/tests/test_tab_escalation_analytics.py.
 *
 * Load order: tabs.jsx → tab_escalation_analytics.jsx → app.jsx
 * Export:     window.DF_TABS.EscalationAnalyticsTab  (additive mutation of
 *             the object created by tabs.jsx; app.jsx destructures it last)
 */
const { useState: uS, useEffect: uE, useMemo } = React;
const DF = window.DF_DATA;
const { ProjectGroup, Segmented, fmtUptime, taskId } = window.DF_SHELL;
const C = window.DF_CHARTS;

// ── Local helpers (tab_escalations.jsx-compatible copies; not exported from
//    any namespace) ──

function useOpenSet(ids, defaultOpen = true, storageKey = null) {
  const [openMap, setOpenMap] = uS(() => {
    let stored = {};
    if (storageKey) {
      try { stored = JSON.parse(localStorage.getItem(storageKey) || '{}') || {}; } catch (e) {}
    }
    const init = {};
    for (const id of ids) init[id] = id in stored ? !!stored[id] : defaultOpen;
    return init;
  });
  // Backfill ids that arrive after mount (per_project starts [] and is
  // populated by the first poll, so groups would otherwise render collapsed).
  const idsKey = ids.join('\0');
  uE(() => {
    setOpenMap(m => {
      let patch = null;
      for (const id of ids) {
        if (!(id in m)) { if (!patch) patch = {}; patch[id] = defaultOpen; }
      }
      return patch ? { ...m, ...patch } : m;
    });
  }, [idsKey]); // eslint-disable-line react-hooks/exhaustive-deps
  uE(() => {
    if (storageKey) {
      try { localStorage.setItem(storageKey, JSON.stringify(openMap)); } catch (e) {}
    }
  }, [storageKey, openMap]);
  const toggle = id => setOpenMap(m => ({ ...m, [id]: !m[id] }));
  const setAll = v => setOpenMap(Object.fromEntries(ids.map(id => [id, v])));
  return [openMap, toggle, setAll];
}

function usePersistedState(storageKey, defaultValue) {
  const [v, setV] = uS(() => {
    try {
      const raw = localStorage.getItem(storageKey);
      return raw === null ? defaultValue : JSON.parse(raw);
    } catch (e) { return defaultValue; }
  });
  uE(() => { try { localStorage.setItem(storageKey, JSON.stringify(v)); } catch (e) {} }, [storageKey, v]);
  return [v, setV];
}

// ── EscalationAnalyticsTab ──

function EscalationAnalyticsTab({ projectFilter }) {
  const analytics = DF.ESCALATION_ANALYTICS || { generated_at: null, parse_failures: 0, regime_markers: [], per_project: [] };

  const projects = (analytics.per_project || []).filter(p => {
    if (!projectFilter || projectFilter.length === 0) return true;
    return projectFilter.includes(p.project);
  });
  const projectIds = projects.map(p => p.project);

  const [openMap, toggle] = useOpenSet(projectIds, true, 'df.open.escanalytics');

  return (
    <div style={{ position: 'relative' }}>
      {projects.map(p => (
        <div key={p.project} style={{ marginBottom: 8 }}>
          <ProjectGroup
            id={p.project}
            label={p.project}
            open={!!openMap[p.project]}
            onToggle={() => toggle(p.project)}
          >
            {/* Origin / Lifespan / Workflow panels land in later steps */}
          </ProjectGroup>
        </div>
      ))}
    </div>
  );
}

// ── Additive export (mutates the object created by tabs.jsx) ──
window.DF_TABS.EscalationAnalyticsTab = EscalationAnalyticsTab;
