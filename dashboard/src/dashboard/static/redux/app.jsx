/* Main app — routes tabs, manages filter state, hosts tweaks */
const { useState: uS, useEffect: uE } = React;
const { Rail, StatStrip, Toolbar } = window.DF_SHELL;
const { OverviewTab } = window.DF_OVERVIEW;
const { OrchTab, PerfTab, MemoryTab, ReconTab, MergeTab, CostsTab, BurnTab } = window.DF_TABS;
const { TasksTab } = window.DF_TASKS;
const DD = window.DF_DATA;

// Tweaks helpers are attached directly to window
const { TweaksPanel, useTweaks, TweakSection, TweakSlider, TweakToggle, TweakRadio, TweakSelect, TweakColor } = window;

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "accent": "#4DA3FF",
  "density": "comfortable",
  "showLiveFeed": true,
  "navStyle": "rail",
  "kpiStyle": "spark",
  "monoNumerics": true,
  "pauseLive": false,
  "tableZebra": false,
  "showHints": true
}/*EDITMODE-END*/;

function App() {
  const [tab, setTab] = uS('overview');
  const [tw, setTw] = useTweaks ? useTweaks(TWEAK_DEFAULTS) : [TWEAK_DEFAULTS, () => {}];

  // Filter state — per tab
  const [win, setWin] = uS('24h');
  const [projects, setProjects] = uS([]);     // [] = all
  const [agents, setAgents] = uS([]);
  const [search, setSearch] = uS('');

  // Apply tweaks: accent
  uE(() => {
    document.documentElement.style.setProperty('--accent', tw.accent || '#4DA3FF');
    document.documentElement.style.setProperty('--accent-bg', `${tw.accent || '#4DA3FF'}40`);
  }, [tw.accent]);

  // Density
  uE(() => {
    document.documentElement.style.setProperty('--pad', tw.density === 'compact' ? '8px' : tw.density === 'roomy' ? '16px' : '12px');
  }, [tw.density]);

  // Last-update tick
  const [now, setNow] = uS(new Date());
  uE(() => {
    if (tw.pauseLive) return;
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, [tw.pauseLive]);

  // Re-render when the data loader refreshes window.DF_DATA.
  const [, setDataTick] = uS(0);
  uE(() => {
    const onRefresh = () => setDataTick(n => n + 1);
    window.addEventListener('df-data-refresh', onRefresh);
    return () => window.removeEventListener('df-data-refresh', onRefresh);
  }, []);

  // Mirror tweaks-panel pauseLive into the data loader so it stops polling.
  uE(() => {
    window.__DF_PAUSE = !!tw.pauseLive;
  }, [tw.pauseLive]);

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'orch',     label: 'Orchestrators' },
    { id: 'tasks',    label: 'Tasks' },
    { id: 'perf',     label: 'Performance' },
    { id: 'memory',   label: 'Memory' },
    { id: 'recon',    label: 'Reconciliation' },
    { id: 'merge',    label: 'Merge Queue' },
    { id: 'cost',     label: 'Costs' },
    { id: 'burn',     label: 'Burndown' },
  ];
  const tabLabel = tabs.find(t => t.id === tab)?.label || 'Overview';

  // Topbar status summary — all derived from real data.  `spend24h` is the
  // current-day total from COSTS.summary.today (server-computed from the
  // cost trend tail).  Falls back to 0 if cost data hasn't loaded yet.
  const summary = {
    orchRunning: DD.ORCHESTRATORS.filter(o => o.running).length,
    orchTotal: DD.ORCHESTRATORS.length,
    tasksActive: DD.ORCHESTRATORS.reduce((s, o) => s + o.summary.in_progress + o.summary.blocked, 0),
    queue: DD.MEMORY_STATUS.queue.counts.pending,
    spend24h: DD.COSTS?.summary?.today ?? 0,
  };

  const railCounts = {
    orch: summary.orchRunning,
    tasks: DD.ACTIVE_TASKS.filter(t => t.status === 'in-progress' || t.status === 'blocked' || t.status === 'pending').length,
    recon: DD.RECON_STATE.runs.filter(r => r.status === 'failed' || r.status === 'partial').length,
    merge: Object.values(DD.MERGE_QUEUE).reduce((s, d) => s + d.active.length, 0),
  };

  const ts = now.toLocaleTimeString('en-GB', { hour12: false });

  function renderTab() {
    switch (tab) {
      case 'overview': return <OverviewTab paused={tw.pauseLive} />;
      case 'orch':     return <OrchTab projectFilter={projects} search={search} />;
      case 'tasks':    return <TasksTab projectFilter={projects} search={search} />;
      case 'perf':     return <PerfTab projectFilter={projects} />;
      case 'memory':   return <MemoryTab projectFilter={projects} />;
      case 'recon':    return <ReconTab projectFilter={projects} search={search} />;
      case 'merge':    return <MergeTab projectFilter={projects} />;
      case 'cost':     return <CostsTab projectFilter={projects} />;
      case 'burn':     return <BurnTab projectFilter={projects} />;
      default: return null;
    }
  }

  // Per-tab toolbar config
  const toolbarConfig = {
    overview: { showAgents: false, search: false },
    orch:     { showAgents: true,  search: true,  searchPlaceholder: 'Search tasks…' },
    tasks:    { showAgents: false, search: true,  searchPlaceholder: 'Search tasks…' },
    perf:     { showAgents: false, search: false },
    memory:   { showAgents: true,  search: false },
    recon:    { showAgents: false, search: true,  searchPlaceholder: 'Search runs…' },
    merge:    { showAgents: false, search: false },
    cost:     { showAgents: false, search: false },
    burn:     { showAgents: false, search: false },
  }[tab] || {};

  return (
    <div className="app" data-density={tw.density}>
      <Rail active={tab} onSelect={setTab} counts={railCounts} />
      <div className="topbar">
        <div className="crumbs">
          <span style={{ color: 'var(--fg-3)' }}>dark-factory</span>
          <span style={{ color: 'var(--fg-3)' }}>/</span>
          <span className="here">{tabLabel}</span>
        </div>
        <StatStrip live={!tw.pauseLive} lastUpdate={ts} summary={summary} />
      </div>
      <div className="main">
        <Toolbar
          window={win} onWindow={setWin}
          projects={projects} onProjects={setProjects}
          agents={agents} onAgents={setAgents}
          showAgents={toolbarConfig.showAgents}
          search={toolbarConfig.search ? search : undefined}
          onSearch={toolbarConfig.search ? setSearch : undefined}
          searchPlaceholder={toolbarConfig.searchPlaceholder}
          extra={
            <button onClick={() => setTw('pauseLive', !tw.pauseLive)}
              className="multi" style={{ cursor: 'pointer' }}>
              {tw.pauseLive ? '▶ resume' : '❚❚ pause live'}
            </button>
          }
        />
        <div className="body" key={tab}>
          {renderTab()}
        </div>
      </div>

      {TweaksPanel && (
        <TweaksPanel title="Tweaks">
          <TweakSection label="Theme">
            <TweakColor label="Accent color" value={tw.accent} onChange={v => setTw('accent', v)} />
            <TweakRadio label="Density" value={tw.density} options={[
              { value: 'compact', label: 'Compact' },
              { value: 'comfortable', label: 'Comfort' },
              { value: 'roomy', label: 'Roomy' },
            ]} onChange={v => setTw('density', v)} />
            <TweakToggle label="Mono numerics" value={tw.monoNumerics} onChange={v => setTw('monoNumerics', v)} />
            <TweakToggle label="Zebra tables" value={tw.tableZebra} onChange={v => setTw('tableZebra', v)} />
          </TweakSection>
          <TweakSection label="Layout">
            <TweakRadio label="KPI style" value={tw.kpiStyle} options={[
              { value: 'spark', label: 'Sparkline' },
              { value: 'minimal', label: 'Minimal' },
              { value: 'bar', label: 'Bar' },
            ]} onChange={v => setTw('kpiStyle', v)} />
            <TweakToggle label="Show live feed" value={tw.showLiveFeed} onChange={v => setTw('showLiveFeed', v)} />
            <TweakToggle label="Show hints" value={tw.showHints} onChange={v => setTw('showHints', v)} />
          </TweakSection>
          <TweakSection label="Live data">
            <TweakToggle label="Pause stream" value={tw.pauseLive} onChange={v => setTw('pauseLive', v)} />
          </TweakSection>
        </TweaksPanel>
      )}
    </div>
  );
}

// Apply density / tweak side-effects via class
const obs = new MutationObserver(() => {});
const root = document.getElementById('root');
ReactDOM.createRoot(root).render(<App />);
