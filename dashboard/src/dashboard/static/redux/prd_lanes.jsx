// prd_lanes.jsx — the Tasks tab's PRD lanes page (plans/dashboard-prd-lanes-prd.md
// §4.11): one project's served PRD view as lanes of rows, its dark matter and
// footer, and its contention line. Every decision is prd_lanes_view.js's.
//
// A BLOCK'S PROVENANCE IS READ ONCE, BY ITS HEAD PIP. The view and the
// contention line are each one served Datum. The shell Pip at the head of each
// block is that Datum's one reading, so its `≥`, age and non-fresh reason
// appear there and nowhere else. Every figure inside is plain text (ViewFigure)
// titled with the rule the server serves for it, plus any row detail, whatever
// the Datum's state. This is how this page reads PRD §4.10's "one small
// component that takes the parent Datum".
//
// A LOWER LAYER than tab_tasks.jsx: it loads before it and never reads it.
// TasksTab passes its task graph in as renderMembers(rows).

const { Pip: Pip_PL, timeago: timeago_PL } = window.DF_SHELL;
const { PALETTE: CP_PL } = window.DF_CHARTS;
const {
  readyCell,
  barSegments,
  doneOfTotal,
  signalCell,
  contentionEntryText,
  heldShareText,
  neverStartedText,
  darkMatterSummary,
  darkMatterPage,
  membersDrawn,
  inputPip,
  figureTitle,
} = window.DF_PRD_LANES_VIEW;
const { useState: uS_PL } = React;

function ViewFigure({ rule, detail, text }) {
  return <span title={figureTitle(rule, detail)}>{text}</span>;
}

// The expanded rows are held here, by PRD key, so a row a poll moves to
// another lane stays expanded.
function ProjectPrdLanes({ project, view, pool, renderMembers, onSelect }) {
  const [expanded, setExpanded] = uS_PL(() => new Set());
  if (view.state === 'unknown') return <div className="empty prd-unknown">{view.reason}</div>;
  const { value } = view;
  const { prds, lanes, rules, segment_legend: legend } = value;
  const byKey = new Map(prds.map(row => [row.key, row]));
  const toggle = key => setExpanded(was => {
    const next = new Set(was);
    if (!next.delete(key)) next.add(key);
    return next;
  });
  const rowProps = { rules, legend, pool, project, renderMembers, expanded, toggle };
  return (
    <div className="prd-lanes">
      <div className="prd-lanes-head">
        <Pip_PL datum={view} format={served => String(served.prds.length)} label="PRDs with open work" rule={rules.lanes} />
        <SegmentLegend legend={legend} rule={rules.segments} />
      </div>
      {lanes.map(entry => (
        <LaneSection key={entry.lane} entry={entry} rows={entry.prd_keys.map(key => byKey.get(key))} rowProps={rowProps} />
      ))}
      <DarkMatterStrip darkMatter={value.dark_matter} rules={rules} pool={pool} project={project} onSelect={onSelect} />
      <ProjectFooter value={value} />
    </div>
  );
}

function SegmentLegend({ legend, rule }) {
  return (
    <div className="prd-legend" title={rule}>
      {legend.map(item => (
        <span key={item.segment} className="prd-legend-item">
          <span className="prd-legend-swatch" style={{ background: CP_PL[item.tone] }}></span>{item.label}
        </span>
      ))}
    </div>
  );
}

// The tail lane starts collapsed; its header carries the served rule.
function LaneSection({ entry, rows, rowProps }) {
  const [open, setOpen] = uS_PL(entry.lane !== 'tail');
  return (
    <section className="prd-lane" data-lane={entry.lane}>
      <div className="prd-lane-head">
        <ViewFigure rule={entry.rule} text={`${entry.label} (${entry.prd_keys.length})`} />
      </div>
      {open
        ? rows.map(row => <PrdRowView key={row.key} row={row} {...rowProps} />)
        : (
          <button type="button" className="prd-tail-line" onClick={() => setOpen(true)}>
            {`${rows.length}: ${rows.map(row => row.title).join(' · ')}`}
          </button>
        )}
    </section>
  );
}

function PrdRowView({ row, rules, legend, pool, project, renderMembers, expanded, toggle }) {
  const open = expanded.has(row.key);
  const total = doneOfTotal(row);
  const signal = signalCell(row.signal);
  const ready = readyCell(row.ready);
  const action = row.next_action;
  return (
    <div className="prd-row" data-key={row.key}>
      <div className="prd-row-head" role="button" onClick={() => toggle(row.key)}>
        <span className="prd-title"><ViewFigure rule={rules.title} detail={row.key} text={row.title} /></span>
        <span className="prd-state"><ViewFigure rule={rules.state_line} text={row.state_line} /></span>
        <SegmentBar row={row} legend={legend} />
        <span className="prd-total"><ViewFigure rule={rules.total} detail={total.detail} text={total.text} /></span>
        <span className="prd-signal"><ViewFigure rule={rules.signal} detail={signal.detail} text={signal.text} /></span>
        <span className={ready.overdue ? 'prd-ready overdue' : 'prd-ready'}>
          <ViewFigure rule={rules.ready} detail={ready.detail} text={ready.text} />
        </span>
        {action.kind !== 'none' && (
          <span className={`prd-action prd-action-${action.kind}`}>
            <ViewFigure rule={rules.next_action} text={action.text} />
          </span>
        )}
      </div>
      {open && <PrdRowDetail row={row} rules={rules} pool={pool} project={project} renderMembers={renderMembers} />}
    </div>
  );
}

// Zero-count segments stay in the DOM, titled, but take no width or gap.
function SegmentBar({ row, legend }) {
  return (
    <span className="prd-bar">
      {barSegments(row, legend).map(seg => (
        <span
          key={seg.segment}
          data-segment={seg.segment}
          className={seg.count === 0 ? 'prd-bar-seg prd-bar-seg-empty' : 'prd-bar-seg'}
          title={seg.title}
          style={{ flexGrow: seg.count, background: CP_PL[seg.tone] }}
        ></span>
      ))}
    </span>
  );
}

function PrdRowDetail({ row, rules, pool, project, renderMembers }) {
  const openIds = [...row.direct_open_ids, ...row.followup_open.map(followup => followup.task_id)];
  const { rows, notDrawn } = membersDrawn(pool, project, openIds);
  const { blocker_refs: blockers, oldest_ready_high_ids: readyHigh, deferred_unrecorded_ids: unrecorded } = row.detail;
  const { direct, followup } = row.activity;
  return (
    <div className="prd-row-body">
      <div className="prd-activity">
        <ViewFigure rule={rules.activity} text={`active: ${direct} direct, ${followup} follow-up`} />
      </div>
      {renderMembers(rows)}
      {notDrawn.length > 0 && <div className="prd-not-drawn muted">not drawn: {notDrawn.map(id => `#${id}`).join(' ')}</div>}
      <div className="prd-detail" title={rules.detail}>
        <ChipSection className="prd-blockers" label="blocked by" chips={blockers.map(ref => `#${ref.task_id} (${ref.prd ?? 'no PRD'})`)} />
        <ChipSection className="prd-ready-high" label="ready high, oldest first" chips={readyHigh.map(id => `#${id}`)} />
        <ChipSection className="prd-deferred-unrecorded" label="deferred with no record" chips={unrecorded.map(id => `#${id}`)} />
      </div>
    </div>
  );
}

function ChipSection({ className, label, chips }) {
  if (chips.length === 0) return null;
  return (
    <div className={className}>
      <span className="muted">{label}</span>
      {chips.map(chip => <span key={chip} className="chip">{chip}</span>)}
    </div>
  );
}

function DarkMatterStrip({ darkMatter, rules, pool, project, onSelect }) {
  const [open, setOpen] = uS_PL(false);
  const { hidden_high: hiddenHigh } = darkMatter;
  const hidden = hiddenHigh.refs.map(ref => `#${ref.task_id} (${ref.prd})`).join(', ');
  return (
    <div className="dark-matter">
      <div className="dark-matter-head">
        <button type="button" className="dark-matter-toggle" onClick={() => setOpen(was => !was)}>{open ? '▾' : '▸'}</button>
        <ViewFigure rule={rules.dark_matter} text={darkMatterSummary(darkMatter)} />
      </div>
      {hiddenHigh.count > 0 && (
        <div className="dark-matter-hidden-high">
          <ViewFigure rule={rules.hidden_high} text={`${hiddenHigh.count} high open under the collapsed tail: ${hidden}`} />
        </div>
      )}
      {open && Object.entries(darkMatter.members).map(([reason, ids]) => (
        <DarkMatterGroup key={reason} reason={reason} ids={ids} pool={pool} project={project} onSelect={onSelect} />
      ))}
    </div>
  );
}

function DarkMatterGroup({ reason, ids, pool, project, onSelect }) {
  const [pages, setPages] = uS_PL(1);
  const page = darkMatterPage(ids, pages);
  const { rows, notDrawn } = membersDrawn(pool, project, page.ids);
  return (
    <div className="dark-matter-group" data-reason={reason}>
      <div className="dark-matter-reason muted">{`${reason} (${ids.length})`}</div>
      {rows.map(row => (
        <button key={row.id} type="button" className="dark-matter-row" data-row-id={row.id} onClick={() => onSelect(row.id)}>
          {row.title}
        </button>
      ))}
      {notDrawn.length > 0 && <div className="prd-not-drawn muted">not drawn: {notDrawn.map(id => `#${id}`).join(' ')}</div>}
      {page.more > 0 && (
        <button type="button" className="dark-matter-more" onClick={() => setPages(was => was + 1)}>{`more (${page.more})`}</button>
      )}
    </div>
  );
}

function ProjectFooter({ value }) {
  const { closed, signal_coverage: coverage, multi_link: multiLink, inputs, rules } = value;
  return (
    <div className="prd-footer">
      <span className="prd-closed">
        <ViewFigure rule={rules.closed} text={`${closed.prds} PRDs closed (${closed.tasks} tasks)`} />
      </span>
      <span className="prd-coverage">
        <ViewFigure rule={rules.signal_coverage} text={`${coverage.named} of ${coverage.of} open-work PRDs name a signal leaf`} />
      </span>
      {multiLink > 0 && (
        <span className="prd-multi-link">
          <ViewFigure rule={rules.multi_link} text={`${multiLink} tasks carry more than one lineage link`} />
        </span>
      )}
      {inputs.map(input => <PrdInputPip key={input.name} input={input} />)}
    </div>
  );
}

function PrdInputPip({ input }) {
  const pip = inputPip(input);
  return (
    <span className="pip" data-input={input.name} title={pip.title}>
      <span className="pip-dot" style={{ background: CP_PL[pip.tone] }}></span>{pip.text}
    </span>
  );
}

function PrdContentionLine({ contention }) {
  if (contention.state === 'unknown') return <div className="prd-contention muted">{contention.reason}</div>;
  const { lock_depth: depth, snapshot_at: snapshotAt, top, held_share: heldShare, never_started: neverStarted, rules } = contention.value;
  return (
    <div className="prd-contention">
      <span className="prd-contention-head">
        <Pip_PL datum={contention} format={served => String(served.top.length)} label="most-wanted lock keys" rule={rules.top} />
      </span>
      <span className="prd-contention-meta" title={`${rules.lock_depth}\n\n${rules.snapshot_at}`}>
        {`locks at depth ${depth} · scheduler state changed ${timeago_PL(snapshotAt)}`}
      </span>
      {top.map(entry => (
        <span key={entry.key} className="prd-contention-entry"><ViewFigure rule={rules.top} text={contentionEntryText(entry)} /></span>
      ))}
      <span className="prd-held-share"><ViewFigure rule={rules.held_share} text={heldShareText(heldShare)} /></span>
      <span className="prd-never-started"><ViewFigure rule={rules.never_started} text={neverStartedText(neverStarted)} /></span>
    </div>
  );
}

window.DF_PRD_LANES = { ProjectPrdLanes, PrdContentionLine };
