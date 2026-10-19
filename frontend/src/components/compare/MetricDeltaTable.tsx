import { useState } from 'react'
import type { CompareMetricCell, CompareMetricRow, ComparedRunMeta } from '../../api/types.ts'

function formatMetricNumber(val: number | null | undefined): string {
  if (val === null || val === undefined) return '—'
  if (Number.isInteger(val)) return val.toLocaleString()
  if (Math.abs(val) >= 100) return val.toFixed(2)
  if (Math.abs(val) >= 1) return val.toFixed(3)
  return val.toFixed(4)
}

function formatSignedDelta(val: number | null | undefined, unit?: string | null): string {
  if (val === null || val === undefined) return '—'
  const sign = val > 0 ? '+' : ''
  const formatted = Number.isInteger(val)
    ? `${sign}${val}`
    : Math.abs(val) >= 10
      ? `${sign}${val.toFixed(2)}`
      : `${sign}${val.toFixed(3)}`
  return unit ? `${formatted} ${unit}` : formatted
}

function formatSignedPct(pct: number | null | undefined): string {
  if (pct === null || pct === undefined) return 'N/A'
  const sign = pct > 0 ? '+' : ''
  return `${sign}${pct.toFixed(2)}%`
}

export function MetricDeltaTable({
  metrics,
  totalMetricsCount,
  runs,
}: {
  metrics: CompareMetricRow[]
  totalMetricsCount: number
  runs: ComparedRunMeta[]
}) {
  const [search, setSearch] = useState('')

  const visibleMetrics = metrics.filter((m) =>
    m.name.toLowerCase().includes(search.toLowerCase())
  )

  const findCell = (row: CompareMetricRow, r: ComparedRunMeta): CompareMetricCell | undefined =>
    row.values.find((c) => c.run_id === r.run_id && c.result_id === r.result_id) ||
    row.values.find((c) => c.run_id === r.run_id)

  return (
    <div className="card compare-section-card" data-testid="metric-delta-table-section">
      <div className="compare-section-header">
        <div>
          <h3 className="section-title" style={{ marginBottom: '2px' }}>
            Metric Delta Table ({visibleMetrics.length}
            {visibleMetrics.length !== totalMetricsCount ? ` of ${totalMetricsCount}` : ''})
          </h3>
          <span className="section-desc">
            Top-level metrics aligned across runs with absolute &amp; percentage deltas vs reference run
          </span>
        </div>

        <div className="search-input-wrapper" style={{ maxWidth: '280px' }}>
          <span className="search-icon">🔍</span>
          <input
            type="text"
            className="search-input"
            placeholder="Filter metrics by name..."
            aria-label="Filter metrics by name"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </div>

      {visibleMetrics.length === 0 ? (
        <div className="text-muted empty-inline-msg">
          No metrics match the active filters.
        </div>
      ) : (
        <div className="table-responsive">
          <table className="data-table metric-delta-table" data-testid="metric-delta-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Direction</th>
                {runs.map((r) => (
                  <th key={`${r.run_id}-${r.result_id}`}>
                    <div className="th-run-header">
                      <span>
                        Run #{r.run_id}
                        {r.baselines && r.baselines.length > 0
                          ? ` ★ ${r.baselines.map((b) => b.name).join(', ')}`
                          : ''}
                        {r.is_baseline ? ' (Reference run)' : ''}
                      </span>
                      {(r.model || r.target_name) && (
                        <span className="th-run-sub font-mono">
                          {r.model || r.target_name}
                        </span>
                      )}
                    </div>
                  </th>
                ))}
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {visibleMetrics.map((m) => {
                const rowClass = m.has_regression
                  ? 'row-regression'
                  : m.has_improvement
                    ? 'row-improvement'
                    : ''

                return (
                  <tr
                    key={m.name}
                    className={`metric-delta-row ${rowClass}`}
                    data-testid={`metric-row-${m.name}`}
                  >
                    <td>
                      <div className="metric-name-cell">
                        <span className="font-mono font-bold">{m.name}</span>
                        {m.unit && <span className="unit-pill font-mono">{m.unit}</span>}
                      </div>
                    </td>
                    <td>
                      {m.better === 'lower' ? (
                        <span className="direction-pill font-mono" title="Lower values are better">
                          ↓ lower
                        </span>
                      ) : m.better === 'higher' ? (
                        <span className="direction-pill font-mono" title="Higher values are better">
                          ↑ higher
                        </span>
                      ) : (
                        <span className="text-muted font-mono">—</span>
                      )}
                    </td>

                    {runs.map((r) => {
                      const cell = findCell(m, r)
                      if (!cell || !cell.present || cell.value === null) {
                        return (
                          <td
                            key={`${r.run_id}-${r.result_id}`}
                            className="cell-missing font-mono"
                          >
                            <span className="delta-badge delta-missing">— (missing)</span>
                          </td>
                        )
                      }

                      if (cell.is_baseline) {
                        return (
                          <td
                            key={`${r.run_id}-${r.result_id}`}
                            className="cell-baseline font-mono"
                          >
                            <div className="metric-cell-val font-semibold">
                              {formatMetricNumber(cell.value)}
                              {m.unit ? <span className="text-muted"> {m.unit}</span> : null}
                            </div>
                            <div className="metric-cell-sub text-muted">Reference run</div>
                          </td>
                        )
                      }

                      const cellClass = cell.regression
                        ? 'cell-regression'
                        : cell.improvement
                          ? 'cell-improvement'
                          : ''

                      return (
                        <td
                          key={`${r.run_id}-${r.result_id}`}
                          className={`font-mono ${cellClass}`}
                          data-testid={`metric-cell-${m.name}-${r.run_id}`}
                        >
                          <div className="metric-cell-val font-semibold">
                            {formatMetricNumber(cell.value)}
                            {m.unit ? <span className="text-muted"> {m.unit}</span> : null}
                          </div>
                          <div className="metric-cell-delta-row">
                            <span
                              className={`delta-badge ${
                                cell.regression
                                  ? 'delta-regression'
                                  : cell.improvement
                                    ? 'delta-improvement'
                                    : 'delta-neutral'
                              }`}
                            >
                              {cell.regression ? '▼ ' : cell.improvement ? '▲ ' : ''}
                              {formatSignedDelta(cell.abs_delta, m.unit)} (
                              {formatSignedPct(cell.pct_delta)})
                            </span>
                          </div>
                        </td>
                      )
                    })}

                    <td>
                      {m.has_regression ? (
                        <span className="delta-badge delta-regression">REGRESSION</span>
                      ) : m.has_improvement ? (
                        <span className="delta-badge delta-improvement">IMPROVED</span>
                      ) : m.missing_in_some ? (
                        <span className="delta-badge delta-missing">MISSING</span>
                      ) : m.has_change ? (
                        <span className="delta-badge delta-neutral">CHANGED</span>
                      ) : (
                        <span className="delta-badge delta-unchanged">UNCHANGED</span>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
