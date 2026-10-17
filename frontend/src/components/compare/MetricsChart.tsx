import { useState } from 'react'
import type { CompareMetricRow, ComparedRunMeta } from '../../api/types.ts'

function formatNum(val: number | null | undefined): string {
  if (val === null || val === undefined) return '—'
  if (Number.isInteger(val)) return val.toLocaleString()
  if (Math.abs(val) >= 100) return val.toFixed(1)
  if (Math.abs(val) >= 1) return val.toFixed(2)
  return val.toFixed(4)
}

function formatPct(pct: number | null | undefined): string {
  if (pct === null || pct === undefined) return ''
  const sign = pct > 0 ? '+' : ''
  return `${sign}${pct.toFixed(1)}%`
}

export function MetricsChart({
  metrics,
  runs,
}: {
  metrics: CompareMetricRow[]
  runs: ComparedRunMeta[]
}) {
  const [chartType, setChartType] = useState<'bar' | 'line'>('bar')
  const [selectedNames, setSelectedNames] = useState<string[] | null>(null)

  if (!metrics || metrics.length === 0) {
    return null
  }

  // Default to up to 6 key metrics (prioritizing regressed/changed metrics first)
  const sortedForDefault = [...metrics].sort((a, b) => {
    const scoreA = a.has_regression ? 3 : a.has_improvement ? 2 : a.has_change ? 1 : 0
    const scoreB = b.has_regression ? 3 : b.has_improvement ? 2 : b.has_change ? 1 : 0
    return scoreB - scoreA
  })
  const defaultMetricNames = sortedForDefault.slice(0, 6).map((m) => m.name)
  const activeNames = selectedNames ?? defaultMetricNames

  const toggleMetric = (name: string) => {
    const current = selectedNames ?? defaultMetricNames
    if (current.includes(name)) {
      if (current.length > 1) {
        setSelectedNames(current.filter((n) => n !== name))
      }
    } else {
      setSelectedNames([...current, name])
    }
  }

  const displayedMetrics = metrics.filter((m) => activeNames.includes(m.name))

  return (
    <div className="card compare-charts-card" data-testid="metrics-chart-section">
      <div className="compare-section-header">
        <div>
          <h3 className="section-title" style={{ marginBottom: '2px' }}>
            Key Metrics Charts
          </h3>
          <span className="section-desc">
            Visual comparison of metric values and deltas across selected runs
          </span>
        </div>

        <div className="chart-controls">
          <div className="btn-group" role="group" aria-label="Chart Type">
            <button
              type="button"
              className={`btn btn-sm ${chartType === 'bar' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setChartType('bar')}
              data-testid="chart-type-bar"
            >
              📊 Bar Chart
            </button>
            <button
              type="button"
              className={`btn btn-sm ${chartType === 'line' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setChartType('line')}
              data-testid="chart-type-line"
            >
              📈 Line Chart
            </button>
          </div>
          {metrics.length > 6 && (
            <button
              type="button"
              className="btn btn-sm btn-secondary"
              onClick={() =>
                setSelectedNames(
                  activeNames.length === metrics.length ? defaultMetricNames : metrics.map((m) => m.name)
                )
              }
            >
              {activeNames.length === metrics.length ? 'Show Top 6' : `Show All (${metrics.length})`}
            </button>
          )}
        </div>
      </div>

      {/* Metric Selector Pills */}
      {metrics.length > 1 && (
        <div className="chart-metric-pills">
          {metrics.map((m) => {
            const isSelected = activeNames.includes(m.name)
            return (
              <button
                key={m.name}
                type="button"
                className={`pill-btn ${isSelected ? 'selected' : ''}`}
                onClick={() => toggleMetric(m.name)}
              >
                <span>{m.name}</span>
                {m.has_regression && <span className="pill-dot red" title="Regression" />}
                {!m.has_regression && m.has_improvement && (
                  <span className="pill-dot green" title="Improvement" />
                )}
              </button>
            )
          })}
        </div>
      )}

      {/* Charts Grid */}
      <div className="charts-grid">
        {displayedMetrics.map((m) => {
          const numericValues = m.values
            .map((v) => v.value)
            .filter((v): v is number => typeof v === 'number')
          const maxVal = numericValues.length > 0 ? Math.max(...numericValues, 0) : 1
          const minVal = numericValues.length > 0 ? Math.min(...numericValues, 0) : 0
          const safeMax = maxVal === 0 ? 1 : maxVal

          return (
            <div
              key={m.name}
              className={`chart-card ${m.has_regression ? 'chart-card-regression' : m.has_improvement ? 'chart-card-improvement' : ''}`}
              data-testid={`metric-chart-${m.name}`}
            >
              <div className="chart-card-header">
                <div>
                  <span className="chart-metric-title font-mono font-bold">{m.name}</span>
                  {m.unit && <span className="chart-metric-unit">({m.unit})</span>}
                </div>
                <div className="chart-card-badges">
                  {m.better && (
                    <span className="better-hint">
                      {m.better === 'lower' ? '↓ lower is better' : '↑ higher is better'}
                    </span>
                  )}
                  {m.has_regression && (
                    <span className="delta-badge delta-regression">REGRESSED</span>
                  )}
                  {!m.has_regression && m.has_improvement && (
                    <span className="delta-badge delta-improvement">IMPROVED</span>
                  )}
                </div>
              </div>

              {chartType === 'bar' ? (
                <div className="bar-chart-container" role="img" aria-label={`${m.name} bar chart`}>
                  <svg
                    viewBox={`0 0 360 ${Math.max(70, m.values.length * 36 + 16)}`}
                    className="metric-bar-svg"
                  >
                    {m.values.map((cell, idx) => {
                      const runMeta = runs.find(
                        (r) => r.run_id === cell.run_id && r.result_id === cell.result_id
                      )
                      const label = `Run #${cell.run_id}${cell.is_baseline ? ' (Base)' : ''}${runMeta?.model ? ` · ${runMeta.model}` : ''}`
                      const val = cell.value ?? 0
                      const ratio = cell.present ? Math.max(0.04, Math.min(1, Math.abs(val) / safeMax)) : 0
                      const barWidth = Math.round(ratio * 170)
                      const y = 12 + idx * 36
                      const fill = cell.is_baseline
                        ? '#3b82f6'
                        : cell.regression
                          ? '#ef4444'
                          : cell.improvement
                            ? '#10b981'
                            : '#64748b'

                      const pctStr = !cell.is_baseline && cell.present ? formatPct(cell.pct_delta) : ''
                      const valStr = cell.present
                        ? `${formatNum(cell.value)}${m.unit ? ` ${m.unit}` : ''}${pctStr ? ` (${pctStr})` : ''}`
                        : 'missing'

                      return (
                        <g key={`${cell.run_id}-${cell.result_id}`}>
                          <text
                            x="4"
                            y={y + 15}
                            fill="#cbd5e1"
                            fontSize="11"
                            fontFamily="monospace"
                          >
                            {label.length > 18 ? `${label.slice(0, 17)}…` : label}
                          </text>
                          <rect
                            x="118"
                            y={y + 2}
                            width={barWidth}
                            height="20"
                            rx="4"
                            fill={fill}
                            opacity={cell.present ? 0.9 : 0.25}
                          />
                          <text
                            x={124 + barWidth}
                            y={y + 16}
                            fill={
                              cell.regression
                                ? '#fca5a5'
                                : cell.improvement
                                  ? '#86efac'
                                  : '#f8fafc'
                            }
                            fontSize="11"
                            fontFamily="monospace"
                          >
                            {valStr}
                          </text>
                        </g>
                      )
                    })}
                  </svg>
                </div>
              ) : (
                <div className="line-chart-container" role="img" aria-label={`${m.name} line chart`}>
                  {(() => {
                    const width = 360
                    const height = 140
                    const padLeft = 42
                    const padRight = 28
                    const padTop = 24
                    const padBottom = 28
                    const plotW = width - padLeft - padRight
                    const plotH = height - padTop - padBottom
                    const range = maxVal - minVal || 1

                    const getX = (idx: number, count: number) =>
                      count <= 1 ? padLeft + plotW / 2 : padLeft + (idx / (count - 1)) * plotW
                    const getY = (val: number) =>
                      padTop + plotH - ((val - minVal) / range) * plotH

                    const presentPoints = m.values
                      .map((cell, idx) => ({
                        cell,
                        idx,
                        x: getX(idx, m.values.length),
                        y: cell.value !== null ? getY(cell.value) : padTop + plotH,
                      }))
                      .filter((p) => p.cell.present && p.cell.value !== null)

                    const polyPoints = presentPoints.map((p) => `${p.x},${p.y}`).join(' ')
                    const baseY =
                      m.baseline_value !== null ? getY(m.baseline_value) : null

                    return (
                      <svg viewBox={`0 0 ${width} ${height}`} className="metric-line-svg">
                        {/* Grid / Baseline reference line */}
                        <line
                          x1={padLeft}
                          y1={padTop + plotH}
                          x2={width - padRight}
                          y2={padTop + plotH}
                          stroke="#334155"
                          strokeWidth="1"
                        />
                        {baseY !== null && (
                          <line
                            x1={padLeft}
                            y1={baseY}
                            x2={width - padRight}
                            y2={baseY}
                            stroke="#3b82f6"
                            strokeDasharray="4 4"
                            strokeWidth="1"
                            opacity="0.6"
                          />
                        )}

                        {/* Connecting line */}
                        {presentPoints.length > 1 && (
                          <polyline
                            fill="none"
                            stroke="#38bdf8"
                            strokeWidth="2.5"
                            points={polyPoints}
                          />
                        )}

                        {/* Data points and labels */}
                        {m.values.map((cell, idx) => {
                          const x = getX(idx, m.values.length)
                          const y =
                            cell.present && cell.value !== null
                              ? getY(cell.value)
                              : padTop + plotH
                          const dotColor = cell.is_baseline
                            ? '#3b82f6'
                            : cell.regression
                              ? '#ef4444'
                              : cell.improvement
                                ? '#10b981'
                                : '#38bdf8'

                          return (
                            <g key={`${cell.run_id}-${cell.result_id}`}>
                              {cell.present && cell.value !== null && (
                                <>
                                  <circle
                                    cx={x}
                                    cy={y}
                                    r="5"
                                    fill={dotColor}
                                    stroke="#0f172a"
                                    strokeWidth="1.5"
                                  />
                                  <text
                                    x={x}
                                    y={Math.max(12, y - 8)}
                                    textAnchor="middle"
                                    fill="#f8fafc"
                                    fontSize="10"
                                    fontFamily="monospace"
                                  >
                                    {formatNum(cell.value)}
                                  </text>
                                </>
                              )}
                              <text
                                x={x}
                                y={height - 8}
                                textAnchor="middle"
                                fill="#94a3b8"
                                fontSize="10"
                                fontFamily="monospace"
                              >
                                #{cell.run_id}
                                {cell.is_baseline ? '*' : ''}
                              </text>
                            </g>
                          )
                        })}
                      </svg>
                    )
                  })()}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
