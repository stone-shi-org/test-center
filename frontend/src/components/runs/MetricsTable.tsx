export function MetricsTable({
  metrics,
}: {
  metrics: Record<string, unknown> | null | undefined
}) {
  if (!metrics || Object.keys(metrics).length === 0) {
    return <div className="text-muted empty-inline-msg">No metrics reported.</div>
  }

  const formatMetricValue = (val: unknown): string => {
    if (typeof val === 'number') {
      if (Number.isInteger(val)) return val.toLocaleString()
      return val.toFixed(3)
    }
    if (typeof val === 'boolean') {
      return val ? 'True' : 'False'
    }
    if (val === null || val === undefined) {
      return '—'
    }
    return String(val)
  }

  const entries = Object.entries(metrics)

  return (
    <div className="metrics-grid">
      {entries.map(([key, val]) => (
        <div key={key} className="metric-card">
          <div className="metric-label" title={key}>
            {key.replace(/_/g, ' ')}
          </div>
          <div className="metric-value font-mono">{formatMetricValue(val)}</div>
        </div>
      ))}
    </div>
  )
}
