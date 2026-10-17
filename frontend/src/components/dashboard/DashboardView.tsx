import { useEffect, useState } from 'react'
import {
  api,
  type Run,
  type Target,
  type TesterInfo,
} from '../../api/index.ts'
import { Link, useRouter } from '../../router/index.ts'
import { Badge } from '../common/Badge.tsx'
import { Spinner } from '../common/Spinner.tsx'

interface TesterStatusSummary {
  tester: TesterInfo
  latestRun: Run | null
}

export function DashboardView() {
  const { navigate } = useRouter()

  const [testers, setTesters] = useState<TesterInfo[]>([])
  const [runs, setRuns] = useState<Run[]>([])
  const [targets, setTargets] = useState<Target[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = async (showLoadingSpinner: boolean) => {
    try {
      if (showLoadingSpinner) setLoading(true)
      setError(null)
      const [testersData, runsData, targetsData] = await Promise.all([
        api.testers.list(),
        api.runs.list({ limit: 50 }),
        api.targets.list(),
      ])
      setTesters(testersData)
      setRuns(runsData)
      setTargets(targetsData)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      if (showLoadingSpinner) setLoading(false)
    }
  }

  useEffect(() => {
    let ignore = false
    Promise.all([
      api.testers.list(),
      api.runs.list({ limit: 50 }),
      api.targets.list(),
    ])
      .then(([testersData, runsData, targetsData]) => {
        if (!ignore) {
          setTesters(testersData)
          setRuns(runsData)
          setTargets(targetsData)
          setLoading(false)
        }
      })
      .catch((err: unknown) => {
        if (!ignore) {
          setError(err instanceof Error ? err.message : String(err))
          setLoading(false)
        }
      })

    return () => {
      ignore = true
    }
  }, [])

  // Auto-refresh when any run is ongoing
  useEffect(() => {
    const hasActiveRuns = runs.some((r) => r.status === 'queued' || r.status === 'running')
    if (!hasActiveRuns) return

    const timer = setInterval(() => {
      fetchData(false)
    }, 4000)

    return () => clearInterval(timer)
  }, [runs])

  // Correlate latest run per tester
  const testerSummaries: TesterStatusSummary[] = testers.map((t) => {
    const testerRuns = runs.filter(
      (r) => (r.config_snapshot?.tester as string)?.toLowerCase() === t.name.toLowerCase()
    )
    return {
      tester: t,
      latestRun: testerRuns.length > 0 ? testerRuns[0] : null,
    }
  })

  // Aggregate metrics
  const totalRuns = runs.length
  const completedRuns = runs.filter((r) => r.status === 'completed')
  const passedResults = runs.filter(
    (r) => r.results && r.results.length > 0 && r.results[0].status === 'pass'
  )
  const ongoingRuns = runs.filter((r) => r.status === 'queued' || r.status === 'running')
  const successRate =
    completedRuns.length > 0 ? Math.round((passedResults.length / completedRuns.length) * 100) : null

  if (loading) {
    return <Spinner message="Loading dashboard statistics..." />
  }

  return (
    <div className="view-container dashboard-view">
      <div className="view-header">
        <div>
          <h2>Evaluation Dashboard</h2>
          <p className="view-subtitle">
            Overview of evaluation plugins, latest benchmark statuses, and test targets.
          </p>
        </div>
        <div className="header-actions">
          <button className="btn btn-secondary" onClick={() => fetchData(true)}>
            ↻ Refresh
          </button>
          <button className="btn btn-primary" onClick={() => navigate('/trigger')}>
            <span className="btn-icon">🚀</span> Trigger Test
          </button>
        </div>
      </div>

      {error && <div className="banner error-banner">{error}</div>}

      {/* Metric Stat Cards */}
      <div className="stats-row">
        <div className="stat-card">
          <span className="stat-label">Active / Queued</span>
          <div className="stat-number">
            {ongoingRuns.length}
            {ongoingRuns.length > 0 && <span className="pulsing-dot blue" />}
          </div>
          <span className="stat-hint">In execution right now</span>
        </div>

        <div className="stat-card">
          <span className="stat-label">Total Executions</span>
          <div className="stat-number">{totalRuns}</div>
          <span className="stat-hint">Recorded test runs</span>
        </div>

        <div className="stat-card">
          <span className="stat-label">Pass Rate</span>
          <div className="stat-number">{successRate !== null ? `${successRate}%` : 'N/A'}</div>
          <span className="stat-hint">Across completed runs</span>
        </div>

        <div className="stat-card">
          <span className="stat-label">Configured Targets</span>
          <div className="stat-number">{targets.length}</div>
          <Link href="/targets" className="stat-link">
            Manage targets →
          </Link>
        </div>

        <div className="stat-card">
          <span className="stat-label">Registered Testers</span>
          <div className="stat-number">{testers.length}</div>
          <span className="stat-hint">Plugin modules</span>
        </div>
      </div>

      {/* Latest Status Per Tester */}
      <div className="dashboard-section">
        <div className="section-header">
          <h3>Tester Plugins & Latest Status</h3>
          <span className="section-desc">Health and latest run performance per plugin</span>
        </div>

        <div className="tester-cards-grid">
          {testerSummaries.map(({ tester, latestRun }) => {
            const primaryResult =
              latestRun?.results && latestRun.results.length > 0 ? latestRun.results[0] : null

            return (
              <div key={tester.name} className="tester-card card">
                <div className="tester-card-header">
                  <div>
                    <h4 className="tester-name">{tester.name.toUpperCase()}</h4>
                    <span className="tester-suites-count">
                      {tester.suites?.length || 0} sub-suites available
                    </span>
                  </div>
                  {latestRun ? (
                    <Badge status={primaryResult ? primaryResult.status : latestRun.status} />
                  ) : (
                    <span className="status-badge badge-neutral badge-md">NO RUNS</span>
                  )}
                </div>

                <p className="tester-desc">{tester.description}</p>

                {tester.suites && tester.suites.length > 0 && (
                  <div className="tester-suites-pills">
                    {tester.suites.map((s) => (
                      <span key={s} className="suite-pill">
                        {s}
                      </span>
                    ))}
                  </div>
                )}

                <div className="tester-latest-run-info">
                  {latestRun ? (
                    <div className="latest-run-box">
                      <div className="latest-run-row">
                        <span className="run-label">Last Run:</span>
                        <Link href={`/runs/${latestRun.id}`} className="run-link font-mono font-bold">
                          Run #{latestRun.id}
                        </Link>
                        <span className="run-time text-muted">
                          {latestRun.finished_at
                            ? new Date(latestRun.finished_at).toLocaleTimeString()
                            : latestRun.started_at
                              ? new Date(latestRun.started_at).toLocaleTimeString()
                              : 'Just now'}
                        </span>
                      </div>

                      {primaryResult?.metrics && Object.keys(primaryResult.metrics).length > 0 && (
                        <div className="metrics-compact">
                          {Object.entries(primaryResult.metrics)
                            .slice(0, 3)
                            .map(([k, v]) => {
                              const rawVal =
                                v !== null && typeof v === 'object' && 'value' in v
                                  ? (v as { value?: unknown }).value
                                  : v
                              return (
                                <span key={k} className="metric-tag font-mono">
                                  {k}: {typeof rawVal === 'number' ? rawVal.toFixed(2) : String(rawVal)}
                                </span>
                              )
                            })}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="latest-run-box text-muted">No runs executed for this tester yet.</div>
                  )}
                </div>

                <div className="tester-card-footer">
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={() => navigate('/trigger')}
                  >
                    🚀 Trigger {tester.name}
                  </button>
                  {latestRun && (
                    <Link href={`/runs/${latestRun.id}`} className="btn-link font-mono">
                      View Last Run →
                    </Link>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Recent Runs Table */}
      <div className="dashboard-section">
        <div className="section-header">
          <h3>Recent Test Runs</h3>
          <Link href="/runs" className="btn-link">
            View All Runs →
          </Link>
        </div>

        {runs.length === 0 ? (
          <div className="empty-state-compact card">
            <p>No test runs executed yet.</p>
            <button className="btn btn-primary btn-sm" onClick={() => navigate('/trigger')}>
              Trigger First Run
            </button>
          </div>
        ) : (
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Status</th>
                  <th>Tester</th>
                  <th>Target</th>
                  <th>Executed At</th>
                  <th>Result</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {runs.slice(0, 5).map((r) => {
                  const tester = (r.config_snapshot?.tester as string) || 'unknown'
                  const target = (r.target_snapshot?.name as string) || 'None'
                  const primaryRes = r.results && r.results.length > 0 ? r.results[0] : null

                  return (
                    <tr key={r.id}>
                      <td>
                        <Link href={`/runs/${r.id}`} className="font-mono font-bold table-link">
                          #{r.id}
                        </Link>
                      </td>
                      <td>
                        <Badge status={r.status} size="sm" />
                      </td>
                      <td>
                        <span className="kind-badge">{tester}</span>
                      </td>
                      <td className="font-mono text-muted">{target}</td>
                      <td className="text-muted">
                        {r.started_at
                          ? new Date(r.started_at).toLocaleString()
                          : r.queued_at
                            ? new Date(r.queued_at).toLocaleString()
                            : '—'}
                      </td>
                      <td>
                        {primaryRes ? (
                          <Badge status={primaryRes.status} size="sm" />
                        ) : (
                          <span className="text-muted">—</span>
                        )}
                      </td>
                      <td>
                        <Link href={`/runs/${r.id}`} className="btn btn-sm btn-secondary">
                          View →
                        </Link>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
