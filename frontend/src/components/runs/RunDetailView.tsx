import { useCallback, useEffect, useState } from 'react'
import { api, type Artifact, type Run } from '../../api/index.ts'
import { useRouter } from '../../router/index.ts'
import { Badge } from '../common/Badge.tsx'
import { ConfirmDialog } from '../common/ConfirmDialog.tsx'
import { Spinner } from '../common/Spinner.tsx'
import { RunArtifactsSection } from './ArtifactViewer.tsx'
import { CasesTable } from './CasesTable.tsx'
import { LiveLogViewer } from './LiveLogViewer.tsx'
import { MetricsTable } from './MetricsTable.tsx'

export function RunDetailView({ runId }: { runId: number }) {
  const { navigate } = useRouter()

  const [run, setRun] = useState<Run | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'logs' | 'results' | 'artifacts' | 'config'>('logs')
  const [selectedResultId, setSelectedResultId] = useState<number | 'all'>('all')

  // Modals
  const [showCancelDialog, setShowCancelDialog] = useState(false)
  const [actionInProgress, setActionInProgress] = useState(false)

  const loadRun = useCallback(async (silent = false) => {
    try {
      if (!silent) setLoading(true)
      const data = await api.runs.get(runId)
      setRun(data)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      if (!silent) setLoading(false)
    }
  }, [runId])

  useEffect(() => {
    let ignore = false
    api.runs
      .get(runId)
      .then((data) => {
        if (!ignore) {
          setRun(data)
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
  }, [runId])

  // Poll run status while queued or running
  const runStatus = run?.status
  useEffect(() => {
    if (!runStatus) return
    const isOngoing = runStatus === 'queued' || runStatus === 'running'
    if (!isOngoing) return

    const timer = setInterval(() => {
      loadRun(true)
    }, 2000)

    return () => clearInterval(timer)
  }, [runStatus, loadRun])

  const handleCancelRun = async () => {
    try {
      setActionInProgress(true)
      await api.runs.cancel(runId)
      setShowCancelDialog(false)
      await loadRun(true)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setActionInProgress(false)
    }
  }

  const handleRerun = async () => {
    if (!run) return
    try {
      setActionInProgress(true)
      const newRun = run.job_id
        ? await api.jobs.trigger(run.job_id)
        : await api.runs.create({
            tester: (run.config_snapshot?.tester as string) || undefined,
            suites: (run.config_snapshot?.suites as string[]) || undefined,
            config: (run.config_snapshot?.config as Record<string, unknown>) || {},
            target_id: (run.target_snapshot?.id as number) || undefined,
            timeout: (run.config_snapshot?.timeout as number) || undefined,
          })
      navigate(`/runs/${newRun.id}`)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setActionInProgress(false)
    }
  }

  if (loading) {
    return <Spinner message={`Loading details for Run #${runId}...`} />
  }

  if (error || !run) {
    return (
      <div className="view-container">
        <div className="banner error-banner">
          {error || `Run #${runId} not found`}
        </div>
        <button className="btn btn-secondary" onClick={() => navigate('/runs')}>
          ← Back to Runs History
        </button>
      </div>
    )
  }

  const isOngoing = run.status === 'queued' || run.status === 'running'
  const testersList = Array.isArray(run.config_snapshot?.testers)
    ? (run.config_snapshot.testers as string[])
    : []
  const testerName =
    testersList.length > 1
      ? testersList.map((t) => t.toUpperCase()).join(', ')
      : ((run.config_snapshot?.tester as string) || 'unknown').toUpperCase()
  const targetName = (run.target_snapshot?.name as string) || 'None'
  const targetUrl = (run.target_snapshot?.base_url as string) || ''
  const jobName = run.job_name || (run.config_snapshot?.job_name as string | undefined) || null
  const jobRunMode = (run.config_snapshot?.run_mode as string | undefined) || null

  // Calculate duration without calling Date.now() during render
  let durationStr = '—'
  if (run.started_at && run.finished_at) {
    const ms = new Date(run.finished_at).getTime() - new Date(run.started_at).getTime()
    durationStr = `${(ms / 1000).toFixed(1)}s`
  } else if (run.status === 'running') {
    durationStr = 'Running...'
  } else if (run.status === 'queued') {
    durationStr = 'Queued'
  }

  const allResults = run.results || []
  const primaryResult = allResults.length > 0 ? allResults[0] : null
  const visibleResults =
    selectedResultId === 'all'
      ? allResults
      : allResults.filter((res) => res.id === selectedResultId)
  const allArtifacts: Artifact[] = allResults.flatMap((res) => res.artifacts || [])

  return (
    <div className="view-container run-detail-view">
      {/* Header Bar */}
      <div className="detail-header-card card">
        <div className="detail-header-top">
          <div className="detail-title-group">
            <button className="back-link-btn" onClick={() => navigate('/runs')}>
              ← Runs
            </button>
            <h2>Run #{run.id}</h2>
            <Badge status={run.status} size="lg" />
            {run.job_id && (
              <span className="job-run-tag">
                Job: {jobName || `#${run.job_id}`}
                {jobRunMode ? ` (${jobRunMode.toUpperCase()})` : ''}
              </span>
            )}
          </div>

          <div className="detail-header-actions">
            {isOngoing && (
              <button
                className="btn btn-danger"
                onClick={() => setShowCancelDialog(true)}
                disabled={actionInProgress}
              >
                Cancel Run
              </button>
            )}
            <button
              className="btn btn-secondary"
              onClick={() => navigate(`/compare?run_ids=${run.id}&baseline_run_id=${run.id}`)}
              disabled={actionInProgress}
              title="Compare this run against other runs"
            >
              ⚖️ Compare
            </button>
            <button
              className="btn btn-secondary"
              onClick={handleRerun}
              disabled={actionInProgress}
              title="Trigger a new run with the exact same configuration"
            >
              ↻ Re-run
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => loadRun(false)}
              disabled={actionInProgress}
              title="Refresh status"
            >
              Refresh
            </button>
          </div>
        </div>

        {/* Info Grid */}
        <div className="detail-meta-grid">
          <div className="meta-item">
            <span className="meta-label">Tester Plugin</span>
            <span className="meta-value font-semibold">{testerName}</span>
          </div>

          <div className="meta-item">
            <span className="meta-label">Target Endpoint</span>
            <span className="meta-value font-mono">
              {targetName} {targetUrl ? `(${targetUrl})` : ''}
            </span>
          </div>

          <div className="meta-item">
            <span className="meta-label">Execution Duration</span>
            <span className="meta-value font-mono">{durationStr}</span>
          </div>

          <div className="meta-item">
            <span className="meta-label">Queued At</span>
            <span className="meta-value text-muted">
              {run.queued_at ? new Date(run.queued_at).toLocaleTimeString() : '—'}
            </span>
          </div>

          <div className="meta-item">
            <span className="meta-label">Started At</span>
            <span className="meta-value text-muted">
              {run.started_at ? new Date(run.started_at).toLocaleTimeString() : '—'}
            </span>
          </div>

          <div className="meta-item">
            <span className="meta-label">Finished At</span>
            <span className="meta-value text-muted">
              {run.finished_at ? new Date(run.finished_at).toLocaleTimeString() : '—'}
            </span>
          </div>
        </div>
      </div>

      {/* Detail Tabs */}
      <div className="detail-nav-tabs">
        <button
          className={`tab-btn ${activeTab === 'logs' ? 'active' : ''}`}
          onClick={() => setActiveTab('logs')}
        >
          Terminal & Live Logs
        </button>
        <button
          className={`tab-btn ${activeTab === 'results' ? 'active' : ''}`}
          onClick={() => setActiveTab('results')}
        >
          Results & Metrics{' '}
          {allResults.length > 1
            ? `(${allResults.length} TESTERS)`
            : primaryResult
              ? `(${primaryResult.status.toUpperCase()})`
              : ''}
        </button>
        <button
          className={`tab-btn ${activeTab === 'artifacts' ? 'active' : ''}`}
          onClick={() => setActiveTab('artifacts')}
          data-testid="run-detail-artifacts-tab"
        >
          Artifacts ({allArtifacts.length})
        </button>
        <button
          className={`tab-btn ${activeTab === 'config' ? 'active' : ''}`}
          onClick={() => setActiveTab('config')}
        >
          Config Snapshot
        </button>
      </div>

      {/* Tab Contents */}
      <div className="detail-tab-content">
        {/* LOGS TAB */}
        {activeTab === 'logs' && (
          <div className="log-tab-wrapper">
            <LiveLogViewer runId={run.id} runStatus={run.status} />
          </div>
        )}

        {/* RESULTS TAB */}
        {activeTab === 'results' && (
          <div className="results-tab-wrapper">
            {allResults.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">⏳</div>
                <h3>No results recorded yet</h3>
                <p>
                  {isOngoing
                    ? 'The run is currently in progress. Results and metrics will be displayed here as soon as tests finish.'
                    : 'This run finished without producing result records.'}
                </p>
                {isOngoing && <button className="btn btn-secondary" onClick={() => setActiveTab('logs')}>Watch Live Logs</button>}
              </div>
            ) : (
              <div className="results-content">
                {/* Per-Tester Result Selector / Grouping Bar when multiple results exist */}
                {allResults.length > 1 && (
                  <div className="grouped-results-bar card">
                    <div className="grouped-results-header">
                      <span className="font-semibold">
                        Grouped Tester Results ({allResults.length}):
                      </span>
                    </div>
                    <div className="grouped-results-pills">
                      <button
                        type="button"
                        className={`tester-result-pill ${selectedResultId === 'all' ? 'active' : ''}`}
                        onClick={() => setSelectedResultId('all')}
                      >
                        All Testers ({allResults.length})
                      </button>
                      {allResults.map((res, idx) => (
                        <button
                          key={res.id}
                          type="button"
                          className={`tester-result-pill ${selectedResultId === res.id ? 'active' : ''}`}
                          onClick={() => setSelectedResultId(res.id)}
                        >
                          <span className="pill-order">{idx + 1}.</span>
                          <span>
                            {res.definition_name
                              ? `${res.definition_name} (${res.tester.toUpperCase()})`
                              : res.tester.toUpperCase()}
                          </span>
                          <Badge status={res.status} size="sm" />
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {visibleResults.map((resultItem, idx) => (
                  <div key={resultItem.id} className="grouped-result-section">
                    {allResults.length > 1 && (
                      <div className="grouped-result-title-bar">
                        <h3>
                          #{idx + 1}{' '}
                          {resultItem.definition_name
                            ? `${resultItem.definition_name} — ${resultItem.tester.toUpperCase()}`
                            : `Tester: ${resultItem.tester.toUpperCase()}`}
                        </h3>
                        <Badge status={resultItem.status} size="md" />
                      </div>
                    )}

                    {/* Result Summary Banner */}
                    <div className="result-summary-card card">
                      <div className="summary-left">
                        <span className="summary-label">Test Outcome:</span>
                        <Badge status={resultItem.status} size="lg" />
                      </div>
                      {resultItem.summary && (
                        <div className="summary-text font-mono">{resultItem.summary}</div>
                      )}
                    </div>

                    {/* Metrics */}
                    <div className="card section-card">
                      <h3 className="section-title">Aggregate Metrics</h3>
                      <MetricsTable metrics={resultItem.metrics} />
                    </div>

                    {/* Test Cases Table */}
                    <div className="card section-card">
                      <h3 className="section-title">
                        Test Cases ({resultItem.cases?.length || 0})
                      </h3>
                      <CasesTable cases={resultItem.cases || []} />
                    </div>

                    {/* Artifacts */}
                    {resultItem.artifacts && resultItem.artifacts.length > 0 && (
                      <RunArtifactsSection
                        runId={run.id}
                        artifacts={resultItem.artifacts}
                      />
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ARTIFACTS TAB */}
        {activeTab === 'artifacts' && (
          <div className="artifacts-tab-wrapper">
            {allArtifacts.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">📂</div>
                <h3>No artifacts recorded for Run #{run.id}</h3>
                <p>
                  Artifacts stored under <code>data/artifacts/{run.id}/</code> will appear here once generated.
                </p>
              </div>
            ) : (
              <RunArtifactsSection
                runId={run.id}
                artifacts={allArtifacts}
                title={`All Run #${run.id} Artifacts`}
              />
            )}
          </div>
        )}

        {/* CONFIG TAB */}
        {activeTab === 'config' && (
          <div className="config-tab-wrapper">
            <div className="card">
              <h3>Configuration Snapshot</h3>
              <p className="text-muted">
                Immutable snapshot of parameters and target state captured when this run was queued.
              </p>
              <div className="drawer-grid">
                <div className="drawer-col">
                  <h4>Config</h4>
                  <pre className="json-block font-mono">
                    {JSON.stringify(run.config_snapshot || {}, null, 2)}
                  </pre>
                </div>
                <div className="drawer-col">
                  <h4>Target Snapshot</h4>
                  <pre className="json-block font-mono">
                    {JSON.stringify(run.target_snapshot || {}, null, 2)}
                  </pre>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Cancel Confirmation */}
      <ConfirmDialog
        isOpen={showCancelDialog}
        title="Cancel Execution"
        message={`Are you sure you want to cancel Run #${run.id}? Any ongoing test subprocess will be terminated immediately.`}
        confirmLabel="Cancel Run"
        danger
        onConfirm={handleCancelRun}
        onCancel={() => setShowCancelDialog(false)}
      />
    </div>
  )
}
