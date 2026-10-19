import { useCallback, useEffect, useState } from 'react'
import { ApiError, api, type Artifact, type Run, type RunBaselineBadge } from '../../api/index.ts'
import { useRouter } from '../../router/index.ts'
import { Badge } from '../common/Badge.tsx'
import { ConfirmDialog } from '../common/ConfirmDialog.tsx'
import { Spinner } from '../common/Spinner.tsx'
import { RunArtifactsSection } from './ArtifactViewer.tsx'
import { CasesTable } from './CasesTable.tsx'
import { LiveLogViewer } from './LiveLogViewer.tsx'
import { MetricsTable } from './MetricsTable.tsx'

function suggestBaselineName(run: Run): string {
  const tester =
    (run.config_snapshot?.tester as string) ||
    run.results?.[0]?.tester ||
    'run'
  const cfgObj = (run.config_snapshot?.config as Record<string, unknown> | undefined) || {}
  const tgtExtra = (run.target_snapshot?.extra as Record<string, unknown> | undefined) || {}
  const model =
    (cfgObj.model as string) ||
    (cfgObj.transcription_model as string) ||
    (tgtExtra.model as string) ||
    run.results?.[0]?.model ||
    (run.target_snapshot?.name as string) ||
    'default'
  const rawDate = run.finished_at || run.started_at || run.queued_at || '2026-01-01'
  const datePart = rawDate.slice(0, 10)
  return `${tester} · ${model} · ${datePart}`
}

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

  // Baseline dialog state
  const [showBaselineDialog, setShowBaselineDialog] = useState(false)
  const [baselineName, setBaselineName] = useState('')
  const [baselineDescription, setBaselineDescription] = useState('')
  const [baselineError, setBaselineError] = useState<string | null>(null)
  const [baselineConflict, setBaselineConflict] = useState<{
    name: string
    runId: number
  } | null>(null)
  const [savingBaseline, setSavingBaseline] = useState(false)

  // Rename baseline state
  const [renamingBaseline, setRenamingBaseline] = useState<RunBaselineBadge | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const [renameError, setRenameError] = useState<string | null>(null)

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

  const openSaveBaselineDialog = () => {
    if (!run) return
    setBaselineName(suggestBaselineName(run))
    setBaselineDescription('')
    setBaselineError(null)
    setBaselineConflict(null)
    setShowBaselineDialog(true)
  }

  const handleSaveBaseline = async (move = false) => {
    if (!run) return
    const trimmedName = baselineName.trim()
    if (!trimmedName) {
      setBaselineError('Baseline name is required.')
      return
    }
    try {
      setSavingBaseline(true)
      setBaselineError(null)
      if (!move) setBaselineConflict(null)
      await api.runs.saveBaseline(run.id, {
        name: trimmedName,
        description: baselineDescription.trim() || null,
        move,
      })
      setShowBaselineDialog(false)
      setBaselineConflict(null)
      await loadRun(true)
    } catch (err: unknown) {
      if (err instanceof ApiError && err.status === 409) {
        const rawObj = (err.detail as Record<string, unknown> | null) || {}
        const nestedObj =
          rawObj.detail && typeof rawObj.detail === 'object'
            ? (rawObj.detail as Record<string, unknown>)
            : {}
        const detailObj = { ...nestedObj, ...rawObj }
        const existingRunId =
          typeof detailObj.run_id === 'number'
            ? detailObj.run_id
            : typeof detailObj.current_run_id === 'number'
              ? detailObj.current_run_id
              : 0
        const existingName =
          typeof detailObj.name === 'string' && detailObj.name
            ? detailObj.name
            : trimmedName
        setBaselineConflict({
          name: existingName,
          runId: existingRunId,
        })
      } else {
        setBaselineError(err instanceof Error ? err.message : String(err))
      }
    } finally {
      setSavingBaseline(false)
    }
  }

  const handleStartRename = (b: RunBaselineBadge) => {
    setRenamingBaseline(b)
    setRenameValue(b.name)
    setRenameError(null)
  }

  const handleConfirmRename = async () => {
    if (!renamingBaseline) return
    const trimmed = renameValue.trim()
    if (!trimmed) {
      setRenameError('Baseline name cannot be empty.')
      return
    }
    try {
      setActionInProgress(true)
      setRenameError(null)
      await api.baselines.update(renamingBaseline.id, { name: trimmed })
      setRenamingBaseline(null)
      await loadRun(true)
    } catch (err: unknown) {
      setRenameError(err instanceof Error ? err.message : String(err))
    } finally {
      setActionInProgress(false)
    }
  }

  const handleRemoveBaseline = async (baselineId: number) => {
    try {
      setActionInProgress(true)
      await api.baselines.delete(baselineId)
      await loadRun(true)
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
  const isFinished =
    run.status === 'completed' ||
    run.status === 'failed' ||
    run.status === 'error' ||
    run.status === 'cancelled'
  const isNonPassingFinished =
    run.status === 'failed' || run.status === 'error' || run.status === 'cancelled'
  const runBaselines = run.baselines || []
  const previousBaselines = run.previous_baselines || []
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
  const bambooMeta = run.bamboo || run.config_snapshot?.bamboo || null
  const junitUrl = run.junit_url || api.runs.getJunitXmlUrl(run.id)

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
          <div className="detail-title-group" style={{ flexWrap: 'wrap' }}>
            <button className="back-link-btn" onClick={() => navigate('/runs')}>
              ← Runs
            </button>
            <h2>Run #{run.id}</h2>
            <Badge status={run.status} size="lg" />
            {runBaselines.map((b) => (
              <span
                key={b.id}
                className="job-run-tag"
                data-testid={`run-baseline-badge-${b.id}`}
                style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}
              >
                <span>★ {b.name}</span>
                <button
                  type="button"
                  className="btn btn-sm btn-secondary"
                  style={{ padding: '1px 6px', fontSize: '0.75rem' }}
                  data-testid={`rename-baseline-btn-${b.id}`}
                  onClick={() => handleStartRename(b)}
                  disabled={actionInProgress}
                >
                  Rename
                </button>
                <button
                  type="button"
                  className="btn btn-sm btn-secondary"
                  style={{ padding: '1px 6px', fontSize: '0.75rem' }}
                  data-testid={`remove-baseline-btn-${b.id}`}
                  onClick={() => handleRemoveBaseline(b.id)}
                  disabled={actionInProgress}
                >
                  Remove
                </button>
              </span>
            ))}
            {previousBaselines.map((pb) => (
              <span
                key={pb.id}
                className="job-run-tag text-muted"
                data-testid={`previous-baseline-tag-${pb.id}`}
              >
                previously baseline '{pb.name}'
              </span>
            ))}
            {run.job_id && (
              <span className="job-run-tag">
                Job: {jobName || `#${run.job_id}`}
                {jobRunMode ? ` (${jobRunMode.toUpperCase()})` : ''}
              </span>
            )}
            {bambooMeta && (
              <span
                className="job-run-tag"
                data-testid="run-bamboo-badge"
                title="Triggered via Bamboo CI integration"
              >
                🎋 Bamboo:{' '}
                {(bambooMeta.build_result_key as string) ||
                  (bambooMeta.plan_key as string) ||
                  'CI'}
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
            {isFinished && (
              <button
                type="button"
                className="btn btn-secondary"
                data-testid="save-baseline-btn"
                onClick={openSaveBaselineDialog}
                disabled={actionInProgress}
                title="Save this finished run as a named baseline protected from retention"
              >
                ★ Save as baseline
              </button>
            )}
            <a
              href={junitUrl}
              download={`run-${run.id}-junit.xml`}
              className="btn btn-secondary"
              data-testid="run-junit-download"
              title="Download JUnit XML report for CI / Bamboo JUnit Parser"
            >
              ⬇ JUnit XML
            </a>
            <button
              className="btn btn-secondary"
              onClick={() =>
                navigate(
                  runBaselines.length > 0
                    ? `/compare?baselines=${encodeURIComponent(runBaselines[0].name)}`
                    : `/compare?run_ids=${run.id}&baseline_run_id=${run.id}`
                )
              }
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

      {/* Save as Baseline Dialog */}
      {showBaselineDialog && (
        <div className="modal-backdrop" onClick={() => setShowBaselineDialog(false)}>
          <div
            className="modal-content"
            data-testid="save-baseline-dialog"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-header">
              <h3>★ Save Run #{run.id} as Baseline</h3>
            </div>
            <div className="modal-body" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {isNonPassingFinished && (
                <div
                  className="banner warning-banner"
                  data-testid="baseline-warning-banner"
                >
                  ⚠️ Warning: This run finished with status <strong>{run.status}</strong>. Please confirm before saving a {run.status} run as a baseline.
                </div>
              )}

              {baselineError && (
                <div className="banner error-banner" data-testid="baseline-error-banner">
                  {baselineError}
                </div>
              )}

              {baselineConflict && (
                <div
                  className="banner warning-banner"
                  data-testid="baseline-conflict-banner"
                  style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}
                >
                  <div>
                    Baseline <strong>'{baselineConflict.name}'</strong> already exists and points to{' '}
                    <strong>Run #{baselineConflict.runId}</strong>. Choose a different name below, or move it to this run:
                  </div>
                  <div>
                    <button
                      type="button"
                      className="btn btn-sm btn-primary"
                      data-testid="move-baseline-btn"
                      disabled={savingBaseline}
                      onClick={() => handleSaveBaseline(true)}
                    >
                      Move '{baselineConflict.name}' to this run
                    </button>
                  </div>
                </div>
              )}

              <div className="form-group">
                <label htmlFor="baseline-name-input" className="form-label">
                  Baseline Name *
                </label>
                <input
                  id="baseline-name-input"
                  data-testid="baseline-name-input"
                  type="text"
                  className="form-input"
                  value={baselineName}
                  onChange={(e) => {
                    setBaselineName(e.target.value)
                    if (baselineConflict) setBaselineConflict(null)
                  }}
                  placeholder="e.g. llm · Qwen2.5-72B · 2026-10-18"
                />
              </div>

              <div className="form-group">
                <label htmlFor="baseline-description-input" className="form-label">
                  Description (optional)
                </label>
                <textarea
                  id="baseline-description-input"
                  data-testid="baseline-description-input"
                  className="form-input"
                  rows={2}
                  value={baselineDescription}
                  onChange={(e) => setBaselineDescription(e.target.value)}
                  placeholder="Optional notes about why this run is a reference baseline"
                />
              </div>
            </div>
            <div className="modal-footer">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setShowBaselineDialog(false)}
                disabled={savingBaseline}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn btn-primary"
                data-testid="confirm-save-baseline-btn"
                onClick={() => handleSaveBaseline(false)}
                disabled={savingBaseline || !baselineName.trim()}
              >
                {savingBaseline ? 'Saving...' : 'Save Baseline'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Rename Baseline Dialog */}
      {renamingBaseline && (
        <div className="modal-backdrop" onClick={() => setRenamingBaseline(null)}>
          <div
            className="modal-content"
            data-testid="rename-baseline-dialog"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-header">
              <h3>Rename Baseline '{renamingBaseline.name}'</h3>
            </div>
            <div className="modal-body" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {renameError && (
                <div className="banner error-banner" data-testid="rename-baseline-error">
                  {renameError}
                </div>
              )}
              <div className="form-group">
                <label htmlFor="rename-baseline-input" className="form-label">
                  New Baseline Name *
                </label>
                <input
                  id="rename-baseline-input"
                  data-testid="rename-baseline-input"
                  type="text"
                  className="form-input"
                  value={renameValue}
                  onChange={(e) => setRenameValue(e.target.value)}
                />
              </div>
            </div>
            <div className="modal-footer">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setRenamingBaseline(null)}
                disabled={actionInProgress}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn btn-primary"
                data-testid="confirm-rename-baseline-btn"
                onClick={handleConfirmRename}
                disabled={actionInProgress || !renameValue.trim()}
              >
                Save Name
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
