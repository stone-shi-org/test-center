import { useEffect, useState } from 'react'
import {
  api,
  type RetentionCleanupResult,
  type RetentionSettings,
} from '../../api/index.ts'

function formatBytes(bytes: number | null | undefined): string {
  if (!bytes || bytes <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let val = bytes
  let uIdx = 0
  while (val >= 1024 && uIdx < units.length - 1) {
    val /= 1024
    uIdx++
  }
  return `${val.toFixed(uIdx === 0 ? 0 : 1)} ${units[uIdx]}`
}

export function RetentionSettingsPanel({
  onCleanedUp,
  onClose,
}: {
  onCleanedUp: () => void
  onClose?: () => void
}) {
  const [enabled, setEnabled] = useState(false)
  const [maxAgeDays, setMaxAgeDays] = useState<string>('')
  const [maxRuns, setMaxRuns] = useState<string>('')
  const [artifactsOnly, setArtifactsOnly] = useState(false)
  const [keepFailed, setKeepFailed] = useState(false)

  const [stats, setStats] = useState<RetentionSettings['stats'] | null>(null)
  const [lastCleanupAt, setLastCleanupAt] = useState<string | null>(null)
  const [cleanupResult, setCleanupResult] = useState<RetentionCleanupResult | null>(null)
  const [saving, setSaving] = useState(false)
  const [cleaning, setCleaning] = useState(false)
  const [statusMessage, setStatusMessage] = useState<string | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const applyLoadedSettings = (data: RetentionSettings) => {
    if (!data || Array.isArray(data) || typeof data !== 'object') return
    if (typeof data.enabled === 'boolean') setEnabled(data.enabled)
    const days = data.max_age_days ?? data.retention_days
    setMaxAgeDays(typeof days === 'number' && days > 0 ? String(days) : '')
    setMaxRuns(typeof data.max_runs === 'number' && data.max_runs > 0 ? String(data.max_runs) : '')
    if (typeof data.artifacts_only === 'boolean') setArtifactsOnly(data.artifacts_only)
    if (typeof data.keep_failed === 'boolean') setKeepFailed(data.keep_failed)
    if (data.stats) setStats(data.stats)
    if (data.last_cleanup_at) setLastCleanupAt(data.last_cleanup_at)
  }

  useEffect(() => {
    let ignore = false
    api.runs
      .getRetention()
      .then((data) => {
        if (!ignore) {
          applyLoadedSettings(data)
        }
      })
      .catch(() => {
        // Keep defaults if endpoint not mocked in basic unit tests
      })
    return () => {
      ignore = true
    }
  }, [])

  const parsedDays = maxAgeDays.trim() ? parseInt(maxAgeDays.trim(), 10) : null
  const parsedRuns = maxRuns.trim() ? parseInt(maxRuns.trim(), 10) : null

  const handleSave = async (applyNow = false) => {
    try {
      setSaving(true)
      setErrorMessage(null)
      setStatusMessage(null)
      const updated = await api.runs.updateRetention({
        enabled,
        max_age_days: parsedDays && parsedDays > 0 ? parsedDays : null,
        retention_days: parsedDays && parsedDays > 0 ? parsedDays : null,
        max_runs: parsedRuns && parsedRuns > 0 ? parsedRuns : null,
        artifacts_only: artifactsOnly,
        keep_failed: keepFailed,
        apply_now: applyNow,
      })
      applyLoadedSettings(updated)
      if (updated.cleanup_result) {
        setCleanupResult(updated.cleanup_result)
        onCleanedUp()
      }
      setStatusMessage(
        applyNow
          ? 'Retention settings saved and cleanup executed.'
          : 'Retention settings saved.'
      )
    } catch (err: unknown) {
      setErrorMessage(err instanceof Error ? err.message : String(err))
    } finally {
      setSaving(false)
    }
  }

  const handleCleanup = async (dryRun: boolean) => {
    try {
      setCleaning(true)
      setErrorMessage(null)
      setStatusMessage(null)
      const res = await api.runs.cleanupRetention({
        max_age_days: parsedDays && parsedDays > 0 ? parsedDays : null,
        retention_days: parsedDays && parsedDays > 0 ? parsedDays : null,
        max_runs: parsedRuns && parsedRuns > 0 ? parsedRuns : null,
        artifacts_only: artifactsOnly,
        keep_failed: keepFailed,
        dry_run: dryRun,
      })
      setCleanupResult(res)
      if (!dryRun) {
        setStatusMessage(
          res.artifacts_only
            ? `Pruned ${res.deleted_artifacts} artifacts from ${res.cleaned_runs} old runs (${formatBytes(res.freed_bytes)} freed).`
            : `Deleted ${res.deleted_runs} old runs and ${res.deleted_artifacts} artifacts (${formatBytes(res.freed_bytes)} freed).`
        )
        onCleanedUp()
        // Refresh stats
        api.runs
          .getRetention()
          .then(applyLoadedSettings)
          .catch(() => {})
      } else {
        setStatusMessage(
          `Dry run preview: ${res.cleaned_runs} runs (${res.deleted_artifacts} artifacts, ${formatBytes(res.freed_bytes)}) eligible for cleanup.`
        )
      }
    } catch (err: unknown) {
      setErrorMessage(err instanceof Error ? err.message : String(err))
    } finally {
      setCleaning(false)
    }
  }

  return (
    <div className="card retention-settings-card" data-testid="retention-settings-panel">
      <div className="compare-section-header">
        <div>
          <h3 className="section-title" style={{ marginBottom: '2px' }}>
            🗄️ Retention Settings for Old Runs & Artifacts
          </h3>
          <span className="section-desc">
            Configure automatic or manual cleanup policies for historical runs and disk artifacts
          </span>
        </div>
        {onClose && (
          <button
            type="button"
            className="btn btn-sm btn-secondary"
            onClick={onClose}
            aria-label="Close retention settings"
          >
            Close ✕
          </button>
        )}
      </div>

      {stats && (
        <div className="retention-stats-bar" data-testid="retention-stats-bar">
          <span className="retention-stat-item">
            <strong>{stats.total_runs}</strong> total runs ({stats.terminal_runs} finished,{' '}
            {stats.active_runs} active)
          </span>
          <span className="retention-stat-item">
            <strong>{stats.total_artifacts}</strong> artifacts ({formatBytes(stats.artifacts_size_bytes)})
          </span>
          <span className="retention-stat-item">
            <strong>{stats.eligible_runs_count}</strong> runs eligible under saved policy
          </span>
          {lastCleanupAt && (
            <span className="retention-stat-item text-muted">
              Last cleanup: {new Date(lastCleanupAt).toLocaleString()}
            </span>
          )}
        </div>
      )}

      {errorMessage && <div className="banner error-banner">{errorMessage}</div>}
      {statusMessage && (
        <div className="banner info-banner" data-testid="retention-status-message">
          {statusMessage}
        </div>
      )}

      <div className="retention-form-grid">
        <div className="filter-group">
          <label htmlFor="retention-max-age-days" className="filter-label">
            Max Age (days):
          </label>
          <input
            id="retention-max-age-days"
            type="number"
            min="0"
            placeholder="Unlimited (e.g. 30)"
            className="filter-select"
            value={maxAgeDays}
            onChange={(e) => setMaxAgeDays(e.target.value)}
          />
        </div>

        <div className="filter-group">
          <label htmlFor="retention-max-runs" className="filter-label">
            Max Runs to Keep:
          </label>
          <input
            id="retention-max-runs"
            type="number"
            min="0"
            placeholder="Unlimited (e.g. 100)"
            className="filter-select"
            value={maxRuns}
            onChange={(e) => setMaxRuns(e.target.value)}
          />
        </div>

        <div className="filter-group">
          <label htmlFor="retention-mode-select" className="filter-label">
            Cleanup Target:
          </label>
          <select
            id="retention-mode-select"
            aria-label="Cleanup Target"
            className="filter-select"
            value={artifactsOnly ? 'artifacts_only' : 'runs_and_artifacts'}
            onChange={(e) => setArtifactsOnly(e.target.value === 'artifacts_only')}
          >
            <option value="runs_and_artifacts">Delete Old Runs & Artifacts</option>
            <option value="artifacts_only">
              Delete Artifacts Only (Keep Run Metrics for Trends)
            </option>
          </select>
        </div>

        <div className="retention-checkboxes">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={enabled}
              onChange={(e) => setEnabled(e.target.checked)}
              aria-label="Enable automatic retention cleanup"
            />
            <span>Auto-cleanup after runs finish</span>
          </label>

          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={keepFailed}
              onChange={(e) => setKeepFailed(e.target.checked)}
              aria-label="Keep failed runs"
            />
            <span>Preserve failed/errored runs</span>
          </label>
        </div>
      </div>

      <div className="retention-actions-row">
        <button
          type="button"
          className="btn btn-sm btn-primary"
          disabled={saving || cleaning}
          data-testid="retention-save-btn"
          onClick={() => handleSave(false)}
        >
          {saving ? 'Saving...' : '💾 Save Retention Policy'}
        </button>
        <button
          type="button"
          className="btn btn-sm btn-secondary"
          disabled={saving || cleaning}
          data-testid="retention-preview-btn"
          onClick={() => handleCleanup(true)}
        >
          🔍 Preview Cleanup (Dry Run)
        </button>
        <button
          type="button"
          className="btn btn-sm btn-secondary"
          disabled={saving || cleaning}
          data-testid="retention-cleanup-now-btn"
          onClick={() => handleCleanup(false)}
        >
          🧹 Clean Up Old Runs Now
        </button>
      </div>

      {cleanupResult && (
        <div className="retention-cleanup-summary" data-testid="retention-cleanup-summary">
          <span className="font-mono text-sm">
            {cleanupResult.dry_run ? '[DRY RUN] ' : '[EXECUTED] '}
            Cleaned runs: <strong>{cleanupResult.cleaned_runs}</strong> • Deleted DB runs:{' '}
            <strong>{cleanupResult.deleted_runs}</strong> • Deleted artifacts:{' '}
            <strong>{cleanupResult.deleted_artifacts}</strong> • Space freed:{' '}
            <strong>{formatBytes(cleanupResult.freed_bytes)}</strong>
          </span>
        </div>
      )}
    </div>
  )
}
