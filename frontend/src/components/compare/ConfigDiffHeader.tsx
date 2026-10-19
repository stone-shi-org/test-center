import { useState } from 'react'
import type { CompareConfigDiff, ComparedRunMeta } from '../../api/types.ts'
import { Link } from '../../router/index.ts'
import { Badge } from '../common/Badge.tsx'

export function ConfigDiffHeader({
  runs,
  configDiff,
  onSelectBaseline,
  onRemoveRun,
}: {
  runs: ComparedRunMeta[]
  configDiff: CompareConfigDiff
  onSelectBaseline: (runId: number) => void
  onRemoveRun?: (runId: number) => void
}) {
  const [showAllConfig, setShowAllConfig] = useState(false)

  const formatConfigVal = (val: unknown): string => {
    if (val === undefined || val === null) return '—'
    if (typeof val === 'string') return val
    if (typeof val === 'number' || typeof val === 'boolean') return String(val)
    try {
      return JSON.stringify(val)
    } catch {
      return String(val)
    }
  }

  // Collect all config keys across compared runs
  const allConfigKeys: string[] = []
  const seenKeys = new Set<string>()
  for (const k of configDiff.config_keys_changed || []) {
    if (!seenKeys.has(k)) {
      allConfigKeys.push(k)
      seenKeys.add(k)
    }
  }
  for (const r of runs) {
    const cfg = r.config || {}
    for (const k of Object.keys(cfg)) {
      if (!seenKeys.has(k)) {
        allConfigKeys.push(k)
        seenKeys.add(k)
      }
    }
  }

  const changedKeySet = new Set(configDiff.config_keys_changed || [])
  const visibleConfigKeys = showAllConfig
    ? allConfigKeys
    : allConfigKeys.filter((k) => changedKeySet.has(k))

  const hasAnyDiff =
    configDiff.targets_differ ||
    configDiff.models_differ ||
    configDiff.suites_differ ||
    changedKeySet.size > 0

  return (
    <div className="card config-diff-header-card" data-testid="config-diff-header">
      <div className="config-diff-top">
        <div>
          <h3 className="section-title" style={{ marginBottom: '4px' }}>
            Config &amp; Target Comparison
          </h3>
          <div className="config-diff-badges">
            <span
              className={`diff-pill ${configDiff.targets_differ ? 'diff-pill-warn' : 'diff-pill-neutral'}`}
            >
              {configDiff.targets_differ ? '⚡ Targets Differ' : '✓ Same Target'}
            </span>
            <span
              className={`diff-pill ${configDiff.models_differ ? 'diff-pill-warn' : 'diff-pill-neutral'}`}
            >
              {configDiff.models_differ ? '⚡ Models Differ' : '✓ Same Model'}
            </span>
            <span
              className={`diff-pill ${configDiff.suites_differ ? 'diff-pill-warn' : 'diff-pill-neutral'}`}
            >
              {configDiff.suites_differ ? '⚡ Suites Differ' : '✓ Same Suites'}
            </span>
            <span
              className={`diff-pill ${changedKeySet.size > 0 ? 'diff-pill-warn' : 'diff-pill-neutral'}`}
            >
              {changedKeySet.size > 0
                ? `⚡ ${changedKeySet.size} Config Key${changedKeySet.size === 1 ? '' : 's'} Changed`
                : '✓ Identical Config'}
            </span>
          </div>
        </div>

        <div className="config-diff-actions">
          <button
            type="button"
            className="btn btn-sm btn-secondary"
            onClick={() => setShowAllConfig((prev) => !prev)}
          >
            {showAllConfig ? 'Show Changed Config Only' : 'Show All Config & Target Fields'}
          </button>
        </div>
      </div>

      {/* Side-by-side Run Metadata Cards */}
      <div className="compare-runs-grid">
        {runs.map((r) => (
          <div
            key={`${r.run_id}-${r.result_id}`}
            className={`compare-run-meta-card ${r.is_baseline ? 'is-baseline' : ''}`}
            data-testid={`compare-run-card-${r.run_id}`}
          >
            <div className="compare-run-card-header">
              <div className="compare-run-title-group">
                <Link href={`/runs/${r.run_id}`} className="font-mono font-bold table-link">
                  Run #{r.run_id}
                </Link>
                {r.baselines &&
                  r.baselines.map((b) => (
                    <span
                      key={b.id}
                      className="baseline-name-badge"
                      data-testid={`card-named-baseline-${r.run_id}-${b.name}`}
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '3px',
                        padding: '1px 6px',
                        borderRadius: '999px',
                        background: 'rgba(245, 158, 11, 0.15)',
                        border: '1px solid rgba(245, 158, 11, 0.4)',
                        color: '#fbbf24',
                        fontSize: '11px',
                        fontWeight: 600,
                      }}
                    >
                      ★ {b.name}
                    </span>
                  ))}
                {r.is_baseline && (
                  <span className="baseline-badge" data-testid={`baseline-badge-${r.run_id}`}>
                    REFERENCE RUN
                  </span>
                )}
                <Badge status={r.status} size="sm" />
              </div>

              <div className="compare-run-card-actions">
                {!r.is_baseline && (
                  <button
                    type="button"
                    className="btn btn-sm btn-secondary"
                    onClick={() => onSelectBaseline(r.run_id)}
                    title={`Set Run #${r.run_id} as Reference run`}
                  >
                    Set Reference run
                  </button>
                )}
                {onRemoveRun && runs.length > 2 && (
                  <button
                    type="button"
                    className="action-btn danger"
                    onClick={() => onRemoveRun(r.run_id)}
                    title={`Remove Run #${r.run_id} from comparison`}
                    aria-label={`Remove Run #${r.run_id}`}
                  >
                    ×
                  </button>
                )}
              </div>
            </div>

            <div className="compare-run-meta-rows">
              <div className="compare-meta-row">
                <span className="compare-meta-label">Tester:</span>
                <span className="kind-badge">{r.tester}</span>
              </div>
              <div
                className={`compare-meta-row ${configDiff.targets_differ ? 'diff-highlight' : ''}`}
              >
                <span className="compare-meta-label">Target:</span>
                <span className="font-mono compare-meta-val">
                  {r.target_name || 'None'}
                  {r.base_url ? (
                    <span className="text-muted"> ({r.base_url})</span>
                  ) : null}
                </span>
              </div>
              <div
                className={`compare-meta-row ${configDiff.models_differ ? 'diff-highlight' : ''}`}
              >
                <span className="compare-meta-label">Model:</span>
                <span className="font-mono font-semibold compare-meta-val">
                  {r.model || '—'}
                </span>
              </div>
              <div
                className={`compare-meta-row ${configDiff.suites_differ ? 'diff-highlight' : ''}`}
              >
                <span className="compare-meta-label">Suites:</span>
                <span className="font-mono compare-meta-val">
                  {r.suites && r.suites.length > 0 ? r.suites.join(', ') : 'all'}
                </span>
              </div>
              {r.summary && (
                <div className="compare-meta-row">
                  <span className="compare-meta-label">Summary:</span>
                  <span className="font-mono text-muted compare-meta-val">{r.summary}</span>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Config & Target Diff Table */}
      {(hasAnyDiff || showAllConfig) && (
        <div className="table-responsive config-diff-table-wrapper">
          <table className="data-table config-diff-table" data-testid="config-diff-table">
            <thead>
              <tr>
                <th>Field / Config Key</th>
                <th>Diff</th>
                {runs.map((r) => (
                  <th key={`${r.run_id}-${r.result_id}`}>
                    Run #{r.run_id}
                    {r.baselines && r.baselines.length > 0
                      ? ` ★ ${r.baselines.map((b) => b.name).join(', ')}`
                      : ''}
                    {r.is_baseline ? ' (Reference run)' : ''}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(configDiff.targets_differ || showAllConfig) && (
                <tr className={configDiff.targets_differ ? 'config-row-changed' : ''}>
                  <td className="font-mono font-semibold">target</td>
                  <td>
                    <span
                      className={`diff-pill ${configDiff.targets_differ ? 'diff-pill-warn' : 'diff-pill-neutral'}`}
                    >
                      {configDiff.targets_differ ? 'CHANGED' : 'SAME'}
                    </span>
                  </td>
                  {runs.map((r) => (
                    <td key={`${r.run_id}-target`} className="font-mono">
                      {r.target_name || 'None'}{' '}
                      {r.base_url ? <span className="text-muted">({r.base_url})</span> : null}
                    </td>
                  ))}
                </tr>
              )}

              {(configDiff.models_differ || showAllConfig) && (
                <tr className={configDiff.models_differ ? 'config-row-changed' : ''}>
                  <td className="font-mono font-semibold">model</td>
                  <td>
                    <span
                      className={`diff-pill ${configDiff.models_differ ? 'diff-pill-warn' : 'diff-pill-neutral'}`}
                    >
                      {configDiff.models_differ ? 'CHANGED' : 'SAME'}
                    </span>
                  </td>
                  {runs.map((r) => (
                    <td key={`${r.run_id}-model`} className="font-mono">
                      {r.model || '—'}
                    </td>
                  ))}
                </tr>
              )}

              {(configDiff.suites_differ || showAllConfig) && (
                <tr className={configDiff.suites_differ ? 'config-row-changed' : ''}>
                  <td className="font-mono font-semibold">suites</td>
                  <td>
                    <span
                      className={`diff-pill ${configDiff.suites_differ ? 'diff-pill-warn' : 'diff-pill-neutral'}`}
                    >
                      {configDiff.suites_differ ? 'CHANGED' : 'SAME'}
                    </span>
                  </td>
                  {runs.map((r) => (
                    <td key={`${r.run_id}-suites`} className="font-mono">
                      {r.suites && r.suites.length > 0 ? r.suites.join(', ') : 'all'}
                    </td>
                  ))}
                </tr>
              )}

              {visibleConfigKeys
                .filter((k) => k !== 'model' || (!configDiff.models_differ && !showAllConfig))
                .map((key) => {
                  const isChanged = changedKeySet.has(key)
                  return (
                    <tr key={key} className={isChanged ? 'config-row-changed' : ''}>
                      <td className="font-mono font-semibold">config.{key}</td>
                      <td>
                        <span
                          className={`diff-pill ${isChanged ? 'diff-pill-warn' : 'diff-pill-neutral'}`}
                        >
                          {isChanged ? 'CHANGED' : 'SAME'}
                        </span>
                      </td>
                      {runs.map((r) => (
                        <td key={`${r.run_id}-cfg-${key}`} className="font-mono">
                          {formatConfigVal((r.config || {})[key])}
                        </td>
                      ))}
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
