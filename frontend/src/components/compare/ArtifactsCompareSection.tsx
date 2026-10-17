import { useState } from 'react'
import { api } from '../../api/client.ts'
import type { CompareArtifactItem, ComparedRunMeta } from '../../api/types.ts'

function formatSize(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return '—'
  if (bytes < 1024) return `${bytes} B`
  return `${(bytes / 1024).toFixed(1)} KB`
}

function isTextPreviewable(mime: string, name: string): boolean {
  const lowerMime = (mime || '').toLowerCase()
  const lowerName = (name || '').toLowerCase()
  return (
    lowerMime.startsWith('text/') ||
    lowerMime.includes('json') ||
    lowerMime.includes('xml') ||
    lowerMime.includes('yaml') ||
    lowerMime.includes('csv') ||
    lowerName.endsWith('.txt') ||
    lowerName.endsWith('.json') ||
    lowerName.endsWith('.log') ||
    lowerName.endsWith('.md') ||
    lowerName.endsWith('.csv')
  )
}

export function ArtifactsCompareSection({
  runs,
}: {
  runs: ComparedRunMeta[]
}) {
  const [activeArtifactName, setActiveArtifactName] = useState<string | null>(null)
  const [contentCache, setContentCache] = useState<Record<string, string>>({})
  const [loadingMap, setLoadingMap] = useState<Record<string, boolean>>({})
  const [errorMap, setErrorMap] = useState<Record<string, string>>({})

  // Collect ordered union of artifact names across all compared runs
  const artifactNames: string[] = []
  const seenNames = new Set<string>()
  for (const r of runs) {
    for (const art of r.artifacts || []) {
      if (!seenNames.has(art.name)) {
        artifactNames.push(art.name)
        seenNames.add(art.name)
      }
    }
  }

  const findArtifact = (
    r: ComparedRunMeta,
    name: string
  ): CompareArtifactItem | undefined =>
    (r.artifacts || []).find((a) => a.name === name)

  const handleToggleSideBySide = async (name: string) => {
    if (activeArtifactName === name) {
      setActiveArtifactName(null)
      return
    }
    setActiveArtifactName(name)

    // Fetch text content for each run's matching artifact if not already cached
    for (const r of runs) {
      const art = findArtifact(r, name)
      if (!art || !isTextPreviewable(art.mime, art.name)) continue
      const cacheKey = `${r.run_id}:${art.id}`
      if (contentCache[cacheKey] !== undefined || loadingMap[cacheKey]) continue

      setLoadingMap((prev) => ({ ...prev, [cacheKey]: true }))
      try {
        const text = await api.runs.getArtifactContent(r.run_id, art.id)
        setContentCache((prev) => ({ ...prev, [cacheKey]: text }))
      } catch (err: unknown) {
        setErrorMap((prev) => ({
          ...prev,
          [cacheKey]: err instanceof Error ? err.message : String(err),
        }))
      } finally {
        setLoadingMap((prev) => ({ ...prev, [cacheKey]: false }))
      }
    }
  }

  return (
    <div className="card compare-section-card" data-testid="artifacts-compare-section">
      <div className="compare-section-header">
        <div>
          <h3 className="section-title" style={{ marginBottom: '2px' }}>
            Artifacts Side by Side ({artifactNames.length})
          </h3>
          <span className="section-desc">
            Compare generated reports, logs, and output artifacts across runs side by side
          </span>
        </div>
      </div>

      {artifactNames.length === 0 ? (
        <div className="text-muted empty-inline-msg">
          No artifacts recorded for the selected runs.
        </div>
      ) : (
        <>
          <div className="table-responsive">
            <table className="data-table artifacts-compare-table" data-testid="artifacts-compare-table">
              <thead>
                <tr>
                  <th>Artifact Name</th>
                  {runs.map((r) => (
                    <th key={`${r.run_id}-${r.result_id}`}>
                      Run #{r.run_id} {r.is_baseline ? '(Baseline)' : ''}
                    </th>
                  ))}
                  <th>Side-by-Side</th>
                </tr>
              </thead>
              <tbody>
                {artifactNames.map((name) => {
                  const isPreviewOpen = activeArtifactName === name
                  return (
                    <tr key={name} data-testid={`artifact-row-${name}`}>
                      <td className="font-mono font-bold">{name}</td>
                      {runs.map((r) => {
                        const art = findArtifact(r, name)
                        if (!art) {
                          return (
                            <td
                              key={`${r.run_id}-${r.result_id}`}
                              className="text-muted font-mono"
                            >
                              — (missing)
                            </td>
                          )
                        }
                        const downloadHref =
                          art.download_url ||
                          api.runs.getArtifactDownloadUrl(r.run_id, art.id)

                        return (
                          <td key={`${r.run_id}-${r.result_id}`}>
                            <div className="artifact-cell-box">
                              <div className="artifact-cell-meta font-mono">
                                <span className="text-muted">{art.mime}</span>
                                <span>•</span>
                                <span>{formatSize(art.size_bytes)}</span>
                              </div>
                              <a
                                href={downloadHref}
                                className="btn btn-sm btn-secondary"
                                download={art.name}
                                target="_blank"
                                rel="noreferrer"
                              >
                                ⬇ Download
                              </a>
                            </div>
                          </td>
                        )
                      })}
                      <td>
                        <button
                          type="button"
                          className={`btn btn-sm ${isPreviewOpen ? 'btn-primary' : 'btn-secondary'}`}
                          onClick={() => handleToggleSideBySide(name)}
                          data-testid={`compare-artifact-btn-${name}`}
                        >
                          {isPreviewOpen ? 'Hide Side by Side' : '👁 Compare Side by Side'}
                        </button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* Side-by-Side Preview Panel */}
          {activeArtifactName && (
            <div
              className="artifact-side-by-side-drawer"
              data-testid={`artifact-side-by-side-${activeArtifactName}`}
            >
              {(() => {
                const baselineRun = runs.find((r) => r.is_baseline) || runs[0]
                const baselineArt = baselineRun
                  ? findArtifact(baselineRun, activeArtifactName)
                  : undefined
                const baselineContent =
                  baselineRun && baselineArt
                    ? contentCache[`${baselineRun.run_id}:${baselineArt.id}`]
                    : undefined
                const baselineLines =
                  baselineContent !== undefined ? baselineContent.split('\n') : []

                const loadedContents = runs
                  .map((r) => {
                    const a = findArtifact(r, activeArtifactName)
                    return a ? contentCache[`${r.run_id}:${a.id}`] : undefined
                  })
                  .filter((c): c is string => c !== undefined)
                const contentsDiffer =
                  loadedContents.length > 1 && new Set(loadedContents).size > 1

                return (
                  <>
                    <div className="drawer-header">
                      <div>
                        <h4>
                          Side-by-Side Artifact: <code>{activeArtifactName}</code>
                        </h4>
                      </div>
                      <div className="artifact-drawer-badges">
                        {loadedContents.length > 1 && (
                          <span
                            className={`diff-pill ${
                              contentsDiffer ? 'diff-pill-warn' : 'diff-pill-neutral'
                            }`}
                          >
                            {contentsDiffer ? '⚡ Contents Differ' : '✓ Contents Identical'}
                          </span>
                        )}
                        <button
                          type="button"
                          className="btn btn-sm btn-secondary"
                          onClick={() => setActiveArtifactName(null)}
                        >
                          Close
                        </button>
                      </div>
                    </div>

                    <div
                      className="artifacts-side-by-side-grid"
                      style={{
                        gridTemplateColumns: `repeat(${Math.min(runs.length, 3)}, minmax(0, 1fr))`,
                      }}
                    >
                      {runs.map((r) => {
                        const art = findArtifact(r, activeArtifactName)
                        if (!art) {
                          return (
                            <div
                              key={`${r.run_id}-${r.result_id}`}
                              className="artifact-preview-col"
                            >
                              <div className="artifact-col-header">
                                <span className="font-mono font-bold">
                                  Run #{r.run_id} {r.is_baseline ? '(Baseline)' : ''}
                                </span>
                              </div>
                              <div className="json-block font-mono text-muted">
                                Artifact not present in Run #{r.run_id}.
                              </div>
                            </div>
                          )
                        }

                        const cacheKey = `${r.run_id}:${art.id}`
                        const isLoading = loadingMap[cacheKey]
                        const err = errorMap[cacheKey]
                        const content = contentCache[cacheKey]
                        const downloadHref =
                          art.download_url ||
                          api.runs.getArtifactDownloadUrl(r.run_id, art.id)

                        return (
                          <div
                            key={`${r.run_id}-${r.result_id}`}
                            className="artifact-preview-col"
                            data-testid={`artifact-col-${r.run_id}`}
                          >
                            <div className="artifact-col-header">
                              <div>
                                <span className="font-mono font-bold">
                                  Run #{r.run_id} {r.is_baseline ? '(Baseline)' : ''}
                                </span>
                                {r.model && (
                                  <span className="text-muted font-mono"> • {r.model}</span>
                                )}
                              </div>
                              <a
                                href={downloadHref}
                                className="btn-link font-mono"
                                download={art.name}
                                target="_blank"
                                rel="noreferrer"
                              >
                                ⬇ {formatSize(art.size_bytes)}
                              </a>
                            </div>

                            {art.mime.startsWith('audio/') ? (
                              <div className="artifact-media-box">
                                <audio controls src={downloadHref} style={{ width: '100%' }} />
                              </div>
                            ) : art.mime.startsWith('image/') ? (
                              <div className="artifact-media-box">
                                <img
                                  src={downloadHref}
                                  alt={art.name}
                                  style={{ maxWidth: '100%', borderRadius: '4px' }}
                                />
                              </div>
                            ) : isLoading ? (
                              <div className="json-block font-mono text-muted">
                                Loading artifact content...
                              </div>
                            ) : err ? (
                              <div className="json-block font-mono text-muted">
                                Preview unavailable ({err}). Use Download link above.
                              </div>
                            ) : content !== undefined ? (
                              <div className="json-block font-mono artifact-diff-lines">
                                {content.split('\n').map((line, idx) => {
                                  const lineDiffers =
                                    !r.is_baseline &&
                                    baselineLines.length > 0 &&
                                    baselineLines[idx] !== line
                                  return (
                                    <div
                                      key={idx}
                                      className={`artifact-line ${
                                        lineDiffers ? 'artifact-line-diff' : ''
                                      }`}
                                    >
                                      <span className="artifact-line-num">{idx + 1}</span>
                                      <span className="artifact-line-text">{line || ' '}</span>
                                    </div>
                                  )
                                })}
                              </div>
                            ) : (
                              <div className="json-block font-mono text-muted">
                                Binary artifact ({art.mime}). Use Download link above.
                              </div>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  </>
                )
              })()}
            </div>
          )}
        </>
      )}
    </div>
  )
}
