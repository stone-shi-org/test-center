import { useEffect, useMemo, useRef, useState } from 'react'
import { api, type Artifact, type ArtifactKind } from '../../api/index.ts'

export function detectArtifactKind(artifact: {
  name: string
  mime?: string
  kind?: string
  viewer_type?: string
}): ArtifactKind {
  const declared = (artifact.kind || artifact.viewer_type || '').toLowerCase()
  if (
    declared === 'html' ||
    declared === 'json' ||
    declared === 'audio' ||
    declared === 'log' ||
    declared === 'image' ||
    declared === 'binary'
  ) {
    return declared
  }

  const lowerMime = (artifact.mime || '').toLowerCase()
  const lowerName = (artifact.name || '').toLowerCase()

  if (
    lowerName.endsWith('.html') ||
    lowerName.endsWith('.htm') ||
    lowerName.endsWith('.xhtml') ||
    lowerMime.includes('text/html') ||
    lowerMime.includes('application/xhtml+xml')
  ) {
    return 'html'
  }

  if (
    lowerName.endsWith('.json') ||
    lowerName.endsWith('.jsonl') ||
    lowerName.endsWith('.ndjson') ||
    lowerMime.includes('json')
  ) {
    return 'json'
  }

  if (
    lowerMime.startsWith('audio/') ||
    lowerName.endsWith('.wav') ||
    lowerName.endsWith('.mp3') ||
    lowerName.endsWith('.ogg') ||
    lowerName.endsWith('.oga') ||
    lowerName.endsWith('.opus') ||
    lowerName.endsWith('.flac') ||
    lowerName.endsWith('.m4a') ||
    lowerName.endsWith('.aac')
  ) {
    return 'audio'
  }

  if (
    lowerMime.startsWith('image/') ||
    lowerName.endsWith('.png') ||
    lowerName.endsWith('.jpg') ||
    lowerName.endsWith('.jpeg') ||
    lowerName.endsWith('.gif') ||
    lowerName.endsWith('.svg') ||
    lowerName.endsWith('.webp')
  ) {
    return 'image'
  }

  if (
    lowerMime.startsWith('text/') ||
    lowerMime.includes('xml') ||
    lowerMime.includes('yaml') ||
    lowerMime.includes('csv') ||
    lowerName.endsWith('.log') ||
    lowerName.endsWith('.txt') ||
    lowerName.endsWith('.out') ||
    lowerName.endsWith('.err') ||
    lowerName.endsWith('.md') ||
    lowerName.endsWith('.csv') ||
    lowerName.endsWith('.tsv') ||
    lowerName.endsWith('.yaml') ||
    lowerName.endsWith('.yml') ||
    lowerName.endsWith('.xml')
  ) {
    return 'log'
  }

  return 'binary'
}

export function formatArtifactSize(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

export function formatArtifactDisplayPath(runId: number, artifact: Artifact): string {
  const rawPath = artifact.path || ''
  const marker = `artifacts/${runId}/`
  const idx = rawPath.indexOf(marker)
  if (idx !== -1) {
    return `data/${rawPath.slice(idx)}`
  }
  return `data/artifacts/${runId}/${artifact.name}`
}

function getKindBadgeLabel(kind: ArtifactKind): { icon: string; label: string; className: string } {
  switch (kind) {
    case 'html':
      return { icon: '🌐', label: 'HTML', className: 'artifact-kind-html' }
    case 'json':
      return { icon: '{}', label: 'JSON', className: 'artifact-kind-json' }
    case 'audio':
      return { icon: '🔊', label: 'AUDIO', className: 'artifact-kind-audio' }
    case 'log':
      return { icon: '📜', label: 'LOG', className: 'artifact-kind-log' }
    case 'image':
      return { icon: '🖼', label: 'IMAGE', className: 'artifact-kind-image' }
    default:
      return { icon: '📦', label: 'BINARY', className: 'artifact-kind-binary' }
  }
}

// ============================================================================
// 1. Safe Sandboxed HTML Preview
// ============================================================================

function HtmlArtifactPreview({
  artifact,
  content,
  previewUrl,
}: {
  artifact: Artifact
  content: string
  previewUrl: string
}) {
  const [viewMode, setViewMode] = useState<'rendered' | 'source'>('rendered')
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // ignore clipboard errors
    }
  }

  const lines = useMemo(() => content.split('\n'), [content])

  return (
    <div className="artifact-viewer-pane" data-testid="artifact-html-viewer">
      <div className="artifact-subtoolbar">
        <div className="artifact-subtoolbar-left">
          <span className="sandbox-security-pill" data-testid="html-sandbox-badge">
            🔒 Sandboxed HTML (scripts disabled)
          </span>
          <span className="text-muted font-mono" style={{ fontSize: '11px' }}>
            {lines.length} lines
          </span>
        </div>
        <div className="artifact-subtoolbar-right">
          <div className="btn-group">
            <button
              type="button"
              className={`btn btn-sm ${viewMode === 'rendered' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setViewMode('rendered')}
              data-testid="html-mode-rendered"
            >
              Rendered Preview
            </button>
            <button
              type="button"
              className={`btn btn-sm ${viewMode === 'source' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setViewMode('source')}
              data-testid="html-mode-source"
            >
              HTML Source
            </button>
          </div>
          <button
            type="button"
            className="btn btn-sm btn-secondary"
            onClick={handleCopy}
            data-testid="html-copy-btn"
          >
            {copied ? '✓ Copied' : 'Copy HTML'}
          </button>
        </div>
      </div>

      {viewMode === 'rendered' ? (
        <div className="artifact-iframe-container">
          <iframe
            data-testid="artifact-html-iframe"
            title={`Sandboxed HTML preview of ${artifact.name}`}
            sandbox=""
            referrerPolicy="no-referrer"
            srcDoc={content}
            src={previewUrl}
            className="artifact-sandbox-iframe"
          />
        </div>
      ) : (
        <div className="log-terminal" data-testid="artifact-html-source">
          <div className="log-lines">
            {lines.map((line, idx) => (
              <div key={idx} className="log-line">
                <span className="log-line-num">{idx + 1}</span>
                <span className="log-line-content">{line || ' '}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ============================================================================
// 2. Interactive JSON Tree Viewer
// ============================================================================

function jsonContainsSearch(value: unknown, queryLower: string, keyName?: string): boolean {
  if (!queryLower) return true
  if (keyName && keyName.toLowerCase().includes(queryLower)) return true
  if (value === null || value === undefined) {
    return 'null'.includes(queryLower)
  }
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value).toLowerCase().includes(queryLower)
  }
  if (Array.isArray(value)) {
    return value.some((item, idx) => jsonContainsSearch(item, queryLower, String(idx)))
  }
  if (typeof value === 'object') {
    return Object.entries(value as Record<string, unknown>).some(([k, v]) =>
      jsonContainsSearch(v, queryLower, k)
    )
  }
  return false
}

function JsonTreeNode({
  name,
  value,
  path,
  depth,
  expandState,
  searchQuery,
}: {
  name?: string
  value: unknown
  path: string
  depth: number
  expandState: { version: number; expandAll: boolean | null }
  searchQuery: string
}) {
  const isArray = Array.isArray(value)
  const isObject = value !== null && typeof value === 'object' && !isArray
  const isCollapsible = isArray || isObject

  const queryLower = searchQuery.trim().toLowerCase()
  const defaultExpanded = depth < 2
  const [manualOpen, setManualOpen] = useState<{ version: number; open: boolean }>({
    version: expandState.version,
    open: defaultExpanded,
  })

  const isExpanded = useMemo(() => {
    if (queryLower) return true
    if (manualOpen.version === expandState.version && expandState.expandAll === null) {
      return manualOpen.open
    }
    if (manualOpen.version === expandState.version && manualOpen.open !== defaultExpanded) {
      return manualOpen.open
    }
    if (expandState.expandAll !== null) {
      return expandState.expandAll
    }
    return defaultExpanded
  }, [queryLower, manualOpen, expandState, defaultExpanded])

  const handleToggle = () => {
    setManualOpen({
      version: expandState.version,
      open: !isExpanded,
    })
  }

  if (queryLower && !jsonContainsSearch(value, queryLower, name)) {
    return null
  }

  const keyHighlighted =
    queryLower && name && name.toLowerCase().includes(queryLower)

  if (!isCollapsible) {
    let formattedVal = String(value)
    let valClass = 'json-val-primitive'
    if (value === null) {
      formattedVal = 'null'
      valClass = 'json-val-null'
    } else if (typeof value === 'string') {
      formattedVal = `"${value}"`
      valClass = 'json-val-string'
    } else if (typeof value === 'number') {
      valClass = 'json-val-number'
    } else if (typeof value === 'boolean') {
      valClass = 'json-val-boolean'
    }

    const valHighlighted =
      queryLower && formattedVal.toLowerCase().includes(queryLower)

    return (
      <div className="json-tree-leaf" data-testid={`json-leaf-${path}`}>
        {name !== undefined && (
          <>
            <span className={`json-tree-key ${keyHighlighted ? 'json-match-highlight' : ''}`}>
              {name}
            </span>
            <span className="json-tree-colon">: </span>
          </>
        )}
        <span className={`${valClass} ${valHighlighted ? 'json-match-highlight' : ''}`}>
          {formattedVal}
        </span>
      </div>
    )
  }

  const entries: Array<[string, unknown]> = isArray
    ? (value as unknown[]).map((item, idx) => [String(idx), item])
    : Object.entries(value as Record<string, unknown>)

  const bracketOpen = isArray ? '[' : '{'
  const bracketClose = isArray ? ']' : '}'
  const countLabel = isArray
    ? `${entries.length} ${entries.length === 1 ? 'item' : 'items'}`
    : `${entries.length} ${entries.length === 1 ? 'key' : 'keys'}`

  return (
    <div className="json-tree-branch" data-testid={`json-branch-${path}`}>
      <div className="json-tree-branch-header">
        <button
          type="button"
          className="json-tree-toggle-btn"
          onClick={handleToggle}
          data-testid={`json-node-toggle-${path}`}
          aria-label={isExpanded ? `Collapse ${name || path}` : `Expand ${name || path}`}
        >
          <span className="json-tree-arrow">{isExpanded ? '▼' : '▶'}</span>
          {name !== undefined && (
            <>
              <span className={`json-tree-key ${keyHighlighted ? 'json-match-highlight' : ''}`}>
                {name}
              </span>
              <span className="json-tree-colon">: </span>
            </>
          )}
          <span className="json-tree-bracket">{bracketOpen}</span>
          {!isExpanded && (
            <span className="json-tree-collapsed-summary">
              … {bracketClose} <span className="json-tree-count">({countLabel})</span>
            </span>
          )}
          {isExpanded && (
            <span className="json-tree-count">
              {countLabel}
            </span>
          )}
        </button>
      </div>

      {isExpanded && (
        <div className="json-tree-children">
          {entries.length === 0 ? (
            <div className="json-tree-empty text-muted">empty</div>
          ) : (
            entries.map(([k, v]) => (
              <JsonTreeNode
                key={k}
                name={k}
                value={v}
                path={`${path}.${k}`}
                depth={depth + 1}
                expandState={expandState}
                searchQuery={searchQuery}
              />
            ))
          )}
          <div className="json-tree-closing-bracket">{bracketClose}</div>
        </div>
      )}
    </div>
  )
}

function JsonArtifactViewer({
  artifact,
  content,
}: {
  artifact: Artifact
  content: string
}) {
  const [viewMode, setViewMode] = useState<'tree' | 'raw'>('tree')
  const [searchQuery, setSearchQuery] = useState('')
  const [expandState, setExpandState] = useState<{
    version: number
    expandAll: boolean | null
  }>({ version: 0, expandAll: null })
  const [copied, setCopied] = useState(false)

  const parsed = useMemo(() => {
    try {
      if (
        artifact.name.toLowerCase().endsWith('.jsonl') ||
        artifact.name.toLowerCase().endsWith('.ndjson')
      ) {
        const rows = content
          .split('\n')
          .filter((l) => l.trim().length > 0)
          .map((l) => JSON.parse(l))
        return { ok: true as const, data: rows, error: null }
      }
      return { ok: true as const, data: JSON.parse(content), error: null }
    } catch (err: unknown) {
      return {
        ok: false as const,
        data: null,
        error: err instanceof Error ? err.message : String(err),
      }
    }
  }, [artifact.name, content])

  const formattedRaw = useMemo(() => {
    if (parsed.ok) {
      return JSON.stringify(parsed.data, null, 2)
    }
    return content
  }, [parsed, content])

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(formattedRaw)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // ignore
    }
  }

  return (
    <div className="artifact-viewer-pane" data-testid="artifact-json-viewer">
      <div className="artifact-subtoolbar">
        <div className="artifact-subtoolbar-left">
          <input
            type="text"
            className="log-filter-input"
            style={{ width: '210px' }}
            placeholder="Search keys or values..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            data-testid="json-tree-search"
          />
          {viewMode === 'tree' && parsed.ok && (
            <div className="btn-group">
              <button
                type="button"
                className="btn btn-sm btn-secondary"
                onClick={() =>
                  setExpandState((prev) => ({ version: prev.version + 1, expandAll: true }))
                }
                data-testid="json-expand-all"
              >
                Expand All
              </button>
              <button
                type="button"
                className="btn btn-sm btn-secondary"
                onClick={() =>
                  setExpandState((prev) => ({ version: prev.version + 1, expandAll: false }))
                }
                data-testid="json-collapse-all"
              >
                Collapse All
              </button>
            </div>
          )}
        </div>

        <div className="artifact-subtoolbar-right">
          <div className="btn-group">
            <button
              type="button"
              className={`btn btn-sm ${viewMode === 'tree' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setViewMode('tree')}
              data-testid="json-mode-tree"
            >
              JSON Tree
            </button>
            <button
              type="button"
              className={`btn btn-sm ${viewMode === 'raw' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setViewMode('raw')}
              data-testid="json-mode-raw"
            >
              Raw JSON
            </button>
          </div>
          <button
            type="button"
            className="btn btn-sm btn-secondary"
            onClick={handleCopy}
            data-testid="json-copy-btn"
          >
            {copied ? '✓ Copied' : 'Copy JSON'}
          </button>
        </div>
      </div>

      {!parsed.ok ? (
        <div className="artifact-parse-warning" data-testid="json-parse-error">
          <div className="banner error-banner" style={{ marginBottom: '10px' }}>
            Invalid JSON ({parsed.error}). Showing raw content below.
          </div>
          <pre className="json-block font-mono" style={{ maxHeight: '420px' }}>
            {content}
          </pre>
        </div>
      ) : viewMode === 'tree' ? (
        <div className="json-tree-container font-mono" data-testid="artifact-json-tree">
          <JsonTreeNode
            value={parsed.data}
            path="root"
            depth={0}
            expandState={expandState}
            searchQuery={searchQuery}
          />
        </div>
      ) : (
        <div className="log-terminal" data-testid="artifact-json-raw">
          <div className="log-lines">
            {formattedRaw.split('\n').map((line, idx) => (
              <div key={idx} className="log-line">
                <span className="log-line-num">{idx + 1}</span>
                <span className="log-line-content">{line || ' '}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ============================================================================
// 3. In-Page Audio Player
// ============================================================================

function AudioArtifactPlayer({
  runId,
  artifact,
  previewUrl,
  downloadUrl,
}: {
  runId: number
  artifact: Artifact
  previewUrl: string
  downloadUrl: string
}) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const [playbackRate, setPlaybackRate] = useState<number>(1)
  const [loop, setLoop] = useState<boolean>(false)

  const handleSpeedChange = (newRate: number) => {
    setPlaybackRate(newRate)
    if (audioRef.current) {
      audioRef.current.playbackRate = newRate
    }
  }

  return (
    <div className="artifact-viewer-pane" data-testid="artifact-audio-viewer">
      <div className="artifact-subtoolbar">
        <div className="artifact-subtoolbar-left">
          <span className="font-mono text-muted" style={{ fontSize: '12px' }}>
            Format: <strong>{artifact.mime || 'audio/wav'}</strong> • Size:{' '}
            <strong>{formatArtifactSize(artifact.size_bytes)}</strong>
          </span>
        </div>
        <div className="artifact-subtoolbar-right">
          <label className="filter-group" style={{ fontSize: '12px' }}>
            <span className="filter-label">Speed:</span>
            <select
              className="filter-select"
              value={String(playbackRate)}
              onChange={(e) => handleSpeedChange(parseFloat(e.target.value))}
              data-testid="audio-speed-select"
            >
              <option value="0.5">0.5x</option>
              <option value="0.75">0.75x</option>
              <option value="1">1.0x</option>
              <option value="1.25">1.25x</option>
              <option value="1.5">1.5x</option>
              <option value="2">2.0x</option>
            </select>
          </label>
          <label className="autoscroll-toggle">
            <input
              type="checkbox"
              checked={loop}
              onChange={(e) => setLoop(e.target.checked)}
              data-testid="audio-loop-toggle"
            />
            <span>Loop</span>
          </label>
        </div>
      </div>

      <div className="artifact-audio-player-card">
        <div className="artifact-audio-header">
          <span className="artifact-audio-icon">🔊</span>
          <div className="artifact-audio-meta">
            <div className="font-mono font-bold">{artifact.name}</div>
            <div className="font-mono text-muted" style={{ fontSize: '11px' }}>
              {formatArtifactDisplayPath(runId, artifact)}
            </div>
          </div>
        </div>
        <audio
          ref={audioRef}
          controls
          preload="metadata"
          src={previewUrl || downloadUrl}
          loop={loop}
          data-testid="artifact-audio-element"
          className="artifact-audio-control"
        />
      </div>
    </div>
  )
}

// ============================================================================
// 4. In-Page Log & Text Viewer
// ============================================================================

function getLogLineLevelClass(line: string): string {
  const upper = line.toUpperCase()
  if (upper.includes('[ERROR]') || upper.includes(' ERROR ') || upper.includes(' status: fail')) {
    return 'artifact-log-line-error'
  }
  if (upper.includes('[WARNING]') || upper.includes('[WARN]')) {
    return 'artifact-log-line-warn'
  }
  if (upper.includes('[PASS]') || upper.includes('-> PASS')) {
    return 'artifact-log-line-pass'
  }
  return ''
}

function LogArtifactViewer({
  content,
}: {
  content: string
}) {
  const [filterText, setFilterText] = useState('')
  const [levelFilter, setLevelFilter] = useState<'ALL' | 'INFO' | 'WARNING' | 'ERROR'>('ALL')
  const [wrapLines, setWrapLines] = useState(true)
  const [copied, setCopied] = useState(false)

  const allLines = useMemo(() => {
    const raw = content.split('\n')
    if (raw.length > 1 && raw[raw.length - 1] === '') {
      return raw.slice(0, -1)
    }
    return raw
  }, [content])

  const filteredLines = useMemo(() => {
    return allLines
      .map((text, idx) => ({ lineNumber: idx + 1, text }))
      .filter((item) => {
        if (levelFilter !== 'ALL') {
          const upper = item.text.toUpperCase()
          if (levelFilter === 'WARNING') {
            if (!upper.includes('[WARNING]') && !upper.includes('[WARN]')) return false
          } else if (!upper.includes(`[${levelFilter}]`)) {
            return false
          }
        }
        if (filterText.trim()) {
          if (!item.text.toLowerCase().includes(filterText.trim().toLowerCase())) {
            return false
          }
        }
        return true
      })
  }, [allLines, levelFilter, filterText])

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(
        filteredLines.map((item) => item.text).join('\n')
      )
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // ignore
    }
  }

  return (
    <div className="artifact-viewer-pane" data-testid="artifact-log-viewer">
      <div className="artifact-subtoolbar">
        <div className="artifact-subtoolbar-left">
          <input
            type="text"
            className="log-filter-input"
            style={{ width: '200px' }}
            placeholder="Filter log lines..."
            value={filterText}
            onChange={(e) => setFilterText(e.target.value)}
            data-testid="artifact-log-filter"
          />
          <select
            className="filter-select"
            value={levelFilter}
            onChange={(e) =>
              setLevelFilter(e.target.value as 'ALL' | 'INFO' | 'WARNING' | 'ERROR')
            }
            data-testid="artifact-log-level"
            aria-label="Log level filter"
          >
            <option value="ALL">All Levels</option>
            <option value="INFO">INFO</option>
            <option value="WARNING">WARNING</option>
            <option value="ERROR">ERROR</option>
          </select>
          <span className="log-line-count" data-testid="log-viewer-line-count">
            {filteredLines.length} / {allLines.length} lines
          </span>
        </div>

        <div className="artifact-subtoolbar-right">
          <label className="autoscroll-toggle">
            <input
              type="checkbox"
              checked={wrapLines}
              onChange={(e) => setWrapLines(e.target.checked)}
              data-testid="artifact-log-wrap"
            />
            <span>Wrap lines</span>
          </label>
          <button
            type="button"
            className="btn btn-sm btn-secondary"
            onClick={handleCopy}
            data-testid="artifact-log-copy"
          >
            {copied ? '✓ Copied' : 'Copy'}
          </button>
        </div>
      </div>

      <div className="log-terminal" data-testid="artifact-log-terminal">
        {filteredLines.length === 0 ? (
          <div className="log-empty-msg">
            {allLines.length === 0 || (allLines.length === 1 && !allLines[0])
              ? 'Artifact file is empty.'
              : 'No log lines match the current filter.'}
          </div>
        ) : (
          <div className="log-lines">
            {filteredLines.map((item) => (
              <div
                key={item.lineNumber}
                className={`log-line ${getLogLineLevelClass(item.text)} ${
                  wrapLines ? '' : 'log-line-nowrap'
                }`}
              >
                <span className="log-line-num">{item.lineNumber}</span>
                <span className="log-line-content">{item.text || ' '}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// ============================================================================
// Main ArtifactViewer Drawer & RunArtifactsSection
// ============================================================================

export function ArtifactViewer({
  runId,
  artifact,
  artifacts,
  onSelectArtifact,
  onClose,
}: {
  runId: number
  artifact: Artifact
  artifacts?: Artifact[]
  onSelectArtifact?: (artifact: Artifact) => void
  onClose?: () => void
}) {
  const kind = detectArtifactKind(artifact)
  const badge = getKindBadgeLabel(kind)
  const needsTextFetch = kind === 'html' || kind === 'json' || kind === 'log'

  const [content, setContent] = useState<string | null>(null)
  const [loading, setLoading] = useState<boolean>(needsTextFetch)
  const [error, setError] = useState<string | null>(null)

  const artifactRef = artifact.id ? artifact.id : artifact.name
  const previewUrl =
    artifact.preview_url || api.runs.getArtifactPreviewUrl(runId, artifactRef)
  const downloadUrl =
    artifact.download_url || api.runs.getArtifactDownloadUrl(runId, artifactRef)

  useEffect(() => {
    if (!needsTextFetch) {
      setContent(null)
      setLoading(false)
      setError(null)
      return
    }

    let ignore = false
    setLoading(true)
    setError(null)

    api.runs
      .getArtifactContent(runId, artifactRef)
      .then((text) => {
        if (!ignore) {
          setContent(text)
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
  }, [runId, artifactRef, needsTextFetch])

  return (
    <div className="artifact-viewer-drawer" data-testid={`artifact-viewer-${artifact.name}`}>
      <div className="artifact-viewer-header">
        <div className="artifact-viewer-title-group">
          <span className={`artifact-kind-badge ${badge.className}`}>
            {badge.icon} {badge.label}
          </span>
          <h4 className="font-mono" style={{ margin: 0 }}>
            {artifact.name}
          </h4>
          <span className="font-mono text-muted" style={{ fontSize: '12px' }}>
            ({formatArtifactDisplayPath(runId, artifact)} • {formatArtifactSize(artifact.size_bytes)})
          </span>
        </div>

        <div className="artifact-viewer-actions">
          <a
            href={downloadUrl}
            className="btn btn-sm btn-secondary"
            download={artifact.name}
            target="_blank"
            rel="noreferrer"
            data-testid="viewer-download-btn"
          >
            ⬇ Download
          </a>
          {onClose && (
            <button
              type="button"
              className="btn btn-sm btn-secondary"
              onClick={onClose}
              data-testid="viewer-close-btn"
            >
              ✕ Close Preview
            </button>
          )}
        </div>
      </div>

      {/* Quick switcher pills when multiple artifacts exist */}
      {artifacts && artifacts.length > 1 && onSelectArtifact && (
        <div className="artifact-switcher-pills">
          {artifacts.map((item) => {
            const itemKind = detectArtifactKind(item)
            const itemBadge = getKindBadgeLabel(itemKind)
            const isActive =
              (item.id && item.id === artifact.id) || item.name === artifact.name
            return (
              <button
                key={`${item.id}-${item.name}`}
                type="button"
                className={`artifact-switcher-pill ${isActive ? 'active' : ''}`}
                onClick={() => onSelectArtifact(item)}
                data-testid={`switch-artifact-pill-${item.name}`}
              >
                <span>{itemBadge.icon}</span>
                <span className="font-mono">{item.name}</span>
              </button>
            )
          })}
        </div>
      )}

      {/* Viewer Body */}
      <div className="artifact-viewer-body">
        {loading ? (
          <div className="artifact-viewer-status font-mono text-muted">
            Loading artifact preview for {artifact.name}...
          </div>
        ) : error ? (
          <div className="artifact-viewer-status" data-testid="artifact-viewer-error">
            <div className="banner error-banner">
              Failed to load preview ({error}). You can still download the file directly.
            </div>
          </div>
        ) : kind === 'html' && content !== null ? (
          <HtmlArtifactPreview
            artifact={artifact}
            content={content}
            previewUrl={previewUrl}
          />
        ) : kind === 'json' && content !== null ? (
          <JsonArtifactViewer artifact={artifact} content={content} />
        ) : kind === 'audio' ? (
          <AudioArtifactPlayer
            runId={runId}
            artifact={artifact}
            previewUrl={previewUrl}
            downloadUrl={downloadUrl}
          />
        ) : kind === 'image' ? (
          <div className="artifact-media-box" data-testid="artifact-image-viewer">
            <img
              src={previewUrl || downloadUrl}
              alt={artifact.name}
              style={{ maxWidth: '100%', borderRadius: '4px' }}
            />
          </div>
        ) : kind === 'log' && content !== null ? (
          <LogArtifactViewer content={content} />
        ) : (
          <div className="artifact-viewer-status font-mono text-muted" data-testid="artifact-binary-fallback">
            Binary artifact ({artifact.mime}, {formatArtifactSize(artifact.size_bytes)}). Use the
            Download button above to save this file.
          </div>
        )}
      </div>
    </div>
  )
}

export function RunArtifactsSection({
  runId,
  artifacts,
  title = 'Run Artifacts',
}: {
  runId: number
  artifacts: Artifact[]
  title?: string
}) {
  const [selectedArtifactKey, setSelectedArtifactKey] = useState<string | null>(null)

  const getKey = (a: Artifact) => `${a.id || 0}:${a.name}`

  const selectedArtifact = useMemo(
    () => artifacts.find((a) => getKey(a) === selectedArtifactKey) || null,
    [artifacts, selectedArtifactKey]
  )

  const handleTogglePreview = (art: Artifact) => {
    const key = getKey(art)
    if (selectedArtifactKey === key) {
      setSelectedArtifactKey(null)
    } else {
      setSelectedArtifactKey(key)
    }
  }

  if (!artifacts || artifacts.length === 0) {
    return null
  }

  return (
    <div className="card section-card" data-testid="run-artifacts-section">
      <div className="compare-section-header" style={{ marginBottom: '12px' }}>
        <div>
          <h3 className="section-title" style={{ marginBottom: '2px' }}>
            {title} ({artifacts.length})
          </h3>
          <span className="section-desc">
            Safe in-page preview (sandboxed HTML, JSON tree, audio player, log viewer) and download for{' '}
            <code>data/artifacts/{runId}/</code>
          </span>
        </div>
      </div>

      <div className="table-responsive">
        <table className="data-table" data-testid="run-artifacts-table">
          <thead>
            <tr>
              <th>Artifact Name</th>
              <th>Type</th>
              <th>Storage Path</th>
              <th>MIME Type</th>
              <th>Size</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {artifacts.map((art) => {
              const kind = detectArtifactKind(art)
              const badge = getKindBadgeLabel(kind)
              const isSelected = selectedArtifactKey === getKey(art)
              const artifactRef = art.id ? art.id : art.name
              const downloadUrl =
                art.download_url || api.runs.getArtifactDownloadUrl(runId, artifactRef)

              return (
                <tr
                  key={getKey(art)}
                  className={isSelected ? 'row-selected' : ''}
                  data-testid={`run-artifact-row-${art.name}`}
                >
                  <td>
                    <strong className="font-mono">{art.name}</strong>
                  </td>
                  <td>
                    <span className={`artifact-kind-badge ${badge.className}`}>
                      {badge.icon} {badge.label}
                    </span>
                  </td>
                  <td className="font-mono text-muted" style={{ fontSize: '12px' }}>
                    {formatArtifactDisplayPath(runId, art)}
                  </td>
                  <td className="font-mono text-muted">{art.mime}</td>
                  <td className="font-mono">{formatArtifactSize(art.size_bytes)}</td>
                  <td>
                    <div className="row-actions">
                      <button
                        type="button"
                        className={`btn btn-sm ${isSelected ? 'btn-primary' : 'btn-secondary'}`}
                        onClick={() => handleTogglePreview(art)}
                        data-testid={`preview-artifact-btn-${art.name}`}
                      >
                        {isSelected ? 'Hide Preview' : '👁 Preview'}
                      </button>
                      <a
                        href={downloadUrl}
                        className="btn btn-sm btn-secondary"
                        download={art.name}
                        target="_blank"
                        rel="noreferrer"
                        data-testid={`download-artifact-btn-${art.name}`}
                      >
                        ⬇ Download
                      </a>
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {selectedArtifact && (
        <ArtifactViewer
          runId={runId}
          artifact={selectedArtifact}
          artifacts={artifacts}
          onSelectArtifact={(art) => setSelectedArtifactKey(getKey(art))}
          onClose={() => setSelectedArtifactKey(null)}
        />
      )}
    </div>
  )
}
