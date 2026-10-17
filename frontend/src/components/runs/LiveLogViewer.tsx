import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type RunStatus } from '../../api/index.ts'

export function LiveLogViewer({
  runId,
  runStatus,
}: {
  runId: number
  runStatus: RunStatus
}) {
  const [logs, setLogs] = useState<string[]>([])
  const [filterText, setFilterText] = useState('')
  const [autoScroll, setAutoScroll] = useState(true)
  const [connected, setConnected] = useState(false)
  const [copied, setCopied] = useState(false)

  const consoleEndRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const loadHistoricalLogs = useCallback(async () => {
    try {
      const data = await api.runs.getLogs(runId)
      if (data.lines && data.lines.length > 0) {
        setLogs(data.lines)
      }
    } catch {
      // Ignore load error
    }
  }, [runId])

  useEffect(() => {
    let ignore = false

    api.runs
      .getLogs(runId)
      .then((data) => {
        if (!ignore && data.lines && data.lines.length > 0) {
          setLogs(data.lines)
        }
      })
      .catch(() => {})

    // If run is already completed/cancelled/failed/error, don't open SSE stream
    const isFinished = ['completed', 'failed', 'cancelled', 'error'].includes(runStatus)
    if (isFinished) {
      return () => {
        ignore = true
      }
    }

    // Connect SSE
    const streamUrl = api.runs.getLogStreamUrl(runId)
    const eventSource = new EventSource(streamUrl)

    eventSource.onopen = () => {
      if (!ignore) setConnected(true)
    }

    eventSource.addEventListener('log', (e: MessageEvent) => {
      if (ignore) return
      try {
        const payload = JSON.parse(e.data)
        if (payload.message) {
          setLogs((prev) => [...prev, payload.message])
        }
      } catch {
        if (e.data) {
          setLogs((prev) => [...prev, e.data])
        }
      }
    })

    eventSource.addEventListener('completed', () => {
      if (ignore) return
      setConnected(false)
      eventSource.close()
      loadHistoricalLogs()
    })

    eventSource.addEventListener('end', () => {
      if (ignore) return
      setConnected(false)
      eventSource.close()
    })

    eventSource.onerror = () => {
      if (ignore) return
      setConnected(false)
      eventSource.close()
      setTimeout(loadHistoricalLogs, 1000)
    }

    return () => {
      ignore = true
      eventSource.close()
    }
  }, [runId, runStatus, loadHistoricalLogs])

  // Handle auto-scroll
  useEffect(() => {
    if (
      autoScroll &&
      consoleEndRef.current &&
      typeof consoleEndRef.current.scrollIntoView === 'function'
    ) {
      consoleEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs, autoScroll])

  const handleCopyLogs = async () => {
    try {
      await navigator.clipboard.writeText(logs.join('\n'))
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // fallback
    }
  }

  const handleDownloadLogs = () => {
    const blob = new Blob([logs.join('\n')], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `run-${runId}.log`
    a.click()
    URL.revokeObjectURL(url)
  }

  const filteredLogs = filterText
    ? logs.filter((line) => line.toLowerCase().includes(filterText.toLowerCase()))
    : logs

  return (
    <div className="log-viewer">
      <div className="log-toolbar">
        <div className="log-status-indicator">
          {connected ? (
            <span className="live-pill active">
              <span className="pulsing-dot green" /> Live Streaming
            </span>
          ) : ['completed', 'failed', 'cancelled', 'error'].includes(runStatus) ? (
            <span className="live-pill finished">Log Finished</span>
          ) : (
            <span className="live-pill disconnected">Connecting...</span>
          )}
          <span className="log-line-count">{logs.length} lines</span>
        </div>

        <div className="log-controls">
          <input
            type="text"
            className="log-filter-input"
            placeholder="Filter logs..."
            value={filterText}
            onChange={(e) => setFilterText(e.target.value)}
          />

          <label className="autoscroll-toggle">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
            />
            <span>Auto-scroll</span>
          </label>

          <button className="btn btn-sm btn-secondary" onClick={handleCopyLogs} title="Copy logs">
            {copied ? '✓ Copied' : 'Copy'}
          </button>
          <button
            className="btn btn-sm btn-secondary"
            onClick={handleDownloadLogs}
            title="Download full log file"
          >
            Download
          </button>
        </div>
      </div>

      <div className="log-terminal" ref={containerRef}>
        {filteredLogs.length === 0 ? (
          <div className="log-empty-msg">
            {logs.length === 0 ? 'Waiting for output...' : 'No lines match filter.'}
          </div>
        ) : (
          <div className="log-lines">
            {filteredLogs.map((line, idx) => (
              <div key={idx} className="log-line">
                <span className="log-line-num">{idx + 1}</span>
                <span className="log-line-content">{line}</span>
              </div>
            ))}
          </div>
        )}
        <div ref={consoleEndRef} />
      </div>
    </div>
  )
}
