import { useEffect, useState } from 'react'
import { api, type HealthResponse } from '../../api/index.ts'
import { Link } from '../../router/index.ts'

export function Header() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState<boolean>(true)

  const checkHealth = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.health.get()
      setHealth(data)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    let ignore = false
    api.health
      .get()
      .then((data) => {
        if (!ignore) {
          setHealth(data)
          setLoading(false)
        }
      })
      .catch((err: unknown) => {
        if (!ignore) {
          setError(err instanceof Error ? err.message : String(err))
          setLoading(false)
        }
      })

    const timer = setInterval(() => {
      api.health
        .get()
        .then((data) => {
          if (!ignore) {
            setHealth(data)
            setError(null)
          }
        })
        .catch((err: unknown) => {
          if (!ignore) {
            setError(err instanceof Error ? err.message : String(err))
          }
        })
    }, 30000)

    return () => {
      ignore = true
      clearInterval(timer)
    }
  }, [])

  return (
    <header className="app-header">
      <div className="brand">
        <Link href="/" className="brand-link">
          <span className="logo-icon">⚡</span>
          <div className="brand-text">
            <h1>Test Center</h1>
            <span className="brand-sub">Infrastructure & Model Evaluation</span>
          </div>
        </Link>
      </div>

      <div className="header-status">
        <div className="health-indicator">
          {loading && <span className="health-badge loading">Checking...</span>}
          {!loading && error && (
            <span className="health-badge error" title={error}>
              <span className="status-dot red" /> Backend Disconnected
            </span>
          )}
          {!loading && !error && health && (
            <span
              className="health-badge healthy"
              title={`App: ${health.app || 'Test Center'} v${health.version || '0.1.0'}`}
            >
              <span className="status-dot green" /> System Healthy
            </span>
          )}
          <button
            className="icon-btn"
            onClick={checkHealth}
            disabled={loading}
            title="Refresh system health"
            aria-label="Refresh system health"
          >
            ↻
          </button>
        </div>
      </div>
    </header>
  )
}
