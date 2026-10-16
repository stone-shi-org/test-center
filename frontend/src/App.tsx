import { useEffect, useState } from 'react'
import './App.css'

interface HealthResponse {
  status: string
  app?: string
  version?: string
  [key: string]: unknown
}

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [activeTab, setActiveTab] = useState<string>('dashboard')

  const handleRefresh = () => {
    setLoading(true)
    setError(null)
    fetch('/api/health')
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`)
        }
        return res.json()
      })
      .then((data: HealthResponse) => {
        setHealth(data)
        setLoading(false)
      })
      .catch((err: Error) => {
        setError(err.message)
        setLoading(false)
      })
  }

  useEffect(() => {
    let ignore = false
    fetch('/api/health')
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`)
        }
        return res.json()
      })
      .then((data: HealthResponse) => {
        if (!ignore) {
          setHealth(data)
          setLoading(false)
        }
      })
      .catch((err: Error) => {
        if (!ignore) {
          setError(err.message)
          setLoading(false)
        }
      })
    return () => {
      ignore = true
    }
  }, [])

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="brand">
          <span className="logo-icon">⚡</span>
          <h1>Test Center</h1>
        </div>
        <div className="health-status">
          {loading && <span className="badge loading">Checking backend...</span>}
          {error && (
            <span className="badge error" title={error}>
              Backend: Disconnected ({error})
            </span>
          )}
          {health && (
            <span className="badge success">
              Backend: Healthy ({health.status})
            </span>
          )}
          <button className="refresh-btn" onClick={handleRefresh} disabled={loading}>
            Refresh
          </button>
        </div>
      </header>

      <nav className="nav-tabs">
        {['dashboard', 'targets', 'jobs', 'runs', 'compare'].map((tab) => (
          <button
            key={tab}
            className={`tab-btn ${activeTab === tab ? 'active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </nav>

      <main className="app-main">
        {activeTab === 'dashboard' && (
          <section className="card-grid">
            <div className="card">
              <h3>Live STT</h3>
              <p>HTTP and gRPC streaming benchmarks, transcription and diarization accuracy on NOTSOFAR-1.</p>
              <span className="tag">Plugin Ready</span>
            </div>
            <div className="card">
              <h3>Embedding</h3>
              <p>Latency percentiles, sequential & concurrent throughput, vector norms, and retrieval recall.</p>
              <span className="tag">Plugin Ready</span>
            </div>
            <div className="card">
              <h3>LLM</h3>
              <p>TTFT, tokens/sec, tool-call JSON, reasoning, summarization, and capability evaluations.</p>
              <span className="tag">Plugin Ready</span>
            </div>
          </section>
        )}

        {activeTab !== 'dashboard' && (
          <div className="placeholder-view">
            <h2>{activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} View</h2>
            <p>Module initialized. Configure and run tests from here.</p>
          </div>
        )}

        <section className="system-info-card">
          <h3>System & Health Status</h3>
          <div className="system-info-content">
            <p><strong>Endpoint:</strong> <code>/api/health</code></p>
            <p>
              <strong>Status:</strong>{' '}
              {loading ? 'Checking...' : error ? `Error: ${error}` : JSON.stringify(health)}
            </p>
          </div>
        </section>
      </main>

      <footer className="app-footer">
        <p>Test Center • FastAPI Backend + Vite React Frontend</p>
      </footer>
    </div>
  )
}

export default App
