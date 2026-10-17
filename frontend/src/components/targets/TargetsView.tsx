import { useEffect, useState } from 'react'
import { api, type Target, type TargetCreate } from '../../api/index.ts'
import { ConfirmDialog } from '../common/ConfirmDialog.tsx'
import { Spinner } from '../common/Spinner.tsx'
import { TargetFormModal } from './TargetFormModal.tsx'

export function TargetsView() {
  const [targets, setTargets] = useState<Target[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [kindFilter, setKindFilter] = useState('all')

  // Modals state
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [editingTarget, setEditingTarget] = useState<Target | null>(null)
  const [deletingTarget, setDeletingTarget] = useState<Target | null>(null)

  const loadTargets = async (showSpinner = false) => {
    try {
      if (showSpinner) setLoading(true)
      setError(null)
      const data = await api.targets.list()
      setTargets(data)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      if (showSpinner) setLoading(false)
    }
  }

  useEffect(() => {
    let ignore = false
    api.targets
      .list()
      .then((data) => {
        if (!ignore) {
          setTargets(data)
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

  const handleSave = async (data: TargetCreate) => {
    if (editingTarget) {
      await api.targets.update(editingTarget.id, data)
    } else {
      await api.targets.create(data)
    }
    await loadTargets(false)
  }

  const handleDelete = async () => {
    if (!deletingTarget) return
    try {
      await api.targets.delete(deletingTarget.id)
      setDeletingTarget(null)
      await loadTargets(false)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const uniqueKinds = Array.from(new Set(targets.map((t) => t.kind)))

  const filteredTargets = targets.filter((t) => {
    const matchesKind = kindFilter === 'all' || t.kind === kindFilter
    const matchesSearch =
      t.name.toLowerCase().includes(search.toLowerCase()) ||
      t.base_url.toLowerCase().includes(search.toLowerCase()) ||
      (t.auth_env_var && t.auth_env_var.toLowerCase().includes(search.toLowerCase()))
    return matchesKind && matchesSearch
  })

  return (
    <div className="view-container targets-view">
      <div className="view-header">
        <div>
          <h2>Target Endpoints</h2>
          <p className="view-subtitle">
            Configure external test targets, models, and endpoints for benchmark suites.
          </p>
        </div>
        <button
          className="btn btn-primary"
          onClick={() => {
            setEditingTarget(null)
            setIsModalOpen(true)
          }}
        >
          <span className="btn-icon">+</span> Add Target
        </button>
      </div>

      {error && <div className="banner error-banner">{error}</div>}

      <div className="filter-bar">
        <div className="search-input-wrapper">
          <span className="search-icon">🔍</span>
          <input
            type="text"
            className="search-input"
            placeholder="Search targets by name or URL..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <div className="filter-group">
          <label htmlFor="kind-filter" className="filter-label">Kind:</label>
          <select
            id="kind-filter"
            className="filter-select"
            value={kindFilter}
            onChange={(e) => setKindFilter(e.target.value)}
          >
            <option value="all">All Kinds ({targets.length})</option>
            {uniqueKinds.map((k) => (
              <option key={k} value={k}>
                {k} ({targets.filter((t) => t.kind === k).length})
              </option>
            ))}
          </select>
        </div>
      </div>

      {loading ? (
        <Spinner message="Loading targets..." />
      ) : filteredTargets.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">🎯</div>
          <h3>No targets found</h3>
          <p>
            {targets.length === 0
              ? 'No targets configured yet. Add your first test target to begin benchmarking.'
              : 'No targets match your current search and filter criteria.'}
          </p>
          {targets.length === 0 && (
            <button
              className="btn btn-primary"
              onClick={() => {
                setEditingTarget(null)
                setIsModalOpen(true)
              }}
            >
              Add Target
            </button>
          )}
        </div>
      ) : (
        <div className="table-responsive">
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Target Name</th>
                <th>Kind</th>
                <th>Base URL</th>
                <th>Auth Env Var</th>
                <th>Extra Config</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredTargets.map((target) => (
                <tr key={target.id}>
                  <td className="font-mono text-muted">#{target.id}</td>
                  <td>
                    <strong>{target.name}</strong>
                  </td>
                  <td>
                    <span className="kind-badge">{target.kind}</span>
                  </td>
                  <td className="font-mono text-break">{target.base_url}</td>
                  <td>
                    {target.auth_env_var ? (
                      <span className="auth-env-badge" title="Environment variable reference">
                        🔒 <code>{target.auth_env_var}</code>
                      </span>
                    ) : (
                      <span className="text-muted">None</span>
                    )}
                  </td>
                  <td>
                    {target.extra && Object.keys(target.extra).length > 0 ? (
                      <span
                        className="extra-summary-tag font-mono"
                        title={JSON.stringify(target.extra, null, 2)}
                      >
                        {Object.entries(target.extra)
                          .map(([k, v]) => `${k}=${v}`)
                          .join(', ')}
                      </span>
                    ) : (
                      <span className="text-muted">—</span>
                    )}
                  </td>
                  <td>
                    <div className="row-actions">
                      <button
                        className="action-btn"
                        title="Edit Target"
                        onClick={() => {
                          setEditingTarget(target)
                          setIsModalOpen(true)
                        }}
                      >
                        Edit
                      </button>
                      <button
                        className="action-btn danger"
                        title="Delete Target"
                        onClick={() => setDeletingTarget(target)}
                      >
                        Delete
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Create / Edit Modal */}
      <TargetFormModal
        isOpen={isModalOpen}
        target={editingTarget}
        onClose={() => setIsModalOpen(false)}
        onSave={handleSave}
      />

      {/* Delete Confirmation */}
      <ConfirmDialog
        isOpen={Boolean(deletingTarget)}
        title="Delete Target"
        message={`Are you sure you want to delete target "${deletingTarget?.name}"? Test definitions referencing this target will have their target reference cleared.`}
        confirmLabel="Delete Target"
        danger
        onConfirm={handleDelete}
        onCancel={() => setDeletingTarget(null)}
      />
    </div>
  )
}
