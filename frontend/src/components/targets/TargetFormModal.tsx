import { useState } from 'react'
import type { Target, TargetCreate } from '../../api/types.ts'
import { Modal } from '../common/Modal.tsx'

const COMMON_KINDS = [
  'openai-compatible',
  'embedding',
  'live-stt',
  'llm',
  'rerank',
  'custom',
]

const ENV_VAR_REGEX = /^[A-Za-z_][A-Za-z0-9_]*$/
const DISALLOWED_SECRET_KEYS = [
  'api_key',
  'apikey',
  'secret',
  'secret_key',
  'password',
  'token',
  'access_token',
  'auth_token',
  'bearer_token',
  'private_key',
]

function TargetForm({
  target,
  onClose,
  onSave,
}: {
  target: Target | null
  onClose: () => void
  onSave: (data: TargetCreate) => Promise<void>
}) {
  const [name, setName] = useState(target?.name || '')
  const [kind, setKind] = useState(() => {
    if (!target) return 'openai-compatible'
    return COMMON_KINDS.includes(target.kind) ? target.kind : 'custom'
  })
  const [customKind, setCustomKind] = useState(() => {
    if (!target) return ''
    return COMMON_KINDS.includes(target.kind) ? '' : target.kind
  })
  const [baseUrl, setBaseUrl] = useState(target?.base_url || '')
  const [authEnvVar, setAuthEnvVar] = useState(target?.auth_env_var || '')
  const [extraJson, setExtraJson] = useState(() =>
    target?.extra ? JSON.stringify(target.extra, null, 2) : '{}'
  )
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)

    const finalKind = kind === 'custom' ? customKind.trim() : kind
    if (!name.trim()) {
      setError('Target name is required')
      return
    }
    if (!finalKind) {
      setError('Target kind is required')
      return
    }
    if (!baseUrl.trim()) {
      setError('Base URL is required')
      return
    }

    const trimmedAuth = authEnvVar.trim()
    if (trimmedAuth) {
      if (
        trimmedAuth.startsWith('sk-') ||
        trimmedAuth.startsWith('ghp_') ||
        trimmedAuth.startsWith('Bearer ') ||
        trimmedAuth.startsWith('token-')
      ) {
        setError(
          'API key must be provided by environment variable reference only (e.g. "OPENAI_API_KEY"), never a raw secret or token.'
        )
        return
      }
      if (!ENV_VAR_REGEX.test(trimmedAuth)) {
        setError(
          'auth_env_var must be a valid environment variable name (letters, numbers, underscores).'
        )
        return
      }
    }

    let parsedExtra: Record<string, unknown> = {}
    if (extraJson.trim()) {
      try {
        parsedExtra = JSON.parse(extraJson)
        if (typeof parsedExtra !== 'object' || Array.isArray(parsedExtra) || parsedExtra === null) {
          setError('Extra parameters must be a valid JSON object')
          return
        }
      } catch {
        setError('Extra parameters is not valid JSON')
        return
      }
    }

    for (const key of Object.keys(parsedExtra)) {
      if (DISALLOWED_SECRET_KEYS.includes(key.toLowerCase())) {
        setError(
          `Direct secrets forbidden in extra ('${key}'). Use Auth Env Var to reference secrets via environment variables.`
        )
        return
      }
    }

    try {
      setSaving(true)
      await onSave({
        name: name.trim(),
        kind: finalKind,
        base_url: baseUrl.trim(),
        auth_env_var: trimmedAuth || null,
        extra: parsedExtra,
      })
      onClose()
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSaving(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="target-form">
      {error && <div className="form-error-alert">{error}</div>}

      <div className="form-group">
        <label htmlFor="target-name">
          Target Name <span className="required-star">*</span>
        </label>
        <input
          id="target-name"
          type="text"
          className="form-input"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. local-vllm-embed"
          required
          autoFocus
        />
      </div>

      <div className="form-row">
        <div className="form-group flex-1">
          <label htmlFor="target-kind">
            Target Kind <span className="required-star">*</span>
          </label>
          <select
            id="target-kind"
            className="form-select"
            value={kind}
            onChange={(e) => setKind(e.target.value)}
          >
            {COMMON_KINDS.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </div>

        {kind === 'custom' && (
          <div className="form-group flex-1">
            <label htmlFor="custom-kind">
              Custom Kind <span className="required-star">*</span>
            </label>
            <input
              id="custom-kind"
              type="text"
              className="form-input"
              value={customKind}
              onChange={(e) => setCustomKind(e.target.value)}
              placeholder="e.g. my-custom-service"
              required
            />
          </div>
        )}
      </div>

      <div className="form-group">
        <label htmlFor="target-base-url">
          Base URL <span className="required-star">*</span>
        </label>
        <input
          id="target-base-url"
          type="text"
          className="form-input"
          value={baseUrl}
          onChange={(e) => setBaseUrl(e.target.value)}
          placeholder="e.g. http://localhost:8000/v1"
          required
        />
        <span className="field-hint">Root URL of the target service or API gateway.</span>
      </div>

      <div className="form-group">
        <label htmlFor="target-auth-env-var">Auth Environment Variable Reference</label>
        <input
          id="target-auth-env-var"
          type="text"
          className="form-input font-mono"
          value={authEnvVar}
          onChange={(e) => setAuthEnvVar(e.target.value)}
          placeholder="e.g. OPENAI_API_KEY"
        />
        <div className="auth-env-notice">
          <span className="notice-icon">🔒</span>
          <span className="notice-text">
            <strong>Secrets are never stored in the database.</strong> Specify the name of the
            environment variable only (e.g. <code>OPENAI_API_KEY</code>). Never paste raw API
            keys or tokens.
          </span>
        </div>
      </div>

      <div className="form-group">
        <div className="field-header">
          <label htmlFor="target-extra-json">Extra Parameters (JSON)</label>
          <span className="field-default-hint">e.g. model name, dimensions</span>
        </div>
        <textarea
          id="target-extra-json"
          className="form-textarea font-mono"
          rows={4}
          value={extraJson}
          onChange={(e) => setExtraJson(e.target.value)}
          placeholder='{ "model": "text-embedding-3-small" }'
        />
      </div>

      <div className="modal-actions">
        <button type="button" className="btn btn-secondary" onClick={onClose} disabled={saving}>
          Cancel
        </button>
        <button type="submit" className="btn btn-primary" disabled={saving}>
          {saving ? 'Saving...' : target ? 'Update Target' : 'Create Target'}
        </button>
      </div>
    </form>
  )
}

export function TargetFormModal({
  isOpen,
  target,
  onClose,
  onSave,
}: {
  isOpen: boolean
  target: Target | null
  onClose: () => void
  onSave: (data: TargetCreate) => Promise<void>
}) {
  return (
    <Modal
      title={target ? `Edit Target: ${target.name}` : 'Create New Target'}
      isOpen={isOpen}
      onClose={onClose}
      maxWidth="560px"
    >
      {isOpen && (
        <TargetForm
          key={target ? target.id : 'new'}
          target={target}
          onClose={onClose}
          onSave={onSave}
        />
      )}
    </Modal>
  )
}
