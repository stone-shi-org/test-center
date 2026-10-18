import { useState } from 'react'
import {
  api,
  type Target,
  type TargetCreate,
  type TargetTestResponse,
} from '../../api/index.ts'
import { Modal } from '../common/Modal.tsx'

const COMMON_KINDS = [
  'openai-compatible',
  'embedding',
  'live-stt',
  'llm',
  'rerank',
  'custom',
]

export const TARGET_CAPABILITY_OPTIONS = [
  {
    id: 'chat',
    label: 'chat',
    title: 'Chat / LLM completions (/chat/completions)',
  },
  {
    id: 'embedding',
    label: 'embedding',
    title: 'Text embeddings (/embeddings)',
  },
  {
    id: 'asr',
    label: 'asr',
    title: 'Speech-to-Text transcription (/v1/audio/transcriptions)',
  },
  {
    id: 'tts',
    label: 'tts',
    title: 'Text-to-Speech synthesis (/v1/audio/speech)',
  },
  {
    id: 'diarization',
    label: 'diarization',
    title: 'Speaker diarization (/v1/audio/diarization)',
  },
  {
    id: 'grpc',
    label: 'grpc',
    title: 'gRPC StreamingASR (GetServerInfo probe on the gRPC port)',
  },
] as const

export function inferDefaultCapabilities(kind: string): string[] {
  const k = (kind || '').trim().toLowerCase()
  if (k === 'embedding' || k === 'embeddings') return ['embedding']
  if (k === 'live-stt' || k === 'live_stt' || k === 'stt' || k === 'asr') {
    return ['asr', 'tts', 'diarization', 'grpc']
  }
  return ['chat']
}

export function getTargetCapabilities(
  target: Pick<Target, 'kind' | 'extra' | 'capabilities'>
): string[] {
  if (Array.isArray(target.capabilities) && target.capabilities.length > 0) {
    return target.capabilities
  }
  const extraCaps = target.extra?.capabilities
  if (Array.isArray(extraCaps) && extraCaps.length > 0) {
    return extraCaps.map((c) => String(c))
  }
  return inferDefaultCapabilities(target.kind)
}

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
  const [capabilities, setCapabilities] = useState<string[]>(() =>
    target ? getTargetCapabilities(target) : inferDefaultCapabilities('openai-compatible')
  )
  const [capabilitiesCustomized, setCapabilitiesCustomized] = useState<boolean>(() =>
    Boolean(target)
  )
  const [grpcPort, setGrpcPort] = useState(() => {
    const p = target?.extra?.grpc_port
    return p === undefined || p === null ? '' : String(p)
  })
  const [extraJson, setExtraJson] = useState(() => {
    if (!target?.extra) return '{}'
    const copy = { ...target.extra }
    delete copy.capabilities
    delete copy.grpc_port
    return JSON.stringify(copy, null, 2)
  })
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<TargetTestResponse | null>(null)

  const handleKindChange = (newKind: string) => {
    setKind(newKind)
    if (!capabilitiesCustomized) {
      const effectiveKind = newKind === 'custom' ? customKind || 'custom' : newKind
      setCapabilities(inferDefaultCapabilities(effectiveKind))
    }
  }

  const toggleCapability = (capId: string) => {
    setCapabilitiesCustomized(true)
    setCapabilities((prev) =>
      prev.includes(capId) ? prev.filter((c) => c !== capId) : [...prev, capId]
    )
  }

  const validateAndBuildPayload = (): TargetCreate | null => {
    setError(null)
    const finalKind = kind === 'custom' ? customKind.trim() : kind
    if (!name.trim()) {
      setError('Target name is required')
      return null
    }
    if (!finalKind) {
      setError('Target kind is required')
      return null
    }
    if (!baseUrl.trim()) {
      setError('Base URL is required')
      return null
    }
    if (capabilities.length === 0) {
      setError('Select at least one target capability (e.g. chat, embedding)')
      return null
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
        return null
      }
      if (!ENV_VAR_REGEX.test(trimmedAuth)) {
        setError(
          'auth_env_var must be a valid environment variable name (letters, numbers, underscores).'
        )
        return null
      }
    }

    let parsedExtra: Record<string, unknown> = {}
    if (extraJson.trim()) {
      try {
        parsedExtra = JSON.parse(extraJson)
        if (typeof parsedExtra !== 'object' || Array.isArray(parsedExtra) || parsedExtra === null) {
          setError('Extra parameters must be a valid JSON object')
          return null
        }
      } catch {
        setError('Extra parameters is not valid JSON')
        return null
      }
    }

    const trimmedGrpcPort = grpcPort.trim()
    if (trimmedGrpcPort) {
      const portNum = Number(trimmedGrpcPort)
      if (!/^\d+$/.test(trimmedGrpcPort) || portNum < 1 || portNum > 65535) {
        setError('gRPC port must be an integer between 1 and 65535')
        return null
      }
      parsedExtra = { ...parsedExtra, grpc_port: portNum }
    } else {
      delete parsedExtra.grpc_port
    }

    for (const key of Object.keys(parsedExtra)) {
      if (DISALLOWED_SECRET_KEYS.includes(key.toLowerCase())) {
        setError(
          `Direct secrets forbidden in extra ('${key}'). Use Auth Env Var to reference secrets via environment variables.`
        )
        return null
      }
    }

    return {
      name: name.trim(),
      kind: finalKind,
      base_url: baseUrl.trim(),
      auth_env_var: trimmedAuth || null,
      extra: parsedExtra,
      capabilities,
    }
  }

  const handleTestInForm = async () => {
    const payload = validateAndBuildPayload()
    if (!payload) return
    try {
      setTesting(true)
      setTestResult(null)
      const res = await api.targets.testAdHoc({
        ...payload,
        capabilities: payload.capabilities,
      })
      setTestResult(res)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setTesting(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const payload = validateAndBuildPayload()
    if (!payload) return

    try {
      setSaving(true)
      await onSave(payload)
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
            onChange={(e) => handleKindChange(e.target.value)}
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
        <div className="field-header">
          <label>
            Target Capabilities <span className="required-star">*</span>
          </label>
          <span className="field-default-hint">Pick capabilities supported by this target</span>
        </div>
        <div className="enum-pills" role="group" aria-label="Target Capabilities">
          {TARGET_CAPABILITY_OPTIONS.map((opt) => {
            const isSelected = capabilities.includes(opt.id)
            return (
              <button
                key={opt.id}
                type="button"
                className={`pill-btn ${isSelected ? 'selected' : ''}`}
                title={opt.title}
                onClick={() => toggleCapability(opt.id)}
              >
                <span>{isSelected ? '✓' : '+'}</span>
                <span>{opt.label}</span>
              </button>
            )
          })}
        </div>
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

      {(capabilities.includes('grpc') || kind === 'live-stt') && (
        <div className="form-group">
          <label htmlFor="target-grpc-port">gRPC Port</label>
          <input
            id="target-grpc-port"
            type="text"
            inputMode="numeric"
            className="form-input font-mono"
            value={grpcPort}
            onChange={(e) => setGrpcPort(e.target.value)}
            placeholder="50051"
          />
          <span className="field-hint">
            Optional. Uses the host of Base URL with this port for gRPC streaming. Leave empty
            for the default (50051).
          </span>
        </div>
      )}

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

      {testResult && (
        <div
          className={`banner ${testResult.ok ? 'success-banner' : 'error-banner'} target-test-inline-banner`}
        >
          <div className="target-test-inline-header">
            <strong>
              {testResult.ok
                ? '✓ Target capability check passed'
                : '✕ Target capability check failed'}
            </strong>
          </div>
          <div className="target-test-inline-list">
            {testResult.results.map((r) => (
              <div key={r.capability} className="target-test-inline-item">
                <span className="kind-badge">{r.capability}</span>{' '}
                <span>{r.ok ? 'PASS' : 'FAIL'}</span> — <span>{r.message}</span>
                {r.error && <div className="font-mono text-break">{r.error}</div>}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="modal-actions">
        <button
          type="button"
          className="btn btn-secondary"
          onClick={handleTestInForm}
          disabled={saving || testing}
        >
          {testing ? 'Testing...' : 'Test Target'}
        </button>
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

