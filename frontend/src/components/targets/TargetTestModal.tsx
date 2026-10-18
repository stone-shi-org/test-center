import { useEffect, useState } from 'react'
import {
  api,
  type CapabilityTestResult,
  type Target,
  type TargetTestResponse,
} from '../../api/index.ts'
import { Modal } from '../common/Modal.tsx'
import {
  getTargetCapabilities,
  TARGET_CAPABILITY_OPTIONS,
} from './TargetFormModal.tsx'

function TargetTestContent({
  target,
  onClose,
  onTestComplete,
  onUpdateCapabilities,
}: {
  target: Target
  onClose: () => void
  onTestComplete: (targetId: number, result: TargetTestResponse) => void
  onUpdateCapabilities?: (targetId: number, capabilities: string[]) => Promise<void>
}) {
  const initialCapabilities = getTargetCapabilities(target)
  const [selectedCapabilities, setSelectedCapabilities] =
    useState<string[]>(initialCapabilities)
  const [modelOverride, setModelOverride] = useState<string>(() =>
    typeof target.extra?.model === 'string' ? target.extra.model : ''
  )
  const [testing, setTesting] = useState(false)
  const [savingCaps, setSavingCaps] = useState(false)
  const [capsSavedMessage, setCapsSavedMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<TargetTestResponse | null>(null)

  const runCapabilityTest = async (capsToTest: string[], model?: string) => {
    if (capsToTest.length === 0) {
      setError('Please select at least one capability to test (e.g. chat, embedding).')
      return
    }
    try {
      setTesting(true)
      setError(null)
      const res = await api.targets.test(target.id, {
        capabilities: capsToTest,
        model: model?.trim() || null,
      })
      setResult(res)
      onTestComplete(target.id, res)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setTesting(false)
    }
  }

  useEffect(() => {
    let ignore = false
    const defaultCaps = getTargetCapabilities(target)
    const defaultModel =
      typeof target.extra?.model === 'string' ? target.extra.model.trim() : null
    setTesting(true)
    setError(null)
    api.targets
      .test(target.id, {
        capabilities: defaultCaps,
        model: defaultModel || null,
      })
      .then((res) => {
        if (!ignore) {
          setResult(res)
          setTesting(false)
          onTestComplete(target.id, res)
        }
      })
      .catch((err: unknown) => {
        if (!ignore) {
          setError(err instanceof Error ? err.message : String(err))
          setTesting(false)
        }
      })
    return () => {
      ignore = true
    }
  }, [target.id])

  const toggleCapability = (capId: string) => {
    setCapsSavedMessage(null)
    setSelectedCapabilities((prev) =>
      prev.includes(capId) ? prev.filter((c) => c !== capId) : [...prev, capId]
    )
  }

  const handleSaveCapabilities = async () => {
    if (!onUpdateCapabilities) return
    if (selectedCapabilities.length === 0) {
      setError('Select at least one capability before saving.')
      return
    }
    try {
      setSavingCaps(true)
      setError(null)
      setCapsSavedMessage(null)
      await onUpdateCapabilities(target.id, selectedCapabilities)
      setCapsSavedMessage('Target capabilities saved.')
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSavingCaps(false)
    }
  }

  const renderCapabilityDetail = (item: CapabilityTestResult) => {
    const detail = item.detail || {}
    return (
      <div key={item.capability} className="target-test-result-card">
        <div className="target-test-result-header">
          <div className="target-test-result-title">
            <span className="kind-badge">{item.capability}</span>
            <span
              className={`status-badge badge-sm ${
                item.ok ? 'badge-success' : 'badge-error'
              }`}
            >
              {item.ok ? 'PASS' : 'FAIL'}
            </span>
            {item.model && (
              <span className="font-mono text-muted" style={{ fontSize: '12px' }}>
                model: {item.model}
              </span>
            )}
          </div>
          <div className="target-test-result-meta font-mono">
            {item.status_code !== null && item.status_code !== undefined && (
              <span>HTTP {item.status_code} · </span>
            )}
            <span>{item.latency_ms.toFixed(1)} ms</span>
          </div>
        </div>

        <div className="target-test-result-message">{item.message}</div>

        {typeof detail.response_preview === 'string' && detail.response_preview && (
          <div className="target-test-result-sub font-mono">
            Response: &ldquo;{detail.response_preview}&rdquo;
          </div>
        )}

        {typeof detail.dimensions === 'number' && detail.dimensions > 0 && (
          <div className="target-test-result-sub font-mono">
            Vector Dimensions: {detail.dimensions}
          </div>
        )}

        {typeof detail.endpoint === 'string' && (
          <div className="target-test-result-sub font-mono text-muted">
            Endpoint: {detail.endpoint}
          </div>
        )}

        {item.error && (
          <div className="form-error-alert font-mono text-break" style={{ marginTop: '6px' }}>
            {item.error}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="target-test-modal-body">
      <div className="target-test-meta-box">
        <div>
          <strong>Target:</strong> {target.name}{' '}
          <span className="kind-badge">{target.kind}</span>
        </div>
        <div className="font-mono text-break" style={{ fontSize: '12px' }}>
          <strong>URL:</strong> {target.base_url}
        </div>
        {target.auth_env_var && (
          <div style={{ fontSize: '12px' }}>
            <strong>Auth Env Var:</strong> <code>{target.auth_env_var}</code>
            {result && !result.auth_configured && (
              <span className="status-badge badge-sm badge-warning" style={{ marginLeft: '8px' }}>
                Env var not set
              </span>
            )}
          </div>
        )}
      </div>

      {error && <div className="form-error-alert">{error}</div>}
      {capsSavedMessage && <div className="banner success-banner">{capsSavedMessage}</div>}

      <div className="form-group" style={{ marginTop: '12px' }}>
        <div className="field-header">
          <label>Capabilities to Test</label>
          <span className="field-default-hint">
            Pick which capabilities this target supports &amp; test against them
          </span>
        </div>
        <div className="enum-pills" role="group" aria-label="Capabilities to Test">
          {TARGET_CAPABILITY_OPTIONS.map((opt) => {
            const isSelected = selectedCapabilities.includes(opt.id)
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
        <label htmlFor="target-test-model-override">Model Override (Optional)</label>
        <input
          id="target-test-model-override"
          type="text"
          className="form-input font-mono"
          value={modelOverride}
          onChange={(e) => setModelOverride(e.target.value)}
          placeholder="Auto-detect or use target extra.model"
        />
      </div>

      {testing && (
        <div className="banner" style={{ background: 'var(--bg-muted)' }}>
          Testing target capabilities ({selectedCapabilities.join(', ')})...
        </div>
      )}

      {!testing && result && (
        <div className="target-test-results-section">
          <div className={`banner ${result.ok ? 'success-banner' : 'error-banner'}`}>
            <strong>
              {result.ok
                ? `✓ All ${result.results.length} capability check(s) passed`
                : '✕ One or more capability checks failed'}
            </strong>
          </div>
          <div className="target-test-results-list">
            {result.results.map(renderCapabilityDetail)}
          </div>
        </div>
      )}

      <div className="modal-actions">
        {onUpdateCapabilities && (
          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleSaveCapabilities}
            disabled={savingCaps || testing}
          >
            {savingCaps ? 'Saving...' : 'Save Capabilities'}
          </button>
        )}
        <button type="button" className="btn btn-secondary" onClick={onClose}>
          Close
        </button>
        <button
          type="button"
          className="btn btn-primary"
          onClick={() => runCapabilityTest(selectedCapabilities, modelOverride)}
          disabled={testing || selectedCapabilities.length === 0}
        >
          {testing ? 'Testing...' : 'Run Test'}
        </button>
      </div>
    </div>
  )
}

export function TargetTestModal({
  isOpen,
  target,
  onClose,
  onTestComplete,
  onUpdateCapabilities,
}: {
  isOpen: boolean
  target: Target | null
  onClose: () => void
  onTestComplete: (targetId: number, result: TargetTestResponse) => void
  onUpdateCapabilities?: (targetId: number, capabilities: string[]) => Promise<void>
}) {
  return (
    <Modal
      title={target ? `Test Target: ${target.name}` : 'Test Target'}
      isOpen={isOpen}
      onClose={onClose}
      maxWidth="620px"
    >
      {isOpen && target && (
        <TargetTestContent
          key={target.id}
          target={target}
          onClose={onClose}
          onTestComplete={onTestComplete}
          onUpdateCapabilities={onUpdateCapabilities}
        />
      )}
    </Modal>
  )
}
