import { useEffect, useState } from 'react'
import {
  api,
  type Target,
  type TestDefinition,
  type TesterInfo,
} from '../../api/index.ts'
import { useRouter } from '../../router/index.ts'
import { Spinner } from '../common/Spinner.tsx'
import { SchemaForm } from '../schema/SchemaForm.tsx'

export function TriggerView() {
  const { navigate } = useRouter()

  const [testers, setTesters] = useState<TesterInfo[]>([])
  const [targets, setTargets] = useState<Target[]>([])
  const [definitions, setDefinitions] = useState<TestDefinition[]>([])

  const [selectedTester, setSelectedTester] = useState<string>('')
  const [definitionName, setDefinitionName] = useState<string>('')
  const [selectedTargetId, setSelectedTargetId] = useState<string>('')
  const [selectedSuites, setSelectedSuites] = useState<string[]>([])
  const [configValues, setConfigValues] = useState<Record<string, unknown>>({})
  const [timeoutSeconds, setTimeoutSeconds] = useState<string>('')

  const [loading, setLoading] = useState<boolean>(true)
  const [submitting, setSubmitting] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)

  // Load initial data
  useEffect(() => {
    const initData = async () => {
      try {
        setLoading(true)
        setError(null)
        const [testersData, targetsData, defsData] = await Promise.all([
          api.testers.list(),
          api.targets.list(),
          api.definitions.list(),
        ])

        setTesters(testersData)
        setTargets(targetsData)
        setDefinitions(defsData)

        if (testersData.length > 0) {
          const first = testersData[0]
          setSelectedTester(first.name)
          setSelectedSuites(first.suites || [])
          setDefinitionName(`${first.name}-run-${new Date().toISOString().slice(0, 10)}`)
        }
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : String(err))
      } finally {
        setLoading(false)
      }
    }
    initData()
  }, [])

  // When tester changes, update suites and default name
  const handleTesterChange = (testerName: string) => {
    setSelectedTester(testerName)
    const t = testers.find((item) => item.name === testerName)
    if (t) {
      setSelectedSuites(t.suites || [])
      setDefinitionName(`${t.name}-run-${new Date().toISOString().slice(0, 10)}`)
      setConfigValues({})
    }
  }

  // Load an existing definition into the form
  const handleLoadDefinition = (defId: string) => {
    if (!defId) return
    const id = parseInt(defId, 10)
    const def = definitions.find((d) => d.id === id)
    if (def) {
      setSelectedTester(def.tester)
      setDefinitionName(`${def.name} (copy)`)
      setSelectedTargetId(def.target_id ? String(def.target_id) : '')
      setSelectedSuites(def.suites || [])
      setConfigValues(def.config || {})
    }
  }

  const currentTesterInfo = testers.find((t) => t.name === selectedTester)

  const toggleSuite = (suiteName: string) => {
    if (selectedSuites.includes(suiteName)) {
      setSelectedSuites(selectedSuites.filter((s) => s !== suiteName))
    } else {
      setSelectedSuites([...selectedSuites, suiteName])
    }
  }

  const handleSubmit = async (startRunImmediately: boolean) => {
    setError(null)
    setSuccessMessage(null)

    if (!selectedTester) {
      setError('Please select a tester plugin')
      return
    }
    if (!definitionName.trim()) {
      setError('Please provide a name for this test definition')
      return
    }

    try {
      setSubmitting(true)
      const targetIdNum = selectedTargetId ? parseInt(selectedTargetId, 10) : null
      const timeoutVal = timeoutSeconds.trim() ? parseFloat(timeoutSeconds) : undefined

      // 1. Create or save definition
      const def = await api.definitions.create({
        name: definitionName.trim(),
        tester: selectedTester,
        suites: selectedSuites,
        config: configValues,
        target_id: targetIdNum,
      })

      if (startRunImmediately) {
        // 2. Trigger run immediately
        const run = await api.definitions.trigger(def.id, {
          timeout: timeoutVal,
        })
        // Navigate to the newly created run's detail page
        navigate(`/runs/${run.id}`)
      } else {
        setSuccessMessage(`Test definition "${def.name}" created successfully (ID #${def.id})!`)
        // Refresh definitions list
        const updatedDefs = await api.definitions.list()
        setDefinitions(updatedDefs)
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSubmitting(false)
    }
  }

  if (loading) {
    return <Spinner message="Loading testers and configuration schemas..." />
  }

  return (
    <div className="view-container trigger-view">
      <div className="view-header">
        <div>
          <h2>Run Test Suite</h2>
          <p className="view-subtitle">
            Configure test parameters via generated schema form, select targets, and trigger runs.
          </p>
        </div>
      </div>

      {error && <div className="banner error-banner">{error}</div>}
      {successMessage && <div className="banner success-banner">{successMessage}</div>}

      <div className="trigger-layout">
        <div className="trigger-form-card card">
          {/* Preset / Load definition selector */}
          {definitions.length > 0 && (
            <div className="form-group load-preset-group">
              <label htmlFor="load-def-select">Load from Existing Definition (Optional)</label>
              <select
                id="load-def-select"
                className="form-select"
                defaultValue=""
                onChange={(e) => handleLoadDefinition(e.target.value)}
              >
                <option value="">-- Choose a saved definition to load settings --</option>
                {definitions.map((d) => (
                  <option key={d.id} value={d.id}>
                    #{d.id}: {d.name} ({d.tester})
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className="form-row">
            {/* Tester Plugin selection */}
            <div className="form-group flex-1">
              <label htmlFor="tester-select">
                Tester Plugin <span className="required-star">*</span>
              </label>
              <select
                id="tester-select"
                className="form-select font-semibold"
                value={selectedTester}
                onChange={(e) => handleTesterChange(e.target.value)}
                disabled={submitting}
              >
                {testers.map((t) => (
                  <option key={t.name} value={t.name}>
                    {t.name.toUpperCase()} — {t.description.slice(0, 60)}...
                  </option>
                ))}
              </select>
            </div>

            {/* Target selection */}
            <div className="form-group flex-1">
              <label htmlFor="target-select">Target Endpoint (Optional)</label>
              <select
                id="target-select"
                className="form-select"
                value={selectedTargetId}
                onChange={(e) => setSelectedTargetId(e.target.value)}
                disabled={submitting}
              >
                <option value="">No Target (Standalone / Default)</option>
                {targets.map((tgt) => (
                  <option key={tgt.id} value={tgt.id}>
                    #{tgt.id} {tgt.name} ({tgt.kind}) — {tgt.base_url}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {currentTesterInfo?.description && (
            <div className="tester-description-box">
              <span className="info-icon">ℹ️</span>
              <p>{currentTesterInfo.description}</p>
            </div>
          )}

          <div className="form-row">
            {/* Definition Name */}
            <div className="form-group flex-2">
              <label htmlFor="definition-name">
                Definition Name <span className="required-star">*</span>
              </label>
              <input
                id="definition-name"
                type="text"
                className="form-input"
                value={definitionName}
                onChange={(e) => setDefinitionName(e.target.value)}
                placeholder="e.g. embedding-eval-baseline"
                disabled={submitting}
                required
              />
            </div>

            {/* Optional Timeout */}
            <div className="form-group flex-1">
              <label htmlFor="run-timeout">Timeout (Seconds)</label>
              <input
                id="run-timeout"
                type="number"
                className="form-input"
                value={timeoutSeconds}
                onChange={(e) => setTimeoutSeconds(e.target.value)}
                placeholder="e.g. 120"
                min={1}
                disabled={submitting}
              />
            </div>
          </div>

          {/* Suites selection */}
          {currentTesterInfo?.suites && currentTesterInfo.suites.length > 0 && (
            <div className="form-group">
              <div className="field-header">
                <label>Sub-suites to Run</label>
                <div className="suite-quick-actions">
                  <button
                    type="button"
                    className="btn-link"
                    onClick={() => setSelectedSuites(currentTesterInfo.suites)}
                  >
                    Select All
                  </button>
                  <span className="divider">|</span>
                  <button
                    type="button"
                    className="btn-link"
                    onClick={() => setSelectedSuites([])}
                  >
                    Clear All
                  </button>
                </div>
              </div>
              <div className="suites-selector-grid">
                {currentTesterInfo.suites.map((suite) => {
                  const isChecked = selectedSuites.includes(suite)
                  return (
                    <label key={suite} className={`suite-checkbox-card ${isChecked ? 'active' : ''}`}>
                      <input
                        type="checkbox"
                        checked={isChecked}
                        onChange={() => toggleSuite(suite)}
                        disabled={submitting}
                      />
                      <span className="suite-name">{suite}</span>
                    </label>
                  )
                })}
              </div>
            </div>
          )}

          {/* Generated Schema Form */}
          {currentTesterInfo?.config_schema && (
            <div className="schema-section">
              <SchemaForm
                schema={currentTesterInfo.config_schema}
                values={configValues}
                onChange={setConfigValues}
                disabled={submitting}
              />
            </div>
          )}

          {/* Form Actions */}
          <div className="form-submit-bar">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => handleSubmit(false)}
              disabled={submitting}
            >
              Save Definition Only
            </button>
            <button
              type="button"
              className="btn btn-primary btn-lg"
              onClick={() => handleSubmit(true)}
              disabled={submitting}
            >
              {submitting ? 'Starting Run...' : '🚀 Save & Start Run Now'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
