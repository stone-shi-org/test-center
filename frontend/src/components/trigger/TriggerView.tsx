import { useEffect, useState } from 'react'
import {
  api,
  type StarterTemplate,
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
  const [templates, setTemplates] = useState<TestDefinition[]>([])
  const [starters, setStarters] = useState<StarterTemplate[]>([])
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>('')

  const [selectedTester, setSelectedTester] = useState<string>('')
  const [selectedTargetId, setSelectedTargetId] = useState<string>('')
  const [selectedSuites, setSelectedSuites] = useState<string[]>([])
  const [configValues, setConfigValues] = useState<Record<string, unknown>>({})
  const [timeoutSeconds, setTimeoutSeconds] = useState<string>('')

  const [loading, setLoading] = useState<boolean>(true)
  const [submitting, setSubmitting] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  // Load initial data
  useEffect(() => {
    const initData = async () => {
      try {
        setLoading(true)
        setError(null)
        const [testersData, targetsData, defsData, startersData] = await Promise.all([
          api.testers.list(),
          api.targets.list(),
          api.definitions.list(),
          api.definitions.listStarters().catch(() => [] as StarterTemplate[]),
        ])

        setTesters(testersData)
        setTargets(targetsData)
        setTemplates(defsData)
        setStarters(startersData)

        if (testersData.length > 0) {
          const first = testersData[0]
          setSelectedTester(first.name)
          setSelectedSuites(first.suites ? [...first.suites] : [])
        }
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : String(err))
      } finally {
        setLoading(false)
      }
    }
    initData()
  }, [])

  // When tester changes, update suites and reset config
  const handleTesterChange = (testerName: string) => {
    setSelectedTester(testerName)
    const t = testers.find((item) => item.name === testerName)
    if (t) {
      setSelectedSuites(t.suites ? [...t.suites] : [])
      setConfigValues({})
    }
  }

  // Load a template or starter into the form without modifying the template
  const handleLoadTemplate = (val: string) => {
    setSelectedTemplateId(val)
    if (!val) return
    if (val.startsWith('starter:')) {
      const idx = parseInt(val.replace('starter:', ''), 10)
      const starter = starters[idx]
      if (starter) {
        setSelectedTester(starter.tester)
        setSelectedTargetId('')
        setSelectedSuites(starter.suites ? [...starter.suites] : [])
        setConfigValues(starter.config ? { ...starter.config } : {})
      }
      return
    }
    const id = parseInt(val, 10)
    const tpl = templates.find((d) => d.id === id)
    if (tpl) {
      setSelectedTester(tpl.tester)
      setSelectedTargetId(tpl.target_id ? String(tpl.target_id) : '')
      setSelectedSuites(tpl.suites ? [...tpl.suites] : [])
      setConfigValues(tpl.config ? { ...tpl.config } : {})
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

  const handleStartRun = async () => {
    setError(null)

    if (!selectedTester) {
      setError('Please select a tester plugin')
      return
    }

    try {
      setSubmitting(true)
      const targetIdNum = selectedTargetId ? parseInt(selectedTargetId, 10) : null
      const timeoutVal = timeoutSeconds.trim() ? parseFloat(timeoutSeconds) : undefined

      const run = await api.runs.create({
        tester: selectedTester,
        suites: selectedSuites,
        config: configValues,
        target_id: targetIdNum,
        timeout: timeoutVal,
      })

      navigate(`/runs/${run.id}`)
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

      {error && (
        <div className="banner error-banner" role="alert">
          {error}
        </div>
      )}

      <div className="trigger-layout">
        <div className="trigger-form-card card">
          {/* Load from template selector */}
          <div className="form-group load-preset-group">
            <label htmlFor="load-template-select">Load from template</label>
            <select
              id="load-template-select"
              className="form-select"
              value={selectedTemplateId}
              onChange={(e) => handleLoadTemplate(e.target.value)}
              disabled={submitting}
            >
              <option value="">-- Choose a template to load settings --</option>
              {starters.length > 0 && (
                <optgroup label="Starters">
                  {starters.map((starter, idx) => (
                    <option key={`starter-${starter.tester}-${starter.name}`} value={`starter:${idx}`}>
                      {starter.name} ({starter.tester})
                    </option>
                  ))}
                </optgroup>
              )}
              {templates.length > 0 && (
                <optgroup label="Saved Templates">
                  {templates.map((tpl) => (
                    <option key={tpl.id} value={String(tpl.id)}>
                      {tpl.name} ({tpl.tester})
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
          </div>

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

          {currentTesterInfo?.description && (
            <div className="tester-description-box">
              <span className="info-icon">ℹ️</span>
              <p>{currentTesterInfo.description}</p>
            </div>
          )}

          {/* Suites selection */}
          {currentTesterInfo?.suites && currentTesterInfo.suites.length > 0 && (
            <div className="form-group">
              <div className="field-header">
                <label>Sub-suites to Run</label>
                <div className="suite-quick-actions">
                  <button
                    type="button"
                    className="btn-link"
                    onClick={() => setSelectedSuites([...currentTesterInfo.suites])}
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
                context={{
                  targetId: selectedTargetId ? parseInt(selectedTargetId, 10) : null,
                }}
              />
            </div>
          )}

          {/* Form Actions */}
          <div className="form-submit-bar">
            <button
              type="button"
              className="btn btn-primary btn-lg"
              onClick={handleStartRun}
              disabled={submitting}
            >
              {submitting ? 'Starting Run...' : 'Start Run'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
