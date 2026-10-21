import { useCallback, useEffect, useState } from 'react'
import {
  api,
  type Job,
  type StarterTemplate,
  type Target,
  type TestDefinition,
  type TesterInfo,
} from '../../api/index.ts'
import { ConfirmDialog } from '../common/ConfirmDialog.tsx'
import { Modal } from '../common/Modal.tsx'
import { Spinner } from '../common/Spinner.tsx'
import { SchemaForm } from '../schema/SchemaForm.tsx'
import { formatDateTime } from '../../utils/date.ts'

function formatUpdatedTime(iso: string | null | undefined): string {
  return formatDateTime(iso)
}

function buildCopyName(baseName: string, existingNames: Set<string>): string {
  const firstCandidate = `${baseName} (copy)`
  if (!existingNames.has(firstCandidate)) {
    return firstCandidate
  }
  let counter = 2
  while (existingNames.has(`${baseName} (copy ${counter})`)) {
    counter += 1
  }
  return `${baseName} (copy ${counter})`
}

function stripSuitesFromConfig(cfg?: Record<string, unknown> | null): Record<string, unknown> {
  if (!cfg) return {}
  const next = { ...cfg }
  delete next.suites
  return next
}

export function TemplatesView() {
  const [templates, setTemplates] = useState<TestDefinition[]>([])
  const [starters, setStarters] = useState<StarterTemplate[]>([])
  const [testers, setTesters] = useState<TesterInfo[]>([])
  const [targets, setTargets] = useState<Target[]>([])
  const [jobs, setJobs] = useState<Job[]>([])

  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)
  const [search, setSearch] = useState<string>('')
  const [testerFilter, setTesterFilter] = useState<string>('all')

  // Create / Edit modal state
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false)
  const [editingTemplate, setEditingTemplate] = useState<TestDefinition | null>(null)
  const [templateName, setTemplateName] = useState<string>('')
  const [selectedTester, setSelectedTester] = useState<string>('')
  const [selectedTargetId, setSelectedTargetId] = useState<string>('')
  const [selectedSuites, setSelectedSuites] = useState<string[]>([])
  const [configValues, setConfigValues] = useState<Record<string, unknown>>({})
  const [formError, setFormError] = useState<string | null>(null)
  const [saving, setSaving] = useState<boolean>(false)
  const [duplicatingId, setDuplicatingId] = useState<number | null>(null)
  const [creatingStarterName, setCreatingStarterName] = useState<string | null>(null)

  // Delete confirmation state
  const [deletingTemplate, setDeletingTemplate] = useState<TestDefinition | null>(null)

  const loadAll = useCallback(async (showSpinner = false) => {
    try {
      if (showSpinner) setLoading(true)
      setError(null)
      const [defsData, testersData, targetsData, jobsData, startersData] = await Promise.all([
        api.definitions.list(),
        api.testers.list(),
        api.targets.list(),
        api.jobs.list().catch(() => [] as Job[]),
        api.definitions.listStarters().catch(() => [] as StarterTemplate[]),
      ])
      setTemplates(defsData)
      setTesters(testersData)
      setTargets(targetsData)
      setJobs(jobsData)
      setStarters(startersData)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      if (showSpinner) setLoading(false)
    }
  }, [])

  useEffect(() => {
    let ignore = false
    Promise.all([
      api.definitions.list(),
      api.testers.list(),
      api.targets.list(),
      api.jobs.list().catch(() => [] as Job[]),
      api.definitions.listStarters().catch(() => [] as StarterTemplate[]),
    ])
      .then(([defsData, testersData, targetsData, jobsData, startersData]) => {
        if (ignore) return
        setTemplates(defsData)
        setTesters(testersData)
        setTargets(targetsData)
        setJobs(jobsData)
        setStarters(startersData)
        setLoading(false)
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

  const handleUseStarter = async (starter: StarterTemplate) => {
    setError(null)
    setSuccessMessage(null)
    try {
      setCreatingStarterName(starter.name)
      const existingNames = new Set(templates.map((t) => t.name))
      const name = existingNames.has(starter.name)
        ? buildCopyName(starter.name, existingNames)
        : starter.name
      const created = await api.definitions.create({
        name,
        tester: starter.tester,
        suites: starter.suites ? [...starter.suites] : [],
        config: stripSuitesFromConfig(starter.config),
        target_id: null,
      })
      setSuccessMessage(`Created template "${created.name}" from starter.`)
      await loadAll(false)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setCreatingStarterName(null)
    }
  }

  const openCreateModal = () => {
    setEditingTemplate(null)
    setFormError(null)
    const firstTester = testers[0]
    setTemplateName('')
    setSelectedTester(firstTester ? firstTester.name : '')
    setSelectedTargetId('')
    setSelectedSuites(firstTester?.suites ? [...firstTester.suites] : [])
    setConfigValues({})
    setIsModalOpen(true)
  }

  const openEditModal = (tpl: TestDefinition) => {
    setEditingTemplate(tpl)
    setFormError(null)
    setTemplateName(tpl.name)
    setSelectedTester(tpl.tester)
    setSelectedTargetId(tpl.target_id ? String(tpl.target_id) : '')
    const fallbackSuites = Array.isArray(tpl.config?.suites)
      ? (tpl.config.suites as string[])
      : []
    setSelectedSuites(tpl.suites ? [...tpl.suites] : [...fallbackSuites])
    setConfigValues(stripSuitesFromConfig(tpl.config))
    setIsModalOpen(true)
  }

  const closeModal = () => {
    setIsModalOpen(false)
    setEditingTemplate(null)
    setFormError(null)
  }

  const handleTesterChange = (testerName: string) => {
    setSelectedTester(testerName)
    const t = testers.find((item) => item.name === testerName)
    if (t) {
      setSelectedSuites(t.suites ? [...t.suites] : [])
      setConfigValues({})
    }
  }

  const toggleSuite = (suiteName: string) => {
    if (selectedSuites.includes(suiteName)) {
      setSelectedSuites(selectedSuites.filter((s) => s !== suiteName))
    } else {
      setSelectedSuites([...selectedSuites, suiteName])
    }
  }

  const handleSaveTemplate = async () => {
    setFormError(null)
    setError(null)
    setSuccessMessage(null)

    if (!templateName.trim()) {
      setFormError('Template name is required')
      return
    }
    if (!selectedTester) {
      setFormError('Please select a tester plugin')
      return
    }

    const payload = {
      name: templateName.trim(),
      tester: selectedTester,
      suites: selectedSuites,
      config: stripSuitesFromConfig(configValues),
      target_id: selectedTargetId ? parseInt(selectedTargetId, 10) : null,
    }

    try {
      setSaving(true)
      if (editingTemplate) {
        const updated = await api.definitions.update(editingTemplate.id, payload)
        setSuccessMessage(`Template "${updated.name}" updated.`)
      } else {
        const created = await api.definitions.create(payload)
        setSuccessMessage(`Template "${created.name}" created.`)
      }
      closeModal()
      await loadAll(false)
    } catch (err: unknown) {
      setFormError(err instanceof Error ? err.message : String(err))
    } finally {
      setSaving(false)
    }
  }

  const handleDuplicate = async (tpl: TestDefinition) => {
    setError(null)
    setSuccessMessage(null)
    try {
      setDuplicatingId(tpl.id)
      const existingNames = new Set(templates.map((t) => t.name))
      const copyName = buildCopyName(tpl.name, existingNames)
      const fallbackSuites = Array.isArray(tpl.config?.suites)
        ? (tpl.config.suites as string[])
        : []
      const created = await api.definitions.create({
        name: copyName,
        tester: tpl.tester,
        suites: tpl.suites ? [...tpl.suites] : [...fallbackSuites],
        config: stripSuitesFromConfig(tpl.config),
        target_id: tpl.target_id ?? null,
      })
      setSuccessMessage(`Duplicated template as "${created.name}".`)
      await loadAll(false)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setDuplicatingId(null)
    }
  }

  const handleDeleteConfirm = async () => {
    if (!deletingTemplate) return
    const targetToDelete = deletingTemplate
    setError(null)
    setSuccessMessage(null)
    try {
      await api.definitions.delete(targetToDelete.id)
      setDeletingTemplate(null)
      setSuccessMessage(`Template "${targetToDelete.name}" deleted.`)
      await loadAll(false)
    } catch (err: unknown) {
      setDeletingTemplate(null)
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const getReferencingJobs = (templateId: number): Job[] =>
    jobs.filter((job) => (job.definition_ids || []).includes(templateId))

  const getTargetLabel = (tpl: TestDefinition): string => {
    if (tpl.target?.name) {
      return tpl.target.name
    }
    if (tpl.target_id !== null && tpl.target_id !== undefined) {
      const found = targets.find((t) => t.id === tpl.target_id)
      return found ? found.name : `#${tpl.target_id}`
    }
    return '—'
  }

  const currentTesterInfo = testers.find((t) => t.name === selectedTester)

  const uniqueTesters = Array.from(new Set(templates.map((t) => t.tester)))

  const filteredTemplates = templates.filter((tpl) => {
    const matchesTester = testerFilter === 'all' || tpl.tester === testerFilter
    const q = search.toLowerCase()
    const targetLabel = getTargetLabel(tpl).toLowerCase()
    const matchesSearch =
      !q ||
      tpl.name.toLowerCase().includes(q) ||
      tpl.tester.toLowerCase().includes(q) ||
      targetLabel.includes(q) ||
      (tpl.suites || []).some((s) => s.toLowerCase().includes(q))
    return matchesTester && matchesSearch
  })

  const filteredStarters = starters.filter((s) => {
    const matchesTester = testerFilter === 'all' || s.tester === testerFilter
    const q = search.toLowerCase()
    const matchesSearch =
      !q ||
      s.name.toLowerCase().includes(q) ||
      s.tester.toLowerCase().includes(q) ||
      s.description.toLowerCase().includes(q) ||
      s.suites.some((suite) => suite.toLowerCase().includes(q))
    return matchesTester && matchesSearch
  })

  const deletingUsedByJobs = deletingTemplate ? getReferencingJobs(deletingTemplate.id) : []
  const deleteDialogMessage = deletingTemplate
    ? deletingUsedByJobs.length > 0
      ? `Template "${deletingTemplate.name}" is referenced by job(s): ${deletingUsedByJobs
          .map((j) => `${j.name} (#${j.id})`)
          .join(', ')}. Remove it from those jobs before deleting.`
      : `Are you sure you want to delete template "${deletingTemplate.name}"?`
    : ''

  if (loading) {
    return <Spinner message="Loading templates..." />
  }

  return (
    <div className="view-container templates-view">
      <div className="view-header">
        <div>
          <h2>Run Templates</h2>
          <p className="view-subtitle">
            Manage reusable test configurations to load into the Run Trigger page or bundle into Jobs.
          </p>
        </div>
        <div className="header-actions">
          <button
            type="button"
            className="btn btn-primary"
            onClick={openCreateModal}
          >
            <span className="btn-icon">+</span> New Template
          </button>
        </div>
      </div>

      {error && (
        <div className="banner error-banner" role="alert">
          {error}
        </div>
      )}
      {successMessage && <div className="banner success-banner">{successMessage}</div>}

      <div className="filter-bar">
        <div className="search-input-wrapper">
          <span className="search-icon">🔍</span>
          <input
            type="text"
            className="search-input"
            placeholder="Search templates by name, tester, target, or suite..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <div className="filter-group">
          <label htmlFor="template-tester-filter" className="filter-label">
            Tester:
          </label>
          <select
            id="template-tester-filter"
            className="filter-select"
            value={testerFilter}
            onChange={(e) => setTesterFilter(e.target.value)}
          >
            <option value="all">All Testers ({templates.length})</option>
            {uniqueTesters.map((t) => (
              <option key={t} value={t}>
                {t} ({templates.filter((tpl) => tpl.tester === t).length})
              </option>
            ))}
          </select>
        </div>
      </div>

      {filteredTemplates.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">📋</div>
          <h3>No templates found</h3>
          <p>
            {templates.length === 0
              ? 'No run templates created yet. Create a template to reuse configurations across runs and jobs.'
              : 'No templates match your current search and filter criteria.'}
          </p>
          {templates.length === 0 && (
            <button type="button" className="btn btn-primary" onClick={openCreateModal}>
              New Template
            </button>
          )}
        </div>
      ) : (
        <div className="table-responsive">
          <table className="data-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Tester</th>
                <th>Target</th>
                <th>Suites</th>
                <th>Updated</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredTemplates.map((tpl) => {
                const usedByJobs = getReferencingJobs(tpl.id)
                return (
                  <tr key={tpl.id}>
                    <td>
                      <div className="target-name-cell">
                        <strong>{tpl.name}</strong>
                        {usedByJobs.length > 0 && (
                          <span
                            className="suite-pill"
                            title={`Used by job(s): ${usedByJobs.map((j) => j.name).join(', ')}`}
                          >
                            {usedByJobs.length} {usedByJobs.length === 1 ? 'job' : 'jobs'}
                          </span>
                        )}
                      </div>
                    </td>
                    <td>
                      <span className="kind-badge">{tpl.tester}</span>
                    </td>
                    <td>{getTargetLabel(tpl)}</td>
                    <td>
                      {tpl.suites && tpl.suites.length > 0 ? (
                        <div className="target-capabilities-badges">
                          {tpl.suites.map((suite) => (
                            <span key={suite} className="suite-pill">
                              {suite}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <span className="text-muted">All</span>
                      )}
                    </td>
                    <td className="text-muted">{formatUpdatedTime(tpl.updated_at)}</td>
                    <td>
                      <div className="row-actions">
                        <button
                          type="button"
                          className="action-btn"
                          title="Edit Template"
                          onClick={() => openEditModal(tpl)}
                        >
                          Edit
                        </button>
                        <button
                          type="button"
                          className="action-btn"
                          title="Duplicate Template"
                          disabled={duplicatingId === tpl.id}
                          onClick={() => handleDuplicate(tpl)}
                        >
                          {duplicatingId === tpl.id ? 'Duplicating...' : 'Duplicate'}
                        </button>
                        <button
                          type="button"
                          className="action-btn danger"
                          title="Delete Template"
                          onClick={() => setDeletingTemplate(tpl)}
                        >
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Built-in Starter Templates Section */}
      <div className="starters-section card">
        <div className="starters-section-header">
          <div>
            <h3>Starter Templates</h3>
            <p className="view-subtitle">
              Built-in starter configurations for each tester plugin. Click &quot;Use as template&quot; to copy any starter into your templates to customize targets, suites, or parameters.
            </p>
          </div>
        </div>

        {filteredStarters.length === 0 ? (
          <p className="text-muted starters-empty-notice">No starter templates match the selected filter.</p>
        ) : (
          <div className="starters-grid">
            {filteredStarters.map((starter) => (
              <div key={`${starter.tester}-${starter.name}`} className="starter-card">
                <div className="starter-card-top">
                  <div className="starter-title-area">
                    <h4 className="starter-title">{starter.name}</h4>
                    <span className="kind-badge">{starter.tester}</span>
                  </div>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    disabled={creatingStarterName === starter.name}
                    onClick={() => handleUseStarter(starter)}
                  >
                    {creatingStarterName === starter.name ? 'Copying...' : 'Use as template'}
                  </button>
                </div>
                <p className="starter-desc">{starter.description}</p>
                <div className="starter-suites">
                  <span className="starter-suites-label">Suites:</span>
                  <div className="target-capabilities-badges">
                    {starter.suites.map((s) => (
                      <span key={s} className="suite-pill">
                        {s}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Create / Edit Template Modal */}
      <Modal
        isOpen={isModalOpen}
        onClose={closeModal}
        title={editingTemplate ? `Edit Template: ${editingTemplate.name}` : 'New Template'}
        maxWidth="720px"
      >
        <div className="template-form">
          {formError && (
            <div className="banner error-banner" role="alert">
              {formError}
            </div>
          )}

          <div className="form-group">
            <label htmlFor="template-name">
              Template Name <span className="required-star">*</span>
            </label>
            <input
              id="template-name"
              type="text"
              className="form-input"
              value={templateName}
              onChange={(e) => setTemplateName(e.target.value)}
              placeholder="e.g. embedding-eval-baseline"
              disabled={saving}
              required
            />
          </div>

          <div className="form-row">
            <div className="form-group flex-1">
              <label htmlFor="template-tester">
                Tester Plugin <span className="required-star">*</span>
              </label>
              <select
                id="template-tester"
                className="form-select font-semibold"
                value={selectedTester}
                onChange={(e) => handleTesterChange(e.target.value)}
                disabled={saving}
              >
                {testers.map((t) => (
                  <option key={t.name} value={t.name}>
                    {t.name.toUpperCase()} — {t.description.slice(0, 60)}...
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group flex-1">
              <label htmlFor="template-target">Target Endpoint (Optional)</label>
              <select
                id="template-target"
                className="form-select"
                value={selectedTargetId}
                onChange={(e) => setSelectedTargetId(e.target.value)}
                disabled={saving}
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
                    <label
                      key={suite}
                      className={`suite-checkbox-card ${isChecked ? 'active' : ''}`}
                    >
                      <input
                        type="checkbox"
                        checked={isChecked}
                        onChange={() => toggleSuite(suite)}
                        disabled={saving}
                      />
                      <span className="suite-name">{suite}</span>
                    </label>
                  )
                })}
              </div>
            </div>
          )}

          {currentTesterInfo?.config_schema && (
            <div className="schema-section">
              <SchemaForm
                schema={currentTesterInfo.config_schema}
                values={configValues}
                onChange={setConfigValues}
                disabled={saving}
                omitFields={
                  currentTesterInfo.suites && currentTesterInfo.suites.length > 0
                    ? ['suites']
                    : undefined
                }
                context={{
                  targetId: selectedTargetId ? parseInt(selectedTargetId, 10) : null,
                }}
              />
            </div>
          )}

          <div className="modal-actions">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={closeModal}
              disabled={saving}
            >
              Cancel
            </button>
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleSaveTemplate}
              disabled={saving}
            >
              {saving
                ? 'Saving...'
                : editingTemplate
                  ? 'Save Changes'
                  : 'Create Template'}
            </button>
          </div>
        </div>
      </Modal>

      {/* Delete Confirmation Dialog */}
      <ConfirmDialog
        isOpen={Boolean(deletingTemplate)}
        title="Delete Template"
        message={deleteDialogMessage}
        confirmLabel="Delete Template"
        danger
        onConfirm={handleDeleteConfirm}
        onCancel={() => setDeletingTemplate(null)}
      />
    </div>
  )
}
