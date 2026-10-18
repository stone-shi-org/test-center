import { useCallback, useEffect, useState } from 'react'
import {
  api,
  type Job,
  type JobCreate,
  type JobRunMode,
  type JobStepOverride,
  type MissedRunPolicy,
  type Run,
  type Target,
  type TestDefinition,
  type TesterInfo,
} from '../../api/index.ts'
import { Link, useRouter } from '../../router/index.ts'
import { Badge } from '../common/Badge.tsx'
import { ConfirmDialog } from '../common/ConfirmDialog.tsx'
import { Spinner } from '../common/Spinner.tsx'
import { SchemaForm } from '../schema/SchemaForm.tsx'
import {
  formatUtcTimestamp,
  isManualSchedule,
  SCHEDULE_PRESETS,
  validateCronSchedule,
} from './cronUtils.ts'

interface BundledStepState {
  test_definition_id: number
  overrideEnabled: boolean
  suitesOverride: string[] | undefined
  targetIdOverride: string
  timeoutOverride: string
  configOverride: Record<string, unknown>
}

export function JobsView({ initialJobId }: { initialJobId?: number }) {
  const { navigate } = useRouter()

  const [jobs, setJobs] = useState<Job[]>([])
  const [definitions, setDefinitions] = useState<TestDefinition[]>([])
  const [testers, setTesters] = useState<TesterInfo[]>([])
  const [targets, setTargets] = useState<Target[]>([])
  const [runs, setRuns] = useState<Run[]>([])

  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)

  const [search, setSearch] = useState<string>('')
  const [modeFilter, setModeFilter] = useState<string>('all')
  const [statusFilter, setStatusFilter] = useState<string>('all')

  // Editor state
  const [isEditorOpen, setIsEditorOpen] = useState<boolean>(false)
  const [editingJob, setEditingJob] = useState<Job | null>(null)
  const [jobName, setJobName] = useState<string>('')
  const [jobDescription, setJobDescription] = useState<string>('')
  const [jobRunMode, setJobRunMode] = useState<JobRunMode>('sequential')
  const [jobSchedule, setJobSchedule] = useState<string>('')
  const [jobMissedRunPolicy, setJobMissedRunPolicy] = useState<MissedRunPolicy>('skip')
  const [jobEnabled, setJobEnabled] = useState<boolean>(true)
  const [steps, setSteps] = useState<BundledStepState[]>([])
  const [expandedStepIdx, setExpandedStepIdx] = useState<number | null>(null)
  const [selectedDefToAdd, setSelectedDefToAdd] = useState<string>('')
  const [savingJob, setSavingJob] = useState<boolean>(false)
  const [runningJobId, setRunningJobId] = useState<number | null>(null)
  const [togglingJobId, setTogglingJobId] = useState<number | null>(null)

  // Notification settings state
  const [notifyEnabled, setNotifyEnabled] = useState<boolean>(false)
  const [notifyOnFailure, setNotifyOnFailure] = useState<boolean>(true)
  const [notifyOnRegression, setNotifyOnRegression] = useState<boolean>(true)
  const [regressionThreshold, setRegressionThreshold] = useState<string>('0')
  const [webhookUrl, setWebhookUrl] = useState<string>('')
  const [webhookSecretEnvVar, setWebhookSecretEnvVar] = useState<string>('')
  const [emailTo, setEmailTo] = useState<string>('')
  const [testingNotification, setTestingNotification] = useState<boolean>(false)
  const [showBambooSnippet, setShowBambooSnippet] = useState<boolean>(false)

  // Inline Quick Definition Builder state (uses SchemaForm)
  const [showDefBuilder, setShowDefBuilder] = useState<boolean>(false)
  const [newDefName, setNewDefName] = useState<string>('')
  const [newDefTester, setNewDefTester] = useState<string>('')
  const [newDefTargetId, setNewDefTargetId] = useState<string>('')
  const [newDefSuites, setNewDefSuites] = useState<string[]>([])
  const [newDefConfig, setNewDefConfig] = useState<Record<string, unknown>>({})
  const [creatingDef, setCreatingDef] = useState<boolean>(false)

  // Delete confirmation state
  const [deletingJob, setDeletingJob] = useState<Job | null>(null)

  const loadAll = useCallback(async (showSpinner = false) => {
    try {
      if (showSpinner) setLoading(true)
      setError(null)
      const [jobsData, defsData, testersData, targetsData, runsData] = await Promise.all([
        api.jobs.list(),
        api.definitions.list(),
        api.testers.list(),
        api.targets.list(),
        api.runs.list({ limit: 100 }),
      ])
      setJobs(jobsData)
      setDefinitions(defsData)
      setTesters(testersData)
      setTargets(targetsData)
      setRuns(runsData)

      if (defsData.length > 0) {
        setSelectedDefToAdd((prev) => prev || String(defsData[0].id))
      }
      if (testersData.length > 0) {
        setNewDefTester((prev) => {
          if (prev) return prev
          const first = testersData[0]
          setNewDefSuites(first.suites || [])
          setNewDefName(`${first.name}-def-${new Date().toISOString().slice(0, 10)}`)
          return first.name
        })
      }
      return jobsData
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
      return []
    } finally {
      if (showSpinner) setLoading(false)
    }
  }, [])

  const openEditorForJob = useCallback((job: Job) => {
    setEditingJob(job)
    setJobName(job.name)
    setJobDescription(job.description || '')
    setJobRunMode(job.run_mode || 'sequential')
    setJobSchedule(job.schedule || '')
    setJobMissedRunPolicy(job.missed_run_policy || 'skip')
    setJobEnabled(job.enabled ?? true)

    const notif = job.notifications
    setNotifyEnabled(
      notif?.enabled ?? Boolean(notif?.webhook_url || (notif?.email_to && notif.email_to.length > 0))
    )
    setNotifyOnFailure(notif?.on_failure ?? job.notify_on_failure ?? true)
    setNotifyOnRegression(notif?.on_regression ?? job.notify_on_regression ?? true)
    setRegressionThreshold(
      String(notif?.regression_threshold ?? job.regression_threshold ?? 0)
    )
    setWebhookUrl(notif?.webhook_url || job.webhook_url || '')
    setWebhookSecretEnvVar(notif?.webhook_secret_env_var || '')
    setEmailTo((notif?.email_to || job.email_to || []).join(', '))

    const initialSteps: BundledStepState[] = (job.definition_ids || []).map((defId, idx) => {
      const defItem = job.definitions?.[idx]
      const rawOv =
        defItem?.overrides ||
        ((job.overrides?.[String(defId)] as JobStepOverride | undefined) ?? {})
      const hasOv = Boolean(
        rawOv &&
          (rawOv.suites !== undefined ||
            (rawOv.config && Object.keys(rawOv.config).length > 0) ||
            rawOv.target_id !== undefined ||
            rawOv.timeout !== undefined)
      )
      return {
        test_definition_id: defId,
        overrideEnabled: hasOv,
        suitesOverride: rawOv?.suites ? [...rawOv.suites] : undefined,
        targetIdOverride:
          rawOv?.target_id !== undefined && rawOv?.target_id !== null
            ? String(rawOv.target_id)
            : '',
        timeoutOverride:
          rawOv?.timeout !== undefined && rawOv?.timeout !== null ? String(rawOv.timeout) : '',
        configOverride: rawOv?.config ? { ...rawOv.config } : {},
      }
    })

    setSteps(initialSteps)
    setExpandedStepIdx(initialSteps.length > 0 ? 0 : null)
    setIsEditorOpen(true)
  }, [])

  useEffect(() => {
    let ignore = false
    Promise.all([
      api.jobs.list(),
      api.definitions.list(),
      api.testers.list(),
      api.targets.list(),
      api.runs.list({ limit: 100 }),
    ])
      .then(([jobsData, defsData, testersData, targetsData, runsData]) => {
        if (ignore) return
        setJobs(jobsData)
        setDefinitions(defsData)
        setTesters(testersData)
        setTargets(targetsData)
        setRuns(runsData)
        if (defsData.length > 0) {
          setSelectedDefToAdd((prev) => prev || String(defsData[0].id))
        }
        if (testersData.length > 0) {
          const first = testersData[0]
          setNewDefTester((prev) => prev || first.name)
          setNewDefSuites((prev) => (prev.length > 0 ? prev : first.suites || []))
          setNewDefName(
            (prev) => prev || `${first.name}-def-${new Date().toISOString().slice(0, 10)}`
          )
        }
        if (initialJobId && jobsData.length > 0) {
          const found = jobsData.find((j) => j.id === initialJobId)
          if (found) {
            openEditorForJob(found)
          }
        }
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
  }, [initialJobId, openEditorForJob])

  const openEditorForNew = () => {
    setEditingJob(null)
    setJobName('')
    setJobDescription('')
    setJobRunMode('sequential')
    setJobSchedule('')
    setJobMissedRunPolicy('skip')
    setJobEnabled(true)
    setNotifyEnabled(false)
    setNotifyOnFailure(true)
    setNotifyOnRegression(true)
    setRegressionThreshold('0')
    setWebhookUrl('')
    setWebhookSecretEnvVar('')
    setEmailTo('')
    setSteps([])
    setExpandedStepIdx(null)
    setShowDefBuilder(definitions.length === 0)
    if (definitions.length > 0 && !selectedDefToAdd) {
      setSelectedDefToAdd(String(definitions[0].id))
    }
    setIsEditorOpen(true)
  }

  const closeEditor = () => {
    setIsEditorOpen(false)
    setEditingJob(null)
    setShowDefBuilder(false)
    if (initialJobId) {
      navigate('/jobs')
    }
  }

  const handleAddStep = (defIdNum?: number) => {
    const idToAdd = defIdNum ?? (selectedDefToAdd ? parseInt(selectedDefToAdd, 10) : NaN)
    if (Number.isNaN(idToAdd)) return
    const def = definitions.find((d) => d.id === idToAdd)
    if (!def) return

    setSteps((prev) => {
      if (prev.some((s) => s.test_definition_id === idToAdd)) {
        return prev
      }
      const next = [
        ...prev,
        {
          test_definition_id: idToAdd,
          overrideEnabled: false,
          suitesOverride: undefined,
          targetIdOverride: '',
          timeoutOverride: '',
          configOverride: { ...(def.config || {}) },
        },
      ]
      setExpandedStepIdx(next.length - 1)
      return next
    })
  }

  const handleRemoveStep = (index: number) => {
    setSteps((prev) => prev.filter((_, idx) => idx !== index))
    setExpandedStepIdx((prev) => (prev === index ? null : prev !== null && prev > index ? prev - 1 : prev))
  }

  const handleMoveStep = (index: number, direction: -1 | 1) => {
    const targetIdx = index + direction
    if (targetIdx < 0 || targetIdx >= steps.length) return
    setSteps((prev) => {
      const copy = [...prev]
      const [item] = copy.splice(index, 1)
      copy.splice(targetIdx, 0, item)
      return copy
    })
    setExpandedStepIdx(targetIdx)
  }

  const updateStepAt = (index: number, updater: (prev: BundledStepState) => BundledStepState) => {
    setSteps((prev) => prev.map((s, idx) => (idx === index ? updater(s) : s)))
  }

  // Inline test definition creation using SchemaForm
  const handleNewDefTesterChange = (testerName: string) => {
    setNewDefTester(testerName)
    const t = testers.find((item) => item.name === testerName)
    if (t) {
      setNewDefSuites(t.suites || [])
      setNewDefName(`${t.name}-def-${new Date().toISOString().slice(0, 10)}`)
      setNewDefConfig({})
    }
  }

  const handleCreateInlineDefinition = async () => {
    setError(null)
    if (!newDefTester) {
      setError('Please select a tester plugin for the new test definition')
      return
    }
    if (!newDefName.trim()) {
      setError('Please enter a name for the new test definition')
      return
    }

    try {
      setCreatingDef(true)
      const created = await api.definitions.create({
        name: newDefName.trim(),
        tester: newDefTester,
        suites: newDefSuites,
        config: newDefConfig,
        target_id: newDefTargetId ? parseInt(newDefTargetId, 10) : null,
      })
      const updatedDefs = await api.definitions.list()
      setDefinitions(updatedDefs)
      setSelectedDefToAdd(String(created.id))
      setSteps((prev) => {
        const next = [
          ...prev,
          {
            test_definition_id: created.id,
            overrideEnabled: false,
            suitesOverride: undefined,
            targetIdOverride: '',
            timeoutOverride: '',
            configOverride: { ...(created.config || {}) },
          },
        ]
        setExpandedStepIdx(next.length - 1)
        return next
      })
      setShowDefBuilder(false)
      setSuccessMessage(`Created definition "${created.name}" and added to job.`)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setCreatingDef(false)
    }
  }

  const buildJobPayload = (): JobCreate => {
    const definition_ids = steps.map((s) => s.test_definition_id)
    const overrides: Record<string, JobStepOverride> = {}

    for (const step of steps) {
      const def = definitions.find((d) => d.id === step.test_definition_id)
      const stepOv: JobStepOverride = {}

      if (step.suitesOverride !== undefined) {
        stepOv.suites = step.suitesOverride
      }
      if (step.targetIdOverride.trim() !== '') {
        stepOv.target_id = parseInt(step.targetIdOverride, 10)
      }
      if (step.timeoutOverride.trim() !== '') {
        stepOv.timeout = parseFloat(step.timeoutOverride)
      }

      // Check if configOverride has keys or differs from base definition config
      const baseConfig = def?.config || {}
      const cfgDiff: Record<string, unknown> = {}
      for (const [k, v] of Object.entries(step.configOverride)) {
        if (v !== undefined && JSON.stringify(baseConfig[k]) !== JSON.stringify(v)) {
          cfgDiff[k] = v
        }
      }
      if (step.overrideEnabled && Object.keys(step.configOverride).length > 0) {
        stepOv.config = { ...step.configOverride }
      } else if (Object.keys(cfgDiff).length > 0) {
        stepOv.config = cfgDiff
      }

      if (Object.keys(stepOv).length > 0) {
        overrides[String(step.test_definition_id)] = stepOv
      }
    }

    const cleanedWebhook = webhookUrl.trim() || null
    const parsedEmails = emailTo
      .split(/[,;]/)
      .map((s) => s.trim())
      .filter(Boolean)
    const parsedThreshold = Number.isNaN(parseFloat(regressionThreshold))
      ? 0
      : Math.max(0, parseFloat(regressionThreshold))
    const hasChannel = Boolean(cleanedWebhook || parsedEmails.length > 0)

    return {
      name: jobName.trim(),
      description: jobDescription.trim() || null,
      schedule: jobSchedule.trim(),
      enabled: jobEnabled,
      run_mode: jobRunMode,
      missed_run_policy: jobMissedRunPolicy,
      definition_ids,
      overrides,
      notifications: {
        enabled: notifyEnabled || hasChannel,
        on_failure: notifyOnFailure,
        on_regression: notifyOnRegression,
        regression_threshold: parsedThreshold,
        webhook_enabled: Boolean(cleanedWebhook),
        webhook_url: cleanedWebhook,
        webhook_secret_env_var: webhookSecretEnvVar.trim() || null,
        email_enabled: parsedEmails.length > 0,
        email_to: parsedEmails,
      },
    }
  }

  const handleTestNotification = async () => {
    if (!editingJob) return
    setError(null)
    setSuccessMessage(null)
    try {
      setTestingNotification(true)
      const cleanedWebhook = webhookUrl.trim() || null
      const parsedEmails = emailTo
        .split(/[,;]/)
        .map((s) => s.trim())
        .filter(Boolean)
      const res = await api.jobs.testNotifications(editingJob.id, {
        webhook_url: cleanedWebhook,
        email_to: parsedEmails,
        notifications: {
          enabled: true,
          on_failure: notifyOnFailure,
          on_regression: notifyOnRegression,
          webhook_url: cleanedWebhook,
          webhook_secret_env_var: webhookSecretEnvVar.trim() || null,
          email_to: parsedEmails,
        },
      })
      if (res.ok || res.sent) {
        setSuccessMessage('Test notification delivered successfully.')
      } else {
        setError('Test notification delivery failed or no channel succeeded.')
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setTestingNotification(false)
    }
  }

  const cronValidation = validateCronSchedule(jobSchedule)

  const handleSaveJob = async (runNow = false) => {
    setError(null)
    setSuccessMessage(null)

    if (!jobName.trim()) {
      setError('Job name is required')
      return
    }
    if (!cronValidation.valid) {
      setError(`Invalid cron expression: ${cronValidation.error}`)
      return
    }
    if (steps.length === 0) {
      setError('Please add at least one test definition to the job')
      return
    }

    try {
      setSavingJob(true)
      const payload = buildJobPayload()
      let saved: Job
      if (editingJob) {
        saved = await api.jobs.update(editingJob.id, payload)
      } else {
        saved = await api.jobs.create(payload)
      }

      if (runNow) {
        const triggeredRun = await api.jobs.trigger(saved.id)
        navigate(`/runs/${triggeredRun.id}`)
        return
      }

      setSuccessMessage(
        editingJob
          ? `Job "${saved.name}" updated successfully.`
          : `Job "${saved.name}" created successfully.`
      )
      setIsEditorOpen(false)
      setEditingJob(null)
      await loadAll(false)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSavingJob(false)
    }
  }

  const handleRunNow = async (job: Job) => {
    setError(null)
    setSuccessMessage(null)
    try {
      setRunningJobId(job.id)
      const run = await api.jobs.trigger(job.id)
      navigate(`/runs/${run.id}`)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setRunningJobId(null)
    }
  }

  const handleToggleJobEnabled = async (job: Job) => {
    setError(null)
    setSuccessMessage(null)
    try {
      setTogglingJobId(job.id)
      const updated = job.enabled
        ? await api.jobs.disable(job.id)
        : await api.jobs.enable(job.id)
      setJobs((prev) => prev.map((j) => (j.id === updated.id ? updated : j)))
      setSuccessMessage(
        `Job "${updated.name}" ${updated.enabled ? 'enabled' : 'disabled'}.`
      )
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setTogglingJobId(null)
    }
  }

  const handleDeleteJob = async () => {
    if (!deletingJob) return
    try {
      await api.jobs.delete(deletingJob.id)
      setSuccessMessage(`Job "${deletingJob.name}" deleted.`)
      setDeletingJob(null)
      await loadAll(false)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const filteredJobs = jobs.filter((j) => {
    const matchesMode = modeFilter === 'all' || j.run_mode === modeFilter
    const matchesStatus =
      statusFilter === 'all' ||
      (statusFilter === 'enabled' && j.enabled) ||
      (statusFilter === 'disabled' && !j.enabled) ||
      (statusFilter === 'scheduled' && !isManualSchedule(j.schedule))
    const q = search.toLowerCase()
    const matchesSearch =
      !q ||
      j.name.toLowerCase().includes(q) ||
      (j.description && j.description.toLowerCase().includes(q))
    return matchesMode && matchesStatus && matchesSearch
  })

  const builderTesterInfo = testers.find((t) => t.name === newDefTester)

  if (loading) {
    return <Spinner message="Loading jobs and test definitions..." />
  }

  return (
    <div className="view-container jobs-view">
      <div className="view-header">
        <div>
          <h2>Test Jobs</h2>
          <p className="view-subtitle">
            Bundle ordered test definitions with cron scheduling, sequential or parallel execution, and schema-driven overrides.
          </p>
        </div>
        <div className="header-actions">
          <button className="btn btn-secondary" onClick={() => loadAll(true)}>
            ↻ Refresh
          </button>
          <button className="btn btn-primary" onClick={openEditorForNew}>
            <span className="btn-icon">+</span> New Job
          </button>
        </div>
      </div>

      {error && <div className="banner error-banner">{error}</div>}
      {successMessage && <div className="banner success-banner">{successMessage}</div>}

      {/* Job Editor Card */}
      {isEditorOpen && (
        <div className="card job-editor-card">
          <div className="job-editor-header">
            <h3>{editingJob ? `Edit Job: ${editingJob.name}` : 'Create New Job'}</h3>
            <button type="button" className="btn btn-sm btn-secondary" onClick={closeEditor}>
              ✕ Close Editor
            </button>
          </div>

          <div className="form-row">
            <div className="form-group flex-2">
              <label htmlFor="job-name-input">
                Job Name <span className="required-star">*</span>
              </label>
              <input
                id="job-name-input"
                type="text"
                className="form-input"
                value={jobName}
                onChange={(e) => setJobName(e.target.value)}
                placeholder="e.g. nightly-model-regression-suite"
                disabled={savingJob}
              />
            </div>

            <div className="form-group flex-1">
              <label htmlFor="job-run-mode-select">
                Run Mode <span className="required-star">*</span>
              </label>
              <select
                id="job-run-mode-select"
                className="form-select font-semibold"
                value={jobRunMode}
                onChange={(e) => setJobRunMode(e.target.value as JobRunMode)}
                disabled={savingJob}
              >
                <option value="sequential">Sequential (Run in order)</option>
                <option value="parallel">Parallel (Run concurrently)</option>
              </select>
            </div>

            <div className="form-group flex-1">
              <label htmlFor="job-schedule-input">Schedule (Optional)</label>
              <input
                id="job-schedule-input"
                type="text"
                className={`form-input font-mono ${!cronValidation.valid ? 'input-error' : ''}`}
                value={jobSchedule}
                onChange={(e) => setJobSchedule(e.target.value)}
                placeholder="e.g. 0 2 * * * or manual"
                disabled={savingJob}
              />
              <div className="cron-presets-row">
                {SCHEDULE_PRESETS.map((preset) => (
                  <button
                    key={preset.label}
                    type="button"
                    className={`cron-preset-chip ${
                      (jobSchedule.trim() || 'manual') === preset.value ? 'active' : ''
                    }`}
                    onClick={() => setJobSchedule(preset.value === 'manual' ? '' : preset.value)}
                    disabled={savingJob}
                    title={preset.desc}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
              {!cronValidation.valid ? (
                <div className="cron-validation-error" role="alert">
                  Invalid cron: {cronValidation.error}
                </div>
              ) : !cronValidation.isManual ? (
                <div className="cron-preview-box">
                  <strong>{cronValidation.description}</strong>
                  {' · '}
                  <span>
                    Next run:{' '}
                    {jobEnabled
                      ? formatUtcTimestamp(cronValidation.nextRunAt)
                      : 'Paused (Disabled)'}
                  </span>
                </div>
              ) : (
                <div className="cron-preview-box text-muted">Manual trigger only</div>
              )}
            </div>
          </div>

          <div className="form-row">
            <div className="form-group flex-2">
              <label htmlFor="job-description-input">Description</label>
              <input
                id="job-description-input"
                type="text"
                className="form-input"
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
                placeholder="Describe the purpose of this bundled test job..."
                disabled={savingJob}
              />
            </div>

            <div className="form-group flex-1">
              <label htmlFor="job-missed-policy-select">
                Missed-Run Policy (After Downtime)
              </label>
              <select
                id="job-missed-policy-select"
                className="form-select"
                value={jobMissedRunPolicy}
                onChange={(e) => setJobMissedRunPolicy(e.target.value as MissedRunPolicy)}
                disabled={savingJob}
              >
                <option value="skip">Skip missed runs (Wait for next future slot)</option>
                <option value="run_once">Run once immediately (Coalesce missed runs)</option>
              </select>
            </div>

            <div className="form-group flex-1 job-enabled-group">
              <label className="checkbox-label" htmlFor="job-enabled-checkbox">
                <input
                  id="job-enabled-checkbox"
                  type="checkbox"
                  checked={jobEnabled}
                  onChange={(e) => setJobEnabled(e.target.checked)}
                  disabled={savingJob}
                />
                <span className="field-label-text">Job Enabled</span>
              </label>
            </div>
          </div>

          {/* Per-Job Notification Settings */}
          <div className="job-notifications-section card" style={{ marginBottom: '16px', padding: '12px 16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <div>
                <h4 style={{ margin: 0 }}>Notifications (Failure & Regression)</h4>
                <p className="text-muted text-sm" style={{ margin: '2px 0 0 0' }}>
                  Trigger webhook or email notifications on failed runs or regressions vs the previous job run.
                </p>
              </div>
              {editingJob && (
                <button
                  type="button"
                  className="btn btn-sm btn-secondary"
                  onClick={handleTestNotification}
                  disabled={testingNotification || savingJob || (!webhookUrl.trim() && !emailTo.trim())}
                >
                  {testingNotification ? 'Sending...' : '🔔 Send Test Notification'}
                </button>
              )}
            </div>

            <div className="form-row" style={{ alignItems: 'center' }}>
              <div className="form-group flex-1">
                <label className="checkbox-label" htmlFor="job-notify-enabled-checkbox">
                  <input
                    id="job-notify-enabled-checkbox"
                    type="checkbox"
                    checked={notifyEnabled}
                    onChange={(e) => setNotifyEnabled(e.target.checked)}
                    disabled={savingJob}
                  />
                  <span className="field-label-text">Enable Notifications</span>
                </label>
              </div>

              <div className="form-group flex-1">
                <label className="checkbox-label" htmlFor="job-notify-failure-checkbox">
                  <input
                    id="job-notify-failure-checkbox"
                    type="checkbox"
                    checked={notifyOnFailure}
                    onChange={(e) => setNotifyOnFailure(e.target.checked)}
                    disabled={savingJob}
                  />
                  <span className="field-label-text">Trigger on Failed Run</span>
                </label>
              </div>

              <div className="form-group flex-1">
                <label className="checkbox-label" htmlFor="job-notify-regression-checkbox">
                  <input
                    id="job-notify-regression-checkbox"
                    type="checkbox"
                    checked={notifyOnRegression}
                    onChange={(e) => setNotifyOnRegression(e.target.checked)}
                    disabled={savingJob}
                  />
                  <span className="field-label-text">Trigger on Regression vs Previous Run</span>
                </label>
              </div>

              <div className="form-group flex-1">
                <label htmlFor="job-regression-threshold-input">
                  Regression Threshold (%)
                </label>
                <input
                  id="job-regression-threshold-input"
                  type="number"
                  min="0"
                  max="100"
                  step="0.5"
                  className="form-input font-mono"
                  value={regressionThreshold}
                  onChange={(e) => setRegressionThreshold(e.target.value)}
                  disabled={savingJob}
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group flex-2">
                <label htmlFor="job-webhook-url-input">Webhook URL</label>
                <input
                  id="job-webhook-url-input"
                  type="url"
                  className="form-input font-mono"
                  value={webhookUrl}
                  onChange={(e) => {
                    setWebhookUrl(e.target.value)
                    if (e.target.value.trim()) {
                      setNotifyEnabled(true)
                    }
                  }}
                  placeholder="https://hooks.example.com/test-center"
                  disabled={savingJob}
                />
              </div>

              <div className="form-group flex-1">
                <label htmlFor="job-webhook-secret-env-input">
                  Webhook Secret Env Var (Optional)
                </label>
                <input
                  id="job-webhook-secret-env-input"
                  type="text"
                  className="form-input font-mono"
                  value={webhookSecretEnvVar}
                  onChange={(e) => setWebhookSecretEnvVar(e.target.value)}
                  placeholder="e.g. WEBHOOK_SIGNING_SECRET"
                  disabled={savingJob}
                />
              </div>

              <div className="form-group flex-1">
                <label htmlFor="job-email-to-input">
                  Email Recipients (Optional)
                </label>
                <input
                  id="job-email-to-input"
                  type="text"
                  className="form-input"
                  value={emailTo}
                  onChange={(e) => {
                    setEmailTo(e.target.value)
                    if (e.target.value.trim()) {
                      setNotifyEnabled(true)
                    }
                  }}
                  placeholder="alerts@example.com, team@example.com"
                  disabled={savingJob}
                />
              </div>
            </div>
          </div>

          {/* Bamboo / CI Integration Section */}
          <div className="job-bamboo-card card" data-testid="job-bamboo-card">
            <div className="job-definitions-header">
              <div>
                <h4>🎋 Bamboo / CI Integration (Optional)</h4>
                <p className="text-muted">
                  Trigger this job from a Bamboo plan step using a token-protected endpoint, poll its result, and export cases as JUnit XML for Bamboo&apos;s JUnit Parser task.
                </p>
              </div>
              <button
                type="button"
                className="btn btn-sm btn-secondary"
                onClick={() => setShowBambooSnippet((prev) => !prev)}
                data-testid="toggle-bamboo-snippet-btn"
              >
                {showBambooSnippet ? 'Hide Bamboo Plan Snippet' : 'Show Bamboo Plan Snippet'}
              </button>
            </div>

            <div className="form-row" style={{ marginTop: '8px' }}>
              <div className="form-group flex-1">
                <label>Trigger Endpoint (POST)</label>
                <code className="font-mono text-sm">
                  /api/bamboo/jobs/{editingJob ? editingJob.id : encodeURIComponent(jobName.trim() || '<job_id>')}/trigger
                </code>
              </div>
              <div className="form-group flex-1">
                <label>Poll Status Endpoint (GET)</label>
                <code className="font-mono text-sm">/api/bamboo/runs/&lt;run_id&gt;</code>
              </div>
              <div className="form-group flex-1">
                <label>JUnit XML Export (GET)</label>
                <code className="font-mono text-sm">/api/bamboo/runs/&lt;run_id&gt;/junit.xml</code>
              </div>
            </div>

            {showBambooSnippet && (
              <div style={{ marginTop: '10px' }} data-testid="bamboo-plan-script-container">
                <p className="text-muted text-sm" style={{ marginBottom: '6px' }}>
                  Set <code>BAMBOO_API_TOKEN</code> on the Test Center server and pass it via{' '}
                  <code>Authorization: Bearer $&#123;bamboo.test_center_token_secret&#125;</code> or{' '}
                  <code>X-Bamboo-Token</code> header in a Bamboo Script task:
                </p>
                <pre className="json-block font-mono text-sm">
                  {`# 1. Trigger job and capture run_id
RESP=$(curl -fsSL -X POST "$TEST_CENTER_URL/api/bamboo/jobs/${editingJob ? editingJob.id : encodeURIComponent(jobName.trim() || '<job_id>')}/trigger" \\
  -H "Authorization: Bearer \${bamboo.test_center_token_secret}" \\
  -H "Content-Type: application/json" \\
  -d '{"plan_key":"\${bamboo.planKey}","build_number":"\${bamboo.buildNumber}","build_result_key":"\${bamboo.buildResultKey}"}')
RUN_ID=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['run_id'])")

# 2. Poll until finished and export JUnit XML for Bamboo JUnit Parser
curl -fsSL "$TEST_CENTER_URL/api/bamboo/runs/$RUN_ID?wait=true&timeout=300" \\
  -H "Authorization: Bearer \${bamboo.test_center_token_secret}"
curl -fsSL "$TEST_CENTER_URL/api/bamboo/runs/$RUN_ID/junit.xml" \\
  -H "Authorization: Bearer \${bamboo.test_center_token_secret}" \\
  -o "test-reports/test-center-run-$RUN_ID.xml"`}
                </pre>
              </div>
            )}
          </div>

          {/* Bundled Test Definitions Section */}
          <div className="job-definitions-section">
            <div className="job-definitions-header">
              <div>
                <h4>Bundled Test Definitions ({steps.length})</h4>
                <p className="text-muted">
                  Order test definitions and configure optional per-definition overrides using schema-generated forms.
                </p>
              </div>
              <div className="job-def-Picker-actions">
                {definitions.length > 0 && (
                  <div className="job-add-def-bar">
                    <select
                      id="add-definition-select"
                      aria-label="Select Test Definition"
                      className="form-select"
                      value={selectedDefToAdd}
                      onChange={(e) => setSelectedDefToAdd(e.target.value)}
                      disabled={savingJob}
                    >
                      {definitions.map((d) => (
                        <option key={d.id} value={d.id}>
                          #{d.id} {d.name} ({d.tester})
                        </option>
                      ))}
                    </select>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      onClick={() => handleAddStep()}
                      disabled={savingJob || !selectedDefToAdd}
                    >
                      + Add Definition
                    </button>
                  </div>
                )}
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => setShowDefBuilder((prev) => !prev)}
                  disabled={savingJob}
                >
                  {showDefBuilder ? 'Hide Definition Builder' : '+ Create New Definition'}
                </button>
              </div>
            </div>

            {/* Inline Quick Definition Builder (uses SchemaForm) */}
            {showDefBuilder && (
              <div className="inline-def-builder card">
                <h5>Create New Test Definition</h5>
                <div className="form-row">
                  <div className="form-group flex-1">
                    <label htmlFor="new-def-name">Definition Name *</label>
                    <input
                      id="new-def-name"
                      type="text"
                      className="form-input"
                      value={newDefName}
                      onChange={(e) => setNewDefName(e.target.value)}
                      placeholder="e.g. smoke-dummy-check"
                      disabled={creatingDef}
                    />
                  </div>
                  <div className="form-group flex-1">
                    <label htmlFor="new-def-tester">Tester Plugin *</label>
                    <select
                      id="new-def-tester"
                      className="form-select"
                      value={newDefTester}
                      onChange={(e) => handleNewDefTesterChange(e.target.value)}
                      disabled={creatingDef}
                    >
                      {testers.map((t) => (
                        <option key={t.name} value={t.name}>
                          {t.name.toUpperCase()} — {t.description}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="form-group flex-1">
                    <label htmlFor="new-def-target">Target Endpoint</label>
                    <select
                      id="new-def-target"
                      className="form-select"
                      value={newDefTargetId}
                      onChange={(e) => setNewDefTargetId(e.target.value)}
                      disabled={creatingDef}
                    >
                      <option value="">No Target (Default)</option>
                      {targets.map((tgt) => (
                        <option key={tgt.id} value={tgt.id}>
                          #{tgt.id} {tgt.name} ({tgt.base_url})
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {builderTesterInfo?.suites && builderTesterInfo.suites.length > 0 && (
                  <div className="form-group">
                    <label>Suites</label>
                    <div className="suites-selector-grid">
                      {builderTesterInfo.suites.map((suite) => {
                        const checked = newDefSuites.includes(suite)
                        return (
                          <label key={suite} className={`suite-checkbox-card ${checked ? 'active' : ''}`}>
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() =>
                                setNewDefSuites((prev) =>
                                  prev.includes(suite) ? prev.filter((s) => s !== suite) : [...prev, suite]
                                )
                              }
                              disabled={creatingDef}
                            />
                            <span className="suite-name">{suite}</span>
                          </label>
                        )
                      })}
                    </div>
                  </div>
                )}

                {builderTesterInfo?.config_schema && (
                  <div className="schema-section">
                    <SchemaForm
                      schema={builderTesterInfo.config_schema}
                      values={newDefConfig}
                      onChange={setNewDefConfig}
                      disabled={creatingDef}
                      context={{
                        targetId: newDefTargetId ? parseInt(newDefTargetId, 10) : null,
                      }}
                    />
                  </div>
                )}

                <div className="inline-def-actions">
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => setShowDefBuilder(false)}
                    disabled={creatingDef}
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    className="btn btn-primary btn-sm"
                    onClick={handleCreateInlineDefinition}
                    disabled={creatingDef}
                  >
                    {creatingDef ? 'Creating...' : 'Save Definition & Add to Job'}
                  </button>
                </div>
              </div>
            )}

            {/* Ordered Steps List */}
            {steps.length === 0 ? (
              <div className="empty-steps-notice">
                <p>No test definitions added to this job yet. Select or create a definition above.</p>
              </div>
            ) : (
              <div className="job-steps-list">
                {steps.map((step, idx) => {
                  const def = definitions.find((d) => d.id === step.test_definition_id)
                  const testerName = def?.tester || 'unknown'
                  const testerInfo = testers.find((t) => t.name === testerName)
                  const isExpanded = expandedStepIdx === idx
                  const activeSuites = step.suitesOverride ?? def?.suites ?? testerInfo?.suites ?? []

                  return (
                    <div key={`${step.test_definition_id}-${idx}`} className="job-step-item card">
                      <div className="job-step-header">
                        <div className="job-step-title">
                          <span className="step-order-badge">#{idx + 1}</span>
                          <strong>{def ? def.name : `Definition #${step.test_definition_id}`}</strong>
                          <span className="kind-badge">{testerName}</span>
                          {step.overrideEnabled && (
                            <span className="override-indicator-badge">Overrides Active</span>
                          )}
                        </div>

                        <div className="job-step-controls">
                          <button
                            type="button"
                            className="btn btn-sm btn-secondary"
                            aria-label={`Move step ${idx + 1} up`}
                            onClick={() => handleMoveStep(idx, -1)}
                            disabled={idx === 0 || savingJob}
                            title="Move Up"
                          >
                            ↑ Up
                          </button>
                          <button
                            type="button"
                            className="btn btn-sm btn-secondary"
                            aria-label={`Move step ${idx + 1} down`}
                            onClick={() => handleMoveStep(idx, 1)}
                            disabled={idx === steps.length - 1 || savingJob}
                            title="Move Down"
                          >
                            ↓ Down
                          </button>
                          <button
                            type="button"
                            className="btn btn-sm btn-secondary"
                            onClick={() => setExpandedStepIdx(isExpanded ? null : idx)}
                            disabled={savingJob}
                          >
                            {isExpanded ? 'Hide Overrides' : '⚙ Overrides & Schema Form'}
                          </button>
                          <button
                            type="button"
                            className="btn btn-sm btn-danger"
                            aria-label={`Remove step ${idx + 1}`}
                            onClick={() => handleRemoveStep(idx)}
                            disabled={savingJob}
                          >
                            Remove
                          </button>
                        </div>
                      </div>

                      {isExpanded && (
                        <div className="job-step-overrides-body">
                          <div className="form-row">
                            <div className="form-group flex-1">
                              <label htmlFor={`step-target-override-${idx}`}>
                                Target Override (Default: {def?.target?.name || 'None'})
                              </label>
                              <select
                                id={`step-target-override-${idx}`}
                                className="form-select"
                                value={step.targetIdOverride}
                                onChange={(e) =>
                                  updateStepAt(idx, (s) => ({
                                    ...s,
                                    overrideEnabled: true,
                                    targetIdOverride: e.target.value,
                                  }))
                                }
                                disabled={savingJob}
                              >
                                <option value="">Use Definition Default</option>
                                {targets.map((tgt) => (
                                  <option key={tgt.id} value={tgt.id}>
                                    #{tgt.id} {tgt.name} ({tgt.base_url})
                                  </option>
                                ))}
                              </select>
                            </div>

                            <div className="form-group flex-1">
                              <label htmlFor={`step-timeout-override-${idx}`}>
                                Step Timeout Override (Seconds)
                              </label>
                              <input
                                id={`step-timeout-override-${idx}`}
                                type="number"
                                className="form-input"
                                value={step.timeoutOverride}
                                onChange={(e) =>
                                  updateStepAt(idx, (s) => ({
                                    ...s,
                                    overrideEnabled: true,
                                    timeoutOverride: e.target.value,
                                  }))
                                }
                                placeholder="Default timeout"
                                min={1}
                                disabled={savingJob}
                              />
                            </div>
                          </div>

                          {testerInfo?.suites && testerInfo.suites.length > 0 && (
                            <div className="form-group">
                              <label>Suites Override</label>
                              <div className="suites-selector-grid">
                                {testerInfo.suites.map((suite) => {
                                  const checked = activeSuites.includes(suite)
                                  return (
                                    <label
                                      key={suite}
                                      className={`suite-checkbox-card ${checked ? 'active' : ''}`}
                                    >
                                      <input
                                        type="checkbox"
                                        checked={checked}
                                        onChange={() => {
                                          const nextSuites = checked
                                            ? activeSuites.filter((s) => s !== suite)
                                            : [...activeSuites, suite]
                                          updateStepAt(idx, (s) => ({
                                            ...s,
                                            overrideEnabled: true,
                                            suitesOverride: nextSuites,
                                          }))
                                        }}
                                        disabled={savingJob}
                                      />
                                      <span className="suite-name">{suite}</span>
                                    </label>
                                  )
                                })}
                              </div>
                            </div>
                          )}

                          {testerInfo?.config_schema && (
                            <div className="schema-section">
                              <SchemaForm
                                schema={testerInfo.config_schema}
                                values={step.configOverride}
                                onChange={(newVals) =>
                                  updateStepAt(idx, (s) => ({
                                    ...s,
                                    overrideEnabled: true,
                                    configOverride: newVals,
                                  }))
                                }
                                disabled={savingJob}
                                context={{
                                  targetId: step.targetIdOverride
                                    ? parseInt(step.targetIdOverride, 10)
                                    : def?.target_id ?? null,
                                }}
                              />
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            )}
          </div>

          <div className="form-submit-bar">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={closeEditor}
              disabled={savingJob}
            >
              Cancel
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => handleSaveJob(false)}
              disabled={savingJob}
            >
              {savingJob ? 'Saving...' : 'Save Job'}
            </button>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => handleSaveJob(true)}
              disabled={savingJob}
            >
              {savingJob ? 'Starting...' : '🚀 Save & Run Now'}
            </button>
          </div>
        </div>
      )}

      {/* Filter Bar */}
      <div className="filter-bar">
        <div className="search-input-wrapper">
          <span className="search-icon">🔍</span>
          <input
            type="text"
            className="search-input"
            placeholder="Search jobs by name or description..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <div className="filter-group">
          <label htmlFor="jobs-status-filter" className="filter-label">
            Status:
          </label>
          <select
            id="jobs-status-filter"
            className="filter-select"
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
          >
            <option value="all">All Jobs</option>
            <option value="enabled">Enabled</option>
            <option value="disabled">Disabled</option>
            <option value="scheduled">Cron Scheduled</option>
          </select>
        </div>

        <div className="filter-group">
          <label htmlFor="jobs-mode-filter" className="filter-label">
            Run Mode:
          </label>
          <select
            id="jobs-mode-filter"
            className="filter-select"
            value={modeFilter}
            onChange={(e) => setModeFilter(e.target.value)}
          >
            <option value="all">All Modes ({jobs.length})</option>
            <option value="sequential">Sequential</option>
            <option value="parallel">Parallel</option>
          </select>
        </div>
      </div>

      {/* Jobs Table */}
      {filteredJobs.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">📦</div>
          <h3>No jobs found</h3>
          <p>
            {jobs.length === 0
              ? 'No test jobs have been created yet. Create a job to bundle test definitions and run them together.'
              : 'No jobs match your search criteria.'}
          </p>
          <button className="btn btn-primary" onClick={openEditorForNew}>
            Create First Job
          </button>
        </div>
      ) : (
        <div className="table-responsive">
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Job Name</th>
                <th>Run Mode</th>
                <th>Ordered Test Definitions</th>
                <th>Schedule & Policy</th>
                <th>Next Run</th>
                <th>Latest Run</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredJobs.map((job) => {
                const jobRuns = runs.filter((r) => r.job_id === job.id)
                const latestRun = jobRuns.length > 0 ? jobRuns[0] : null
                const isManual = isManualSchedule(job.schedule)
                const schedValidation = !isManual ? validateCronSchedule(job.schedule) : null
                const schedDesc =
                  job.schedule_description || schedValidation?.description || 'Manual trigger only'
                const effectiveNextRun =
                  job.next_run_at !== undefined
                    ? job.next_run_at
                    : job.enabled && schedValidation?.valid
                      ? schedValidation.nextRunAt
                      : null
                const policy = job.missed_run_policy || 'skip'

                return (
                  <tr key={job.id}>
                    <td className="font-mono text-muted">#{job.id}</td>
                    <td>
                      <div>
                        <strong>{job.name}</strong>
                        {!job.enabled && (
                          <span className="status-badge badge-neutral badge-sm" style={{ marginLeft: '6px' }}>
                            DISABLED
                          </span>
                        )}
                      </div>
                      {job.description && <div className="text-muted text-sm">{job.description}</div>}
                      {job.notifications?.enabled &&
                        (job.notifications.webhook_url ||
                          (job.notifications.email_to && job.notifications.email_to.length > 0)) && (
                          <div className="schedule-meta-row" style={{ marginTop: '4px' }}>
                            {job.notifications.webhook_url && (
                              <span
                                className="missed-policy-badge"
                                title={`Webhook: ${job.notifications.webhook_url}`}
                              >
                                🔔 Webhook
                              </span>
                            )}
                            {job.notifications.email_to && job.notifications.email_to.length > 0 && (
                              <span
                                className="missed-policy-badge"
                                title={`Email: ${job.notifications.email_to.join(', ')}`}
                              >
                                ✉ Email ({job.notifications.email_to.length})
                              </span>
                            )}
                          </div>
                        )}
                    </td>
                    <td>
                      <span className={`run-mode-badge mode-${job.run_mode}`}>
                        {job.run_mode.toUpperCase()}
                      </span>
                    </td>
                    <td>
                      {job.definitions && job.definitions.length > 0 ? (
                        <div className="job-defs-pill-list">
                          {job.definitions.map((d, i) => (
                            <span key={`${d.id}-${i}`} className="job-def-pill">
                              <span className="pill-order">{i + 1}.</span> {d.name}{' '}
                              <span className="text-muted font-mono">({d.tester})</span>
                            </span>
                          ))}
                        </div>
                      ) : (
                        <span className="text-muted">
                          {job.definition_ids?.length || 0} definitions
                        </span>
                      )}
                    </td>
                    <td>
                      <div className="schedule-cell">
                        <span className="font-mono">{isManual ? 'Manual' : job.schedule}</span>
                        {!isManual && (
                          <div className="schedule-meta-row">
                            <span className="text-muted text-sm">{schedDesc}</span>
                            <span
                              className="missed-policy-badge"
                              title={
                                policy === 'run_once'
                                  ? 'Coalesce missed runs after downtime into 1 run'
                                  : 'Skip missed runs after downtime'
                              }
                            >
                              Missed: {policy}
                            </span>
                          </div>
                        )}
                      </div>
                    </td>
                    <td>
                      {isManual ? (
                        <span className="text-muted text-sm">Manual only</span>
                      ) : !job.enabled ? (
                        <span className="text-muted text-sm">Paused (Disabled)</span>
                      ) : effectiveNextRun ? (
                        <span className="font-mono text-sm next-run-timestamp">
                          {formatUtcTimestamp(effectiveNextRun)}
                        </span>
                      ) : (
                        <span className="text-muted text-sm">—</span>
                      )}
                    </td>
                    <td>
                      {latestRun ? (
                        <div className="result-pill-group">
                          <Link href={`/runs/${latestRun.id}`} className="font-mono font-bold table-link">
                            #{latestRun.id}
                          </Link>
                          <Badge status={latestRun.status} size="sm" />
                          {latestRun.results && latestRun.results.length > 1 && (
                            <span className="cases-badge text-muted">
                              {latestRun.results.length} testers
                            </span>
                          )}
                        </div>
                      ) : (
                        <span className="text-muted">No runs yet</span>
                      )}
                    </td>
                    <td>
                      <div className="row-actions">
                        <button
                          type="button"
                          className="btn btn-sm btn-primary"
                          onClick={() => handleRunNow(job)}
                          disabled={runningJobId === job.id}
                        >
                          {runningJobId === job.id ? 'Starting...' : '🚀 Run Now'}
                        </button>
                        <button
                          type="button"
                          className="btn btn-sm btn-secondary"
                          onClick={() => handleToggleJobEnabled(job)}
                          disabled={togglingJobId === job.id}
                          aria-label={job.enabled ? `Disable job ${job.name}` : `Enable job ${job.name}`}
                        >
                          {togglingJobId === job.id
                            ? '...'
                            : job.enabled
                              ? 'Disable'
                              : 'Enable'}
                        </button>
                        <button
                          type="button"
                          className="btn btn-sm btn-secondary"
                          onClick={() => openEditorForJob(job)}
                        >
                          Edit
                        </button>
                        <button
                          type="button"
                          className="action-btn danger"
                          onClick={() => setDeletingJob(job)}
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

      <ConfirmDialog
        isOpen={Boolean(deletingJob)}
        title="Delete Job"
        message={`Are you sure you want to delete job "${deletingJob?.name}"? Past runs triggered by this job will be preserved.`}
        confirmLabel="Delete Job"
        danger
        onConfirm={handleDeleteJob}
        onCancel={() => setDeletingJob(null)}
      />
    </div>
  )
}
