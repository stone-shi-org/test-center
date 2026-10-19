import type {
  AdHocTargetModelsRequest,
  AdHocTargetTestRequest,
  Artifact,
  ArtifactContentResponse,
  BambooIntegrationInfo,
  Baseline,
  BaselinesFilterParams,
  CaseResult,
  CompareParams,
  CompareResponse,
  CreateBaselineRequest,
  CreateRunRequest,
  CronValidationResult,
  HealthResponse,
  Job,
  JobBambooInfo,
  JobCreate,
  JobNotificationSettings,
  JobUpdate,
  RetentionCleanupRequest,
  RetentionCleanupResult,
  RetentionSettings,
  RetentionUpdateRequest,
  Run,
  RunLogsResponse,
  RunsFilterParams,
  StarterTemplate,
  Target,
  TargetCreate,
  TargetModelInfo,
  TargetTestRequest,
  TargetTestResponse,
  TargetUpdate,
  TestDefinition,
  TestDefinitionCreate,
  TestDefinitionUpdate,
  TesterInfo,
  TestJobNotificationResponse,
  TrendQueryParams,
  TrendResponse,
  TriggerDefinitionRunRequest,
  TriggerJobRunRequest,
  UpdateBaselineRequest,
} from './types.ts'

export class ApiError extends Error {
  status: number
  detail: unknown

  constructor(status: number, message: string, detail?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const url = path.startsWith('/api') ? path : `/api${path.startsWith('/') ? '' : '/'}${path}`
  const headers = new Headers(options.headers || {})
  if (!headers.has('Content-Type') && options.body && typeof options.body === 'string') {
    headers.set('Content-Type', 'application/json')
  }

  const response = await fetch(url, {
    ...options,
    headers,
  })

  if (response.status === 204) {
    return undefined as unknown as T
  }

  if (!response.ok) {
    let errDetail: unknown = null
    let errMsg = `Request failed with status ${response.status}`
    try {
      const data = await response.json()
      errDetail = data
      if (typeof data.detail === 'string') {
        errMsg = data.detail
      } else if (Array.isArray(data.detail)) {
        errMsg = data.detail.map((d: { msg?: string }) => d.msg || JSON.stringify(d)).join(', ')
      } else if (data.message) {
        errMsg = data.message
      }
    } catch {
      // Use status text if body is not JSON
      errMsg = response.statusText || errMsg
    }
    throw new ApiError(response.status, errMsg, errDetail)
  }

  return response.json() as Promise<T>
}

export const api = {
  health: {
    get: () => request<HealthResponse>('/health'),
  },

  targets: {
    list: (params?: { kind?: string; limit?: number; offset?: number }) => {
      const query = new URLSearchParams()
      if (params?.kind) query.set('kind', params.kind)
      if (params?.limit) query.set('limit', String(params.limit))
      if (params?.offset) query.set('offset', String(params.offset))
      const qs = query.toString()
      return request<Target[]>(`/targets${qs ? `?${qs}` : ''}`)
    },
    get: (id: number) => request<Target>(`/targets/${id}`),
    create: (data: TargetCreate) =>
      request<Target>('/targets', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    update: (id: number, data: TargetUpdate) =>
      request<Target>(`/targets/${id}`, {
        method: 'PUT',
        body: JSON.stringify(data),
      }),
    delete: (id: number) =>
      request<void>(`/targets/${id}`, {
        method: 'DELETE',
      }),
    test: (id: number, data?: TargetTestRequest) =>
      request<TargetTestResponse>(`/targets/${id}/test`, {
        method: 'POST',
        body: JSON.stringify(data || {}),
      }),
    testAdHoc: (data: AdHocTargetTestRequest) =>
      request<TargetTestResponse>('/targets/test', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    listModels: (id: number) => request<TargetModelInfo[]>(`/targets/${id}/models`),
    listModelsAdHoc: (data: AdHocTargetModelsRequest) =>
      request<TargetModelInfo[]>('/targets/models', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
  },

  testers: {
    list: () => request<TesterInfo[]>('/testers'),
    get: (name: string) => request<TesterInfo>(`/testers/${encodeURIComponent(name)}`),
  },

  definitions: {
    list: (params?: { tester?: string; target_id?: number; limit?: number; offset?: number }) => {
      const query = new URLSearchParams()
      if (params?.tester) query.set('tester', params.tester)
      if (params?.target_id !== undefined) query.set('target_id', String(params.target_id))
      if (params?.limit) query.set('limit', String(params.limit))
      if (params?.offset) query.set('offset', String(params.offset))
      const qs = query.toString()
      return request<TestDefinition[]>(`/test-definitions${qs ? `?${qs}` : ''}`)
    },
    listStarters: (tester?: string) => {
      const query = new URLSearchParams()
      if (tester) query.set('tester', tester)
      const qs = query.toString()
      return request<StarterTemplate[]>(`/test-definitions/starters${qs ? `?${qs}` : ''}`)
    },
    get: (id: number) => request<TestDefinition>(`/test-definitions/${id}`),
    create: (data: TestDefinitionCreate) =>
      request<TestDefinition>('/test-definitions', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    update: (id: number, data: TestDefinitionUpdate) =>
      request<TestDefinition>(`/test-definitions/${id}`, {
        method: 'PUT',
        body: JSON.stringify(data),
      }),
    delete: (id: number) =>
      request<void>(`/test-definitions/${id}`, {
        method: 'DELETE',
      }),
    trigger: (id: number, data?: TriggerDefinitionRunRequest) =>
      request<Run>(`/test-definitions/${id}/run`, {
        method: 'POST',
        body: data ? JSON.stringify(data) : undefined,
      }),
  },

  jobs: {
    list: (params?: {
      enabled?: boolean
      run_mode?: string
      scheduled?: boolean
      limit?: number
      offset?: number
    }) => {
      const query = new URLSearchParams()
      if (params?.enabled !== undefined) query.set('enabled', String(params.enabled))
      if (params?.run_mode) query.set('run_mode', params.run_mode)
      if (params?.scheduled !== undefined) query.set('scheduled', String(params.scheduled))
      if (params?.limit) query.set('limit', String(params.limit))
      if (params?.offset) query.set('offset', String(params.offset))
      const qs = query.toString()
      return request<Job[]>(`/jobs${qs ? `?${qs}` : ''}`)
    },
    get: (id: number) => request<Job>(`/jobs/${id}`),
    create: (data: JobCreate) =>
      request<Job>('/jobs', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    update: (id: number, data: JobUpdate) =>
      request<Job>(`/jobs/${id}`, {
        method: 'PUT',
        body: JSON.stringify(data),
      }),
    enable: (id: number) =>
      request<Job>(`/jobs/${id}/enable`, {
        method: 'POST',
      }),
    disable: (id: number) =>
      request<Job>(`/jobs/${id}/disable`, {
        method: 'POST',
      }),
    toggle: (id: number) =>
      request<Job>(`/jobs/${id}/toggle`, {
        method: 'POST',
      }),
    validateCron: (schedule: string, count = 5) =>
      request<CronValidationResult>('/jobs/validate-cron', {
        method: 'POST',
        body: JSON.stringify({ schedule, count }),
      }),
    reorder: (id: number, definition_ids: number[]) =>
      request<Job>(`/jobs/${id}/order`, {
        method: 'PUT',
        body: JSON.stringify({ definition_ids }),
      }),
    delete: (id: number) =>
      request<void>(`/jobs/${id}`, {
        method: 'DELETE',
      }),
    trigger: (id: number, data?: TriggerJobRunRequest) =>
      request<Run>(`/jobs/${id}/run`, {
        method: 'POST',
        body: data ? JSON.stringify(data) : undefined,
      }),
    getNotifications: (id: number) =>
      request<JobNotificationSettings>(`/jobs/${id}/notifications`),
    updateNotifications: (id: number, data: Partial<JobNotificationSettings>) =>
      request<JobNotificationSettings>(`/jobs/${id}/notifications`, {
        method: 'PUT',
        body: JSON.stringify(data),
      }),
    testNotifications: (
      id: number,
      data?: {
        run_id?: number
        webhook_url?: string | null
        email_to?: string[] | string
        notifications?: Partial<JobNotificationSettings>
      } & Partial<JobNotificationSettings>
    ) =>
      request<TestJobNotificationResponse>(`/jobs/${id}/notifications/test`, {
        method: 'POST',
        body: data ? JSON.stringify(data) : undefined,
      }),
    listRuns: (id: number, params?: { limit?: number; offset?: number }) => {
      const query = new URLSearchParams()
      if (params?.limit) query.set('limit', String(params.limit))
      if (params?.offset) query.set('offset', String(params.offset))
      const qs = query.toString()
      return request<Run[]>(`/jobs/${id}/runs${qs ? `?${qs}` : ''}`)
    },
    getBambooInfo: (id: number, baseUrl?: string) => {
      const qs = baseUrl ? `?base_url=${encodeURIComponent(baseUrl)}` : ''
      return request<JobBambooInfo>(`/jobs/${id}/bamboo${qs}`)
    },
  },

  runs: {
    list: (params?: RunsFilterParams) => {
      const query = new URLSearchParams()
      if (params?.status) query.set('status', params.status)
      if (params?.tester) query.set('tester', params.tester)
      if (params?.target_id !== undefined) query.set('target_id', String(params.target_id))
      if (params?.target) query.set('target', params.target)
      if (params?.model) query.set('model', params.model)
      if (params?.job_id !== undefined) query.set('job_id', String(params.job_id))
      if (params?.job) query.set('job', params.job)
      if (params?.baseline !== undefined) query.set('baseline', String(params.baseline))
      if (params?.date_from) query.set('date_from', params.date_from)
      if (params?.date_to) query.set('date_to', params.date_to)
      if (params?.date) query.set('date', params.date)
      if (params?.limit) query.set('limit', String(params.limit))
      if (params?.offset) query.set('offset', String(params.offset))
      const qs = query.toString()
      return request<Run[]>(`/runs${qs ? `?${qs}` : ''}`)
    },
    get: (id: number) => request<Run>(`/runs/${id}`),
    create: (data: CreateRunRequest) =>
      request<Run>('/runs', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    saveBaseline: (id: number, data: CreateBaselineRequest) =>
      request<Baseline>(`/runs/${id}/baseline`, {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    delete: (id: number) =>
      request<void>(`/runs/${id}`, {
        method: 'DELETE',
      }),
    cancel: (id: number) =>
      request<{ status: string; message: string }>(`/runs/${id}/cancel`, {
        method: 'POST',
      }),
    getTrends: (params?: TrendQueryParams) => {
      const query = new URLSearchParams()
      if (params?.metric) query.set('metric', params.metric)
      if (params?.tester) query.set('tester', params.tester)
      if (params?.target_id !== undefined) query.set('target_id', String(params.target_id))
      if (params?.target) query.set('target', params.target)
      if (params?.model) query.set('model', params.model)
      if (params?.job_id !== undefined) query.set('job_id', String(params.job_id))
      if (params?.job) query.set('job', params.job)
      if (params?.status) query.set('status', params.status)
      if (params?.date_from) query.set('date_from', params.date_from)
      if (params?.date_to) query.set('date_to', params.date_to)
      if (params?.date) query.set('date', params.date)
      if (params?.limit) query.set('limit', String(params.limit))
      const qs = query.toString()
      return request<TrendResponse>(`/runs/trends${qs ? `?${qs}` : ''}`)
    },
    getRetention: () => request<RetentionSettings>('/runs/retention'),
    updateRetention: (data: RetentionUpdateRequest) =>
      request<RetentionSettings>('/runs/retention', {
        method: 'PUT',
        body: JSON.stringify(data),
      }),
    cleanupRetention: (data?: RetentionCleanupRequest) =>
      request<RetentionCleanupResult>('/runs/retention/cleanup', {
        method: 'POST',
        body: data ? JSON.stringify(data) : undefined,
      }),
    getLogs: (id: number) => request<RunLogsResponse>(`/runs/${id}/logs`),
    getLogStreamUrl: (id: number) => `/api/runs/${id}/logs/stream`,
    getJunitXmlUrl: (runId: number) => `/api/runs/${runId}/junit.xml`,
    getJunitXml: async (runId: number): Promise<string> => {
      const response = await fetch(`/api/runs/${runId}/junit.xml`)
      if (!response.ok) {
        throw new ApiError(response.status, `Failed to export JUnit XML for Run #${runId}`)
      }
      return response.text()
    },
    listArtifacts: (runId: number, kind?: string) => {
      const qs = kind ? `?kind=${encodeURIComponent(kind)}` : ''
      return request<Artifact[]>(`/runs/${runId}/artifacts${qs}`)
    },
    getArtifactPreviewUrl: (runId: number, artifactId: number | string) =>
      `/api/runs/${runId}/artifacts/${artifactId}/preview`,
    getArtifactDownloadUrl: (runId: number, artifactId: number | string) =>
      `/api/runs/${runId}/artifacts/${artifactId}/download`,
    getArtifactContent: async (runId: number, artifactId: number | string): Promise<string> => {
      const response = await fetch(`/api/runs/${runId}/artifacts/${artifactId}/download`)
      if (!response.ok) {
        throw new ApiError(response.status, `Failed to load artifact #${artifactId}`)
      }
      return response.text()
    },
    getArtifactStructuredContent: (
      runId: number,
      artifactId: number | string,
      params?: { search?: string; level?: string; tail?: number }
    ) => {
      const query = new URLSearchParams()
      if (params?.search) query.set('search', params.search)
      if (params?.level) query.set('level', params.level)
      if (params?.tail !== undefined) query.set('tail', String(params.tail))
      const qs = query.toString()
      return request<ArtifactContentResponse>(
        `/runs/${runId}/artifacts/${artifactId}/content${qs ? `?${qs}` : ''}`
      )
    },
  },

  baselines: {
    list: (params?: BaselinesFilterParams) => {
      const query = new URLSearchParams()
      if (params?.q) query.set('q', params.q)
      if (params?.tester) query.set('tester', params.tester)
      if (params?.target) query.set('target', params.target)
      if (params?.model) query.set('model', params.model)
      const qs = query.toString()
      return request<Baseline[]>(`/baselines${qs ? `?${qs}` : ''}`)
    },
    get: (id: number) => request<Baseline>(`/baselines/${id}`),
    update: (id: number, data: UpdateBaselineRequest) =>
      request<Baseline>(`/baselines/${id}`, {
        method: 'PATCH',
        body: JSON.stringify(data),
      }),
    delete: (id: number) =>
      request<void>(`/baselines/${id}`, {
        method: 'DELETE',
      }),
  },

  results: {
    getCases: (resultId: number, params?: { suite?: string; status?: string }) => {
      const query = new URLSearchParams()
      if (params?.suite) query.set('suite', params.suite)
      if (params?.status) query.set('status', params.status)
      const qs = query.toString()
      return request<CaseResult[]>(`/results/${resultId}/cases${qs ? `?${qs}` : ''}`)
    },
    getArtifacts: (resultId: number) => request<Artifact[]>(`/results/${resultId}/artifacts`),
    getJunitXmlUrl: (resultId: number) => `/api/results/${resultId}/junit.xml`,
  },

  bamboo: {
    getInfo: () => request<BambooIntegrationInfo>('/bamboo/info'),
  },

  compare: {
    get: (params: CompareParams) => {
      const query = new URLSearchParams()
      if (params.run_ids && params.run_ids.length > 0) {
        query.set('run_ids', params.run_ids.join(','))
      }
      if (params.baselines && params.baselines.length > 0) {
        query.set('baselines', params.baselines.join(','))
      }
      if (params.result_ids && params.result_ids.length > 0) {
        query.set('result_ids', params.result_ids.join(','))
      }
      if (params.baseline_run_id !== undefined && params.baseline_run_id !== null) {
        query.set('baseline_run_id', String(params.baseline_run_id))
      }
      if (params.baseline_result_id !== undefined && params.baseline_result_id !== null) {
        query.set('baseline_result_id', String(params.baseline_result_id))
      }
      if (params.threshold !== undefined && params.threshold > 0) {
        query.set('threshold', String(params.threshold))
      }
      if (params.tester) query.set('tester', params.tester)
      if (params.suite) query.set('suite', params.suite)
      if (params.changed_only) query.set('changed_only', 'true')
      if (params.regressed_only) query.set('regressed_only', 'true')
      if (params.allow_different_testers) query.set('allow_different_testers', 'true')
      const qs = query.toString()
      return request<CompareResponse>(`/compare${qs ? `?${qs}` : ''}`)
    },
    post: (data: CompareParams) =>
      request<CompareResponse>('/compare', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
  },
}
