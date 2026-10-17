import type {
  Artifact,
  CaseResult,
  CompareParams,
  CompareResponse,
  CreateRunRequest,
  HealthResponse,
  Job,
  JobCreate,
  JobUpdate,
  Run,
  RunLogsResponse,
  Target,
  TargetCreate,
  TargetUpdate,
  TestDefinition,
  TestDefinitionCreate,
  TestDefinitionUpdate,
  TesterInfo,
  TriggerDefinitionRunRequest,
  TriggerJobRunRequest,
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
    list: (params?: { enabled?: boolean; run_mode?: string; limit?: number; offset?: number }) => {
      const query = new URLSearchParams()
      if (params?.enabled !== undefined) query.set('enabled', String(params.enabled))
      if (params?.run_mode) query.set('run_mode', params.run_mode)
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
    listRuns: (id: number, params?: { limit?: number; offset?: number }) => {
      const query = new URLSearchParams()
      if (params?.limit) query.set('limit', String(params.limit))
      if (params?.offset) query.set('offset', String(params.offset))
      const qs = query.toString()
      return request<Run[]>(`/jobs/${id}/runs${qs ? `?${qs}` : ''}`)
    },
  },

  runs: {
    list: (params?: { status?: string; job_id?: number; limit?: number; offset?: number }) => {
      const query = new URLSearchParams()
      if (params?.status) query.set('status', params.status)
      if (params?.job_id !== undefined) query.set('job_id', String(params.job_id))
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
    cancel: (id: number) =>
      request<{ status: string; message: string }>(`/runs/${id}/cancel`, {
        method: 'POST',
      }),
    getLogs: (id: number) => request<RunLogsResponse>(`/runs/${id}/logs`),
    getLogStreamUrl: (id: number) => `/api/runs/${id}/logs/stream`,
    getArtifactDownloadUrl: (runId: number, artifactId: number) =>
      `/api/runs/${runId}/artifacts/${artifactId}/download`,
    getArtifactContent: async (runId: number, artifactId: number): Promise<string> => {
      const response = await fetch(`/api/runs/${runId}/artifacts/${artifactId}/download`)
      if (!response.ok) {
        throw new ApiError(response.status, `Failed to load artifact #${artifactId}`)
      }
      return response.text()
    },
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
  },

  compare: {
    get: (params: CompareParams) => {
      const query = new URLSearchParams()
      if (params.run_ids && params.run_ids.length > 0) {
        query.set('run_ids', params.run_ids.join(','))
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
