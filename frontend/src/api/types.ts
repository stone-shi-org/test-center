export type RunStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'error'
export type ResultStatus = 'pass' | 'fail' | 'error' | 'cancelled'
export type CaseStatus = 'pass' | 'fail' | 'error' | 'skip'

export interface Target {
  id: number
  name: string
  kind: string
  base_url: string
  auth_env_var: string | null
  extra: Record<string, unknown>
  created_at: string
  updated_at: string
}

export interface TargetCreate {
  name: string
  kind: string
  base_url: string
  auth_env_var?: string | null
  extra?: Record<string, unknown>
}

export interface TargetUpdate {
  name?: string
  kind?: string
  base_url?: string
  auth_env_var?: string | null
  extra?: Record<string, unknown>
}

export interface JSONSchemaProperty {
  type?: 'string' | 'number' | 'integer' | 'boolean' | 'array' | 'object'
  description?: string
  default?: unknown
  minimum?: number
  maximum?: number
  enum?: string[]
  items?: {
    type?: string
    enum?: string[]
  }
  properties?: Record<string, JSONSchemaProperty>
  required?: string[]
}

export interface JSONSchemaObject {
  type: 'object'
  properties: Record<string, JSONSchemaProperty>
  required?: string[]
  description?: string
}

export interface TesterInfo {
  name: string
  description: string
  suites: string[]
  config_schema: JSONSchemaObject
}

export interface TestDefinition {
  id: number
  name: string
  tester: string
  suites: string[]
  config: Record<string, unknown>
  target_id: number | null
  created_at: string
  updated_at: string
  target?: Target | null
}

export interface TestDefinitionCreate {
  name: string
  tester: string
  suites?: string[]
  config?: Record<string, unknown>
  target_id?: number | null
}

export interface TestDefinitionUpdate {
  name?: string
  tester?: string
  suites?: string[]
  config?: Record<string, unknown>
  target_id?: number | null
}

export interface TriggerDefinitionRunRequest {
  suites?: string[]
  config?: Record<string, unknown>
  target_id?: number | null
  timeout?: number
}

export interface CreateRunRequest {
  test_definition_id?: number | null
  tester?: string | null
  suites?: string[] | null
  config?: Record<string, unknown>
  target_id?: number | null
  target?: Record<string, unknown> | null
  job_id?: number | null
  timeout?: number | null
}

export interface CaseResult {
  id: number
  suite: string
  case_id: string
  status: CaseStatus | string
  metrics: Record<string, unknown>
  detail: Record<string, unknown>
}

export interface Artifact {
  id: number
  name: string
  path: string
  mime: string
  size_bytes: number | null
  created_at: string | null
}

export interface RunResult {
  id: number
  tester: string
  status: ResultStatus | string
  metrics: Record<string, unknown>
  summary: string | null
  cases: CaseResult[]
  artifacts: Artifact[]
}

export interface Run {
  id: number
  job_id: number | null
  status: RunStatus
  queued_at: string | null
  started_at: string | null
  finished_at: string | null
  config_snapshot: {
    tester?: string
    suites?: string[]
    config?: Record<string, unknown>
    test_definition_id?: number
    timeout?: number
    [key: string]: unknown
  }
  target_snapshot: {
    id?: number
    name?: string
    kind?: string
    base_url?: string
    auth_env_var?: string | null
    extra?: Record<string, unknown>
    [key: string]: unknown
  }
  git_info: Record<string, unknown> | null
  results: RunResult[]
}

export interface RunLogsResponse {
  run_id: number
  status: string
  lines: string[]
  raw: string
}

export interface HealthResponse {
  status: string
  app?: string
  version?: string
  [key: string]: unknown
}
