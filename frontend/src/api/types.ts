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

export type JobRunMode = 'sequential' | 'parallel'
export type MissedRunPolicy = 'skip' | 'run_once'

export interface CronValidationResult {
  valid: boolean
  schedule: string
  normalized: string | null
  is_manual: boolean
  description: string | null
  next_run_at: string | null
  upcoming_runs: string[]
  error: string | null
}

export interface JobStepOverride {
  suites?: string[]
  config?: Record<string, unknown>
  target_id?: number | null
  timeout?: number | null
}

export interface JobDefinitionItem {
  order: number
  id: number
  test_definition_id: number
  name: string
  tester: string
  suites: string[]
  config: Record<string, unknown>
  target_id: number | null
  target?: Target | null
  overrides: JobStepOverride
  effective_suites: string[]
  effective_config: Record<string, unknown>
  effective_target_id: number | null
  effective_target?: Target | null
  timeout?: number | null
}

export interface Job {
  id: number
  name: string
  description: string | null
  schedule: string
  is_scheduled?: boolean
  schedule_description?: string | null
  enabled: boolean
  run_mode: JobRunMode
  missed_run_policy?: MissedRunPolicy
  next_run_at?: string | null
  upcoming_runs?: string[]
  last_scheduled_at?: string | null
  is_running?: boolean
  active_run_id?: number | null
  definition_ids: number[]
  overrides: Record<string, JobStepOverride | unknown>
  definitions: JobDefinitionItem[]
  created_at: string
  updated_at: string
}

export interface JobCreate {
  name: string
  description?: string | null
  schedule?: string
  enabled?: boolean
  run_mode?: JobRunMode
  missed_run_policy?: MissedRunPolicy
  definition_ids?: number[]
  overrides?: Record<string, unknown>
  definitions?: Array<{
    test_definition_id: number
    suites?: string[]
    config?: Record<string, unknown>
    target_id?: number | null
    timeout?: number | null
    overrides?: Record<string, unknown>
  }>
}

export interface JobUpdate {
  name?: string
  description?: string | null
  schedule?: string
  enabled?: boolean
  run_mode?: JobRunMode
  missed_run_policy?: MissedRunPolicy
  definition_ids?: number[]
  overrides?: Record<string, unknown>
  definitions?: Array<{
    test_definition_id: number
    suites?: string[]
    config?: Record<string, unknown>
    target_id?: number | null
    timeout?: number | null
    overrides?: Record<string, unknown>
  }>
}

export interface TriggerJobRunRequest {
  run_mode?: JobRunMode
  overrides?: Record<string, unknown>
  config?: Record<string, unknown>
  suites?: string[]
  target_id?: number | null
  timeout?: number | null
  definition_ids?: number[]
  prevent_overlap?: boolean
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
  test_definition_id?: number | null
  definition_name?: string | null
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
  job_name?: string | null
  status: RunStatus
  queued_at: string | null
  started_at: string | null
  finished_at: string | null
  config_snapshot: {
    job_id?: number
    job_name?: string
    run_mode?: JobRunMode
    tester?: string
    testers?: string[]
    suites?: string[]
    config?: Record<string, unknown>
    test_definition_id?: number
    definition_ids?: number[]
    definitions?: Array<Record<string, unknown>>
    overrides?: Record<string, unknown>
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
    targets?: Array<Record<string, unknown>>
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

export interface CompareMetricCell {
  run_id: number
  result_id: number
  is_baseline: boolean
  value: number | null
  present: boolean
  delta: number | null
  abs_delta: number | null
  delta_abs?: number | null
  pct_delta: number | null
  delta_pct?: number | null
  percent_delta?: number | null
  regression: boolean
  is_regression?: boolean
  improvement: boolean
  is_improvement?: boolean
  flag: 'baseline' | 'regression' | 'improvement' | 'changed' | 'unchanged' | 'missing' | string
  status?: string
}

export interface CompareMetricRow {
  name: string
  metric?: string
  unit: string | null
  better: 'higher' | 'lower' | null
  threshold: number
  baseline_run_id: number
  baseline_result_id: number
  baseline_value: number | null
  target_value: number | null
  delta: number | null
  abs_delta: number | null
  pct_delta: number | null
  regression: boolean
  has_regression: boolean
  improvement: boolean
  has_improvement: boolean
  changed: boolean
  has_change: boolean
  missing_in_some: boolean
  flag: string
  status?: string
  values: CompareMetricCell[]
  comparisons: CompareMetricCell[]
  by_run_id?: Record<string, CompareMetricCell>
  by_result_id?: Record<string, CompareMetricCell>
}

export interface CompareCaseMetricDelta extends CompareMetricCell {
  name: string
  unit: string | null
  better: 'higher' | 'lower' | null
  baseline_value: number | null
}

export interface CompareCaseCell {
  run_id: number
  result_id: number
  case_result_id: number | null
  original_case_id: string | null
  is_baseline: boolean
  present: boolean
  status: CaseStatus | string | null
  status_change:
    | 'baseline'
    | 'unchanged'
    | 'regression'
    | 'improvement'
    | 'changed'
    | 'added'
    | 'missing'
    | string
  status_changed: boolean
  status_regression: boolean
  status_improvement: boolean
  metric_regression: boolean
  metric_improvement: boolean
  regression: boolean
  improvement: boolean
  metrics: Record<string, unknown>
  metric_deltas: Record<string, CompareCaseMetricDelta>
  detail: Record<string, unknown> | null
}

export interface CompareCaseRow {
  case_id: string
  suite: string
  baseline_run_id: number
  baseline_result_id: number
  baseline_status: string | null
  target_status: string | null
  status_change: string
  status_changed: boolean
  status_regression: boolean
  status_improvement: boolean
  has_status_regression: boolean
  has_status_improvement: boolean
  has_metric_regression: boolean
  has_metric_improvement: boolean
  has_metric_change: boolean
  regression: boolean
  has_regression: boolean
  improvement: boolean
  has_improvement: boolean
  changed: boolean
  has_change: boolean
  missing_in_some: boolean
  metrics: CompareMetricRow[]
  metric_changes: Record<string, CompareMetricRow>
  values: CompareCaseCell[]
  comparisons: CompareCaseCell[]
  by_run_id?: Record<string, CompareCaseCell>
  by_result_id?: Record<string, CompareCaseCell>
}

export interface CompareArtifactItem {
  id: number
  result_id: number
  run_id: number
  name: string
  path: string
  mime: string
  size_bytes: number | null
  created_at: string | null
  download_url: string
}

export interface ComparedRunMeta {
  run_id: number
  result_id: number
  job_id: number | null
  test_definition_id: number | null
  is_baseline: boolean
  tester: string
  status: ResultStatus | string
  result_status: ResultStatus | string
  run_status: RunStatus | string
  summary: string | null
  target_id: number | null
  target_name: string | null
  target_kind: string | null
  base_url: string | null
  model: string | null
  models: string[]
  suites: string[]
  config: Record<string, unknown>
  config_snapshot: Record<string, unknown>
  target_snapshot: Record<string, unknown>
  git_info: Record<string, unknown> | null
  queued_at: string | null
  started_at: string | null
  finished_at: string | null
  case_count: number
  artifact_count: number
  artifacts: CompareArtifactItem[]
}

export interface CompareConfigDiff {
  targets_differ: boolean
  models_differ: boolean
  suites_differ: boolean
  config_keys_changed: string[]
  config_changes: Record<
    string,
    Array<{
      run_id: number
      result_id: number
      value: unknown
    }>
  >
  targets: Array<{
    run_id: number
    result_id: number
    target_id: number | null
    target_name: string | null
    base_url: string | null
    model: string | null
  }>
}

export interface CompareRegressionItem {
  type: 'metric' | 'case_status'
  name?: string
  unit?: string | null
  better?: 'higher' | 'lower' | null
  suite?: string
  case_id?: string
  baseline_run_id: number
  baseline_result_id: number
  baseline_value?: number | null
  baseline_status?: string | null
  run_id: number
  result_id: number
  value?: number | null
  status?: string | null
  abs_delta?: number | null
  pct_delta?: number | null
}

export interface CompareSummary {
  has_regression: boolean
  has_improvement: boolean
  total_metrics: number
  metrics_changed: number
  metrics_regressed: number
  metrics_improved: number
  metrics_unchanged: number
  metrics_missing: number
  total_cases: number
  cases_changed: number
  cases_status_changed: number
  cases_regressed: number
  cases_status_regressed: number
  cases_metric_regressed: number
  cases_improved: number
  cases_status_improved: number
  cases_unchanged: number
  cases_missing: number
  regressions: CompareRegressionItem[]
}

export interface CompareResponse {
  tester: string
  threshold: number
  baseline_run_id: number
  baseline_result_id: number
  run_ids: number[]
  result_ids: number[]
  runs: ComparedRunMeta[]
  config_diff: CompareConfigDiff
  summary: CompareSummary
  metrics: CompareMetricRow[]
  metric_map: Record<string, CompareMetricRow>
  cases: CompareCaseRow[]
  case_map: Record<string, CompareCaseRow>
}

export interface CompareParams {
  run_ids?: number[]
  result_ids?: number[]
  baseline_run_id?: number | null
  baseline_result_id?: number | null
  threshold?: number
  metric_thresholds?: Record<string, number>
  tester?: string
  suite?: string
  changed_only?: boolean
  regressed_only?: boolean
  allow_different_testers?: boolean
}

