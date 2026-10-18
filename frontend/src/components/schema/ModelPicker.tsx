import { useEffect, useMemo, useState, type KeyboardEvent } from 'react'
import { api, type JSONSchemaProperty, type TargetModelInfo } from '../../api/index.ts'

export type ModelPreset = 'per_engine' | 'all' | 'default'

export const BUILTIN_DEFAULT_MODELS: TargetModelInfo[] = [
  {
    id: 'parakeet-ctc-0.6b',
    type: 'asr',
    engine: 'parakeet',
    is_default: true,
    sample_rate: 16000,
    supports_streaming: true,
    supports_num_speakers_hint: false,
    default_voice: null,
    supported_voices: [],
    grpc_streaming: true,
  },
  {
    id: 'pyannote/speaker-diarization-community-1',
    type: 'diarization',
    engine: 'pyannote',
    is_default: true,
    sample_rate: 16000,
    supports_streaming: false,
    supports_num_speakers_hint: true,
    default_voice: null,
    supported_voices: [],
    grpc_streaming: false,
  },
  {
    id: 'kokoro-82m',
    type: 'tts',
    engine: 'kokoro',
    is_default: true,
    sample_rate: 24000,
    supports_streaming: true,
    supports_num_speakers_hint: false,
    default_voice: 'af_heart',
    supported_voices: ['af_heart'],
    grpc_streaming: false,
  },
]

const KNOWN_BATCH_ONLY_PREFIXES = ['whisper-', 'whisper_', 'parakeet-tdt', 'parakeet_tdt']

export function isBatchOnlyGrpcModelId(modelId: string): boolean {
  const low = modelId.trim().toLowerCase()
  if (!low) return false
  return KNOWN_BATCH_ONLY_PREFIXES.some((prefix) => low.startsWith(prefix))
}

export function inferModelInfoFromId(rawId: string): TargetModelInfo {
  const id = rawId.trim()
  const low = id.toLowerCase()
  if (low.includes('pyannote') || low.includes('sortformer') || low.includes('diar')) {
    const engine = low.includes('sortformer') ? 'nemo_sortformer' : 'pyannote'
    return {
      id,
      type: 'diarization',
      engine,
      is_default: false,
      sample_rate: 16000,
      supports_streaming: false,
      supports_num_speakers_hint: engine === 'pyannote',
      default_voice: null,
      supported_voices: [],
      grpc_streaming: false,
    }
  }
  if (low.includes('kokoro') || low.includes('piper') || low.includes('tts')) {
    const engine = low.includes('piper') ? 'piper' : 'kokoro'
    return {
      id,
      type: 'tts',
      engine,
      is_default: false,
      sample_rate: engine === 'piper' ? 22050 : 24000,
      supports_streaming: true,
      supports_num_speakers_hint: false,
      default_voice: engine === 'kokoro' ? 'af_heart' : null,
      supported_voices: engine === 'kokoro' ? ['af_heart'] : [],
      grpc_streaming: false,
    }
  }
  const engine = low.includes('whisper')
    ? 'whisper'
    : low.includes('tdt')
      ? 'parakeet_tdt'
      : 'parakeet'
  return {
    id,
    type: 'asr',
    engine,
    is_default: false,
    sample_rate: 16000,
    supports_streaming: true,
    supports_num_speakers_hint: false,
    default_voice: null,
    supported_voices: [],
    grpc_streaming: !isBatchOnlyGrpcModelId(id),
  }
}

export function selectPresetModelsForGroup(
  groupModels: TargetModelInfo[],
  mode: ModelPreset
): string[] {
  if (groupModels.length === 0) return []
  if (mode === 'all') {
    return groupModels.map((m) => m.id)
  }
  if (mode === 'default') {
    const def = groupModels.find((m) => m.is_default) || groupModels[0]
    return def ? [def.id] : []
  }
  // per_engine: one model per distinct engine, preferring is_default
  const byEngine = new Map<string, TargetModelInfo>()
  for (const m of groupModels) {
    const existing = byEngine.get(m.engine)
    if (!existing || (m.is_default && !existing.is_default)) {
      byEngine.set(m.engine, m)
    }
  }
  return Array.from(byEngine.values()).map((m) => m.id)
}

function extractLegacyModelsConfig(
  allValues: Record<string, unknown> | undefined
): ModelPreset | string[] | undefined {
  if (!allValues) return undefined
  const explicit: string[] = []
  const addId = (val: unknown) => {
    if (typeof val === 'string' && val.trim() && !explicit.includes(val.trim())) {
      explicit.push(val.trim())
    }
  }
  const addList = (val: unknown) => {
    if (Array.isArray(val)) {
      for (const item of val) {
        addId(item)
      }
    }
  }

  addList(allValues.transcription_models)
  addId(allValues.transcription_model)
  addList(allValues.diarization_models)
  addId(allValues.diarization_model)
  addList(allValues.tts_models)
  addId(allValues.tts_model)
  addList(allValues.grpc_models)
  addId(allValues.grpc_model)

  if (explicit.length > 0) {
    return explicit
  }
  const legacyPreset = allValues.model_parametrization
  if (
    legacyPreset === 'per_engine' ||
    legacyPreset === 'all' ||
    legacyPreset === 'default'
  ) {
    return legacyPreset
  }
  return undefined
}

const PRESET_LABELS: Record<ModelPreset, string> = {
  all: 'All (resolved when the run starts)',
  default: 'Default (resolved when the run starts)',
  per_engine: 'Per engine (resolved when the run starts)',
}

const GROUP_TITLES: { type: 'asr' | 'diarization' | 'tts'; label: string }[] = [
  { type: 'asr', label: 'ASR' },
  { type: 'diarization', label: 'Diarization' },
  { type: 'tts', label: 'TTS' },
]

export interface ModelPickerProps {
  fieldKey: string
  property: JSONSchemaProperty
  value: unknown
  allValues?: Record<string, unknown>
  targetId?: number | string | null
  disabled?: boolean
  onChange: (newValue: ModelPreset | string[]) => void
}

export function ModelPicker({
  fieldKey,
  property,
  value,
  allValues,
  targetId,
  disabled = false,
  onChange,
}: ModelPickerProps) {
  const [targetModels, setTargetModels] = useState<TargetModelInfo[] | null>(null)
  const [loadingTarget, setLoadingTarget] = useState<boolean>(false)
  const [targetError, setTargetError] = useState<string | null>(null)
  const [customModels, setCustomModels] = useState<TargetModelInfo[]>([])
  const [manualInput, setManualInput] = useState<string>('')

  const numericTargetId = useMemo(() => {
    if (typeof targetId === 'number' && !Number.isNaN(targetId) && targetId > 0) {
      return targetId
    }
    if (typeof targetId === 'string' && targetId.trim()) {
      const parsed = parseInt(targetId.trim(), 10)
      if (!Number.isNaN(parsed) && parsed > 0) {
        return parsed
      }
    }
    return null
  }, [targetId])

  useEffect(() => {
    if (numericTargetId === null) {
      setTargetModels(null)
      setTargetError(null)
      setLoadingTarget(false)
      return
    }

    let cancelled = false
    setLoadingTarget(true)
    setTargetError(null)

    api.targets
      .listModels(numericTargetId)
      .then((models) => {
        if (cancelled) return
        setTargetModels(Array.isArray(models) ? models : [])
        setTargetError(null)
        setLoadingTarget(false)
      })
      .catch((err: unknown) => {
        if (cancelled) return
        setTargetModels(null)
        setTargetError(err instanceof Error ? err.message : String(err))
        setLoadingTarget(false)
      })

    return () => {
      cancelled = true
    }
  }, [numericTargetId])

  // Resolve effective config value (supporting legacy keys if `models` was not explicitly set)
  const effectiveValue = useMemo<ModelPreset | string[]>(() => {
    if (value === 'per_engine' || value === 'all' || value === 'default') {
      return value
    }
    if (Array.isArray(value)) {
      return value.map((v) => String(v).trim()).filter(Boolean)
    }
    const legacy = extractLegacyModelsConfig(allValues)
    if (legacy !== undefined) {
      return legacy
    }
    return 'per_engine'
  }, [value, allValues])

  const activePreset: ModelPreset | null =
    typeof effectiveValue === 'string' ? effectiveValue : null

  const hasTargetCatalog =
    numericTargetId !== null && targetModels !== null && targetModels.length > 0
  const isFallbackMode = !hasTargetCatalog

  const baseCatalog = useMemo<TargetModelInfo[]>(() => {
    return hasTargetCatalog ? targetModels : BUILTIN_DEFAULT_MODELS
  }, [hasTargetCatalog, targetModels])

  const baseCatalogIds = useMemo(
    () => new Set(baseCatalog.map((m) => m.id)),
    [baseCatalog]
  )

  // Combine base catalog + manually added models + any explicit models from template/config not in catalog
  const allDisplayedModels = useMemo<TargetModelInfo[]>(() => {
    const map = new Map<string, TargetModelInfo>()
    for (const m of baseCatalog) {
      map.set(m.id, m)
    }
    for (const m of customModels) {
      if (!map.has(m.id)) {
        map.set(m.id, m)
      }
    }
    if (Array.isArray(effectiveValue)) {
      for (const id of effectiveValue) {
        if (!map.has(id)) {
          map.set(id, inferModelInfoFromId(id))
        }
      }
    }
    return Array.from(map.values())
  }, [baseCatalog, customModels, effectiveValue])

  // Determine which model IDs are currently checked (either previewed via preset or explicitly selected)
  const checkedModelIds = useMemo<string[]>(() => {
    if (Array.isArray(effectiveValue)) {
      return effectiveValue
    }
    // Preset preview across each type group from the base catalog
    const preview: string[] = []
    for (const { type } of GROUP_TITLES) {
      const groupModels = baseCatalog.filter((m) => m.type === type)
      preview.push(...selectPresetModelsForGroup(groupModels, effectiveValue))
    }
    return preview
  }, [effectiveValue, baseCatalog])

  const checkedSet = useMemo(() => new Set(checkedModelIds), [checkedModelIds])

  const handleSelectPreset = (preset: ModelPreset) => {
    onChange(preset)
  }

  const handleClear = () => {
    onChange([])
  }

  const handleToggleCheckbox = (modelId: string) => {
    const currentList = [...checkedModelIds]
    const nextList = currentList.includes(modelId)
      ? currentList.filter((id) => id !== modelId)
      : [...currentList, modelId]
    onChange(nextList)
  }

  const handleAddManualModel = () => {
    const trimmed = manualInput.trim()
    if (!trimmed) return
    const inferred = inferModelInfoFromId(trimmed)
    setCustomModels((prev) =>
      prev.some((m) => m.id === inferred.id) ? prev : [...prev, inferred]
    )
    const currentList = [...checkedModelIds]
    const nextList = currentList.includes(inferred.id)
      ? currentList
      : [...currentList, inferred.id]
    onChange(nextList)
    setManualInput('')
  }

  const handleManualKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleAddManualModel()
    }
  }

  return (
    <div
      className="schema-field-group model-picker-field"
      data-testid={`model-picker-${fieldKey}`}
    >
      <div className="model-picker-header">
        <div>
          <label className="field-title">{fieldKey}</label>
          {property.description && <p className="field-desc">{property.description}</p>}
        </div>

        <div className="model-picker-presets" role="group" aria-label="Model selection presets">
          <button
            type="button"
            className={`btn btn-sm model-preset-btn ${activePreset === 'all' ? 'active' : ''}`}
            disabled={disabled}
            onClick={() => handleSelectPreset('all')}
          >
            All
          </button>
          <button
            type="button"
            className={`btn btn-sm model-preset-btn ${activePreset === 'default' ? 'active' : ''}`}
            disabled={disabled}
            onClick={() => handleSelectPreset('default')}
          >
            Default
          </button>
          <button
            type="button"
            className={`btn btn-sm model-preset-btn ${activePreset === 'per_engine' ? 'active' : ''}`}
            disabled={disabled}
            onClick={() => handleSelectPreset('per_engine')}
          >
            Per engine
          </button>
          <button
            type="button"
            className="btn btn-sm model-preset-btn model-preset-clear"
            disabled={disabled}
            onClick={handleClear}
          >
            Clear
          </button>
        </div>
      </div>

      {activePreset && (
        <div className="model-picker-preset-banner" data-testid="model-picker-preset-status">
          {PRESET_LABELS[activePreset]}
        </div>
      )}

      {loadingTarget && (
        <div className="model-picker-status-note">Querying target /v1/models...</div>
      )}

      {targetError && (
        <div className="model-picker-fallback-notice" role="status">
          Target models unavailable ({targetError}) — showing built-in defaults.
        </div>
      )}

      {numericTargetId === null && !loadingTarget && (
        <div className="model-picker-fallback-notice" role="status">
          No target selected — showing built-in defaults. Presets are resolved against the target when the run starts.
        </div>
      )}

      <div className="model-picker-groups">
        {GROUP_TITLES.map(({ type, label }) => {
          const groupModels = allDisplayedModels.filter((m) => m.type === type)
          return (
            <div
              key={type}
              className="model-picker-group"
              data-testid={`model-picker-group-${type}`}
            >
              <div className="model-picker-group-title">{label}</div>
              <div className="model-picker-list">
                {groupModels.map((model) => {
                  const checked = checkedSet.has(model.id)
                  const isMissingOnTarget =
                    hasTargetCatalog && !baseCatalogIds.has(model.id)
                  const voicesList =
                    model.supported_voices && model.supported_voices.length > 0
                      ? model.supported_voices
                      : model.default_voice
                        ? [model.default_voice]
                        : []

                  return (
                    <label
                      key={model.id}
                      className={`model-picker-item ${checked ? 'checked' : ''} ${
                        isMissingOnTarget ? 'mismatch' : ''
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        disabled={disabled}
                        onChange={() => handleToggleCheckbox(model.id)}
                      />
                      <span className="model-picker-id">{model.id}</span>
                      <span className="model-picker-badges">
                        {model.engine && (
                          <span className="model-badge model-engine-badge">
                            {model.engine}
                          </span>
                        )}
                        {model.is_default && (
                          <span className="model-badge model-default-badge">
                            default
                          </span>
                        )}
                        {model.type === 'asr' && !model.grpc_streaming && (
                          <span
                            className="model-badge model-batch-badge"
                            title="Not usable for gRPC streaming"
                          >
                            batch only
                          </span>
                        )}
                        {model.type === 'tts' && voicesList.length > 0 && (
                          <span className="model-badge model-voices-badge">
                            voices: {voicesList.join(', ')}
                          </span>
                        )}
                        {isMissingOnTarget && (
                          <span className="badge badge-warning model-picker-mismatch-badge">
                            not on this target
                          </span>
                        )}
                      </span>
                    </label>
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>

      {isFallbackMode && (
        <div className="model-picker-manual-add">
          <input
            type="text"
            className="form-input model-picker-manual-input"
            placeholder="Add model ID"
            aria-label="Add model ID"
            value={manualInput}
            disabled={disabled}
            onChange={(e) => setManualInput(e.target.value)}
            onKeyDown={handleManualKeyDown}
          />
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            disabled={disabled || !manualInput.trim()}
            onClick={handleAddManualModel}
          >
            Add
          </button>
        </div>
      )}
    </div>
  )
}
