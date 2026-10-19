import { useState, useEffect, useRef, useMemo, type ChangeEvent, type ReactNode } from 'react'
import type { JSONSchemaObject, JSONSchemaProperty } from '../../api/types.ts'
import { ModelPicker } from './ModelPicker.tsx'

export interface SchemaFormContext {
  targetId?: number | string | null
}

export interface CustomWidgetRenderProps {
  fieldKey: string
  property: JSONSchemaProperty
  value: unknown
  allValues: Record<string, unknown>
  context?: SchemaFormContext
  disabled: boolean
  onChange: (newValue: unknown) => void
}

function isFieldNonDefault(val: unknown, prop: JSONSchemaProperty): boolean {
  if (val === undefined || val === null) {
    return false
  }
  if (prop.default !== undefined) {
    if (typeof prop.default === 'object' && prop.default !== null) {
      return JSON.stringify(val) !== JSON.stringify(prop.default)
    }
    return val !== prop.default
  }
  if (typeof val === 'string') {
    return val.trim().length > 0
  }
  if (Array.isArray(val)) {
    return val.length > 0
  }
  if (typeof val === 'object' && val !== null) {
    return Object.keys(val as Record<string, unknown>).length > 0
  }
  return true
}

export function SchemaForm({
  schema,
  values,
  onChange,
  disabled = false,
  context,
  targetId,
  widgets,
  omitFields,
}: {
  schema: JSONSchemaObject
  values: Record<string, unknown>
  onChange: (newValues: Record<string, unknown>) => void
  disabled?: boolean
  context?: SchemaFormContext
  targetId?: number | string | null
  widgets?: Record<string, (props: CustomWidgetRenderProps) => ReactNode>
  omitFields?: string[]
}) {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const prevSchemaRef = useRef(schema)
  const lastEmittedValuesRef = useRef<Record<string, unknown> | null>(null)
  const resolvedTargetId = context?.targetId !== undefined ? context.targetId : targetId
  const omittedSet = useMemo(() => new Set(omitFields ?? []), [omitFields])

  const handleFieldChange = (key: string, value: unknown) => {
    const next: Record<string, unknown> = { ...values }
    if (value === undefined) {
      delete next[key]
    } else {
      next[key] = value
    }
    lastEmittedValuesRef.current = next
    onChange(next)
  }

  const handleResetDefaults = () => {
    if (!schema?.properties) return
    const defaults: Record<string, unknown> = {}
    for (const [key, prop] of Object.entries(schema.properties)) {
      if (omittedSet.has(key)) continue
      if (prop.default !== undefined) {
        defaults[key] = prop.default
      }
    }
    lastEmittedValuesRef.current = defaults
    onChange(defaults)
  }

  // Split properties into essential vs advanced
  const { essentialEntries, advancedEntries } = useMemo(() => {
    if (!schema?.properties) {
      return { essentialEntries: [], advancedEntries: [] }
    }
    const essential: [string, JSONSchemaProperty][] = []
    const advanced: [string, JSONSchemaProperty][] = []
    for (const [key, prop] of Object.entries(schema.properties)) {
      if (omittedSet.has(key)) continue
      if (prop['x-advanced'] === true) {
        advanced.push([key, prop])
      } else {
        essential.push([key, prop])
      }
    }
    return { essentialEntries: essential, advancedEntries: advanced }
  }, [schema, omittedSet])

  // Count changed non-default advanced options
  const changedAdvancedCount = useMemo(() => {
    if (advancedEntries.length === 0) return 0
    let count = 0
    for (const [key, prop] of advancedEntries) {
      if (isFieldNonDefault(values[key], prop)) {
        count++
      }
    }
    return count
  }, [advancedEntries, values])

  // Auto-expand if external values contain non-default advanced options
  useEffect(() => {
    if (schema !== prevSchemaRef.current) {
      prevSchemaRef.current = schema
      setShowAdvanced(changedAdvancedCount > 0)
      return
    }

    if (values !== lastEmittedValuesRef.current) {
      if (changedAdvancedCount > 0) {
        setShowAdvanced(true)
      }
    }
  }, [schema, values, changedAdvancedCount])

  if (!schema?.properties || Object.keys(schema.properties).length === 0) {
    return (
      <div className="empty-schema-notice">
        <p>This tester does not require any additional configuration parameters.</p>
      </div>
    )
  }

  const renderField = (key: string, prop: JSONSchemaProperty) => {
    const isRequired = schema.required?.includes(key)
    const val = values[key] !== undefined ? values[key] : prop.default
    const widgetName = prop['x-widget']

    if (widgetName && widgets?.[widgetName]) {
      return (
        <div key={key} className="schema-custom-widget-wrapper">
          {widgets[widgetName]({
            fieldKey: key,
            property: prop,
            value: val,
            allValues: values,
            context: { targetId: resolvedTargetId },
            disabled,
            onChange: (newVal) => handleFieldChange(key, newVal),
          })}
        </div>
      )
    }

    if (widgetName === 'model-picker') {
      return (
        <ModelPicker
          key={key}
          fieldKey={key}
          property={prop}
          value={val}
          allValues={values}
          targetId={resolvedTargetId}
          disabled={disabled}
          onChange={(newVal) => handleFieldChange(key, newVal)}
        />
      )
    }

    // 1. Boolean field
    if (prop.type === 'boolean') {
      const boolVal = Boolean(val)
      return (
        <div key={key} className="schema-field-group schema-boolean-field">
          <label className="checkbox-label">
            <input
              type="checkbox"
              id={`schema-field-${key}`}
              checked={boolVal}
              disabled={disabled}
              onChange={(e) => handleFieldChange(key, e.target.checked)}
            />
            <span className="checkbox-custom" />
            <div className="field-info">
              <span className="field-label-text">
                {key} {isRequired && <span className="required-star">*</span>}
              </span>
              {prop.description && <span className="field-desc">{prop.description}</span>}
            </div>
          </label>
        </div>
      )
    }

    // 2. Array with enum items (multi-select)
    if (prop.type === 'array' && prop.items?.enum) {
      const currentArr: string[] = Array.isArray(val) ? (val as string[]) : []
      const toggleItem = (item: string) => {
        if (currentArr.includes(item)) {
          handleFieldChange(
            key,
            currentArr.filter((i) => i !== item)
          )
        } else {
          handleFieldChange(key, [...currentArr, item])
        }
      }

      return (
        <div key={key} className="schema-field-group">
          <label className="field-title" htmlFor={`schema-field-${key}`}>
            {key} {isRequired && <span className="required-star">*</span>}
          </label>
          {prop.description && <p className="field-desc">{prop.description}</p>}
          <div className="enum-pills" id={`schema-field-${key}`}>
            {prop.items.enum.map((option) => {
              const selected = currentArr.includes(option)
              return (
                <button
                  type="button"
                  key={option}
                  className={`pill-btn ${selected ? 'selected' : ''}`}
                  disabled={disabled}
                  onClick={() => toggleItem(option)}
                >
                  <span className="pill-check">{selected ? '✓' : '+'}</span>
                  {option}
                </button>
              )
            })}
          </div>
        </div>
      )
    }

    // 3. String with Enum dropdown
    if (prop.type === 'string' && prop.enum && prop.enum.length > 0) {
      const showUnsetOption = !isRequired && prop.default === undefined
      const currentStringVal = typeof val === 'string' ? val : ''

      return (
        <div key={key} className="schema-field-group">
          <label className="field-title" htmlFor={`schema-field-${key}`}>
            {key} {isRequired && <span className="required-star">*</span>}
          </label>
          {prop.description && <p className="field-desc">{prop.description}</p>}
          <select
            id={`schema-field-${key}`}
            className="form-select"
            value={currentStringVal}
            disabled={disabled}
            onChange={(e: ChangeEvent<HTMLSelectElement>) => {
              const selected = e.target.value
              if (selected === '' && showUnsetOption) {
                handleFieldChange(key, undefined)
              } else {
                handleFieldChange(key, selected)
              }
            }}
          >
            {showUnsetOption && (
              <option value="">— Unset —</option>
            )}
            {prop.enum.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </div>
      )
    }

    // 4. Number / Integer
    if (prop.type === 'integer' || prop.type === 'number') {
      const isInt = prop.type === 'integer'
      const numVal = val !== undefined && val !== null ? String(val) : ''

      return (
        <div key={key} className="schema-field-group">
          <div className="field-header">
            <label className="field-title" htmlFor={`schema-field-${key}`}>
              {key} {isRequired && <span className="required-star">*</span>}
            </label>
            {prop.default !== undefined && (
              <span className="field-default-hint">default: {String(prop.default)}</span>
            )}
          </div>
          {prop.description && <p className="field-desc">{prop.description}</p>}
          <input
            type="number"
            id={`schema-field-${key}`}
            className="form-input"
            value={numVal}
            min={prop.minimum}
            max={prop.maximum}
            step={isInt ? 1 : 'any'}
            disabled={disabled}
            onChange={(e) => {
              const strVal = e.target.value
              if (strVal === '') {
                handleFieldChange(key, undefined)
              } else {
                handleFieldChange(key, isInt ? parseInt(strVal, 10) : parseFloat(strVal))
              }
            }}
          />
        </div>
      )
    }

    // 5. General String
    if (prop.type === 'string' || !prop.type) {
      return (
        <div key={key} className="schema-field-group">
          <div className="field-header">
            <label className="field-title" htmlFor={`schema-field-${key}`}>
              {key} {isRequired && <span className="required-star">*</span>}
            </label>
            {prop.default !== undefined && (
              <span className="field-default-hint">default: {String(prop.default)}</span>
            )}
          </div>
          {prop.description && <p className="field-desc">{prop.description}</p>}
          <input
            type="text"
            id={`schema-field-${key}`}
            className="form-input"
            value={typeof val === 'string' ? val : ''}
            placeholder={prop.default ? String(prop.default) : ''}
            disabled={disabled}
            onChange={(e) => handleFieldChange(key, e.target.value)}
          />
        </div>
      )
    }

    // 6. Generic Array / Object fallback (JSON text)
    return (
      <div key={key} className="schema-field-group">
        <label className="field-title" htmlFor={`schema-field-${key}`}>
          {key} ({prop.type}) {isRequired && <span className="required-star">*</span>}
        </label>
        {prop.description && <p className="field-desc">{prop.description}</p>}
        <textarea
          id={`schema-field-${key}`}
          className="form-textarea"
          rows={3}
          value={typeof val === 'object' ? JSON.stringify(val, null, 2) : String(val ?? '')}
          disabled={disabled}
          onChange={(e) => {
            try {
              const parsed = JSON.parse(e.target.value)
              handleFieldChange(key, parsed)
            } catch {
              handleFieldChange(key, e.target.value)
            }
          }}
        />
      </div>
    )
  }

  return (
    <div className="schema-form">
      <div className="schema-form-header">
        <h4>Tester Configuration Options</h4>
        <button
          type="button"
          className="btn-link"
          onClick={handleResetDefaults}
          disabled={disabled}
        >
          Reset to Defaults
        </button>
      </div>

      <div className="schema-fields-grid essential-fields">
        {essentialEntries.map(([key, prop]) => renderField(key, prop))}
      </div>

      {advancedEntries.length > 0 && (
        <div className="schema-advanced-section">
          <div className="schema-advanced-controls">
            <button
              type="button"
              className="btn btn-secondary schema-advanced-toggle-btn"
              onClick={() => setShowAdvanced((prev) => !prev)}
              disabled={disabled}
            >
              {showAdvanced
                ? `Hide advanced options (${advancedEntries.length})`
                : `Show advanced options (${advancedEntries.length})`}
            </button>
            {changedAdvancedCount > 0 && (
              <span className="badge badge-warning schema-advanced-badge">
                {changedAdvancedCount} advanced option{changedAdvancedCount === 1 ? '' : 's'} changed
              </span>
            )}
          </div>
          {showAdvanced && (
            <div className="schema-fields-grid advanced-fields">
              {advancedEntries.map(([key, prop]) => renderField(key, prop))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
