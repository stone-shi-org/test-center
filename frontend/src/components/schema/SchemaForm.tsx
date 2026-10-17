import type { ChangeEvent } from 'react'
import type { JSONSchemaObject, JSONSchemaProperty } from '../../api/types.ts'

export function SchemaForm({
  schema,
  values,
  onChange,
  disabled = false,
}: {
  schema: JSONSchemaObject
  values: Record<string, unknown>
  onChange: (newValues: Record<string, unknown>) => void
  disabled?: boolean
}) {
  const handleFieldChange = (key: string, value: unknown) => {
    onChange({
      ...values,
      [key]: value,
    })
  }

  const handleResetDefaults = () => {
    if (!schema?.properties) return
    const defaults: Record<string, unknown> = {}
    for (const [key, prop] of Object.entries(schema.properties)) {
      if (prop.default !== undefined) {
        defaults[key] = prop.default
      }
    }
    onChange(defaults)
  }

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
      return (
        <div key={key} className="schema-field-group">
          <label className="field-title" htmlFor={`schema-field-${key}`}>
            {key} {isRequired && <span className="required-star">*</span>}
          </label>
          {prop.description && <p className="field-desc">{prop.description}</p>}
          <select
            id={`schema-field-${key}`}
            className="form-select"
            value={typeof val === 'string' ? val : ''}
            disabled={disabled}
            onChange={(e: ChangeEvent<HTMLSelectElement>) => handleFieldChange(key, e.target.value)}
          >
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
      <div className="schema-fields-grid">
        {Object.entries(schema.properties).map(([key, prop]) => renderField(key, prop))}
      </div>
    </div>
  )
}
