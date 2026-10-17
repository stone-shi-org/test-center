export function Badge({
  status,
  size = 'md',
}: {
  status: string
  size?: 'sm' | 'md' | 'lg'
}) {
  const s = status.toLowerCase()
  let variant = 'default'

  if (s === 'completed' || s === 'pass' || s === 'success' || s === 'healthy') {
    variant = 'success'
  } else if (s === 'running' || s === 'in_progress') {
    variant = 'running'
  } else if (s === 'queued') {
    variant = 'warning'
  } else if (s === 'failed' || s === 'fail' || s === 'error') {
    variant = 'error'
  } else if (s === 'cancelled' || s === 'skip') {
    variant = 'neutral'
  }

  return (
    <span className={`status-badge badge-${variant} badge-${size}`}>
      {s === 'running' && <span className="pulsing-dot" />}
      {status.toUpperCase()}
    </span>
  )
}
