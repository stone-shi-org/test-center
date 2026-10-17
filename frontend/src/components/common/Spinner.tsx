export function Spinner({
  size = 'md',
  message,
}: {
  size?: 'sm' | 'md' | 'lg'
  message?: string
}) {
  return (
    <div className={`spinner-container spinner-${size}`}>
      <div className="spinner-circle" />
      {message && <span className="spinner-message">{message}</span>}
    </div>
  )
}
