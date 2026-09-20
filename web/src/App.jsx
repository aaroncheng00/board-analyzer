import { useRef, useState } from 'react'
import BoardCanvas from './BoardCanvas.jsx'
import { TABS, VIEW } from './views.js'

async function post(path, file) {
  const body = new FormData()
  body.append('file', file)
  const res = await fetch(path, { method: 'POST', body })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error(detail.error || `${path} failed (${res.status})`)
  }
  return res
}

export default function App() {
  const [file, setFile] = useState(null)
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [view, setView] = useState(VIEW.LABELS)
  const picker = useRef(null)

  // Flip the busy flag, run, put any failure on the error line
  async function guard(fn) {
    setBusy(true)
    try {
      await fn()
    } catch (exc) {
      setError(exc.message)
    } finally {
      setBusy(false)
    }
  }

  async function upload(chosen) {
    if (!chosen) return
    setError(null)
    setData(null)
    setFile(chosen)
    await guard(async () => {
      const res = await post('/api/analyze', chosen)
      setData(await res.json())
    })
  }

  async function download() {
    await guard(async () => {
      const res = await post('/api/cells', file)
      const url = URL.createObjectURL(await res.blob())
      const a = document.createElement('a')
      a.href = url
      a.download = 'board.zip'
      a.click()
      URL.revokeObjectURL(url)
    })
  }

  return (
    <div className="app">
      <h1>Board analyzer</h1>
      <p className="sub">
        Upload a board screenshot. The pipeline finds the board, infers the grid, and
        classifies each cell.
      </p>
      <p className="sub">
        Supported games: chess, checkers, tango, and queens.
      </p>

      <div className="controls">
        <input
          ref={picker}
          type="file"
          accept="image/*"
          style={{ display: 'none' }}
          onChange={(e) => upload(e.target.files[0])}
        />
        <button onClick={() => picker.current.click()} disabled={busy}>
          {busy ? 'Analyzing...' : 'Upload screenshot'}
        </button>

        {data && (
          <>
            <div className="tabs">
              {TABS.map(([key, label]) => (
                <button
                  key={key}
                  className={view === key ? 'active' : ''}
                  onClick={() => setView(key)}
                >
                  {label}
                </button>
              ))}
            </div>
            <button onClick={download} disabled={busy}>Download cells</button>
          </>
        )}
      </div>

      {error && <div className="error">{error}</div>}

      {data && (
        <p className="meta">
          {data.rows}&times;{data.cols} grid &middot; board at ({data.bbox[0]},{' '}
          {data.bbox[1]}) {data.bbox[2]}&times;{data.bbox[3]}px
        </p>
      )}

      <BoardCanvas data={data} view={view} />
    </div>
  )
}
