import { useEffect, useRef } from 'react'
import { VIEW } from './views.js'
import { STYLE } from './canvasStyle.js'

const GAMES = ['chess', 'tango', 'queens', 'checkers', 'go']
const COLOURS = ['black', 'white', 'red']

// 'chess_black_bishop' -> 'b_bishop'
// Remove game names from labels
export function shortLabel(name) {
  if (!name) return ''
  let parts = name.split('_')
  if (GAMES.includes(parts[0])) parts = parts.slice(1)
  if (parts.length > 1 && COLOURS.includes(parts[0])) {
    parts = [parts[0][0], ...parts.slice(1)]
  }
  return parts.join('_')
}

// White fill over a dark outline, which stays legible on any square colour --
// the boards run from pale wood to near-black.
function outlined(ctx, text, x, y, font, fill) {
  ctx.font = font
  ctx.strokeStyle = STYLE.textOutline
  ctx.strokeText(text, x, y)
  ctx.fillStyle = fill
  ctx.fillText(text, x, y)
}

function drawGrid(ctx, data) {
  const { row_lines: rl, col_lines: cl } = data
  // Lines span the fitted grid, not the whole crop
  const [x0, x1] = [cl[0], cl[cl.length - 1]]
  const [y0, y1] = [rl[0], rl[rl.length - 1]]

  ctx.strokeStyle = STYLE.gridLine
  ctx.lineWidth = Math.max(1, Math.round(Math.min(x1 - x0, y1 - y0) / 400))
  ctx.beginPath()
  for (const y of rl) { ctx.moveTo(x0, y + 0.5); ctx.lineTo(x1, y + 0.5) }
  for (const x of cl) { ctx.moveTo(x + 0.5, y0); ctx.lineTo(x + 0.5, y1) }
  ctx.stroke()
}

// Shrink a uniform font until the widest label fits inside a cell
function fitFont(ctx, texts, cellW, start) {
  let size = start
  for (const text of texts) {
    ctx.font = `${STYLE.weight} ${size}px ${STYLE.font}`
    const width = ctx.measureText(text).width
    if (width > cellW * 0.92) size *= (cellW * 0.92) / width
  }
  return Math.max(7, size)
}

function drawLabels(ctx, data) {
  const { row_lines: rl, col_lines: cl, labels, confidences } = data
  // Size off the smallest cell to avoid overflow
  const cellH = Math.min(...rl.slice(1).map((v, i) => v - rl[i]))
  const cellW = Math.min(...cl.slice(1).map((v, i) => v - cl[i]))

  const names = labels.flat().filter((n) => n && n !== 'empty').map(shortLabel)
  if (!names.length) return
  const size = fitFont(ctx, names, cellW, Math.min(cellW / 5.5, cellH / 5.5))

  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.lineJoin = 'round'
  ctx.lineWidth = Math.max(2, size / 4)

  for (let r = 0; r < data.rows; r++) {
    for (let c = 0; c < data.cols; c++) {
      const name = labels[r][c]
      // 'empty' and a null crop are skipped
      if (!name || name === 'empty') continue
      const cx = (cl[c] + cl[c + 1]) / 2
      const cy = (rl[r] + rl[r + 1]) / 2
      outlined(ctx, shortLabel(name), cx, cy,
               `${STYLE.weight} ${size}px ${STYLE.font}`, STYLE.labelFill)
      outlined(ctx, confidences[r][c].toFixed(2), cx, cy + size * 1.2,
               `${size * 0.85}px ${STYLE.font}`, STYLE.confidenceFill)
    }
  }
}

export default function BoardCanvas({ data, view }) {
  const ref = useRef(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas || !data) return

    // cancelled is specific to each render instance
    // Prevent old canvas from rendering after a tab switch or equivalent
    let cancelled = false

    const img = new Image()
    img.onload = () => {
      if (cancelled) return
      // Canvas is sized to the crop's true pixels and scaled down by CSS
      // Overlays use board coordinates directly with no transform
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      const ctx = canvas.getContext('2d')
      ctx.drawImage(img, 0, 0)
      if (view !== VIEW.ORIGINAL) drawGrid(ctx, data)
      if (view === VIEW.LABELS) drawLabels(ctx, data)
    }
    img.src = data.board_crop_png

    return () => { cancelled = true }
  }, [data, view])

  return <canvas ref={ref} />
}
