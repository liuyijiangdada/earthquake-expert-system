/** 上传前压缩图片，降低带宽与多模态推理显存占用 */

const DEFAULTS = {
  maxEdge: 1280,
  maxBytes: 1024 * 1024,
  quality: 0.82,
  mimeType: 'image/jpeg',
}

function scaleSize(width, height, maxEdge) {
  if (Math.max(width, height) <= maxEdge) {
    return { width, height }
  }
  const scale = maxEdge / Math.max(width, height)
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  }
}

function canvasToBlob(canvas, mimeType, quality) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('图片编码失败'))),
      mimeType,
      quality,
    )
  })
}

async function bitmapToCompressed(bitmap, options) {
  const { maxEdge, maxBytes, quality, mimeType } = { ...DEFAULTS, ...options }
  const { width, height } = scaleSize(bitmap.width, bitmap.height, maxEdge)

  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('无法创建画布')
  ctx.drawImage(bitmap, 0, 0, width, height)
  bitmap.close?.()

  let q = quality
  let blob = await canvasToBlob(canvas, mimeType, q)
  while (blob.size > maxBytes && q > 0.45) {
    q -= 0.08
    blob = await canvasToBlob(canvas, mimeType, q)
  }

  const dataUrl = await new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result)
    reader.onerror = () => reject(new Error('预览生成失败'))
    reader.readAsDataURL(blob)
  })

  const baseName = (options.fileName || 'image').replace(/\.[^.]+$/, '')
  return {
    blob,
    dataUrl,
    name: `${baseName}.jpg`,
    width,
    height,
    size: blob.size,
  }
}

export async function compressImageFile(file, options = {}) {
  const bitmap = await createImageBitmap(file)
  return bitmapToCompressed(bitmap, { ...options, fileName: file.name })
}

export async function compressImageDataUrl(dataUrl, options = {}) {
  const res = await fetch(dataUrl)
  const blob = await res.blob()
  const bitmap = await createImageBitmap(blob)
  return bitmapToCompressed(bitmap, options)
}
