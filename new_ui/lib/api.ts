const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

export interface Detection {
  species: string
  count: number
}

export interface DetectionResult {
  detections: Detection[]
  totalCount: number
  annotatedImage: string
}

export async function detectAnimals(file: File, confidence = 0.25): Promise<DetectionResult> {
  const formData = new FormData()
  formData.append("file", file)

  const response = await fetch(`${API_URL}/api/detect?confidence=${confidence}`, {
    method: "POST",
    body: formData,
  })

  if (!response.ok) {
    throw new Error("Error al procesar la imagen")
  }

  return response.json()
}

export async function checkModelHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}/api/health`)
    return response.ok
  } catch {
    return false
  }
}
