"use client"

import { useState, useCallback, useEffect } from "react"
import { Header } from "@/components/header"
import { ImageUploader } from "@/components/image-uploader"
import { ResultsPanel } from "@/components/results-panel"
import { ModelInfoPanel } from "@/components/model-info-panel"
import { detectAnimals, checkModelHealth } from "@/lib/api"

export interface DetectionResult {
  annotatedImage: string | null
  counts: Record<string, number>
  totalCount: number
}

export default function Home() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [results, setResults] = useState<DetectionResult | null>(null)
  const [modelStatus, setModelStatus] = useState<"online" | "offline" | "checking">("checking")
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const checkHealth = async () => {
      const isOnline = await checkModelHealth()
      setModelStatus(isOnline ? "online" : "offline")
    }
    checkHealth()
  }, [])

  const handleImageUpload = useCallback((imageDataUrl: string, file: File) => {
    setUploadedImage(imageDataUrl)
    setUploadedFile(file)
    setResults(null)
    setError(null)
  }, [])

  const handleRunDetection = useCallback(async () => {
    if (!uploadedFile) return

    setIsProcessing(true)
    setError(null)

    try {
      const apiResult = await detectAnimals(uploadedFile)

      // Transform API response to our format
      const counts: Record<string, number> = {}
      apiResult.detections.forEach((d) => {
        counts[d.species] = d.count
      })

      setResults({
        annotatedImage: apiResult.annotatedImage,
        counts,
        totalCount: apiResult.totalCount,
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error al procesar la imagen")
    } finally {
      setIsProcessing(false)
    }
  }, [uploadedFile])

  const handleClear = useCallback(() => {
    setUploadedImage(null)
    setUploadedFile(null)
    setResults(null)
    setError(null)
  }, [])

  return (
    <div className="min-h-screen bg-background">
      <Header modelStatus={modelStatus} />

      <main className="container mx-auto px-4 py-8 max-w-7xl">
        {error && (
          <div className="mb-6 p-4 bg-destructive/10 border border-destructive/20 rounded-lg text-destructive text-sm">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column - Upload */}
          <div className="space-y-6">
            <ImageUploader
              onImageUpload={handleImageUpload}
              uploadedImage={uploadedImage}
              onClear={handleClear}
              isProcessing={isProcessing}
              onRunDetection={handleRunDetection}
              hasImage={!!uploadedFile}
            />
            <ModelInfoPanel />
          </div>

          {/* Right Column - Results */}
          <ResultsPanel results={results} isProcessing={isProcessing} />
        </div>
      </main>
    </div>
  )
}
