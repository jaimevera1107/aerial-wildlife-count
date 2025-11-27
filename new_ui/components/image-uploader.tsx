"use client"

import type React from "react"

import { useCallback, useRef } from "react"
import { Upload, X, Play, Loader2, ImageIcon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"

interface ImageUploaderProps {
  onImageUpload: (imageDataUrl: string, file: File) => void
  uploadedImage: string | null
  onClear: () => void
  isProcessing: boolean
  onRunDetection: () => void
  hasImage: boolean
}

export function ImageUploader({
  onImageUpload,
  uploadedImage,
  onClear,
  isProcessing,
  onRunDetection,
  hasImage,
}: ImageUploaderProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      const file = e.dataTransfer.files[0]
      if (file && file.type.startsWith("image/")) {
        const reader = new FileReader()
        reader.onload = (event) => {
          onImageUpload(event.target?.result as string, file)
        }
        reader.readAsDataURL(file)
      }
    },
    [onImageUpload],
  )

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) {
        const reader = new FileReader()
        reader.onload = (event) => {
          onImageUpload(event.target?.result as string, file)
        }
        reader.readAsDataURL(file)
      }
    },
    [onImageUpload],
  )

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
  }, [])

  return (
    <Card className="bg-card border-border">
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <ImageIcon className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold text-foreground">Imagen Aérea</h2>
          </div>
          {uploadedImage && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onClear}
              disabled={isProcessing}
              className="text-muted-foreground hover:text-foreground"
            >
              <X className="h-4 w-4 mr-1" />
              Limpiar
            </Button>
          )}
        </div>

        {!uploadedImage ? (
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-border rounded-xl p-12 text-center cursor-pointer transition-all duration-200 hover:border-primary/50 hover:bg-primary/5 group"
          >
            <div className="flex flex-col items-center gap-4">
              <div className="p-4 rounded-full bg-secondary group-hover:bg-primary/10 transition-colors">
                <Upload className="h-8 w-8 text-muted-foreground group-hover:text-primary transition-colors" />
              </div>
              <div>
                <p className="text-foreground font-medium mb-1">Arrastra una imagen o haz clic para seleccionar</p>
                <p className="text-sm text-muted-foreground">PNG, JPG, TIFF hasta 50MB</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="relative rounded-xl overflow-hidden border border-border">
            <img
              src={uploadedImage || "/placeholder.svg"}
              alt="Imagen cargada"
              className="w-full h-auto max-h-80 object-contain bg-secondary/30"
            />
            {isProcessing && (
              <div className="absolute inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center">
                <div className="flex flex-col items-center gap-3">
                  <Loader2 className="h-8 w-8 text-primary animate-spin" />
                  <span className="text-sm text-foreground font-medium">Procesando imagen...</span>
                </div>
              </div>
            )}
          </div>
        )}

        <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileSelect} className="hidden" />

        <Button
          onClick={onRunDetection}
          disabled={!hasImage || isProcessing}
          className="w-full mt-4 bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          size="lg"
        >
          {isProcessing ? (
            <>
              <Loader2 className="h-5 w-5 mr-2 animate-spin" />
              Detectando fauna...
            </>
          ) : (
            <>
              <Play className="h-5 w-5 mr-2" />
              Ejecutar Detección
            </>
          )}
        </Button>
      </CardContent>
    </Card>
  )
}
