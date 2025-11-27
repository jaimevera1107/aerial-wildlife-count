"use client"

import { Download, Target, BarChart3 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { DetectionResult } from "@/app/page"

interface ResultsPanelProps {
  results: DetectionResult | null
  isProcessing: boolean
}

const SPECIES_COLORS: Record<string, string> = {
  Buffalo: "bg-chart-1",
  Elephant: "bg-chart-2",
  Kob: "bg-chart-3",
  Topi: "bg-chart-4",
  Warthog: "bg-chart-5",
  Waterbuck: "bg-chart-6",
}

const SPECIES_ICONS: Record<string, string> = {
  Buffalo: "🦬",
  Elephant: "🐘",
  Kob: "🦌",
  Topi: "🫎",
  Warthog: "🐗",
  Waterbuck: "🦌",
}

export function ResultsPanel({ results, isProcessing }: ResultsPanelProps) {
  const handleDownloadCSV = () => {
    if (!results) return

    const csvContent = [
      "Especie,Conteo",
      ...Object.entries(results.counts).map(([species, count]) => `${species},${count}`),
      `Total,${results.totalCount}`,
    ].join("\n")

    const blob = new Blob([csvContent], { type: "text/csv" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "detecciones_fauna.csv"
    a.click()
    URL.revokeObjectURL(url)
  }

  if (!results && !isProcessing) {
    return (
      <Card className="bg-card border-border h-full">
        <CardContent className="p-6 h-full flex items-center justify-center min-h-96">
          <div className="text-center">
            <div className="p-4 rounded-full bg-secondary inline-block mb-4">
              <Target className="h-8 w-8 text-muted-foreground" />
            </div>
            <h3 className="text-lg font-medium text-foreground mb-2">Sin resultados aún</h3>
            <p className="text-sm text-muted-foreground max-w-xs">
              Sube una imagen aérea y ejecuta la detección para ver los resultados del análisis
            </p>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (isProcessing) {
    return (
      <Card className="bg-card border-border h-full">
        <CardContent className="p-6 h-full flex items-center justify-center min-h-96">
          <div className="text-center">
            <div className="relative mb-4">
              <div className="w-16 h-16 rounded-full border-4 border-secondary border-t-primary animate-spin" />
            </div>
            <h3 className="text-lg font-medium text-foreground mb-2">Analizando imagen...</h3>
            <p className="text-sm text-muted-foreground">El modelo HerdNet está procesando tu imagen</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  const maxCount = Math.max(...Object.values(results!.counts), 1)

  return (
    <div className="space-y-6">
      {/* Annotated Image */}
      <Card className="bg-card border-border">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-lg">
              <Target className="h-5 w-5 text-primary" />
              Detecciones
            </CardTitle>
            <span className="text-sm text-muted-foreground">{results!.totalCount} animales detectados</span>
          </div>
        </CardHeader>
        <CardContent>
          <div className="rounded-xl overflow-hidden border border-border">
            <img
              src={results!.annotatedImage! || "/placeholder.svg"}
              alt="Imagen con detecciones"
              className="w-full h-auto max-h-72 object-contain bg-secondary/30"
            />
          </div>
        </CardContent>
      </Card>

      {/* Species Count */}
      <Card className="bg-card border-border">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-lg">
              <BarChart3 className="h-5 w-5 text-primary" />
              Conteo por Especie
            </CardTitle>
            <Button variant="outline" size="sm" onClick={handleDownloadCSV}>
              <Download className="h-4 w-4 mr-2" />
              CSV
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {Object.entries(results!.counts).map(([species, count]) => (
              <div key={species} className="flex items-center gap-3">
                <span className="text-xl w-8">{SPECIES_ICONS[species] || "🐾"}</span>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-foreground">{species}</span>
                    <span className="text-sm font-bold text-primary">{count}</span>
                  </div>
                  <div className="h-2 bg-secondary rounded-full overflow-hidden">
                    <div
                      className={`h-full ${SPECIES_COLORS[species] || "bg-primary"} rounded-full transition-all duration-500`}
                      style={{ width: `${(count / maxCount) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Total */}
          <div className="mt-4 pt-4 border-t border-border flex items-center justify-between">
            <span className="text-sm font-medium text-muted-foreground">Total detectado</span>
            <span className="text-2xl font-bold text-primary">{results!.totalCount}</span>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
