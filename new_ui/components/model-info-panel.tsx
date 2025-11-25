"use client"

import { useState } from "react"
import { ChevronDown, Info, Cpu, Database, BarChart } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"

const MODEL_METRICS = [
  { label: "F1-Score", value: "0.8405", icon: BarChart },
  { label: "Precision", value: "0.8407", icon: BarChart },
  { label: "Recall", value: "0.8404", icon: BarChart },
  { label: "MAE", value: "1.80", icon: BarChart },
]

const SPECIES = ["Buffalo", "Elephant", "Kob", "Topi", "Warthog", "Waterbuck"]

export function ModelInfoPanel() {
  const [isExpanded, setIsExpanded] = useState(false)

  return (
    <Card className="bg-card border-border">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-4 flex items-center justify-between text-left hover:bg-secondary/30 transition-colors rounded-lg"
      >
        <div className="flex items-center gap-2">
          <Info className="h-5 w-5 text-primary" />
          <span className="font-medium text-foreground">Información del Modelo</span>
        </div>
        <ChevronDown
          className={cn("h-5 w-5 text-muted-foreground transition-transform duration-200", isExpanded && "rotate-180")}
        />
      </button>

      {isExpanded && (
        <CardContent className="pt-0 pb-4 px-4">
          <div className="space-y-4">
            {/* Model Architecture */}
            <div className="flex items-start gap-3 p-3 rounded-lg bg-secondary/30">
              <Cpu className="h-5 w-5 text-primary mt-0.5" />
              <div>
                <p className="text-sm font-medium text-foreground">HerdNet (FPN + Density Maps)</p>
                <p className="text-xs text-muted-foreground">Última actualización: 07 Nov 2025</p>
              </div>
            </div>

            {/* Dataset */}
            <div className="flex items-start gap-3 p-3 rounded-lg bg-secondary/30">
              <Database className="h-5 w-5 text-accent mt-0.5" />
              <div>
                <p className="text-sm font-medium text-foreground">Dataset ULiège-AIR</p>
                <p className="text-xs text-muted-foreground">6 especies: {SPECIES.join(", ")}</p>
              </div>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 gap-2">
              {MODEL_METRICS.map((metric) => (
                <div key={metric.label} className="p-3 rounded-lg bg-secondary/30 text-center">
                  <p className="text-lg font-bold text-primary">{metric.value}</p>
                  <p className="text-xs text-muted-foreground">{metric.label}</p>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      )}
    </Card>
  )
}
