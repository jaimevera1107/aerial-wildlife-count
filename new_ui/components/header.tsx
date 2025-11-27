import { Eye } from "lucide-react"

interface HeaderProps {
  modelStatus?: "online" | "offline" | "checking"
}

export function Header({ modelStatus = "checking" }: HeaderProps) {
  const statusConfig = {
    online: { color: "bg-accent", text: "Modelo activo" },
    offline: { color: "bg-destructive", text: "Modelo offline" },
    checking: { color: "bg-muted-foreground", text: "Verificando..." },
  }

  const status = statusConfig[modelStatus]

  return (
    <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4 max-w-7xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary/10 border border-primary/20">
              <Eye className="h-6 w-6 text-primary" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-foreground">Wildlife Vision</h1>
              <p className="text-xs text-muted-foreground">wildlife.vision</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="hidden sm:flex items-center gap-2 text-sm text-muted-foreground">
              <span
                className={`inline-block w-2 h-2 rounded-full ${status.color} ${modelStatus === "online" ? "animate-pulse" : ""}`}
              />
              <span>{status.text}</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}
