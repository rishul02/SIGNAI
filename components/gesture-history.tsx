"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Download, Trash2 } from "lucide-react"

interface GestureEntry {
  id: string
  gesture: string
  confidence: number
  timestamp: Date
}

interface GestureHistoryProps {
  history: GestureEntry[]
  onClear: () => void
  onExport: () => void
}

export function GestureHistory({ history, onClear, onExport }: GestureHistoryProps) {
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    })
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return "bg-green-100 text-green-800"
    if (confidence >= 0.7) return "bg-yellow-100 text-yellow-800"
    return "bg-red-100 text-red-800"
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Gesture History</CardTitle>
            <CardDescription>Complete log of recognized gestures with confidence scores</CardDescription>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={onExport} disabled={history.length === 0}>
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
            <Button variant="outline" size="sm" onClick={onClear} disabled={history.length === 0}>
              <Trash2 className="h-4 w-4 mr-2" />
              Clear
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {history.length > 0 ? (
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {history
              .slice()
              .reverse()
              .map((entry) => (
                <div key={entry.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <span className="font-medium">{entry.gesture}</span>
                    <Badge variant="secondary" className={getConfidenceColor(entry.confidence)}>
                      {(entry.confidence * 100).toFixed(1)}%
                    </Badge>
                  </div>
                  <span className="text-sm text-gray-500">{formatTime(entry.timestamp)}</span>
                </div>
              ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <p>No gestures recorded yet</p>
            <p className="text-sm">Start signing to build your history</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
