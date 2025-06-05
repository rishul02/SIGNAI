"use client"

import { useState } from "react"
import { Settings, Volume2, Brain } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"

interface SettingsPanelProps {
  speechEnabled: boolean
  onSpeechEnabledChange: (enabled: boolean) => void
}

export function SettingsPanel({ speechEnabled, onSpeechEnabledChange }: SettingsPanelProps) {
  const [speechRate, setSpeechRate] = useState([0.8])
  const [speechVolume, setSpeechVolume] = useState([0.7])
  const [confidenceThreshold, setConfidenceThreshold] = useState([0.7])
  const [language, setLanguage] = useState("en-US")

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="outline" size="icon">
          <Settings className="h-4 w-4" />
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
          <DialogDescription>Configure your SignSpeak AI preferences</DialogDescription>
        </DialogHeader>

        <div className="space-y-6">
          {/* Speech Settings */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Volume2 className="h-4 w-4" />
                Speech Output
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <Label htmlFor="speech-enabled">Enable Speech</Label>
                <Switch id="speech-enabled" checked={speechEnabled} onCheckedChange={onSpeechEnabledChange} />
              </div>

              <div className="space-y-2">
                <Label>Speech Rate: {speechRate[0]}</Label>
                <Slider
                  value={speechRate}
                  onValueChange={setSpeechRate}
                  max={2}
                  min={0.1}
                  step={0.1}
                  disabled={!speechEnabled}
                />
              </div>

              <div className="space-y-2">
                <Label>Volume: {speechVolume[0]}</Label>
                <Slider
                  value={speechVolume}
                  onValueChange={setSpeechVolume}
                  max={1}
                  min={0}
                  step={0.1}
                  disabled={!speechEnabled}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="language">Language</Label>
                <Select value={language} onValueChange={setLanguage}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="en-US">English (US)</SelectItem>
                    <SelectItem value="en-GB">English (UK)</SelectItem>
                    <SelectItem value="es-ES">Spanish</SelectItem>
                    <SelectItem value="fr-FR">French</SelectItem>
                    <SelectItem value="de-DE">German</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* AI Settings */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Brain className="h-4 w-4" />
                AI Recognition
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Confidence Threshold: {confidenceThreshold[0]}</Label>
                <Slider
                  value={confidenceThreshold}
                  onValueChange={setConfidenceThreshold}
                  max={1}
                  min={0.1}
                  step={0.05}
                />
              </div>
            </CardContent>
          </Card>
        </div>
      </DialogContent>
    </Dialog>
  )
}
