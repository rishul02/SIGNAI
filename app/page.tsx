"use client"

import { useState, useRef, useEffect } from "react"
import { Camera, Volume2, VolumeX, Info } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"

export default function SignSpeakAI() {
  const [isRecording, setIsRecording] = useState(false)
  const [isSpeechEnabled, setIsSpeechEnabled] = useState(true)
  const [recognizedText, setRecognizedText] = useState("")
  const [confidence, setConfidence] = useState(0)
  const [currentGesture, setCurrentGesture] = useState("")
  const [gestureHistory, setGestureHistory] = useState<string[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [cameraError, setCameraError] = useState("")

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)

  // Initialize camera
  const startCamera = async () => {
    try {
      setCameraError("")
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream
      }
      setIsRecording(true)
    } catch (error) {
      setCameraError("Camera access denied. Please allow camera permissions.")
      console.error("Camera error:", error)
    }
  }

  // Stop camera
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    setIsRecording(false)
  }

  // Simulate gesture recognition (in real implementation, this would use MediaPipe/TensorFlow)
  const processGesture = async () => {
    if (!isRecording || isProcessing) return

    setIsProcessing(true)

    // Simulate AI processing delay
    await new Promise((resolve) => setTimeout(resolve, 500))

    // Mock gesture recognition results
    const mockGestures = ["Hello", "Thank you", "Please", "Yes", "No", "Help", "Good", "Bad"]
    const randomGesture = mockGestures[Math.floor(Math.random() * mockGestures.length)]
    const mockConfidence = Math.random() * 0.4 + 0.6 // 60-100% confidence

    setCurrentGesture(randomGesture)
    setConfidence(mockConfidence)

    if (mockConfidence > 0.7) {
      setRecognizedText((prev) => prev + (prev ? " " : "") + randomGesture)
      setGestureHistory((prev) => [...prev.slice(-9), randomGesture])

      // Text-to-speech
      if (isSpeechEnabled && "speechSynthesis" in window) {
        const utterance = new SpeechSynthesisUtterance(randomGesture)
        utterance.rate = 0.8
        utterance.volume = 0.7
        speechSynthesis.speak(utterance)
      }
    }

    setIsProcessing(false)
  }

  // Auto-process gestures when recording
  useEffect(() => {
    let interval: NodeJS.Timeout
    if (isRecording) {
      interval = setInterval(processGesture, 2000)
    }
    return () => clearInterval(interval)
  }, [isRecording, isProcessing, isSpeechEnabled])

  const clearText = () => {
    setRecognizedText("")
    setGestureHistory([])
    setCurrentGesture("")
    setConfidence(0)
  }

  const speakText = () => {
    if (recognizedText && "speechSynthesis" in window) {
      const utterance = new SpeechSynthesisUtterance(recognizedText)
      utterance.rate = 0.8
      utterance.volume = 0.7
      speechSynthesis.speak(utterance)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold text-gray-900">SignSpeak AI</h1>
          <p className="text-lg text-gray-600">Real-Time Sign Language to Speech/Text Interpreter</p>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Camera Feed */}
          <Card className="overflow-hidden">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Camera className="h-5 w-5" />
                Camera Feed
              </CardTitle>
              <CardDescription>Position your hands clearly in view for optimal recognition</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {cameraError && (
                <Alert>
                  <Info className="h-4 w-4" />
                  <AlertDescription>{cameraError}</AlertDescription>
                </Alert>
              )}

              <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
                <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
                <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />

                {/* Overlay indicators */}
                {isRecording && (
                  <div className="absolute top-4 left-4 flex items-center gap-2">
                    <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
                    <span className="text-white text-sm font-medium">Recording</span>
                  </div>
                )}

                {currentGesture && (
                  <div className="absolute bottom-4 left-4 bg-black/70 text-white px-3 py-2 rounded-lg">
                    <div className="text-sm font-medium">{currentGesture}</div>
                    <div className="text-xs opacity-75">Confidence: {(confidence * 100).toFixed(1)}%</div>
                  </div>
                )}
              </div>

              <div className="flex gap-2">
                <Button
                  onClick={isRecording ? stopCamera : startCamera}
                  variant={isRecording ? "destructive" : "default"}
                  className="flex-1"
                >
                  <Camera className="h-4 w-4 mr-2" />
                  {isRecording ? "Stop Camera" : "Start Camera"}
                </Button>

                <Button onClick={() => setIsSpeechEnabled(!isSpeechEnabled)} variant="outline" size="icon">
                  {isSpeechEnabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Recognition Results */}
          <div className="space-y-6">
            {/* Current Recognition */}
            <Card>
              <CardHeader>
                <CardTitle>Recognition Output</CardTitle>
                <CardDescription>Interpreted text from your sign language gestures</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="min-h-[120px] p-4 bg-gray-50 rounded-lg border-2 border-dashed border-gray-200">
                  {recognizedText ? (
                    <p className="text-lg leading-relaxed text-gray-900">{recognizedText}</p>
                  ) : (
                    <p className="text-gray-500 italic">Start signing to see recognized text here...</p>
                  )}
                </div>

                <div className="flex gap-2">
                  <Button onClick={speakText} disabled={!recognizedText} className="flex-1">
                    <Volume2 className="h-4 w-4 mr-2" />
                    Speak Text
                  </Button>
                  <Button onClick={clearText} variant="outline" disabled={!recognizedText}>
                    Clear
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Gesture History */}
            <Card>
              <CardHeader>
                <CardTitle>Recent Gestures</CardTitle>
                <CardDescription>Last 10 recognized gestures</CardDescription>
              </CardHeader>
              <CardContent>
                {gestureHistory.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {gestureHistory.map((gesture, index) => (
                      <Badge key={index} variant="secondary">
                        {gesture}
                      </Badge>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-500 italic">No gestures recognized yet</p>
                )}
              </CardContent>
            </Card>

            {/* Status */}
            <Card>
              <CardHeader>
                <CardTitle>System Status</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Camera Status</span>
                  <Badge variant={isRecording ? "default" : "secondary"}>{isRecording ? "Active" : "Inactive"}</Badge>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Speech Output</span>
                  <Badge variant={isSpeechEnabled ? "default" : "secondary"}>
                    {isSpeechEnabled ? "Enabled" : "Disabled"}
                  </Badge>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">AI Processing</span>
                  <Badge variant={isProcessing ? "default" : "secondary"}>
                    {isProcessing ? "Processing" : "Ready"}
                  </Badge>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Instructions */}
        <Card>
          <CardHeader>
            <CardTitle>How to Use SignSpeak AI</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center space-y-2">
                <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto">
                  <span className="text-blue-600 font-bold">1</span>
                </div>
                <h3 className="font-medium">Start Camera</h3>
                <p className="text-sm text-gray-600">Click "Start Camera" to begin capturing your gestures</p>
              </div>

              <div className="text-center space-y-2">
                <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto">
                  <span className="text-blue-600 font-bold">2</span>
                </div>
                <h3 className="font-medium">Sign Clearly</h3>
                <p className="text-sm text-gray-600">Make clear gestures within the camera frame</p>
              </div>

              <div className="text-center space-y-2">
                <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto">
                  <span className="text-blue-600 font-bold">3</span>
                </div>
                <h3 className="font-medium">Get Results</h3>
                <p className="text-sm text-gray-600">View recognized text and hear speech output</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
