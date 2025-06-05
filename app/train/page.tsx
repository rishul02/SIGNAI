"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Download, Upload, Brain, Database, CheckCircle, AlertCircle } from "lucide-react"

export default function TrainingPage() {
  const [trainingStep, setTrainingStep] = useState(0)
  const [progress, setProgress] = useState(0)
  const [isTraining, setIsTraining] = useState(false)

  const trainingSteps = [
    {
      title: "Download Dataset",
      description: "Download ASL Alphabet dataset from Kaggle",
      icon: Download,
      status: "pending",
    },
    {
      title: "Preprocess Data",
      description: "Extract hand landmarks and prepare images",
      icon: Database,
      status: "pending",
    },
    {
      title: "Train Models",
      description: "Train CNN and landmark-based models",
      icon: Brain,
      status: "pending",
    },
    {
      title: "Convert Models",
      description: "Convert to TensorFlow.js format",
      icon: Upload,
      status: "pending",
    },
  ]

  const startTraining = async () => {
    setIsTraining(true)
    setProgress(0)

    // Simulate training process
    for (let step = 0; step < trainingSteps.length; step++) {
      setTrainingStep(step)

      // Simulate step progress
      for (let i = 0; i <= 100; i += 10) {
        setProgress((step * 100 + i) / trainingSteps.length)
        await new Promise((resolve) => setTimeout(resolve, 200))
      }
    }

    setIsTraining(false)
    setProgress(100)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold text-gray-900">ASL Model Training</h1>
          <p className="text-lg text-gray-600">Train your custom ASL recognition models with Kaggle dataset</p>
        </div>

        {/* Dataset Information */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              Dataset Information
            </CardTitle>
            <CardDescription>ASL Alphabet Dataset from Kaggle</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center p-4 bg-blue-50 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">87,000+</div>
                <div className="text-sm text-gray-600">Training Images</div>
              </div>
              <div className="text-center p-4 bg-green-50 rounded-lg">
                <div className="text-2xl font-bold text-green-600">29</div>
                <div className="text-sm text-gray-600">Classes (A-Z + 3 special)</div>
              </div>
              <div className="text-center p-4 bg-purple-50 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">200x200</div>
                <div className="text-sm text-gray-600">Image Resolution</div>
              </div>
            </div>

            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                To get started, you'll need to download the ASL Alphabet dataset from Kaggle. Make sure you have a
                Kaggle account and API key set up.
              </AlertDescription>
            </Alert>
          </CardContent>
        </Card>

        {/* Training Steps */}
        <Card>
          <CardHeader>
            <CardTitle>Training Pipeline</CardTitle>
            <CardDescription>Follow these steps to train your ASL recognition models</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {trainingSteps.map((step, index) => {
              const Icon = step.icon
              const isActive = index === trainingStep && isTraining
              const isCompleted = index < trainingStep || (!isTraining && progress === 100)

              return (
                <div key={index} className="flex items-center gap-4 p-4 border rounded-lg">
                  <div
                    className={`w-10 h-10 rounded-full flex items-center justify-center ${
                      isCompleted
                        ? "bg-green-100 text-green-600"
                        : isActive
                          ? "bg-blue-100 text-blue-600"
                          : "bg-gray-100 text-gray-400"
                    }`}
                  >
                    {isCompleted ? <CheckCircle className="h-5 w-5" /> : <Icon className="h-5 w-5" />}
                  </div>
                  <div className="flex-1">
                    <h3 className="font-medium">{step.title}</h3>
                    <p className="text-sm text-gray-600">{step.description}</p>
                  </div>
                  <Badge variant={isCompleted ? "default" : isActive ? "secondary" : "outline"}>
                    {isCompleted ? "Completed" : isActive ? "In Progress" : "Pending"}
                  </Badge>
                </div>
              )
            })}
          </CardContent>
        </Card>

        {/* Training Progress */}
        {isTraining && (
          <Card>
            <CardHeader>
              <CardTitle>Training Progress</CardTitle>
              <CardDescription>Current step: {trainingSteps[trainingStep]?.title}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Overall Progress</span>
                  <span>{Math.round(progress)}%</span>
                </div>
                <Progress value={progress} className="w-full" />
              </div>
            </CardContent>
          </Card>
        )}

        {/* Training Controls */}
        <Card>
          <CardHeader>
            <CardTitle>Training Controls</CardTitle>
            <CardDescription>Start the training process or download pre-trained models</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-4">
              <Button onClick={startTraining} disabled={isTraining} className="flex-1">
                <Brain className="h-4 w-4 mr-2" />
                {isTraining ? "Training in Progress..." : "Start Training"}
              </Button>
              <Button variant="outline" className="flex-1">
                <Download className="h-4 w-4 mr-2" />
                Download Pre-trained Models
              </Button>
            </div>

            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                Training can take several hours depending on your hardware. Make sure you have a GPU available for
                faster training.
              </AlertDescription>
            </Alert>
          </CardContent>
        </Card>

        {/* Code Examples */}
        <Card>
          <CardHeader>
            <CardTitle>Setup Instructions</CardTitle>
            <CardDescription>Commands to set up your training environment</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <h4 className="font-medium mb-2">1. Install Dependencies</h4>
                <div className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm font-mono">
                  pip install tensorflow opencv-python mediapipe scikit-learn matplotlib kaggle tensorflowjs
                </div>
              </div>

              <div>
                <h4 className="font-medium mb-2">2. Download Dataset</h4>
                <div className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm font-mono">
                  kaggle datasets download -d grassknoted/asl-alphabet
                  <br />
                  unzip asl-alphabet.zip -d asl_dataset/
                </div>
              </div>

              <div>
                <h4 className="font-medium mb-2">3. Run Training Script</h4>
                <div className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm font-mono">
                  python train_asl_model.py
                </div>
              </div>

              <div>
                <h4 className="font-medium mb-2">4. Convert for Web</h4>
                <div className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm font-mono">
                  python model_converter.py
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
