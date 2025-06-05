"use client"

import type React from "react"

import { useEffect, useRef, useState } from "react"
import * as tf from "@tensorflow/tfjs"

interface ASLModelIntegrationProps {
  onPrediction: (gesture: string, confidence: number) => void
  videoRef: React.RefObject<HTMLVideoElement>
  isActive: boolean
}

export function ASLModelIntegration({ onPrediction, videoRef, isActive }: ASLModelIntegrationProps) {
  const [models, setModels] = useState<{
    cnn: tf.LayersModel | null
    landmark: tf.LayersModel | null
  }>({
    cnn: null,
    landmark: null,
  })
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const processingRef = useRef(false)

  // ASL alphabet classes (matching your training data)
  const ASL_CLASSES = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
  ]

  useEffect(() => {
    loadModels()
  }, [])

  const loadModels = async () => {
    try {
      setIsLoading(true)
      setError(null)

      // Load TensorFlow.js models
      console.log("Loading ASL models...")

      // In production, these would be your actual trained models
      // For now, we'll create mock models with the correct structure
      const cnnModel = await createMockCNNModel()
      const landmarkModel = await createMockLandmarkModel()

      setModels({
        cnn: cnnModel,
        landmark: landmarkModel,
      })

      console.log("Models loaded successfully!")
    } catch (err) {
      console.error("Error loading models:", err)
      setError("Failed to load ASL models")
    } finally {
      setIsLoading(false)
    }
  }

  const createMockCNNModel = async () => {
    // Create a mock CNN model structure (replace with actual model loading)
    const model = tf.sequential({
      layers: [
        tf.layers.conv2d({
          inputShape: [224, 224, 3],
          filters: 32,
          kernelSize: 3,
          activation: "relu",
        }),
        tf.layers.maxPooling2d({ poolSize: 2 }),
        tf.layers.flatten(),
        tf.layers.dense({ units: 128, activation: "relu" }),
        tf.layers.dense({ units: ASL_CLASSES.length, activation: "softmax" }),
      ],
    })

    model.compile({
      optimizer: "adam",
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    })

    return model
  }

  const createMockLandmarkModel = async () => {
    // Create a mock landmark model structure
    const model = tf.sequential({
      layers: [
        tf.layers.dense({
          inputShape: [63], // 21 landmarks * 3 coordinates
          units: 128,
          activation: "relu",
        }),
        tf.layers.dropout({ rate: 0.3 }),
        tf.layers.dense({ units: 64, activation: "relu" }),
        tf.layers.dense({ units: ASL_CLASSES.length, activation: "softmax" }),
      ],
    })

    model.compile({
      optimizer: "adam",
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    })

    return model
  }

  const preprocessImage = (imageData: ImageData): tf.Tensor => {
    // Convert ImageData to tensor and preprocess
    const tensor = tf.browser.fromPixels(imageData).resizeNearestNeighbor([224, 224]).toFloat().div(255.0).expandDims(0)

    return tensor
  }

  const extractLandmarks = async (imageData: ImageData): Promise<number[] | null> => {
    // In a real implementation, this would use MediaPipe
    // For now, return mock landmarks
    const mockLandmarks = Array.from({ length: 63 }, () => Math.random())
    return mockLandmarks
  }

  const predictGesture = async () => {
    if (!videoRef.current || !models.cnn || !models.landmark || processingRef.current) {
      return
    }

    processingRef.current = true

    try {
      const video = videoRef.current
      const canvas = document.createElement("canvas")
      const ctx = canvas.getContext("2d")

      if (!ctx) return

      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      ctx.drawImage(video, 0, 0)

      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)

      // CNN prediction
      const imageTensor = preprocessImage(imageData)
      const cnnPrediction = models.cnn.predict(imageTensor) as tf.Tensor
      const cnnProbs = await cnnPrediction.data()

      // Landmark prediction
      const landmarks = await extractLandmarks(imageData)
      let landmarkProbs: Float32Array | null = null

      if (landmarks) {
        const landmarkTensor = tf.tensor2d([landmarks])
        const landmarkPrediction = models.landmark.predict(landmarkTensor) as tf.Tensor
        landmarkProbs = await landmarkPrediction.data()
        landmarkTensor.dispose()
      }

      // Ensemble prediction (combine CNN and landmark predictions)
      let finalProbs: Float32Array
      if (landmarkProbs) {
        finalProbs = new Float32Array(ASL_CLASSES.length)
        for (let i = 0; i < ASL_CLASSES.length; i++) {
          finalProbs[i] = 0.6 * cnnProbs[i] + 0.4 * landmarkProbs[i]
        }
      } else {
        finalProbs = cnnProbs as Float32Array
      }

      // Get best prediction
      const maxIndex = finalProbs.indexOf(Math.max(...finalProbs))
      const confidence = finalProbs[maxIndex]
      const gesture = ASL_CLASSES[maxIndex]

      // Only report high-confidence predictions
      if (confidence > 0.7) {
        onPrediction(gesture, confidence)
      }

      // Cleanup tensors
      imageTensor.dispose()
      cnnPrediction.dispose()
    } catch (err) {
      console.error("Prediction error:", err)
    } finally {
      processingRef.current = false
    }
  }

  useEffect(() => {
    if (!isActive || isLoading || error) return

    const interval = setInterval(predictGesture, 1000) // Predict every second
    return () => clearInterval(interval)
  }, [isActive, isLoading, error, models])

  if (isLoading) {
    return <div className="text-sm text-gray-600">Loading ASL models...</div>
  }

  if (error) {
    return <div className="text-sm text-red-600">Error: {error}</div>
  }

  return <div className="text-sm text-green-600">ASL models loaded and ready</div>
}
