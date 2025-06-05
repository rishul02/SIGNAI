#!/bin/bash

echo "🚀 Setting up SignSpeak AI Project..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    exit 1
fi

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python first."
    exit 1
fi

echo "✅ Node.js and Python found"

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Create Python virtual environment
echo "🐍 Setting up Python environment..."
python -m venv asl_env

# Activate virtual environment and install dependencies
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source asl_env/Scripts/activate
else
    # macOS/Linux
    source asl_env/bin/activate
fi

echo "📦 Installing Python dependencies..."
pip install tensorflow opencv-python mediapipe scikit-learn matplotlib pandas numpy
pip install tensorflowjs kaggle joblib

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p python_scripts
mkdir -p public/models
mkdir -p asl_dataset

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Run 'npm run dev' to start the Next.js development server"
echo "2. Set up Kaggle API credentials for dataset download"
echo "3. Download the ASL dataset using the Python scripts"
echo "4. Train your models"
echo ""
echo "🌐 The web app will be available at http://localhost:3000"
