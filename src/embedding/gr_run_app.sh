#!/bin/bash

# Arabic Text Embeddings Visualizer - Gradio Version Startup Script
echo "🤗 Arabic Text Embeddings Visualizer (Gradio)"
echo "=============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

Install/upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📋 Installing requirements..."
pip install -r requirements.txt

# Check if CUDA is available
echo "🔍 Checking for CUDA support..."
python3 -c "import torch; print('✅ CUDA available:', torch.cuda.is_available())"

# Download essential models (optional, will be downloaded on first use)
echo "🤖 Pre-downloading essential models (optional)..."
python3 -c "
try:
    from sentence_transformers import SentenceTransformer
    print('📥 Downloading multilingual model...')
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    print('✅ Model downloaded successfully!')
except Exception as e:
    print('⚠️ Model will be downloaded on first use:', str(e))
"

echo ""
echo "🚀 Starting Arabic Text Embeddings Visualizer (Gradio)..."
echo "🌐 The app will open in your browser at http://localhost:7860"
echo "📱 For mobile access, use your local IP address"
echo ""


# Run the Gradio app
python3 src/embedding/gradio_ar_emb.py