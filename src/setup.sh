

# RAG Pipeline Setup Script

echo "🚀 Setting up Arabic Educational RAG Pipeline..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p data/{uploads,processed,chunks,chroma_db}
mkdir -p logs
mkdir -p ssl

# Set permissions
chmod 755 data logs ssl
chmod -R 777 data/

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📄 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file and add your Gemini API key!"
fi

# Build and start services
echo "🏗️  Building Docker images..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🔍 Checking service health..."
if curl -f http://localhost:8000/api/v1/health &> /dev/null; then
    echo "✅ RAG Pipeline is running successfully!"
    echo "📖 API Documentation: http://localhost:8000/docs"
    echo "🔍 Health Check: http://localhost:8000/api/v1/health"
else
    echo "❌ Service health check failed. Check logs:"
    echo "docker-compose logs rag-pipeline"
fi

echo "🎉 Setup complete!"

