# startup.py
"""
Simple startup script for the RAG Pipeline
"""
import asyncio
import subprocess
import sys
import os
from pathlib import Path
import shutil

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'sentence-transformers',
        'google-generativeai', 'rank-bm25', 'pydantic-settings'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def setup_directories():
    """Create necessary directories"""
    directories = [
        'data/uploads',
        'data/processed', 
        'data/chunks',
        'data/chroma_db',
        'logs',
        'temp_logs/requests',
        'temp_logs/metrics',
        'temp_logs/assets',
        'temp_logs/errors'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_file = Path('.env')
    
    if not env_file.exists():
        print("⚠️  .env file not found. Creating from template...")
        template_file = Path('.env.example')
        if template_file.exists():
            shutil.copy(template_file, env_file)
            print("📄 Please edit .env file and add your Gemini API key!")
        else:
            # Create a basic .env file
            with open('.env', 'w') as f:
                f.write("""# RAG Pipeline Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp
API_HOST=0.0.0.0
API_PORT=8000
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
VECTOR_SIZE=384
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MIN_CHUNK_SIZE=100
RERANK_TOP_K=20
FINAL_TOP_K=5
""")
            print("📄 Created basic .env file. Please add your Gemini API key!")
        return False
    
    return True

def check_vector_db():
    """Check which vector database to use"""
    try:
        import chromadb
        print("✅ ChromaDB available")
        return "chromadb"
    except ImportError:
        pass
    
    try:
        from qdrant_client import QdrantClient
        print("✅ Qdrant client available")
        return "qdrant"
    except ImportError:
        pass
    
    print("❌ No vector database found. Install ChromaDB or Qdrant:")
    print("  ChromaDB: pip install chromadb")
    print("  Qdrant: pip install qdrant-client")
    return None

def install_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        print("📥 Downloading NLTK data...")
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        print("✅ NLTK data downloaded")
    except ImportError:
        print("⚠️  NLTK not found. Some text processing features may be limited.")

def main():
    """Main startup function"""
    print("🚀 Starting Arabic Educational RAG Pipeline Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        return False
    
    # Setup directories
    print("\n📁 Setting up directories...")
    setup_directories()
    
    # Check environment
    print("\n🔧 Checking environment configuration...")
    env_ok = check_env_file()
    
    # Check vector database
    print("\n🗃️ Checking vector database...")
    vector_db = check_vector_db()
    if not vector_db:
        return False
    
    # Install NLTK data
    print("\n📚 Setting up NLTK data...")
    install_nltk_data()
    
    # Load environment variables
    print("\n🔑 Loading environment variables...")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment variables loaded")
    except ImportError:
        print("⚠️  python-dotenv not installed. Using system environment variables.")
    
    # Check Gemini API key
    gemini_key = os.getenv('GEMINI_API_KEY')
    if not gemini_key or gemini_key == 'your_gemini_api_key_here':
        print("❌ GEMINI_API_KEY not found or not set!")
        print("Please set it in your .env file")
        if not env_ok:
            return False
    else:
        print("✅ Gemini API key found")
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Ensure your Gemini API key is set in .env file")
    if vector_db == "qdrant":
        print("2. Start Qdrant server: docker run -p 6333:6333 qdrant/qdrant")
    print("3. Run the application: python main.py")
    print("4. Open http://localhost:8000/docs for API documentation")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please fix the issues above and try again.")
        sys.exit(1)
    
    # Ask if user wants to start the application
    response = input("\n🚀 Start the application now? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        print("\n🚀 Starting application...")
        try:
            import uvicorn
            from main import app, settings
            
            uvicorn.run(
                app,
                host=settings.API_HOST,
                port=settings.API_PORT,
                reload=True,
                log_level="info"
            )
        except KeyboardInterrupt:
            print("\n🛑 Application stopped by user")
        except Exception as e:
            print(f"\n❌ Error starting application: {str(e)}")
    else:
        print("\n✨ Setup complete. Run 'python main.py' when ready.")