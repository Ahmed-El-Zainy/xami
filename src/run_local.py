
# run_local.py
"""
Local development runner for the RAG Pipeline
Run this instead of Docker for development
"""

import asyncio
import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'chromadb', 'sentence-transformers',
        'easyocr', 'google-generativeai', 'rank-bm25'
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
    
    return True

def setup_directories():
    """Create necessary directories"""
    directories = [
        'data/uploads',
        'data/processed', 
        'data/chunks',
        'data/chroma_db',
        'logs'
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
            import shutil
            shutil.copy(template_file, env_file)
            print("📄 Please edit .env file and add your Gemini API key!")
        else:
            print("❌ .env.example template not found!")
            return False
    
    return True

async def main():
    """Main setup and run function"""
    print("🚀 Starting Arabic Educational RAG Pipeline (Local Development)")
    print("=" * 60)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        return
    
    # Setup directories
    print("📁 Setting up directories...")
    setup_directories()
    
    # Check environment
    print("🔧 Checking environment configuration...")
    if not check_env_file():
        return
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("⚠️  python-dotenv not installed. Environment variables from shell will be used.")
    
    # Check Gemini API key
    gemini_key = os.getenv('GEMINI_API_KEY')
    if not gemini_key:
        print("❌ GEMINI_API_KEY not found in environment!")
        print("Please set it in your .env file or environment variables.")
        return
    
    print("✅ All checks passed!")
    print("🚀 Starting the application...")
    print("📖 API Documentation will be available at: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/api/v1/health")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    # Import and run the application
    try:
        from ocr.main import app
        import uvicorn
        
        # Run the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())

