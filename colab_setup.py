#!/usr/bin/env python3
"""
Automated setup script for Arabic Educational RAG Pipeline in Google Colab
Uses UV for fast package management and handles DotsOCR model download
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path
import shutil
import json
import requests
from typing import Dict, List, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ColabSetupManager:
    """Manages the complete setup process for Colab environment"""
    
    def __init__(self):
        self.project_dir = Path("/content/xami")
        self.requirements = [
            "fastapi==0.104.1",
            "uvicorn[standard]==0.24.0",
            "python-multipart==0.0.6",
            "aiofiles==23.2.0",
            "sentence-transformers==2.2.2",
            "transformers==4.35.2",
            "torch==2.1.1",
            "torchvision==0.16.1",
            "torchaudio==2.1.1",
            "google-generativeai==0.3.2",
            "rank-bm25==0.2.2",
            "pyarabic==0.6.15",
            "nltk==3.8.1",
            "python-dotenv==1.0.0",
            "pydantic==2.5.0",
            "pydantic-settings==2.1.0",
            "huggingface-hub==0.19.4",
            "qwen-vl-utils",
            "pdf2image==1.16.3",
            "pillow==10.1.0",
            "opencv-python==4.8.1.78",
            "PyPDF2==3.0.1",
            "chromadb==0.4.18",
            "numpy==1.24.3",
            "requests==2.31.0",
            "colorlog==6.8.0",
            "tqdm==4.66.1",
            "pyyaml==6.0.1"
        ]
        self.github_repo = "https://github.com/your-username/xami.git"  # Replace with actual repo
        
    def install_uv(self) -> bool:
        """Install UV package manager"""
        try:
            logger.info("🚀 Installing UV package manager...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "uv"
            ], check=True, capture_output=True)
            
            # Verify installation
            result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ UV installed successfully: {result.stdout.strip()}")
                return True
            else:
                logger.error("❌ UV installation verification failed")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install UV: {e}")
            return False
    
    def setup_project_structure(self) -> bool:
        """Create project directory structure"""
        try:
            logger.info("📁 Setting up project structure...")
            
            directories = [
                "src/api",
                "src/config", 
                "src/models",
                "src/services",
                "src/utilities",
                "src/tests",
                "src/weights/DotsOCR",
                "data/uploads",
                "data/processed",
                "data/chunks",
                "data/chroma_db",
                "logs",
                "temp_logs/requests",
                "temp_logs/metrics", 
                "temp_logs/assets",
                "temp_logs/errors",
                "logger"
            ]
            
            for directory in directories:
                dir_path = self.project_dir / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                
                # Create __init__.py files for Python packages
                if directory.startswith("src/") and not directory.endswith("weights/DotsOCR"):
                    (dir_path / "__init__.py").touch()
            
            logger.info("✅ Project structure created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create project structure: {e}")
            return False
    
    def install_dependencies_with_uv(self) -> bool:
        """Install dependencies using UV"""
        try:
            logger.info("📦 Installing dependencies with UV...")
            
            # Change to project directory
            os.chdir(self.project_dir)
            
            # Initialize UV project
            subprocess.run(["uv", "init", "--no-readme"], check=True, capture_output=True)
            
            # Install dependencies
            for package in self.requirements:
                logger.info(f"Installing {package}...")
                result = subprocess.run([
                    "uv", "add", package
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.warning(f"⚠️ Failed to install {package} with UV, trying pip...")
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", package
                    ], check=True)
            
            logger.info("✅ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    
    def download_system_files(self) -> bool:
        """Download system files from GitHub or create them"""
        try:
            logger.info("📥 Setting up system files...")
            
            # Create essential files
            files_content = {
                ".env": self._get_env_content(),
                "main.py": self._get_main_content(),
                "requirements.txt": "\n".join(self.requirements),
                "src/config/config_settings.py": self._get_settings_content(),
                "src/config/logging_config.py": self._get_logging_config_content(),
                "src/models/schemas.py": self._get_schemas_content(),
                "pyproject.toml": self._get_pyproject_content()
            }
            
            for file_path, content in files_content.items():
                full_path = self.project_dir / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Created {file_path}")
            
            logger.info("✅ System files created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create system files: {e}")
            return False
    
    async def download_dotsocr_model(self) -> bool:
        """Download DotsOCR model using huggingface_hub"""
        try:
            logger.info("🤖 Downloading DotsOCR model...")
            
            from huggingface_hub import snapshot_download
            
            model_dir = self.project_dir / "src/weights/DotsOCR"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Download model
            snapshot_download(
                repo_id="rednote-hilab/dots.ocr",
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            # Create additional configuration files
            await self._create_dotsocr_config_files(model_dir)
            
            logger.info("✅ DotsOCR model downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download DotsOCR model: {e}")
            return False
    
    async def _create_dotsocr_config_files(self, model_dir: Path):
        """Create additional configuration files for DotsOCR"""
        try:
            # Configuration files content
            config_files = {
                "configuration_dots.py": self._get_dotsocr_config(),
                "modeling_dots_ocr.py": self._get_dotsocr_modeling(),
                "modeling_dots_vision.py": self._get_dotsocr_vision()
            }
            
            for filename, content in config_files.items():
                file_path = model_dir / filename
                if not file_path.exists():
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.info(f"Created {filename}")
                    
        except Exception as e:
            logger.error(f"Error creating DotsOCR config files: {e}")
    
    def setup_colab_environment(self) -> bool:
        """Setup Colab specific environment"""
        try:
            logger.info("🔧 Setting up Colab environment...")
            
            # Install system dependencies
            system_packages = [
                "poppler-utils",
                "libgl1-mesa-glx", 
                "libglib2.0-0"
            ]
            
            for package in system_packages:
                subprocess.run([
                    "apt-get", "install", "-y", package
                ], check=True, capture_output=True)
            
            # Set environment variables
            os.environ["PYTHONPATH"] = str(self.project_dir)
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            # Download NLTK data
            import nltk
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            
            logger.info("✅ Colab environment setup completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Colab environment: {e}")
            return False
    
    def verify_installation(self) -> bool:
        """Verify that everything is installed correctly"""
        try:
            logger.info("🔍 Verifying installation...")
            
            # Check Python path
            sys.path.insert(0, str(self.project_dir))
            
            # Test imports
            test_imports = [
                "fastapi",
                "transformers", 
                "torch",
                "sentence_transformers",
                "google.generativeai",
                "chromadb",
                "huggingface_hub"
            ]
            
            for module in test_imports:
                try:
                    __import__(module)
                    logger.info(f"✅ {module} imported successfully")
                except ImportError as e:
                    logger.error(f"❌ Failed to import {module}: {e}")
                    return False
            
            # Check model directory
            model_dir = self.project_dir / "src/weights/DotsOCR"
            if not model_dir.exists() or not any(model_dir.iterdir()):
                logger.error("❌ DotsOCR model directory is empty")
                return False
            
            logger.info("✅ Installation verification completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Installation verification failed: {e}")
            return False
    
    def create_startup_script(self) -> bool:
        """Create a startup script for easy launching"""
        try:
            startup_script = self.project_dir / "start_rag_pipeline.py"
            
            startup_content = '''#!/usr/bin/env python3
"""
Startup script for Arabic Educational RAG Pipeline in Colab
"""
import sys
import os
from pathlib import Path
import asyncio

# Add project to Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

async def main():
    """Main startup function"""
    print("🚀 Starting Arabic Educational RAG Pipeline...")
    
    # Set environment variables
    os.environ["PYTHONPATH"] = str(project_dir)
    
    try:
        # Import and start the application
        from main import app
        from src.config.config_settings import settings
        import uvicorn
        
        print(f"📖 API Documentation: http://localhost:{settings.API_PORT}/docs")
        print(f"🔍 Health Check: http://localhost:{settings.API_PORT}/api/v1/health")
        
        # Start the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=settings.API_PORT,
            reload=False,
            log_level="info"
        )
        
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(main())
'''
            
            with open(startup_script, 'w') as f:
                f.write(startup_content)
            
            # Make executable
            os.chmod(startup_script, 0o755)
            
            logger.info("✅ Startup script created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create startup script: {e}")
            return False
    
    # Content generation methods
    def _get_env_content(self) -> str:
        return '''# Arabic Educational RAG Pipeline Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false

# Vector Database (ChromaDB)
CHROMA_PERSIST_DIR=data/chroma_db
CHROMA_COLLECTION_NAME=arabic_educational_content
VECTOR_SIZE=384

# Embedding Configuration
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# OCR Configuration
OCR_LANGUAGES=["ar", "en"]
OCR_GPU=true
DOTS_OCR_MODEL_PATH=src/weights/DotsOCR

# Text Processing
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MIN_CHUNK_SIZE=100

# Reranking
RERANK_TOP_K=20
FINAL_TOP_K=5

# File Paths
UPLOAD_DIR=data/uploads
PROCESSED_DIR=data/processed
CHUNKS_DIR=data/chunks
LOGS_DIR=logs

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
'''
    
    def _get_main_content(self) -> str:
        return '''import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import sys
import os

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config.config_settings import settings
from src.config.logging_config import get_logger

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    try:
        logger.info("Starting RAG Pipeline application...")
        logger.info("RAG Pipeline application started successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise
    finally:
        logger.info("Shutting down RAG Pipeline application...")

# Create FastAPI application
app = FastAPI(
    title="Arabic Educational RAG Pipeline",
    description="A complete RAG pipeline for Arabic educational content.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Arabic Educational RAG Pipeline API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time()
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level="info"
    )
'''
    
    def _get_settings_content(self) -> str:
        return '''import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = False
    
    # Gemini API
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash-exp"
    
    # Vector Database (ChromaDB)
    CHROMA_PERSIST_DIR: str = "data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "arabic_educational_content"
    VECTOR_SIZE: int = 384
    
    # Embedding Models
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # OCR Configuration
    OCR_LANGUAGES: List[str] = ["ar", "en"]
    OCR_GPU: bool = True
    DOTS_OCR_MODEL_PATH: str = "src/weights/DotsOCR"
    
    # Text Processing
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MIN_CHUNK_SIZE: int = 100
    
    # Reranking
    RERANK_TOP_K: int = 20
    FINAL_TOP_K: int = 5
    
    # File Paths
    UPLOAD_DIR: str = "data/uploads"
    PROCESSED_DIR: str = "data/processed"
    CHUNKS_DIR: str = "data/chunks"
    LOGS_DIR: str = "logs"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Ensure directories exist
for directory in [settings.UPLOAD_DIR, settings.PROCESSED_DIR, 
                 settings.CHUNKS_DIR, settings.LOGS_DIR, settings.CHROMA_PERSIST_DIR]:
    os.makedirs(directory, exist_ok=True)
'''
    
    def _get_logging_config_content(self) -> str:
        return '''import logging
import colorlog
import os
from datetime import datetime

class CustomLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        # Console handler with colors
        console_handler = colorlog.StreamHandler()
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def get_logger(self):
        return self.logger

def get_logger(name: str) -> logging.Logger:
    return CustomLogger(name).get_logger()

def log_execution_time(logger):
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator
'''
    
    def _get_schemas_content(self) -> str:
        return '''from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class OCRResult(BaseModel):
    text: str
    confidence: float
    processing_time: float
    language_detected: str
    page_count: int

class DocumentUpload(BaseModel):
    filename: str
    content_type: str = "application/pdf"

class TextChunk(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]
    chunk_index: int
    source_file: str
    page_number: Optional[int] = None
    
class EmbeddingResult(BaseModel):
    chunk_id: str
    embedding: List[float]
    text: str
    metadata: Dict[str, Any]

class SearchResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    rerank_score: Optional[float] = None

class RAGResponse(BaseModel):
    query: str
    answer: str
    sources: List[SearchResult]
    processing_time: float
    confidence_score: float

class ProcessingStatus(BaseModel):
    status: str
    message: str
    progress: float
    timestamp: datetime
    file_id: str

class DocumentInfo(BaseModel):
    id: str
    filename: str
    upload_time: datetime
    processing_status: str
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    file_size: int
    language: Optional[str] = None

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, bool]
'''
    
    def _get_pyproject_content(self) -> str:
        return '''[project]
name = "arabic-rag-pipeline"
version = "1.0.0"
description = "Arabic Educational RAG Pipeline"
readme = "README.md"
requires-python = ">=3.8"
dependencies = [
    "fastapi>=0.104.1",
    "uvicorn[standard]>=0.24.0",
    "python-multipart>=0.0.6",
    "aiofiles>=23.2.0",
    "sentence-transformers>=2.2.2",
    "transformers>=4.35.2",
    "torch>=2.1.1",
    "google-generativeai>=0.3.2",
    "rank-bm25>=0.2.2",
    "pyarabic>=0.6.15",
    "nltk>=3.8.1",
    "python-dotenv>=1.0.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "huggingface-hub>=0.19.4",
    "pdf2image>=1.16.3",
    "pillow>=10.1.0",
    "opencv-python>=4.8.1.78",
    "PyPDF2>=3.0.1",
    "chromadb>=0.4.18",
    "numpy>=1.24.3",
    "requests>=2.31.0",
    "colorlog>=6.8.0",
    "tqdm>=4.66.1",
    "pyyaml>=6.0.1"
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
'''
    
    def _get_dotsocr_config(self) -> str:
        return '''# DotsOCR Configuration
from typing import Any, Optional
from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2 import Qwen2Config

class DotsVisionConfig(PretrainedConfig):
    model_type: str = "dots_vit"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add vision config parameters here

class DotsOCRConfig(Qwen2Config):
    model_type = "dots_ocr"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add OCR config parameters here
'''
    
    def _get_dotsocr_modeling(self) -> str:
        return '''# DotsOCR Model Implementation
from transformers.models.qwen2 import Qwen2ForCausalLM

class DotsOCRForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # Add DotsOCR specific implementation
'''
    
    def _get_dotsocr_vision(self) -> str:
        return '''# DotsOCR Vision Transformer Implementation
import torch
import torch.nn as nn

class DotsVisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Add vision transformer implementation
        
    def forward(self, pixel_values, grid_thw):
        # Add forward pass implementation
        pass
'''

async def main():
    """Main setup function for Colab"""
    print("🚀 Starting Arabic Educational RAG Pipeline Setup for Colab")
    print("=" * 70)
    
    setup_manager = ColabSetupManager()
    
    # Step 1: Install UV
    print("\n📦 Step 1: Installing UV package manager...")
    if not setup_manager.install_uv():
        print("❌ Setup failed at UV installation")
        return False
    
    # Step 2: Create project structure
    print("\n📁 Step 2: Creating project structure...")
    if not setup_manager.setup_project_structure():
        print("❌ Setup failed at project structure creation")
        return False
    
    # Step 3: Install dependencies
    print("\n🔧 Step 3: Installing dependencies...")
    if not setup_manager.install_dependencies_with_uv():
        print("❌ Setup failed at dependency installation")
        return False
    
    # Step 4: Setup Colab environment
    print("\n🌐 Step 4: Setting up Colab environment...")
    if not setup_manager.setup_colab_environment():
        print("❌ Setup failed at Colab environment setup")
        return False
    
    # Step 5: Download system files
    print("\n📥 Step 5: Creating system files...")
    if not setup_manager.download_system_files():
        print("❌ Setup failed at system files creation")
        return False
    
    # Step 6: Download DotsOCR model
    print("\n🤖 Step 6: Downloading DotsOCR model...")
    if not await setup_manager.download_dotsocr_model():
        print("❌ Setup failed at DotsOCR model download")
        return False
    
    # Step 7: Create startup script
    print("\n🚀 Step 7: Creating startup script...")
    if not setup_manager.create_startup_script():
        print("❌ Setup failed at startup script creation")
        return False
    
    # Step 8: Verify installation
    print("\n🔍 Step 8: Verifying installation...")
    if not setup_manager.verify_installation():
        print("❌ Setup failed at installation verification")
        return False
    
    print("\n" + "=" * 70)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Set your Gemini API key in /content/xami/.env")
    print("2. Run: cd /content/xami && python start_rag_pipeline.py")
    print("3. Open the API docs at: http://localhost:8000/docs")
    print("\n✨ Your Arabic Educational RAG Pipeline is ready!")
    
    return True


def run_colab_setup():
    """Entry point for Colab setup"""
    import asyncio
    return asyncio.run(main())


if __name__ == "__main__":
    # Run the setup
    success = run_colab_setup()
    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)
    
    print("\n🚀 Setup complete! Ready to start the RAG pipeline.")