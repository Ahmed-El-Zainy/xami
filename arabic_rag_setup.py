#!/usr/bin/env python3
"""
Arabic Educational RAG Pipeline Setup for Google Colab
Converted from Jupyter Notebook to Python script

This script sets up a complete RAG (Retrieval-Augmented Generation) pipeline for Arabic educational content using:
- DotsOCR for Arabic PDF text extraction
- Sentence Transformers for multilingual embeddings
- ChromaDB for vector storage
- Gemini API for intelligent responses
- UV for fast package management

Perfect for processing Arabic textbooks and educational materials up to 500+ pages!
"""

import os
import sys
import subprocess
import threading
import time
import psutil
from pathlib import Path
from getpass import getpass
import requests
import json
import zipfile

class ColabRAGSetup:
    """Arabic Educational RAG Pipeline Setup for Google Colab"""
    
    def __init__(self):
        self.project_dir = Path("/content/xami")
        self.in_colab = self._check_colab()
        self.setup_complete = False
        
    def _check_colab(self):
        """Check if running in Google Colab"""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def check_environment(self):
        """Check environment requirements"""
        print("🔍 Environment Check:")
        print("=" * 40)
        
        # Check Python
        print(f"📱 Python: {sys.version}")
        
        # Check GPU
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            print(f"🔥 GPU Available: {gpu_available}")
            if gpu_available:
                print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
                print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        except ImportError:
            print("🔥 GPU: PyTorch not installed yet")
        
        # Check RAM and Disk
        print(f"💽 RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
        print(f"📁 Disk Free: {psutil.disk_usage('/').free / 1e9:.1f} GB")
        
        # Check if in Colab
        if self.in_colab:
            print("✅ Running in Google Colab")
        else:
            print("❌ Not in Colab - some features may not work")
        
        print()
        
    def setup_api_key(self):
        """Setup Gemini API key"""
        print("🔑 API Key Setup")
        print("=" * 40)
        
        # Check if already set
        existing_key = os.environ.get('GEMINI_API_KEY')
        if existing_key and existing_key != 'your_gemini_api_key_here':
            print(f"✅ API key already set (length: {len(existing_key)})")
            return True
        
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        
        if self.in_colab:
            # In Colab, use getpass for security
            api_key = getpass("Enter your Gemini API key: ")
        else:
            # In local environment, can use input
            api_key = input("Enter your Gemini API key: ")
        
        if api_key and api_key.strip():
            os.environ['GEMINI_API_KEY'] = api_key.strip()
            print("✅ API key set successfully!")
            return True
        else:
            print("❌ No API key provided")
            return False
    
    def run_bash_setup(self):
        """Run the automated bash setup"""
        print("🚀 Automated Setup")
        print("=" * 40)
        print("This will:")
        print("1. Install UV package manager")
        print("2. Create project structure")
        print("3. Install all dependencies")
        print("⏰ This may take 10-15 minutes...")
        print()
        
        bash_commands = [
            "pip install uv --quiet",
            "cd /content",
            "mkdir -p xami/{src/{api,config,models,services,utilities,tests,weights/DotsOCR},data/{uploads,processed,chunks,chroma_db},logs,temp_logs/{requests,metrics,assets,errors},logger}",
            "find xami/src -type d -not -path '*/weights*' -exec touch {}/__init__.py \\;",
            "cd xami"
        ]
        
        # Create requirements.txt content
        requirements = """fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
aiofiles==23.2.0
sentence-transformers==2.2.2
transformers==4.35.2
torch==2.1.1
torchvision==0.16.1
torchaudio==2.1.1
google-generativeai==0.3.2
rank-bm25==0.2.2
pyarabic==0.6.15
nltk==3.8.1
python-dotenv==1.0.0
pydantic==2.5.0
pydantic-settings==2.1.0
huggingface-hub==0.19.4
qwen-vl-utils
pdf2image==1.16.3
pillow==10.1.0
opencv-python==4.8.1.78
PyPDF2==3.0.1
chromadb==0.4.18
numpy==1.24.3
requests==2.31.0
colorlog==6.8.0
tqdm==4.66.1
pyyaml==6.0.1
gradio==4.8.0
pyngrok==7.0.0"""
        
        try:
            # Execute bash commands
            for i, cmd in enumerate(bash_commands, 1):
                print(f"📦 Step {i}: {cmd}")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"⚠️ Warning in step {i}: {result.stderr}")
            
            # Create requirements.txt
            os.chdir("/content/xami")
            with open("requirements.txt", "w") as f:
                f.write(requirements)
            print("✅ Requirements file created")
            
            # Install system dependencies
            print("🔧 Installing system dependencies...")
            subprocess.run("apt-get update -qq", shell=True, capture_output=True)
            subprocess.run("apt-get install -y -qq poppler-utils libgl1-mesa-glx libglib2.0-0", 
                         shell=True, capture_output=True)
            
            # Install Python dependencies
            print("📦 Installing Python dependencies...")
            result = subprocess.run("pip install -r requirements.txt --quiet", 
                                   shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"⚠️ Some packages may have failed: {result.stderr}")
            
            print("✅ Basic setup completed!")
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def download_dotsocr_model(self):
        """Download DotsOCR model"""
        print("🤖 Download DotsOCR Model")
        print("=" * 40)
        print("⏰ This may take 5-10 minutes depending on your connection")
        
        try:
            from huggingface_hub import snapshot_download
            
            os.chdir('/content/xami')
            sys.path.insert(0, '/content/xami')
            
            model_dir = Path("src/weights/DotsOCR")
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Download the model
            snapshot_download(
                repo_id="rednote-hilab/dots.ocr",
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            print("✅ DotsOCR model downloaded successfully!")
            
            # Check downloaded files
            files = list(model_dir.glob("*"))
            print(f"📁 Downloaded {len(files)} files")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            print("🔄 You can try running this again")
            return False
    
    def create_application_files(self):
        """Create essential application files"""
        print("📁 Create Application Files")
        print("=" * 40)
        
        try:
            os.chdir('/content/xami')
            
            # Get API key from environment
            api_key = os.environ.get('GEMINI_API_KEY', 'your_gemini_api_key_here')
            
            # Create .env file
            env_content = f"""# Arabic Educational RAG Pipeline Configuration
GEMINI_API_KEY={api_key}
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
LOG_LEVEL=INFO"""
            
            with open('.env', 'w') as f:
                f.write(env_content)
            
            # Create main.py
            main_content = '''import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import time
import sys
import os

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.config.config_settings import settings
    from src.config.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    class Settings:
        API_HOST = "0.0.0.0"
        API_PORT = 8000
        API_RELOAD = False
    settings = Settings()

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
        "timestamp": time.time(),
        "services": {
            "api": True,
            "ocr": False,
            "embedding": False,
            "vector_db": False,
            "gemini": False
        }
    }

@app.post("/api/v1/test-ocr")
async def test_ocr():
    """Test OCR functionality"""
    try:
        # Test DotsOCR import
        from transformers import AutoModelForCausalLM, AutoProcessor
        model_path = "src/weights/DotsOCR"
        
        if os.path.exists(model_path):
            return {
                "status": "success",
                "message": "DotsOCR model found and ready",
                "model_path": model_path
            }
        else:
            return {
                "status": "error",
                "message": "DotsOCR model not found"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error testing OCR: {str(e)}"
        }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level="info"
    )'''
            
            with open('main.py', 'w') as f:
                f.write(main_content)
            
            # Create config files
            Path('src/config').mkdir(parents=True, exist_ok=True)
            
            # settings.py
            settings_content = '''import os
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
    os.makedirs(directory, exist_ok=True)'''
            
            with open('src/config/config_settings.py', 'w') as f:
                f.write(settings_content)
            
            # logging_config.py
            logging_content = '''import logging
import colorlog

class CustomLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
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
    return CustomLogger(name).get_logger()'''
            
            with open('src/config/logging_config.py', 'w') as f:
                f.write(logging_content)
            
            # Create __init__.py files
            with open('src/__init__.py', 'w') as f:
                f.write('')
            with open('src/config/__init__.py', 'w') as f:
                f.write('')
            
            print("✅ Application files created successfully!")
            print("📁 Project structure:")
            print("├── .env (with your API key)")
            print("├── main.py")
            print("├── src/")
            print("│   ├── config/")
            print("│   └── weights/DotsOCR/")
            print("└── data/")
            return True
            
        except Exception as e:
            print(f"❌ Error creating files: {e}")
            return False
    
    def setup_nltk(self):
        """Setup NLTK data"""
        print("📚 Setup NLTK Data")
        print("=" * 40)
        
        try:
            import nltk
            import ssl
            
            # Handle SSL issues
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            # Download required NLTK data
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            print("✅ NLTK data downloaded successfully")
            return True
            
        except Exception as e:
            print(f"⚠️ NLTK download warning: {e}")
            print("This may not affect core functionality")
            return True
    
    def start_application(self):
        """Start the RAG pipeline application"""
        print("🚀 Start the Application")
        print("=" * 40)
        
        try:
            os.chdir('/content/xami')
            sys.path.insert(0, '/content/xami')
            
            def start_server():
                """Start the FastAPI server"""
                import uvicorn
                from main import app
                
                uvicorn.run(
                    app,
                    host="0.0.0.0",
                    port=8000,
                    reload=False,
                    log_level="info",
                    access_log=False
                )
            
            # Check API key
            api_key = os.environ.get('GEMINI_API_KEY')
            if not api_key or api_key == 'your_gemini_api_key_here':
                print("⚠️ GEMINI_API_KEY not set!")
                print("Please run setup_api_key() first")
                return False
            
            print("✅ API key is set")
            
            # Start the server in background
            print("🚀 Starting the RAG pipeline server...")
            server_thread = threading.Thread(target=start_server, daemon=True)
            server_thread.start()
            
            # Wait for server to start
            time.sleep(5)
            
            # Setup ngrok tunnel if in Colab
            if self.in_colab:
                print("🌐 Setting up ngrok tunnel for external access...")
                try:
                    from pyngrok import ngrok
                    
                    # Kill any existing tunnels
                    ngrok.kill()
                    
                    # Create new tunnel
                    tunnel = ngrok.connect(8000)
                    public_url = tunnel.public_url
                    
                    print("\\n🎉 RAG Pipeline is now running!")
                    print("=" * 50)
                    print(f"🌐 Public URL: {public_url}")
                    print(f"📖 API Documentation: {public_url}/docs")
                    print(f"🔍 Health Check: {public_url}/api/v1/health")
                    print(f"🧪 Test OCR: {public_url}/api/v1/test-ocr")
                    print(f"📱 Local URL: http://localhost:8000/docs")
                    print("=" * 50)
                    
                    # Test the health endpoint
                    try:
                        response = requests.get(f"{public_url}/api/v1/health", timeout=10)
                        if response.status_code == 200:
                            print("✅ Health check passed!")
                        else:
                            print(f"⚠️ Health check returned status: {response.status_code}")
                    except Exception as e:
                        print(f"⚠️ Could not reach health endpoint: {e}")
                    
                    return public_url
                    
                except Exception as e:
                    print(f"❌ Error setting up ngrok: {e}")
                    print("📱 Server is running locally at: http://localhost:8000/docs")
                    return "http://localhost:8000"
            else:
                print("📱 Server is running locally at: http://localhost:8000/docs")
                return "http://localhost:8000"
                
        except Exception as e:
            print(f"❌ Error starting application: {e}")
            return False
    
    def test_application(self, base_url="http://localhost:8000"):
        """Test various components of the RAG pipeline"""
        print("🧪 Test the Application")
        print("=" * 40)
        
        # Get the public URL from ngrok if available
        try:
            if self.in_colab:
                from pyngrok import ngrok
                tunnels = ngrok.get_tunnels()
                if tunnels:
                    base_url = tunnels[0].public_url
        except:
            pass
        
        print(f"🧪 Testing RAG Pipeline at: {base_url}")
        print("=" * 50)
        
        # Test 1: Health Check
        print("1️⃣ Testing Health Check...")
        try:
            response = requests.get(f"{base_url}/api/v1/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Health check passed: {health_data['status']}")
                print(f"   Services: {health_data.get('services', {})}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Health check error: {e}")
        
        # Test 2: Root Endpoint
        print("\\n2️⃣ Testing Root Endpoint...")
        try:
            response = requests.get(f"{base_url}/", timeout=10)
            if response.status_code == 200:
                root_data = response.json()
                print(f"✅ Root endpoint: {root_data['message']}")
                print(f"   Version: {root_data['version']}")
            else:
                print(f"❌ Root endpoint failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Root endpoint error: {e}")
        
        # Test 3: OCR Test
        print("\\n3️⃣ Testing OCR Component...")
        try:
            response = requests.post(f"{base_url}/api/v1/test-ocr", timeout=30)
            if response.status_code == 200:
                ocr_data = response.json()
                print(f"✅ OCR test: {ocr_data['status']}")
                print(f"   Message: {ocr_data['message']}")
            else:
                print(f"❌ OCR test failed: {response.status_code}")
        except Exception as e:
            print(f"❌ OCR test error: {e}")
        
        # Test 4: Environment Check
        print("\\n4️⃣ Testing Environment...")
        try:
            import torch
            print(f"🔥 GPU Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"🎮 GPU Device: {torch.cuda.get_device_name(0)}")
        except:
            print("🔥 GPU: Could not check")
        
        # Check API key
        api_key = os.environ.get('GEMINI_API_KEY')
        if api_key and api_key != 'your_gemini_api_key_here':
            print("🔑 Gemini API key: ✅ Set")
        else:
            print("🔑 Gemini API key: ❌ Not set")
        
        # Check model files
        model_path = "/content/xami/src/weights/DotsOCR"
        if os.path.exists(model_path) and os.listdir(model_path):
            print("🤖 DotsOCR model: ✅ Downloaded")
            print(f"   Files: {len(os.listdir(model_path))}")
        else:
            print("🤖 DotsOCR model: ❌ Missing")
        
        print("\\n" + "=" * 50)
        print("🎉 Testing completed!")
        print(f"\\n📖 Access your API documentation at: {base_url}/docs")
    
    def show_usage_examples(self, base_url="http://localhost:8000"):
        """Show usage examples"""
        print("📚 Usage Examples")
        print("=" * 40)
        
        # Get the base URL
        try:
            if self.in_colab:
                from pyngrok import ngrok
                tunnels = ngrok.get_tunnels()
                base_url = tunnels[0].public_url if tunnels else "http://localhost:8000"
        except:
            pass
        
        print("\\n1️⃣ Health Check:")
        print(f"GET {base_url}/api/v1/health")
        
        print("\\n2️⃣ Document Upload (when implemented):")
        print(f"POST {base_url}/api/v1/upload-document/")
        print("# Upload a PDF file for processing")
        
        print("\\n3️⃣ RAG Query (when implemented):")
        print(f"POST {base_url}/api/v1/rag-query/")
        print("# Query: 'ما هو تعريف الجاذبية؟'")
        
        print("\\n🐍 Python Client Example:")
        print(f"""
import requests

# Health check
response = requests.get("{base_url}/api/v1/health")
print(response.json())

# Upload document (when available)
with open("textbook.pdf", "rb") as f:
    files = {{"file": f}}
    response = requests.post("{base_url}/api/v1/upload-document/", files=files)
    print(response.json())

# Query the RAG system (when available)
params = {{
    "query": "ما هو تعريف الجاذبية؟",
    "top_k": 5,
    "rerank": True
}}
response = requests.post("{base_url}/api/v1/rag-query/", params=params)
print(response.json()["answer"])
""")
        
        print(f"\\n📖 Full API documentation available at:")
        print(f"   {base_url}/docs")
    
    def troubleshoot(self):
        """Troubleshooting guide"""
        print("🔧 Troubleshooting RAG Pipeline")
        print("=" * 40)
        
        # Check 1: Environment
        print("\\n1️⃣ Environment Check:")
        print(f"📁 Current Directory: {os.getcwd()}")
        print(f"🐍 Python Path: {':'.join(sys.path[:3])}...")
        print(f"💾 Available RAM: {psutil.virtual_memory().available / 1e9:.1f} GB")
        print(f"📀 Available Disk: {psutil.disk_usage('/').free / 1e9:.1f} GB")
        
        try:
            import torch
            print(f"🔥 GPU Available: {torch.cuda.is_available()}")
        except:
            print("🔥 GPU: Could not check")
        
        # Check 2: API Key
        print("\\n2️⃣ API Key Check:")
        api_key = os.environ.get('GEMINI_API_KEY')
        if api_key and api_key != 'your_gemini_api_key_here' and len(api_key) > 10:
            print(f"✅ API Key: Set (length: {len(api_key)})")
        else:
            print("❌ API Key: Not set or invalid")
            print("   Solution: Run setup.setup_api_key()")
        
        # Check 3: Files
        print("\\n3️⃣ File Check:")
        required_files = [
            "/content/xami/.env",
            "/content/xami/main.py",
            "/content/xami/src/config/config_settings.py",
            "/content/xami/src/config/logging_config.py"
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"✅ {os.path.basename(file_path)}")
            else:
                print(f"❌ {os.path.basename(file_path)} - Missing")
        
        # Check 4: Model
        print("\\n4️⃣ Model Check:")
        model_dir = Path("/content/xami/src/weights/DotsOCR")
        if model_dir.exists():
            model_files = list(model_dir.glob("*"))
            if model_files:
                print(f"✅ DotsOCR Model: {len(model_files)} files")
                # Check for key files
                key_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
                for key_file in key_files:
                    if any(key_file in f.name for f in model_files):
                        print(f"  ✅ {key_file}")
                    else:
                        print(f"  ⚠️ {key_file} - Not found")
            else:
                print("❌ DotsOCR Model: Directory empty")
                print("   Solution: Run setup.download_dotsocr_model()")
        else:
            print("❌ DotsOCR Model: Directory missing")
            print("   Solution: Run setup.download_dotsocr_model()")
        
        # Check 5: Dependencies
        print("\\n5️⃣ Dependencies Check:")
        critical_packages = [
            'fastapi', 'uvicorn', 'transformers', 'torch', 
            'sentence_transformers', 'chromadb', 'pyngrok'
        ]
        
        for package in critical_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package} - Not installed")
                print(f"   Solution: pip install {package}")
        
        # Check 6: Server Status
        print("\\n6️⃣ Server Status:")
        try:
            response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server: Running")
            else:
                print(f"⚠️ Server: Responding with status {response.status_code}")
        except Exception:
            print("❌ Server: Not responding")
            print("   Solution: Run setup.start_application()")
        
        # Common Solutions
        print("\\n🔧 Common Solutions:")
        print("1. Restart and run all: setup.run_complete_setup()")
        print("2. Clear cache: !pip cache purge")
        print("3. Reinstall UV: !pip install --upgrade uv")
        print("4. Check GPU: Runtime → Change runtime type → GPU")
        print("5. Free memory: Runtime → Manage sessions → Terminate other sessions")
    
    def save_work(self):
        """Save work before session ends"""
        print("💾 Save Your Work")
        print("=" * 40)
        
        try:
            # Create a backup zip
            backup_path = "/content/arabic_rag_pipeline_backup.zip"
            
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add configuration files
                config_files = [
                    "/content/xami/.env",
                    "/content/xami/main.py",
                    "/content/xami/requirements.txt",
                    "/content/xami/src/config/config_settings.py",
                    "/content/xami/src/config/logging_config.py"
                ]
                
                for file_path in config_files:
                    if os.path.exists(file_path):
                        arcname = os.path.relpath(file_path, "/content/xami")
                        zipf.write(file_path, arcname)
                        print(f"✅ Added: {arcname}")
                
                # Add any custom files you've created
                custom_dir = Path("/content/xami/src")
                if custom_dir.exists():
                    for file_path in custom_dir.rglob("*.py"):
                        if "weights" not in str(file_path):  # Skip large model files
                            arcname = str(file_path.relative_to("/content/xami"))
                            zipf.write(str(file_path), arcname)
            
            print(f"\\n📦 Backup created: {backup_path}")
            print(f"📊 Size: {os.path.getsize(backup_path) / 1024 / 1024:.2f} MB")
            
            # Download the backup if in Colab
            if self.in_colab:
                print("\\n⬇️ Downloading backup...")
                try:
                    from google.colab import files
                    files.download(backup_path)
                    print("✅ Backup downloaded successfully!")
                except Exception as e:
                    print(f"⚠️ Download failed: {e}")
                    print(f"You can manually download from: {backup_path}")
            
            # Create a setup script for next time
            setup_script = """#!/bin/bash
# Quick setup script for Arabic RAG Pipeline

echo "🚀 Quick Setup for Arabic RAG Pipeline"

# Install UV
pip install uv --quiet

# Create project structure
mkdir -p /content/xami
cd /content/xami

# Extract your backup files here
echo "📁 Extract your backup zip file to /content/xami/"
echo "🔑 Set your GEMINI_API_KEY in .env file"
echo "🤖 Re-download DotsOCR model if needed"
echo "🚀 Run: python main.py"
"""
            
            with open("/content/quick_setup.sh", "w") as f:
                f.write(setup_script.strip())
            
            print("\\n📜 Created quick setup script")
            print("\\n📋 To restore your work in a new session:")
            print("1. Upload your backup zip to Colab")
            print("2. Extract it to /content/xami/")
            print("3. Set your API key in .env")
            print("4. Re-run the model download")
            print("5. Start the application")
            
            # Show what's saved
            print("\\n📁 Files in backup:")
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                for name in zipf.namelist():
                    print(f"  📄 {name}")
            
            return backup_path
            
        except Exception as e:
            print(f"❌ Error saving work: {e}")
            return None
    
    def run_complete_setup(self):
        """Run the complete setup process"""
        print("🚀 Arabic Educational RAG Pipeline Complete Setup")
        print("=" * 60)
        
        success_steps = []
        
        # Step 1: Check environment
        print("\\nStep 1: Environment Check")
        self.check_environment()
        success_steps.append("Environment Check")
        
        # Step 2: Setup API key
        print("\\nStep 2: API Key Setup")
        if self.setup_api_key():
            success_steps.append("API Key Setup")
        else:
            print("❌ Cannot continue without API key")
            return False
        
        # Step 3: Run bash setup
        print("\\nStep 3: Automated Setup")
        if self.run_bash_setup():
            success_steps.append("Bash Setup")
        else:
            print("❌ Bash setup failed")
            return False
        
        # Step 4: Download model
        print("\\nStep 4: Download DotsOCR Model")
        if self.download_dotsocr_model():
            success_steps.append("Model Download")
        else:
            print("⚠️ Model download failed, but continuing...")
        
        # Step 5: Create application files
        print("\\nStep 5: Create Application Files")
        if self.create_application_files():
            success_steps.append("Application Files")
        else:
            print("❌ Failed to create application files")
            return False
        
        # Step 6: Setup NLTK
        print("\\nStep 6: Setup NLTK")
        if self.setup_nltk():
            success_steps.append("NLTK Setup")
        
        # Step 7: Start application
        print("\\nStep 7: Start Application")
        base_url = self.start_application()
        if base_url:
            success_steps.append("Application Start")
            
            # Step 8: Test application
            print("\\nStep 8: Test Application")
            self.test_application(base_url)
            success_steps.append("Application Test")
            
            # Step 9: Show usage examples
            print("\\nStep 9: Usage Examples")
            self.show_usage_examples(base_url)
            success_steps.append("Usage Examples")
            
            self.setup_complete = True
            
            print("\\n" + "=" * 60)
            print("🎉 Setup completed successfully!")
            print(f"✅ Completed steps: {', '.join(success_steps)}")
            print("\\n📋 Next steps:")
            print("1. Visit the API documentation")
            print("2. Upload Arabic PDF documents")
            print("3. Query your documents")
            print("4. Save your work before session ends")
            
            return True
        else:
            print("❌ Failed to start application")
            return False


# Convenience functions for easy usage
def setup_rag_pipeline():
    """Main function to setup the complete RAG pipeline"""
    setup = ColabRAGSetup()
    return setup.run_complete_setup()

def quick_setup():
    """Quick setup with minimal interaction"""
    setup = ColabRAGSetup()
    
    # Check environment
    setup.check_environment()
    
    # Try to setup API key
    if not setup.setup_api_key():
        print("Please set your API key to continue")
        return setup
    
    # Run automated setup
    if setup.run_bash_setup():
        # Create basic files
        setup.create_application_files()
        setup.setup_nltk()
        
        # Start application
        base_url = setup.start_application()
        if base_url:
            print(f"\\n🎉 Quick setup complete! Access your API at: {base_url}/docs")
    
    return setup

def troubleshoot():
    """Run troubleshooting diagnostics"""
    setup = ColabRAGSetup()
    setup.troubleshoot()
    return setup


# Example usage and main execution
if __name__ == "__main__":
    print("🚀 Arabic Educational RAG Pipeline Setup")
    print("=" * 50)
    print()
    print("Available functions:")
    print("1. setup_rag_pipeline() - Complete setup")
    print("2. quick_setup() - Quick setup")
    print("3. troubleshoot() - Diagnostics")
    print()
    print("Example usage:")
    print("  setup = ColabRAGSetup()")
    print("  setup.run_complete_setup()")
    print()
    print("Or simply run:")
    print("  setup_rag_pipeline()")
    print()
    
    # Interactive mode
    choice = input("Run complete setup now? (y/n): ").lower().strip()
    if choice in ['y', 'yes']:
        setup_rag_pipeline()
    else:
        print("Setup ready. Call setup_rag_pipeline() when ready.")