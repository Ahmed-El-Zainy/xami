#!/bin/bash

# Enhanced run.sh script for Arabic Educational RAG Pipeline
# Handles automated setup with UV, model downloads, and Colab compatibility

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="xami"
PYTHON_MIN_VERSION="3.8"
COLAB_ENV="/content"
CURRENT_DIR=$(pwd)

# Functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}🚀 Arabic Educational RAG Pipeline Setup${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_step() {
    echo -e "${GREEN}📋 Step $1: $2${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Check if running in Colab
check_environment() {
    if [[ -d "/content" ]] && [[ -n "${COLAB_GPU}" || -n "${COLAB_TPU_ADDR}" ]]; then
        echo "🌐 Detected Google Colab environment"
        export IS_COLAB=true
        export PROJECT_DIR="/content/${PROJECT_NAME}"
    else
        echo "💻 Detected local environment"
        export IS_COLAB=false
        export PROJECT_DIR="${CURRENT_DIR}"
    fi
}

# Check Python version
check_python() {
    print_step "1" "Checking Python version"
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python ${PYTHON_MIN_VERSION} or higher."
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [[ $PYTHON_MAJOR -eq 3 ]] && [[ $PYTHON_MINOR -ge 8 ]]; then
        print_success "Python ${PYTHON_VERSION} detected"
        export PYTHON_CMD
    else
        print_error "Python ${PYTHON_MIN_VERSION} or higher is required. Found: ${PYTHON_VERSION}"
    fi
}

# Install system dependencies
install_system_deps() {
    print_step "2" "Installing system dependencies"
    
    if [[ "$IS_COLAB" == "true" ]]; then
        # Colab-specific system dependencies
        apt-get update -qq
        apt-get install -y -qq \
            poppler-utils \
            libgl1-mesa-glx \
            libglib2.0-0 \
            libsm6 \
            libxext6 \
            libxrender1 \
            libgomp1 \
            tesseract-ocr \
            tesseract-ocr-ara \
            tesseract-ocr-eng
    else
        # Local environment - check OS and install accordingly
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            if command -v apt-get &> /dev/null; then
                sudo apt-get update
                sudo apt-get install -y \
                    poppler-utils \
                    libgl1-mesa-glx \
                    libglib2.0-0 \
                    tesseract-ocr \
                    tesseract-ocr-ara
            elif command -v yum &> /dev/null; then
                sudo yum install -y \
                    poppler-utils \
                    tesseract \
                    tesseract-langpack-ara
            fi
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            if command -v brew &> /dev/null; then
                brew install poppler tesseract tesseract-lang
            else
                print_warning "Homebrew not found. Please install system dependencies manually."
            fi
        fi
    fi
    
    print_success "System dependencies installed"
}

# Install UV package manager
install_uv() {
    print_step "3" "Installing UV package manager"
    
    if command -v uv &> /dev/null; then
        print_success "UV already installed: $(uv --version)"
        return
    fi
    
    # Install UV using the official installer
    if [[ "$IS_COLAB" == "true" ]]; then
        # In Colab, install via pip
        $PYTHON_CMD -m pip install uv --quiet
    else
        # Use official installer for local environments
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source ~/.bashrc 2>/dev/null || source ~/.zshrc 2>/dev/null || true
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    
    # Verify installation
    if command -v uv &> /dev/null; then
        print_success "UV installed successfully: $(uv --version)"
    else
        print_error "Failed to install UV package manager"
    fi
}

# Create project structure
create_project_structure() {
    print_step "4" "Creating project structure"
    
    # Create project directory if in Colab
    if [[ "$IS_COLAB" == "true" ]]; then
        mkdir -p "$PROJECT_DIR"
        cd "$PROJECT_DIR"
    fi
    
    # Create directory structure
    DIRECTORIES=(
        "src/api"
        "src/config"
        "src/models"
        "src/services"
        "src/utilities"
        "src/tests"
        "src/weights/DotsOCR"
        "data/uploads"
        "data/processed"
        "data/chunks"
        "data/chroma_db"
        "logs"
        "temp_logs/requests"
        "temp_logs/metrics"
        "temp_logs/assets"
        "temp_logs/errors"
        "logger"
    )
    
    for dir in "${DIRECTORIES[@]}"; do
        mkdir -p "$dir"
        # Create __init__.py for Python packages
        if [[ "$dir" == src/* ]] && [[ "$dir" != *"weights"* ]]; then
            touch "$dir/__init__.py"
        fi
    done
    
    print_success "Project structure created"
}

# Setup Python environment with UV
setup_python_env() {
    print_step "5" "Setting up Python environment with UV"
    
    cd "$PROJECT_DIR"
    
    # Initialize UV project if pyproject.toml doesn't exist
    if [[ ! -f "pyproject.toml" ]]; then
        uv init --no-readme --quiet
    fi
    
    # Create requirements list
    cat > requirements.txt << 'EOF'
fastapi==0.104.1
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
pytest==7.4.3
pytest-asyncio==0.21.1
gradio==4.8.0
streamlit==1.28.0
EOF
    
    # Install dependencies with UV
    echo "📦 Installing Python dependencies..."
    
    # Read requirements and install each package
    while IFS= read -r package; do
        if [[ ! -z "$package" ]] && [[ ! "$package" =~ ^# ]]; then
            echo "Installing $package..."
            if ! uv add "$package" --quiet; then
                print_warning "Failed to install $package with UV, trying pip..."
                $PYTHON_CMD -m pip install "$package" --quiet
            fi
        fi
    done < requirements.txt
    
    print_success "Python environment setup completed"
}

# Download and setup DotsOCR model
setup_dotsocr_model() {
    print_step "6" "Setting up DotsOCR model"
    
    cd "$PROJECT_DIR"
    
    # Create model download script
    cat > download_dotsocr.py << 'EOF'
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

def download_dotsocr_model():
    """Download DotsOCR model from HuggingFace"""
    model_dir = Path("src/weights/DotsOCR")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("🤖 Downloading DotsOCR model from HuggingFace...")
    
    try:
        snapshot_download(
            repo_id="rednote-hilab/dots.ocr",
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("✅ DotsOCR model downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to download DotsOCR model: {e}")
        return False

def create_config_files():
    """Create additional configuration files for DotsOCR"""
    model_dir = Path("src/weights/DotsOCR")
    
    # Configuration files content
    config_content = '''from typing import Any, Optional
from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2 import Qwen2Config
from transformers import Qwen2_5_VLProcessor, AutoProcessor
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

class DotsVisionConfig(PretrainedConfig):
    model_type: str = "dots_vit"

    def __init__(
        self,
        embed_dim: int = 1536,
        hidden_size: int = 1536,
        intermediate_size: int = 4224,
        num_hidden_layers: int = 42,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 1,
        rms_norm_eps: float = 1e-5,
        use_bias: bool = False,
        attn_implementation="flash_attention_2",
        initializer_range=0.02,
        init_merger_std=0.02,
        is_causal=False,
        post_norm=True,
        gradient_checkpointing=False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.rms_norm_eps = rms_norm_eps
        self.use_bias = use_bias
        self.attn_implementation = attn_implementation
        self.initializer_range = initializer_range
        self.init_merger_std = init_merger_std
        self.is_causal = is_causal
        self.post_norm = post_norm
        self.gradient_checkpointing = gradient_checkpointing

class DotsOCRConfig(Qwen2Config):
    model_type = "dots_ocr"
    def __init__(self, 
        image_token_id = 151665, 
        video_token_id = 151656,
        vision_config: Optional[dict] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_config = DotsVisionConfig(**(vision_config or {}))

    def save_pretrained(self, save_directory, **kwargs):
        self._auto_class = None
        super().save_pretrained(save_directory, **kwargs)

class DotsVLProcessor(Qwen2_5_VLProcessor):
    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.image_token = "<|imgpad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token

AutoProcessor.register("dots_ocr", DotsVLProcessor)
CONFIG_MAPPING.register("dots_ocr", DotsOCRConfig)
'''
    
    # Write configuration file
    config_file = model_dir / "configuration_dots.py"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print("✅ Configuration files created")

if __name__ == "__main__":
    if download_dotsocr_model():
        create_config_files()
        print("🎉 DotsOCR setup completed successfully!")
    else:
        print("❌ DotsOCR setup failed!")
        sys.exit(1)
EOF
    
    # Run the download script
    $PYTHON_CMD download_dotsocr.py
    
    # Clean up download script
    rm download_dotsocr.py
    
    print_success "DotsOCR model setup completed"
}

# Create essential application files
create_app_files() {
    print_step "7" "Creating application files"
    
    cd "$PROJECT_DIR"
    
    # Create .env file
    cat > .env << 'EOF'
# Arabic Educational RAG Pipeline Configuration
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
EOF
    
    # Create main.py
    cat > main.py << 'EOF'
import asyncio
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
    # Create minimal settings
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
            "ocr": False,  # Will be updated when services are loaded
            "embedding": False,
            "vector_db": False,
            "gemini": False
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level="info"
    )
EOF
    
    # Create basic config files
    mkdir -p src/config
    
    cat > src/config/config_settings.py << 'EOF'
import os
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
EOF
    
    cat > src/config/logging_config.py << 'EOF'
import logging
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
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return func
    return decorator
EOF
    
    # Create startup script
    cat > start_rag_pipeline.py << 'EOF'
#!/usr/bin/env python3
"""
Startup script for Arabic Educational RAG Pipeline
"""
import sys
import os
from pathlib import Path
import asyncio

# Add project to Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

def check_api_key():
    """Check if Gemini API key is set"""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_gemini_api_key_here':
        print("⚠️  GEMINI_API_KEY not set!")
        print("Please set it in your .env file:")
        print("1. Open .env file")
        print("2. Replace 'your_gemini_api_key_here' with your actual API key")
        print("3. Save the file and run this script again")
        return False
    return True

async def main():
    """Main startup function"""
    print("🚀 Starting Arabic Educational RAG Pipeline...")
    
    # Check API key
    if not check_api_key():
        return False
    
    # Set environment variables
    os.environ["PYTHONPATH"] = str(project_dir)
    
    try:
        # Import and start the application
        from main import app
        from src.config.config_settings import settings
        import uvicorn
        
        print(f"📖 API Documentation: http://localhost:{settings.API_PORT}/docs")
        print(f"🔍 Health Check: http://localhost:{settings.API_PORT}/api/v1/health")
        print(f"🌐 API Root: http://localhost:{settings.API_PORT}/")
        
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
EOF
    
    # Make startup script executable
    chmod +x start_rag_pipeline.py
    
    print_success "Application files created"
}

# Download NLTK data
setup_nltk() {
    print_step "8" "Setting up NLTK data"
    
    cd "$PROJECT_DIR"
    
    cat > download_nltk.py << 'EOF'
import nltk
import ssl

# Handle SSL issues in some environments
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
EOF
    
    $PYTHON_CMD download_nltk.py
    rm download_nltk.py
    
    print_success "NLTK data setup completed"
}

# Verify installation
verify_installation() {
    print_step "9" "Verifying installation"
    
    cd "$PROJECT_DIR"
    
    # Create verification script
    cat > verify_setup.py << 'EOF'
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test critical imports"""
    test_modules = [
        'fastapi',
        'transformers',
        'torch',
        'sentence_transformers',
        'google.generativeai',
        'chromadb',
        'huggingface_hub',
        'nltk',
        'cv2',
        'PIL'
    ]
    
    failed_imports = []
    
    for module in test_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def check_files():
    """Check essential files exist"""
    essential_files = [
        '.env',
        'main.py',
        'start_rag_pipeline.py',
        'requirements.txt',
        'src/config/config_settings.py',
        'src/config/logging_config.py'
    ]
    
    missing_files = []
    
    for file_path in essential_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} missing")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def check_model():
    """Check DotsOCR model"""
    model_dir = Path("src/weights/DotsOCR")
    if model_dir.exists() and any(model_dir.iterdir()):
        print("✅ DotsOCR model directory exists and contains files")
        return True
    else:
        print("❌ DotsOCR model directory is missing or empty")
        return False

def main():
    print("🔍 Verifying installation...")
    print("\n📦 Checking imports:")
    imports_ok = test_imports()
    
    print("\n📁 Checking files:")
    files_ok = check_files()
    
    print("\n🤖 Checking model:")
    model_ok = check_model()
    
    if imports_ok and files_ok and model_ok:
        print("\n🎉 Installation verification successful!")
        print("\n📋 Next steps:")
        print("1. Set your Gemini API key in .env file")
        print("2. Run: python start_rag_pipeline.py")
        print("3. Open http://localhost:8000/docs")
        return True
    else:
        print("\n❌ Installation verification failed!")
        print("Please check the errors above and re-run the setup.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF
    
    $PYTHON_CMD verify_setup.py
    VERIFY_EXIT_CODE=$?
    
    rm verify_setup.py
    
    if [[ $VERIFY_EXIT_CODE -eq 0 ]]; then
        print_success "Installation verification passed"
        return 0
    else
        print_error "Installation verification failed"
        return 1
    fi
}

# Create Colab notebook launcher
create_colab_launcher() {
    if [[ "$IS_COLAB" == "true" ]]; then
        print_step "10" "Creating Colab launcher"
        
        cd "$PROJECT_DIR"
        
        cat > launch_in_colab.py << 'EOF'
"""
Colab-specific launcher for Arabic Educational RAG Pipeline
Run this in a Colab cell to start the application
"""

import os
import sys
import subprocess
from pathlib import Path
import time

# Setup environment
project_dir = Path("/content/xami")
os.chdir(project_dir)
sys.path.insert(0, str(project_dir))

def show_urls():
    """Display access URLs with ngrok"""
    try:
        # Install and setup ngrok for external access
        subprocess.run(["pip", "install", "pyngrok"], check=True, capture_output=True)
        
        from pyngrok import ngrok
        
        # Create tunnel
        tunnel = ngrok.connect(8000)
        public_url = tunnel.public_url
        
        print("🌐 RAG Pipeline URLs:")
        print(f"📖 API Documentation: {public_url}/docs")
        print(f"🔍 Health Check: {public_url}/api/v1/health")
        print(f"🌐 API Root: {public_url}/")
        print(f"📱 Local: http://localhost:8000/docs")
        
        return public_url
        
    except Exception as e:
        print(f"⚠️ Could not setup ngrok: {e}")
        print("📱 Local access only: http://localhost:8000/docs")
        return None

def start_pipeline():
    """Start the RAG pipeline"""
    print("🚀 Starting Arabic Educational RAG Pipeline in Colab...")
    
    # Check API key
    if not os.getenv('GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY') == 'your_gemini_api_key_here':
        print("⚠️ Please set your Gemini API key:")
        print("1. Get API key from: https://makersuite.google.com/app/apikey")
        print("2. Run: os.environ['GEMINI_API_KEY'] = 'your_actual_api_key'")
        return
    
    # Show URLs
    public_url = show_urls()
    
    # Start the application
    try:
        exec(open('start_rag_pipeline.py').read())
    except KeyboardInterrupt:
        print("\n🛑 Pipeline stopped")
    except Exception as e:
        print(f"❌ Error: {e}")

# Auto-start if run directly
if __name__ == "__main__":
    start_pipeline()

# For notebook usage
def launch():
    """Launch function for notebook cells"""
    start_pipeline()

print("📝 To start the pipeline, run: launch()")
EOF
        
        print_success "Colab launcher created"
    fi
}

# Main execution
main() {
    print_header
    
    # Check environment
    check_environment
    
    # Run setup steps
    check_python
    install_system_deps
    install_uv
    create_project_structure
    setup_python_env
    setup_dotsocr_model
    create_app_files
    setup_nltk
    create_colab_launcher
    
    # Verify installation
    if verify_installation; then
        print_header
        print_success "🎉 Arabic Educational RAG Pipeline setup completed successfully!"
        echo ""
        echo "📋 Next Steps:"
        echo "1. Set your Gemini API key in .env file:"
        echo "   GEMINI_API_KEY=your_actual_api_key_here"
        echo ""
        echo "2. Start the pipeline:"
        if [[ "$IS_COLAB" == "true" ]]; then
            echo "   python launch_in_colab.py"
            echo "   OR run launch() in a notebook cell"
        else
            echo "   python start_rag_pipeline.py"
        fi
        echo ""
        echo "3. Open the API documentation:"
        echo "   http://localhost:8000/docs"
        echo ""
        print_success "Setup completed! 🚀"
    else
        print_error "Setup failed during verification. Please check the errors above."
    fi
}

# Run main function
main "$@"