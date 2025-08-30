# config/settings.py
import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # Gemini API
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash-exp"
    
    # Vector Database (Qdrant)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "arabic_educational_content"
    CHROMA_PERSIST_DIR: str = "data/chroma_db"  # legacy, kept for compatibility
    CHROMA_COLLECTION_NAME: str = "arabic_educational_content"  # legacy
    VECTOR_SIZE: int = 1024  # Match Qwen/Qwen3-Embedding-0.6B output dimension
    
    # Embedding Models
    EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-0.6B"
    EMBEDDING_BACKEND: str = "hf"  # options: hf | llama_cpp | sentence_transformers
    EMBEDDING_MODEL_PATH: str = ""  # optional GGUF path if using llama_cpp
    
    # Reranker Model
    RERANKER_MODEL: str = "Qwen/Qwen3-Reranker-0.6B"
    RERANKER_DEVICE: str = "auto"  # auto|cpu|cuda
    # Alternative Arabic-optimized models:
    # "aubmindlab/bert-base-arabertv02"
    # "CAMeL-Lab/bert-base-arabic-camelbert-mix"
    
    # OCR Configuration
    OCR_LANGUAGES: List[str] = ["ar", "en"]  # Arabic and English
    OCR_GPU: bool = True
    DOTS_OCR_MODEL_PATH : str = "src/dotsOCR/DotsOCR"  # Local path to DotsOCR model assets
    DOTS_OCR_DEVICE: str = "auto" # Device for DotsOCR, e.g., "cuda", "cpu", "auto"
    DOTS_OCR_AUTO_DOWNLOAD: bool = False  # Do not auto-download when local path is used
    DOTS_OCR_MAX_NEW_TOKENS: int = 256 # Reduced for testing
    DOTS_OCR_DPI: int = 120 # Reduced for testing
    DOTS_OCR_MAX_IMAGE_WIDTH: int = 1280
    
    
    # Text Processing
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MIN_CHUNK_SIZE: int = 100
    
    # Reranking
    RERANK_TOP_K: int = 20
    FINAL_TOP_K: int = 5

    # LLM (Ollama)
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "custom-qwen3_30b_a3b_Q3_K_S:latest"
    
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