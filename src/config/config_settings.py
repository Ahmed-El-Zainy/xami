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
    
    # Vector Database (ChromaDB)
    CHROMA_PERSIST_DIR: str = "data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "arabic_educational_content"
    VECTOR_SIZE: int = 384  # For sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    
    # Embedding Models
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Alternative Arabic-optimized models:
    # "aubmindlab/bert-base-arabertv02"
    # "CAMeL-Lab/bert-base-arabic-camelbert-mix"
    
    # OCR Configuration
    OCR_LANGUAGES: List[str] = ["ar", "en"]  # Arabic and English
    OCR_GPU: bool = True
    
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