# config/gemini_config.py
"""
Configuration settings for Gemini Service
"""
import os
from typing import Optional

class GeminiSettings:
    """Gemini-specific configuration settings"""
    
    # API Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    
    # Generation Parameters
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
    GEMINI_TOP_P: float = float(os.getenv("GEMINI_TOP_P", "0.8"))
    GEMINI_TOP_K: int = int(os.getenv("GEMINI_TOP_K", "40"))
    GEMINI_MAX_OUTPUT_TOKENS: int = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "2048"))
    
    # RAG Settings
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    MAX_CONVERSATION_HISTORY: int = int(os.getenv("MAX_CONVERSATION_HISTORY", "3"))
    
    # Performance Settings
    REQUEST_TIMEOUT: int = int(os.getenv("GEMINI_REQUEST_TIMEOUT", "30"))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
    
    # Language Settings
    DEFAULT_LANGUAGE: str = os.getenv("DEFAULT_LANGUAGE", "ar")
    SUPPORTED_LANGUAGES: list = ["ar", "en", "mixed"]
    
    @classmethod
    def validate_settings(cls) -> bool:
        """Validate that all required settings are present"""
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required")
        
        if cls.GEMINI_MODEL not in ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]:
            raise ValueError(f"Unsupported model: {cls.GEMINI_MODEL}")
        
        return True
