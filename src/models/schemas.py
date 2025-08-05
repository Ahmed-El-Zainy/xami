# models/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class DocumentUpload(BaseModel):
    filename: str
    content_type: str = "application/pdf"

class OCRResult(BaseModel):
    text: str
    confidence: float
    processing_time: float
    language_detected: str
    page_count: int

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

class SearchQuery(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    include_metadata: bool = True
    rerank: bool = True

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
    status: str  # "processing", "completed", "failed"
    message: str
    progress: float  # 0.0 to 1.0
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
    services: Dict[str, bool]  # Service availability status