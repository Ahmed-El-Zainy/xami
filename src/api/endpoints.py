# api/endpoints.py
import asyncio
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config_settings import settings
from src.config.logging_config import get_logger
from src.models.schemas import (
    DocumentUpload, OCRResult, SearchQuery, SearchResult, 
    RAGResponse, ProcessingStatus, DocumentInfo, HealthCheck
)
from src.services.ocr_service import ocr_service
from src.services.text_processor import text_processor
from src.services.embedding_service import embedding_service
from src.services.vector_db import vector_db_service
from src.services.reranker import reranker_service
from src.services.gemini_service import gemini_service
from src.services.llm_service import llm_service
from src.utilities.file_utils import file_manager

# Initialize logger
logger = get_logger(__name__)

# Create router
router = APIRouter()

# Global state for tracking processing
processing_status = {}
document_registry = {}

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    rerank: bool = True
    conversation_history: Optional[List[Dict[str, str]]] = None

class HierarchyMetadata(BaseModel):
    curriculum: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    term: Optional[str] = None  # first | second
    book: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None

class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 200

class QuestionGenerationRequest(BaseModel):
    text: str
    num_questions: int = 3

class AnswerEvaluationRequest(BaseModel):
    question: str
    student_answer: str
    correct_answer: str

# Health check endpoint
@router.get("/health/", response_model=HealthCheck)
async def health_check():
    """Check health of all services"""
    try:
        services_status = {}
        
        # Check OCR service
        try:
            services_status["ocr_service"] = ocr_service.model is not None
        except:
            services_status["ocr_service"] = False
        
        # Check embedding service
        try:
            services_status["embedding_service"] = embedding_service.model is not None
        except:
            services_status["embedding_service"] = False
        
        # Check vector database
        try:
            services_status["vector_db"] = await vector_db_service.health_check()
        except:
            services_status["vector_db"] = False
        
        # Check Gemini service
        try:
            services_status["gemini_service"] = await gemini_service.health_check()
        except:
            services_status["gemini_service"] = False
        
        # Overall status
        all_healthy = all(services_status.values())
        status = "healthy" if all_healthy else "degraded"
        
        return HealthCheck(
            status=status,
            timestamp=datetime.now(),
            services=services_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

# Document upload endpoint
@router.post("/upload-document/")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    curriculum: Optional[str] = Query(None),
    grade: Optional[str] = Query(None),
    subject: Optional[str] = Query(None),
    term: Optional[str] = Query(None),
    book: Optional[str] = Query(None),
    chapter: Optional[str] = Query(None),
    section: Optional[str] = Query(None)
):
    """Upload and process a PDF document"""
    try:
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        if file.size > 100 * 1024 * 1024:  # 100MB limit
            raise HTTPException(status_code=400, detail="File too large (max 100MB)")
        
        # Generate file ID
        file_id = str(uuid.uuid4())
        
        # Read file content
        file_content = await file.read()
        
        # Save file
        file_path = await file_manager.save_uploaded_file(file_content, file.filename)
        
        # Initialize processing status
        processing_status[file_id] = {
            "status": "uploaded",
            "message": "File uploaded successfully",
            "progress": 0.1,
            "file_path": str(file_path),
            "filename": file.filename,
            "start_time": time.time()
        }
        
        # Register document
        document_registry[file_id] = DocumentInfo(
            id=file_id,
            filename=file.filename,
            upload_time=datetime.now(),
            processing_status="uploaded",
            file_size=file.size
        )
        
        # Start background processing with hierarchy metadata
        background_tasks.add_task(
            process_document_background,
            file_id,
            file_path,
            {
                "curriculum": curriculum,
                "grade": grade,
                "subject": subject,
                "term": term,
                "book": book,
                "chapter": chapter,
                "section": section,
            }
        )
        
        logger.info(f"Document uploaded: {file.filename} with ID: {file_id}")
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "message": "File uploaded successfully. Processing started.",
            "size_mb": round(file.size / (1024 * 1024), 2)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def process_document_background(file_id: str, file_path: Path, hierarchy: Optional[Dict[str, Any]] = None):
    """Background task to process uploaded document"""
    try:
        logger.info(f"Starting background processing for file_id: {file_id}")
        
        # Update status
        processing_status[file_id].update({
            "status": "processing",
            "message": "Starting OCR processing...",
            "progress": 0.2
        })
        
        # Step 1: OCR Processing
        logger.info(f"Starting OCR for {file_id}")
        ocr_result = await ocr_service.extract_text_from_pdf(str(file_path))
        
        if not ocr_result.text:
            processing_status[file_id].update({
                "status": "failed",
                "message": "OCR processing failed - no text extracted",
                "progress": 0.0
            })
            return
        
        processing_status[file_id].update({
            "message": "OCR completed. Processing text...",
            "progress": 0.4
        })
        
        # Step 2: Text Processing and Chunking
        logger.info(f"Starting text processing for {file_id}")
        text_chunks = await text_processor.process_and_chunk_text(
            ocr_result.text,
            str(file_path),
            metadata={
                "file_id": file_id,
                "ocr_confidence": ocr_result.confidence,
                "language": ocr_result.language_detected,
                "page_count": ocr_result.page_count,
                **(hierarchy or {})
            }
        )
        
        if not text_chunks:
            processing_status[file_id].update({
                "status": "failed",
                "message": "Text processing failed - no chunks created",
                "progress": 0.0
            })
            return
        
        processing_status[file_id].update({
            "message": f"Created {len(text_chunks)} text chunks. Generating embeddings...",
            "progress": 0.6
        })
        
        # Step 3: Generate Embeddings
        logger.info(f"Generating embeddings for {file_id}")
        embeddings = await embedding_service.generate_embeddings(text_chunks)
        
        processing_status[file_id].update({
            "message": "Embeddings generated. Storing in vector database...",
            "progress": 0.8
        })
        
        # Step 4: Store in Vector Database
        logger.info(f"Storing embeddings for {file_id}")
        success = await vector_db_service.add_embeddings(embeddings)
        
        if not success:
            processing_status[file_id].update({
                "status": "failed",
                "message": "Failed to store embeddings in vector database",
                "progress": 0.0
            })
            return
        
        # Update final status
        processing_time = time.time() - processing_status[file_id]["start_time"]
        
        processing_status[file_id].update({
            "status": "completed",
            "message": "Document processing completed successfully",
            "progress": 1.0,
            "chunks_created": len(text_chunks),
            "embeddings_stored": len(embeddings),
            "processing_time": round(processing_time, 2)
        })
        
        # Update document registry
        document_registry[file_id].processing_status = "completed"
        document_registry[file_id].page_count = ocr_result.page_count
        document_registry[file_id].chunk_count = len(text_chunks)
        document_registry[file_id].language = ocr_result.language_detected
        
        logger.info(f"Document processing completed for {file_id}")
        
    except Exception as e:
        logger.error(f"Background processing failed for {file_id}: {str(e)}")
        processing_status[file_id].update({
            "status": "failed",
            "message": f"Processing failed: {str(e)}",
            "progress": 0.0
        })

# Processing status endpoint
@router.get("/processing-status/{file_id}")
async def get_processing_status(file_id: str):
    """Get processing status for a specific file"""
    if file_id not in processing_status:
        raise HTTPException(status_code=404, detail="File ID not found")
    
    status_data = processing_status[file_id].copy()
    
    return ProcessingStatus(
        status=status_data["status"],
        message=status_data["message"],
        progress=status_data["progress"],
        timestamp=datetime.now(),
        file_id=file_id
    )

# RAG Query endpoint
@router.post("/rag-query/")
async def rag_query(
    query: str = Query(..., description="The query to search for"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to return"),
    rerank: bool = Query(True, description="Whether to rerank results"),
    conversation_history: Optional[str] = Query(None, description="JSON string of conversation history"),
    curriculum: Optional[str] = Query(None),
    grade: Optional[str] = Query(None),
    subject: Optional[str] = Query(None),
    term: Optional[str] = Query(None),
    book: Optional[str] = Query(None),
    chapter: Optional[str] = Query(None),
    section: Optional[str] = Query(None)
):
    """Perform RAG query against the knowledge base"""
    try:
        start_time = time.time()
        
        # Parse conversation history if provided
        parsed_history = None
        if conversation_history:
            try:
                import json
                parsed_history = json.loads(conversation_history)
            except:
                logger.warning("Failed to parse conversation history")
        
        logger.info(f"RAG query: {query[:50]}...")
        
        # Step 1: Generate query embedding
        query_embedding = await embedding_service.generate_query_embedding(query)
        
        # Step 2: Search vector database
        filters = {
            "curriculum": curriculum,
            "grade": grade,
            "subject": subject,
            "term": term,
            "book": book,
            "chapter": chapter,
            "section": section,
        }
        # remove Nones
        filters = {k: v for k, v in filters.items() if v is not None}

        search_results = await vector_db_service.search_similar(
            query_vector=query_embedding.tolist(),
            limit=settings.RERANK_TOP_K if rerank else top_k,
            score_threshold=0.3,
            filters=filters if filters else None
        )
        
        if not search_results:
            return RAGResponse(
                query=query,
                answer="عذراً، لم أجد أي معلومات ذات صلة بسؤالك في قاعدة المعرفة المتوفرة.",
                sources=[],
                processing_time=time.time() - start_time,
                confidence_score=0.0
            )
        
        # Step 3: Rerank if requested
        if rerank:
            # Prefer Qwen reranker as main; fallback handled inside service
            search_results = await reranker_service.rerank_results(
                query, search_results, method="qwen"
            )
            search_results = search_results[:top_k]
        
        # Step 4: Generate response using Ollama (primary) or Gemini fallback
        try:
            rag_response = await llm_service.generate_rag_response(
                query, search_results, parsed_history
            )
        except Exception:
            rag_response = await gemini_service.generate_rag_response(
                query, search_results, parsed_history
            )
        
        logger.info(f"RAG query completed in {rag_response.processing_time:.2f}s")
        return rag_response
        
    except Exception as e:
        logger.error(f"RAG query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

# Search endpoint
@router.post("/search/", response_model=List[SearchResult])
async def search_documents(request: QueryRequest):
    """Search documents without generating response"""
    try:
        logger.info(f"Search query: {request.query[:50]}...")
        
        # Generate query embedding
        query_embedding = await embedding_service.generate_query_embedding(request.query)
        
        # Search vector database
        search_results = await vector_db_service.search_similar(
            query_vector=query_embedding.tolist(),
            limit=request.top_k * 2 if request.rerank else request.top_k,
            score_threshold=0.3
        )
        
        # Rerank if requested
        if request.rerank:
            search_results = await reranker_service.rerank_results(
                request.query, search_results, method="qwen"
            )
            search_results = search_results[:request.top_k]
        
        return search_results
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Text summarization endpoint
@router.post("/summarize-text/")
async def summarize_text(request: SummarizeRequest):
    """Summarize provided text"""
    try:
        try:
            summary = await llm_service.summarize_text(request.text, request.max_length)
        except Exception:
            summary = await gemini_service.summarize_text(request.text, request.max_length)
        
        return {
            "original_text": request.text[:200] + "..." if len(request.text) > 200 else request.text,
            "summary": summary,
            "original_length": len(request.text),
            "summary_length": len(summary),
            "compression_ratio": round(len(summary) / len(request.text), 2) if request.text else 0
        }
        
    except Exception as e:
        logger.error(f"Text summarization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

# Question generation endpoint
@router.post("/generate-questions/")
async def generate_questions(request: QuestionGenerationRequest):
    """Generate questions from provided text"""
    try:
        try:
            questions = await llm_service.generate_questions(request.text, request.num_questions)
        except Exception:
            questions = await gemini_service.generate_questions(request.text, request.num_questions)
        
        return {
            "text": request.text[:200] + "..." if len(request.text) > 200 else request.text,
            "questions": questions,
            "num_generated": len(questions)
        }
        
    except Exception as e:
        logger.error(f"Question generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")

# Answer evaluation endpoint
@router.post("/evaluate-answer/")
async def evaluate_answer(request: AnswerEvaluationRequest):
    """Evaluate student answer against correct answer"""
    try:
        try:
            evaluation = await llm_service.evaluate_answer(
                request.question, request.student_answer, request.correct_answer
            )
        except Exception:
            evaluation = await gemini_service.evaluate_answer(
                request.question, request.student_answer, request.correct_answer
            )
        
        return evaluation
        
    except Exception as e:
        logger.error(f"Answer evaluation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Answer evaluation failed: {str(e)}")

# Document management endpoints
@router.get("/documents/")
async def list_documents():
    """List all uploaded documents"""
    try:
        documents = list(document_registry.values())
        return {
            "documents": documents,
            "total_count": len(documents)
        }
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list documents")

@router.delete("/documents/{file_id}")
async def delete_document(file_id: str):
    """Delete a document and all related data"""
    try:
        if file_id not in document_registry:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from vector database
        doc_info = document_registry[file_id]
        await vector_db_service.delete_by_source(doc_info.filename)
        
        # Delete files
        success = await file_manager.delete_file_and_related(file_id)
        
        if success:
            # Remove from registries
            del document_registry[file_id]
            if file_id in processing_status:
                del processing_status[file_id]
            
            return {"message": f"Document {file_id} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete document files")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

# Statistics endpoint
@router.get("/statistics/")
async def get_statistics():
    """Get system statistics"""
    try:
        # Vector database stats
        vector_stats = await vector_db_service.get_collection_stats()
        
        # File system stats
        disk_usage = await file_manager.get_disk_usage()
        
        # Processing stats
        total_documents = len(document_registry)
        completed_documents = sum(1 for doc in document_registry.values() 
                                if doc.processing_status == "completed")
        processing_documents = sum(1 for status in processing_status.values() 
                                 if status["status"] == "processing")
        
        # Service health
        services_health = {}
        try:
            services_health["ocr_service"] = ocr_service.model is not None
            services_health["embedding_service"] = embedding_service.model is not None
            services_health["vector_db"] = await vector_db_service.health_check()
            services_health["gemini_service"] = await gemini_service.health_check()
        except:
            pass
        
        return {
            "vector_database": vector_stats,
            "file_system": disk_usage,
            "documents": {
                "total": total_documents,
                "completed": completed_documents,
                "processing": processing_documents,
                "failed": total_documents - completed_documents - processing_documents
            },
            "services_health": services_health,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

# Cleanup endpoint
@router.post("/cleanup/")
async def cleanup_old_data(days_old: int = Query(30, ge=1, le=365)):
    """Clean up old files and data"""
    try:
        # Clean up files
        cleanup_results = await file_manager.cleanup_old_files(days_old)
        
        # Clean up processing status (keep only recent ones)
        current_time = time.time()
        old_statuses = [
            file_id for file_id, status in processing_status.items()
            if current_time - status.get("start_time", current_time) > (days_old * 24 * 60 * 60)
        ]
        
        for file_id in old_statuses:
            del processing_status[file_id]
        
        cleanup_results["processing_status_cleaned"] = len(old_statuses)
        
        return {
            "message": f"Cleanup completed for files older than {days_old} days",
            "cleanup_results": cleanup_results
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

# Reindex endpoint
@router.post("/reindex/")
async def reindex_documents():
    """Reindex all documents in the vector database"""
    try:
        # This is a simplified reindexing - in production, you'd want more sophisticated logic
        await vector_db_service.optimize_collection()
        
        return {
            "message": "Reindexing completed successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Reindexing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {str(e)}")

# System info endpoint
@router.get("/system-info/")
async def get_system_info():
    """Get system information"""
    try:
        # Get service information
        gemini_info = await gemini_service.get_service_info()
        llm_info = await llm_service.get_service_info()
        
        # Get embedding service info
        embedding_stats = {}
        if embedding_service.model:
            embedding_stats = {
                "model_name": settings.EMBEDDING_MODEL,
                "device": embedding_service.device,
                "vector_size": settings.VECTOR_SIZE
            }
        
        return {
            "application": {
                "name": "Arabic Educational RAG Pipeline",
                "version": "1.0.0",
                "environment": "development",  # You can make this configurable
            },
            "services": {
                "gemini": gemini_info,
                "ollama_llm": llm_info,
                "embedding": embedding_stats,
                "vector_db": {
                    "type": "Qdrant",
                    "collection": settings.QDRANT_COLLECTION_NAME
                },
                "resource_manager": {"note": "Ensures only one heavy GPU module active at a time"}
            },
            "configuration": {
                "chunk_size": settings.CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP,
                "min_chunk_size": settings.MIN_CHUNK_SIZE,
                "rerank_top_k": settings.RERANK_TOP_K,
                "final_top_k": settings.FINAL_TOP_K
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get system info")

# Root endpoint for API
@router.get("/")
async def api_root():
    """API root endpoint"""
    return {
        "message": "Arabic Educational RAG Pipeline API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health/",
            "upload": "/upload-document/",
            "query": "/rag-query/",
            "search": "/search/",
            "documents": "/documents/",
            "statistics": "/statistics/",
            "docs": "/docs"
        },
        "timestamp": datetime.now().isoformat()
    }