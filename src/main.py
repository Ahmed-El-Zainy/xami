# main.py
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time

from config.settings import settings
from config.logging_config import get_logger
from api.endpoints import router
from services.ocr_service import ocr_service
from services.embedding_service import embedding_service
from services.vector_db import vector_db_service
from services.gemini_service import gemini_service

# Initialize logger
logger = get_logger(__name__)

# Global services initialization status
services_initialized = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global services_initialized
    
    try:
        logger.info("Starting RAG Pipeline application...")
        
        # Initialize all services
        await initialize_services()
        services_initialized = True
        
        logger.info("RAG Pipeline application started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down RAG Pipeline application...")
        await cleanup_services()
        logger.info("Application shutdown complete")

async def initialize_services():
    """Initialize all services"""
    try:
        logger.info("Initializing services...")
        
        # Initialize services in order of dependency
        
        # 1. Vector Database (no dependencies)
        logger.info("Initializing Vector Database...")
        await vector_db_service.initialize()
        
        # 2. OCR Service (no dependencies)
        logger.info("Initializing OCR Service...")
        await ocr_service.initialize()
        
        # 3. Embedding Service (no dependencies)
        logger.info("Initializing Embedding Service...")
        await embedding_service.initialize()
        
        # 4. Gemini Service (no dependencies)
        logger.info("Initializing Gemini Service...")
        await gemini_service.initialize()
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Service initialization failed: {str(e)}")
        raise

async def cleanup_services():
    """Cleanup all services"""
    try:
        logger.info("Cleaning up services...")
        
        # Cleanup in reverse order
        gemini_service.cleanup() if hasattr(gemini_service, 'cleanup') else None
        embedding_service.cleanup()
        ocr_service.cleanup()
        vector_db_service.cleanup()
        
        logger.info("Services cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

# Create FastAPI application
app = FastAPI(
    title="Arabic Educational RAG Pipeline",
    description="""
    A complete RAG (Retrieval-Augmented Generation) pipeline for Arabic educational content.
    
    Features:
    - PDF upload and OCR processing with EasyOCR
    - Arabic text processing and chunking
    - Multilingual embeddings with sentence-transformers
    - ChromaDB vector database for efficient similarity search
    - Advanced reranking with BM25 and hybrid methods
    - Gemini API integration for intelligent responses
    - Custom logging and monitoring
    
    Perfect for secondary school Arabic educational materials up to 500+ pages.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    GZipMiddleware, 
    minimum_size=1000
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["RAG Pipeline"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Arabic Educational RAG Pipeline API",
        "version": "1.0.0",
        "status": "running" if services_initialized else "initializing",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled Exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": time.time()
        }
    )

# Middleware for request logging
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests"""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.2f}s")
    
    # Add process time header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Startup event (additional to lifespan)
@app.on_event("startup")
async def startup_event():
    """Additional startup tasks"""
    logger.info("Performing additional startup tasks...")
    
    # Create any additional directories if needed
    # Warm up services if needed
    
    logger.info("Startup tasks completed")

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_config=None,  # Use our custom logging
        access_log=False  # We handle request logging in middleware
    )