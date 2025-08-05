# services/embedding_service.py
import asyncio
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import torch
from concurrent.futures import ThreadPoolExecutor
import os 
import sys 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"SCRIPT_DIR: {SCRIPT_DIR}")
print(f"os.path.dirname(os.path.dirname(SCRIPT_DIR)): {os.path.dirname(os.path.dirname(SCRIPT_DIR))}")
print(f"os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))): {os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))}")
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))


from ocr.config.custom_logger import get_logger, log_execution_time, CustomLoggerTracker

try:
    # from logger.custom_logger import CustomLoggerTracker
    custom = CustomLoggerTracker()
    logger = custom.get_logger("embedding_service")
    logger.info("Custom Logger Start Working.....")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("embedding_service")
    logger.info("Using standard logger - custom logger not available")




from src.ocr.config.config_settings import settings
from src.ocr.config.logging_config import get_logger, log_execution_time
from models.schemas import TextChunk, EmbeddingResult
from utils.arabic_utils import arabic_processor

class EmbeddingService:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    async def initialize(self):
        """Initialize the embedding model"""
        try:
            self.logger.info(f"Initializing embedding model: {settings.EMBEDDING_MODEL}")
            self.logger.info(f"Using device: {self.device}")
            
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self.executor,
                lambda: SentenceTransformer(settings.EMBEDDING_MODEL, device=self.device)
            )
            
            # Test the model
            test_embedding = await self._encode_text(["test"])
            self.logger.info(f"Model initialized successfully. Embedding dimension: {len(test_embedding[0])}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise
    
    @log_execution_time
    async def generate_embeddings(self, chunks: List[TextChunk]) -> List[EmbeddingResult]:
        """Generate embeddings for text chunks"""
        if not self.model:
            await self.initialize()
        
        if not chunks:
            return []
        
        try:
            self.logger.info(f"Generating embeddings for {len(chunks)} chunks")
            
            # Prepare texts for embedding
            texts = []
            for chunk in chunks:
                # Preprocess text for better embeddings
                processed_text = self._preprocess_for_embedding(chunk.text)
                texts.append(processed_text)
            
            # Generate embeddings in batches
            batch_size = 32  # Adjust based on GPU memory
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                batch_embeddings = await self._encode_text(batch_texts)
                all_embeddings.extend(batch_embeddings)
            
            # Create EmbeddingResult objects
            embedding_results = []
            for chunk, embedding in zip(chunks, all_embeddings):
                result = EmbeddingResult(
                    chunk_id=chunk.id,
                    embedding=embedding.tolist(),
                    text=chunk.text,
                    metadata=chunk.metadata
                )
                embedding_results.append(result)
            
            self.logger.info(f"Generated {len(embedding_results)} embeddings successfully")
            return embedding_results
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    async def _encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor,
            lambda: self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # Important for cosine similarity
            )
        )
        return embeddings
    
    def _preprocess_for_embedding(self, text: str) -> str:
        """Preprocess text for better embedding quality"""
        # Clean the text
        processed_text = arabic_processor.clean_arabic_text(text)
        
        # Truncate if too long (model context limits)
        max_length = 512  # Adjust based on model
        if len(processed_text) > max_length:
            # Try to truncate at sentence boundary
            sentences = arabic_processor.segment_sentences(processed_text)
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence) <= max_length:
                    truncated += sentence + " "
                else:
                    break
            processed_text = truncated.strip() if truncated else processed_text[:max_length]
        
        return processed_text
    
    @log_execution_time
    async def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a search query"""
        if not self.model:
            await self.initialize()
        
        try:
            # Preprocess query
            processed_query = self._preprocess_query(query)
            
            # Generate embedding
            embedding = await self._encode_text([processed_query])
            return embedding[0]
            
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {str(e)}")
            raise
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better matching"""
        # Clean and normalize
        processed_query = arabic_processor.normalize_for_search(query)
        
        # Remove stopwords for better matching
        language = arabic_processor.detect_language(query)
        processed_query = arabic_processor.remove_stopwords(processed_query, language)
        
        return processed_query
    
    async def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            # Ensure embeddings are normalized
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error computing similarity: {str(e)}")
            return 0.0
    
    async def find_similar_chunks(
        self, 
        query_embedding: np.ndarray, 
        chunk_embeddings: List[EmbeddingResult],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Find most similar chunks to a query"""
        try:
            similarities = []
            
            for chunk_embedding in chunk_embeddings:
                embedding = np.array(chunk_embedding.embedding)
                similarity = await self.compute_similarity(query_embedding, embedding)
                
                similarities.append({
                    "chunk_id": chunk_embedding.chunk_id,
                    "text": chunk_embedding.text,
                    "metadata": chunk_embedding.metadata,
                    "similarity": similarity
                })
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error finding similar chunks: {str(e)}")
            return []
    
    async def batch_similarity_search(
        self,
        query_embeddings: List[np.ndarray],
        chunk_embeddings: List[EmbeddingResult],
        top_k: int = 10
    ) -> List[List[Dict[str, Any]]]:
        """Perform batch similarity search"""
        try:
            results = []
            
            for query_embedding in query_embeddings:
                similar_chunks = await self.find_similar_chunks(
                    query_embedding, chunk_embeddings, top_k
                )
                results.append(similar_chunks)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch similarity search: {str(e)}")
            return []
    
    def get_embedding_stats(self, embeddings: List[EmbeddingResult]) -> Dict[str, Any]:
        """Get statistics about embeddings"""
        if not embeddings:
            return {}
        
        embedding_matrix = np.array([emb.embedding for emb in embeddings])
        
        stats = {
            "total_embeddings": len(embeddings),
            "embedding_dimension": len(embeddings[0].embedding),
            "mean_norm": float(np.mean(np.linalg.norm(embedding_matrix, axis=1))),
            "std_norm": float(np.std(np.linalg.norm(embedding_matrix, axis=1))),
            "model_used": settings.EMBEDDING_MODEL,
            "device_used": self.device
        }
        
        return stats
    
    def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        # Clear GPU memory if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Global embedding service instance
embedding_service = EmbeddingService()