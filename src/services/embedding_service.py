# services/embedding_service.py
import asyncio
import numpy as np
import gc
from typing import List, Dict, Any
import torch
from concurrent.futures import ThreadPoolExecutor
import os 
import sys 


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.config.logging_config import get_logger, log_execution_time, CustomLoggerTracker
from src.config.config_settings import settings
# from config.logging_config import CustomLoggerTracker, get_logger, log_execution_time
from src.models.schemas import TextChunk, EmbeddingResult
from src.utilities.arabic_utils import arabic_processor
from src.services.resource_manager import resource_manager


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



class EmbeddingService:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.backend = settings.EMBEDDING_BACKEND
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = False
        self.model = None
        self.hf_tokenizer = None
        self.llama = None
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def initialize(self):
        """Initialize the embedding model."""
        try:
            await resource_manager.claim("embedding")
            self.logger.info(f"Initializing embedding backend: {self.backend}")
            loop = asyncio.get_event_loop()
            
            # Decide backend with robust fallback
            selected_backend = self.backend
            if selected_backend == 'llama_cpp':
                try:
                    from llama_cpp import Llama  # type: ignore
                except Exception as e:
                    self.logger.warning(f"llama_cpp not available ({e}). Falling back to sentence-transformers backend.")
                    selected_backend = 'sentence_transformers'

                model_path = getattr(settings, 'EMBEDDING_MODEL_PATH', '')
                if selected_backend == 'llama_cpp' and (not model_path or not os.path.exists(model_path)):
                    self.logger.warning("EMBEDDING_MODEL_PATH not set or missing. Falling back to sentence-transformers backend.")
                    selected_backend = 'sentence_transformers'

                if selected_backend == 'llama_cpp':
                    self.logger.info(f"Loading GGUF embedding model from: {model_path}")
                    def _load_llama():
                        return Llama(
                            model_path=model_path,
                            embedding=True,
                            n_gpu_layers=-1 if self.device == 'cuda' else 0,
                        )
                    self.llama = await loop.run_in_executor(self.executor, _load_llama)
                    self.backend = 'llama_cpp'
                    # Test
                    test_vec = await self._encode_text(["test text for embedding"])
                    self.logger.info(f"llama.cpp embedding initialized. Dim: {len(test_vec[0])}")
                else:
                    selected_backend = 'sentence_transformers'

            if selected_backend == 'hf':
                # Hugging Face Transformers backend for encoder embeddings
                self.logger.info(f"Initializing HF embedding model: {settings.EMBEDDING_MODEL} on {self.device}")
                def _load_hf():
                    from transformers import AutoModel, AutoTokenizer  # type: ignore
                    tok = AutoTokenizer.from_pretrained(settings.EMBEDDING_MODEL, trust_remote_code=True)
                    mdl = AutoModel.from_pretrained(settings.EMBEDDING_MODEL, trust_remote_code=True, device_map='auto' if self.device=='cuda' else None)
                    return tok, mdl
                self.hf_tokenizer, self.hf_model = await loop.run_in_executor(self.executor, _load_hf)
                self.backend = 'hf'
                test_embedding = await self._encode_text(["test"])
                self.logger.info(f"HF embedding model initialized. Dim: {len(test_embedding[0])}")
            elif selected_backend == 'sentence_transformers':
                # Default or fallback to Sentence-Transformers
                self.logger.info(f"Initializing embedding model: {settings.EMBEDDING_MODEL} on {self.device}")
                def _load_st():
                    from sentence_transformers import SentenceTransformer  # lazy import
                    return SentenceTransformer(settings.EMBEDDING_MODEL, device=self.device)
                self.model = await loop.run_in_executor(self.executor, _load_st)
                self.backend = 'sentence_transformers'
                test_embedding = await self._encode_text(["test"])
                self.logger.info(f"Model initialized. Embedding dim: {len(test_embedding[0])}")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise
    
    @log_execution_time
    async def generate_embeddings(self, chunks: List[TextChunk]) -> List[EmbeddingResult]:
        """Generate embeddings for text chunks"""
        if self.backend == 'llama_cpp':
            if not self.llama:
                await self.initialize()
        else:
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
        # llama.cpp backend
        if self.backend == 'llama_cpp':
            try:
                # Try OpenAI-compatible API
                if hasattr(self.llama, 'create_embedding'):
                    resp = self.llama.create_embedding(input=texts)  # type: ignore
                    vectors = [np.array(item["embedding"], dtype=np.float32) for item in resp["data"]]
                else:
                    # Fallback to per-text embed
                    vectors = [np.array(self.llama.embed(t), dtype=np.float32) for t in texts]  # type: ignore
                # Normalize
                vectors = [v / (np.linalg.norm(v) + 1e-12) for v in vectors]
                return np.stack(vectors)
            except Exception as e:
                self.logger.error(f"llama.cpp embedding failed: {str(e)}")
                raise
        
        # HF Transformers encoder backend (AutoModel)
        if self.backend == 'hf':
            with torch.no_grad():
                inputs = self.hf_tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=1024)
                device = next(self.hf_model.parameters()).device if hasattr(self.hf_model, 'parameters') else torch.device('cpu')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = self.hf_model(**inputs)
                # Try standard pooling: mean of last_hidden_state masked by attention mask
                last_hidden = outputs.last_hidden_state
                mask = inputs.get('attention_mask', torch.ones_like(last_hidden[:, :, 0]))
                mask = mask.unsqueeze(-1).expand(last_hidden.size())
                summed = torch.sum(last_hidden * mask, dim=1)
                counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                embeddings = (summed / counts).cpu().numpy().astype(np.float32)
                # Normalize
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
                embeddings = embeddings / norms
                return embeddings

        # Sentence-Transformers backend
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor,
            lambda: self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
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
        if self.backend == 'llama_cpp':
            if not self.llama:
                await self.initialize()
        else:
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
        """Releases the embedding model from memory."""
        if not self._initialized:
            return
        self.logger.info("Cleaning up EmbeddingService resources...")
        try:
            del self.model
            del self.hf_tokenizer
            del self.llama
            self.model = None
            self.hf_tokenizer = None
            self.llama = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            self.logger.info("EmbeddingService resources cleaned up successfully.")
        except Exception as e:
            self.logger.error(f"Error during EmbeddingService cleanup: {e}", exc_info=True)
        finally:
            self._initialized = False


# Global embedding service instance
embedding_service = EmbeddingService()
try:
    resource_manager.register("embedding", embedding_service.cleanup)
except Exception:
    pass

if __name__ =="__main__":
    async def test_embedding_service():
        try:
            # Initialize the embedding service
            await embedding_service.initialize()
            
            # Test embeddings generation
            test_chunks = [TextChunk(
                id="test-chunk-1",
                text="This is a test chunk",
                metadata={"test": True, "source": "test"},
                chunk_index=0,
                source_file="test.pdf"
            )]
            embeddings = await embedding_service.generate_embeddings(test_chunks)
            assert len(embeddings) == 1, "Expected 1 embedding for 1 chunk"
            # Dimension check uses configured vector size when available
            if hasattr(settings, 'VECTOR_SIZE'):
                assert len(embeddings[0].embedding) == settings.VECTOR_SIZE, f"Expected embedding dim {settings.VECTOR_SIZE}"
            
            # Test batch similarity search
            d = getattr(settings, 'VECTOR_SIZE', 384)
            query_embeddings = [np.random.rand(d) for _ in range(2)]  # Mock query embeddings
            chunk_embeddings = [np.random.rand(d) for _ in range(5)]  # Mock chunk embeddings
            results = await embedding_service.batch_similarity_search(query_embeddings, chunk_embeddings, top_k=3)
            assert len(results) == 2, "Expected 2 sets of results for 2 query embeddings"
            assert all(len(result) == 3 for result in results), "Expected top 3 similar chunks for each query"
            
            print("All tests passed successfully.")
        except Exception as e:
            print(f"Test failed: {str(e)}")
    
    asyncio.run(test_embedding_service())