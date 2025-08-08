# services/reranker.py
import asyncio
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor
import os 
import sys 


SCR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCR_DIR)))

from src.config.config_settings import settings
from src.config.logging_config import get_logger, log_execution_time, CustomLoggerTracker
from src.models.schemas import SearchResult, RAGResponse
from src.utilities.arabic_utils import arabic_processor


try:
    # from logger.custom_logger import CustomLoggerTracker
    custom = CustomLoggerTracker()
    logger = custom.get_logger("reranker")
    logger.info("Custom Logger Start Working.....")

except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("reranker")
    logger.info("Using standard logger - custom logger not available")


from src.config.config_settings import settings
from config.logging_config import get_logger, log_execution_time
from models.schemas import SearchResult
from src.utilities.arabic_utils import arabic_processor



class RerankerService:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    @log_execution_time
    async def rerank_results(
        self, 
        query: str, 
        search_results: List[SearchResult],
        method: str = "hybrid"
    ) -> List[SearchResult]:
        """Rerank search results using various methods"""
        
        if not search_results:
            return search_results
        
        if len(search_results) <= 1:
            return search_results
        
        try:
            self.logger.info(f"Reranking {len(search_results)} results using {method} method")
            
            if method == "bm25":
                return await self._bm25_rerank(query, search_results)
            elif method == "semantic":
                return await self._semantic_rerank(query, search_results)
            elif method == "hybrid":
                return await self._hybrid_rerank(query, search_results)
            elif method == "arabic_specific":
                return await self._arabic_specific_rerank(query, search_results)
            else:
                self.logger.warning(f"Unknown reranking method: {method}, using hybrid")
                return await self._hybrid_rerank(query, search_results)
                
        except Exception as e:
            self.logger.error(f"Error during reranking: {str(e)}")
            return search_results  # Return original results if reranking fails
    
    async def _bm25_rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank using BM25 algorithm"""
        try:
            # Prepare documents for BM25
            documents = [result.text for result in results]
            
            # Tokenize documents
            tokenized_docs = await self._tokenize_documents(documents)
            
            # Create BM25 index
            loop = asyncio.get_event_loop()
            bm25 = await loop.run_in_executor(
                self.executor,
                lambda: BM25Okapi(tokenized_docs)
            )
            
            # Tokenize query
            query_tokens = await self._tokenize_text(query)
            
            # Get BM25 scores
            bm25_scores = await loop.run_in_executor(
                self.executor,
                lambda: bm25.get_scores(query_tokens)
            )
            
            # Update results with BM25 scores
            for i, result in enumerate(results):
                result.rerank_score = float(bm25_scores[i])
            
            # Sort by BM25 score
            reranked_results = sorted(results, key=lambda x: x.rerank_score, reverse=True)
            
            self.logger.info("BM25 reranking completed")
            return reranked_results[:settings.FINAL_TOP_K]
            
        except Exception as e:
            self.logger.error(f"Error in BM25 reranking: {str(e)}")
            return results
    
    async def _semantic_rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank based on semantic similarity (using original scores)"""
        try:
            # For semantic reranking, we use the original similarity scores
            # but apply additional semantic analysis
            
            query_keywords = arabic_processor.extract_keywords(query, top_k=5)
            
            for result in results:
                # Extract keywords from result text
                result_keywords = arabic_processor.extract_keywords(result.text, top_k=10)
                
                # Calculate keyword overlap score
                keyword_overlap = len(set(query_keywords) & set(result_keywords))
                keyword_score = keyword_overlap / max(len(query_keywords), 1)
                
                # Combine original score with keyword score
                semantic_score = (result.score * 0.7) + (keyword_score * 0.3)
                result.rerank_score = semantic_score
            
            # Sort by semantic score
            reranked_results = sorted(results, key=lambda x: x.rerank_score, reverse=True)
            
            self.logger.info("Semantic reranking completed")
            return reranked_results[:settings.FINAL_TOP_K]
            
        except Exception as e:
            self.logger.error(f"Error in semantic reranking: {str(e)}")
            return results
    
    async def _hybrid_rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Hybrid reranking combining multiple methods"""
        try:
            # Get BM25 scores
            bm25_results = await self._bm25_rerank(query, results.copy())
            bm25_scores = {r.chunk_id: r.rerank_score for r in bm25_results}
            
            # Get semantic scores
            semantic_results = await self._semantic_rerank(query, results.copy())
            semantic_scores = {r.chunk_id: r.rerank_score for r in semantic_results}
            
            # Normalize scores to 0-1 range
            bm25_values = list(bm25_scores.values())
            semantic_values = list(semantic_scores.values())
            
            bm25_min, bm25_max = min(bm25_values), max(bm25_values)
            semantic_min, semantic_max = min(semantic_values), max(semantic_values)
            
            # Combine scores with weights
            for result in results:
                chunk_id = result.chunk_id
                
                # Normalize BM25 score
                bm25_norm = 0
                if bm25_max > bm25_min:
                    bm25_norm = (bm25_scores.get(chunk_id, 0) - bm25_min) / (bm25_max - bm25_min)
                
                # Normalize semantic score
                semantic_norm = 0
                if semantic_max > semantic_min:
                    semantic_norm = (semantic_scores.get(chunk_id, 0) - semantic_min) / (semantic_max - semantic_min)
                
                # Weighted combination
                hybrid_score = (
                    result.score * 0.4 +      # Original embedding similarity
                    bm25_norm * 0.35 +        # BM25 lexical matching
                    semantic_norm * 0.25      # Semantic keyword matching
                )
                
                result.rerank_score = hybrid_score
            
            # Sort by hybrid score
            reranked_results = sorted(results, key=lambda x: x.rerank_score, reverse=True)
            
            self.logger.info("Hybrid reranking completed")
            return reranked_results[:settings.FINAL_TOP_K]
            
        except Exception as e:
            self.logger.error(f"Error in hybrid reranking: {str(e)}")
            return results
    
    async def _arabic_specific_rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Arabic-specific reranking considering Arabic language features"""
        try:
            # Clean and normalize query
            normalized_query = arabic_processor.normalize_for_search(query)
            query_keywords = arabic_processor.extract_keywords(normalized_query, top_k=10)
            
            for result in results:
                # Calculate various Arabic-specific scores
                scores = await self._calculate_arabic_scores(
                    normalized_query, query_keywords, result
                )
                
                # Weighted combination of scores
                arabic_score = (
                    result.score * 0.3 +                    # Original similarity
                    scores["keyword_match"] * 0.25 +        # Keyword matching
                    scores["root_similarity"] * 0.2 +       # Root-based similarity
                    scores["context_relevance"] * 0.15 +    # Context relevance
                    scores["language_bonus"] * 0.1          # Arabic language bonus
                )
                
                result.rerank_score = arabic_score
            
            # Sort by Arabic-specific score
            reranked_results = sorted(results, key=lambda x: x.rerank_score, reverse=True)
            
            self.logger.info("Arabic-specific reranking completed")
            return reranked_results[:settings.FINAL_TOP_K]
            
        except Exception as e:
            self.logger.error(f"Error in Arabic-specific reranking: {str(e)}")
            return results
    
    async def _calculate_arabic_scores(
        self, 
        query: str, 
        query_keywords: List[str], 
        result: SearchResult
    ) -> Dict[str, float]:
        """Calculate Arabic-specific scoring metrics"""
        
        result_text_normalized = arabic_processor.normalize_for_search(result.text)
        result_keywords = arabic_processor.extract_keywords(result_text_normalized, top_k=15)
        
        scores = {}
        
        # 1. Keyword matching score
        keyword_matches = len(set(query_keywords) & set(result_keywords))
        scores["keyword_match"] = keyword_matches / max(len(query_keywords), 1)
        
        # 2. Root similarity (simplified - based on common 3-letter patterns)
        scores["root_similarity"] = await self._calculate_root_similarity(
            query_keywords, result_keywords
        )
        
        # 3. Context relevance (based on educational keywords)
        educational_keywords = [
            "درس", "تمرين", "سؤال", "جواب", "شرح", "مثال", "تعريف", "قاعدة", "نظرية",
            "مسألة", "حل", "طريقة", "أسلوب", "منهج", "كتاب", "فصل", "باب", "وحدة"
        ]
        
        educational_matches = sum(1 for keyword in educational_keywords 
                                if keyword in result_text_normalized)
        scores["context_relevance"] = min(educational_matches / 5.0, 1.0)
        
        # 4. Language bonus (prefer Arabic content)
        arabic_ratio = sum(1 for char in result.text if '\u0600' <= char <= '\u06ff') / max(len(result.text), 1)
        scores["language_bonus"] = arabic_ratio
        
        return scores
    
    async def _calculate_root_similarity(self, keywords1: List[str], keywords2: List[str]) -> float:
        """Calculate similarity based on Arabic root patterns"""
        try:
            # Simplified root extraction - look for common 3-letter patterns
            def extract_potential_roots(word):
                if len(word) < 3:
                    return []
                
                # Remove common prefixes and suffixes
                word = re.sub(r'^(ال|و|ف|ب|ل|ك)', '', word)
                word = re.sub(r'(ة|ه|ها|هم|هن|ك|كم|كن|ت|تم|تن|ي|ين|ون|وا)$', '', word)
                
                if len(word) >= 3:
                    return [word[:3]]
                return []
            
            roots1 = set()
            for keyword in keywords1:
                roots1.update(extract_potential_roots(keyword))
            
            roots2 = set()
            for keyword in keywords2:
                roots2.update(extract_potential_roots(keyword))
            
            if not roots1 or not roots2:
                return 0.0
            
            common_roots = len(roots1 & roots2)
            return common_roots / max(len(roots1), len(roots2))
            
        except Exception:
            return 0.0
    
    async def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for BM25"""
        # Clean and normalize
        cleaned_text = arabic_processor.normalize_for_search(text)
        
        # Remove stopwords
        language = arabic_processor.detect_language(text)
        without_stopwords = arabic_processor.remove_stopwords(cleaned_text, language)
        
        # Split into tokens
        tokens = without_stopwords.split()
        
        # Filter tokens (minimum length 2)
        filtered_tokens = [token for token in tokens if len(token) >= 2]
        
        return filtered_tokens
    
    async def _tokenize_documents(self, documents: List[str]) -> List[List[str]]:
        """Tokenize multiple documents"""
        tokenized_docs = []
        
        for doc in documents:
            tokens = await self._tokenize_text(doc)
            tokenized_docs.append(tokens)
        
        return tokenized_docs
    
    async def diversify_results(self, results: List[SearchResult], diversity_threshold: float = 0.8) -> List[SearchResult]:
        """Remove very similar results to increase diversity"""
        if len(results) <= 1:
            return results
        
        try:
            diversified = [results[0]]  # Always include the top result
            
            for candidate in results[1:]:
                is_diverse = True
                
                # Check similarity with already selected results
                for selected in diversified:
                    # Simple text similarity check
                    similarity = await self._calculate_text_similarity(
                        candidate.text, selected.text
                    )
                    
                    if similarity > diversity_threshold:
                        is_diverse = False
                        break
                
                if is_diverse:
                    diversified.append(candidate)
                
                # Stop if we have enough diverse results
                if len(diversified) >= settings.FINAL_TOP_K:
                    break
            
            self.logger.info(f"Diversified from {len(results)} to {len(diversified)} results")
            return diversified
            
        except Exception as e:
            self.logger.error(f"Error during diversification: {str(e)}")
            return results[:settings.FINAL_TOP_K]
    
    async def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        try:
            # Tokenize both texts
            tokens1 = set(await self._tokenize_text(text1))
            tokens2 = set(await self._tokenize_text(text2))
            
            if not tokens1 or not tokens2:
                return 0.0
            
            # Jaccard similarity
            intersection = len(tokens1 & tokens2)
            union = len(tokens1 | tokens2)
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    async def get_reranking_explanation(
        self, 
        query: str, 
        original_results: List[SearchResult],
        reranked_results: List[SearchResult]
    ) -> Dict[str, Any]:
        """Provide explanation for reranking decisions"""
        try:
            explanation = {
                "query": query,
                "original_count": len(original_results),
                "reranked_count": len(reranked_results),
                "method_used": "hybrid",
                "ranking_changes": []
            }
            
            # Track position changes
            original_positions = {r.chunk_id: i for i, r in enumerate(original_results)}
            
            for new_pos, result in enumerate(reranked_results):
                old_pos = original_positions.get(result.chunk_id, -1)
                
                change_info = {
                    "chunk_id": result.chunk_id,
                    "original_position": old_pos,
                    "new_position": new_pos,
                    "position_change": old_pos - new_pos if old_pos >= 0 else 0,
                    "original_score": result.score,
                    "rerank_score": getattr(result, 'rerank_score', result.score),
                    "text_preview": result.text[:100] + "..." if len(result.text) > 100 else result.text
                }
                
                explanation["ranking_changes"].append(change_info)
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error creating reranking explanation: {str(e)}")
            return {"error": str(e)}
    
    def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)

# Global reranker service instance
reranker_service = RerankerService()