# services/vector_db.py
import asyncio
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, CreateCollection, PointStruct,
    Filter, FieldCondition, MatchValue, SearchRequest
)
import numpy as np
import uuid
import os 
import sys 


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config_settings import settings
from src.config.logging_config import get_logger, log_execution_time, CustomLoggerTracker
from src.models.schemas import SearchResult, RAGResponse
from src.utilities.arabic_utils import arabic_processor


try:
    # from logger.custom_logger import CustomLoggerTracker
    custom = CustomLoggerTracker()
    logger = custom.get_logger("vector_db")
    logger.info("Custom Logger Start Working.....")

except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("vector_db")
    logger.info("Using standard logger - custom logger not available")



from src.config.config_settings import settings
from config.logging_config import get_logger, log_execution_time
from models.schemas import EmbeddingResult, SearchResult

class VectorDBService:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.client = None
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        
    async def initialize(self):
        """Initialize Qdrant client and create collection if needed"""
        try:
            self.logger.info(f"Connecting to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
            
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=30
            )
            
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(col.name == self.collection_name for col in collections)
            
            if not collection_exists:
                await self._create_collection()
            else:
                self.logger.info(f"Collection '{self.collection_name}' already exists")
            
            # Get collection info
            collection_info = self.client.get_collection(self.collection_name)
            self.logger.info(f"Collection info: {collection_info}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Qdrant: {str(e)}")
            raise
    
    async def _create_collection(self):
        """Create a new collection with appropriate settings"""
        try:
            self.logger.info(f"Creating collection '{self.collection_name}'")
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.VECTOR_SIZE,
                    distance=Distance.COSINE,  # Best for normalized embeddings
                    on_disk=True  # Store vectors on disk for large datasets
                ),
                # Optimize for Arabic content
                optimizers_config=models.OptimizersConfig(
                    deleted_threshold=0.2,
                    vacuum_min_vector_number=1000,  # Suitable for large documents
                    default_segment_number=4,  # Good for ~500 page books
                ),
                # HNSW configuration for large datasets
                hnsw_config=models.HnswConfig(
                    m=16,  # Number of connections
                    ef_construct=200,  # Construction parameter
                    full_scan_threshold=10000,  # When to use full scan
                    max_indexing_threads=0  # Use all available threads
                )
            )
            
            self.logger.info(f"Collection '{self.collection_name}' created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create collection: {str(e)}")
            raise
    
    @log_execution_time
    async def add_embeddings(self, embeddings: List[EmbeddingResult]) -> bool:
        """Add embeddings to the vector database"""
        if not self.client:
            await self.initialize()
        
        if not embeddings:
            self.logger.warning("No embeddings to add")
            return True
        
        try:
            self.logger.info(f"Adding {len(embeddings)} embeddings to vector database")
            
            # Prepare points for insertion
            points = []
            for embedding_result in embeddings:
                # Create payload with metadata
                payload = {
                    "text": embedding_result.text,
                    "chunk_id": embedding_result.chunk_id,
                    **embedding_result.metadata
                }
                
                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),  # Qdrant point ID
                    vector=embedding_result.embedding,
                    payload=payload
                )
                points.append(point)
            
            # Insert points in batches
            batch_size = 100  # Adjust based on performance
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.logger.info(f"Inserting batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}")
                
                operation_info = self.client.upsert(
                    collection_name=self.collection_name,
                    wait=True,
                    points=batch
                )
                
                if operation_info.status != models.UpdateStatus.COMPLETED:
                    self.logger.error(f"Batch insertion failed: {operation_info}")
                    return False
            
            # Get collection stats
            collection_info = self.client.get_collection(self.collection_name)
            self.logger.info(f"Total vectors in collection: {collection_info.points_count}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding embeddings: {str(e)}")
            raise
    
    @log_execution_time
    async def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors"""
        if not self.client:
            await self.initialize()
        
        try:
            # Prepare filter if provided
            query_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        # Multiple values - use should condition
                        should_conditions = [
                            FieldCondition(key=key, match=MatchValue(value=v))
                            for v in value
                        ]
                        conditions.append(models.Filter(should=should_conditions))
                    else:
                        # Single value
                        conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )
                
                if conditions:
                    query_filter = models.Filter(must=conditions)
            
            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False  # We don't need vectors in response
            )
            
            # Convert to SearchResult objects
            results = []
            for scored_point in search_result:
                result = SearchResult(
                    chunk_id=scored_point.payload.get("chunk_id", ""),
                    text=scored_point.payload.get("text", ""),
                    score=float(scored_point.score),
                    metadata={k: v for k, v in scored_point.payload.items() 
                             if k not in ["text", "chunk_id"]}
                )
                results.append(result)
            
            self.logger.info(f"Found {len(results)} similar results")
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching similar vectors: {str(e)}")
            return []
    
    async def search_with_scroll(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        offset: Optional[str] = None
    ) -> tuple[List[SearchResult], Optional[str]]:
        """Search with pagination support"""
        if not self.client:
            await self.initialize()
        
        try:
            # This is a simplified version - Qdrant has more advanced pagination
            results = await self.search_similar(query_vector, limit, filters=filters)
            return results, None  # No pagination in this simple implementation
            
        except Exception as e:
            self.logger.error(f"Error in paginated search: {str(e)}")
            return [], None
    
    async def delete_by_source(self, source_file: str) -> bool:
        """Delete all vectors from a specific source file"""
        if not self.client:
            await self.initialize()
        
        try:
            self.logger.info(f"Deleting vectors from source: {source_file}")
            
            # Create filter for source file
            filter_condition = models.Filter(
                must=[
                    FieldCondition(
                        key="source_file",
                        match=MatchValue(value=source_file)
                    )
                ]
            )
            
            # Delete points
            operation_info = self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=filter_condition),
                wait=True
            )
            
            success = operation_info.status == models.UpdateStatus.COMPLETED
            if success:
                self.logger.info(f"Successfully deleted vectors from {source_file}")
            else:
                self.logger.error(f"Failed to delete vectors: {operation_info}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error deleting vectors: {str(e)}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self.client:
            await self.initialize()
        
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            stats = {
                "collection_name": self.collection_name,
                "points_count": collection_info.points_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.value,
                "status": collection_info.status.value
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {str(e)}")
            return {}
    
    async def create_index(self, field_name: str) -> bool:
        """Create an index on a payload field for faster filtering"""
        if not self.client:
            await self.initialize()
        
        try:
            self.logger.info(f"Creating index on field: {field_name}")
            
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True
            )
            
            self.logger.info(f"Index created successfully on {field_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating index: {str(e)}")
            return False
    
    async def health_check(self) -> bool:
        """Check if Qdrant is healthy"""
        try:
            if not self.client:
                return False
            
            # Try to get collection info
            self.client.get_collection(self.collection_name)
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False
    
    async def get_points_by_filter(
        self,
        filters: Dict[str, Any],
        limit: int = 100,
        with_payload: bool = True
    ) -> List[Dict[str, Any]]:
        """Get points by filter criteria"""
        if not self.client:
            await self.initialize()
        
        try:
            # Create filter conditions
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            
            query_filter = models.Filter(must=conditions) if conditions else None
            
            # Scroll through points
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=limit,
                with_payload=with_payload,
                with_vectors=False
            )
            
            return [
                {
                    "id": point.id,
                    "payload": point.payload if with_payload else {}
                }
                for point in points
            ]
            
        except Exception as e:
            self.logger.error(f"Error getting points by filter: {str(e)}")
            return []
    
    async def update_payload(self, point_id: str, payload_updates: Dict[str, Any]) -> bool:
        """Update payload for a specific point"""
        if not self.client:
            await self.initialize()
        
        try:
            operation_info = self.client.set_payload(
                collection_name=self.collection_name,
                payload=payload_updates,
                points=[point_id],
                wait=True
            )
            
            return operation_info.status == models.UpdateStatus.COMPLETED
            
        except Exception as e:
            self.logger.error(f"Error updating payload: {str(e)}")
            return False
    
    async def backup_collection(self, backup_path: str) -> bool:
        """Create a snapshot of the collection"""
        try:
            snapshot_info = self.client.create_snapshot(
                collection_name=self.collection_name
            )
            
            self.logger.info(f"Snapshot created: {snapshot_info}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {str(e)}")
            return False
    
    async def optimize_collection(self) -> bool:
        """Optimize the collection for better performance"""
        if not self.client:
            await self.initialize()
        
        try:
            # Force indexing optimization
            operation_info = self.client.update_collection(
                collection_name=self.collection_name,
                optimizer_config=models.OptimizersConfig(
                    deleted_threshold=0.2,
                    vacuum_min_vector_number=1000,
                    default_segment_number=4
                )
            )
            
            self.logger.info("Collection optimization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error optimizing collection: {str(e)}")
            return False
    
    async def get_vector_by_id(self, point_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific vector by its ID"""
        if not self.client:
            await self.initialize()
        
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=True
            )
            
            if points:
                point = points[0]
                return {
                    "id": point.id,
                    "vector": point.vector,
                    "payload": point.payload
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving vector: {str(e)}")
            return None
    
    async def count_points(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count points in collection with optional filters"""
        if not self.client:
            await self.initialize()
        
        try:
            query_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                query_filter = models.Filter(must=conditions)
            
            count_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=query_filter,
                exact=True
            )
            
            return count_result.count
            
        except Exception as e:
            self.logger.error(f"Error counting points: {str(e)}")
            return 0
    
    async def clear_collection(self) -> bool:
        """Clear all points from the collection"""
        if not self.client:
            await self.initialize()
        
        try:
            self.logger.warning("Clearing all points from collection")
            
            operation_info = self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(must=[])  # Empty filter matches all
                ),
                wait=True
            )
            
            success = operation_info.status == models.UpdateStatus.COMPLETED
            if success:
                self.logger.info("Collection cleared successfully")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error clearing collection: {str(e)}")
            return False
    
    def cleanup(self):
        """Cleanup resources"""
        if self.client:
            self.client.close()
            self.client = None


if __name__=="__main__":
    # Global vector database service instance
    vector_db_service = VectorDBService()