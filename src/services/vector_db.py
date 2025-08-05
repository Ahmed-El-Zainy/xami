# # services/vector_db.py
# import asyncio
# from typing import List, Dict, Any, Optional, Union
# import chromadb
# from chromadb.config import Settings as ChromaSettings
# from chromadb.utils import embedding_functions
# import numpy as np
# import uuid
# import json

# from config.settings import settings
# from config.logging_config import get_logger, log_execution_time
# from models.schemas import EmbeddingResult, SearchResult

# class VectorDBService:
#     def __init__(self):
#         self.logger = get_logger(__name__)
#         self.client = None
#         self.collection = None
#         self.collection_name = settings.CHROMA_COLLECTION_NAME
        
#     async def initialize(self):
#         """Initialize ChromaDB client and create collection if needed"""
#         try:
#             self.logger.info(f"Initializing ChromaDB with persist directory: {settings.CHROMA_PERSIST_DIR}")
            
#             # Initialize ChromaDB client with persistence
#             self.client = chromadb.PersistentClient(
#                 path=settings.CHROMA_PERSIST_DIR,
#                 settings=ChromaSettings(
#                     anonymized_telemetry=False,
#                     allow_reset=True
#                 )
#             )
            
#             # Try to get existing collection or create new one
#             try:
#                 self.collection = self.client.get_collection(
#                     name=self.collection_name
#                 )
#                 self.logger.info(f"Found existing collection '{self.collection_name}'")
#             except Exception:
#                 # Collection doesn't exist, create it
#                 await self._create_collection()
            
#             # Get collection info
#             count = self.collection.count()
#             self.logger.info(f"Collection '{self.collection_name}' has {count} documents")
            
#         except Exception as e:
#             self.logger.error(f"Failed to initialize ChromaDB: {str(e)}")
#             raise
    
#     async def _create_collection(self):
#         """Create a new collection"""
#         try:
#             self.logger.info(f"Creating collection '{self.collection_name}'")
            
#             # Create collection without embedding function (we'll provide embeddings)
#             self.collection = self.client.create_collection(
#                 name=self.collection_name,
#                 metadata={
#                     "description": "Arabic educational content for secondary schools",
#                     "embedding_model": settings.EMBEDDING_MODEL,
#                     "vector_size": settings.VECTOR_SIZE,
#                     "created_by": "RAG Pipeline"
#                 }
#             )
            
#             self.logger.info(f"Collection '{self.collection_name}' created successfully")
            
#         except Exception as e:
#             self.logger.error(f"Failed to create collection: {str(e)}")
#             raise
    
#     @log_execution_time
#     async def add_embeddings(self, embeddings: List[EmbeddingResult]) -> bool:
#         """Add embeddings to the vector database"""
#         if not self.client or not self.collection:
#             await self.initialize()
        
#         if not embeddings:
#             self.logger.warning("No embeddings to add")
#             return True
        
#         try:
#             self.logger.info(f"Adding {len(embeddings)} embeddings to ChromaDB")
            
#             # Prepare data for ChromaDB
#             ids = []
#             documents = []
#             metadatas = []
#             embedding_vectors = []
            
#             for embedding_result in embeddings:
#                 # Use chunk_id as the document ID, but ensure it's unique
#                 doc_id = f"{embedding_result.chunk_id}_{uuid.uuid4().hex[:8]}"
#                 ids.append(doc_id)
                
#                 # Document text
#                 documents.append(embedding_result.text)
                
#                 # Metadata (ChromaDB requires JSON-serializable values)
#                 metadata = self._prepare_metadata(embedding_result.metadata)
#                 metadata["chunk_id"] = embedding_result.chunk_id
#                 metadatas.append(metadata)
                
#                 # Embedding vector
#                 embedding_vectors.append(embedding_result.embedding)
            
#             # Add to collection in batches
#             batch_size = 100
#             for i in range(0, len(ids), batch_size):
#                 batch_end = min(i + batch_size, len(ids))
#                 self.logger.info(f"Adding batch {i//batch_size + 1}/{(len(ids) + batch_size - 1)//batch_size}")
                
#                 self.collection.add(
#                     ids=ids[i:batch_end],
#                     documents=documents[i:batch_end],
#                     metadatas=metadatas[i:batch_end],
#                     embeddings=embedding_vectors[i:batch_end]
#                 )
            
#             # Get updated count
#             total_count = self.collection.count()
#             self.logger.info(f"Successfully added embeddings. Total documents in collection: {total_count}")
            
#             return True
            
#         except Exception as e:
#             self.logger.error(f"Error adding embeddings: {str(e)}")
#             raise
    
#     def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
#         """Prepare metadata for ChromaDB (must be JSON serializable)"""
#         prepared = {}
        
#         for key, value in metadata.items():
#             if isinstance(value, (str, int, float, bool)):
#                 prepared[key] = value
#             elif isinstance(value, list):
#                 # Convert lists to JSON strings
#                 prepared[key] = json.dumps(value)
#             elif isinstance(value, dict):
#                 # Convert dicts to JSON strings
#                 prepared[key] = json.dumps(value)
#             else:
#                 # Convert other types to string
#                 prepared[key] = str(value)
        
#         return prepared
    
#     @log_execution_time
#     async def search_similar(
#         self,
#         query_vector: List[float],
#         limit: int = 10,
#         score_threshold: float = 0.7,
#         filters: Optional[Dict[str, Any]] = None
#     ) -> List[SearchResult]:
#         """Search for similar vectors"""
#         if not self.client or not self.collection:
#             await self.initialize()
        
#         try:
#             # Prepare where clause for filtering
#             where_clause = None
#             if filters:
#                 where_clause = self._prepare_where_clause(filters)
            
#             # Perform similarity search
#             results = self.collection.query(
#                 query_embeddings=[query_vector],
#                 n_results=limit,
#                 where=where_clause,
#                 include=["documents", "metadatas", "distances"]
#             )
            
#             # Convert to SearchResult objects
#             search_results = []
#             if results["ids"] and results["ids"][0]:  # Check if we have results
#                 for i, doc_id in enumerate(results["ids"][0]):
#                     # Convert distance to similarity score (ChromaDB returns distances)
#                     distance = results["distances"][0][i]
#                     similarity_score = 1.0 / (1.0 + distance)  # Convert distance to similarity
                    
#                     # Apply score threshold
#                     if similarity_score >= score_threshold:
#                         metadata = results["metadatas"][0][i]
                        
#                         # Parse JSON strings back to objects if needed
#                         parsed_metadata = self._parse_metadata(metadata)
                        
#                         result = SearchResult(
#                             chunk_id=metadata.get("chunk_id", doc_id),
#                             text=results["documents"][0][i],
#                             score=similarity_score,
#                             metadata=parsed_metadata
#                         )
#                         search_results.append(result)
            
#             self.logger.info(f"Found {len(search_results)} similar results above threshold {score_threshold}")
#             return search_results
            
#         except Exception as e:
#             self.logger.error(f"Error searching similar vectors: {str(e)}")
#             return []
    
#     def _prepare_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
#         """Prepare where clause for ChromaDB filtering"""
#         where_clause = {}
        
#         for key, value in filters.items():
#             if isinstance(value, list):
#                 # Use $in operator for list values
#                 where_clause[key] = {"$in": value}
#             else:
#                 # Direct equality
#                 where_clause[key] = value
        
#         return where_clause
    
#     def _parse_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
#         """Parse metadata back from JSON strings if needed"""
#         parsed = {}
        
#         for key, value in metadata.items():
#             if key == "chunk_id":
#                 parsed[key] = value
#                 continue
                
#             if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
#                 try:
#                     # Try to parse as JSON
#                     parsed[key] = json.loads(value)
#                 except json.JSONDecodeError:
#                     parsed[key] = value
#             else:
#                 parsed[key] = value
        
#         return parsed
    
#     async def search_with_filters(
#         self,
#         query_vector: List[float],
#         limit: int = 10,
#         where: Optional[Dict[str, Any]] = None,
#         where_document: Optional[Dict[str, Any]] = None
#     ) -> List[SearchResult]:
#         """Advanced search with document content filtering"""
#         if not self.client or not self.collection:
#             await self.initialize()
        
#         try:
#             results = self.collection.query(
#                 query_embeddings=[query_vector],
#                 n_results=limit,
#                 where=where,
#                 where_document=where_document,
#                 include=["documents", "metadatas", "distances"]
#             )
            
#             # Convert to SearchResult objects
#             search_results = []
#             if results["ids"] and results["ids"][0]:
#                 for i, doc_id in enumerate(results["ids"][0]):
#                     distance = results["distances"][0][i]
#                     similarity_score = 1.0 / (1.0 + distance)
                    
#                     metadata = results["metadatas"][0][i]
#                     parsed_metadata = self._parse_metadata(metadata)
                    
#                     result = SearchResult(
#                         chunk_id=metadata.get("chunk_id", doc_id),
#                         text=results["documents"][0][i],
#                         score=similarity_score,
#                         metadata=parsed_metadata
#                     )
#                     search_results.append(result)
            
#             return search_results
            
#         except Exception as e:
#             self.logger.error(f"Error in advanced search: {str(e)}")
#             return []
    
#     async def delete_by_source(self, source_file: str) -> bool:
#         """Delete all vectors from a specific source file"""
#         if not self.client or not self.collection:
#             await self.initialize()
        
#         try:
#             self.logger.info(f"Deleting vectors from source: {source_file}")
            
#             # First, get all IDs for this source file
#             results = self.collection.get(
#                 where={"source_file": source_file},
#                 include=["metadatas"]
#             )
            
#             if results["ids"]:
#                 # Delete all documents from this source
#                 self.collection.delete(
#                     where={"source_file": source_file}
#                 )
                
#                 deleted_count = len(results["ids"])
#                 self.logger.info(f"Successfully deleted {deleted_count} vectors from {source_file}")
#                 return True
#             else:
#                 self.logger.info(f"No vectors found for source file: {source_file}")
#                 return True
            
#         except Exception as e:
#             self.logger.error(f"Error deleting vectors: {str(e)}")
#             return False
    
#     async def get_collection_stats(self) -> Dict[str, Any]:
#         """Get collection statistics"""
#         if not self.client or not self.collection:
#             await self.initialize()
        
#         try:
#             count = self.collection.count()
#             metadata = self.collection.metadata
            
#             # Get sample to understand data distribution
#             sample_results = self.collection.peek(limit=5)
            
#             stats = {
#                 "collection_name": self.collection_name,
#                 "total_documents": count,
#                 "collection_metadata": metadata,
#                 "persist_directory": settings.CHROMA_PERSIST_DIR,
#                 "sample_count": len(sample_results["ids"]) if sample_results["ids"] else 0
#             }
            
#             # Add language distribution if we have documents
#             if count > 0:
#                 try:
#                     # Get all documents to analyze language distribution
#                     all_results = self.collection.get(include=["metadatas"])
#                     language_counts = {}
                    
#                     for metadata in all_results["metadatas"]:
#                         lang = metadata.get("language", "unknown")
#                         language_counts[lang] = language_counts.get(lang, 0) + 1
                    
#                     stats["language_distribution"] = language_counts
#                 except Exception as e:
#                     self.logger.warning(f"Could not get language distribution: {str(e)}")
            
#             return stats
            
#         except Exception as e:
#             self.logger.error(f"Error getting collection stats: {str(e)}")
#             return {}
    
#     async def update_document(
#         self, 
#         doc_id: str, 
#         document: str, 
#         metadata: Dict[str, Any], 
#         embedding: List[float]
#     ) -> bool:
#         """Update a specific document"""
#         if not self.client or not self.collection:
#             await self.initialize()
        
#         try:
#             prepared_metadata = self._prepare_metadata(metadata)
            
#             self.collection.upsert(
#                 ids=[doc_id],
#                 documents=[document],
#                 metadatas=[prepared_metadata],
#                 embeddings=[embedding]
#             )
            
#             self.logger.info(f"Successfully updated document: {doc_id}")
#             return True
            
#         except Exception as e:
#             self.logger.error(f"Error updating document: {str(e)}")
#             return False
    
#     async def health_check(self) -> bool:
#         """Check if ChromaDB is healthy"""
#         try:
#             if not self.client:
#                 return False
            
#             # Try to get collection info
#             if self.collection:
#                 count = self.collection.count()
#                 self.logger.debug(f"Health check passed. Collection has {count} documents")
#                 return True
#             else:
#                 return False
            
#         except Exception as e:
#             self.logger.error(f"Health check failed: {str(e)}")
#             return False
    
#     async def reset_collection(self) -> bool:
#         """Reset (delete and recreate) the collection"""
#         try:
#             if self.client:
#                 # Delete existing collection
#                 try:
#                     self.client.delete_collection(name=self.collection_name)
#                     self.logger.info(f"Deleted existing collection: {self.collection_name}")
#                 except Exception:
#                     self.logger.info("No existing collection to delete")
                
#                 # Create new collection
#                 await self._create_collection()
#                 return True
            
#             return False
            
#         except Exception as e:
#             self.logger.error(f"Error resetting collection: {str(e)}")
#             return False
    
#     def cleanup(self):
#         """Cleanup resources"""
#         try:
#             if self.client:
#                 # ChromaDB automatically persists data
#                 self.logger.info("ChromaDB cleanup completed")
#         except Exception as e:
#             self.logger.error(f"Error during cleanup: {str(e)}")

# # Global vector database service instance
# vector_db_service = VectorDBService()


