# services/text_processor.py
import uuid
import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
import os 
import sys 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.config.logging_config import get_logger, log_execution_time, CustomLoggerTracker
from src.config.config_settings import settings
from src.models.schemas import TextChunk, EmbeddingResult
from src.utilities.arabic_utils import arabic_processor


try:
    # from logger.custom_logger import CustomLoggerTracker
    custom = CustomLoggerTracker()
    logger = custom.get_logger("text_processor")
    logger.info("Custom Logger Start Working.....")

except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("text_processor")
    logger.info("Using standard logger - custom logger not available")



from src.config.config_settings import settings
from src.config.logging_config import get_logger, log_execution_time
from src.models.schemas import TextChunk
from src.utilities.arabic_utils import arabic_processor

class TextProcessor:
    def __init__(self):
        self.logger = get_logger(__name__)
    
    @log_execution_time
    async def process_and_chunk_text(
        self, 
        text: str, 
        source_file: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextChunk]:
        """Process text and split into chunks"""
        
        if not text or len(text.strip()) < settings.MIN_CHUNK_SIZE:
            self.logger.warning(f"Text too short for processing: {len(text)} characters")
            return []
        
        try:
            self.logger.info(f"Processing text from {source_file}")
            
            # Clean the text
            cleaned_text = self._clean_text(text)
            
            # Get text statistics
            text_stats = arabic_processor.get_text_statistics(cleaned_text)
            self.logger.info(f"Text stats: {text_stats}")
            
            # Split into chunks
            chunks = await self._split_text_into_chunks(cleaned_text)
            
            # Create TextChunk objects
            text_chunks = []
            base_metadata = metadata or {}
            base_metadata.update({
                "source_file": source_file,
                "text_stats": text_stats,
                "processing_method": "semantic_chunking"
            })
            
            for i, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) >= settings.MIN_CHUNK_SIZE:
                    chunk_id = str(uuid.uuid4())
                    
                    # Extract keywords for this chunk
                    keywords = arabic_processor.extract_keywords(chunk_text, top_k=5)
                    
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata.update({
                        "chunk_length": len(chunk_text),
                        "keywords": keywords,
                        "language": arabic_processor.detect_language(chunk_text),
                        "chunk_number": i + 1,
                        "total_chunks": len(chunks)
                    })
                    
                    text_chunk = TextChunk(
                        id=chunk_id,
                        text=chunk_text.strip(),
                        metadata=chunk_metadata,
                        chunk_index=i,
                        source_file=source_file
                    )
                    
                    text_chunks.append(text_chunk)
            
            self.logger.info(f"Created {len(text_chunks)} chunks from {source_file}")
            
            # Save chunks to files
            await self._save_chunks_to_files(text_chunks, source_file)
            
            return text_chunks
            
        except Exception as e:
            self.logger.error(f"Error processing text: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'\n\s*صفحة\s*\d+\s*\n', '\n', text)
        text = arabic_processor.clean_arabic_text(text)
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if len(line.strip()) > 5]
        return '\n'.join(cleaned_lines).strip()
    
    async def _split_text_into_chunks(self, text: str) -> List[str]:
        # First, try to split by natural boundaries
        chunks = await self._semantic_chunking(text)
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= settings.CHUNK_SIZE:
                final_chunks.append(chunk)
            else:
                sub_chunks = self._sliding_window_split(chunk)
                final_chunks.extend(sub_chunks)
        return final_chunks
    

    async def _semantic_chunking(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = ""
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > settings.CHUNK_SIZE:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    sentences = arabic_processor.segment_sentences(paragraph)
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) > settings.CHUNK_SIZE:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence
                        else:
                            current_chunk += " " + sentence if current_chunk else sentence
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    

    def _sliding_window_split(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 10
        words_per_chunk = int(settings.CHUNK_SIZE / avg_word_length)
        overlap_words = int(settings.CHUNK_OVERLAP / avg_word_length)
        start = 0
        while start < len(words):
            end = min(start + words_per_chunk, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            if len(chunk_text.strip()) >= settings.MIN_CHUNK_SIZE:
                chunks.append(chunk_text)
            if end >= len(words):
                break
            start = end - overlap_words if overlap_words > 0 else end
        return chunks
    

    async def _save_chunks_to_files(self, chunks: List[TextChunk], source_file: str):
        try:
            doc_name = Path(source_file).stem
            chunks_dir = Path(settings.CHUNKS_DIR) / doc_name
            chunks_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each chunk
            for chunk in chunks:
                chunk_file = chunks_dir / f"chunk_{chunk.chunk_index:04d}.json"
                
                chunk_data = {
                    "id": chunk.id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "chunk_index": chunk.chunk_index,
                    "source_file": chunk.source_file
                }
                
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk_data, f, ensure_ascii=False, indent=2)
            
            # Save summary file
            summary_file = chunks_dir / "summary.json"
            summary_data = {
                "source_file": source_file,
                "total_chunks": len(chunks),
                "chunk_files": [f"chunk_{i:04d}.json" for i in range(len(chunks))],
                "processing_settings": {
                    "chunk_size": settings.CHUNK_SIZE,
                    "chunk_overlap": settings.CHUNK_OVERLAP,
                    "min_chunk_size": settings.MIN_CHUNK_SIZE
                }
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Saved {len(chunks)} chunks to {chunks_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving chunks: {str(e)}")
            raise
    
    async def load_chunks_from_files(self, source_file: str) -> List[TextChunk]:
        """Load chunks from saved files"""
        try:
            doc_name = Path(source_file).stem
            chunks_dir = Path(settings.CHUNKS_DIR) / doc_name
            
            if not chunks_dir.exists():
                self.logger.warning(f"Chunks directory not found: {chunks_dir}")
                return []
            
            summary_file = chunks_dir / "summary.json"
            if not summary_file.exists():
                self.logger.warning(f"Summary file not found: {summary_file}")
                return []
            
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            chunks = []
            for chunk_file_name in summary["chunk_files"]:
                chunk_file = chunks_dir / chunk_file_name
                if chunk_file.exists():
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk_data = json.load(f)
                    
                    chunk = TextChunk(
                        id=chunk_data["id"],
                        text=chunk_data["text"],
                        metadata=chunk_data["metadata"],
                        chunk_index=chunk_data["chunk_index"],
                        source_file=chunk_data["source_file"]
                    )
                    chunks.append(chunk)
            
            self.logger.info(f"Loaded {len(chunks)} chunks from {chunks_dir}")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error loading chunks: {str(e)}")
            return []
    
    async def get_chunk_statistics(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Get statistics about the chunks"""
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk.text) for chunk in chunks]
        languages = [chunk.metadata.get("language", "unknown") for chunk in chunks]
        
        stats = {
            "total_chunks": len(chunks),
            "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "total_text_length": sum(chunk_lengths),
            "language_distribution": {}
        }
        
        # Count languages
        for lang in languages:
            stats["language_distribution"][lang] = stats["language_distribution"].get(lang, 0) + 1
        
        return stats


if __name__=="__main__":
    # Global text processor instance
    text_processor = TextProcessor()