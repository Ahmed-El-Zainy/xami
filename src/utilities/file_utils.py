# utils/file_utils.py
import os
import shutil
import aiofiles
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import mimetypes

from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)

class FileManager:
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.processed_dir = Path(settings.PROCESSED_DIR)
        self.chunks_dir = Path(settings.CHUNKS_DIR)
        
    async def save_uploaded_file(self, file_content: bytes, filename: str) -> Path:
        """Save uploaded file and return path"""
        try:
            # Generate unique filename
            file_hash = hashlib.md5(file_content).hexdigest()[:8]
            safe_filename = self._sanitize_filename(filename)
            unique_filename = f"{file_hash}_{safe_filename}"
            
            file_path = self.upload_dir / unique_filename
            
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_content)
            
            logger.info(f"File saved: {unique_filename}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage"""
        # Remove directory path
        filename = os.path.basename(filename)
        
        # Replace unsafe characters
        unsafe_chars = '<>:"/\\|?*'
        for char in unsafe_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        name, ext = os.path.splitext(filename)
        if len(name) > 100:
            name = name[:100]
        
        return name + ext
    
    async def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get file information"""
        try:
            if not file_path.exists():
                return {}
            
            stat = file_path.stat()
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            return {
                "filename": file_path.name,
                "size": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "mime_type": mime_type,
                "extension": file_path.suffix
            }
            
        except Exception as e:
            logger.error(f"Error getting file info: {str(e)}")
            return {}
    
    async def delete_file_and_related(self, file_id: str) -> bool:
        """Delete file and all related data"""
        try:
            deleted_items = []
            
            # Delete uploaded PDF
            for pdf_file in self.upload_dir.glob(f"{file_id}_*.pdf"):
                pdf_file.unlink(missing_ok=True)
                deleted_items.append(f"upload: {pdf_file.name}")
            
            # Delete processed text
            for txt_file in self.processed_dir.glob(f"{file_id}_*"):
                txt_file.unlink(missing_ok=True)
                deleted_items.append(f"processed: {txt_file.name}")
            
            # Delete chunks directory
            chunks_path = self.chunks_dir / file_id
            if chunks_path.exists():
                shutil.rmtree(chunks_path)
                deleted_items.append(f"chunks: {chunks_path}")
            
            logger.info(f"Deleted items for {file_id}: {deleted_items}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting files: {str(e)}")
            return False
    
    async def get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage statistics"""
        try:
            def get_dir_size(path: Path) -> int:
                total = 0
                for item in path.rglob('*'):
                    if item.is_file():
                        total += item.stat().st_size
                return total
            
            upload_size = get_dir_size(self.upload_dir)
            processed_size = get_dir_size(self.processed_dir)
            chunks_size = get_dir_size(self.chunks_dir)
            
            total_size = upload_size + processed_size + chunks_size
            
            return {
                "upload_dir_mb": round(upload_size / (1024 * 1024), 2),
                "processed_dir_mb": round(processed_size / (1024 * 1024), 2),
                "chunks_dir_mb": round(chunks_size / (1024 * 1024), 2),
                "total_mb": round(total_size / (1024 * 1024), 2),
                "upload_files_count": len(list(self.upload_dir.glob("*.pdf"))),
                "processed_files_count": len(list(self.processed_dir.glob("*"))),
                "chunks_dirs_count": len(list(self.chunks_dir.iterdir()))
            }
            
        except Exception as e:
            logger.error(f"Error getting disk usage: {str(e)}")
            return {}
    
    async def cleanup_old_files(self, days_old: int = 30) -> Dict[str, int]:
        """Clean up files older than specified days"""
        try:
            import time
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)
            
            cleaned = {"uploads": 0, "processed": 0, "chunks": 0}
            
            # Clean uploads
            for file_path in self.upload_dir.iterdir():
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink(missing_ok=True)
                    cleaned["uploads"] += 1
            
            # Clean processed
            for file_path in self.processed_dir.iterdir():
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink(missing_ok=True)
                    cleaned["processed"] += 1
            
            # Clean chunks
            for dir_path in self.chunks_dir.iterdir():
                if dir_path.is_dir() and dir_path.stat().st_mtime < cutoff_time:
                    shutil.rmtree(dir_path, ignore_errors=True)
                    cleaned["chunks"] += 1
            
            logger.info(f"Cleanup completed: {cleaned}")
            return cleaned
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            return {"error": str(e)}
    
    async def validate_pdf_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate PDF file"""
        try:
            validation_result = {
                "is_valid": False,
                "file_size_mb": 0,
                "page_count": 0,
                "errors": []
            }
            
            if not file_path.exists():
                validation_result["errors"].append("File does not exist")
                return validation_result
            
            # Check file size
            file_size = file_path.stat().st_size
            validation_result["file_size_mb"] = round(file_size / (1024 * 1024), 2)
            
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                validation_result["errors"].append("File too large (>100MB)")
            
            # Check if it's a PDF
            if not file_path.suffix.lower() == '.pdf':
                validation_result["errors"].append("Not a PDF file")
            
            # Try to read PDF
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    validation_result["page_count"] = len(pdf_reader.pages)
                    
                    if validation_result["page_count"] == 0:
                        validation_result["errors"].append("PDF has no pages")
                    elif validation_result["page_count"] > 1000:
                        validation_result["errors"].append("PDF has too many pages (>1000)")
                        
            except Exception as e:
                validation_result["errors"].append(f"PDF read error: {str(e)}")
            
            validation_result["is_valid"] = len(validation_result["errors"]) == 0
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating PDF: {str(e)}")
            return {
                "is_valid": False,
                "errors": [f"Validation error: {str(e)}"]
            }

# Global file manager instance
file_manager = FileManager()