import logging
import os
import yaml
import json
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import sys

# fmt: off
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))


@dataclass
class RAGMetrics:
    """Data class to store RAG pipeline metrics."""
    request_id: str
    timestamp: str
    query: str
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    documents_retrieved: int = 0
    similarity_scores: List[float] = None
    tokens_used: int = 0
    response_length: int = 0
    success: bool = True
    error_message: str = ""
    pipeline_stage: str = ""
    
    def __post_init__(self):
        if self.similarity_scores is None:
            self.similarity_scores = []


class RAGPipelineLogger:
    """Enhanced logger specifically designed for RAG pipeline operations."""
    
    def __init__(self, config_path='rag_logging_config.yaml'):
        """Initialize the RAG pipeline logger with enhanced configuration."""
        self.config = self._load_config(config_path)
        self.loggers = {}
        self.base_log_dir = self.config.get('base_log_dir', 'logs')
        self.temp_log_dir = self.config.get('temp_log_dir', 'temp_logs')
        self.active_requests = {}  # Track active requests
        self._setup_directories()
        
    def _load_config(self, config_path):
        """Load configuration from YAML file with RAG-specific defaults."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            # Enhanced default configuration for RAG pipeline
            return {
                'base_log_dir': 'logs',
                'temp_log_dir': 'temp_logs',
                'default_level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s',
                'console_output': True,
                'save_metrics': True,
                'metrics_format': 'json',  # json or csv
                'cleanup_temp_logs': False,  # Set to True to auto-cleanup temp logs
                'temp_log_retention_hours': 24,
                'modules': {
                    'main': {'level': 'INFO'},
                    'rag_pipeline': {'level': 'DEBUG'},
                    'document_retrieval': {'level': 'INFO'},
                    'vector_search': {'level': 'DEBUG'},
                    'embedding': {'level': 'INFO'},
                    'text_generation': {'level': 'INFO'},
                    'preprocessing': {'level': 'INFO'},
                    'postprocessing': {'level': 'INFO'},
                    'query_processing': {'level': 'DEBUG'},
                    'response_validation': {'level': 'INFO'},
                    'gradio': {'level': 'DEBUG'},
                    'streamlit': {'level': 'DEBUG'},
                    'fastapi': {'level': 'DEBUG'}
                }
            }

    def _setup_directories(self):
        """Setup directory structure for logs and temporary files."""
        for directory in [self.base_log_dir, self.temp_log_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Create subdirectories for different types of temp logs
        temp_subdirs = ['requests', 'metrics', 'assets', 'errors']
        for subdir in temp_subdirs:
            path = os.path.join(self.temp_log_dir, subdir)
            if not os.path.exists(path):
                os.makedirs(path)

    def _get_log_path(self, module_name):
        """Generate hierarchical path for log files."""
        now = datetime.now()
        year_dir = os.path.join(self.base_log_dir, str(now.year))
        month_dir = os.path.join(year_dir, f"{now.month:02d}")
        day_dir = os.path.join(month_dir, f"{now.day:02d}")
        os.makedirs(day_dir, exist_ok=True)
        return os.path.join(day_dir, f"{module_name}.log")

    def _get_temp_log_path(self, log_type, filename):
        """Generate path for temporary log files."""
        return os.path.join(self.temp_log_dir, log_type, filename)

    def get_logger(self, module_name, request_id=None):
        """Get or create a logger for a specific module with optional request ID."""
        logger_key = f"{module_name}_{request_id}" if request_id else module_name
        
        if logger_key in self.loggers:
            return self.loggers[logger_key]

        # Create new logger
        logger = logging.getLogger(logger_key)
        module_config = self.config['modules'].get(module_name, {})
        level = getattr(logging, module_config.get('level', self.config['default_level']))
        logger.setLevel(level)

        # Create custom formatter that includes request_id
        class RAGFormatter(logging.Formatter):
            def format(self, record):
                if not hasattr(record, 'request_id'):
                    record.request_id = request_id or 'N/A'
                return super().format(record)

        formatter = RAGFormatter(self.config.get('format'))

        # Create file handler
        log_path = self._get_log_path(module_name)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Optionally add console handler
        if self.config.get('console_output', True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        self.loggers[logger_key] = logger
        return logger

    def start_rag_request(self, query: str, user_id: str = None) -> str:
        """Start tracking a new RAG pipeline request."""
        request_id = str(uuid.uuid4())
        
        metrics = RAGMetrics(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            query=query
        )
        
        self.active_requests[request_id] = {
            'metrics': metrics,
            'start_time': time.time(),
            'user_id': user_id,
            'stages': []
        }
        
        # Log request start
        logger = self.get_logger('rag_pipeline', request_id)
        logger.info(f"RAG request started - Query: '{query[:100]}...' - User: {user_id}")
        
        # Save request to temp file
        self._save_temp_request_log(request_id, 'started', {'query': query, 'user_id': user_id})
        
        return request_id

    def log_stage(self, request_id: str, stage: str, data: Dict[str, Any] = None):
        """Log a specific stage in the RAG pipeline."""
        if request_id not in self.active_requests:
            return
        
        logger = self.get_logger('rag_pipeline', request_id)
        stage_info = {'stage': stage, 'timestamp': datetime.now().isoformat()}
        
        if data:
            stage_info.update(data)
            
        self.active_requests[request_id]['stages'].append(stage_info)
        self.active_requests[request_id]['metrics'].pipeline_stage = stage
        
        logger.info(f"Pipeline stage: {stage} - {json.dumps(data) if data else ''}")
        
        # Save stage to temp file
        self._save_temp_request_log(request_id, f'stage_{stage}', stage_info)

    def log_retrieval_metrics(self, request_id: str, retrieval_time: float, 
                            documents_retrieved: int, similarity_scores: List[float] = None):
        """Log document retrieval metrics."""
        if request_id not in self.active_requests:
            return
            
        metrics = self.active_requests[request_id]['metrics']
        metrics.retrieval_time = retrieval_time
        metrics.documents_retrieved = documents_retrieved
        metrics.similarity_scores = similarity_scores or []
        
        logger = self.get_logger('document_retrieval', request_id)
        logger.info(f"Retrieved {documents_retrieved} documents in {retrieval_time:.3f}s")
        
        if similarity_scores:
            avg_score = sum(similarity_scores) / len(similarity_scores)
            logger.debug(f"Average similarity score: {avg_score:.3f}")

    def log_generation_metrics(self, request_id: str, generation_time: float, 
                             tokens_used: int, response_length: int):
        """Log text generation metrics."""
        if request_id not in self.active_requests:
            return
            
        metrics = self.active_requests[request_id]['metrics']
        metrics.generation_time = generation_time
        metrics.tokens_used = tokens_used
        metrics.response_length = response_length
        
        logger = self.get_logger('text_generation', request_id)
        logger.info(f"Generated response in {generation_time:.3f}s - Tokens: {tokens_used} - Length: {response_length}")

    def log_error(self, request_id: str, error_message: str, stage: str = None):
        """Log an error in the RAG pipeline."""
        if request_id in self.active_requests:
            metrics = self.active_requests[request_id]['metrics']
            metrics.success = False
            metrics.error_message = error_message
            if stage:
                metrics.pipeline_stage = stage
        
        logger = self.get_logger('rag_pipeline', request_id)
        logger.error(f"Pipeline error in stage '{stage}': {error_message}")
        
        # Save error to temp file
        error_data = {
            'error_message': error_message,
            'stage': stage,
            'timestamp': datetime.now().isoformat()
        }
        self._save_temp_log('errors', f'error_{request_id}_{int(time.time())}.json', error_data)

    def finish_rag_request(self, request_id: str, success: bool = True) -> Dict[str, Any]:
        """Finish tracking a RAG pipeline request and return metrics."""
        if request_id not in self.active_requests:
            return {}
        
        request_data = self.active_requests[request_id]
        metrics = request_data['metrics']
        
        # Calculate total time
        total_time = time.time() - request_data['start_time']
        metrics.total_time = total_time
        metrics.success = success
        
        logger = self.get_logger('rag_pipeline', request_id)
        logger.info(f"RAG request completed - Total time: {total_time:.3f}s - Success: {success}")
        
        # Save final metrics
        if self.config.get('save_metrics', True):
            self._save_metrics(request_id, metrics)
        
        # Save complete request log
        complete_request_data = {
            'metrics': asdict(metrics),
            'stages': request_data['stages'],
            'user_id': request_data['user_id']
        }
        self._save_temp_request_log(request_id, 'completed', complete_request_data)
        
        # Clean up active request
        metrics_dict = asdict(metrics)
        del self.active_requests[request_id]
        
        return metrics_dict

    @contextmanager
    def rag_request_context(self, query: str, user_id: str = None):
        """Context manager for RAG requests."""
        request_id = self.start_rag_request(query, user_id)
        try:
            yield request_id
            self.finish_rag_request(request_id, success=True)
        except Exception as e:
            self.log_error(request_id, str(e))
            self.finish_rag_request(request_id, success=False)
            raise

    def _save_temp_request_log(self, request_id: str, event: str, data: Dict[str, Any]):
        """Save request event to temporary log file."""
        filename = f"request_{request_id}_{event}.json"
        filepath = self._get_temp_log_path('requests', filename)
        
        log_entry = {
            'request_id': request_id,
            'event': event,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_entry, f, indent=2, default=str)

    def _save_temp_log(self, log_type: str, filename: str, data: Dict[str, Any]):
        """Save data to temporary log file."""
        filepath = self._get_temp_log_path(log_type, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _save_metrics(self, request_id: str, metrics: RAGMetrics):
        """Save metrics to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if self.config.get('metrics_format', 'json') == 'json':
            filename = f"metrics_{request_id}_{timestamp}.json"
            filepath = self._get_temp_log_path('metrics', filename)
            with open(filepath, 'w') as f:
                json.dump(asdict(metrics), f, indent=2, default=str)
        else:
            # CSV format
            filename = f"metrics_{timestamp}.csv"
            filepath = self._get_temp_log_path('metrics', filename)
            import csv
            
            file_exists = os.path.exists(filepath)
            with open(filepath, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=asdict(metrics).keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(asdict(metrics))

    def get_temp_logs_for_analysis(self, request_id: str = None, 
                                  log_type: str = None) -> List[Dict[str, Any]]:
        """Retrieve temporary logs for analysis."""
        logs = []
        
        search_dirs = [log_type] if log_type else ['requests', 'metrics', 'assets', 'errors']
        
        for search_dir in search_dirs:
            dir_path = self._get_temp_log_path(search_dir, '')
            if not os.path.exists(dir_path):
                continue
                
            for filename in os.listdir(dir_path):
                if request_id and request_id not in filename:
                    continue
                    
                filepath = os.path.join(dir_path, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        logs.append({
                            'filename': filename,
                            'log_type': search_dir,
                            'data': data
                        })
                except json.JSONDecodeError:
                    continue
        
        return logs

    def cleanup_temp_logs(self, older_than_hours: int = None):
        """Clean up old temporary log files."""
        if older_than_hours is None:
            older_than_hours = self.config.get('temp_log_retention_hours', 24)
        
        cutoff_time = time.time() - (older_than_hours * 3600)
        cleaned_count = 0
        
        for root, dirs, files in os.walk(self.temp_log_dir):
            for filename in files:
                filepath = os.path.join(root, filename)
                if os.path.getmtime(filepath) < cutoff_time:
                    os.remove(filepath)
                    cleaned_count += 1
        
        logger = self.get_logger('main')
        logger.info(f"Cleaned up {cleaned_count} temporary log files older than {older_than_hours} hours")

    def get_pipeline_analytics(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get analytics for RAG pipeline performance."""
        logs = self.get_temp_logs_for_analysis(log_type='metrics')
        
        cutoff_time = datetime.now().timestamp() - (time_range_hours * 3600)
        recent_logs = []
        
        for log in logs:
            try:
                log_time = datetime.fromisoformat(log['data']['timestamp']).timestamp()
                if log_time > cutoff_time:
                    recent_logs.append(log['data'])
            except (KeyError, ValueError):
                continue
        
        if not recent_logs:
            return {}
        
        # Calculate analytics
        total_requests = len(recent_logs)
        successful_requests = sum(1 for log in recent_logs if log.get('success', False))
        avg_total_time = sum(log.get('total_time', 0) for log in recent_logs) / total_requests
        avg_retrieval_time = sum(log.get('retrieval_time', 0) for log in recent_logs) / total_requests
        avg_generation_time = sum(log.get('generation_time', 0) for log in recent_logs) / total_requests
        avg_documents_retrieved = sum(log.get('documents_retrieved', 0) for log in recent_logs) / total_requests
        
        return {
            'time_range_hours': time_range_hours,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'success_rate': successful_requests / total_requests if total_requests > 0 else 0,
            'avg_total_time': avg_total_time,
            'avg_retrieval_time': avg_retrieval_time,
            'avg_generation_time': avg_generation_time,
            'avg_documents_retrieved': avg_documents_retrieved
        }


# Example usage and testing
if __name__ == "__main__":
    # Create RAG pipeline logger instance
    rag_logger = RAGPipelineLogger()
    
    # Example 1: Using context manager
    print("Testing with context manager:")
    try:
        with rag_logger.rag_request_context("What is machine learning?", "user123") as request_id:
            print(f"Request ID: {request_id}")
            
            # Log different stages
            rag_logger.log_stage(request_id, "query_preprocessing", {"processed_query": "machine learning"})
            time.sleep(0.1)  # Simulate processing time
            
            rag_logger.log_stage(request_id, "document_retrieval")
            rag_logger.log_retrieval_metrics(request_id, 0.25, 5, [0.85, 0.82, 0.78, 0.75, 0.70])
            time.sleep(0.1)
            
            rag_logger.log_stage(request_id, "text_generation")
            rag_logger.log_generation_metrics(request_id, 0.8, 150, 500)
            
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Manual request tracking
    print("\nTesting manual request tracking:")
    request_id = rag_logger.start_rag_request("Explain neural networks", "user456")
    rag_logger.log_stage(request_id, "embedding_generation", {"embedding_model": "sentence-transformers"})
    rag_logger.log_retrieval_metrics(request_id, 0.15, 3, [0.92, 0.88, 0.84])
    rag_logger.log_generation_metrics(request_id, 1.2, 200, 750)
    final_metrics = rag_logger.finish_rag_request(request_id)
    print(f"Final metrics: {json.dumps(final_metrics, indent=2)}")
    
    # Show analytics
    print("\nPipeline Analytics:")
    analytics = rag_logger.get_pipeline_analytics()
    print(json.dumps(analytics, indent=2))
    
    # Show temp logs for analysis
    print("\nTemp logs available for analysis:")
    temp_logs = rag_logger.get_temp_logs_for_analysis()
    print(f"Found {len(temp_logs)} temp log files")
    for log in temp_logs[:3]:  # Show first 3
        print(f"- {log['log_type']}/{log['filename']}")