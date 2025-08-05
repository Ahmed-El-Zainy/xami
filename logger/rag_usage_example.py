"""
Complete Working RAG Pipeline Example with Enhanced Logging
This example fixes common errors and provides a fully functional demonstration.
"""

import logging
import os
import yaml
import json
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import sys

# Create necessary directories
os.makedirs('logs', exist_ok=True)
os.makedirs('temp_logs/requests', exist_ok=True)
os.makedirs('temp_logs/metrics', exist_ok=True)
os.makedirs('temp_logs/assets', exist_ok=True)
os.makedirs('temp_logs/errors', exist_ok=True)

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
    similarity_scores: Optional[List[float]] = None
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
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the RAG pipeline logger with enhanced configuration."""
        self.config = self._load_config(config_path)
        self.loggers = {}
        self.base_log_dir = self.config.get('base_log_dir', 'logs')
        self.temp_log_dir = self.config.get('temp_log_dir', 'temp_logs')
        self.active_requests = {}  # Track active requests
        self._setup_directories()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file with RAG-specific defaults."""
        default_config = {
            'base_log_dir': 'logs',
            'temp_log_dir': 'temp_logs',
            'default_level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s',
            'console_output': True,
            'save_metrics': True,
            'metrics_format': 'json',
            'cleanup_temp_logs': False,
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
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as file:
                    loaded_config = yaml.safe_load(file)
                    # Merge with defaults
                    default_config.update(loaded_config)
            except Exception as e:
                print(f"Warning: Could not load config file {config_path}: {e}")
                print("Using default configuration")
        
        return default_config

    def _setup_directories(self):
        """Setup directory structure for logs and temporary files."""
        for directory in [self.base_log_dir, self.temp_log_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Create subdirectories for different types of temp logs
        temp_subdirs = ['requests', 'metrics', 'assets', 'errors']
        for subdir in temp_subdirs:
            path = os.path.join(self.temp_log_dir, subdir)
            os.makedirs(path, exist_ok=True)

    def _get_log_path(self, module_name: str) -> str:
        """Generate hierarchical path for log files."""
        now = datetime.now()
        year_dir = os.path.join(self.base_log_dir, str(now.year))
        month_dir = os.path.join(year_dir, f"{now.month:02d}")
        day_dir = os.path.join(month_dir, f"{now.day:02d}")
        os.makedirs(day_dir, exist_ok=True)
        return os.path.join(day_dir, f"{module_name}.log")

    def _get_temp_log_path(self, log_type: str, filename: str) -> str:
        """Generate path for temporary log files."""
        return os.path.join(self.temp_log_dir, log_type, filename)

    def get_logger(self, module_name: str, request_id: Optional[str] = None):
        """Get or create a logger for a specific module with optional request ID."""
        logger_key = f"{module_name}_{request_id}" if request_id else module_name
        
        if logger_key in self.loggers:
            return self.loggers[logger_key]

        # Create new logger
        logger = logging.getLogger(logger_key)
        
        # Clear any existing handlers to avoid duplicates
        logger.handlers.clear()
        
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

        # Prevent propagation to avoid duplicate logs
        logger.propagate = False
        
        self.loggers[logger_key] = logger
        return logger

    def start_rag_request(self, query: str, user_id: Optional[str] = None) -> str:
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

    def log_stage(self, request_id: str, stage: str, data: Optional[Dict[str, Any]] = None):
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
                            documents_retrieved: int, similarity_scores: Optional[List[float]] = None):
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

    def log_error(self, request_id: str, error_message: str, stage: Optional[str] = None):
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
    def rag_request_context(self, query: str, user_id: Optional[str] = None):
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
        
        try:
            with open(filepath, 'w') as f:
                json.dump(log_entry, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save temp log: {e}")

    def _save_temp_log(self, log_type: str, filename: str, data: Dict[str, Any]):
        """Save data to temporary log file."""
        filepath = self._get_temp_log_path(log_type, filename)
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save temp log: {e}")

    def _save_metrics(self, request_id: str, metrics: RAGMetrics):
        """Save metrics to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"metrics_{request_id}_{timestamp}.json"
        filepath = self._get_temp_log_path('metrics', filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(asdict(metrics), f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save metrics: {e}")

    def get_temp_logs_for_analysis(self, request_id: Optional[str] = None, 
                                  log_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve temporary logs for analysis."""
        logs = []
        
        search_dirs = [log_type] if log_type else ['requests', 'metrics', 'assets', 'errors']
        
        for search_dir in search_dirs:
            dir_path = self._get_temp_log_path(search_dir, '')
            if not os.path.exists(dir_path):
                continue
                
            try:
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
                    except (json.JSONDecodeError, FileNotFoundError):
                        continue
            except OSError:
                continue
        
        return logs


# Mock RAG Components for Testing
class MockEmbeddingModel:
    """Mock embedding model for demonstration."""
    
    def __init__(self, logger: RAGPipelineLogger):
        self.logger = logger
        self.model_logger = logger.get_logger('embedding')
    
    def encode(self, text: str, request_id: str) -> List[float]:
        """Generate mock embeddings."""
        self.model_logger.info(f"Generating embeddings for text length: {len(text)}")
        time.sleep(0.05)  # Simulate processing time
        return [0.1 + i * 0.01 for i in range(384)]  # Mock 384-dimensional embedding


class MockVectorDatabase:
    """Mock vector database for demonstration."""
    
    def __init__(self, logger: RAGPipelineLogger):
        self.logger = logger
        self.db_logger = logger.get_logger('vector_search')
        # Mock documents with metadata
        self.documents = [
            {"id": "doc1", "content": "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.", "score": 0.95},
            {"id": "doc2", "content": "Neural networks are computational models inspired by biological neural networks that constitute animal brains.", "score": 0.88},
            {"id": "doc3", "content": "Deep learning uses multiple layers of artificial neural networks to model and understand complex patterns.", "score": 0.82},
            {"id": "doc4", "content": "Supervised learning requires labeled training data to learn the mapping from inputs to outputs.", "score": 0.78},
            {"id": "doc5", "content": "Unsupervised learning finds hidden patterns or structures in data without labeled examples.", "score": 0.72}
        ]
    
    def search(self, query_embedding: List[float], top_k: int, request_id: str) -> Tuple[List[Dict], List[float]]:
        """Search for similar documents."""
        self.db_logger.info(f"Searching for top {top_k} similar documents")
        time.sleep(0.1)  # Simulate search time
        
        results = self.documents[:top_k]
        scores = [doc["score"] for doc in results]
        
        self.db_logger.debug(f"Found {len(results)} documents with scores: {scores}")
        return results, scores


class MockLLM:
    """Mock Language Model for demonstration."""
    
    def __init__(self, logger: RAGPipelineLogger):
        self.logger = logger
        self.llm_logger = logger.get_logger('text_generation')
    
    def generate(self, prompt: str, context_docs: List[Dict], request_id: str) -> str:
        """Generate response based on prompt and context."""
        self.llm_logger.info(f"Generating response with {len(context_docs)} context documents")
        time.sleep(0.5)  # Simulate generation time
        
        # Create a more realistic response
        if "machine learning" in prompt.lower():
            response = "Machine learning is a powerful subset of artificial intelligence that enables computers to learn and improve their performance on tasks through experience, without being explicitly programmed for every scenario. Based on the retrieved documents, machine learning systems can identify patterns in data and make predictions or decisions."
        elif "neural network" in prompt.lower():
            response = "Neural networks are computational models inspired by the biological neural networks found in animal brains. They consist of interconnected nodes (neurons) that process information through weighted connections. These networks can learn complex patterns and relationships in data through training."
        else:
            response = f"Based on the provided context documents, I can provide information about your query: '{prompt[:50]}...'. The retrieved documents contain relevant information that helps answer your question comprehensively."
        
        self.llm_logger.debug(f"Generated response length: {len(response)} characters")
        return response


class RAGPipeline:
    """Complete RAG Pipeline with integrated logging."""
    
    def __init__(self, config_path: Optional[str] = None):
        # Initialize logger
        self.logger = RAGPipelineLogger(config_path)
        self.pipeline_logger = self.logger.get_logger('rag_pipeline')
        
        # Initialize components
        self.embedding_model = MockEmbeddingModel(self.logger)
        self.vector_db = MockVectorDatabase(self.logger)
        self.llm = MockLLM(self.logger)
        
        # Component loggers
        self.preprocessing_logger = self.logger.get_logger('preprocessing')
        self.postprocessing_logger = self.logger.get_logger('postprocessing')
        self.validation_logger = self.logger.get_logger('response_validation')
    
    def preprocess_query(self, query: str, request_id: str) -> str:
        """Preprocess the input query."""
        self.preprocessing_logger.info("Starting query preprocessing")
        
        # Mock preprocessing steps
        processed_query = query.strip().lower()
        
        self.logger.log_stage(request_id, "query_preprocessing", {
            "original_query": query,
            "processed_query": processed_query,
            "preprocessing_steps": ["strip", "lowercase"]
        })
        
        self.preprocessing_logger.debug(f"Query preprocessed: '{query}' -> '{processed_query}'")
        return processed_query
    
    def retrieve_documents(self, query: str, request_id: str, top_k: int = 5) -> Tuple[List[Dict], List[float]]:
        """Retrieve relevant documents for the query."""
        retrieval_start = time.time()
        
        self.logger.log_stage(request_id, "embedding_generation")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, request_id)
        
        self.logger.log_stage(request_id, "vector_search", {"top_k": top_k})
        
        # Search for similar documents
        documents, scores = self.vector_db.search(query_embedding, top_k, request_id)
        
        retrieval_time = time.time() - retrieval_start
        
        # Log retrieval metrics
        self.logger.log_retrieval_metrics(request_id, retrieval_time, len(documents), scores)
        
        return documents, scores
    
    def generate_response(self, query: str, documents: List[Dict], request_id: str) -> str:
        """Generate response using retrieved documents."""
        generation_start = time.time()
        
        self.logger.log_stage(request_id, "response_generation", {
            "num_context_docs": len(documents),
            "context_doc_ids": [doc["id"] for doc in documents]
        })
        
        # Generate response
        response = self.llm.generate(query, documents, request_id)
        
        generation_time = time.time() - generation_start
        
        # Mock token counting (in real implementation, use actual tokenizer)
        estimated_tokens = int(len(response.split()) * 1.3)  # Rough approximation
        
        # Log generation metrics
        self.logger.log_generation_metrics(request_id, generation_time, estimated_tokens, len(response))
        
        return response
    
    def validate_response(self, response: str, query: str, request_id: str) -> bool:
        """Validate the generated response quality."""
        self.validation_logger.info("Starting response validation")
        
        # Mock validation checks
        validation_results = {
            "length_check": len(response) > 50,
            "relevance_check": True,  # Mock check
            "safety_check": True,     # Mock check
            "completeness_check": len(response) > 100
        }
        
        all_passed = all(validation_results.values())
        
        self.logger.log_stage(request_id, "response_validation", {
            "validation_results": validation_results,
            "overall_passed": all_passed
        })
        
        if not all_passed:
            failed_checks = [check for check, passed in validation_results.items() if not passed]
            self.validation_logger.warning(f"Validation failed for checks: {failed_checks}")
        
        return all_passed
    
    def postprocess_response(self, response: str, request_id: str) -> str:
        """Post-process the generated response."""
        self.postprocessing_logger.info("Starting response postprocessing")
        
        # Mock postprocessing steps
        processed_response = response.strip()
        
        self.logger.log_stage(request_id, "response_postprocessing", {
            "original_length": len(response),
            "processed_length": len(processed_response),
            "postprocessing_steps": ["strip_whitespace"]
        })
        
        return processed_response
    
    def process_query(self, query: str, user_id: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """Process a complete RAG query with full logging."""
        
        # Use context manager for automatic request tracking
        with self.logger.rag_request_context(query, user_id) as request_id:
            try:
                # Step 1: Preprocess query
                processed_query = self.preprocess_query(query, request_id)
                
                # Step 2: Retrieve documents
                documents, scores = self.retrieve_documents(processed_query, request_id, top_k)
                
                # Step 3: Generate response
                response = self.generate_response(processed_query, documents, request_id)
                
                # Step 4: Validate response
                is_valid = self.validate_response(response, processed_query, request_id)
                
                if not is_valid:
                    self.logger.log_error(request_id, "Response validation failed", "response_validation")
                    # In a real system, you might retry or use a fallback
                
                # Step 5: Post-process response
                final_response = self.postprocess_response(response, request_id)
                
                return {
                    "request_id": request_id,
                    "query": query,
                    "response": final_response,
                    "documents_used": len(documents),
                    "similarity_scores": scores,
                    "validation_passed": is_valid,
                    "success": True
                }
                
            except Exception as e:
                self.logger.log_error(request_id, str(e), "unknown")
                raise


def create_sample_config():
    """Create a sample configuration file."""
    config = {
        'base_log_dir': 'logs',
        'temp_log_dir': 'temp_logs',
        'default_level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s',
        'console_output': True,
        'save_metrics': True,
        'metrics_format': 'json',
        'cleanup_temp_logs': False,
        'temp_log_retention_hours': 24,
        'modules': {
            'main': {'level': 'INFO'},
            'rag_pipeline': {'level': 'INFO'},
            'document_retrieval': {'level': 'INFO'},
            'vector_search': {'level': 'INFO'},
            'embedding': {'level': 'INFO'},
            'text_generation': {'level': 'INFO'},
            'preprocessing': {'level': 'INFO'},
            'postprocessing': {'level': 'INFO'},
            'query_processing': {'level': 'INFO'},
            'response_validation': {'level': 'INFO'}
        }
    }
    
    with open('rag_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Created rag_config.yaml")


def run_example():
    """Run the complete RAG pipeline example."""
    print("🚀 Starting RAG Pipeline Logger Example")
    print("=" * 50)
    
    # Create sample config
    create_sample_config()
    
    # Initialize RAG pipeline
    try:
        rag_system = RAGPipeline('rag_config.yaml')
        print("✅ RAG Pipeline initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing RAG Pipeline: {e}")
        print("🔄 Trying with default configuration...")
        rag_system = RAGPipeline()
        print("✅ RAG Pipeline initialized with default config")
    
    # Test queries
    test_queries = [
        "What is machine learning and how does it work?",
        "Explain neural networks in simple terms",
        "What are the differences between supervised and unsupervised learning?",
    ]
    
    print(f"\n📋 Processing {len(test_queries)} test queries...")
    print("-" * 50)
    
    # Process queries
    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        try:
            result = rag_system.process_query(query, f"test_user_{i}")
            results.append(result)
            
            print(f"✅ Success!")
            print(f"   Request ID: {result['request_id'][:8]}...")
            print(f"   Documents used: {result['documents_used']}")
            print(f"   Response length: {len(result['response'])} chars")
            print(f"   Validation passed: {result['validation_passed']}")
            print(f"   Response preview: {result['response'][:100]}...")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    # Show logs analysis
    print(f"\n📊 Analyzing logs...")
    print("-" * 50)
    
    # Wait for logs to be written
    time.sleep(1)
    
    # Get temp logs for analysis
    temp_logs = rag_system.logger.get_temp_logs_for_analysis()
    print(f"📁 Found {len(temp_logs)} temporary log files")
    
    # Group by type
    log_types = {}
    for log in temp_logs:
        log_type = log['log_type']
        log_types[log_type] = log_types.get(log_type, 0) + 1
    
    for log_type, count in log_types.items():
        print(f"   📂 {log_type}: {count} files")
    
    # Show sample metrics
    metrics_logs = [log for log in temp_logs if log['log_type'] == 'metrics']
    if metrics_logs:
        print(f"\n📈 Sample Metrics from last request:")
        sample_metrics = metrics_logs[-1]['data']
        print(f"   ⏱️ Total time: {sample_metrics.get('total_time', 0):.3f}s")
        print(f"   🔍 Retrieval time: {sample_metrics.get('retrieval_time', 0):.3f}s")
        print(f"   🤖 Generation time: {sample_metrics.get('generation_time', 0):.3f}s")
        print(f"   📄 Documents retrieved: {sample_metrics.get('documents_retrieved', 0)}")
        print(f"   ✅ Success: {sample_metrics.get('success', False)}")
    
    # Show directory structure
    print(f"\n📁 Log Directory Structure:")
    for root, dirs, files in os.walk('logs'):
        level = root.replace('logs', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}📂 {os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:3]:  # Show only first 3 files
            print(f"{subindent}📄 {file}")
        if len(files) > 3:
            print(f"{subindent}... and {len(files) - 3} more files")
    
    # Show temp logs structure
    print(f"\n📁 Temp Logs Structure:")
    for root, dirs, files in os.walk('temp_logs'):
        level = root.replace('temp_logs', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}📂 {os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:3]:  # Show only first 3 files
            print(f"{subindent}📄 {file}")
        if len(files) > 3:
            print(f"{subindent}... and {len(files) - 3} more files")
    
    print(f"\n🎉 RAG Pipeline Logger Example Complete!")
    print(f"✨ Key Features Demonstrated:")
    print(f"   • ✅ Request lifecycle tracking with unique IDs")
    print(f"   • ✅ Stage-by-stage pipeline logging")
    print(f"   • ✅ Performance metrics collection")
    print(f"   • ✅ Error handling and logging")
    print(f"   • ✅ Temporary files for analysis")
    print(f"   • ✅ Hierarchical log organization")
    
    print(f"\n📚 Next Steps:")
    print(f"   1. 📊 Run log analysis: python rag_log_analyzer.py --export-report --generate-dashboard")
    print(f"   2. 🔍 Start monitoring: python rag_log_analyzer.py --monitor --monitor-duration 30")
    print(f"   3. 🧪 Test analyzer: python rag_log_analyzer.py --test")
    print(f"   4. 📈 Quick summary: from rag_log_analyzer import quick_performance_summary; quick_performance_summary()")


if __name__ == "__main__":
    run_example()