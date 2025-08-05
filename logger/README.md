# RAG Pipeline Logger - Complete Guide

## 📋 Requirements

### Core Requirements (Essential)
```bash
pip install pyyaml
```

### Optional Requirements (For Full Features)
```bash
pip install pandas matplotlib seaborn numpy
```

## 🚀 Quick Start

### 1. Basic Setup
```python
from rag_pipeline_logger import RAGPipelineLogger

# Initialize logger
logger = RAGPipelineLogger()

# Use with context manager (recommended)
with logger.rag_request_context("What is machine learning?", "user123") as request_id:
    # Your RAG pipeline code here
    logger.log_stage(request_id, "retrieval")
    logger.log_retrieval_metrics(request_id, 0.25, 5, [0.85, 0.82])
    logger.log_generation_metrics(request_id, 0.8, 150, 500)
```

### 2. Run the Complete Example
```bash
python complete_rag_example.py
```

### 3. Analyze Logs
```bash
# Basic analysis
python fixed_rag_analyzer.py --days 7 --export-report

# Generate dashboard (requires matplotlib)
python fixed_rag_analyzer.py --days 7 --generate-dashboard --export-report

# Start monitoring
python fixed_rag_analyzer.py --monitor --monitor-duration 60

# Test with sample data
python fixed_rag_analyzer.py --test
```

## 📁 File Structure

```
project/
├── rag_pipeline_logger.py      # Core logger implementation
├── fixed_rag_analyzer.py       # Analysis and monitoring tools
├── complete_rag_example.py     # Complete working example
├── rag_config.yaml            # Configuration file
├── logs/                      # Hierarchical log files
│   └── 2024/01/15/           # Organized by date
│       ├── rag_pipeline.log
│       ├── retrieval.log
│       └── generation.log
└── temp_logs/                 # Temporary analysis files
    ├── requests/              # Request lifecycle logs
    ├── metrics/               # Performance metrics
    ├── assets/                # Query/response content
    └── errors/                # Error details
```

## 🔧 Configuration

### Basic Configuration (`rag_config.yaml`)
```yaml
base_log_dir: logs
temp_log_dir: temp_logs
default_level: INFO
console_output: true
save_metrics: true

modules:
  rag_pipeline: {level: INFO}
  document_retrieval: {level: INFO}
  text_generation: {level: INFO}
  # ... more modules
```

## 💡 Key Features

### ✅ Request Tracking
- **Unique Request IDs**: Every request gets a UUID for tracking
- **Context Managers**: Automatic request lifecycle management
- **Stage Logging**: Track each pipeline stage separately

### ✅ Performance Metrics
- **Timing**: Retrieval, generation, and total response times
- **Quality**: Document relevance scores, token usage
- **Success Tracking**: Request success/failure with error details

### ✅ Analysis & Monitoring
- **Performance Patterns**: Identify trends and bottlenecks
- **Real-time Monitoring**: Health checks with configurable alerts
- **Visual Dashboards**: Performance charts and graphs

### ✅ Flexible Storage
- **Hierarchical Logs**: Organized by date for easy management
- **Temporary Analysis Files**: JSON format for easy processing
- **Configurable Retention**: Automatic cleanup policies

## 🎯 Usage Patterns

### Pattern 1: Simple Logging
```python
logger = RAGPipelineLogger()
request_id = logger.start_rag_request("User query", "user_id")
logger.log_stage(request_id, "preprocessing")
# ... your code ...
logger.finish_rag_request(request_id, success=True)
```

### Pattern 2: Context Manager (Recommended)
```python
with logger.rag_request_context("User query", "user_id") as request_id:
    # Automatic error handling and cleanup
    logger.log_stage(request_id, "retrieval")
    # ... your code ...
```

### Pattern 3: Component Integration
```python
class MyRAGComponent:
    def __init__(self, logger):
        self.logger = logger.get_logger('my_component')
    
    def process(self, data, request_id):
        self.logger.info("Processing started")
        # ... your code ...
```

## 📊 Analysis Examples

### Performance Summary
```python
from fixed_rag_analyzer import quick_performance_summary
quick_performance_summary(days=1)
```

### Custom Analysis
```python
from fixed_rag_analyzer import RAGLogAnalyzer

analyzer = RAGLogAnalyzer(logger)
data = analyzer.load_all_metrics(days=7)
patterns = analyzer.analyze_performance_patterns(data)
bottlenecks = analyzer.identify_performance_bottlenecks(data)
```

### Real-time Monitoring
```python
from fixed_rag_analyzer import monitor_system_health
monitor_system_health(duration_minutes=30)
```

## ⚠️ Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Missing dependencies
   pip install pyyaml pandas matplotlib seaborn
   ```

2. **Permission Errors**
   ```bash
   # Ensure write permissions for log directories
   chmod 755 logs temp_logs
   ```

3. **No Data for Analysis**
   ```python
   # Generate sample data for testing
   python fixed_rag_analyzer.py --test
   ```

4. **Dashboard Generation Fails**
   ```bash
   # Install plotting libraries
   pip install matplotlib seaborn
   ```

### Fallback Mode
The system gracefully degrades when optional dependencies are missing:
- **No pandas**: Uses basic Python data structures
- **No matplotlib**: Skips dashboard generation
- **No config file**: Uses built-in defaults

## 🔄 Integration Examples

### FastAPI Integration
```python
from fastapi import FastAPI
from rag_pipeline_logger import RAGPipelineLogger

app = FastAPI()
logger = RAGPipelineLogger()

@app.post("/query")
async def process_query(query: str, user_id: str):
    with logger.rag_request_context(query, user_id) as request_id:
        # Your RAG pipeline logic
        result = await process_rag_pipeline(query, request_id)
        return result
```

### Gradio Integration
```python
import gradio as gr
from rag_pipeline_logger import RAGPipelineLogger

logger = RAGPipelineLogger()

def rag_interface(query, user_id="gradio_user"):
    with logger.rag_request_context(query, user_id) as request_id:
        # Your RAG pipeline logic
        return process_query(query, request_id)

gr.Interface(fn=rag_interface, inputs="text", outputs="text").launch()
```

## 📈 Performance Optimization

### Best Practices
1. **Use Context Managers**: Automatic cleanup and error handling
2. **Configure Log Levels**: Reduce verbosity in production
3. **Regular Cleanup**: Enable automatic temp log cleanup
4. **Monitor Metrics**: Set up alerts for performance thresholds

### Performance Thresholds
```yaml
performance_thresholds:
  retrieval_time_warning: 2.0    # seconds
  generation_time_warning: 5.0   # seconds
  total_time_warning: 10.0      # seconds
  low_similarity_warning: 0.5   # similarity threshold
```

## 🚀 Advanced Features

### Custom Formatters
```yaml
formatters:
  detailed:
    format: "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(funcName)s:%(lineno)d - %(message)s"
    modules: ["rag_pipeline", "vector_search"]
```

### Asset Tracking
```yaml
asset_tracking:
  enabled: true
  track_queries: true
  track_responses: true
  max_query_length: 1000
  max_response_length: 5000
```

### Health Monitoring
```python
monitor = RAGLogMonitor(logger, check_interval=30)
monitor.alert_thresholds = {
    'success_rate_min': 0.95,
    'avg_response_time_max': 5.0,
    'error_rate_max': 0.05
}
monitor.start_monitoring(duration_minutes=60)
```

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Run the test suite: `python fixed_rag_analyzer.py --test`
3. Review log files in the `logs/` directory
4. Check temp files in `temp_logs/` for analysis data

---

## 🎉 That's It!

You now have a complete, production-ready RAG pipeline logging system with:
- ✅ Comprehensive request tracking
- ✅ Performance monitoring
- ✅ Error handling and analysis
- ✅ Visual dashboards
- ✅ Real-time health monitoring
- ✅ Flexible configuration
- ✅ Easy integration with existing systems

Happy logging! 🚀