
<!-- 
# Directory structure
"""
ocr/
├── main.py                 # FastAPI application entry point
├── config/
│   ├── __init__.py
│   ├── settings.py         # Configuration settings
│   └── logging_config.py   # Custom logger configuration
├── services/
│   ├── __init__.py
│   ├── ocr_service.py      # OCR processing with EasyOCR
│   ├── text_processor.py   # Text cleaning and chunking
│   ├── embedding_service.py # Embedding generation
│   ├── vector_db.py        # Qdrant vector database operations
│   ├── reranker.py         # Reranking service
│   └── gemini_service.py   # Gemini API integration
├── models/
│   ├── __init__.py
│   └── schemas.py          # Pydantic models
├── utils/
│   ├── __init__.py
│   ├── arabic_utils.py     # Arabic text utilities
│   └── file_utils.py       # File handling utilities
├── api/
│   ├── __init__.py
│   └── endpoints.py        # API endpoints
├── data/
│   ├── uploads/           # Uploaded PDFs
│   ├── processed/         # Processed text files
│   └── chunks/           # Text chunks
└── logs/                 # Application logs

""" -->





# Arabic Educational RAG Pipeline 📚🤖

A complete RAG (Retrieval-Augmented Generation) pipeline specifically designed for Arabic educational content, perfect for secondary school materials up to 500+ pages.

## ✨ Features

- **📄 PDF Processing**: Upload and OCR processing with EasyOCR
- **🔤 Arabic Text Processing**: Advanced Arabic text cleaning, normalization, and chunking
- **🧠 Smart Embeddings**: Multilingual embeddings optimized for Arabic content
- **🗃️ Vector Database**: ChromaDB for efficient similarity search
- **🎯 Advanced Reranking**: BM25 and hybrid reranking methods
- **🤖 AI Responses**: Gemini API integration for intelligent answers
- **📊 Custom Logging**: Comprehensive logging and monitoring
- **🚀 FastAPI Backend**: High-performance async API

## 🏗️ Architecture




```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Upload    │───▶│   OCR Service   │───▶│ Text Processor  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Gemini Response │◀───│   Reranker      │◀───│ Embedding Gen   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Search API    │◀───│   ChromaDB      │
                       └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)

1. **Clone the repository**
```bash
git clone <repository-url>
cd rag-pipeline
```

2. **Set up environment**
```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

3. **Run setup script**
```bash
chmod +x setup.sh
./setup.sh
```

4. **Access the API**
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/v1/health

### Option 2: Local Development

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Run local setup**
```bash
python run_local.py
```

## 📖 API Usage

### Upload a Document
```python
import requests

with open("textbook.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/upload-document/",
        files={"file": f}
    )
print(response.json())
```

### Query the RAG System
```python
response = requests.post(
    "http://localhost:8000/api/v1/rag-query/",
    params={
        "query": "ما هو تعريف الجاذبية؟",
        "top_k": 5,
        "rerank": True
    }
)
print(response.json()["answer"])
```

### Search Documents
```python
response = requests.post(
    "http://localhost:8000/api/v1/search/",
    json={
        "query": "الرياضيات",
        "top_k": 10,
        "rerank": True
    }
)
```

## 🔧 Configuration

Key settings in `.env`:

```env
# Gemini API
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Text Processing
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MIN_CHUNK_SIZE=100

# Reranking
RERANK_TOP_K=20
FINAL_TOP_K=5
```

## 📊 Monitoring

- **Logs**: Check `logs/` directory for detailed application logs
- **Health Check**: `GET /api/v1/health`
- **Statistics**: `GET /api/v1/statistics`
- **Processing Status**: `GET /api/v1/processing-status/{file_id}`

## 🛠️ Advanced Features

### Custom Arabic Text Processing
- Diacritic removal and normalization
- Stopword filtering
- Root-based similarity matching
- Educational keyword detection

### Hybrid Reranking
- BM25 lexical matching
- Semantic similarity
- Arabic-specific scoring
- Diversity filtering

### Vector Database Optimization
- Persistent storage with ChromaDB
- Optimized for large documents
- Efficient similarity search
- Metadata filtering

## 📁 Project Structure

```
rag_pipeline/
├── main.py                 # FastAPI application
├── config/
│   ├── settings.py         # Configuration
│   └── logging_config.py   # Custom logger
├── services/
│   ├── ocr_service.py      # OCR processing
│   ├── text_processor.py   # Text chunking
│   ├── embedding_service.py # Embeddings
│   ├── vector_db.py        # ChromaDB operations
│   ├── reranker.py         # Reranking service
│   └── gemini_service.py   # Gemini API
├── utils/
│   ├── arabic_utils.py     # Arabic text processing
│   └── file_utils.py       # File operations
├── api/
│   └── endpoints.py        # API routes
└── data/                   # Data storage
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
- Check the logs in `logs/` directory
- Use the health check endpoint
- Review the API documentation at `/docs`
