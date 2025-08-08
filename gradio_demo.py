# enhanced_gradio_demo.py
import gradio as gr
import requests
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Enhanced version with chat interface and analytics

API_BASE_URL = "http://localhost:8000/api/v1"

class EnhancedRAGClient:
    """Enhanced client with additional features"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.conversation_history = []
        
    def health_check(self) -> Dict:
        try:
            response = requests.get(f"{self.base_url}/health/", timeout=10)
            return response.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def upload_document(self, file_path: str) -> Tuple[bool, str, str]:
        try:
            with open(file_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(
                    f"{self.base_url}/upload-document/",
                    files=files,
                    timeout=300
                )
            
            if response.status_code == 200:
                result = response.json()
                return True, result.get('file_id', ''), result.get('message', '')
            else:
                return False, '', f"Upload failed: {response.text}"
        except Exception as e:
            return False, '', f"Error: {str(e)}"
    
    def get_processing_status(self, file_id: str) -> Dict:
        try:
            response = requests.get(f"{self.base_url}/processing-status/{file_id}")
            return response.json() if response.status_code == 200 else {"status": "error"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def query_rag_with_history(self, query: str, top_k: int = 5) -> Dict:
        try:
            # Include conversation history
            params = {
                "query": query,
                "top_k": top_k,
                "rerank": True,
                "conversation_history": self.conversation_history[-3:]  # Last 3 turns
            }
            response = requests.post(f"{self.base_url}/rag-query/", params=params)
            
            if response.status_code == 200:
                result = response.json()
                # Update conversation history
                self.conversation_history.append({
                    "user": query,
                    "assistant": result.get("answer", ""),
                    "timestamp": datetime.now().isoformat()
                })
                return result
            else:
                return {"error": f"Query failed: {response.text}"}
        except Exception as e:
            return {"error": f"Error: {str(e)}"}
    
    def get_statistics(self) -> Dict:
        try:
            response = requests.get(f"{self.base_url}/statistics/")
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            return {"error": str(e)}
    
    def summarize_text(self, text: str, max_length: int = 200) -> str:
        try:
            data = {"text": text, "max_length": max_length}
            response = requests.post(f"{self.base_url}/summarize-text/", json=data)
            if response.status_code == 200:
                return response.json().get("summary", "")
            return "Error summarizing text"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_questions(self, text: str, num_questions: int = 3) -> List[str]:
        try:
            data = {"text": text, "num_questions": num_questions}
            response = requests.post(f"{self.base_url}/generate-questions/", json=data)
            if response.status_code == 200:
                return response.json().get("questions", [])
            return []
        except Exception as e:
            return []

# Initialize enhanced client
client = EnhancedRAGClient()

# Global state
upload_progress = {}
chat_history = []

def create_status_plot(documents_stats: Dict) -> gr.Plot:
    """Create a status plot for document processing"""
    try:
        if not documents_stats:
            return gr.Plot()
        
        # Sample data for demonstration
        statuses = ["Completed", "Processing", "Failed"]
        counts = [5, 2, 1]  # This would come from real data
        colors = ["#2ecc71", "#f39c12", "#e74c3c"]
        
        fig = go.Figure(data=[
            go.Bar(x=statuses, y=counts, marker_color=colors)
        ])
        
        fig.update_layout(
            title="Document Processing Status",
            xaxis_title="Status",
            yaxis_title="Count",
            template="plotly_white"
        )
        
        return fig
    except:
        return gr.Plot()

def chat_interface(message: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
    """Chat interface for conversational queries"""
    if not message.strip():
        return "", history
    
    try:
        # Query the RAG system
        result = client.query_rag_with_history(message)
        
        if "error" in result:
            response = f"❌ {result['error']}"
        else:
            answer = result.get("answer", "No answer generated")
            confidence = result.get("confidence_score", 0.0)
            sources_count = len(result.get("sources", []))
            
            response = f"{answer}\n\n*Confidence: {confidence:.2f} | Sources: {sources_count}*"
        
        # Update history
        history.append([message, response])
        
        return "", history
        
    except Exception as e:
        history.append([message, f"❌ Error: {str(e)}"])
        return "", history

def upload_with_progress(file) -> Tuple[str, Dict]:
    """Upload file with progress tracking"""
    if file is None:
        return "❌ No file selected", {}
    
    if not file.name.lower().endswith('.pdf'):
        return "❌ Please upload a PDF file only", {}
    
    try:
        success, file_id, message = client.upload_document(file.name)
        
        if success:
            progress_info = {
                "file_id": file_id,
                "filename": Path(file.name).name,
                "status": "processing",
                "progress": 0.0,
                "start_time": time.time()
            }
            upload_progress[file_id] = progress_info
            
            return f"✅ **Upload Successful!**\n\n📄 **File:** {Path(file.name).name}\n🆔 **ID:** {file_id}\n📝 **Message:** {message}", progress_info
        else:
            return f"❌ Upload failed: {message}", {}
            
    except Exception as e:
        return f"❌ Error: {str(e)}", {}

def check_all_uploads_status() -> str:
    """Check status of all uploads"""
    if not upload_progress:
        return "📭 No uploads in progress"
    
    status_text = "## 📊 Upload Status Overview\n\n"
    
    for file_id, info in upload_progress.items():
        try:
            status_data = client.get_processing_status(file_id)
            status = status_data.get("status", "unknown")
            progress = status_data.get("progress", 0.0)
            message = status_data.get("message", "")
            
            # Update local info
            info["status"] = status
            info["progress"] = progress
            
            # Status emoji
            emoji_map = {
                "processing": "🔄",
                "completed": "✅", 
                "failed": "❌",
                "unknown": "❓"
            }
            emoji = emoji_map.get(status, "❓")
            
            # Progress bar
            if status == "processing":
                progress_bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
                progress_text = f"{progress*100:.1f}% `{progress_bar}`"
            else:
                progress_text = "Complete" if status == "completed" else status.title()
            
            status_text += f"### {emoji} {info['filename']}\n"
            status_text += f"**Status:** {progress_text}\n"
            status_text += f"**Message:** {message}\n"
            status_text += f"**ID:** `{file_id}`\n\n"
            
        except Exception as e:
            status_text += f"### ❌ {info['filename']}\n**Error:** {str(e)}\n\n"
    
    return status_text

def generate_analytics_dashboard() -> Tuple[str, gr.Plot]:
    """Generate analytics dashboard"""
    