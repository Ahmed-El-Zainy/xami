import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import re
import io
import PyPDF2
from docx import Document
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import umap
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import os
import requests
from huggingface_hub import login, HfApi
import warnings
from scipy.interpolate import griddata
import json
warnings.filterwarnings('ignore')

# Global variables for session management
global_state = {
    'embeddings': None,
    'texts': None,
    'reduced_embeddings_2d': None,
    'reduced_embeddings_3d': None,
    'similarity_matrix': None,
    'hf_token': None,
    'authenticated_user': None
}

class ArabicTextProcessor:
    """Class for processing Arabic text and generating embeddings"""
    
    def __init__(self):
        self.models_info = {
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
                "name": "Multilingual MiniLM-L12-v2",
                "description": "Fast and efficient multilingual sentence transformer (384 dim)",
                "type": "sentence_transformer",
                "requires_token": False
            },
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {
                "name": "Multilingual MPNet Base",
                "description": "Higher quality multilingual embeddings (768 dim)",
                "type": "sentence_transformer",
                "requires_token": False
            },
            "aubmindlab/bert-base-arabertv02": {
                "name": "AraBERT v0.2",
                "description": "Arabic-specific BERT model",
                "type": "transformer",
                "requires_token": False
            },
            "CAMeL-Lab/bert-base-arabic-camelbert-mix": {
                "name": "CamelBERT Mix",
                "description": "Mixed-domain Arabic BERT",
                "type": "transformer",
                "requires_token": False
            },
            "sentence-transformers/distiluse-base-multilingual-cased": {
                "name": "DistilUSE Multilingual",
                "description": "Distilled multilingual Universal Sentence Encoder (512 dim)",
                "type": "sentence_transformer",
                "requires_token": False
            }
        }
        
    def set_hf_token(self, token: str) -> Tuple[bool, str]:
        """Set and validate Hugging Face token"""
        if not token or not token.strip():
            return False, "Please enter a token"
            
        token = token.strip()
        if not token.startswith('hf_'):
            return False, "Token must start with 'hf_'"
            
        try:
            # Test the token by making a simple API call
            api = HfApi(token=token)
            user_info = api.whoami()
            
            # Set token for transformers library
            os.environ['HUGGINGFACE_HUB_TOKEN'] = token
            login(token=token, add_to_git_credential=False)
            
            global_state['hf_token'] = token
            global_state['authenticated_user'] = user_info.get('name', 'Unknown')
            
            return True, f"✅ Authenticated as: {global_state['authenticated_user']}"
            
        except Exception as e:
            return False, f"❌ Authentication failed: {str(e)}"
    
    def logout(self) -> str:
        """Logout and clear token"""
        global_state['hf_token'] = None
        global_state['authenticated_user'] = None
        if 'HUGGINGFACE_HUB_TOKEN' in os.environ:
            del os.environ['HUGGINGFACE_HUB_TOKEN']
        return "🔒 Logged out successfully"
    
    def check_model_access(self, model_name: str) -> Tuple[bool, str]:
        """Check if model is accessible with current token"""
        try:
            if global_state['hf_token']:
                headers = {"Authorization": f"Bearer {global_state['hf_token']}"}
            else:
                headers = {}
                
            response = requests.get(
                f"https://huggingface.co/api/models/{model_name}",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return True, "✅ Model accessible"
            elif response.status_code == 401:
                return False, "❌ Model requires authentication"
            elif response.status_code == 404:
                return False, "❌ Model not found"
            else:
                return False, f"⚠️ Unexpected response: {response.status_code}"
                
        except Exception as e:
            return False, f"⚠️ Could not verify model: {str(e)}"
    
    def preprocess_arabic_text(self, text: str) -> str:
        """Preprocess Arabic text by removing diacritics and normalizing"""
        # Remove diacritics
        text = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)
        # Keep only Arabic letters, spaces, and basic punctuation
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s\.\!\?\،\؛\:\"\'\(\)\[\]\{\}]', ' ', text)
        # Normalize spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def segment_text(self, text: str) -> List[str]:
        """Segment text into meaningful chunks"""
        # Split by sentences and paragraphs
        segments = re.split(r'[.!?؟\n]+', text)
        # Filter and clean segments
        processed_segments = []
        for segment in segments:
            cleaned = self.preprocess_arabic_text(segment)
            # Keep segments with at least 3 words and 15 characters
            if len(cleaned) > 15 and len(cleaned.split()) > 3:
                processed_segments.append(cleaned)
        return processed_segments
    
    def load_model(self, model_name: str, model_type: str = None):
        """Load the selected model"""
        try:
            # Determine model type
            if model_type is None:
                if model_name in self.models_info:
                    model_type = self.models_info[model_name]["type"]
                else:
                    # Default to sentence_transformer for custom models
                    model_type = "sentence_transformer"
            
            token = global_state.get('hf_token')
            
            if model_type == "sentence_transformer":
                if token:
                    model = SentenceTransformer(model_name, use_auth_token=token)
                else:
                    model = SentenceTransformer(model_name)
                return model, None
            else:
                if token:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
                    model = AutoModel.from_pretrained(model_name, use_auth_token=token)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModel.from_pretrained(model_name)
                return model, tokenizer
                
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    def generate_embeddings(self, texts: List[str], model_name: str, model_type: str = None) -> np.ndarray:
        """Generate embeddings for texts using the selected model"""
        model, tokenizer = self.load_model(model_name, model_type)
        embeddings = []
        
        # Determine model type
        if model_type is None:
            if model_name in self.models_info:
                model_type = self.models_info[model_name]["type"]
            else:
                model_type = "sentence_transformer"
        
        if model_type == "sentence_transformer":
            # Use SentenceTransformer
            batch_size = 16
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_embeddings = model.encode(batch, convert_to_tensor=False)
                embeddings.extend(batch_embeddings)
        else:
            # Use regular transformer with mean pooling
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()
            
            for text in texts:
                inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                                 padding=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    embeddings.append(embedding[0])
        
        return np.array(embeddings)

class TextFileProcessor:
    """Class for processing different file types"""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='windows-1256') as file:
                    return file.read()
            except Exception as e:
                raise Exception(f"Error reading TXT file: {str(e)}")

class EmbeddingVisualizer:
    """Class for creating various visualizations of embeddings"""
    
    def __init__(self, embeddings: np.ndarray, texts: List[str]):
        self.embeddings = embeddings
        self.texts = texts
        
    def reduce_dimensions(self, method='PCA', n_components=2):
        """Reduce embedding dimensions for visualization"""
        if method == 'PCA':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 'TSNE':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(self.embeddings)-1))
        elif method == 'UMAP':
            reducer = umap.UMAP(n_components=n_components, random_state=42)
        else:
            raise ValueError("Method must be 'PCA', 'TSNE', or 'UMAP'")
        
        reduced = reducer.fit_transform(self.embeddings)
        
        # Store in global state
        if n_components == 2:
            global_state['reduced_embeddings_2d'] = reduced
        elif n_components == 3:
            global_state['reduced_embeddings_3d'] = reduced
            
        return reduced
    
    def calculate_similarity_matrix(self):
        """Calculate cosine similarity matrix"""
        similarity_matrix = cosine_similarity(self.embeddings)
        global_state['similarity_matrix'] = similarity_matrix
        return similarity_matrix
    
    def create_scatter_plot(self, method='PCA', show_3d=False, plot_3d_type="scatter", 
                          color_scheme="viridis", plot_theme="plotly"):
        """Create interactive scatter plot"""
        n_components = 3 if show_3d else 2
        reduced = self.reduce_dimensions(method, n_components)
        
        df = pd.DataFrame(reduced, columns=[f'{method}_{i+1}' for i in range(n_components)])
        df['text'] = [f"Segment {i+1}: {text[:100]}..." if len(text) > 100 else f"Segment {i+1}: {text}" 
                     for i, text in enumerate(self.texts)]
        df['segment_id'] = range(1, len(self.texts) + 1)
        
        if show_3d:
            if plot_3d_type == "scatter":
                fig = px.scatter_3d(
                    df, x=f'{method}_1', y=f'{method}_2', z=f'{method}_3',
                    hover_data=['text'], color='segment_id',
                    title=f"3D {method} Visualization of Text Embeddings",
                    color_continuous_scale=color_scheme
                )
            elif plot_3d_type == "surface":
                # Create surface plot
                xi = np.linspace(reduced[:, 0].min(), reduced[:, 0].max(), 20)
                yi = np.linspace(reduced[:, 1].min(), reduced[:, 1].max(), 20)
                xi, yi = np.meshgrid(xi, yi)
                zi = griddata((reduced[:, 0], reduced[:, 1]), reduced[:, 2], (xi, yi), method='cubic')
                
                fig = go.Figure(data=[go.Surface(x=xi, y=yi, z=zi, colorscale=color_scheme)])
                fig.add_trace(go.Scatter3d(
                    x=reduced[:, 0], y=reduced[:, 1], z=reduced[:, 2],
                    mode='markers',
                    marker=dict(size=8, color=range(len(reduced)), colorscale=color_scheme),
                    text=df['text'], name='Text Points'
                ))
                fig.update_layout(title=f"3D Surface {method} Visualization")
            
            fig.update_layout(height=700, template=plot_theme)
        else:
            fig = px.scatter(df, x=f'{method}_1', y=f'{method}_2',
                           hover_data=['text'], color='segment_id',
                           title=f"2D {method} Visualization of Text Embeddings",
                           color_continuous_scale=color_scheme)
            fig.update_layout(height=600, template=plot_theme)
        
        fig.update_layout(showlegend=False)
        return fig
    
    def create_similarity_heatmap(self, plot_theme="plotly"):
        """Create similarity matrix heatmap"""
        similarity_matrix = self.calculate_similarity_matrix()
        labels = [f"T{i+1}" for i in range(len(self.texts))]
        
        fig = ff.create_annotated_heatmap(
            z=similarity_matrix,
            x=labels, y=labels,
            colorscale='Viridis',
            showscale=True
        )
        fig.update_layout(
            title="Text Similarity Matrix (Cosine Similarity)",
            height=600, template=plot_theme
        )
        return fig
    
    def create_cluster_visualization(self, n_clusters=4, method='PCA', clustering_method='kmeans', 
                                   color_scheme='viridis', plot_theme='plotly'):
        """Create cluster visualization"""
        reduced = self.reduce_dimensions(method, 2)
        
        # Perform clustering
        if clustering_method == 'kmeans':
            clusterer = KMeans(n_clusters=min(n_clusters, len(self.texts)), random_state=42)
        elif clustering_method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=min(n_clusters, len(self.texts)))
        elif clustering_method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=2)
        
        clusters = clusterer.fit_predict(self.embeddings)
        
        df = pd.DataFrame(reduced, columns=[f'{method}_1', f'{method}_2'])
        df['cluster'] = clusters
        df['text'] = [f"Segment {i+1}: {text[:100]}..." if len(text) > 100 else f"Segment {i+1}: {text}" 
                     for i, text in enumerate(self.texts)]
        
        if clustering_method == 'dbscan':
            df['cluster_name'] = df['cluster'].apply(lambda x: 'Noise' if x == -1 else f'Cluster {x+1}')
        else:
            df['cluster_name'] = df['cluster'].apply(lambda x: f'Cluster {x+1}')
        
        fig = px.scatter(df, x=f'{method}_1', y=f'{method}_2',
                        color='cluster_name', hover_data=['text'],
                        title=f"{clustering_method.title()} Clustering - {method}",
                        color_discrete_sequence=px.colors.qualitative.Set1)
        
        fig.update_layout(height=600, template=plot_theme)
        return fig
    
    def create_network_graph(self, similarity_threshold=0.5, layout='spring', plot_theme='plotly'):
        """Create network graph"""
        similarity_matrix = self.calculate_similarity_matrix()
        
        G = nx.Graph()
        for i, text in enumerate(self.texts):
            G.add_node(i, text=text[:50] + "..." if len(text) > 50 else text)
        
        edges_added = 0
        for i in range(len(self.texts)):
            for j in range(i+1, len(self.texts)):
                if similarity_matrix[i][j] > similarity_threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i][j])
                    edges_added += 1
        
        if edges_added == 0:
            return None
        
        # Create layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create traces
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=2, color='#888'),
                              hoverinfo='none', mode='lines')
        
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = [f"T{node+1}" for node in G.nodes()]
        node_info = [G.nodes[node]['text'] for node in G.nodes()]
        
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                              hovertext=node_info, text=node_text,
                              textposition="middle center", hoverinfo="text",
                              marker=dict(size=20, color=list(range(len(node_x))),
                                        colorscale='viridis', line=dict(width=2)))
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(title=f'Network Graph (Similarity > {similarity_threshold})',
                                      showlegend=False, hovermode='closest',
                                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      height=600, template=plot_theme))
        return fig

# Initialize processors
text_processor = ArabicTextProcessor()
file_processor = TextFileProcessor()

def authenticate_user(token):
    """Authenticate user with HF token"""
    if not token:
        return "Please enter a token", ""
    
    success, message = text_processor.set_hf_token(token)
    if success:
        return message, gr.update(visible=True)
    else:
        return message, gr.update(visible=False)

def logout_user():
    """Logout current user"""
    message = text_processor.logout()
    return message, gr.update(visible=False)

def check_custom_model(model_id, model_type):
    """Check if custom model is accessible"""
    if not model_id:
        return "Please enter a model ID"
    
    accessible, message = text_processor.check_model_access(model_id)
    return message

def process_file(file):
    """Process uploaded file and extract text"""
    if file is None:
        return ""
    
    try:
        file_extension = file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            return file_processor.extract_text_from_pdf(file.name)
        elif file_extension == 'docx':
            return file_processor.extract_text_from_docx(file.name)
        elif file_extension == 'txt':
            return file_processor.extract_text_from_txt(file.name)
        else:
            return "Unsupported file type"
    except Exception as e:
        return f"Error processing file: {str(e)}"

def process_text_and_generate_embeddings(text_input, file_input, model_selection, custom_model, custom_model_type):
    """Main processing function"""
    try:
        # Get text content
        text_content = text_input or ""
        if file_input is not None:
            file_text = process_file(file_input)
            text_content = text_content + "\n" + file_text
        
        if not text_content.strip():
            return "Please enter text or upload a file", None, None, None, None
        
        # Determine model to use
        if model_selection == "Custom Model":
            if not custom_model:
                return "Please enter a custom model ID", None, None, None, None
            selected_model = custom_model
            model_type = custom_model_type
        else:
            selected_model = model_selection
            model_type = None
        
        # Segment text
        segments = text_processor.segment_text(text_content)
        if len(segments) == 0:
            return "No valid text segments found", None, None, None, None
        
        if len(segments) > 50:
            segments = segments[:50]
        
        # Generate embeddings
        embeddings = text_processor.generate_embeddings(segments, selected_model, model_type)
        
        # Store in global state
        global_state['embeddings'] = embeddings
        global_state['texts'] = segments
        
        # Create visualizer
        visualizer = EmbeddingVisualizer(embeddings, segments)
        
        # Generate basic statistics
        avg_similarity = np.mean(cosine_similarity(embeddings))
        stats = f"""
        ✅ **Processing Complete!**
        
        📊 **Statistics:**
        - **Segments**: {len(segments)}
        - **Embedding Dimension**: {embeddings.shape[1]}
        - **Average Similarity**: {avg_similarity:.3f}
        - **Model Used**: {selected_model}
        """
        
        # Create initial visualizations
        scatter_fig = visualizer.create_scatter_plot()
        similarity_fig = visualizer.create_similarity_heatmap()
        cluster_fig = visualizer.create_cluster_visualization()
        network_fig = visualizer.create_network_graph()
        
        return stats, scatter_fig, similarity_fig, cluster_fig, network_fig
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None, None, None, None

def update_scatter_plot(method, show_3d, plot_3d_type, color_scheme, plot_theme):
    """Update scatter plot with new parameters"""
    if global_state['embeddings'] is None:
        return None
    
    try:
        visualizer = EmbeddingVisualizer(global_state['embeddings'], global_state['texts'])
        fig = visualizer.create_scatter_plot(method, show_3d, plot_3d_type, color_scheme, plot_theme)
        return fig
    except Exception as e:
        return None

def update_cluster_plot(n_clusters, method, clustering_method, color_scheme, plot_theme):
    """Update cluster plot with new parameters"""
    if global_state['embeddings'] is None:
        return None
    
    try:
        visualizer = EmbeddingVisualizer(global_state['embeddings'], global_state['texts'])
        fig = visualizer.create_cluster_visualization(n_clusters, method, clustering_method, color_scheme, plot_theme)
        return fig
    except Exception as e:
        return None

def update_network_plot(similarity_threshold, layout, plot_theme):
    """Update network plot with new parameters"""
    if global_state['embeddings'] is None:
        return None
    
    try:
        visualizer = EmbeddingVisualizer(global_state['embeddings'], global_state['texts'])
        fig = visualizer.create_network_graph(similarity_threshold, layout, plot_theme)
        return fig
    except Exception as e:
        return None

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="🤗 Arabic Text Embeddings Visualizer", theme=gr.themes.Soft()) as app:
        
        # Header
        gr.Markdown("""
        # 🤗 Arabic Text Embeddings Visualizer
        ### Generate high-quality semantic embeddings using state-of-the-art Arabic language models
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Authentication Section
                gr.Markdown("## 🔑 Authentication (Optional)")
                with gr.Group():
                    token_input = gr.Textbox(
                        label="Hugging Face Token",
                        placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxx",
                        type="password",
                        info="Get your free token from https://huggingface.co/settings/tokens"
                    )
                    
                    with gr.Row():
                        auth_btn = gr.Button("🔓 Authenticate", variant="primary")
                        logout_btn = gr.Button("🔒 Logout", visible=False)
                    
                    auth_status = gr.Textbox(label="Status", interactive=False)
                
                # Model Selection
                gr.Markdown("## 🤖 Model Selection")
                with gr.Group():
                    model_method = gr.Radio(
                        choices=["Pre-configured", "Custom Model"],
                        value="Pre-configured",
                        label="Model Selection Method"
                    )
                    
                    # Pre-configured models
                    model_dropdown = gr.Dropdown(
                        choices=list(text_processor.models_info.keys()),
                        value="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        label="Pre-configured Models",
                        visible=True
                    )
                    
                    # Custom model inputs
                    custom_model_input = gr.Textbox(
                        label="Custom Model ID",
                        placeholder="e.g., sentence-transformers/all-MiniLM-L6-v2",
                        visible=False
                    )
                    
                    custom_model_type = gr.Dropdown(
                        choices=["sentence_transformer", "transformer"],
                        value="sentence_transformer",
                        label="Model Type",
                        visible=False
                    )
                    
                    check_model_btn = gr.Button("🔍 Check Model", visible=False)
                    model_status = gr.Textbox(label="Model Status", visible=False, interactive=False)
                
                # Input Section
                gr.Markdown("## 📝 Text Input")
                with gr.Group():
                    text_input = gr.Textbox(
                        label="Arabic Text",
                        placeholder="أدخل النص العربي هنا...",
                        lines=8,
                        rtl=True
                    )
                    
                    file_input = gr.File(
                        label="Upload File",
                        file_types=[".txt", ".pdf", ".docx"]
                    )
                    
                    process_btn = gr.Button("🚀 Generate Embeddings", variant="primary", size="lg")
                
                # Visualization Controls
                gr.Markdown("## 📊 Visualization Settings")
                with gr.Group():
                    with gr.Row():
                        reduction_method = gr.Dropdown(
                            choices=["PCA", "TSNE", "UMAP"],
                            value="PCA",
                            label="Dimensionality Reduction"
                        )
                        
                        color_scheme = gr.Dropdown(
                            choices=["viridis", "plasma", "inferno", "magma", "cividis", "turbo"],
                            value="viridis",
                            label="Color Scheme"
                        )
                    
                    with gr.Row():
                        show_3d = gr.Checkbox(label="3D Visualization", value=False)
                        plot_3d_type = gr.Dropdown(
                            choices=["scatter", "surface"],
                            value="scatter",
                            label="3D Plot Type"
                        )
                    
                    with gr.Row():
                        n_clusters = gr.Slider(
                            minimum=2, maximum=10, value=4, step=1,
                            label="Number of Clusters"
                        )
                        
                        clustering_method = gr.Dropdown(
                            choices=["kmeans", "hierarchical", "dbscan"],
                            value="kmeans",
                            label="Clustering Method"
                        )
                    
                    with gr.Row():
                        similarity_threshold = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                            label="Similarity Threshold"
                        )
                        
                        network_layout = gr.Dropdown(
                            choices=["spring", "circular", "kamada_kawai"],
                            value="spring",
                            label="Network Layout"
                        )
                    
                    plot_theme = gr.Dropdown(
                        choices=["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"],
                        value="plotly",
                        label="Plot Theme"
                    )
            
            with gr.Column(scale=2):
                # Results Section
                gr.Markdown("## 🎯 Results & Visualizations")
                
                # Processing Status
                status_output = gr.Markdown(value="Ready to process text...")
                
                # Visualization Tabs
                with gr.Tabs():
                    with gr.TabItem("📈 Scatter Plot"):
                        scatter_plot = gr.Plot(label="Scatter Plot Visualization")
                        
                        with gr.Row():
                            update_scatter_btn = gr.Button("🔄 Update Scatter Plot", variant="secondary")
                        
                    with gr.TabItem("🔥 Similarity Matrix"):
                        similarity_plot = gr.Plot(label="Similarity Matrix")
                        
                        with gr.Group():
                            gr.Markdown("### 📊 Similarity Statistics")
                            similarity_stats = gr.Markdown("Process text to see similarity statistics...")
                    
                    with gr.TabItem("🎯 Clustering"):
                        cluster_plot = gr.Plot(label="Cluster Visualization")
                        
                        with gr.Row():
                            update_cluster_btn = gr.Button("🔄 Update Clustering", variant="secondary")
                        
                        cluster_stats = gr.Markdown("Process text to see cluster statistics...")
                    
                    with gr.TabItem("🕸️ Network Graph"):
                        network_plot = gr.Plot(label="Network Graph")
                        
                        with gr.Row():
                            update_network_btn = gr.Button("🔄 Update Network", variant="secondary")
                        
                        network_info = gr.Markdown("Process text to see network analysis...")
                    
                    with gr.TabItem("📋 Text Segments"):
                        segments_display = gr.Markdown("Process text to see segments...")
                        
                        with gr.Group():
                            gr.Markdown("### 📈 Advanced Analytics")
                            analytics_output = gr.Markdown("Detailed analytics will appear here after processing...")
        
        # Event Handlers
        
        # Authentication events
        auth_btn.click(
            fn=authenticate_user,
            inputs=[token_input],
            outputs=[auth_status, logout_btn]
        )
        
        logout_btn.click(
            fn=logout_user,
            outputs=[auth_status, logout_btn]
        )
        
        # Model selection events
        def toggle_model_inputs(method):
            if method == "Custom Model":
                return (
                    gr.update(visible=False),  # model_dropdown
                    gr.update(visible=True),   # custom_model_input
                    gr.update(visible=True),   # custom_model_type
                    gr.update(visible=True),   # check_model_btn
                    gr.update(visible=True)    # model_status
                )
            else:
                return (
                    gr.update(visible=True),   # model_dropdown
                    gr.update(visible=False),  # custom_model_input
                    gr.update(visible=False),  # custom_model_type
                    gr.update(visible=False),  # check_model_btn
                    gr.update(visible=False)   # model_status
                )
        
        model_method.change(
            fn=toggle_model_inputs,
            inputs=[model_method],
            outputs=[model_dropdown, custom_model_input, custom_model_type, check_model_btn, model_status]
        )
        
        check_model_btn.click(
            fn=check_custom_model,
            inputs=[custom_model_input, custom_model_type],
            outputs=[model_status]
        )
        
        # Main processing event
        def process_and_update_all(text_input, file_input, model_method, model_dropdown, custom_model_input, custom_model_type):
            # Determine which model to use
            if model_method == "Custom Model":
                model_to_use = custom_model_input
                model_type_to_use = custom_model_type
            else:
                model_to_use = model_dropdown
                model_type_to_use = None
            
            # Process text and generate embeddings
            status, scatter_fig, similarity_fig, cluster_fig, network_fig = process_text_and_generate_embeddings(
                text_input, file_input, model_to_use, custom_model_input, model_type_to_use
            )
            
            # Generate segments display
            segments_md = ""
            analytics_md = ""
            similarity_stats_md = ""
            cluster_stats_md = ""
            network_info_md = ""
            
            if global_state['texts'] is not None and global_state['embeddings'] is not None:
                # Segments display
                segments_md = "### 📋 Text Segments\n\n"
                for i, segment in enumerate(global_state['texts'][:10]):  # Show first 10
                    segments_md += f"**Segment {i+1}:** {segment[:200]}{'...' if len(segment) > 200 else ''}\n\n"
                
                if len(global_state['texts']) > 10:
                    segments_md += f"... and {len(global_state['texts']) - 10} more segments\n"
                
                # Analytics
                embeddings = global_state['embeddings']
                texts = global_state['texts']
                
                text_lengths = [len(text.split()) for text in texts]
                total_chars = sum(len(text) for text in texts)
                
                analytics_md = f"""
                ### 📊 Detailed Analytics
                
                **Text Statistics:**
                - Total Segments: {len(texts)}
                - Average Words/Segment: {np.mean(text_lengths):.1f}
                - Min/Max Words: {min(text_lengths)} / {max(text_lengths)}
                - Total Characters: {total_chars:,}
                
                **Embedding Statistics:**
                - Embedding Dimension: {embeddings.shape[1]}
                - Embedding Norm (avg): {np.mean(np.linalg.norm(embeddings, axis=1)):.3f}
                """
                
                # Similarity statistics
                sim_matrix = cosine_similarity(embeddings)
                upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
                
                similarity_stats_md = f"""
                **Similarity Analysis:**
                - Mean Similarity: {np.mean(upper_triangle):.3f}
                - Std Similarity: {np.std(upper_triangle):.3f}
                - Min Similarity: {np.min(upper_triangle):.3f}
                - Max Similarity: {np.max(upper_triangle):.3f}
                """
                
                # Cluster statistics (default kmeans with 4 clusters)
                try:
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=min(4, len(texts)), random_state=42)
                    clusters = kmeans.fit_predict(embeddings)
                    cluster_counts = pd.Series(clusters).value_counts().sort_index()
                    
                    cluster_stats_md = f"""
                    **Cluster Analysis (K-means, k=4):**
                    - Silhouette Score: {calculate_silhouette_score(embeddings, clusters):.3f}
                    - Cluster Distribution: {dict(cluster_counts)}
                    """
                except:
                    cluster_stats_md = "Cluster analysis unavailable"
                
                # Network statistics
                edges_count = np.sum(sim_matrix > 0.5) // 2  # Symmetric matrix
                network_info_md = f"""
                **Network Analysis (threshold=0.5):**
                - Total Connections: {edges_count}
                - Network Density: {edges_count / (len(texts) * (len(texts) - 1) / 2):.3f}
                - Average Degree: {2 * edges_count / len(texts):.1f}
                """
            
            return (
                status,
                scatter_fig,
                similarity_fig,
                cluster_fig,
                network_fig,
                segments_md,
                analytics_md,
                similarity_stats_md,
                cluster_stats_md,
                network_info_md
            )
        
        def calculate_silhouette_score(embeddings, clusters):
            """Calculate silhouette score for clustering"""
            try:
                from sklearn.metrics import silhouette_score
                if len(set(clusters)) > 1:
                    return silhouette_score(embeddings, clusters)
                else:
                    return 0.0
            except:
                return 0.0
        
        process_btn.click(
            fn=process_and_update_all,
            inputs=[text_input, file_input, model_method, model_dropdown, custom_model_input, custom_model_type],
            outputs=[
                status_output,
                scatter_plot,
                similarity_plot,
                cluster_plot,
                network_plot,
                segments_display,
                analytics_output,
                similarity_stats,
                cluster_stats,
                network_info
            ]
        )
        
        # Update visualization events
        update_scatter_btn.click(
            fn=update_scatter_plot,
            inputs=[reduction_method, show_3d, plot_3d_type, color_scheme, plot_theme],
            outputs=[scatter_plot]
        )
        
        update_cluster_btn.click(
            fn=update_cluster_plot,
            inputs=[n_clusters, reduction_method, clustering_method, color_scheme, plot_theme],
            outputs=[cluster_plot]
        )
        
        update_network_btn.click(
            fn=update_network_plot,
            inputs=[similarity_threshold, network_layout, plot_theme],
            outputs=[network_plot]
        )
        
        # Auto-update on parameter changes
        for component in [reduction_method, show_3d, plot_3d_type, color_scheme, plot_theme]:
            component.change(
                fn=update_scatter_plot,
                inputs=[reduction_method, show_3d, plot_3d_type, color_scheme, plot_theme],
                outputs=[scatter_plot]
            )
        
        for component in [n_clusters, clustering_method]:
            component.change(
                fn=update_cluster_plot,
                inputs=[n_clusters, reduction_method, clustering_method, color_scheme, plot_theme],
                outputs=[cluster_plot]
            )
        
        for component in [similarity_threshold, network_layout]:
            component.change(
                fn=update_network_plot,
                inputs=[similarity_threshold, network_layout, plot_theme],
                outputs=[network_plot]
            )
        
        # Add example section
        with gr.Row():
            gr.Markdown("""
            ## 📘 Usage Guide
            
            ### 🚀 Quick Start
            1. **Optional**: Enter your HF token for better access
            2. **Select Model**: Choose pre-configured or enter custom model
            3. **Input Text**: Paste Arabic text or upload file (TXT/PDF/DOCX)
            4. **Generate**: Click "Generate Embeddings" to process
            5. **Explore**: View different visualizations and adjust parameters
            
            ### 🤖 Model Recommendations
            - **Speed**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
            - **Quality**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
            - **Arabic-Specific**: `aubmindlab/bert-base-arabertv02`
            
            ### 📊 Visualization Types
            - **Scatter Plot**: 2D/3D semantic space visualization
            - **Similarity Matrix**: Heatmap showing text relationships
            - **Clustering**: Automatic grouping of similar texts
            - **Network Graph**: Connected graph based on similarity
            
            ### 🎯 Advanced Features
            - **3D Plots**: Surface and scatter plots with interactive controls
            - **Multiple Clustering**: K-means, Hierarchical, DBSCAN
            - **Custom Models**: Use any Hugging Face model
            - **Real-time Updates**: Adjust parameters and see instant results
            """)
    
    return app

# Create and launch the interface
if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True,
        show_error=True
    )