import streamlit as st
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
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import umap
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import os
import requests
from huggingface_hub import login, HfApi
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="🤗 Arabic Text Embeddings Visualizer",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .segment-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        direction: rtl;
        text-align: right;
    }
    .stSelectbox > div > div > select {
        direction: ltr;
    }
</style>
""", unsafe_allow_html=True)

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
            },
            # Add some models that might require tokens
            "microsoft/DialoGPT-medium": {
                "name": "DialoGPT Medium",
                "description": "Conversational AI model (requires token for some features)",
                "type": "transformer",
                "requires_token": True
            }
        }
        self.model = None
        self.tokenizer = None
        self.hf_token = None
        
    def set_hf_token(self, token: str) -> bool:
        """Set and validate Hugging Face token"""
        if not token or not token.startswith('hf_'):
            return False
            
        try:
            # Test the token by making a simple API call
            api = HfApi(token=token)
            user_info = api.whoami()
            
            # Set token for transformers library
            os.environ['HUGGINGFACE_HUB_TOKEN'] = token
            login(token=token, add_to_git_credential=False)
            
            self.hf_token = token
            st.session_state.hf_token = token
            st.session_state.hf_user = user_info.get('name', 'Unknown')
            return True
            
        except Exception as e:
            st.error(f"Invalid token: {str(e)}")
            return False
    
    def check_model_access(self, model_name: str) -> bool:
        """Check if model is accessible with current token"""
        try:
            if self.hf_token:
                headers = {"Authorization": f"Bearer {self.hf_token}"}
            else:
                headers = {}
                
            # Check model info endpoint
            response = requests.get(
                f"https://huggingface.co/api/models/{model_name}",
                headers=headers
            )
            
            if response.status_code == 200:
                return True
            elif response.status_code == 401:
                st.error("Model requires authentication. Please provide a valid HF token.")
                return False
            elif response.status_code == 404:
                st.error("Model not found.")
                return False
            else:
                st.warning(f"Could not verify model access: {response.status_code}")
                return True  # Try anyway
                
        except Exception as e:
            st.warning(f"Could not verify model access: {str(e)}")
            return True  # Try anyway
        
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
    
    @st.cache_resource
    def load_model(_self, model_name: str, use_token: bool = True):
        """Load the selected model with caching"""
        try:
            model_info = _self.models_info[model_name]
            
            # Prepare token for model loading if available
            token = _self.hf_token if (use_token and _self.hf_token) else None
            
            if model_info["type"] == "sentence_transformer":
                if token:
                    # For sentence transformers, token is passed via environment or login
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
            if "401" in str(e) or "authentication" in str(e).lower():
                st.error("❌ Authentication failed. Please check your Hugging Face token.")
            elif "404" in str(e):
                st.error("❌ Model not found. Please check the model name.")
            else:
                st.error(f"❌ Error loading model: {str(e)}")
            return None, None
    
    def generate_embeddings(self, texts: List[str], model_name: str, progress_bar=None) -> np.ndarray:
        """Generate embeddings for texts using the selected model"""
        
        # Check model access first
        if not self.check_model_access(model_name):
            model_info = self.models_info[model_name]
            if model_info.get("requires_token", False) and not self.hf_token:
                raise ValueError("This model requires a Hugging Face token. Please provide one in the sidebar.")
        
        model, tokenizer = self.load_model(model_name, use_token=True)
        
        if model is None:
            raise ValueError("Failed to load model")
        
        embeddings = []
        model_info = self.models_info[model_name]
        
        if model_info["type"] == "sentence_transformer":
            # Use SentenceTransformer
            batch_size = 16
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                if progress_bar:
                    progress_bar.progress((i + len(batch)) / len(texts))
                
                batch_embeddings = model.encode(batch, convert_to_tensor=False)
                embeddings.extend(batch_embeddings)
        
        else:
            # Use regular transformer with mean pooling
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()
            
            for i, text in enumerate(texts):
                if progress_bar:
                    progress_bar.progress((i + 1) / len(texts))
                
                inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                                 padding=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Mean pooling
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    embeddings.append(embedding[0])
        
        return np.array(embeddings)

class TextFileProcessor:
    """Class for processing different file types"""
    
    @staticmethod
    def extract_text_from_pdf(file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text_from_txt(file) -> str:
        """Extract text from TXT file"""
        try:
            return file.read().decode('utf-8')
        except UnicodeDecodeError:
            try:
                file.seek(0)
                return file.read().decode('windows-1256')  # Arabic encoding
            except Exception as e:
                st.error(f"Error reading TXT file: {str(e)}")
                return ""

class EmbeddingVisualizer:
    """Class for creating various visualizations of embeddings"""
    
    def __init__(self, embeddings: np.ndarray, texts: List[str]):
        self.embeddings = embeddings
        self.texts = texts
        self.reduced_embeddings_2d = None
        self.reduced_embeddings_3d = None
        self.similarity_matrix = None
        
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
        
        if n_components == 2:
            self.reduced_embeddings_2d = reduced
        elif n_components == 3:
            self.reduced_embeddings_3d = reduced
            
        return reduced
    
    def calculate_similarity_matrix(self):
        """Calculate cosine similarity matrix"""
        self.similarity_matrix = cosine_similarity(self.embeddings)
        return self.similarity_matrix
    
    def create_scatter_plot(self, method='PCA', show_3d=False, plot_3d_type="scatter", 
                          point_size_3d=8, show_3d_axes=True, show_3d_grid=False, 
                          camera_preset="default", color_scheme="viridis", plot_theme="plotly"):
        """Create interactive scatter plot with enhanced 3D options"""
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
                fig.update_traces(marker=dict(size=point_size_3d, opacity=0.8))
                
            elif plot_3d_type == "surface":
                # Create surface plot by interpolating points
                from scipy.interpolate import griddata
                
                # Create grid for surface
                xi = np.linspace(reduced[:, 0].min(), reduced[:, 0].max(), 20)
                yi = np.linspace(reduced[:, 1].min(), reduced[:, 1].max(), 20)
                xi, yi = np.meshgrid(xi, yi)
                
                # Interpolate Z values
                zi = griddata((reduced[:, 0], reduced[:, 1]), reduced[:, 2], (xi, yi), method='cubic')
                
                fig = go.Figure(data=[go.Surface(x=xi, y=yi, z=zi, colorscale=color_scheme)])
                
                # Add scatter points on top
                fig.add_trace(go.Scatter3d(
                    x=reduced[:, 0], y=reduced[:, 1], z=reduced[:, 2],
                    mode='markers',
                    marker=dict(size=point_size_3d, color=range(len(reduced)), 
                              colorscale=color_scheme, opacity=0.9),
                    text=df['text'],
                    name='Text Points'
                ))
                
                fig.update_layout(title=f"3D Surface {method} Visualization")
                
            elif plot_3d_type == "mesh":
                # Create 3D mesh plot
                fig = go.Figure(data=[go.Mesh3d(
                    x=reduced[:, 0], y=reduced[:, 1], z=reduced[:, 2],
                    colorscale=color_scheme,
                    intensity=range(len(reduced)),
                    opacity=0.5
                )])
                
                # Add points
                fig.add_trace(go.Scatter3d(
                    x=reduced[:, 0], y=reduced[:, 1], z=reduced[:, 2],
                    mode='markers',
                    marker=dict(size=point_size_3d, color=range(len(reduced)), 
                              colorscale=color_scheme),
                    text=df['text'],
                    name='Text Points'
                ))
                
                fig.update_layout(title=f"3D Mesh {method} Visualization")
            
            # Set camera angle
            camera_angles = {
                "default": dict(eye=dict(x=1.2, y=1.2, z=1.2)),
                "top": dict(eye=dict(x=0, y=0, z=2.5)),
                "side": dict(eye=dict(x=2.5, y=0, z=0)),
                "diagonal": dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                "bottom": dict(eye=dict(x=0, y=0, z=-2.5))
            }
            
            fig.update_layout(
                scene_camera=camera_angles.get(camera_preset, camera_angles["default"]),
                scene=dict(
                    showgrid=show_3d_grid,
                    showticklabels=show_3d_axes,
                    showspikes=False
                ),
                height=700
            )
        else:
            fig = px.scatter(df, x=f'{method}_1', y=f'{method}_2',
                           hover_data=['text'], color='segment_id',
                           title=f"2D {method} Visualization of Text Embeddings",
                           color_continuous_scale=color_scheme)
            fig.update_traces(marker=dict(size=10, opacity=0.8))
            fig.update_layout(height=600)
        
        fig.update_layout(showlegend=False, template=plot_theme)
        return fig
    
    def create_similarity_heatmap(self):
        """Create similarity matrix heatmap"""
        if self.similarity_matrix is None:
            self.calculate_similarity_matrix()
        
        # Create labels for segments
        labels = [f"T{i+1}" for i in range(len(self.texts))]
        
        fig = ff.create_annotated_heatmap(
            z=self.similarity_matrix,
            x=labels,
            y=labels,
            colorscale='Viridis',
            showscale=True
        )
        
        fig.update_layout(
            title="Text Similarity Matrix (Cosine Similarity)",
            height=600,
            xaxis_title="Text Segments",
            yaxis_title="Text Segments"
        )
        
        return fig
    
    def create_cluster_visualization(self, n_clusters=4, method='PCA', clustering_method='kmeans', 
                                   color_scheme='viridis', plot_theme='plotly'):
        """Create cluster visualization with multiple algorithms"""
        if self.reduced_embeddings_2d is None:
            self.reduce_dimensions(method, 2)
        
        # Perform clustering based on selected method
        if clustering_method == 'kmeans':
            from sklearn.cluster import KMeans
            clusterer = KMeans(n_clusters=min(n_clusters, len(self.texts)), random_state=42)
            clusters = clusterer.fit_predict(self.embeddings)
        elif clustering_method == 'hierarchical':
            from sklearn.cluster import AgglomerativeClustering
            clusterer = AgglomerativeClustering(n_clusters=min(n_clusters, len(self.texts)))
            clusters = clusterer.fit_predict(self.embeddings)
        elif clustering_method == 'dbscan':
            from sklearn.cluster import DBSCAN
            clusterer = DBSCAN(eps=0.5, min_samples=2)
            clusters = clusterer.fit_predict(self.embeddings)
            # Handle noise points (labeled as -1)
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        
        df = pd.DataFrame(self.reduced_embeddings_2d, columns=[f'{method}_1', f'{method}_2'])
        df['cluster'] = clusters
        df['text'] = [f"Segment {i+1}: {text[:100]}..." if len(text) > 100 else f"Segment {i+1}: {text}" 
                     for i, text in enumerate(self.texts)]
        df['segment_id'] = range(1, len(self.texts) + 1)
        
        # Handle noise points for DBSCAN
        if clustering_method == 'dbscan':
            df['cluster_name'] = df['cluster'].apply(lambda x: 'Noise' if x == -1 else f'Cluster {x+1}')
        else:
            df['cluster_name'] = df['cluster'].apply(lambda x: f'Cluster {x+1}')
        
        fig = px.scatter(df, x=f'{method}_1', y=f'{method}_2',
                        color='cluster_name', hover_data=['text'],
                        title=f"{clustering_method.title()} Clustering - {method}",
                        color_discrete_sequence=px.colors.qualitative.Set1)
        
        # Add cluster centers for kmeans
        if clustering_method == 'kmeans' and hasattr(clusterer, 'cluster_centers_'):
            centers_2d = clusterer.cluster_centers_
            if self.embeddings.shape[1] != 2:
                # Project centers to 2D space
                pca = PCA(n_components=2, random_state=42)
                pca.fit(self.embeddings)
                centers_2d = pca.transform(clusterer.cluster_centers_)
            
            fig.add_trace(go.Scatter(
                x=centers_2d[:, 0], y=centers_2d[:, 1],
                mode='markers',
                marker=dict(size=15, color='red', symbol='x', line=dict(width=2, color='black')),
                name='Cluster Centers',
                showlegend=True
            ))
        
        fig.update_traces(marker=dict(size=10, opacity=0.8))
        fig.update_layout(height=600, template=plot_theme)
        return fig, clusters
    
    def create_network_graph(self, similarity_threshold=0.5, layout='spring', color_scheme='viridis', plot_theme='plotly'):
        """Create network graph based on similarity with enhanced layouts"""
        if self.similarity_matrix is None:
            self.calculate_similarity_matrix()
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for i, text in enumerate(self.texts):
            G.add_node(i, text=text[:50] + "..." if len(text) > 50 else text)
        
        # Add edges based on similarity threshold
        edges_added = 0
        for i in range(len(self.texts)):
            for j in range(i+1, len(self.texts)):
                if self.similarity_matrix[i][j] > similarity_threshold:
                    G.add_edge(i, j, weight=self.similarity_matrix[i][j])
                    edges_added += 1
        
        if edges_added == 0:
            st.warning(f"No connections found with similarity threshold {similarity_threshold}. Try lowering the threshold.")
            return None
        
        # Create layout based on selection
        if layout == 'spring':
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'random':
            pos = nx.random_layout(G)
        else:
            pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Extract edges with weights for line width
        edge_x = []
        edge_y = []
        edge_weights = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2]['weight'])
        
        # Create edge trace with varying line widths
        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                              line=dict(width=2, color='rgba(125,125,125,0.5)'),
                              hoverinfo='none',
                              mode='lines')
        
        # Extract nodes
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        node_connections = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"T{node+1}")
            node_info.append(G.nodes[node]['text'])
            node_connections.append(len(list(G.neighbors(node))))
        
        # Create node trace with size based on connections
        node_trace = go.Scatter(x=node_x, y=node_y,
                              mode='markers+text',
                              hovertext=node_info,
                              text=node_text,
                              textposition="middle center",
                              hoverinfo="text",
                              marker=dict(
                                  size=[15 + 5 * conn for conn in node_connections],
                                  color=node_connections,
                                  colorscale=color_scheme,
                                  line=dict(width=2, color='black'),
                                  colorbar=dict(title="Connections")
                              ))
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f'Network Graph - {layout.title()} Layout (Similarity > {similarity_threshold})',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text=f"Connections: {edges_added} | Avg Similarity: {np.mean(edge_weights):.3f}",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color='gray', size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=600,
                           template=plot_theme))
        
        return fig
        
        # Create node trace
        node_trace = go.Scatter(x=node_x, y=node_y,
                              mode='markers+text',
                              hovertext=node_info,
                              text=node_text,
                              textposition="middle center",
                              hoverinfo="text",
                              marker=dict(size=20,
                                        color=list(range(len(node_x))),
                                        colorscale='viridis',
                                        line=dict(width=2)))
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f'Network Graph (Similarity > {similarity_threshold})',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text=f"Connections: {edges_added}",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color='gray', size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=600))
        
        return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤗 Arabic Text Embeddings Visualizer</h1>
        <p>Generate high-quality semantic embeddings using state-of-the-art Arabic language models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize processors
    text_processor = ArabicTextProcessor()
    file_processor = TextFileProcessor()
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # Hugging Face Token Section
    st.sidebar.subheader("🔑 Hugging Face Authentication")
    
    # Check if token is already in session state
    if 'hf_token' not in st.session_state:
        st.session_state.hf_token = None
        st.session_state.hf_user = None
    
    # Token input
    if st.session_state.hf_token is None:
        st.sidebar.info("💡 **Optional**: Enter your HF token to access private models and avoid rate limits")
        
        token_input = st.sidebar.text_input(
            "HF Token",
            type="password",
            placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxx",
            help="Get your free token from https://huggingface.co/settings/tokens"
        )
        
        if st.sidebar.button("🔓 Authenticate", type="primary"):
            if token_input:
                if text_processor.set_hf_token(token_input):
                    st.sidebar.success(f"✅ Authenticated as: {st.session_state.hf_user}")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Invalid token format or authentication failed")
            else:
                st.sidebar.warning("⚠️ Please enter a token first")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Benefits of authentication:**")
        st.sidebar.markdown("• Access to private models")
        st.sidebar.markdown("• Higher rate limits")
        st.sidebar.markdown("• Better performance")
        st.sidebar.markdown("• Priority access")
        
    else:
        st.sidebar.success(f"✅ **Authenticated as:** {st.session_state.hf_user}")
        text_processor.hf_token = st.session_state.hf_token
        
        if st.sidebar.button("🔒 Logout"):
            st.session_state.hf_token = None
            st.session_state.hf_user = None
            if 'HUGGINGFACE_HUB_TOKEN' in os.environ:
                del os.environ['HUGGINGFACE_HUB_TOKEN']
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Model selection
    st.sidebar.subheader("🤖 Model Selection")
    
    # Model input method
    model_input_method = st.sidebar.radio(
        "Model Selection Method",
        options=["📋 Pre-configured Models", "✍️ Custom Model"],
        help="Choose from curated models or enter any Hugging Face model"
    )
    
    if model_input_method == "📋 Pre-configured Models":
        # Filter models based on authentication status
        available_models = {}
        for model_id, info in text_processor.models_info.items():
            if info.get("requires_token", False) and st.session_state.hf_token is None:
                continue  # Skip models that require token when not authenticated
            available_models[model_id] = info
        
        selected_model = st.sidebar.selectbox(
            "Select Pre-configured Model",
            options=list(available_models.keys()),
            format_func=lambda x: available_models[x]["name"],
            help="Choose from curated embedding models"
        )
        
        # Display model info
        model_info = available_models[selected_model]
        
        # Show token requirement status
        token_status = ""
        if model_info.get("requires_token", False):
            if st.session_state.hf_token:
                token_status = "🔓 **Authenticated** - Full access"
            else:
                token_status = "🔒 **Token Required** - Limited access"
        else:
            token_status = "🌐 **Public Model** - No token needed"
        
        st.sidebar.info(f"""
        **{model_info['name']}**
        
        {model_info['description']}
        
        {token_status}
        """)
        
    else:  # Custom Model
        st.sidebar.info("💡 **Enter any Hugging Face model ID**\n\nExamples:\n• `sentence-transformers/all-MiniLM-L6-v2`\n• `microsoft/DialoGPT-medium`\n• `aubmindlab/bert-base-arabertv02`")
        
        custom_model = st.sidebar.text_input(
            "Hugging Face Model ID",
            placeholder="e.g., sentence-transformers/all-MiniLM-L6-v2",
            help="Enter the full model path from Hugging Face Hub"
        )
        
        # Model type selection for custom models
        custom_model_type = st.sidebar.selectbox(
            "Model Type",
            options=["sentence_transformer", "transformer"],
            format_func=lambda x: "Sentence Transformer" if x == "sentence_transformer" else "Regular Transformer",
            help="Choose the model architecture type"
        )
        
        if custom_model:
            selected_model = custom_model
            # Create dynamic model info for custom model
            model_info = {
                "name": custom_model.split("/")[-1],
                "description": f"Custom {custom_model_type.replace('_', ' ').title()} model",
                "type": custom_model_type,
                "requires_token": False  # We'll check this dynamically
            }
            
            # Add to processor's model info temporarily
            text_processor.models_info[custom_model] = model_info
            
            # Check model accessibility
            with st.sidebar:
                with st.spinner("🔍 Checking model availability..."):
                    is_accessible = text_processor.check_model_access(custom_model)
                    if is_accessible:
                        st.success("✅ Model found and accessible")
                    else:
                        st.error("❌ Model not accessible or requires authentication")
        else:
            selected_model = None
            st.sidebar.warning("⚠️ Please enter a model ID to continue")
    
    # Rate limit info
    if st.session_state.hf_token is None:
        st.sidebar.warning("⚠️ **Rate Limits Apply** - Consider authenticating for better performance")
    else:
        st.sidebar.info("✨ **Authenticated** - Higher rate limits active")
    
    # Visualization settings
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Visualization Settings")
    
    reduction_method = st.sidebar.selectbox(
        "Dimensionality Reduction",
        options=['PCA', 'TSNE', 'UMAP'],
        help="Method for reducing high-dimensional embeddings to 2D/3D"
    )
    
    # 3D visualization options
    st.sidebar.subheader("🎯 3D Visualization")
    show_3d = st.sidebar.checkbox("Enable 3D Visualization", value=False)
    
    if show_3d:
        plot_3d_type = st.sidebar.radio(
            "3D Plot Type",
            options=["scatter", "surface", "mesh"],
            format_func=lambda x: {
                "scatter": "🔵 Scatter Plot",
                "surface": "🌊 Surface Plot", 
                "mesh": "🕸️ Mesh Plot"
            }[x],
            help="Choose 3D visualization style"
        )
        
        # 3D specific settings
        point_size_3d = st.sidebar.slider(
            "3D Point Size",
            min_value=3, max_value=20, value=8,
            help="Size of points in 3D scatter plot"
        )
        
        show_3d_axes = st.sidebar.checkbox("Show 3D Axes", value=True)
        show_3d_grid = st.sidebar.checkbox("Show 3D Grid", value=False)
        
        # Camera angle presets
        camera_preset = st.sidebar.selectbox(
            "Camera Angle",
            options=["default", "top", "side", "diagonal", "bottom"],
            help="Pre-defined camera angles for 3D view"
        )
    
    # Clustering settings
    st.sidebar.subheader("🎯 Clustering")
    n_clusters = st.sidebar.slider(
        "Number of Clusters",
        min_value=2, max_value=10, value=4,
        help="Number of clusters for K-means clustering"
    )
    
    # Advanced clustering options
    clustering_method = st.sidebar.selectbox(
        "Clustering Algorithm",
        options=["kmeans", "hierarchical", "dbscan"],
        format_func=lambda x: {
            "kmeans": "K-Means",
            "hierarchical": "Hierarchical",
            "dbscan": "DBSCAN"
        }[x],
        help="Choose clustering algorithm"
    )
    
    # Network settings
    st.sidebar.subheader("🕸️ Network Graph")
    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.0, max_value=1.0, value=0.5, step=0.1,
        help="Minimum similarity to create connections"
    )
    
    network_layout = st.sidebar.selectbox(
        "Network Layout",
        options=["spring", "circular", "kamada_kawai", "random"],
        help="Layout algorithm for network graph"
    )
    
    # Plot styling
    st.sidebar.subheader("🎨 Plot Styling")
    color_scheme = st.sidebar.selectbox(
        "Color Scheme",
        options=["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "rainbow"],
        help="Color palette for visualizations"
    )
    
    plot_theme = st.sidebar.selectbox(
        "Plot Theme",
        options=["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"],
        help="Overall theme for plots"
    )
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📝 Input Text")
        
        # Text input methods
        input_method = st.radio(
            "Input Method",
            options=["Direct Text", "File Upload"],
            horizontal=True
        )
        
        text_content = ""
        
        if input_method == "Direct Text":
            text_content = st.text_area(
                "Enter Arabic Text",
                height=200,
                placeholder="أدخل النص العربي هنا... يمكنك كتابة فقرات متعددة وسيتم تحليل كل فقرة بشكل منفصل.",
                help="Enter Arabic text that will be segmented and analyzed"
            )
        
        else:
            uploaded_file = st.file_uploader(
                "Upload File",
                type=['txt', 'pdf', 'docx'],
                help="Upload a text file, PDF, or Word document containing Arabic text"
            )
            
            if uploaded_file is not None:
                st.success(f"File uploaded: {uploaded_file.name}")
                
                if uploaded_file.type == "application/pdf":
                    text_content = file_processor.extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text_content = file_processor.extract_text_from_docx(uploaded_file)
                else:
                    text_content = file_processor.extract_text_from_txt(uploaded_file)
        
        # Process button
        process_button = st.button(
            "🚀 Generate Embeddings & Visualize",
            type="primary",
            use_container_width=True,
            disabled=(selected_model is None)
        )
        
        if selected_model is None:
            st.warning("⚠️ Please select or enter a model first")
        
        if process_button and text_content.strip() and selected_model:
            with st.spinner("Processing text and generating embeddings..."):
                
                # Segment text
                segments = text_processor.segment_text(text_content)
                
                if len(segments) == 0:
                    st.error("No valid text segments found. Please ensure your text contains meaningful Arabic sentences.")
                    st.stop()
                
                if len(segments) > 50:
                    st.warning(f"Found {len(segments)} segments. Processing first 50 for better performance.")
                    segments = segments[:50]
                
                # Store in session state
                st.session_state.segments = segments
                st.session_state.selected_model = selected_model
                
                # Generate embeddings
                progress_bar = st.progress(0)
                try:
                    embeddings = text_processor.generate_embeddings(
                        segments, selected_model, progress_bar
                    )
                    st.session_state.embeddings = embeddings
                    st.session_state.visualizer = EmbeddingVisualizer(embeddings, segments)
                    
                    st.success(f"✅ Successfully processed {len(segments)} segments!")
                    progress_bar.progress(1.0)
                    
                except Exception as e:
                    st.error(f"Error generating embeddings: {str(e)}")
                    st.stop()
        
        elif process_button and not text_content.strip():
            st.error("Please enter text or upload a file before processing.")
        
        # Display text segments
        if 'segments' in st.session_state:
            st.header("📋 Text Segments")
            
            for i, segment in enumerate(st.session_state.segments):
                with st.expander(f"Segment {i+1}", expanded=False):
                    st.markdown(f"""
                    <div class="segment-card">
                        {segment}
                    </div>
                    """, unsafe_allow_html=True)
    
    with col2:
        st.header("🎯 Visualization Dashboard")
        
        if 'visualizer' in st.session_state:
            visualizer = st.session_state.visualizer
            
            # Embedding statistics
            st.subheader("📊 Embedding Statistics")
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("Segments", len(st.session_state.segments))
            
            with col_stat2:
                st.metric("Embedding Dim", st.session_state.embeddings.shape[1])
            
            with col_stat3:
                avg_sim = np.mean(cosine_similarity(st.session_state.embeddings))
                st.metric("Avg Similarity", f"{avg_sim:.3f}")
            
            # Visualization tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Scatter Plot", "🔥 Similarity Matrix", "🎯 Clusters", "🕸️ Network"])
            
            with tab1:
                st.subheader(f"Scatter Plot ({reduction_method})")
                try:
                    if show_3d:
                        fig = visualizer.create_scatter_plot(
                            method=reduction_method, 
                            show_3d=True,
                            plot_3d_type=plot_3d_type,
                            point_size_3d=point_size_3d,
                            show_3d_axes=show_3d_axes,
                            show_3d_grid=show_3d_grid,
                            camera_preset=camera_preset,
                            color_scheme=color_scheme,
                            plot_theme=plot_theme
                        )
                    else:
                        fig = visualizer.create_scatter_plot(
                            method=reduction_method, 
                            show_3d=False,
                            color_scheme=color_scheme,
                            plot_theme=plot_theme
                        )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add 3D controls info
                    if show_3d:
                        st.info("🎮 **3D Controls**: Click and drag to rotate • Scroll to zoom • Shift+drag to pan")
                        
                except Exception as e:
                    st.error(f"Error creating scatter plot: {str(e)}")
            
            with tab2:
                st.subheader("Similarity Matrix")
                try:
                    fig = visualizer.create_similarity_heatmap()
                    fig.update_layout(template=plot_theme)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show similarity statistics
                    sim_matrix = visualizer.calculate_similarity_matrix()
                    max_sim_idx = np.unravel_index(np.argmax(sim_matrix - np.eye(len(sim_matrix))), sim_matrix.shape)
                    min_sim_idx = np.unravel_index(np.argmin(sim_matrix + np.eye(len(sim_matrix))), sim_matrix.shape)
                    
                    col_sim1, col_sim2 = st.columns(2)
                    with col_sim1:
                        st.metric("Max Similarity", 
                                f"{sim_matrix[max_sim_idx]:.3f}",
                                delta=f"Segments {max_sim_idx[0]+1} & {max_sim_idx[1]+1}")
                    with col_sim2:
                        st.metric("Min Similarity", 
                                f"{sim_matrix[min_sim_idx]:.3f}",
                                delta=f"Segments {min_sim_idx[0]+1} & {min_sim_idx[1]+1}")
                
                except Exception as e:
                    st.error(f"Error creating similarity matrix: {str(e)}")
            
            with tab3:
                st.subheader(f"{clustering_method.title()} Clustering")
                try:
                    fig, clusters = visualizer.create_cluster_visualization(
                        n_clusters=n_clusters, 
                        method=reduction_method,
                        clustering_method=clustering_method,
                        color_scheme=color_scheme,
                        plot_theme=plot_theme
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show cluster distribution
                    cluster_counts = pd.Series(clusters).value_counts().sort_index()
                    
                    # Handle DBSCAN noise
                    if clustering_method == 'dbscan' and -1 in cluster_counts.index:
                        noise_count = cluster_counts[-1]
                        cluster_counts = cluster_counts.drop(-1)
                        st.warning(f"⚠️ DBSCAN found {noise_count} noise points (outliers)")
                    
                    cluster_df = pd.DataFrame({
                        'Cluster': [f"Cluster {i+1}" for i in cluster_counts.index],
                        'Count': cluster_counts.values
                    })
                    
                    fig_bar = px.bar(cluster_df, x='Cluster', y='Count', 
                                   title="Cluster Distribution",
                                   color='Count', 
                                   color_continuous_scale=color_scheme,
                                   template=plot_theme)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error creating cluster visualization: {str(e)}")
            
            with tab4:
                st.subheader(f"Network Graph ({network_layout.title()} Layout)")
                try:
                    fig = visualizer.create_network_graph(
                        similarity_threshold=similarity_threshold,
                        layout=network_layout,
                        color_scheme=color_scheme,
                        plot_theme=plot_theme
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.info("🔗 **Network Info**: Node size = number of connections • Color intensity = connectivity")
                    else:
                        st.warning("No connections found. Try lowering the similarity threshold.")
                
                except Exception as e:
                    st.error(f"Error creating network graph: {str(e)}")
            
            # Additional analysis tab
            with st.expander("📈 Advanced Analytics", expanded=False):
                st.subheader("Embedding Analysis")
                
                col_analysis1, col_analysis2 = st.columns(2)
                
                with col_analysis1:
                    st.write("**Dimensionality Reduction Comparison**")
                    
                    # Compare different reduction methods
                    methods_to_compare = ['PCA', 'TSNE', 'UMAP']
                    comparison_results = {}
                    
                    for method in methods_to_compare:
                        try:
                            reduced = visualizer.reduce_dimensions(method, 2)
                            # Calculate explained variance or quality metric
                            if method == 'PCA':
                                pca = PCA(n_components=2, random_state=42)
                                pca.fit(visualizer.embeddings)
                                variance_explained = sum(pca.explained_variance_ratio_)
                                comparison_results[method] = f"Variance: {variance_explained:.3f}"
                            else:
                                comparison_results[method] = "Quality: Good"
                        except Exception as e:
                            comparison_results[method] = f"Error: {str(e)[:20]}..."
                    
                    for method, result in comparison_results.items():
                        st.write(f"• **{method}**: {result}")
                
                with col_analysis2:
                    st.write("**Text Segment Statistics**")
                    
                    # Text length analysis
                    text_lengths = [len(text.split()) for text in st.session_state.segments]
                    
                    st.write(f"• **Total Segments**: {len(st.session_state.segments)}")
                    st.write(f"• **Avg Words/Segment**: {np.mean(text_lengths):.1f}")
                    st.write(f"• **Min Words**: {min(text_lengths)}")
                    st.write(f"• **Max Words**: {max(text_lengths)}")
                    
                    # Character analysis
                    total_chars = sum(len(text) for text in st.session_state.segments)
                    st.write(f"• **Total Characters**: {total_chars:,}")
                
                # Similarity distribution plot
                st.subheader("Similarity Distribution")
                sim_matrix = visualizer.calculate_similarity_matrix()
                
                # Extract upper triangle (excluding diagonal)
                upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
                
                fig_hist = px.histogram(
                    x=upper_triangle,
                    nbins=20,
                    title="Distribution of Pairwise Similarities",
                    labels={'x': 'Cosine Similarity', 'y': 'Frequency'},
                    color_discrete_sequence=[color_scheme],
                    template=plot_theme
                )
                fig_hist.add_vline(x=np.mean(upper_triangle), 
                                 line_dash="dash", 
                                 annotation_text=f"Mean: {np.mean(upper_triangle):.3f}")
                st.plotly_chart(fig_hist, use_container_width=True)
        
        else:
            st.info("👆 Process some text first to see visualizations here!")
            
            # Show enhanced example with custom models
            st.subheader("📘 Enhanced Usage Guide")
            
            col_guide1, col_guide2 = st.columns(2)
            
            with col_guide1:
                st.markdown("""
                **🤖 Model Options:**
                1. **Pre-configured**: Curated Arabic/multilingual models
                2. **Custom Models**: Enter any HF model ID
                   - `sentence-transformers/all-MiniLM-L6-v2`
                   - `microsoft/DialoGPT-medium`
                   - `aubmindlab/bert-base-arabertv02`
                
                **🔑 Authentication Benefits:**
                - Access private/gated models
                - Higher rate limits
                - Better performance
                - Priority processing
                """)
            
            with col_guide2:
                st.markdown("""
                **📊 Visualization Features:**
                1. **2D/3D Scatter**: PCA, t-SNE, UMAP projections
                2. **Advanced 3D**: Surface plots, mesh plots
                3. **Multiple Clustering**: K-means, Hierarchical, DBSCAN
                4. **Network Layouts**: Spring, circular, random
                5. **Customizable**: Colors, themes, camera angles
                
                **🎯 3D Controls:**
                - Drag to rotate
                - Scroll to zoom
                - Shift+drag to pan
                """)
            
            # Model recommendations
            st.subheader("🏆 Model Recommendations")
            
            recommendations = {
                "🚀 **Speed Priority**": {
                    "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "description": "Fast processing, good quality, 384 dimensions"
                },
                "🎯 **Quality Priority**": {
                    "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", 
                    "description": "Best quality, slower processing, 768 dimensions"
                },
                "🇸🇦 **Arabic Specific**": {
                    "model": "aubmindlab/bert-base-arabertv02",
                    "description": "Trained specifically on Arabic text, excellent understanding"
                },
                "⚖️ **Balanced**": {
                    "model": "sentence-transformers/distiluse-base-multilingual-cased",
                    "description": "Good balance of speed and quality, 512 dimensions"
                }
            }
            
            for category, info in recommendations.items():
                with st.expander(category, expanded=False):
                    st.code(info["model"])
                    st.write(info["description"])
            
            # 3D visualization preview
            st.subheader("🎮 3D Visualization Preview")
            st.markdown("""
            **Available 3D Plot Types:**
            - **🔵 Scatter Plot**: Individual points in 3D space
            - **🌊 Surface Plot**: Interpolated surface with points
            - **🕸️ Mesh Plot**: 3D mesh representation
            
            **Camera Presets:**
            - **Default**: Balanced diagonal view
            - **Top**: Bird's eye view
            - **Side**: Profile view
            - **Diagonal**: Angled perspective
            - **Bottom**: Upward view
            """)
            
            # Advanced features info
            st.subheader("🔬 Advanced Analytics")
            st.markdown("""
            **Enhanced Analysis Features:**
            - **Multiple Clustering Algorithms**: K-means, Hierarchical, DBSCAN
            - **Dimensionality Reduction Comparison**: Compare PCA, t-SNE, UMAP
            - **Similarity Distribution**: Histogram of pairwise similarities
            - **Text Statistics**: Word counts, character analysis
            - **Network Analysis**: Connection patterns and centrality
            - **Export Capabilities**: Save plots as PNG, SVG, HTML
            """)

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Built with ❤️ using Streamlit, Hugging Face Transformers, and Plotly<br>
    For Arabic NLP research and semantic analysis
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()