## Complete Streamlit RAG Application

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Fix OpenMP library conflict (common with FAISS/PyTorch)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Configure Unicode support for Windows
import sys
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except:
            pass

import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import os
import tempfile
import time
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import our proven multi-source RAG architecture
import logging
from typing import Dict, List, Any, Optional

# Core components - proven architecture
from src.rag_system import RAGSystem
from src.salesforce_connector import SalesforceConnector
from src.colpali_retriever import ColPaliRetriever
from src.cross_encoder_reranker import CrossEncoderReRanker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress excessive httpx logging from OpenAI API calls
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Configure Streamlit page with file upload settings
st.set_page_config(
    page_title="🤖 Smart Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure Streamlit to handle file uploads properly and prevent 400 errors
os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '200'  # 200MB max upload
os.environ['STREAMLIT_SERVER_MAX_MESSAGE_SIZE'] = '200'  # 200MB max message
os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'  # Disable CORS to prevent conflicts
os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'  # Disable XSRF for upload stability
os.environ['STREAMLIT_CLIENT_TOOLBAR_MODE'] = 'minimal'  # Reduce UI overhead
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'  # Reduce file system monitoring

# Production Glassmorphism CSS - Complete System from Prototype
def load_glassmorphism_css():
    st.markdown("""
    <style>
    /* TRUE GLASSMORPHISM v9.0 - COMPLETE REDESIGN */
    /* Import elegant fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    /* VIBRANT GLASSMORPHIC BACKGROUND - Essential for glass effect */
    .stApp {
        background: 
            radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.2) 0%, transparent 50%),
            linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%) !important;
        font-family: 'Inter', sans-serif !important;
        min-height: 100vh !important;
    }
    
    /* Enhanced glassmorphic base styles */
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* QUERY INPUT FIELD ENHANCEMENT - High contrast for 2560x1440 displays */
    .stTextInput > div > div > input {
        background: rgba(0, 0, 0, 0.45) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        padding: 0.75rem 1rem !important;
        box-shadow: 
            0 4px 16px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        transition: all 0.3s ease !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.9) !important;
    }
    
    .stTextInput > div > div > input:focus {
        background: rgba(0, 0, 0, 0.55) !important;
        border-color: #64b5f6 !important;
        box-shadow: 
            0 0 0 2px rgba(100, 181, 246, 0.4), 
            0 6px 20px rgba(0, 0, 0, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
        outline: none !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.8) !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.7) !important;
    }
    
    /* Ensure text input labels are visible */
    .stTextInput > label {
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 500 !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Main content area styling */
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: none !important;
    }
    
    /* Enhanced Glass container base class - 2025 Design */
    .glass-container {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border-radius: 18px !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        padding: 0.75rem !important;
        margin: 0.4rem 0 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
    }
    
    .glass-container::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        border-radius: 18px !important;
        padding: 1px !important;
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05)) !important;
        mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0) !important;
        mask-composite: exclude !important;
        -webkit-mask-composite: xor !important;
        pointer-events: none !important;
    }
    
    .glass-container:hover {
        background: rgba(255, 255, 255, 0.12) !important;
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 
            0 16px 48px rgba(0, 0, 0, 0.35),
            inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
    }
    
    /* Glass header styling */
    .glass-header {
        background: rgba(255, 255, 255, 0.12) !important;
        backdrop-filter: blur(20px) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        padding: 1rem 1.5rem !important;
        margin-bottom: 1rem !important;
        text-align: center !important;
    }
    
    .glass-header h1 {
        color: #ffffff !important;
        font-size: 2.2rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3) !important;
    }
    
    .glass-header p {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 1rem !important;
        margin: 0.5rem 0 0 0 !important;
        font-weight: 300 !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(0, 0, 0, 0.4) !important;
        backdrop-filter: blur(15px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Sidebar content styling */
    .css-1d391kg .stMarkdown,
    .css-1d391kg .stButton,
    .css-1d391kg h1,
    .css-1d391kg h2,
    .css-1d391kg h3,
    .css-1d391kg h4 {
        color: #ffffff !important;
    }
    
    /* Sidebar button styling */
    .css-1d391kg .stButton > button {
        background: rgba(100, 181, 246, 0.2) !important;
        border: 1px solid rgba(100, 181, 246, 0.4) !important;
        color: #64b5f6 !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Sidebar text elements */
    .css-1d391kg p,
    .css-1d391kg span,
    .css-1d391kg div {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Sidebar specific overrides */
    [data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.4) !important;
        backdrop-filter: blur(15px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(100, 181, 246, 0.2) !important;
        border: 1px solid rgba(100, 181, 246, 0.4) !important;
        color: #64b5f6 !important;
        backdrop-filter: blur(10px) !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(100, 181, 246, 0.3) !important;
        border-color: rgba(100, 181, 246, 0.6) !important;
        color: #ffffff !important;
    }
    
    /* Text styling improvements */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    p, div, span, label {
        color: rgba(255, 255, 255, 0.85) !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Chat bubble styling */
    .chat-bubble {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(12px) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        padding: 0.75rem 1rem !important;
        margin: 0.5rem 0 !important;
        transition: all 0.3s ease !important;
    }
    
    .chat-bubble:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        border-color: rgba(100, 181, 246, 0.4) !important;
    }
    
    .chat-query {
        color: #81c784 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .chat-answer {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 0.9rem !important;
        line-height: 1.5 !important;
        margin-bottom: 0.75rem !important;
    }
    
    .chat-meta {
        color: rgba(255, 255, 255, 0.6) !important;
        font-size: 0.75rem !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: rgba(100, 181, 246, 0.2) !important;
        border: 1px solid rgba(100, 181, 246, 0.3) !important;
        border-radius: 12px !important;
        color: #64b5f6 !important;
        font-weight: 500 !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: rgba(100, 181, 246, 0.3) !important;
        border-color: rgba(100, 181, 246, 0.5) !important;
        transform: translateY(-2px) !important;
        color: #ffffff !important;
    }
    
    /* Form Submit Button styling - Fix for Search All Sources button */
    [data-testid="stFormSubmitButton"] > button,
    .stFormSubmitButton > button {
        background: rgba(100, 181, 246, 0.2) !important;
        border: 1px solid rgba(100, 181, 246, 0.3) !important;
        border-radius: 12px !important;
        color: #64b5f6 !important;
        font-weight: 500 !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFormSubmitButton"] > button:hover,
    .stFormSubmitButton > button:hover {
        background: rgba(100, 181, 246, 0.3) !important;
        border-color: rgba(100, 181, 246, 0.5) !important;
        transform: translateY(-2px) !important;
        color: #ffffff !important;
    }
    
    /* File Uploader styling - Better text readability */
    .stFileUploader {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    .stFileUploader > label {
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 500 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
    }
    
    .stFileUploader small {
        color: rgba(255, 255, 255, 0.7) !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5) !important;
    }
    
    /* File uploader drag and drop area */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 2px dashed rgba(100, 181, 246, 0.3) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    [data-testid="stFileUploaderDropzone"] > div {
        color: rgba(255, 255, 255, 0.8) !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5) !important;
    }
    
    /* Success/Warning/Error message styling for glassmorphism */
    .source-selected {
        background: rgba(129, 199, 132, 0.15) !important;
        border: 1px solid rgba(129, 199, 132, 0.3) !important;
        color: #81c784 !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 12px !important;
        padding: 0.75rem !important;
        margin: 1rem 0 !important;
    }
    
    .rejected-sources {
        background: rgba(255, 183, 77, 0.15) !important;
        border: 1px solid rgba(255, 183, 77, 0.3) !important;
        color: #ffb74d !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 12px !important;
        padding: 0.5rem !important;
        margin: 0.5rem 0 !important;
        font-size: 0.9rem !important;
    }
    
    /* Hide Streamlit branding but keep native sidebar toggle functional */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* HIDE sidebar toggle completely for fixed layout */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    button[data-testid="collapsedControl"],
    button[data-testid="stSidebarCollapseButton"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Hide the button container entirely */
    .css-1y4p8pa,
    .css-1vencpc {
        display: none !important;
    }
    
    /* Comprehensive white box removal and background override */
    .element-container {
        margin: 0 !important;
        background: transparent !important;
    }
    
    /* Ensure no white backgrounds appear */
    .stApp > div {
        background: transparent !important;
    }
    
    /* Remove extra spacing and white containers */
    .block-container > div {
        gap: 0.5rem !important;
        background: transparent !important;
    }
    
    /* NUCLEAR APPROACH - FORCE EVERY POSSIBLE ELEMENT TO BE TRANSPARENT */
    /* Maximum specificity override for EVERYTHING in sidebar */
    html body div[data-testid="stSidebar"] * {
        background-color: transparent !important;
        background: transparent !important;
    }
    
    html body div[data-testid="stSidebar"] div {
        background-color: transparent !important;
        background: transparent !important;
    }
    
    html body div[data-testid="stSidebar"] .stMarkdown {
        background-color: transparent !important;
        background: transparent !important;
    }
    
    html body div[data-testid="stSidebar"] .stMarkdown div {
        background-color: transparent !important;
        background: transparent !important;
    }
    
    html body div[data-testid="stSidebar"] .element-container {
        background-color: transparent !important;
        background: transparent !important;
    }
    
    /* FORCE specific white box areas */
    html body div[data-testid="stSidebar"] [style*="background"] {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Exception: Keep only our glass elements */
    html body div[data-testid="stSidebar"] .status-card {
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(16px) saturate(160%) !important;
    }
    
    html body div[data-testid="stSidebar"] .token-counter {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Load glassmorphism CSS
load_glassmorphism_css()

# Proven Multi-Source RAG Architecture Components
class TokenCounter:
    """Comprehensive token counting for all sources and operations"""
    
    def __init__(self):
        try:
            import tiktoken
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            self.available = True
        except ImportError:
            logger.warning("tiktoken not available - install with: pip install tiktoken")
            self.available = False
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if not self.available or not text:
            return 0
        try:
            return len(self.encoding.encode(str(text)))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            return len(str(text).split()) * 1.3  # Rough estimate
    
    def get_comprehensive_breakdown(self, query: str, answer: str, 
                                   vlm_tokens: int = 0, sf_tokens: int = 0, 
                                   text_tokens: int = 0, reranker_tokens: int = 10) -> Dict[str, int]:
        """Get complete token breakdown"""
        query_tokens = self.count_tokens(query)
        response_tokens = self.count_tokens(answer)
        
        return {
            'query_tokens': query_tokens,
            'vlm_analysis_tokens': vlm_tokens,
            'salesforce_api_tokens': sf_tokens,
            'text_rag_tokens': text_tokens,
            'reranker_tokens': reranker_tokens,
            'response_tokens': response_tokens,
            'total_tokens': query_tokens + vlm_tokens + sf_tokens + text_tokens + reranker_tokens + response_tokens
        }

class SimpleRAGOrchestrator:
    """Main query orchestrator implementing re-ranker architecture"""
    
    def __init__(self):
        self.token_counter = TokenCounter()
        self.text_rag = None
        self.colpali_retriever = None
        self.sf_connector = None
        self.reranker = None
        
        logger.info("🎯 SimpleRAGOrchestrator initialized")
    
    def ensure_initialized(self) -> bool:
        """Ensure components are initialized, with user-friendly loading messages"""
        if st.session_state.components_initialized:
            return True
        
        # Check for Docker pre-loaded models
        is_docker_preloaded = self._check_preloaded_models()
        
        if is_docker_preloaded:
            # Docker container with pre-loaded models - instant startup
            with st.spinner("⚡ Activating pre-loaded AI systems..."):
                st.info("🐳 **Docker container detected** - Models pre-loaded for instant startup!")
                success = self.initialize_components()
                
                if success:
                    st.session_state.components_initialized = True
                    st.success("✅ **All AI systems activated instantly!** (Pre-loaded models)")
                    return True
                else:
                    st.error("❌ **System activation failed** - Please restart container")
                    return False
        else:
            # Standard initialization - download and load models
            with st.spinner("🚀 Starting up AI systems for the first time..."):
                st.info("**First-time setup:** Loading AI models (BGE re-ranker, ColPali vision, etc.)")
                
                import torch
                if torch.cuda.is_available():
                    st.info("🔥 **GPU detected** - Loading models with GPU acceleration")
                else:
                    st.info("💻 **CPU mode** - Models loading (may take 30-60 seconds)")
                
                success = self.initialize_components()
                
                if success:
                    st.session_state.components_initialized = True
                    st.success("✅ **All AI systems loaded and ready!**")
                    return True
                else:
                    st.error("❌ **System initialization failed** - Please refresh and try again")
                    return False
    
    def _check_preloaded_models(self) -> bool:
        """Check if models are pre-loaded (Docker environment)"""
        import os
        
        # Check for Docker environment variable
        docker_preloaded = os.getenv('DOCKER_PRELOADED_MODELS', 'false').lower() == 'true'
        
        # Check for model manifest file
        manifest_exists = os.path.exists('/app/models/model_manifest.json')
        
        # Check for HuggingFace cache directory
        hf_cache_exists = os.path.exists('/app/models/huggingface')
        
        logger.info(f"🔍 Pre-loaded model check: Docker flag={docker_preloaded}, Manifest={manifest_exists}, HF cache={hf_cache_exists}")
        
        return docker_preloaded and (manifest_exists or hf_cache_exists)
    
    def initialize_components(self):
        """Initialize ALL components for multi-source search"""
        try:
            logger.info("🔧 Initializing all components for multi-source search")
            
            # Initialize CrossEncoderReRanker (always needed)
            logger.info("📊 Initializing cross-encoder re-ranker...")
            self.reranker = CrossEncoderReRanker(
                model_name='BAAI/bge-reranker-base',
                relevance_threshold=0.3
            )
            if not self.reranker.initialize():
                logger.warning("⚠️ Re-ranker initialization failed - using fallback scoring")
            
            # Initialize Text RAG system (always) with enhanced configuration
            logger.info("📝 Initializing Text RAG system with OpenAI embeddings...")
            text_config = {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'embedding_model': 'openai',  # Updated to use OpenAI embeddings
                'generation_model': 'gpt-3.5-turbo',
                'max_retrieved_chunks': 5,
                'temperature': 0.1
            }
            self.text_rag = RAGSystem(text_config)
            
            # Reinitialize vector database to ensure correct dimensions
            logger.info("🔄 Ensuring vector database has correct dimensions...")
            self.text_rag.reinitialize_vector_database()
            
            # Initialize ColPali retriever (production settings with GPU detection)
            import torch
            if torch.cuda.is_available():
                # Get GPU memory information for dynamic configuration
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"🖼️ Initializing ColPali retriever (GPU: {gpu_name}, {gpu_memory_gb:.1f}GB)...")
                
                # Memory-aware configuration for different GPU sizes
                if gpu_memory_gb <= 6.5:  # 6GB GPU (like GTX 1060)
                    max_pages = 20  # Conservative for 6GB
                    logger.info("🎯 6GB GPU detected - using memory-optimized settings")
                elif gpu_memory_gb <= 8.5:  # 8GB GPU
                    max_pages = 30
                    logger.info("🚀 8GB GPU detected - using balanced settings")
                else:  # 10GB+ GPU
                    max_pages = 50
                    logger.info("⚡ High-memory GPU detected - using maximum settings")
                
                colpali_config = {
                    'model_name': 'vidore/colqwen2-v1.0',
                    'device': 'auto',
                    'max_pages_per_doc': max_pages,
                    'cache_embeddings': True,
                    'cache_dir': 'cache/embeddings'
                }
            else:
                logger.info("🖼️ Initializing ColPali retriever (CPU mode - lightweight testing)...")
                logger.warning("⚠️ CPU-only processing: expect slower performance")
                colpali_config = {
                    'model_name': 'vidore/colqwen2-v1.0',
                    'device': 'cpu',
                    'max_pages_per_doc': 10,  # Reduced for CPU
                    'batch_size': 1,
                    'torch_dtype': 'float32',
                    'cache_embeddings': True,
                    'cache_dir': 'cache/embeddings',
                    'timeout_seconds': 300
                }
            
            try:
                self.colpali_retriever = ColPaliRetriever(colpali_config)
                
                # Check poppler availability during initialization
                poppler_available = self.colpali_retriever.check_poppler_availability()
                
                if poppler_available:
                    if torch.cuda.is_available():
                        logger.info("✅ ColPali retriever initialized (GPU mode)")
                    else:
                        logger.info("✅ ColPali retriever initialized (CPU testing mode)")
                else:
                    logger.warning("⚠️ ColPali initialized but poppler unavailable")
                    logger.warning("Visual document processing will be limited")
                    
            except Exception as e:
                logger.warning(f"⚠️ ColPali initialization failed: {e}")
                logger.warning("Multi-source search will use text + Salesforce only")
                self.colpali_retriever = None
            
            # Initialize Salesforce connector (always available)
            logger.info("🏢 Initializing Salesforce connector...")
            try:
                self.sf_connector = SalesforceConnector()
                connection_status = self.sf_connector.test_connection()
                if connection_status:
                    logger.info("✅ Salesforce connected successfully")
                else:
                    logger.warning("⚠️ Salesforce connection failed - check credentials")
            except Exception as e:
                logger.warning(f"⚠️ Salesforce connector failed to initialize: {e}")
                self.sf_connector = None
            
            logger.info("✅ Component initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            return False
    
    def get_system_capabilities(self) -> Dict[str, Any]:
        """Get current system capabilities for UI display."""
        capabilities = {
            'text_rag': {
                'available': self.text_rag is not None,
                'status': '✅ Available' if self.text_rag else '❌ Unavailable'
            },
            'visual_rag': {
                'available': False,
                'status': '❌ Unavailable',
                'poppler_available': False
            },
            'salesforce': {
                'available': self.sf_connector is not None,
                'status': '✅ Connected' if self.sf_connector else '❌ Disconnected'
            },
            'reranker': {
                'available': self.reranker is not None,
                'status': '✅ Active' if self.reranker else '❌ Inactive'
            }
        }
        
        # Check ColPali/Visual RAG detailed status
        if self.colpali_retriever is not None:
            poppler_available = getattr(self.colpali_retriever, 'poppler_available', False)
            if poppler_available:
                capabilities['visual_rag'] = {
                    'available': True,
                    'status': '✅ Available',
                    'poppler_available': True
                }
            else:
                capabilities['visual_rag'] = {
                    'available': False,
                    'status': '⚠️ Limited (Poppler unavailable)',
                    'poppler_available': False
                }
        
        return capabilities
    
    def reinitialize_with_fresh_embeddings(self) -> bool:
        """
        Reinitialize all embedding systems with fresh state.
        Call this after clearing cache or changing embedding models.
        """
        logger.info("🔄 Reinitializing all embedding systems with fresh state...")
        
        try:
            # Clear any existing vector databases
            if hasattr(self, 'text_rag') and self.text_rag:
                if hasattr(self.text_rag, 'vector_db'):
                    logger.info("🧹 Clearing text RAG vector database...")
                    self.text_rag.vector_db = None
            
            if hasattr(self, 'colpali_retriever') and self.colpali_retriever:
                logger.info("🧹 Clearing ColPali embeddings...")
                if hasattr(self.colpali_retriever, 'document_embeddings'):
                    self.colpali_retriever.document_embeddings = {}
                if hasattr(self.colpali_retriever, 'document_metadata'):
                    self.colpali_retriever.document_metadata = {}
            
            # Reinitialize components
            logger.info("🔧 Reinitializing components...")
            self.initialize_components()
            
            logger.info("✅ Reinitialization complete - ready for fresh document processing")
            return True
            
        except Exception as e:
            logger.error(f"❌ Reinitialization failed: {e}")
            return False
    
    def _enhance_query_for_search(self, query: str) -> str:
        """
        Enhance query for better vector search across all sources.
        Adds relevant synonyms and context to improve embedding matching.
        """
        query_lower = query.lower()
        enhanced_terms = []
        
        # Person/role queries - expand with synonyms
        if any(term in query_lower for term in ['who is', 'who are', 'person', 'people']):
            if 'recruitment' in query_lower:
                enhanced_terms.extend(['recruitment specialist', 'hiring manager', 'HR recruiter', 'human resources'])
            elif 'senior dev' in query_lower or 'senior developer' in query_lower:
                enhanced_terms.extend(['senior developer', 'lead developer', 'principal engineer', 'senior engineer'])
            elif 'manager' in query_lower:
                enhanced_terms.extend(['manager', 'supervisor', 'team lead', 'director'])
        
        # Performance/chart queries - add technical terms
        if any(term in query_lower for term in ['time', 'performance', 'speed', 'latency']):
            if 'retrieval' in query_lower:
                enhanced_terms.extend(['retrieval time', 'query performance', 'search latency', 'response time'])
            elif 'colpali' in query_lower:
                enhanced_terms.extend(['colpali', 'visual embedding', 'document processing', 'pipeline performance'])
        
        # Chart/visual queries - add visual terms
        if any(term in query_lower for term in ['chart', 'graph', 'diagram', 'figure']):
            enhanced_terms.extend(['visualization', 'chart data', 'graph analysis', 'performance metrics'])
        
        # Build enhanced query
        if enhanced_terms:
            # Remove duplicates and join with original query
            unique_terms = list(set(enhanced_terms))
            enhanced_query = f"{query} {' '.join(unique_terms)}"
            return enhanced_query
        
        return query
    
    def _clean_response_for_user(self, response: str, source_type: str) -> str:
        """
        Clean and format responses from all sources for better user experience.
        Removes technical metadata and converts raw data to natural language.
        """
        if not response or not response.strip():
            return response
        
        import re
        
        # Remove CSV/Excel technical headers
        cleaned = re.sub(r'### CSV: [^-]+ - Entry \d+\n\n', '', response)
        cleaned = re.sub(r'### Sheet: [^-]+ - Entry \d+\n\n', '', cleaned)
        
        # Remove "Key fields:" technical metadata
        cleaned = re.sub(r'\n\nKey fields: [^\n]+', '', cleaned)
        
        # Remove page markers from PDFs (but keep the content)
        cleaned = re.sub(r'--- Page \d+ ---\n?', '', cleaned)
        
        # Clean ColPali prefixes but keep content type context
        if source_type == 'colpali':
            # Keep helpful prefixes like "Performance chart analysis:" but remove technical stuff
            cleaned = re.sub(r'\(from [^)]+\)', '', cleaned)
        
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        cleaned = cleaned.strip()
        
        # ENHANCED: Convert raw structured data patterns to natural language
        cleaned = self._convert_raw_data_to_natural_language(cleaned, source_type)
        
        # For person queries, make response more conversational
        if any(indicator in response.lower() for indicator in ['works as', 'specialist', 'manager', 'developer']):
            # This is a person query - make it more natural
            if not cleaned.startswith('The person') and not cleaned.startswith('Based on'):
                # Add context if it's just a name and role
                if len(cleaned.split('.')) == 1 and ' works as ' in cleaned:
                    cleaned = f"Based on the data, {cleaned.lower()}"
        
        return cleaned
    
    def _convert_raw_data_to_natural_language(self, content: str, source_type: str) -> str:
        """Convert raw structured data patterns into natural language answers."""
        import re
        
        # Pattern 1: Technical specification format (like the user's example)
        # "Policy Validation Engine: Given that a traveler is making a booking When they select travel options..."
        technical_spec_pattern = r'([A-Z][^:]+):\s*Given that\s+([^.]+)\s+When\s+([^.]+)\s+Then\s+([^.]+)'
        match = re.search(technical_spec_pattern, content)
        if match:
            system_name, given_condition, when_condition, then_result = match.groups()
            natural_response = f"The {system_name.lower()} is designed to handle the following scenario: when {given_condition.lower()}, and specifically when {when_condition.lower()}, the system will {then_result.lower()}."
            
            # If there's more content after the technical spec, include it
            remaining_content = content[match.end():].strip()
            if remaining_content:
                # Clean up remaining content and add it
                remaining_content = re.sub(r'Given that.*?Then\s+[^.]+\.?\s*', '', remaining_content)
                remaining_content = remaining_content.strip()
                if remaining_content:
                    natural_response += f" Additionally, {remaining_content}"
            
            return natural_response
        
        # Pattern 2: Raw CSV/table data format
        # "Row 1: Name: John Smith | Department: Engineering | Role: Manager"
        row_data_pattern = r'Row \d+:\s*(.+?)(?=Row \d+:|$)'
        if 'Row ' in content and '|' in content:
            # Extract meaningful information from table rows
            rows = re.findall(row_data_pattern, content, re.DOTALL)
            if rows:
                summary_parts = []
                for row in rows[:3]:  # Limit to first 3 rows for brevity
                    # Parse key-value pairs
                    pairs = [pair.strip() for pair in row.split('|') if ':' in pair]
                    if pairs:
                        row_info = {}
                        for pair in pairs:
                            if ':' in pair:
                                key, value = pair.split(':', 1)
                                row_info[key.strip().lower()] = value.strip()
                        
                        # Create natural sentence from the data
                        if 'name' in row_info:
                            sentence = f"{row_info['name']}"
                            if 'role' in row_info or 'position' in row_info:
                                role = row_info.get('role', row_info.get('position', ''))
                                sentence += f" works as {role}"
                            if 'department' in row_info:
                                sentence += f" in the {row_info['department']} department"
                            summary_parts.append(sentence)
                
                if summary_parts:
                    return f"Based on the data, here's what I found: {'. '.join(summary_parts)}."
        
        # Pattern 3: Business rule format
        # "Rule: [condition] Result: [outcome]"
        rule_pattern = r'Rule:\s*([^.]+)\s+Result:\s*([^.]+)'
        match = re.search(rule_pattern, content)
        if match:
            condition, outcome = match.groups()
            return f"According to the business rules, {condition.lower().strip()}, and as a result, {outcome.lower().strip()}."
        
        # Pattern 4: Process step format
        if 'Step 1:' in content or 'Phase 1:' in content:
            # Convert step-by-step format to flowing description
            steps = re.findall(r'(?:Step|Phase)\s+\d+:\s*([^.]+)', content)
            if steps:
                if len(steps) == 1:
                    return f"The process involves {steps[0].lower()}."
                else:
                    return f"The process consists of several steps: {', '.join(step.lower() for step in steps[:-1])}, and finally {steps[-1].lower()}."
        
        # Pattern 5: Acceptance criteria format (like the user's example)
        if 'acceptance criteria' in content.lower():
            # Clean up technical jargon and make it more readable
            cleaned = re.sub(r'Given that\s+', 'When ', content)
            cleaned = re.sub(r'When they\s+', 'and when they ', cleaned)
            cleaned = re.sub(r'Then the\s+', 'the ', cleaned)
            
            # If it's still very technical, add a natural introduction
            if not cleaned.lower().startswith(('the', 'this', 'according', 'based on')):
                cleaned = f"The acceptance criteria specify that {cleaned.lower()}"
            
            return cleaned
        
        # Pattern 6: Configuration or policy data
        config_patterns = ['Policy:', 'Configuration:', 'Setting:', 'Rule:']
        for pattern in config_patterns:
            if pattern in content:
                parts = content.split(pattern)
                if len(parts) > 1:
                    policy_content = parts[1].strip()
                    return f"According to the {pattern.lower().rstrip(':')}, {policy_content}"
        
        # If no specific patterns match, try to make the content more natural
        # Remove technical markers and improve flow
        if content.count(':') > 2 and content.count('|') > 1:
            # This looks like structured data - try to make it more readable
            content = re.sub(r'\s*\|\s*', ', ', content)  # Replace | with commas
            content = re.sub(r'([A-Z][^:]+):\s*', r'\1 is ', content)  # Convert "Field:" to "Field is"
        
        return content
    
    def query_all_sources(self, user_query: str) -> Dict[str, Any]:
        """Query ALL sources and use re-ranker to select best response"""
        
        # Enhance query for better vector search across all sources
        enhanced_query = self._enhance_query_for_search(user_query)
        
        logger.info(f"🔍 Querying ALL sources for: '{user_query[:50]}...'")
        if enhanced_query != user_query:
            logger.debug(f"📝 Enhanced query: '{enhanced_query[:50]}...'")
        
        candidates = []
        
        # 1. Query Text RAG system (if available)
        if self.text_rag:
            text_start_time = time.time()
            try:
                logger.info("📝 Querying text RAG system...")
                text_results = self.text_rag.query(enhanced_query)
                
                if text_results.get('success'):
                    candidates.append({
                        'success': True,
                        'answer': text_results['answer'],
                        'source_type': 'text',
                        'score': text_results.get('confidence', 0.5),
                        'sources': text_results.get('sources', []),
                        'metadata': {'chunks_used': text_results.get('chunks_used', 0)},
                        'token_info': {
                            'query_time': time.time() - text_start_time,
                            'text_tokens': text_results.get('tokens_used', 0)
                        }
                    })
                    logger.info(f"✅ Text RAG returned result (confidence: {text_results.get('confidence', 0):.3f})")
                else:
                    logger.info("ℹ️ Text RAG found no relevant content")
                    
            except Exception as e:
                logger.error(f"❌ Text RAG search failed: {e}")
        
        # 2. Query ColPali retriever (if available)
        if self.colpali_retriever:
            colpali_start_time = time.time()
            try:
                logger.info("🖼️ Querying ColPali retriever...")
                colpali_results, metrics = self.colpali_retriever.retrieve(enhanced_query, top_k=5)
                
                if colpali_results:
                    best_result = colpali_results[0]
                    candidates.append({
                        'success': True,
                        'answer': best_result.content,
                        'source_type': 'colpali',
                        'score': best_result.score,
                        'sources': [{'filename': best_result.metadata.get('filename', 'Unknown'),
                                   'page': best_result.metadata.get('page', 1),
                                   'score': best_result.score}],
                        'metadata': best_result.metadata,
                        'token_info': {'query_time': metrics.query_time, 'vlm_tokens': best_result.metadata.get('vlm_tokens', 0)}
                    })
                    logger.info(f"✅ ColPali returned {len(colpali_results)} results (best score: {best_result.score:.3f})")
                    logger.info(f"🎯 ColPali content preview: {best_result.content[:100]}...")
                else:
                    logger.info("ℹ️ ColPali found no relevant visual content")
                    
            except Exception as e:
                logger.error(f"❌ ColPali search failed: {e}")
        
        # 3. Query Salesforce (always attempt if available)
        if self.sf_connector:
            sf_start_time = time.time()
            try:
                logger.info("🏢 Querying Salesforce knowledge base...")
                sf_results = self.sf_connector.search_knowledge_with_intent(enhanced_query, limit=3)
                
                if sf_results:
                    best_sf = max(sf_results, key=lambda x: x.get('relevance_score', 0))
                    
                    # Use enhanced Salesforce response generation with LLM synthesis
                    enhanced_answer, sf_tokens_used = self.sf_connector.generate_enhanced_sf_response(user_query, [best_sf])
                    
                    candidates.append({
                        'success': True,
                        'answer': enhanced_answer,
                        'source_type': 'salesforce',
                        'score': best_sf.get('relevance_score', 0.5),
                        'sources': [{'title': best_sf.get('title', 'Unknown KB'),
                                   'score': best_sf.get('relevance_score', 0.5)}],
                        'metadata': {'article_id': best_sf.get('id', ''), 'source_url': best_sf.get('source_url', '')},
                        'token_info': {'query_time': time.time() - sf_start_time, 'sf_tokens': sf_tokens_used}
                    })
                    logger.info(f"✅ Salesforce returned {len(sf_results)} articles (best score: {best_sf.get('relevance_score', 0):.3f})")
                else:
                    logger.info("ℹ️ Salesforce found no relevant articles")
                    
            except Exception as e:
                logger.error(f"❌ Salesforce search failed: {e}")
        
        # 4. Re-rank all candidates
        if not candidates:
            return {
                'success': False,
                'error': 'No sources returned valid results',
                'attempted_sources': ['text', 'colpali', 'salesforce']
            }
        
        logger.info(f"🎯 Re-ranking {len(candidates)} candidates...")
        
        if self.reranker and self.reranker.is_initialized:
            ranking_result = self.reranker.rank_all_sources(user_query, candidates)
            
            if ranking_result['success']:
                selected = ranking_result['selected_source']
                
                token_breakdown = self.token_counter.get_comprehensive_breakdown(
                    query=user_query,
                    answer=selected['answer'],
                    vlm_tokens=selected.get('token_info', {}).get('vlm_tokens', 0),
                    sf_tokens=selected.get('token_info', {}).get('sf_tokens', 0),
                    text_tokens=selected.get('token_info', {}).get('text_tokens', 0),
                    reranker_tokens=10
                )
                
                # Clean response for better user experience
                cleaned_answer = self._clean_response_for_user(selected['answer'], selected['source_type'])
                
                return {
                    'success': True,
                    'answer': cleaned_answer,
                    'selected_source': selected['source_type'],
                    'rerank_score': selected['rerank_score'],
                    'sources': selected['sources'],
                    'confidence': selected['rerank_score'],  # For compatibility
                    'token_breakdown': token_breakdown,
                    'rejected_sources': [
                        {
                            'type': r['source_type'],
                            'score': r['rerank_score'],
                            'reason': r['reason']
                        }
                        for r in ranking_result['rejected_sources']
                    ],
                    'reasoning': ranking_result['reasoning'],
                    'model_used': ranking_result.get('model_used', 'BGE re-ranker')
                }
        
        # Fallback: simple score-based selection
        logger.warning("⚠️ Using fallback scoring (re-ranker not available)")
        best_candidate = max(candidates, key=lambda x: x['score'])
        
        # Clean response for better user experience
        cleaned_answer = self._clean_response_for_user(best_candidate['answer'], best_candidate['source_type'])
        
        token_breakdown = self.token_counter.get_comprehensive_breakdown(
            query=user_query,
            answer=cleaned_answer,
            vlm_tokens=best_candidate.get('token_info', {}).get('vlm_tokens', 0),
            sf_tokens=best_candidate.get('token_info', {}).get('sf_tokens', 0),
            text_tokens=best_candidate.get('token_info', {}).get('text_tokens', 0)
        )
        
        return {
            'success': True,
            'answer': cleaned_answer,
            'selected_source': best_candidate['source_type'],
            'rerank_score': best_candidate['score'],
            'sources': best_candidate['sources'],
            'confidence': best_candidate['score'],  # For compatibility
            'token_breakdown': token_breakdown,
            'reasoning': f"Fallback selection: {best_candidate['source_type']} had highest score: {best_candidate['score']:.3f}"
        }
    
    
    def add_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Add documents to ALL available systems (text + ColPali)"""
        logger.info(f"📄 Processing {len(file_paths)} documents for ALL systems")
        
        start_time = time.time()
        results = {
            'successful': [],
            'failed': [],
            'text_processed': 0,
            'colpali_processed': 0,
            'processing_time': 0
        }
        
        for file_path in file_paths:
            file_result = {
                'path': file_path,
                'filename': os.path.basename(file_path),
                'text_success': False,
                'colpali_success': False,
                'text_chunks': 0,
                'colpali_pages': 0,
                'errors': []
            }
            
            # Process with Text RAG system
            if self.text_rag:
                try:
                    logger.info(f"📝 Processing {file_result['filename']} for text RAG...")
                    text_result = self.text_rag.add_documents([file_path])
                    
                    if text_result['successful']:
                        file_result['text_success'] = True
                        file_result['text_chunks'] = text_result['successful'][0].get('chunks', 0)
                        results['text_processed'] += 1
                        logger.info(f"✅ Text processing: {file_result['text_chunks']} chunks")
                    else:
                        file_result['errors'].append(f"Text: {text_result['failed'][0].get('error', 'Unknown error')}")
                        
                except Exception as e:
                    file_result['errors'].append(f"Text processing failed: {str(e)}")
                    logger.error(f"❌ Text processing failed for {file_result['filename']}: {e}")
            
            # Process with ColPali retriever  
            if self.colpali_retriever:
                try:
                    logger.info(f"🖼️ Processing {file_result['filename']} for ColPali...")
                    colpali_result = self.colpali_retriever.add_documents([file_path])
                    
                    if colpali_result['successful']:
                        file_result['colpali_success'] = True
                        file_result['colpali_pages'] = colpali_result['successful'][0].get('pages', 0)
                        results['colpali_processed'] += 1
                        logger.info(f"✅ ColPali processing: {file_result['colpali_pages']} pages")
                    else:
                        file_result['errors'].append(f"ColPali: {colpali_result['failed'][0].get('error', 'Unknown error')}")
                        
                except Exception as e:
                    file_result['errors'].append(f"ColPali processing failed: {str(e)}")
                    logger.error(f"❌ ColPali processing failed for {file_result['filename']}: {e}")
            
            # Determine overall success
            if file_result['text_success'] or file_result['colpali_success']:
                results['successful'].append(file_result)
            else:
                results['failed'].append({
                    'path': file_path,
                    'error': '; '.join(file_result['errors']) if file_result['errors'] else 'No processing systems available'
                })
        
        results['processing_time'] = time.time() - start_time
        
        logger.info(f"📊 Document processing complete: {len(results['successful'])}/{len(file_paths)} successful")
        logger.info(f"   Text: {results['text_processed']} docs, ColPali: {results['colpali_processed']} docs")
        
        return results

# Initialize session state with our proven architecture
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = SimpleRAGOrchestrator()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'components_initialized' not in st.session_state:
    st.session_state.components_initialized = False
if 'query_input_key' not in st.session_state:
    st.session_state.query_input_key = 0
if 'show_clear_confirmation' not in st.session_state:
    st.session_state.show_clear_confirmation = False

# Helper functions for UI formatting

def get_confidence_color(score):
    """Get color class based on confidence score."""
    if score >= 0.8:
        return "confidence-high"
    elif score >= 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"

def format_sources(sources):
    """Format sources for display."""
    if not sources:
        return "No sources available"
    
    formatted = []
    for i, source in enumerate(sources, 1):
        # Handle different source structures
        if isinstance(source, dict):
            filename = source.get('filename', source.get('source_number', f'Source {i}'))
            score = source.get('relevance_score', source.get('score', 0.0))
            content = source.get('chunk_text', source.get('content', 'No content available'))
        else:
            filename = f'Source {i}'
            score = 0.0
            content = str(source)
            
        confidence_color = get_confidence_color(score)
        formatted.append(f"""
        <div class="source-card">
            <h4>📄 Source {i}: {filename}</h4>
            <p class="{confidence_color}">Relevance: {score:.3f}</p>
            <p>{content}</p>
        </div>
        """)
    
    return "".join(formatted)

def extract_question_specific_content(user_query, sf_results):
    """IMPROVED: Extract content from the most relevant Salesforce result with better article selection."""
    import re
    import html
    
    if not sf_results:
        return "No information found in Salesforce knowledge articles."
    
    # IMPROVED: Better article selection based on query context
    query_lower = user_query.lower()
    
    # Define service type keywords for better matching
    service_keywords = {
        'hotel': ['hotel', 'accommodation', 'room', 'stay'],
        'air': ['air', 'flight', 'airline', 'aviation'],
        'car': ['car', 'rental', 'vehicle', 'auto']
    }
    
    # Define action type keywords
    action_keywords = {
        'cancel': ['cancel', 'cancellation', 'refund', 'void'],
        'modify': ['modify', 'modification', 'change', 'update'],
        'book': ['book', 'booking', 'new', 'create'],
        'handle': ['handle', 'handling', 'manage', 'process']
    }
    
    # Determine query service type and action type
    query_service_type = None
    query_action_type = None
    
    for service_type, keywords in service_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            query_service_type = service_type
            break
    
    for action_type, keywords in action_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            query_action_type = action_type
            break
    
    # Score articles based on how well they match the query context
    scored_articles = []
    for result in sf_results:
        context_score = result['relevance_score']
        title_lower = result['title'].lower()
        
        # Bonus for matching service type
        if query_service_type and query_service_type in title_lower:
            context_score += 0.3
        
        # Bonus for matching action type (look for action in title)
        if query_action_type:
            action_words = action_keywords[query_action_type]
            if any(word in title_lower for word in action_words):
                context_score += 0.4
        
        scored_articles.append((result, context_score))
    
    # Sort by context score and select the best match
    scored_articles.sort(key=lambda x: x[1], reverse=True)
    best_match = scored_articles[0][0]
    
    # If the best match has very low relevance, return a helpful message
    if best_match['relevance_score'] < 0.15:
        return f"No highly relevant information found for '{user_query}'. You may need to check additional resources or contact your supervisor."
    
    # Clean HTML and decode HTML entities
    clean_content = re.sub(r'<[^>]+>', ' ', best_match['content'])
    clean_content = html.unescape(clean_content)  # Decode HTML entities like &amp;
    clean_content = re.sub(r'\s+', ' ', clean_content).strip()
    
    # If content is very short, return it as-is
    if len(clean_content) < 200:
        return f"Based on '{best_match['title']}': {clean_content}"
    
    # For longer content, extract the most relevant sections
    sentences = [s.strip() for s in clean_content.split('.') if len(s.strip()) > 15]
    
    # IMPROVED: Better query-aware content extraction
    query_words = [word.lower() for word in user_query.split() if len(word) > 2]
    
    # Define action-specific keywords to look for
    action_keywords = {
        'cancel': ['cancel', 'cancellation', 'refund', 'void', 'terminate'],
        'modify': ['modify', 'change', 'update', 'amend', 'alter'],
        'book': ['book', 'reserve', 'create', 'make', 'new'],
        'handle': ['handle', 'manage', 'process', 'deal with', 'address']
    }
    
    # Determine the primary action from the query
    primary_action = None
    for action, keywords in action_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            primary_action = action
            break
    
    scored_sentences = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = 0
        
        # Score based on query words
        score += sum(1 for word in query_words if word in sentence_lower)
        
        # Higher score for sentences containing the primary action
        if primary_action:
            action_words = action_keywords[primary_action]
            if any(word in sentence_lower for word in action_words):
                score += 3  # High bonus for action-related sentences
        
        # Bonus for procedural language
        if any(proc_word in sentence_lower for proc_word in ['step', 'process', 'follow', 'ensure', 'must', 'should', 'contact', 'check']):
            score += 1
        
        # Penalty for introductory fluff
        if any(fluff in sentence_lower for fluff in ['purpose', 'this article', 'overview', 'introduction', 'general information']):
            score -= 2
        
        # Penalty for sentences that seem to be about different actions
        if primary_action:
            other_actions = [act for act in action_keywords.keys() if act != primary_action]
            for other_action in other_actions:
                other_words = action_keywords[other_action]
                if any(word in sentence_lower for word in other_words):
                    score -= 1  # Penalty for sentences about different actions
        
        if score > 0:
            scored_sentences.append((sentence, score))
    
    # Sort by relevance and take top sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    if scored_sentences:
        # Return the most relevant content
        answer = f"Based on '{best_match['title']}' (relevance: {best_match['relevance_score']:.2f}):\n\n"
        
        # Take up to 4 most relevant sentences
        for i, (sentence, score) in enumerate(scored_sentences[:4], 1):
            sentence = sentence.strip()
            if not sentence.endswith('.'):
                sentence += '.'
            answer += f"{i}. {sentence}\n"
        
        return answer
    else:
        # Fallback: return first few sentences
        answer = f"Based on '{best_match['title']}' (relevance: {best_match['relevance_score']:.2f}):\n\n"
        for i, sentence in enumerate(sentences[:3], 1):
            sentence = sentence.strip()
            if not sentence.endswith('.'):
                sentence += '.'
            answer += f"{i}. {sentence}\n"
        
        return answer

def clear_all_data():
    """Properly clear all data including documents from vector database."""
    try:
        # First, clear documents from the RAG system if it exists
        if st.session_state.rag_system is not None:
            try:
                success = st.session_state.rag_system.clear_documents()
                if success:
                    print("✅ Documents cleared from vector database")
                else:
                    print("⚠️ Warning: Could not clear vector database")
            except Exception as e:
                print(f"⚠️ Error clearing vector database: {str(e)}")
        
        # Clear enhanced ColPali system if it exists
        if st.session_state.get('enhanced_colpali') is not None:
            try:
                success = st.session_state.enhanced_colpali.clear_documents()
                if success:
                    print("✅ Enhanced ColPali documents cleared")
                else:
                    print("⚠️ Warning: Could not clear Enhanced ColPali")
            except Exception as e:
                print(f"⚠️ Error clearing Enhanced ColPali: {str(e)}")
        
        # Clear ALL session state variables thoroughly
        keys_to_keep = ['initialize_rag_system', 'initialize_enhanced_colpali']  # Keep the cache functions
        keys_to_remove = [key for key in st.session_state.keys() if key not in keys_to_keep]
        
        for key in keys_to_remove:
            del st.session_state[key]
        
        # Reinitialize ALL essential session state
        st.session_state.chat_history = []
        st.session_state.processed_files = []
        st.session_state.rag_system = None
        st.session_state.multimodal_rag = None
        st.session_state.enhanced_colpali = None
        st.session_state.enhanced_mode_enabled = False
        st.session_state.retrieval_mode = 'text'
        st.session_state.query_input_key = 0
        st.session_state.show_clear_confirmation = False
        
        # Session state cleared - cached functions no longer exist
        print("✅ Session state cleared")
        
        print("🗑️ All data cleared successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error clearing data: {str(e)}")
        return False


def main():
    # Glassmorphic Header
    st.markdown("""
    <div class="glass-header">
        <h1>🤖 Smart Document Assistant</h1>
        <p>Multi-source search with intelligent re-ranking - Text + Visual + Salesforce</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System Architecture Info
    with st.expander("🎯 Multi-Source Architecture", expanded=False):
        st.markdown("""
        **🔄 Current Architecture: Multi-Source with BGE Re-ranker**
        
        **How it works:**
        - 📝 **Text RAG**: Traditional document chunking and embedding search
        - 🖼️ **ColPali Visual**: Vision-language model for visual document understanding  
        - 🏢 **Salesforce**: Knowledge base search for organizational content
        - 🎯 **BGE Re-ranker**: Cross-encoder model selects single best source
        
        **Query Flow:**
        1. Search ALL sources simultaneously (Text + ColPali + Salesforce)
        2. BGE cross-encoder re-ranks all candidates semantically
        3. Return single best answer (no mixing of sources)
        4. Show which source was selected and why
        
        **Benefits:**
        - **No source mixing**: Single coherent answer from best source
        - **Semantic understanding**: BGE model understands query intent better than rules
        - **Visual comprehension**: ColPali handles tables, charts, complex layouts
        - **Transparent selection**: See why each source was chosen or rejected
        """)

    # Sidebar - System Configuration  
    with st.sidebar:
        st.header("🎛️ System Config")
        
        
        # Feature Status with Glassmorphic Cards
        with stylable_container(
            key="feature_status_panel",
            css_styles="""
            {
                background: rgba(255, 255, 255, 0.06);
                backdrop-filter: blur(20px) saturate(180%);
                -webkit-backdrop-filter: blur(20px) saturate(180%);
                border: 1px solid rgba(100, 181, 246, 0.2);
                border-radius: 18px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                box-shadow: 
                    0 12px 40px rgba(100, 181, 246, 0.1),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
            }
            """
        ):
            st.markdown('<h4 style="color: #64b5f6; font-size: 1rem; margin-bottom: 1rem; text-shadow: 0 1px 3px rgba(0,0,0,0.5);">📋 Feature Status</h4>', unsafe_allow_html=True)
            
            if st.session_state.components_initialized:
                # Get system capabilities
                capabilities = st.session_state.orchestrator.get_system_capabilities()
                
                # Create status cards mapping
                status_cards = []
                feature_labels = {
                    'text_rag': ('Text RAG', '📝'),
                    'visual_rag': ('ColPali', '🖼️'), 
                    'salesforce': ('Salesforce', '🏢'),
                    'reranker': ('Re-ranker', '🎯')
                }
                
                for feature_name, feature_info in capabilities.items():
                    if feature_name in feature_labels:
                        label, icon = feature_labels[feature_name]
                        status = feature_info['status']
                        
                        if '✅' in status:
                            # Ready status
                            value = "✅ Ready"
                            color = "#81c784"
                            bg_color = "rgba(129, 199, 132, 0.1)"
                            border_color = "rgba(129, 199, 132, 0.2)"
                        elif '⚠️' in status:
                            # Warning status  
                            value = "⚠️ Config"
                            color = "#ffb74d"
                            bg_color = "rgba(255, 183, 77, 0.1)"
                            border_color = "rgba(255, 183, 77, 0.2)"
                        else:
                            # Error status
                            value = "❌ Error"
                            color = "#ef5350"
                            bg_color = "rgba(239, 83, 80, 0.1)"
                            border_color = "rgba(239, 83, 80, 0.2)"
                        
                        status_cards.append((label, value, color, bg_color, border_color))
                
                # Display glassmorphic status cards with perfect centering
                for label, value, color, bg_color, border_color in status_cards:
                    status_card_html = f"""
                    <div style="
                        background: {bg_color};
                        backdrop-filter: blur(15px);
                        border: 1px solid {border_color};
                        border-radius: 12px;
                        margin-bottom: 0.5rem;
                        text-align: center;
                        height: 60px;
                        position: relative;
                        transition: all 0.3s ease;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                        align-items: center;
                    ">
                        <div style="
                            color: rgba(255,255,255,0.7); 
                            font-size: 0.75rem; 
                            text-transform: uppercase; 
                            line-height: 1;
                            margin-bottom: 4px;
                        ">{label}</div>
                        <div style="
                            color: {color}; 
                            font-weight: 600; 
                            font-size: 1rem;
                            line-height: 1;
                        ">{value}</div>
                    </div>
                    """
                    st.markdown(status_card_html, unsafe_allow_html=True)
            else:
                # Loading state card
                loading_card_html = """
                <div style="
                    background: rgba(100, 181, 246, 0.1);
                    backdrop-filter: blur(15px);
                    border: 1px solid rgba(100, 181, 246, 0.2);
                    border-radius: 12px;
                    margin-bottom: 0.5rem;
                    text-align: center;
                    height: 60px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                ">
                    <div style="
                        color: rgba(255,255,255,0.7); 
                        font-size: 0.75rem; 
                        text-transform: uppercase; 
                        line-height: 1;
                        margin-bottom: 4px;
                    ">Systems</div>
                    <div style="
                        color: #64b5f6; 
                        font-weight: 600; 
                        font-size: 1rem;
                        line-height: 1;
                    ">🔄 Loading</div>
                </div>
                """
                st.markdown(loading_card_html, unsafe_allow_html=True)
        
        st.divider()
        
        # System Status Glass Panel
        with stylable_container(
            key="system_status_panel",
            css_styles="""
            {
                background: rgba(255, 255, 255, 0.04);
                backdrop-filter: blur(22px) saturate(170%);
                -webkit-backdrop-filter: blur(22px) saturate(170%);
                border: 1px solid rgba(255, 183, 77, 0.15);
                border-radius: 16px;
                padding: 1.5rem 1rem;
                margin-bottom: 1.5rem;
                box-shadow: 
                    0 10px 35px rgba(255, 183, 77, 0.1),
                    inset 0 1px 0 rgba(255, 183, 77, 0.1);
            }
            """
        ):
            st.markdown('<h4 style="color: #ffb74d; font-size: 1rem; margin-bottom: 1rem; text-shadow: 0 1px 3px rgba(0,0,0,0.5);">📊 System Status</h4>', unsafe_allow_html=True)
            
            if st.session_state.components_initialized:
                st.markdown("""
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem; padding: 0.5rem 0;">
                    <span style="color: rgba(255, 255, 255, 0.8); font-size: 0.85rem;">Mode:</span>
                    <span style="color: #64b5f6; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">GPU</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem; padding: 0.5rem 0;">
                    <span style="color: rgba(255, 255, 255, 0.8); font-size: 0.85rem;">Sources:</span>
                    <span style="color: #64b5f6; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">3 Active</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.5rem 0;">
                    <span style="color: rgba(255, 255, 255, 0.8); font-size: 0.85rem;">Status:</span>
                    <span style="color: #81c784; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">Ready</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem; padding: 0.5rem 0;">
                    <span style="color: rgba(255, 255, 255, 0.8); font-size: 0.85rem;">Systems:</span>
                    <span style="color: #64b5f6; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">Loading</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.5rem 0;">
                    <span style="color: rgba(255, 255, 255, 0.8); font-size: 0.85rem;">Status:</span>
                    <span style="color: #ffb74d; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">Initializing</span>
                </div>
                """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Query Interface Glass Panel
        with stylable_container(
            key="query_interface_panel",
            css_styles="""
            {
                background: rgba(255, 255, 255, 0.06);
                backdrop-filter: blur(20px) saturate(180%);
                -webkit-backdrop-filter: blur(20px) saturate(180%);
                border: 1px solid rgba(100, 181, 246, 0.2);
                border-radius: 18px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                box-shadow: 
                    0 12px 40px rgba(100, 181, 246, 0.1),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
            }
            """
        ):
            st.markdown('<h3 style="color: #64b5f6; margin-bottom: 1.5rem; font-size: 1.2rem; text-align: center; text-shadow: 0 2px 8px rgba(0,0,0,0.3);">💬 Query Interface</h3>', unsafe_allow_html=True)
            
            # Query form
            with st.form(f"query_form_{st.session_state.query_input_key}"):
                user_query = st.text_input(
                    "**Ask a question:**",
                    placeholder="e.g., What is the cancellation policy?",
                    help="This will search all sources and select the most relevant one",
                    key=f"query_input_{st.session_state.query_input_key}"
                )
                
                submitted = st.form_submit_button("🔍 Search All Sources", type="primary")
        
        # Process query
        if submitted and user_query.strip():
            # Auto-initialize systems if needed
            if not st.session_state.orchestrator.ensure_initialized():
                st.stop()  # Stop execution if initialization failed
            
            with st.spinner("🤔 Searching all sources and re-ranking..."):
                start_time = time.time()
                
                result = st.session_state.orchestrator.query_all_sources(user_query)
                
                processing_time = time.time() - start_time
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'query': user_query,
                    'result': result,
                    'timestamp': datetime.now(),
                    'processing_time': processing_time
                })
                
                # Clear the input field by incrementing the key
                st.session_state.query_input_key += 1
                
                # Safe rerun only for query (avoid during file uploads)
                if not st.session_state.get('file_upload_in_progress', False):
                    st.rerun()
        
        # Recent Results Glass Panel
        if st.session_state.chat_history:
            with stylable_container(
                key="results_header_panel",
                css_styles="""
                {
                    background: rgba(255, 255, 255, 0.04);
                    backdrop-filter: blur(18px) saturate(160%);
                    -webkit-backdrop-filter: blur(18px) saturate(160%);
                    border: 1px solid rgba(100, 181, 246, 0.15);
                    border-radius: 16px;
                    padding: 1rem 1.5rem;
                    margin-bottom: 1.5rem;
                    box-shadow: 
                        0 8px 30px rgba(100, 181, 246, 0.08),
                        inset 0 1px 0 rgba(255, 255, 255, 0.08);
                }
                """
            ):
                st.markdown('<h3 style="color: #64b5f6; margin: 0; font-size: 1.2rem; text-align: center; text-shadow: 0 2px 8px rgba(0,0,0,0.3);">📝 Recent Results</h3>', unsafe_allow_html=True)
            
            # Show last 3 results as glassmorphic chat bubbles
            for i, chat in enumerate(reversed(st.session_state.chat_history[-3:])):
                # Create individual glassmorphic chat bubble
                with stylable_container(
                    key=f"chat_bubble_{i}",
                    css_styles="""
                    {
                        background: rgba(255, 255, 255, 0.05);
                        backdrop-filter: blur(15px) saturate(150%);
                        -webkit-backdrop-filter: blur(15px) saturate(150%);
                        border: 1px solid rgba(255, 255, 255, 0.12);
                        border-radius: 16px;
                        padding: 1.5rem;
                        margin-bottom: 1rem;
                        box-shadow: 
                            0 6px 25px rgba(0, 0, 0, 0.1),
                            inset 0 1px 0 rgba(255, 255, 255, 0.08);
                        transition: all 0.3s ease;
                    }
                    """
                ):
                    # Query
                    st.markdown(f'<div style="color: #81c784; font-weight: 500; font-size: 0.95rem; margin-bottom: 0.75rem; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">🔍 {chat["query"]}</div>', unsafe_allow_html=True)
                    
                    result = chat['result']
                    if result['success']:
                        # Source selection info
                        st.markdown(f"""
                        <div class="source-selected">
                            <strong>📍 Selected Source:</strong> {result['selected_source'].upper()} 
                            (Re-rank score: {result['rerank_score']:.3f})
                            <br><small>{result['reasoning']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Answer
                        st.markdown(f'<div style="color: rgba(255, 255, 255, 0.9); font-size: 0.9rem; line-height: 1.6; margin-bottom: 1rem;">{result["answer"]}</div>', unsafe_allow_html=True)
                        
                        # Perfect Token Counter with line-height centering
                        if 'token_breakdown' in result:
                            tokens = result['token_breakdown']
                            
                            with stylable_container(
                                key=f"token_counter_{i}",
                                css_styles="""
                                {
                                    background: rgba(255, 255, 255, 0.08);
                                    backdrop-filter: blur(12px);
                                    -webkit-backdrop-filter: blur(12px);
                                    border: 1px solid rgba(255, 255, 255, 0.15);
                                    border-radius: 12px;
                                    margin: 1rem 0;
                                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                                    overflow: hidden;
                                }
                                """
                            ):
                                # Perfect Token Counter - Line-height based centering
                                perfect_token_counter = f"""
                                <div style="padding: 8px 12px; margin: 8px 0 4px 0;">
                                    <div style="display: flex; justify-content: space-between; align-items: center; height: 40px;">
                                        <div style="text-align: center; flex: 1;">
                                            <div style="color: rgba(255, 255, 255, 0.6); font-size: 11px; text-transform: uppercase; line-height: 1; margin: 0;">Query</div>
                                            <div style="color: #64b5f6; font-weight: 600; font-size: 13px; line-height: 1.8; margin: 0;">{tokens['query_tokens']}</div>
                                        </div>
                                        <div style="text-align: center; flex: 1;">
                                            <div style="color: rgba(255, 255, 255, 0.6); font-size: 11px; text-transform: uppercase; line-height: 1; margin: 0;">Text RAG</div>
                                            <div style="color: #64b5f6; font-weight: 600; font-size: 13px; line-height: 1.8; margin: 0;">{tokens['text_rag_tokens']}</div>
                                        </div>
                                        <div style="text-align: center; flex: 1;">
                                            <div style="color: rgba(255, 255, 255, 0.6); font-size: 11px; text-transform: uppercase; line-height: 1; margin: 0;">VLM</div>
                                            <div style="color: #64b5f6; font-weight: 600; font-size: 13px; line-height: 1.8; margin: 0;">{tokens['vlm_analysis_tokens']}</div>
                                        </div>
                                        <div style="text-align: center; flex: 1;">
                                            <div style="color: rgba(255, 255, 255, 0.6); font-size: 11px; text-transform: uppercase; line-height: 1; margin: 0;">SF</div>
                                            <div style="color: #64b5f6; font-weight: 600; font-size: 13px; line-height: 1.8; margin: 0;">{tokens['salesforce_api_tokens']}</div>
                                        </div>
                                        <div style="text-align: center; flex: 1;">
                                            <div style="color: rgba(255, 255, 255, 0.6); font-size: 11px; text-transform: uppercase; line-height: 1; margin: 0;">Re-rank</div>
                                            <div style="color: #64b5f6; font-weight: 600; font-size: 13px; line-height: 1.8; margin: 0;">{tokens['reranker_tokens']}</div>
                                        </div>
                                        <div style="text-align: center; flex: 1;">
                                            <div style="color: rgba(255, 255, 255, 0.6); font-size: 11px; text-transform: uppercase; line-height: 1; margin: 0;">Response</div>
                                            <div style="color: #64b5f6; font-weight: 600; font-size: 13px; line-height: 1.8; margin: 0;">{tokens['response_tokens']}</div>
                                        </div>
                                        <div style="text-align: center; flex: 1; border-left: 1px solid rgba(255, 255, 255, 0.2); padding-left: 8px;">
                                            <div style="color: rgba(255, 255, 255, 0.6); font-size: 11px; text-transform: uppercase; line-height: 1; margin: 0;">Total</div>
                                            <div style="color: #81c784; font-weight: 600; font-size: 13px; line-height: 1.8; margin: 0;">{tokens['total_tokens']}</div>
                                        </div>
                                    </div>
                                </div>
                                """
                                st.markdown(perfect_token_counter, unsafe_allow_html=True)
                        
                        # Rejected sources (transparency)
                        if result.get('rejected_sources'):
                            with st.expander("🔍 Why other sources weren't selected"):
                                for rejected in result['rejected_sources']:
                                    st.markdown(f"""
                                    <div class="rejected-sources">
                                        <strong>{rejected['type'].upper()}:</strong> {rejected['reason']}
                                    </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.error(f"❌ {result['error']}")
                    
                    # Source and timestamp
                    st.markdown(f'<div style="color: rgba(255, 255, 255, 0.6); font-size: 0.75rem; text-align: center; margin-top: 0.75rem;">Multi-source | Time: {chat["processing_time"]:.2f}s | {chat["timestamp"].strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
    
    with col2:
        # Document Management Glass Panel
        with stylable_container(
            key="document_management_panel",
            css_styles="""
            {
                background: rgba(255, 255, 255, 0.06);
                backdrop-filter: blur(20px) saturate(180%);
                -webkit-backdrop-filter: blur(20px) saturate(180%);
                border: 1px solid rgba(100, 181, 246, 0.2);
                border-radius: 18px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                box-shadow: 
                    0 12px 40px rgba(100, 181, 246, 0.1),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
            }
            """
        ):
            st.markdown('<h3 style="color: #64b5f6; margin-bottom: 1.5rem; font-size: 1.2rem; text-align: center; text-shadow: 0 2px 8px rgba(0,0,0,0.3);">📁 Document Management</h3>', unsafe_allow_html=True)
            
            
            # Streamlined File Upload with glassmorphic styling
            try:
                uploaded_files = st.file_uploader(
                    "📤 Upload Documents",
                    accept_multiple_files=True,
                    type=['pdf', 'txt', 'docx', 'doc', 'xlsx', 'xls', 'csv'],
                    help="Upload documents for multi-source processing (PDF, DOCX, TXT, CSV supported)",
                    key="main_file_uploader"  # Static key prevents AxiosError 400
                )
                
                # If upload fails, show helpful troubleshooting info
                if uploaded_files is None and 'upload_error_shown' not in st.session_state:
                    st.info("💡 **Upload Tips**: If you encounter errors, try refreshing the page or uploading one file at a time.")
                    
            except Exception as e:
                st.error(f"❌ File upload error: {str(e)}")
                st.error("🔧 **Troubleshooting**: This might be a browser/network issue. Try:")
                st.error("   • Refresh the page and try again")
                st.error("   • Upload files one at a time") 
                st.error("   • Check your internet connection")
                st.error("   • Try a smaller file first")
                st.session_state.upload_error_shown = True
                uploaded_files = None
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files selected")
            
            # Add some spacing and then the process button
            st.write("")
            process_docs = st.button(
                "📤 Process Documents", 
                type="primary",
                help="Process uploaded documents with multi-source RAG"
            )
            
            if process_docs:
                # Set flag to prevent query rerun during file upload
                st.session_state.file_upload_in_progress = True
                
                # Auto-initialize systems if needed
                if not st.session_state.orchestrator.ensure_initialized():
                    st.stop()  # Stop execution if initialization failed
                
                import torch
                if torch.cuda.is_available():
                    spinner_msg = "Processing documents (GPU acceleration)..."
                else:
                    spinner_msg = "Processing documents (CPU mode - may take longer)..."
                    st.info("⏱️ **CPU Processing**: ColPali analysis may take 1-2 minutes per document.")
                
                with st.spinner(spinner_msg):
                    # Save uploaded files to temporary paths with robust error handling
                    temp_paths = []
                    upload_success_count = 0
                    upload_error_count = 0
                    
                    try:
                        for i, uploaded_file in enumerate(uploaded_files):
                            try:
                                # Enhanced file validation
                                file_size = len(uploaded_file.getvalue())
                                
                                # File size validation
                                if file_size > 200 * 1024 * 1024:  # 200MB limit
                                    st.error(f"❌ File {uploaded_file.name} too large ({file_size/1024/1024:.1f}MB). Max 200MB.")
                                    upload_error_count += 1
                                    continue
                                
                                if file_size == 0:
                                    st.warning(f"⚠️ File {uploaded_file.name} is empty, skipping.")
                                    upload_error_count += 1
                                    continue
                                
                                # File type validation (additional check)
                                allowed_extensions = ['.pdf', '.txt', '.docx', '.doc', '.xlsx', '.xls', '.csv']
                                file_extension = Path(uploaded_file.name).suffix.lower()
                                if file_extension not in allowed_extensions:
                                    st.error(f"❌ File {uploaded_file.name} has unsupported extension: {file_extension}")
                                    upload_error_count += 1
                                    continue
                                
                                # Save to temporary file with retry mechanism
                                max_retries = 3
                                for retry in range(max_retries):
                                    try:
                                        # Create secure temporary file
                                        with tempfile.NamedTemporaryFile(
                                            delete=False, 
                                            suffix=f"_{uploaded_file.name}",
                                            mode='wb'
                                        ) as tmp_file:
                                            tmp_file.write(uploaded_file.getvalue())
                                            temp_paths.append(tmp_file.name)
                                            
                                        st.success(f"✅ {uploaded_file.name} ({file_size/1024:.1f}KB) ready for processing")
                                        upload_success_count += 1
                                        break  # Success, exit retry loop
                                        
                                    except Exception as retry_err:
                                        if retry < max_retries - 1:
                                            st.warning(f"⚠️ Retry {retry + 1}/3 for {uploaded_file.name}: {str(retry_err)}")
                                            time.sleep(1)  # Brief delay before retry
                                        else:
                                            st.error(f"❌ Failed to save {uploaded_file.name} after {max_retries} attempts: {str(retry_err)}")
                                            upload_error_count += 1
                                
                            except Exception as file_err:
                                st.error(f"❌ Error processing {uploaded_file.name}: {str(file_err)}")
                                upload_error_count += 1
                                continue
                        
                        # Upload summary
                        total_files = len(uploaded_files)
                        if upload_success_count > 0:
                            st.info(f"📊 **Upload Summary**: {upload_success_count}/{total_files} files ready for processing")
                        
                        if upload_error_count > 0:
                            st.warning(f"⚠️ {upload_error_count} files had issues and were skipped")
                        
                        # Check if we have valid files to process
                        if not temp_paths:
                            st.error("❌ No valid files to process. Please check file formats and sizes.")
                            st.info("💡 **Supported formats**: PDF, TXT, DOCX (max 200MB each)")
                            return
                        
                        # Process documents (VLM analysis happens here while files exist)
                        result = st.session_state.orchestrator.add_documents(temp_paths)
                        
                        # Display results
                        if result['successful']:
                            st.success(f"✅ Processed {len(result['successful'])} documents successfully")
                            for doc in result['successful']:
                                filename = doc.get('filename', doc.get('path', 'Unknown'))
                                text_info = f"📝 {doc.get('text_chunks', 0)} chunks" if doc.get('text_success') else ""
                                colpali_info = f"🖼️ {doc.get('colpali_pages', 0)} pages" if doc.get('colpali_success') else ""
                                
                                if text_info and colpali_info:
                                    st.info(f"📄 {filename}: {text_info} + {colpali_info}")
                                elif text_info:
                                    st.info(f"📄 {filename}: {text_info}")
                                elif colpali_info:
                                    st.info(f"📄 {filename}: {colpali_info}")
                                else:
                                    st.info(f"📄 {filename}: processed")
                        
                        if result['failed']:
                            st.error(f"❌ Failed to process {len(result['failed'])} documents")
                            for doc in result['failed']:
                                st.error(f"• {os.path.basename(doc.get('path', ''))}: {doc.get('error', 'Unknown error')}")
                        
                        st.caption(f"Processing time: {result.get('processing_time', 0):.2f}s")
                        
                    except Exception as e:
                        st.error(f"❌ Document processing failed: {e}")
                    finally:
                        # Clear file upload flag
                        st.session_state.file_upload_in_progress = False
                        # Clean up temporary files
                        for temp_path in temp_paths:
                            try:
                                os.unlink(temp_path)
                            except Exception:
                                pass
        
        st.divider()
        
        # Processing Status Glass Panel - Matching prototype styling
        with stylable_container(
            key="processing_status_panel",
            css_styles="""
            {
                background: rgba(255, 255, 255, 0.04);
                backdrop-filter: blur(18px) saturate(160%);
                -webkit-backdrop-filter: blur(18px) saturate(160%);
                border: 1px solid rgba(100, 181, 246, 0.15);
                border-radius: 18px;
                padding: 1.5rem;
                margin-top: 0rem;
                margin-bottom: 1.5rem;
                box-shadow: 
                    0 8px 30px rgba(100, 181, 246, 0.08),
                    inset 0 1px 0 rgba(255, 255, 255, 0.08);
            }
            """
        ):
            st.markdown('<h3 style="color: #64b5f6; margin-bottom: 1.5rem; font-size: 1.2rem; text-align: center; text-shadow: 0 2px 8px rgba(0,0,0,0.3);">⚡ Processing Status</h3>', unsafe_allow_html=True)
            
            # Dynamic status based on hardware
            import torch
            gpu_mode = torch.cuda.is_available()
            
            # Create glassmorphic status display matching prototype style
            st.markdown(f"""
            <div style="margin: 0.5rem 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem; padding: 0.5rem 0;">
                    <span style="color: rgba(255, 255, 255, 0.8); font-size: 0.85rem;">Mode:</span>
                    <span style="color: #64b5f6; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">{"GPU" if gpu_mode else "CPU"}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem; padding: 0.5rem 0;">
                    <span style="color: rgba(255, 255, 255, 0.8); font-size: 0.85rem;">Systems:</span>
                    <span style="color: #81c784; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">{"All Ready" if st.session_state.components_initialized else "Initializing"}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem; padding: 0.5rem 0;">
                    <span style="color: rgba(255, 255, 255, 0.8); font-size: 0.85rem;">Queries:</span>
                    <span style="color: #64b5f6; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">{len(st.session_state.chat_history)}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem; padding: 0.5rem 0;">
                    <span style="color: rgba(255, 255, 255, 0.8); font-size: 0.85rem;">ColPali:</span>
                    <span style="color: #64b5f6; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">{"Fast ⚡" if gpu_mode else "Functional 🐌"}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.5rem 0;">
                    <span style="color: rgba(255, 255, 255, 0.8); font-size: 0.85rem;">Sources:</span>
                    <span style="color: rgba(255, 255, 255, 0.9); font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">3 Active</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Clear History Button with glassmorphic styling
        if st.session_state.chat_history:
            with stylable_container(
                key="clear_button",
                css_styles="""
                {
                    background: rgba(244, 67, 54, 0.1);
                    backdrop-filter: blur(15px);
                    border: 1px solid rgba(244, 67, 54, 0.2);
                    border-radius: 12px;
                    padding: 0.75rem;
                    transition: all 0.3s ease;
                }
                """
            ):
                if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
                    st.session_state.chat_history = []
                    # Note: Removed st.rerun() to prevent interference with file uploads


# Run the app
if __name__ == "__main__":
    main()
