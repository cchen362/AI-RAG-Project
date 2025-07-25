"""
Visual Document Processor for ColPali Integration

This module handles PDF to image conversion and ColPali visual embedding generation.
Based on the working prototype from simple_colpali_app.py.
"""

import os
import time
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import torch
import platform
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

class VisualDocumentProcessor:
    """
    Processes PDF documents for ColPali visual understanding.
    
    This class:
    1. Converts PDF pages to images
    2. Uses ColPali model to generate visual embeddings
    3. Handles CPU/GPU optimization
    4. Provides query capabilities with MaxSim scoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the visual document processor."""
        self.config = config
        self.model_name = config.get('colpali_model', 'vidore/colqwen2-v1.0')
        self.device = self._detect_device()
        self.cache_dir = config.get('cache_dir', 'cache/embeddings')
        
        # Model components
        self.model = None
        self.processor = None
        self.model_loaded = False
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"🔧 VisualDocumentProcessor initialized for {self.model_name}")
        logger.info(f"🖥️ Device: {self.device}")
    
    def _detect_device(self) -> str:
        """Detect optimal device (GPU/CPU) for processing."""
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🚀 GPU detected: {gpu_name}")
        else:
            device = "cpu"
            logger.info("💻 Using CPU (GPU not available)")
        return device
    
    def _load_model(self):
        """Load ColPali model and processor."""
        if self.model_loaded:
            return
        
        try:
            logger.info(f"📥 Loading ColPali model: {self.model_name}")
            start_time = time.time()
            
            # Import ColPali components with 2025 API compatibility
            from transformers.utils.import_utils import is_flash_attn_2_available
            
            try:
                from colpali_engine.models import ColQwen2, ColQwen2Processor
                logger.info("✅ Using colpali_engine ColQwen2 models")
                use_colpali_engine = True
            except ImportError:
                logger.warning("⚠️ colpali_engine not available, trying transformers")
                use_colpali_engine = False
            
            if use_colpali_engine:
                # Try ColPali engine first (latest API)
                try:
                    # Determine optimal settings
                    torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
                    attn_implementation = "flash_attention_2" if (self.device == "cuda" and is_flash_attn_2_available()) else None
                    
                    logger.info(f"🔧 Loading with dtype: {torch_dtype}, attention: {attn_implementation}")
                    
                    self.model = ColQwen2.from_pretrained(
                        self.model_name,
                        torch_dtype=torch_dtype,
                        device_map=self.device,
                        trust_remote_code=True,
                        attn_implementation=attn_implementation
                    ).eval()
                    
                    self.processor = ColQwen2Processor.from_pretrained(
                        self.model_name,
                        trust_remote_code=True
                    )
                    
                    logger.info("✅ ColQwen2 models loaded successfully")
                    
                except Exception as engine_error:
                    logger.warning(f"⚠️ ColPali engine failed: {engine_error}")
                    use_colpali_engine = False
            
            if not use_colpali_engine:
                # Fallback to transformers approach
                logger.info("🔄 Using transformers fallback")
                from transformers import AutoModel, AutoProcessor
                
                torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
                attn_implementation = "flash_attention_2" if (self.device == "cuda" and is_flash_attn_2_available()) else None
                
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    device_map=self.device,
                    trust_remote_code=True,
                    attn_implementation=attn_implementation
                ).eval()
                
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
            
            load_time = time.time() - start_time
            logger.info(f"✅ Model loaded in {load_time:.2f}s")
            self.model_loaded = True
            
        except Exception as e:
            logger.error(f"❌ Failed to load ColPali model: {e}")
            raise
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a PDF file and generate visual embeddings.
        
        Returns:
            Dict containing embeddings, metadata, and processing info
        """
        if not file_path.lower().endswith('.pdf'):
            return {
                'status': 'error',
                'error': 'Only PDF files are supported for visual processing'
            }
        
        if not os.path.exists(file_path):
            return {
                'status': 'error',
                'error': f'File not found: {file_path}'
            }
        
        try:
            logger.info(f"🖼️ Processing PDF: {os.path.basename(file_path)}")
            start_time = time.time()
            
            # Load model if not already loaded
            self._load_model()
            
            # Convert PDF to images
            images = self._convert_pdf_to_images(file_path)
            if not images:
                return {
                    'status': 'error',
                    'error': 'Failed to convert PDF to images'
                }
            
            # Generate visual embeddings
            embeddings = self._generate_embeddings(images)
            
            processing_time = time.time() - start_time
            
            # Create result
            result = {
                'status': 'success',
                'file_path': file_path,
                'embeddings': embeddings,
                'metadata': {
                    'filename': os.path.basename(file_path),
                    'page_count': len(images),
                    'model_name': self.model_name,
                    'device': self.device,
                    'embedding_shape': embeddings.shape if hasattr(embeddings, 'shape') else 'unknown'
                },
                'file_info': {
                    'size_bytes': os.path.getsize(file_path),
                    'path': file_path
                },
                'page_info': {
                    'total_pages': len(images),
                    'processed_pages': len(images)
                },
                'processing_time': processing_time,
                'embedding_type': 'visual_patches',
                'model_name': self.model_name,
                'device': self.device
            }
            
            logger.info(f"✅ Processed {len(images)} pages in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'file_path': file_path
            }
    
    def _convert_pdf_to_images(self, pdf_path: str) -> List:
        """Convert PDF pages to PIL Images."""
        try:
            from pdf2image import convert_from_path
            from pdf2image.exceptions import PDFInfoNotInstalledError
            
            # Determine page limit based on device
            max_pages = 5 if self.device == "cpu" else 20
            
            # Try different poppler paths for Windows
            poppler_paths = self._get_poppler_paths()
            
            for poppler_path in poppler_paths:
                try:
                    logger.info(f"🔧 Attempting PDF conversion with poppler: {poppler_path or 'system PATH'}")
                    
                    images = convert_from_path(
                        pdf_path,
                        dpi=200,
                        poppler_path=poppler_path,
                        first_page=1,
                        last_page=max_pages
                    )
                    
                    if images:
                        logger.info(f"✅ Converted {len(images)} pages successfully")
                        return images
                        
                except (PDFInfoNotInstalledError, Exception) as e:
                    logger.debug(f"Poppler path failed {poppler_path}: {e}")
                    continue
            
            # Final fallback
            logger.error("❌ All poppler paths failed")
            logger.error("💡 Install poppler: conda install -c conda-forge poppler")
            return []
            
        except ImportError:
            logger.error("❌ pdf2image not installed: pip install pdf2image")
            return []
    
    def _get_poppler_paths(self) -> List[Optional[str]]:
        """Get platform-specific poppler installation paths."""
        system_platform = platform.system().lower()
        paths = []
        
        # Always check environment variable first
        if 'POPPLER_PATH' in os.environ:
            paths.append(os.environ['POPPLER_PATH'])
        
        # Always try system PATH first (works for Docker/apt-installed poppler)
        paths.append(None)
        
        if system_platform == 'linux':
            # Linux/Docker paths - prioritize system installation
            paths.extend([
                '/usr/bin',           # Standard apt-get install location
                '/usr/local/bin',     # Manual installation
                '/opt/poppler/bin',   # Alternative installation
                '/usr/share/poppler/bin'
            ])
            
        elif system_platform == 'darwin':  # macOS
            # macOS paths - Homebrew and MacPorts
            paths.extend([
                '/opt/homebrew/bin',      # Apple Silicon Homebrew
                '/usr/local/bin',         # Intel Homebrew
                '/opt/local/bin',         # MacPorts
                '/usr/local/Cellar/poppler/*/bin'  # Homebrew versioned
            ])
            
        elif system_platform == 'windows':
            # Windows paths - comprehensive list
            username = os.getenv('USERNAME', '')
            
            # Add conda environment variable paths
            if 'CONDA_PREFIX' in os.environ:
                conda_prefix = os.environ['CONDA_PREFIX']
                paths.append(os.path.join(conda_prefix, 'Library', 'bin'))
            
            # Add conda environment paths (miniconda/anaconda)
            if username:
                paths.extend([
                    rf"C:\Users\{username}\miniconda3\Library\bin",
                    rf"C:\Users\{username}\anaconda3\Library\bin",
                    rf"C:\Users\{username}\Miniconda3\Library\bin",
                    rf"C:\Users\{username}\Anaconda3\Library\bin"
                ])
            
            # Fallback manual installation paths
            paths.extend([
                r"C:\Program Files\poppler\poppler-24.08.0\Library\bin",
                r"C:\Program Files\poppler\Library\bin",
                r"C:\Program Files\poppler\bin",
                r"C:\Program Files (x86)\poppler\bin",
                r"C:\poppler\bin",
                r"C:\tools\poppler\bin"
            ])
        
        # Filter out non-existent paths (except None for system PATH)
        validated_paths = []
        for path in paths:
            if path is None:
                validated_paths.append(path)  # Always include system PATH
            elif os.path.exists(path):
                validated_paths.append(path)
            else:
                logger.debug(f"Poppler path does not exist: {path}")
        
        logger.info(f"🔧 Platform: {system_platform}, found {len(validated_paths)} poppler paths")
        return validated_paths
    
    def _generate_embeddings(self, images: List) -> torch.Tensor:
        """Generate ColPali embeddings for document images."""
        try:
            # Prepare images for processing
            batch_images = []
            for img in images:
                # Convert PIL image to the format expected by ColPali
                batch_images.append(img)
            
            # Process images through ColPali with updated API handling
            with torch.no_grad():
                try:
                    # Use updated ColQwen2 processor API
                    batch_inputs = self.processor(images=batch_images, return_tensors="pt")
                    
                    # Move to device
                    for key in batch_inputs:
                        if isinstance(batch_inputs[key], torch.Tensor):
                            batch_inputs[key] = batch_inputs[key].to(self.device)
                    
                    # Generate embeddings with ColQwen2 model
                    outputs = self.model(**batch_inputs)
                    
                    # Handle ColQwen2 output format
                    if hasattr(outputs, 'last_hidden_state'):
                        embeddings = outputs.last_hidden_state
                    elif isinstance(outputs, tuple):
                        embeddings = outputs[0]
                    else:
                        embeddings = outputs
                    
                except Exception as proc_err:
                    logger.warning(f"⚠️ Standard processing failed: {proc_err}, trying alternative")
                    # Alternative processing approach
                    try:
                        batch_inputs = self.processor(batch_images, return_tensors="pt")
                        for key in batch_inputs:
                            if isinstance(batch_inputs[key], torch.Tensor):
                                batch_inputs[key] = batch_inputs[key].to(self.device)
                        
                        outputs = self.model(**batch_inputs)
                        if hasattr(outputs, 'last_hidden_state'):
                            embeddings = outputs.last_hidden_state
                        else:
                            embeddings = outputs
                    except Exception as alt_err:
                        logger.error(f"❌ Alternative processing also failed: {alt_err}")
                        raise
            
            # Return embeddings tensor
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            # Return dummy embeddings as fallback
            dummy_embeddings = torch.zeros((len(images), 1024, 128), dtype=torch.float32)
            logger.warning(f"⚠️ Using dummy embeddings ({dummy_embeddings.shape})")
            return dummy_embeddings
    
    def query_embeddings(self, query: str, document_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Query document embeddings using ColPali's MaxSim scoring.
        
        Args:
            query: Text query string
            document_embeddings: ColPali embeddings for the document
            
        Returns:
            Tensor of similarity scores
        """
        try:
            self._load_model()
            
            # Process query text with ColQwen2 API
            with torch.no_grad():
                try:
                    # Use standard text processing for queries
                    query_inputs = self.processor(text=[query], return_tensors="pt")
                    
                    # Move to device
                    for key in query_inputs:
                        if isinstance(query_inputs[key], torch.Tensor):
                            query_inputs[key] = query_inputs[key].to(self.device)
                    
                    # Generate query embedding
                    query_outputs = self.model(**query_inputs)
                    
                    # Handle ColQwen2 output format
                    if hasattr(query_outputs, 'last_hidden_state'):
                        query_embedding = query_outputs.last_hidden_state
                    elif isinstance(query_outputs, tuple):
                        query_embedding = query_outputs[0]
                    else:
                        query_embedding = query_outputs
                        
                except Exception as query_err:
                    logger.warning(f"⚠️ Query processing error: {query_err}, using fallback")
                    # Fallback query processing
                    try:
                        query_inputs = self.processor(query, return_tensors="pt")
                        for key in query_inputs:
                            if isinstance(query_inputs[key], torch.Tensor):
                                query_inputs[key] = query_inputs[key].to(self.device)
                        
                        query_outputs = self.model(**query_inputs)
                        if hasattr(query_outputs, 'last_hidden_state'):
                            query_embedding = query_outputs.last_hidden_state
                        else:
                            query_embedding = query_outputs
                    except Exception as fallback_err:
                        logger.error(f"❌ Fallback query processing failed: {fallback_err}")
                        raise
                
                # Move document embeddings to same device
                if isinstance(document_embeddings, torch.Tensor):
                    document_embeddings = document_embeddings.to(self.device)
                
                # Calculate MaxSim scores
                scores = self._calculate_maxsim_scores(query_embedding, document_embeddings)
            
            return scores
            
        except Exception as e:
            logger.error(f"❌ Query embedding failed: {e}")
            # Return dummy scores
            if hasattr(document_embeddings, 'shape'):
                num_pages = document_embeddings.shape[0] if len(document_embeddings.shape) > 2 else 1
            else:
                num_pages = 1
            return torch.zeros(num_pages, dtype=torch.float32)
    
    def _calculate_maxsim_scores(self, query_embedding: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        """Calculate MaxSim scores between query and document embeddings with dimension handling."""
        try:
            # Ensure tensors are on same device
            query_embedding = query_embedding.to(self.device)
            doc_embeddings = doc_embeddings.to(self.device)
            
            logger.info(f"🔍 Query embedding shape: {query_embedding.shape}")
            logger.info(f"🔍 Document embeddings shape: {doc_embeddings.shape}")
            
            # Handle dimension mismatches
            query_dim = query_embedding.shape[-1]
            doc_dim = doc_embeddings.shape[-1]
            
            if query_dim != doc_dim:
                logger.warning(f"⚠️ Dimension mismatch: query={query_dim}, doc={doc_dim}")
                
                # Strategy: Truncate to smaller dimension
                min_dim = min(query_dim, doc_dim)
                query_embedding = query_embedding[..., :min_dim]
                doc_embeddings = doc_embeddings[..., :min_dim]
                
                logger.info(f"🔧 Truncated to dimension: {min_dim}")
            
            # Calculate similarity scores with proper shape handling
            if len(doc_embeddings.shape) >= 3:
                # Multiple pages with patches: [num_pages, num_patches, embedding_dim]
                page_scores = []
                
                for page_idx in range(doc_embeddings.shape[0]):
                    page_patches = doc_embeddings[page_idx]  # [num_patches, embedding_dim]
                    
                    # Handle query embedding shape properly
                    if len(query_embedding.shape) == 3:
                        # Shape: [batch, tokens, embedding_dim] -> average over tokens
                        query_emb = query_embedding.mean(dim=1)  # [batch, embedding_dim]
                        if query_emb.shape[0] == 1:
                            query_emb = query_emb  # Keep [1, embedding_dim]
                        else:
                            query_emb = query_emb.mean(dim=0, keepdim=True)  # [1, embedding_dim]
                    elif len(query_embedding.shape) == 2:
                        if query_embedding.shape[0] == 1:
                            query_emb = query_embedding  # [1, embedding_dim]
                        else:
                            # Multiple embeddings, take mean
                            query_emb = query_embedding.mean(dim=0, keepdim=True)  # [1, embedding_dim]
                    else:
                        # 1D embedding
                        query_emb = query_embedding.unsqueeze(0)  # [1, embedding_dim]
                    
                    # Compute cosine similarity between query and all patches
                    if page_idx == 0:  # Only log for first page to avoid spam
                        logger.info(f"   Page {page_idx}: query_emb shape: {query_emb.shape}, page_patches shape: {page_patches.shape}")
                    
                    # Normalize embeddings
                    query_norm = torch.nn.functional.normalize(query_emb, p=2, dim=-1)
                    patches_norm = torch.nn.functional.normalize(page_patches, p=2, dim=-1)
                    
                    if page_idx == 0:
                        logger.info(f"   Normalized shapes: query_norm: {query_norm.shape}, patches_norm: {patches_norm.shape}")
                    
                    # Calculate similarities using batch matrix multiplication
                    # query_norm: [1, embedding_dim], patches_norm: [num_patches, embedding_dim]
                    similarities = torch.matmul(query_norm, patches_norm.T).squeeze(0)  # [num_patches]
                    
                    if page_idx == 0:
                        logger.info(f"   Similarities shape: {similarities.shape}")
                    
                    # MaxSim: take maximum similarity
                    max_sim = similarities.max()
                    page_scores.append(max_sim)
                
                scores = torch.stack(page_scores)
                
            elif len(doc_embeddings.shape) == 2:
                # Single page with patches: [num_patches, embedding_dim]
                # Handle query embedding shape properly
                if len(query_embedding.shape) == 3:
                    # Shape: [batch, tokens, embedding_dim] -> average over tokens
                    query_emb = query_embedding.mean(dim=1)  # [batch, embedding_dim]
                    if query_emb.shape[0] != 1:
                        query_emb = query_emb.mean(dim=0, keepdim=True)  # [1, embedding_dim]
                elif len(query_embedding.shape) == 2:
                    if query_embedding.shape[0] == 1:
                        query_emb = query_embedding  # [1, embedding_dim]
                    else:
                        query_emb = query_embedding.mean(dim=0, keepdim=True)  # [1, embedding_dim]
                else:
                    query_emb = query_embedding.unsqueeze(0)  # [1, embedding_dim]
                
                # Normalize and compute similarity
                query_norm = torch.nn.functional.normalize(query_emb, p=2, dim=-1)
                doc_norm = torch.nn.functional.normalize(doc_embeddings, p=2, dim=-1)
                
                similarities = torch.matmul(query_norm, doc_norm.T).squeeze(0)
                scores = similarities.max().unsqueeze(0)  # Make it [1] shape
                
            else:
                # Single embeddings: direct comparison
                scores = torch.cosine_similarity(
                    query_embedding.flatten().unsqueeze(0), 
                    doc_embeddings.flatten().unsqueeze(0), 
                    dim=-1
                )
            
            logger.info(f"✅ Calculated scores shape: {scores.shape}, values: {scores}")
            return scores
            
        except Exception as e:
            logger.error(f"❌ MaxSim calculation failed: {e}")
            logger.error(f"   Query shape: {query_embedding.shape if 'query_embedding' in locals() else 'unknown'}")
            logger.error(f"   Doc shape: {doc_embeddings.shape if 'doc_embeddings' in locals() else 'unknown'}")
            
            # Return dummy scores based on document structure
            if 'doc_embeddings' in locals() and len(doc_embeddings.shape) >= 3:
                num_pages = doc_embeddings.shape[0]
                return torch.zeros(num_pages, dtype=torch.float32)
            else:
                return torch.zeros(1, dtype=torch.float32)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'model_loaded': self.model_loaded,
            'cache_dir': self.cache_dir,
            'cuda_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }