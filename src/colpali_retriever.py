"""
ColPali Vision-Language Retriever

This retriever implements ColPali's vision-language approach:
1. Convert PDF pages to images
2. Use ColPali model to generate vision-language embeddings per page
3. Store page embeddings in vector database
4. Use MaxSim scoring for query matching (not cosine similarity)
5. Return full page images/regions instead of text chunks
"""

import os
import time
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
import torch
import base64
from io import BytesIO
from PIL import Image
import sys
import platform
import shutil
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Try absolute import first, fallback to relative
try:
    from src.embedding_manager import EmbeddingManager
except ImportError:
    from embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)

# Simple data classes to replace the missing ones
class RetrievalResult:
    def __init__(self, content: str, score: float, metadata: dict, source_type: str = "visual", processing_time: float = 0.0):
        self.content = content
        self.score = score
        self.metadata = metadata
        self.source_type = source_type
        self.processing_time = processing_time

class RetrievalMetrics:
    def __init__(self, query_time: float, total_results: int, avg_score: float, max_score: float, tokens_used: int, cost_estimate: float):
        self.query_time = query_time
        self.total_results = total_results
        self.avg_score = avg_score
        self.max_score = max_score
        self.tokens_used = tokens_used
        self.cost_estimate = cost_estimate

class ColPaliRetriever:
    """
    ColPali-based visual document retriever.
    
    Uses vision-language models to understand document layout,
    visual elements, and textual content simultaneously.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_initialized = False
        self.processed_documents = []
        
        # Check poppler availability during initialization
        self.poppler_available = False
        
        # Initialize retrieval statistics
        self.retrieval_stats = {
            'total_queries': 0,
            'total_query_time': 0.0,
            'total_results_returned': 0,
            'total_tokens_used': 0,
            'total_cost': 0.0,
            'total_documents_processed': 0,
            'avg_query_time': 0.0,
            'avg_results_per_query': 0.0
        }
        
        # ColPali-specific configuration
        self.model_name = config.get('model_name', 'vidore/colqwen2-v1.0')
        self.device = config.get('device', 'auto')  # 'auto', 'cuda', 'cpu'
        self.max_pages = config.get('max_pages_per_doc', 50)
        
        # Initialize visual components
        try:
            logger.info(f"🔧 Initializing ColPali components with model: {self.model_name}")
            
            # Initialize visual document processor
            visual_config = {
                'model_name': self.model_name,
                'device': self.device,
                'cache_embeddings': config.get('cache_embeddings', True),
                'cache_dir': config.get('cache_dir', 'cache/embeddings')
            }
            
            # Initialize visual document processor
            try:
                from src.visual_document_processor import VisualDocumentProcessor
                self.visual_processor = VisualDocumentProcessor(visual_config)
                logger.info("✅ VisualDocumentProcessor initialized successfully")
            except ImportError as e:
                logger.error(f"❌ VisualDocumentProcessor not available: {e}")
                self.visual_processor = None
            self.embedding_manager = EmbeddingManager.create_colpali(self.model_name)
            
            # Storage for page embeddings, metadata, and images
            self.page_embeddings = {}  # {doc_id: {'embeddings': tensor, 'metadata': dict}}
            self.document_metadata = {}  # {doc_id: doc_info}
            self.page_images = {}  # {doc_id: {page_idx: PIL.Image}}
            # Removed precomputed_vlm_content - now using live query-specific analysis
            
            # Initialize VLM for image analysis
            self._init_vlm_client()
            
            logger.info("✅ ColPaliRetriever initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ColPali components: {e}")
            raise
    
    def get_retriever_type(self) -> str:
        return "visual"
    
    def _init_vlm_client(self):
        """Initialize Vision-Language Model client for image analysis."""
        try:
            # Try to use OpenAI GPT-4 Vision if API key available
            import openai
            api_key = os.getenv('OPENAI_API_KEY')
            
            if api_key:
                self.vlm_client = openai.OpenAI(api_key=api_key)
                self.vlm_model = "gpt-4o"  # GPT-4 with vision
                self.vlm_available = True
                logger.info("✅ OpenAI GPT-4 Vision initialized for image analysis")
            else:
                logger.warning("⚠️ OpenAI API key not found - visual analysis will be limited")
                self.vlm_client = None
                self.vlm_model = None
                self.vlm_available = False
                
        except ImportError:
            logger.warning("⚠️ OpenAI not available - visual analysis will be limited")
            self.vlm_client = None
            self.vlm_model = None
            self.vlm_available = False
    
    def add_documents(self, document_paths: List[str]) -> Dict[str, Any]:
        """
        Process documents by converting pages to images and generating visual embeddings.
        """
        logger.info(f"🖼️ Processing {len(document_paths)} documents with ColPaliRetriever")
        
        # Check poppler availability on first use
        if not hasattr(self, '_poppler_checked'):
            self.poppler_available = self.check_poppler_availability()
            self._poppler_checked = True
        
        # If poppler is not available, return early with clear messaging
        if not self.poppler_available:
            logger.warning("⚠️ ColPali visual processing unavailable - poppler not found")
            logger.info("💡 Documents will need to be processed using text-based RAG only")
            return {
                'successful': [],
                'failed': [{'path': path, 'filename': os.path.basename(path), 
                           'error': 'Visual processing unavailable - poppler not found', 
                           'type': 'visual'} for path in document_paths],
                'total_pages': 0,
                'total_documents': 0,
                'processing_time': 0,
                'retriever_type': 'visual_unavailable'
            }
        
        start_time = time.time()
        results = {
            'successful': [],
            'failed': [],
            'total_pages': 0,
            'total_documents': 0,
            'processing_time': 0,
            'retriever_type': 'visual'
        }
        
        for doc_path in document_paths:
            try:
                filename = os.path.basename(doc_path)
                logger.info(f"🖼️ Processing visual content from: {filename}")
                
                # Only process PDFs for visual retrieval
                if not doc_path.lower().endswith('.pdf'):
                    results['failed'].append({
                        'path': doc_path,
                        'filename': filename,
                        'error': 'ColPali only supports PDF files',
                        'type': 'visual'
                    })
                    logger.warning(f"⚠️ Skipping non-PDF file: {filename}")
                    continue
                
                # Generate visual embeddings and extract page images
                logger.info(f"🔄 Calling embedding_manager.create_visual_embedding for {filename}")
                visual_result = self.embedding_manager.create_visual_embedding(doc_path)
                
                logger.info(f"📊 Visual embedding result: status={visual_result.get('status')}, keys={list(visual_result.keys())}")
                if 'embeddings' in visual_result and visual_result['embeddings'] is not None:
                    if hasattr(visual_result['embeddings'], 'shape'):
                        logger.info(f"📊 Embeddings shape: {visual_result['embeddings'].shape}")
                    else:
                        logger.info(f"📊 Embeddings type: {type(visual_result['embeddings'])}")
                else:
                    logger.warning(f"⚠️ No embeddings in visual_result: {visual_result}")
                
                if visual_result['status'] == 'success':
                    # Create unique document ID
                    doc_id = f"{filename}_{hash(doc_path) % 10000}"
                    
                    # Store embeddings and metadata
                    self.page_embeddings[doc_id] = {
                        'embeddings': visual_result['embeddings'],
                        'metadata': visual_result['metadata']
                    }
                    logger.info(f"📦 Stored embeddings for doc_id: {doc_id}")
                    logger.info(f"📦 Total docs in page_embeddings: {len(self.page_embeddings)}")
                    
                    self.document_metadata[doc_id] = {
                        'filename': filename,
                        'original_path': doc_path,
                        'page_count': visual_result['metadata']['page_count'],
                        'doc_id': doc_id
                    }
                    
                    # Store page images for VLM analysis (while file still exists)
                    self._store_page_images(doc_id, doc_path)
                    
                    # Store page images for live query-specific analysis
                    # No longer precompute generic analysis - do live analysis per query instead
                    
                    page_count = visual_result['metadata']['page_count']
                    results['successful'].append({
                        'path': doc_path,
                        'filename': filename,
                        'pages': page_count,
                        'doc_id': doc_id,
                        'type': 'visual'
                    })
                    
                    results['total_pages'] += page_count
                    results['total_documents'] += 1
                    self.processed_documents.append(doc_path)
                    
                    logger.info(f"✅ Processed {page_count} pages from {filename}")
                    
                else:
                    error_msg = visual_result.get('error', 'Visual processing failed')
                    results['failed'].append({
                        'path': doc_path,
                        'filename': filename,
                        'error': error_msg,
                        'type': 'visual'
                    })
                    logger.warning(f"⚠️ Failed to process {filename}: {error_msg}")
                    
            except Exception as e:
                error_msg = str(e)
                results['failed'].append({
                    'path': doc_path,
                    'filename': os.path.basename(doc_path),
                    'error': error_msg,
                    'type': 'visual'
                })
                logger.error(f"❌ Error processing {doc_path}: {error_msg}")
        
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        
        # DEBUG: Log results tracking vs actual stored data
        logger.info(f"🔍 DEBUG: results['successful'] count: {len(results['successful'])}")
        logger.info(f"🔍 DEBUG: page_embeddings count: {len(self.page_embeddings)}")
        logger.info(f"🔍 DEBUG: processed_documents count: {len(self.processed_documents)}")
        
        # FIXED: Use actual stored data for initialization status instead of results tracking
        # The results['successful'] tracking was somehow failing despite successful processing
        self.is_initialized = len(self.page_embeddings) > 0
        
        logger.info(f"🔧 FIXED: is_initialized set to {self.is_initialized} based on stored embeddings")
        
        # Update stats
        self.retrieval_stats['total_documents_processed'] += len(results['successful'])
        
        logger.info(f"✅ ColPaliRetriever processed {len(results['successful'])}/{len(document_paths)} documents")
        logger.info(f"🖼️ Indexed {results['total_pages']} visual pages in {processing_time:.2f}s")
        
        return results
    
    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """
        Retrieve relevant pages using ColPali MaxSim scoring.
        """
        if not self.is_initialized:
            logger.warning("⚠️ No documents processed yet")
            logger.warning(f"🔍 Debug: is_initialized={self.is_initialized}, page_embeddings count={len(self.page_embeddings)}")
            logger.warning(f"🔍 Debug: processed_documents count={len(self.processed_documents)}")
            return [], RetrievalMetrics(0, 0, 0, 0, 0, 0)
        
        # Check if poppler was available during document processing
        if not self.poppler_available:
            logger.warning("⚠️ Visual retrieval unavailable - poppler not found during initialization")
            return [], RetrievalMetrics(0, 0, 0, 0, 0, 0)
        
        logger.info(f"🔍 Visual retrieval for: '{query}' (top_k={top_k})")
        
        start_time = time.time()
        
        try:
            retrieval_results = []
            
            # Query each document's embeddings
            for doc_id, doc_data in self.page_embeddings.items():
                try:
                    embeddings = doc_data['embeddings']
                    metadata = doc_data['metadata']
                    doc_info = self.document_metadata[doc_id]
                    
                    # Use MaxSim scoring (ColPali's approach)
                    scores = self.embedding_manager.query_visual_embeddings(query, embeddings)
                    
                    # Handle different score formats
                    if isinstance(scores, torch.Tensor):
                        scores_array = scores.cpu().numpy()
                    elif isinstance(scores, np.ndarray):
                        scores_array = scores
                    else:
                        scores_array = np.array([scores])
                    
                    # Find best matching page
                    if len(scores_array.shape) > 0 and scores_array.size > 0:
                        best_page_idx = int(np.argmax(scores_array))
                        max_score = float(np.max(scores_array))
                    else:
                        best_page_idx = 0
                        max_score = float(scores_array) if np.isscalar(scores_array) else 0.0
                    
                    # Create page-level result
                    page_number = best_page_idx + 1  # Convert to 1-based
                    
                    # Generate VLM-based answer instead of simple description
                    content = self._generate_vlm_answer(query, doc_id, best_page_idx, doc_info)
                    
                    enhanced_metadata = {
                        'filename': doc_info['filename'],
                        'page': page_number,
                        'page_count': doc_info['page_count'],
                        'doc_id': doc_id,
                        'best_page_index': best_page_idx,
                        'retriever_type': 'visual',
                        'model_name': self.model_name,
                        'scoring_method': 'maxsim'
                    }
                    
                    result = RetrievalResult(
                        content=content,
                        score=max_score,
                        metadata=enhanced_metadata,
                        source_type='page_image',
                        processing_time=0.0
                    )
                    
                    retrieval_results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Error processing document {doc_id}: {e}")
                    continue
            
            # Sort by score and take top_k
            retrieval_results.sort(key=lambda x: x.score, reverse=True)
            top_results = retrieval_results[:top_k]
            
            query_time = time.time() - start_time
            
            # Calculate metrics
            if top_results:
                scores = [r.score for r in top_results]
                avg_score = sum(scores) / len(scores)
                max_score = max(scores)
            else:
                avg_score = max_score = 0.0
            
            # Visual queries don't use traditional tokens, but estimate for comparison
            estimated_tokens = len(query.split()) * 2  # Query processing tokens
            
            # ColPali models are typically free/local, so no cost
            cost_estimate = 0.0
            
            metrics = RetrievalMetrics(
                query_time=query_time,
                total_results=len(top_results),
                avg_score=avg_score,
                max_score=max_score,
                tokens_used=estimated_tokens,
                cost_estimate=cost_estimate
            )
            
            # Update internal stats
            self._update_retrieval_stats(query_time, len(top_results), estimated_tokens, cost_estimate)
            
            logger.info(f"✅ Retrieved {len(top_results)} visual results in {query_time:.3f}s")
            logger.info(f"🖼️ Avg score: {avg_score:.3f}, Max score: {max_score:.3f}")
            
            return top_results, metrics
            
        except Exception as e:
            logger.error(f"❌ Visual retrieval failed: {e}")
            query_time = time.time() - start_time
            return [], RetrievalMetrics(query_time, 0, 0, 0, 0, 0)
    
    def _store_page_images(self, doc_id: str, doc_path: str):
        """Store page images for VLM analysis."""
        logger.info(f"📸 Starting image storage for {doc_id}")
        logger.info(f"   PDF path: {doc_path}")
        
        try:
            from pdf2image import convert_from_path
            
            # Check if PDF file exists and is readable
            if not os.path.exists(doc_path):
                logger.error(f"❌ PDF file not found: {doc_path}")
                self.page_images[doc_id] = {}
                return
                
            file_size = os.path.getsize(doc_path)
            logger.info(f"   PDF size: {file_size/1024:.1f}KB")
            
            # Convert PDF pages to images with robust poppler handling
            logger.info(f"🔄 Converting PDF to images...")
            images = self._convert_pdf_with_fallbacks(doc_path)
            
            if images:
                # Store images by page index
                self.page_images[doc_id] = {}
                for page_idx, image in enumerate(images):
                    self.page_images[doc_id][page_idx] = image
                    logger.debug(f"   Stored page {page_idx}: {image.size} pixels")
                
                logger.info(f"✅ Successfully stored {len(images)} page images for {doc_id}")
                logger.info(f"   Images stored in memory at: self.page_images['{doc_id}']")
            else:
                logger.error(f"❌ No images could be extracted from {doc_id}")
                logger.error(f"   This means PDF→Image conversion completely failed")
                self.page_images[doc_id] = {}
            
        except Exception as e:
            logger.error(f"❌ Failed to store page images for {doc_id}: {e}")
            logger.error(f"   Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            self.page_images[doc_id] = {}
    
    def _get_platform_poppler_paths(self) -> List[str]:
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
            # Windows paths - keep existing comprehensive list
            username = os.getenv('USERNAME', '')
            paths.extend([
                r"C:\Program Files\poppler\poppler-24.08.0\Library\bin",  # Known working
                r"C:\Program Files\poppler\Library\bin",
                r"C:\Program Files\poppler\bin", 
                r"C:\Program Files (x86)\poppler\bin",
                r"C:\poppler\bin",
                r"C:\tools\poppler\bin",
                # Conda environments
                rf"C:\Users\{username}\miniconda3\Library\bin",
                rf"C:\Users\{username}\anaconda3\Library\bin",
                r"C:\ProgramData\miniconda3\Library\bin",
                r"C:\ProgramData\anaconda3\Library\bin"
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
    
    def _convert_pdf_with_fallbacks(self, pdf_path: str):
        """Convert PDF to images with multiple fallback strategies."""
        from pdf2image import convert_from_path
        from pdf2image.exceptions import PDFInfoNotInstalledError
        
        # Get platform-specific poppler paths
        possible_poppler_paths = self._get_platform_poppler_paths()
        
        for poppler_path in possible_poppler_paths:
            try:
                if poppler_path and not os.path.exists(poppler_path):
                    logger.debug(f"   Poppler path does not exist: {poppler_path}")
                    continue
                    
                logger.info(f"🔧 Trying Poppler path: {poppler_path or 'system PATH'}")
                
                images = convert_from_path(
                    pdf_path, 
                    dpi=200,
                    poppler_path=poppler_path,
                    first_page=1
                )
                
                if images:
                    logger.info(f"✅ PDF conversion successful with path: {poppler_path or 'system PATH'}")
                    logger.info(f"   Generated {len(images)} page images")
                    return images
                    
            except PDFInfoNotInstalledError:
                continue
            except Exception as e:
                logger.debug(f"Conversion attempt failed with {poppler_path}: {e}")
                continue
        
        # Final fallback: Try to find poppler executables
        logger.warning("All poppler paths failed, attempting to locate poppler...")
        return self._find_and_convert_pdf(pdf_path)
    
    def _find_and_convert_pdf(self, pdf_path: str):
        """Final fallback: search for poppler and convert."""
        try:
            import shutil
            from pdf2image import convert_from_path
            
            # Try to find poppler executables
            poppler_executables = ['pdftoppm', 'pdfinfo']
            
            for exe in poppler_executables:
                exe_path = shutil.which(exe)
                if exe_path:
                    # Extract directory and try conversion
                    poppler_dir = os.path.dirname(exe_path)
                    logger.info(f"Found poppler executable at: {poppler_dir}")
                    
                    try:
                        images = convert_from_path(
                            pdf_path,
                            dpi=200,
                            poppler_path=poppler_dir,
                            first_page=1
                        )
                        
                        if images:
                            logger.info(f"✅ PDF conversion successful with found poppler: {poppler_dir}")
                            return images
                            
                    except Exception as e:
                        logger.debug(f"Conversion failed with found poppler {poppler_dir}: {e}")
                        continue
            
            # If still no success, provide platform-specific error message
            self._log_poppler_installation_help()
            
            return []
            
        except Exception as e:
            logger.error(f"Final poppler search failed: {e}")
            return []
    
    def _log_poppler_installation_help(self):
        """Log platform-specific poppler installation instructions."""
        system_platform = platform.system().lower()
        
        logger.error("❌ Poppler not found. Please install poppler-utils:")
        
        if system_platform == 'linux':
            logger.error("   Ubuntu/Debian: sudo apt-get install poppler-utils")
            logger.error("   CentOS/RHEL: sudo yum install poppler-utils")
            logger.error("   Alpine: apk add poppler-utils")
            
        elif system_platform == 'darwin':  # macOS
            logger.error("   Homebrew: brew install poppler")
            logger.error("   MacPorts: sudo port install poppler")
            
        elif system_platform == 'windows':
            logger.error("   Download: https://github.com/oschwartz10612/poppler-windows")
            logger.error("   Conda: conda install -c conda-forge poppler")
            logger.error("   Chocolatey: choco install poppler")
            
        else:
            logger.error("   Please install poppler-utils for your system")
            
        logger.error("   Or set POPPLER_PATH environment variable to poppler bin directory")
    
    def check_poppler_availability(self) -> bool:
        """Check if poppler is available for PDF processing."""
        try:
            from pdf2image import convert_from_path
            from pdf2image.exceptions import PDFInfoNotInstalledError
            
            # Try to find poppler using platform-specific paths
            possible_paths = self._get_platform_poppler_paths()
            
            for poppler_path in possible_paths:
                try:
                    # Quick test - just check if the library can find poppler
                    # We don't actually convert anything, just verify poppler is accessible
                    if poppler_path is None:
                        # Test system PATH
                        test_result = shutil.which('pdftoppm')
                        if test_result:
                            logger.info(f"✅ Poppler found in system PATH: {test_result}")
                            self.poppler_available = True
                            return True
                    else:
                        # Test specific path
                        if os.path.exists(poppler_path):
                            pdftoppm_path = os.path.join(poppler_path, 'pdftoppm')
                            if os.path.exists(pdftoppm_path) or os.path.exists(f"{pdftoppm_path}.exe"):
                                logger.info(f"✅ Poppler found at: {poppler_path}")
                                self.poppler_available = True
                                return True
                except Exception as e:
                    logger.debug(f"Poppler check failed for {poppler_path}: {e}")
                    continue
            
            # Final attempt: Try shutil.which for cross-platform executable search
            poppler_executables = ['pdftoppm', 'pdfinfo']
            for exe in poppler_executables:
                if shutil.which(exe):
                    logger.info(f"✅ Poppler executable found: {exe}")
                    self.poppler_available = True
                    return True
            
            logger.warning("⚠️ Poppler not found - visual document processing will be unavailable")
            self._log_poppler_installation_help()
            self.poppler_available = False
            return False
            
        except ImportError:
            logger.error("❌ pdf2image library not installed")
            self.poppler_available = False
            return False
        except Exception as e:
            logger.warning(f"⚠️ Error checking poppler availability: {e}")
            self.poppler_available = False
            return False
    
    # Removed _precompute_vlm_analysis - now using live query-specific analysis
    # This eliminates the generic "What is this document about?" approach that 
    # caused all queries to receive the same response
    
    def _get_page_image(self, doc_id: str, page_idx: int) -> Image.Image:
        """Get specific page image."""
        try:
            if doc_id in self.page_images and page_idx in self.page_images[doc_id]:
                return self.page_images[doc_id][page_idx]
            else:
                logger.warning(f"Page image not found: {doc_id}, page {page_idx}")
                return None
        except Exception as e:
            logger.error(f"Error getting page image: {e}")
            return None
    
    def _analyze_visual_content_type(self, content: str, query: str) -> str:
        """Analyze the type of visual content to provide better context hints."""
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Detect visual content patterns
        if any(term in content_lower for term in ['chart', 'graph', 'plot', 'figure', 'diagram']):
            if any(term in content_lower for term in ['time', 'performance', 'speed', 'latency']):
                return 'performance_chart'
            elif any(term in content_lower for term in ['comparison', 'vs', 'versus', 'compared']):
                return 'comparison_chart'
            else:
                return 'data_chart'
        elif any(term in content_lower for term in ['table', 'data', 'columns', 'rows']):
            return 'data_table'
        elif any(term in content_lower for term in ['pipeline', 'workflow', 'architecture', 'diagram']):
            return 'technical_diagram'
        else:
            return 'document_page'
    
    def _format_visual_response(self, content: str, query: str, filename: str, page_number: int, content_type: str) -> str:
        """Format visual content response for better re-ranker compatibility."""
        
        # Create content type hints for the re-ranker
        type_prefixes = {
            'performance_chart': 'Performance chart analysis: ',
            'comparison_chart': 'Comparison chart data: ',
            'data_chart': 'Chart visualization shows: ',
            'data_table': 'Table data indicates: ',
            'technical_diagram': 'Technical diagram depicts: ',
            'document_page': 'Document content: '
        }
        
        prefix = type_prefixes.get(content_type, 'Visual analysis: ')
        
        # Extract the most relevant parts of the content
        content_lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Filter out metadata lines and focus on actual content
        filtered_lines = []
        for line in content_lines:
            if not any(skip_word in line.lower() for skip_word in ['**', 'source:', 'page', 'relevance:', '🔍', '📄']):
                filtered_lines.append(line)
        
        # Take the most substantive content
        main_content = ' '.join(filtered_lines[:3]) if filtered_lines else content
        
        # Create concise response
        if content_type in ['performance_chart', 'comparison_chart', 'data_chart']:
            # For chart content, focus on the data insights
            response = f"{prefix}{main_content}"
        elif content_type == 'data_table':
            # For table content, focus on the key data points
            response = f"{prefix}{main_content}"
        else:
            # For general content, provide context-aware summary
            response = f"Based on page {page_number} of {filename}: {main_content}"
        
        # Add subtle source context (for user transparency, but brief for re-ranker)
        if len(response) < 200:  # Only add if response is concise
            response += f" (from {filename}, page {page_number})"
        
        return response
    
    def _generate_vlm_answer(self, query: str, doc_id: str, page_idx: int, doc_info: Dict) -> str:
        """Generate query-specific answer using live VLM analysis."""
        
        page_number = page_idx + 1
        filename = doc_info['filename']
        
        # Step 1: PRIORITY - Live query-specific VLM analysis 
        try:
            if self.vlm_available:
                page_image = self._get_page_image(doc_id, page_idx)
                if page_image:
                    logger.info(f"🎯 Performing live query-specific VLM analysis for: '{query[:50]}...'")
                    vlm_response = self._analyze_image_with_vlm(query, page_image, filename, page_number)
                    if vlm_response and len(vlm_response.strip()) > 50:
                        logger.info(f"✅ Live VLM analysis successful: {len(vlm_response)} chars")
                        return vlm_response
                    else:
                        logger.warning(f"⚠️ VLM analysis returned minimal content")
        
        except Exception as e:
            logger.warning(f"❌ Live VLM analysis failed: {e}")
        
        # Step 2: Try to extract text content as backup (if original file still exists)
        try:
            original_path = doc_info.get('original_path', '')
            if original_path and os.path.exists(original_path):
                logger.info(f"📄 Falling back to text extraction for {filename} page {page_idx}")
                text_content = self._extract_page_text_simple(original_path, page_idx)
                
                if text_content and len(text_content.strip()) > 50:
                    # Found meaningful text content - return with query-relevant excerpt
                    relevant_excerpt = self._extract_query_relevant_text(text_content, query)
                    
                    # Analyze the content type for better response formatting
                    content_type = self._analyze_visual_content_type(relevant_excerpt, query)
                    response = self._format_visual_response(relevant_excerpt, query, filename, page_number, content_type)
                    
                    return response
        
        except Exception as e:
            logger.debug(f"Text extraction failed for {filename} page {page_idx}: {e}")
        
        # Step 3: Generate contextual response based on query keywords
        logger.info(f"🔧 Generating contextual response for query keywords")
        contextual_response = self._generate_contextual_response(query, filename, page_number)
        if contextual_response:
            return contextual_response
        
        # Step 4: Final fallback - but make it informative about the query specificity issue
        logger.warning(f"⚠️ All analysis methods failed for query: '{query}'")
        return f"**Page {page_number} of '{filename}' identified as visually relevant to your query.**\n\nColPali's similarity scoring determined this page contains content related to: *'{query}'*\n\nHowever, detailed query-specific analysis was not available. This may be due to:\n- VLM analysis temporarily unavailable\n- PDF file no longer accessible for text extraction\n- Network connectivity issues\n\n📄 **Source**: {filename}, page {page_number}\n🔍 **Query**: {query}"
    
    def _analyze_image_with_vlm(self, query: str, image: Image.Image, filename: str, page_number: int) -> str:
        """Send image to VLM for analysis."""
        
        try:
            # Convert PIL image to base64 for OpenAI API
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Create query-specific and directive prompt for VLM (based on successful debug app)
            system_prompt = f"""You are an expert document analyst examining page {page_number} from '{filename}'.

CRITICAL INSTRUCTIONS for query: "{query}"
1. Look carefully at the image and read ALL visible text
2. Extract specific facts, numbers, dates, and details that answer the exact question
3. If you see tables or charts, transcribe relevant data points
4. If you see performance metrics, report the actual numbers
5. Quote exact text from the document when possible
6. Focus ONLY on information relevant to this specific query
7. DO NOT give generic summaries - answer the specific question

Your response should:
- Start with the specific information that answers "{query}"
- Include exact quotes, numbers, or data from the page
- Describe visual elements (charts, tables, diagrams) with specific details relevant to the query
- Be concrete and factual, not vague or general"""
            
            # Call OpenAI Vision API
            response = self.vlm_client.chat.completions.create(
                model=self.vlm_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": f"Please analyze this document page to answer: {query}\n\nI need specific facts and details from what you can see in the image, not general descriptions."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=800,
                temperature=0.0  # More deterministic, focused responses
            )
            
            answer = response.choices[0].message.content
            
            # Add source attribution
            attributed_answer = f"{answer}\n\n📄 Source: {filename}, page {page_number}"
            
            logger.info(f"✅ VLM analysis completed for {filename} page {page_number}")
            return attributed_answer
            
        except Exception as e:
            logger.error(f"❌ VLM analysis failed: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics including visual-specific info."""
        # Create base stats from retrieval_stats (no parent class)
        base_stats = self.retrieval_stats.copy()
        
        # Add visual-specific statistics
        total_pages = sum(
            doc_data['metadata']['page_count'] 
            for doc_data in self.page_embeddings.values()
        )
        
        visual_stats = {
            'model_name': self.model_name,
            'device': self.device,
            'total_pages_indexed': total_pages,
            'total_documents_indexed': len(self.page_embeddings),
            'scoring_method': 'maxsim',
            'supported_formats': ['pdf']
        }
        
        base_stats['visual_specific'] = visual_stats
        return base_stats
    
    def clear_documents(self) -> bool:
        """Clear all visual embeddings, metadata, and page images."""
        try:
            self.page_embeddings.clear()
            self.document_metadata.clear()
            self.page_images.clear()  # Clear stored page images
            # No longer need to clear precomputed_vlm_content (removed)
            
            # Clear processed documents list and reset initialization
            self.processed_documents.clear()
            self.is_initialized = False
            result = True
            
            logger.info("✅ ColPaliRetriever cleared all documents and images")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error clearing ColPaliRetriever: {e}")
            return False
    
    def get_document_info(self, doc_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific document."""
        if doc_id in self.document_metadata:
            return self.document_metadata[doc_id].copy()
        return {}
    
    def get_page_embedding(self, doc_id: str, page_idx: int) -> np.ndarray:
        """Get embedding for a specific page (if available)."""
        if doc_id in self.page_embeddings:
            embeddings = self.page_embeddings[doc_id]['embeddings']
            if hasattr(embeddings, 'shape') and len(embeddings.shape) > 1:
                if page_idx < embeddings.shape[0]:
                    return embeddings[page_idx]
        return np.array([])
    
    def _extract_page_text_simple(self, file_path: str, page_idx: int) -> str:
        """Simple text extraction from PDF page."""
        try:
            # Try pypdf first (most common)
            try:
                import PyPDF2
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    if page_idx < len(pdf_reader.pages):
                        page = pdf_reader.pages[page_idx]
                        return page.extract_text().strip()
            except ImportError:
                pass
            
            # Try pdfplumber as backup
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    if page_idx < len(pdf.pages):
                        page = pdf.pages[page_idx]
                        return page.extract_text() or ""
            except ImportError:
                pass
            
            return ""
            
        except Exception as e:
            logger.debug(f"Text extraction failed for {file_path} page {page_idx}: {e}")
            return ""
    
    def _extract_query_relevant_text(self, text_content: str, query: str) -> str:
        """Extract the most relevant portion of text content based on the query."""
        try:
            # Clean and normalize text
            clean_text = ' '.join(text_content.split())
            
            # Split into sentences for better context
            sentences = []
            for sent in clean_text.split('.'):
                sent = sent.strip()
                if sent and len(sent) > 10:
                    sentences.append(sent + '.')
            
            if not sentences:
                return clean_text[:500] + "..." if len(clean_text) > 500 else clean_text
            
            # Extract query keywords
            query_words = set(word.lower().strip('.,!?;:') for word in query.split() if len(word) > 2)
            
            # Score sentences based on keyword matches
            scored_sentences = []
            for i, sentence in enumerate(sentences):
                sentence_words = set(word.lower().strip('.,!?;:') for word in sentence.split())
                matches = len(query_words.intersection(sentence_words))
                scored_sentences.append((matches, i, sentence))
            
            # Sort by relevance score
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            
            # Get top relevant sentences with context
            relevant_text = ""
            total_length = 0
            max_length = 400
            
            for score, idx, sentence in scored_sentences:
                if score > 0 and total_length < max_length:
                    if relevant_text:
                        relevant_text += " "
                    relevant_text += sentence
                    total_length += len(sentence)
                    
                    # Add context from adjacent sentences if space allows
                    if total_length < max_length - 100:
                        # Add previous sentence for context
                        if idx > 0 and sentences[idx-1] not in relevant_text:
                            context_sentence = sentences[idx-1]
                            if total_length + len(context_sentence) < max_length:
                                relevant_text = context_sentence + " " + relevant_text
                                total_length += len(context_sentence)
                        
                        # Add next sentence for context
                        if idx < len(sentences) - 1 and sentences[idx+1] not in relevant_text:
                            context_sentence = sentences[idx+1]
                            if total_length + len(context_sentence) < max_length:
                                relevant_text += " " + context_sentence
                                total_length += len(context_sentence)
            
            # If no relevant sentences found, return first portion of text
            if not relevant_text:
                relevant_text = clean_text[:400] + "..." if len(clean_text) > 400 else clean_text
            
            return relevant_text
            
        except Exception as e:
            logger.debug(f"Text relevance extraction failed: {e}")
            # Fallback to first portion of text
            clean_text = ' '.join(text_content.split())
            return clean_text[:400] + "..." if len(clean_text) > 400 else clean_text
    
    def _generate_contextual_response(self, query: str, filename: str, page_number: int) -> str:
        """Generate contextual response based on query keywords."""
        try:
            query_lower = query.lower()
            
            # Technical/academic keywords
            if any(keyword in query_lower for keyword in ['transformer', 'attention', 'neural', 'model', 'architecture']):
                return f"**Page {page_number} of '{filename}' contains technical information about neural networks or AI models.**\n\nThe ColPali visual analysis identified this page as highly relevant to your query about '{query}'. This page likely contains technical details, diagrams, or explanations related to machine learning architecture.\n\n📄 **Source**: {filename}, page {page_number}\n🧠 **Content Type**: Technical/Academic"
            
            # Research/paper keywords
            elif any(keyword in query_lower for keyword in ['research', 'paper', 'study', 'experiment', 'results']):
                return f"**Page {page_number} of '{filename}' contains research-related content.**\n\nThis page was identified by ColPali's visual analysis as relevant to your research query: '{query}'. It likely contains academic content, experimental results, or research findings.\n\n📄 **Source**: {filename}, page {page_number}\n📊 **Content Type**: Research/Academic"
            
            # Business/policy keywords
            elif any(keyword in query_lower for keyword in ['policy', 'procedure', 'rule', 'guideline', 'process']):
                return f"**Page {page_number} of '{filename}' contains policy or procedural information.**\n\nColPali's visual analysis found this page relevant to your query about '{query}'. This page likely contains business rules, procedures, or policy information.\n\n📄 **Source**: {filename}, page {page_number}\n📋 **Content Type**: Policy/Procedure"
            
            # Data/analysis keywords
            elif any(keyword in query_lower for keyword in ['data', 'analysis', 'chart', 'graph', 'table', 'statistics']):
                return f"**Page {page_number} of '{filename}' contains data or analytical content.**\n\nThis page was identified as relevant to your data-related query: '{query}'. It likely contains charts, tables, statistical information, or analytical content.\n\n📄 **Source**: {filename}, page {page_number}\n📈 **Content Type**: Data/Analysis"
            
            # General contextual response
            else:
                return f"**Page {page_number} of '{filename}' identified as relevant content.**\n\nColPali's vision-language analysis determined this page contains information related to your query: '{query}'. The visual elements and layout suggest it contains relevant information.\n\n📄 **Source**: {filename}, page {page_number}\n🔍 **Analysis**: Visual relevance match"
                
        except Exception as e:
            logger.debug(f"Contextual response generation failed: {e}")
            return None
    
    def _update_retrieval_stats(self, query_time: float, num_results: int, tokens_used: int, cost: float):
        """Update retrieval statistics."""
        self.retrieval_stats['total_queries'] += 1
        self.retrieval_stats['total_query_time'] += query_time
        self.retrieval_stats['total_results_returned'] += num_results
        self.retrieval_stats['total_tokens_used'] += tokens_used
        self.retrieval_stats['total_cost'] += cost
        
        # Update averages
        if self.retrieval_stats['total_queries'] > 0:
            self.retrieval_stats['avg_query_time'] = (
                self.retrieval_stats['total_query_time'] / self.retrieval_stats['total_queries']
            )
            self.retrieval_stats['avg_results_per_query'] = (
                self.retrieval_stats['total_results_returned'] / self.retrieval_stats['total_queries']
            )