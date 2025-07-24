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
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from retriever_base import BaseRetriever, RetrievalResult, RetrievalMetrics
from visual_document_processor import VisualDocumentProcessor
from embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)

class ColPaliRetriever(BaseRetriever):
    """
    ColPali-based visual document retriever.
    
    Uses vision-language models to understand document layout,
    visual elements, and textual content simultaneously.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
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
            
            self.visual_processor = VisualDocumentProcessor(visual_config)
            self.embedding_manager = EmbeddingManager.create_colpali(self.model_name)
            
            # Storage for page embeddings, metadata, and images
            self.page_embeddings = {}  # {doc_id: {'embeddings': tensor, 'metadata': dict}}
            self.document_metadata = {}  # {doc_id: doc_info}
            self.page_images = {}  # {doc_id: {page_idx: PIL.Image}}
            
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
                visual_result = self.embedding_manager.create_visual_embedding(doc_path)
                
                if visual_result['status'] == 'success':
                    # Create unique document ID
                    doc_id = f"{filename}_{hash(doc_path) % 10000}"
                    
                    # Store embeddings and metadata
                    self.page_embeddings[doc_id] = {
                        'embeddings': visual_result['embeddings'],
                        'metadata': visual_result['metadata']
                    }
                    
                    self.document_metadata[doc_id] = {
                        'filename': filename,
                        'original_path': doc_path,
                        'page_count': visual_result['metadata']['page_count'],
                        'doc_id': doc_id
                    }
                    
                    # Store page images for VLM analysis
                    self._store_page_images(doc_id, doc_path)
                    
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
        
        # Update initialization status
        self.is_initialized = len(results['successful']) > 0
        
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
            self._update_stats(query_time, len(top_results), estimated_tokens, cost_estimate)
            
            logger.info(f"✅ Retrieved {len(top_results)} visual results in {query_time:.3f}s")
            logger.info(f"🖼️ Avg score: {avg_score:.3f}, Max score: {max_score:.3f}")
            
            return top_results, metrics
            
        except Exception as e:
            logger.error(f"❌ Visual retrieval failed: {e}")
            query_time = time.time() - start_time
            return [], RetrievalMetrics(query_time, 0, 0, 0, 0, 0)
    
    def _store_page_images(self, doc_id: str, doc_path: str):
        """Store page images for VLM analysis."""
        try:
            from pdf2image import convert_from_path
            
            # Convert PDF pages to images with robust poppler handling
            logger.info(f"📸 Storing page images for {doc_id}")
            
            # Try different poppler paths and configurations
            images = self._convert_pdf_with_fallbacks(doc_path)
            
            if images:
                # Store images by page index
                self.page_images[doc_id] = {}
                for page_idx, image in enumerate(images):
                    self.page_images[doc_id][page_idx] = image
                
                logger.info(f"✅ Stored {len(images)} page images for {doc_id}")
            else:
                logger.warning(f"⚠️ No images could be extracted from {doc_id}")
                self.page_images[doc_id] = {}
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to store page images for {doc_id}: {e}")
            self.page_images[doc_id] = {}
    
    def _convert_pdf_with_fallbacks(self, pdf_path: str):
        """Convert PDF to images with multiple fallback strategies."""
        from pdf2image import convert_from_path
        from pdf2image.exceptions import PDFInfoNotInstalledError
        
        # Common Windows Poppler installation paths
        possible_poppler_paths = [
            None,  # Try system PATH first
            r"C:\Program Files\poppler\poppler-24.08.0\Library\bin",  # Your specific installation
            r"C:\Program Files\poppler\Library\bin",
            r"C:\Program Files\poppler\bin", 
            r"C:\Program Files (x86)\poppler\bin",
            r"C:\poppler\bin",
            r"C:\tools\poppler\bin",
            # Conda environments
            r"C:\Users\{}\miniconda3\Library\bin".format(os.getenv('USERNAME', '')),
            r"C:\Users\{}\anaconda3\Library\bin".format(os.getenv('USERNAME', '')),
        ]
        
        # Also check environment variables
        if 'POPPLER_PATH' in os.environ:
            possible_poppler_paths.insert(1, os.environ['POPPLER_PATH'])
        
        for poppler_path in possible_poppler_paths:
            try:
                if poppler_path and not os.path.exists(poppler_path):
                    continue
                    
                logger.info(f"Trying Poppler path: {poppler_path or 'system PATH'}")
                
                images = convert_from_path(
                    pdf_path, 
                    dpi=200,
                    poppler_path=poppler_path,
                    first_page=1,
                    last_page=5  # Limit to first 5 pages for memory efficiency
                )
                
                if images:
                    logger.info(f"✅ PDF conversion successful with path: {poppler_path or 'system PATH'}")
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
                            first_page=1,
                            last_page=5
                        )
                        
                        if images:
                            logger.info(f"✅ PDF conversion successful with found poppler: {poppler_dir}")
                            return images
                            
                    except Exception as e:
                        logger.debug(f"Conversion failed with found poppler {poppler_dir}: {e}")
                        continue
            
            # If still no success, provide helpful error message
            logger.error("❌ Poppler not found. Please install poppler-utils:")
            logger.error("   Windows: Download from https://github.com/oschwartz10612/poppler-windows")
            logger.error("   Or: conda install -c conda-forge poppler")
            logger.error("   Or: choco install poppler")
            
            return []
            
        except Exception as e:
            logger.error(f"Final poppler search failed: {e}")
            return []
    
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
    
    def _generate_vlm_answer(self, query: str, doc_id: str, page_idx: int, doc_info: Dict) -> str:
        """Generate answer using VLM to analyze the page image."""
        
        # Fallback description
        page_number = page_idx + 1
        filename = doc_info['filename']
        fallback = f"Found relevant visual content on page {page_number} of '{filename}'"
        
        if not self.vlm_available:
            return fallback + " (detailed visual analysis not available - OpenAI API key required)"
        
        try:
            # Get the page image
            page_image = self._get_page_image(doc_id, page_idx)
            if page_image is None:
                return fallback + " (page image not available)"
            
            # Analyze with VLM
            vlm_response = self._analyze_image_with_vlm(query, page_image, filename, page_number)
            
            if vlm_response:
                return vlm_response
            else:
                return fallback + " (visual analysis failed)"
                
        except Exception as e:
            logger.warning(f"VLM analysis failed: {e}")
            return fallback + f" (analysis error: {str(e)[:50]})"
    
    def _analyze_image_with_vlm(self, query: str, image: Image.Image, filename: str, page_number: int) -> str:
        """Send image to VLM for analysis."""
        
        try:
            # Convert PIL image to base64 for OpenAI API
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Create more specific and directive prompt for VLM
            system_prompt = f"""You are an expert document analyst examining page {page_number} from '{filename}'.

CRITICAL INSTRUCTIONS:
1. Look carefully at the image and read ALL visible text
2. Extract specific facts, numbers, dates, names, and details that answer the question
3. If you see tables, transcribe relevant data
4. If you see charts/graphs, describe the specific data points
5. Quote exact text from the document when possible
6. DO NOT give generic responses - be specific about what you observe

User Question: "{query}"

Your response should:
- Start with the specific information that answers the question
- Include exact quotes, numbers, or data from the page
- Describe visual elements (charts, tables, diagrams) with specific details
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
                                "text": f"Examine this document page carefully. Read all text, analyze all visual elements, and provide specific details to answer: {query}\n\nI need concrete facts, not general descriptions. Extract exact information from what you can see."
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
        base_stats = super().get_stats()
        
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
            
            # Clear base class state
            result = super().clear_documents()
            
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