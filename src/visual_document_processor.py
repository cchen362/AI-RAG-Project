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
from PIL import Image, ImageStat, ImageFilter
import cv2

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
        
        # Dynamic batching configuration for memory optimization
        self.max_batch_size = 4 if self.device == "cuda" else 2
        self.min_batch_size = 1
        self.current_batch_size = self.max_batch_size
        
        logger.info(f"🔧 VisualDocumentProcessor initialized for {self.model_name}")
        logger.info(f"🖥️ Device: {self.device}")
        logger.info(f"🔄 Dynamic batching: max={self.max_batch_size}, current={self.current_batch_size}")
    
    def _detect_device(self) -> str:
        """Detect optimal device (GPU/CPU) for processing."""
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🚀 GPU detected: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
            
            # Enhanced GPU memory optimization for 6GB RTX 1060 constraints
            memory_config = 'expandable_segments:True,max_split_size_mb:256,roundup_power2_divisions:8'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = memory_config
            
            # Additional memory optimization environment variables
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async operations
            os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # CuDNN optimizations
            
            logger.info("🔧 Enhanced GPU memory optimization configured for RTX 1060 6GB")
            logger.info(f"   - Memory allocation: {memory_config}")
            
            # Set memory management parameters
            self.gpu_memory_limit = gpu_memory * 0.85  # Use 85% of available VRAM
            self.memory_cleanup_threshold = gpu_memory * 0.75  # Cleanup at 75%
            
            logger.info(f"   - Memory limit: {self.gpu_memory_limit:.2f}GB")
            logger.info(f"   - Cleanup threshold: {self.memory_cleanup_threshold:.2f}GB")
            
        else:
            device = "cpu"
            logger.info("💻 Using CPU (GPU not available)")
        return device
    
    def _get_current_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated(0) / 1024**3
        return 0.0
    
    def _cleanup_gpu_memory(self, force: bool = False):
        """Clean up GPU memory when needed."""
        if self.device != "cuda":
            return
            
        current_memory = self._get_current_memory_usage()
        
        if force or current_memory > self.memory_cleanup_threshold:
            logger.info(f"🧹 GPU memory cleanup triggered: {current_memory:.2f}GB")
            torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            new_memory = self._get_current_memory_usage()
            freed = current_memory - new_memory
            logger.info(f"   - Freed: {freed:.2f}GB, Current: {new_memory:.2f}GB")
    
    def _check_memory_available(self, required_mb: float = 530) -> bool:
        """Check if enough GPU memory is available for operation."""
        if self.device != "cuda":
            return True
            
        current_memory_gb = self._get_current_memory_usage()
        required_gb = required_mb / 1024
        available_gb = self.gpu_memory_limit - current_memory_gb
        
        logger.info(f"🔍 Memory check: Current={current_memory_gb:.2f}GB, Required={required_gb:.2f}GB, Available={available_gb:.2f}GB")
        
        return available_gb >= required_gb
    
    def _adjust_batch_size(self, memory_used_gb: float):
        """Dynamically adjust batch size based on memory usage."""
        if self.device != "cuda":
            return
            
        # Calculate optimal batch size based on available memory
        available_memory = self.gpu_memory_limit - memory_used_gb
        
        if available_memory > 1.5:  # Plenty of memory
            new_batch_size = self.max_batch_size
        elif available_memory > 1.0:  # Moderate memory
            new_batch_size = max(2, self.max_batch_size // 2)
        elif available_memory > 0.5:  # Limited memory
            new_batch_size = 1
        else:  # Critical memory
            logger.warning("⚠️ Critical memory situation, consider CPU fallback")
            new_batch_size = 1
            
        if new_batch_size != self.current_batch_size:
            logger.info(f"🔄 Adjusting batch size: {self.current_batch_size} → {new_batch_size}")
            self.current_batch_size = new_batch_size
    
    def analyze_visual_complexity(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze visual complexity of a page to determine processing needs.
        Returns analysis with visual_score and processing recommendations.
        """
        try:
            # Convert PIL to numpy array for analysis
            img_array = np.array(image)
            
            # Convert to grayscale for analysis
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # 1. Color variance analysis (high variance = complex visuals)
            variance_score = np.var(gray) / 10000.0  # Normalize to 0-1 range
            
            # 2. Edge density analysis (charts/diagrams have many edges)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # 3. Contour analysis (detect non-text elements)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter out small contours (likely text)
            significant_contours = [c for c in contours if cv2.contourArea(c) > 500]
            contour_score = len(significant_contours) / 100.0  # Normalize
            
            # 4. Layout analysis using image statistics
            # High standard deviation in pixel values suggests varied content
            img_stats = ImageStat.Stat(image)
            if hasattr(img_stats, 'stddev'):
                layout_complexity = np.mean(img_stats.stddev) / 128.0
            else:
                layout_complexity = 0.3  # Default moderate complexity
            
            # Combine scores (weighted)
            visual_score = (
                variance_score * 0.3 +
                edge_density * 0.3 +
                contour_score * 0.2 +
                layout_complexity * 0.2
            )
            
            # Normalize to 0-1 range
            visual_score = min(1.0, max(0.0, visual_score))
            
            # Classification
            if visual_score > 0.6:
                complexity_level = "high"
                recommendation = "visual_processing"
            elif visual_score > 0.3:
                complexity_level = "medium" 
                recommendation = "hybrid_processing"
            else:
                complexity_level = "low"
                recommendation = "text_processing"
            
            return {
                'visual_score': visual_score,
                'complexity_level': complexity_level,
                'recommendation': recommendation,
                'metrics': {
                    'variance_score': variance_score,
                    'edge_density': edge_density,
                    'contour_count': len(significant_contours),
                    'layout_complexity': layout_complexity
                }
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Visual complexity analysis failed: {e}")
            # Return safe defaults
            return {
                'visual_score': 0.5,
                'complexity_level': "medium",
                'recommendation': "hybrid_processing",
                'metrics': {}
            }
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine if visual processing is beneficial."""
        visual_keywords = {
            'chart': 0.8, 'graph': 0.8, 'diagram': 0.9, 'figure': 0.7, 'table': 0.8,
            'plot': 0.8, 'visualization': 0.9, 'architecture': 0.7, 'flowchart': 0.9,
            'schema': 0.8, 'performance': 0.6, 'comparison': 0.5, 'results': 0.4,
            'data': 0.3, 'analysis': 0.3, 'layout': 0.7, 'design': 0.6, 'structure': 0.5
        }
        
        query_lower = query.lower()
        max_visual_intent = 0.0
        matched_keywords = []
        
        for keyword, weight in visual_keywords.items():
            if keyword in query_lower:
                max_visual_intent = max(max_visual_intent, weight)
                matched_keywords.append(keyword)
        
        # Determine intent level
        if max_visual_intent > 0.7:
            intent_level = "high"
            recommendation = "prioritize_visual"
        elif max_visual_intent > 0.4:
            intent_level = "medium"
            recommendation = "hybrid_processing"
        else:
            intent_level = "low" 
            recommendation = "text_preferred"
        
        return {
            'visual_intent_score': max_visual_intent,
            'intent_level': intent_level,
            'recommendation': recommendation,
            'matched_keywords': matched_keywords
        }
    
    def process_file_adaptive(self, file_path: str, query: str = None) -> Dict[str, Any]:
        """
        Adaptive processing that combines visual complexity analysis with memory optimization.
        
        Args:
            file_path: Path to PDF file
            query: Optional query to guide processing decisions
            
        Returns:
            Dict containing processing results with adaptive decisions
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
            logger.info(f"🎯 Adaptive processing: {os.path.basename(file_path)}")
            start_time = time.time()
            
            # Analyze query intent if provided
            query_analysis = self.analyze_query_intent(query) if query else {
                'visual_intent_score': 0.5,
                'intent_level': 'medium',
                'recommendation': 'hybrid_processing',
                'matched_keywords': []
            }
            
            logger.info(f"🔍 Query analysis: {query_analysis['intent_level']} intent ({query_analysis['visual_intent_score']:.2f})")
            
            # Convert PDF to images for analysis
            images = self._convert_pdf_to_images(file_path)
            if not images:
                return {
                    'status': 'error',
                    'error': 'Failed to convert PDF to images'
                }
            
            # Analyze each page for visual complexity
            page_analyses = []
            total_visual_score = 0
            
            for i, image in enumerate(images):
                page_analysis = self.analyze_visual_complexity(image)
                page_analyses.append(page_analysis)
                total_visual_score += page_analysis['visual_score']
                
                logger.info(f"   Page {i+1}: {page_analysis['complexity_level']} complexity ({page_analysis['visual_score']:.2f})")
            
            avg_visual_score = total_visual_score / len(images)
            
            # Make adaptive processing decision
            processing_decision = self._make_processing_decision(
                query_analysis, page_analyses, avg_visual_score
            )
            
            logger.info(f"🎯 Processing decision: {processing_decision['strategy']} - {processing_decision['reason']}")
            
            # Execute processing based on decision
            if processing_decision['strategy'] == 'visual_priority':
                result = self._process_visual_priority(images, file_path, processing_decision)
            elif processing_decision['strategy'] == 'hybrid':
                result = self._process_hybrid(images, file_path, processing_decision)
            elif processing_decision['strategy'] == 'text_fallback':
                result = self._process_text_fallback(images, file_path, processing_decision)
            else:
                # Default to memory-aware visual processing
                result = self._process_memory_aware_visual(images, file_path, processing_decision)
            
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            result['adaptive_analysis'] = {
                'query_analysis': query_analysis,
                'page_analyses': page_analyses,
                'avg_visual_score': avg_visual_score,
                'processing_decision': processing_decision
            }
            
            logger.info(f"✅ Adaptive processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Adaptive processing failed: {e}")
            # Fallback to original processing method
            return self.process_file(file_path)
    
    def _make_processing_decision(self, query_analysis: Dict, page_analyses: List[Dict], avg_visual_score: float) -> Dict[str, Any]:
        """Make intelligent processing decision based on analysis."""
        
        # Check memory availability
        memory_available = self._check_memory_available(required_mb=530)
        current_memory = self._get_current_memory_usage()
        
        # Count high-complexity pages
        high_complexity_pages = sum(1 for p in page_analyses if p['visual_score'] > 0.6)
        medium_complexity_pages = sum(1 for p in page_analyses if 0.3 <= p['visual_score'] <= 0.6)
        
        # Decision logic
        if not memory_available:
            return {
                'strategy': 'text_fallback',
                'reason': f'Insufficient GPU memory ({current_memory:.2f}GB used)',
                'pages_to_process': []
            }
        
        if query_analysis['visual_intent_score'] > 0.7:
            # High query intent for visual - try to process all pages
            return {
                'strategy': 'visual_priority',
                'reason': f'High visual query intent ({query_analysis["visual_intent_score"]:.2f})',
                'pages_to_process': list(range(len(page_analyses)))
            }
        
        if avg_visual_score > 0.5 and high_complexity_pages > len(page_analyses) * 0.3:
            # Many visually complex pages
            visual_pages = [i for i, p in enumerate(page_analyses) if p['visual_score'] > 0.4]
            return {
                'strategy': 'hybrid',
                'reason': f'Mixed content ({high_complexity_pages} high-complexity pages)',
                'pages_to_process': visual_pages
            }
        
        if avg_visual_score < 0.3:
            # Mostly text content
            return {
                'strategy': 'text_fallback',
                'reason': f'Low visual complexity ({avg_visual_score:.2f})',
                'pages_to_process': []
            }
        
        # Default: Memory-aware visual processing
        return {
            'strategy': 'memory_aware_visual',
            'reason': f'Balanced approach (avg score: {avg_visual_score:.2f})',
            'pages_to_process': list(range(len(page_analyses)))
        }
    
    def _process_visual_priority(self, images: List[Image.Image], file_path: str, decision: Dict) -> Dict[str, Any]:
        """Process with visual priority - attempt to process all pages with ColPali."""
        logger.info("🎯 Processing with visual priority strategy")
        
        try:
            self._load_model()
            
            # Process with aggressive memory management
            embeddings = self._generate_embeddings_memory_aware(images)
            
            return {
                'status': 'success',
                'file_path': file_path,
                'embeddings': embeddings,
                'strategy': 'visual_priority',
                'metadata': {
                    'page_count': len(images),
                    'model_name': self.model_name,
                    'device': self.device,
                    'batch_size': self.current_batch_size
                }
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Visual priority failed: {e}, falling back to hybrid")
            return self._process_hybrid(images, file_path, decision)
    
    def _process_hybrid(self, images: List[Image.Image], file_path: str, decision: Dict) -> Dict[str, Any]:
        """Process using hybrid approach - visual for complex pages, text for simple ones."""
        logger.info("🎯 Processing with hybrid strategy")
        
        try:
            pages_to_process = decision.get('pages_to_process', [])
            
            if not pages_to_process:
                # No visual pages identified, fall back to text
                return self._process_text_fallback(images, file_path, decision)
            
            self._load_model()
            
            # Process only selected pages with visual analysis
            selected_images = [images[i] for i in pages_to_process]
            embeddings = self._generate_embeddings_memory_aware(selected_images)
            
            # For simplicity, map embeddings back to full page indices
            # In a more sophisticated implementation, you'd handle text processing for other pages
            full_embeddings = embeddings  # Simplified for now
            
            return {
                'status': 'success', 
                'file_path': file_path,
                'embeddings': full_embeddings,
                'strategy': 'hybrid',
                'metadata': {
                    'page_count': len(images),
                    'visual_pages': pages_to_process,
                    'model_name': self.model_name,
                    'device': self.device,
                    'batch_size': self.current_batch_size
                }
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Hybrid processing failed: {e}, falling back to text")
            return self._process_text_fallback(images, file_path, decision)
    
    def _process_text_fallback(self, images: List[Image.Image], file_path: str, decision: Dict) -> Dict[str, Any]:
        """Process using text extraction fallback when visual processing not suitable."""
        logger.info("🎯 Processing with text fallback strategy")
        
        try:
            # For text fallback, return empty embeddings but provide metadata
            # The higher-level system will handle text-only processing
            return {
                'status': 'success',
                'file_path': file_path,
                'embeddings': None,  # No visual embeddings
                'strategy': 'text_fallback',
                'metadata': {
                    'page_count': len(images),
                    'reason': decision.get('reason', 'Low visual complexity'),
                    'recommendation': 'use_text_rag'
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Text fallback failed: {e}")
            return {
                'status': 'error',
                'error': f'All processing strategies failed: {e}',
                'file_path': file_path
            }
    
    def _process_memory_aware_visual(self, images: List[Image.Image], file_path: str, decision: Dict) -> Dict[str, Any]:
        """Process with memory-aware visual processing."""
        logger.info("🎯 Processing with memory-aware visual strategy")
        
        try:
            self._load_model()
            
            # Use the enhanced memory-aware embedding generation
            embeddings = self._generate_embeddings_memory_aware(images)
            
            return {
                'status': 'success',
                'file_path': file_path,
                'embeddings': embeddings,
                'strategy': 'memory_aware_visual',
                'metadata': {
                    'page_count': len(images),
                    'model_name': self.model_name,
                    'device': self.device,
                    'batch_size': self.current_batch_size
                }
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Memory-aware visual failed: {e}, falling back to text")
            return self._process_text_fallback(images, file_path, decision)
    
    def _generate_embeddings_memory_aware(self, images: List[Image.Image]) -> torch.Tensor:
        """Generate embeddings with aggressive memory management."""
        logger.info(f"🧠 Generating embeddings for {len(images)} pages (batch_size={self.current_batch_size})")
        
        all_embeddings = []
        
        try:
            # Process images in batches with memory monitoring
            for i in range(0, len(images), self.current_batch_size):
                batch_end = min(i + self.current_batch_size, len(images))
                batch_images = images[i:batch_end]
                
                logger.info(f"   Processing batch {i//self.current_batch_size + 1}: pages {i+1}-{batch_end}")
                
                # Check memory before processing
                current_memory = self._get_current_memory_usage()
                
                if current_memory > self.memory_cleanup_threshold:
                    logger.info(f"🧹 Memory cleanup before batch: {current_memory:.2f}GB")
                    self._cleanup_gpu_memory(force=True)
                    
                    # Adjust batch size if still high memory usage
                    if current_memory > self.gpu_memory_limit * 0.8:
                        self._adjust_batch_size(current_memory)
                        if self.current_batch_size < len(batch_images):
                            # Re-process with smaller batch
                            batch_images = batch_images[:self.current_batch_size]
                            batch_end = i + self.current_batch_size
                
                # Generate embeddings for batch with mixed precision
                with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                    batch_embeddings = self._generate_batch_embeddings(batch_images)
                    
                all_embeddings.append(batch_embeddings)
                
                # Cleanup after each batch
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                logger.info(f"   ✅ Batch completed, memory: {self._get_current_memory_usage():.2f}GB")
            
            # Concatenate all embeddings
            if all_embeddings:
                final_embeddings = torch.cat(all_embeddings, dim=0)
                logger.info(f"✅ Generated embeddings: {final_embeddings.shape}")
                return final_embeddings
            else:
                logger.error("❌ No embeddings generated")
                return torch.empty(0)
                
        except Exception as e:
            logger.error(f"❌ Memory-aware embedding generation failed: {e}")
            raise
    
    def _generate_batch_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """Generate embeddings for a batch of images."""
        try:
            # Process images with the model processor
            inputs = self.processor.process_images(images).to(self.device)
            
            # Generate embeddings with memory-efficient forward pass
            with torch.no_grad():
                embeddings = self.model(**inputs)
                
            # Convert to appropriate format and move to CPU to save GPU memory
            if isinstance(embeddings, dict) and 'last_hidden_state' in embeddings:
                embeddings = embeddings['last_hidden_state']
            
            # Keep on GPU for now, will move to CPU later if needed
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Batch embedding generation failed: {e}")
            raise
    
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
                    # Enhanced memory-aware model loading for RTX 1060 6GB
                    self._cleanup_gpu_memory(force=True)  # Aggressive cleanup before loading
                    
                    # Optimal settings for 6GB GPU
                    torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
                    attn_implementation = "flash_attention_2" if (self.device == "cuda" and is_flash_attn_2_available()) else None
                    
                    # Memory monitoring
                    if self.device == "cuda":
                        initial_memory = self._get_current_memory_usage()
                        logger.info(f"🧹 GPU memory before loading: {initial_memory:.2f}GB")
                    
                    logger.info(f"🔧 Memory-optimized loading: dtype={torch_dtype}, flash_attn={attn_implementation is not None}")
                    
                    # Check if we have enough memory for model loading
                    if self.device == "cuda" and not self._check_memory_available(required_mb=2048):  # Model needs ~2GB
                        logger.warning("⚠️ Insufficient memory for GPU loading, falling back to CPU")
                        self.device = "cpu"
                        torch_dtype = torch.float32
                        attn_implementation = None
                    
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
                    
                    # GPU memory status after loading
                    if self.device == "cuda":
                        final_memory = torch.cuda.memory_allocated(0) / 1024**3
                        model_memory = final_memory - initial_memory
                        logger.info(f"🎮 Model loaded: {model_memory:.2f}GB VRAM used")
                    
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
                
                # GPU memory cleanup before fallback loading
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    initial_memory = torch.cuda.memory_allocated(0) / 1024**3
                    logger.info(f"🧹 GPU memory before fallback loading: {initial_memory:.2f}GB")
                
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
                
                # GPU memory status after fallback loading
                if self.device == "cuda":
                    final_memory = torch.cuda.memory_allocated(0) / 1024**3
                    model_memory = final_memory - initial_memory
                    logger.info(f"🎮 Fallback model loaded: {model_memory:.2f}GB VRAM used")
            
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
            # Validate inputs
            if not images:
                logger.error("❌ No images provided for embedding generation")
                raise ValueError("No images to process")
            
            logger.info(f"🔄 Generating embeddings for {len(images)} images")
            
            # Prepare images for processing - ensure they're valid PIL images
            batch_images = []
            for i, img in enumerate(images):
                if img is None:
                    logger.warning(f"⚠️ Skipping None image at index {i}")
                    continue
                batch_images.append(img)
            
            if not batch_images:
                logger.error("❌ No valid images found after filtering")
                raise ValueError("No valid images to process")
            
            logger.info(f"✅ Prepared {len(batch_images)} valid images for processing")
            
            # Process images through ColPali with memory-optimized batch processing
            with torch.no_grad():
                embeddings = None
                
                # GPU memory cleanup before processing
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    pre_process_memory = torch.cuda.memory_allocated(0) / 1024**3
                    available_memory = (torch.cuda.get_device_properties(0).total_memory / 1024**3) - pre_process_memory
                    logger.info(f"🧹 GPU memory before processing: {pre_process_memory:.2f}GB, available: {available_memory:.2f}GB")
                    
                    # Dynamic batch sizing based on available memory
                    if available_memory < 1.0:  # Less than 1GB available
                        max_batch_size = 1  # Process one page at a time
                        logger.info("⚠️ Low memory detected - using single page processing")
                    elif available_memory < 2.0:  # Less than 2GB available  
                        max_batch_size = 2  # Process 2 pages at a time
                        logger.info("🔧 Medium memory - processing 2 pages per batch")
                    else:
                        max_batch_size = 3  # Process 3 pages at a time (conservative)
                        logger.info("🚀 Good memory - processing 3 pages per batch")
                else:
                    max_batch_size = len(batch_images)  # CPU can handle full batch
                
                # Process images in memory-optimized batches
                all_embeddings = []
                
                for i in range(0, len(batch_images), max_batch_size):
                    batch_subset = batch_images[i:i + max_batch_size]
                    logger.info(f"🔄 Processing batch {i//max_batch_size + 1}/{(len(batch_images)-1)//max_batch_size + 1} ({len(batch_subset)} pages)")
                    
                    # GPU memory cleanup before each batch
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    
                    # Try standard processing first
                    try:
                        batch_inputs = self.processor.process_images(batch_subset)
                        
                        # Validate batch_inputs
                        if batch_inputs is None:
                            raise ValueError("Processor returned None for batch_inputs")
                        
                        # BatchFeature is the correct return type from ColQwen2 processor
                        if not hasattr(batch_inputs, 'keys'):
                            raise ValueError(f"Expected BatchFeature or dict from processor, got {type(batch_inputs)}")
                        
                        logger.info(f"✅ Processor returned: {list(batch_inputs.keys())}")
                        
                        # Move to device
                        for key in batch_inputs:
                            if isinstance(batch_inputs[key], torch.Tensor):
                                batch_inputs[key] = batch_inputs[key].to(self.device)
                        
                        # Generate embeddings with ColQwen2 model
                        outputs = self.model(**batch_inputs)
                        
                        # Handle ColQwen2 output format
                        if hasattr(outputs, 'last_hidden_state'):
                            batch_embeddings = outputs.last_hidden_state
                        elif isinstance(outputs, tuple):
                            batch_embeddings = outputs[0]
                        else:
                            batch_embeddings = outputs
                        
                        # Store batch embeddings
                        all_embeddings.append(batch_embeddings)
                        
                        # Memory cleanup after batch processing
                        del batch_inputs, outputs
                        if self.device == "cuda":
                            torch.cuda.empty_cache()
                            current_memory = torch.cuda.memory_allocated(0) / 1024**3
                            logger.info(f"🧹 Memory after batch {i//max_batch_size + 1}: {current_memory:.2f}GB")
                        
                        logger.info(f"✅ Batch {i//max_batch_size + 1} processing succeeded")
                    
                    except Exception as proc_err:
                        logger.warning(f"⚠️ Batch {i//max_batch_size + 1} failed: {proc_err}, trying single image fallback")
                        
                        # Fallback to single image processing for this batch
                        batch_embeddings_list = []
                        for img_idx, img in enumerate(batch_subset):
                            try:
                                # GPU memory cleanup before each image
                                if self.device == "cuda":
                                    torch.cuda.empty_cache()
                                
                                single_input = self.processor.process_images([img])
                                if single_input is not None and hasattr(single_input, 'keys'):
                                    for key in single_input:
                                        if isinstance(single_input[key], torch.Tensor):
                                            single_input[key] = single_input[key].to(self.device)
                                    
                                    single_output = self.model(**single_input)
                                    if hasattr(single_output, 'last_hidden_state'):
                                        batch_embeddings_list.append(single_output.last_hidden_state)
                                    else:
                                        batch_embeddings_list.append(single_output)
                                    
                                    # Clean up after each image
                                    del single_input, single_output
                                    if self.device == "cuda":
                                        torch.cuda.empty_cache()
                                
                            except Exception as img_err:
                                logger.error(f"❌ Image {img_idx} in batch {i//max_batch_size + 1} failed: {img_err}")
                                continue
                        
                        if batch_embeddings_list:
                            batch_embeddings = torch.cat(batch_embeddings_list, dim=0)
                            all_embeddings.append(batch_embeddings)
                            logger.info(f"✅ Single image fallback succeeded for batch {i//max_batch_size + 1}")
                        else:
                            logger.error(f"❌ All images failed in batch {i//max_batch_size + 1}")
                            raise RuntimeError(f"Batch {i//max_batch_size + 1} completely failed")
                
                # Combine all batch embeddings
                if all_embeddings:
                    embeddings = torch.cat(all_embeddings, dim=0)
                    logger.info(f"✅ Combined {len(all_embeddings)} batches into final embeddings")
                else:
                    raise ValueError("No embeddings generated from any batch")
                
                # Validate final embeddings
                if embeddings is None:
                    raise ValueError("All processing attempts returned None embeddings")
                
                if not isinstance(embeddings, torch.Tensor):
                    raise ValueError(f"Expected torch.Tensor, got {type(embeddings)}")
                
                logger.info(f"✅ Generated embeddings shape: {embeddings.shape}")
                logger.info(f"✅ Embeddings dtype: {embeddings.dtype}, device: {embeddings.device}")
                
                # Final GPU memory cleanup after processing
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    final_memory = torch.cuda.memory_allocated(0) / 1024**3
                    logger.info(f"🧹 GPU memory after processing: {final_memory:.2f}GB")
                
                return embeddings
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # CRITICAL: Do not use dummy embeddings - they cause zero similarity scores
            # Instead, raise the exception to identify and fix the root cause
            raise RuntimeError(f"ColPali embedding generation failed: {e}. Check logs for details.")
    
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
            
            # Process query text with ColQwen2 API and memory optimization
            with torch.no_grad():
                query_embedding = None
                
                # GPU memory cleanup before query processing
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    pre_query_memory = torch.cuda.memory_allocated(0) / 1024**3
                    logger.info(f"🧹 GPU memory before query: {pre_query_memory:.2f}GB")
                
                try:
                    # Use ColQwen2 specific query processing API
                    logger.info("🔧 Processing query with ColQwen2 process_queries API...")
                    query_inputs = self.processor.process_queries([query])
                    
                    # Validate query_inputs
                    if query_inputs is None:
                        raise ValueError("Processor returned None for query_inputs")
                    
                    # BatchFeature is the correct return type from ColQwen2 processor
                    if not hasattr(query_inputs, 'keys'):
                        raise ValueError(f"Expected BatchFeature or dict from processor, got {type(query_inputs)}")
                    
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
                    
                    # Clean up query processing tensors
                    del query_inputs, query_outputs
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    
                    logger.info("✅ Standard query processing succeeded")
                        
                except Exception as query_err:
                    logger.warning(f"⚠️ Query processing error: {query_err}, using fallback")
                    # Fallback query processing
                    try:
                        logger.info("🔧 Processing query with ColQwen2 process_queries fallback...")
                        query_inputs = self.processor.process_queries([query])
                        
                        # Validate fallback query_inputs
                        if query_inputs is None:
                            raise ValueError("Fallback processor also returned None")
                        
                        # BatchFeature is the correct return type from ColQwen2 processor
                        if not hasattr(query_inputs, 'keys'):
                            raise ValueError(f"Fallback processor returned {type(query_inputs)}, expected BatchFeature or dict")
                        
                        # Safely move to device
                        for key in query_inputs:
                            if isinstance(query_inputs[key], torch.Tensor):
                                query_inputs[key] = query_inputs[key].to(self.device)
                        
                        query_outputs = self.model(**query_inputs)
                        if hasattr(query_outputs, 'last_hidden_state'):
                            query_embedding = query_outputs.last_hidden_state
                        else:
                            query_embedding = query_outputs
                        
                        logger.info("✅ Fallback query processing succeeded")
                            
                    except Exception as fallback_err:
                        logger.error(f"❌ Fallback query processing failed: {fallback_err}")
                        raise
                
                # Validate query embedding
                if query_embedding is None:
                    raise ValueError("All query processing attempts returned None")
                
                # Move document embeddings to same device
                if isinstance(document_embeddings, torch.Tensor):
                    document_embeddings = document_embeddings.to(self.device)
                
                # Calculate MaxSim scores
                scores = self._calculate_maxsim_scores(query_embedding, document_embeddings)
                
                # GPU memory cleanup after query processing
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    post_query_memory = torch.cuda.memory_allocated(0) / 1024**3
                    logger.info(f"🧹 GPU memory after query: {post_query_memory:.2f}GB")
            
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