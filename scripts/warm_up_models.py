#!/usr/bin/env python3
"""
Model Warm-up Script for Docker Pre-loading
Pre-loads all AI models during Docker build time for instant startup
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment for model loading"""
    # Add src directory to path
    current_dir = Path(__file__).parent.parent
    src_dir = current_dir / 'src'
    sys.path.insert(0, str(src_dir))
    
    # Set model cache directories
    model_cache_dir = current_dir / 'models'
    model_cache_dir.mkdir(exist_ok=True)
    
    # Set environment variables for model caching
    os.environ['TRANSFORMERS_CACHE'] = str(model_cache_dir / 'transformers')
    os.environ['HF_HOME'] = str(model_cache_dir / 'huggingface') 
    os.environ['TORCH_HOME'] = str(model_cache_dir / 'torch')
    
    # Detect and configure device
    model_device = os.getenv('MODEL_DEVICE', 'auto')
    if model_device == 'auto':
        try:
            import torch
            model_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            os.environ['MODEL_DEVICE'] = model_device
        except ImportError:
            model_device = 'cpu'
            os.environ['MODEL_DEVICE'] = model_device
    
    logger.info(f"Model cache directory: {model_cache_dir}")
    logger.info(f"Target device: {model_device}")
    
    # GPU-specific optimizations
    if model_device == 'cuda':
        try:
            import torch
            if torch.cuda.is_available():
                # Optimize GPU memory usage
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA version: {torch.version.cuda}")
            else:
                logger.warning("CUDA device requested but not available, falling back to CPU")
                model_device = 'cpu'
                os.environ['MODEL_DEVICE'] = model_device
        except Exception as e:
            logger.warning(f"GPU setup failed: {e}, falling back to CPU")
            model_device = 'cpu'
            os.environ['MODEL_DEVICE'] = model_device
    
    return model_cache_dir, model_device

def warm_up_sentence_transformer():
    """Pre-load SentenceTransformer model"""
    logger.info("🔄 Warming up SentenceTransformer (all-MiniLM-L6-v2)...")
    start_time = time.time()
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load the model - this will download and cache it
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test inference to ensure model works
        test_text = ["This is a test sentence for warm-up"]
        embeddings = model.encode(test_text)
        
        load_time = time.time() - start_time
        logger.info(f"✅ SentenceTransformer loaded successfully ({load_time:.2f}s)")
        logger.info(f"   Model dimension: {embeddings.shape[1]}")
        
        return True, load_time
        
    except Exception as e:
        logger.error(f"❌ SentenceTransformer warm-up failed: {e}")
        return False, 0

def warm_up_bge_reranker():
    """Pre-load BGE Re-ranker model"""
    logger.info("🔄 Warming up BGE Re-ranker (BAAI/bge-reranker-base)...")
    start_time = time.time()
    
    try:
        from sentence_transformers import CrossEncoder
        
        # Load the cross-encoder model
        model = CrossEncoder('BAAI/bge-reranker-base')
        
        # Test inference
        test_pairs = [("What is RAG?", "RAG stands for Retrieval Augmented Generation")]
        scores = model.predict(test_pairs)
        
        load_time = time.time() - start_time
        logger.info(f"✅ BGE Re-ranker loaded successfully ({load_time:.2f}s)")
        logger.info(f"   Test score: {scores[0]:.4f}")
        
        # GPU memory cleanup if applicable
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        return True, load_time
        
    except Exception as e:
        logger.error(f"❌ BGE Re-ranker warm-up failed: {e}")
        return False, 0

def warm_up_colpali(target_device='auto'):
    """Pre-load ColPali model with device-specific optimization"""
    logger.info("🔄 Warming up ColPali (vidore/colqwen2-v1.0)...")
    start_time = time.time()
    
    try:
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        import torch
        
        # Determine device
        if target_device == 'auto':
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = target_device
            
        logger.info(f"   Using device: {device}")
        
        # Load model with device-specific optimizations
        if device == "cuda":
            try:
                # GPU-optimized loading
                model = ColQwen2.from_pretrained(
                    'vidore/colqwen2-v1.0',
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
                
                # Test GPU memory allocation
                gpu_memory = torch.cuda.memory_allocated(0) / 1024**3  # GB
                logger.info(f"   GPU memory allocated: {gpu_memory:.2f}GB")
                
            except Exception as gpu_error:
                logger.warning(f"GPU loading failed: {gpu_error}, trying CPU fallback")
                device = "cpu"
                model = ColQwen2.from_pretrained(
                    'vidore/colqwen2-v1.0',
                    torch_dtype=torch.float32
                )
                model = model.to(device)
        else:
            # CPU-optimized loading
            model = ColQwen2.from_pretrained(
                'vidore/colqwen2-v1.0',
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            model = model.to(device)
        
        # Load processor
        processor = ColQwen2Processor.from_pretrained('vidore/colqwen2-v1.0')
        
        # Performance test
        if device == "cuda":
            # Quick GPU inference test
            test_input = torch.randn(1, 3, 224, 224).to(device)
            with torch.no_grad():
                _ = model.forward(test_input)
            torch.cuda.empty_cache()
        
        load_time = time.time() - start_time
        logger.info(f"✅ ColPali loaded successfully ({load_time:.2f}s)")
        logger.info(f"   Device: {device}, Model type: {type(model).__name__}")
        
        return True, load_time
        
    except Exception as e:
        logger.error(f"❌ ColPali warm-up failed: {e}")
        return False, 0

def warm_up_torch_dependencies():
    """Pre-load PyTorch and related dependencies"""
    logger.info("🔄 Warming up PyTorch dependencies...")
    start_time = time.time()
    
    try:
        import torch
        import torch.nn.functional as F
        import numpy as np
        
        # Test basic operations
        if torch.cuda.is_available():
            logger.info(f"   CUDA available: {torch.cuda.get_device_name(0)}")
            # Test GPU tensor operations
            x = torch.randn(10, 10).cuda()
            y = F.relu(x)
        else:
            logger.info("   Using CPU")
            # Test CPU tensor operations
            x = torch.randn(10, 10)
            y = F.relu(x)
        
        load_time = time.time() - start_time
        logger.info(f"✅ PyTorch dependencies loaded ({load_time:.2f}s)")
        
        return True, load_time
        
    except Exception as e:
        logger.error(f"❌ PyTorch warm-up failed: {e}")
        return False, 0

def create_model_manifest(results: Dict[str, Any], model_cache_dir: Path, target_device: str):
    """Create a manifest file with model loading information"""
    
    manifest = {
        "warm_up_timestamp": time.time(),
        "target_device": target_device,
        "models_loaded": results,
        "cache_directory": str(model_cache_dir),
        "total_models": len([r for r in results.values() if r["success"]]),
        "total_load_time": sum([r["load_time"] for r in results.values() if r["success"]]),
        "docker_optimized": os.getenv('DOCKER_PRELOADED_MODELS', 'false') == 'true'
    }
    
    manifest_path = model_cache_dir / "model_manifest.json"
    
    try:
        import json
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"📄 Model manifest created: {manifest_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to create manifest: {e}")

def main():
    """Main warm-up execution"""
    logger.info("🚀 Starting model warm-up for Docker pre-loading...")
    logger.info("=" * 60)
    
    # Setup environment and get target device
    model_cache_dir, target_device = setup_environment()
    
    # Track results
    results = {}
    total_start_time = time.time()
    
    # Warm up each model with device awareness
    if target_device == 'cuda':
        warm_up_functions = [
            ("pytorch", lambda: warm_up_torch_dependencies()),
            ("sentence_transformer", lambda: warm_up_sentence_transformer()),
            ("bge_reranker", lambda: warm_up_bge_reranker()),
            ("colpali", lambda: warm_up_colpali(target_device))
        ]
        logger.info("🚀 GPU acceleration enabled - full model loading")
    else:
        warm_up_functions = [
            ("pytorch", lambda: warm_up_torch_dependencies()),
            ("sentence_transformer", lambda: warm_up_sentence_transformer()),
            ("bge_reranker", lambda: warm_up_bge_reranker()),
            ("colpali", lambda: warm_up_colpali(target_device))
        ]
        logger.info("💻 CPU mode - optimized model loading")
    
    for model_name, warm_up_func in warm_up_functions:
        try:
            success, load_time = warm_up_func()
            results[model_name] = {
                "success": success,
                "load_time": load_time,
                "device": target_device
            }
        except Exception as e:
            logger.error(f"❌ Critical error warming up {model_name}: {e}")
            results[model_name] = {
                "success": False,
                "load_time": 0,
                "device": target_device,
                "error": str(e)
            }
    
    # Calculate totals
    total_time = time.time() - total_start_time
    successful_models = [k for k, v in results.items() if v["success"]]
    failed_models = [k for k, v in results.items() if not v["success"]]
    
    # Create manifest
    create_model_manifest(results, model_cache_dir, target_device)
    
    # Final report
    logger.info("=" * 60)
    logger.info("📊 WARM-UP SUMMARY")
    logger.info(f"✅ Successful models: {len(successful_models)}/{len(results)}")
    if successful_models:
        logger.info(f"   {', '.join(successful_models)}")
    
    if failed_models:
        logger.info(f"❌ Failed models: {len(failed_models)}")
        logger.info(f"   {', '.join(failed_models)}")
    
    logger.info(f"⏱️ Total warm-up time: {total_time:.2f}s")
    logger.info(f"📁 Models cached in: {model_cache_dir}")
    logger.info(f"💻 Target device: {target_device}")
    
    # Exit with appropriate code
    if failed_models:
        logger.warning("⚠️ Some models failed to load - Docker build may continue but functionality will be limited")
        if len(failed_models) == len(results):
            logger.error("❌ All models failed to load - critical failure")
            sys.exit(1)
        else:
            logger.info(f"✅ {len(successful_models)} models loaded successfully despite {len(failed_models)} failures")
            sys.exit(0)
    else:
        if target_device == 'cuda':
            logger.info("🚀 All models warmed up successfully with GPU acceleration! Container will have instant startup.")
        else:
            logger.info("💻 All models warmed up successfully with CPU optimization! Container will have instant startup.")
        sys.exit(0)

if __name__ == "__main__":
    main()