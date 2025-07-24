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
    
    logger.info(f"Model cache directory: {model_cache_dir}")
    return model_cache_dir

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
        
        return True, load_time
        
    except Exception as e:
        logger.error(f"❌ BGE Re-ranker warm-up failed: {e}")
        return False, 0

def warm_up_colpali():
    """Pre-load ColPali model"""
    logger.info("🔄 Warming up ColPali (vidore/colqwen2-v1.0)...")
    start_time = time.time()
    
    try:
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        import torch
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"   Using device: {device}")
        
        # Load model with appropriate settings
        if device == "cuda":
            model = ColQwen2.from_pretrained(
                'vidore/colqwen2-v1.0',
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        else:
            model = ColQwen2.from_pretrained(
                'vidore/colqwen2-v1.0',
                torch_dtype=torch.float32
            )
            model = model.to(device)
        
        # Load processor
        processor = ColQwen2Processor.from_pretrained('vidore/colqwen2-v1.0')
        
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

def create_model_manifest(results: Dict[str, Any], model_cache_dir: Path):
    """Create a manifest file with model loading information"""
    
    manifest = {
        "warm_up_timestamp": time.time(),
        "models_loaded": results,
        "cache_directory": str(model_cache_dir),
        "total_models": len([r for r in results.values() if r["success"]]),
        "total_load_time": sum([r["load_time"] for r in results.values() if r["success"]])
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
    
    # Setup environment
    model_cache_dir = setup_environment()
    
    # Track results
    results = {}
    total_start_time = time.time()
    
    # Warm up each model
    warm_up_functions = [
        ("pytorch", warm_up_torch_dependencies),
        ("sentence_transformer", warm_up_sentence_transformer),
        ("bge_reranker", warm_up_bge_reranker),
        ("colpali", warm_up_colpali)
    ]
    
    for model_name, warm_up_func in warm_up_functions:
        try:
            success, load_time = warm_up_func()
            results[model_name] = {
                "success": success,
                "load_time": load_time
            }
        except Exception as e:
            logger.error(f"❌ Critical error warming up {model_name}: {e}")
            results[model_name] = {
                "success": False,
                "load_time": 0,
                "error": str(e)
            }
    
    # Calculate totals
    total_time = time.time() - total_start_time
    successful_models = [k for k, v in results.items() if v["success"]]
    failed_models = [k for k, v in results.items() if not v["success"]]
    
    # Create manifest
    create_model_manifest(results, model_cache_dir)
    
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
    
    # Exit with appropriate code
    if failed_models:
        logger.warning("⚠️ Some models failed to load - Docker build may continue but functionality will be limited")
        sys.exit(1) if len(failed_models) == len(results) else sys.exit(0)
    else:
        logger.info("🎉 All models warmed up successfully! Docker container will have instant startup.")
        sys.exit(0)

if __name__ == "__main__":
    main()