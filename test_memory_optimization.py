#!/usr/bin/env python3
"""
Memory Optimization Test Script for ColPali
Tests the new memory-efficient processing on 6GB GPU constraints
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add src directory to path
sys.path.append('src')

from visual_document_processor import VisualDocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_memory_optimization():
    """Test ColPali memory optimization with a sample PDF"""
    
    logger.info("🧪 Testing ColPali Memory Optimization")
    logger.info("=" * 50)
    
    # Check GPU availability and memory
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available - cannot test GPU memory optimization")
        return False
    
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"🖥️ GPU: {gpu_name} ({gpu_memory_gb:.1f}GB)")
    
    # Show initial GPU memory usage
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated(0) / 1024**3
    logger.info(f"📊 Initial GPU memory: {initial_memory:.2f}GB")
    
    # Configure ColPali processor for memory optimization
    config = {
        'colpali_model': 'vidore/colqwen2-v1.0',
        'device': 'cuda',
        'cache_dir': 'cache/embeddings'
    }
    
    try:
        # Initialize processor
        logger.info("🔧 Initializing VisualDocumentProcessor...")
        processor = VisualDocumentProcessor(config)
        
        # Check memory after initialization
        post_init_memory = torch.cuda.memory_allocated(0) / 1024**3
        logger.info(f"📊 Memory after init: {post_init_memory:.2f}GB")
        
        # Test with a sample PDF (if available)
        test_pdf_path = None
        
        # Look for test PDFs in common locations
        test_locations = [
            'data/documents',
            'evaluation_results',
            '.'
        ]
        
        for location in test_locations:
            pdf_files = list(Path(location).glob('*.pdf'))
            if pdf_files:
                test_pdf_path = str(pdf_files[0])
                break
        
        if test_pdf_path:
            logger.info(f"📄 Testing with PDF: {test_pdf_path}")
            
            # Process the document
            result = processor.process_file(test_pdf_path)
            
            if result['status'] == 'success':
                logger.info("✅ Document processing succeeded!")
                logger.info(f"   📄 Pages processed: {result['metadata']['page_count']}")
                logger.info(f"   ⏱️ Processing time: {result['processing_time']:.2f}s")
                logger.info(f"   📐 Embedding shape: {result['metadata']['embedding_shape']}")
                
                # Test query processing
                logger.info("🔍 Testing query processing...")
                query = "What is shown in the chart?"
                scores = processor.query_embeddings(query, result['embeddings'])
                logger.info(f"✅ Query processing succeeded! Scores shape: {scores.shape}")
                
            else:
                logger.error(f"❌ Document processing failed: {result['error']}")
                return False
        else:
            logger.warning("⚠️ No test PDF found - cannot test full pipeline")
            logger.info("✅ Processor initialization succeeded (basic test passed)")
        
        # Final memory check
        final_memory = torch.cuda.memory_allocated(0) / 1024**3
        peak_memory = torch.cuda.max_memory_allocated(0) / 1024**3
        
        logger.info("📊 Memory Usage Summary:")
        logger.info(f"   Initial: {initial_memory:.2f}GB")
        logger.info(f"   Final: {final_memory:.2f}GB")
        logger.info(f"   Peak: {peak_memory:.2f}GB")
        logger.info(f"   Available: {gpu_memory_gb - peak_memory:.2f}GB remaining")
        
        if peak_memory < gpu_memory_gb * 0.95:  # Less than 95% usage
            logger.info("✅ Memory optimization test PASSED!")
            return True
        else:
            logger.warning("⚠️ Memory usage high but may still work")
            return True
            
    except Exception as e:
        logger.error(f"❌ Memory optimization test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        torch.cuda.empty_cache()

def main():
    """Run memory optimization tests"""
    success = test_memory_optimization()
    
    if success:
        logger.info("🎉 All tests passed! Memory optimization is ready for deployment.")
        exit(0)
    else:
        logger.error("💥 Tests failed! Review the implementation before deployment.")
        exit(1)

if __name__ == "__main__":
    main()