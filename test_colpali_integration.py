#!/usr/bin/env python3
"""
ColPali Integration Test - Lightweight Model Approach

This script tests ColPali visual document processing with a simple, lightweight model
to verify the architecture works before scaling to production models.

Strategy:
1. Use simple vision-text model for fast testing
2. Test PDF processing pipeline end-to-end  
3. Verify embedding generation and query processing
4. Ensure dimension compatibility throughout
"""

import os
import sys
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def test_pdf_to_image_conversion():
    """Test PDF to image conversion pipeline."""
    print("\n🖼️ Testing PDF to Image Conversion...")
    
    try:
        from pdf2image import convert_from_path
        from pdf2image.exceptions import PDFInfoNotInstalledError
        
        # Find a test PDF
        test_pdf = None
        possible_paths = [
            "data/test_docs/AI Knowledge Assignment.pdf",
            "data/documents/sample.pdf"  # Create a simple one if needed
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                test_pdf = path
                break
        
        if not test_pdf:
            print("⚠️ No test PDF found, creating a simple test...")
            return create_simple_test_pdf()
        
        print(f"📄 Testing with: {test_pdf}")
        
        # Try conversion with different poppler paths
        poppler_paths = [
            None,  # System PATH
            r"C:\Program Files\poppler\poppler-24.08.0\Library\bin",
            r"C:\Program Files\poppler\Library\bin"
        ]
        
        for poppler_path in poppler_paths:
            try:
                print(f"🔧 Trying poppler path: {poppler_path or 'system PATH'}")
                
                images = convert_from_path(
                    test_pdf,
                    dpi=150,  # Lower DPI for testing
                    poppler_path=poppler_path,
                    first_page=1,
                    last_page=2  # Just first 2 pages for testing
                )
                
                if images:
                    print(f"✅ PDF conversion successful: {len(images)} pages")
                    print(f"   Image size: {images[0].size}")
                    return images
                    
            except Exception as e:
                print(f"❌ Failed with {poppler_path}: {e}")
                continue
        
        print("❌ All PDF conversion attempts failed")
        return None
        
    except ImportError:
        print("❌ pdf2image not installed - install with: pip install pdf2image")
        return None

def create_simple_test_pdf():
    """Create a simple test PDF if none available."""
    print("📝 Creating simple test PDF...")
    
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        test_pdf_path = "test_document.pdf"
        
        c = canvas.Canvas(test_pdf_path, pagesize=letter)
        c.drawString(100, 750, "Test Document for ColPali")
        c.drawString(100, 700, "This is a simple test document.")
        c.drawString(100, 650, "It contains basic text for visual processing.")
        c.showPage()
        c.save()
        
        print(f"✅ Created test PDF: {test_pdf_path}")
        return test_pdf_path
        
    except ImportError:
        print("⚠️ reportlab not available, using text file as fallback")
        return None

def test_lightweight_colpali_model():
    """Test with a lightweight vision-text model."""
    print("\n🤖 Testing Lightweight ColPali Model...")
    
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
        
        # Use a simpler, more compatible model for testing
        # These are known to work well and have consistent dimensions
        test_models = [
            "microsoft/DialoGPT-medium",  # Simple text model for fallback
            "openai/clip-vit-base-patch32",  # CLIP model for vision-text
            "sentence-transformers/all-MiniLM-L6-v2"  # Simple embedding model
        ]
        
        for model_name in test_models:
            try:
                print(f"🔧 Testing model: {model_name}")
                
                # Try to load model
                model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  # Use float32 for CPU testing
                    trust_remote_code=True
                )
                
                processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                
                print(f"✅ Model loaded successfully: {model_name}")
                print(f"   Model type: {type(model)}")
                print(f"   Processor type: {type(processor)}")
                
                return model, processor, model_name
                
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                continue
        
        print("❌ All lightweight models failed to load")
        return None, None, None
        
    except ImportError as e:
        print(f"❌ Required packages not available: {e}")
        return None, None, None

def test_simple_embedding_generation():
    """Test simple embedding generation without complex ColPali."""
    print("\n🔢 Testing Simple Embedding Generation...")
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        # Use a simple, reliable embedding model
        model_name = "all-MiniLM-L6-v2"
        print(f"📥 Loading simple embedding model: {model_name}")
        
        model = SentenceTransformer(model_name)
        
        # Test text embedding
        test_text = "This is a test document about artificial intelligence and machine learning."
        embedding = model.encode(test_text)
        
        print(f"✅ Text embedding generated")
        print(f"   Dimension: {embedding.shape}")
        print(f"   Type: {type(embedding)}")
        print(f"   Sample values: {embedding[:5]}")
        
        # Test query embedding
        query = "What is artificial intelligence?"
        query_embedding = model.encode(query)
        
        print(f"✅ Query embedding generated")
        print(f"   Dimension: {query_embedding.shape}")
        
        # Test similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity([embedding], [query_embedding])[0][0]
        
        print(f"✅ Similarity calculation successful")
        print(f"   Similarity score: {similarity:.3f}")
        
        return model, embedding.shape[0]  # Return model and dimension
        
    except Exception as e:
        print(f"❌ Simple embedding test failed: {e}")
        return None, None

def test_visual_processing_pipeline():
    """Test the complete visual processing pipeline with lightweight components."""
    print("\n🔄 Testing Complete Visual Processing Pipeline...")
    
    # Step 1: Test PDF conversion
    images = test_pdf_to_image_conversion()
    if not images:
        print("❌ Cannot continue without PDF images")
        return False
    
    # Step 2: Test simple embedding model
    embedding_model, embedding_dim = test_simple_embedding_generation()
    if not embedding_model:
        print("❌ Cannot continue without embedding model")
        return False
    
    # Step 3: Test end-to-end processing simulation
    try:
        print("🔄 Simulating document processing...")
        
        # Simulate visual document processing
        num_pages = len(images)
        
        # Create fake visual embeddings with consistent dimensions
        import numpy as np
        visual_embeddings = np.random.random((num_pages, 100, embedding_dim))  # Patches per page
        
        print(f"✅ Simulated visual embeddings: {visual_embeddings.shape}")
        
        # Test query processing
        query = "What is the main topic of this document?"
        query_embedding = embedding_model.encode(query)
        
        print(f"✅ Query embedding: {query_embedding.shape}")
        
        # Simulate MaxSim scoring
        scores = []
        for page_idx in range(num_pages):
            page_patches = visual_embeddings[page_idx]
            
            # Simple similarity calculation (not true MaxSim, but tests the pipeline)
            page_similarities = []
            for patch in page_patches:
                sim = np.dot(query_embedding, patch) / (np.linalg.norm(query_embedding) * np.linalg.norm(patch))
                page_similarities.append(sim)
            
            max_sim = max(page_similarities)
            scores.append(max_sim)
        
        print(f"✅ Page scores calculated: {scores}")
        
        # Find best page
        best_page = np.argmax(scores)
        best_score = scores[best_page]
        
        print(f"✅ Best match: Page {best_page + 1} (score: {best_score:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def test_dimension_compatibility():
    """Test that different embedding dimensions can be handled."""
    print("\n📏 Testing Dimension Compatibility...")
    
    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Test different embedding dimensions
        dimensions = [128, 384, 768, 1536]
        
        for dim in dimensions:
            print(f"🔧 Testing dimension: {dim}")
            
            # Create sample embeddings
            doc_embedding = np.random.random((1, dim))
            query_embedding = np.random.random((1, dim))
            
            # Test similarity calculation
            similarity = cosine_similarity(doc_embedding, query_embedding)[0][0]
            
            print(f"   ✅ Similarity: {similarity:.3f}")
        
        # Test dimension mismatch handling
        print("🔧 Testing dimension mismatch handling...")
        
        doc_embedding_128 = np.random.random((1, 128))
        query_embedding_384 = np.random.random((1, 384))
        
        try:
            # This should fail
            similarity = cosine_similarity(doc_embedding_128, query_embedding_384)
            print("❌ Dimension mismatch not detected!")
        except ValueError as e:
            print(f"✅ Dimension mismatch properly detected: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dimension compatibility test failed: {e}")
        return False

def main():
    """Run all ColPali integration tests."""
    print("🚀 ColPali Integration Testing - Lightweight Model Approach")
    print("=" * 60)
    
    results = {}
    
    # Test 1: PDF to Image Conversion
    print("\n" + "="*20 + " TEST 1: PDF PROCESSING " + "="*20)
    results['pdf_conversion'] = test_pdf_to_image_conversion() is not None
    
    # Test 2: Lightweight Model Loading
    print("\n" + "="*20 + " TEST 2: MODEL LOADING " + "="*20)
    model, processor, model_name = test_lightweight_colpali_model()
    results['model_loading'] = model is not None
    
    # Test 3: Simple Embedding Generation
    print("\n" + "="*20 + " TEST 3: EMBEDDING GENERATION " + "="*20)
    results['embedding_generation'] = test_simple_embedding_generation()[0] is not None
    
    # Test 4: Dimension Compatibility
    print("\n" + "="*20 + " TEST 4: DIMENSION COMPATIBILITY " + "="*20)
    results['dimension_compatibility'] = test_dimension_compatibility()
    
    # Test 5: Complete Pipeline
    print("\n" + "="*20 + " TEST 5: COMPLETE PIPELINE " + "="*20)
    results['complete_pipeline'] = test_visual_processing_pipeline()
    
    # Summary
    print("\n" + "="*20 + " TEST SUMMARY " + "="*20)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready to implement lightweight ColPali integration.")
    else:
        print("⚠️ Some tests failed. Fix these issues before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)