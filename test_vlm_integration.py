#!/usr/bin/env python3
"""
Test VLM Integration for ColPali

This tests the Vision-Language Model integration for generating actual answers.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def test_vlm_integration():
    """Test VLM integration for answer generation."""
    
    try:
        from colpali_retriever import ColPaliRetriever
        
        print("🚀 Testing VLM Integration")
        print("=" * 50)
        
        # Check if OpenAI is available
        try:
            import openai
            print("✅ OpenAI package installed")
        except ImportError:
            print("❌ OpenAI package not installed")
            return False
        
        # Check for API key
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            print(f"✅ OpenAI API key found: {api_key[:8]}...{api_key[-4:]}")
        else:
            print("⚠️ No OpenAI API key found in environment")
            print("   Set OPENAI_API_KEY environment variable to test VLM")
            
            # Test without API key - should gracefully degrade
            print("\n📋 Testing VLM initialization without API key...")
        
        # Create retriever
        config = {
            'model_name': 'vidore/colqwen2-v1.0',
            'device': 'cpu',
            'max_pages_per_doc': 5
        }
        
        print("\n1️⃣ Creating ColPali retriever...")
        retriever = ColPaliRetriever(config)
        
        # Check VLM availability
        print(f"\n2️⃣ VLM Status:")
        print(f"   VLM available: {retriever.vlm_available}")
        print(f"   VLM client: {retriever.vlm_client}")
        print(f"   VLM model: {retriever.vlm_model}")
        
        # Test image storage first
        doc_id = 'test_transformer_pdf'
        pdf_path = 'data/test_docs/AI Knowledge Assignment.pdf'
        
        print(f"\n3️⃣ Storing PDF images...")
        retriever._store_page_images(doc_id, pdf_path)
        
        if doc_id in retriever.page_images and len(retriever.page_images[doc_id]) > 0:
            print(f"✅ Images stored: {len(retriever.page_images[doc_id])} pages")
            
            # Test VLM answer generation
            print(f"\n4️⃣ Testing VLM answer generation...")
            query = "Help me understand the model architecture of the Transformer"
            page_idx = 0
            doc_info = {
                'filename': 'AI Knowledge Assignment.pdf',
                'original_path': pdf_path,
                'page_count': 5
            }
            
            # Test the VLM answer generation method
            print(f"   Query: {query}")
            print(f"   Testing page {page_idx}")
            
            answer = retriever._generate_vlm_answer(query, doc_id, page_idx, doc_info)
            
            print(f"\n5️⃣ Generated Answer:")
            print(f"   Length: {len(answer)} characters")
            print(f"   Preview: {answer[:200]}...")
            
            # Check if it's a real answer or generic fallback
            if "Found relevant visual content" in answer:
                print("⚠️ Got generic fallback response")
                print("   This means VLM analysis is not working")
                
                if not api_key:
                    print("💡 Reason: No OpenAI API key provided")
                    print("   To get real answers, set OPENAI_API_KEY environment variable")
                
                return False
            else:
                print("🎉 Got VLM-generated answer!")
                return True
        
        else:
            print("❌ Image storage failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def test_image_analysis_directly():
    """Test image analysis directly with a stored image."""
    
    print("\n🖼️ Testing Direct Image Analysis")
    print("=" * 50)
    
    try:
        from colpali_retriever import ColPaliRetriever
        
        # Create retriever
        config = {'model_name': 'vidore/colqwen2-v1.0', 'device': 'cpu'}
        retriever = ColPaliRetriever(config)
        
        # Store images first
        doc_id = 'test_direct'
        pdf_path = 'data/test_docs/AI Knowledge Assignment.pdf'
        retriever._store_page_images(doc_id, pdf_path)
        
        if retriever.vlm_available and doc_id in retriever.page_images:
            # Get first page image
            page_image = retriever.page_images[doc_id][0]
            print(f"✅ Got page image: {page_image.size} pixels")
            
            # Test direct VLM analysis
            query = "What is this document about?"
            filename = "AI Knowledge Assignment.pdf"
            page_number = 1
            
            answer = retriever._analyze_image_with_vlm(query, page_image, filename, page_number)
            
            if answer:
                print(f"✅ VLM analysis successful!")
                print(f"   Answer length: {len(answer)} characters")
                print(f"   Answer preview: {answer[:300]}...")
                return True
            else:
                print("❌ VLM analysis returned None")
                return False
        else:
            print("⚠️ Cannot test - VLM not available or no images stored")
            return False
            
    except Exception as e:
        print(f"❌ Direct image analysis failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 VLM Integration Test Suite")
    print("=" * 60)
    
    # Test 1: Basic VLM integration
    success1 = test_vlm_integration()
    
    # Test 2: Direct image analysis (if VLM available)
    success2 = test_image_analysis_directly()
    
    print(f"\n📊 Test Results:")
    print(f"   VLM Integration: {'PASS' if success1 else 'FAIL'}")
    print(f"   Direct Analysis: {'PASS' if success2 else 'FAIL'}")
    
    if success1 or success2:
        print("\n🎯 VLM integration test PASSED!")
    else:
        print("\n💥 VLM integration test FAILED!")
        print("💡 To enable VLM: Set OPENAI_API_KEY environment variable")
    
    exit(0 if (success1 or success2) else 1)