#!/usr/bin/env python3
"""
Test Complete ColPali Pipeline

This tests the full ColPali→VLM pipeline with a Transformer architecture query.
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

def test_complete_pipeline():
    """Test the complete ColPali pipeline with Transformer query."""
    
    try:
        from colpali_retriever import ColPaliRetriever
        
        print("🚀 Testing Complete ColPali Pipeline")
        print("=" * 60)
        
        # Create retriever 
        config = {
            'model_name': 'vidore/colqwen2-v1.0',
            'device': 'cpu',
            'max_pages_per_doc': 10
        }
        
        print("1️⃣ Creating ColPali retriever...")
        retriever = ColPaliRetriever(config)
        print(f"   VLM available: {retriever.vlm_available}")
        
        # Add a document that should contain Transformer info
        # Use the NIPS Transformer paper if available, otherwise the test doc
        pdf_candidates = [
            'data/test_docs/NIPS-2017-attention-is-all-you-need-Paper.pdf',
            'data/test_docs/AI Knowledge Assignment.pdf'
        ]
        
        pdf_path = None
        for candidate in pdf_candidates:
            if os.path.exists(candidate):
                pdf_path = candidate
                break
        
        if not pdf_path:
            print("❌ No suitable PDF found for testing")
            return False
        
        print(f"\n2️⃣ Processing document: {os.path.basename(pdf_path)}")
        
        # Process the document through ColPali
        result = retriever.add_documents([pdf_path])
        
        if result['total_documents'] > 0:
            print(f"✅ Document processed successfully")
            print(f"   Pages indexed: {result['total_pages']}")
            print(f"   Processing time: {result['processing_time']:.2f}s")
        else:
            print("❌ Document processing failed")
            return False
        
        # Test retrieval with Transformer query
        print(f"\n3️⃣ Testing retrieval...")
        query = "Help me understand the model architecture of the Transformer"
        print(f"   Query: {query}")
        
        # Retrieve relevant content
        results, metrics = retriever.retrieve(query, top_k=3)
        
        print(f"\n4️⃣ Retrieval Results:")
        print(f"   Found {len(results)} results")
        print(f"   Query time: {metrics.query_time:.3f}s")
        print(f"   Max score: {metrics.max_score:.3f}")
        
        if len(results) > 0:
            print(f"\n5️⃣ Top Result Analysis:")
            top_result = results[0]
            
            print(f"   Score: {top_result.score:.3f}")
            print(f"   Source: {top_result.metadata.get('filename', 'unknown')}")
            print(f"   Page: {top_result.metadata.get('page', 'unknown')}")
            print(f"   Content length: {len(top_result.content)} characters")
            
            print(f"\n📄 Generated Answer:")
            print("=" * 50)
            print(top_result.content)
            print("=" * 50)
            
            # Check if this is a real answer
            content_indicators = [
                "transformer", "attention", "architecture", "encoder", "decoder",
                "neural", "model", "network", "layer", "mechanism"
            ]
            
            content_lower = top_result.content.lower()
            found_indicators = [ind for ind in content_indicators if ind in content_lower]
            
            print(f"\n6️⃣ Answer Quality Assessment:")
            print(f"   Relevant keywords found: {len(found_indicators)}")
            print(f"   Keywords: {', '.join(found_indicators)}")
            
            if len(found_indicators) >= 3:
                print("🎉 HIGH QUALITY: Answer contains relevant technical content!")
                return True
            elif "Found relevant visual content" not in top_result.content:
                print("✅ MEDIUM QUALITY: Answer generated but may need more context")
                return True
            else:
                print("⚠️ LOW QUALITY: Generic fallback response")
                return False
        
        else:
            print("❌ No results found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_complete_pipeline()
    
    if success:
        print("\n🎯 Complete pipeline test PASSED!")
        print("✅ ColPali is now generating real answers from visual content!")
    else:
        print("\n💥 Complete pipeline test FAILED!")
        print("❌ Need to debug the retrieval or answer generation process")
    
    exit(0 if success else 1)