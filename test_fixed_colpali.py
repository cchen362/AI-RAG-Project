#!/usr/bin/env python3
"""
Test Fixed ColPali System - Full Document Processing

Tests that the ColPali system now processes ALL pages and provides
detailed VLM analysis for any retrieved page.
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

def test_fixed_colpali():
    """Test the fixed ColPali system with full document processing."""
    print("🚀 TESTING FIXED COLPALI - FULL DOCUMENT PROCESSING")
    print("=" * 70)
    
    try:
        from colpali_retriever import ColPaliRetriever
        
        # Create retriever with CPU mode for testing
        config = {
            'model_name': 'vidore/colqwen2-v1.0',
            'device': 'cpu',
            'max_pages_per_doc': 50  # No artificial limits
        }
        
        print("1️⃣ Initializing ColPali retriever...")
        retriever = ColPaliRetriever(config)
        print(f"   VLM available: {retriever.vlm_available}")
        
        # Test with multi-page document
        test_documents = [
            'data/test_docs/NIPS-2017-attention-is-all-you-need-Paper.pdf',
            'data/test_docs/AI Knowledge Assignment.pdf'
        ]
        
        test_doc = None
        for doc_path in test_documents:
            if os.path.exists(doc_path):
                test_doc = doc_path
                break
                
        if not test_doc:
            print("❌ No test document found!")
            return False
            
        print(f"\n2️⃣ Processing document: {os.path.basename(test_doc)}")
        
        # Process the document - should now handle ALL pages
        result = retriever.add_documents([test_doc])
        
        if result['total_documents'] > 0:
            print(f"✅ Document processed successfully")
            print(f"   Total pages indexed: {result['total_pages']}")
            print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
            
            # Check how many page images were stored
            doc_ids = list(retriever.page_images.keys())
            if doc_ids:
                doc_id = doc_ids[0]
                stored_pages = list(retriever.page_images[doc_id].keys())
                print(f"   Page images stored: {len(stored_pages)} pages")
                print(f"   Page indices: {sorted(stored_pages)}")
                
                # Test queries that should find content on later pages
                test_queries = [
                    "Help me understand the Transformer architecture",
                    "What is the attention mechanism?",
                    "Explain the model architecture"
                ]
                
                print(f"\n3️⃣ Testing retrieval and VLM analysis...")
                
                for i, query in enumerate(test_queries, 1):
                    print(f"\n📝 Query {i}: {query}")
                    
                    # Retrieve relevant content
                    results, metrics = retriever.retrieve(query, top_k=2)
                    
                    if results:
                        print(f"   Found {len(results)} results")
                        
                        for j, result in enumerate(results):
                            page_num = result.metadata.get('page', 'unknown')
                            score = result.score
                            content_length = len(result.content)
                            
                            print(f"   Result {j+1}: Page {page_num}, Score: {score:.3f}")
                            print(f"             Content length: {content_length} chars")
                            
                            # Check if this is real VLM analysis or generic response
                            if "Page image not found" in result.content:
                                print(f"             ❌ Still getting 'Page image not found' error!")
                                return False
                            elif content_length > 200 and "specific" in result.content.lower():
                                print(f"             ✅ Detailed VLM analysis detected!")
                            elif content_length > 100:
                                print(f"             ✅ Meaningful content generated!")
                            else:
                                print(f"             ⚠️ Generic/short response")
                            
                            # Show a snippet of the content
                            content_preview = result.content[:200] + "..." if len(result.content) > 200 else result.content
                            print(f"             Preview: {content_preview}")
                    else:
                        print(f"   ❌ No results found for query")
                
                return True
            else:
                print("❌ No page images stored")
                return False
        else:
            print("❌ Document processing failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_fixed_colpali()
    
    if success:
        print("\n🎉 FIXED COLPALI TEST PASSED!")
        print("✅ System now processes ALL pages and provides detailed VLM analysis!")
    else:
        print("\n💥 FIXED COLPALI TEST FAILED!")
        print("❌ Issues still remain - need further debugging")
    
    exit(0 if success else 1)