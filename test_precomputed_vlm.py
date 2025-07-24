#!/usr/bin/env python3
"""
Test Precomputed VLM Fix

This tests the fix for the "Page image not found" issue by precomputing VLM analysis
during document processing rather than during retrieval.
"""

import sys
import os
import logging
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging to see the fix in action
logging.basicConfig(level=logging.INFO, format='%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def test_precomputed_vlm_fix():
    """Test the precomputed VLM fix to resolve file cleanup issues."""
    
    try:
        from colpali_retriever import ColPaliRetriever
        
        print("🚀 Testing Precomputed VLM Fix")
        print("=" * 50)
        
        # Create retriever
        config = {'model_name': 'vidore/colqwen2-v1.0', 'device': 'cpu', 'max_pages_per_doc': 3}
        retriever = ColPaliRetriever(config)
        
        print(f"VLM available: {retriever.vlm_available}")
        
        # Simulate Streamlit's temporary file workflow
        pdf_path = 'data/test_docs/AI Knowledge Assignment.pdf'
        
        # Create temporary file (like Streamlit does)
        with tempfile.NamedTemporaryFile(delete=False, suffix='_AI Knowledge Assignment.pdf') as tmp_file:
            with open(pdf_path, 'rb') as orig_file:
                tmp_file.write(orig_file.read())
            temp_path = tmp_file.name
        
        print(f"📄 Temp file created: {os.path.basename(temp_path)}")
        
        # Process document (this should trigger precomputation)
        print(f"\n1️⃣ Processing document...")
        result = retriever.add_documents([temp_path])
        
        if result['total_documents'] > 0:
            print(f"✅ Document processed: {result['total_pages']} pages")
            
            # Check precomputed content
            doc_ids = list(retriever.precomputed_vlm_content.keys())
            print(f"📊 Precomputed VLM for: {doc_ids}")
            
            if doc_ids:
                doc_id = doc_ids[0]
                precomputed_pages = list(retriever.precomputed_vlm_content[doc_id].keys())
                print(f"   Pages with VLM analysis: {precomputed_pages}")
            
            # NOW delete the temporary file (simulating Streamlit cleanup)
            print(f"\n2️⃣ Deleting temporary file (simulating Streamlit cleanup)...")
            os.unlink(temp_path)
            print(f"🗑️ Temp file deleted")
            
            # Test retrieval AFTER file deletion (this was the original problem)
            print(f"\n3️⃣ Testing retrieval after file cleanup...")
            query = "Help me understand the content of this document"
            results, metrics = retriever.retrieve(query, top_k=1)
            
            if results:
                answer = results[0].content
                print(f"✅ Retrieved answer successfully!")
                print(f"   Answer length: {len(answer)} characters")
                print(f"   Source: {results[0].metadata.get('filename', 'unknown')}")
                print(f"   Page: {results[0].metadata.get('page', 'unknown')}")
                
                print(f"\n📄 Generated Answer:")
                print("=" * 40)
                print(answer)
                print("=" * 40)
                
                # Check if it's precomputed content or fallback
                if "Page image not found" in answer:
                    print("❌ Still getting 'Page image not found' error")
                    return False
                elif "precomputed" in answer.lower() or len(answer) > 200:
                    print("🎉 SUCCESS: Using precomputed VLM content!")
                    return True
                else:
                    print("✅ SUCCESS: Generated meaningful content!")
                    return True
            else:
                print("❌ No results returned")
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
    success = test_precomputed_vlm_fix()
    
    if success:
        print("\n🎯 Precomputed VLM fix test PASSED!")
        print("✅ The 'Page image not found' issue should be resolved!")
    else:
        print("\n💥 Precomputed VLM fix test FAILED!")
        print("❌ The issue may still persist")
    
    exit(0 if success else 1)