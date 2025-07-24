#!/usr/bin/env python3
"""
Test Image Storage in ColPali Retriever

This tests the PDF→Image conversion and storage specifically.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def test_image_storage():
    """Test PDF→Image storage in ColPali retriever."""
    
    try:
        from colpali_retriever import ColPaliRetriever
        
        print("🚀 Testing ColPali Image Storage")
        print("=" * 50)
        
        # Create a minimal config for testing
        config = {
            'model_name': 'vidore/colqwen2-v1.0',
            'device': 'cpu',  # Use CPU for testing
            'max_pages_per_doc': 5
        }
        
        print("1️⃣ Creating ColPali retriever...")
        try:
            retriever = ColPaliRetriever(config)
            print("✅ ColPaliRetriever created successfully")
        except Exception as e:
            print(f"❌ Failed to create ColPaliRetriever: {e}")
            return False
        
        # Test with a known PDF
        doc_id = 'test_pdf_123'
        pdf_path = 'data/test_docs/AI Knowledge Assignment.pdf'
        
        print(f"\n2️⃣ Testing image storage...")
        print(f"   PDF: {pdf_path}")
        print(f"   Doc ID: {doc_id}")
        print(f"   PDF exists: {os.path.exists(pdf_path)}")
        
        # Test the image storage method directly
        print(f"\n🖼️ Calling _store_page_images()...")
        retriever._store_page_images(doc_id, pdf_path)
        
        # Check results
        print(f"\n3️⃣ Checking storage results...")
        if doc_id in retriever.page_images:
            stored_images = retriever.page_images[doc_id]
            print(f"✅ Doc ID found in page_images")
            print(f"   Number of images stored: {len(stored_images)}")
            
            for page_idx, image in stored_images.items():
                print(f"   Page {page_idx}: {image.size} pixels")
            
            if len(stored_images) > 0:
                print("🎉 SUCCESS: Images stored correctly!")
                
                # Test image retrieval
                print(f"\n4️⃣ Testing image retrieval...")
                test_image = retriever._get_page_image(doc_id, 0)
                if test_image:
                    print(f"✅ Image retrieval works: {test_image.size} pixels")
                else:
                    print("❌ Image retrieval failed")
                
                return True
            else:
                print("❌ No images were stored")
                return False
        else:
            print(f"❌ Doc ID '{doc_id}' not found in page_images")
            print(f"   Available keys: {list(retriever.page_images.keys())}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_image_storage()
    if success:
        print("\n🎯 Image storage test PASSED!")
    else:
        print("\n💥 Image storage test FAILED!")
    
    exit(0 if success else 1)