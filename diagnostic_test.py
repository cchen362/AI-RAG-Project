#!/usr/bin/env python3
"""
Diagnostic Test for Text RAG and ColPali Systems

This test will isolate both systems to identify why text processing shows 0 documents.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

def test_text_rag_isolated():
    """Test ONLY the text RAG system in isolation."""
    print("🔍 TESTING TEXT RAG SYSTEM IN ISOLATION")
    print("=" * 60)
    
    try:
        # Import and test individual components first
        print("\n1️⃣ Testing individual components...")
        
        # Test DocumentProcessor
        print("   📄 Testing DocumentProcessor...")
        from document_processor import DocumentProcessor
        doc_processor = DocumentProcessor()
        print("   ✅ DocumentProcessor imported successfully")
        
        # Test TextChunker
        print("   ✂️ Testing TextChunker...")
        from text_chunker import TextChunker
        text_chunker = TextChunker()
        print("   ✅ TextChunker imported successfully")
        
        # Test EmbeddingManager
        print("   🔢 Testing EmbeddingManager...")
        from embedding_manager import EmbeddingManager
        embedding_manager = EmbeddingManager.create_local()
        print("   ✅ EmbeddingManager imported successfully")
        
        # Test RAGSystem
        print("   🎯 Testing RAGSystem...")
        from rag_system import RAGSystem
        text_config = {
            'chunk_size': 800,
            'chunk_overlap': 150,
            'model_name': 'gpt-3.5-turbo',
            'temperature': 0.1
        }
        rag_system = RAGSystem(text_config)
        print("   ✅ RAGSystem initialized successfully")
        
        # Test with an actual document
        print("\n2️⃣ Testing document processing...")
        test_documents = [
            'data/test_docs/NIPS-2017-attention-is-all-you-need-Paper.pdf',
            'data/test_docs/AI Knowledge Assignment.pdf',
            'data/documents/test_document.pdf'  # fallback
        ]
        
        test_doc = None
        for doc_path in test_documents:
            if os.path.exists(doc_path):
                test_doc = doc_path
                break
                
        if not test_doc:
            print("   ❌ No test document found!")
            return False
            
        print(f"   📄 Using test document: {os.path.basename(test_doc)}")
        
        # Process the document
        result = rag_system.add_documents([test_doc])
        
        print("\n3️⃣ Processing Results:")
        print(f"   Total processed: {len(result['successful'])}")
        print(f"   Failed: {len(result['failed'])}")
        print(f"   Total chunks: {result['total_chunks']}")
        print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
        
        if result['successful']:
            print("   ✅ TEXT RAG WORKING - Document processed successfully!")
            successful_doc = result['successful'][0]
            print(f"      Filename: {successful_doc['filename']}")
            print(f"      Chunks created: {successful_doc['chunks']}")
            return True
        else:
            print("   ❌ TEXT RAG FAILED - No successful processing")
            if result['failed']:
                error_info = result['failed'][0]
                print(f"      Error: {error_info.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"   ❌ CRITICAL ERROR: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def test_colpali_isolated():
    """Test ONLY the ColPali system in isolation."""
    print("\n🔍 TESTING COLPALI SYSTEM IN ISOLATION")
    print("=" * 60)
    
    try:
        print("\n1️⃣ Testing ColPali components...")
        
        from colpali_retriever import ColPaliRetriever
        config = {
            'model_name': 'vidore/colqwen2-v1.0',
            'device': 'cpu',  # Use CPU for testing
            'max_pages_per_doc': 5
        }
        
        colpali = ColPaliRetriever(config)
        print("   ✅ ColPaliRetriever initialized successfully")
        print(f"   VLM Available: {colpali.vlm_available}")
        
        # Test with document
        print("\n2️⃣ Testing ColPali document processing...")
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
            print("   ❌ No test document found!")
            return False
            
        print(f"   📄 Using test document: {os.path.basename(test_doc)}")
        
        result = colpali.add_documents([test_doc])
        
        print("\n3️⃣ ColPali Processing Results:")
        print(f"   Total documents: {result['total_documents']}")
        print(f"   Total pages: {result['total_pages']}")
        print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
        
        if result['total_documents'] > 0:
            print("   ✅ COLPALI WORKING - Document processed successfully!")
            return True
        else:
            print("   ❌ COLPALI FAILED - No documents processed")
            return False
            
    except Exception as e:
        print(f"   ❌ CRITICAL ERROR: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run comprehensive diagnostic tests."""
    print("🚀 COMPREHENSIVE RAG SYSTEM DIAGNOSTICS")
    print("=" * 80)
    
    # Test text RAG
    text_success = test_text_rag_isolated()
    
    # Test ColPali
    colpali_success = test_colpali_isolated()
    
    # Summary
    print("\n📊 DIAGNOSTIC SUMMARY")
    print("=" * 40)
    print(f"Text RAG System:   {'✅ WORKING' if text_success else '❌ BROKEN'}")
    print(f"ColPali System:    {'✅ WORKING' if colpali_success else '❌ BROKEN'}")
    
    if not text_success:
        print("\n🔥 CRITICAL ISSUE IDENTIFIED:")
        print("   Text RAG system is not processing documents correctly!")
        print("   This explains why 'Text: 0 docs' appears in logs.")
        print("   Need to debug text processing pipeline before integration.")
    
    if not colpali_success:
        print("\n⚠️ ColPali system also has issues that need addressing.")
    
    if text_success and colpali_success:
        print("\n🎉 Both systems working independently!")
        print("   Issue may be in the integration layer (SimpleRAGOrchestrator).")
    
    return text_success and colpali_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)