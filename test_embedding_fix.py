#!/usr/bin/env python3
"""
Test script to validate the embedding dimension fix.
This will help verify that our OpenAI dimension validation works correctly.
"""

import sys
import os
import logging

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging to see all debug info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_embedding_dimensions():
    """Test that our embedding manager properly handles dimensions."""
    print("Testing OpenAI Embedding Dimensions Fix")
    print("=" * 50)
    
    try:
        from embedding_manager import EmbeddingManager
        
        print("\nCreating EmbeddingManager with custom dimensions (512)...")
        embedding_manager = EmbeddingManager.create_openai("text-embedding-3-large", dimensions=512)
        
        print(f"\nEmbedding Manager Details:")
        print(f"   - Model: {embedding_manager.model_name}")
        print(f"   - Requested dimensions: {embedding_manager.dimensions}")
        print(f"   - Actual dimensions: {embedding_manager.embedding_dimension}")
        print(f"   - Dimension match: {embedding_manager.dimensions == embedding_manager.embedding_dimension}")
        
        print(f"\nTesting single embedding creation...")
        test_text = "This is a test document about artificial intelligence and machine learning."
        embedding = embedding_manager.create_embedding(test_text)
        
        print(f"Single Embedding Results:")
        print(f"   - Embedding shape: {embedding.shape}")
        print(f"   - Expected dimensions: {embedding_manager.embedding_dimension}")
        print(f"   - Shape matches: {embedding.shape[0] == embedding_manager.embedding_dimension}")
        
        print(f"\nTesting batch embedding creation...")
        test_texts = [
            "Document about machine learning algorithms",
            "Article on natural language processing", 
            "Research paper on deep learning"
        ]
        batch_embeddings = embedding_manager.create_batch_embeddings(test_texts)
        
        print(f"Batch Embedding Results:")
        print(f"   - Number of embeddings: {len(batch_embeddings)}")
        for i, emb in enumerate(batch_embeddings):
            print(f"   - Embedding {i+1} shape: {emb.shape}")
            if emb.shape[0] != embedding_manager.embedding_dimension:
                print(f"   ERROR: Dimension mismatch for embedding {i+1}")
                return False
        
        print(f"\nSUCCESS: All dimension tests passed!")
        print(f"OpenAI embeddings working correctly with {embedding_manager.embedding_dimension} dimensions")
        return True
        
    except Exception as e:
        print(f"\nERROR: Test failed with error: {str(e)}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return False

def test_rag_system_initialization():
    """Test that RAG system initializes with correct dimensions."""
    print(f"\nTesting RAG System Initialization")
    print("=" * 30)
    
    try:
        from rag_system import RAGSystem
        
        config = {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'embedding_model': 'openai'
        }
        
        print("Creating RAG System...")
        rag = RAGSystem(config)
        
        print(f"RAG System Details:")
        print(f"   - Embedding dimensions: {rag.embedding_manager.embedding_dimension}")
        print(f"   - Vector DB dimensions: {rag.vector_db.dimension}")
        print(f"   - Dimensions match: {rag.embedding_manager.embedding_dimension == rag.vector_db.dimension}")
        
        if rag.embedding_manager.embedding_dimension != rag.vector_db.dimension:
            print(f"ERROR: RAG system dimension mismatch!")
            return False
            
        print(f"SUCCESS: RAG system initialized correctly!")
        return True
        
    except Exception as e:
        print(f"ERROR: RAG system test failed: {str(e)}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("Starting Embedding Dimension Fix Tests")
    print("=" * 60)
    
    # Test 1: Embedding dimensions
    test1_success = test_embedding_dimensions()
    
    # Test 2: RAG system initialization  
    test2_success = test_rag_system_initialization()
    
    print(f"\nTest Results Summary")
    print("=" * 30)
    print(f"Embedding Dimensions: {'PASS' if test1_success else 'FAIL'}")
    print(f"RAG System Init: {'PASS' if test2_success else 'FAIL'}")
    
    if test1_success and test2_success:
        print(f"\nSUCCESS: All tests passed! The embedding dimension fix is working correctly.")
        print(f"Next step: Test with actual document processing")
    else:
        print(f"\nWARNING: Some tests failed. Please check the logs above for details.")
        sys.exit(1)