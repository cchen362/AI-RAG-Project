#!/usr/bin/env python3
"""
Debug test for document processing issues
"""

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import tempfile
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def test_rag_system_initialization():
    """Test RAG system initialization with debug logging"""
    
    print("=== Testing RAG System Initialization ===")
    
    try:
        from rag_system import RAGSystem
        
        # Test config matching streamlit app
        text_config = {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'embedding_model': 'openai',
            'generation_model': 'gpt-3.5-turbo',
            'max_retrieved_chunks': 5,
            'temperature': 0.1
        }
        
        print("Creating RAG system...")
        rag_system = RAGSystem(text_config)
        print("RAG system created successfully")
        
        # Test with a simple text file
        test_content = "This is a test document for debugging. It contains multiple sentences to test the chunking process. The embedding system should process this content correctly."
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_file_path = f.name
        
        print(f"Testing document processing with: {temp_file_path}")
        
        # Test document processing
        result = rag_system._process_document_simple(temp_file_path)
        
        print("Processing result:")
        print(f"  Success: {result.get('success', False)}")
        print(f"  Error: {result.get('error', 'None')}")
        if result.get('success'):
            print(f"  Chunks: {result.get('chunks_created', 0)}")
            print(f"  Embeddings: {result.get('embeddings_created', 0)}")
        
        # Cleanup
        os.unlink(temp_file_path)
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"ERROR during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rag_system_initialization()
    if success:
        print("\\n🎉 Document processing test PASSED")
    else:
        print("\\n💥 Document processing test FAILED")
    
    # Clean up
    try:
        os.remove(__file__)
        print("Test file cleaned up")
    except:
        pass