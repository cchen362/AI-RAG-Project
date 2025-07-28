#!/usr/bin/env python3
"""
Test the complete document processing and search pipeline
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
logging.basicConfig(level=logging.INFO)

def test_complete_pipeline():
    """Test complete pipeline: process document -> search -> get results"""
    
    print("=== Testing Complete Pipeline ===")
    
    try:
        from rag_system import RAGSystem
        
        # Test config matching streamlit app
        text_config = {
            'chunk_size': 500,  # Smaller chunks to force multiple chunks
            'chunk_overlap': 100,
            'embedding_model': 'openai',
            'generation_model': 'gpt-3.5-turbo',
            'max_retrieved_chunks': 5,
            'temperature': 0.1
        }
        
        print("Creating RAG system...")
        rag_system = RAGSystem(text_config)
        print(f"RAG system created successfully - embedding dimension: {rag_system.embedding_manager.embedding_dimension}")
        
        # Create longer content to force multiple chunks
        test_content = ""
        for i in range(5):
            test_content += f"""
Section {i+1}: This is a detailed section about topic {i+1}.
This section contains comprehensive information about various aspects of the subject matter.
It includes detailed explanations, examples, and technical specifications that are relevant to the topic.
The content is designed to be substantial enough to create multiple text chunks during processing.
Each section should be independently searchable and retrievable through the RAG system.
This particular section focuses on specific aspects and provides detailed insights.
The information here is crucial for understanding the complete picture of the subject.
It contains both theoretical concepts and practical applications that users might search for.
The section concludes with important takeaways and actionable recommendations.

"""
        
        print(f"Test content: {len(test_content)} characters")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_file_path = f.name
        
        print(f"Processing document: {temp_file_path}")
        
        # Use the proper add_documents method
        result = rag_system.add_documents([temp_file_path])
        
        print("\\nProcessing result:")
        print(f"  Successful: {len(result['successful'])}")
        print(f"  Failed: {len(result['failed'])}")
        print(f"  Total chunks: {result['total_chunks']}")
        
        if result['successful']:
            chunks = result['successful'][0]['chunks']
            print(f"  Chunks created: {chunks}")
        
        # Test search capability
        if result['successful']:
            print("\\n=== Testing Search Capability ===")
            test_queries = [
                "What is section 1 about?",
                "Tell me about topic 3",
                "What are the technical specifications?"
            ]
            
            for query in test_queries:
                print(f"\\nQuery: {query}")
                search_result = rag_system.query(query)
                print(f"  Success: {search_result.get('success', False)}")
                if search_result.get('success'):
                    print(f"  Chunks used: {search_result.get('chunks_used', 0)}")
                    print(f"  Confidence: {search_result.get('confidence', 0):.3f}")
                    answer_preview = search_result.get('answer', '')[:150]
                    print(f"  Answer preview: {answer_preview}...")
                else:
                    print(f"  Search failed: {search_result.get('error', 'Unknown error')}")
        
        # Cleanup
        os.unlink(temp_file_path)
        
        return len(result['successful']) > 0
        
    except Exception as e:
        print(f"ERROR during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_pipeline()
    if success:
        print("\\nSUCCESS: Complete pipeline test PASSED")
    else:
        print("\\nERROR: Complete pipeline test FAILED")
    
    # Clean up
    try:
        os.remove(__file__)
        print("Test file cleaned up")
    except:
        pass