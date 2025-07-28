#!/usr/bin/env python3
"""
End-to-end test to verify document processing and query functionality.
This tests the complete pipeline: document upload -> processing -> embedding -> storage -> query.
"""

import sys
import os
import logging
import tempfile

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_test_documents():
    """Create test documents for processing."""
    print("Creating test documents...")
    
    # Create temporary directory for test documents
    temp_dir = tempfile.mkdtemp()
    
    # Create a simple text document
    txt_content = """
    # Artificial Intelligence Overview
    
    Artificial Intelligence (AI) is the simulation of human intelligence in machines 
    that are programmed to think like humans and mimic their actions.
    
    ## Types of AI
    1. Narrow AI: AI designed for specific tasks
    2. General AI: AI with human-like cognitive abilities
    3. Super AI: AI that exceeds human intelligence
    
    ## Machine Learning
    Machine Learning is a subset of AI that enables computers to learn without 
    explicit programming. It uses statistical techniques to give computers the 
    ability to progressively improve performance on a specific task with data.
    
    ## Applications
    AI is used in various fields including:
    - Healthcare: Medical diagnosis and drug discovery
    - Finance: Fraud detection and algorithmic trading
    - Transportation: Autonomous vehicles and traffic optimization
    - Entertainment: Content recommendation and game playing
    """
    
    # Create CSV document
    csv_content = """Company,Industry,Revenue,Employees,Founded
Apple,Technology,365000000000,154000,1976
Microsoft,Technology,168000000000,181000,1975
Google,Technology,257000000000,139995,1998
Amazon,E-commerce,469822000000,1541000,1994
Tesla,Automotive,81462000000,127855,2003
Netflix,Entertainment,29697000000,11300,1997"""
    
    # Write test files
    txt_path = os.path.join(temp_dir, "ai_overview.txt")
    csv_path = os.path.join(temp_dir, "companies.csv")
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(txt_content)
    
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(csv_content)
    
    print(f"Created test documents:")
    print(f"  - Text: {txt_path}")
    print(f"  - CSV: {csv_path}")
    
    return [txt_path, csv_path]

def test_document_processing():
    """Test document processing and query functionality."""
    print("\nTesting Document Processing and Query")
    print("=" * 50)
    
    try:
        from rag_system import RAGSystem
        
        # Create RAG system
        config = {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'embedding_model': 'openai',
            'max_retrieved_chunks': 5
        }
        
        print("Creating RAG system...")
        rag = RAGSystem(config)
        
        # Check initial state
        print(f"Initial state: initialized = {rag.is_initialized}")
        
        # Create test documents
        test_docs = create_test_documents()
        
        # Process documents
        print(f"\nProcessing {len(test_docs)} documents...")
        results = rag.add_documents(test_docs)
        
        print(f"Processing Results:")
        print(f"  - Successful: {len(results['successful'])}")
        print(f"  - Failed: {len(results['failed'])}")
        print(f"  - Total chunks: {results['total_chunks']}")
        print(f"  - Processing time: {results['processing_time']:.2f}s")
        
        if results['failed']:
            print(f"  - Failed documents:")
            for failed in results['failed']:
                print(f"    * {failed['path']}: {failed['error']}")
        
        # Check system status after processing
        system_info = rag.get_system_info()
        print(f"\nSystem Status After Processing:")
        print(f"  - Initialized: {system_info['is_initialized']}")
        print(f"  - Total documents: {system_info['total_documents']}")
        print(f"  - Total chunks: {system_info['total_chunks']}")
        
        if not system_info['is_initialized']:
            print("ERROR: System not initialized after document processing!")
            return False
        
        # Test queries
        print(f"\nTesting Queries:")
        
        # Query 1: About AI
        print(f"\nQuery 1: What is artificial intelligence?")
        response1 = rag.query("What is artificial intelligence?")
        print(f"  - Success: {response1['success']}")
        if response1['success']:
            print(f"  - Answer length: {len(response1['answer'])} chars")
            print(f"  - Sources found: {len(response1['sources'])}")
            print(f"  - Confidence: {response1.get('confidence', 0):.3f}")
            print(f"  - Answer preview: {response1['answer'][:200]}...")
        else:
            print(f"  - Error: {response1.get('error', 'Unknown error')}")
        
        # Query 2: About companies (CSV data)
        print(f"\nQuery 2: Which company has the highest revenue?")
        response2 = rag.query("Which company has the highest revenue?")
        print(f"  - Success: {response2['success']}")
        if response2['success']:
            print(f"  - Answer length: {len(response2['answer'])} chars")
            print(f"  - Sources found: {len(response2['sources'])}")
            print(f"  - Confidence: {response2.get('confidence', 0):.3f}")
            print(f"  - Answer preview: {response2['answer'][:200]}...")
        else:
            print(f"  - Error: {response2.get('error', 'Unknown error')}")
        
        # Query 3: Mixed query
        print(f"\nQuery 3: How is AI used in healthcare?")
        response3 = rag.query("How is AI used in healthcare?")
        print(f"  - Success: {response3['success']}")
        if response3['success']:
            print(f"  - Answer length: {len(response3['answer'])} chars")
            print(f"  - Sources found: {len(response3['sources'])}")
            print(f"  - Confidence: {response3.get('confidence', 0):.3f}")
            print(f"  - Answer preview: {response3['answer'][:200]}...")
        else:
            print(f"  - Error: {response3.get('error', 'Unknown error')}")
        
        # Cleanup
        print(f"\nCleaning up test documents...")
        for doc_path in test_docs:
            try:
                os.remove(doc_path)
                os.rmdir(os.path.dirname(doc_path))
            except:
                pass
        
        # Check if all queries succeeded
        all_queries_successful = all([
            response1['success'], 
            response2['success'], 
            response3['success']
        ])
        
        if all_queries_successful:
            print(f"\nSUCCESS: All end-to-end tests passed!")
            return True
        else:
            print(f"\nWARNING: Some queries failed")
            return False
        
    except Exception as e:
        print(f"ERROR: End-to-end test failed: {str(e)}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("Starting End-to-End RAG System Test")
    print("=" * 60)
    
    # Test document processing and queries
    success = test_document_processing()
    
    print(f"\nFinal Result: {'PASS' if success else 'FAIL'}")
    
    if success:
        print("The RAG system is working correctly end-to-end!")
        print("Both dimension fixes and query functionality are operational.")
    else:
        print("Some issues were detected. Check the logs above for details.")
        sys.exit(1)