#!/usr/bin/env python3
"""
Basic Functionality Test - No Dependencies Required

This script tests the core functionality that should work without heavy ML dependencies.
It focuses on validating the system architecture and file handling.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def test_src_imports():
    """Test that all our src modules can be imported."""
    print("\n📦 Testing Source Module Imports...")
    
    modules_to_test = [
        'document_processor',
        'text_chunker', 
        'embedding_manager',
        'rag_system',
        'colpali_retriever',
        'visual_document_processor',
        'cross_encoder_reranker',
        'salesforce_connector'
    ]
    
    results = {}
    
    for module_name in modules_to_test:
        try:
            module = __import__(module_name)
            print(f"✅ {module_name}")
            results[module_name] = True
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            results[module_name] = False
        except Exception as e:
            print(f"⚠️ {module_name}: {e}")
            results[module_name] = False
    
    return results

def test_class_instantiation():
    """Test that key classes can be instantiated without heavy dependencies."""
    print("\n🏗️ Testing Class Instantiation...")
    
    results = {}
    
    # Test DocumentProcessor
    try:
        from document_processor import DocumentProcessor
        doc_processor = DocumentProcessor()
        print("✅ DocumentProcessor created")
        results['DocumentProcessor'] = True
    except Exception as e:
        print(f"❌ DocumentProcessor: {e}")
        results['DocumentProcessor'] = False
    
    # Test TextChunker
    try:
        from text_chunker import TextChunker
        chunker = TextChunker(chunk_size=500, overlap=100)
        print("✅ TextChunker created")
        results['TextChunker'] = True
    except Exception as e:
        print(f"❌ TextChunker: {e}")
        results['TextChunker'] = False
    
    # Test basic VisualDocumentProcessor creation (should handle missing deps gracefully)
    try:
        from visual_document_processor import VisualDocumentProcessor
        config = {'colpali_model': 'test-model', 'cache_dir': 'cache/embeddings'}
        processor = VisualDocumentProcessor(config)
        print("✅ VisualDocumentProcessor created (may not be loaded)")
        results['VisualDocumentProcessor'] = True
    except Exception as e:
        print(f"❌ VisualDocumentProcessor: {e}")
        results['VisualDocumentProcessor'] = False
    
    return results

def test_file_handling():
    """Test basic file handling capabilities."""
    print("\n📁 Testing File Handling...")
    
    results = {}
    
    # Test creating temp files
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
            tmp_file.write(b"Test content for document processing")
            temp_path = tmp_file.name
        
        print(f"✅ Temporary file created: {temp_path}")
        
        # Test reading the file
        with open(temp_path, 'r') as f:
            content = f.read()
        
        print(f"✅ File read successfully: {len(content)} characters")
        
        # Cleanup
        os.unlink(temp_path)
        print("✅ File cleanup successful")
        
        results['file_handling'] = True
        
    except Exception as e:
        print(f"❌ File handling failed: {e}")
        results['file_handling'] = False
    
    return results

def test_text_processing():
    """Test basic text processing without heavy ML dependencies."""
    print("\n📝 Testing Text Processing...")
    
    results = {}
    
    try:
        from text_chunker import TextChunker
        
        # Create sample text
        sample_text = """
        This is a test document for validating text processing capabilities.
        
        It contains multiple paragraphs to test the chunking functionality.
        The text chunker should be able to split this into smaller pieces
        while maintaining context and avoiding awkward breaks.
        
        This is particularly important for RAG systems where chunk quality
        directly impacts retrieval performance and answer generation quality.
        """
        
        chunker = TextChunker(chunk_size=200, overlap=50)
        
        # Test chunking
        chunks = chunker.chunk_text(sample_text, metadata={'filename': 'test.txt'})
        
        print(f"✅ Text chunked into {len(chunks)} pieces")
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get('text', '')
            print(f"   Chunk {i+1}: {len(chunk_text)} chars")
        
        results['text_chunking'] = len(chunks) > 0
        
    except Exception as e:
        print(f"❌ Text processing failed: {e}")
        results['text_chunking'] = False
    
    return results

def test_document_processing():
    """Test document processing with simple text files."""
    print("\n📄 Testing Document Processing...")
    
    results = {}
    
    try:
        from document_processor import DocumentProcessor
        
        # Create a simple text file for testing
        test_content = "This is a simple test document for validating document processing."
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt") as tmp_file:
            tmp_file.write(test_content)
            temp_path = tmp_file.name
        
        processor = DocumentProcessor()
        
        # Test processing
        result = processor.process_file(temp_path)
        
        print(f"✅ Document processed")
        print(f"   Success: {result.get('success', False)}")
        print(f"   Content length: {len(result.get('content', ''))}")
        
        # Cleanup
        os.unlink(temp_path)
        
        results['document_processing'] = result.get('success', False)
        
    except Exception as e:
        print(f"❌ Document processing failed: {e}")
        results['document_processing'] = False
    
    return results

def test_rag_system_creation():
    """Test creating RAG system components."""
    print("\n🤖 Testing RAG System Creation...")
    
    results = {}
    
    try:
        from rag_system import RAGSystem
        
        # Create basic config
        config = {
            'chunk_size': 500,
            'chunk_overlap': 100,
            'embedding_model': 'local',
            'max_retrieved_chunks': 5
        }
        
        # This will likely fail on embedding manager, but we can test the structure
        try:
            rag = RAGSystem(config)
            print("✅ RAGSystem created successfully")
            results['rag_system'] = True
        except Exception as rag_error:
            print(f"⚠️ RAGSystem creation failed (expected): {rag_error}")
            print("   This is expected without ML dependencies")
            results['rag_system'] = False
        
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        results['rag_system'] = False
    
    return results

def test_streamlit_app_structure():
    """Test that the main Streamlit app can be imported."""
    print("\n🌟 Testing Streamlit App Structure...")
    
    results = {}
    
    try:
        # Test if the main app file exists and has correct structure
        app_path = "streamlit_rag_app.py"
        
        if os.path.exists(app_path):
            print(f"✅ Main app file exists: {app_path}")
            
            # Read and check for key components
            with open(app_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            key_components = [
                'SimpleRAGOrchestrator',
                'st.file_uploader',
                'st.form_submit_button',
                'ColPaliRetriever',
                'SalesforceConnector'
            ]
            
            found_components = []
            for component in key_components:
                if component in content:
                    found_components.append(component)
                    print(f"   ✅ {component}")
                else:
                    print(f"   ❌ {component}")
            
            results['app_structure'] = len(found_components) >= 3
            
        else:
            print(f"❌ Main app file not found: {app_path}")
            results['app_structure'] = False
        
    except Exception as e:
        print(f"❌ App structure test failed: {e}")
        results['app_structure'] = False
    
    return results

def main():
    """Run all basic functionality tests."""
    print("🚀 Basic Functionality Testing - No Heavy Dependencies")
    print("=" * 60)
    
    all_results = {}
    
    # Test 1: Module Imports
    print("\n" + "="*20 + " TEST 1: MODULE IMPORTS " + "="*20)
    all_results.update(test_src_imports())
    
    # Test 2: Class Instantiation
    print("\n" + "="*20 + " TEST 2: CLASS INSTANTIATION " + "="*20)
    all_results.update(test_class_instantiation())
    
    # Test 3: File Handling
    print("\n" + "="*20 + " TEST 3: FILE HANDLING " + "="*20)
    all_results.update(test_file_handling())
    
    # Test 4: Text Processing
    print("\n" + "="*20 + " TEST 4: TEXT PROCESSING " + "="*20)
    all_results.update(test_text_processing())
    
    # Test 5: Document Processing  
    print("\n" + "="*20 + " TEST 5: DOCUMENT PROCESSING " + "="*20)
    all_results.update(test_document_processing())
    
    # Test 6: RAG System Creation
    print("\n" + "="*20 + " TEST 6: RAG SYSTEM CREATION " + "="*20)
    all_results.update(test_rag_system_creation())
    
    # Test 7: App Structure
    print("\n" + "="*20 + " TEST 7: APP STRUCTURE " + "="*20)
    all_results.update(test_streamlit_app_structure())
    
    # Summary
    print("\n" + "="*20 + " TEST SUMMARY " + "="*20)
    
    passed = sum(all_results.values())
    total = len(all_results)
    
    for test_name, passed_test in all_results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed >= total * 0.7:  # 70% threshold
        print("🎉 Most tests passed! Core architecture is sound.")
        print("💡 Next step: Test with dependencies in virtual environment.")
    else:
        print("⚠️ Many tests failed. Core architecture needs fixes.")
    
    return passed >= total * 0.7

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)