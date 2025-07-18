"""
Comprehensive RAG System Testing Suite

This file combines all your testing functions in one place.
Run this to test your entire RAG system health.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.embedding_manager import EmbeddingManager, VectorDatabase

def monitor_embedding_quality(embedder: EmbeddingManager):
    """Monitor embedding quality metrics - from the guide."""
    
    # Test with known similar/dissimilar pairs
    test_pairs = [
        ("password reset", "forgot password", True),  # Should be similar
        ("login help", "cooking recipe", False),      # Should be different
        ("customer service", "client support", True), # Should be similar
        ("data analysis", "banana recipe", False),    # Should be different
    ]
    
    print("🔍 EMBEDDING QUALITY MONITOR")
    print("=" * 50)
    
    for text1, text2, should_be_similar in test_pairs:
        emb1 = embedder.create_embedding(text1)
        emb2 = embedder.create_embedding(text2)
        similarity = embedder.calculate_similarity(emb1, emb2)
        
        print(f"'{text1}' vs '{text2}': {similarity:.3f}")
        
        if should_be_similar and similarity < 0.7:
            print("⚠️  Warning: Similar texts have low similarity")
        elif not should_be_similar and similarity > 0.5:
            print("⚠️  Warning: Different texts have high similarity")

def test_basic_embedding_functionality():
    """Test basic embedding creation and similarity."""
    
    print("\n🧪 BASIC FUNCTIONALITY TEST")
    print("=" * 50)
    
    try:
        # Initialize with local model (free) - using new default
        embedder = EmbeddingManager()  # Now defaults to local
        
        # Test single embedding creation
        text = "This is a test sentence"
        embedding = embedder.create_embedding(text)
        
        print(f"✅ Created embedding with dimension: {len(embedding)}")
        print(f"✅ Embedding type: {type(embedding)}")
        
        # Test similarity calculation
        text1 = "Hello world"
        text2 = "Hi universe"
        
        emb1 = embedder.create_embedding(text1)
        emb2 = embedder.create_embedding(text2)
        similarity = embedder.calculate_similarity(emb1, emb2)
        
        print(f"✅ Similarity between '{text1}' and '{text2}': {similarity:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {str(e)}")
        return False

def test_vector_database():
    """Test vector database operations."""
    
    print("\n🗄️ VECTOR DATABASE TEST")
    print("=" * 50)
    
    try:
        # Initialize components
        embedder = EmbeddingManager()  # Now defaults to local
        vector_db = VectorDatabase(dimension=embedder.embedding_dimension)
        
        # Test data
        test_texts = [
            "Python programming tutorial",
            "JavaScript web development",
            "Database management with SQL",
            "Machine learning basics"
        ]
        
        # Create embeddings and metadata
        embeddings = []
        metadata = []
        
        for i, text in enumerate(test_texts):
            emb = embedder.create_embedding(text)
            embeddings.append(emb)
            metadata.append({
                'content': text,
                'id': i,
                'category': 'programming'
            })
        
        # Add to database
        vector_db.add_vectors(embeddings, metadata)
        print(f"✅ Added {len(embeddings)} vectors to database")
        
        # Test search
        query = "web development programming"
        query_emb = embedder.create_embedding(query)
        results = vector_db.search(query_emb, top_k=2)
        
        print(f"✅ Search query: '{query}'")
        print("📋 Top results:")
        for result in results:
            print(f"  Score: {result['score']:.3f} - {result['metadata']['content']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector database test failed: {str(e)}")
        return False

def run_all_tests():
    """Run all embedding tests."""
    
    print("🚀 STARTING COMPREHENSIVE EMBEDDING TESTS")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_embedding_functionality),
        ("Vector Database", test_vector_database),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name} Test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {str(e)}")
            results[test_name] = False
    
    # Run quality monitoring if basic tests pass
    if results.get("Basic Functionality", False):
        print(f"\n🔬 Running Embedding Quality Monitor...")
        try:
            embedder = EmbeddingManager()  # Now defaults to local
            monitor_embedding_quality(embedder)
            results["Quality Monitor"] = True
        except Exception as e:
            print(f"❌ Quality monitor failed: {str(e)}")
            results["Quality Monitor"] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    overall_success = all(results.values())
    
    if overall_success:
        print("\n🎉 ALL TESTS PASSED! Your embedding system is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
    
    return overall_success

if __name__ == "__main__":
    """Run when executed directly: python tests/test_comprehensive.py"""
    
    print("Welcome to RAG System Testing!")
    print("This will test your embedding system comprehensively.\n")
    
    # Check if required packages are installed
    try:
        import sentence_transformers
        import faiss
        print("✅ Required packages detected")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("💡 Install with: pip install sentence-transformers faiss-cpu")
        sys.exit(1)
    
    # Run all tests
    success = run_all_tests()
    
    if success:
        print("\n🚀 Your embedding system is ready for the next RAG components!")
    else:
        print("\n🔧 Fix the issues above before proceeding.")
