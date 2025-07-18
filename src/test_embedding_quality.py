"""
Embedding Quality Monitoring and Testing

This file contains utility functions to test and monitor 
the quality of your embedding system.

Think of this as a "health check" for your RAG system.
"""

from src.embedding_manager import EmbeddingManager

def monitor_embedding_quality(embedder: EmbeddingManager):
    """
    Monitor embedding quality metrics.
    
    This function tests if your embedding system can properly distinguish
    between similar and different texts. Like testing if your smart 
    filing system can tell the difference between related and unrelated documents.
    """
    
    # Test with known similar/dissimilar pairs
    test_pairs = [
        ("password reset", "forgot password", True),  # Should be similar
        ("login help", "cooking recipe", False),      # Should be different
        ("customer service", "client support", True), # Should be similar
        ("data analysis", "banana recipe", False),    # Should be different
        ("troubleshooting guide", "problem solving help", True), # Should be similar
    ]
    
    print("🔍 EMBEDDING QUALITY MONITOR")
    print("=" * 50)
    
    for text1, text2, should_be_similar in test_pairs:
        emb1 = embedder.create_embedding(text1)
        emb2 = embedder.create_embedding(text2)
        similarity = embedder.calculate_similarity(emb1, emb2)
        
        # Determine the status
        status = "✅"
        if should_be_similar and similarity < 0.7:
            status = "⚠️  Warning: Similar texts have low similarity"
        elif not should_be_similar and similarity > 0.5:
            status = "⚠️  Warning: Different texts have high similarity"
        
        print(f"'{text1}' vs '{text2}': {similarity:.3f} {status}")
    
    print("\n📊 INTERPRETATION:")
    print("- Similar texts should score above 0.7")
    print("- Different texts should score below 0.5")
    print("- Scores between 0.5-0.7 are in the 'gray area'")

def test_different_embedding_models():
    """Compare quality between different embedding models."""
    
    print("\n🏆 MODEL COMPARISON")
    print("=" * 50)
    
    # Test with both local and OpenAI models
    models_to_test = [
        ("local", "all-MiniLM-L6-v2"),
        # ("openai", "text-embedding-ada-002"),  # Uncomment if you have API key
    ]
    
    test_query = "password reset"
    test_docs = [
        "forgot password recovery",
        "login troubleshooting", 
        "cooking pasta recipe"
    ]
    
    for model_type, model_name in models_to_test:
        print(f"\n🔬 Testing {model_type} model: {model_name}")
        
        try:
            embedder = EmbeddingManager(
                embedding_model=model_type,
                model_name=model_name
            )
            
            query_emb = embedder.create_embedding(test_query)
            
            for doc in test_docs:
                doc_emb = embedder.create_embedding(doc)
                similarity = embedder.calculate_similarity(query_emb, doc_emb)
                print(f"  '{test_query}' vs '{doc}': {similarity:.3f}")
                
        except Exception as e:
            print(f"  ❌ Error with {model_type}: {str(e)}")

if __name__ == "__main__":
    """
    This runs when you execute this file directly.
    Like running: python src/test_embedding_quality.py
    """
    
    print("🚀 Starting Embedding Quality Tests...")
    
    # Test with local model (free, no API key needed)
    try:
        print("Testing with LOCAL embedding model...")
        embedder = EmbeddingManager(embedding_model="local")
        monitor_embedding_quality(embedder)
        test_different_embedding_models()
        
        print("\n✅ Quality testing completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        print("💡 Make sure you have installed: pip install sentence-transformers")
