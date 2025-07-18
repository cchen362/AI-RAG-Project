import os 
import sys
import time
from pathlib import Path

# Make sure we can import our src folder
# This is like telling Python where to find your kitchen appliances
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

# Import your RAG system (the master chef)
try:
    from src.rag_system import create_rag_system, create_sample_documents
    print("✅ Successfully imported RAG system components!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have created the rag_system.py file in your src folder")
    sys.exit(1)

def print_header(title: str):
    """Print a nice header for different sections."""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)

def print_step(step_num: int, description: str):
    """Print step information."""
    print(f"\n📍 Step {step_num}: {description}")
    print("-" * 40)

def display_processing_results(results: dict):
    """Display document processing results in a user-friendly way."""
    print(f"\n📊 Processing Results:")
    print(f"   ✅ Successfully processed: {len(results['successful'])} documents")
    print(f"   ❌ Failed to process: {len(results['failed'])} documents")
    print(f"   📦 Total chunks created: {results['total_chunks']}")
    print(f"   ⏱️ Processing time: {results['processing_time']:.2f} seconds")
    
    if results['successful']:
        print(f"\n📚 Successfully processed documents:")
        for doc in results['successful']:
            print(f"   • {doc['filename']} ({doc['chunks']} chunks)")
    
    if results['failed']:
        print(f"\n❌ Failed documents:")
        for doc in results['failed']:
            print(f"   • {os.path.basename(doc['path'])}: {doc['error']}")

def display_query_results(results: dict):
    """Display query results in a beautiful format."""
    if not results['success']:
        print(f"❌ Query failed: {results['error']}")
        return
    
    print(f"\n🤖 AI Assistant's Answer:")
    print("-" * 40)
    print(results['answer'])

    print(f"\n📚 Sources Used (Confidence: {results.get('confidence', 0)*100:.1f}%):")
    print("-" * 40)
    
    if results['sources']:
        for source in results['sources']:
            print(f"📄 Source {source['source_number']}: {source['filename']}")
            print(f"   📍 Page: {source['page']} | Relevance: {source['relevance_score']}")
            print(f"   📝 Preview: {source['chunk_text']}")
            print()
    else:
        print("   No specific sources found")

def interactive_demo(rag_system):
    """Run an interactive demo where users can ask questions."""
    print_header("Interactive RAG Demo")
    print("Ask questions about your document! Type 'quit' to exit.")

    while True:
        print("\n" + "="*50)
        user_question = input("🤔 Your question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("👋 Thanks for using the RAG system!")
            break
        
        if not user_question:
            print("Please enter a question!")
            continue
        
        print(f"\n🔍 Searching for information about: '{user_question}'")
        print("⏳ Processing...")

        # Get answer from RAG system
        start_time = time.time()
        results = rag_system.query(user_question)
        query_time = time.time() - start_time

        print(f"⚡ Query completed in {query_time:.2f} seconds")
        display_query_results(results)

def main():
    """
    Main demo function that showcases your complete RAG system.
    
    This is like the grand opening of your smart restaurant!
    """
    print_header("🚀 Welcome to Your Complete RAG System Demo!")
    print("This demo will show you everything your RAG system can do!")

    # Step 1: System Configuration
    print_step(1, "Configuring RAG System")

    # There are like the settings for your smart assistant
    config = {
        'chunk_size': 800,          # How big pieces of text to remember
        'chunk_overlap': 150,       # How much overlap between pieces
        'embedding_model': 'local', # Which brain to use (local or openai)
        'max_retrieved_chunks': 5   # How many pieces to consider for each answer
    }

    print("⚙️ RAG Configuration:")
    for key, value in config.items():
        print(f"   • {key}: {value}")

    # Step 2: Initialize RAG System
    print_step(2, "Initializing RAG System")

    try:
        rag = create_rag_system(config)
        print("✅ RAG system initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        return
    
    # Step 3: Prepare Documents
    print_step(3, "Preparing Document Collection")

    # Check if documents directory exists
    docs_dir = "data/documents"
    if not os.path.exists(docs_dir):
        print(f"📁 Creating documents directory: {docs_dir}")
        os.makedirs(docs_dir, exist_ok=True)

        # Create sample documents for demonstration
        print("📝 Creating sample documents for demo...")
        create_sample_documents(docs_dir)
        print("✅ Sample documents created!")

    # Find available documents
    doc_extensions = ['.txt', '.pdf', '.docx']
    available_docs = []

    for ext in doc_extensions:
        available_docs.extend(list(Path(docs_dir).glob(f"*{ext}")))

    if not available_docs:
        print(f"⚠️ No documents found in {docs_dir}")
        print("Please add some documents (.txt, .pdf, .docx) to test the system!")
        return
    
    print(f"📚 Found {len(available_docs)} documents:")
    for doc in available_docs:
        print(f"   • {doc.name}")

    # Step 4: Process Documents
    print_step(4, "Processing Documents into RAG System")

    print("🔄 Processing documents... This may take a moment.")

    # Conver Path objects to string for processing
    doc_paths = [str(doc) for doc in available_docs]
    results = rag.add_documents(doc_paths)

    display_processing_results(results)

    if not results['successful']:
        print("❌ No documents were processed successfully. Cannot continue with demo.")
        return
    
    # Step 5: System Information
    print_step(5, "System Status Check")

    system_info = rag.get_system_info()
    print("📊 RAG System Status:")
    print(f"   • System Ready: {'✅ Yes' if system_info['is_initialized'] else '❌ No'}")
    print(f"   • Documents Loaded: {system_info['total_documents']}")
    print(f"   • Total Searchable Chunks: {system_info['total_chunks']}")
    print(f"   • Embedding Model: {system_info['config']['embedding_model']}")

    # Step 6: Demo Quries
    print_step(6, "Testing with Sample Questions")

    # Pre-defined demo questions
    demo_questions = [
        "What is articifial intelligence?",
        "How does RAG work",
        "What are the benefits of using RAG?",
        "What is machine learning?"
    ]

    print("🧪 Running sample queries to test the system...")

    for i, question in enumerate(demo_questions, 1):
        print(f"\n🔍 Demo Query {i}: '{question}'")
        print("⏳ Processing...")

        start_time = time.time()
        results = rag.query(question)
        query_time = time.time() - start_time

        print(f"⚡ Completed in {query_time:.2f} seconds")
        display_query_results(results)

        # Small pause between queries for better readability
        time.sleep(1)

    # Step 7: Interactive Demo
    print_step(7, "Interactive Demo Mode")

    try:
        interactive_demo(rag)
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    
    # Final Summary
    print_header("🎉 Demo Complete - Summary")
    
    final_info = rag.get_system_info()
    print("✨ Your RAG system successfully demonstrated:")
    print(f"   📚 Document Processing: {final_info['total_documents']} documents")
    print(f"   🔍 Semantic Search: {final_info['total_chunks']} searchable chunks")
    print(f"   🤖 Question Answering: With source citations")
    print(f"   ⚡ Real-time Performance: Fast query responses")
    
    print("\n🎯 Next Steps:")
    print("   • Add more documents to expand knowledge")
    print("   • Try the Streamlit UI for a better interface")
    print("   • Experiment with different configuration settings")
    print("   • Integrate with enterprise systems")
    
    print("\n🙏 Thanks for exploring your RAG system!")

if __name__ == "__main__":
    """
    This runs when you execute this script directly.
    
    Like opening the doors to your restaurant for the grand opening!
    """
    try:
        main()
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("\nDebugging tips:")
        print("1. Make sure all required packages are installed")
        print("2. Check that your .env file has the correct API keys")
        print("3. Verify that all component files exist in the src folder")
        print("4. Try running the individual component tests first")
    except KeyboardInterrupt:
        print("\n\n👋 Demo stopped by user. Goodbye!")

