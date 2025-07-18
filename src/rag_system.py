"""
RAG System Orchestrator - The Master Chef of Your RAG Kitchen

This file is like a master chef who knows how to coordinate all the kitchen appliances
(document_processor, embedding_manager, etc.) to create a complete RAG meal.

Think of it this way:
- Your individual components are like kitchen appliances (blender, oven, mixer)
- This rag_system.py is the master chef who knows the recipe
- The 05_Complete_RAG_System.py is the customer who just wants to order food

Real-life analogy: 
Imagine you're running a smart library system:
1. The document processor is like the librarian who sorts and catalogs books
2. The text chunker is like someone who creates detailed index cards
3. The embedding manager is like a system that understands the meaning of each card
4. This rag_system ties it all together - like the head librarian who can answer any question
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add the current directory to Python path for imports
# This is like telling Python where to find your kitchen appliances
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import all your RAG components that you've built
# We'll use absolute imports to avoid the relative import issue
try:
    from rag_pipeline import RAGPipeline
    from document_processor import DocumentProcessor
    from text_chunker import TextChunker
    from embedding_manager import EmbeddingManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all component files exist in the src folder:")
    print("  - rag_pipeline.py")
    print("  - document_processor.py") 
    print("  - text_chunker.py")
    print("  - embedding_manager.py")
    sys.exit(1)

# Set up logging to track what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """
    The complete RAG system that brings everything together.
    
    Like a smart assistant that can:
    1. Read and understand documents
    2. Remember what it learned
    3. Answer questions based on what it knows
    4. Show you where it found the information
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RAG system with configuration.
        
        Think of this like setting up your smart assistant's personality:
        - How big chunks of text should it remember?
        - Which brain (embedding model) should it use?
        - How many pieces of information should it consider when answering?
        """
        self.config = config
        self.is_initialized = False
        
        # Initialize the core RAG pipeline (your main engine)
        logger.info("🔧 Setting up RAG pipeline...")
        try:
            self.rag_pipeline = RAGPipeline(
                chunk_size=config.get('chunk_size', 800),
                overlap=config.get('chunk_overlap', 150),
                embedding_model=config.get('embedding_model', 'local')
            )
            logger.info("✅ RAG pipeline created successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to create RAG pipeline: {e}")
            raise
        
        # Track what documents we've processed
        self.processed_documents = []
        
        logger.info("✅ RAG system created successfully!")
    
    def add_documents(self, document_paths: List[str]) -> Dict[str, Any]:
        """
        Add documents to the RAG system.
        
        Like feeding books to your smart librarian:
        - Give it a list of document paths
        - It reads, understands, and remembers them
        - Returns a summary of what it learned
        """
        logger.info(f"📚 Processing {len(document_paths)} documents...")
        
        results = {
            'successful': [],
            'failed': [],
            'total_chunks': 0,
            'processing_time': 0
        }
        
        import time
        start_time = time.time()
        
        for doc_path in document_paths:
            try:
                logger.info(f"📄 Processing: {os.path.basename(doc_path)}")
                
                # Process the document through your pipeline
                doc_results = self.rag_pipeline.process_document(doc_path)
                
                if doc_results.get('success', False):
                    results['successful'].append({
                        'path': doc_path,
                        'chunks': doc_results.get('chunks_created', 0),
                        'filename': os.path.basename(doc_path)
                    })
                    results['total_chunks'] += doc_results.get('chunks_created', 0)
                    self.processed_documents.append(doc_path)
                else:
                    results['failed'].append({
                        'path': doc_path,
                        'error': doc_results.get('error', 'Unknown error')
                    })
                    
            except Exception as e:
                logger.error(f"❌ Failed to process {doc_path}: {str(e)}")
                results['failed'].append({
                    'path': doc_path,
                    'error': str(e)
                })
        
        results['processing_time'] = time.time() - start_time
        self.is_initialized = len(results['successful']) > 0
        
        logger.info(f"✅ Processed {len(results['successful'])} documents successfully")
        logger.info(f"📊 Created {results['total_chunks']} searchable chunks")
        
        return results
    
    def query(self, question: str, max_chunks: Optional[int] = None) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources.
        
        Like asking your smart librarian a question:
        - You ask something
        - It searches through all the books it has read
        - It gives you an answer and tells you which books it used
        """
        if not self.is_initialized:
            return {
                'success': False,
                'error': 'No documents have been processed yet. Please add documents first.'
            }
        
        logger.info(f"🔍 Searching for: {question}")
        
        try:
            # Use your RAG pipeline to find relevant information
            max_chunks = max_chunks or self.config.get('max_retrieved_chunks', 5)
            
            # Search for relevant chunks
            search_results = self.rag_pipeline.search(question, top_k=max_chunks)
            
            if not search_results:
                return {
                    'success': True,
                    'answer': "I couldn't find any relevant information in the documents to answer your question.",
                    'sources': [],
                    'confidence': 0.0
                }
            
            # Generate answer using the found information
            response = self.rag_pipeline.generate_response(question, search_results)
            
            return {
                'success': True,
                'question': question,
                'answer': response.get('answer', ''),
                'sources': self._format_sources(search_results),
                'confidence': response.get('confidence', 0.0),
                'chunks_used': len(search_results)
            }
            
        except Exception as e:
            logger.error(f"❌ Query failed: {str(e)}")
            return {
                'success': False,
                'error': f"Query processing failed: {str(e)}"
            }
    
    def _format_sources(self, search_results: List[Dict]) -> List[Dict]:
        """
        Format the source information in a user-friendly way.
        
        Like creating footnotes that show exactly where information came from.
        """
        sources = []
        
        for i, result in enumerate(search_results, 1):
            metadata = result.get('metadata', {})
            sources.append({
                'source_number': i,
                'filename': metadata.get('filename', 'Unknown'),
                'page': metadata.get('page', 'Unknown'),
                'chunk_text': result.get('content', '')[:200] + '...',  # First 200 chars
                'relevance_score': round(result.get('score', 0.0), 3)
            })
        
        return sources
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the current state of the RAG system.
        
        Like checking the status of your smart librarian:
        - How many books has it read?
        - What settings is it using?
        - Is it ready to answer questions?
        """
        total_chunks = 0
        try:
            if hasattr(self.rag_pipeline, 'vector_db') and hasattr(self.rag_pipeline.vector_db, 'total_chunks'):
                total_chunks = self.rag_pipeline.vector_db.total_chunks
        except:
            total_chunks = 0
            
        return {
            'is_initialized': self.is_initialized,
            'total_documents': len(self.processed_documents),
            'config': self.config,
            'processed_documents': [os.path.basename(doc) for doc in self.processed_documents],
            'total_chunks': total_chunks
        }
    
    def clear_documents(self) -> bool:
        """
        Clear all processed documents and start fresh.
        
        Like telling your librarian to forget everything and start over.
        """
        try:
            logger.info("🗑️ Clearing all documents...")
            
            # Clear the vector database
            if hasattr(self.rag_pipeline, 'vector_db'):
                self.rag_pipeline.vector_db.clear()
            
            # Reset tracking
            self.processed_documents = []
            self.is_initialized = False
            
            logger.info("✅ All documents cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to clear documents: {str(e)}")
            return False


def create_rag_system(config: Dict[str, Any]) -> RAGSystem:
    """
    Factory function to create a new RAG system.
    
    This is like calling a service to set up your smart librarian with specific preferences.
    
    Example usage:
    ```python
    config = {
        'chunk_size': 800,
        'chunk_overlap': 150,
        'embedding_model': 'local',  # or 'openai'
        'max_retrieved_chunks': 5
    }
    rag = create_rag_system(config)
    ```
    """
    logger.info("🏗️ Creating new RAG system...")
    
    # Validate configuration
    required_keys = ['chunk_size', 'chunk_overlap', 'embedding_model']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Create and return the system
    rag_system = RAGSystem(config)
    logger.info("🎉 RAG system ready to use!")
    
    return rag_system


def create_sample_documents(docs_dir: str) -> None:
    """
    Create sample documents for testing the RAG system.
    
    Like putting some practice books in your library for testing.
    """
    logger.info(f"📝 Creating sample documents in {docs_dir}")
    
    # Make sure the directory exists
    os.makedirs(docs_dir, exist_ok=True)
    
    # Sample AI document
    ai_content = """
    # Artificial Intelligence Overview
    
    Artificial Intelligence (AI) is the simulation of human intelligence in machines.
    
    ## Key Types of AI:
    
    1. **Narrow AI**: AI designed for specific tasks (like voice assistants)
    2. **General AI**: AI with human-like cognitive abilities (still theoretical)
    3. **Super AI**: AI that exceeds human intelligence (hypothetical)
    
    ## Machine Learning
    Machine Learning is a subset of AI that enables computers to learn without explicit programming.
    
    ### Types of Machine Learning:
    - Supervised Learning: Learning with labeled data
    - Unsupervised Learning: Finding patterns in unlabeled data
    - Reinforcement Learning: Learning through trial and error
    
    ## Applications
    AI is used in healthcare, finance, transportation, and entertainment.
    """
    
    # Sample RAG document
    rag_content = """
    # Retrieval-Augmented Generation (RAG)
    
    RAG combines information retrieval with text generation to create more accurate AI responses.
    
    ## How RAG Works:
    
    1. **Document Processing**: Break documents into chunks
    2. **Embedding**: Convert text chunks into numerical vectors
    3. **Storage**: Store vectors in a searchable database
    4. **Retrieval**: Find relevant chunks for user queries
    5. **Generation**: Use retrieved context to generate accurate responses
    
    ## Benefits of RAG:
    - Reduces hallucinations
    - Provides source citations
    - Updates knowledge without retraining
    - Cost-effective compared to fine-tuning
    
    ## RAG vs Fine-tuning:
    RAG is better for dynamic knowledge, while fine-tuning is better for style adaptation.
    """
    
    # Write sample files
    try:
        with open(os.path.join(docs_dir, "ai_overview.txt"), "w", encoding='utf-8') as f:
            f.write(ai_content)
        
        with open(os.path.join(docs_dir, "rag_explanation.txt"), "w", encoding='utf-8') as f:
            f.write(rag_content)
        
        logger.info("✅ Sample documents created successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to create sample documents: {e}")


# Example usage and testing
if __name__ == "__main__":
    """
    This runs when you execute this file directly.
    
    Like testing your kitchen equipment before opening the restaurant!
    """
    print("🧪 Testing RAG System Components...")
    print("="*50)
    
    # Test configuration
    test_config = {
        'chunk_size': 500,
        'chunk_overlap': 100,
        'embedding_model': 'local',
        'max_retrieved_chunks': 3
    }
    
    try:
        print("⚙️ Creating RAG system with test configuration...")
        rag = create_rag_system(test_config)
        
        # Test system info
        info = rag.get_system_info()
        print(f"✅ System ready: {info['is_initialized']}")
        print(f"📊 Configuration: {info['config']}")
        print(f"📚 Documents loaded: {info['total_documents']}")
        
        print("\n🎉 RAG system test completed successfully!")
        print("You can now run the complete demo with: python 05_Complete_RAG_System.py")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure all component files exist in the src folder")
        print("2. Check that all dependencies are installed")
        print("3. Verify your .env file has the correct settings")
