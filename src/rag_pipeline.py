"""
RAG Pipeline - Improved version with better response generation

This file processes documents and enables semantic search with intelligent responses.
Think of this as a smart librarian system that can actually understand and answer questions.
"""

import os
import sys
from typing import List, Dict, Any
from pathlib import Path

# Configure Unicode support for Windows
if sys.platform.startswith('win'):
    # Set UTF-8 encoding for Windows
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # Handle stdout encoding issues
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except:
            pass

# Safe print function for Unicode characters
def safe_print(text):
    """Print text safely, handling Unicode encoding issues"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback to ASCII version
        ascii_text = text.encode('ascii', 'replace').decode('ascii')
        print(ascii_text)

import os
import sys
from typing import List, Dict, Any
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Use absolute imports instead of relative imports
from document_processor import DocumentProcessor
from text_chunker import TextChunker
from embedding_manager import EmbeddingManager, VectorDatabase

# For intelligent response generation
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    safe_print("⚠️ OpenAI not installed. Install with: pip install openai")

class RAGPipeline:
    """
    Complete RAG pipeline that processes documents and enables semantic search.
    
    Think of this as a smart librarian system that:
    1. Reads and understands documents
    2. Organizes them by meaning
    3. Quickly finds relevant information when asked
    4. Provides intelligent, well-formatted answers
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 overlap: int = 200,
                 embedding_model: str = "local"):
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.text_chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        self.embedding_manager = EmbeddingManager(embedding_model=embedding_model)
        
        # Initialize vector database
        self.vector_db = VectorDatabase(
            dimension=self.embedding_manager.embedding_dimension
        )
        
        # Initialize OpenAI client for intelligent response generation
        self.openai_client = None
        if OpenAI:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                try:
                    self.openai_client = OpenAI(api_key=api_key)
                    safe_print(f"✅ OpenAI client initialized for intelligent response generation")
                except Exception as e:
                    safe_print(f"⚠️ Warning: Could not initialize OpenAI client: {e}")
            else:
                safe_print("⚠️ No OpenAI API key found in .env file")
        
        # Track processed documents
        self.processed_documents = {}
        
        print(f"✅ RAG Pipeline initialized with {embedding_model} embeddings")
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single document through the complete pipeline.
        
        Like feeding a book to your smart librarian - it reads, understands,
        and files it away for future reference.
        """
        try:
            print(f"📄 Processing document: {os.path.basename(file_path)}")
            
            # Step 1: Extract text from document
            doc_result = self.document_processor.process_file(file_path)
            
            if doc_result['status'] != 'success':
                return {
                    'success': False,
                    'error': f"Failed to extract text: {doc_result.get('error', 'Unknown error')}"
                }
            
            # Step 2: Chunk the text
            chunks = self.text_chunker.chunk_text(
                doc_result['content'],
                source_metadata=doc_result['metadata']
            )
            
            if not chunks:
                return {
                    'success': False,
                    'error': 'No chunks were created from the document'
                }
            
            # Step 3: Create embeddings
            embeddings = []
            chunk_metadata = []
            
            for chunk in chunks:
                try:
                    embedding = self.embedding_manager.create_embedding(chunk.content)
                    embeddings.append(embedding)
                    
                    # Combine chunk and document metadata
                    metadata = {
                        'chunk_id': chunk.chunk_id,
                        'content': chunk.content,
                        'file_path': file_path,
                        'filename': doc_result['file_info']['filename'],
                        'chunk_metadata': chunk.metadata,
                        'document_metadata': doc_result['metadata']
                    }
                    chunk_metadata.append(metadata)
                except Exception as chunk_error:
                    print(f"⚠️ Warning: Failed to process chunk: {str(chunk_error)}")
                    continue
            
            # Step 4: Store in vector database
            if embeddings:
                self.vector_db.add_vectors(embeddings, chunk_metadata)
            
            # Track processed document
            self.processed_documents[file_path] = {
                'chunks_count': len(chunks),
                'processing_time': doc_result.get('processing_time', 0),
                'file_info': doc_result['file_info']
            }
            
            result = {
                'success': True,
                'chunks_created': len(chunks),
                'embeddings_created': len(embeddings),
                'filename': os.path.basename(file_path)
            }
            
            print(f"✅ Successfully processed {len(embeddings)}/{len(chunks)} chunks")
            return result
            
        except Exception as e:
            print(f"❌ Error processing document: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks based on query.
        
        Like asking your librarian to find books related to a specific topic.
        """
        try:
            print(f"🔍 Searching for: '{query}' (top {top_k} results)")
            
            # Create query embedding
            query_embedding = self.embedding_manager.create_embedding(query)
            
            # Search vector database
            results = self.vector_db.search(query_embedding, top_k)
            
            # Format results for consistency
            formatted_results = []
            for result in results:
                formatted_result = {
                    'content': result['metadata']['content'],
                    'score': result['score'],
                    'rank': result.get('rank', 0),
                    'metadata': {
                        'filename': result['metadata']['filename'],
                        'chunk_id': result['metadata']['chunk_id'],
                        'file_path': result['metadata']['file_path']
                    }
                }
                formatted_results.append(formatted_result)
            
            print(f"📚 Found {len(formatted_results)} relevant chunks")
            return formatted_results
            
        except Exception as e:
            print(f"❌ Search failed: {str(e)}")
            return []

    def generate_response(self, query: str, context_chunks: List[Dict]) -> Dict[str, Any]:
        """
        Generate a response using the retrieved context.
        
        Like having your librarian read the relevant books and give you a comprehensive
        answer with proper citations.
        """
        try:
            if not context_chunks:
                return {
                    'answer': "I couldn't find any relevant information in the documents to answer your question.",
                    'confidence': 0.0,
                    'sources_used': 0
                }
            
            # Use OpenAI if available, otherwise fallback
            if self.openai_client:
                return self._generate_openai_response(query, context_chunks)
            else:
                return self._generate_fallback_response(query, context_chunks)
                
        except Exception as e:
            print(f"❌ Response generation failed: {str(e)}")
            return {
                'answer': f"I apologize, but I encountered an error while generating a response to '{query}'. Please try rephrasing your question.",
                'confidence': 0.0,
                'sources_used': 0
            }
    
    def _generate_openai_response(self, query: str, context_chunks: List[Dict]) -> Dict[str, Any]:
        """Generate intelligent response using OpenAI GPT."""
        
        # Prepare context from chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            filename = chunk.get('metadata', {}).get('filename', 'Unknown')
            content = chunk.get('content', '')
            score = chunk.get('score', 0.0)
            
            # Clean up content
            clean_content = content.replace('\n', ' ').replace('\t', ' ').replace('  ', ' ').strip()
            
            context_parts.append(f"Document {i} (from {filename}, relevance: {score:.3f}):\n{clean_content}")
        
        context_text = "\n\n".join(context_parts)
        
        # Create a comprehensive prompt for better responses
        prompt = f"""You are a helpful AI assistant that answers questions based on provided documents. 

Question: {query}

Available context from documents:
{context_text}

Instructions:
- Answer the question based ONLY on the information provided in the context above
- If the context doesn't contain enough information to answer the question, say so clearly
- Be specific and detailed in your answer
- When referencing information, mention which document it came from
- If multiple documents provide related information, synthesize them in your response
- Use a clear, professional tone
- Format your response with proper headings and bullet points when appropriate

Answer:"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for factual responses
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Calculate confidence based on relevance scores
            avg_score = sum(chunk.get('score', 0) for chunk in context_chunks) / len(context_chunks)
            confidence = min(avg_score * 1.3, 1.0)  # Boost for OpenAI responses
            
            return {
                'answer': answer,
                'confidence': confidence,
                'sources_used': len(context_chunks),
                'generation_method': 'openai_gpt'
            }
            
        except Exception as e:
            print(f"⚠️ OpenAI generation failed: {e}")
            print("Falling back to basic response generation...")
            return self._generate_fallback_response(query, context_chunks)
    
    def _generate_fallback_response(self, query: str, context_chunks: List[Dict]) -> Dict[str, Any]:
        """Generate response using basic text processing when OpenAI is not available."""
        
        # Create a more intelligent response by analyzing the content
        answer_parts = []
        source_info = []
        
        # Analyze each chunk for relevant information
        for i, chunk in enumerate(context_chunks, 1):
            filename = chunk.get('metadata', {}).get('filename', 'Unknown')
            content = chunk.get('content', '')
            score = chunk.get('score', 0.0)
            
            # Add source information
            source_info.append(f"Source {i}: {filename} (relevance: {score:.3f})")
            
            # Extract relevant content (not just truncate)
            if len(content) > 400:
                # Try to find the most relevant part based on the query
                query_lower = query.lower()
                content_lower = content.lower()
                
                # Find the best excerpt
                best_start = 0
                best_score = 0
                
                # Look for query terms in the content
                query_words = query_lower.split()
                for word in query_words:
                    if word in content_lower:
                        pos = content_lower.find(word)
                        # Score based on how early the word appears
                        word_score = 1.0 / (pos + 1) * 1000
                        if word_score > best_score:
                            best_score = word_score
                            best_start = max(0, pos - 100)
                
                # Extract excerpt around the best position
                excerpt = content[best_start:best_start + 400]
                if best_start > 0:
                    excerpt = "..." + excerpt
                if best_start + 400 < len(content):
                    excerpt = excerpt + "..."
                    
                # Clean up the excerpt for better readability
                excerpt = excerpt.replace('\n', ' ').replace('\t', ' ').replace('  ', ' ').strip()
            else:
                excerpt = content.replace('\n', ' ').replace('\t', ' ').replace('  ', ' ').strip()
            
            answer_parts.append(f"**From {filename}:**\n{excerpt}")
        
        # Combine into a coherent answer
        if len(answer_parts) == 1:
            main_answer = f"**Based on your question '{query}', here's what I found:**\n\n{answer_parts[0]}"
        else:
            main_answer = f"**Based on your question '{query}', I found relevant information from {len(answer_parts)} sources:**\n\n"
            main_answer += "\n\n".join(answer_parts)
        
        # Add source summary
        main_answer += f"\n\n**📚 Information Sources:** {', '.join(source_info)}"
        
        # Calculate confidence based on relevance scores
        avg_score = sum(chunk.get('score', 0) for chunk in context_chunks) / len(context_chunks)
        confidence = min(avg_score * 1.0, 0.8)  # Cap fallback confidence at 0.8
        
        return {
            'answer': main_answer,
            'confidence': confidence,
            'sources_used': len(context_chunks),
            'generation_method': 'fallback'
        }
    
    def clear_documents(self):
        """
        Clear all processed documents and vector database.
        
        Like emptying your entire library and starting fresh.
        """
        try:
            print("🗑️ Clearing all documents and vector database...")
            
            # Clear vector database
            self.vector_db.clear()
            
            # Clear processed documents tracking
            self.processed_documents = {}
            
            print("✅ All documents cleared successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to clear documents: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.
        
        Like checking the status of your library system.
        """
        try:
            return {
                'processed_documents': len(self.processed_documents),
                'total_chunks': self.vector_db.get_stats().get('total_vectors', 0),
                'embedding_stats': self.embedding_manager.get_stats(),
                'database_stats': self.vector_db.get_stats()
            }
        except Exception as e:
            return {
                'processed_documents': len(self.processed_documents),
                'total_chunks': 0,
                'error': str(e)
            }


# Test function
if __name__ == "__main__":
    """
    This runs when you execute this file directly.
    
    Like testing your library system before opening to the public!
    """
    print("🧪 Testing Improved RAG Pipeline...")
    print("="*50)
    
    try:
        # Test with local embeddings
        pipeline = RAGPipeline(
            chunk_size=500,
            overlap=100,
            embedding_model="local"
        )
        
        stats = pipeline.get_stats()
        print(f"📊 Pipeline Stats: {stats}")
        print("✅ RAG Pipeline test completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        print("Make sure all component files exist in the src folder")
