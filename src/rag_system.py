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
    from embedding_manager import EmbeddingManager, MultiModalVectorDatabase
    from visual_document_processor import VisualDocumentProcessor
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


class MultiModalRAGSystem:
    """
    Multi-modal RAG system supporting both text and visual retrieval modes.
    
    This allows comparison between:
    1. Traditional text-only retrieval (existing approach)
    2. Visual document retrieval using ColPali
    3. Hybrid mode combining both approaches
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize multi-modal RAG system."""
        self.config = config
        self.mode = config.get('retrieval_mode', 'text')  # 'text', 'visual', 'hybrid'
        self.is_initialized = False
        
        # Track processed documents
        self.processed_documents = []
        
        # Storage for visual embeddings (to avoid re-reading temp files)
        self.visual_embeddings = {}
        
        # Initialize components based on mode
        self._initialize_components()
        
        # Load existing visual embeddings from cache if in visual mode
        if self.mode in ['visual', 'hybrid']:
            self._load_existing_visual_embeddings()
        
        logger.info(f"✅ Multi-modal RAG system created in {self.mode} mode!")
    
    def _load_existing_visual_embeddings(self):
        """Load existing visual embeddings from cache."""
        try:
            # Look for cached visual embedding files
            import glob
            from pathlib import Path
            cache_dir = "cache/embeddings"
            if os.path.exists(cache_dir):
                visual_cache_files = glob.glob(os.path.join(cache_dir, "visual_*.pkl"))
                logger.info(f"🔍 Found {len(visual_cache_files)} cached visual embedding files")
                
                for cache_file in visual_cache_files:
                    try:
                        # Try to load cached embedding
                        import pickle
                        with open(cache_file, 'rb') as f:
                            cached_data = pickle.load(f)
                        
                        # Extract filename from cache data
                        if isinstance(cached_data, dict):
                            # Try to get filename from metadata first
                            filename = cached_data.get('metadata', {}).get('filename')
                            
                            # If not found, extract from file_path
                            if not filename and 'file_path' in cached_data:
                                file_path = cached_data['file_path']
                                # Skip temporary matplotlib files and similar
                                if 'mpl-data' not in file_path and file_path.endswith('.pdf'):
                                    filename = Path(file_path).name
                            
                            # Use cache file name as fallback for recently processed files
                            if not filename:
                                # For recently processed files, use a generic name with timestamp
                                cache_basename = Path(cache_file).stem
                                if cache_basename.startswith('visual_'):
                                    filename = f"Document_{cache_basename[7:15]}.pdf"  # Use part of hash as ID
                            
                            if filename and 'embeddings' in cached_data:
                                # Enhance metadata with filename
                                metadata = cached_data.get('metadata', {})
                                metadata['filename'] = filename
                                
                                self.visual_embeddings[filename] = {
                                    'embeddings': cached_data.get('embeddings'),
                                    'metadata': metadata,
                                    'original_path': cached_data.get('file_path', cache_file)
                                }
                                self.processed_documents.append(filename)
                                logger.info(f"📊 Loaded visual embeddings for: {filename}")
                    except Exception as e:
                        logger.warning(f"Failed to load cache file {cache_file}: {e}")
                
                if self.visual_embeddings:
                    self.is_initialized = True
                    logger.info(f"✅ Loaded {len(self.visual_embeddings)} visual document embeddings from cache")
        except Exception as e:
            logger.warning(f"Failed to load existing visual embeddings: {e}")
    
    def _initialize_components(self):
        """Initialize components based on retrieval mode."""
        
        if self.mode in ['text', 'hybrid']:
            # Initialize text-based RAG pipeline
            logger.info("🔧 Setting up text RAG pipeline...")
            self.text_pipeline = RAGPipeline(
                chunk_size=self.config.get('chunk_size', 800),
                overlap=self.config.get('chunk_overlap', 150),
                embedding_model=self.config.get('text_embedding_model', 'local')
            )
        
        if self.mode in ['visual', 'hybrid']:
            # Initialize visual components
            logger.info("🔧 Setting up visual processing...")
            self.visual_processor = VisualDocumentProcessor(self.config)
            self.visual_embedding_manager = EmbeddingManager.create_colpali(
                self.config.get('visual_model', 'vidore/colqwen2-v1.0')
            )
        
        if self.mode == 'hybrid':
            # Initialize multi-modal database
            text_dim = 384 if self.config.get('text_embedding_model', 'local') == 'local' else 1536
            self.multimodal_db = MultiModalVectorDatabase(
                text_dimension=text_dim,
                visual_dimension=128
            )
    
    def add_documents(self, document_paths: List[str]) -> Dict[str, Any]:
        """Add documents using the selected retrieval mode."""
        logger.info(f"📚 Processing {len(document_paths)} documents in {self.mode} mode...")
        
        results = {
            'successful': [],
            'failed': [],
            'total_chunks': 0,
            'visual_documents': 0,
            'processing_time': 0,
            'mode': self.mode
        }
        
        import time
        start_time = time.time()
        
        for doc_path in document_paths:
            try:
                filename = os.path.basename(doc_path)
                logger.info(f"📄 Processing: {filename}")
                
                if self.mode == 'text':
                    # Text-only processing
                    doc_results = self.text_pipeline.process_document(doc_path)
                    if doc_results.get('success', False):
                        results['successful'].append({
                            'path': doc_path,
                            'chunks': doc_results.get('chunks_created', 0),
                            'filename': filename,
                            'type': 'text'
                        })
                        results['total_chunks'] += doc_results.get('chunks_created', 0)
                        self.processed_documents.append(doc_path)
                    else:
                        results['failed'].append({
                            'path': doc_path,
                            'error': doc_results.get('error', 'Unknown error'),
                            'type': 'text'
                        })
                
                elif self.mode == 'visual':
                    # Visual-only processing (PDFs only)
                    if doc_path.lower().endswith('.pdf'):
                        visual_result = self.visual_embedding_manager.create_visual_embedding(doc_path)
                        if visual_result['status'] == 'success':
                            # Store embeddings with original filename for later retrieval
                            if not hasattr(self, 'visual_embeddings'):
                                self.visual_embeddings = {}
                            
                            self.visual_embeddings[filename] = {
                                'embeddings': visual_result['embeddings'],
                                'metadata': visual_result['metadata'],
                                'original_path': doc_path
                            }
                            
                            results['successful'].append({
                                'path': doc_path,
                                'pages': visual_result['metadata']['page_count'],
                                'filename': filename,
                                'type': 'visual'
                            })
                            results['visual_documents'] += 1
                            # Store filename instead of temp path
                            self.processed_documents.append(filename)
                        else:
                            results['failed'].append({
                                'path': doc_path,
                                'error': visual_result.get('error', 'Visual processing failed'),
                                'type': 'visual'
                            })
                    else:
                        results['failed'].append({
                            'path': doc_path,
                            'error': 'Visual mode only supports PDF files',
                            'type': 'visual'
                        })
                
                elif self.mode == 'hybrid':
                    # Process both text and visual (for PDFs)
                    text_success = False
                    visual_success = False
                    
                    # Text processing
                    doc_results = self.text_pipeline.process_document(doc_path)
                    if doc_results.get('success', False):
                        # Add text embeddings to multimodal database
                        # This would require accessing the embeddings from the pipeline
                        text_success = True
                        results['total_chunks'] += doc_results.get('chunks_created', 0)
                    
                    # Visual processing (if PDF)
                    if doc_path.lower().endswith('.pdf'):
                        visual_result = self.visual_embedding_manager.create_visual_embedding(doc_path)
                        if visual_result['status'] == 'success':
                            # Add visual embeddings to multimodal database
                            self.multimodal_db.add_visual_document(
                                visual_result['embeddings'],
                                visual_result['metadata']
                            )
                            visual_success = True
                            results['visual_documents'] += 1
                    
                    if text_success or visual_success:
                        results['successful'].append({
                            'path': doc_path,
                            'chunks': doc_results.get('chunks_created', 0) if text_success else 0,
                            'pages': visual_result['metadata']['page_count'] if visual_success else 0,
                            'filename': filename,
                            'type': 'hybrid',
                            'text_processed': text_success,
                            'visual_processed': visual_success
                        })
                        self.processed_documents.append(doc_path)
                    else:
                        results['failed'].append({
                            'path': doc_path,
                            'error': 'Both text and visual processing failed',
                            'type': 'hybrid'
                        })
                        
            except Exception as e:
                logger.error(f"❌ Failed to process {doc_path}: {str(e)}")
                results['failed'].append({
                    'path': doc_path,
                    'error': str(e),
                    'type': self.mode
                })
        
        results['processing_time'] = time.time() - start_time
        self.is_initialized = len(results['successful']) > 0
        
        logger.info(f"✅ Processed {len(results['successful'])} documents successfully")
        if self.mode in ['visual', 'hybrid']:
            logger.info(f"🖼️ Created {results['visual_documents']} visual document embeddings")
        if self.mode in ['text', 'hybrid']:
            logger.info(f"📊 Created {results['total_chunks']} text chunks")
        
        return results
    
    def query(self, question: str, max_chunks: Optional[int] = None) -> Dict[str, Any]:
        """Query using the selected retrieval mode."""
        if not self.is_initialized:
            return {
                'success': False,
                'error': 'No documents have been processed yet. Please add documents first.',
                'mode': self.mode
            }
        
        logger.info(f"🔍 Querying in {self.mode} mode: {question}")
        max_chunks = max_chunks or self.config.get('max_retrieved_chunks', 5)
        
        try:
            if self.mode == 'text':
                # Standard text-based query
                search_results = self.text_pipeline.search(question, top_k=max_chunks)
                if not search_results:
                    return self._no_results_response(question)
                
                response = self.text_pipeline.generate_response(question, search_results)
                return {
                    'success': True,
                    'question': question,
                    'answer': response.get('answer', ''),
                    'sources': self._format_sources(search_results),
                    'confidence': response.get('confidence', 0.0),
                    'chunks_used': len(search_results),
                    'mode': 'text'
                }
            
            elif self.mode == 'visual':
                # Visual-based query using ColPali
                visual_results = []
                
                logger.info(f"🔍 Starting visual query: '{question}'")
                logger.info(f"📊 Available visual embeddings: {len(self.visual_embeddings) if hasattr(self, 'visual_embeddings') else 0}")
                
                # Use stored embeddings instead of re-reading files
                if hasattr(self, 'visual_embeddings'):
                    for filename, stored_data in self.visual_embeddings.items():
                        try:
                            logger.info(f"🔍 Querying embeddings for: {filename}")
                            embeddings = stored_data['embeddings']
                            logger.info(f"📊 Embedding shape: {embeddings.shape if hasattr(embeddings, 'shape') else type(embeddings)}")
                            
                            scores = self.visual_embedding_manager.query_visual_embeddings(
                                question, embeddings
                            )
                            
                            logger.info(f"📊 Query scores shape: {scores.shape if hasattr(scores, 'shape') else type(scores)}")
                            logger.info(f"📊 Score statistics: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
                            
                            # Create result format similar to text results
                            max_score = float(scores.max()) if hasattr(scores, 'max') else float(scores)
                            
                            # Get more detailed page information if available
                            best_page_idx = int(scores.argmax()) if hasattr(scores, 'argmax') else 0
                            page_num = best_page_idx + 1  # Convert to 1-based page numbering
                            
                            logger.info(f"✅ Best match: {filename} page {page_num} (score: {max_score:.4f})")
                            
                            visual_results.append({
                                'score': max_score,
                                'content': f"Visual analysis of {filename}, page {page_num}",
                                'metadata': {
                                    'filename': filename,
                                    'page': page_num,
                                    'page_count': stored_data['metadata']['page_count'],
                                    'type': 'visual',
                                    'best_page_index': best_page_idx
                                }
                            })
                        except Exception as e:
                            logger.error(f"❌ Error querying visual embeddings for {filename}: {e}")
                            import traceback
                            logger.error(f"Full traceback: {traceback.format_exc()}")
                
                logger.info(f"📊 Total visual results found: {len(visual_results)}")
                
                if not visual_results:
                    logger.warning("⚠️ No visual results found, returning no_results_response")
                    return self._no_results_response(question)
                
                # Sort by score and take top results
                visual_results.sort(key=lambda x: x['score'], reverse=True)
                top_results = visual_results[:max_chunks]
                
                # Generate more meaningful visual answer with context
                answer = self._generate_enhanced_visual_answer(question, top_results)
                
                return {
                    'success': True,
                    'question': question,
                    'answer': answer,
                    'sources': self._format_sources(top_results),
                    'confidence': top_results[0]['score'] if top_results else 0.0,
                    'documents_used': len(top_results),
                    'mode': 'visual'
                }
            
            elif self.mode == 'hybrid':
                # Hybrid query combining text and visual
                # Get text results
                text_results = []
                visual_results = []
                
                # Try text search first
                try:
                    text_search_results = self.text_pipeline.search(question, top_k=max_chunks)
                    for result in text_search_results:
                        text_results.append({
                            'score': result.get('score', 0.0),
                            'content': result.get('content', ''),
                            'source': 'text',
                            'metadata': result.get('metadata', {})
                        })
                except Exception as e:
                    logger.warning(f"Text search failed in hybrid mode: {e}")
                
                # Try visual search
                try:
                    if hasattr(self, 'visual_embeddings'):
                        for filename, stored_data in self.visual_embeddings.items():
                            try:
                                scores = self.visual_embedding_manager.query_visual_embeddings(
                                    question, stored_data['embeddings']
                                )
                                max_score = float(scores.max()) if hasattr(scores, 'max') else float(scores)
                                best_page_idx = int(scores.argmax()) if hasattr(scores, 'argmax') else 0
                                page_num = best_page_idx + 1
                                
                                visual_results.append({
                                    'score': max_score,
                                    'content': f"Visual analysis of {filename}, page {page_num}",
                                    'source': 'visual',
                                    'metadata': {
                                        'filename': filename,
                                        'page': page_num,
                                        'page_count': stored_data['metadata']['page_count'],
                                        'type': 'visual'
                                    }
                                })
                            except Exception as e:
                                logger.warning(f"Error processing visual embeddings for {filename}: {e}")
                except Exception as e:
                    logger.warning(f"Visual search failed in hybrid mode: {e}")
                
                # Combine results with weighting
                text_weight = self.config.get('text_weight', 0.5)
                visual_weight = 1.0 - text_weight
                
                combined_results = []
                
                # Add weighted text results
                for result in text_results:
                    result['weighted_score'] = result['score'] * text_weight
                    combined_results.append(result)
                
                # Add weighted visual results
                for result in visual_results:
                    result['weighted_score'] = result['score'] * visual_weight
                    combined_results.append(result)
                
                # Sort by weighted score
                combined_results.sort(key=lambda x: x['weighted_score'], reverse=True)
                top_results = combined_results[:max_chunks]
                
                if not top_results:
                    return self._no_results_response(question)
                
                # Generate a meaningful hybrid answer
                text_count = sum(1 for r in top_results if r['source'] == 'text')
                visual_count = sum(1 for r in top_results if r['source'] == 'visual')
                
                best_result = top_results[0]
                confidence = best_result['weighted_score']
                
                answer = f"""Found {len(top_results)} relevant results combining text and visual analysis:

🔍 **Search Results Summary:**
- {text_count} text-based matches
- {visual_count} visual-based matches  
- Best match: {best_result['source']} analysis with confidence {confidence:.3f}

📊 **Combined Analysis:**
The hybrid search analyzed both textual content and visual document structure to find the most relevant information for your query: "{question}"

💡 **Recommendation:** 
For more detailed content, try switching to "Text" mode for specific excerpts or "Visual" mode for layout-aware analysis."""
                
                return {
                    'success': True,
                    'question': question,
                    'answer': answer,
                    'sources': self._format_hybrid_sources(top_results),
                    'confidence': confidence,
                    'results_used': len(top_results),
                    'mode': 'hybrid'
                }
                
        except Exception as e:
            logger.error(f"❌ Query failed: {str(e)}")
            return {
                'success': False,
                'error': f"Query processing failed: {str(e)}",
                'mode': self.mode
            }
    
    def _no_results_response(self, question: str) -> Dict[str, Any]:
        """Standard response when no results are found."""
        return {
            'success': True,
            'question': question,
            'answer': "I couldn't find any relevant information in the documents to answer your question.",
            'sources': [],
            'confidence': 0.0,
            'mode': self.mode
        }
    
    def _format_sources(self, search_results: List[Dict]) -> List[Dict]:
        """Format source information."""
        sources = []
        for i, result in enumerate(search_results, 1):
            metadata = result.get('metadata', {})
            
            # Handle different content types
            content = result.get('content', '')
            if metadata.get('type') == 'visual':
                # For visual results, create a descriptive chunk text
                page_info = f"page {metadata.get('page', 'unknown')}" if metadata.get('page') != 'Unknown' else ""
                chunk_text = f"Visual content from {metadata.get('filename', 'document')} {page_info}"
            else:
                # For text results, use actual content
                chunk_text = content[:200] + ('...' if len(content) > 200 else '')
            
            sources.append({
                'source_number': i,
                'filename': metadata.get('filename', 'Unknown'),
                'page': metadata.get('page', 'Unknown'),
                'type': metadata.get('type', 'text'),
                'chunk_text': chunk_text,
                'relevance_score': round(result.get('score', 0.0), 3)
            })
        return sources
    
    def _format_hybrid_sources(self, hybrid_results: List[Dict]) -> List[Dict]:
        """Format hybrid search results."""
        sources = []
        for i, result in enumerate(hybrid_results, 1):
            metadata = result.get('metadata', {})
            # Create content description based on source type
            source_type = result.get('source', 'unknown')
            if source_type == 'text':
                chunk_text = result.get('content', '')[:200] + '...'
            elif source_type == 'visual':
                chunk_text = f"Visual content from page {metadata.get('page', 'unknown')}"
            else:
                chunk_text = f"Hybrid content ({source_type})"
            
            sources.append({
                'source_number': i,
                'filename': metadata.get('filename', 'Unknown'),
                'page': metadata.get('page', 'Unknown'),
                'type': source_type,
                'chunk_text': chunk_text,
                'relevance_score': round(result.get('weighted_score', 0.0), 3),
                'original_score': round(result.get('score', 0.0), 3)
            })
        return sources
    
    def _generate_visual_answer(self, question: str, visual_results: List[Dict]) -> str:
        """Generate a more detailed answer based on visual search results."""
        if not visual_results:
            return "No visually relevant content found."
        
        logger.info(f"🖼️ Generating visual answer for {len(visual_results)} results")
        
        # Get the best matching document(s)
        best_result = visual_results[0]
        score = best_result['score']
        filename = best_result['metadata']['filename']
        page_count = best_result['metadata']['page_count']
        best_page = best_result['metadata'].get('page', 'unknown')
        
        logger.info(f"📊 Best match: {filename} page {best_page} (score: {score:.3f})")
        
        # Create more informative response with specific page information
        if score > 0.7:
            confidence_level = "high"
            confidence_desc = "I found highly relevant visual content"
        elif score > 0.5:
            confidence_level = "medium" 
            confidence_desc = "I found moderately relevant visual content"
        else:
            confidence_level = "low"
            confidence_desc = "I found potentially relevant visual content"
        
        # Create a more specific answer based on what we found
        multiple_docs = len(visual_results) > 1
        doc_summary = []
        
        for i, result in enumerate(visual_results[:3]):  # Show top 3 results
            fname = result['metadata']['filename']
            page = result['metadata'].get('page', 'unknown')
            rscore = result['score']
            doc_summary.append(f"• {fname} (page {page}) - score: {rscore:.3f}")
        
        docs_text = "\n".join(doc_summary)
        
        answer = f"""{confidence_desc} that may answer your question:

🎯 **Question:** {question}

📋 **Visual Matches Found:**
{docs_text}

🔍 **Primary Result:** 
The most relevant content appears to be on page {best_page} of "{filename}" with a confidence score of {score:.3f} ({confidence_level} confidence).

🖼️ **Visual Analysis Details:**
- ColPali analyzed the visual layout, text positioning, and document structure
- The system identified content based on both textual elements and visual arrangement
- {'Multiple documents' if multiple_docs else 'One document'} matched your query with varying degrees of relevance

💡 **To get more specific content:**
- Switch to "Text" mode for detailed text excerpts
- Use "Hybrid" mode to combine visual and text analysis
- The visual mode identifies relevant documents but doesn't extract specific text passages"""
        
        logger.info(f"✅ Generated enhanced visual answer ({len(answer)} chars)")
        return answer
    
    def _generate_enhanced_visual_answer(self, question: str, visual_results: List[Dict]) -> str:
        """Generate enhanced visual answer with more meaningful content analysis."""
        if not visual_results:
            return "No visually relevant content found."
        
        logger.info(f"🖼️ Generating enhanced visual answer for {len(visual_results)} results")
        
        # Analyze the query to provide more contextual responses
        query_lower = question.lower()
        
        # Best result details
        best_result = visual_results[0]
        score = best_result['score']
        filename = best_result['metadata']['filename']
        best_page = best_result['metadata'].get('page', 1)
        page_count = best_result['metadata']['page_count']
        
        # Confidence assessment
        if score > 0.7:
            confidence_desc = "found highly relevant visual content"
            confidence_level = "high"
        elif score > 0.5:
            confidence_desc = "found moderately relevant visual content"
            confidence_level = "medium"
        else:
            confidence_desc = "found potentially relevant visual content"
            confidence_level = "low"
        
        # Create contextual response based on query patterns
        context_hint = ""
        if any(word in query_lower for word in ['table', 'chart', 'graph', 'data']):
            context_hint = "\n🔍 **Visual Element Detection**: Your query suggests you're looking for tabular or graphical content. The visual analysis identified structural elements that may contain the data you're seeking."
        elif any(word in query_lower for word in ['figure', 'image', 'picture', 'diagram']):
            context_hint = "\n🖼️ **Image Analysis**: Your query indicates interest in visual elements. The system analyzed the document's visual layout and imagery."
        elif any(word in query_lower for word in ['section', 'heading', 'title']):
            context_hint = "\n📋 **Document Structure**: Your query relates to document organization. The visual analysis examined headers, sections, and document structure."
        
        # Build comprehensive answer
        answer = f"""Based on visual document analysis, I {confidence_desc} that addresses your question:

🎯 **Query Analysis**: "{question}"

📊 **Primary Match**:
- Document: {filename} (page {best_page} of {page_count})
- Confidence: {score:.3f} ({confidence_level})
- Analysis: ColPali visual-language model identified relevant content based on both text positioning and visual document structure

📋 **Visual Search Results**:"""
        
        # Add details for multiple results
        for i, result in enumerate(visual_results[:3], 1):
            fname = result['metadata']['filename']
            page = result['metadata'].get('page', 'unknown')
            rscore = result['score']
            answer += f"\n{i}. {fname} (page {page}) - confidence: {rscore:.3f}"
        
        answer += f"""{context_hint}

🧠 **How Visual Search Works**:
ColPali analyzed your documents by:
- Converting PDF pages to visual representations
- Understanding both text content AND visual layout
- Matching your query against document structure, not just text
- Identifying the most relevant pages based on visual-textual similarity

💡 **Next Steps**:
- For specific text content: Switch to "Text" mode to get exact quotations
- For comprehensive analysis: Try "Hybrid" mode combining both approaches
- This visual mode excels at finding relevant sections when you're unsure of exact wording"""
        
        logger.info(f"✅ Generated enhanced contextual visual answer ({len(answer)} chars)")
        return answer
    
    def switch_mode(self, new_mode: str) -> bool:
        """Switch retrieval mode and reinitialize components."""
        if new_mode not in ['text', 'visual', 'hybrid']:
            logger.error(f"❌ Invalid mode: {new_mode}")
            return False
        
        if new_mode == self.mode:
            logger.info(f"Already in {new_mode} mode")
            return True
        
        logger.info(f"🔄 Switching from {self.mode} to {new_mode} mode...")
        
        old_mode = self.mode
        self.mode = new_mode
        
        try:
            self._initialize_components()
            
            # Clear and reprocess documents if needed
            if self.processed_documents:
                logger.info("📚 Reprocessing documents for new mode...")
                docs_to_reprocess = self.processed_documents.copy()
                self.processed_documents = []
                self.is_initialized = False
                
                # Reprocess with new mode
                self.add_documents(docs_to_reprocess)
            
            logger.info(f"✅ Successfully switched to {new_mode} mode")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to switch modes: {str(e)}")
            # Revert to old mode
            self.mode = old_mode
            self._initialize_components()
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        info = {
            'mode': self.mode,
            'is_initialized': self.is_initialized,
            'total_documents': len(self.processed_documents),
            'config': self.config,
            'processed_documents': [os.path.basename(doc) for doc in self.processed_documents]
        }
        
        if self.mode in ['text', 'hybrid'] and hasattr(self, 'text_pipeline'):
            info['text_chunks'] = getattr(self.text_pipeline.vector_db, 'total_chunks', 0) if hasattr(self.text_pipeline, 'vector_db') else 0
        
        if self.mode in ['visual', 'hybrid'] and hasattr(self, 'visual_embedding_manager'):
            info['visual_stats'] = self.visual_embedding_manager.get_stats()
        
        if self.mode == 'hybrid' and hasattr(self, 'multimodal_db'):
            info['multimodal_stats'] = self.multimodal_db.get_stats()
        
        return info


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


def create_multimodal_rag_system(config: Dict[str, Any]) -> MultiModalRAGSystem:
    """
    Factory function to create a multi-modal RAG system.
    
    Example configurations:
    
    # Text-only mode (original behavior)
    config = {
        'retrieval_mode': 'text',
        'chunk_size': 800,
        'chunk_overlap': 150,
        'text_embedding_model': 'local',
        'max_retrieved_chunks': 5
    }
    
    # Visual-only mode (ColPali)
    config = {
        'retrieval_mode': 'visual',
        'visual_model': 'vidore/colqwen2-v1.0',
        'max_retrieved_chunks': 5
    }
    
    # Hybrid mode (both text and visual)
    config = {
        'retrieval_mode': 'hybrid',
        'chunk_size': 800,
        'chunk_overlap': 150,
        'text_embedding_model': 'local',
        'visual_model': 'vidore/colqwen2-v1.0',
        'text_weight': 0.5,
        'max_retrieved_chunks': 5
    }
    """
    logger.info("🏗️ Creating new multi-modal RAG system...")
    
    # Validate configuration
    retrieval_mode = config.get('retrieval_mode', 'text')
    if retrieval_mode not in ['text', 'visual', 'hybrid']:
        raise ValueError(f"Invalid retrieval_mode: {retrieval_mode}. Must be 'text', 'visual', or 'hybrid'")
    
    # Validate mode-specific requirements
    if retrieval_mode in ['text', 'hybrid']:
        text_required = ['chunk_size', 'chunk_overlap', 'text_embedding_model']
        for key in text_required:
            if key not in config:
                raise ValueError(f"Missing required configuration for text mode: {key}")
    
    if retrieval_mode in ['visual', 'hybrid']:
        if 'visual_model' not in config:
            config['visual_model'] = 'vidore/colqwen2-v1.0'  # Default
    
    # Create and return the system
    multimodal_rag = MultiModalRAGSystem(config)
    logger.info("🎉 Multi-modal RAG system ready to use!")
    
    return multimodal_rag


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
