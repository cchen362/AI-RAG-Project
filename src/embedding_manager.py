import os
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime
import pickle
import hashlib

# Heavy imports moved to methods for faster startup
try:
    from visual_document_processor import VisualDocumentProcessor
except ImportError:
    VisualDocumentProcessor = None

class EmbeddingManager:
    """
    Manages text embeddings for RAG system.
    
    Think of this as a universal translator that converts text 
    into a language that computers can understand and compare - 
    numbers that represent meaning.
    """

    def __init__(self,
                 embedding_model: str = "local",  # Changed default to "local" for beginners
                 model_name: str = None,  # Will auto-select appropriate model
                 cache_embeddings: bool = True,
                 cache_dir: str = "cache/embeddings",
                 dimensions: int = None):  # OpenAI embeddings dimension control
        
        self.embedding_model = embedding_model
        
        # Auto-select appropriate model name if not provided
        if model_name is None:
            if embedding_model == "openai":
                self.model_name = "text-embedding-ada-002"
            elif embedding_model == "colpali":
                self.model_name = "vidore/colqwen2-v1.0"
            else:  # local
                self.model_name = "all-MiniLM-L6-v2"
        else:
            self.model_name = model_name
            
        self.cache_embeddings = cache_embeddings
        self.cache_dir = cache_dir
        self.dimensions = dimensions  # Store dimensions for OpenAI models

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize embedding model
        self._initialize_model()

        # Create cache directory
        if self.cache_embeddings:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Track statistics
        self.stats = {
            'embeddings_created': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_tokens_processed': 0
        }
    
    @classmethod
    def create_local(cls, model_name: str = "all-MiniLM-L6-v2"):
        """Create a local embedding manager (free, no API key needed)."""
        return cls(embedding_model="local", model_name=model_name)
    
    @classmethod
    def create_openai(cls, model_name: str = "text-embedding-ada-002", dimensions: int = None):
        """Create an OpenAI embedding manager (requires API key)."""
        return cls(embedding_model="openai", model_name=model_name, dimensions=dimensions)
    
    @classmethod
    def create_colpali(cls, model_name: str = "vidore/colqwen2-v1.0"):
        """Create a ColPali visual embedding manager (requires GPU for best performance)."""
        return cls(embedding_model="colpali", model_name=model_name)

    def _initialize_model(self):
        """Initialize the embedding model based on configuration."""

        if self.embedding_model == "openai":
            # Lazy import OpenAI
            from openai import OpenAI
            
            # Initialize OpenAI client
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            
            self.client = OpenAI(api_key=api_key)
            
            # Validate actual embedding dimensions by testing a small sample
            self.embedding_dimension = self._validate_openai_dimensions()
            
            if self.dimensions and self.embedding_dimension != self.dimensions:
                self.logger.warning(f"⚠️ DIMENSION MISMATCH: Requested {self.dimensions} but OpenAI returns {self.embedding_dimension}")
                self.logger.warning(f"   - This suggests the dimensions parameter is not working for model: {self.model_name}")
                self.logger.warning(f"   - Proceeding with actual dimensions: {self.embedding_dimension}")
            
            self.logger.info(f"✅ Initialized OpenAI embeddings with model: {self.model_name}, actual dimensions: {self.embedding_dimension}")

        elif self.embedding_model == "local":
            # Lazy import SentenceTransformer
            from sentence_transformers import SentenceTransformer
            
            # Initialize local sentence transformer with proper model name
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"✅ Initialized local embeddings with model: {self.model_name}")

        elif self.embedding_model == "colpali":
            # Initialize ColPali visual processor
            if VisualDocumentProcessor is None:
                raise ImportError("VisualDocumentProcessor not available. Check visual_document_processor.py")
            
            config = {
                'colpali_model': self.model_name,
                'cache_dir': self.cache_dir
            }
            self.visual_processor = VisualDocumentProcessor(config)
            # ColPali uses 128-dimensional embeddings per patch
            self.embedding_dimension = 128
            self.logger.info(f"✅ Initialized ColPali visual embeddings with model: {self.model_name}")

        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model}")
    
    def _validate_openai_dimensions(self) -> int:
        """
        Validate actual OpenAI embedding dimensions by making a test API call.
        This helps us detect if the dimensions parameter is working correctly.
        """
        self.logger.info("🔍 Validating OpenAI embedding dimensions with test call...")
        
        try:
            # Test with a simple text
            test_text = "test"
            
            # Prepare embedding request parameters
            embedding_params = {
                'model': self.model_name,
                'input': test_text
            }
            
            # Add dimensions parameter if requested
            if self.dimensions and 'text-embedding-3' in self.model_name:
                embedding_params['dimensions'] = self.dimensions
                self.logger.info(f"🔧 Testing with dimensions parameter: {self.dimensions}")
            else:
                self.logger.info(f"🔧 Testing without dimensions parameter")
            
            # Make the test API call
            response = self.client.embeddings.create(**embedding_params)
            
            # Check actual dimensions
            test_embedding = np.array(response.data[0].embedding, dtype=np.float32)
            actual_dimensions = test_embedding.shape[0]
            
            self.logger.info(f"✅ OpenAI dimension validation complete:")
            self.logger.info(f"   - Model: {self.model_name}")
            self.logger.info(f"   - Requested dimensions: {self.dimensions}")
            self.logger.info(f"   - Actual dimensions: {actual_dimensions}")
            
            return actual_dimensions
            
        except Exception as e:
            self.logger.error(f"❌ Failed to validate OpenAI dimensions: {str(e)}")
            
            # Fallback to model defaults if validation fails
            if 'text-embedding-3-large' in self.model_name:
                fallback_dim = 3072
            elif 'text-embedding-3-small' in self.model_name:
                fallback_dim = 1536
            else:  # ada-002 and others
                fallback_dim = 1536
                
            self.logger.warning(f"⚠️ Using fallback dimensions: {fallback_dim}")
            return fallback_dim
        
    def create_embedding(self, text: str, cache_key: str = None) -> np.ndarray:
        """
        Create embedding for a single text.
        
        Like converting a sentence into a unique fingerprint that 
        captures its meaning and can be compared with other fingerprints.
        """

        if not text or not text.strip():
            return np.zeros(self.embedding_dimension)
        
        # Generate cache key
        if cache_key is None:
            cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()

        # Check cache first
        if self.cache_embeddings:
            cached_embedding = self._load_from_cache(cache_key)
            if cached_embedding is not None:
                self.stats['cache_hits'] += 1
                return cached_embedding
            
        # Create new embedding
        try:
            if self.embedding_model == "openai":
                embedding = self._create_openai_embedding(text)
            elif self.embedding_model == "colpali":
                # For ColPali, text-only embedding isn't the primary use case
                # Return zero vector or raise error for text-only queries
                self.logger.warning("ColPali is designed for visual documents. Consider using create_visual_embedding instead.")
                return np.zeros(self.embedding_dimension)
            else:
                embedding = self._create_local_embedding(text)

            # Cache the result
            if self.cache_embeddings:
                self._save_to_cache(cache_key, embedding)

            # Update statistics
            self.stats['embeddings_created'] += 1
            self.stats['cache_misses'] += 1
            self.stats['total_tokens_processed'] += len(text.split())

            return embedding
        
        except Exception as e:
            self.logger.error(f"Failed to create embedding for text: {str(e)}")
            return np.zeros(self.embedding_dimension)
        
    def _create_openai_embedding(self, text: str) -> np.ndarray:
        """Create embedding using OpenAI API."""

        try: 
            # Prepare embedding request parameters
            embedding_params = {
                'model': self.model_name,
                'input': text.strip()
            }
            
            # Add dimensions parameter for supported models (text-embedding-3-*)
            # Note: We use self.dimensions (requested) not self.embedding_dimension (actual)
            # because we want to be consistent in our API calls
            if self.dimensions and 'text-embedding-3' in self.model_name:
                embedding_params['dimensions'] = self.dimensions
                
            response = self.client.embeddings.create(**embedding_params)
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            # Validate that we get expected dimensions (should match our validated dimension)
            actual_dimensions = embedding.shape[0]
            if actual_dimensions != self.embedding_dimension:
                self.logger.error(f"❌ UNEXPECTED DIMENSION CHANGE!")
                self.logger.error(f"   - Expected: {self.embedding_dimension} (from validation)")
                self.logger.error(f"   - Received: {actual_dimensions}")
                raise ValueError(f"Embedding dimensions changed unexpectedly: {actual_dimensions} vs {self.embedding_dimension}")
            
            return embedding
        
        except Exception as e:
            self.logger.error(f"OpenAI embedding failed: {str(e)}")
            raise

    def _create_local_embedding(self, text: str) -> np.ndarray:
        """Create embedding using local model."""

        try: 
            embedding = self.model.encode(text.strip())
            return np.array(embedding, dtype=np.float32)
        
        except Exception as e:
            self.logger.error(f"Local embedding failed: {str(e)}")
            raise

    def create_visual_embedding(self, file_path: str, cache_key: str = None) -> Dict[str, Any]:
        """
        Create visual embeddings for a document using ColPali.
        
        Returns a dictionary containing multiple embeddings (one per page/patch)
        and associated metadata for multi-vector search.
        """
        if self.embedding_model != "colpali":
            return {
                'status': 'error',
                'error': 'Visual embeddings are only supported with ColPali model'
            }
        
        if not file_path or not os.path.exists(file_path):
            return {
                'status': 'error',
                'error': f'Invalid file path: {file_path}'
            }
        
        # Generate cache key if not provided
        if cache_key is None:
            try:
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                    cache_key = hashlib.md5(file_content).hexdigest()
            except Exception as e:
                self.logger.error(f"Failed to generate cache key: {e}")
                cache_key = hashlib.md5(file_path.encode()).hexdigest()
        
        # Check cache first
        if self.cache_embeddings:
            cached_result = self._load_visual_from_cache(cache_key)
            if cached_result is not None:
                self.stats['cache_hits'] += 1
                return cached_result
        
        try:
            # Process document with visual processor
            if self.visual_processor is None:
                return {
                    'status': 'error',
                    'error': 'VisualDocumentProcessor not initialized'
                }
            
            result = self.visual_processor.process_file(file_path)
            
            if result['status'] == 'error':
                self.logger.error(f"Visual processor error: {result['error']}")
                return result
            
            # Cache the result
            if self.cache_embeddings:
                try:
                    self._save_visual_to_cache(cache_key, result)
                except Exception as e:
                    self.logger.warning(f"Failed to cache visual embedding: {e}")
            
            # Update statistics
            self.stats['embeddings_created'] += 1
            self.stats['cache_misses'] += 1
            self.stats['total_tokens_processed'] += result['metadata'].get('page_count', 0) * 1024  # Estimate tokens per page
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to create visual embedding for {file_path}: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'file_path': file_path
            }

    def query_visual_embeddings(self, query: str, document_embeddings: Any) -> Any:
        """Query visual embeddings using ColPali."""
        if self.embedding_model != "colpali":
            self.logger.error("Visual querying is only supported with ColPali model")
            import torch
            return torch.tensor([0.0])
        
        try:
            if self.visual_processor is None:
                self.logger.error("VisualDocumentProcessor not initialized for querying")
                import torch
                return torch.tensor([0.0])
            
            scores = self.visual_processor.query_embeddings(query, document_embeddings)
            return scores
        except Exception as e:
            self.logger.error(f"Failed to query visual embeddings: {str(e)}")
            # Return a default score instead of raising
            import torch
            return torch.tensor([0.0])

    def create_batch_embeddings(self, texts: List[str], batch_size: int = 10) -> List[np.ndarray]:
        """
        Create embeddings for multiple texts efficiently.
        
        Like processing a batch of photos at once instead of one by one - 
        much more efficient for large collections.
        """

        if not texts:
            return []
        
        embeddings = []

        # Process in batches for efficiency
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            if self.embedding_model == "openai":
                # OpenAI supports batch processing
                batch_embeddings = self._create_openai_batch_embeddings(batch)
            else:
                # Local model batch processing
                batch_embeddings = self._create_local_batch_embeddings(batch)

            embeddings.extend(batch_embeddings)

            # Progress update
            if i % 50 == 0:
                self.logger.info(f"Processed {i + len(batch)}/{len(texts)} embeddings")

        return embeddings
    
    def _create_openai_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Create batch embeddings using OpenAI API."""

        try:
            # Prepare embedding request parameters
            embedding_params = {
                'model': self.model_name,
                'input': [text.strip() for text in texts]
            }
            
            # Add dimensions parameter for supported models (text-embedding-3-*)
            if self.dimensions and 'text-embedding-3' in self.model_name:
                embedding_params['dimensions'] = self.dimensions
                
            response = self.client.embeddings.create(**embedding_params)

            embeddings = []
            for i, data in enumerate(response.data):
                embedding = np.array(data.embedding, dtype=np.float32)
                
                # Validate dimensions consistency
                actual_dimensions = embedding.shape[0]
                if actual_dimensions != self.embedding_dimension:
                    self.logger.error(f"❌ BATCH DIMENSION MISMATCH!")
                    self.logger.error(f"   - Expected: {self.embedding_dimension}")
                    self.logger.error(f"   - Received: {actual_dimensions}")
                    raise ValueError(f"Batch embedding {i} has wrong dimensions: {actual_dimensions} vs {self.embedding_dimension}")
                
                embeddings.append(embedding)

            return embeddings
        
        except Exception as e:
            self.logger.error(f"OpenAI batch embedding failed: {str(e)}")
            # Fallback to individual processing
            return [self.create_embedding(text) for text in texts]
        
    def _create_local_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Create batch embeddings using local model."""

        try:
            embeddings = self.model.encode([text.strip() for text in texts])
            return [np.array(emb, dtype=np.float32) for emb in embeddings]
        
        except Exception as e: 
            self.logger.error(f"Local batch embedding failed: {str(e)}")
            # Fallback to individual processing
            return [self.create_embedding(text) for text in texts]

    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from cache."""

        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache {cache_key}: {str(e)}")

        return None

    def _save_to_cache(self, cache_key: str, embedding: np.ndarray):
        """Save embedding to cache."""

        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            self.logger.warning(f"Failed to save cache {cache_key}: {str(e)}")

    def _load_visual_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load visual embedding result from cache."""
        cache_path = os.path.join(self.cache_dir, f"visual_{cache_key}.pkl")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load visual cache {cache_key}: {str(e)}")

        return None

    def _save_visual_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """Save visual embedding result to cache."""
        cache_path = os.path.join(self.cache_dir, f"visual_{cache_key}.pkl")

        try:
            # Don't cache the PIL images, only the embeddings and metadata
            cache_result = {
                'file_path': result['file_path'],
                'file_info': result['file_info'],
                'page_info': result['page_info'],
                'embeddings': result['embeddings'],
                'metadata': result['metadata'],
                'processing_time': result['processing_time'],
                'status': result['status'],
                'embedding_type': result['embedding_type'],
                'model_name': result['model_name'],
                'device': result['device']
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_result, f)
        except Exception as e:
            self.logger.warning(f"Failed to save visual cache {cache_key}: {str(e)}")

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Like measuring how similar two people's fingerprints are - 
        returns a score from 0 (completely different) to 1 (identical).
        """
        # Lazy import sklearn
        from sklearn.metrics.pairwise import cosine_similarity

        # Handle zero vectors
        if np.all(embedding1 == 0) or np.all(embedding2 == 0):
            return 0.0
        
        # Calculate cosine similarity
        similarity = cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        )[0][0]

        return float(similarity)
    
    def find_most_similar(self, query_embedding: np.ndarray,
                          candidate_embeddings: List[np.ndarray],
                          top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find the most similar embeddings to a query.
        
        Like finding the most similar photos in a collection
        based on visual similarity.
        """

        if not candidate_embeddings:
            return []
        
        similarities = []

        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.calculate_similarity(query_embedding, candidate)
            similarities.append({
                'index': i,
                'similarity': similarity
            })

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        # Return top K results
        return similarities[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding statistics."""

        cache_hit_rate = 0
        if self.stats['cache_hits'] + self.stats['cache_misses'] > 0:
            cache_hit_rate = self.stats['cache_hits'] / (
                self.stats['cache_hits'] + self.stats['cache_misses']
            )

        return {
            **self.stats,
            'cache_hit_rate': cache_hit_rate,
            'embedding_dimension': self.embedding_dimension,
            'model_type': self.embedding_model,
            'model_name': self.model_name
        }


class VectorDatabase:
    """
    Stores and retrieves embeddings efficiently.
    
    Think of this as a specialized filing cabinet that organizes 
    documents by meaning rather than alphabetically.
    """
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        """

        Initialize vector database.
        
        Args:
            dimension: Embedding dimension
            index_type: "flat" (exact search) or "ivf" (approximate, faster)
        """
        # Lazy import faiss
        import faiss
        
        self.dimension = dimension
        self.index_type = index_type
        
        # Initialize FAISS index
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 clusters
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Store metadata alongside vectors
        self.metadata = []
        self.id_to_index = {}
        self.next_id = 0
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def add_vectors(self, embeddings: List[np.ndarray], metadata: List[Dict[str, Any]]):
        """
        Add vectors to the database.
        
        Like adding new books to your library with their catalog information.
        """
        try:
            # Lazy import faiss
            import faiss
            
            if len(embeddings) != len(metadata):
                raise ValueError(f"Number of embeddings ({len(embeddings)}) must match number of metadata entries ({len(metadata)})")
            
            # Validate embeddings before processing
            for i, embedding in enumerate(embeddings):
                if not isinstance(embedding, np.ndarray):
                    raise ValueError(f"Embedding {i} is not a numpy array, got {type(embedding)}")
                if embedding.shape != (self.dimension,):
                    raise ValueError(f"Embedding {i} has shape {embedding.shape}, expected ({self.dimension},)")
                if not np.isfinite(embedding).all():
                    raise ValueError(f"Embedding {i} contains non-finite values")
            
            # Convert to numpy array for FAISS
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Validate final array shape
            expected_shape = (len(embeddings), self.dimension)
            if embeddings_array.shape != expected_shape:
                raise ValueError(f"Embeddings array has shape {embeddings_array.shape}, expected {expected_shape}")
            
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            # Train index if needed (for IVF)
            if self.index_type == "ivf" and not self.index.is_trained:
                self.index.train(embeddings_array)
            
            # Add to index
            start_idx = self.index.ntotal
            self.index.add(embeddings_array)
            
        except Exception as e:
            # Log detailed error information
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"❌ Vector database add_vectors failed: {str(e)}")
            logger.error(f"   - Embeddings count: {len(embeddings) if embeddings else 0}")
            logger.error(f"   - Metadata count: {len(metadata) if metadata else 0}")
            logger.error(f"   - Expected dimension: {self.dimension}")
            if embeddings:
                logger.error(f"   - First embedding shape: {embeddings[0].shape if hasattr(embeddings[0], 'shape') else 'N/A'}")
                logger.error(f"   - First embedding type: {type(embeddings[0])}")
            raise e
        
        # Store metadata
        for i, meta in enumerate(metadata):
            doc_id = self.next_id
            self.id_to_index[doc_id] = start_idx + i
            
            # Add ID to metadata
            meta_with_id = {**meta, 'id': doc_id}
            self.metadata.append(meta_with_id)
            
            self.next_id += 1
        
        self.logger.info(f"Added {len(embeddings)} vectors to database. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Like asking a librarian to find books similar to one you're holding.
        """
        # Lazy import faiss
        import faiss
        
        if self.index.ntotal == 0:
            return []
        
        # Normalize query vector
        query_normalized = query_embedding.copy().reshape(1, -1)
        faiss.normalize_L2(query_normalized)
        
        # Search
        scores, indices = self.index.search(query_normalized, top_k)
        
        # Combine results with metadata
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid result
                result = {
                    'score': float(score),
                    'rank': i + 1,
                    'metadata': self.metadata[idx]
                }
                results.append(result)
        
        return results
    
    def get_vector_by_id(self, doc_id: int) -> Optional[np.ndarray]:
        """Get vector by document ID."""
        
        if doc_id not in self.id_to_index:
            return None
        
        index_pos = self.id_to_index[doc_id]
        vector = self.index.reconstruct(index_pos)
        
        return vector
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metadata_count': len(self.metadata),
            'is_trained': getattr(self.index, 'is_trained', True)
        }
    

    def clear(self):
        """
        Clear all vectors and metadata from the database.
        
        Like emptying your entire filing cabinet and starting fresh.
        """
        # Lazy import faiss
        import faiss
        
        # Reinitialize the FAISS index
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        
        # Clear all metadata
        self.metadata = []
        self.id_to_index = {}
        self.next_id = 0
        
        self.logger.info("🗑️ Cleared all vectors and metadata from database")

    def save_to_disk(self, filepath: str):
        """Save database to disk."""
        # Lazy import faiss
        import faiss
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.index")
        
        # Save metadata
        with open(f"{filepath}.metadata", 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'id_to_index': self.id_to_index,
                'next_id': self.next_id,
                'dimension': self.dimension,
                'index_type': self.index_type
            }, f)
        
        self.logger.info(f"Saved database to {filepath}")
    
    def load_from_disk(self, filepath: str):
        """Load database from disk."""
        # Lazy import faiss
        import faiss
        
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.index")
        
        # Load metadata
        with open(f"{filepath}.metadata", 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.id_to_index = data['id_to_index']
            self.next_id = data['next_id']
            self.dimension = data['dimension']
            self.index_type = data['index_type']
        
        self.logger.info(f"Loaded database from {filepath}")


class MultiModalVectorDatabase:
    """
    Multi-modal vector database supporting both text and visual embeddings.
    
    Handles ColPali's multi-vector approach where each document generates
    multiple embeddings (one per patch/page) with late interaction scoring.
    """
    
    def __init__(self, text_dimension: int = None, visual_dimension: int = 128):
        """
        Initialize multi-modal vector database.
        
        Args:
            text_dimension: Dimension for text embeddings (if using hybrid mode)
            visual_dimension: Dimension for visual embeddings (128 for ColPali)
        """
        import faiss
        
        self.text_dimension = text_dimension
        self.visual_dimension = visual_dimension
        
        # Separate indices for different embedding types
        self.text_index = None
        self.visual_index = None
        
        if text_dimension:
            self.text_index = faiss.IndexFlatIP(text_dimension)
        
        if visual_dimension:
            self.visual_index = faiss.IndexFlatIP(visual_dimension)
        
        # Metadata storage
        self.text_metadata = []
        self.visual_metadata = []
        self.visual_documents = {}  # Store multi-vector documents
        
        # ID management
        self.text_id_to_index = {}
        self.visual_id_to_index = {}
        self.next_text_id = 0
        self.next_visual_id = 0
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def add_text_vectors(self, embeddings: List[np.ndarray], metadata: List[Dict[str, Any]]):
        """Add text embeddings to the database."""
        if not self.text_index:
            raise ValueError("Text index not initialized. Provide text_dimension during initialization.")
        
        import faiss
        
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        # Convert to numpy array for FAISS
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Add to index
        start_idx = self.text_index.ntotal
        self.text_index.add(embeddings_array)
        
        # Store metadata
        for i, meta in enumerate(metadata):
            doc_id = self.next_text_id
            self.text_id_to_index[doc_id] = start_idx + i
            
            # Add ID to metadata
            meta_with_id = {**meta, 'id': doc_id, 'type': 'text'}
            self.text_metadata.append(meta_with_id)
            
            self.next_text_id += 1
        
        self.logger.info(f"Added {len(embeddings)} text vectors. Total text: {self.text_index.ntotal}")
    
    def add_visual_document(self, embeddings: Any, metadata: Dict[str, Any]):
        """
        Add visual document with multi-vector embeddings.
        
        Args:
            embeddings: ColPali multi-vector embeddings (torch.Tensor)
            metadata: Document metadata including file info and page info
        """
        import torch
        import faiss
        
        doc_id = self.next_visual_id
        
        # Store the full multi-vector embeddings for the document
        self.visual_documents[doc_id] = {
            'embeddings': embeddings.cpu().detach() if torch.is_tensor(embeddings) else embeddings,
            'metadata': {**metadata, 'id': doc_id, 'type': 'visual'}
        }
        
        # For search efficiency, we can also flatten and add individual patches to FAISS
        if torch.is_tensor(embeddings):
            # Flatten the multi-vector embeddings
            if len(embeddings.shape) > 2:
                flat_embeddings = embeddings.view(-1, embeddings.shape[-1])
            else:
                flat_embeddings = embeddings
            
            flat_embeddings = flat_embeddings.cpu().numpy().astype(np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(flat_embeddings)
            
            # Add to visual index
            start_idx = self.visual_index.ntotal
            self.visual_index.add(flat_embeddings)
            
            # Create metadata for each patch
            for i in range(len(flat_embeddings)):
                patch_metadata = {
                    'document_id': doc_id,
                    'patch_index': i,
                    'type': 'visual_patch',
                    'parent_metadata': metadata
                }
                self.visual_metadata.append(patch_metadata)
                self.visual_id_to_index[f"{doc_id}_{i}"] = start_idx + i
        
        self.next_visual_id += 1
        
        self.logger.info(f"Added visual document {doc_id}. Total visual docs: {len(self.visual_documents)}")
    
    def search_text(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search text embeddings."""
        if not self.text_index or self.text_index.ntotal == 0:
            return []
        
        import faiss
        
        # Normalize query vector
        query_normalized = query_embedding.copy().reshape(1, -1)
        faiss.normalize_L2(query_normalized)
        
        # Search
        scores, indices = self.text_index.search(query_normalized, top_k)
        
        # Combine results with metadata
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid result
                result = {
                    'score': float(score),
                    'rank': i + 1,
                    'metadata': self.text_metadata[idx],
                    'type': 'text'
                }
                results.append(result)
        
        return results
    
    def search_visual(self, query: str, embedding_manager: 'EmbeddingManager', top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search visual documents using ColPali's multi-vector approach.
        
        Args:
            query: Text query string
            embedding_manager: EmbeddingManager with ColPali model
            top_k: Number of documents to return
        """
        if not self.visual_documents:
            return []
        
        # Calculate scores for each document using ColPali's scoring method
        document_scores = []
        
        for doc_id, doc_data in self.visual_documents.items():
            try:
                # Use ColPali's scoring method
                scores = embedding_manager.query_visual_embeddings(query, doc_data['embeddings'])
                
                # Get the maximum score for this document
                if hasattr(scores, 'max'):
                    max_score = float(scores.max())
                else:
                    max_score = float(scores)
                
                document_scores.append({
                    'document_id': doc_id,
                    'score': max_score,
                    'metadata': doc_data['metadata']
                })
                
            except Exception as e:
                self.logger.warning(f"Error scoring document {doc_id}: {e}")
                continue
        
        # Sort by score and return top_k
        document_scores.sort(key=lambda x: x['score'], reverse=True)
        
        results = []
        for i, doc_score in enumerate(document_scores[:top_k]):
            result = {
                'score': doc_score['score'],
                'rank': i + 1,
                'metadata': doc_score['metadata'],
                'type': 'visual'
            }
            results.append(result)
        
        return results
    
    def search_hybrid(self, query_embedding: np.ndarray, query: str, 
                     embedding_manager: 'EmbeddingManager', top_k: int = 5, 
                     text_weight: float = 0.5) -> List[Dict[str, Any]]:
        """
        Hybrid search combining text and visual results.
        
        Args:
            query_embedding: Text embedding for the query
            query: Original query string for visual search
            embedding_manager: EmbeddingManager with models
            top_k: Total number of results to return
            text_weight: Weight for text results (0.0 to 1.0)
        """
        text_results = []
        visual_results = []
        
        # Get text results if available
        if self.text_index and self.text_index.ntotal > 0:
            text_results = self.search_text(query_embedding, top_k * 2)  # Get more to rerank
        
        # Get visual results if available
        if self.visual_documents:
            visual_results = self.search_visual(query, embedding_manager, top_k * 2)
        
        # Combine and rerank results
        combined_results = []
        
        # Add text results with weight
        for result in text_results:
            result['weighted_score'] = result['score'] * text_weight
            result['source'] = 'text'
            combined_results.append(result)
        
        # Add visual results with weight
        visual_weight = 1.0 - text_weight
        for result in visual_results:
            result['weighted_score'] = result['score'] * visual_weight
            result['source'] = 'visual'
            combined_results.append(result)
        
        # Sort by weighted score and return top_k
        combined_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        # Re-rank the combined results
        final_results = []
        for i, result in enumerate(combined_results[:top_k]):
            result['rank'] = i + 1
            final_results.append(result)
        
        return final_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            'text_vectors': self.text_index.ntotal if self.text_index else 0,
            'visual_documents': len(self.visual_documents),
            'visual_patches': self.visual_index.ntotal if self.visual_index else 0,
            'text_dimension': self.text_dimension,
            'visual_dimension': self.visual_dimension,
            'total_metadata_entries': len(self.text_metadata) + len(self.visual_metadata)
        }
    
    def clear(self):
        """Clear all vectors and metadata."""
        import faiss
        
        # Reinitialize indices
        if self.text_dimension and self.text_index:
            self.text_index = faiss.IndexFlatIP(self.text_dimension)
        
        if self.visual_dimension and self.visual_index:
            self.visual_index = faiss.IndexFlatIP(self.visual_dimension)
        
        # Clear all data
        self.text_metadata = []
        self.visual_metadata = []
        self.visual_documents = {}
        self.text_id_to_index = {}
        self.visual_id_to_index = {}
        self.next_text_id = 0
        self.next_visual_id = 0
        
        self.logger.info("🗑️ Cleared all multi-modal vectors and metadata")
