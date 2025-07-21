import os
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime
import pickle
import hashlib

# Heavy imports moved to methods for faster startup

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
                 cache_dir: str = "cache/embeddings"):
        
        self.embedding_model = embedding_model
        
        # Auto-select appropriate model name if not provided
        if model_name is None:
            if embedding_model == "openai":
                self.model_name = "text-embedding-ada-002"
            else:  # local
                self.model_name = "all-MiniLM-L6-v2"
        else:
            self.model_name = model_name
            
        self.cache_embeddings = cache_embeddings
        self.cache_dir = cache_dir

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
    def create_openai(cls, model_name: str = "text-embedding-ada-002"):
        """Create an OpenAI embedding manager (requires API key)."""
        return cls(embedding_model="openai", model_name=model_name)

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
            self.embedding_dimension = 1536  # ada-002 dimension
            self.logger.info(f"✅ Initialized OpenAI embeddings with model: {self.model_name}")

        elif self.embedding_model == "local":
            # Lazy import SentenceTransformer
            from sentence_transformers import SentenceTransformer
            
            # Initialize local sentence transformer with proper model name
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"✅ Initialized local embeddings with model: {self.model_name}")

        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model}")
        
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
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text.strip()
            )

            embedding = np.array(response.data[0].embedding, dtype=np.float32)
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
            response = self.client.embeddings.create(
                model=self.model_name,
                input=[text.strip() for text in texts]
            )

            embeddings = []
            for data in response.data:
                embedding = np.array(data.embedding, dtype=np.float32)
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
        # Lazy import faiss
        import faiss
        
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        # Convert to numpy array for FAISS
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(embeddings_array)
        
        # Add to index
        start_idx = self.index.ntotal
        self.index.add(embeddings_array)
        
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
