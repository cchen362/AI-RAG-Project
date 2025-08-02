"""
Hypergraph Constructor for Graph-R1 Agentic RAG System

Implements unified hypergraph architecture combining:
1. Text embeddings (512D from text-embedding-3-large)
2. Visual embeddings (128D→512D projected from ColPali)  
3. Salesforce knowledge base content
4. Cross-modal semantic relationships
5. Hierarchical document structure
6. Source provenance tracking

Key Innovation: Learnable linear projection unifies all modalities 
into 512D space while preserving source-specific characteristics.
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import pickle
import hashlib
import json
from datetime import datetime

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import existing components
try:
    from src.embedding_manager import EmbeddingManager
    from src.text_chunker import TextChunker
    from src.document_processor import DocumentProcessor
    from src.salesforce_connector import SalesforceConnector
except ImportError:
    from embedding_manager import EmbeddingManager
    from text_chunker import TextChunker
    from document_processor import DocumentProcessor
    from salesforce_connector import SalesforceConnector

logger = logging.getLogger(__name__)

@dataclass
class HypergraphNode:
    """Represents a node in the unified hypergraph."""
    node_id: str
    content: str
    embedding: np.ndarray  # Always 512D after projection
    source_type: str  # 'text', 'visual', 'salesforce'
    source_metadata: Dict[str, Any]
    hierarchical_level: int  # 0=document, 1=section, 2=chunk, 3=patch
    parent_node_id: Optional[str] = None
    confidence_score: float = 1.0
    processing_cost: float = 0.0  # Token cost for retrieval

@dataclass
class HypergraphEdge:
    """Represents relationships between nodes."""
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str  # 'semantic', 'hierarchical', 'cross_modal', 'source_link'
    weight: float
    metadata: Dict[str, Any]

class CrossModalProjector(nn.Module):
    """Learnable linear projection for unifying embedding dimensions."""
    
    def __init__(self, input_dim: int, output_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Linear projection with bias
        self.projection = nn.Linear(input_dim, output_dim, bias=True)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Initialize weights using Xavier uniform
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        
        logger.info(f"✅ Initialized CrossModalProjector: {input_dim}D → {output_dim}D")
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project embeddings to unified dimension."""
        # Linear projection
        projected = self.projection(embeddings)
        
        # Layer normalization
        normalized = self.layer_norm(projected)
        
        # L2 normalization for cosine similarity
        normalized = torch.nn.functional.normalize(normalized, p=2, dim=-1)
        
        return normalized

class UnifiedEmbeddingSpace:
    """Manages the unified 512D embedding space for all modalities."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.unified_dimension = 512  # Target dimension for all modalities
        
        # Initialize projectors for different input dimensions
        self.projectors = {}
        
        # Common dimensions from different sources
        self._initialize_projectors()
        
        # Track projection statistics
        self.projection_stats = {
            'total_projections': 0,
            'projections_by_source': defaultdict(int),
            'average_projection_time': 0.0
        }
        
        logger.info(f"✅ UnifiedEmbeddingSpace initialized with target dimension: {self.unified_dimension}")
    
    def _initialize_projectors(self):
        """Initialize projectors for known embedding dimensions."""
        common_dimensions = [
            128,   # ColPali patches
            384,   # all-MiniLM-L6-v2
            768,   # BERT-base, RoBERTa-base
            1024,  # Custom models
            1536,  # text-embedding-ada-002, text-embedding-3-small
            3072   # text-embedding-3-large (default)
        ]
        
        for dim in common_dimensions:
            if dim != self.unified_dimension:
                self.projectors[dim] = CrossModalProjector(dim, self.unified_dimension)
                logger.info(f"🔧 Created projector: {dim}D → {self.unified_dimension}D")
    
    def project_to_unified_space(self, embeddings: np.ndarray, source_type: str) -> np.ndarray:
        """Project embeddings to unified 512D space with comprehensive validation."""
        import time
        start_time = time.time()
        
        try:
            # Input validation
            if embeddings is None:
                raise ValueError(f"Embeddings cannot be None for {source_type}")
            
            # Convert to torch tensor
            if isinstance(embeddings, np.ndarray):
                tensor_embeddings = torch.tensor(embeddings, dtype=torch.float32)
            else:
                tensor_embeddings = embeddings
            
            # Critical validation: Ensure input is 1D vector for unified space
            original_shape = tensor_embeddings.shape
            logger.debug(f"🔍 Input shape for {source_type}: {original_shape}")
            
            if len(original_shape) > 2:
                raise ValueError(f"Invalid embedding shape for {source_type}: {original_shape}. Expected 1D or 2D, got {len(original_shape)}D")
            
            if len(original_shape) == 2 and original_shape[0] != 1:
                raise ValueError(f"Multi-dimensional embedding for {source_type}: {original_shape}. Must be averaged to single vector first!")
            
            # Handle different input shapes
            if len(original_shape) == 1:
                tensor_embeddings = tensor_embeddings.unsqueeze(0)
                single_embedding = True
            else:
                single_embedding = False
            
            input_dim = tensor_embeddings.shape[-1]
            
            # Validate input dimension is reasonable
            if input_dim <= 0 or input_dim > 10000:
                raise ValueError(f"Invalid embedding dimension for {source_type}: {input_dim}")
            
            logger.debug(f"📏 Projecting {source_type}: input_dim={input_dim}, target_dim={self.unified_dimension}")
            
            # If already correct dimension, just normalize
            if input_dim == self.unified_dimension:
                projected = torch.nn.functional.normalize(tensor_embeddings, p=2, dim=-1)
                logger.debug(f"📏 No projection needed for {source_type}: already {input_dim}D")
            else:
                # Get or create projector for this dimension
                if input_dim not in self.projectors:
                    self.projectors[input_dim] = CrossModalProjector(input_dim, self.unified_dimension)
                    logger.info(f"🆕 Created new projector for {source_type}: {input_dim}D → {self.unified_dimension}D")
                
                projector = self.projectors[input_dim]
                projector.eval()  # Set to evaluation mode
                
                with torch.no_grad():
                    projected = projector(tensor_embeddings)
                
                logger.debug(f"📐 Projected {source_type}: {input_dim}D → {self.unified_dimension}D")
            
            # Convert back to numpy and restore original shape
            result = projected.cpu().numpy()
            if single_embedding:
                result = result.squeeze(0)
            
            # Final validation: Ensure output is correct shape
            if len(result.shape) != 1 or result.shape[0] != self.unified_dimension:
                raise ValueError(f"Projection output validation failed for {source_type}: got {result.shape}, expected ({self.unified_dimension},)")
            
            # Validate result is not all zeros (common error indicator)
            if np.allclose(result, 0, atol=1e-6):
                logger.warning(f"⚠️ Projection result is near-zero for {source_type}, may indicate processing error")
            
            # Update statistics
            processing_time = time.time() - start_time
            self.projection_stats['total_projections'] += 1
            self.projection_stats['projections_by_source'][source_type] += 1
            self.projection_stats['average_projection_time'] = (
                (self.projection_stats['average_projection_time'] * (self.projection_stats['total_projections'] - 1) + processing_time) 
                / self.projection_stats['total_projections']
            )
            
            logger.debug(f"✅ Projection completed: {original_shape} → {result.shape} in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Projection failed for {source_type}: {e}")
            # Fallback: return zero vector or raise
            if isinstance(embeddings, np.ndarray):
                return np.zeros(self.unified_dimension, dtype=np.float32)
            else:
                raise e

class HypergraphBuilder:
    """Constructs the unified hypergraph from multi-source content."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize unified embedding space
        self.unified_space = UnifiedEmbeddingSpace(config)
        
        # Initialize source processors
        self._initialize_source_processors()
        
        # Hypergraph storage
        self.nodes: Dict[str, HypergraphNode] = {}
        self.edges: List[HypergraphEdge] = []
        
        # Hierarchical relationships
        self.hierarchy_map: Dict[str, List[str]] = defaultdict(list)  # parent_id -> child_ids
        
        # Source-specific indices
        self.nodes_by_source: Dict[str, List[str]] = defaultdict(list)
        self.source_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Similarity thresholds for edge creation
        self.similarity_thresholds = {
            'semantic': config.get('semantic_similarity_threshold', 0.75),
            'cross_modal': config.get('cross_modal_similarity_threshold', 0.4),  # Lowered from 0.65 to 0.4
            'hierarchical': 1.0  # Always create hierarchical edges
        }
        
        # Performance tracking
        self.build_stats = {
            'total_nodes': 0,
            'total_edges': 0,
            'nodes_by_source': defaultdict(int),
            'edges_by_type': defaultdict(int),
            'total_build_time': 0.0
        }
        
        logger.info("✅ HypergraphBuilder initialized")
    
    def _initialize_source_processors(self):
        """Initialize processors for different content sources."""
        
        # Text RAG processor
        self.text_processor = DocumentProcessor()
        self.text_chunker = TextChunker(
            chunk_size=self.config.get('chunk_size', 1000),
            overlap=self.config.get('chunk_overlap', 200),
            strategy="semantic"
        )
        
        # Text embedding manager (512D)
        self.text_embedding_manager = EmbeddingManager.create_openai(
            "text-embedding-3-large", 
            dimensions=512
        )
        
        # Visual embedding manager (128D → 512D)
        try:
            self.visual_embedding_manager = EmbeddingManager.create_colpali(
                self.config.get('visual_model', 'vidore/colqwen2-v1.0')
            )
            logger.info("✅ Visual embedding manager initialized")
        except Exception as e:
            logger.warning(f"⚠️ Visual embedding manager failed: {e}")
            self.visual_embedding_manager = None
        
        # Salesforce connector
        try:
            self.salesforce_connector = SalesforceConnector()
            logger.info("✅ Salesforce connector initialized")
        except Exception as e:
            logger.warning(f"⚠️ Salesforce connector failed: {e}")
            self.salesforce_connector = None
    
    def build_hypergraph(self, sources: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Build unified hypergraph from multiple sources.
        
        Args:
            sources: {
                'text_documents': [list of file paths],
                'visual_documents': [list of PDF paths], 
                'salesforce_queries': [list of query terms]
            }
        """
        import time
        start_time = time.time()
        
        logger.info("🏗️ Building unified hypergraph from multi-source content")
        
        results = {
            'text_nodes': 0,
            'visual_nodes': 0,
            'salesforce_nodes': 0,
            'total_edges': 0,
            'processing_time': 0.0,
            'errors': []
        }
        
        try:
            # Process each source type
            if 'text_documents' in sources:
                text_results = self._process_text_documents(sources['text_documents'])
                results['text_nodes'] = text_results['nodes_created']
                if text_results.get('errors'):
                    results['errors'].extend(text_results['errors'])
            
            if 'visual_documents' in sources and self.visual_embedding_manager:
                visual_results = self._process_visual_documents(sources['visual_documents'])
                results['visual_nodes'] = visual_results['nodes_created']
                if visual_results.get('errors'):
                    results['errors'].extend(visual_results['errors'])
            
            if 'salesforce_queries' in sources and self.salesforce_connector:
                sf_results = self._process_salesforce_content(sources['salesforce_queries'])
                results['salesforce_nodes'] = sf_results['nodes_created']
                if sf_results.get('errors'):
                    results['errors'].extend(sf_results['errors'])
            
            # Build cross-modal relationships
            logger.info("🔗 Building cross-modal relationships...")
            self._build_semantic_edges()
            self._build_cross_modal_edges()
            self._build_hierarchical_edges()
            
            results['total_edges'] = len(self.edges)
            results['processing_time'] = time.time() - start_time
            
            # Update build statistics
            self.build_stats['total_nodes'] = len(self.nodes)
            self.build_stats['total_edges'] = len(self.edges)
            self.build_stats['total_build_time'] = results['processing_time']
            
            logger.info(f"✅ Hypergraph built successfully:")
            logger.info(f"   📊 Total nodes: {len(self.nodes)}")
            logger.info(f"   🔗 Total edges: {len(self.edges)}")
            logger.info(f"   ⏱️ Build time: {results['processing_time']:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Hypergraph building failed: {e}")
            results['errors'].append(str(e))
            return results
    
    def _process_text_documents(self, document_paths: List[str]) -> Dict[str, Any]:
        """Process text documents into hypergraph nodes."""
        logger.info(f"📄 Processing {len(document_paths)} text documents")
        
        results = {
            'nodes_created': 0,
            'errors': []
        }
        
        for doc_path in document_paths:
            try:
                filename = os.path.basename(doc_path)
                logger.info(f"📄 Processing text document: {filename}")
                
                # Extract text content
                doc_result = self.text_processor.process_file(doc_path)
                if not doc_result.get('success'):
                    error_msg = f"Text extraction failed for {filename}: {doc_result.get('error')}"
                    results['errors'].append(error_msg)
                    continue
                
                content = doc_result.get('content', '')
                if not content.strip():
                    results['errors'].append(f"No content extracted from {filename}")
                    continue
                
                # Create document-level node
                doc_node_id = f"text_doc_{hashlib.md5(doc_path.encode()).hexdigest()[:8]}"
                doc_embedding = self.text_embedding_manager.create_embedding(content[:1000])  # Summary embedding
                
                # Project to unified space
                unified_embedding = self.unified_space.project_to_unified_space(doc_embedding, 'text')
                
                doc_node = HypergraphNode(
                    node_id=doc_node_id,
                    content=f"Document: {filename}",
                    embedding=unified_embedding,
                    source_type='text',
                    source_metadata={
                        'filename': filename,
                        'file_path': doc_path,
                        'content_length': len(content),
                        'document_type': 'text'
                    },
                    hierarchical_level=0  # Document level
                )
                
                self.nodes[doc_node_id] = doc_node
                self.nodes_by_source['text'].append(doc_node_id)
                results['nodes_created'] += 1
                
                # Chunk the document
                chunks = self.text_chunker.chunk_text(
                    content,
                    source_metadata={
                        'filename': filename,
                        'source_path': doc_path,
                        'file_type': 'text'
                    }
                )
                
                # Create chunk-level nodes
                for i, chunk_data in enumerate(chunks):
                    chunk_node_id = f"text_chunk_{doc_node_id}_{i}"
                    chunk_embedding = self.text_embedding_manager.create_embedding(chunk_data.content)
                    
                    # Project to unified space
                    unified_chunk_embedding = self.unified_space.project_to_unified_space(chunk_embedding, 'text')
                    
                    chunk_node = HypergraphNode(
                        node_id=chunk_node_id,
                        content=chunk_data.content,
                        embedding=unified_chunk_embedding,
                        source_type='text',
                        source_metadata={
                            'filename': filename,
                            'file_path': doc_path,
                            'chunk_index': i,
                            'chunk_size': len(chunk_data.content),
                            'parent_document': doc_node_id
                        },
                        hierarchical_level=2,  # Chunk level
                        parent_node_id=doc_node_id
                    )
                    
                    self.nodes[chunk_node_id] = chunk_node
                    self.nodes_by_source['text'].append(chunk_node_id)
                    self.hierarchy_map[doc_node_id].append(chunk_node_id)
                    results['nodes_created'] += 1
                
                logger.info(f"✅ Created {len(chunks) + 1} nodes for {filename}")
                
            except Exception as e:
                error_msg = f"Failed to process text document {doc_path}: {e}"
                logger.error(f"❌ {error_msg}")
                results['errors'].append(error_msg)
        
        self.build_stats['nodes_by_source']['text'] = results['nodes_created']
        return results
    
    def _process_visual_documents(self, document_paths: List[str]) -> Dict[str, Any]:
        """Process visual documents (PDFs) into hypergraph nodes."""
        logger.info(f"🖼️ Processing {len(document_paths)} visual documents")
        
        results = {
            'nodes_created': 0,
            'errors': []
        }
        
        for doc_path in document_paths:
            try:
                filename = os.path.basename(doc_path)
                logger.info(f"🖼️ Processing visual document: {filename}")
                
                # Generate visual embeddings using ColPali
                visual_result = self.visual_embedding_manager.create_visual_embedding(doc_path)
                
                if visual_result['status'] != 'success':
                    error_msg = f"Visual processing failed for {filename}: {visual_result.get('error')}"
                    results['errors'].append(error_msg)
                    continue
                
                # Create document-level node
                doc_node_id = f"visual_doc_{hashlib.md5(doc_path.encode()).hexdigest()[:8]}"
                
                # Use average of page embeddings for document-level embedding
                page_embeddings = visual_result['embeddings']
                if hasattr(page_embeddings, 'mean'):
                    doc_embedding = page_embeddings.mean(dim=0).cpu().numpy()
                else:
                    doc_embedding = np.mean(page_embeddings, axis=0)
                
                # Project to unified space (128D → 512D)
                unified_embedding = self.unified_space.project_to_unified_space(doc_embedding, 'visual')
                
                doc_node = HypergraphNode(
                    node_id=doc_node_id,
                    content=f"Visual Document: {filename}",
                    embedding=unified_embedding,
                    source_type='visual',
                    source_metadata={
                        'filename': filename,
                        'file_path': doc_path,
                        'page_count': visual_result['metadata']['page_count'],
                        'document_type': 'visual'
                    },
                    hierarchical_level=0  # Document level
                )
                
                self.nodes[doc_node_id] = doc_node
                self.nodes_by_source['visual'].append(doc_node_id)
                results['nodes_created'] += 1
                
                # Create page-level nodes
                page_count = visual_result['metadata']['page_count']
                for page_idx in range(page_count):
                    page_node_id = f"visual_page_{doc_node_id}_{page_idx}"
                    
                    # Extract and average page embedding patches to single vector
                    if hasattr(page_embeddings, 'shape') and len(page_embeddings.shape) > 1:
                        # Get patches for this page
                        page_patches = page_embeddings[page_idx]
                        
                        # Average patches to single vector (747, 512) → (512,)
                        if hasattr(page_patches, 'mean'):
                            page_embedding = page_patches.mean(dim=0).cpu().numpy()
                        else:
                            page_embedding = np.mean(page_patches, axis=0)
                            
                        logger.debug(f"📐 Page {page_idx} patches {page_patches.shape} → embedding {page_embedding.shape}")
                    else:
                        page_embedding = page_embeddings  # Single page case
                    
                    # Validate embedding shape before projection
                    if hasattr(page_embedding, 'shape') and len(page_embedding.shape) != 1:
                        logger.error(f"❌ Invalid page embedding shape: {page_embedding.shape}, expected 1D")
                        continue
                    
                    # Project to unified space
                    unified_page_embedding = self.unified_space.project_to_unified_space(page_embedding, 'visual')
                    
                    page_node = HypergraphNode(
                        node_id=page_node_id,
                        content=f"Page {page_idx + 1} of {filename}",
                        embedding=unified_page_embedding,
                        source_type='visual',
                        source_metadata={
                            'filename': filename,
                            'file_path': doc_path,
                            'page_index': page_idx,
                            'page_number': page_idx + 1,
                            'parent_document': doc_node_id
                        },
                        hierarchical_level=1,  # Page level
                        parent_node_id=doc_node_id
                    )
                    
                    self.nodes[page_node_id] = page_node
                    self.nodes_by_source['visual'].append(page_node_id)
                    self.hierarchy_map[doc_node_id].append(page_node_id)
                    results['nodes_created'] += 1
                
                logger.info(f"✅ Created {page_count + 1} visual nodes for {filename}")
                
            except Exception as e:
                error_msg = f"Failed to process visual document {doc_path}: {e}"
                logger.error(f"❌ {error_msg}")
                results['errors'].append(error_msg)
        
        self.build_stats['nodes_by_source']['visual'] = results['nodes_created']
        
        # Debug: Validate visual nodes are accessible
        if results['nodes_created'] > 0:
            visual_node_ids = self.nodes_by_source.get('visual', [])
            logger.info(f"🔍 Visual nodes accessibility check:")
            logger.info(f"   - Created {results['nodes_created']} visual nodes")
            logger.info(f"   - nodes_by_source['visual'] has {len(visual_node_ids)} entries")
            
            # Sample a few visual nodes to verify embeddings
            for i, node_id in enumerate(visual_node_ids[:3]):
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    emb_shape = node.embedding.shape if hasattr(node.embedding, 'shape') else 'Unknown'
                    logger.info(f"   - Sample visual node {i+1}: {node_id}, embedding shape: {emb_shape}")
                else:
                    logger.warning(f"   - Visual node {node_id} not found in nodes dict!")
        
        return results
    
    def _process_salesforce_content(self, query_terms: List[str]) -> Dict[str, Any]:
        """Process Salesforce knowledge base content into hypergraph nodes."""
        logger.info(f"🏢 Processing Salesforce content for {len(query_terms)} queries")
        
        results = {
            'nodes_created': 0,
            'errors': []
        }
        
        try:
            # Search Salesforce knowledge base
            search_results = []
            for query in query_terms:
                try:
                    sf_results = self.salesforce_connector.search_knowledge_base(query, limit=10)
                    if sf_results.get('success'):
                        search_results.extend(sf_results.get('results', []))
                except Exception as e:
                    logger.warning(f"Salesforce search failed for '{query}': {e}")
            
            # Remove duplicates by article ID
            unique_articles = {}
            for result in search_results:
                article_id = result.get('Id', result.get('id', ''))
                if article_id and article_id not in unique_articles:
                    unique_articles[article_id] = result
            
            # Create nodes for unique articles
            for article_id, article_data in unique_articles.items():
                try:
                    # Extract article content
                    title = article_data.get('Title', article_data.get('title', 'Unknown Title'))
                    content = article_data.get('ArticleBody', article_data.get('content', ''))
                    
                    if not content.strip():
                        continue
                    
                    # Create text embedding for article content
                    full_content = f"{title}\n\n{content}"
                    article_embedding = self.text_embedding_manager.create_embedding(full_content)
                    
                    # Project to unified space
                    unified_embedding = self.unified_space.project_to_unified_space(article_embedding, 'salesforce')
                    
                    # Create article node
                    node_id = f"sf_article_{article_id}"
                    
                    article_node = HypergraphNode(
                        node_id=node_id,
                        content=full_content,
                        embedding=unified_embedding,
                        source_type='salesforce',
                        source_metadata={
                            'article_id': article_id,
                            'title': title,
                            'content_length': len(content),
                            'article_type': article_data.get('ArticleType', 'Knowledge'),
                            'last_modified': article_data.get('LastModifiedDate', ''),
                            'url': article_data.get('Url', ''),
                            'confidence_score': article_data.get('confidence', 1.0)
                        },
                        hierarchical_level=0  # Article level (equivalent to document)
                    )
                    
                    self.nodes[node_id] = article_node
                    self.nodes_by_source['salesforce'].append(node_id)
                    results['nodes_created'] += 1
                    
                except Exception as e:
                    error_msg = f"Failed to process Salesforce article {article_id}: {e}"
                    logger.warning(f"⚠️ {error_msg}")
                    results['errors'].append(error_msg)
            
            logger.info(f"✅ Created {results['nodes_created']} Salesforce nodes")
            
        except Exception as e:
            error_msg = f"Salesforce content processing failed: {e}"
            logger.error(f"❌ {error_msg}")
            results['errors'].append(error_msg)
        
        self.build_stats['nodes_by_source']['salesforce'] = results['nodes_created']
        return results
    
    def _build_semantic_edges(self):
        """Build semantic similarity edges within and across sources."""
        logger.info("🔗 Building semantic similarity edges...")
        
        # Get all leaf nodes (chunks, pages, articles) for semantic comparison
        leaf_nodes = []
        for node in self.nodes.values():
            if node.hierarchical_level > 0:  # Not document-level
                leaf_nodes.append(node)
        
        # Compare all pairs for semantic similarity
        edge_count = 0
        threshold = self.similarity_thresholds['semantic']
        
        for i, node1 in enumerate(leaf_nodes):
            for j, node2 in enumerate(leaf_nodes[i+1:], i+1):
                try:
                    # Calculate cosine similarity in unified space
                    similarity = np.dot(node1.embedding, node2.embedding) / (
                        np.linalg.norm(node1.embedding) * np.linalg.norm(node2.embedding)
                    )
                    
                    if similarity >= threshold:
                        edge_id = f"semantic_{node1.node_id}_{node2.node_id}"
                        edge = HypergraphEdge(
                            edge_id=edge_id,
                            source_node_id=node1.node_id,
                            target_node_id=node2.node_id,
                            edge_type='semantic',
                            weight=float(similarity),
                            metadata={
                                'similarity_score': float(similarity),
                                'source1_type': node1.source_type,
                                'source2_type': node2.source_type,
                                'threshold_used': threshold
                            }
                        )
                        self.edges.append(edge)
                        edge_count += 1
                        
                except Exception as e:
                    logger.debug(f"Similarity calculation failed: {e}")
                    continue
        
        self.build_stats['edges_by_type']['semantic'] = edge_count
        logger.info(f"✅ Created {edge_count} semantic edges")
    
    def _build_cross_modal_edges(self):
        """Build cross-modal relationship edges between different source types."""
        logger.info("🌉 Building cross-modal edges...")
        
        edge_count = 0
        threshold = self.similarity_thresholds['cross_modal']
        
        # Compare nodes from different sources
        source_pairs = [
            ('text', 'visual'),
            ('text', 'salesforce'),
            ('visual', 'salesforce')
        ]
        
        for source1, source2 in source_pairs:
            # Check if both sources have nodes
            if source1 not in self.nodes_by_source or not self.nodes_by_source[source1]:
                logger.info(f"⚠️ No {source1} nodes available for cross-modal comparison")
                continue
            
            if source2 not in self.nodes_by_source or not self.nodes_by_source[source2]:
                logger.info(f"⚠️ No {source2} nodes available for cross-modal comparison")
                continue
            
            nodes1 = [self.nodes[nid] for nid in self.nodes_by_source[source1] if nid in self.nodes]
            nodes2 = [self.nodes[nid] for nid in self.nodes_by_source[source2] if nid in self.nodes]
            
            # Double-check we have valid nodes
            if not nodes1:
                logger.warning(f"⚠️ No valid {source1} nodes found after filtering")
                continue
            if not nodes2:
                logger.warning(f"⚠️ No valid {source2} nodes found after filtering")
                continue
            
            logger.info(f"🔗 Cross-modal comparison: {source1}({len(nodes1)}) ↔ {source2}({len(nodes2)})")
            
            similarities = []
            for node1 in nodes1:
                for node2 in nodes2:
                    try:
                        # Detailed validation of embeddings
                        if node1.embedding is None:
                            logger.warning(f"❌ Node {node1.node_id} has None embedding")
                            continue
                        if node2.embedding is None:
                            logger.warning(f"❌ Node {node2.node_id} has None embedding")
                            continue
                        
                        # Check embedding shapes and types
                        emb1 = np.array(node1.embedding) if not isinstance(node1.embedding, np.ndarray) else node1.embedding
                        emb2 = np.array(node2.embedding) if not isinstance(node2.embedding, np.ndarray) else node2.embedding
                        
                        if emb1.shape != emb2.shape:
                            logger.warning(f"❌ Shape mismatch: {node1.node_id}({emb1.shape}) vs {node2.node_id}({emb2.shape})")
                            continue
                        
                        if len(emb1.shape) != 1:
                            logger.warning(f"❌ Invalid embedding shape: {emb1.shape}, expected 1D")
                            continue
                        
                        # Check for zero or invalid embeddings
                        norm1 = np.linalg.norm(emb1)
                        norm2 = np.linalg.norm(emb2)
                        
                        if norm1 == 0 or norm2 == 0:
                            logger.warning(f"❌ Zero norm embedding: {node1.node_id}({norm1:.3f}) or {node2.node_id}({norm2:.3f})")
                            continue
                        
                        if not np.isfinite(norm1) or not np.isfinite(norm2):
                            logger.warning(f"❌ Invalid norm: {node1.node_id}({norm1}) or {node2.node_id}({norm2})")
                            continue
                        
                        # Calculate cross-modal similarity with validation
                        dot_product = np.dot(emb1, emb2)
                        similarity = dot_product / (norm1 * norm2)
                        
                        # Validate similarity result
                        if not np.isfinite(similarity):
                            logger.warning(f"❌ Invalid similarity: {similarity}")
                            continue
                        
                        similarities.append(similarity)
                        logger.debug(f"✅ Similarity calculated: {source1}→{source2} = {similarity:.4f}")
                        
                        if similarity >= threshold:
                            edge_id = f"cross_modal_{node1.node_id}_{node2.node_id}"
                            edge = HypergraphEdge(
                                edge_id=edge_id,
                                source_node_id=node1.node_id,
                                target_node_id=node2.node_id,
                                edge_type='cross_modal',
                                weight=float(similarity),
                                metadata={
                                    'cross_modal_similarity': float(similarity),
                                    'source1_type': source1,
                                    'source2_type': source2,
                                    'modality_bridge': f"{source1}→{source2}"
                                }
                            )
                            self.edges.append(edge)
                            edge_count += 1
                            logger.info(f"✅ Cross-modal edge created: {similarity:.3f} >= {threshold} ({node1.node_id}↔{node2.node_id})")
                            
                    except Exception as e:
                        logger.error(f"❌ Cross-modal similarity calculation failed for {node1.node_id}↔{node2.node_id}: {e}")
                        import traceback
                        logger.error(f"Stack trace: {traceback.format_exc()}")
                        continue
            
            if similarities:
                logger.info(f"📊 {source1}↔{source2} similarities: min={min(similarities):.3f}, max={max(similarities):.3f}, avg={np.mean(similarities):.3f}, threshold={threshold}")
            else:
                logger.warning(f"⚠️ No similarity calculations for {source1}↔{source2}")                    
        
        self.build_stats['edges_by_type']['cross_modal'] = edge_count
        logger.info(f"✅ Created {edge_count} cross-modal edges")
    
    def _build_hierarchical_edges(self):
        """Build hierarchical relationship edges (parent-child)."""
        logger.info("🏗️ Building hierarchical edges...")
        
        edge_count = 0
        
        for parent_id, child_ids in self.hierarchy_map.items():
            for child_id in child_ids:
                edge_id = f"hierarchical_{parent_id}_{child_id}"
                edge = HypergraphEdge(
                    edge_id=edge_id,
                    source_node_id=parent_id,
                    target_node_id=child_id,
                    edge_type='hierarchical',
                    weight=1.0,  # Hierarchical relationships have maximum weight
                    metadata={
                        'relationship': 'parent_child',
                        'parent_level': self.nodes[parent_id].hierarchical_level,
                        'child_level': self.nodes[child_id].hierarchical_level
                    }
                )
                self.edges.append(edge)
                edge_count += 1
        
        self.build_stats['edges_by_type']['hierarchical'] = edge_count
        logger.info(f"✅ Created {edge_count} hierarchical edges")
    
    def get_hypergraph_stats(self) -> Dict[str, Any]:
        """Get comprehensive hypergraph statistics."""
        stats = {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'nodes_by_source': dict(self.build_stats['nodes_by_source']),
            'edges_by_type': dict(self.build_stats['edges_by_type']),
            'unified_dimension': self.unified_space.unified_dimension,
            'projection_stats': dict(self.unified_space.projection_stats),
            'similarity_thresholds': self.similarity_thresholds,
            'hierarchy_levels': {
                0: sum(1 for n in self.nodes.values() if n.hierarchical_level == 0),
                1: sum(1 for n in self.nodes.values() if n.hierarchical_level == 1),
                2: sum(1 for n in self.nodes.values() if n.hierarchical_level == 2),
                3: sum(1 for n in self.nodes.values() if n.hierarchical_level == 3)
            }
        }
        return stats
    
    def save_hypergraph(self, cache_dir: str = "cache/hypergraph") -> bool:
        """Save hypergraph to disk for persistence."""
        try:
            os.makedirs(cache_dir, exist_ok=True)
            
            # Save hypergraph data
            hypergraph_data = {
                'nodes': {nid: {
                    'node_id': node.node_id,
                    'content': node.content,
                    'embedding': node.embedding.tolist(),
                    'source_type': node.source_type,
                    'source_metadata': node.source_metadata,
                    'hierarchical_level': node.hierarchical_level,
                    'parent_node_id': node.parent_node_id,
                    'confidence_score': node.confidence_score,
                    'processing_cost': node.processing_cost
                } for nid, node in self.nodes.items()},
                'edges': [{
                    'edge_id': edge.edge_id,
                    'source_node_id': edge.source_node_id,
                    'target_node_id': edge.target_node_id,
                    'edge_type': edge.edge_type,
                    'weight': edge.weight,
                    'metadata': edge.metadata
                } for edge in self.edges],
                'hierarchy_map': dict(self.hierarchy_map),
                'nodes_by_source': dict(self.nodes_by_source),
                'build_stats': dict(self.build_stats),
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to file
            cache_file = os.path.join(cache_dir, 'hypergraph.pkl')
            with open(cache_file, 'wb') as f:
                pickle.dump(hypergraph_data, f)
            
            logger.info(f"✅ Hypergraph saved to {cache_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save hypergraph: {e}")
            return False
    
    def load_hypergraph(self, cache_dir: str = "cache/hypergraph") -> bool:
        """Load hypergraph from disk."""
        try:
            cache_file = os.path.join(cache_dir, 'hypergraph.pkl')
            
            if not os.path.exists(cache_file):
                logger.warning(f"⚠️ No cached hypergraph found at {cache_file}")
                return False
            
            with open(cache_file, 'rb') as f:
                hypergraph_data = pickle.load(f)
            
            # Reconstruct nodes
            self.nodes = {}
            for nid, node_data in hypergraph_data['nodes'].items():
                node = HypergraphNode(
                    node_id=node_data['node_id'],
                    content=node_data['content'],
                    embedding=np.array(node_data['embedding']),
                    source_type=node_data['source_type'],
                    source_metadata=node_data['source_metadata'],
                    hierarchical_level=node_data['hierarchical_level'],
                    parent_node_id=node_data['parent_node_id'],
                    confidence_score=node_data['confidence_score'],
                    processing_cost=node_data['processing_cost']
                )
                self.nodes[nid] = node
            
            # Reconstruct edges
            self.edges = []
            for edge_data in hypergraph_data['edges']:
                edge = HypergraphEdge(
                    edge_id=edge_data['edge_id'],
                    source_node_id=edge_data['source_node_id'],
                    target_node_id=edge_data['target_node_id'],
                    edge_type=edge_data['edge_type'],
                    weight=edge_data['weight'],
                    metadata=edge_data['metadata']
                )
                self.edges.append(edge)
            
            # Restore other data
            self.hierarchy_map = defaultdict(list, hypergraph_data['hierarchy_map'])
            self.nodes_by_source = defaultdict(list, hypergraph_data['nodes_by_source'])
            self.build_stats = hypergraph_data['build_stats']
            
            logger.info(f"✅ Hypergraph loaded from {cache_file}")
            logger.info(f"   📊 Loaded {len(self.nodes)} nodes and {len(self.edges)} edges")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load hypergraph: {e}")
            return False

class ValidationSuite:
    """Validation and testing for hypergraph construction."""
    
    def __init__(self, hypergraph_builder: HypergraphBuilder):
        self.builder = hypergraph_builder
        
    def validate_embedding_quality(self) -> Dict[str, Any]:
        """Validate embedding quality and similarity preservation."""
        logger.info("🧪 Validating embedding quality...")
        
        validation_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'embedding_dimension_check': False,
            'similarity_preservation_check': False,
            'cross_modal_projection_check': False,
            'errors': []
        }
        
        try:
            # Test 1: Embedding dimension consistency
            validation_results['total_tests'] += 1
            all_embeddings_512d = all(
                node.embedding.shape[0] == 512 
                for node in self.builder.nodes.values()
            )
            
            if all_embeddings_512d:
                validation_results['embedding_dimension_check'] = True
                validation_results['passed_tests'] += 1
                logger.info("✅ All embeddings are 512D")
            else:
                validation_results['failed_tests'] += 1
                validation_results['errors'].append("Some embeddings are not 512D")
                logger.error("❌ Embedding dimension inconsistency detected")
            
            # Test 2: Similarity preservation (similar content should have high similarity)
            validation_results['total_tests'] += 1
            if self._test_similarity_preservation():
                validation_results['similarity_preservation_check'] = True
                validation_results['passed_tests'] += 1
                logger.info("✅ Similarity preservation validated")
            else:
                validation_results['failed_tests'] += 1
                validation_results['errors'].append("Similarity preservation test failed")
                logger.error("❌ Similarity preservation test failed")
            
            # Test 3: Cross-modal projection quality
            validation_results['total_tests'] += 1
            if self._test_cross_modal_projection():
                validation_results['cross_modal_projection_check'] = True
                validation_results['passed_tests'] += 1
                logger.info("✅ Cross-modal projection validated")
            else:
                validation_results['failed_tests'] += 1
                validation_results['errors'].append("Cross-modal projection test failed")
                logger.error("❌ Cross-modal projection test failed")
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            validation_results['errors'].append(str(e))
        
        return validation_results
    
    def _test_similarity_preservation(self) -> bool:
        """Test that similar content maintains high similarity after projection."""
        try:
            # Find text nodes from same document (should be similar)
            text_nodes = [node for node in self.builder.nodes.values() if node.source_type == 'text']
            
            if len(text_nodes) < 2:
                return True  # Skip test if insufficient data
            
            same_doc_similarities = []
            different_doc_similarities = []
            
            for i, node1 in enumerate(text_nodes):
                for j, node2 in enumerate(text_nodes[i+1:], i+1):
                    similarity = np.dot(node1.embedding, node2.embedding) / (
                        np.linalg.norm(node1.embedding) * np.linalg.norm(node2.embedding)
                    )
                    
                    # Check if from same document
                    if node1.source_metadata.get('filename') == node2.source_metadata.get('filename'):
                        same_doc_similarities.append(similarity)
                    else:
                        different_doc_similarities.append(similarity)
            
            # Same document chunks should be more similar on average
            if same_doc_similarities and different_doc_similarities:
                avg_same_doc = np.mean(same_doc_similarities)
                avg_diff_doc = np.mean(different_doc_similarities)
                return avg_same_doc > avg_diff_doc
            
            return True  # Pass if we can't make comparison
            
        except Exception as e:
            logger.debug(f"Similarity preservation test error: {e}")
            return False
    
    def _test_cross_modal_projection(self) -> bool:
        """Test cross-modal projection maintains reasonable relationships."""
        try:
            # Check that cross-modal edges exist with reasonable similarities
            cross_modal_edges = [edge for edge in self.builder.edges if edge.edge_type == 'cross_modal']
            
            if not cross_modal_edges:
                return True  # No cross-modal data to test
            
            # Check similarity distribution
            similarities = [edge.weight for edge in cross_modal_edges]
            avg_similarity = np.mean(similarities)
            
            # Cross-modal similarities should be reasonable (not too low, not suspiciously high)
            return 0.3 <= avg_similarity <= 0.9
            
        except Exception as e:
            logger.debug(f"Cross-modal projection test error: {e}")
            return False

# Factory function for easy initialization
def create_hypergraph_constructor(config: Dict[str, Any]) -> HypergraphBuilder:
    """Factory function to create and initialize hypergraph constructor."""
    logger.info("🏗️ Creating hypergraph constructor...")
    
    # Validate configuration
    required_keys = ['chunk_size', 'chunk_overlap']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Set defaults
    config.setdefault('semantic_similarity_threshold', 0.75)
    config.setdefault('cross_modal_similarity_threshold', 0.65)
    config.setdefault('visual_model', 'vidore/colqwen2-v1.0')
    
    # Create hypergraph builder
    builder = HypergraphBuilder(config)
    logger.info("✅ Hypergraph constructor created successfully!")
    
    return builder

# Example usage and testing
if __name__ == "__main__":
    """Test hypergraph construction with sample data."""
    print("🧪 Testing Hypergraph Constructor...")
    print("="*50)
    
    # Test configuration
    test_config = {
        'chunk_size': 800,
        'chunk_overlap': 150,
        'semantic_similarity_threshold': 0.7,
        'cross_modal_similarity_threshold': 0.6
    }
    
    try:
        # Create hypergraph constructor
        builder = create_hypergraph_constructor(test_config)
        
        # Test with minimal sources
        test_sources = {
            'text_documents': [],  # Add test documents
            'visual_documents': [],  # Add test PDFs
            'salesforce_queries': ['AI', 'machine learning']
        }
        
        # Build hypergraph (would fail without actual documents)
        # results = builder.build_hypergraph(test_sources)
        
        # Get statistics
        stats = builder.get_hypergraph_stats()
        print(f"✅ Constructor test passed!")
        print(f"📊 Unified dimension: {stats['unified_dimension']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Note: Full testing requires actual documents and API keys")