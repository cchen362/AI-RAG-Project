"""
Debug ColPali Test App
Simple standalone test to isolate and fix the 0.000 scores issue.

This minimal app focuses solely on ColPali functionality without 
the complexity of the full RAG system.
"""

import os
import sys
import time
import logging
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from pdf2image import convert_from_path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class SimpleColPaliDebugger:
    """Minimal ColPali implementation for debugging."""
    
    def __init__(self):
        self.model_name = 'vidore/colqwen2-v1.0'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.processor = None
        
        logger.info(f"🔧 Device: {self.device}")
        
    def load_model(self):
        """Load ColPali model components."""
        logger.info(f"📥 Loading ColPali model: {self.model_name}")
        start_time = time.time()
        
        try:
            # Try colpali_engine first
            from colpali_engine.models import ColQwen2, ColQwen2Processor
            
            torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            
            self.model = ColQwen2.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=self.device,
                trust_remote_code=True
            ).eval()
            
            self.processor = ColQwen2Processor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            load_time = time.time() - start_time
            logger.info(f"✅ Model loaded successfully in {load_time:.2f}s")
            
        except ImportError:
            logger.error("❌ colpali_engine not available")
            raise
            
    def convert_pdf_to_images(self, pdf_path: str) -> List:
        """Convert PDF to images with simple approach."""
        logger.info(f"🔄 Converting PDF to images: {os.path.basename(pdf_path)}")
        
        try:
            images = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=5)
            logger.info(f"✅ Converted {len(images)} pages")
            return images
        except Exception as e:
            logger.error(f"❌ PDF conversion failed: {e}")
            return []
    
    def generate_document_embeddings(self, images: List) -> torch.Tensor:
        """Generate ColPali embeddings for document images."""
        logger.info(f"🔄 Generating embeddings for {len(images)} images")
        
        try:
            with torch.no_grad():
                # Process images with ColQwen2
                batch_inputs = self.processor.process_images(images)
                
                logger.info(f"📊 Processor input keys: {list(batch_inputs.keys())}")
                logger.info(f"📊 Input tensor shapes: {[(k, v.shape) for k, v in batch_inputs.items() if hasattr(v, 'shape')]}")
                
                # Move to device
                for key in batch_inputs:
                    if isinstance(batch_inputs[key], torch.Tensor):
                        batch_inputs[key] = batch_inputs[key].to(self.device)
                
                # Generate embeddings
                outputs = self.model(**batch_inputs)
                
                # Get embeddings
                if hasattr(outputs, 'last_hidden_state'):
                    embeddings = outputs.last_hidden_state
                else:
                    embeddings = outputs
                
                logger.info(f"✅ Generated embeddings shape: {embeddings.shape}")
                logger.info(f"📊 Embeddings dtype: {embeddings.dtype}, device: {embeddings.device}")
                logger.info(f"📊 Embeddings stats: min={embeddings.min().item():.4f}, max={embeddings.max().item():.4f}, mean={embeddings.mean().item():.4f}")
                
                return embeddings
                
        except Exception as e:
            logger.error(f"❌ Document embedding generation failed: {e}")
            raise
    
    def generate_query_embedding(self, query: str) -> torch.Tensor:
        """Generate ColPali embedding for query."""
        logger.info(f"🔄 Processing query: '{query}'")
        
        try:
            with torch.no_grad():
                # Process query with ColQwen2
                query_inputs = self.processor.process_queries([query])
                
                logger.info(f"📊 Query processor keys: {list(query_inputs.keys())}")
                logger.info(f"📊 Query tensor shapes: {[(k, v.shape) for k, v in query_inputs.items() if hasattr(v, 'shape')]}")
                
                # Move to device
                for key in query_inputs:
                    if isinstance(query_inputs[key], torch.Tensor):
                        query_inputs[key] = query_inputs[key].to(self.device)
                
                # Generate query embedding
                query_outputs = self.model(**query_inputs)
                
                # Get query embeddings
                if hasattr(query_outputs, 'last_hidden_state'):
                    query_embedding = query_outputs.last_hidden_state
                else:
                    query_embedding = query_outputs
                
                logger.info(f"✅ Generated query embedding shape: {query_embedding.shape}")
                logger.info(f"📊 Query embeddings dtype: {query_embedding.dtype}, device: {query_embedding.device}")
                logger.info(f"📊 Query stats: min={query_embedding.min().item():.4f}, max={query_embedding.max().item():.4f}, mean={query_embedding.mean().item():.4f}")
                
                return query_embedding
                
        except Exception as e:
            logger.error(f"❌ Query embedding generation failed: {e}")
            raise
    
    def calculate_maxsim_debug(self, query_embedding: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        """Debug version of MaxSim calculation with detailed logging."""
        logger.info("🔍 Starting MaxSim calculation debug...")
        
        try:
            # Log initial shapes
            logger.info(f"📊 Query embedding shape: {query_embedding.shape}")
            logger.info(f"📊 Document embeddings shape: {doc_embeddings.shape}")
            
            # Handle query embedding shape
            if len(query_embedding.shape) == 3:
                # Average over tokens: [batch, tokens, dim] -> [batch, dim]
                query_emb = query_embedding.mean(dim=1)
                logger.info(f"📊 Query averaged over tokens: {query_emb.shape}")
            else:
                query_emb = query_embedding
            
            if query_emb.shape[0] != 1:
                query_emb = query_emb.mean(dim=0, keepdim=True)
                logger.info(f"📊 Query averaged over batch: {query_emb.shape}")
            
            # Process each page
            page_scores = []
            
            for page_idx in range(doc_embeddings.shape[0]):
                page_patches = doc_embeddings[page_idx]  # [num_patches, embedding_dim]
                
                logger.info(f"📊 Page {page_idx}: query_emb {query_emb.shape}, page_patches {page_patches.shape}")
                
                # Check for zero embeddings
                query_norm_before = torch.norm(query_emb, dim=-1)
                patches_norm_before = torch.norm(page_patches, dim=-1)
                
                logger.info(f"📊 Page {page_idx}: Query norm before: {query_norm_before.mean():.6f}")
                logger.info(f"📊 Page {page_idx}: Patches norm before: {patches_norm_before.mean():.6f}")
                
                # Check for zero vectors
                zero_query = (query_norm_before < 1e-8).sum()
                zero_patches = (patches_norm_before < 1e-8).sum()
                logger.info(f"📊 Page {page_idx}: Zero query vectors: {zero_query}, Zero patch vectors: {zero_patches}")
                
                # Normalize embeddings with epsilon for numerical stability
                epsilon = 1e-8
                query_norm = torch.nn.functional.normalize(query_emb + epsilon, p=2, dim=-1)
                patches_norm = torch.nn.functional.normalize(page_patches + epsilon, p=2, dim=-1)
                
                logger.info(f"📊 Page {page_idx}: After normalization - query: {query_norm.shape}, patches: {patches_norm.shape}")
                
                # Calculate similarities using batch matrix multiplication
                similarities = torch.matmul(query_norm, patches_norm.T).squeeze(0)  # [num_patches]
                
                logger.info(f"📊 Page {page_idx}: Similarities shape: {similarities.shape}")
                logger.info(f"📊 Page {page_idx}: Similarities stats: min={similarities.min():.6f}, max={similarities.max():.6f}, mean={similarities.mean():.6f}")
                
                # MaxSim: take maximum similarity
                max_sim = similarities.max()
                page_scores.append(max_sim)
                
                logger.info(f"📊 Page {page_idx}: MaxSim score: {max_sim:.6f}")
            
            scores = torch.stack(page_scores)
            logger.info(f"✅ Final scores: {scores}")
            
            return scores
            
        except Exception as e:
            logger.error(f"❌ MaxSim calculation failed: {e}")
            raise
    
    def test_simple_similarity(self, query_embedding: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        """Test simple cosine similarity as baseline."""
        logger.info("🔍 Testing simple cosine similarity...")
        
        try:
            # Flatten embeddings and compute simple cosine similarity
            query_flat = query_embedding.flatten()
            doc_flat = doc_embeddings.flatten()
            
            # Normalize
            query_norm = torch.nn.functional.normalize(query_flat.unsqueeze(0), p=2, dim=-1)
            doc_norm = torch.nn.functional.normalize(doc_flat.unsqueeze(0), p=2, dim=-1)
            
            # Cosine similarity
            simple_sim = torch.cosine_similarity(query_norm, doc_norm, dim=-1)
            
            logger.info(f"📊 Simple cosine similarity: {simple_sim}")
            return simple_sim
            
        except Exception as e:
            logger.error(f"❌ Simple similarity failed: {e}")
            return torch.tensor([0.0])

def main():
    """Main debug test function."""
    logger.info("🚀 Starting ColPali Debug Test")
    
    # Initialize debugger
    debugger = SimpleColPaliDebugger()
    debugger.load_model()
    
    # Test document - you can change this path
    test_pdf = "data/documents/RAG_ColPali_Visual_Diagram_Only.pdf"
    
    if not os.path.exists(test_pdf):
        logger.error(f"❌ Test PDF not found: {test_pdf}")
        logger.info("💡 Please place a test PDF at 'data/documents/RAG_ColPali_Visual_Diagram_Only.pdf'")
        return
    
    try:
        # 1. Convert PDF to images
        images = debugger.convert_pdf_to_images(test_pdf)
        if not images:
            return
        
        # 2. Generate document embeddings
        doc_embeddings = debugger.generate_document_embeddings(images)
        
        # 3. Test queries
        test_queries = [
            "What is the retrieval time?",
            "chart performance diagram",
            "ColPali RAG pipeline",
            "visual embedding model"
        ]
        
        for query in test_queries:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 Testing query: '{query}'")
            logger.info(f"{'='*60}")
            
            # Generate query embedding
            query_embedding = debugger.generate_query_embedding(query)
            
            # Calculate MaxSim with debug info
            maxsim_scores = debugger.calculate_maxsim_debug(query_embedding, doc_embeddings)
            
            # Test simple similarity as baseline
            simple_scores = debugger.test_simple_similarity(query_embedding, doc_embeddings)
            
            logger.info(f"🎯 RESULTS for '{query}':")
            logger.info(f"   MaxSim scores: {maxsim_scores}")
            logger.info(f"   Simple similarity: {simple_scores}")
            
            if maxsim_scores.max() > 0.01:
                logger.info(f"✅ Non-zero scores detected!")
            else:
                logger.warning(f"⚠️ Zero/near-zero scores - debugging needed")
        
        logger.info(f"\n🎉 Debug test completed!")
        
    except Exception as e:
        logger.error(f"❌ Debug test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()