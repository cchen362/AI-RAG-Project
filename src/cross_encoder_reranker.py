"""
Cross-Encoder Re-ranker for Multi-Source RAG
Uses BGE model to rank and select single best source
"""

import logging
import time
import os
import torch
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class CrossEncoderReRanker:
    """
    Re-ranker using cross-encoder models for intelligent source selection.
    
    This component replaces rule-based intent logic with semantic understanding
    to select the single most relevant source for each query.
    """
    
    def __init__(self, model_name: str = 'BAAI/bge-reranker-base', 
                 relevance_threshold: float = 0.0,
                 gpu_only_mode: bool = False):
        """
        Initialize the cross-encoder re-ranker.
        
        Args:
            model_name: HuggingFace model for cross-encoding
            relevance_threshold: Deprecated - threshold removed for better reliability
            gpu_only_mode: Force GPU-only execution for old CPU compatibility
        """
        self.model_name = model_name
        self.threshold = relevance_threshold  # Kept for compatibility but not used
        self.gpu_only_mode = gpu_only_mode or (os.getenv('MODEL_DEVICE') == 'cuda')
        self.model = None
        self.is_initialized = False
        
        if self.gpu_only_mode:
            logger.info(f"🎯 CrossEncoderReRanker configured with model: {model_name} (GPU-only mode)")
        else:
            logger.info(f"🎯 CrossEncoderReRanker configured with model: {model_name}")
    
    
    
    
    def _calculate_simple_combined_score(self, original_score: float, rerank_score: float, source_type: str, bias_colpali: bool) -> float:
        """
        Simple combined scoring with direct bias for chart/performance queries.
        
        This replaces the complex fitness scoring with straightforward logic.
        """
        is_explicit_chart = getattr(self, '_current_is_explicit_chart', False)
        # Source-specific minimum thresholds
        min_thresholds = {
            'text': 0.15,
            'colpali': 0.20,  # Further reduced to 0.20 to help ColPali compete
            'salesforce': 0.2
        }
        
        min_threshold = min_thresholds.get(source_type, 0.2)
        
        # Basic combined score with source-specific weighting
        if original_score < min_threshold:
            # Heavy penalty for low original scores
            combined = (original_score * 0.4 + rerank_score * 0.6) * 0.3
        else:
            # Different weighting for visual vs text sources
            if source_type == 'colpali':
                # For ColPali, weight original score more heavily since cross-encoder 
                # doesn't handle visual content descriptions well
                combined = original_score * 0.6 + rerank_score * 0.4
            else:
                # Standard balanced combination for text sources
                combined = original_score * 0.3 + rerank_score * 0.7
        
        # Apply direct bias for chart/performance queries
        if bias_colpali and source_type == 'colpali':
            if is_explicit_chart:
                # Very strong bias for explicit chart queries like "based on the chart"
                combined *= 2.5  # 150% boost - increased from 100%
                logger.info(f"  Applied STRONG ColPali bias for explicit chart query: {combined:.3f}")
            else:
                # Strong bias for general chart/performance queries
                combined *= 2.0  # 100% boost - increased from 50%
                logger.info(f"  Applied ColPali performance/chart bias: {combined:.3f}")
        elif source_type == 'text' and original_score > 0.4:
            if not is_explicit_chart:  # Don't boost TEXT for explicit chart queries
                combined *= 1.05  # Small boost for good text sources
                logger.debug(f"  Applied TEXT boost: {combined:.3f}")
        
        # Ensure reasonable range
        return max(0.0, min(1.0, combined))
    
        
    def initialize(self) -> bool:
        """Initialize the cross-encoder model"""
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"🔧 Loading cross-encoder model: {self.model_name}")
            
            # GPU-only mode configuration
            if self.gpu_only_mode:
                if not torch.cuda.is_available():
                    raise RuntimeError("GPU-only mode required but CUDA not available for CrossEncoder!")
                
                logger.info("🎮 Initializing CrossEncoder in GPU-only mode")
                device = 'cuda'
                
                # GPU memory cleanup before loading
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated(0) / 1024**3
                logger.info(f"🧹 GPU memory before CrossEncoder loading: {initial_memory:.2f}GB")
                
                # Load the BGE reranker model on GPU
                self.model = CrossEncoder(self.model_name, device=device)
                
                # GPU memory status after loading
                final_memory = torch.cuda.memory_allocated(0) / 1024**3
                model_memory = final_memory - initial_memory
                logger.info(f"🎮 CrossEncoder loaded: {model_memory:.2f}GB VRAM used")
                
            else:
                # Standard initialization
                self.model = CrossEncoder(self.model_name)
            
            self.is_initialized = True
            logger.info("✅ Cross-encoder model loaded successfully")
            return True
            
        except ImportError:
            logger.error("❌ sentence-transformers not installed. Run: pip install sentence-transformers")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to load cross-encoder model: {e}")
            return False
    
    def rank_all_sources(self, query: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Rank all candidate sources and select the single best one.
        
        Args:
            query: User's query string
            candidates: List of candidate results from different sources
            
        Returns:
            Dict with success status, selected source, and reasoning
        """
        if not self.is_initialized:
            if not self.initialize():
                return {
                    'success': False,
                    'error': 'Re-ranker model not available',
                    'selected_source': None
                }
        
        if not candidates:
            return {
                'success': False,
                'error': 'No candidates provided for re-ranking',
                'selected_source': None
            }
        
        logger.info(f"🔍 Re-ranking {len(candidates)} sources for query: '{query[:50]}...'")
        start_time = time.time()
        
        # Simple query analysis for direct bias
        query_lower = query.lower()
        is_chart_query = any(term in query_lower for term in ['chart', 'graph', 'diagram', 'figure', 'visualization'])
        is_performance_query = any(term in query_lower for term in ['time', 'performance', 'speed', 'latency', 'rate'])
        is_explicit_chart = 'based on the chart' in query_lower or 'from the chart' in query_lower
        
        bias_colpali = is_chart_query or is_performance_query
        if bias_colpali:
            logger.info(f"🎯 Chart/performance query detected - will boost ColPali scores (explicit_chart: {is_explicit_chart})")
        
        # Store for stronger bias logic
        self._current_is_explicit_chart = is_explicit_chart
        
        try:
            scored_candidates = []
            
            # Score each candidate with cross-encoder
            for candidate in candidates:
                if not candidate.get('success', False) or not candidate.get('answer'):
                    # Skip failed candidates
                    continue
                
                source_type = candidate.get('source_type', 'unknown')
                answer_content = candidate.get('answer', '')
                original_score = candidate.get('score', 0.0)
                
                # Use content as-is - let the source formatting help the re-ranker
                normalized_content = answer_content
                
                # Skip sources with very low original confidence (likely irrelevant)
                if original_score < 0.05:  # Very low threshold to filter obvious misses
                    logger.debug(f"  {source_type}: Skipped due to low original score ({original_score:.3f})")
                    continue
                
                # Cross-encoder prediction: query + normalized content → relevance score
                try:
                    score = self.model.predict([(query, normalized_content)])
                    
                    # Handle different score formats
                    if isinstance(score, (list, np.ndarray)):
                        score = float(score[0])
                    else:
                        score = float(score)
                    
                    # Simple combined scoring with direct bias
                    combined_score = self._calculate_simple_combined_score(original_score, score, source_type, bias_colpali)
                    
                    scored_candidates.append({
                        'source_type': source_type,
                        'answer': answer_content,
                        'sources': candidate.get('sources', []),
                        'rerank_score': score,
                        'original_score': original_score,
                        'combined_score': combined_score,
                        'metadata': candidate.get('metadata', {}),
                        'token_info': candidate.get('token_info', {}),
                        'normalized_content_length': len(normalized_content)
                    })
                    
                    logger.debug(f"  {source_type}: original={original_score:.3f}, rerank={score:.3f}, combined={combined_score:.3f}")
                    
                except Exception as e:
                    logger.warning(f"❌ Failed to score {source_type}: {e}")
                    continue
            
            ranking_time = time.time() - start_time
            
            if not scored_candidates:
                return {
                    'success': False,
                    'error': 'No candidates could be scored by re-ranker',
                    'ranking_time': ranking_time
                }
            
            # Sort by combined score (highest first) for better selection
            scored_candidates.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Get the best candidate
            best_candidate = scored_candidates[0]
            rejected_candidates = scored_candidates[1:]
            
            # Always select the best candidate - let users judge quality
            # Removed artificial threshold that was causing fallback behavior
            
            # Successful selection
            result = {
                'success': True,
                'selected_source': best_candidate,
                'rejected_sources': [
                    {
                        'source_type': c['source_type'],
                        'combined_score': c['combined_score'],
                        'original_score': c['original_score'],
                        'rerank_score': c['rerank_score'],
                        'reason': f"Lower combined score ({c['combined_score']:.3f} vs {best_candidate['combined_score']:.3f})"
                    }
                    for c in rejected_candidates
                ],
                'ranking_time': ranking_time,
                'reasoning': f"Selected {best_candidate['source_type']} (original: {best_candidate['original_score']:.3f}, rerank: {best_candidate['rerank_score']:.3f}, combined: {best_candidate['combined_score']:.3f})",
                'model_used': self.model_name,
                'total_candidates': len(scored_candidates)
            }
            
            logger.info(f"✅ Re-ranking complete: {best_candidate['source_type']} selected (combined: {best_candidate['combined_score']:.3f}) in {ranking_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Re-ranking failed: {e}")
            return {
                'success': False,
                'error': f'Re-ranking error: {str(e)}',
                'ranking_time': time.time() - start_time
            }
    
    def compare_sources(self, query: str, source1: Dict[str, Any], 
                       source2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two sources and return the better one.
        
        Useful for A/B testing or pairwise comparisons.
        """
        candidates = [source1, source2]
        result = self.rank_all_sources(query, candidates)
        
        if result['success']:
            winner = result['selected_source']
            loser = result['rejected_sources'][0] if result['rejected_sources'] else None
            
            return {
                'winner': winner['source_type'],
                'winner_score': winner['rerank_score'],
                'loser': loser['source_type'] if loser else None,
                'loser_score': loser['rerank_score'] if loser else None,
                'margin': winner['rerank_score'] - (loser['rerank_score'] if loser else 0),
                'confidence': 'high' if winner['rerank_score'] > 0.7 else 'medium' if winner['rerank_score'] > 0.5 else 'low'
            }
        else:
            return {
                'winner': None,
                'error': result['error']
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get re-ranker statistics and model info"""
        return {
            'model_name': self.model_name,
            'is_initialized': self.is_initialized,
            'relevance_threshold': self.threshold,
            'model_type': 'cross_encoder',
            'capabilities': [
                'multi_source_ranking',
                'semantic_relevance_scoring',
                'single_source_selection'
            ]
        }
    
    def update_threshold(self, new_threshold: float):
        """Update relevance threshold for source selection"""
        old_threshold = self.threshold
        self.threshold = new_threshold
        logger.info(f"🎛️ Relevance threshold updated: {old_threshold:.3f} → {new_threshold:.3f}")
    
    def clear_cache(self):
        """Clear any internal caches (if applicable)"""
        # BGE models don't typically have caches to clear
        # But this method provides interface consistency
        logger.info("🧹 Re-ranker cache cleared (no-op for BGE models)")

# Convenience function for easy integration
def create_reranker(model_name: str = 'BAAI/bge-reranker-base', 
                   threshold: float = 0.3) -> CrossEncoderReRanker:
    """
    Factory function to create and initialize a re-ranker.
    
    Args:
        model_name: BGE model name
        threshold: Minimum relevance threshold
        
    Returns:
        Initialized CrossEncoderReRanker instance
    """
    reranker = CrossEncoderReRanker(model_name, threshold)
    
    # Try to initialize immediately
    if reranker.initialize():
        logger.info("✅ Re-ranker created and initialized successfully")
    else:
        logger.warning("⚠️ Re-ranker created but initialization failed - will retry on first use")
    
    return reranker