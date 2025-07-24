"""
Cross-Encoder Re-ranker for Multi-Source RAG
Uses BGE model to rank and select single best source
"""

import logging
import time
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
                 relevance_threshold: float = 0.3):
        """
        Initialize the cross-encoder re-ranker.
        
        Args:
            model_name: HuggingFace model for cross-encoding
            relevance_threshold: Minimum score to consider a source relevant
        """
        self.model_name = model_name
        self.threshold = relevance_threshold
        self.model = None
        self.is_initialized = False
        
        logger.info(f"🎯 CrossEncoderReRanker configured with model: {model_name}")
        
    def initialize(self) -> bool:
        """Initialize the cross-encoder model"""
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"🔧 Loading cross-encoder model: {self.model_name}")
            
            # Load the BGE reranker model
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
        
        try:
            scored_candidates = []
            
            # Score each candidate with cross-encoder
            for candidate in candidates:
                if not candidate.get('success', False) or not candidate.get('answer'):
                    # Skip failed candidates
                    continue
                
                source_type = candidate.get('source_type', 'unknown')
                answer_content = candidate.get('answer', '')
                
                # Cross-encoder prediction: query + content → relevance score
                try:
                    score = self.model.predict([(query, answer_content)])
                    
                    # Handle different score formats
                    if isinstance(score, (list, np.ndarray)):
                        score = float(score[0])
                    else:
                        score = float(score)
                    
                    scored_candidates.append({
                        'source_type': source_type,
                        'answer': answer_content,
                        'sources': candidate.get('sources', []),
                        'rerank_score': score,
                        'original_score': candidate.get('score', 0.0),
                        'metadata': candidate.get('metadata', {}),
                        'token_info': candidate.get('token_info', {})
                    })
                    
                    logger.debug(f"  {source_type}: {score:.3f}")
                    
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
            
            # Sort by re-rank score (highest first)
            scored_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            # Get the best candidate
            best_candidate = scored_candidates[0]
            rejected_candidates = scored_candidates[1:]
            
            # Check if best candidate meets threshold
            if best_candidate['rerank_score'] < self.threshold:
                return {
                    'success': False,
                    'error': f'Best source score ({best_candidate["rerank_score"]:.3f}) below threshold ({self.threshold})',
                    'all_scores': [(c['source_type'], c['rerank_score']) for c in scored_candidates],
                    'ranking_time': ranking_time
                }
            
            # Successful selection
            result = {
                'success': True,
                'selected_source': best_candidate,
                'rejected_sources': [
                    {
                        'source_type': c['source_type'],
                        'rerank_score': c['rerank_score'],
                        'reason': f"Lower relevance ({c['rerank_score']:.3f} vs {best_candidate['rerank_score']:.3f})"
                    }
                    for c in rejected_candidates
                ],
                'ranking_time': ranking_time,
                'reasoning': f"Cross-encoder selected {best_candidate['source_type']} with highest semantic relevance: {best_candidate['rerank_score']:.3f}",
                'model_used': self.model_name,
                'total_candidates': len(scored_candidates)
            }
            
            logger.info(f"✅ Re-ranking complete: {best_candidate['source_type']} selected (score: {best_candidate['rerank_score']:.3f}) in {ranking_time:.3f}s")
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