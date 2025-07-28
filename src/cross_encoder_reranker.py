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
    
    def _normalize_content_for_ranking(self, content: str, source_type: str) -> str:
        """
        Normalize content to remove formatting bias in re-ranking.
        
        The BGE re-ranker should focus on content relevance, not presentation quality.
        This prevents well-formatted but irrelevant sources from getting unfair advantage.
        """
        import re
        
        if not content or not content.strip():
            return ""
        
        # Clean the content for fair comparison
        normalized = content
        
        # Remove markdown formatting that can bias the re-ranker
        normalized = re.sub(r'\*\*(.*?)\*\*', r'\1', normalized)  # **bold** -> bold
        normalized = re.sub(r'\*(.*?)\*', r'\1', normalized)      # *italic* -> italic
        normalized = re.sub(r'#{1,6}\s*', '', normalized)         # ### headers -> headers
        normalized = re.sub(r'`(.*?)`', r'\1', normalized)        # `code` -> code
        
        # Remove common ColPali formatting patterns that bias scoring
        normalized = re.sub(r'\*\*Page \d+ of [\'"][^\'"]*[\'"]:\*\*\s*', '', normalized)
        normalized = re.sub(r'📄 \*\*Source\*\*:.*?(?=\n|$)', '', normalized)
        normalized = re.sub(r'🔍 \*\*Query Context\*\*:.*?(?=\n|$)', '', normalized)
        normalized = re.sub(r'🔍 \*\*Relevance\*\*:.*?(?=\n|$)', '', normalized)
        
        # Remove excessive emojis and special characters that don't add content value
        normalized = re.sub(r'[📄🔍📊🎯✅❌⚠️💡🖼️🏢📝]', '', normalized)
        
        # Clean up whitespace
        normalized = re.sub(r'\n\s*\n\s*\n+', '\n\n', normalized)  # Multiple newlines -> double
        normalized = re.sub(r'^\s+|\s+$', '', normalized)          # Trim
        
        # Truncate very long content to prevent length bias
        max_length = 1000  # BGE models work best with shorter text
        if len(normalized) > max_length:
            # Try to break at sentence boundary
            truncated = normalized[:max_length]
            last_period = truncated.rfind('.')
            if last_period > max_length * 0.7:  # If period is reasonably close to end
                normalized = truncated[:last_period+1]
            else:
                normalized = truncated + "..."
        
        # Add source type context for better scoring
        type_prefix = {
            'text': 'Document content:',
            'colpali': 'Visual document content:',
            'salesforce': 'Knowledge base content:'
        }.get(source_type, 'Content:')
        
        result = f"{type_prefix} {normalized}".strip()
        
        logger.debug(f"📝 Normalized {source_type} content: {len(content)} -> {len(result)} chars")
        return result
    
    def _calculate_combined_score(self, original_score: float, rerank_score: float, source_type: str) -> float:
        """
        Calculate combined score from original retrieval confidence and re-ranker score.
        
        This prevents the re-ranker from completely overriding retrieval quality,
        while still allowing semantic re-ranking to improve selection.
        """
        # Source-specific minimum thresholds (stricter for visual sources)
        min_thresholds = {
            'text': 0.15,        # Text RAG is generally reliable
            'colpali': 0.4,      # ColPali needs higher confidence to be relevant
            'salesforce': 0.2    # Salesforce varies in quality
        }
        
        min_threshold = min_thresholds.get(source_type, 0.2)
        
        # If original score is below threshold, heavily penalize
        if original_score < min_threshold:
            penalty_factor = 0.3  # Reduce score significantly
            combined = (original_score * 0.4 + rerank_score * 0.6) * penalty_factor
        else:
            # Balanced combination for good original scores
            # Weight slightly toward re-ranker for semantic understanding
            combined = original_score * 0.3 + rerank_score * 0.7
        
        # Boost text sources slightly for factual queries
        if source_type == 'text' and original_score > 0.4:
            combined *= 1.1  # 10% boost for good text sources
        
        # Ensure score stays in reasonable range
        combined = max(0.0, min(1.0, combined))
        
        return combined
        
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
                original_score = candidate.get('score', 0.0)
                
                # Normalize content for fair re-ranking comparison
                normalized_content = self._normalize_content_for_ranking(answer_content, source_type)
                
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
                    
                    # Combine original confidence with re-ranker score
                    # Original score indicates retrieval relevance, re-ranker score indicates semantic relevance
                    combined_score = self._calculate_combined_score(original_score, score, source_type)
                    
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
            
            # Check if best candidate meets threshold (using combined score)
            if best_candidate['combined_score'] < self.threshold:
                return {
                    'success': False,
                    'error': f'Best source combined score ({best_candidate["combined_score"]:.3f}) below threshold ({self.threshold})',
                    'all_scores': [(c['source_type'], c['combined_score'], c['original_score'], c['rerank_score']) for c in scored_candidates],
                    'ranking_time': ranking_time
                }
            
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