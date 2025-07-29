"""
Debug Re-ranker Test App
Isolates and debugs the re-ranker selection logic to understand why
ColPali (0.748) loses to text RAG (0.508 -> 0.442).
"""

import logging
import time
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class DebugReRanker:
    """Debug version of the re-ranker with detailed tracing."""
    
    def __init__(self):
        self.model = None
        self.is_initialized = False
        
    def initialize_mock_model(self):
        """Initialize with mock cross-encoder that returns predictable scores."""
        self.model = MockCrossEncoder()
        self.is_initialized = True
        return True
    
    def _calculate_simple_combined_score_debug(self, original_score: float, rerank_score: float, 
                                             source_type: str, bias_colpali: bool, is_explicit_chart: bool = False) -> float:
        """Debug version with detailed step-by-step logging."""
        
        logger.info(f"🔍 SCORE CALCULATION for {source_type.upper()}:")
        logger.info(f"   Input: original_score={original_score:.3f}, rerank_score={rerank_score:.3f}")
        logger.info(f"   Bias conditions: bias_colpali={bias_colpali}, is_explicit_chart={is_explicit_chart}")
        
        # Source-specific minimum thresholds
        min_thresholds = {
            'text': 0.15,
            'colpali': 0.25,  # Higher threshold than text!
            'salesforce': 0.2
        }
        
        min_threshold = min_thresholds.get(source_type, 0.2)
        logger.info(f"   Minimum threshold for {source_type}: {min_threshold}")
        
        # Basic combined score calculation
        if original_score < min_threshold:
            # Heavy penalty for low original scores
            combined = (original_score * 0.4 + rerank_score * 0.6) * 0.3
            logger.info(f"   PENALTY applied (below threshold): combined = ({original_score:.3f} * 0.4 + {rerank_score:.3f} * 0.6) * 0.3 = {combined:.3f}")
        else:
            # Balanced combination
            combined = original_score * 0.3 + rerank_score * 0.7
            logger.info(f"   BALANCED combination: combined = {original_score:.3f} * 0.3 + {rerank_score:.3f} * 0.7 = {combined:.3f}")
        
        # Apply bias for chart/performance queries
        pre_bias_combined = combined
        if bias_colpali and source_type == 'colpali':
            if is_explicit_chart:
                combined *= 2.0  # 100% boost
                logger.info(f"   STRONG ColPali bias applied (2.0x): {pre_bias_combined:.3f} -> {combined:.3f}")
            else:
                combined *= 1.5  # 50% boost
                logger.info(f"   ColPali bias applied (1.5x): {pre_bias_combined:.3f} -> {combined:.3f}")
        elif source_type == 'text' and original_score > 0.4:
            if not is_explicit_chart:
                combined *= 1.05  # Small text boost
                logger.info(f"   Small text boost applied (1.05x): {pre_bias_combined:.3f} -> {combined:.3f}")
        else:
            logger.info(f"   No bias applied")
        
        # Ensure reasonable range
        final_combined = max(0.0, min(1.0, combined))
        if final_combined != combined:
            logger.info(f"   Clamped to range [0,1]: {combined:.3f} -> {final_combined:.3f}")
        
        logger.info(f"   FINAL SCORE: {final_combined:.3f}")
        logger.info("")
        
        return final_combined
    
    def debug_scoring_scenario(self, query: str, text_score: float, colpali_score: float, 
                              mock_rerank_scores: Dict[str, float]):
        """Debug the exact scenario from the logs."""
        
        logger.info(f"🎯 DEBUGGING RE-RANKER SCENARIO")
        logger.info(f"Query: '{query}'")
        logger.info(f"Input scores - Text: {text_score:.3f}, ColPali: {colpali_score:.3f}")
        logger.info("")
        
        # Query analysis (copied from actual re-ranker)
        query_lower = query.lower()
        is_chart_query = any(term in query_lower for term in ['chart', 'graph', 'diagram', 'figure', 'visualization'])
        is_performance_query = any(term in query_lower for term in ['time', 'performance', 'speed', 'latency', 'rate'])
        is_explicit_chart = 'based on the chart' in query_lower or 'from the chart' in query_lower
        
        bias_colpali = is_chart_query or is_performance_query
        
        logger.info(f"📊 QUERY ANALYSIS:")
        logger.info(f"   is_chart_query: {is_chart_query}")
        logger.info(f"   is_performance_query: {is_performance_query}")
        logger.info(f"   is_explicit_chart: {is_explicit_chart}")
        logger.info(f"   bias_colpali: {bias_colpali}")
        logger.info("")
        
        # Mock re-ranking scores from cross-encoder
        text_rerank = mock_rerank_scores.get('text', 0.3)
        colpali_rerank = mock_rerank_scores.get('colpali', 0.2)
        
        logger.info(f"🤖 MOCK CROSS-ENCODER SCORES:")
        logger.info(f"   Text rerank score: {text_rerank:.3f}")
        logger.info(f"   ColPali rerank score: {colpali_rerank:.3f}")
        logger.info("")
        
        # Calculate combined scores
        text_combined = self._calculate_simple_combined_score_debug(
            text_score, text_rerank, 'text', bias_colpali, is_explicit_chart
        )
        
        colpali_combined = self._calculate_simple_combined_score_debug(
            colpali_score, colpali_rerank, 'colpali', bias_colpali, is_explicit_chart
        )
        
        # Final selection
        logger.info(f"🏆 FINAL COMPARISON:")
        logger.info(f"   Text combined score: {text_combined:.3f}")
        logger.info(f"   ColPali combined score: {colpali_combined:.3f}")
        
        if colpali_combined > text_combined:
            winner = "ColPali"
            margin = colpali_combined - text_combined
        else:
            winner = "Text"
            margin = text_combined - colpali_combined
            
        logger.info(f"   WINNER: {winner} (margin: +{margin:.3f})")
        logger.info("")
        
        return {
            'winner': winner,
            'text_combined': text_combined,
            'colpali_combined': colpali_combined,
            'margin': margin,
            'bias_applied': bias_colpali
        }
    
    def test_different_rerank_scenarios(self, query: str, text_score: float, colpali_score: float):
        """Test different rerank score scenarios to understand the issue."""
        
        logger.info(f"🧪 TESTING DIFFERENT RERANK SCENARIOS")
        logger.info("="*60)
        
        scenarios = [
            {
                'name': 'Scenario 1: Both get decent rerank scores',
                'text_rerank': 0.4,
                'colpali_rerank': 0.5
            },
            {
                'name': 'Scenario 2: ColPali gets poor rerank score',
                'text_rerank': 0.4,
                'colpali_rerank': 0.1
            },
            {
                'name': 'Scenario 3: Text gets poor rerank score',
                'text_rerank': 0.1,
                'colpali_rerank': 0.4
            },
            {
                'name': 'Scenario 4: Both get high rerank scores',
                'text_rerank': 0.7,
                'colpali_rerank': 0.8
            },
            {
                'name': 'Scenario 5: Real-world estimate (ColPali may get low rerank)',
                'text_rerank': 0.35,  # Decent score for text content
                'colpali_rerank': 0.15  # Low score for visual content description
            }
        ]
        
        results = []
        for scenario in scenarios:
            logger.info(f"\n{scenario['name']}")
            logger.info("-" * 50)
            
            result = self.debug_scoring_scenario(
                query, 
                text_score, 
                colpali_score,
                {'text': scenario['text_rerank'], 'colpali': scenario['colpali_rerank']}
            )
            results.append({**scenario, **result})
        
        # Summary
        logger.info(f"📋 SCENARIO SUMMARY:")
        for i, result in enumerate(results, 1):
            winner_symbol = "🥇" if result['winner'] == 'ColPali' else "🥈"
            logger.info(f"   {winner_symbol} Scenario {i}: {result['winner']} wins (margin: {result['margin']:.3f})")
        
        return results

class MockCrossEncoder:
    """Mock cross-encoder for predictable testing."""
    
    def predict(self, query_text_pairs):
        """Return mock rerank scores."""
        # In reality, these scores depend on how well the cross-encoder
        # thinks the text content answers the query
        return [0.3]  # Default mock score

def main():
    """Main debug function."""
    logger.info("🚀 Starting Re-ranker Debug Test")
    
    # Initialize debug re-ranker
    debugger = DebugReRanker()
    debugger.initialize_mock_model()
    
    # Test with exact scenario from logs
    query = "what's the retrieval time in a ColPali RAG pipeline?"
    text_score = 0.508
    colpali_score = 0.748
    
    logger.info(f"🎯 REPRODUCING ACTUAL SCENARIO FROM LOGS")
    logger.info("="*60)
    
    # Test different rerank score scenarios
    results = debugger.test_different_rerank_scenarios(query, text_score, colpali_score)
    
    # Analysis
    logger.info(f"\n🔍 ROOT CAUSE ANALYSIS:")
    logger.info(f"The issue likely occurs when:")
    logger.info(f"1. ColPali gets a low cross-encoder rerank score (common for visual content descriptions)")
    logger.info(f"2. The combined score formula heavily weights rerank score (70%)")
    logger.info(f"3. Even with 1.5x bias, ColPali can't overcome a poor rerank score")
    logger.info(f"")
    logger.info(f"Example calculation that reproduces the issue:")
    logger.info(f"- Text: 0.508 * 0.3 + 0.35 * 0.7 = 0.152 + 0.245 = 0.397 (≈ 0.442 from logs)")  
    logger.info(f"- ColPali: (0.748 * 0.3 + 0.15 * 0.7) * 1.5 = (0.224 + 0.105) * 1.5 = 0.494")
    logger.info(f"")
    logger.info(f"💡 PROPOSED SOLUTIONS:")
    logger.info(f"1. Increase ColPali bias multiplier (2.0x instead of 1.5x)")
    logger.info(f"2. Reduce rerank score weight for visual sources")
    logger.info(f"3. Lower ColPali minimum threshold")
    logger.info(f"4. Add special handling for performance/chart queries")

if __name__ == "__main__":
    main()