"""
Re-ranker Integration Strategies for Agentic RAG

This module implements three different strategies for integrating the BGE Cross-Encoder
re-ranker into the agentic reasoning process for A/B testing and performance comparison.
"""

import sys
import os
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Import base agentic orchestrator
from agentic_orchestrator import (
    AgenticOrchestrator, AgentStep, SourceType, AgentAction, AgentResponse
)

class RerankerStrategy(Enum):
    """Available re-ranker integration strategies"""
    PURE_AGENTIC = "pure_agentic"           # No re-ranker (current baseline)
    RERANKER_ENHANCED = "reranker_enhanced" # Re-ranker for source result evaluation
    HYBRID_MODE = "hybrid_mode"             # Re-ranker for final synthesis evaluation

@dataclass
class StrategyPerformanceMetrics:
    """Performance metrics for strategy comparison"""
    response_quality_score: float = 0.0
    confidence_accuracy: float = 0.0
    execution_time: float = 0.0
    total_tokens: int = 0
    reranker_tokens: int = 0
    sources_used_count: int = 0
    reasoning_steps: int = 0
    final_confidence: float = 0.0

class StrategyTestOrchestrator(AgenticOrchestrator):
    """
    Enhanced agentic orchestrator with configurable re-ranker integration strategies
    for A/B testing and performance comparison.
    """
    
    def __init__(self, strategy: RerankerStrategy = RerankerStrategy.PURE_AGENTIC, **kwargs):
        """
        Initialize with specific re-ranker integration strategy.
        
        Args:
            strategy: Re-ranker integration strategy to use
            **kwargs: Arguments passed to base AgenticOrchestrator
        """
        super().__init__(**kwargs)
        self.strategy = strategy
        self.reranker_token_count = 0
        
        self.logger.info(f"Strategy Test Orchestrator initialized with strategy: {strategy.value}")
    
    def query_with_strategy(self, user_query: str, context: Optional[Dict] = None) -> AgentResponse:
        """
        Query with specified re-ranker integration strategy.
        
        Args:
            user_query: User's question/request
            context: Optional context from previous interactions
            
        Returns:
            AgentResponse with strategy-specific processing
        """
        self.reranker_token_count = 0  # Reset token counting
        
        if self.strategy == RerankerStrategy.PURE_AGENTIC:
            return self._pure_agentic_query(user_query, context)
        elif self.strategy == RerankerStrategy.RERANKER_ENHANCED:
            return self._reranker_enhanced_query(user_query, context)
        elif self.strategy == RerankerStrategy.HYBRID_MODE:
            return self._hybrid_mode_query(user_query, context)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _pure_agentic_query(self, user_query: str, context: Optional[Dict] = None) -> AgentResponse:
        """
        Strategy 1: Pure agentic reasoning without re-ranker integration.
        This is the current baseline implementation.
        """
        self.logger.info("STRATEGY 1: Pure Agentic (No Re-ranker)")
        return super().query(user_query, context)
    
    def _reranker_enhanced_query(self, user_query: str, context: Optional[Dict] = None) -> AgentResponse:
        """
        Strategy 2: Re-ranker enhanced - use BGE re-ranker to evaluate individual source results.
        """
        self.logger.info("STRATEGY 2: Re-ranker Enhanced (Source Result Evaluation)")
        
        start_time = time.time()
        reasoning_chain = []
        
        # Step 1: Initial THINK - same as baseline
        think_action = self._think_step(user_query, context)
        reasoning_chain.append(think_action)
        
        current_knowledge = {}
        step_count = 1
        
        while step_count < self.max_steps:
            # Step 2: RETRIEVE with re-ranker evaluation
            retrieve_action = self._reranker_enhanced_retrieve_step(
                user_query, think_action.reasoning, current_knowledge
            )
            reasoning_chain.append(retrieve_action)
            
            if retrieve_action.result:
                current_knowledge[retrieve_action.source.value] = retrieve_action.result
            
            # Step 3: RETHINK with enhanced confidence from re-ranker
            rethink_action = self._enhanced_rethink_step(
                user_query, current_knowledge, reasoning_chain
            )
            reasoning_chain.append(rethink_action)
            
            # Enhanced stopping criteria using re-ranker confidence
            if rethink_action.confidence >= self.confidence_threshold:
                self.logger.info(f"Enhanced confidence threshold reached: {rethink_action.confidence}")
                break
                
            if "CONTINUE" not in rethink_action.reasoning.upper():
                break
                
            step_count += 1
        
        # Step 4: GENERATE - standard synthesis
        generate_action = self._generate_step(user_query, current_knowledge, reasoning_chain)
        reasoning_chain.append(generate_action)
        
        return self._build_response(user_query, reasoning_chain, start_time)
    
    def _hybrid_mode_query(self, user_query: str, context: Optional[Dict] = None) -> AgentResponse:
        """
        Strategy 3: Hybrid mode - use re-ranker for final synthesis evaluation and confidence boosting.
        """
        self.logger.info("STRATEGY 3: Hybrid Mode (Final Synthesis Evaluation)")
        
        start_time = time.time()
        reasoning_chain = []
        
        # Steps 1-3: Standard agentic reasoning
        think_action = self._think_step(user_query, context)
        reasoning_chain.append(think_action)
        
        current_knowledge = {}
        step_count = 1
        
        while step_count < self.max_steps:
            retrieve_action = self._retrieve_step(
                user_query, think_action.reasoning, current_knowledge
            )
            reasoning_chain.append(retrieve_action)
            
            if retrieve_action.result:
                current_knowledge[retrieve_action.source.value] = retrieve_action.result
            
            rethink_action = self._rethink_step(
                user_query, current_knowledge, reasoning_chain
            )
            reasoning_chain.append(rethink_action)
            
            if rethink_action.confidence >= self.confidence_threshold:
                break
                
            if "CONTINUE" not in rethink_action.reasoning.upper():
                break
                
            step_count += 1
        
        # Step 4: GENERATE with hybrid re-ranker evaluation
        generate_action = self._hybrid_generate_step(user_query, current_knowledge, reasoning_chain)
        reasoning_chain.append(generate_action)
        
        return self._build_response(user_query, reasoning_chain, start_time)
    
    def _reranker_enhanced_retrieve_step(self, query: str, plan: str, current_knowledge: Dict) -> AgentAction:
        """
        Enhanced retrieve step that uses re-ranker to evaluate source result quality.
        """
        source = self._select_source(plan, current_knowledge)
        self.logger.info(f"RETRIEVE (Enhanced): Querying {source.value} with re-ranker evaluation")
        
        try:
            # Get source result using parent class method
            raw_result = self._query_single_source(source, query)
            
            # Re-ranker evaluation for confidence scoring
            if self.reranker and raw_result and 'answer' in raw_result:
                try:
                    # Score individual result quality
                    ranking_score = self._score_single_result(query, raw_result['answer'])
                    confidence = min(ranking_score, 0.95)  # Cap for multi-step reasoning
                    self.reranker_token_count += 5  # Estimate for single result scoring
                    
                    reasoning = f"Retrieved from {source.value} with re-ranker confidence {confidence:.2f}"
                    
                except Exception as e:
                    self.logger.warning(f"Re-ranker evaluation failed: {e}")
                    confidence = 0.5
                    reasoning = f"Retrieved from {source.value} (re-ranker evaluation failed)"
            else:
                confidence = 0.5
                reasoning = f"Retrieved from {source.value} (no re-ranker evaluation)"
            
            return AgentAction(
                step=AgentStep.RETRIEVE,
                source=source,
                query=query,
                reasoning=reasoning,
                result=raw_result,
                confidence=confidence,
                timestamp=time.time()
            )
            
        except Exception as e:
            return AgentAction(
                step=AgentStep.RETRIEVE,
                source=source,
                query=query,
                reasoning=f"Failed to retrieve from {source.value}: {str(e)}",
                result=None,
                confidence=0.1,
                timestamp=time.time()
            )
    
    def _enhanced_rethink_step(self, query: str, knowledge: Dict, reasoning_chain: List) -> AgentAction:
        """
        Enhanced rethink step that uses re-ranker confidence scores for better decision making.
        """
        # Get confidence scores from re-ranker enhanced retrievals
        retrieval_confidences = [
            action.confidence for action in reasoning_chain 
            if action.step == AgentStep.RETRIEVE and action.confidence > 0
        ]
        
        if retrieval_confidences:
            avg_confidence = sum(retrieval_confidences) / len(retrieval_confidences)
            max_confidence = max(retrieval_confidences)
        else:
            avg_confidence = 0.3
            max_confidence = 0.3
        
        # Enhanced decision logic using re-ranker insights
        successful_results = len([k for k in knowledge.values() if k])
        available_sources = self._count_available_sources()
        
        if max_confidence >= 0.8:
            reasoning = f"High-quality result found (confidence: {max_confidence:.2f}). SUFFICIENT"
            confidence = max_confidence
        elif successful_results >= available_sources:
            reasoning = f"All available sources queried (avg confidence: {avg_confidence:.2f}). SUFFICIENT"
            confidence = min(avg_confidence + 0.1, 0.85)
        elif successful_results >= 2 and avg_confidence >= 0.6:
            reasoning = f"Multiple good results obtained (avg confidence: {avg_confidence:.2f}). SUFFICIENT"
            confidence = avg_confidence
        else:
            reasoning = f"Need more information (current confidence: {avg_confidence:.2f}). CONTINUE"
            confidence = avg_confidence
        
        return AgentAction(
            step=AgentStep.RETHINK,
            query=query,
            reasoning=reasoning,
            confidence=confidence,
            timestamp=time.time()
        )
    
    def _hybrid_generate_step(self, query: str, knowledge: Dict, reasoning_chain: List) -> AgentAction:
        """
        Hybrid generate step that uses re-ranker to evaluate final synthesis vs individual sources.
        """
        try:
            # Generate synthesis using parent class method
            synthesis_prompt = self._build_synthesis_prompt(query, knowledge)
            synthesized_answer = self._generate_response(synthesis_prompt)
            
            # Re-ranker evaluation for final answer confidence
            if self.reranker and knowledge:
                try:
                    # Prepare candidates: synthesis + individual source responses
                    candidates = []
                    
                    # Add synthesis as candidate
                    candidates.append({
                        "source": "synthesis",
                        "answer": synthesized_answer,
                        "confidence": 0.7  # Base synthesis confidence
                    })
                    
                    # Add individual source responses
                    for source_name, result in knowledge.items():
                        if result and 'answer' in result:
                            candidates.append({
                                "source": source_name,
                                "answer": result['answer'],
                                "confidence": 0.6  # Base individual source confidence
                            })
                    
                    if len(candidates) > 1:
                        # Use re-ranker to evaluate all candidates
                        ranking_result = self._rank_candidates(query, candidates)
                        self.reranker_token_count += 10  # Estimate for candidate ranking
                        
                        if ranking_result and ranking_result.get('success'):
                            selected = ranking_result['selected_source']
                            base_confidence = ranking_result.get('confidence', 0.6)
                            
                            # Boost confidence if synthesis was selected
                            if selected.get('source') == 'synthesis':
                                final_confidence = min(base_confidence * 1.15, 0.95)
                                reasoning = f"Synthesis selected as best answer (confidence boosted to {final_confidence:.2f})"
                            else:
                                final_confidence = base_confidence * 0.9
                                reasoning = f"Individual source '{selected.get('source')}' outperformed synthesis (confidence: {final_confidence:.2f})"
                                # Could optionally use individual source answer here
                        else:
                            final_confidence = 0.6
                            reasoning = "Re-ranker evaluation failed, using synthesis with default confidence"
                    else:
                        final_confidence = 0.6
                        reasoning = "Only synthesis available, no comparison possible"
                        
                except Exception as e:
                    self.logger.warning(f"Hybrid re-ranker evaluation failed: {e}")
                    final_confidence = 0.6
                    reasoning = f"Re-ranker evaluation error: {str(e)[:100]}"
            else:
                final_confidence = 0.6
                reasoning = "No re-ranker available for hybrid evaluation"
            
            return AgentAction(
                step=AgentStep.GENERATE,
                query=query,
                reasoning=reasoning,
                result=synthesized_answer,
                confidence=final_confidence,
                timestamp=time.time()
            )
            
        except Exception as e:
            return AgentAction(
                step=AgentStep.GENERATE,
                query=query,
                reasoning=f"Hybrid synthesis generation failed: {str(e)}",
                result="I apologize, but I encountered an error generating the response.",
                confidence=0.1,
                timestamp=time.time()
            )
    
    def _score_single_result(self, query: str, answer: str) -> float:
        """
        Score a single result using the re-ranker.
        """
        if not self.reranker:
            return 0.5
        
        try:
            # Use re-ranker to score query-answer pair
            score = self.reranker.score_single_pair(query, answer)
            return max(0.1, min(0.95, score))  # Clamp between 0.1 and 0.95
        except Exception as e:
            self.logger.warning(f"Single result scoring failed: {e}")
            return 0.5
    
    def _rank_candidates(self, query: str, candidates: List[Dict]) -> Optional[Dict]:
        """
        Rank multiple candidates using the re-ranker.
        """
        if not self.reranker or len(candidates) < 2:
            return None
        
        try:
            # Convert candidates to format expected by re-ranker
            candidate_list = [
                {"source": c["source"], "answer": c["answer"]}
                for c in candidates
            ]
            
            # Use re-ranker to rank all candidates
            ranking_result = self.reranker.rank_all_sources(query, candidate_list)
            return ranking_result
            
        except Exception as e:
            self.logger.warning(f"Candidate ranking failed: {e}")
            return None
    
    def _build_response(self, query: str, reasoning_chain: List, start_time: float) -> AgentResponse:
        """
        Build agent response with strategy-specific metrics.
        """
        execution_time = time.time() - start_time
        sources_used = [action.source for action in reasoning_chain 
                       if action.source and action.result]
        
        # Get final answer from last generate step
        generate_actions = [a for a in reasoning_chain if a.step == AgentStep.GENERATE]
        final_answer = generate_actions[-1].result if generate_actions else "No response generated"
        final_confidence = generate_actions[-1].confidence if generate_actions else 0.1
        
        response = AgentResponse(
            final_answer=final_answer,
            reasoning_chain=reasoning_chain,
            total_steps=len(reasoning_chain),
            execution_time=execution_time,
            sources_used=sources_used,
            confidence_score=final_confidence
        )
        
        # Add strategy-specific metadata
        response.strategy = self.strategy.value
        response.reranker_tokens = self.reranker_token_count
        
        self.logger.info(f"Strategy {self.strategy.value} completed: {len(reasoning_chain)} steps, "
                        f"{execution_time:.2f}s, {self.reranker_token_count} re-ranker tokens")
        
        return response

class StrategyComparator:
    """
    Utility class for running A/B tests comparing different re-ranker integration strategies.
    """
    
    def __init__(self, rag_system=None, colpali_retriever=None, 
                 salesforce_connector=None, reranker=None):
        """
        Initialize strategy comparator with RAG components.
        """
        self.components = {
            'rag_system': rag_system,
            'colpali_retriever': colpali_retriever,
            'salesforce_connector': salesforce_connector,
            'reranker': reranker
        }
        
        self.logger = logging.getLogger(__name__)
    
    def compare_strategies(self, test_query: str, 
                          strategies: List[RerankerStrategy] = None) -> Dict[str, Any]:
        """
        Compare multiple strategies on a single test query.
        
        Args:
            test_query: Query to test all strategies on
            strategies: List of strategies to test (default: all three)
            
        Returns:
            Dictionary with results for each strategy
        """
        if strategies is None:
            strategies = list(RerankerStrategy)
        
        results = {}
        
        for strategy in strategies:
            self.logger.info(f"Testing strategy: {strategy.value}")
            
            try:
                # Create orchestrator with specific strategy
                orchestrator = StrategyTestOrchestrator(
                    strategy=strategy,
                    **self.components,
                    max_steps=10,
                    confidence_threshold=0.7
                )
                
                # Run query with strategy
                start_time = time.time()
                response = orchestrator.query_with_strategy(test_query)
                total_time = time.time() - start_time
                
                # Collect metrics
                results[strategy.value] = {
                    'response': response,
                    'metrics': StrategyPerformanceMetrics(
                        execution_time=total_time,
                        total_tokens=getattr(response, 'reranker_tokens', 0) + 100,  # Estimate other tokens
                        reranker_tokens=getattr(response, 'reranker_tokens', 0),
                        sources_used_count=len(response.sources_used),
                        reasoning_steps=response.total_steps,
                        final_confidence=response.confidence_score
                    ),
                    'success': True
                }
                
            except Exception as e:
                self.logger.error(f"Strategy {strategy.value} failed: {e}")
                results[strategy.value] = {
                    'response': None,
                    'metrics': None,
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def run_comprehensive_test(self, test_scenarios: List[Dict]) -> Dict[str, Any]:
        """
        Run comprehensive A/B test across multiple scenarios.
        
        Args:
            test_scenarios: List of test scenarios with query and metadata
            
        Returns:
            Comprehensive test results with aggregated metrics
        """
        all_results = {}
        strategy_totals = {strategy.value: [] for strategy in RerankerStrategy}
        
        for i, scenario in enumerate(test_scenarios, 1):
            self.logger.info(f"Running test scenario {i}/{len(test_scenarios)}: {scenario.get('name', 'Unnamed')}")
            
            scenario_results = self.compare_strategies(scenario['query'])
            all_results[scenario.get('name', f'Scenario_{i}')] = scenario_results
            
            # Aggregate results for each strategy
            for strategy_name, result in scenario_results.items():
                if result['success']:
                    strategy_totals[strategy_name].append(result['metrics'])
        
        # Calculate aggregate statistics
        aggregate_stats = {}
        for strategy_name, metrics_list in strategy_totals.items():
            if metrics_list:
                aggregate_stats[strategy_name] = {
                    'avg_execution_time': sum(m.execution_time for m in metrics_list) / len(metrics_list),
                    'avg_reasoning_steps': sum(m.reasoning_steps for m in metrics_list) / len(metrics_list),
                    'avg_final_confidence': sum(m.final_confidence for m in metrics_list) / len(metrics_list),
                    'avg_reranker_tokens': sum(m.reranker_tokens for m in metrics_list) / len(metrics_list),
                    'success_rate': len(metrics_list) / len(test_scenarios),
                    'total_tests': len(metrics_list)
                }
        
        return {
            'individual_results': all_results,
            'aggregate_statistics': aggregate_stats,
            'test_summary': {
                'total_scenarios': len(test_scenarios),
                'strategies_tested': len(RerankerStrategy),
                'timestamp': time.time()
            }
        }

# Usage example and testing utilities
if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Simple Technical Query",
            "query": "What is a transformer architecture in machine learning?",
            "category": "technical"
        },
        {
            "name": "Attention Mechanism Query", 
            "query": "What is attention mechanism in transformers?",
            "category": "technical"
        }
    ]
    
    print("Re-ranker Integration Strategies module loaded successfully!")
    print(f"Available strategies: {[s.value for s in RerankerStrategy]}")
    print(f"Test scenarios loaded: {len(test_scenarios)}")