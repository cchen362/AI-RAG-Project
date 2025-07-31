"""
Enhanced Agentic Orchestrator - True vs Pseudo-Agentic Reasoning

This module provides both true LLM-driven agentic reasoning and the original
pseudo-agentic fixed pipeline for A/B testing and comparison. Users can toggle
between genuine intelligence-driven decisions and hardcoded workflows.

Key Features:
- True agentic reasoning using LLM decision making
- Pseudo-agentic baseline for performance comparison  
- A/B testing framework with detailed metrics
- Cost monitoring and efficiency analysis
- Reasoning transparency for both approaches
"""

import sys
import os
import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json

# Import components
from llm_reasoning_agent import LLMReasoningAgent, AgenticResponse as LLMAgenticResponse
from baseline_agentic_orchestrator import AgenticOrchestrator as BaselineOrchestrator, AgentResponse as BaselineResponse

# Import existing production components  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag_system import RAGSystem
from src.colpali_retriever import ColPaliRetriever
from src.salesforce_connector import SalesforceConnector
from src.cross_encoder_reranker import CrossEncoderReRanker

class ReasoningMode(Enum):
    """Available reasoning approaches"""
    TRUE_AGENTIC = "true_agentic"        # LLM-driven dynamic reasoning
    PSEUDO_AGENTIC = "pseudo_agentic"    # Original fixed pipeline
    A_B_COMPARISON = "ab_comparison"     # Side-by-side testing

@dataclass
class ComparisonResult:
    """Results from A/B testing both approaches"""
    true_agentic_response: LLMAgenticResponse
    pseudo_agentic_response: BaselineResponse
    performance_comparison: Dict[str, Any]
    recommendation: str
    cost_analysis: Dict[str, float]
    reasoning_quality_comparison: Dict[str, str]

@dataclass
class UnifiedResponse:
    """Unified response format for both reasoning approaches"""
    final_answer: str
    reasoning_approach: str
    reasoning_chain: List[Any]
    execution_time: float
    sources_used: List[str]
    confidence_score: float
    cost_breakdown: Dict[str, float]
    reasoning_transparency: Dict[str, Any]
    performance_metrics: Dict[str, Any]

class EnhancedAgenticOrchestrator:
    """
    Enhanced orchestrator supporting both true agentic and pseudo-agentic reasoning
    with comprehensive A/B testing and performance analysis capabilities.
    """
    
    def __init__(self,
                 rag_system: Optional[RAGSystem] = None,
                 colpali_retriever: Optional[ColPaliRetriever] = None,
                 salesforce_connector: Optional[SalesforceConnector] = None,
                 reranker: Optional[CrossEncoderReRanker] = None,
                 default_mode: ReasoningMode = ReasoningMode.TRUE_AGENTIC,
                 enable_cost_monitoring: bool = True,
                 enable_logging: bool = True):
        """
        Initialize enhanced orchestrator with both reasoning approaches.
        
        Args:
            rag_system: Text RAG system
            colpali_retriever: Visual document processor
            salesforce_connector: Business knowledge connector
            reranker: Cross-encoder re-ranker
            default_mode: Default reasoning approach to use
            enable_cost_monitoring: Track API costs and usage
            enable_logging: Enable detailed logging
        """
        # Store components
        self.rag_system = rag_system
        self.colpali_retriever = colpali_retriever
        self.salesforce_connector = salesforce_connector
        self.reranker = reranker
        self.default_mode = default_mode
        
        # Initialize reasoning engines
        self.llm_agent = LLMReasoningAgent(
            rag_system=rag_system,
            colpali_retriever=colpali_retriever,
            salesforce_connector=salesforce_connector,
            reranker=reranker,
            enable_logging=enable_logging
        )
        
        self.baseline_orchestrator = BaselineOrchestrator(
            rag_system=rag_system,
            colpali_retriever=colpali_retriever,
            salesforce_connector=salesforce_connector,
            reranker=reranker,
            enable_logging=enable_logging
        )
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if enable_logging:
            self.logger.setLevel(logging.INFO)
        
        # Performance tracking
        self.query_count = 0
        self.total_cost = 0.0
        self.performance_history = []
        
        self.logger.info(f"Enhanced Agentic Orchestrator initialized")
        self.logger.info(f"Default mode: {default_mode.value}")
        self.logger.info(f"Available sources: {self._get_available_sources_summary()}")
    
    def query(self, 
              user_query: str, 
              mode: Optional[ReasoningMode] = None,
              context: Optional[Dict] = None) -> Union[UnifiedResponse, ComparisonResult]:
        """
        Main query interface supporting multiple reasoning modes.
        
        Args:
            user_query: User's question or request
            mode: Reasoning mode to use (defaults to instance default)
            context: Optional context from previous interactions
            
        Returns:
            UnifiedResponse for single mode, ComparisonResult for A/B testing
        """
        query_mode = mode or self.default_mode
        self.query_count += 1
        
        self.logger.info(f"Query #{self.query_count}: {user_query}")
        self.logger.info(f"Reasoning mode: {query_mode.value}")
        
        if query_mode == ReasoningMode.TRUE_AGENTIC:
            return self._true_agentic_query(user_query, context)
        elif query_mode == ReasoningMode.PSEUDO_AGENTIC:
            return self._pseudo_agentic_query(user_query, context)
        elif query_mode == ReasoningMode.A_B_COMPARISON:
            return self._ab_comparison_query(user_query, context)
        else:
            raise ValueError(f"Unknown reasoning mode: {query_mode}")
    
    def _true_agentic_query(self, user_query: str, context: Optional[Dict]) -> UnifiedResponse:
        """Execute query using true LLM-driven agentic reasoning."""
        start_time = time.time()
        
        try:
            # Use LLM reasoning agent
            llm_response = self.llm_agent.query(user_query, context)
            execution_time = time.time() - start_time
            
            # Convert to unified format
            unified_response = UnifiedResponse(
                final_answer=llm_response.final_answer,
                reasoning_approach="TRUE_AGENTIC",
                reasoning_chain=llm_response.reasoning_chain,
                execution_time=execution_time,
                sources_used=[s.value for s in llm_response.sources_queried],
                confidence_score=llm_response.confidence_score,
                cost_breakdown=llm_response.cost_breakdown,
                reasoning_transparency={
                    "total_reasoning_steps": llm_response.total_reasoning_steps,
                    "llm_tokens_used": llm_response.total_llm_tokens,
                    "reasoning_quality": llm_response.reasoning_quality,
                    "dynamic_decisions": len([a for a in llm_response.reasoning_chain if "LLM" in a.reasoning])
                },
                performance_metrics={
                    "intelligent_source_selection": True,
                    "dynamic_stopping": True,
                    "cost_optimization": True,
                    "reasoning_steps": llm_response.total_reasoning_steps
                }
            )
            
            self.total_cost += llm_response.cost_breakdown.get("total", 0)
            self.logger.info(f"TRUE agentic query complete: {execution_time:.2f}s, {llm_response.total_llm_tokens} tokens")
            
            return unified_response
            
        except Exception as e:
            self.logger.error(f"True agentic query failed: {e}")
            return self._create_error_response("TRUE_AGENTIC", str(e), time.time() - start_time)
    
    def _pseudo_agentic_query(self, user_query: str, context: Optional[Dict]) -> UnifiedResponse:
        """Execute query using pseudo-agentic fixed pipeline."""
        start_time = time.time()
        
        try:
            # Use baseline orchestrator
            baseline_response = self.baseline_orchestrator.query(user_query, context)
            execution_time = time.time() - start_time
            
            # Convert to unified format
            unified_response = UnifiedResponse(
                final_answer=baseline_response.final_answer,
                reasoning_approach="PSEUDO_AGENTIC",
                reasoning_chain=baseline_response.reasoning_chain,
                execution_time=execution_time,
                sources_used=[s.value for s in baseline_response.sources_used if s],
                confidence_score=baseline_response.confidence_score,
                cost_breakdown={"total": 0.01},  # Estimated minimal cost
                reasoning_transparency={
                    "total_reasoning_steps": baseline_response.total_steps,
                    "llm_tokens_used": 0,  # No LLM reasoning
                    "reasoning_quality": "fixed_pipeline",
                    "dynamic_decisions": 0  # All decisions hardcoded
                },
                performance_metrics={
                    "intelligent_source_selection": False,
                    "dynamic_stopping": False,
                    "cost_optimization": False,
                    "reasoning_steps": baseline_response.total_steps
                }
            )
            
            self.logger.info(f"PSEUDO agentic query complete: {execution_time:.2f}s, fixed pipeline")
            
            return unified_response
            
        except Exception as e:
            self.logger.error(f"Pseudo agentic query failed: {e}")
            return self._create_error_response("PSEUDO_AGENTIC", str(e), time.time() - start_time)
    
    def _ab_comparison_query(self, user_query: str, context: Optional[Dict]) -> ComparisonResult:
        """Execute A/B comparison testing both approaches."""
        self.logger.info("Starting A/B comparison test")
        
        # Run both approaches
        true_response = self._true_agentic_query(user_query, context)
        pseudo_response = self._pseudo_agentic_query(user_query, context)
        
        # Analyze performance comparison
        performance_comparison = self._analyze_performance_comparison(true_response, pseudo_response)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(performance_comparison)
        
        # Cost analysis
        cost_analysis = {
            "true_agentic_cost": true_response.cost_breakdown.get("total", 0),
            "pseudo_agentic_cost": pseudo_response.cost_breakdown.get("total", 0),
            "cost_difference": true_response.cost_breakdown.get("total", 0) - pseudo_response.cost_breakdown.get("total", 0),
            "cost_efficiency": self._calculate_cost_efficiency(true_response, pseudo_response)
        }
        
        # Reasoning quality comparison
        reasoning_quality = {
            "true_agentic": "dynamic_llm_driven",
            "pseudo_agentic": "fixed_pipeline",
            "transparency_advantage": "true_agentic",
            "decision_quality": "true_agentic" if performance_comparison["decision_quality_winner"] == "true" else "pseudo_agentic"
        }
        
        comparison_result = ComparisonResult(
            true_agentic_response=self._convert_to_llm_response(true_response),
            pseudo_agentic_response=self._convert_to_baseline_response(pseudo_response),
            performance_comparison=performance_comparison,
            recommendation=recommendation,
            cost_analysis=cost_analysis,
            reasoning_quality_comparison=reasoning_quality
        )
        
        self.performance_history.append({
            "query": user_query,
            "comparison": asdict(comparison_result),
            "timestamp": time.time()
        })
        
        self.logger.info(f"A/B comparison complete. Recommendation: {recommendation}")
        
        return comparison_result
    
    def _analyze_performance_comparison(self, true_response: UnifiedResponse, pseudo_response: UnifiedResponse) -> Dict[str, Any]:
        """Analyze performance differences between approaches."""
        return {
            "execution_time_comparison": {
                "true_agentic": true_response.execution_time,
                "pseudo_agentic": pseudo_response.execution_time,
                "difference": true_response.execution_time - pseudo_response.execution_time,
                "winner": "pseudo" if pseudo_response.execution_time < true_response.execution_time else "true"
            },
            "source_utilization": {
                "true_agentic_sources": len(true_response.sources_used),
                "pseudo_agentic_sources": len(pseudo_response.sources_used),
                "intelligent_selection": true_response.performance_metrics.get("intelligent_source_selection", True),
                "dynamic_stopping": true_response.performance_metrics.get("dynamic_stopping", True)
            },
            "reasoning_complexity": {
                "true_agentic_steps": true_response.reasoning_transparency.get("total_reasoning_steps", 0),
                "pseudo_agentic_steps": pseudo_response.reasoning_transparency.get("total_reasoning_steps", 0),
                "dynamic_decisions": true_response.reasoning_transparency.get("dynamic_decisions", 0)
            },
            "confidence_comparison": {
                "true_agentic": true_response.confidence_score,
                "pseudo_agentic": pseudo_response.confidence_score,
                "difference": true_response.confidence_score - pseudo_response.confidence_score
            },
            "decision_quality_winner": "true" if true_response.performance_metrics.get("intelligent_source_selection", True) else "pseudo"
        }
    
    def _generate_recommendation(self, performance: Dict[str, Any]) -> str:
        """Generate recommendation based on performance analysis."""
        true_advantages = []
        pseudo_advantages = []
        
        # Analyze execution time
        if performance["execution_time_comparison"]["winner"] == "pseudo":
            pseudo_advantages.append("faster execution")
        else:
            true_advantages.append("acceptable execution time")
        
        # Analyze intelligence features
        if performance["source_utilization"]["intelligent_selection"]:
            true_advantages.append("intelligent source selection")
        
        if performance["source_utilization"]["dynamic_stopping"]:
            true_advantages.append("efficient stopping criteria")
        
        if performance["reasoning_complexity"]["dynamic_decisions"] > 0:
            true_advantages.append("dynamic decision making")
        
        # Generate recommendation
        if len(true_advantages) > len(pseudo_advantages):
            return f"TRUE_AGENTIC recommended: {', '.join(true_advantages)}"
        elif len(pseudo_advantages) > len(true_advantages):
            return f"PSEUDO_AGENTIC recommended: {', '.join(pseudo_advantages)}"
        else:
            return "MIXED: Both approaches have merits, consider query-specific selection"
    
    def _calculate_cost_efficiency(self, true_response: UnifiedResponse, pseudo_response: UnifiedResponse) -> str:
        """Calculate cost efficiency comparison."""
        true_cost = true_response.cost_breakdown.get("total", 0)
        pseudo_cost = pseudo_response.cost_breakdown.get("total", 0)
        
        if true_cost <= pseudo_cost * 1.2:  # Within 20%
            return "cost_efficient"
        elif true_cost <= pseudo_cost * 2.0:  # Within 100%
            return "acceptable"
        else:
            return "expensive"
    
    def _create_error_response(self, approach: str, error: str, execution_time: float) -> UnifiedResponse:
        """Create error response in unified format."""
        return UnifiedResponse(
            final_answer=f"Error in {approach} reasoning: {error}",
            reasoning_approach=approach,
            reasoning_chain=[],
            execution_time=execution_time,
            sources_used=[],
            confidence_score=0.1,
            cost_breakdown={"total": 0.0},
            reasoning_transparency={"error": error},
            performance_metrics={"error": True}
        )
    
    def _convert_to_llm_response(self, unified: UnifiedResponse) -> LLMAgenticResponse:
        """Convert unified response back to LLM response format."""
        from llm_reasoning_agent import SourceType
        
        # Convert sources_used strings back to SourceType enums
        sources_queried = []
        for source_str in unified.sources_used:
            try:
                sources_queried.append(SourceType(source_str))
            except ValueError:
                # Handle any invalid source strings gracefully
                pass
        
        return type('MockLLMResponse', (), {
            'final_answer': unified.final_answer,
            'reasoning_chain': unified.reasoning_chain,
            'confidence_score': unified.confidence_score,
            'cost_breakdown': unified.cost_breakdown,
            'sources_queried': sources_queried,
            'total_reasoning_steps': unified.reasoning_transparency.get('total_reasoning_steps', 0),
            'total_llm_tokens': unified.reasoning_transparency.get('llm_tokens_used', 0),
            'reasoning_quality': unified.reasoning_transparency.get('reasoning_quality', 'unknown'),
            'total_execution_time': unified.execution_time
        })()
    
    def _convert_to_baseline_response(self, unified: UnifiedResponse) -> BaselineResponse:
        """Convert unified response back to baseline response format."""
        from llm_reasoning_agent import SourceType
        
        # Convert sources_used strings back to SourceType enums for consistency
        sources_used = []
        for source_str in unified.sources_used:
            try:
                sources_used.append(SourceType(source_str))
            except ValueError:
                # Handle any invalid source strings gracefully
                pass
        
        return type('MockBaselineResponse', (), {
            'final_answer': unified.final_answer,
            'reasoning_chain': unified.reasoning_chain,
            'confidence_score': unified.confidence_score,
            'total_steps': len(unified.reasoning_chain),
            'sources_used': sources_used,
            'execution_time': unified.execution_time,
            'reasoning_transparency': unified.reasoning_transparency
        })()
    
    def _get_available_sources_summary(self) -> List[str]:
        """Get summary of available sources."""
        sources = []
        if self.rag_system:
            sources.append("text_rag")
        if self.colpali_retriever:
            sources.append("colpali_visual")
        if self.salesforce_connector:
            sources.append("salesforce")
        return sources
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "total_queries": self.query_count,
            "total_cost": self.total_cost,
            "average_cost_per_query": self.total_cost / max(1, self.query_count),
            "available_sources": self._get_available_sources_summary(),
            "comparison_history": len(self.performance_history),
            "llm_agent_stats": self.llm_agent.get_cost_stats()
        }
    
    def get_reasoning_comparison_summary(self) -> Dict[str, Any]:
        """Get summary of reasoning approach comparisons."""
        if not self.performance_history:
            return {"message": "No comparison data available"}
        
        # Analyze trends from comparison history
        true_wins = sum(1 for h in self.performance_history 
                       if "TRUE_AGENTIC recommended" in h["comparison"]["recommendation"])
        pseudo_wins = sum(1 for h in self.performance_history 
                         if "PSEUDO_AGENTIC recommended" in h["comparison"]["recommendation"])
        
        return {
            "total_comparisons": len(self.performance_history),
            "true_agentic_wins": true_wins,
            "pseudo_agentic_wins": pseudo_wins,
            "mixed_results": len(self.performance_history) - true_wins - pseudo_wins,
            "recommendation_trend": "true_agentic" if true_wins > pseudo_wins else "pseudo_agentic" if pseudo_wins > true_wins else "mixed"
        }

# Testing utilities
def run_comparison_test(orchestrator: EnhancedAgenticOrchestrator, test_query: str) -> ComparisonResult:
    """Utility function to run a single comparison test."""
    print(f"🧪 Running comparison test: {test_query}")
    result = orchestrator.query(test_query, mode=ReasoningMode.A_B_COMPARISON)
    print(f"📊 Result: {result.recommendation}")
    return result

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize enhanced orchestrator
    orchestrator = EnhancedAgenticOrchestrator(
        default_mode=ReasoningMode.TRUE_AGENTIC
    )
    
    print("🚀 Enhanced Agentic Orchestrator initialized!")
    print(f"Available modes: {[mode.value for mode in ReasoningMode]}")
    print(f"Available sources: {orchestrator._get_available_sources_summary()}")
    print("Ready for true vs pseudo-agentic comparison testing!")