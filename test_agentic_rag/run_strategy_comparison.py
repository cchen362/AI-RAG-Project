"""
Strategy Comparison Test Runner

Comprehensive testing script for comparing re-ranker integration strategies
in the agentic RAG system. Runs A/B tests and generates performance analysis.
"""

import sys
import os
import json
import time
import logging
from typing import Dict, List, Any
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import testing components
from test_harness import TestHarness
from reranker_integration_strategies import (
    StrategyComparator, RerankerStrategy, StrategyPerformanceMetrics
)

# Import production components
from src.rag_system import RAGSystem
from src.colpali_retriever import ColPaliRetriever
from src.salesforce_connector import SalesforceConnector
from src.cross_encoder_reranker import CrossEncoderReRanker

class StrategyComparisonRunner:
    """
    Main test runner for comprehensive strategy comparison and analysis.
    """
    
    def __init__(self, output_dir: str = "strategy_comparison_results"):
        """
        Initialize the strategy comparison runner.
        
        Args:
            output_dir: Directory to save results and analysis
        """
        self.output_dir = output_dir
        self.setup_logging()
        self.create_output_directory()
        
        # Test scenarios for comprehensive evaluation
        self.test_scenarios = [
            {
                "name": "Simple Technical Query",
                "query": "What is a transformer architecture in machine learning?",
                "category": "technical",
                "description": "Basic technical query testing fundamental capabilities",
                "expected_sources": ["text_rag"],
                "complexity": "low"
            },
            {
                "name": "Attention Mechanism Deep Dive",
                "query": "What is attention mechanism in transformers and how does it work mathematically?",
                "category": "technical", 
                "description": "Complex technical query requiring detailed mathematical understanding",
                "expected_sources": ["text_rag"],
                "complexity": "high"
            },
            {
                "name": "Multi-hop Complex Query",
                "query": "How do transformers use attention mechanisms for language modeling and what are the computational advantages over RNNs?",
                "category": "complex",
                "description": "Multi-step reasoning requiring cross-source synthesis",
                "expected_sources": ["text_rag", "salesforce"],
                "complexity": "high"
            },
            {
                "name": "Business Context Query",
                "query": "What are the latest trends in artificial intelligence for business applications?",
                "category": "business",
                "description": "Business-focused query testing Salesforce integration",
                "expected_sources": ["salesforce", "text_rag"],
                "complexity": "medium"
            },
            {
                "name": "Comparative Analysis Query",
                "query": "Compare transformer models with traditional RNN architectures in terms of performance and efficiency",
                "category": "comparative",
                "description": "Comparative analysis requiring synthesis of multiple concepts",
                "expected_sources": ["text_rag"],
                "complexity": "high"
            }
        ]
        
        self.logger.info(f"Strategy Comparison Runner initialized with {len(self.test_scenarios)} test scenarios")
    
    def setup_logging(self):
        """Setup detailed logging for the test run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"strategy_comparison_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Strategy comparison logging initialized")
    
    def create_output_directory(self):
        """Create output directory for results."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.logger.info(f"Created output directory: {self.output_dir}")
    
    def initialize_components(self) -> Dict[str, Any]:
        """
        Initialize all RAG components for testing.
        
        Returns:
            Dictionary of initialized components
        """
        self.logger.info("Initializing RAG components for strategy testing...")
        
        components = {}
        
        # Initialize Text RAG system
        try:
            text_config = {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'embedding_model': 'openai',
                'generation_model': 'gpt-3.5-turbo',
                'max_retrieved_chunks': 5,
                'temperature': 0.1
            }
            components['rag_system'] = RAGSystem(text_config)
            
            # Process documents
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'documents')
            if os.path.exists(data_dir):
                doc_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                           if f.endswith(('.txt', '.pdf', '.docx'))]
                if doc_files:
                    result = components['rag_system'].add_documents(doc_files)
                    self.logger.info(f"Processed {result.get('documents_processed', 0)} documents for Text RAG")
            
            self.logger.info("✅ Text RAG system initialized")
        except Exception as e:
            self.logger.error(f"❌ Text RAG initialization failed: {e}")
            components['rag_system'] = None
        
        # Initialize ColPali (skip for faster testing)
        self.logger.info("⏸️ Skipping ColPali initialization for faster testing")
        components['colpali_retriever'] = None
        
        # Initialize Salesforce connector
        try:
            components['salesforce_connector'] = SalesforceConnector()
            if components['salesforce_connector'].test_connection():
                self.logger.info("✅ Salesforce connector initialized")
            else:
                self.logger.warning("⚠️ Salesforce connection test failed")
                components['salesforce_connector'] = None
        except Exception as e:
            self.logger.warning(f"⚠️ Salesforce initialization failed: {e}")
            components['salesforce_connector'] = None
        
        # Initialize Re-ranker (CRITICAL for this test)
        try:
            components['reranker'] = CrossEncoderReRanker(
                model_name='BAAI/bge-reranker-base',
                relevance_threshold=0.3
            )
            if components['reranker'].initialize():
                self.logger.info("✅ BGE Re-ranker initialized - READY FOR STRATEGY TESTING")
            else:
                self.logger.error("❌ Re-ranker initialization failed - CANNOT RUN STRATEGY TESTS")
                components['reranker'] = None
        except Exception as e:
            self.logger.error(f"❌ Re-ranker initialization failed: {e}")
            components['reranker'] = None
        
        # Verify critical components
        if not components['rag_system']:
            raise RuntimeError("Text RAG system is required for testing")
        
        if not components['reranker']:
            self.logger.warning("⚠️ Re-ranker not available - some strategies will not work properly")
        
        return components
    
    def run_single_scenario_test(self, scenario: Dict, components: Dict) -> Dict[str, Any]:
        """
        Run all three strategies on a single test scenario.
        
        Args:
            scenario: Test scenario dictionary
            components: Initialized RAG components
            
        Returns:
            Results for all strategies on this scenario
        """
        self.logger.info(f"🧪 Testing scenario: {scenario['name']}")
        self.logger.info(f"   Query: {scenario['query']}")
        self.logger.info(f"   Category: {scenario['category']} | Complexity: {scenario['complexity']}")
        
        # Create strategy comparator
        comparator = StrategyComparator(**components)
        
        # Run comparison
        start_time = time.time()
        results = comparator.compare_strategies(scenario['query'])
        total_time = time.time() - start_time
        
        # Add scenario metadata to results
        for strategy_name in results:
            if results[strategy_name]['success']:
                results[strategy_name]['scenario_info'] = {
                    'name': scenario['name'],
                    'category': scenario['category'],
                    'complexity': scenario['complexity'],
                    'query_length': len(scenario['query']),
                    'total_test_time': total_time
                }
        
        self.logger.info(f"✅ Scenario completed in {total_time:.2f}s")
        return results
    
    def run_comprehensive_comparison(self) -> Dict[str, Any]:
        """
        Run comprehensive comparison across all test scenarios and strategies.
        
        Returns:
            Complete test results with analysis
        """
        self.logger.info("🚀 Starting comprehensive strategy comparison...")
        self.logger.info(f"Testing {len(RerankerStrategy)} strategies on {len(self.test_scenarios)} scenarios")
        
        # Initialize components
        components = self.initialize_components()
        
        # Track results
        all_scenario_results = {}
        strategy_aggregates = {strategy.value: [] for strategy in RerankerStrategy}
        
        # Run tests for each scenario
        for i, scenario in enumerate(self.test_scenarios, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"SCENARIO {i}/{len(self.test_scenarios)}: {scenario['name']}")
            self.logger.info(f"{'='*60}")
            
            try:
                scenario_results = self.run_single_scenario_test(scenario, components)
                all_scenario_results[scenario['name']] = scenario_results
                
                # Aggregate successful results
                for strategy_name, result in scenario_results.items():
                    if result['success'] and result['metrics']:
                        strategy_aggregates[strategy_name].append({
                            'scenario': scenario['name'],
                            'metrics': result['metrics'],
                            'response_quality': self.assess_response_quality(
                                result['response'], scenario
                            )
                        })
                
            except Exception as e:
                self.logger.error(f"❌ Scenario {scenario['name']} failed: {e}")
                all_scenario_results[scenario['name']] = {
                    'error': str(e),
                    'scenario_failed': True
                }
        
        # Generate comprehensive analysis
        analysis = self.generate_comprehensive_analysis(strategy_aggregates, all_scenario_results)
        
        # Save results
        self.save_results(all_scenario_results, analysis)
        
        self.logger.info("\n🎉 Comprehensive strategy comparison completed!")
        return {
            'scenario_results': all_scenario_results,
            'analysis': analysis,
            'summary': analysis['executive_summary']
        }
    
    def assess_response_quality(self, response, scenario: Dict) -> Dict[str, float]:
        """
        Assess response quality using multiple metrics.
        
        Args:
            response: Agent response object
            scenario: Test scenario information
            
        Returns:
            Dictionary of quality metrics
        """
        if not response or not hasattr(response, 'final_answer'):
            return {
                'relevance_score': 0.0,
                'completeness_score': 0.0,
                'accuracy_score': 0.0,
                'overall_quality': 0.0
            }
        
        answer = response.final_answer
        query = scenario['query']
        
        # Simple heuristic-based quality assessment
        relevance_score = self.calculate_relevance_score(answer, query)
        completeness_score = self.calculate_completeness_score(answer, scenario)
        accuracy_score = self.calculate_accuracy_score(answer, scenario)
        
        overall_quality = (relevance_score + completeness_score + accuracy_score) / 3.0
        
        return {
            'relevance_score': relevance_score,
            'completeness_score': completeness_score, 
            'accuracy_score': accuracy_score,
            'overall_quality': overall_quality
        }
    
    def calculate_relevance_score(self, answer: str, query: str) -> float:
        """Calculate relevance score based on keyword overlap and length."""
        if not answer or len(answer) < 10:
            return 0.0
        
        # Extract key terms from query
        query_terms = set(query.lower().split())
        answer_terms = set(answer.lower().split())
        
        # Calculate overlap
        overlap = len(query_terms.intersection(answer_terms))
        relevance = min(overlap / len(query_terms), 1.0)
        
        # Bonus for adequate length
        length_bonus = min(len(answer) / 200, 1.0) * 0.3
        
        return min(relevance + length_bonus, 1.0)
    
    def calculate_completeness_score(self, answer: str, scenario: Dict) -> float:
        """Calculate completeness score based on expected content."""
        if not answer:
            return 0.0
        
        # Complexity-based expectations
        complexity_expectations = {
            'low': 100,    # Minimum characters expected
            'medium': 200,
            'high': 300
        }
        
        expected_length = complexity_expectations.get(scenario.get('complexity', 'medium'), 200)
        length_score = min(len(answer) / expected_length, 1.0)
        
        # Category-specific keywords
        category_keywords = {
            'technical': ['algorithm', 'model', 'architecture', 'mechanism', 'process'],
            'business': ['business', 'application', 'industry', 'market', 'enterprise'],
            'complex': ['compare', 'analysis', 'advantage', 'relationship', 'impact'],
            'comparative': ['versus', 'compared', 'difference', 'better', 'advantage']
        }
        
        expected_keywords = category_keywords.get(scenario.get('category', ''), [])
        if expected_keywords:
            keyword_matches = sum(1 for kw in expected_keywords if kw in answer.lower())
            keyword_score = min(keyword_matches / len(expected_keywords), 1.0)
        else:
            keyword_score = 0.7  # Default if no category keywords
        
        return (length_score + keyword_score) / 2.0
    
    def calculate_accuracy_score(self, answer: str, scenario: Dict) -> float:
        """Calculate accuracy score based on factual correctness indicators."""
        if not answer:
            return 0.0
        
        # Look for indicators of accurate technical information
        accuracy_indicators = {
            'technical': ['attention', 'transformer', 'neural', 'layer', 'matrix'],
            'business': ['trend', 'application', 'solution', 'technology', 'innovation'],
            'mathematical': ['formula', 'equation', 'calculation', 'algorithm', 'function']
        }
        
        # Count accuracy indicators present
        total_indicators = 0
        found_indicators = 0
        
        for category, indicators in accuracy_indicators.items():
            if category in scenario.get('description', '').lower():
                total_indicators += len(indicators)
                found_indicators += sum(1 for ind in indicators if ind in answer.lower())
        
        if total_indicators > 0:
            accuracy = found_indicators / total_indicators
        else:
            accuracy = 0.7  # Default accuracy score
        
        # Penalty for obvious errors or "I don't know" responses
        if any(phrase in answer.lower() for phrase in ['i don\'t know', 'cannot answer', 'insufficient information']):
            accuracy *= 0.5
        
        return min(accuracy, 1.0)
    
    def generate_comprehensive_analysis(self, strategy_aggregates: Dict, scenario_results: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive analysis of strategy performance.
        
        Args:
            strategy_aggregates: Aggregated results by strategy
            scenario_results: Individual scenario results
            
        Returns:
            Comprehensive analysis dictionary
        """
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'test_configuration': {
                'total_scenarios': len(self.test_scenarios),
                'strategies_tested': len(RerankerStrategy),
                'scenario_categories': list(set(s['category'] for s in self.test_scenarios))
            }
        }
        
        # Strategy performance comparison
        strategy_performance = {}
        
        for strategy_name, results in strategy_aggregates.items():
            if not results:
                strategy_performance[strategy_name] = {
                    'success_rate': 0.0,
                    'error': 'No successful tests'
                }
                continue
            
            # Calculate averages
            avg_execution_time = sum(r['metrics'].execution_time for r in results) / len(results)
            avg_reasoning_steps = sum(r['metrics'].reasoning_steps for r in results) / len(results)
            avg_final_confidence = sum(r['metrics'].final_confidence for r in results) / len(results)
            avg_reranker_tokens = sum(r['metrics'].reranker_tokens for r in results) / len(results)
            avg_overall_quality = sum(r['response_quality']['overall_quality'] for r in results) / len(results)
            
            strategy_performance[strategy_name] = {
                'success_rate': len(results) / len(self.test_scenarios),
                'avg_execution_time': avg_execution_time,
                'avg_reasoning_steps': avg_reasoning_steps,
                'avg_final_confidence': avg_final_confidence,
                'avg_reranker_tokens': avg_reranker_tokens,
                'avg_overall_quality': avg_overall_quality,
                'total_successful_tests': len(results),
                'performance_score': self.calculate_performance_score({
                    'quality': avg_overall_quality,
                    'confidence': avg_final_confidence,
                    'efficiency': 1.0 / max(avg_execution_time, 1.0),
                    'token_efficiency': 1.0 / max(avg_reranker_tokens + 50, 50)  # Add base tokens
                })
            }
        
        analysis['strategy_performance'] = strategy_performance
        
        # Determine winner
        winner = max(strategy_performance.keys(), 
                    key=lambda k: strategy_performance[k].get('performance_score', 0))
        
        analysis['recommendation'] = {
            'recommended_strategy': winner,
            'rationale': self.generate_recommendation_rationale(winner, strategy_performance),
            'confidence_level': 'High' if strategy_performance[winner]['success_rate'] >= 0.8 else 'Medium'
        }
        
        # Executive summary
        analysis['executive_summary'] = self.generate_executive_summary(strategy_performance, winner)
        
        return analysis
    
    def calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate weighted performance score.
        
        Args:
            metrics: Dictionary of normalized metrics
            
        Returns:
            Weighted performance score
        """
        weights = {
            'quality': 0.4,        # Response quality is most important
            'confidence': 0.3,     # Confidence accuracy is important
            'efficiency': 0.2,     # Execution time efficiency
            'token_efficiency': 0.1 # Token usage efficiency
        }
        
        score = sum(weights[key] * metrics[key] for key in weights if key in metrics)
        return min(score, 1.0)
    
    def generate_recommendation_rationale(self, winner: str, performance: Dict) -> str:
        """Generate rationale for strategy recommendation."""
        winner_stats = performance[winner]
        
        rationale_parts = [
            f"Strategy '{winner}' achieved the highest performance score of {winner_stats['performance_score']:.3f}."
        ]
        
        if winner_stats['avg_overall_quality'] > 0.7:
            rationale_parts.append(f"Excellent response quality ({winner_stats['avg_overall_quality']:.2f}).")
        
        if winner_stats['avg_final_confidence'] > 0.6:
            rationale_parts.append(f"Strong confidence accuracy ({winner_stats['avg_final_confidence']:.2f}).")
        
        if winner_stats['avg_execution_time'] < 12.0:
            rationale_parts.append(f"Efficient execution time ({winner_stats['avg_execution_time']:.1f}s average).")
        
        return " ".join(rationale_parts)
    
    def generate_executive_summary(self, performance: Dict, winner: str) -> Dict[str, Any]:
        """Generate executive summary of findings."""
        return {
            'key_findings': [
                f"Tested {len(RerankerStrategy)} re-ranker integration strategies across {len(self.test_scenarios)} scenarios",
                f"'{winner}' strategy showed best overall performance",
                f"Average execution time across strategies: {sum(p.get('avg_execution_time', 0) for p in performance.values()) / len(performance):.1f}s",
                f"Re-ranker integration impact: {sum(p.get('avg_reranker_tokens', 0) for p in performance.values()) / len(performance):.1f} tokens average"
            ],
            'recommendation': f"Implement '{winner}' strategy for production agentic RAG system",
            'next_steps': [
                "Integrate winning strategy into main agentic orchestrator",
                "Conduct user acceptance testing with new strategy",
                "Monitor performance metrics in production environment",
                "Consider fine-tuning re-ranker confidence thresholds"
            ]
        }
    
    def save_results(self, scenario_results: Dict, analysis: Dict):
        """Save test results and analysis to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        results_file = os.path.join(self.output_dir, f"strategy_comparison_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(scenario_results, f, indent=2, default=str)
        
        # Save analysis
        analysis_file = os.path.join(self.output_dir, f"strategy_analysis_{timestamp}.json")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save summary report
        report_file = os.path.join(self.output_dir, f"strategy_comparison_report_{timestamp}.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(self.generate_markdown_report(analysis))
        
        self.logger.info(f"Results saved:")
        self.logger.info(f"  - Raw results: {results_file}")
        self.logger.info(f"  - Analysis: {analysis_file}")
        self.logger.info(f"  - Report: {report_file}")
    
    def generate_markdown_report(self, analysis: Dict) -> str:
        """Generate markdown report from analysis."""
        report = f"""# Re-ranker Integration Strategy Comparison Report

**Generated:** {analysis['timestamp']}

## Executive Summary

### Key Findings
"""
        for finding in analysis['executive_summary']['key_findings']:
            report += f"- {finding}\n"
        
        report += f"""
### Recommendation
**{analysis['recommendation']['recommended_strategy']}** - {analysis['recommendation']['rationale']}

**Confidence Level:** {analysis['recommendation']['confidence_level']}

## Strategy Performance Comparison

"""
        
        for strategy, perf in analysis['strategy_performance'].items():
            report += f"""### {strategy.replace('_', ' ').title()}
- **Success Rate:** {perf.get('success_rate', 0):.1%}
- **Average Quality Score:** {perf.get('avg_overall_quality', 0):.2f}
- **Average Confidence:** {perf.get('avg_final_confidence', 0):.2f}
- **Average Execution Time:** {perf.get('avg_execution_time', 0):.1f}s
- **Average Re-ranker Tokens:** {perf.get('avg_reranker_tokens', 0):.0f}
- **Performance Score:** {perf.get('performance_score', 0):.3f}

"""
        
        report += """## Next Steps

"""
        for step in analysis['executive_summary']['next_steps']:
            report += f"1. {step}\n"
        
        return report

def main():
    """Main entry point for strategy comparison testing."""
    print("🧪 Re-ranker Integration Strategy Comparison")
    print("=" * 50)
    
    # Create test runner
    runner = StrategyComparisonRunner()
    
    try:
        # Run comprehensive comparison
        results = runner.run_comprehensive_comparison()
        
        # Print summary
        print("\n📊 COMPARISON COMPLETE!")
        print(f"Recommended Strategy: {results['summary']['recommendation']}")
        print(f"Key Findings: {len(results['summary']['key_findings'])} insights generated")
        
        return results
        
    except Exception as e:
        print(f"❌ Test run failed: {e}")
        raise

if __name__ == "__main__":
    main()