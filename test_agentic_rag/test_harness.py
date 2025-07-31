"""
Test Harness - Simple Testing Interface for Agentic RAG

This module provides a simple command-line and programmatic interface for testing
the agentic RAG orchestrator with various scenarios and comparing performance
against the baseline single-turn approach.
"""

import sys
import os
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import asdict

# Add parent directory to path to import existing components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_agentic_rag.agentic_orchestrator import AgenticOrchestrator, AgentResponse
from test_agentic_rag.agent_memory import AgentMemory
from test_agentic_rag.evaluation_metrics import EvaluationMetrics

# Import existing production components
from src.rag_system import RAGSystem
from src.colpali_retriever import ColPaliRetriever
from src.salesforce_connector import SalesforceConnector
from src.cross_encoder_reranker import CrossEncoderReRanker

class TestHarness:
    """
    Simple testing interface for agentic RAG system that allows easy comparison
    between single-turn and multi-turn approaches.
    """
    
    def __init__(self, test_config: Optional[Dict] = None):
        """
        Initialize test harness with optional configuration.
        
        Args:
            test_config: Configuration dictionary for testing parameters
        """
        self.config = test_config or self._get_default_config()
        
        # Initialize components (will be set up lazily)
        self.rag_system = None
        self.colpali_retriever = None  
        self.salesforce_connector = None
        self.reranker = None
        
        # Initialize agentic system
        self.agentic_orchestrator = None
        self.agent_memory = AgentMemory(
            max_conversation_length=self.config.get("max_conversation_length", 10),
            memory_persistence_file=self.config.get("memory_file")
        )
        
        # Initialize evaluation system
        self.evaluator = EvaluationMetrics()
        
        # Test results storage
        self.test_results = []
        self.baseline_results = []
        
        print("Test Harness initialized")
        print(f"Configuration: {json.dumps(self.config, indent=2)}")
    
    def setup_components(self, init_all: bool = False):
        """
        Set up RAG components. Initialize only what's available.
        
        Args:
            init_all: If True, attempt to initialize all components
        """
        print("\nSetting up RAG components...")
        
        # Initialize Text RAG (usually always available)
        try:
            text_config = {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'embedding_model': 'openai',
                'generation_model': 'gpt-3.5-turbo',
                'max_retrieved_chunks': 5,
                'temperature': 0.1
            }
            self.rag_system = RAGSystem(text_config)
            # Process documents from data folder
            import os
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'documents')
            if os.path.exists(data_dir):
                print(f"[INIT] Processing documents from {data_dir}")
                doc_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                           if f.endswith(('.txt', '.pdf', '.docx'))]
                if doc_files:
                    result = self.rag_system.add_documents(doc_files)
                    print(f"[OK] Processed {result.get('documents_processed', 0)} documents")
            print("[OK] Text RAG system initialized")
        except Exception as e:
            print(f"[ERROR] Text RAG initialization failed: {e}")
            if init_all:
                raise
        
        # Initialize ColPali (may not be available in all environments)
        # Skip ColPali for initial testing to avoid timeouts
        print("[SKIP] ColPali initialization skipped for fast testing")
        self.colpali_retriever = None
        
        # Initialize Salesforce (requires credentials)
        try:
            self.salesforce_connector = SalesforceConnector()
            print("[OK] Salesforce connector initialized")
        except Exception as e:
            print(f"[WARN] Salesforce initialization failed (optional): {e}")
            self.salesforce_connector = None
        
        # Initialize re-ranker
        try:
            self.reranker = CrossEncoderReRanker()
            print("[OK] Cross-encoder re-ranker initialized")
        except Exception as e:
            print(f"[WARN] Re-ranker initialization failed: {e}")
            self.reranker = None
        
        # Initialize agentic orchestrator
        self.agentic_orchestrator = AgenticOrchestrator(
            rag_system=self.rag_system,
            colpali_retriever=self.colpali_retriever,
            salesforce_connector=self.salesforce_connector,
            reranker=self.reranker,
            max_steps=self.config.get("max_agent_steps", 5),
            confidence_threshold=self.config.get("confidence_threshold", 0.8)
        )
        
        print("[OK] Agentic orchestrator initialized")
        print(f"[CONFIDENCE] Available sources: {self._get_available_sources()}")
    
    def run_single_test(self, query: str, expected_answer: Optional[str] = None,
                       test_both: bool = True) -> Dict[str, Any]:
        """
        Run a single test query through both agentic and baseline systems.
        
        Args:
            query: Test query to run
            expected_answer: Optional expected answer for evaluation
            test_both: Whether to test both agentic and baseline approaches
            
        Returns:
            Test results dictionary
        """
        print(f"\n[QUERY] Testing Query: {query}")
        print("=" * 80)
        
        results = {
            "query": query,
            "expected_answer": expected_answer,
            "timestamp": time.time()
        }
        
        # Test agentic approach
        if self.agentic_orchestrator:
            print("\n[AGENTIC] AGENTIC APPROACH:")
            start_time = time.time()
            
            try:
                agentic_response = self.agentic_orchestrator.query(query)
                results["agentic"] = {
                    "answer": agentic_response.final_answer,
                    "execution_time": agentic_response.execution_time,
                    "steps": agentic_response.total_steps,
                    "sources_used": [s.value for s in agentic_response.sources_used],
                    "confidence": agentic_response.confidence_score,
                    "reasoning_chain": [asdict(action) for action in agentic_response.reasoning_chain]
                }
                
                print(f"[OK] Answer: {agentic_response.final_answer}")
                print(f"[TIME] Time: {agentic_response.execution_time:.2f}s")
                print(f"[STEPS] Steps: {agentic_response.total_steps}")
                print(f"[CONFIDENCE] Confidence: {agentic_response.confidence_score:.2f}")
                print(f"[QUERY] Sources: {', '.join([s.value for s in agentic_response.sources_used])}")
                
                # Show reasoning chain
                print("\n[REASONING] Reasoning Chain:")
                reasoning_summary = self.agentic_orchestrator.get_reasoning_summary(agentic_response)
                print(reasoning_summary)
                
                # Add to memory
                self.agent_memory.add_conversation_turn(
                    user_query=query,
                    agent_response=agentic_response.final_answer,
                    reasoning_chain=results["agentic"]["reasoning_chain"],
                    sources_used=results["agentic"]["sources_used"],
                    confidence_score=agentic_response.confidence_score,
                    execution_time=agentic_response.execution_time
                )
                
            except Exception as e:
                print(f"[ERROR] Agentic approach failed: {e}")
                results["agentic"] = {"error": str(e)}
        
        # Test baseline approach (single-turn)
        if test_both and self.rag_system:
            print(f"\n[BASELINE] BASELINE APPROACH (Single-turn):")
            
            try:
                start_time = time.time()
                baseline_response = self.rag_system.query(query)
                baseline_time = time.time() - start_time
                
                results["baseline"] = {
                    "answer": baseline_response.get("answer", ""),
                    "execution_time": baseline_time,
                    "steps": 1,
                    "sources_used": ["text_rag"],
                    "confidence": baseline_response.get("confidence", 0.0),
                    "relevant_chunks": len(baseline_response.get("relevant_chunks", []))
                }
                
                print(f"[OK] Answer: {baseline_response.get('answer', 'No answer')}")
                print(f"[TIME] Time: {baseline_time:.2f}s")
                print(f"[CONFIDENCE] Confidence: {baseline_response.get('confidence', 0.0):.2f}")
                print(f"[INFO] Chunks: {len(baseline_response.get('relevant_chunks', []))}")
                
            except Exception as e:
                print(f"[ERROR] Baseline approach failed: {e}")
                results["baseline"] = {"error": str(e)}
        
        # Evaluate results if expected answer provided
        if expected_answer:
            print(f"\n[EVAL] EVALUATION:")
            if "agentic" in results and "answer" in results["agentic"]:
                agentic_eval = self.evaluator.evaluate_response(
                    results["agentic"]["answer"], expected_answer, query
                )
                results["agentic"]["evaluation"] = agentic_eval
                print(f"[AGENTIC] Agentic Score: {agentic_eval.get('overall_score', 0):.2f}")
            
            if "baseline" in results and "answer" in results["baseline"]:
                baseline_eval = self.evaluator.evaluate_response(
                    results["baseline"]["answer"], expected_answer, query
                )
                results["baseline"]["evaluation"] = baseline_eval
                print(f"[BASELINE] Baseline Score: {baseline_eval.get('overall_score', 0):.2f}")
        
        # Store results
        self.test_results.append(results)
        
        return results
    
    def run_test_suite(self, test_scenarios_file: str = "test_scenarios.json") -> Dict[str, Any]:
        """
        Run a complete test suite from scenarios file.
        
        Args:
            test_scenarios_file: Path to test scenarios JSON file
            
        Returns:
            Aggregated test results
        """
        print(f"\n[TEST] Running Test Suite from {test_scenarios_file}")
        print("=" * 80)
        
        # Load test scenarios
        scenarios_path = os.path.join(os.path.dirname(__file__), test_scenarios_file)
        try:
            with open(scenarios_path, 'r') as f:
                scenarios = json.load(f)
        except FileNotFoundError:
            print(f"[ERROR] Test scenarios file not found: {scenarios_path}")
            return {"error": "Test scenarios file not found"}
        
        suite_results = {
            "total_tests": len(scenarios),
            "passed": 0,
            "failed": 0,
            "test_results": [],
            "summary": {}
        }
        
        # Run each test scenario
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n[TEST] Test {i}/{len(scenarios)}: {scenario.get('name', 'Unnamed Test')}")
            
            try:
                result = self.run_single_test(
                    query=scenario["query"],
                    expected_answer=scenario.get("expected_answer"),
                    test_both=scenario.get("test_both", True)
                )
                
                result["test_name"] = scenario.get("name", f"Test_{i}")
                result["test_category"] = scenario.get("category", "general")
                
                # Determine if test passed
                if "expected_answer" in scenario:
                    agentic_score = result.get("agentic", {}).get("evaluation", {}).get("overall_score", 0)
                    baseline_score = result.get("baseline", {}).get("evaluation", {}).get("overall_score", 0)
                    
                    # Test passes if agentic approach performs better or achieves good score
                    if agentic_score > max(baseline_score, 0.7):
                        suite_results["passed"] += 1
                        result["test_status"] = "PASSED"
                    else:
                        suite_results["failed"] += 1
                        result["test_status"] = "FAILED"
                else:
                    suite_results["passed"] += 1
                    result["test_status"] = "COMPLETED"
                
                suite_results["test_results"].append(result)
                
            except Exception as e:
                print(f"[ERROR] Test {i} failed with error: {e}")
                suite_results["failed"] += 1
                suite_results["test_results"].append({
                    "test_name": scenario.get("name", f"Test_{i}"),
                    "error": str(e),
                    "test_status": "ERROR"
                })
        
        # Generate summary
        suite_results["summary"] = self._generate_test_summary(suite_results)
        
        print(f"\n[CONFIDENCE] TEST SUITE COMPLETE")
        print(f"[OK] Passed: {suite_results['passed']}")
        print(f"[ERROR] Failed: {suite_results['failed']}")
        print(f"[RATE] Success Rate: {suite_results['passed']/suite_results['total_tests']*100:.1f}%")
        
        return suite_results
    
    def interactive_mode(self):
        """
        Run interactive testing mode for manual query testing.
        """
        print("\n[INTERACTIVE] Interactive Testing Mode")
        print("Commands: 'quit' to exit, 'memory' for conversation summary, 'clear' to clear memory")
        print("=" * 80)
        
        while True:
            try:
                query = input("\n[QUERY] Enter your query (or command): ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                elif query.lower() == 'memory':
                    summary = self.agent_memory.get_conversation_summary()
                    print(f"\n[CONFIDENCE] Memory Summary:")
                    print(json.dumps(summary, indent=2))
                    continue
                elif query.lower() == 'clear':
                    self.agent_memory.clear_session()
                    print("[CLEAR] Memory cleared")
                    continue
                elif not query:
                    continue
                
                # Run the query
                result = self.run_single_test(query, test_both=True)
                
                # Show comparison if both approaches worked
                if "agentic" in result and "baseline" in result:
                    if "answer" in result["agentic"] and "answer" in result["baseline"]:
                        print(f"\n[COMPARISON] COMPARISON:")
                        print(f"[AGENTIC] Agentic: {result['agentic']['execution_time']:.2f}s, {result['agentic']['steps']} steps")
                        print(f"[BASELINE] Baseline: {result['baseline']['execution_time']:.2f}s, 1 step")
                        
                        if result["agentic"]["execution_time"] < result["baseline"]["execution_time"]:
                            print("[WINNER] Agentic was faster")
                        else:
                            print("[WINNER] Baseline was faster")
                
            except KeyboardInterrupt:
                print("\n[BYE] Goodbye!")
                break
            except Exception as e:
                print(f"[ERROR] Error: {e}")
    
    def save_results(self, filename: str = None):
        """Save test results to file."""
        if not filename:
            timestamp = int(time.time())
            filename = f"test_results_{timestamp}.json"
        
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        results_data = {
            "config": self.config,
            "test_results": self.test_results,
            "memory_summary": self.agent_memory.get_conversation_summary(),
            "timestamp": time.time()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            print(f"[SAVE] Results saved to {filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to save results: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default test configuration."""
        return {
            "max_agent_steps": 5,
            "confidence_threshold": 0.8,
            "max_conversation_length": 10,
            "memory_file": "test_memory.json",
            "enable_logging": True,
            "timeout_seconds": 300
        }
    
    def _get_available_sources(self) -> List[str]:
        """Get list of available sources."""
        sources = []
        if self.rag_system:
            sources.append("text_rag")
        if self.colpali_retriever:
            sources.append("colpali_visual")
        if self.salesforce_connector:
            sources.append("salesforce")
        return sources
    
    def _generate_test_summary(self, suite_results: Dict) -> Dict[str, Any]:
        """Generate summary statistics from test suite results."""
        if not suite_results["test_results"]:
            return {}
        
        # Calculate average metrics
        agentic_times = []
        baseline_times = []
        agentic_scores = []
        baseline_scores = []
        
        for result in suite_results["test_results"]:
            if "agentic" in result and "execution_time" in result["agentic"]:
                agentic_times.append(result["agentic"]["execution_time"])
                if "evaluation" in result["agentic"]:
                    agentic_scores.append(result["agentic"]["evaluation"].get("overall_score", 0))
            
            if "baseline" in result and "execution_time" in result["baseline"]:
                baseline_times.append(result["baseline"]["execution_time"])
                if "evaluation" in result["baseline"]:
                    baseline_scores.append(result["baseline"]["evaluation"].get("overall_score", 0))
        
        summary = {
            "success_rate": suite_results["passed"] / suite_results["total_tests"],
            "avg_agentic_time": sum(agentic_times) / len(agentic_times) if agentic_times else 0,
            "avg_baseline_time": sum(baseline_times) / len(baseline_times) if baseline_times else 0,
            "avg_agentic_score": sum(agentic_scores) / len(agentic_scores) if agentic_scores else 0,
            "avg_baseline_score": sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0,
            "agentic_faster_count": sum(1 for i, at in enumerate(agentic_times) 
                                       if i < len(baseline_times) and at < baseline_times[i]),
            "agentic_better_count": sum(1 for i, ascore in enumerate(agentic_scores)
                                       if i < len(baseline_scores) and ascore > baseline_scores[i])
        }
        
        return summary

def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Harness for Agentic RAG System")
    parser.add_argument("--mode", choices=["interactive", "suite", "single"], 
                       default="interactive", help="Testing mode")
    parser.add_argument("--query", type=str, help="Single query to test")
    parser.add_argument("--scenarios", type=str, default="test_scenarios.json",
                       help="Test scenarios file")
    parser.add_argument("--config", type=str, help="Config file path")
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config from {args.config}: {e}")
    
    # Initialize test harness
    harness = TestHarness(config)
    harness.setup_components()
    
    # Run based on mode
    if args.mode == "interactive":
        harness.interactive_mode()
    elif args.mode == "suite":
        results = harness.run_test_suite(args.scenarios)
        harness.save_results()
    elif args.mode == "single" and args.query:
        result = harness.run_single_test(args.query)
        print(f"\n[RESULT] Result: {json.dumps(result, indent=2, default=str)}")
    else:
        print("[ERROR] Invalid mode or missing query for single mode")

if __name__ == "__main__":
    main()