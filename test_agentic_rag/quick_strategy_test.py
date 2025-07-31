"""
Quick Strategy Test - Simplified runner for immediate testing

This script provides a simplified interface for quickly testing the three
re-ranker integration strategies without full comprehensive analysis.
"""

import sys
import os
import json
import time
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components (simplified imports to avoid circular dependencies)
try:
    from reranker_integration_strategies import RerankerStrategy, StrategyComparator
    from test_harness import TestHarness
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the test_agentic_rag directory")
    sys.exit(1)

def quick_test_single_query(query: str = "What is attention mechanism in transformers?"):
    """
    Quick test of all three strategies on a single query.
    
    Args:
        query: Test query to run
        
    Returns:
        Results dictionary
    """
    print(f"🧪 Quick Strategy Test")
    print(f"Query: {query}")
    print("=" * 60)
    
    # Initialize test harness (simplified)
    print("🚀 Initializing test environment...")
    
    try:
        # Use test harness for component initialization
        harness = TestHarness({
            "max_conversation_length": 5,
            "confidence_threshold": 0.7,
            "max_reasoning_steps": 8
        })
        
        # Setup components (skip ColPali for speed)
        harness.setup_components(init_all=False)
        
        # Setup agentic orchestrator
        harness.setup_agentic_orchestrator()
        
        print("✅ Test environment ready!")
        
        # Extract components for strategy comparator
        components = {
            'rag_system': harness.rag_system,
            'colpali_retriever': harness.colpali_retriever,
            'salesforce_connector': harness.salesforce_connector,
            'reranker': harness.reranker
        }
        
        # Create strategy comparator
        comparator = StrategyComparator(**components)
        
        # Test all strategies
        print("\n🔬 Testing strategies...")
        
        strategies_to_test = [
            RerankerStrategy.PURE_AGENTIC,
            RerankerStrategy.RERANKER_ENHANCED, 
            RerankerStrategy.HYBRID_MODE
        ]
        
        results = {}
        
        for i, strategy in enumerate(strategies_to_test, 1):
            print(f"\n📋 Strategy {i}/{len(strategies_to_test)}: {strategy.value}")
            print("-" * 40)
            
            try:
                start_time = time.time()
                strategy_result = comparator.compare_strategies(query, [strategy])
                end_time = time.time()
                
                if strategy.value in strategy_result and strategy_result[strategy.value]['success']:
                    response = strategy_result[strategy.value]['response']
                    metrics = strategy_result[strategy.value]['metrics']
                    
                    print(f"✅ Success! ({end_time - start_time:.1f}s)")
                    print(f"   Steps: {response.total_steps}")
                    print(f"   Confidence: {response.confidence_score:.2f}")
                    print(f"   Sources: {len(response.sources_used)}")
                    if hasattr(response, 'reranker_tokens'):
                        print(f"   Re-ranker tokens: {response.reranker_tokens}")
                    print(f"   Answer preview: {response.final_answer[:100]}...")
                    
                    results[strategy.value] = {
                        'success': True,
                        'execution_time': end_time - start_time,
                        'response': response,
                        'metrics': metrics
                    }
                else:
                    error = strategy_result[strategy.value].get('error', 'Unknown error')
                    print(f"❌ Failed: {error}")
                    results[strategy.value] = {
                        'success': False,
                        'error': error
                    }
                    
            except Exception as e:
                print(f"❌ Exception: {str(e)}")
                results[strategy.value] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Generate quick comparison
        print("\n📊 QUICK COMPARISON SUMMARY")
        print("=" * 60)
        
        successful_strategies = {k: v for k, v in results.items() if v['success']}
        
        if successful_strategies:
            # Find best performing strategy
            best_strategy = min(successful_strategies.keys(), 
                              key=lambda k: successful_strategies[k]['execution_time'])
            
            print(f"🏆 Fastest: {best_strategy} ({successful_strategies[best_strategy]['execution_time']:.1f}s)")
            
            # Find highest confidence
            best_confidence = max(successful_strategies.keys(),
                                key=lambda k: successful_strategies[k]['response'].confidence_score)
            
            print(f"🎯 Highest Confidence: {best_confidence} ({successful_strategies[best_confidence]['response'].confidence_score:.2f})")
            
            # Find most thorough (most steps)
            most_thorough = max(successful_strategies.keys(),
                              key=lambda k: successful_strategies[k]['response'].total_steps)
            
            print(f"🔍 Most Thorough: {most_thorough} ({successful_strategies[most_thorough]['response'].total_steps} steps)")
            
            # Performance summary table
            print(f"\n📈 PERFORMANCE TABLE")
            print(f"{'Strategy':<20} {'Time (s)':<10} {'Steps':<8} {'Confidence':<12} {'Success'}")
            print("-" * 65)
            
            for strategy_name in strategies_to_test:
                strategy_key = strategy_name.value
                if strategy_key in results:
                    result = results[strategy_key]
                    if result['success']:
                        resp = result['response']
                        time_str = f"{result['execution_time']:.1f}"
                        steps_str = str(resp.total_steps)
                        conf_str = f"{resp.confidence_score:.2f}"
                        success_str = "✅"
                    else:
                        time_str = "N/A"
                        steps_str = "N/A"
                        conf_str = "N/A"
                        success_str = "❌"
                    
                    print(f"{strategy_key:<20} {time_str:<10} {steps_str:<8} {conf_str:<12} {success_str}")
        
        else:
            print("❌ No strategies completed successfully")
        
        return results
        
    except Exception as e:
        print(f"❌ Test initialization failed: {e}")
        return None

def interactive_test():
    """Interactive testing interface."""
    print("🧪 Interactive Strategy Testing")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Test default query (attention mechanism)")
        print("2. Test custom query")
        print("3. Test transformer architecture query")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            quick_test_single_query()
        elif choice == "2":
            custom_query = input("Enter your query: ").strip()
            if custom_query:
                quick_test_single_query(custom_query)
        elif choice == "3":
            quick_test_single_query("What is a transformer architecture in machine learning?")
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")

def main():
    """Main entry point."""
    print("🧪 Quick Strategy Test Runner")
    print("This tool quickly tests all three re-ranker integration strategies.")
    print()
    
    # Check if running interactively
    if len(sys.argv) > 1:
        if sys.argv[1] == "--query" and len(sys.argv) > 2:
            query = " ".join(sys.argv[2:])
            quick_test_single_query(query)
        elif sys.argv[1] == "--interactive":
            interactive_test()
        else:
            print("Usage:")
            print("  python quick_strategy_test.py                    # Run with default query")
            print("  python quick_strategy_test.py --query <text>     # Run with custom query")
            print("  python quick_strategy_test.py --interactive      # Interactive mode")
    else:
        # Run default test
        quick_test_single_query()

if __name__ == "__main__":
    main()