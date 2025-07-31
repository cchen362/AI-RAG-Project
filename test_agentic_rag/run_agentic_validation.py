"""
Agentic Validation Test Runner

Quick validation script to test the true LLM reasoning agent against
the pseudo-agentic baseline. Demonstrates dynamic reasoning vs fixed pipeline.

Usage:
    python run_agentic_validation.py
"""

import sys
import os
import logging
from typing import Dict, Any

# Setup path
sys.path.append(os.path.dirname(__file__))

try:
    from enhanced_test_harness import EnhancedTestHarness, ReasoningMode
    print("✅ Enhanced test harness imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please ensure you're running from test_agentic_rag directory")
    sys.exit(1)

def run_quick_validation():
    """Run quick validation of true vs pseudo-agentic reasoning."""
    print("🚀 AGENTIC REASONING VALIDATION")
    print("=" * 50)
    print("Testing TRUE LLM-driven vs PSEUDO fixed pipeline")
    print()
    
    # Initialize test harness
    print("🔧 Initializing test environment...")
    harness = EnhancedTestHarness({
        "enable_cost_monitoring": True,
        "max_reasoning_steps": 8,
        "confidence_threshold": 0.7
    })
    
    try:
        # Setup components (skip ColPali for speed)
        harness.setup_components(init_colpali=False)
        print("✅ Test environment ready!")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("Check your OpenAI API key and try again")
        return None
    
    # Test query that should show different behavior
    test_query = "What is attention mechanism in transformers?"
    
    print(f"\n🧪 VALIDATION TEST")
    print(f"Query: {test_query}")
    print("Expected: TRUE agentic should use fewer sources intelligently")
    print("Expected: PSEUDO agentic should use fixed sequence (all sources)")
    print()
    
    try:
        # Run A/B comparison
        print("🔄 Running TRUE vs PSEUDO comparison...")
        comparison_result = harness.run_single_comparison_test(test_query)
        
        # Analyze results
        print(f"\n🔍 VALIDATION ANALYSIS")
        print("-" * 40)
        
        # Check for true agentic behavior
        true_reasoning = comparison_result.performance_comparison
        intelligent_selection = true_reasoning["source_utilization"]["intelligent_selection"]
        dynamic_stopping = true_reasoning["source_utilization"]["dynamic_stopping"]
        dynamic_decisions = true_reasoning["reasoning_complexity"]["dynamic_decisions"]
        
        print(f"✅ Intelligent Source Selection: {intelligent_selection}")
        print(f"✅ Dynamic Stopping: {dynamic_stopping}")
        print(f"✅ Dynamic Decisions: {dynamic_decisions > 0}")
        
        # Validation score
        validation_score = sum([
            intelligent_selection,
            dynamic_stopping,
            dynamic_decisions > 0
        ])
        
        print(f"\n🎯 VALIDATION SCORE: {validation_score}/3")
        
        if validation_score >= 2:
            print("🎉 SUCCESS: True agentic reasoning is working!")
            print("   System shows intelligent behavior vs fixed pipeline")
        elif validation_score == 1:
            print("⚠️  PARTIAL: Some agentic behavior detected")
            print("   May need prompt tuning or threshold adjustment")
        else:
            print("❌ FAILED: No clear agentic behavior detected")
            print("   Check LLM prompts and reasoning logic")
        
        # Cost analysis
        cost_analysis = comparison_result.cost_analysis
        print(f"\n💰 COST ANALYSIS")
        print(f"   True Agentic: ${cost_analysis['true_agentic_cost']:.4f}")
        print(f"   Pseudo Agentic: ${cost_analysis['pseudo_agentic_cost']:.4f}")
        print(f"   Efficiency: {cost_analysis['cost_efficiency']}")
        
        # Recommendation
        print(f"\n🏆 SYSTEM RECOMMENDATION:")
        print(f"   {comparison_result.recommendation}")
        
        return comparison_result
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_comprehensive_validation():
    """Run comprehensive validation across multiple query types."""
    print("\n🌟 COMPREHENSIVE VALIDATION")
    print("=" * 50)
    print("Testing across diverse query types...")
    
    harness = EnhancedTestHarness({
        "enable_cost_monitoring": True,
        "max_reasoning_steps": 10,
        "confidence_threshold": 0.75
    })
    
    try:
        harness.setup_components(init_colpali=False)
        
        # Run diversity test
        diversity_results = harness.run_query_diversity_test()
        
        # Save results
        results_file = harness.save_test_results()
        
        print(f"\n📊 COMPREHENSIVE VALIDATION COMPLETE")
        print(f"   Status: {diversity_results['validation_summary']['status'].upper()}")
        print(f"   Results saved: {results_file}")
        
        return diversity_results
        
    except Exception as e:
        print(f"❌ Comprehensive validation failed: {e}")
        return None

def demo_reasoning_modes():
    """Demonstrate different reasoning modes side by side."""
    print("\n🎭 REASONING MODE DEMONSTRATION")
    print("=" * 50)
    
    harness = EnhancedTestHarness()
    
    try:
        harness.setup_components(init_colpali=False)
        
        test_query = "How do transformers use attention mechanisms for language modeling?"
        
        print(f"Query: {test_query}")
        print()
        
        # Test TRUE agentic
        print("🧠 TRUE AGENTIC MODE:")
        true_result = harness.run_single_mode_test(test_query, ReasoningMode.TRUE_AGENTIC)
        
        # Test PSEUDO agentic
        print("\n🤖 PSEUDO AGENTIC MODE:")
        pseudo_result = harness.run_single_mode_test(test_query, ReasoningMode.PSEUDO_AGENTIC)
        
        # Compare
        print(f"\n📊 COMPARISON:")
        print(f"   TRUE - Sources: {len(true_result.sources_used)}, Steps: {true_result.reasoning_transparency.get('total_reasoning_steps', 0)}")
        print(f"   PSEUDO - Sources: {len(pseudo_result.sources_used)}, Steps: {pseudo_result.reasoning_transparency.get('total_reasoning_steps', 0)}")
        print(f"   Different behavior: {true_result.sources_used != pseudo_result.sources_used}")
        
        return {"true": true_result, "pseudo": pseudo_result}
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return None

def main():
    """Main validation runner."""
    print("🧪 AGENTIC RAG VALIDATION SUITE")
    print("=" * 60)
    print("Choose validation type:")
    print("1. Quick validation (single query A/B test)")
    print("2. Comprehensive validation (multiple query types)")  
    print("3. Reasoning mode demonstration")
    print("4. Run all validations")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            run_quick_validation()
        elif choice == "2":
            run_comprehensive_validation()
        elif choice == "3":
            demo_reasoning_modes()
        elif choice == "4":
            print("🚀 Running all validations...")
            run_quick_validation()
            run_comprehensive_validation()
            demo_reasoning_modes()
        else:
            print("❌ Invalid choice")
            return
            
        print(f"\n✅ Validation complete!")
        
    except KeyboardInterrupt:
        print("\n👋 Validation interrupted by user")
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()