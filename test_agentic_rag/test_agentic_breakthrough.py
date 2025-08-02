"""
Agentic Breakthrough Validation Test

Quick validation script to test the fixes for the true agentic system.
Tests that the "0 chunks" issue is resolved and response quality matches pseudo-agentic.
"""

import sys
import os
import logging
from typing import Dict, Any

# Setup path
sys.path.append(os.path.dirname(__file__))

try:
    from enhanced_test_harness import EnhancedTestHarness, ReasoningMode
    print("[SUCCESS] Enhanced test harness imported successfully")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("Please ensure you're running from test_agentic_rag directory")
    sys.exit(1)

def test_content_detection_fix():
    """Test that content detection now properly recognizes synthesized content."""
    print("\n[TEST 1] CONTENT DETECTION FIX VALIDATION")
    print("=" * 60)
    
    # Initialize test harness
    harness = EnhancedTestHarness({
        "enable_cost_monitoring": True,
        "max_reasoning_steps": 5,
        "confidence_threshold": 0.7,
        "enable_detailed_logging": False  # Reduce noise for this test
    })
    
    try:
        harness.setup_components(init_colpali=False)
        
        # Test a simple technical query that should find synthesized content
        test_query = "What is attention mechanism?"
        
        print(f"Testing query: {test_query}")
        
        # Run true agentic test
        result = harness.run_single_mode_test(test_query, ReasoningMode.TRUE_AGENTIC)
        
        # Validate fixes
        validation_results = {
            "content_detected": len(result.sources_used) > 0,
            "confidence_reasonable": result.confidence_score > 0.5,
            "response_quality": len(result.final_answer) > 100,
            "no_insufficient_data_error": "don't have access to relevant documents" not in result.final_answer.lower()
        }
        
        print("\n📊 VALIDATION RESULTS:")
        for check, passed in validation_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {check}: {status}")
        
        passed_checks = sum(validation_results.values())
        total_checks = len(validation_results)
        
        print(f"\n🎯 OVERALL: {passed_checks}/{total_checks} checks passed")
        
        if passed_checks >= 3:
            print("✅ CONTENT DETECTION FIX: SUCCESS")
            return True
        else:
            print("❌ CONTENT DETECTION FIX: NEEDS WORK")
            print(f"Response preview: {result.final_answer[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_response_quality_parity():
    """Test that true agentic response quality now matches pseudo-agentic."""
    print("\n[TEST 2] RESPONSE QUALITY PARITY VALIDATION")
    print("=" * 60)
    
    # Initialize test harness
    harness = EnhancedTestHarness({
        "enable_cost_monitoring": True,
        "max_reasoning_steps": 5,
        "confidence_threshold": 0.7,
        "enable_detailed_logging": False
    })
    
    try:
        harness.setup_components(init_colpali=False)
        
        # Test query that should show quality differences
        test_query = "How do transformers use attention mechanisms for language modeling?"
        
        print(f"Testing query: {test_query}")
        
        # Run A/B comparison test
        comparison = harness.run_single_comparison_test(test_query)
        
        # Extract responses
        true_response = comparison.true_agentic_response.final_answer
        pseudo_response = comparison.pseudo_agentic_response.final_answer
        
        # Quality metrics
        true_metrics = {
            "length": len(true_response),
            "has_technical_content": any(term in true_response.lower() for term in ["attention", "mechanism", "transformer"]),
            "not_error_message": "don't have access" not in true_response.lower(),
            "structured_content": true_response.count('\n') > 0 or true_response.count('.') > 2
        }
        
        pseudo_metrics = {
            "length": len(pseudo_response),
            "has_technical_content": any(term in pseudo_response.lower() for term in ["attention", "mechanism", "transformer"]),
            "not_error_message": "don't have access" not in pseudo_response.lower(),
            "structured_content": pseudo_response.count('\n') > 0 or pseudo_response.count('.') > 2
        }
        
        print(f"\n📊 TRUE AGENTIC METRICS:")
        for metric, value in true_metrics.items():
            print(f"   {metric}: {value}")
        
        print(f"\n📊 PSEUDO AGENTIC METRICS:")
        for metric, value in pseudo_metrics.items():
            print(f"   {metric}: {value}")
        
        # Compare quality
        true_score = sum(1 for v in true_metrics.values() if (isinstance(v, bool) and v) or (isinstance(v, int) and v > 50))
        pseudo_score = sum(1 for v in pseudo_metrics.values() if (isinstance(v, bool) and v) or (isinstance(v, int) and v > 50))
        
        quality_parity = true_score >= pseudo_score * 0.8  # Allow 20% tolerance
        
        print(f"\n🎯 QUALITY COMPARISON:")
        print(f"   True Agentic Score: {true_score}/4")
        print(f"   Pseudo Agentic Score: {pseudo_score}/4")
        print(f"   Quality Parity: {'✅ ACHIEVED' if quality_parity else '❌ NEEDS IMPROVEMENT'}")
        
        if quality_parity:
            print("✅ RESPONSE QUALITY PARITY: SUCCESS")
        else:
            print("❌ RESPONSE QUALITY PARITY: NEEDS WORK")
            print(f"\nTrue Response Preview: {true_response[:300]}...")
            print(f"\nPseudo Response Preview: {pseudo_response[:300]}...")
        
        return quality_parity
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_reasoning_consistency():
    """Test that LLM reasoning is now more consistent."""
    print("\n[TEST 3] REASONING CONSISTENCY VALIDATION")
    print("=" * 60)
    
    # Initialize test harness
    harness = EnhancedTestHarness({
        "enable_cost_monitoring": True,
        "max_reasoning_steps": 8,
        "confidence_threshold": 0.7,
        "enable_detailed_logging": True  # Need to see reasoning
    })
    
    try:
        harness.setup_components(init_colpali=False)
        
        # Test the same query multiple times to check consistency
        test_query = "What is attention mechanism in transformers?"
        
        print(f"Testing reasoning consistency for: {test_query}")
        
        results = []
        for i in range(2):  # Run twice to check consistency
            print(f"\n--- Run {i+1} ---")
            result = harness.run_single_mode_test(test_query, ReasoningMode.TRUE_AGENTIC)
            results.append({
                "sources_used": result.sources_used,
                "confidence": result.confidence_score,
                "reasoning_steps": result.reasoning_transparency.get("total_reasoning_steps", 0),
                "response_length": len(result.final_answer)
            })
        
        # Check consistency
        consistency_checks = {
            "same_sources": results[0]["sources_used"] == results[1]["sources_used"],
            "similar_confidence": abs(results[0]["confidence"] - results[1]["confidence"]) < 0.3,
            "reasonable_steps": all(r["reasoning_steps"] >= 2 for r in results),
            "quality_responses": all(r["response_length"] > 100 for r in results)
        }
        
        print(f"\n📊 CONSISTENCY CHECKS:")
        for check, passed in consistency_checks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {check}: {status}")
        
        passed_checks = sum(consistency_checks.values())
        total_checks = len(consistency_checks)
        
        print(f"\n🎯 CONSISTENCY SCORE: {passed_checks}/{total_checks}")
        
        if passed_checks >= 3:
            print("✅ REASONING CONSISTENCY: SUCCESS")
            return True
        else:
            print("❌ REASONING CONSISTENCY: NEEDS WORK")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def main():
    """Run all breakthrough validation tests."""
    print("[MAIN] AGENTIC BREAKTHROUGH VALIDATION SUITE")
    print("=" * 80)
    print("Testing fixes for true agentic system improvements")
    print()
    
    # Configure logging to show key info
    logging.getLogger("test_agentic_rag.llm_reasoning_agent").setLevel(logging.INFO)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Content Detection Fix", test_content_detection_fix),
        ("Response Quality Parity", test_response_quality_parity),
        ("Reasoning Consistency", test_reasoning_consistency)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        result = test_func()
        test_results.append((test_name, result))
    
    # Overall results
    print(f"\n{'='*80}")
    print("[FINAL] BREAKTHROUGH VALIDATION RESULTS")
    print("=" * 80)
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    total = len(test_results)
    success_rate = passed / total
    
    print(f"\n🎯 OVERALL SUCCESS RATE: {passed}/{total} ({success_rate:.1%})")
    
    if success_rate >= 0.67:  # 2/3 tests must pass
        print("\n🚀 BREAKTHROUGH VALIDATED: True agentic system improvements successful!")
    else:
        print("\n⚠️  BREAKTHROUGH PARTIAL: Some fixes still needed")
    
    return success_rate >= 0.67

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    success = main()
    sys.exit(0 if success else 1)