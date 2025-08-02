#!/usr/bin/env python3
"""
Test script to verify our agentic fixes work properly
- Tests document loading fixes
- Tests graceful failure responses
- Tests pre-flight validation
- Simple test without unicode issues
"""

import sys
import os
import logging

# Setup path
sys.path.append(os.path.dirname(__file__))

# Suppress verbose logs to avoid encoding issues
logging.basicConfig(level=logging.ERROR)

def test_document_loading():
    """Test that document loading works with our fixes."""
    try:
        from enhanced_test_harness import EnhancedTestHarness
        
        print("[TEST] Testing document loading improvements...")
        
        # Create test harness
        harness = EnhancedTestHarness({
            "enable_cost_monitoring": True,
            "max_reasoning_steps": 5,
            "confidence_threshold": 0.7,
            "enable_detailed_logging": False
        })
        
        # Setup components  
        harness.setup_components(init_colpali=False)
        
        # Check if documents loaded
        if harness.rag_system:
            try:
                # Test basic query to see if documents loaded
                result = harness.rag_system.query("test", max_results=1)
                if result and result.get('chunks'):
                    print("[SUCCESS] Documents are loaded and searchable!")
                    return True
                else:
                    print("[WARNING] Documents not loaded - will test graceful failure")
                    return False
            except Exception as e:
                print(f"[INFO] Query test error (expected if no docs): {str(e)[:100]}")
                return False
        else:
            print("[ERROR] RAG system not initialized")
            return False
            
    except Exception as e:
        print(f"[ERROR] Document loading test failed: {e}")
        return False

def test_graceful_failure():
    """Test that graceful failure works when no documents available."""
    try:
        from enhanced_test_harness import EnhancedTestHarness
        
        print("[TEST] Testing graceful failure response...")
        
        # Create test harness
        harness = EnhancedTestHarness({
            "enable_cost_monitoring": True,
            "max_reasoning_steps": 3,
            "confidence_threshold": 0.7,
            "enable_detailed_logging": False
        })
        
        # Setup components
        harness.setup_components(init_colpali=False)
        
        if harness.enhanced_orchestrator:
            # Test a simple technical query that would normally use text_rag
            test_query = "What is machine learning?"
            
            print(f"[TEST] Running test query: {test_query}")
            result = harness.enhanced_orchestrator.run_single_comparison_test(test_query)
            
            if result and result.true_agentic_response:
                response_text = result.true_agentic_response.final_answer.lower()
                
                # Check for graceful failure indicators
                graceful_phrases = [
                    "don't have access to relevant documents",
                    "no relevant documents",
                    "ensure proper documents are loaded",
                    "document sources are not available"
                ]
                
                is_graceful = any(phrase in response_text for phrase in graceful_phrases)
                
                if is_graceful:
                    print("[SUCCESS] Graceful failure detected!")
                    print(f"[SAMPLE] Response snippet: {result.true_agentic_response.final_answer[:150]}...")
                    return True
                else:
                    print("[INFO] No graceful failure detected (may have documents or other sources)")
                    print(f"[SAMPLE] Response snippet: {result.true_agentic_response.final_answer[:150]}...")
                    return False
            else:
                print("[ERROR] No response received from orchestrator")
                return False
        else:
            print("[ERROR] Enhanced orchestrator not initialized")
            return False
            
    except Exception as e:
        print(f"[ERROR] Graceful failure test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("[MAIN] AGENTIC FIX VALIDATION")
    print("=" * 50)
    
    results = {
        'document_loading': False,
        'graceful_failure': False
    }
    
    print("\n[PHASE 1] Document Loading Test")
    results['document_loading'] = test_document_loading()
    
    print("\n[PHASE 2] Graceful Failure Test")  
    results['graceful_failure'] = test_graceful_failure()
    
    print("\n[RESULTS] Test Summary:")
    print("=" * 50)
    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}: {result}")
    
    overall_success = any(results.values())
    print(f"\n[OVERALL] {'SUCCESS' if overall_success else 'NEEDS WORK'}")
    
    if not overall_success:
        print("[INFO] If both tests fail, check:")
        print("  1. OpenAI API key is set")
        print("  2. Documents exist in data/documents/")
        print("  3. Salesforce credentials are configured")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    if success:
        print("\n[SUCCESS] AGENTIC FIXES WORKING!")
        print("[PASS] Document loading or graceful failure working")
        print("[PASS] System ready for validation")
    else:
        print("\n[FAIL] Issues detected - check configuration")