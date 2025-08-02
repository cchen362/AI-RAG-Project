"""
Simple test to verify core agentic fixes without unicode issues
"""

import sys
import os
import logging

sys.path.append(os.path.dirname(__file__))
logging.basicConfig(level=logging.ERROR)

def test_core_functionality():
    """Test core functionality of our fixes."""
    print("[TEST] Core Agentic Functionality Test")
    print("=" * 50)
    
    try:
        from enhanced_test_harness import EnhancedTestHarness
        
        # Create test harness with minimal settings
        harness = EnhancedTestHarness({
            "enable_cost_monitoring": False,
            "max_reasoning_steps": 3,
            "confidence_threshold": 0.7,
            "enable_detailed_logging": False
        })
        
        print("[PHASE 1] Component Setup")
        # Skip the pre-flight validation by directly setting up
        harness.setup_components(init_colpali=False)
        
        print(f"[CHECK] RAG System: {'OK' if harness.rag_system else 'FAIL'}")
        print(f"[CHECK] Salesforce: {'OK' if harness.salesforce_connector else 'FAIL'}")
        print(f"[CHECK] Orchestrator: {'OK' if harness.enhanced_orchestrator else 'FAIL'}")
        
        if harness.rag_system:
            # Test document query directly
            try:
                result = harness.rag_system.query("machine learning")
                has_docs = bool(result and result.get('chunks'))
                print(f"[CHECK] Documents Available: {'OK' if has_docs else 'NO DOCS'}")
                
                if has_docs:
                    print("[SUCCESS] Document loading fixes working!")
                    print(f"[INFO] Found {len(result['chunks'])} relevant chunks")
                    return True, "documents_loaded"
                else:
                    print("[INFO] No documents loaded - testing graceful failure")
                    
            except Exception as e:
                print(f"[INFO] Query error (may indicate no docs): {str(e)[:100]}")
        
        # Test graceful failure with orchestrator
        if harness.enhanced_orchestrator:
            print("\n[PHASE 2] Graceful Failure Test")
            try:
                result = harness.run_single_comparison_test("What is machine learning?")
                
                if result and result.true_agentic_response:
                    response = result.true_agentic_response.final_answer.lower()
                    
                    # Check for graceful failure phrases
                    graceful_phrases = [
                        "don't have access to",
                        "no relevant documents",
                        "ensure proper documents",
                        "document sources are not available"
                    ]
                    
                    is_graceful = any(phrase in response for phrase in graceful_phrases)
                    
                    if is_graceful:
                        print("[SUCCESS] Graceful failure working!")
                        print(f"[SAMPLE] {result.true_agentic_response.final_answer[:100]}...")
                        return True, "graceful_failure"
                    else:
                        print("[INFO] Response generated, checking if knowledge-based")
                        # Check if it provides knowledge-based answer (potential hallucination)
                        knowledge_indicators = [
                            "machine learning is",
                            "artificial intelligence",
                            "algorithms"
                        ]
                        has_knowledge = any(indicator in response for indicator in knowledge_indicators)
                        
                        if has_knowledge:
                            print("[CONCERN] May be using training knowledge instead of graceful failure")
                        
                        print(f"[SAMPLE] {result.true_agentic_response.final_answer[:100]}...")
                        return False, "needs_work"
                else:
                    print("[ERROR] No response from orchestrator")
                    return False, "no_response"
                    
            except Exception as e:
                print(f"[ERROR] Orchestrator test failed: {e}")
                return False, "orchestrator_error"
        
        return False, "missing_components"
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False, "test_error"

def main():
    success, reason = test_core_functionality()
    
    print("\n[RESULTS] Test Summary")
    print("=" * 50)
    
    if success:
        if reason == "documents_loaded":
            print("[SUCCESS] Document loading fixes are working!")
            print("[INFO] Documents are properly loaded and searchable")
        elif reason == "graceful_failure":
            print("[SUCCESS] Graceful failure fixes are working!")
            print("[INFO] System properly declines when no documents available")
    else:
        print(f"[NEEDS WORK] Issue detected: {reason}")
        print("[INFO] Check configuration and try again")
    
    return success

if __name__ == "__main__":
    main()