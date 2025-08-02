"""
Quick test script to validate Graph-R1 fixes.
"""
import sys
import logging
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def test_salesforce_method():
    """Test 1: Verify SalesforceConnector has search_knowledge_base method."""
    try:
        from src.salesforce_connector import SalesforceConnector
        connector = SalesforceConnector()
        
        # Check if method exists
        if hasattr(connector, 'search_knowledge_base'):
            logger.info("✅ Test 1 PASSED: search_knowledge_base method exists")
            return True
        else:
            logger.error("❌ Test 1 FAILED: search_knowledge_base method missing")
            return False
    except Exception as e:
        logger.error(f"❌ Test 1 ERROR: {e}")
        return False

def test_hypergraph_thresholds():
    """Test 2: Verify hypergraph constructor has lower thresholds."""
    try:
        from src.hypergraph_constructor import HypergraphBuilder
        
        # Create builder with default config
        builder = HypergraphBuilder({})
        
        # Check cross-modal threshold
        cross_modal_threshold = builder.similarity_thresholds.get('cross_modal', 1.0)
        if cross_modal_threshold <= 0.5:
            logger.info(f"✅ Test 2 PASSED: Cross-modal threshold is {cross_modal_threshold} (≤ 0.5)")
            return True
        else:
            logger.error(f"❌ Test 2 FAILED: Cross-modal threshold is {cross_modal_threshold} (> 0.5)")
            return False
    except Exception as e:
        logger.error(f"❌ Test 2 ERROR: {e}")
        return False

def test_traversal_thresholds():
    """Test 3: Verify traversal engine has lower confidence thresholds."""
    try:
        from src.graph_traversal_engine import LLMPathPlanner, QueryType
        
        # Create path planner
        planner = LLMPathPlanner({})
        
        # Check factual threshold
        factual_threshold = planner.traversal_strategies[QueryType.FACTUAL]['confidence_threshold']
        if factual_threshold <= 0.4:
            logger.info(f"✅ Test 3 PASSED: Factual threshold is {factual_threshold} (≤ 0.4)")
            return True
        else:
            logger.error(f"❌ Test 3 FAILED: Factual threshold is {factual_threshold} (> 0.4)")
            return False
    except Exception as e:
        logger.error(f"❌ Test 3 ERROR: {e}")
        return False

def test_visual_query_detection():
    """Test 4: Verify visual query pattern detection."""
    try:
        from src.graph_traversal_engine import LLMPathPlanner
        
        # Create path planner
        planner = LLMPathPlanner({})
        
        # Check if visual patterns exist
        if hasattr(planner, 'visual_query_patterns') and len(planner.visual_query_patterns) > 0:
            logger.info(f"✅ Test 4 PASSED: Visual query patterns found ({len(planner.visual_query_patterns)} patterns)")
            return True
        else:
            logger.error("❌ Test 4 FAILED: Visual query patterns missing")
            return False
    except Exception as e:
        logger.error(f"❌ Test 4 ERROR: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🧪 Starting Graph-R1 fixes validation tests...")
    
    tests = [
        ("Salesforce Method", test_salesforce_method),
        ("Hypergraph Thresholds", test_hypergraph_thresholds),
        ("Traversal Thresholds", test_traversal_thresholds),
        ("Visual Query Detection", test_visual_query_detection)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running: {test_name}")
        if test_func():
            passed += 1
    
    logger.info(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Graph-R1 fixes are ready for testing.")
    else:
        logger.error(f"⚠️ {total - passed} tests failed. Please review the fixes.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)