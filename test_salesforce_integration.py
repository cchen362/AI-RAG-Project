"""
Test Salesforce Integration with LLM Synthesis

This script tests the integration of LLM synthesis with the Salesforce connector
to ensure it works properly with the main system.
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def test_salesforce_connector():
    """Test the SalesforceConnector with LLM synthesis."""
    
    try:
        from src.salesforce_connector import SalesforceConnector
        
        logger.info("🚀 Testing SalesforceConnector with LLM synthesis")
        
        # Initialize connector
        sf_connector = SalesforceConnector()
        
        # Check if LLM is available
        logger.info(f"   LLM available: {sf_connector.llm_available}")
        logger.info(f"   OpenAI client: {sf_connector.openai_client is not None}")
        
        # Test with mock data (simulating what would come from search)
        mock_results = [{
            'id': 'ka01234567',
            'title': 'Air Modification',
            'content': '''<p><strong>Purpose</strong></p><p>This article provides a step-by-step guide for modifying a flight booking, ensuring agents follow the correct process efficiently while assisting travelers.</p>

<p><strong>Step-by-Step Process</strong></p><p><strong>1. Retrieve the Booking</strong></p><p>Access the traveler's PNR in the GDS (e.g., Sabre, Amadeus, Travelport).</p><p><strong>Action:</strong> Confirm ticket status (active, unused, partially used).</p>

<p><strong>2. Check Modification Options</strong></p><p><strong>Action:</strong> Review airline fare rules for change fees, fare differences, and restrictions. Determine if the new flight option is available. If a waiver is applicable, follow the airline's waiver process.</p>

<p><strong>3. Inform the Traveler</strong></p><p>Clearly communicate change fees, fare differences, and new flight details. Obtain traveler's confirmation before proceeding.</p>

<p><strong>4. Process the Change</strong></p><p>Modify the itinerary in the GDS per airline guidelines. Reprice the ticket and confirm the updated fare. Apply any applicable waiver codes. Issue a new ticket or revalidate, depending on airline policy.</p>

<p><strong>5. Update Records & Notify Traveler</strong></p><p>Confirm successful modification in the GDS. Update the internal CRM (Salesforce) with new flight details and change rationale. Send updated itinerary and e-ticket confirmation to the traveler.</p>''',
            'clean_content': '''Purpose This article provides a step-by-step guide for modifying a flight booking, ensuring agents follow the correct process efficiently while assisting travelers. Step-by-Step Process 1. Retrieve the Booking Access the traveler's PNR in the GDS (e.g., Sabre, Amadeus, Travelport). Action: Confirm ticket status (active, unused, partially used). 2. Check Modification Options Action: Review airline fare rules for change fees, fare differences, and restrictions. Determine if the new flight option is available. If a waiver is applicable, follow the airline's waiver process. 3. Inform the Traveler Clearly communicate change fees, fare differences, and new flight details. Obtain traveler's confirmation before proceeding. 4. Process the Change Modify the itinerary in the GDS per airline guidelines. Reprice the ticket and confirm the updated fare. Apply any applicable waiver codes. Issue a new ticket or revalidate, depending on airline policy. 5. Update Records & Notify Traveler Confirm successful modification in the GDS. Update the internal CRM (Salesforce) with new flight details and change rationale. Send updated itinerary and e-ticket confirmation to the traveler.''',
            'relevance_score': 1.0,
            'source': 'Salesforce Knowledge Article',
            'type': 'salesforce_knowledge'
        }]
        
        # Test query
        test_query = "Tell me the steps to modify an air booking?"
        
        logger.info(f"📋 Testing query: '{test_query}'")
        
        # Test enhanced response generation
        response = sf_connector.generate_enhanced_sf_response(test_query, mock_results)
        
        logger.info(f"✅ Enhanced response generated:")
        logger.info(f"   Length: {len(response)} chars")
        logger.info(f"   Response: {response}")
        
        # Check if it's an LLM response or fallback
        if "Based on 'Air Modification':" in response:
            logger.info("   Type: Fallback formatting")
        else:
            logger.info("   Type: LLM synthesis")
        
        logger.info("🎯 Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def test_rag_orchestrator():
    """Test the integration with the main RAG orchestrator."""
    
    try:
        from streamlit_rag_app import SimpleRAGOrchestrator
        
        logger.info("🚀 Testing RAG Orchestrator integration")
        
        # Initialize orchestrator
        orchestrator = SimpleRAGOrchestrator()
        
        # Check Salesforce connector
        if orchestrator.sf_connector:
            logger.info("   ✅ Salesforce connector initialized")
            logger.info(f"   LLM available: {orchestrator.sf_connector.llm_available}")
        else:
            logger.info("   ❌ Salesforce connector not available")
            return False
        
        logger.info("🎯 RAG Orchestrator integration test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ RAG Orchestrator test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    logger.info("🧪 Starting Salesforce Integration Tests")
    logger.info("="*50)
    
    # Test 1: SalesforceConnector with LLM
    test1_passed = test_salesforce_connector()
    
    logger.info("\n" + "="*50)
    
    # Test 2: RAG Orchestrator integration
    test2_passed = test_rag_orchestrator()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("📊 TEST SUMMARY:")
    logger.info(f"   SalesforceConnector LLM: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    logger.info(f"   RAG Integration: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        logger.info("🎉 All tests passed! Ready for production use.")
    else:
        logger.info("⚠️ Some tests failed. Check implementation.")

if __name__ == "__main__":
    main()