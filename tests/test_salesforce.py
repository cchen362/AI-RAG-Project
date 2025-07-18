import sys
import os
# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from salesforce_connector import SalesforceConnector
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_salesforce_integration():
    """
    Test your Salesforce connection and data extraction
    """
    print("Testing Salesforce Integration...")

    # Test 1: Authentication
    print("1. Testing authentication...")
    sf_connector = SalesforceConnector()
    if sf_connector.test_connection():
        print("[SUCCESS] Authentication successful!")
    else:
        print("[FAILED] Authentication failed!")
        print("Check your .env file has correct Salesforce credentials")
        return
    
    # Test 2: Knowledge Article Extraction
    print("2. Testing knowledge article extraction...")
    articles = sf_connector.get_knowledge_articles(limit=5)
    if articles:
        print(f"[SUCCESS] Retrieved {len(articles)} knowledge articles")
        print(f"   Example title: {articles[0]['title']}")
    else:
        print("[INFO] No knowledge articles found (this might be normal for new orgs)")
    
    # Test 3: Case Solution Extraction
    print("3. Testing case solution extraction...")
    cases = sf_connector.get_case_solutions(limit=5)
    if cases:
        print(f"[SUCCESS] Retrieved {len(cases)} case solutions")
        print(f"   Example case: {cases[0]['title']}")
    else:
        print("[INFO] No case solutions found (this might be normal for new orgs)")
    
    # Test 4: Data Processing for RAG
    print("4. Testing data processing for RAG...")
    processed_docs = sf_connector.process_for_rag(articles, cases)
    if processed_docs:
        print(f"[SUCCESS] Processed {len(processed_docs)} documents for RAG")
        print(f"   Document types: {set(doc['metadata']['type'] for doc in processed_docs)}")
    elif not articles and not cases:
        print("[INFO] Data processing skipped - no source data available (normal for new orgs)")
    else:
        print("[FAILED] Data processing failed")
    
    # Test 5: Complete integration method
    print("5. Testing complete integration method...")
    all_knowledge = sf_connector.get_all_knowledge_for_rag()
    if all_knowledge:
        print(f"[SUCCESS] Retrieved {len(all_knowledge)} total knowledge items")
        print(f"   First item preview: {all_knowledge[0]['content'][:100]}...")
    elif not articles and not cases:
        print("[INFO] Complete integration successful but no data available (normal for new orgs)")
    else:
        print("[FAILED] Complete integration failed")
    
    print("\nSalesforce integration test completed!")
    print("\n[SUMMARY]")
    print(f"- Authentication: SUCCESS")
    print(f"- Knowledge Articles: {len(articles) if articles else 0} found")
    print(f"- Cases: {len(cases) if cases else 0} found")
    print(f"- Total RAG Documents: {len(all_knowledge) if all_knowledge else 0}")
    
    return sf_connector, all_knowledge if all_knowledge else []

if __name__ == "__main__":
    connector, docs = test_salesforce_integration()